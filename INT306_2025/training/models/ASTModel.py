# -*- coding: utf-8 -*-
# @Time    : 6/10/21 5:04 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : ast_models.py

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import wget
import timm
from timm.models.layers import to_2tuple, trunc_normal_
import torchaudio
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()

        # 转换 img_size 和 patch_size 为二元组
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size

        # 计算补丁的数量
        self.num_patches_height = img_size[0] // patch_size[0]
        self.num_patches_width = img_size[1] // patch_size[1]
        num_patches = self.num_patches_height * self.num_patches_width
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ASTModel(nn.Module):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    """

    def __init__(
            self,
            label_dim=527,
            fstride=10,
            tstride=10,
            input_fdim=128,
            input_tdim=1024,
            imagenet_pretrain=True,
            audioset_pretrain=False,
            model_size="base384",
            verbose=True,
    ):
        super(ASTModel, self).__init__()
        assert (
                timm.__version__ == "0.4.5"
        ), "Please use timm == 0.4.5, the code might not be compatible with newer versions."

        if verbose == True:
            print("---------------AST Model Summary---------------")
            print(
                "ImageNet pretraining: {:s}, AudioSet pretraining: {:s}".format(
                    str(imagenet_pretrain), str(audioset_pretrain)
                )
            )
        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        if audioset_pretrain == False:
            if model_size == "tiny224":
                self.v = timm.create_model(
                    "vit_deit_tiny_distilled_patch16_224", pretrained=imagenet_pretrain
                )
            elif model_size == "small224":
                self.v = timm.create_model(
                    "vit_deit_small_distilled_patch16_224", pretrained=imagenet_pretrain
                )
            elif model_size == "base224":
                self.v = timm.create_model(
                    "vit_deit_base_distilled_patch16_224", pretrained=imagenet_pretrain
                )
            elif model_size == "base384":
                self.v = timm.create_model(
                    "vit_deit_base_distilled_patch16_384", pretrained=imagenet_pretrain
                )
            else:
                raise Exception(
                    "Model size must be one of tiny224, small224, base224, base384."
                )

            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.original_embedding_dim),
                nn.Linear(self.original_embedding_dim, label_dim),
            )

            # automatcially get the intermediate shape
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print(
                    "frequncey stride={:d}, time stride={:d}".format(fstride, tstride)
                )
                print("number of patches={:d}".format(num_patches))

            # the linear projection layer
            new_proj = torch.nn.Conv2d(
                1,
                self.original_embedding_dim,
                kernel_size=(16, 16),
                stride=(fstride, tstride),
            )
            if imagenet_pretrain == True:
                new_proj.weight = torch.nn.Parameter(
                    torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1)
                )
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            # the positional embedding
            if imagenet_pretrain == True:
                # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
                new_pos_embed = (
                    self.v.pos_embed[:, 2:, :]
                    .detach()
                    .reshape(1, self.original_num_patches, self.original_embedding_dim)
                    .transpose(1, 2)
                    .reshape(
                        1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw
                    )
                )
                # cut (from middle) or interpolate the second dimension of the positional embedding
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[
                                    :,
                                    :,
                                    :,
                                    int(self.oringal_hw / 2)
                                    - int(t_dim / 2): int(self.oringal_hw / 2)
                                                      - int(t_dim / 2)
                                                      + t_dim,
                                    ]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(
                        new_pos_embed, size=(self.oringal_hw, t_dim), mode="bilinear"
                    )
                # cut (from middle) or interpolate the first dimension of the positional embedding
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[
                                    :,
                                    :,
                                    int(self.oringal_hw / 2)
                                    - int(f_dim / 2): int(self.oringal_hw / 2)
                                                      - int(f_dim / 2)
                                                      + f_dim,
                                    :,
                                    ]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(
                        new_pos_embed, size=(f_dim, t_dim), mode="bilinear"
                    )
                # flatten the positional embedding
                new_pos_embed = new_pos_embed.reshape(
                    1, self.original_embedding_dim, num_patches
                ).transpose(1, 2)
                # concatenate the above positional embedding with the cls token and distillation token of the deit model.
                self.v.pos_embed = nn.Parameter(
                    torch.cat(
                        [self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1
                    )
                )
            else:
                # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
                # TODO can use sinusoidal positional embedding instead
                new_pos_embed = nn.Parameter(
                    torch.zeros(
                        1,
                        self.v.patch_embed.num_patches + 2,
                        self.original_embedding_dim,
                    )
                )
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=0.02)

        # now load a model that is pretrained on both ImageNet and AudioSet
        elif audioset_pretrain == True:
            if audioset_pretrain == True and imagenet_pretrain == False:
                raise ValueError(
                    "currently model pretrained on only audioset is not supported, please set imagenet_pretrain = True to use audioset pretrained model."
                )
            if model_size != "base384":
                raise ValueError(
                    "currently only has base384 AudioSet pretrained model."
                )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            """
            if (
                os.path.exists("../../pretrained_models/audioset_10_10_0.4593.pth")
                == False
            ):
                # this model performs 0.4593 mAP on the audioset eval set
                audioset_mdl_url = (
                    "https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1"
                )
                wget.download(
                    audioset_mdl_url,
                    out="../../pretrained_models/audioset_10_10_0.4593.pth",
                )
            """
            sd = torch.load(
                "../pretrained_models/audioset_10_10_0.4593.pth", map_location=device
                # "/home/yons/文档/music-repro/pretrained_models/audioset_10_10_0.4593.pth", map_location=device
            )
            audio_model = ASTModel(
                label_dim=527,
                fstride=10,
                tstride=10,
                input_fdim=128,
                input_tdim=1024,
                imagenet_pretrain=False,
                audioset_pretrain=False,
                model_size="base384",
                verbose=False,
            )
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.original_embedding_dim),
                nn.Linear(self.original_embedding_dim, label_dim),
            )

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print(
                    "frequncey stride={:d}, time stride={:d}".format(fstride, tstride)
                )
                print("number of patches={:d}".format(num_patches))

            new_pos_embed = (
                self.v.pos_embed[:, 2:, :]
                .detach()
                .reshape(1, 1212, 768)
                .transpose(1, 2)
                .reshape(1, 768, 12, 101)
            )
            # if the input sequence length is larger than the original audioset (10s), then cut the positional embedding
            if t_dim < 101:
                new_pos_embed = new_pos_embed[
                                :, :, :, 50 - int(t_dim / 2): 50 - int(t_dim / 2) + t_dim
                                ]
            # otherwise interpolate
            else:
                new_pos_embed = torch.nn.functional.interpolate(
                    new_pos_embed, size=(12, t_dim), mode="bilinear"
                )
            if f_dim < 12:
                new_pos_embed = new_pos_embed[
                                :, :, 6 - int(f_dim / 2): 6 - int(f_dim / 2) + f_dim, :
                                ]
            # otherwise interpolate
            elif f_dim > 12:
                new_pos_embed = torch.nn.functional.interpolate(
                    new_pos_embed, size=(f_dim, t_dim), mode="bilinear"
                )
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(
                torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1)
            )

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(
            1,
            self.original_embedding_dim,
            kernel_size=(16, 16),
            stride=(fstride, tstride),
        )
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    @autocast()
    def forward(self, x, skip=None):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """

        skip_x = None  ##//TODO xintianjiade
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for i, blk in enumerate(self.v.blocks):
            x = blk(x)

            if skip is not None and i == skip[0]:
                skip_x = skip[1](skip[2](x[:, 2:].permute(0, 2, 1)).mean(-1))

        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2

        if skip_x is not None:
            _x = x * skip_x[:, :768] + skip_x[:, 768:]
            oup = self.mlp_head(_x)
            return oup, x, _x
        else:
            x = self.mlp_head(x)

            return x


class WrappedModel(nn.Module):
    def __init__(self, label_dim, input_tdim, imagenet_pretrain, audioset_pretrain):
        super(WrappedModel, self).__init__()

        self.module = ASTModel(
            label_dim=label_dim,
            input_tdim=input_tdim,
            imagenet_pretrain=imagenet_pretrain,
            audioset_pretrain=audioset_pretrain,
        )

    def forward(self, x, skip=None):
        return self.module(x, skip)


class AST(torch.nn.Module):
    def __init__(self, map_num=5, n_class=16, reprog_front=None, is_cuda=False):
        super().__init__()

        self.input_tdim = 1024  # 130#
        label_dim = 527  # 35 #
        self.class_num = n_class
        self.map_num = map_num
        self.reprog_front = reprog_front
        self.ast_mdl = WrappedModel(
            label_dim=label_dim,
            input_tdim=self.input_tdim,
            imagenet_pretrain=True,
            audioset_pretrain=True,
        )

        # for name, param in self.ast_mdl.named_parameters():
        #    # if 'mlp_head' not in name:
        #    param.requires_grad = False //TODO wogaile

        if reprog_front == "uni_noise":
            self.delta = torch.nn.Parameter(torch.Tensor(1, 160000), requires_grad=True)
            torch.nn.init.xavier_uniform_(self.delta)
        elif reprog_front == "condi":
            n_channel = 135
            self.linear = nn.Linear(n_channel, 128)
            self.conv = nn.Sequential(
                nn.Conv1d(128, n_channel, 3, 1, 1),
                nn.ReLU(),
                nn.BatchNorm1d(n_channel),
                nn.Conv1d(n_channel, n_channel, 3, 1, 1),
                nn.ReLU(),
                nn.BatchNorm1d(n_channel),
                nn.Conv1d(n_channel, n_channel, 3, 1, 1),
                nn.ReLU(),
                nn.BatchNorm1d(n_channel),
                nn.Conv1d(n_channel, n_channel, 3, 1, 1),
            )
        elif reprog_front == "skip":
            n_channel = 54
            self.linear = nn.Linear(n_channel, 768 * 2)
            self.conv = nn.Sequential(
                nn.Conv1d(768, n_channel, 3, 1, 1),
                nn.ReLU(),
                nn.BatchNorm1d(n_channel),
                nn.Conv1d(n_channel, n_channel, 3, 1, 1),
                nn.ReLU(),
                nn.BatchNorm1d(n_channel),
                nn.Conv1d(n_channel, n_channel, 3, 1, 1),
                nn.ReLU(),
                nn.BatchNorm1d(n_channel),
                nn.Conv1d(n_channel, n_channel, 3, 1, 1),
            )

    def preprocess(self, waveform, target_length=130):
        fbank = []

        for i in range(waveform.shape[0]):

            data = torchaudio.compliance.kaldi.fbank(
                waveform[i][
                    None,
                ],
                htk_compat=True,
                sample_frequency=16000,
                use_energy=False,
                window_type="hanning",
                num_mel_bins=128,
                dither=0.0,
                frame_shift=10,
            )

            n_frames = data.shape[0]
            p = target_length - n_frames
            if p > 0:
                m = torch.nn.ZeroPad2d((0, 0, 0, p))
                data = m(data)
            elif p < 0:
                data = data[0:target_length, :]

            fbank.append(
                data[
                    None,
                ]
            )
        fbank = torch.cat(fbank, 0)
        fbank = (fbank + 4.2677393) / (4.5689974 * 2)

        return fbank

    def forward(self, wav):
        n_batch = wav.shape[0]

        if self.reprog_front == "uni_noise":
            wav = wav + self.delta


        features = self.preprocess(wav, self.input_tdim)


        if self.reprog_front == "condi":
            features = self.linear(
                self.conv(features.permute(0, 2, 1)).permute(0, 2, 1)
            )

        if self.reprog_front == "skip":
            predicted, ori_emb, tra_emb = self.ast_mdl(features, [8, self.linear, self.conv])
            predicted = predicted[:, : self.class_num * self.map_num]
            predicted = predicted.view(-1, self.class_num, self.map_num).sum(dim=-1)
            return predicted, ori_emb, tra_emb
        else:
            predicted = self.ast_mdl(features)[:, : self.class_num * self.map_num]
            predicted = predicted.view(-1, self.class_num, self.map_num).sum(dim=-1)

            return predicted
