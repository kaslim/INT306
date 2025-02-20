import torch
import numpy as np
import librosa
import os

# create AST model
from models.ASTModel import AST


# Function to load the model and weights
def load_model(model_path):
    model = AST(n_class=10, reprog_front='skip', map_num=5)
    ckpt_path = torch.load(model_path, map_location='cpu')
    model.load_state_dict(ckpt_path)
    model.eval()
    return model


# Function to extract features from the audio file
def extract_audio_features(file, seq_length=160000):
    "Extract audio features from an audio file for genre classification"
    audio, sr = librosa.load(file, sr=16000)

    n_chunk = len(audio) // seq_length
    audio_chunks = np.split(audio[:n_chunk * seq_length], n_chunk)

    # Pad the last chunk to make sure it has the same length
    if len(audio) % seq_length != 0:
        last_chunk = audio[n_chunk * seq_length:]
        padded_last_chunk = np.pad(last_chunk, (0, seq_length - len(last_chunk)), mode='constant')
        audio_chunks.append(padded_last_chunk)

    return torch.from_numpy(np.array(audio_chunks))


# Function to get genre prediction
def get_genre(model, file):
    "Predict genre of music using the trained AST model"
    audio_chunks = extract_audio_features(file)
    output, ori_emb, transformed_emb = model(audio_chunks)
    output = torch.sigmoid(output).detach().cpu().numpy()

    # Get the predicted genre
    idx = np.argmax(output.mean(0), axis=0)
    return idx


# Function to extract the actual genre from the filename
def extract_actual_genre(file):
    "Extract the actual genre label from the filename"
    # Assuming the genre is the first part of the filename, like 'classical.00001.wav'
    genre = os.path.basename(file).split('.')[0]
    genre_map = {
        "blues": 0,
        "classical": 1,
        "country": 2,
        "disco": 3,
        "hiphop": 4,
        "jazz": 5,
        "metal": 6,
        "pop": 7,
        "reggae": 8,
        "rock": 9,
    }
    return genre_map.get(genre, -1)  # Returns -1 if the genre is not in the map


# Function to predict a single audio file and compare with actual label
def predict_and_compare(model, file):
    "Predict the genre of a single audio file and compare with actual label"
    predicted_idx = get_genre(model, file)
    actual_idx = extract_actual_genre(file)

    if actual_idx == -1:
        print(f"Unknown genre for file: {file}")
        return

    genre_map = {
        0: "blues",
        1: "classical",
        2: "country",
        3: "disco",
        4: "hiphop",
        5: "jazz",
        6: "metal",
        7: "pop",
        8: "reggae",
        9: "rock",
    }

    predicted_genre = genre_map.get(predicted_idx, "Unknown")
    actual_genre = genre_map.get(actual_idx, "Unknown")

    print(f"File: {file}")
    print(f"Predicted: {predicted_genre}")
    print(f"Actual: {actual_genre}")


# Example usage
if __name__ == "__main__":
    model_path = '../models/best_model.pth'  # Change this to the path to your model
    model = load_model(model_path)

    # Replace with your audio file path
    file_path = '/home/yons/文档/CSI_dataset/GTZAN/genres_original/classical/classical.00006.wav'  # Change this to the path to a single audio file

    # Predict the genre and compare with the actual label
    predict_and_compare(model, file_path)
