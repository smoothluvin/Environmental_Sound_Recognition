import torch
import torchaudio
import numpy as np
from audio_processing import load_audio, extract_mel_spectrogram, extract_mfcc
from model import SoundCNN
from config import TARGET_CLASSES


# Loading the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SoundCNN(num_classes=10).to(device)

# Running a dummy forward pass to initialize fc1 correctly from model.py
dummy_input = torch.randn(1, 1, 64, 800).to(device)
_ = model(dummy_input)

# Load the saved model state
model.load_state_dict(torch.load("./models/sound_cnn.pth", map_location=device))
model.eval()

# Load and preprocess test audio
audio_path = "data/Filtered_Dataset/gun_shot/174290-6-1-0.wav"
waveform = load_audio(audio_path)
mel_spectrogram = extract_mel_spectrogram(waveform)
mel_spectrogram = mel_spectrogram.unsqueeze(0).to(device)  # Add batch and channel dimensions

# Making the prediction
with torch.no_grad():
    output = model(mel_spectrogram)
    predicted_class_idx = torch.argmax(output, dim=1).item()

predicted_class_name = TARGET_CLASSES[predicted_class_idx]

print(f"Predicted Sound Class: {predicted_class_name}")