import torch
import torchaudio
from audio_processing import load_audio, extract_mel_spectrogram, extract_mfcc
from model import SoundCNN

# Loading the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SoundCNN(num_classes=10).to(device)
model.load_state_dict(torch.load("sound_cnn.pth", map_location=device))
model.eval()

# Load and preprocess test audio
audio_path = "./data/test.wav"
waveform, _ = load_audio(audio_path)
mel_spectrogram = extract_mel_spectrogram(waveform).unsqueeze(0).to(device)

# Making the prediction
with torch.no_grad():
    output = model(mel_spectrogram)
    predicted_class = torch.argmax(output, dim=1).item()

print(f"Predicted Sound Class: {predicted_class}")