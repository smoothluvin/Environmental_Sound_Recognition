import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from audio_processing import load_audio, extract_mel_spectrogram, extract_mfcc

# Load an audio file
audio_path = "./data/test.wav"
waveform, sample_rate = load_audio(audio_path)

# Extract Mel Spectrogram
mel_spectrogram = extract_mel_spectrogram(waveform, sample_rate)

# Extract MFCC
mfccs = extract_mfcc(waveform, sample_rate)

# Convert Mel Spectrogram to numpy for visualization
mel_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram).numpy()

# Plot and display Mel Spectrogram
plt.figure(figsize=(10, 4))
plt.imshow(mel_spectrogram_db[0], aspect='auto', origin='lower')
plt.colorbar(label = 'dB')
plt.title("Mel Spectrogram")
plt.xlabel("Time")
plt.ylabel("Mel Frequency")

# Plot and display MFCC
plt.figure(figsize=(10, 4))
plt.imshow(mfccs, aspect='auto', origin='lower')
plt.colorbar(label = 'dB')
plt.title("MFCCs")
plt.xlabel("Time")
plt.ylabel("MFCC Coefficients")
plt.show()


