import torch
import torchaudio
import torchaudio.transforms as T
import librosa
import numpy as np
from torch.utils.data import Dataset
import os

# Default Audio Settings
SAMPLE_RATE = 16000
N_MELS = 64

class AudioDataSet(Dataset):
    def __init__(self, root_dir, sample_rate=16000, n_mels=64, transform=None):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.transform = transform
        # Using os.listdir to get the list of audio files in the directory, but dataset should be structured with subdirectories for each class
        self.classes = sorted(os.listdir(root_dir))
        self.files = []

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for file in os.listdir(class_dir):
                if file.endswith(".wav"):
                    self.files.append((os.path.join(class_dir, file), label))
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path, label = self.files[idx]
        waveform, _ = load_audio(file_path)
        mel_spectrogram = extract_mel_spectrogram(waveform)

        if self.transform:
            mel_spectrogram = self.transform(mel_spectrogram)

        return mel_spectrogram, label

def load_audio(audio_path, target_sr = SAMPLE_RATE):
    """
    Loads an audio file and resamples it to the target sample rate if needed
    """
    waveform, sample_rate = torchaudio.load(audio_path)

    if sample_rate != target_sr:
        resample = T.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resample(waveform)

    return waveform, target_sr

def extract_mel_spectrogram(waveform, sample_rate = SAMPLE_RATE, n_mels = N_MELS):
    """
    Convert an audio waveform into a Mel Spectrogram
    """
    mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
    mel_spectrogram = mel_transform(waveform)

    return mel_spectrogram

def extract_mfcc(waveform, sample_rate = SAMPLE_RATE, n_mfcc = 13):
    """
    Convert an audio waveform into Mel Frequency Cepstral Coefficients (MFCC)
    """
    waveform_np = waveform.numpy().squeeze()
    mfccs = librosa.feature.mfcc(y=waveform_np, sr=sample_rate, n_mfcc=n_mfcc)

    return mfccs