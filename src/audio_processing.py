import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import librosa
import numpy as np
from torch.utils.data import Dataset
import os
from config import TARGET_CLASSES # Importing class list from config.py

# Default Audio Settings
SAMPLE_RATE = 16000
N_MELS = 64
MAX_FRAMES = 800

def load_audio(audio_path, target_sr = SAMPLE_RATE):
    """
    Loads an audio file and resamples it to the target sample rate if needed
    """
    waveform, sample_rate = torchaudio.load(audio_path)

    # Converting stereo to mono if needed since I'm getting a runtime error when loading stereo files
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sample_rate != target_sr:
        waveform = T.Resample(orig_freq=sample_rate, new_freq=target_sr)(waveform)

    return waveform

def extract_mel_spectrogram(waveform, sample_rate = SAMPLE_RATE, n_mels = N_MELS, max_frames = MAX_FRAMES):
    """
    Convert an audio waveform into a Mel Spectrogram
    """
    mel_spectrogram = T.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)(waveform)

    # Ensuring all spectrograms have the same time dimensions
    num_frames = mel_spectrogram.shape[2]

    if num_frames > max_frames:
        mel_spectrogram = mel_spectrogram[:, :, :max_frames] # Truncating
    elif num_frames < max_frames:
        pad_amount = max_frames - num_frames
        mel_spectrogram = F.pad(mel_spectrogram, (0, pad_amount)) # Padding
    
    return mel_spectrogram

def extract_mfcc(waveform, sample_rate = SAMPLE_RATE, n_mfcc = 13):
    """
    Convert an audio waveform into Mel Frequency Cepstral Coefficients (MFCC)
    """
    waveform_np = waveform.numpy().squeeze()
    mfccs = librosa.feature.mfcc(y=waveform_np, sr=sample_rate, n_mfcc=n_mfcc)

    return mfccs

class AudioDataSet(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # Using os.listdir to get the list of audio files in the directory, but dataset should be structured with subdirectories for each class
        self.classes = TARGET_CLASSES
        self.class_mapping = {class_name: i for i, class_name in enumerate(self.classes)}

        self.files = []
        for class_name, label in self.class_mapping.items():
            class_dir = os.path.join(root_dir, class_name)
            for file in os.listdir(class_dir):
                if file.endswith(".wav"):
                    self.files.append((os.path.join(class_dir, file), label))

        # For Debugging: Prints out the first few files
        print("Loaded Dataset files:", self.files[:10]) # Prints the first 10 files

        if len(self.files) == 0:
            raise ValueError("No audio files found in the dataset directory.")
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            file_path, label = self.files[idx]  
            waveform = load_audio(file_path)
            mel_spectrogram = extract_mel_spectrogram(waveform)
            return mel_spectrogram, label
        except ValueError as e:
            print(f"🚨 Error loading file {file_path}: {e}")
            return None  # Skip problematic filesfile_path, label = self.files[idx]
            
