import torch
import torchaudio
import torchaudio.transforms as T
import librosa
import numpy as np

# Default Audio Settings
SAMPLE_RATE = 16000
N_MELS = 64

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