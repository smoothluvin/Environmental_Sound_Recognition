import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import librosa
import numpy as np
from matplotlib import pyplot as plt
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
            
# This section of audio_processing.py is for analyzing the waveforms in our dataset. What I want to accomplish by doing this
# is to figure out dimensional and general statistical information about the audio files so that I can restructure the architecture of the
# model to best fit the data. I'll be using this information to determine the number of layers, filters, and other hyperparameters

def analyze_audio_features(audio_path):
    """
    Analyze the audio features of a given audio file
    """

    # First we want to load the audio file and extract the features from them
    waveform = load_audio(audio_path)
    mel_spectrogram = extract_mel_spectrogram(waveform)
    mfccs = extract_mfcc(waveform)

    # Print out the shapes of the features, first we start with the original waveform
    print('\n ----- Audio Feature Analysis ----- \n')
    print(f"Original Waveform shape: {waveform.shape}")
    print(f"Sampling rate used: {SAMPLE_RATE}")

    # Mel Spectrogram analysis
    print('\n ----- Mel Spectrogram Analysis ----- \n')
    print(f"Mel Spectrogram shape: {mel_spectrogram.shape}")
    print(f"Number of Mel bins: {N_MELS}")
    print(f"Time Frames: {mel_spectrogram.shape[2]}")

    # Calculating actual frequency range from Mel Spectrogram
    print(f"Frequency Range: 0 - ~{SAMPLE_RATE / 2} Hz")
    print(f"Time Resolution: {mel_spectrogram.shape[2]} frames over ~{waveform.shape[1] / SAMPLE_RATE} seconds")
    print(f"Time per frame: ~{(waveform.shape[1]/SAMPLE_RATE)/mel_spectrogram.shape[2]*1000:.2f} ms")
    
    # MFCC details
    print("\n--- MFCC Analysis ---")
    print(f"MFCC shape: {mfccs.shape}")
    print(f"Number of MFCC coefficients: {mfccs.shape[0]}")
    
    # Visualize
    plt.figure(figsize=(15, 10))
    
    # Plot waveform
    plt.subplot(3, 1, 1)
    plt.plot(waveform[0].numpy())
    plt.title('Waveform')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    
    # Plot mel spectrogram
    plt.subplot(3, 1, 2)
    plt.imshow(librosa.power_to_db(mel_spectrogram[0].numpy()), aspect='auto', origin='lower')
    plt.title('Mel Spectrogram')
    plt.xlabel('Time Frames')
    plt.ylabel('Mel Frequency Bins')
    plt.colorbar(format='%+2.0f dB')
    
    # Plot MFCCs
    plt.subplot(3, 1, 3)
    plt.imshow(mfccs, aspect='auto', origin='lower')
    plt.title('MFCCs')
    plt.xlabel('Time Frames')
    plt.ylabel('MFCC Coefficients')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('audio_features_analysis.png')
    plt.show()
    
    # Calculate numerical stats
    print("\n--- Numerical Statistics ---")
    print(f"Mel Spectrogram - Min: {mel_spectrogram.min().item()}, Max: {mel_spectrogram.max().item()}")
    print(f"Mel Spectrogram - Mean: {mel_spectrogram.mean().item()}, Std: {mel_spectrogram.std().item()}")
    print(f"MFCC - Min: {np.min(mfccs)}, Max: {np.max(mfccs)}")
    print(f"MFCC - Mean: {np.mean(mfccs)}, Std: {np.std(mfccs)}")
    
    return {
        "waveform_shape": waveform.shape,
        "mel_shape": mel_spectrogram.shape,
        "mfcc_shape": mfccs.shape,
        "sample_rate": SAMPLE_RATE,
        "n_mels": N_MELS,
        "max_frames": MAX_FRAMES
    }

# Use this function with a few example files
def analyze_dataset(dataset_path, num_samples=5):
    """
    Analyze several files from the dataset to understand feature characteristics
    """
    # Get a list of classes from the directory structure
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    results = []
    
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
        
        # Take a few samples from each class
        samples = files[:min(num_samples, len(files))]
        
        for sample in samples:
            file_path = os.path.join(class_path, sample)
            print(f"\nAnalyzing file: {file_path}")
            result = analyze_audio_features(file_path)
            results.append(result)
    
    # Compute average shapes across samples
    avg_mel_shape = tuple(np.mean([r["mel_shape"][i] for r in results], axis=0).astype(int) for i in range(len(results[0]["mel_shape"])))
    avg_mfcc_shape = tuple(np.mean([r["mfcc_shape"][i] for r in results], axis=0).astype(int) for i in range(len(results[0]["mfcc_shape"])))
    
    print("\n--- Dataset Summary ---")
    print(f"Average Mel Spectrogram shape: {avg_mel_shape}")
    print(f"Average MFCC shape: {avg_mfcc_shape}")
    
    return results

# Example usage
if __name__ == "__main__":
    # Replace with your dataset path
    dataset_path = "./data/Filtered_Dataset"
    results = analyze_dataset(dataset_path)