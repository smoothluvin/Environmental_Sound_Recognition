import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import os
from torch.utils.data import Dataset
from config import TARGET_CLASSES_MUSIC

# Default Audio Settings
SAMPLE_RATE = 16000
N_MELS = 64
MAX_FRAMES = 800
N_MFCC = 20  # Number of MFCC coefficients

def load_audio(audio_path, target_sr=SAMPLE_RATE):
    """
    Loads an audio file and resamples it to the target sample rate if needed
    """
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Check if the waveform is empty
        if waveform.numel() == 0:
            raise ValueError(f"Empty waveform in file: {audio_path}")
            
        # Converting stereo to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Ensure the waveform has at least some minimal length
        min_length = 200  # Minimum sample length to process
        if waveform.shape[1] < min_length:
            # Pad with zeros if too short
            waveform = F.pad(waveform, (0, min_length - waveform.shape[1]))
            
        if sample_rate != target_sr:
            waveform = T.Resample(orig_freq=sample_rate, new_freq=target_sr)(waveform)
            
        return waveform
        
    except Exception as e:
        print(f"Error loading file {audio_path}: {str(e)}")
        # Return a dummy waveform of appropriate length if loading fails
        return torch.zeros(1, SAMPLE_RATE)  # 1 second of silence

def extract_mel_spectrogram(waveform, sample_rate=SAMPLE_RATE, n_mels=N_MELS, max_frames=MAX_FRAMES):
    """
    Convert an audio waveform into a Mel Spectrogram
    """
    try:
        mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=1024,
            hop_length=512
        )
        
        mel_spectrogram = mel_transform(waveform)
        
        # Apply log-mel transform (common practice for audio)
        mel_spectrogram = torch.log(mel_spectrogram + 1e-9)  # Add small constant to avoid log(0)
        
        # Ensuring all spectrograms have the same time dimensions
        num_frames = mel_spectrogram.shape[2]
        
        if num_frames > max_frames:
            mel_spectrogram = mel_spectrogram[:, :, :max_frames]  # Truncating
        elif num_frames < max_frames:
            pad_amount = max_frames - num_frames
            mel_spectrogram = F.pad(mel_spectrogram, (0, pad_amount))  # Padding
            
        return mel_spectrogram
        
    except Exception as e:
        print(f"Error extracting mel spectrogram: {str(e)}")
        # Return a dummy spectrogram of appropriate dimensions
        return torch.zeros(1, n_mels, max_frames)

def extract_mfcc(waveform, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC, max_frames=MAX_FRAMES):
    """
    Convert an audio waveform into Mel Frequency Cepstral Coefficients (MFCC)
    using torchaudio for consistency
    """
    try:
        # Using torchaudio's MFCC transform
        mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            log_mels=True,
            melkwargs={"n_fft": 1024, "hop_length": 512, "n_mels": N_MELS}
        )
        
        mfccs = mfcc_transform(waveform)  # Shape: [1, n_mfcc, time]
        
        # Ensure consistent time dimension
        num_frames = mfccs.shape[2]
        if num_frames > max_frames:
            mfccs = mfccs[:, :, :max_frames]  # Truncate
        elif num_frames < max_frames:
            pad_amount = max_frames - num_frames
            mfccs = F.pad(mfccs, (0, pad_amount))  # Pad
            
        return mfccs
        
    except Exception as e:
        print(f"Error extracting MFCCs: {str(e)}")
        # Return dummy MFCCs of appropriate dimensions
        return torch.zeros(1, n_mfcc, max_frames)

class AudioDataSet(Dataset):
    def __init__(self, root_dir, use_mfcc=False, use_mel=True):
        """
        Initialize dataset with options to include MFCC, mel spectrogram, or both
        
        Args:
            root_dir: Root directory containing audio files in class folders
            use_mfcc: Whether to include MFCC features
            use_mel: Whether to include mel spectrogram features
        """
        self.root_dir = root_dir
        self.use_mfcc = use_mfcc
        self.use_mel = use_mel
        
        if not (use_mfcc or use_mel):
            raise ValueError("At least one of use_mfcc or use_mel must be True")
        
        self.classes = TARGET_CLASSES_MUSIC
        self.class_mapping = {class_name: i for i, class_name in enumerate(self.classes)}
        
        self.files = []
        for class_name, label in self.class_mapping.items():
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for file in os.listdir(class_dir):
                    if file.endswith(".wav"):
                        file_path = os.path.join(class_dir, file)
                        # Check if file is valid and non-empty
                        try:
                            info = torchaudio.info(file_path)
                            if info.num_frames > 0:
                                self.files.append((file_path, label))
                            else:
                                print(f"Skipping empty audio file: {file_path}")
                        except Exception as e:
                            print(f"Error reading file info {file_path}: {str(e)}")
            else:
                print(f"Warning: Directory {class_dir} not found")
                
        # Print some info for debugging
        print(f"Loaded Dataset files: {self.files[:10]}")  # First 10 files
        
        if len(self.files) == 0:
            raise ValueError("No audio files found in the dataset directory.")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        try:
            file_path, label = self.files[idx]
            waveform = load_audio(file_path)
            
            features = []
            
            if self.use_mel:
                mel_spectrogram = extract_mel_spectrogram(waveform)
                features.append(mel_spectrogram)
            
            if self.use_mfcc:
                mfccs = extract_mfcc(waveform)
                features.append(mfccs)
            
            # Combine features if using both
            if len(features) > 1:
                # Concatenate along the channel dimension
                combined_features = torch.cat(features, dim=1)
                return combined_features, label
            
            # If only using one feature type
            return features[0], label
            
        except Exception as e:
            print(f"Error processing item {idx}, file {self.files[idx][0]}: {str(e)}")
            # Return a default tensor and its label to avoid crashing
            if self.use_mfcc and self.use_mel:
                return torch.zeros(1, N_MELS + N_MFCC, MAX_FRAMES), label
            elif self.use_mel:
                return torch.zeros(1, N_MELS, MAX_FRAMES), label
            else:  # use_mfcc only
                return torch.zeros(1, N_MFCC, MAX_FRAMES), label
            