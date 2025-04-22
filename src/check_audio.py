import numpy as np
import librosa
import soundfile as sf
import os
from tqdm import tqdm
import glob
import shutil

def augment_glass_breaking_samples(input_dir, output_dir, num_augmentations=5):
    """Generate augmented versions of glass breaking audio files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all glass breaking audio files
    audio_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    
    print(f"Found {len(audio_files)} glass breaking audio files")
    
    for audio_file in tqdm(audio_files, desc="Augmenting glass breaking samples"):
        try:
            # Load the audio file using librosa instead of torchaudio
            file_path = os.path.join(input_dir, audio_file)
            waveform, sample_rate = librosa.load(file_path, sr=None)  # Keep original sample rate
            
            # Skip empty files
            if len(waveform) == 0:
                print(f"Skipping empty file: {audio_file}")
                continue
            
            # Base filename without extension
            base_name = os.path.splitext(audio_file)[0]
            
            for i in range(num_augmentations):
                # Apply random augmentations
                augmented = waveform.copy()
                
                # 1. Time shifting: shift the sound within the clip
                shift_amount = int(np.random.uniform(-0.2, 0.2) * len(augmented))
                if shift_amount > 0:
                    augmented = np.pad(augmented, (0, shift_amount), mode='constant')[shift_amount:]
                else:
                    augmented = np.pad(augmented, (-shift_amount, 0), mode='constant')[:shift_amount]
                
                # 2. Time stretching: make the sound slightly faster/slower
                stretch_factor = np.random.uniform(0.8, 1.2)
                try:
                    augmented = librosa.effects.time_stretch(augmented, rate=stretch_factor)
                except Exception as e:
                    print(f"Error during time stretching: {e}. Skipping this augmentation.")
                    continue
                
                # 3. Add background noise at a low level
                noise = np.random.normal(0, 0.01, size=len(augmented))
                noise_level = np.random.uniform(0.001, 0.01)
                augmented = augmented + (noise * noise_level)
                
                # 4. Random volume adjustment
                volume_factor = np.random.uniform(0.8, 1.2)
                augmented = augmented * volume_factor
                
                # 5. Pitch shift (optional, may change sound characteristics)
                if np.random.random() > 0.5:
                    try:
                        pitch_steps = np.random.uniform(-2, 2)
                        augmented = librosa.effects.pitch_shift(augmented, sr=sample_rate, n_steps=pitch_steps)
                    except Exception as e:
                        print(f"Error during pitch shifting: {e}. Skipping this step.")
                
                # Save using soundfile instead of torchaudio
                output_path = os.path.join(output_dir, f"{base_name}_aug_{i+1}.wav")
                sf.write(output_path, augmented, sample_rate)
                
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue
            
    # Count the actual augmented files created
    augmented_files = len([f for f in os.listdir(output_dir) if f.endswith('.wav')])
    print(f"Created {augmented_files} augmented samples")

def copy_augmented_files(augmented_dir, target_dir):
    """Copy all augmented files to the target directory."""
    augmented_files = glob.glob(os.path.join(augmented_dir, "*.wav"))
    
    print(f"Copying {len(augmented_files)} augmented files to {target_dir}")
    
    for file in tqdm(augmented_files, desc="Copying files"):
        shutil.copy(file, target_dir)
    
    print("Done!")

def main():
    # Define directories
    base_dir = "./data/AudioSet"
    input_dir = os.path.join(base_dir, "train/glass_breaking")
    output_dir = os.path.join(base_dir, "train/glass_breaking_augmented")
    
    # Set number of augmentations per file
    num_augmentations = 5
    
    # Run augmentation
    augment_glass_breaking_samples(input_dir, output_dir, num_augmentations)
    
    # Copy augmented files to training directory
    copy_augmented_files(output_dir, input_dir)

if __name__ == "__main__":
    main()