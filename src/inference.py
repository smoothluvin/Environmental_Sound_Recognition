import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import pyaudio
import datetime
import soundfile as sf
import argparse
import threading
import ctypes
import subprocess

from cnn import SoundCNN
from config import TARGET_CLASSES_MUSIC, SAMPLE_RATE, N_MELS, N_MFCC, MAX_FRAMES
from audio_processing import extract_features_from_file
# Import filtering module but don't use it
# from filtering import process_audio

# Suppress ALSA warnings
asound = ctypes.CDLL('libasound.so')
asound.snd_lib_error_set_handler(ctypes.CFUNCTYPE(None, ctypes.c_char_p)(lambda x: None))

class RealTimeAudioProcessor:
    def __init__(self, device_index=None, sample_rate=44100, target_sample_rate=16000, window_duration=3.0):
        self.device_sample_rate = sample_rate
        self.target_sample_rate = target_sample_rate
        self.window_duration = window_duration
        self.device_index = device_index
        
        # List available audio capture devices
        self.list_audio_devices()
        
        # If device_index is None, attempt to find a USB audio device
        if self.device_index is None:
            self.find_usb_audio_device()

    def list_audio_devices(self):
        """List available audio devices using arecord"""
        try:
            print("\nAvailable audio capture devices:")
            result = subprocess.run(["arecord", "-l"], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"Error listing devices: {result.stderr}")
        except Exception as e:
            print(f"Failed to list audio devices: {e}")

    def find_usb_audio_device(self):
        """Attempt to find a USB audio device from the arecord -l output"""
        try:
            result = subprocess.run(["arecord", "-l"], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                # Parse the output to find USB audio devices
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'USB Audio' in line and 'card' in line:
                        # Extract card number
                        card_info = line.split(':')[0]
                        card_num = card_info.split('card ')[1].strip()
                        self.device_index = int(card_num)
                        print(f"Automatically selected USB Audio Device at card {self.device_index}")
                        return
                
                # If no USB Audio device found, use card 0 as default
                print("No USB Audio Device found. Using default card 0.")
                self.device_index = 0
            else:
                print(f"Error finding USB audio device: {result.stderr}")
                print("Defaulting to card 0")
                self.device_index = 0
        except Exception as e:
            print(f"Failed to find USB audio device: {e}")
            print("Defaulting to card 0")
            self.device_index = 0

    def record_audio_arecord(self, output_path="temp_arecord.wav"):
        """Record audio using arecord at 16000Hz sample rate"""
        # Ensure the device_index is not None and is a valid integer
        if self.device_index is None:
            self.device_index = 0  # Default to card 0 if not set
            
        device_str = f"plughw:{self.device_index},0"
        duration = self.window_duration
        
        # Explicitly set format to 16-bit signed little-endian at 16000Hz
        cmd = [
            "arecord",
            "-D", device_str,
            "-f", "S16_LE",        # 16-bit signed little-endian format
            "-r", "16000",         # 16kHz sample rate (matching your model)
            "-c", "1",             # Mono
            "-t", "wav",
            "-d", str(int(duration)),
            output_path
        ]
        
        print(f"[arecord] Capturing {duration} seconds using: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True)
            print(f"[arecord] Saved to: {output_path}")
            
            # Verify the recorded file
            info = sf.info(output_path)
            print(f"[arecord] Recording info: {info.samplerate}Hz, {info.channels} channels, {info.frames} frames, {info.duration:.2f} seconds")
            
        except subprocess.CalledProcessError as e:
            print(f"[arecord] Error recording audio: {e}")
            raise

    def get_audio(self):
        """Record audio and read it into numpy array"""
        try:
            # Record audio at the correct sample rate
            self.record_audio_arecord("temp_arecord.wav")
            
            # Read the recorded audio
            audio_data, sample_rate = sf.read("temp_arecord.wav")
            
            # Verify sample rate
            if sample_rate != self.target_sample_rate:
                print(f"[Warning] Recorded sample rate ({sample_rate}Hz) differs from expected ({self.target_sample_rate}Hz)")
                # No need to resample since we're explicitly recording at 16000Hz
            
            # If the recording is stereo, convert to mono
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Verify the audio length matches our expectation
            expected_samples = int(self.window_duration * self.target_sample_rate)
            actual_samples = len(audio_data)
            
            if abs(expected_samples - actual_samples) > 100:  # Allow slight variation
                print(f"[Warning] Audio length ({actual_samples} samples) differs from expected ({expected_samples} samples)")
                print(f"[Warning] Duration: {actual_samples/self.target_sample_rate:.2f}s vs expected {self.window_duration:.2f}s")
            
            return audio_data.astype(np.float32)
            
        except Exception as e:
            print(f"[Error] Failed to get audio: {e}")
            # Return a silent buffer instead of crashing
            return np.zeros(int(self.window_duration * self.target_sample_rate), dtype=np.float32)

class AudioInference:
    def __init__(self, model_path, use_mel=True, use_mfcc=True, threshold=0.5, device_index=None):
        # Create directory for logged audio
        os.makedirs("./logged_audio", exist_ok=True)
        
        self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        checkpoint = torch.load(model_path, map_location=self.device)
        self.input_shape = checkpoint.get('input_shape', (1, 84, MAX_FRAMES))
        self.use_mel = checkpoint.get('use_mel', use_mel)
        self.use_mfcc = checkpoint.get('use_mfcc', use_mfcc)

        print(f"Model input shape: {self.input_shape}")
        print(f"Using features: Mel Spectrogram = {self.use_mel}, MFCC = {self.use_mfcc}")

        self.model = SoundCNN(input_shape=self.input_shape)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.audio_processor = RealTimeAudioProcessor(
            device_index=device_index,
            sample_rate=16000,  # Set the desired recording sample rate to match the model
            target_sample_rate=SAMPLE_RATE,
            window_duration=3.0
        )

        self.threshold = threshold
        self.running = False
        self.thread = None

    def inference_thread(self):
        """Thread function to process audio and make predictions"""
        prediction_interval = self.audio_processor.window_duration
        last_prediction_time = 0
        
        while self.running:
            current_time = time.time()
            
            # Process audio every interval seconds
            if current_time - last_prediction_time >= prediction_interval:
                try:
                    # Get audio from the processor
                    audio_data = self.audio_processor.get_audio()
                    
                    # Extract features
                    features = self.extract_features(audio_data)
                    
                    if features is not None:
                        print(f"Feature shape before model: {features.shape}")
                        
                        # Make prediction
                        with torch.no_grad():
                            outputs = self.model(features)
                            probabilities = F.softmax(outputs, dim=1)[0]
                            confidence, predicted_idx = torch.max(probabilities, 0)
                            class_name = TARGET_CLASSES_MUSIC[predicted_idx.item()]
                            confidence_value = confidence.item()
                            
                            # Display prediction if confidence exceeds threshold
                            if confidence_value >= self.threshold:
                                print(f"Detected: {class_name} ({confidence_value:.2f})")
                    
                    last_prediction_time = current_time
                except Exception as e:
                    print(f"Error in inference thread: {e}")
                    time.sleep(1)  # Wait a bit before trying again
            
            time.sleep(0.1)

        print("Inference thread exiting.")

    def start(self):
        """Start real-time inference"""
        self.running = True
        self.thread = threading.Thread(target=self.inference_thread)
        self.thread.daemon = True
        self.thread.start()
        
        print("Starting real-time inference. Press Ctrl+C to stop.")
        
        try:
            # Keep running until interrupted
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping inference.")
        finally:
            self.stop()

    def stop(self):
        """Stop inference"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def extract_features(self, audio_data):
        """Extract features from raw audio data without filtering"""
        try:
            if not np.all(np.isfinite(audio_data)):
                print("[Error] Audio buffer is not finite everywhere.")
                return None

            # Save audio data to temporary file
            temp_file = "_temp_inference.wav"
            sf.write(temp_file, audio_data, SAMPLE_RATE, subtype='PCM_16')

            # Generate timestamp for logging
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save raw audio for manual inspection
            raw_path = f"./logged_audio/raw_{timestamp}.wav"
            sf.write(raw_path, audio_data, SAMPLE_RATE, subtype='PCM_16')
            print(f"[Saved] Logged raw audio to: {raw_path}")
            
            # Extract features directly from the raw audio
            features = extract_features_from_file(temp_file, use_mel=self.use_mel, use_mfcc=self.use_mfcc)
            
            # Clean up temporary file
            os.remove(temp_file)
            
            # Return features tensor
            return features.unsqueeze(0).to(self.device) if features is not None else None
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def process_file(self, file_path):
        """Process an audio file and return the prediction"""
        try:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return None
                
            # Extract features directly from the file
            features = extract_features_from_file(file_path, use_mel=self.use_mel, use_mfcc=self.use_mfcc)
            
            if features is None:
                return None
                
            # Reshape features for model input
            features = features.unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(features)
                probabilities = F.softmax(outputs, dim=1)[0]
                confidence, predicted_idx = torch.max(probabilities, 0)
                
                class_name = TARGET_CLASSES_MUSIC[predicted_idx.item()]
                confidence_value = confidence.item()
                
                return class_name, confidence_value
                
        except Exception as e:
            print(f"Error processing file: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description='Real-time audio classification')
    parser.add_argument('--model', type=str, default='./models/Music_Models/best_model.pth', 
                        help='Path to the saved model file')
    parser.add_argument('--device', type=int, default=None, 
                        help='Index of the audio input device to use')
    parser.add_argument('--file', type=str, default=None, 
                        help='Path to an audio file to classify (instead of using microphone)')
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='Confidence threshold for predictions')
    
    args = parser.parse_args()
    
    # Initialize the audio inference system
    inference = AudioInference(
        model_path=args.model,
        device_index=args.device,
        threshold=args.threshold
    )
    
    # Either process a file or start real-time inference
    if args.file:
        print(f"Processing file: {args.file}")
        result = inference.process_file(args.file)
        if result:
            class_name, confidence = result
            print(f"Prediction: {class_name} with confidence {confidence:.2f}")
        else:
            print("Could not make a prediction for this file.")
    else:
        # Start real-time inference with the microphone
        inference.start()

if __name__ == "__main__":
    main()
