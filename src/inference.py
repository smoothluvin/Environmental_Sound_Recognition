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
from collections import deque

from cnn import SoundCNN
from config import TARGET_CLASSES_MUSIC, SAMPLE_RATE, N_MELS, N_MFCC, MAX_FRAMES
from audio_processing import extract_features_from_file, extract_features_from_waveform, normalize_waveform
# Import filtering module but don't use it
# from filtering import process_audio

# Suppress ALSA warnings
asound = ctypes.CDLL('libasound.so')
asound.snd_lib_error_set_handler(ctypes.CFUNCTYPE(None, ctypes.c_char_p)(lambda x: None))

def calibrated_softmax(logits, temperature=1.5):
    """
    Apply temperature scaling to logits before softmax
    Higher temperature = softer predictions (less confident)
    """
    return F.softmax(logits / temperature, dim=1)

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
            "-q",                  # Add quiet flag
            output_path
        ]

        try:
            # Use subprocess with stdout and stderr redirected to suppress messages
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            # Still handle errors but silently
            return False

        return True

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

class PredictionSmoother:
    def __init__(self, window_size=5, threshold=0.7):
    	"""
    	Initialize with class-specific thresholds
    	"""
    	self.window_size = window_size
    	self.threshold = threshold
    	self.recent_predictions = deque(maxlen=window_size)
    
    	# Higher threshold for background_noise to reduce false positives
    	self.class_thresholds = {
        	0: 0.85,  # Acoustic_Guitar
        	1: 0.7,  # Drum_set
        	2: 0.6,  # Harmonica (lower due to fewer samples)
        	3: 0.7,  # Piano
        	4: 0.8,  # background_noise (higher to reduce overconfidence)
        	5: 0.6   # silence
    	}
    
    def update_class_threshold(self, class_idx, confidence):
        """Update running average of confidence for a class"""
        if class_idx not in self.class_thresholds:
            self.class_thresholds[class_idx] = confidence
        else:
            # Exponential moving average with alpha=0.3
            alpha = 0.3
            self.class_thresholds[class_idx] = alpha * confidence + (1 - alpha) * self.class_thresholds[class_idx]
    
    def get_smoothed_prediction(self, class_idx, confidence):
        """
        Add current prediction and return smoothed result
        
        Returns:
            tuple of (class_idx, confidence) or None if no stable prediction
        """
        # Update class threshold
        self.update_class_threshold(class_idx, confidence)
        
        # Add current prediction
        self.recent_predictions.append((class_idx, confidence))
        
        # Not enough predictions yet
        if len(self.recent_predictions) < 3:
            return None
            
        # Count occurrences of each class
        class_counts = {}
        class_confs = {}
        
        for idx, conf in self.recent_predictions:
            if idx not in class_counts:
                class_counts[idx] = 0
                class_confs[idx] = []
            class_counts[idx] += 1
            class_confs[idx].append(conf)
        
        # Find most common class
        most_common = max(class_counts.items(), key=lambda x: x[1])
        most_common_idx = most_common[0]
        count = most_common[1]
        
        # Only return a prediction if it appears in majority of recent predictions
        if count >= len(self.recent_predictions) * self.threshold:
            avg_conf = sum(class_confs[most_common_idx]) / len(class_confs[most_common_idx])
            
            # Only return if confidence is above the adaptive threshold
            if most_common_idx in self.class_thresholds:
                adaptive_threshold = max(0.5, self.class_thresholds[most_common_idx] * 0.7)
                if avg_conf >= adaptive_threshold:
                    return (most_common_idx, avg_conf)
        
        return None

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
        self.prediction_smoother = PredictionSmoother(window_size=5, threshold=0.6)
        
        # Class-specific thresholds (will be auto-adjusted during inference)
        self.class_thresholds = {i: threshold for i in range(len(TARGET_CLASSES_MUSIC))}
        
        # Debug mode for logging predictions
        self.debug_mode = True
        self.log_predictions = []

    def inference_thread(self):
    	"""Thread function to process audio and make predictions"""
    	prediction_interval = 0.5  # Make predictions more frequently than window duration
    	last_prediction_time = 0
    
    	while self.running:
        	current_time = time.time()
        
        	# Process audio every interval seconds
        	if current_time - last_prediction_time >= prediction_interval:
            		try:
                		# Get audio from the processor
                		audio_data = self.audio_processor.get_audio()
                
                		# Convert numpy array to torch tensor
                		waveform = torch.tensor(audio_data).float().unsqueeze(0)
                
                		# Normalize waveform (ensure consistency with training)
                		waveform = normalize_waveform(waveform)
                
                		# Extract features directly from waveform
                		features = extract_features_from_waveform(
                    		waveform, 
                    		use_mel=self.use_mel, 
                    		use_mfcc=self.use_mfcc
                		)
                
                		if features is not None:
                    			# Save raw audio for manual inspection (only in debug mode)
                    			if self.debug_mode:
                        			timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        			raw_path = f"./logged_audio/raw_{timestamp}.wav"
                        			sf.write(raw_path, audio_data, SAMPLE_RATE, subtype='PCM_16')
                    
                    			# Make prediction
                    			with torch.no_grad():
                        			outputs = self.model(features.unsqueeze(0).to(self.device))
                        			probabilities = calibrated_softmax(outputs, temperature=1.5)[0]
                        
                        			# Get top 2 predictions for debugging
                        			confidences, indices = torch.topk(probabilities, 2)
                        
                        			# Use two-stage prediction
                        			predicted_idx, confidence = self.two_stage_predict(features)
                        			class_name = TARGET_CLASSES_MUSIC[predicted_idx]
                        
                        			# Get second-best prediction (for logging)
                        			confidence2, predicted_idx2 = confidences[1].item(), indices[1].item()
                        			class_name2 = TARGET_CLASSES_MUSIC[predicted_idx2]
                        
                        			# Apply smoothing
                        			smoothed_prediction = self.prediction_smoother.get_smoothed_prediction(
                            			predicted_idx, confidence
                        			)
                        
                        			# Simplified output - just show the prediction and confidence
                        			if smoothed_prediction:
                            				smooth_idx, smooth_conf = smoothed_prediction
                            				smooth_class = TARGET_CLASSES_MUSIC[smooth_idx]
                            				print(f"{smooth_class}: {smooth_conf:.2f}")
                        			else:
                            				# No stable prediction yet, just show the current prediction
                            				print(f"{class_name}: {confidence:.2f} (unstable)")
                        
                        			# Log prediction for debugging
                        			if self.debug_mode:
                            				self.log_predictions.append({
                                				'time': timestamp if 'timestamp' in locals() else datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                                				'class': class_name,
                                				'confidence': confidence,
                                				'second_class': class_name2,
                                				'second_confidence': confidence2,
                                				'audio_path': raw_path if 'raw_path' in locals() else None
                            				})
                            
                    			last_prediction_time = current_time
                    
            		except Exception as e:
                		print(f"Error: {e}")
                		time.sleep(1)  # Wait a bit before trying again
            
        	time.sleep(0.1)

    	print("Inference thread exiting.")
    
    	# Save debug logs if enabled
    	if self.debug_mode and self.log_predictions:
        	import json
        	log_path = f"./logged_audio/predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        	with open(log_path, 'w') as f:
            		json.dump(self.log_predictions, f, indent=2)
        	print(f"Saved prediction logs to {log_path}")

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

            # Convert numpy array to torch tensor
            waveform = torch.tensor(audio_data).float().unsqueeze(0)
            
            # Normalize waveform (ensure consistency with training)
            waveform = normalize_waveform(waveform)
            
            # Extract features directly from waveform
            features = extract_features_from_waveform(
                waveform, 
                use_mel=self.use_mel, 
                use_mfcc=self.use_mfcc
            )
            
            # Generate timestamp for logging
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save raw audio for manual inspection (if needed)
            raw_path = f"./logged_audio/raw_{timestamp}.wav"
            sf.write(raw_path, audio_data, SAMPLE_RATE, subtype='PCM_16')
            print(f"[Saved] Logged raw audio to: {raw_path}")
            
            return features
            
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
                probabilities = calibrated_softmax(outputs, temperature=1.5)[0]
                
                # Get top 2 predictions
                confidences, indices = torch.topk(probabilities, 2)
                
                # Primary prediction
                confidence, predicted_idx = confidences[0].item(), indices[0].item()
                class_name = TARGET_CLASSES_MUSIC[predicted_idx]
                
                # Second best prediction
                confidence2, predicted_idx2 = confidences[1].item(), indices[1].item()
                class_name2 = TARGET_CLASSES_MUSIC[predicted_idx2]
                
                print(f"Top prediction: {class_name} ({confidence:.2f})")
                print(f"Second prediction: {class_name2} ({confidence2:.2f})")
                
                return class_name, confidence
                
        except Exception as e:
            print(f"Error processing file: {e}")
            return None
        
    def two_stage_predict(self, features):
        """
        First determine if it's an instrument vs non-instrument
        Then classify specific class
        """
        with torch.no_grad():
            outputs = self.model(features.unsqueeze(0).to(self.device))
            probs = calibrated_softmax(outputs, temperature=1.5)[0]
        
            # Get background noise and silence probabilities
            bg_idx = TARGET_CLASSES_MUSIC.index("background_noise")
            silence_idx = TARGET_CLASSES_MUSIC.index("silence")
            bg_prob = probs[bg_idx]
            silence_prob = probs[silence_idx]
        
            # If background noise probability is very high, check second prediction
            if bg_prob > 0.75:
                # Get second highest prediction
                probs_copy = probs.clone()
                probs_copy[bg_idx] = 0  # Zero out background_noise
                second_conf, second_idx = torch.max(probs_copy, 0)
            
                # If second prediction is strong enough, return it instead
                if second_conf > 0.2:
                    return second_idx.item(), second_conf.item()
                
            # Return the normal prediction
            conf, pred_idx = torch.max(probs, 0)
            return pred_idx.item(), conf.item()

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
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with detailed logging')
    
    args = parser.parse_args()
    
    # Initialize the audio inference system
    inference = AudioInference(
        model_path=args.model,
        device_index=args.device,
        threshold=args.threshold
    )
    
    # Set debug mode
    inference.debug_mode = args.debug
    
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
