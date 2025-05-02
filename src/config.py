# Audio Classes
TARGET_CLASSES_ESR = ["crying_baby", "glass_breaking", "gun_shot", "siren", "Screaming", "Slam", "Smoke_detector_smoke_alarm"]
TARGET_CLASSES_MUSIC = ["Acoustic_Guitar", "Drum_set", "Harmonica", "Piano", "background_noise", "silence"]

# Default Audio Settings
SAMPLE_RATE = 16000
MAX_FRAMES = 336        # This calculation 313 is assuming we're using a sample rate of 16000 and hop length of 512 for the transforms. Changing this to 336 (84 x 4) for training using both features and preserving a nice aspect ratio
N_MELS = 64             # Number of mel bins
N_MFCC = 20             # Number of MFCCs

