import numpy as np
import noisereduce as nr
import librosa
import soundfile as sf

# --- Step 1: Load your noisy audio ---
audio, sr = librosa.load('noisy_audio.wav', sr=None)

# --- Step 2: Baseline denoising with spectral subtraction ---
# The noisereduce library applies spectral subtraction internally.
reduced_noise = nr.reduce_noise(y=audio, sr=sr)

# --- Step 3: Estimate a noise reference ---
# For demonstration, we assume the residual (original - reduced) approximates the noise.
noise_ref = audio - reduced_noise

# --- Step 4: Define an LMS adaptive filter ---
def lms_filter(desired, reference, mu=0.001, order=64):
    n_samples = len(reference)
    w = np.zeros(order)      # filter coefficients
    output = np.zeros(n_samples)
    error = np.zeros(n_samples)
    # Zero-pad the reference for initial conditions
    padded_ref = np.concatenate((np.zeros(order - 1), reference))
    for n in range(n_samples):
        # Get current window (reversed for convolution)
        x = padded_ref[n:n + order][::-1]
        output[n] = np.dot(w, x)
        error[n] = desired[n] - output[n]
        w += 2 * mu * error[n] * x
    return output, error, w

# --- Step 5: Apply LMS filter ---
# Here, 'desired' is our baseline denoised signal and 'reference' is the estimated noise.
_, error, _ = lms_filter(desired=reduced_noise, reference=noise_ref, mu=0.001, order=64)

# The LMS output (error) is our refined denoised signal.
final_denoised = error

# --- Step 6: Save the final output ---
sf.write('final_denoised.wav', final_denoised, sr)
