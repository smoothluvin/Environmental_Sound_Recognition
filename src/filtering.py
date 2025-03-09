import numpy as np
import librosa
import noisereduce as nr
import pywt
from scipy.stats import kurtosis
from scipy.signal import wiener
import soundfile as sf
import os

# ----------------------- Filtering Functions -----------------------

def wavelet_threshold_filter(signal, wavelet='db4', level=5):
    """
    Applies wavelet thresholding to the signal.
    (Intended for sharp transient/impulse-like sounds.)
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    threshold = np.median(np.abs(coeffs[-1]))  # basic threshold estimate
    new_coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    filtered_signal = pywt.waverec(new_coeffs, wavelet)
    return filtered_signal

def adaptive_lms_filter(signal, mu=0.01, filter_order=4):
    """
    A simple adaptive LMS filter.
    (Designed to further refine the output for impulse-like signals.)
    """
    N = len(signal)
    w = np.zeros(filter_order)
    filtered_signal = np.zeros(N)
    for n in range(filter_order, N):
        x = signal[n - filter_order:n][::-1]
        y = np.dot(w, x)
        e = signal[n] - y
        w = w + 2 * mu * e * x
        filtered_signal[n] = y
    filtered_signal[:filter_order] = signal[:filter_order]
    return filtered_signal

def adaptive_wiener_filter(signal, mysize=None):
    """
    Uses the Wiener filter for adaptive noise reduction.
    (Well suited for speech-like, tonal signals.)
    """
    return wiener(signal, mysize=mysize)

def kalman_filter(signal, process_variance=1e-5, measurement_variance=0.01):
    """
    A simple 1D Kalman filter implementation.
    (Used here to further smooth signals in the smooth/tonal category.)
    """
    n = len(signal)
    xhat = np.zeros(n)   # posterior estimate
    P = np.zeros(n)      # posterior error estimate
    xhat[0] = signal[0]
    P[0] = 1.0
    Q = process_variance
    R = measurement_variance
    for k in range(1, n):
        # Time update
        xhat_minus = xhat[k-1]
        P_minus = P[k-1] + Q
        # Measurement update
        K = P_minus / (P_minus + R)
        xhat[k] = xhat_minus + K*(signal[k] - xhat_minus)
        P[k] = (1 - K) * P_minus
    return xhat

def rls_filter(signal, lam=0.99, delta=0.1, filter_order=4):
    """
    A simple Recursive Least Squares (RLS) adaptive filter.
    (Intended for handling rapid noise variations.)
    """
    N = len(signal)
    w = np.zeros(filter_order)
    P = np.eye(filter_order) / delta
    filtered_signal = np.zeros(N)
    for n in range(filter_order, N):
        x = signal[n - filter_order:n][::-1]
        x_vec = x.reshape(-1, 1)
        y = np.dot(w, x)
        e = signal[n] - y
        Px = np.dot(P, x_vec)
        k = Px / (lam + np.dot(x_vec.T, Px))
        w = w + (k.flatten() * e)
        P = (P - np.dot(k, (x_vec.T @ P))) / lam
        filtered_signal[n] = y
    filtered_signal[:filter_order] = signal[:filter_order]
    return filtered_signal

def spectral_subtraction_filter(signal, sr, noise_estimation_duration=0.5):
    """
    Applies spectral subtraction with adaptive noise estimation.
    (Useful for non-stationary noise in noisy/chaotic signals.)
    """
    noise_samples = int(noise_estimation_duration * sr)
    noise_profile = np.mean(np.abs(librosa.stft(signal[:noise_samples])), axis=1)
    S = librosa.stft(signal)
    magnitude, phase = np.abs(S), np.angle(S)
    new_magnitude = np.maximum(magnitude - noise_profile[:, None], 0.0)
    S_filtered = new_magnitude * np.exp(1j * phase)
    filtered_signal = librosa.istft(S_filtered)
    return filtered_signal

def dynamic_filter(signal, sr, frame_length=1024, hop_length=512):
    """
    Performs dynamic filtering on a per-frame basis.
    For each frame, the filtering is chosen based on local features.
    (This handles hybrid/mixed signals, e.g., music or footsteps.)
    """
    num_frames = int(np.ceil((len(signal) - frame_length) / hop_length)) + 1
    filtered_signal = np.zeros(len(signal) + frame_length)
    window = np.hanning(frame_length)
    normalization = np.zeros(len(signal) + frame_length)
    
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = signal[start:end]
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)), mode='constant')
        # Compute local Short-Time Energy (STE)
        ste = np.sum(frame**2)
        # A simple decision based on STE (you can add spectral flux/ZCR for finer control)
        if ste > 0.02:
            # Likely impulse-like frame: use wavelet thresholding + adaptive LMS
            frame_filtered = adaptive_lms_filter(wavelet_threshold_filter(frame))
        elif ste < 0.01:
            # Likely smooth frame: use adaptive Wiener + Kalman
            frame_filtered = kalman_filter(adaptive_wiener_filter(frame))
        else:
            # In-between: blend both approaches
            impulse_filtered = adaptive_lms_filter(wavelet_threshold_filter(frame))
            smooth_filtered = kalman_filter(adaptive_wiener_filter(frame))
            frame_filtered = 0.5 * (impulse_filtered + smooth_filtered)
        # Overlap-add with windowing
        filtered_signal[start:end] += frame_filtered * window
        normalization[start:end] += window
    normalized_signal = filtered_signal[:len(signal)] / (normalization[:len(signal)] + 1e-8)
    return normalized_signal

# ----------------------- Main Processing Function -----------------------

def process_audio(file_path):
    # Load audio file
    y, sr = librosa.load(file_path, sr=None)
    
    # Baseline noise reduction
    y_denoised = nr.reduce_noise(y=y, sr=sr)
    
    # ----------------------- Feature Extraction -----------------------
    
    # 1. Wavelet Transform Coefficients (5-level DWT using db4)
    wavelet_coeffs = pywt.wavedec(y_denoised, 'db4', level=5)
    
    # 2. Spectral Flux: computed from STFT frame differences
    S = np.abs(librosa.stft(y_denoised))
    flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
    avg_spectral_flux = np.mean(flux)
    
    # 3. Spectral Centroid
    spec_centroid = librosa.feature.spectral_centroid(y=y_denoised, sr=sr)
    avg_spectral_centroid = np.mean(spec_centroid)
    
    # 4. Spectral Kurtosis: using the flattened magnitude spectrogram
    spec_kurtosis = kurtosis(S.flatten())
    
    # 5. Short-Time Energy (STE)
    frame_length = 1024
    hop_length = 512
    frames = librosa.util.frame(y_denoised, frame_length=frame_length, hop_length=hop_length)
    ste = np.sum(frames**2, axis=0)
    avg_ste = np.mean(ste)
    
    # 6. Zero Crossing Rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y_denoised, frame_length=frame_length, hop_length=hop_length)
    avg_zcr = np.mean(zcr)
    
    # Crest Factor: max amplitude divided by RMS
    rms = np.sqrt(np.mean(y_denoised**2))
    crest_factor = np.max(np.abs(y_denoised)) / (rms + 1e-8)
    
    # Debug prints for feature values
    print("Average STE:", avg_ste)
    print("Average Spectral Flux:", avg_spectral_flux)
    print("Average Spectral Centroid:", avg_spectral_centroid)
    print("Spectral Kurtosis:", spec_kurtosis)
    print("Average ZCR:", avg_zcr)
    print("Crest Factor:", crest_factor)

    analysis_dir = "/mnt/c/Users/eqtng/Dropbox/SFSU/697/analysis_results"
    os.makedirs(analysis_dir, exist_ok=True)

    analysis_file_path = os.path.join(analysis_dir, "analysis_results.txt")

    with open("analysis_results.txt", "w") as f:
        f.write(f"Average STE: {avg_ste}\n")
        f.write(f"Average Spectral Flux: {avg_spectral_flux}\n")
        f.write(f"Average Spectral Centroid: {avg_spectral_centroid}\n")
        f.write(f"Spectral Kurtosis: {spec_kurtosis}\n")
        f.write(f"Average ZCR: {avg_zcr}\n")
        f.write(f"Crest Factor: {crest_factor}\n")

    
    # ----------------------- Categorization -----------------------
    # Using threshold criteria per your table:
    if avg_ste > 0.02 and spec_kurtosis > 3 and crest_factor > 5:
        category = "Impulse-Like"
    elif avg_ste < 0.01 and avg_spectral_flux < 0.5 and avg_zcr < 0.1:
        category = "Smooth/Tonal"
    elif avg_spectral_flux > 1.0 and avg_zcr > 0.15 and (3 < crest_factor < 5):
        category = "Noisy/Chaotic"
    else:
        category = "Hybrid (Mixed)"
    
    print("Audio Category:", category)
    
    # ----------------------- Filtering Chain Selection -----------------------
    if category == "Impulse-Like":
        # Chain: Wavelet Thresholding then Adaptive LMS
        filtered_audio = adaptive_lms_filter(wavelet_threshold_filter(y_denoised))
        filter_used = "Wavelet Thresholding + Adaptive LMS"
    elif category == "Smooth/Tonal":
        # Chain: Adaptive Wiener Filter then Kalman Filter
        filtered_audio = kalman_filter(adaptive_wiener_filter(y_denoised))
        filter_used = "Adaptive Wiener Filter + Kalman Filter"
    elif category == "Noisy/Chaotic":
        # Chain: Spectral Subtraction then RLS Filter
        filtered_audio = rls_filter(spectral_subtraction_filter(y_denoised, sr))
        filter_used = "Spectral Subtraction + RLS"
    elif category == "Hybrid (Mixed)":
        # Chain: Dynamic filtering based on detected features
        filtered_audio = dynamic_filter(y_denoised, sr)
        filter_used = "Dynamic Filtering Based on Detected Features"
    else:
        # Fallback (should not occur)
        filtered_audio = y_denoised
        filter_used = "None (Fallback)"
    
    print("Filter applied:", filter_used)
    
    # Save filtered audio to output folder
    output_dir = r"/mnt/c/Users/eqtng/Dropbox/SFSU/697/audio clips/filtered output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "filtered_output.wav")
    sf.write(output_path, filtered_audio, sr)

    return filtered_audio, sr

# ----------------------- Main Execution -----------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python process_audio.py <audio_file_path>")
    else:
        audio_file = sys.argv[1]
        process_audio(audio_file)
 