import numpy as np
import librosa
import noisereduce as nr
import pywt
from scipy.stats import kurtosis
from scipy.signal import wiener
import soundfile as sf
import os

# ----------------------- High-End Blending -----------------------

def preserve_high_end_blend(original, filtered, sr, cutoff_freq=5000, alpha=0.7):
    """
    Blend original and filtered high-end: final = α*orig + (1-α)*filt
    for STFT bins ≥ cutoff_freq.
    """
    S_orig = librosa.stft(original)
    S_filt = librosa.stft(filtered)
    freqs = librosa.fft_frequencies(sr=sr)

    mask = freqs >= cutoff_freq
    mag_orig, phase_orig = np.abs(S_orig), np.angle(S_orig)
    mag_filt, phase_filt = np.abs(S_filt), np.angle(S_filt)

    # blend magnitudes above cutoff, keep filtered phase
    mag_final = mag_filt.copy()
    mag_final[mask, :] = alpha * mag_orig[mask, :] + (1 - alpha) * mag_filt[mask, :]

    S_final = mag_final * np.exp(1j * phase_filt)
    return librosa.istft(S_final)

# ----------------------- Filtering Functions -----------------------

def wavelet_threshold_filter(signal, wavelet='db4', level=5, thresh_scale=0.5):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    base_thresh = np.median(np.abs(coeffs[-1]))
    scaled = []
    for i, c in enumerate(coeffs):
        t = base_thresh * thresh_scale if i == len(coeffs)-1 else base_thresh
        scaled.append(pywt.threshold(c, t, mode='soft'))
    return pywt.waverec(scaled, wavelet)

def adaptive_lms_filter(signal, mu=0.005, filter_order=4):
    N = len(signal)
    w = np.zeros(filter_order)
    out = np.zeros(N)
    for n in range(filter_order, N):
        x = signal[n-filter_order:n][::-1]
        y = w.dot(x)
        e = signal[n] - y
        w += 2 * mu * e * x
        out[n] = y
    out[:filter_order] = signal[:filter_order]
    return out

def adaptive_wiener_filter(signal, mysize=7):
    return wiener(signal, mysize=mysize)

def kalman_filter(signal, process_variance=1e-6, measurement_variance=0.02):
    n = len(signal)
    xhat = np.zeros(n)
    P = np.zeros(n)
    xhat[0], P[0] = signal[0], 1.0
    Q, R = process_variance, measurement_variance
    for k in range(1, n):
        xhat_minus = xhat[k-1]
        P_minus = P[k-1] + Q
        K = P_minus / (P_minus + R)
        xhat[k] = xhat_minus + K*(signal[k] - xhat_minus)
        P[k] = (1 - K)*P_minus
    return xhat

def rls_filter(signal, lam=0.995, delta=0.1, filter_order=4):
    N = len(signal)
    w = np.zeros(filter_order)
    P = np.eye(filter_order) / delta
    out = np.zeros(N)
    for n in range(filter_order, N):
        x = signal[n-filter_order:n][::-1].reshape(-1,1)
        y = w.dot(x).item()
        e = signal[n] - y
        Px = P.dot(x)
        k = Px / (lam + x.T.dot(Px))
        w += (k.flatten() * e)
        P = (P - k.dot(x.T.dot(P))) / lam
        out[n] = y
    out[:filter_order] = signal[:filter_order]
    return out

def spectral_subtraction_filter(signal, sr, noise_estimation_duration=0.5, sub_scale=0.7):
    noise_samples = int(noise_estimation_duration * sr)
    noise_profile = np.mean(np.abs(librosa.stft(signal[:noise_samples])), axis=1)
    S = librosa.stft(signal)
    mag, phase = np.abs(S), np.angle(S)
    new_mag = np.maximum(mag - sub_scale*noise_profile[:,None], 0.0)
    return librosa.istft(new_mag * np.exp(1j*phase))

def dynamic_filter(signal, sr, frame_length=1024, hop_length=512):
    num_frames = int(np.ceil((len(signal) - frame_length) / hop_length)) + 1
    out = np.zeros(len(signal)+frame_length)
    win = np.hanning(frame_length)
    norm = np.zeros_like(out)
    for i in range(num_frames):
        s, e = i*hop_length, i*hop_length+frame_length
        frame = signal[s:e]
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length-len(frame)), mode='constant')
        ste = np.sum(frame**2)
        if ste > 0.02:
            f = adaptive_lms_filter(wavelet_threshold_filter(frame))
        elif ste < 0.01:
            f = kalman_filter(adaptive_wiener_filter(frame))
        else:
            imp = adaptive_lms_filter(wavelet_threshold_filter(frame))
            sm  = kalman_filter(adaptive_wiener_filter(frame))
            f = 0.5*(imp + sm)
        out[s:e] += f * win
        norm[s:e] += win
    return out[:len(signal)] / (norm[:len(signal)] + 1e-8)

# ----------------------- Main Processing Function -----------------------
def process_audio(file_path, cutoff_freq=5000, alpha=0.7):
    # 1. Load & denoise
    y, sr = librosa.load(file_path, sr=None)
    y_denoised = nr.reduce_noise(y=y, sr=sr)

    # 2. Feature extraction
    wavelet_coeffs = pywt.wavedec(y_denoised, 'db4', level=5)
    S = np.abs(librosa.stft(y_denoised))
    flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
    avg_spectral_flux = np.mean(flux)
    spec_centroid = librosa.feature.spectral_centroid(y=y_denoised, sr=sr)
    avg_spectral_centroid = np.mean(spec_centroid)
    spec_kurtosis = kurtosis(S.flatten())
    frame_length, hop_length = 1024, 512
    frames = librosa.util.frame(y_denoised, frame_length=frame_length, hop_length=hop_length)
    ste = np.sum(frames**2, axis=0)
    avg_ste = np.mean(ste)
    zcr = librosa.feature.zero_crossing_rate(y_denoised,
                                             frame_length=frame_length,
                                             hop_length=hop_length)
    avg_zcr = np.mean(zcr)
    rms = np.sqrt(np.mean(y_denoised**2))
    crest_factor = np.max(np.abs(y_denoised)) / (rms + 1e-8)

    # 3. Categorization
    if avg_ste > 0.02 and spec_kurtosis > 3 and crest_factor > 5:
        category = "Impulse-Like"
    elif avg_ste < 0.01 and avg_spectral_flux < 0.5 and avg_zcr < 0.1:
        category = "Smooth/Tonal"
    elif avg_spectral_flux > 1.0 and avg_zcr > 0.15 and (3 < crest_factor < 5):
        category = "Noisy/Chaotic"
    else:
        category = "Hybrid (Mixed)"
    print("Audio Category:", category)

    # 4. Apply the filtering chain based on category
    if category == "Impulse-Like":
        filtered = adaptive_lms_filter(wavelet_threshold_filter(y_denoised))
    elif category == "Smooth/Tonal":
        filtered = kalman_filter(adaptive_wiener_filter(y_denoised))
    elif category == "Noisy/Chaotic":
        filtered = rls_filter(spectral_subtraction_filter(y_denoised, sr))
    else:  # Hybrid (Mixed)
        filtered = dynamic_filter(y_denoised, sr)

    # 5. Blend high-end back in
    filtered = preserve_high_end_blend(
        original=y_denoised,
        filtered=filtered,
        sr=sr,
        cutoff_freq=cutoff_freq,
        alpha=alpha
    )

    # 6. Save output
    output_path = "filtered_output.wav"
    sf.write(output_path, filtered, sr)
    print(f"Saved filtered audio (with high-end blend) to {output_path}")

    return filtered, sr


# ----------------------- Main Execution -----------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python process_audio.py <audio_file_path>")
    else:
        process_audio(sys.argv[1])
