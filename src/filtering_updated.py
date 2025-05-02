import numpy as np
import librosa
import noisereduce as nr
import pywt
from scipy.stats import kurtosis
from scipy.signal import wiener
import soundfile as sf
import os

# ——— Parameter Definitions ———

# Categorization thresholds
STE_THRESHOLD_IMPULSE   = 0.02   # Higher → only very loud transients count as “impulse-like”
STE_THRESHOLD_SMOOTH    = 0.01   # Lower  → even quieter segments can still be “smooth/tonal”
KURTOSIS_THRESHOLD      = 3.0    # Higher → only very peaky spectra (transients) qualify as impulses
CREST_THRESHOLD         = 5.0    # Higher → only signals with large peak/RMS ratios qualify as impulses
FLUX_THRESHOLD_SMOOTH   = 0.5    # Lower  → only very stable spectra count as smooth/tonal
FLUX_THRESHOLD_NOISY    = 1.0    # Higher → only rapidly changing spectra count as noisy/chaotic
ZCR_THRESHOLD_SMOOTH    = 0.10   # Lower  → fewer zero-crossings required to be “smooth/tonal”
ZCR_THRESHOLD_NOISY     = 0.15   # Higher → more zero-crossings required to be “noisy/chaotic”

# STFT settings
FRAME_LENGTH = 1024   # Larger → better freq. resolution, worse time resolution
HOP_LENGTH   = 512    # Larger → more overlap (smoother features), higher computation

# Wavelet denoising
WAVELET       = 'db4'  # Choice of wavelet—affects how transients vs. tonal parts are separated
WAVELET_LEVEL = 4      # Higher → coarser approximation, more aggressive denoising
THRESH_SCALE  = 0.8    # Closer to 1.0 → gentler thresholding; Closer to 0 → more detail removed

# Adaptive LMS
LMS_MU    = 0.002  # Step size: lower → slower adaptation, less distortion; higher → faster noise removal, more risk
LMS_ORDER = 3      # Filter order: higher → can model more complex noise, may remove desired signal

# Wiener
WIENER_MYSIZE = 3  # Window size: smaller → preserves more short-term detail; larger → smoother but more detail lost

# Kalman
KALMAN_Q = 1e-5    # Process variance: lower → assumes more stable signal (smoother), higher → more responsive (noisier)
KALMAN_R = 0.005   # Measurement variance: lower → trusts measurements more (less smoothing), higher → trusts model (more smoothing)

# Spectral Subtraction
NOISE_EST_DUR = 0.3  # Seconds of initial noise-only audio for noise profile; longer → more accurate noise estimate
SUB_SCALE     = 0.5  # Subtraction scale: closer to 1.0 → full subtraction (more noise removed, risk of artifacts), closer to 0 → gentler subtraction

# Recursive Least Squares (RLS)
RLS_LAMBDA = 0.997  # Forgetting factor: closer to 1.0 → slower adaptation (less distortion), lower → faster but riskier adaptation
RLS_DELTA  = 0.01   # Initial diagonal loading: lower → more aggressive weight updates, higher → more conservative
RLS_ORDER  = 3      # Filter order: higher → can cancel more complex noise, may remove desired signal

# High-end blending
CUTOFF_FREQ = 4000   # Hz: frequencies above this are blended back from original; lower → preserves more mid-high detail
BLEND_ALPHA = 0.9    # Blend factor: 1.0 = all original highs, 0.0 = all filtered highs; adjust to balance clarity vs. noise reduction
# ——— High-End Blend Helper ———

def preserve_high_end_blend(original, filtered, sr,
                            cutoff_freq=CUTOFF_FREQ,
                            alpha=BLEND_ALPHA):
    """
    Blend back the top end of the spectrum from the original signal
    into the filtered signal.
    """
    S_orig = librosa.stft(original)
    S_filt = librosa.stft(filtered)
    freqs  = librosa.fft_frequencies(sr=sr)
    mask   = freqs >= cutoff_freq

    mag_orig   = np.abs(S_orig)
    mag_filt   = np.abs(S_filt)
    phase_filt = np.angle(S_filt)

    mag_final = mag_filt.copy()
    mag_final[mask, :] = alpha * mag_orig[mask, :] + (1 - alpha) * mag_filt[mask, :]

    S_final = mag_final * np.exp(1j * phase_filt)
    return librosa.istft(S_final)

# ——— Filter Functions ———

def wavelet_threshold_filter(sig):
    coeffs      = pywt.wavedec(sig, WAVELET, level=WAVELET_LEVEL)
    base_thresh = np.median(np.abs(coeffs[-1]))
    new_coeffs  = [
        pywt.threshold(c,
                       base_thresh * (THRESH_SCALE if i == len(coeffs)-1 else 1.0),
                       mode='soft')
        for i, c in enumerate(coeffs)
    ]
    return pywt.waverec(new_coeffs, WAVELET)

def adaptive_lms_filter(sig):
    N   = len(sig)
    w   = np.zeros(LMS_ORDER)
    out = np.zeros(N)
    for n in range(LMS_ORDER, N):
        x = sig[n-LMS_ORDER:n][::-1]
        y = w.dot(x)
        e = sig[n] - y
        w += 2 * LMS_MU * e * x
        out[n] = y
    out[:LMS_ORDER] = sig[:LMS_ORDER]
    return out

def adaptive_wiener_filter(sig):
    return wiener(sig, mysize=WIENER_MYSIZE)

def kalman_filter(sig):
    n    = len(sig)
    xhat = np.zeros(n)
    P    = np.zeros(n)
    xhat[0], P[0] = sig[0], 1.0
    for k in range(1, n):
        P_minus    = P[k-1] + KALMAN_Q
        Kf         = P_minus / (P_minus + KALMAN_R)
        xhat[k]    = xhat[k-1] + Kf * (sig[k] - xhat[k-1])
        P[k]       = (1 - Kf) * P_minus
    return xhat

def spectral_subtraction_filter(sig, sr):
    noise_samples = int(NOISE_EST_DUR * sr)
    noise_prof    = np.mean(np.abs(librosa.stft(sig[:noise_samples])), axis=1)
    S             = librosa.stft(sig)
    mag, phase    = np.abs(S), np.angle(S)
    new_mag       = np.maximum(mag - SUB_SCALE * noise_prof[:, None], 0.0)
    return librosa.istft(new_mag * np.exp(1j * phase))

def rls_filter(sig):
    N   = len(sig)
    w   = np.zeros(RLS_ORDER)
    P   = np.eye(RLS_ORDER) / RLS_DELTA
    out = np.zeros(N)
    for n in range(RLS_ORDER, N):
        x   = sig[n-RLS_ORDER:n][::-1].reshape(-1,1)
        y   = w.dot(x).item()
        e   = sig[n] - y
        Px  = P.dot(x)
        k   = Px / (RLS_LAMBDA + x.T.dot(Px))
        w  += (k.flatten() * e)
        P   = (P - k.dot(x.T.dot(P))) / RLS_LAMBDA
        out[n] = y
    out[:RLS_ORDER] = sig[:RLS_ORDER]
    return out

def dynamic_filter(sig, sr,
                   frame_length=FRAME_LENGTH,
                   hop_length=HOP_LENGTH):
    num_frames = int(np.ceil((len(sig) - frame_length) / hop_length)) + 1
    out        = np.zeros(len(sig) + frame_length)
    win        = np.hanning(frame_length)
    norm       = np.zeros_like(out)
    for i in range(num_frames):
        s, e = i*hop_length, i*hop_length + frame_length
        frame = sig[s:e]
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length-len(frame)), mode='constant')
        ste = np.sum(frame**2)
        if ste > STE_THRESHOLD_IMPULSE:
            f = adaptive_lms_filter(wavelet_threshold_filter(frame))
        elif ste < STE_THRESHOLD_SMOOTH:
            f = kalman_filter(adaptive_wiener_filter(frame))
        else:
            imp = adaptive_lms_filter(wavelet_threshold_filter(frame))
            sm  = kalman_filter(adaptive_wiener_filter(frame))
            f   = 0.5 * (imp + sm)
        out[s:e]   += f * win
        norm[s:e]  += win
    return out[:len(sig)] / (norm[:len(sig)] + 1e-8)

# ——— Main Processing Function ———

def process_audio(path,
                  cutoff_freq=CUTOFF_FREQ,
                  alpha=BLEND_ALPHA):
    # 1. Load & denoise
    y, sr        = librosa.load(path, sr=None)
    y_denoised   = nr.reduce_noise(y=y, sr=sr)

    # 2. Feature extraction
    S            = np.abs(librosa.stft(y_denoised))
    flux         = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
    avg_flux     = np.mean(flux)
    spec_cent    = librosa.feature.spectral_centroid(y=y_denoised, sr=sr)
    avg_cent     = np.mean(spec_cent)
    spec_kurt    = kurtosis(S.flatten())
    frames       = librosa.util.frame(y_denoised,
                                      frame_length=FRAME_LENGTH,
                                      hop_length=HOP_LENGTH)
    ste_vals     = np.sum(frames**2, axis=0)
    avg_ste      = np.mean(ste_vals)
    zcr_vals     = librosa.feature.zero_crossing_rate(y_denoised,
                                                       frame_length=FRAME_LENGTH,
                                                       hop_length=HOP_LENGTH)
    avg_zcr      = np.mean(zcr_vals)
    rms          = np.sqrt(np.mean(y_denoised**2))
    crest_factor = np.max(np.abs(y_denoised)) / (rms + 1e-8)

    # 3. Categorization
    if (avg_ste > STE_THRESHOLD_IMPULSE
        and spec_kurt > KURTOSIS_THRESHOLD
        and crest_factor > CREST_THRESHOLD):
        category = "Impulse-Like"
    elif (avg_ste < STE_THRESHOLD_SMOOTH
          and avg_flux < FLUX_THRESHOLD_SMOOTH
          and avg_zcr < ZCR_THRESHOLD_SMOOTH):
        category = "Smooth/Tonal"
    elif (avg_flux > FLUX_THRESHOLD_NOISY
          and avg_zcr > ZCR_THRESHOLD_NOISY
          and CREST_THRESHOLD*0.6 < crest_factor < CREST_THRESHOLD):
        category = "Noisy/Chaotic"
    else:
        category = "Hybrid (Mixed)"
    print("Audio Category:", category)

    # 4. Apply filtering chain
    if category == "Impulse-Like":
        filtered = adaptive_lms_filter(wavelet_threshold_filter(y_denoised))
    elif category == "Smooth/Tonal":
        filtered = kalman_filter(adaptive_wiener_filter(y_denoised))
    elif category == "Noisy/Chaotic":
        filtered = rls_filter(spectral_subtraction_filter(y_denoised, sr))
    else:
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
    print(f"Saved filtered audio to {output_path}")

    return filtered, sr

# ——— Command-Line Interface ———

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python filtering_updated.py <audio_file_path>")
        sys.exit(1)
    process_audio(sys.argv[1])
