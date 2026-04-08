"""
Feature extraction for AI music detection.
Uses librosa for professional-grade audio analysis.
"""

import numpy as np
# librosa is imported lazily inside functions to reduce startup memory


def extract_features(audio_path: str, sr: int = 22050, duration: float = 30) -> dict:
    import librosa
    """
    Extract 80+ audio features optimized for AI music detection.
    Returns a flat dictionary of named features.
    """
    y, sr = librosa.load(audio_path, sr=sr, duration=duration, mono=False)

    is_stereo = y.ndim == 2
    if is_stereo:
        y_left, y_right = y[0], y[1]
        y_mono = librosa.to_mono(y)
    else:
        y_mono = y
        y_left = y_right = None

    features = {}

    # ── 1. RMS & Dynamic Range ──
    rms = librosa.feature.rms(y=y_mono)[0]
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))
    features["rms_cv"] = float(np.std(rms) / (np.mean(rms) + 1e-10))
    features["dynamic_range_db"] = float(
        20 * np.log10(np.max(np.abs(y_mono)) / (np.sqrt(np.mean(y_mono**2)) + 1e-10))
    )

    # ── 2. MFCC (13 coefficients) ──
    mfcc = librosa.feature.mfcc(y=y_mono, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc_{i}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i}_std"] = float(np.std(mfcc[i]))

    # ── 3. Delta-MFCC (rate of timbre change) ──
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    for i in range(13):
        features[f"delta_mfcc_{i}_std"] = float(np.std(delta_mfcc[i]))
        features[f"delta2_mfcc_{i}_std"] = float(np.std(delta2_mfcc[i]))

    # ── 4. Spectral features ──
    cent = librosa.feature.spectral_centroid(y=y_mono, sr=sr)[0]
    features["spectral_centroid_mean"] = float(np.mean(cent))
    features["spectral_centroid_std"] = float(np.std(cent))

    bw = librosa.feature.spectral_bandwidth(y=y_mono, sr=sr)[0]
    features["spectral_bandwidth_mean"] = float(np.mean(bw))
    features["spectral_bandwidth_std"] = float(np.std(bw))

    rolloff = librosa.feature.spectral_rolloff(y=y_mono, sr=sr)[0]
    features["spectral_rolloff_mean"] = float(np.mean(rolloff))

    flat = librosa.feature.spectral_flatness(y=y_mono)[0]
    features["spectral_flatness_mean"] = float(np.mean(flat))
    features["spectral_flatness_std"] = float(np.std(flat))

    # ── 5. Spectral contrast (7 frequency bands) ──
    contrast = librosa.feature.spectral_contrast(y=y_mono, sr=sr)
    for i in range(min(7, contrast.shape[0])):
        features[f"spectral_contrast_{i}_mean"] = float(np.mean(contrast[i]))
        features[f"spectral_contrast_{i}_std"] = float(np.std(contrast[i]))

    # ── 6. Chroma features ──
    chroma = librosa.feature.chroma_stft(y=y_mono, sr=sr)
    features["chroma_mean"] = float(np.mean(chroma))
    features["chroma_std"] = float(np.std(chroma))

    # ── 7. Zero crossing rate ──
    zcr = librosa.feature.zero_crossing_rate(y_mono)[0]
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))

    # ── 8. Tempo & beat regularity ──
    tempo, beat_frames = librosa.beat.beat_track(y=y_mono, sr=sr)
    features["tempo"] = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    if len(beat_frames) > 2:
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        ibi = np.diff(beat_times)
        features["beat_regularity_cv"] = float(np.std(ibi) / (np.mean(ibi) + 1e-10))
        features["beat_ibi_mean"] = float(np.mean(ibi))
        features["beat_ibi_std"] = float(np.std(ibi))
    else:
        features["beat_regularity_cv"] = 0.5
        features["beat_ibi_mean"] = 0.0
        features["beat_ibi_std"] = 0.0

    # ── 9. Onset analysis ──
    onset_env = librosa.onset.onset_strength(y=y_mono, sr=sr)
    features["onset_strength_mean"] = float(np.mean(onset_env))
    features["onset_strength_std"] = float(np.std(onset_env))
    onsets = librosa.onset.onset_detect(y=y_mono, sr=sr, units="time")
    features["onset_count"] = len(onsets)
    if len(onsets) > 2:
        ioi = np.diff(onsets)
        features["onset_ioi_cv"] = float(np.std(ioi) / (np.mean(ioi) + 1e-10))
    else:
        features["onset_ioi_cv"] = 0.5

    # ── 10. SNR & noise floor ──
    noise_floor = float(np.percentile(np.abs(y_mono), 5))
    rms_val = float(np.sqrt(np.mean(y_mono**2)))
    features["snr_db"] = float(20 * np.log10(rms_val / (noise_floor + 1e-10)))
    features["noise_floor"] = noise_floor

    # ── 11. Stereo analysis ──
    features["is_stereo"] = is_stereo
    if is_stereo and y_left is not None:
        win = 4096
        corrs = []
        for i in range(0, min(len(y_left), len(y_right)) - win, win):
            c = np.corrcoef(y_left[i : i + win], y_right[i : i + win])[0, 1]
            if not np.isnan(c):
                corrs.append(abs(c))
        if corrs:
            features["stereo_corr_mean"] = float(np.mean(corrs))
            features["stereo_corr_std"] = float(np.std(corrs))
        else:
            features["stereo_corr_mean"] = 0.5
            features["stereo_corr_std"] = 0.1
    else:
        features["stereo_corr_mean"] = 0.5
        features["stereo_corr_std"] = 0.1

    # ── 12. Harmonic / Percussive ratio ──
    y_harm, y_perc = librosa.effects.hpss(y_mono)
    harm_e = float(np.mean(np.abs(y_harm)))
    perc_e = float(np.mean(np.abs(y_perc)))
    features["harmonic_ratio"] = harm_e / (harm_e + perc_e + 1e-10)
    features["percussive_ratio"] = perc_e / (harm_e + perc_e + 1e-10)

    # ── 13. Tonnetz (tonal centroid) ──
    tonnetz = librosa.feature.tonnetz(y=y_mono, sr=sr)
    for i in range(6):
        features[f"tonnetz_{i}_mean"] = float(np.mean(tonnetz[i]))

    # ── 14. Spectral flux ──
    S = np.abs(librosa.stft(y_mono))
    flux = np.sqrt(np.mean(np.diff(S, axis=1) ** 2, axis=0))
    features["spectral_flux_mean"] = float(np.mean(flux))
    features["spectral_flux_std"] = float(np.std(flux))

    return features


def extract_visuals(audio_path: str, sr: int = 22050, duration: float = 30,
                    n_waveform: int = 300, n_spectrum: int = 512) -> dict:
    import librosa
    """
    Extract waveform envelope and spectrum data for frontend visualisation.
    Returns small arrays suitable for JSON transport.
    """
    y, sr = librosa.load(audio_path, sr=sr, duration=duration, mono=True)

    # Waveform: RMS envelope (n_waveform points, values in [-1, 1])
    hop = max(1, len(y) // n_waveform)
    waveform = []
    for i in range(n_waveform):
        chunk = y[i * hop: i * hop + hop]
        if len(chunk) == 0:
            waveform.append(0.0)
        else:
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            # Alternate sign for visual symmetry (mimic oscilloscope look)
            waveform.append(round(rms, 4))

    # Spectrum: FFT magnitude (n_spectrum bins, log-scaled)
    fft_size = 4096
    start = len(y) // 5
    frame = y[start: start + fft_size]
    if len(frame) < fft_size:
        frame = np.pad(frame, (0, fft_size - len(frame)))
    window = np.hanning(fft_size)
    mag = np.abs(np.fft.rfft(frame * window))
    mag = mag[:n_spectrum]
    mag_max = np.max(mag) or 1.0
    spectrum = [round(float(v / mag_max), 4) for v in mag]

    return {
        "waveform": waveform,
        "spectrum": spectrum,
        "sample_rate": sr,
    }


def extract_mel_spectrogram(
    audio_path: str, sr: int = 22050, duration: float = 10,
    n_mels: int = 128, fixed_length: int = 216
) -> np.ndarray:
    import librosa
    """
    Extract normalized mel spectrogram for CNN model.
    Returns array of shape (n_mels, fixed_length).
    """
    y, sr = librosa.load(audio_path, sr=sr, duration=duration, mono=True)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize to zero mean, unit variance
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-10)

    # Pad or truncate to fixed_length
    if mel_db.shape[1] < fixed_length:
        mel_db = np.pad(mel_db, ((0, 0), (0, fixed_length - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :fixed_length]

    return mel_db


def get_feature_names() -> list[str]:
    """Return ordered list of numeric feature names used by the model."""
    names = []
    # RMS
    names += ["rms_mean", "rms_std", "rms_cv", "dynamic_range_db"]
    # MFCC
    for i in range(13):
        names += [f"mfcc_{i}_mean", f"mfcc_{i}_std"]
    # Delta-MFCC
    for i in range(13):
        names += [f"delta_mfcc_{i}_std", f"delta2_mfcc_{i}_std"]
    # Spectral
    names += [
        "spectral_centroid_mean", "spectral_centroid_std",
        "spectral_bandwidth_mean", "spectral_bandwidth_std",
        "spectral_rolloff_mean",
        "spectral_flatness_mean", "spectral_flatness_std",
    ]
    # Spectral contrast
    for i in range(7):
        names += [f"spectral_contrast_{i}_mean", f"spectral_contrast_{i}_std"]
    # Chroma
    names += ["chroma_mean", "chroma_std"]
    # ZCR
    names += ["zcr_mean", "zcr_std"]
    # Beat
    names += ["tempo", "beat_regularity_cv", "beat_ibi_mean", "beat_ibi_std"]
    # Onset
    names += ["onset_strength_mean", "onset_strength_std", "onset_count", "onset_ioi_cv"]
    # SNR
    names += ["snr_db", "noise_floor"]
    # Stereo
    names += ["stereo_corr_mean", "stereo_corr_std"]
    # Harmonic
    names += ["harmonic_ratio", "percussive_ratio"]
    # Tonnetz
    for i in range(6):
        names += [f"tonnetz_{i}_mean"]
    # Flux
    names += ["spectral_flux_mean", "spectral_flux_std"]

    return names
