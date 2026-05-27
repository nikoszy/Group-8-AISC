# =============================================================================
# src/rppg_scorer.py  —  Remote photoplethysmography (rPPG) pulse check
#
# Real human faces contain a faint but measurable periodic signal caused
# by haemoglobin absorption changes during the cardiac cycle.  Under normal
# lighting, the green channel of a face video carries this pulse at roughly
# 0.7–4 Hz (42–240 BPM).
#
# Deepfakes generated frame-by-frame (GAN/diffusion) destroy this signal:
# the generator does not model the subtle blood-flow colour variation, so
# the output signal is dominated by noise with no coherent periodic component.
#
# This module:
#   1. Extracts the mean green-channel value from a forehead/cheek ROI
#      across all frames with a detected face
#   2. Bandpass-filters the signal to 0.7–4 Hz using scipy
#   3. Computes the signal-to-noise ratio (SNR) of the filtered signal
#   4. Returns a coherence score:  HIGH = clear pulse = REAL signal
#                                  LOW  = no pulse    = FAKE signal
#   5. The score is inverted for use as a fakeness indicator:
#      rppg_fake_score = 1 - coherence_score
#
# Requirements:
#   scipy >= 1.0  (already in requirements.txt)
#
# Limitations:
#   - Needs >= 3 seconds of video (fps * 3 frames with detected face)
#   - Assumes constant or slowly-varying illumination
#   - Does not work on grainy / highly compressed footage
# =============================================================================

import numpy as np

# Minimum frames required for reliable rPPG analysis
_MIN_FRAMES_FOR_RPPG = 30   # ~1 s at 30 fps; absolute minimum
_DEFAULT_FPS         = 30.0

# Bandpass limits for human heart rate (42–240 BPM)
_LOW_HZ  = 0.7
_HIGH_HZ = 4.0

# ROI: central portion of the face crop (avoid hair, ears, background)
# Values are fractions of the crop dimensions
_ROI_TOP    = 0.10   # skip forehead hairline
_ROI_BOTTOM = 0.55   # stop above mouth
_ROI_LEFT   = 0.20   # skip ear region
_ROI_RIGHT  = 0.80


def _extract_roi_signal(face_crops_bgr):
    """
    Extract mean green-channel value from the central face ROI per frame.

    Args:
        face_crops_bgr: list of (H × W × 3) BGR arrays

    Returns:
        numpy array of shape (N,) — green-channel mean per frame
    """
    signal = []
    for crop in face_crops_bgr:
        if crop is None or crop.size == 0:
            continue
        h, w = crop.shape[:2]
        t = int(h * _ROI_TOP)
        b = int(h * _ROI_BOTTOM)
        l = int(w * _ROI_LEFT)
        r = int(w * _ROI_RIGHT)
        roi = crop[t:b, l:r, 1]   # green channel (index 1 in BGR)
        signal.append(float(np.mean(roi)))
    return np.array(signal, dtype=np.float64)


def _bandpass_filter(signal, fps, low_hz=_LOW_HZ, high_hz=_HIGH_HZ):
    """
    Butterworth bandpass filter using scipy.signal.

    Returns filtered signal, or None if scipy unavailable / signal too short.
    """
    try:
        from scipy.signal import butter, filtfilt
    except ImportError:
        return None

    nyq = fps / 2.0
    low  = low_hz  / nyq
    high = high_hz / nyq

    # Clamp to valid range
    low  = float(np.clip(low,  1e-4, 0.99))
    high = float(np.clip(high, 1e-4, 0.99))
    if low >= high:
        return None

    try:
        b, a = butter(4, [low, high], btype="band")
        filtered = filtfilt(b, a, signal)
        return filtered
    except Exception:
        return None


def _compute_snr(raw_signal, filtered_signal):
    """
    Signal-to-noise ratio: power of the filtered pulse signal vs residual noise.

    SNR = 10 * log10(P_signal / P_noise)

    Real face: SNR >= 3 dB (coherent pulse visible)
    Deepfake:  SNR < 0 dB  (pulse buried in noise)
    """
    noise = raw_signal - filtered_signal
    p_signal = float(np.mean(filtered_signal ** 2))
    p_noise  = float(np.mean(noise ** 2))
    if p_noise < 1e-12:
        return 0.0
    snr_db = 10.0 * np.log10(max(p_signal, 1e-12) / p_noise)
    return float(snr_db)


def _compute_peak_coherence(filtered_signal, fps):
    """
    Estimate how coherent/periodic the filtered signal is.

    Uses the ratio of the dominant frequency peak power to total power
    in the bandpass range.  A clear heartbeat produces a sharp peak.

    Returns float in [0, 1].  1.0 = perfect single-frequency signal.
    """
    try:
        from scipy.fft import rfft, rfftfreq
    except ImportError:
        try:
            from numpy.fft import rfft, rfftfreq
        except ImportError:
            return 0.5

    n = len(filtered_signal)
    freqs = rfftfreq(n, d=1.0 / fps)
    power = np.abs(rfft(filtered_signal)) ** 2

    mask = (freqs >= _LOW_HZ) & (freqs <= _HIGH_HZ)
    if not mask.any():
        return 0.5

    in_band_power  = power[mask]
    total_in_band  = float(in_band_power.sum())
    if total_in_band < 1e-12:
        return 0.5

    peak_power = float(in_band_power.max())
    coherence  = peak_power / total_in_band
    return float(np.clip(coherence, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rppg_score(face_crops_bgr, fps=None):
    """
    Compute an rPPG-based liveness score from a sequence of face crops.

    Args:
        face_crops_bgr : list of BGR arrays — one face crop per frame
                         (crops should be aligned to the same face region)
        fps            : frames-per-second of the source video.  If None,
                         assumed 30.0.

    Returns:
        dict with:
            "fake_score"  : float [0,1] — HIGH = no pulse detected = more fake
            "coherence"   : float [0,1] — periodic signal strength
            "snr_db"      : float       — signal-to-noise ratio
            "n_frames"    : int         — number of valid frames analysed
            "available"   : bool        — False if too few frames or scipy missing
            "note"        : str         — human-readable explanation
    """
    fps = float(fps) if fps and fps > 0 else _DEFAULT_FPS

    # Filter out None crops
    valid_crops = [c for c in face_crops_bgr if c is not None and c.size > 0]
    n = len(valid_crops)

    if n < _MIN_FRAMES_FOR_RPPG:
        return {
            "fake_score": 0.5, "coherence": 0.5, "snr_db": 0.0,
            "n_frames": n, "available": False,
            "note": f"need >= {_MIN_FRAMES_FOR_RPPG} face frames, got {n}",
        }

    # 1. Extract ROI green signal
    raw = _extract_roi_signal(valid_crops)
    if len(raw) < _MIN_FRAMES_FOR_RPPG:
        return {
            "fake_score": 0.5, "coherence": 0.5, "snr_db": 0.0,
            "n_frames": len(raw), "available": False,
            "note": "insufficient non-empty crops",
        }

    # 2. Detrend (remove slow drift)
    raw = raw - np.mean(raw)

    # 3. Bandpass filter
    filtered = _bandpass_filter(raw, fps)
    if filtered is None:
        return {
            "fake_score": 0.5, "coherence": 0.5, "snr_db": 0.0,
            "n_frames": len(raw), "available": False,
            "note": "bandpass filter failed (scipy issue or signal too short)",
        }

    # 4. Metrics
    snr_db    = _compute_snr(raw, filtered)
    coherence = _compute_peak_coherence(filtered, fps)

    # 5. Map to fake_score:
    #    coherence ≈ 0.5-1.0 = clear pulse = REAL → fake_score low
    #    coherence ≈ 0.0-0.2 = no pulse    = FAKE → fake_score high
    #    SNR > 3 dB  = real,  SNR < 0 = fake
    snr_score = float(np.clip(1.0 - (snr_db + 5.0) / 15.0, 0.0, 1.0))
    coh_score = float(np.clip(1.0 - coherence * 1.5, 0.0, 1.0))
    fake_score = round(0.60 * coh_score + 0.40 * snr_score, 4)

    return {
        "fake_score": fake_score,
        "coherence" : round(coherence, 4),
        "snr_db"    : round(snr_db, 2),
        "n_frames"  : len(raw),
        "available" : True,
        "note"      : f"pulse coherence={coherence:.3f}, SNR={snr_db:.1f} dB",
    }


def rppg_available():
    """Return True if scipy is importable (always True in this project)."""
    try:
        import scipy.signal  # noqa: F401
        return True
    except ImportError:
        return False
