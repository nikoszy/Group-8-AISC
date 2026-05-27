# =============================================================================
# src/freq_analysis/anomaly_scorer.py
# =============================================================================
# Converts a raw face image into a single 0-1 anomaly score using the FFT
# pipeline from fft_extractor.py.
#
# This is the function that ensemble.py calls.
#
# IMPLEMENTATION — power spectrum slope (replaces mean peripheral energy)
# -----------------------------------------------------------------------
# The previous version measured the mean log-magnitude in the high-frequency
# periphery and normalised it with hardcoded constants.  That gave a delta
# of only ~0.012 between real and fake on FF++ C23 because:
#   (a) the single-number mean discards all shape information, and
#   (b) the normalization constants were not tuned for H.264-compressed faces.
#
# The current version measures the SLOPE of the radial power spectrum in
# log-log space instead.  This is the theoretically correct measurement:
#
#   Natural images follow a "1/f² power law":
#       P(f) ∝ f^(-2)
#       log|FFT(f)| ≈ -1 · log(f) + const    (slope ≈ -1)
#
#   FF++ Deepfake autoencoder decoders apply repeated bilinear upsampling +
#   convolution.  These act as low-pass filters, producing faces that are
#   SMOOTHER than real photographs.  Smoother → less high-frequency energy
#   → STEEPER (more-negative) slope.
#
#   Score = how much the fitted slope is steeper than the natural baseline.
#   Higher score = steeper decay = smoother face = more suspicious.
#
# The slope is computed by fitting a line to (log freq, log|FFT|) across
# 40 radial frequency bands, skipping the noisy DC region.  np.polyfit
# (least-squares) is used — no external dependencies.
# =============================================================================

import numpy as np

from src.freq_analysis.fft_extractor import (
    to_grayscale,
    compute_log_magnitude_spectrum,
    compute_radial_power_spectrum,
)
from src.freq_analysis.utils import resize_to_square


# ---------------------------------------------------------------------------
# Calibration constants
# ---------------------------------------------------------------------------
# WHY THE SLOPE DIRECTION MATTERS FOR FF++ DEEPFAKES
# ---------------------------------------------------
# GAN-based deepfakes (StyleGAN, DALL·E, etc.) produce checkerboard upsampling
# artifacts → MORE high-frequency energy → FLATTER spectral slope.
#
# FF++ Deepfakes use an autoencoder-decoder, NOT a GAN.  Every decoder layer
# applies bilinear upsampling followed by convolution — this acts as a
# low-pass filter and produces faces that are SMOOTHER than the originals.
# Smoother face → LESS high-frequency energy → STEEPER (more-negative) slope.
#
# SCORING DIRECTION (for FF++ autoencoder-based deepfakes):
#   steeper slope (more negative) = smoother face = more suspicious = higher score
#   formula:  score = (_NATURAL_SLOPE - measured_slope) / _SLOPE_RANGE
#
# _NATURAL_SLOPE : expected slope for a real camera-captured face image
#                  following the 1/f² law.  Theoretically -1.0; H.264
#                  compression steepens it slightly, so -1.0 is a safe
#                  conservative floor (images steeper than -1.0 are flagged).
#
# _SLOPE_RANGE   : denominator for normalisation.  1.5 means a face with
#                  slope -2.5 maps to score 1.0, giving generous headroom.
#                  StandardScaler in ensemble.py handles final rescaling so
#                  the absolute values matter less than the real/fake gap.
#
# _N_BANDS       : number of radial rings.  40 gives ~2.8 px rings on 224px.
# _SKIP_BANDS    : innermost rings to drop — dominated by the DC component,
#                  which biases the line fit.
_NATURAL_SLOPE = -1.0
_SLOPE_RANGE   =  1.5
_N_BANDS       = 40
_SKIP_BANDS    = 3


def fft_anomaly_score(face_img, target_size=224):
    """
    Score a single face image for FFT-based deepfake anomalies.

    Returns a float in [0.0, 1.0].  Higher means more suspicious.

    HOW IT WORKS — step by step
    ---------------------------
    1. Resize     : scale to target_size × target_size so every image lives
                    on the same frequency grid.
    2. Grayscale  : collapse BGR to luminance — FFT artifacts are
                    colour-invariant and grayscale reduces compute by 3×.
    3. FFT        : 2D DFT → complex frequency domain.
    4. Log-mag    : log(1 + |FFT|) compresses the dynamic range.
    5. Radial bins: divide the spectrum into 40 concentric rings.
                    Each ring covers one radial-frequency band.
    6. Slope fit  : fit a line to (log ring_radius, mean log-mag) across
                    rings 3–38 (skipping the noisy DC region).
    7. Score      : deviation of the fitted slope from the natural -1.0
                    baseline, normalised to [0, 1].

    WHY SLOPE IS BETTER THAN MEAN PERIPHERAL ENERGY
    ------------------------------------------------
    The old version measured the average log-magnitude in the outer part of
    the spectrum.  That single number is hard to calibrate and had only
    ~0.012 delta between real and fake on FF++ C23, because it blends all
    high-frequency bands together and shifts with overall brightness.

    The slope captures the *shape* of the spectral decay, which is largely
    invariant to overall brightness and H.264 compression level.

    FOR FF++ AUTOENCODER DEEPFAKES SPECIFICALLY:
    The autoencoder decoder applies bilinear upsampling + convolution (a
    low-pass filter), producing faces that are smoother than originals.
    Smoother → less high-frequency energy → steeper (more-negative) slope.
    Score = how much the slope is steeper than the natural -1.0 baseline:
        score = (_NATURAL_SLOPE - measured_slope) / _SLOPE_RANGE
    Higher score → steeper rolloff → smoother face → more suspicious.

    Args:
        face_img    : numpy array  (H × W × 3, BGR, any size)
        target_size : resize to this square before FFT (default 224)

    Returns:
        float in [0.0, 1.0], rounded to 4 decimal places.
    """
    # 1 & 2 — resize + grayscale
    img  = resize_to_square(face_img, target_size)
    gray = to_grayscale(img)

    # 3 & 4 — FFT + log-magnitude
    spectrum = compute_log_magnitude_spectrum(gray)

    # 5 — radial binning
    band_centers, band_means = compute_radial_power_spectrum(spectrum, n_bands=_N_BANDS)

    # 6 — slope fit (skip DC-side bands and the last 2 noisy edge bands)
    freqs    = band_centers[_SKIP_BANDS:-2]
    energies = band_means[_SKIP_BANDS:-2]

    if len(freqs) < 4:
        return 0.5  # fallback for pathologically small images

    log_freqs = np.log(freqs + 1e-8)
    slope, _  = np.polyfit(log_freqs, energies, 1)

    # 7 — score: steeper (more-negative) slope → smoother face → higher score
    #     for FF++ autoencoder deepfakes, fake faces are smoother than real.
    score = float(np.clip((_NATURAL_SLOPE - slope) / _SLOPE_RANGE, 0.0, 1.0))
    return round(score, 4)
