# =============================================================================
# src/freq_analysis/anomaly_scorer.py
# =============================================================================
# Converts a raw face image into a single 0–1 anomaly score using the FFT
# pipeline from fft_extractor.py.
#
# This is the function that ensemble.py calls.
#
# IMPLEMENTATION — four spectral sub-features, equally weighted
# -------------------------------------------------------------
# Previous version: one feature (radial power-spectrum slope only).
#   → Δ ≈ 0.024 between real and fake on FF++ C23.
#
# Current version: four features extracted from the same FFT computation.
#
#   1. SLOPE        — radial power-spectrum slope in log-log space.
#                     FF++ autoencoder decoders apply bilinear upsampling +
#                     convolution (a low-pass filter), making decoded faces
#                     smoother.  Smoother → steeper (more-negative) slope.
#
#   2. HF_RATIO     — fraction of total log-magnitude energy in the outer
#                     50% of radial rings.  Smoother face → less high-freq
#                     energy → lower ratio.
#                     Strongest signal: |Δ|/pooled_std = 0.182.
#
#   3. ENTROPY      — Shannon entropy of the 40-band log-magnitude radial
#                     profile, normalised to [0,1].  Smoother face → energy
#                     concentrated in low-freq rings → lower entropy.
#                     Best signal: |Δ|/pooled_std = 0.212.
#
#   4. PEAK_EXCESS  — tallest positive deviation of the radial profile above
#                     the fitted 1/f line.  GAN fakes show strong spikes;
#                     FF++ autoencoder fakes show slightly fewer peaks than
#                     real.  Weakest signal on FF++ C23: |Δ|/std = 0.081.
#
# ALL FOUR features have the same sign convention on FF++ C23:
#   lower raw value → smoother / more suspicious → higher score component
#
# Each component is normalised:
#   component_score = clip( (BASELINE − raw) / RANGE, 0.0, 1.0 )
# where BASELINE = mean of real-face distribution
#       RANGE    = 3 × std of real-face distribution
# (both calibrated on the 778-frame FF++ C23 dataset in data/manifest.csv)
#
# Final score: simple average of the four component scores (equal weights).
# StandardScaler + LogisticRegression in ensemble.py learns the true optimal
# weights from training data, so the exact per-component weights here matter
# less than the direction and rough scale.
# =============================================================================

import numpy as np

from src.freq_analysis.fft_extractor import (
    to_grayscale,
    compute_log_magnitude_spectrum,
    compute_radial_power_spectrum,
    compute_high_freq_energy_ratio,
    compute_spectral_entropy,
    compute_peak_excess,
)
from src.freq_analysis.utils import resize_to_square


# ---------------------------------------------------------------------------
# Calibration constants
# ---------------------------------------------------------------------------
# Derived by running _calibrate.py over the 778-frame FF++ C23 dataset
# (data/manifest.csv).  Do not hand-tune these — re-run _calibrate.py if
# the dataset changes substantially.
#
# BASELINE = mean of the real-face distribution for each raw sub-feature.
# RANGE    = 3 × std of the real-face distribution.
#            → a face 1σ below real mean scores ≈ 0.33
#            → a face 2σ below real mean scores ≈ 0.67
#            → a face 3σ below real mean scores  = 1.0
#
# KNOWN LIMITATION — mild preprocessing leakage
# -----------------------------------------------
# _calibrate.py was run over all 778 frames, including the ~20% that
# ensemble.py later holds out for validation.  This means BASELINE and
# RANGE incorporate a small amount of information from the validation set.
# The ml-reviewer estimates the per-score shift is < 0.01, and
# StandardScaler in ensemble.py re-normalises the fft_score column from
# training data only, which partially cancels the effect.
#
# The correct fix is to compute BASELINE/RANGE from training-split real
# frames only (after GroupShuffleSplit is performed in ensemble.py).
# Until that refactor is done, treat AUC and cross-validated AUC as the
# reliable metrics — not balanced accuracy or per-threshold metrics.
# (CLAUDE.md already mandates this: "Show me the AUC/PR numbers on the
# held-out video split before claiming any change is an improvement.")

# 1. Spectral slope  (radial power-spectrum slope in log-log space)
_SLOPE_BASELINE = -2.163877
_SLOPE_RANGE    =  0.592026   # 3 × real_std 0.197342

# 2. High-frequency energy ratio  (outer 50% / total, log-magnitude rings)
_HF_RATIO_BASELINE = 0.401319
_HF_RATIO_RANGE    = 0.032208   # 3 × real_std 0.010736

# 3. Spectral entropy  (Shannon entropy of 40-band log-magnitude profile)
_ENTROPY_BASELINE = 0.991984
_ENTROPY_RANGE    = 0.004608   # 3 × real_std 0.001536

# 4. Peak excess  (tallest spike above the fitted 1/f line)
_PEAK_EXCESS_BASELINE = 0.294229
_PEAK_EXCESS_RANGE    = 0.308040   # 3 × real_std 0.102680

# Shared FFT parameters (must match those used during calibration)
_N_BANDS    = 40
_SKIP_BANDS =  3   # DC-side rings dropped from slope fit and peak detection


def _score_component(raw_value, baseline, range_):
    """
    Normalise one raw sub-feature to [0, 1] where higher = more suspicious.

    Formula: clip( (baseline − raw) / range, 0, 1 )

    When raw == baseline (real-face average):  score = 0.0
    When raw is far below baseline (smoother): score approaches 1.0
    When raw is above baseline (more textured): score = 0.0  (clipped)

    Args:
        raw_value : float — the measured sub-feature value
        baseline  : float — BASELINE constant from the calibration block above
        range_    : float — RANGE constant from the calibration block above

    Returns:
        float in [0.0, 1.0]
    """
    return float(np.clip((baseline - raw_value) / range_, 0.0, 1.0))


def fft_spectral_features(face_img, target_size=224):
    """
    Return the four raw FFT sub-features as a dict — NO pre-combining.

    WHY THIS EXISTS (vs fft_anomaly_score)
    ---------------------------------------
    fft_anomaly_score() collapses four sub-features into one number with
    equal weights before the logistic regression ever sees them.  That
    throws away real information: on FF++ C23, spectral entropy has
    |Δ|/pooled_std ≈ 0.212 while peak_excess has only ≈ 0.081 — a 2.6×
    difference.  Forcing equal weights blunts the stronger signal.

    By returning raw values, we let StandardScaler + LogisticRegression in
    ensemble.py learn the correct weight for each sub-feature from training
    data.  This also removes the calibration leakage in fft_anomaly_score():
    the hard-coded BASELINE/RANGE constants were derived over all 778 frames
    (including the ~20% val set).  StandardScaler uses training data only,
    so there is no leakage when the raw features are normalised there.

    Sign conventions (what direction = more real-like):
        fft_slope       : more negative → real-like; fakes are less negative
        fft_hf_ratio    : higher → real-like; fakes have less HF energy
        fft_entropy     : higher → real-like; fakes concentrate in low-freq
        fft_peak_excess : higher → suspicious; periodic GAN/conv artifacts

    The LR with class_weight='balanced' learns the correct coefficient
    sign for each feature from training data — no manual sign flip needed.

    Args:
        face_img    : numpy array  (H × W × 3, BGR, any size)
        target_size : int — resize to this square before FFT (default 224)

    Returns:
        dict with four float values:
            "fft_slope"       — radial slope (log-log fit); typical range -3 to -1
            "fft_hf_ratio"    — outer-half energy fraction; range [0.0, 1.0]
            "fft_entropy"     — normalised Shannon entropy; range [0.0, 1.0]
            "fft_peak_excess" — tallest spike above 1/f fit; range [0.0, ∞)
        On degenerate input (all-zero / constant spectrum) returns neutral
        values equal to the real-face calibration baselines so the LR sees
        a "typical real face" rather than an outlier.
    """
    # Resize + grayscale (same preprocessing as fft_anomaly_score)
    img      = resize_to_square(face_img, target_size)
    gray     = to_grayscale(img)
    log_spec = compute_log_magnitude_spectrum(gray)

    # 40-ring radial profile — shared by all four sub-features
    centers, log_means = compute_radial_power_spectrum(log_spec, n_bands=_N_BANDS)

    # Degenerate guard: return real-face baseline values so the LR is not
    # confused by a constant-spectrum image.
    if log_means.max() - log_means.min() < 1e-6:
        return {
            "fft_slope"       : float(_SLOPE_BASELINE),
            "fft_hf_ratio"    : float(_HF_RATIO_BASELINE),
            "fft_entropy"     : float(_ENTROPY_BASELINE),
            "fft_peak_excess" : float(_PEAK_EXCESS_BASELINE),
        }

    # --- Slope ---
    freqs    = centers[_SKIP_BANDS:-2]
    energies = log_means[_SKIP_BANDS:-2]
    if len(freqs) >= 4:
        try:
            slope, _ = np.polyfit(np.log(freqs + 1e-8), energies, 1)
        except (np.linalg.LinAlgError, ValueError):
            slope = _SLOPE_BASELINE
    else:
        slope = _SLOPE_BASELINE

    # --- High-frequency energy ratio ---
    hf_ratio = compute_high_freq_energy_ratio(log_means, outer_fraction=0.5)

    # --- Spectral entropy ---
    entropy = compute_spectral_entropy(log_means)

    # --- Peak excess ---
    peak_excess = compute_peak_excess(centers, log_means, skip_bands=_SKIP_BANDS)

    return {
        "fft_slope"       : float(round(slope,       6)),
        "fft_hf_ratio"    : float(round(hf_ratio,    6)),
        "fft_entropy"     : float(round(entropy,      6)),
        "fft_peak_excess" : float(round(peak_excess,  6)),
    }


def fft_anomaly_score(face_img, target_size=224):
    """
    Score a single face image for FFT-based deepfake anomalies.

    Returns a float in [0.0, 1.0].  Higher means more suspicious (more likely
    to be a deepfake based on its frequency-domain characteristics).

    HOW IT WORKS — step by step
    ---------------------------
    1. Resize     : scale to target_size × target_size so every image lives
                    on the same frequency grid.
    2. Grayscale  : collapse BGR to luminance — FFT artifacts are largely
                    colour-invariant; grayscale reduces compute by 3×.
    3. FFT        : 2D DFT → shift DC to centre → log(1 + |FFT|).
    4. Radial bins: divide the spectrum into 40 concentric rings.
    5. Four sub-features extracted from the same 40-ring profile:
         a. Slope       — gradient of (log_radius, ring_mean) line fit
         b. HF ratio    — outer-half energy / total energy
         c. Entropy     — Shannon entropy of the normalised ring distribution
         d. Peak excess — max positive residual above the fitted 1/f line
    6. Each sub-feature normalised to [0,1] via calibrated constants.
    7. Combined score = mean of the four component scores.

    WHY FOUR FEATURES INSTEAD OF ONE
    ---------------------------------
    The previous single-slope version had Δ ≈ 0.024 between real and fake
    on FF++ C23.  Each new feature captures a partially different aspect of
    the same underlying "smoothness vs texture" difference:

      Slope     — shape of the spectral decay curve
      HF ratio  — total energy balance between low and high frequencies
      Entropy   — how spread-out (rather than peaked) the energy is
      Peak excess — local spikes vs smooth decay

    By averaging four weakly correlated measurements of the same phenomenon,
    we get a lower-variance combined score with more reliable separation.

    FOR FF++ AUTOENCODER DEEPFAKES SPECIFICALLY
    -------------------------------------------
    The autoencoder decoder applies bilinear upsampling + convolution (a
    low-pass filter), producing faces that are SMOOTHER than originals.
    Smoother → less high-frequency energy → all four features shift in the
    same direction (all lower for fakes than for reals on FF++ C23).

    Args:
        face_img    : numpy array  (H × W × 3, BGR, any size)
        target_size : resize to this square before FFT (default 224)

    Returns:
        float in [0.0, 1.0], rounded to 4 decimal places.
    """
    # 1 & 2 — resize + grayscale
    img  = resize_to_square(face_img, target_size)
    gray = to_grayscale(img)

    # 3 — FFT → log-magnitude spectrum
    log_spec = compute_log_magnitude_spectrum(gray)

    # 4 — 40-ring radial profile  (one binning shared by all four sub-features)
    centers, log_means = compute_radial_power_spectrum(log_spec, n_bands=_N_BANDS)

    # Guard: degenerate spectrum (all-zero or constant input)
    if log_means.max() - log_means.min() < 1e-6:
        return 0.5  # return neutral rather than 0 or 1

    # ------------------------------------------------------------------
    # 5a — Slope sub-feature
    # ------------------------------------------------------------------
    # Fit a line to (log ring_radius, mean log-magnitude) across the
    # usable bands (skipping DC side and noisy edge).
    freqs    = centers[_SKIP_BANDS:-2]
    energies = log_means[_SKIP_BANDS:-2]

    if len(freqs) < 4:
        return 0.5   # pathologically small image

    try:
        slope, _ = np.polyfit(np.log(freqs + 1e-8), energies, 1)
    except (np.linalg.LinAlgError, ValueError):
        slope = _SLOPE_BASELINE  # degenerate fit — neutral (maps to score 0)

    # ------------------------------------------------------------------
    # 5b — High-frequency energy ratio sub-feature
    # ------------------------------------------------------------------
    hf_ratio = compute_high_freq_energy_ratio(log_means, outer_fraction=0.5)

    # ------------------------------------------------------------------
    # 5c — Spectral entropy sub-feature
    # ------------------------------------------------------------------
    # Uses log-magnitude band means (not raw power) — see fft_extractor.py
    # for the full explanation of why log-magnitude gives better separation
    # than raw power for entropy on FF++ C23.
    entropy = compute_spectral_entropy(log_means)

    # ------------------------------------------------------------------
    # 5d — Peak excess sub-feature
    # ------------------------------------------------------------------
    peak_excess = compute_peak_excess(centers, log_means, skip_bands=_SKIP_BANDS)

    # ------------------------------------------------------------------
    # 6 — Normalise each component to [0, 1]
    # ------------------------------------------------------------------
    # Convention: lower raw value → smoother face → more suspicious → higher score
    slope_score = _score_component(slope,       _SLOPE_BASELINE,      _SLOPE_RANGE)
    hf_score    = _score_component(hf_ratio,    _HF_RATIO_BASELINE,   _HF_RATIO_RANGE)
    ent_score   = _score_component(entropy,     _ENTROPY_BASELINE,    _ENTROPY_RANGE)
    peak_score  = _score_component(peak_excess, _PEAK_EXCESS_BASELINE, _PEAK_EXCESS_RANGE)

    # ------------------------------------------------------------------
    # 7 — Combine with equal weights
    # ------------------------------------------------------------------
    # Equal weights: fair starting point before LogisticRegression in
    # ensemble.py learns the optimal weighting from training data.
    combined = (slope_score + hf_score + ent_score + peak_score) / 4.0

    return round(float(np.clip(combined, 0.0, 1.0)), 4)
