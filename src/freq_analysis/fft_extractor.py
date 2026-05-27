# =============================================================================
# src/freq_analysis/fft_extractor.py
# =============================================================================
# Pure FFT mathematics.  No scoring, no I/O — only the signal-processing math.
# These three functions are the building blocks for every other part of the
# frequency analysis pipeline.
#
# BACKGROUND: HOW 2D FFT WORKS
# -----------------------------
# The Fast Fourier Transform (FFT) decomposes an image into sine and cosine
# waves at different spatial frequencies.
#
#   Low  frequencies  -> gradual changes across the image (background, skin tone)
#   High frequencies  -> rapid changes (edges, fine texture, noise)
#
# The output is a grid the same size as the input.  Each cell (u, v) in the
# grid represents "how much energy is at horizontal frequency u and vertical
# frequency v."
#
# After np.fft.fftshift, the DC (zero-frequency) component sits at the
# center of the grid.  Moving outward from the center means higher and higher
# spatial frequencies.
#
# WHY DOES THIS DETECT DEEPFAKES?
# --------------------------------
# Real photographs follow a "1/f" power law: low frequencies carry most of
# the image energy, and energy falls off predictably as frequency rises.
#
# GAN-generated and diffusion-model faces often violate this law.  The
# neural network that generates the face is not trained to produce exactly
# the right high-frequency texture — it focuses on perceptual realism at
# medium scale.  The result is a characteristic "footprint" in the FFT:
# either too much or too little energy at certain high-frequency bands, or
# periodic spikes that repeat across the frequency grid (from up-sampling
# artifacts in the GAN architecture).
#
# By measuring the mean peripheral (high-frequency) energy in the log-
# magnitude spectrum, we can flag images that deviate from the expected
# natural camera pattern.
# =============================================================================

import numpy as np
import cv2


def to_grayscale(img_bgr):
    """
    Convert a BGR colour image to grayscale.

    We use grayscale because:
      - FFT is a 2D operation — one channel at a time.
      - Frequency-domain artifacts from deepfakes are largely colour-invariant
        (they affect luminance more than chrominance).
      - Using grayscale reduces computation by 3x with minimal information loss.

    Args:
        img_bgr : numpy array  (H x W x 3, BGR colour order)

    Returns:
        numpy array  (H x W, single channel, dtype=uint8)
    """
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def compute_log_magnitude_spectrum(gray):
    """
    Compute the 2D FFT of a grayscale image and return the log-magnitude
    spectrum with the DC component shifted to the centre.

    STEPS
    -----
    1. np.fft.fft2     — 2D discrete Fourier transform.
                         Output is complex (real + imaginary parts).
    2. np.fft.fftshift — Rearrange the output so the zero-frequency (DC)
                         component is at the centre rather than the corner.
    3. np.abs          — Magnitude = sqrt(real^2 + imag^2).
                         Tells us how much energy is at each frequency.
    4. np.log(1 + x)   — Logarithm compresses the dynamic range.
                         Without this, the DC component (centre pixel) is
                         so much larger than everything else that the plot
                         looks blank except for one bright dot.
                         Adding 1 prevents log(0) = -inf.

    Args:
        gray : numpy array  (H x W, grayscale, any dtype)

    Returns:
        numpy float32 array  (H x W)  — log-magnitude spectrum
        Values typically in range 3 – 12 for 224x224 face images.
    """
    # Cast to float so FFT arithmetic is precise (uint8 would overflow).
    f32 = gray.astype(np.float32)

    # 2D FFT: transform from spatial domain to frequency domain.
    fft2 = np.fft.fft2(f32)

    # Shift DC to centre (cosmetic rearrangement, does not change values).
    fshift = np.fft.fftshift(fft2)

    # Log-magnitude: the number we actually analyse.
    log_mag = np.log(1.0 + np.abs(fshift))

    return log_mag.astype(np.float32)


def compute_power_spectrum_2d(gray):
    """
    Compute the 2D power spectrum of a grayscale image WITHOUT a log transform.

    DESIGN NOTE — NOT USED FOR ENTROPY IN anomaly_scorer.py
    --------------------------------------------------------
    This function was written to test whether raw power space (|FFT|²) gives
    better real/fake separation for spectral entropy than log-magnitude space.

    Hypothesis going in: the large dynamic range of raw power (inner rings
    carry ~100–1000x more power than outer rings) would make the entropy
    distribution more sensitive to the smoothing that FF++ autoencoder
    decoders introduce.

    What the data showed (run on 778 FF++ C23 frames, _smoke_test_entropy.py):
        Config          |Delta|/pooled_std   Direction
        log_all             0.212             correct (fake < real)   <- USED
        power_all           0.079             WRONG  (fake > real)
        power_skip3         0.143             correct but weaker

    Log-magnitude entropy turned out to have the best signal despite its
    narrow absolute range (0.9917 vs 0.9920).  The DC component dominates
    the raw-power distribution so completely that the residual variation
    across non-DC rings mostly reflects image-specific differences unrelated
    to deepfake smoothing.

    This function is kept as a documented building block for future experiments
    (e.g., per-channel power spectrum analysis, or power-domain features other
    than entropy).  anomaly_scorer.py deliberately uses log-magnitude band means
    for all four sub-features.

    WHAT "POWER" MEANS
    ------------------
    Power at frequency (u, v) = |FFT(u, v)|²  (squared magnitude).

    Args:
        gray : numpy array  (H × W, grayscale, any dtype)

    Returns:
        numpy float64 array  (H × W)  — raw power spectrum, DC at centre.
        float64 used to avoid precision loss at large magnitudes.
    """
    f32    = gray.astype(np.float32)
    fft2   = np.fft.fft2(f32)
    fshift = np.fft.fftshift(fft2)

    # Squared magnitude — the standard definition of "power spectrum".
    power  = (np.abs(fshift) ** 2)

    return power.astype(np.float64)


def make_center_mask(shape, center_fraction=0.1):
    """
    Build a binary mask that is 0 inside a circular central region and 1 outside.

    WHY CIRCULAR, NOT SQUARE?
    -------------------------
    In a 2D FFT, the spatial frequency at pixel (u, v) is proportional to its
    Euclidean distance from the DC centre: freq = sqrt(u² + v²).  Iso-frequency
    contours are therefore circles, not squares.  A square blank zone incorrectly
    includes corner pixels that are actually high-frequency (distance > radius)
    and excludes ring pixels along the axes that are genuinely low-frequency.
    A circular blank zone ensures every masked pixel really is in the low-
    frequency DC neighbourhood, and every unmasked pixel really is high-frequency.

    Args:
        shape           : (H, W) tuple — should match the spectrum shape.
        center_fraction : float  0-1 — diameter of the blank circle as a
                          fraction of the shorter image axis.
                          0.10 blanks a circle of radius = 5% of min(H,W).
                          Default 0.10 discards the dominant DC region.

    Returns:
        numpy float32 array  (H x W)  — 0 inside the circle, 1 everywhere else.
    """
    h, w = shape
    cy, cx = h // 2, w // 2

    # Radius of the circular blank zone.
    radius = max(1.0, min(h, w) * center_fraction / 2.0)

    # Squared distance of every pixel from the centre (avoids a sqrt per pixel).
    Y, X = np.ogrid[:h, :w]
    dist_sq = (Y - cy).astype(np.float32) ** 2 + (X - cx).astype(np.float32) ** 2

    mask = np.ones((h, w), dtype=np.float32)
    mask[dist_sq <= radius ** 2] = 0.0

    return mask


def compute_radial_power_spectrum(log_mag_spectrum, n_bands=40):
    """
    Bin the log-magnitude spectrum into concentric radial frequency bands.

    Each band is an annular ring at a given distance from the DC centre.
    The returned band_means array gives the mean log-magnitude energy in each
    ring — this is the radial power spectrum of the image.

    WHY THIS IS BETTER THAN A SINGLE PERIPHERAL MEAN
    -------------------------------------------------
    A single peripheral mean collapses all high-frequency information into one
    number.  The radial profile preserves *how* energy changes with frequency,
    which lets us fit the 1/f decay slope — a much stronger discriminator.

    Args:
        log_mag_spectrum : (H x W) float32 array — output of
                           compute_log_magnitude_spectrum().
        n_bands          : number of radial rings to divide the spectrum into.
                           40 gives ~2.8 pixel-wide rings for a 224x224 image.

    Returns:
        band_centers : float32 array (n_bands,) — radius (pixels) at ring centre.
        band_means   : float32 array (n_bands,) — mean log-magnitude per ring.
    """
    h, w = log_mag_spectrum.shape
    cy, cx = h // 2, w // 2

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt(
        (Y - cy).astype(np.float32) ** 2 + (X - cx).astype(np.float32) ** 2
    )

    max_r  = float(min(cy, cx))
    edges  = np.linspace(0.0, max_r, n_bands + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means   = np.zeros(n_bands, dtype=np.float32)

    for i in range(n_bands):
        ring = (dist >= edges[i]) & (dist < edges[i + 1])
        if ring.any():
            means[i] = float(np.mean(log_mag_spectrum[ring]))

    return centers.astype(np.float32), means


def compute_high_freq_energy_ratio(band_means, outer_fraction=0.5):
    """
    Compute the fraction of total radial energy that lives in the outer rings.

    WHY THIS HELPS
    --------------
    The 40 rings are ordered from lowest frequency (ring 0, near DC) to
    highest frequency (ring 39, near Nyquist).  We split them in half at
    `outer_fraction` and ask: what share of the total log-magnitude energy
    is in the HIGH-frequency half?

    Natural images follow a 1/f² power law, so most energy is in the inner
    (low-frequency) rings, but there is still a predictable amount in the
    outer rings.  FF++ autoencoder decoders apply bilinear upsampling +
    convolution (a low-pass filter), making decoded faces SMOOTHER than the
    originals.  Smoother = less high-frequency energy = LOWER ratio.

    Interpretation:
        low  ratio  → smoother face → more suspicious (fake-like)
        high ratio  → more textured → more real-like

    Note: the scoring *direction* (low ratio = suspicious) is handled in
    anomaly_scorer.py, not here.  This function just returns the raw ratio.

    Args:
        band_means     : float array (n_bands,) — output of
                         compute_radial_power_spectrum().  Values are mean
                         log-magnitudes per ring (all non-negative).
        outer_fraction : float in (0, 1) — fraction of rings treated as
                         "high frequency."  Default 0.5 = outer half.

    Returns:
        float in [0.0, 1.0]
    """
    n = len(band_means)
    split = int(n * (1.0 - outer_fraction))   # first index of the outer zone

    outer_energy = float(band_means[split:].sum())
    total_energy = float(band_means.sum())

    if total_energy < 1e-8:
        return 0.5   # degenerate (all-zero) spectrum — return neutral

    return float(np.clip(outer_energy / total_energy, 0.0, 1.0))


def compute_spectral_entropy(band_means):
    """
    Compute the normalised Shannon entropy of the radial power distribution.

    WHAT IS ENTROPY HERE?
    ---------------------
    Shannon entropy measures how "spread out" a distribution is.

      H = -sum( p_i * log(p_i) )

    where p_i = band_means[i] / sum(band_means)  (normalised to a
    probability distribution).

    If all 40 rings have equal energy, H = log(40) — maximum spread,
    maximum entropy.

    If all energy is in one ring, H = 0 — perfectly concentrated.

    We divide H by log(n_bands) so the result is always in [0, 1].

    WHY THIS HELPS FOR FF++ DEEPFAKES
    ------------------------------------
    Autoencoder smoothing concentrates energy in the low-frequency inner
    rings.  Concentrated = lower entropy.  Real faces have energy spread
    more evenly across the spectrum, so they have higher entropy.

    Interpretation:
        low  entropy → energy concentrated in low-freq → smoother → suspicious
        high entropy → energy spread out → more textured → real-like

    Note: the scoring direction is handled in anomaly_scorer.py.

    Args:
        band_means : float array (n_bands,) — output of
                     compute_radial_power_spectrum().

    Returns:
        float in [0.0, 1.0]
    """
    total = float(band_means.sum())
    if total < 1e-8:
        return 0.5   # degenerate spectrum — return neutral

    # Normalise to a probability distribution.
    p = band_means / total

    # Clip to avoid log(0).  Any p_i == 0 contributes 0 to the sum
    # (0 * log(0) = 0 by convention), which np.where handles cleanly.
    log_p = np.where(p > 1e-12, np.log(p + 1e-12), 0.0)
    entropy = float(-np.sum(p * log_p))

    # Normalise by maximum possible entropy for n_bands rings.
    max_entropy = float(np.log(len(band_means)))
    if max_entropy < 1e-8:
        return 0.5

    return float(np.clip(entropy / max_entropy, 0.0, 1.0))


def compute_peak_excess(band_centers, band_means, skip_bands=3):
    """
    Measure the largest positive deviation of the radial profile above the
    fitted smooth 1/f decay line.

    HOW IT WORKS
    ------------
    Step 1. Fit the expected log-linear decay:
                log|FFT(f)| ≈ slope * log(f) + intercept
            using least-squares on the usable bands (skipping the DC region).

    Step 2. Compute residuals:
                residual_i = band_means[i] - fitted_i

    Step 3. Return max(residuals), clamped to 0.
            This is the height of the tallest spike that sticks up *above*
            where the smooth natural decay would predict.

    WHY THIS HELPS
    --------------
    A perfectly smooth face produces a spectrum that follows the 1/f line
    closely — residuals near zero.  Upsampling artifacts (common in GAN-
    style deepfakes) create periodic bright spikes in the frequency domain
    that shoot well above the fitted line.  Even FF++ autoencoder fakes
    can produce mild convolutional ringing that shows up as local bumps.

    Note: the raw value returned here is in log-magnitude units and is NOT
    normalised to [0, 1].  anomaly_scorer.py handles calibration.

    Args:
        band_centers : float array (n_bands,) — ring radii, output of
                       compute_radial_power_spectrum().
        band_means   : float array (n_bands,) — mean log-mag per ring.
        skip_bands   : int — skip this many DC-side bands before fitting
                       (same default as anomaly_scorer.py uses for slope).

    Returns:
        float >= 0.0 — height of the tallest spike above the fitted line.
        Returns 0.0 on degenerate input (fewer than 4 usable bands or flat
        spectrum).
    """
    freqs    = band_centers[skip_bands:-2]
    energies = band_means[skip_bands:-2]

    if len(freqs) < 4:
        return 0.0

    if energies.max() - energies.min() < 1e-6:
        return 0.0   # flat spectrum — no peaks possible

    log_freqs = np.log(freqs + 1e-8)

    try:
        slope, intercept = np.polyfit(log_freqs, energies, 1)
    except (np.linalg.LinAlgError, ValueError):
        return 0.0

    fitted    = slope * log_freqs + intercept
    residuals = energies - fitted

    # Only positive residuals (spikes above the line) are anomalous.
    peak = float(np.max(residuals))
    return max(peak, 0.0)
