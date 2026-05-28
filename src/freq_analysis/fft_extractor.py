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
