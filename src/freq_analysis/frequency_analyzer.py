# =============================================================================
# src/freq_analysis/frequency_analyzer.py
# =============================================================================
# Public API for batch scoring and visualisation.
# main.py and ensemble.py import from here, not from the lower-level files.
# =============================================================================

import os
import numpy as np
import cv2

from src.freq_analysis.anomaly_scorer import fft_anomaly_score
from src.freq_analysis.fft_extractor import (
    to_grayscale,
    compute_log_magnitude_spectrum,
    make_center_mask,
)
from src.freq_analysis.utils import resize_to_square, load_face_image, normalize_to_uint8


def compute_fft_score_batch(image_paths, target_size=224, verbose=True):
    """
    Score a list of image files and return (path, score) pairs.

    Args:
        image_paths : list of file path strings
        target_size : resize each image to this square before FFT
        verbose     : if True, print progress every 20 files

    Returns:
        list of (path: str, score: float) tuples — in the same order as input.
        Images that failed to load are silently omitted.
    """
    results = []

    for i, path in enumerate(image_paths):
        img = load_face_image(path, target_size)

        if img is None:
            if verbose:
                print(f"  [WARN] Could not load: {path}")
            continue

        score = fft_anomaly_score(img, target_size=target_size)
        results.append((path, score))

        if verbose and ((i + 1) % 20 == 0 or i == 0):
            print(f"  FFT scored {i + 1}/{len(image_paths)}  "
                  f"latest={score:.4f}  file={os.path.basename(path)}")

    return results


def visualize_spectrum(face_img, save_path=None, label=None):
    """
    Create a 3-panel side-by-side image for teaching / demo purposes.

        Panel 1 — ORIGINAL    : the resized face photograph
        Panel 2 — LOG SPECTRUM: the full FFT log-magnitude (bright = high energy)
        Panel 3 — HIGH-FREQ   : spectrum with low-freq centre masked out
                                (the region we actually measure for anomalies)

    How to read the spectrum:
      - Centre pixel = DC (average brightness of the image).
      - A bright ring near the centre = dominant low spatial frequencies.
      - Bright spikes away from centre = periodic patterns (GAN upsampling
        artifacts often show as a cross or grid of spikes here).

    Args:
        face_img  : numpy array (H x W x 3, BGR)
        save_path : if given, write the image to this path; else cv2.imshow.
        label     : optional string shown in top-left ("REAL" or "FAKE").

    Returns:
        numpy array (224 x 672 x 3) — the combined panel image.
    """
    SIZE = 224

    # Standardise size.
    img = resize_to_square(face_img, SIZE)

    # Greyscale and spectrum.
    gray     = to_grayscale(img)
    spectrum = compute_log_magnitude_spectrum(gray)
    mask     = make_center_mask(gray.shape, center_fraction=0.1)

    # Normalise spectrum to 0-255 uint8 for display.
    spec_u8   = normalize_to_uint8(spectrum)
    masked_u8 = normalize_to_uint8(spectrum * mask)

    # Convert everything to 3-channel BGR so np.hstack works.
    orig_panel   = img.copy()
    spec_panel   = cv2.cvtColor(spec_u8, cv2.COLOR_GRAY2BGR)
    masked_panel = cv2.cvtColor(masked_u8, cv2.COLOR_GRAY2BGR)

    # Text labels for each panel.
    font   = cv2.FONT_HERSHEY_SIMPLEX
    panels = [
        (orig_panel,   "ORIGINAL"),
        (spec_panel,   "LOG SPECTRUM"),
        (masked_panel, "HIGH-FREQ ONLY"),
    ]
    for panel, text in panels:
        cv2.putText(panel, text, (4, 18), font, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

    # Optional class label on the first panel.
    if label is not None:
        colour = (50, 200, 50) if label.upper() == "REAL" else (50, 50, 220)
        cv2.putText(orig_panel, label.upper(), (4, SIZE - 6),
                    font, 0.60, colour, 2, cv2.LINE_AA)

    combined = np.hstack([orig_panel, spec_panel, masked_panel])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        cv2.imwrite(save_path, combined)
        print(f"  Spectrum visualisation saved: {save_path}")
    else:
        cv2.imshow("FFT Spectrum Visualisation", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return combined
