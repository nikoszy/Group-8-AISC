# =============================================================================
# src/quality_scorer.py  —  Per-frame quality scoring for deepfake detection
#
# Quality-aware weighting is the biggest single accuracy gain:
#   - Blurry frames produce unreliable FFT / texture features
#   - Tiny faces produce unreliable CNN predictions
#   - Over/underexposed frames confuse all detectors
#
# By weighting each frame's P(fake) by its quality before averaging,
# low-quality frames have minimal impact on the final verdict.
# =============================================================================

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Quality components
# ---------------------------------------------------------------------------

def _sharpness_score(gray_face):
    """
    Laplacian variance of a grayscale face crop.
    Well-focused faces have high Laplacian variance (~200-2000).
    Returns float in [0, 1].  Full score at variance >= 500.
    """
    lap_var = float(cv2.Laplacian(gray_face, cv2.CV_64F).var())
    return float(np.clip(lap_var / 500.0, 0.0, 1.0))


def _size_score(bbox, frame_shape):
    """
    Face area as a fraction of the frame.  Larger = more reliable.
    Target: face fills 10-25% of frame.  Full score at >= 20%.
    Returns float in [0, 1].
    """
    h_frame, w_frame = frame_shape[:2]
    x, y, bw, bh = bbox
    face_ratio = (bw * bh) / max(float(h_frame * w_frame), 1.0)
    return float(np.clip(face_ratio / 0.20, 0.0, 1.0))


def _brightness_score(gray_face):
    """
    Penalise frames that are too dark (mean < 50) or overexposed (mean > 210).
    Returns 1.0 (good), 0.3 (marginal), or 0.0 (unusable).
    """
    mean_val = float(np.mean(gray_face))
    if 50.0 <= mean_val <= 210.0:
        return 1.0
    if mean_val < 20.0 or mean_val > 240.0:
        return 0.0
    return 0.3


def _contrast_score(gray_face):
    """
    Standard deviation of pixel values.  Flat/washed-out frames have low std.
    Returns float in [0, 1].  Full score at std >= 40.
    """
    std_val = float(np.std(gray_face.astype(np.float32)))
    return float(np.clip(std_val / 40.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_frame_quality(face_crop_bgr, bbox, frame_shape):
    """
    Compute a quality score for a single face frame.

    Args:
        face_crop_bgr : numpy array (H × W × 3, BGR) — cropped face region
        bbox          : (x, y, w, h) — face bounding box in the original frame
        frame_shape   : (H, W [, C]) of the *original* frame

    Returns:
        float in [0.0, 1.0].  Higher = more reliable for deepfake detection.

    Weight breakdown:
        0.40  sharpness  (Laplacian variance)
        0.25  face size  (fraction of frame)
        0.20  brightness (avoid dark / overexposed)
        0.15  contrast   (std dev of pixels)
    """
    if face_crop_bgr is None or face_crop_bgr.size == 0:
        return 0.0

    # Convert to grayscale once (shared across all sub-scores)
    if len(face_crop_bgr.shape) == 3:
        gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_crop_bgr

    sharp   = _sharpness_score(gray)
    size    = _size_score(bbox, frame_shape)
    bright  = _brightness_score(gray)
    contrast= _contrast_score(gray)

    quality = (
        0.40 * sharp    +
        0.25 * size     +
        0.20 * bright   +
        0.15 * contrast
    )
    return round(float(np.clip(quality, 0.0, 1.0)), 4)


def quality_weighted_mean(probs, qualities, min_quality=0.10):
    """
    Compute quality-weighted mean of per-frame probabilities.

    Frames with quality < min_quality are excluded entirely — they are
    too unreliable to contribute even with low weight.

    Args:
        probs      : list of float — P(fake) per face frame
        qualities  : list of float — quality score per face frame
        min_quality: float — frames below this are excluded

    Returns:
        float — quality-weighted mean P(fake), or simple mean if all qualities
                are below min_quality.
    """
    if not probs:
        return 0.5

    valid = [(p, q) for p, q in zip(probs, qualities) if q >= min_quality]
    if not valid:
        # All frames below quality floor — fall back to simple mean
        return float(np.mean(probs))

    ps, qs = zip(*valid)
    total_w = sum(qs)
    if total_w < 1e-8:
        return float(np.mean(ps))

    return float(sum(p * q for p, q in zip(ps, qs)) / total_w)


def quality_label(score):
    """Human-readable quality tier for display."""
    if score >= 0.70:
        return "High"
    if score >= 0.40:
        return "Medium"
    if score >= 0.15:
        return "Low"
    return "Poor"
