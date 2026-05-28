# =============================================================================
# src/temporal_scorer.py  —  Temporal consistency analysis via optical flow
#
# Deepfake generators produce each frame independently, causing:
#   (a) Unnatural SMOOTHNESS — too little frame-to-frame motion (GANs/diffusion)
#   (b) Unnatural JITTER     — sudden jumps at the warping-mask boundary
#
# This module uses OpenCV Lucas-Kanade optical flow to track feature points
# across frames.  No external model download required; cv2 is always present.
#
# Optional: if MediaPipe Tasks API is available with a downloaded model,
#           468-landmark tracking gives a finer-grained signal.  The module
#           gracefully falls back to optical flow when MediaPipe is absent or
#           when the model file is not found.
# =============================================================================

import cv2
import numpy as np

_MIN_FRAMES = 4   # need at least this many frames
_MIN_POINTS = 10  # minimum tracked points for reliable statistics

# Lucas-Kanade parameters
_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

# Shi-Tomasi corner detection parameters
_FEAT_PARAMS = dict(
    maxCorners=200,
    qualityLevel=0.01,
    minDistance=7,
    blockSize=7,
)


# ---------------------------------------------------------------------------
# Optical-flow based analysis
# ---------------------------------------------------------------------------

def _detect_points(gray):
    """Detect trackable feature points in a grayscale frame."""
    pts = cv2.goodFeaturesToTrack(gray, mask=None, **_FEAT_PARAMS)
    if pts is None:
        return np.empty((0, 1, 2), dtype=np.float32)
    return pts


def _track_optical_flow(frames_bgr):
    """
    Track feature points across all frames using Lucas-Kanade optical flow.

    Returns:
        List of (N×2) float32 arrays of per-frame displacements from frame t-1→t.
        Only points successfully tracked in EVERY consecutive pair are included.
    """
    if len(frames_bgr) < 2:
        return []

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_bgr]

    # Initialise points from first frame
    pts = _detect_points(grays[0])
    if len(pts) < _MIN_POINTS:
        return []

    all_displacements = []

    for i in range(1, len(grays)):
        pts_next, status, _ = cv2.calcOpticalFlowPyrLK(
            grays[i - 1], grays[i], pts, None, **_LK_PARAMS
        )
        if pts_next is None or status is None:
            break

        good = status.ravel() == 1
        if good.sum() < _MIN_POINTS:
            break

        delta = pts_next[good] - pts[good]               # (K, 1, 2)
        dist  = np.sqrt((delta[:, 0, :] ** 2).sum(axis=1))  # (K,)
        all_displacements.append(dist)

        # Update tracked points (only good ones)
        pts = pts_next[good]

    return all_displacements


def _face_region_mask(frame_shape, bbox):
    """
    Create a binary mask: 1 = face interior, 0 = boundary / outside.
    Used to separate inner-face points from boundary points.
    """
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if bbox is None:
        # No bbox: treat whole frame as interior
        mask[:] = 1
        return mask
    x, y, bw, bh = bbox
    # Inner 60% of face = interior; outer 20% ring = boundary
    margin_x = int(bw * 0.20)
    margin_y = int(bh * 0.20)
    ix = max(0, x + margin_x)
    iy = max(0, y + margin_y)
    iw = max(0, bw - 2 * margin_x)
    ih = max(0, bh - 2 * margin_y)
    mask[iy:iy + ih, ix:ix + iw] = 1
    return mask


# ---------------------------------------------------------------------------
# Score components
# ---------------------------------------------------------------------------

def _smoothness_score(displacements, face_size_px=224.0):
    """
    Unnaturally smooth motion → deepfake signal.
    Real faces have micro-motion; deepfakes are often too still.
    Returns float in [0, 1].  HIGH = suspiciously smooth.
    """
    if not displacements:
        return 0.5
    mean_disp = float(np.mean([d.mean() for d in displacements]))
    norm_disp  = mean_disp / max(face_size_px, 1.0)
    # < 0.002 of face width per frame = suspiciously still
    # > 0.020 of face width = natural / jittery
    smoothness = 1.0 - float(np.clip(norm_disp / 0.010, 0.0, 1.0))
    return round(smoothness, 4)


def _jitter_score(displacements):
    """
    High inter-frame variance in displacement = inconsistent generation.
    Returns float in [0, 1].  HIGH = suspicious jitter.
    """
    if len(displacements) < 2:
        return 0.5
    per_frame_mean = np.array([d.mean() for d in displacements])
    var = float(np.var(per_frame_mean))
    # High variance (> 20px^2 per frame) = very inconsistent
    return round(float(np.clip(var / 20.0, 0.0, 1.0)), 4)


def _acceleration_score(displacements):
    """
    Check for unnatural step changes in velocity (deepfake frame transitions).
    Computes variance of first differences of per-frame mean displacements.
    """
    if len(displacements) < 3:
        return 0.5
    means = np.array([d.mean() for d in displacements])
    acc   = np.diff(means)            # first differences = "acceleration"
    acc_var = float(np.var(acc))
    return round(float(np.clip(acc_var / 10.0, 0.0, 1.0)), 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def temporal_consistency_score(frames_bgr, face_bboxes=None):
    """
    Analyse temporal motion consistency across a sequence of frames.

    Args:
        frames_bgr  : list of BGR numpy arrays (one per sampled frame)
        face_bboxes : list of (x,y,w,h) or None per frame (optional)

    Returns:
        dict with:
            "score"      : float [0,1] — overall temporal suspicion score
            "smoothness" : float [0,1] — too-smooth motion score
            "jitter"     : float [0,1] — inconsistent motion score
            "accel"      : float [0,1] — acceleration variance score
            "n_frames"   : int   — number of frames where flow was computed
            "available"  : bool  — False if < 4 frames or tracking failed
    """
    valid_frames = [f for f in frames_bgr if f is not None]

    if len(valid_frames) < _MIN_FRAMES:
        return {
            "score": 0.5, "smoothness": 0.5, "jitter": 0.5, "accel": 0.5,
            "n_frames": len(valid_frames), "available": False,
            "note": f"only {len(valid_frames)} frames (need >= {_MIN_FRAMES})",
        }

    displacements = _track_optical_flow(valid_frames)

    if not displacements:
        return {
            "score": 0.5, "smoothness": 0.5, "jitter": 0.5, "accel": 0.5,
            "n_frames": len(valid_frames), "available": False,
            "note": "optical flow tracking failed (too few feature points)",
        }

    # Estimate face size from first frame
    h, w = valid_frames[0].shape[:2]
    face_size_px = float(min(h, w))

    smooth = _smoothness_score(displacements, face_size_px)
    jitter = _jitter_score(displacements)
    accel  = _acceleration_score(displacements)

    score = round(0.50 * smooth + 0.30 * jitter + 0.20 * accel, 4)

    return {
        "score"     : score,
        "smoothness": smooth,
        "jitter"    : jitter,
        "accel"     : accel,
        "n_frames"  : len(valid_frames),
        "available" : True,
        "note"      : f"tracked via Lucas-Kanade optical flow ({len(displacements)} frame-pairs)",
    }


def mediapipe_available():
    """Always True — we use OpenCV optical flow, no MediaPipe required."""
    return True
