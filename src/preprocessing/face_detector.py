"""Haar-cascade face detection with shared train/serve crop logic."""

from __future__ import annotations

import cv2
import numpy as np

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Minimum Haar cascade level-weight to accept a detection.
CONF_THRESHOLD = 0.7

# Bounding-box geometry limits
ASPECT_MIN = 0.7
ASPECT_MAX = 1.4
CENTER_Y_MAX = 0.60

# Reject if face area is below this fraction of the frame.
MIN_FACE_FRAC = 0.04

# detectMultiScale3 parameters (must match inspect_dataset.py / training crops)
_SCALE_FACTOR = 1.1
_MIN_SIZE = (30, 30)


def _pick_best_face(rects, weights, frame_h, frame_w):
    """
    From Haar detections return (x, y, w, h) of the best candidate, or None.

    Selection order:
      1. Drop detections whose level_weight < CONF_THRESHOLD.
      2. Among survivors, pick the largest bounding-box area.
      3. Reject if box centre falls below the top 60 % of the frame.
      4. Reject if aspect ratio (w/h) is outside [ASPECT_MIN, ASPECT_MAX].
      5. Reject if face area < MIN_FACE_FRAC of the frame.
    """
    if len(rects) == 0:
        return None

    confident = [
        (rect, float(wt))
        for rect, wt in zip(rects, weights)
        if float(wt) >= CONF_THRESHOLD
    ]
    if not confident:
        return None

    (x, y, bw, bh), _ = max(confident, key=lambda rw: rw[0][2] * rw[0][3])

    if (y + bh / 2) / frame_h > CENTER_Y_MAX:
        return None
    if not (ASPECT_MIN <= bw / bh <= ASPECT_MAX):
        return None
    if (bw * bh) < (MIN_FACE_FRAC * frame_h * frame_w):
        return None

    return (x, y, bw, bh)


def detect_face_crop_with_bbox(frame, target_size: int = 224):
    """
    Detect the best frontal face and return a padded square crop plus bbox.

    Returns:
        (crop, bbox) where bbox = (x, y, w, h) in original frame coords,
        or (None, None) when no face passes all quality gates.
    """
    frame_h, frame_w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects, _, weights = _face_cascade.detectMultiScale3(
        gray,
        scaleFactor=_SCALE_FACTOR,
        minNeighbors=5,
        minSize=_MIN_SIZE,
        outputRejectLevels=True,
    )

    best = _pick_best_face(rects, weights, frame_h, frame_w)
    if best is None:
        return None, None

    x, y, fw, fh = best
    pad = int(max(fw, fh) * 0.15)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame_w, x + fw + pad)
    y2 = min(frame_h, y + fh + pad)
    crop = frame[y1:y2, x1:x2]

    crop = cv2.resize(
        crop, (target_size, target_size), interpolation=cv2.INTER_AREA
    )
    return crop, (x, y, fw, fh)


def detect_face_crop(frame, target_size: int = 224):
    """Return padded 224×224 face crop, or None if no face is found."""
    crop, _ = detect_face_crop_with_bbox(frame, target_size=target_size)
    return crop


def detect_faces(frame):
    """
    Backward-compatible alias: return face crop only (no bbox).

    Used by main.py and scripts/calibrate_cnn.py.
    """
    return detect_face_crop(frame, target_size=224)
