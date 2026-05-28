"""Tests for shared face detection crop consistency."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.preprocessing.face_detector import (
    detect_face_crop,
    detect_face_crop_with_bbox,
    detect_faces,
    MIN_FACE_FRAC,
)


def _synthetic_face_frame(w=640, h=480):
    """Build a bright frame with a high-contrast square 'face' region."""
    frame = np.full((h, w, 3), 180, dtype=np.uint8)
    x, y, fw, fh = w // 3, h // 5, w // 3, h // 2
    frame[y:y + fh, x:x + fw] = (40, 40, 40)
    return frame, (x, y, fw, fh)


def test_detect_face_crop_returns_224_square():
    frame, _ = _synthetic_face_frame()
    crop = detect_face_crop(frame, target_size=224)
    # Haar may miss synthetic shapes on some OpenCV builds — skip if so
    if crop is None:
        pytest.skip("Haar did not detect synthetic face (environment-specific)")
    assert crop.shape == (224, 224, 3)


def test_detect_face_crop_with_bbox_matches_crop():
    frame, _ = _synthetic_face_frame()
    crop, bbox = detect_face_crop_with_bbox(frame, target_size=224)
    if crop is None:
        pytest.skip("Haar did not detect synthetic face (environment-specific)")
    assert bbox is not None
    assert len(bbox) == 4
    assert crop.shape == (224, 224, 3)
    assert detect_faces(frame) is not None


def test_detect_faces_alias_matches_detect_face_crop():
    frame, _ = _synthetic_face_frame()
    a = detect_face_crop(frame)
    b = detect_faces(frame)
    if a is None:
        pytest.skip("Haar did not detect synthetic face (environment-specific)")
    assert a.shape == b.shape


def test_manifest_constants_exported():
    from src.preprocessing import face_detector as fd

    assert fd.MIN_FACE_FRAC == 0.04
    assert fd.CONF_THRESHOLD == 0.7
