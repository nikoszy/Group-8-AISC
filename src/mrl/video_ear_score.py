"""Video-level MRL blink-rate → ear_score for live inference."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_MRL_AVAILABLE = False
_EyeCropper = None
_classify_frame = None
_BlinkCounter = None
_bpm_to_confidence = None

try:
    from src.mrl.inference import EyeCropper, BlinkCounter, classify_frame
    from src.mrl.score import bpm_to_confidence

    _EyeCropper = EyeCropper
    _classify_frame = classify_frame
    _BlinkCounter = BlinkCounter
    _bpm_to_confidence = bpm_to_confidence
    _MRL_AVAILABLE = True
except ImportError as exc:
    logger.debug("MRL not available for video_ear_score: %s", exc)


def mrl_available() -> bool:
    return _MRL_AVAILABLE


def video_ear_score(
    frames_bgr: list[np.ndarray | None],
    duration_sec: float,
    mrl_model: Any = None,
    mrl_img_size: int = 84,
    mrl_idx_to_label: dict | None = None,
    mrl_device: Any = None,
    *,
    use_full_frames: bool = True,
) -> float:
    """
    Run Module 1 blink detection on a sequence of BGR frames.

    Counts blink transitions (open → closed → open), extrapolates to
    blinks-per-minute, then maps via bpm_to_confidence() to a 0–1 fake score.

    Returns 0.5 (neutral) when MRL is unavailable, the model is missing,
    fewer than 2 valid frames are supplied, or any error occurs.

    Args:
        frames_bgr: BGR frames — full video frames or face crops (both work
                    with MediaPipe EyeCropper).
        duration_sec: video duration used for BPM extrapolation.
        use_full_frames: kept for API compatibility; EyeCropper accepts either.
    """
    del use_full_frames  # both full frames and face crops are supported

    if not _MRL_AVAILABLE or mrl_model is None:
        return 0.5

    valid = [f for f in frames_bgr if f is not None]
    if len(valid) < 2:
        return 0.5

    if mrl_idx_to_label is None:
        mrl_idx_to_label = {}

    try:
        blink_counter = _BlinkCounter()

        with _EyeCropper() as cropper:
            for frame_bgr in valid:
                try:
                    crops = cropper.crop_eyes(frame_bgr, img_size=mrl_img_size)
                    eye_state = _classify_frame(
                        mrl_model, crops, mrl_idx_to_label, mrl_device
                    )
                except Exception as frame_exc:
                    logger.debug("MRL frame error (skipping): %s", frame_exc)
                    eye_state = "unknown"
                blink_counter.update(eye_state)

        blink_count = blink_counter.blink_count
        if duration_sec > 0:
            bpm = blink_count / (duration_sec / 60.0)
        else:
            bpm = blink_count * 20.0

        ear = float(_bpm_to_confidence(bpm))
        logger.debug(
            "MRL: %d blinks in %.1fs → %.1f BPM → ear_score=%.4f",
            blink_count, duration_sec, bpm, ear,
        )
        return ear

    except Exception as exc:
        logger.warning("video_ear_score failed: %s — using 0.5", exc)
        return 0.5
