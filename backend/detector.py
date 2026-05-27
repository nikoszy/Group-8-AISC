"""
backend/detector.py — Single-video analysis orchestrator.

Wraps the existing deepfake detector modules (imported from the repo root)
and provides the `analyze_video()` function called by the FastAPI endpoint.

IMPORTANT: This file does NOT modify any existing detector code.
It only imports and calls functions from the existing modules.
"""

from __future__ import annotations

import base64
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so we can import the existing modules.
# backend/detector.py lives one level below the repo root.
# ---------------------------------------------------------------------------
_REPO_ROOT = str(Path(__file__).parent.parent.resolve())
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Import existing detector modules (DO NOT modify these files)
from src.preprocessing.video_loader import load_video        # noqa: E402
from src.preprocessing.face_detector import detect_faces     # noqa: E402
from artifact_module import get_artifact_score_for_frame     # noqa: E402
from src.freq_analysis.anomaly_scorer import fft_anomaly_score  # noqa: E402
from src.freq_analysis.texture_scorer import laplacian_score    # noqa: E402
from ensemble import ensemble_score_learned, ensemble_score_equal_weights  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module 1 (MRL blink detection) — soft dependency
# ---------------------------------------------------------------------------
# We try to import at module load time so we can log once whether it's available.
# All MRL imports are wrapped in try/except so missing MediaPipe / MobileNetV2
# checkpoint degrades gracefully to ear_score = 0.5 instead of crashing.

_MRL_AVAILABLE = False
_EyeCropper = None
_classify_frame = None
_BlinkCounter = None
_bpm_to_confidence = None

try:
    from src.mrl.inference import EyeCropper, classify_frame, BlinkCounter  # noqa: E402
    from src.mrl.score import bpm_to_confidence                              # noqa: E402
    _EyeCropper       = EyeCropper
    _classify_frame   = classify_frame
    _BlinkCounter     = BlinkCounter
    _bpm_to_confidence = bpm_to_confidence
    _MRL_AVAILABLE    = True
    logger.info("Module 1 (MRL blink detection) loaded successfully.")
except ImportError as _mrl_err:
    logger.info(
        "Module 1 (MRL) not available (%s) — ear_score will be 0.5 for all videos.",
        _mrl_err,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_face_b64(face_bgr: np.ndarray) -> Optional[str]:
    """
    Encode a BGR face-crop array as a base64 JPEG string.

    Returns None if encoding fails.
    """
    try:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
        success, buf = cv2.imencode(".jpg", face_bgr, encode_params)
        if not success:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception as exc:
        logger.warning("Face crop encoding failed: %s", exc)
        return None


def _get_ear_score_for_video(
    frames_bgr: list[np.ndarray],
    duration_sec: float,
    mrl_model: Any,
    mrl_img_size: int,
    mrl_idx_to_label: dict,
    mrl_device: Any,
) -> float:
    """
    Run Module 1 (MRL blink detection) on a list of BGR face frames.

    Counts blink transitions (open → closed → open) across the sampled
    frames, extrapolates to blinks-per-minute using video duration, then
    applies the sigmoid bpm_to_confidence() to get a 0–1 fake score.

    Returns 0.5 (neutral) if:
    - MRL libraries are not installed
    - MRL model checkpoint was not loaded
    - Fewer than 2 frames are available
    - Any exception occurs during processing

    Args:
        frames_bgr      : list of BGR frames (face crops OK, full frames OK)
        duration_sec    : total video duration used to compute BPM
        mrl_model       : loaded MobileNetV2 (from load_model()) or None
        mrl_img_size    : expected input size (e.g. 84)
        mrl_idx_to_label: {0: "sleepy", 1: "awake"} label map
        mrl_device      : torch device

    Returns:
        float [0, 1] — deepfake_confidence (high = likely fake / unnatural blink)
    """
    if not _MRL_AVAILABLE or mrl_model is None or len(frames_bgr) < 2:
        return 0.5

    try:
        blink_counter = _BlinkCounter()

        with _EyeCropper() as cropper:
            for frame_bgr in frames_bgr:
                if frame_bgr is None:
                    continue
                try:
                    crops = cropper.crop_eyes(frame_bgr, img_size=mrl_img_size)
                    eye_state = _classify_frame(
                        mrl_model, crops, mrl_idx_to_label, mrl_device
                    )
                except Exception as frame_exc:
                    logger.debug("MRL frame error (skipping frame): %s", frame_exc)
                    eye_state = "unknown"
                blink_counter.update(eye_state)

        blink_count = blink_counter.blink_count
        # Compute BPM: if we have duration, extrapolate; otherwise use raw count
        if duration_sec > 0:
            bpm = blink_count / (duration_sec / 60.0)
        else:
            # Rough fallback: assume 1 blink per second for a 3-second clip
            bpm = blink_count * 20.0

        ear_score = _bpm_to_confidence(bpm)
        logger.debug(
            "MRL: %d blinks in %.1fs → %.1f BPM → ear_score=%.4f",
            blink_count, duration_sec, bpm, ear_score,
        )
        return float(ear_score)

    except Exception as exc:
        logger.warning("MRL ear_score computation failed: %s — using 0.5", exc)
        return 0.5


def _score_face(
    face: np.ndarray,
    model: Any,
    scaler: Any,
    frame_idx: int,
    ear_score: float = 0.5,
) -> tuple[float, float, float, float, float]:
    """
    Run all four feature scorers on a single face crop.

    ear_score is computed once per video and passed in (not re-computed
    per frame) because blink detection requires the full frame sequence.

    Returns (prob_fake, ear_score, artifact, fft, laplacian).
    Any scorer that raises is logged and replaced with a neutral fallback.
    """
    try:
        artifact = float(get_artifact_score_for_frame(face))
    except Exception as exc:
        logger.warning("Frame %d artifact scorer error: %s", frame_idx, exc)
        artifact = 0.0

    try:
        fft = float(fft_anomaly_score(face))
    except Exception as exc:
        logger.warning("Frame %d FFT scorer error: %s", frame_idx, exc)
        fft = 0.0

    try:
        lap = float(laplacian_score(face))
    except Exception as exc:
        logger.warning("Frame %d Laplacian scorer error: %s", frame_idx, exc)
        lap = 0.0

    if model is not None and scaler is not None:
        try:
            # arg order: (model, scaler, artifact, fft, laplacian, ear_score)
            prob_fake = ensemble_score_learned(model, scaler, artifact, fft, lap, ear_score)
        except Exception as exc:
            logger.warning("Frame %d ensemble_score_learned error: %s", frame_idx, exc)
            prob_fake = ensemble_score_equal_weights(artifact, fft, lap)
    else:
        prob_fake = ensemble_score_equal_weights(artifact, fft, lap)

    return prob_fake, ear_score, artifact, fft, lap


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_video(
    video_path: str,
    model: Any = None,
    scaler: Any = None,
    n_frames: int = 12,
    app_state: dict | None = None,
    mrl_model: Any = None,
    mrl_img_size: int = 84,
    mrl_idx_to_label: dict | None = None,
    mrl_device: Any = None,
) -> dict:
    """
    Analyze a video file for deepfake signals.

    Seeks to n_frames evenly-spaced positions in the video, runs face detection
    on each, scores each detected face with the existing module scorers, and
    returns an aggregate result dict matching the AnalysisResponse schema.

    Module 1 (MRL blink detection) is run across all sampled frames to produce
    a single video-level ear_score, then that score is passed into the LR
    ensemble for every frame. If the MRL model is unavailable, ear_score = 0.5.

    Args:
        video_path       : Absolute path to a video file (temp file on disk).
        model            : Trained sklearn model (or None for equal-weights).
        scaler           : Fitted sklearn StandardScaler (or None).
        n_frames         : Number of frames to sample from the video.
        app_state        : Dict of registry metadata (model_id, model_type,
                           model_f1) to pass through to the response.
        mrl_model        : Loaded MobileNetV2 checkpoint or None.
        mrl_img_size     : Eye crop size expected by the MRL model (default 84).
        mrl_idx_to_label : {0: "sleepy", 1: "awake"} from load_model().
        mrl_device       : torch device for MRL inference.

    Returns:
        A dict matching AnalysisResponse (models.py).

    Raises:
        ValueError: If no faces were detected in any sampled frame.
        IOError:    Propagated from load_video() if cv2 cannot open the file.
    """
    if app_state is None:
        app_state = {}
    if mrl_idx_to_label is None:
        mrl_idx_to_label = {}

    cap = load_video(video_path)

    # Video metadata ─────────────────────────────────────────────────────────
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    duration_sec = (total_frames / fps) if total_frames > 0 and fps > 0 else 0.0

    # Compute evenly-spaced seek positions ───────────────────────────────────
    if total_frames > 0:
        positions = [int(i * total_frames / n_frames) for i in range(n_frames)]
    else:
        positions = list(range(n_frames))

    # ── Pass 1: collect face crops for all frames ────────────────────────────
    # We need all BGR face crops before calling _get_ear_score_for_video()
    # so that the blink counter sees the full temporal sequence.
    raw_frames: list[dict] = []   # {pos, timestamp, face_bgr | None, ok}

    for idx, pos in enumerate(positions):
        timestamp = round(pos / fps, 3) if fps > 0 else 0.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(pos))
        ok, frame = cap.read()

        if not ok or frame is None:
            raw_frames.append({"idx": idx, "pos": pos, "timestamp": timestamp,
                                "face_bgr": None, "read_ok": False})
            continue

        face = detect_faces(frame)
        raw_frames.append({"idx": idx, "pos": pos, "timestamp": timestamp,
                            "face_bgr": face, "read_ok": True})

    cap.release()

    # ── Module 1: compute video-level ear_score ──────────────────────────────
    detected_bgrs = [r["face_bgr"] for r in raw_frames if r["face_bgr"] is not None]
    ear_score_video = _get_ear_score_for_video(
        frames_bgr=detected_bgrs,
        duration_sec=duration_sec,
        mrl_model=mrl_model,
        mrl_img_size=mrl_img_size,
        mrl_idx_to_label=mrl_idx_to_label,
        mrl_device=mrl_device,
    )
    logger.info("Video-level ear_score (MRL): %.4f", ear_score_video)

    # ── Pass 2: score each frame using the pre-computed ear_score ────────────
    frame_results: list[dict] = []
    no_face_count = 0
    warnings_list: list[str] = []

    for r in raw_frames:
        idx = r["idx"]
        timestamp = r["timestamp"]
        pos = r["pos"]

        if not r["read_ok"]:
            no_face_count += 1
            warnings_list.append(f"Frame {idx}: seek to position {pos} failed")
            frame_results.append({
                "frame_index": idx, "timestamp_sec": timestamp,
                "prob_fake": 0.5, "ear_score": 0.5,
                "artifact_score": 0.0, "fft_score": 0.0, "laplacian_score": 0.0,
                "face_detected": False, "face_crop_b64": None,
            })
            continue

        face = r["face_bgr"]
        if face is None:
            no_face_count += 1
            frame_results.append({
                "frame_index": idx, "timestamp_sec": timestamp,
                "prob_fake": 0.5, "ear_score": 0.5,
                "artifact_score": 0.0, "fft_score": 0.0, "laplacian_score": 0.0,
                "face_detected": False, "face_crop_b64": None,
            })
            continue

        prob_fake, ear, artifact, fft, lap = _score_face(
            face, model, scaler, idx, ear_score=ear_score_video
        )
        face_b64 = _encode_face_b64(face)

        frame_results.append({
            "frame_index": idx, "timestamp_sec": timestamp,
            "prob_fake": prob_fake, "ear_score": round(ear, 4),
            "artifact_score": round(artifact, 4),
            "fft_score": round(fft, 4),
            "laplacian_score": round(lap, 4),
            "face_detected": True, "face_crop_b64": face_b64,
        })

    # Guard: at least one face must have been detected ────────────────────────
    detected_frames = [f for f in frame_results if f["face_detected"]]
    if not detected_frames:
        raise ValueError(
            f"No faces detected in any of the {n_frames} sampled frames. "
            "Try a different video or increase n_frames."
        )

    if no_face_count > 0:
        warnings_list.append(
            f"{no_face_count}/{n_frames} sampled frames had no detectable face"
        )

    # Aggregates ─────────────────────────────────────────────────────────────
    prob_fakes = [f["prob_fake"] for f in detected_frames]
    lap_scores = [f["laplacian_score"] for f in detected_frames]

    prob_fake_mean = round(float(np.mean(prob_fakes)), 4)

    # Quality-weighted mean: laplacian_score as quality weight.
    lap_sum = sum(lap_scores)
    if lap_sum > 1e-6:
        quality_weighted_prob_fake = round(
            float(np.average(prob_fakes, weights=lap_scores)), 4
        )
    else:
        quality_weighted_prob_fake = prob_fake_mean

    temporal_score = round(float(np.std(prob_fakes)), 4)
    confidence = round(abs(prob_fake_mean - 0.5) * 2, 4)

    # Verdict thresholds (0.6 / 0.4) ─────────────────────────────────────────
    if prob_fake_mean >= 0.6:
        verdict = "FAKE"
    elif prob_fake_mean <= 0.4:
        verdict = "REAL"
    else:
        verdict = "UNCERTAIN"

    model_used = (
        "ensemble_learned" if (model is not None and scaler is not None)
        else "equal_weights"
    )

    # Registry metadata (passed in from main.py via app_state) ───────────────
    model_id   = app_state.get("active_model_id",   "unknown")
    model_type = app_state.get("active_model_type", model_used)
    model_f1   = app_state.get("active_model_f1",   None)

    return {
        "video_name": os.path.basename(video_path),
        "verdict": verdict,
        "confidence": confidence,
        "prob_fake_mean": prob_fake_mean,
        "quality_weighted_prob_fake": quality_weighted_prob_fake,
        "temporal_score": temporal_score,
        "rppg_fake_score": 0.5,
        "rppg_available": False,
        "model_used": model_used,
        "model_id":   model_id,
        "model_type": model_type,
        "model_f1":   model_f1,
        "cnn_active": False,
        "frames_analyzed": len(detected_frames),
        "frames_sampled": n_frames,
        "fps": round(fps, 2),
        "duration_sec": round(duration_sec, 2),
        "frames": frame_results,
        "warnings": warnings_list,
    }
