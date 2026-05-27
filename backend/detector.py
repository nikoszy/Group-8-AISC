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


def _score_face(
    face: np.ndarray,
    model: Any,
    scaler: Any,
    frame_idx: int,
) -> tuple[float, float, float, float, float]:
    """
    Run all four feature scorers on a single face crop.

    Returns (prob_fake, ear_score, artifact, fft, laplacian).
    Any scorer that raises is logged and replaced with a neutral fallback.
    """
    ear_score = 0.5  # stub until Module 1 is integrated

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
            prob_fake = ensemble_score_learned(model, scaler, ear_score, artifact, fft, lap)
        except Exception as exc:
            logger.warning("Frame %d ensemble_score_learned error: %s", frame_idx, exc)
            prob_fake = ensemble_score_equal_weights(ear_score, artifact, fft, lap)
    else:
        prob_fake = ensemble_score_equal_weights(ear_score, artifact, fft, lap)

    return prob_fake, ear_score, artifact, fft, lap


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_video(
    video_path: str,
    model: Any = None,
    scaler: Any = None,
    n_frames: int = 12,
) -> dict:
    """
    Analyze a video file for deepfake signals.

    Seeks to n_frames evenly-spaced positions in the video, runs face detection
    on each, scores each detected face with the existing module scorers, and
    returns an aggregate result dict matching the AnalysisResponse schema.

    Args:
        video_path: Absolute path to a video file (temp file on disk).
        model:      Trained sklearn LogisticRegression (or None for equal-weights).
        scaler:     Fitted sklearn StandardScaler (or None for equal-weights).
        n_frames:   Number of frames to sample from the video.

    Returns:
        A dict matching AnalysisResponse (models.py).

    Raises:
        ValueError: If no faces were detected in any sampled frame.
        IOError:    Propagated from load_video() if cv2 cannot open the file.
    """
    cap = load_video(video_path)

    # Video metadata ─────────────────────────────────────────────────────────
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    duration_sec = (total_frames / fps) if total_frames > 0 and fps > 0 else 0.0

    # Compute evenly-spaced seek positions ───────────────────────────────────
    # Guard against videos with unknown/zero frame count by sampling sequentially.
    if total_frames > 0:
        positions = [int(i * total_frames / n_frames) for i in range(n_frames)]
    else:
        positions = list(range(n_frames))

    frame_results: list[dict] = []
    no_face_count = 0
    warnings_list: list[str] = []

    # Per-frame loop ─────────────────────────────────────────────────────────
    for idx, pos in enumerate(positions):
        timestamp = round(pos / fps, 3) if fps > 0 else 0.0

        # Seek and read the frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(pos))
        ok, frame = cap.read()

        if not ok or frame is None:
            no_face_count += 1
            warnings_list.append(f"Frame {idx}: seek to position {pos} failed")
            frame_results.append(
                {
                    "frame_index": idx,
                    "timestamp_sec": timestamp,
                    "prob_fake": 0.5,
                    "ear_score": 0.5,
                    "artifact_score": 0.0,
                    "fft_score": 0.0,
                    "laplacian_score": 0.0,
                    "face_detected": False,
                    "face_crop_b64": None,
                }
            )
            continue

        face = detect_faces(frame)

        if face is None:
            no_face_count += 1
            frame_results.append(
                {
                    "frame_index": idx,
                    "timestamp_sec": timestamp,
                    "prob_fake": 0.5,
                    "ear_score": 0.5,
                    "artifact_score": 0.0,
                    "fft_score": 0.0,
                    "laplacian_score": 0.0,
                    "face_detected": False,
                    "face_crop_b64": None,
                }
            )
            continue

        # Score the face crop ────────────────────────────────────────────────
        prob_fake, ear, artifact, fft, lap = _score_face(face, model, scaler, idx)
        face_b64 = _encode_face_b64(face)

        frame_results.append(
            {
                "frame_index": idx,
                "timestamp_sec": timestamp,
                "prob_fake": prob_fake,
                "ear_score": ear,
                "artifact_score": round(artifact, 4),
                "fft_score": round(fft, 4),
                "laplacian_score": round(lap, 4),
                "face_detected": True,
                "face_crop_b64": face_b64,
            }
        )

    cap.release()

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
    # Higher laplacian = sharper frame = more informative.
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
        "cnn_active": False,
        "frames_analyzed": len(detected_frames),
        "frames_sampled": n_frames,
        "fps": round(fps, 2),
        "duration_sec": round(duration_sec, 2),
        "frames": frame_results,
        "warnings": warnings_list,
    }
