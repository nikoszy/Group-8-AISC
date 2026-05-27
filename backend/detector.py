"""
backend/detector.py — Single-video analysis orchestrator.

Wraps the existing deepfake detector modules and provides ``analyze_video()``
used by the FastAPI endpoints.  Scoring logic matches predict.py via
``src/inference_combine.py`` and shared face detection.
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

_REPO_ROOT = str(Path(__file__).parent.parent.resolve())
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.preprocessing.video_loader import load_video  # noqa: E402
from src.preprocessing.face_detector import detect_face_crop_with_bbox  # noqa: E402
from src.quality_scorer import compute_frame_quality, quality_weighted_mean  # noqa: E402
from src.inference_combine import (  # noqa: E402
    sample_temporal_burst,
    run_rppg,
    combine_video_score,
    module_scores_dict,
)
from src.mrl.video_ear_score import video_ear_score  # noqa: E402
from artifact_module import get_artifact_score_for_frame  # noqa: E402
from src.freq_analysis.anomaly_scorer import fft_anomaly_score  # noqa: E402
from src.freq_analysis.texture_scorer import laplacian_score  # noqa: E402
from ensemble import ensemble_score_learned, ensemble_score_equal_weights  # noqa: E402

logger = logging.getLogger(__name__)

_CNN_PREDICT_FN = None

try:
    from src.cnn_runner import cnn_predict as _cnn_predict_imported  # noqa: E402
    _CNN_PREDICT_FN = _cnn_predict_imported
    logger.info("CNN inference module loaded.")
except ImportError as _cnn_err:
    logger.info("CNN inference module not importable (%s) — CNN disabled.", _cnn_err)


def _encode_face_b64(face_bgr: np.ndarray) -> Optional[str]:
    try:
        success, buf = cv2.imencode(".jpg", face_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
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
    ear_score: float = 0.5,
    cnn_model: Any = None,
    cnn_alpha: float = 0.65,
) -> tuple[float, float, float, float, float, float, Optional[float], float]:
    """
    Score one face crop.  Returns
    (prob_fake, ear, artifact, fft, lap, lr_prob, cnn_prob, quality).
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
            lr_prob = ensemble_score_learned(model, scaler, artifact, fft, lap, ear_score)
        except Exception as exc:
            logger.warning("Frame %d ensemble_score_learned error: %s", frame_idx, exc)
            lr_prob = ensemble_score_equal_weights(artifact, fft, lap)
    else:
        lr_prob = ensemble_score_equal_weights(artifact, fft, lap)

    cnn_prob: Optional[float] = None
    if cnn_model is not None and _CNN_PREDICT_FN is not None:
        try:
            cnn_prob = _CNN_PREDICT_FN(cnn_model, face)
        except Exception as exc:
            logger.warning("Frame %d CNN inference error: %s", frame_idx, exc)

    if cnn_prob is not None:
        prob_fake = float(cnn_alpha * cnn_prob + (1.0 - cnn_alpha) * lr_prob)
    else:
        prob_fake = lr_prob

    return prob_fake, ear_score, artifact, fft, lap, lr_prob, cnn_prob, lap


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
    cnn_model: Any = None,
    cnn_alpha: float = 0.65,
    min_quality: float = 0.10,
) -> dict:
    """
    Analyze a video file for deepfake signals (parity with predict.py).

    Pipeline:
      1. Sample n_frames evenly; shared Haar face crop per frame
      2. Video-level MRL ear_score from full sampled frames
      3. Per-frame LR + CNN blend with ear in LR features
      4. Quality-weighted mean (compute_frame_quality, not laplacian alone)
      5. Temporal burst (2 s optical flow) + rPPG nudges
    """
    if app_state is None:
        app_state = {}
    if mrl_idx_to_label is None:
        mrl_idx_to_label = {}

    cap = load_video(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    duration_sec = (total_frames / fps) if total_frames > 0 and fps > 0 else 0.0

    if total_frames > 0:
        positions = [int(i * total_frames / n_frames) for i in range(n_frames)]
    else:
        positions = list(range(n_frames))

    raw_frames: list[dict] = []

    for idx, pos in enumerate(positions):
        timestamp = round(pos / fps, 3) if fps > 0 else 0.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(pos))
        ok, frame = cap.read()

        if not ok or frame is None:
            raw_frames.append({
                "idx": idx, "pos": pos, "timestamp": timestamp,
                "frame_bgr": None, "crop": None, "bbox": None, "read_ok": False,
            })
            continue

        crop, bbox = detect_face_crop_with_bbox(frame)
        raw_frames.append({
            "idx": idx, "pos": pos, "timestamp": timestamp,
            "frame_bgr": frame, "crop": crop, "bbox": bbox, "read_ok": True,
        })

    cap.release()

    sampled_bgrs = [r["frame_bgr"] for r in raw_frames if r["frame_bgr"] is not None]
    ear_score_video = video_ear_score(
        sampled_bgrs,
        duration_sec,
        mrl_model=mrl_model,
        mrl_img_size=mrl_img_size,
        mrl_idx_to_label=mrl_idx_to_label,
        mrl_device=mrl_device,
    )
    logger.info("Video-level ear_score (MRL): %.4f", ear_score_video)

    frame_results: list[dict] = []
    no_face_count = 0
    warnings_list: list[str] = []
    face_crops_for_rppg: list = []
    qualities_for_qw: list[float] = []
    probs_for_qw: list[float] = []

    for r in raw_frames:
        idx = r["idx"]
        timestamp = r["timestamp"]
        pos = r["pos"]

        if not r["read_ok"]:
            no_face_count += 1
            warnings_list.append(f"Frame {idx}: seek to position {pos} failed")
            frame_results.append({
                "frame_index": idx, "timestamp_sec": timestamp,
                "prob_fake": 0.5, "ear_score": round(ear_score_video, 4),
                "artifact_score": 0.0, "fft_score": 0.0, "laplacian_score": 0.0,
                "quality_score": 0.0,
                "face_detected": False, "face_crop_b64": None,
                "lr_prob": None, "cnn_prob": None,
            })
            continue

        crop = r["crop"]
        bbox = r["bbox"]
        frame_bgr = r["frame_bgr"]

        if crop is None or bbox is None:
            no_face_count += 1
            frame_results.append({
                "frame_index": idx, "timestamp_sec": timestamp,
                "prob_fake": 0.5, "ear_score": round(ear_score_video, 4),
                "artifact_score": 0.0, "fft_score": 0.0, "laplacian_score": 0.0,
                "quality_score": 0.0,
                "face_detected": False, "face_crop_b64": None,
                "lr_prob": None, "cnn_prob": None,
            })
            continue

        quality = compute_frame_quality(crop, bbox, frame_bgr.shape)
        prob_fake, ear, artifact, fft, lap, lr_prob, cnn_prob, _ = _score_face(
            crop, model, scaler, idx,
            ear_score=ear_score_video,
            cnn_model=cnn_model,
            cnn_alpha=cnn_alpha,
        )

        probs_for_qw.append(prob_fake)
        qualities_for_qw.append(quality)
        face_crops_for_rppg.append(crop)

        frame_results.append({
            "frame_index": idx,
            "timestamp_sec": timestamp,
            "prob_fake": round(prob_fake, 4),
            "ear_score": round(ear, 4),
            "artifact_score": round(artifact, 4),
            "fft_score": round(fft, 4),
            "laplacian_score": round(lap, 4),
            "quality_score": round(quality, 4),
            "face_detected": True,
            "face_crop_b64": _encode_face_b64(crop),
            "lr_prob": round(lr_prob, 4),
            "cnn_prob": round(cnn_prob, 4) if cnn_prob is not None else None,
        })

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

    prob_fakes = [f["prob_fake"] for f in detected_frames]
    prob_fake_mean = round(float(np.mean(prob_fakes)), 4)

    qw_prob = quality_weighted_mean(
        probs_for_qw, qualities_for_qw, min_quality=min_quality
    )
    quality_weighted_prob_fake = round(float(qw_prob), 4)

    temporal_result = sample_temporal_burst(video_path, fps)
    rppg_result = run_rppg(face_crops_for_rppg, fps=fps)
    final_prob, signal_count = combine_video_score(
        qw_prob, temporal_result, rppg_result
    )
    final_prob = round(final_prob, 4)

    temporal_score = 0.5
    temporal_available = False
    if temporal_result and temporal_result.get("available"):
        temporal_score = round(float(temporal_result["score"]), 4)
        temporal_available = True

    rppg_fake_score = 0.5
    rppg_available = False
    if rppg_result and rppg_result.get("available"):
        rppg_fake_score = round(float(rppg_result["fake_score"]), 4)
        rppg_available = True

    confidence = round(abs(final_prob - 0.5) * 2, 4)

    if final_prob >= 0.6:
        verdict = "FAKE"
    elif final_prob <= 0.4:
        verdict = "REAL"
    else:
        verdict = "UNCERTAIN"

    lr_probs = [f["lr_prob"] for f in detected_frames if f["lr_prob"] is not None]
    cnn_probs = [f["cnn_prob"] for f in detected_frames if f["cnn_prob"] is not None]
    cnn_active = cnn_model is not None and len(cnn_probs) > 0

    model_used = (
        "ensemble_learned" if (model is not None and scaler is not None)
        else "equal_weights"
    )
    if cnn_active:
        model_used = "cnn_lr_stacked"

    model_id = app_state.get("active_model_id", "unknown")
    model_type = app_state.get("active_model_type", model_used)
    model_f1 = app_state.get("active_model_f1", None)

    module_scores = module_scores_dict(lr_probs, cnn_probs, temporal_result, rppg_result)

    return {
        "video_name": os.path.basename(video_path),
        "verdict": verdict,
        "confidence": confidence,
        "prob_fake_mean": prob_fake_mean,
        "final_prob_fake": final_prob,
        "quality_weighted_prob_fake": quality_weighted_prob_fake,
        "temporal_score": temporal_score,
        "temporal_available": temporal_available,
        "temporal_detail": temporal_result,
        "rppg_fake_score": rppg_fake_score,
        "rppg_available": rppg_available,
        "rppg_detail": rppg_result,
        "ear_score_video": round(ear_score_video, 4),
        "signal_count": signal_count,
        "model_used": model_used,
        "model_id": model_id,
        "model_type": model_type,
        "model_f1": model_f1,
        "cnn_active": cnn_active,
        "frames_analyzed": len(detected_frames),
        "frames_sampled": n_frames,
        "fps": round(fps, 2),
        "duration_sec": round(duration_sec, 2),
        "frames": frame_results,
        "warnings": warnings_list,
        "module_scores": module_scores,
    }
