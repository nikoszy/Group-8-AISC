"""Shared video-level score combination for predict.py and backend."""

from __future__ import annotations

import os
import pickle
from typing import Any

import cv2
import numpy as np


def load_cnn_alpha(bundle_path: str, fallback: float = 0.65) -> float:
    """Load CNN blend weight from stacking_bundle.pkl."""
    if not os.path.exists(bundle_path):
        return fallback
    try:
        with open(bundle_path, "rb") as fh:
            sb = pickle.load(fh)
        if sb.get("alpha_reliable", False):
            return float(sb["alpha"])
        return fallback
    except Exception:
        return fallback


def sample_temporal_burst(
    video_path: str,
    fps: float,
    *,
    burst_seconds: float = 2.0,
    max_frames: int = 60,
) -> dict | None:
    """
    Sample a dense consecutive frame burst from the video centre for optical flow.

    Returns the dict from temporal_consistency_score(), or None on failure.
    """
    try:
        from src.temporal_scorer import temporal_consistency_score
    except ImportError:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_t = cap.get(cv2.CAP_PROP_FPS) or fps
        n_burst = min(int(fps_t * burst_seconds), max_frames)
        start = max(0, total // 2 - n_burst // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        burst_frames = []
        for _ in range(n_burst):
            ret, frm = cap.read()
            if ret:
                burst_frames.append(frm)
        return temporal_consistency_score(burst_frames)
    except Exception:
        return None
    finally:
        cap.release()


def run_rppg(face_crops: list, fps: float) -> dict | None:
    """Run rPPG pulse check; returns scorer dict or None."""
    try:
        from src.rppg_scorer import rppg_score
    except ImportError:
        return None

    valid = [c for c in face_crops if c is not None]
    try:
        return rppg_score(valid, fps=fps)
    except Exception:
        return None


def combine_video_score(
    qw_prob: float,
    temporal_result: dict | None,
    rppg_result: dict | None,
    *,
    temporal_weight: float = 0.15,
    rppg_weight: float = 0.10,
) -> tuple[float, int]:
    """
    Combine quality-weighted frame mean with temporal and rPPG nudges.

    Formula (matches predict.py):
        combined = qw + w_t * (temporal - qw) + w_r * (rppg - qw), clipped [0, 1]

    Returns:
        (combined_prob, signal_count) where signal_count starts at 1 (qw always).
    """
    combined = qw_prob
    signal_count = 1

    if temporal_result and temporal_result.get("available"):
        combined = combined + temporal_weight * (
            float(temporal_result["score"]) - combined
        )
        signal_count += 1

    if rppg_result and rppg_result.get("available"):
        combined = combined + rppg_weight * (
            float(rppg_result["fake_score"]) - combined
        )
        signal_count += 1

    return float(np.clip(combined, 0.0, 1.0)), signal_count


def module_scores_dict(
    lr_probs: list[float],
    cnn_probs: list[float | None],
    temporal_result: dict | None,
    rppg_result: dict | None,
) -> dict[str, Any]:
    """Build module_scores payload for API responses."""
    temporal_val = None
    if temporal_result and temporal_result.get("available"):
        temporal_val = round(float(temporal_result["score"]), 4)

    rppg_val = None
    if rppg_result and rppg_result.get("available"):
        rppg_val = round(float(rppg_result["fake_score"]), 4)

    return {
        "cnn": round(float(np.mean(cnn_probs)), 4) if cnn_probs else None,
        "lr": round(float(np.mean(lr_probs)), 4) if lr_probs else None,
        "temporal": temporal_val,
        "rppg": rppg_val,
    }
