"""
Pydantic v2 response models for the deepfake detector API.

These mirror the JSON schema defined in the plan's API contract.
FastAPI uses them for automatic serialization and OpenAPI docs.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class FrameResult(BaseModel):
    """Per-frame analysis result for one sampled video frame."""

    frame_index: int = Field(..., description="0-based index within sampled frames")
    timestamp_sec: float = Field(..., description="Position in the video (seconds)")
    prob_fake: float = Field(..., description="Blended ensemble fake probability [0, 1]")
    ear_score: float = Field(0.5, description="MRL blink-rate score [0,1] (video-level; 0.5 if MRL unavailable)")
    artifact_score: float = Field(..., description="JPEG recompression artifact score [0, 1]")
    fft_score: float = Field(..., description="FFT spectral slope anomaly score [0, 1]")
    laplacian_score: float = Field(..., description="Laplacian texture sharpness score [0, 1]")
    face_detected: bool = Field(..., description="Whether a face was detected in this frame")
    face_crop_b64: Optional[str] = Field(
        None, description="Base64-encoded JPEG of the face crop, or null if no face"
    )
    lr_prob: Optional[float] = Field(
        None, description="LR ensemble P(fake) before CNN blend [0, 1]"
    )
    cnn_prob: Optional[float] = Field(
        None, description="CNN EfficientNet-B0 P(fake) [0, 1], or null if CNN not active"
    )


class AnalysisResponse(BaseModel):
    """Complete analysis result for one uploaded video."""

    # ── Identity ─────────────────────────────────────────────────────────────
    video_name: str = Field(..., description="Original uploaded filename")

    # ── Top-level verdict ────────────────────────────────────────────────────
    verdict: str = Field(..., description='"FAKE", "REAL", or "UNCERTAIN"')
    confidence: float = Field(
        ...,
        description="How far from the 0.5 boundary: abs(prob_fake_mean - 0.5) * 2",
    )
    prob_fake_mean: float = Field(
        ..., description="Simple mean of per-frame prob_fake across detected faces"
    )

    # ── Signal breakdown ─────────────────────────────────────────────────────
    quality_weighted_prob_fake: float = Field(
        ...,
        description="prob_fake weighted by laplacian_score (higher laplacian = sharper = more trustworthy)",
    )
    temporal_score: float = Field(
        ...,
        description="Std-dev of per-frame prob_fake — high value = inconsistent across frames",
    )
    rppg_fake_score: float = Field(
        0.5, description="rPPG-based fake score (stub — not implemented)"
    )
    rppg_available: bool = Field(False, description="Always false — rPPG not integrated")

    # ── Model metadata ───────────────────────────────────────────────────────
    model_used: str = Field(
        ...,
        description='"ensemble_learned" if data/ensemble_model.pkl was loaded, else "equal_weights"',
    )
    cnn_active: bool = Field(False, description="True when EfficientNet-B0 contributed to the verdict")

    # ── Registry provenance (new in v2) ──────────────────────────────────────
    model_id: str = Field(
        "unknown",
        description="Registry model_id of the model that produced this verdict",
    )
    model_type: str = Field(
        "equal_weights",
        description="lr | cnn | stacked | stacked_with_blink | equal_weights",
    )
    model_f1: Optional[float] = Field(
        None,
        description="Validation F1 of the active model from the registry (None if unavailable)",
    )

    # ── Frame counts + video metadata ────────────────────────────────────────
    frames_analyzed: int = Field(..., description="Frames where a face was detected")
    frames_sampled: int = Field(..., description="Total frames attempted (= n_frames param)")
    fps: float = Field(..., description="Video frame rate reported by cv2")
    duration_sec: float = Field(..., description="Total video duration in seconds")

    # ── Per-frame data ───────────────────────────────────────────────────────
    frames: List[FrameResult] = Field(..., description="Per-frame analysis results")

    # ── Warnings ─────────────────────────────────────────────────────────────
    warnings: List[str] = Field(
        default_factory=list, description="Non-fatal processing issues"
    )
