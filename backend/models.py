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
    prob_fake: float = Field(..., description="Ensemble fake probability [0, 1]")
    ear_score: float = Field(
        0.5,
        description=(
            "Module 1 video-level blink/EAR suspicion [0, 1] — higher = more "
            "abnormal blink pattern; same value on all frames for one upload"
        ),
    )
    artifact_score: float = Field(..., description="JPEG recompression artifact score [0, 1]")
    fft_score: float = Field(..., description="FFT spectral slope anomaly score [0, 1]")
    laplacian_score: float = Field(..., description="Laplacian texture sharpness score [0, 1]")
    face_detected: bool = Field(..., description="Whether a face was detected in this frame")
    face_crop_b64: Optional[str] = Field(
        None, description="Base64-encoded JPEG of the face crop, or null if no face"
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
    threshold: float = Field(
        ...,
        description="Calibrated fake probability threshold used for verdicting",
    )
    uncertain_band: float = Field(
        ...,
        description="Symmetric half-band around threshold used for UNCERTAIN verdicts",
    )
    verdict_hi: float = Field(..., description="Upper FAKE cutoff = threshold + uncertain_band")
    verdict_lo: float = Field(..., description="Lower REAL cutoff = threshold - uncertain_band")

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
        description=(
            'One of "ensemble_learned", "equal_weights", "cnn_fallback", '
            'or "cnn_fallback_degraded"'
        ),
    )
    cnn_active: bool = Field(
        False,
        description="True when CNN fallback path is selected for this analysis",
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
