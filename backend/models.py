"""
Pydantic v2 response models for the deepfake detector API.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class FrameResult(BaseModel):
    """Per-frame analysis result for one sampled video frame."""

    frame_index: int = Field(..., description="0-based index within sampled frames")
    timestamp_sec: float = Field(..., description="Position in the video (seconds)")
    prob_fake: float = Field(..., description="Blended ensemble fake probability [0, 1]")
    ear_score: float = Field(0.5, description="MRL blink-rate score [0,1] (video-level)")
    artifact_score: float = Field(..., description="JPEG recompression artifact score [0, 1]")
    fft_score: float = Field(..., description="FFT spectral slope anomaly score [0, 1]")
    laplacian_score: float = Field(..., description="Laplacian texture sharpness score [0, 1]")
    quality_score: float = Field(0.0, description="Frame quality score [0, 1] for weighting")
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

    video_name: str = Field(..., description="Original uploaded filename")

    verdict: str = Field(..., description='"FAKE", "REAL", or "UNCERTAIN"')
    confidence: float = Field(
        ...,
        description="How far from the 0.5 boundary: abs(final_prob_fake - 0.5) * 2",
    )
    prob_fake_mean: float = Field(
        ..., description="Simple mean of per-frame prob_fake across detected faces"
    )
    final_prob_fake: float = Field(
        ...,
        description="Final P(fake) after quality weighting + temporal/rPPG nudges [0, 1]",
    )

    quality_weighted_prob_fake: float = Field(
        ...,
        description="Quality-weighted mean P(fake) before temporal/rPPG nudges",
    )
    temporal_score: float = Field(
        0.5,
        description="Optical-flow temporal fake score [0, 1]; 0.5 when unavailable",
    )
    temporal_available: bool = Field(
        False, description="True when optical-flow temporal signal was computed"
    )
    rppg_fake_score: float = Field(
        0.5, description="rPPG-based fake score; 0.5 when unavailable"
    )
    rppg_available: bool = Field(False, description="True when rPPG pulse check ran")

    model_used: str = Field(
        ...,
        description='"ensemble_learned" if data/ensemble_model.pkl was loaded, else "equal_weights"',
    )
    cnn_active: bool = Field(False, description="True when EfficientNet-B0 contributed")

    model_id: str = Field("unknown", description="Registry model_id for this verdict")
    model_type: str = Field("equal_weights", description="lr | cnn | stacked | equal_weights")
    model_f1: Optional[float] = Field(None, description="Validation F1 from registry")

    frames_analyzed: int = Field(..., description="Frames where a face was detected")
    frames_sampled: int = Field(..., description="Total frames attempted")
    fps: float = Field(..., description="Video frame rate reported by cv2")
    duration_sec: float = Field(..., description="Total video duration in seconds")

    frames: List[FrameResult] = Field(..., description="Per-frame analysis results")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal issues")
