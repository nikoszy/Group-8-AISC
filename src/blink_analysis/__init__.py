# Module 1 — video-level EAR / blink analysis
from src.blink_analysis.ear_scorer import (
    compute_frame_ear,
    compute_video_ear_score,
    detect_blinks,
    load_video_ear_scores,
    resolve_source_video_path,
    write_video_ear_scores_csv,
)

__all__ = [
    "compute_frame_ear",
    "compute_video_ear_score",
    "detect_blinks",
    "load_video_ear_scores",
    "resolve_source_video_path",
    "write_video_ear_scores_csv",
]
