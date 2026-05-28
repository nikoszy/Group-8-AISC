# =============================================================================
# src/blink_analysis/ear_scorer.py — Module 1 video-level EAR / blink scoring
# =============================================================================
#
# Per-frame Eye Aspect Ratio (EAR) from facial landmarks.  Prefer MediaPipe
# Face Mesh when installed; otherwise OpenCV Haar eyes inside a face bbox.
#
# Video-level score in [0, 1]:  HIGHER = more suspicious for deepfake.
#
# CALIBRATION (documented defaults — tune on FF++ val if needed)
# ---------------------------------------------------------------
# Natural blink rate is roughly 15–20 blinks/min (~0.25–0.33 Hz).  Many
# deepfake videos show reduced or irregular blinking.
#
#   blink_rate_hz = blinks / video_duration_sec
#   rate_deficit  = max(0, REF_BLINK_RATE_HZ - blink_rate_hz) / REF_BLINK_RATE_HZ
#   ear_std_low   = low temporal std of EAR → static / pasted eyes
#
#   raw_suspicion = 0.65 * rate_deficit + 0.35 * (1 - norm_ear_std)
#   score         = clip(raw_suspicion, 0, 1)
#
# If no valid EAR frames, returns 0.5 (neutral — same as former stub).
# =============================================================================

from __future__ import annotations

import csv
import os
import re
from typing import Iterator, Optional, Sequence, Union

import cv2
import numpy as np

# MediaPipe Face Mesh eye landmark indices (6 points per eye)
_LEFT_EYE_IDX = (33, 160, 158, 133, 153, 144)
_RIGHT_EYE_IDX = (362, 385, 387, 263, 373, 380)

# Blink detection
EAR_BLINK_THRESHOLD = 0.21
BLINK_MIN_CONSEC_FRAMES = 2

# Video-level calibration
REF_BLINK_RATE_HZ = 0.20       # ~12 blinks/min reference
MIN_VALID_EAR_FRAC = 0.15      # need EAR on at least 15% of sampled frames
EAR_STD_FLOOR = 0.008          # below → suspiciously static eyes

VIDEO_EAR_CSV_DEFAULT = os.path.join("data", "video_ear_scores.csv")
FF_DATASET_DIR_CANDIDATES = [
    os.path.join("data", "archive", "FaceForensics"),
    os.path.join("data", "FaceForensics++_C23"),
]

_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

_mediapipe_face_mesh = None
_mediapipe_checked = False


def _get_face_mesh():
    """Lazy-load MediaPipe FaceMesh; return None if unavailable."""
    global _mediapipe_face_mesh, _mediapipe_checked
    if _mediapipe_checked:
        return _mediapipe_face_mesh
    _mediapipe_checked = True
    try:
        import mediapipe as mp  # type: ignore

        _mediapipe_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
    except Exception:
        _mediapipe_face_mesh = None
    return _mediapipe_face_mesh


def _dist(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


def _ear_from_six_points(pts: np.ndarray) -> float:
    """Standard EAR from six (x, y) points: vertical / horizontal ratios."""
    # pts order: p1..p6 — horizontal: p1-p4, vertical: p2-p6, p3-p5
    v1 = _dist(pts[1], pts[5])
    v2 = _dist(pts[2], pts[4])
    h = _dist(pts[0], pts[3]) + 1e-8
    return (v1 + v2) / (2.0 * h)


def _ear_mediapipe(frame_bgr: np.ndarray) -> Optional[float]:
    mesh = _get_face_mesh()
    if mesh is None:
        return None
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_bgr.shape[:2]
    result = mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None
    lm = result.multi_face_landmarks[0].landmark

    def pts(indices):
        return np.array(
            [[lm[i].x * w, lm[i].y * h] for i in indices], dtype=np.float32
        )

    left = _ear_from_six_points(pts(_LEFT_EYE_IDX))
    right = _ear_from_six_points(pts(_RIGHT_EYE_IDX))
    return (left + right) / 2.0


def _ear_haar_fallback(frame_bgr: np.ndarray) -> Optional[float]:
    """Approximate EAR from Haar face + eye boxes inside the face ROI."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    roi = gray[y : y + fh, x : x + fw]
    if roi.size == 0:
        return None
    eyes = _EYE_CASCADE.detectMultiScale(
        roi, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20)
    )
    if len(eyes) < 2:
        return None
    # Two largest eye regions — aspect ratio as pseudo-EAR
    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
    ratios = []
    for ex, ey, ew, eh in eyes:
        if ew > 0:
            ratios.append(eh / float(ew))
    if not ratios:
        return None
    # Scale to typical EAR range (~0.2–0.35 open)
    return float(np.clip(np.mean(ratios) * 0.35, 0.05, 0.45))


def compute_frame_ear(frame_bgr: np.ndarray) -> Optional[float]:
    """
    Per-frame EAR in roughly [0.05, 0.45] when eyes are visible.

    Returns None if landmarks/eyes cannot be estimated.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    ear = _ear_mediapipe(frame_bgr)
    if ear is not None:
        return float(ear)
    return _ear_haar_fallback(frame_bgr)


def detect_blinks(
    ear_series: Sequence[Optional[float]],
    threshold: float = EAR_BLINK_THRESHOLD,
    min_consecutive: int = BLINK_MIN_CONSEC_FRAMES,
) -> int:
    """
    Count blinks: EAR below threshold for at least min_consecutive frames,
    with recovery above threshold between blinks.
    """
    blinks = 0
    below = 0
    in_blink = False
    for ear in ear_series:
        if ear is None:
            below = 0
            continue
        if ear < threshold:
            below += 1
            if below >= min_consecutive and not in_blink:
                blinks += 1
                in_blink = True
        else:
            below = 0
            in_blink = False
    return blinks


def _iter_sampled_frames(
    cap: cv2.VideoCapture,
    max_frames: int = 60,
) -> Iterator[tuple[int, np.ndarray]]:
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total > 1:
        indices = np.linspace(0, total - 1, min(max_frames, total), dtype=int)
    else:
        indices = list(range(max_frames))
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            yield int(idx), frame


def compute_video_ear_score(
    frames_or_cap: Union[cv2.VideoCapture, Sequence[np.ndarray], str, None] = None,
    *,
    video_path: Optional[str] = None,
    fps: float = 30.0,
    max_frames: int = 60,
    ear_blink_threshold: float = EAR_BLINK_THRESHOLD,
    blink_min_frames: int = BLINK_MIN_CONSEC_FRAMES,
) -> float:
    """
    Video-level deepfake suspicion from blink / EAR dynamics.

    Args:
        frames_or_cap: cv2.VideoCapture, list of BGR frames, or path string.
        video_path:    Alternative path if cap not provided.
        fps:           Frame rate for blink-rate calibration.
        max_frames:    Max frames to sample from video.

    Returns:
        float in [0, 1] — higher = more suspicious (abnormal blink / static eyes).
        0.5 if insufficient signal (neutral fallback).
    """
    cap_owned = False
    cap: Optional[cv2.VideoCapture] = None
    frame_list: Optional[list[np.ndarray]] = None

    if isinstance(frames_or_cap, cv2.VideoCapture):
        cap = frames_or_cap
    elif isinstance(frames_or_cap, str):
        video_path = frames_or_cap
    elif isinstance(frames_or_cap, (list, tuple)):
        frame_list = list(frames_or_cap)

    path = video_path or (
        frames_or_cap if isinstance(frames_or_cap, str) else None
    )
    if cap is None and frame_list is None and path:
        cap = cv2.VideoCapture(path)
        cap_owned = True
        if not cap.isOpened():
            if cap_owned:
                cap.release()
            return 0.5

    ear_values: list[Optional[float]] = []

    if frame_list is not None:
        step = max(1, len(frame_list) // max_frames)
        sampled = frame_list[::step][:max_frames]
        for fr in sampled:
            ear_values.append(compute_frame_ear(fr))
        n_sampled = len(sampled)
        if fps <= 0:
            fps = 30.0
        duration_sec = n_sampled / fps
    elif cap is not None:
        if fps <= 0:
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        n_sampled = 0
        for _, frame in _iter_sampled_frames(cap, max_frames=max_frames):
            ear_values.append(compute_frame_ear(frame))
            n_sampled += 1
        duration_sec = (n_sampled / fps) if n_sampled > 0 and fps > 0 else 0.0
    else:
        return 0.5

    if cap_owned and cap is not None:
        cap.release()

    valid = [e for e in ear_values if e is not None]
    if len(valid) < max(3, int(MIN_VALID_EAR_FRAC * max(len(ear_values), 1))):
        return 0.5

    blinks = detect_blinks(ear_values, ear_blink_threshold, blink_min_frames)
    duration_sec = max(duration_sec, 1e-3)
    blink_rate_hz = blinks / duration_sec

    rate_deficit = max(0.0, REF_BLINK_RATE_HZ - blink_rate_hz) / REF_BLINK_RATE_HZ
    ear_std = float(np.std(valid))
    static_eye = max(0.0, 1.0 - ear_std / EAR_STD_FLOOR) if EAR_STD_FLOOR > 0 else 0.0

    raw = 0.65 * rate_deficit + 0.35 * static_eye
    return round(float(np.clip(raw, 0.0, 1.0)), 4)


def _resolve_ff_subdir(video_id: str, source_dataset: str) -> Optional[str]:
    ds = (source_dataset or "").replace("\\", "/")
    vid = (video_id or "").strip()
    looks_like_ff = (
        "FaceForensics" in ds
        or "FaceForensics++_C23" in ds
        or vid.startswith(("real_", "fake_"))
    )
    if not looks_like_ff:
        return None
    # EAR should always be computed from the original source actor videos.
    return "original"


def _candidate_original_mp4_names(video_id: str) -> list[str]:
    """
    Return candidate original video filenames for this manifest video_id.

    Supports:
      - real_000 -> [000.mp4]
      - fake_000_003 -> [000.mp4, 003.mp4]
      - fake_042 -> [042.mp4] (legacy fallback)
    """
    vid = (video_id or "").strip().replace("\\", "/")
    if not vid:
        return []
    stem = os.path.splitext(vid.split("/")[-1])[0]
    lowered = stem.lower()

    for prefix in (
        "real_",
        "fake_",
        "deepfakes_",
        "face2face_",
        "faceswap_",
        "neuraltextures_",
    ):
        if lowered.startswith(prefix):
            stem = stem[len(prefix) :]
            lowered = lowered[len(prefix) :]
            break

    pair = re.fullmatch(r"(\d{3})_(\d{3})", stem)
    if pair:
        a, b = pair.groups()
        if a == b:
            return [f"{a}.mp4"]
        return [f"{a}.mp4", f"{b}.mp4"]

    if stem.isdigit():
        return [f"{stem.zfill(3)}.mp4"]

    pair_anywhere = re.search(r"(?<!\d)(\d{3})_(\d{3})(?!\d)", stem)
    if pair_anywhere:
        a, b = pair_anywhere.groups()
        if a == b:
            return [f"{a}.mp4"]
        return [f"{a}.mp4", f"{b}.mp4"]

    single_anywhere = re.search(r"(?<!\d)(\d{3})(?!\d)", stem)
    if single_anywhere:
        return [f"{single_anywhere.group(1)}.mp4"]

    return []


def resolve_source_video_paths(video_id: str, source_dataset: str) -> list[str]:
    """Resolve all plausible FF++ source paths for the given manifest video_id."""
    sub = _resolve_ff_subdir(video_id, source_dataset)
    if not sub:
        return []
    names = _candidate_original_mp4_names(video_id)
    if not names:
        return []
    resolved: list[str] = []
    for dataset_dir in FF_DATASET_DIR_CANDIDATES:
        base = os.path.join(dataset_dir, sub)
        for mp4_name in names:
            path = os.path.join(base, mp4_name)
            if os.path.isfile(path):
                resolved.append(path)
    # preserve order while deduping
    seen = set()
    unique = []
    for p in resolved:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def resolve_source_video_path(video_id: str, source_dataset: str) -> Optional[str]:
    """
    Map manifest video_id + source_dataset to an FF++ source .mp4 path.

    video_id examples: real_000, fake_042  →  000.mp4 / 042.mp4
    """
    paths = resolve_source_video_paths(video_id, source_dataset)
    return paths[0] if paths else None


def load_video_ear_scores(path: str = VIDEO_EAR_CSV_DEFAULT) -> dict[str, float]:
    """Load video_id → ear_score from CSV; empty dict if missing."""
    scores: dict[str, float] = {}
    if not os.path.isfile(path):
        return scores
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            vid = row.get("video_id", "").strip()
            if vid:
                try:
                    scores[vid] = float(row["ear_score"])
                except (KeyError, ValueError):
                    pass
    return scores


def write_video_ear_scores_csv(
    video_jobs: list[tuple[str, str]],
    path: str = VIDEO_EAR_CSV_DEFAULT,
    max_frames: int = 60,
    verbose: bool = True,
) -> dict[str, float]:
    """
    Compute and write video-level EAR scores for unique (video_id, source_dataset).

    video_jobs: list of (video_id, source_dataset) — duplicates skipped.
    """
    seen: set[str] = set()
    rows: list[dict] = []
    scores: dict[str, float] = {}

    for video_id, source_dataset in video_jobs:
        if video_id in seen:
            continue
        seen.add(video_id)
        source_paths = resolve_source_video_paths(video_id, source_dataset)
        score = compute_ear_from_source_paths(source_paths, max_frames=max_frames)
        if score is None:
            score = 0.5
        scores[video_id] = score
        rows.append({"video_id": video_id, "ear_score": score, "source_dataset": source_dataset})
        if verbose:
            source_txt = ", ".join(source_paths) if source_paths else "no source video"
            print(f"  EAR {video_id}: {score:.4f}  ({source_txt})")

    if rows:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["video_id", "ear_score", "source_dataset"]
            )
            writer.writeheader()
            writer.writerows(rows)
        if verbose:
            print(f"  Wrote {path} ({len(rows)} videos)")
    return scores


def compute_ear_from_source_paths(source_paths: Sequence[str], max_frames: int = 60) -> Optional[float]:
    """
    Compute a robust EAR score from one or more source videos.

    For FF++ Deepfakes pairs, averages available source-video EAR scores.
    Returns None if no input file could be scored.
    """
    vals: list[float] = []
    for path in source_paths:
        if not path or not os.path.isfile(path):
            continue
        vals.append(float(compute_video_ear_score(video_path=path, max_frames=max_frames)))
    if not vals:
        return None
    return round(float(np.mean(vals)), 4)
