"""MRL Eye inference: per-frame awake/sleepy prediction with blink counting.

Loads a trained MobileNetV2 checkpoint, crops eye regions from JPEG frames
using MediaPipe FaceLandmarker (Tasks API ≥ 0.10), classifies each frame,
counts blinks (open → closed → open), and writes a per-video CSV.

Usage (called from ``run_inference.py``)::

    from src.mrl.inference import load_model, process_video
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.mrl.train import build_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MediaPipe eye landmarks (FaceMesh 478-point topology)
# ---------------------------------------------------------------------------
LEFT_EYE_LANDMARKS = [33, 133, 159, 145]
RIGHT_EYE_LANDMARKS = [362, 263, 386, 374]

_DARK_FRAME_THRESHOLD = 30
_EYE_PAD_FRACTION = 0.25

IMAGE_EXTENSIONS = {".jpg", ".jpeg"}

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LANDMARKER_MODEL = _PROJECT_ROOT / "models" / "face_landmarker.task"


# =========================================================================
# A. Model loading
# =========================================================================

DeviceLike = Union[torch.device, object]


def is_directml_device(device: DeviceLike) -> bool:
    """Return True when *device* is a torch-directml device."""
    device_type = getattr(device, "type", None)
    if device_type == "privateuseone":
        return True
    return type(device).__module__ == "torch_directml"


def resolve_device(device_name: str | None = None) -> DeviceLike:
    """Resolve a device string to a torch/cuda/cpu/directml device."""
    if device_name is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name = device_name.lower()
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    if name == "directml":
        import torch_directml

        return torch_directml.device()

    return torch.device(device_name)


def load_model(
    checkpoint_path: Path,
    device: DeviceLike | None = None,
) -> Tuple[torch.nn.Module, int, Dict[int, str]]:
    """Load a trained MobileNetV2 checkpoint.

    Args:
        checkpoint_path: path to ``best_model.pth``
        device: target device (auto-detected when *None*)

    Returns:
        (model, img_size, idx_to_label) where *idx_to_label* maps
        integer model-output indices to human-readable labels, e.g.
        ``{0: "sleepy", 1: "awake"}``.

    The Kaggle checkpoint stores ``class_to_label`` as **name → index**
    (``{"awake": 1, "sleepy": 0}``).  This function inverts it to
    **index → name** for convenient use during inference.
    """
    if device is None:
        device = resolve_device(None)

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # DirectML does not support every torch.load map_location target.
    map_location = "cpu" if is_directml_device(device) else device
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    img_size: int = ckpt["img_size"]

    # Checkpoint stores name→index; invert to index→name for inference
    name_to_idx: Dict[str, int] = ckpt["class_to_label"]
    idx_to_label: Dict[int, str] = {v: k for k, v in name_to_idx.items()}

    model = build_model()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    logger.info(
        "Loaded checkpoint %s  (img_size=%d, classes=%s, device=%s)",
        checkpoint_path.name, img_size, idx_to_label, device,
    )
    return model, img_size, idx_to_label


# =========================================================================
# B. MediaPipe eye cropper
# =========================================================================

class EyeCropper:
    """Crop left and/or right eye regions using MediaPipe FaceLandmarker.

    Uses the MediaPipe Tasks API (≥ 0.10).  Requires the
    ``face_landmarker.task`` model file, downloaded automatically to
    ``models/face_landmarker.task`` if missing.
    """

    _LANDMARKER_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    )

    def __init__(
        self,
        model_path: Path | str | None = None,
        min_detection_confidence: float = 0.5,
    ):
        import mediapipe as mp

        model_path = Path(model_path) if model_path else _DEFAULT_LANDMARKER_MODEL
        if not model_path.is_file():
            self._download_model(model_path)

        base_opts = mp.tasks.BaseOptions(
            model_asset_path=str(model_path),
        )
        opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_opts,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
        self._mp = mp

    @classmethod
    def _download_model(cls, dest: Path) -> None:
        """Download the face-landmarker model if not already present."""
        import urllib.request

        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading face_landmarker.task → %s", dest)
        urllib.request.urlretrieve(cls._LANDMARKER_URL, str(dest))

    def crop_eyes(
        self, frame_bgr: np.ndarray, img_size: int = 84,
    ) -> List[np.ndarray]:
        """Return 0, 1, or 2 grayscale eye crops resized to *img_size*."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb,
        )
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return []

        lms = result.face_landmarks[0]
        crops: List[np.ndarray] = []

        for eye_ids in (LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS):
            crop = self._crop_single_eye(frame_bgr, lms, eye_ids, h, w, img_size)
            if crop is not None:
                crops.append(crop)

        return crops

    @staticmethod
    def _crop_single_eye(
        frame_bgr: np.ndarray,
        landmarks,
        eye_ids: List[int],
        h: int,
        w: int,
        img_size: int,
    ) -> Optional[np.ndarray]:
        xs = [landmarks[i].x * w for i in eye_ids]
        ys = [landmarks[i].y * h for i in eye_ids]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        bw = x_max - x_min
        bh = y_max - y_min
        if bw < 2 or bh < 2:
            return None

        pad_x = bw * _EYE_PAD_FRACTION
        pad_y = bh * _EYE_PAD_FRACTION
        x1 = max(0, int(x_min - pad_x))
        y1 = max(0, int(y_min - pad_y))
        x2 = min(w, int(x_max + pad_x))
        y2 = min(h, int(y_max + pad_y))

        if x2 - x1 < 2 or y2 - y1 < 2:
            return None

        crop = frame_bgr[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(
            gray, (img_size, img_size), interpolation=cv2.INTER_AREA,
        )
        return resized

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# =========================================================================
# C. Per-frame classifier
# =========================================================================

def classify_frame(
    model: torch.nn.Module,
    eye_crops: List[np.ndarray],
    class_to_label: Dict[int, str],
    device: DeviceLike,
) -> str:
    """Classify a frame as awake/sleepy from its eye crop(s).

    If two crops are available, the prediction with higher confidence wins.
    Returns ``"unknown"`` when no crops are provided.
    """
    if not eye_crops:
        return "unknown"

    best_label: str = "unknown"
    best_conf: float = -1.0

    for crop in eye_crops:
        tensor = torch.from_numpy(crop.astype(np.float32) / 255.0)
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)

        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)

        conf, idx = probs.max(dim=1)
        conf_val = conf.item()
        if conf_val > best_conf:
            best_conf = conf_val
            best_label = class_to_label[idx.item()]

    return best_label


# =========================================================================
# D. Blink counter
# =========================================================================

class BlinkCounter:
    """State machine that counts blinks (open → closed → open)."""

    def __init__(self):
        self._state = "open"
        self.blink_count = 0

    def update(self, eye_state: str) -> int:
        """Feed one frame's eye state; return running blink count."""
        if eye_state == "unknown":
            return self.blink_count

        if eye_state == "sleepy":
            self._state = "closed"
        elif eye_state == "awake" and self._state == "closed":
            self._state = "open"
            self.blink_count += 1

        return self.blink_count


# =========================================================================
# E. Video directory processor
# =========================================================================

def process_video(
    video_dir: Path,
    model: torch.nn.Module,
    img_size: int,
    class_to_label: Dict[int, str],
    device: DeviceLike,
    fps: float = 30.0,
) -> pd.DataFrame:
    """Run inference on every JPEG frame in *video_dir*.

    Returns a DataFrame with columns:
        video_id, frame_id, timestamp, eye_state, blink_count, blinks_per_minute
    """
    video_dir = Path(video_dir)
    video_id = video_dir.name

    frames = sorted(
        p for p in video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not frames:
        logger.warning("No JPEG frames in %s — skipping", video_dir)
        return pd.DataFrame(
            columns=["video_id", "frame_id", "timestamp",
                      "eye_state", "blink_count", "blinks_per_minute"],
        )

    rows: List[Dict] = []
    blink_counter = BlinkCounter()
    active_device = device

    with EyeCropper() as cropper:
        for idx, frame_path in enumerate(frames):
            frame_bgr = cv2.imread(str(frame_path))

            # Corrupted file
            if frame_bgr is None:
                eye_state = "unknown"
                logger.warning("Could not read %s", frame_path.name)
            # Too dark
            elif float(np.mean(frame_bgr)) < _DARK_FRAME_THRESHOLD:
                eye_state = "unknown"
                logger.debug("Frame too dark: %s", frame_path.name)
            else:
                crops = cropper.crop_eyes(frame_bgr, img_size=img_size)
                try:
                    eye_state = classify_frame(
                        model, crops, class_to_label, active_device,
                    )
                except RuntimeError as err:
                    if is_directml_device(active_device):
                        logger.warning(
                            "DirectML op failed (%s); falling back to CPU",
                            err,
                        )
                        active_device = torch.device("cpu")
                        model.to(active_device)
                        eye_state = classify_frame(
                            model, crops, class_to_label, active_device,
                        )
                    else:
                        raise

            blink_count = blink_counter.update(eye_state)
            timestamp = idx / fps
            bpm = (blink_count / timestamp * 60.0) if timestamp > 0 else 0.0

            rows.append({
                "video_id": video_id,
                "frame_id": frame_path.stem,
                "timestamp": round(timestamp, 4),
                "eye_state": eye_state,
                "blink_count": blink_count,
                "blinks_per_minute": round(bpm, 2),
            })

    return pd.DataFrame(rows)
