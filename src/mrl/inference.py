"""Inference helpers for the MRL eye-state model."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = Path(__file__).resolve().with_name("best_model.pth")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EYE_STATE_UNKNOWN = "unknown"
EYE_STATE_AWAKE = "awake"
EYE_STATE_SLEEPY = "sleepy"


class EyeCropper:
    """Crop candidate eye regions from RGB or BGR frames using OpenCV cascades."""

    def __init__(self, eye_size: int = 84, max_eyes: int = 2) -> None:
        self.eye_size = eye_size
        self.max_eyes = max_eyes
        cascade_dir = Path(cv2.data.haarcascades)
        self.face_cascade = cv2.CascadeClassifier(
            str(cascade_dir / "haarcascade_frontalface_default.xml")
        )
        self.eye_cascade = cv2.CascadeClassifier(str(cascade_dir / "haarcascade_eye.xml"))

    def crop_eyes(self, frame: np.ndarray) -> list[np.ndarray]:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return []

        gray = _to_grayscale(frame)
        if gray.size == 0:
            return []

        regions = self._face_regions(gray)
        crops: list[np.ndarray] = []
        for x, y, w, h in regions:
            face_gray = gray[y : y + h, x : x + w]
            eyes = self.eye_cascade.detectMultiScale(
                face_gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(max(8, w // 12), max(8, h // 12)),
            )
            for ex, ey, ew, eh in sorted(eyes, key=lambda box: box[0]):
                crop = face_gray[ey : ey + eh, ex : ex + ew]
                if crop.size == 0:
                    continue
                crops.append(cv2.resize(crop, (self.eye_size, self.eye_size)))
                if len(crops) >= self.max_eyes:
                    return crops
        return crops

    def _face_regions(self, gray: np.ndarray) -> list[tuple[int, int, int, int]]:
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
        if len(faces) == 0:
            return [(0, 0, gray.shape[1], gray.shape[0])]
        return [tuple(map(int, face)) for face in faces]


class BlinkCounter:
    """Count sleepy transitions as blinks while ignoring unknown frames."""

    def __init__(self) -> None:
        self.blink_count = 0
        self._currently_sleepy = False
        self.total_frames = 0

    def update(self, eye_state: str) -> int:
        self.total_frames += 1
        if eye_state == EYE_STATE_SLEEPY:
            if not self._currently_sleepy:
                self.blink_count += 1
            self._currently_sleepy = True
        elif eye_state == EYE_STATE_AWAKE:
            self._currently_sleepy = False
        return self.blink_count


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH, device: str | None = None):
    """Load the trained grayscale MobileNetV2 eye-state classifier."""

    try:
        import torch
        import torch.nn as nn
        from torchvision import models
    except ImportError as exc:  # pragma: no cover - exercised in lean environments.
        raise ImportError("load_model requires torch and torchvision to be installed.") from exc

    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(Path(model_path), map_location=device_obj)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    img_size = int(checkpoint.get("img_size", 84)) if isinstance(checkpoint, dict) else 84
    class_to_label = (
        checkpoint.get("class_to_label", {EYE_STATE_AWAKE: 1, EYE_STATE_SLEEPY: 0})
        if isinstance(checkpoint, dict)
        else {EYE_STATE_AWAKE: 1, EYE_STATE_SLEEPY: 0}
    )

    model = models.mobilenet_v2(weights=None, num_classes=2)
    model.features[0][0] = nn.Conv2d(
        1,
        model.features[0][0].out_channels,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=False,
    )
    model.load_state_dict(state_dict)
    model.img_size = img_size
    model.class_to_label = class_to_label
    model.label_to_class = {int(v): str(k) for k, v in class_to_label.items()}
    model.to(device_obj)
    model.eval()
    return model


def classify_frame(
    frame: np.ndarray,
    model=None,
    cropper: EyeCropper | None = None,
    device: str | None = None,
    dark_threshold: float = 30.0,
) -> dict[str, object]:
    """Classify one video frame as awake, sleepy, or unknown."""

    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return _classification(EYE_STATE_UNKNOWN, 0.0, 0)

    gray = _to_grayscale(frame)
    if gray.size == 0 or float(np.mean(gray)) < dark_threshold:
        return _classification(EYE_STATE_UNKNOWN, 0.0, 0)

    cropper = cropper or EyeCropper()
    eye_crops = cropper.crop_eyes(frame)
    if not eye_crops or model is None:
        return _classification(EYE_STATE_UNKNOWN, 0.0, len(eye_crops))

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - load_model has the same guard.
        raise ImportError("classify_frame with a model requires torch to be installed.") from exc

    device_obj = torch.device(device or next(model.parameters()).device)
    tensors = [_eye_to_tensor(crop, getattr(model, "img_size", 84)) for crop in eye_crops]
    batch = torch.stack(tensors).to(device_obj)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1).mean(dim=0)
    label = int(torch.argmax(probs).item())
    eye_state = getattr(model, "label_to_class", {}).get(label, str(label))
    confidence = float(probs[label].item())
    if eye_state not in {EYE_STATE_AWAKE, EYE_STATE_SLEEPY}:
        eye_state = EYE_STATE_UNKNOWN
    return _classification(eye_state, confidence, len(eye_crops))


def process_video(
    frame_dir: str | Path,
    output_csv: str | Path | None = None,
    model=None,
    cropper: EyeCropper | None = None,
    fps: float = 30.0,
    device: str | None = None,
) -> pd.DataFrame:
    """Run frame-directory inference and optionally save per-frame CSV output."""

    frame_dir = Path(frame_dir)
    output_csv = Path(output_csv) if output_csv is not None else None
    rows: list[dict[str, object]] = []
    counter = BlinkCounter()
    cropper = cropper or EyeCropper()

    if frame_dir.is_dir():
        for idx, path in enumerate(_iter_frame_paths(frame_dir), start=1):
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                result = _classification(EYE_STATE_UNKNOWN, 0.0, 0)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = classify_frame(frame, model=model, cropper=cropper, device=device)
            blink_count = counter.update(str(result["eye_state"]))
            elapsed_minutes = (idx / fps) / 60.0 if fps > 0 else 0.0
            blinks_per_minute = blink_count / elapsed_minutes if elapsed_minutes > 0 else 0.0
            if not math.isfinite(blinks_per_minute):
                blinks_per_minute = 0.0
            rows.append(
                {
                    "frame": path.name,
                    "eye_state": result["eye_state"],
                    "confidence": result["confidence"],
                    "eye_count": result["eye_count"],
                    "blink_count": blink_count,
                    "blinks_per_minute": blinks_per_minute,
                }
            )

    df = pd.DataFrame(
        rows,
        columns=[
            "frame",
            "eye_state",
            "confidence",
            "eye_count",
            "blink_count",
            "blinks_per_minute",
        ],
    )
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
    return df


def _classification(eye_state: str, confidence: float, eye_count: int) -> dict[str, object]:
    return {"eye_state": eye_state, "confidence": confidence, "eye_count": eye_count}


def _iter_frame_paths(frame_dir: Path) -> Iterable[Path]:
    return sorted(
        path for path in frame_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def _to_grayscale(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame.astype(np.uint8, copy=False)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame.astype(np.uint8, copy=False), cv2.COLOR_RGB2GRAY)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame.astype(np.uint8, copy=False), cv2.COLOR_RGBA2GRAY)
    return np.asarray([], dtype=np.uint8)


def _eye_to_tensor(eye_crop: np.ndarray, img_size: int):
    import torch

    resized = cv2.resize(eye_crop, (img_size, img_size))
    arr = resized.astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)
