"""MRL eye-state inference helpers for frame and directory processing."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = Path(__file__).resolve().with_name("best_model.pth")
DEFAULT_FPS = 30.0
UNKNOWN_DARK_THRESHOLD = 30.0


class EyeCropper:
    """Detect faces and eyes, returning eye crops as RGB numpy arrays."""

    def __init__(self) -> None:
        haar_dir = Path(cv2.data.haarcascades)
        self.face_detector = cv2.CascadeClassifier(
            str(haar_dir / "haarcascade_frontalface_default.xml")
        )
        self.eye_detector = cv2.CascadeClassifier(str(haar_dir / "haarcascade_eye.xml"))

    def crop_eyes(self, frame: np.ndarray) -> list[np.ndarray]:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return []

        image = _ensure_rgb_uint8(frame)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        crops: list[np.ndarray] = []
        for x, y, w, h in faces:
            roi_gray = gray[y : y + h, x : x + w]
            roi_rgb = image[y : y + h, x : x + w]
            eyes = self.eye_detector.detectMultiScale(
                roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(12, 12)
            )
            for ex, ey, ew, eh in sorted(eyes, key=lambda box: box[0])[:2]:
                crop = roi_rgb[ey : ey + eh, ex : ex + ew]
                if crop.size:
                    crops.append(crop.copy())
        return crops


class BlinkCounter:
    """Count blink events as transitions from awake/open to sleepy/closed."""

    def __init__(self) -> None:
        self.count = 0
        self._last_known_state: str | None = None

    def update(self, eye_state: str) -> int:
        if eye_state not in {"awake", "sleepy", "unknown"}:
            eye_state = "unknown"

        if eye_state == "unknown":
            return self.count

        if self._last_known_state == "awake" and eye_state == "sleepy":
            self.count += 1
        self._last_known_state = eye_state
        return self.count

    @property
    def blink_count(self) -> int:
        return self.count


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH, device: str | None = None):
    """Load the bundled grayscale MobileNetV2 MRL checkpoint."""

    import torch

    path = Path(model_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    checkpoint = torch.load(path, map_location=device or "cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model = _build_mrl_mobilenet(num_classes=2)
    model.load_state_dict(state_dict)
    model.eval()

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(resolved_device)
    model.img_size = int(checkpoint.get("img_size", 84))
    model.class_to_label = checkpoint.get("class_to_label", {"awake": 1, "sleepy": 0})
    model.device = resolved_device
    return model


def classify_frame(
    frame: np.ndarray,
    model=None,
    cropper: EyeCropper | None = None,
    *,
    dark_threshold: float = UNKNOWN_DARK_THRESHOLD,
) -> dict[str, object]:
    """Classify one RGB frame into awake, sleepy, or unknown."""

    image = _ensure_rgb_uint8(frame)
    if float(np.mean(image)) < dark_threshold:
        return {"eye_state": "unknown", "confidence": 0.0, "num_eyes": 0}

    eye_cropper = cropper or EyeCropper()
    eye_crops = eye_cropper.crop_eyes(image)
    if not eye_crops:
        return {"eye_state": "unknown", "confidence": 0.0, "num_eyes": 0}

    if model is None:
        model = load_model()

    probs = _predict_eye_probabilities(model, eye_crops)
    mean_probs = probs.mean(axis=0)
    class_to_label = getattr(model, "class_to_label", {"awake": 1, "sleepy": 0})
    label_to_class = {int(v): str(k) for k, v in class_to_label.items()}
    predicted_idx = int(np.argmax(mean_probs))
    eye_state = label_to_class.get(predicted_idx, "unknown")
    return {
        "eye_state": eye_state,
        "confidence": float(mean_probs[predicted_idx]),
        "num_eyes": len(eye_crops),
    }


def process_video(
    frame_dir: str | Path,
    output_csv: str | Path | None = None,
    *,
    model=None,
    fps: float = DEFAULT_FPS,
) -> pd.DataFrame:
    """Run inference over JPEG frames in a directory and optionally write CSV."""

    directory = Path(frame_dir)
    frame_paths = _iter_frame_paths(directory)
    columns = [
        "frame",
        "frame_index",
        "eye_state",
        "confidence",
        "num_eyes",
        "blink_count",
        "blinks_per_minute",
    ]
    if not frame_paths:
        df = pd.DataFrame(columns=columns)
        if output_csv is not None:
            _write_csv(df, output_csv)
        return df

    loaded_model = model or load_model()
    cropper = EyeCropper()
    counter = BlinkCounter()
    safe_fps = fps if fps and fps > 0 else DEFAULT_FPS
    rows: list[dict[str, object]] = []

    for idx, path in enumerate(frame_paths):
        bgr = cv2.imread(str(path))
        frame = (
            np.zeros((1, 1, 3), dtype=np.uint8)
            if bgr is None
            else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        )
        result = classify_frame(frame, loaded_model, cropper)
        blink_count = counter.update(str(result["eye_state"]))
        elapsed_minutes = max((idx + 1) / safe_fps / 60.0, np.finfo(float).eps)
        rows.append(
            {
                "frame": path.name,
                "frame_index": idx,
                "eye_state": result["eye_state"],
                "confidence": result["confidence"],
                "num_eyes": result["num_eyes"],
                "blink_count": blink_count,
                "blinks_per_minute": blink_count / elapsed_minutes,
            }
        )

    df = pd.DataFrame(rows, columns=columns)
    if output_csv is not None:
        _write_csv(df, output_csv)
    return df


def _build_mrl_mobilenet(num_classes: int):
    import torch.nn as nn
    from torchvision import models

    model = models.mobilenet_v2(weights=None)
    first_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        1,
        first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=False,
    )
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def _predict_eye_probabilities(model, eye_crops: Iterable[np.ndarray]) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    tensors = [_preprocess_eye(crop, getattr(model, "img_size", 84)) for crop in eye_crops]
    batch = torch.stack(tensors).to(getattr(model, "device", torch.device("cpu")))
    with torch.no_grad():
        logits = model(batch)
        return F.softmax(logits, dim=1).cpu().numpy()


def _preprocess_eye(crop: np.ndarray, img_size: int):
    import torch

    image = _ensure_rgb_uint8(crop)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    array = resized.astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _ensure_rgb_uint8(frame: np.ndarray) -> np.ndarray:
    image = np.asarray(frame)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a grayscale or 3-channel image frame.")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _iter_frame_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in suffixes)


def _write_csv(df: pd.DataFrame, output_csv: str | Path) -> None:
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
