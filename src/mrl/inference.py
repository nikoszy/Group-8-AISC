"""MRL Eye inference: eye cropping, frame classification, blink counting, and
video-level processing pipeline.

Classes:
    EyeCropper    -- Haar-cascade eye region extractor
    BlinkCounter  -- Tracks awake->sleepy transitions as blinks

Functions:
    load_model      -- Load a TinyCNN checkpoint from disk
    classify_frame  -- Classify a single eye/frame image as awake/sleepy/unknown
    process_video   -- End-to-end pipeline over a directory of JPEG frames
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch

from src.mrl.preprocess import TinyCNN, default_img_size

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class EyeCropper:
    """Extracts eye regions from BGR frames using the Haar eye cascade."""

    def __init__(self) -> None:
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

    def crop_eyes(self, frame: np.ndarray) -> List[np.ndarray]:
        """Return a (possibly empty) list of eye-region crops."""
        if frame is None or frame.size == 0:
            return []
        try:
            if frame.ndim == 3 and frame.shape[2] >= 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            elif frame.ndim == 2:
                gray = frame
            else:
                return []
        except cv2.error:
            return []

        eyes = self._cascade.detectMultiScale(gray, 1.3, 5)
        crops: List[np.ndarray] = []
        for x, y, w, h in eyes:
            crops.append(frame[y : y + h, x : x + w])
        return crops


def load_model(
    model_path: str,
    img_size: Optional[int] = None,
    device: Optional[str] = None,
) -> Tuple[TinyCNN, torch.device]:
    """Load a TinyCNN checkpoint and return ``(model, device)``."""
    if img_size is None:
        img_size = default_img_size()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    model = TinyCNN(num_classes=2, in_ch=1)
    state_dict = torch.load(model_path, map_location=dev, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(dev)
    model.eval()
    return model, dev


def classify_frame(
    frame: np.ndarray,
    model: Optional[TinyCNN] = None,
    device: str = "cpu",
    img_size: Optional[int] = None,
    dark_threshold: float = 30.0,
) -> str:
    """Classify a single eye image as ``'awake'``, ``'sleepy'``, or ``'unknown'``.

    Returns ``'unknown'`` when the frame is too dark, empty, or no model is
    provided.
    """
    if frame is None or frame.size == 0:
        return "unknown"

    if float(frame.mean()) < dark_threshold:
        return "unknown"

    if model is None:
        return "unknown"

    if img_size is None:
        img_size = default_img_size()

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        resized = cv2.resize(gray, (img_size, img_size))
        tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        tensor = tensor.to(device)

        with torch.no_grad():
            logits = model(tensor)
            pred = int(torch.argmax(logits, dim=1).item())

        return "awake" if pred == 0 else "sleepy"
    except Exception:
        return "unknown"


class BlinkCounter:
    """Counts blinks defined as *awake -> sleepy* state transitions."""

    def __init__(self) -> None:
        self._blink_count: int = 0
        self._prev_state: Optional[str] = None
        self._total_frames: int = 0

    @property
    def blink_count(self) -> int:
        return self._blink_count

    @property
    def total_frames(self) -> int:
        return self._total_frames

    def update(self, state: str) -> int:
        """Feed the next eye state and return the running blink count."""
        self._total_frames += 1
        if self._prev_state == "awake" and state == "sleepy":
            self._blink_count += 1
        self._prev_state = state
        return self._blink_count

    def reset(self) -> None:
        self._blink_count = 0
        self._prev_state = None
        self._total_frames = 0


def process_video(
    frames_dir: str,
    model: Optional[TinyCNN] = None,
    device: str = "cpu",
    img_size: Optional[int] = None,
    fps: float = 30.0,
) -> pd.DataFrame:
    """Run the full inference pipeline on a directory of image frames.

    Returns a :class:`~pandas.DataFrame` with columns:
    ``frame_file``, ``eye_state``, ``blink_count``, ``blinks_per_minute``.
    """
    empty = pd.DataFrame(
        columns=["frame_file", "eye_state", "blink_count", "blinks_per_minute"]
    )
    frames_path = Path(frames_dir)
    if not frames_path.is_dir():
        return empty

    frame_files = sorted(
        p
        for p in frames_path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not frame_files:
        return empty

    cropper = EyeCropper()
    counter = BlinkCounter()
    rows = []

    for frame_file in frame_files:
        frame = cv2.imread(str(frame_file))
        if frame is None:
            state = "unknown"
        else:
            eyes = cropper.crop_eyes(frame)
            if eyes:
                state = classify_frame(
                    eyes[0], model=model, device=device, img_size=img_size
                )
            else:
                state = classify_frame(
                    frame, model=model, device=device, img_size=img_size
                )

        blink_count = counter.update(state)
        elapsed_seconds = counter.total_frames / max(fps, 1e-9)
        bpm = blink_count / (elapsed_seconds / 60.0) if elapsed_seconds > 0 else 0.0

        rows.append(
            {
                "frame_file": frame_file.name,
                "eye_state": state,
                "blink_count": blink_count,
                "blinks_per_minute": bpm,
            }
        )

    return pd.DataFrame(rows)
