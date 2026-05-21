from __future__ import annotations

import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.mrl.inference import (
    BlinkCounter,
    EyeCropper,
    classify_frame,
    load_model,
    process_video,
)

TMP_ROOT = Path(__file__).resolve().parent / "tmp"
FRAME_SHAPE = (480, 640, 3)


@pytest.fixture(autouse=True)
def print_stress_summary(request):
    start = time.perf_counter()
    outcome = "PASS"
    try:
        yield
    except BaseException:
        outcome = "FAIL"
        raise
    finally:
        elapsed = time.perf_counter() - start
        print(f"[stress-summary] {request.node.name}: {outcome} in {elapsed:.3f}s")


@pytest.fixture()
def tmp_stress_dir(request):
    test_dir = TMP_ROOT / request.node.name
    shutil.rmtree(test_dir, ignore_errors=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield test_dir
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)
        if TMP_ROOT.exists() and not any(TMP_ROOT.iterdir()):
            TMP_ROOT.rmdir()


def test_synthetic_face_frames_crop_eyes_returns_list():
    rng = np.random.default_rng(100)
    cropper = EyeCropper()

    for _ in range(100):
        frame = rng.integers(0, 256, size=FRAME_SHAPE, dtype=np.uint8)
        eyes = cropper.crop_eyes(frame)
        assert isinstance(eyes, list)


def test_dark_frames_return_unknown_eye_state():
    rng = np.random.default_rng(101)
    cropper = EyeCropper()

    for _ in range(50):
        frame = rng.integers(0, 30, size=FRAME_SHAPE, dtype=np.uint8)
        assert float(frame.mean()) < 30.0
        result = classify_frame(frame, cropper=cropper)
        assert result["eye_state"] == "unknown"


def test_blank_white_frames_do_not_crash():
    cropper = EyeCropper()
    frame = np.full(FRAME_SHAPE, 255, dtype=np.uint8)

    for _ in range(50):
        eyes = cropper.crop_eyes(frame)
        result = classify_frame(frame, cropper=cropper)
        assert isinstance(eyes, list)
        assert result["eye_state"] in {"awake", "sleepy", "unknown"}


def test_blink_sequence_random_stress_never_decreases_or_exceeds_frames():
    rng = np.random.default_rng(102)
    counter = BlinkCounter()
    previous = 0

    for frame_idx, state in enumerate(
        rng.choice(["awake", "sleepy", "unknown"], size=1000),
        start=1,
    ):
        blink_count = counter.update(str(state))
        assert blink_count >= previous
        assert blink_count <= frame_idx
        previous = blink_count


def test_rapid_state_changes_count_alternating_sleepy_transitions():
    counter = BlinkCounter()

    for state in ["awake", "sleepy"] * 250:
        counter.update(state)

    assert counter.blink_count == 250


def test_all_unknown_sequence_keeps_blink_count_zero():
    counter = BlinkCounter()

    for _ in range(100):
        assert counter.update("unknown") == 0

    assert counter.blink_count == 0


def test_single_frame_video_outputs_one_csv_row(tmp_stress_dir):
    frame_dir = tmp_stress_dir / "frames"
    frame_dir.mkdir()
    csv_path = tmp_stress_dir / "single_frame.csv"
    _write_jpeg(frame_dir / "frame_000001.jpg", np.zeros(FRAME_SHAPE, dtype=np.uint8))

    df = process_video(frame_dir, output_csv=csv_path)
    csv_df = pd.read_csv(csv_path)

    assert len(df) == 1
    assert len(csv_df) == 1
    assert np.isfinite(csv_df["blinks_per_minute"]).all()


def test_empty_directory_returns_empty_dataframe(tmp_stress_dir):
    frame_dir = tmp_stress_dir / "empty"
    frame_dir.mkdir()

    df = process_video(frame_dir)

    assert df.empty


def test_large_batch_process_video_outputs_500_nonnegative_rows(tmp_stress_dir):
    rng = np.random.default_rng(103)
    frame_dir = tmp_stress_dir / "large_batch"
    frame_dir.mkdir()
    csv_path = tmp_stress_dir / "large_batch.csv"

    for idx in range(500):
        frame = rng.integers(0, 256, size=(120, 160, 3), dtype=np.uint8)
        _write_jpeg(frame_dir / f"frame_{idx:06d}.jpg", frame)

    df = process_video(frame_dir, output_csv=csv_path)
    csv_df = pd.read_csv(csv_path)

    assert len(df) == 500
    assert len(csv_df) == 500
    assert np.isfinite(csv_df["blinks_per_minute"]).all()
    assert (csv_df["blinks_per_minute"] >= 0).all()


def test_model_load_stress_outputs_are_identical():
    import torch
    import torchvision  # noqa: F401

    model_path = Path("src/mrl/best_model.pth")
    input_tensor = torch.zeros((1, 1, 84, 84), dtype=torch.float32)
    outputs = []

    for _ in range(10):
        model = load_model(model_path, device="cpu")
        with torch.no_grad():
            outputs.append(model(input_tensor).detach().cpu())

    first = outputs[0]
    for output in outputs[1:]:
        assert torch.equal(output, first)


def _write_jpeg(path: Path, rgb_frame: np.ndarray) -> None:
    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), bgr_frame)
    assert ok, f"failed to write synthetic frame: {path}"
