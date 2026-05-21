from __future__ import annotations

import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from src.mrl.inference import (
    BlinkCounter,
    EyeCropper,
    classify_frame,
    load_model,
    process_video,
)

TMP_ROOT = Path(__file__).resolve().parent / "tmp"


@pytest.fixture
def tmp_test_dir(request):
    path = TMP_ROOT / request.node.name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


@contextmanager
def timed_test(name: str):
    start = time.perf_counter()
    try:
        yield
    except Exception:
        print(f"[stress-summary] FAIL {name} {time.perf_counter() - start:.3f}s")
        raise
    print(f"[stress-summary] PASS {name} {time.perf_counter() - start:.3f}s")


def _rgb_frame(rng: np.random.Generator, height: int = 480, width: int = 640):
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _dark_frame(rng: np.random.Generator, height: int = 480, width: int = 640):
    return rng.integers(0, 30, size=(height, width, 3), dtype=np.uint8)


def _write_jpeg(path: Path, rgb: np.ndarray) -> None:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    assert cv2.imwrite(str(path), bgr)


def test_synthetic_face_frames_crop_eyes_returns_lists():
    with timed_test("synthetic face frames"):
        rng = np.random.default_rng(101)
        cropper = EyeCropper()
        for _ in range(100):
            result = cropper.crop_eyes(_rgb_frame(rng))
            assert isinstance(result, list)


def test_dark_frames_return_unknown():
    with timed_test("dark frames"):
        rng = np.random.default_rng(202)
        for _ in range(50):
            frame = _dark_frame(rng)
            assert frame.mean() < 30
            result = classify_frame(frame)
            assert result["eye_state"] == "unknown"


def test_blank_white_frames_do_not_crash():
    with timed_test("blank white frames"):
        frame = np.full((480, 640, 3), 255, dtype=np.uint8)
        cropper = EyeCropper()
        for _ in range(50):
            assert isinstance(cropper.crop_eyes(frame), list)
            result = classify_frame(frame, cropper=cropper)
            assert result["eye_state"] in {"awake", "sleepy", "unknown"}


def test_blink_sequence_stress_count_monotonic_and_bounded():
    with timed_test("blink sequence stress"):
        rng = np.random.default_rng(303)
        states = rng.choice(["awake", "sleepy", "unknown"], size=1000)
        counter = BlinkCounter()
        previous = 0
        for index, state in enumerate(states, start=1):
            current = counter.update(str(state))
            assert current >= previous
            assert current <= index
            previous = current


def test_rapid_state_changes_count_every_awake_to_sleepy_transition():
    with timed_test("rapid state changes"):
        counter = BlinkCounter()
        for state in ["awake", "sleepy"] * 250:
            counter.update(state)
        assert counter.blink_count == 250


def test_all_unknown_sequence_stays_zero():
    with timed_test("all unknown sequence"):
        counter = BlinkCounter()
        for _ in range(100):
            assert counter.update("unknown") == 0
        assert counter.blink_count == 0


def test_single_frame_video_outputs_one_csv_row(tmp_test_dir):
    with timed_test("single frame video"):
        rng = np.random.default_rng(404)
        frame_dir = tmp_test_dir / "frames"
        frame_dir.mkdir()
        csv_path = tmp_test_dir / "single.csv"
        _write_jpeg(frame_dir / "frame_000.jpg", _rgb_frame(rng, 64, 64))

        df = process_video(frame_dir, csv_path, fps=30.0)
        written = pd.read_csv(csv_path)

        assert len(df) == 1
        assert len(written) == 1
        assert np.isfinite(df["blinks_per_minute"]).all()
        assert (df["blinks_per_minute"] >= 0).all()


def test_empty_directory_returns_empty_dataframe(tmp_test_dir):
    with timed_test("empty directory"):
        frame_dir = tmp_test_dir / "frames"
        frame_dir.mkdir()
        df = process_video(frame_dir, tmp_test_dir / "empty.csv")

        assert df.empty
        assert list(df.columns) == [
            "frame",
            "frame_index",
            "eye_state",
            "confidence",
            "num_eyes",
            "blink_count",
            "blinks_per_minute",
        ]


def test_large_batch_process_video_outputs_500_rows(tmp_test_dir):
    with timed_test("large batch"):
        rng = np.random.default_rng(505)
        frame_dir = tmp_test_dir / "frames"
        frame_dir.mkdir()
        csv_path = tmp_test_dir / "large.csv"
        for index in range(500):
            _write_jpeg(frame_dir / f"frame_{index:04d}.jpg", _rgb_frame(rng, 64, 64))

        df = process_video(frame_dir, csv_path, fps=30.0)
        written = pd.read_csv(csv_path)

        assert len(df) == 500
        assert len(written) == 500
        assert np.isfinite(df["blinks_per_minute"]).all()
        assert not df["blinks_per_minute"].isna().any()
        assert (df["blinks_per_minute"] >= 0).all()


def test_model_load_stress_identical_outputs_on_same_tensor():
    with timed_test("model load stress"):
        torch.manual_seed(606)
        sample = torch.rand(1, 1, 84, 84)
        expected = None

        for _ in range(10):
            model = load_model(device="cpu")
            with torch.no_grad():
                output = model(sample)
            if expected is None:
                expected = output
            else:
                assert torch.allclose(output, expected, atol=0, rtol=0)
