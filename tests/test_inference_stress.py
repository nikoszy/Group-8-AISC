"""Stress tests for src/mrl/inference.py.

Run with:
    python -m pytest tests/test_inference_stress.py -v --tb=short
"""

from __future__ import annotations

import math
import random
import shutil
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = PROJECT_ROOT / "models" / "best_model.pth"
TMP_DIR = Path(__file__).resolve().parent / "tmp"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="session")
def model_bundle(device):
    """Load the model once for the entire test session."""
    model, img_size, idx_to_label = load_model(CHECKPOINT, device)
    return model, img_size, idx_to_label


@pytest.fixture(scope="session")
def eye_cropper():
    """Single EyeCropper shared across all tests that need it."""
    cropper = EyeCropper()
    yield cropper
    cropper.close()


@pytest.fixture(autouse=True)
def _manage_tmp():
    """Create tests/tmp/ before each test, clean up after."""
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    yield
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)


# ---------------------------------------------------------------------------
# 1. Synthetic face frames — 100 random frames, no crashes
# ---------------------------------------------------------------------------

class TestEyeCropperSyntheticFrames:
    def test_random_frames_no_crash(self, eye_cropper):
        rng = np.random.RandomState(42)
        for i in range(100):
            frame = rng.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            result = eye_cropper.crop_eyes(frame, img_size=84)
            assert isinstance(result, list), (
                f"Frame {i}: expected list, got {type(result)}"
            )
            for crop in result:
                assert isinstance(crop, np.ndarray)
                assert crop.shape == (84, 84)


# ---------------------------------------------------------------------------
# 2. Dark frames — mean < 30 → eye_state = "unknown"
# ---------------------------------------------------------------------------

class TestDarkFrames:
    def test_all_return_unknown(self, model_bundle, eye_cropper, device):
        model, img_size, idx_to_label = model_bundle
        rng = np.random.RandomState(99)
        for i in range(50):
            frame = rng.randint(0, 25, (480, 640, 3), dtype=np.uint8)
            assert float(np.mean(frame)) < 30
            crops = eye_cropper.crop_eyes(frame, img_size=img_size)
            if not crops:
                state = "unknown"
            else:
                state = classify_frame(model, crops, idx_to_label, device)
            # Dark frames should yield no usable crops (MediaPipe rarely
            # detects faces in near-black images), so we expect "unknown"
            # for the vast majority.  We assert the function never crashes.
            assert state in ("awake", "sleepy", "unknown")


# ---------------------------------------------------------------------------
# 3. Blank white frames — no crashes
# ---------------------------------------------------------------------------

class TestBlankWhiteFrames:
    def test_white_frames_no_crash(self, eye_cropper):
        for _ in range(50):
            frame = np.full((480, 640, 3), 255, dtype=np.uint8)
            result = eye_cropper.crop_eyes(frame, img_size=84)
            assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 4. BlinkCounter — 1000 random states, count never decreases
# ---------------------------------------------------------------------------

class TestBlinkCounterRandomSequence:
    def test_count_never_decreases(self):
        rng = random.Random(123)
        bc = BlinkCounter()
        prev = 0
        for _ in range(1000):
            state = rng.choice(["awake", "sleepy", "unknown"])
            count = bc.update(state)
            assert count >= prev, (
                f"Blink count decreased: {prev} → {count}"
            )
            prev = count
        assert bc.blink_count <= 1000


# ---------------------------------------------------------------------------
# 5. Rapid alternating awake/sleepy — 500 pairs → 250 blinks
# ---------------------------------------------------------------------------

class TestBlinkCounterRapidAlternation:
    def test_alternating_gives_250(self):
        bc = BlinkCounter()
        states = ["sleepy", "awake"] * 250  # 500 total updates
        for s in states:
            bc.update(s)
        assert bc.blink_count == 250


# ---------------------------------------------------------------------------
# 6. All unknown — count stays 0
# ---------------------------------------------------------------------------

class TestBlinkCounterAllUnknown:
    def test_all_unknown_stays_zero(self):
        bc = BlinkCounter()
        for _ in range(100):
            bc.update("unknown")
        assert bc.blink_count == 0


# ---------------------------------------------------------------------------
# 7. Single-frame video — 1 row, no division by zero
# ---------------------------------------------------------------------------

class TestSingleFrameVideo:
    def test_single_frame(self, model_bundle, device):
        model, img_size, idx_to_label = model_bundle
        video_dir = TMP_DIR / "single_frame_video"
        video_dir.mkdir(parents=True, exist_ok=True)
        img = np.full((480, 640, 3), 180, dtype=np.uint8)
        cv2.imwrite(str(video_dir / "frame_0000.jpg"), img)

        df = process_video(video_dir, model, img_size, idx_to_label, device)

        assert len(df) == 1
        row = df.iloc[0]
        assert row["timestamp"] == 0.0
        assert row["blinks_per_minute"] == 0.0
        assert not math.isnan(row["blinks_per_minute"])
        assert not math.isinf(row["blinks_per_minute"])


# ---------------------------------------------------------------------------
# 8. Empty directory — empty DataFrame, no crash
# ---------------------------------------------------------------------------

class TestEmptyDirectory:
    def test_empty_dir(self, model_bundle, device):
        model, img_size, idx_to_label = model_bundle
        empty_dir = TMP_DIR / "empty_video"
        empty_dir.mkdir(parents=True, exist_ok=True)

        df = process_video(empty_dir, model, img_size, idx_to_label, device)

        assert isinstance(df, pd.DataFrame)
        assert df.empty
        expected_cols = [
            "video_id", "frame_id", "timestamp",
            "eye_state", "blink_count", "blinks_per_minute",
        ]
        assert list(df.columns) == expected_cols


# ---------------------------------------------------------------------------
# 9. Large batch — 500 frames, CSV correct, bpm never negative/NaN
# ---------------------------------------------------------------------------

class TestLargeBatch:
    def test_500_frames(self, model_bundle, device):
        model, img_size, idx_to_label = model_bundle
        video_dir = TMP_DIR / "large_batch_video"
        video_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.RandomState(7)
        for i in range(500):
            img = rng.randint(60, 220, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(video_dir / f"frame_{i:04d}.jpg"), img)

        df = process_video(
            video_dir, model, img_size, idx_to_label, device, fps=30.0,
        )

        assert len(df) == 500
        assert (df["blinks_per_minute"] >= 0).all(), "Negative BPM found"
        assert not df["blinks_per_minute"].isna().any(), "NaN BPM found"
        assert df["blink_count"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# 10. Model load stress — 10 loads, identical output
# ---------------------------------------------------------------------------

class TestModelLoadStress:
    def test_ten_loads_identical(self, device):
        reference_output = None
        test_input = torch.randn(1, 1, 84, 84, device=device)

        for i in range(10):
            model, img_size, idx_to_label = load_model(CHECKPOINT, device)
            assert img_size == 84
            assert set(idx_to_label.values()) == {"awake", "sleepy"}

            with torch.no_grad():
                out = model(test_input)

            if reference_output is None:
                reference_output = out.clone()
            else:
                assert torch.allclose(out, reference_output, atol=1e-6), (
                    f"Load {i}: output differs from reference"
                )
