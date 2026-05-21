"""Stress tests for src.mrl.inference — EyeCropper, classify_frame,
BlinkCounter, load_model, and process_video.

All synthetic images are generated in-memory (numpy + cv2) and written to
``tests/tmp/`` when disk files are needed.  The directory is cleaned up after
each test via the ``tmp_dir`` fixture.

Run:
    pytest tests/test_inference_stress.py -v
"""

from __future__ import annotations

import shutil
import time
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
from src.mrl.preprocess import TinyCNN

TMP_ROOT = Path(__file__).resolve().parent / "tmp"


@pytest.fixture(autouse=True)
def tmp_dir():
    """Create and tear down tests/tmp/ for every test."""
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    yield TMP_ROOT
    shutil.rmtree(TMP_ROOT, ignore_errors=True)


def _save_model(path: Path) -> None:
    """Save a fresh TinyCNN checkpoint to *path*."""
    model = TinyCNN(num_classes=2, in_ch=1)
    torch.save(model.state_dict(), str(path))


# ── helpers ──────────────────────────────────────────────────────────────


def _random_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _dark_frame(h: int = 480, w: int = 640, max_val: int = 20) -> np.ndarray:
    return np.random.randint(0, max_val, (h, w, 3), dtype=np.uint8)


def _white_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.full((h, w, 3), 255, dtype=np.uint8)


# ── 1. Synthetic face frames ────────────────────────────────────────────


def test_synthetic_face_frames():
    """100 random 640×480 RGB frames → crop_eyes() never crashes, always
    returns a list."""
    t0 = time.perf_counter()
    cropper = EyeCropper()
    for _ in range(100):
        frame = _random_frame()
        result = cropper.crop_eyes(frame)
        assert isinstance(result, list)
    elapsed = time.perf_counter() - t0
    print(f"\n[PASS] test_synthetic_face_frames — {elapsed:.3f}s")


# ── 2. Dark frames ──────────────────────────────────────────────────────


def test_dark_frames_return_unknown():
    """50 frames with mean pixel < 30 → classify_frame returns 'unknown'."""
    t0 = time.perf_counter()
    model_path = TMP_ROOT / "dark_model.pth"
    _save_model(model_path)
    model, dev = load_model(str(model_path))

    for _ in range(50):
        frame = _dark_frame()
        assert frame.mean() < 30, "dark_frame fixture must have mean < 30"
        state = classify_frame(frame, model=model, device=str(dev))
        assert state == "unknown", f"Expected 'unknown' for dark frame, got '{state}'"
    elapsed = time.perf_counter() - t0
    print(f"\n[PASS] test_dark_frames_return_unknown — {elapsed:.3f}s")


# ── 3. Blank white frames ───────────────────────────────────────────────


def test_blank_white_frames_no_crash():
    """50 pure-white frames → no crashes."""
    t0 = time.perf_counter()
    cropper = EyeCropper()
    model_path = TMP_ROOT / "white_model.pth"
    _save_model(model_path)
    model, dev = load_model(str(model_path))

    for _ in range(50):
        frame = _white_frame()
        eyes = cropper.crop_eyes(frame)
        assert isinstance(eyes, list)
        state = classify_frame(frame, model=model, device=str(dev))
        assert state in ("awake", "sleepy", "unknown")
    elapsed = time.perf_counter() - t0
    print(f"\n[PASS] test_blank_white_frames_no_crash — {elapsed:.3f}s")


# ── 4. Blink sequence stress test ───────────────────────────────────────


def test_blink_sequence_stress():
    """1000 random states → blink count never decreases and never exceeds
    total frames."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(42)
    states = rng.choice(["awake", "sleepy", "unknown"], size=1000).tolist()

    counter = BlinkCounter()
    prev_count = 0
    for i, state in enumerate(states):
        count = counter.update(state)
        assert count >= prev_count, (
            f"Blink count decreased at frame {i}: {prev_count} -> {count}"
        )
        assert count <= counter.total_frames, (
            f"Blink count {count} exceeds total frames {counter.total_frames}"
        )
        prev_count = count
    elapsed = time.perf_counter() - t0
    print(
        f"\n[PASS] test_blink_sequence_stress — {elapsed:.3f}s "
        f"(final blinks={counter.blink_count})"
    )


# ── 5. Rapid state changes ──────────────────────────────────────────────


def test_rapid_state_changes():
    """500 alternating awake/sleepy → exactly 250 blinks."""
    t0 = time.perf_counter()
    counter = BlinkCounter()
    for i in range(500):
        state = "awake" if i % 2 == 0 else "sleepy"
        counter.update(state)

    assert counter.blink_count == 250, (
        f"Expected 250 blinks, got {counter.blink_count}"
    )
    elapsed = time.perf_counter() - t0
    print(f"\n[PASS] test_rapid_state_changes — {elapsed:.3f}s")


# ── 6. All unknown sequence ─────────────────────────────────────────────


def test_all_unknown_sequence():
    """100 'unknown' states → blink count stays 0."""
    t0 = time.perf_counter()
    counter = BlinkCounter()
    for _ in range(100):
        counter.update("unknown")
    assert counter.blink_count == 0, (
        f"Expected 0 blinks for all-unknown, got {counter.blink_count}"
    )
    elapsed = time.perf_counter() - t0
    print(f"\n[PASS] test_all_unknown_sequence — {elapsed:.3f}s")


# ── 7. Single frame video ───────────────────────────────────────────────


def test_single_frame_video():
    """Directory with exactly one JPEG → CSV has one row, no division-by-zero
    errors."""
    t0 = time.perf_counter()
    frame_dir = TMP_ROOT / "single_frame"
    frame_dir.mkdir(parents=True, exist_ok=True)

    img = _random_frame(224, 224)
    cv2.imwrite(str(frame_dir / "frame_000.jpg"), img)

    df = process_video(str(frame_dir))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1, f"Expected 1 row, got {len(df)}"
    assert not df["blinks_per_minute"].isna().any(), "blinks_per_minute contains NaN"
    assert np.isfinite(df["blinks_per_minute"].values).all(), (
        "blinks_per_minute contains non-finite values"
    )
    elapsed = time.perf_counter() - t0
    print(f"\n[PASS] test_single_frame_video — {elapsed:.3f}s")


# ── 8. Empty directory ──────────────────────────────────────────────────


def test_empty_directory():
    """Empty directory → returns empty DataFrame without crashing."""
    t0 = time.perf_counter()
    empty_dir = TMP_ROOT / "empty"
    empty_dir.mkdir(parents=True, exist_ok=True)

    df = process_video(str(empty_dir))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0, f"Expected 0 rows, got {len(df)}"
    assert list(df.columns) == [
        "frame_file",
        "eye_state",
        "blink_count",
        "blinks_per_minute",
    ]
    elapsed = time.perf_counter() - t0
    print(f"\n[PASS] test_empty_directory — {elapsed:.3f}s")


# ── 9. Large batch ──────────────────────────────────────────────────────


def test_large_batch():
    """500 synthetic JPEG frames → full pipeline → 500 rows,
    blinks_per_minute is never negative or NaN."""
    t0 = time.perf_counter()
    batch_dir = TMP_ROOT / "large_batch"
    batch_dir.mkdir(parents=True, exist_ok=True)

    for i in range(500):
        img = _random_frame(224, 224)
        cv2.imwrite(str(batch_dir / f"frame_{i:04d}.jpg"), img)

    model_path = TMP_ROOT / "batch_model.pth"
    _save_model(model_path)
    model, dev = load_model(str(model_path))

    df = process_video(str(batch_dir), model=model, device=str(dev))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 500, f"Expected 500 rows, got {len(df)}"
    assert not df["blinks_per_minute"].isna().any(), "blinks_per_minute has NaN"
    assert (df["blinks_per_minute"] >= 0).all(), "blinks_per_minute has negatives"
    elapsed = time.perf_counter() - t0
    print(f"\n[PASS] test_large_batch — {elapsed:.3f}s")


# ── 10. Model load stress ───────────────────────────────────────────────


def test_model_load_stress():
    """Load best_model.pth 10 times → each load produces identical output on
    the same input tensor."""
    t0 = time.perf_counter()
    model_path = TMP_ROOT / "best_model.pth"
    _save_model(model_path)

    rng = np.random.default_rng(99)
    test_input = torch.from_numpy(
        rng.random((1, 1, 84, 84), dtype=np.float32)
    )

    outputs = []
    for _ in range(10):
        model, dev = load_model(str(model_path))
        with torch.no_grad():
            out = model(test_input.to(dev))
        outputs.append(out.cpu())

    for i in range(1, len(outputs)):
        assert torch.allclose(outputs[0], outputs[i], atol=1e-6), (
            f"Load {i} produced different output: "
            f"{outputs[0]} vs {outputs[i]}"
        )
    elapsed = time.perf_counter() - t0
    print(f"\n[PASS] test_model_load_stress — {elapsed:.3f}s")
