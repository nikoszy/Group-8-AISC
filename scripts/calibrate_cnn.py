#!/usr/bin/env python3
"""
scripts/calibrate_cnn.py
========================
Fit Platt scaling calibration on the trained CNN deepfake detector.

The CNN (EfficientNet-B0, cnn_detector.py) outputs raw logits.  After
training on FF++ C23, those logits may carry a systematic bias -- the model
can be over-confident in one direction on fresh real-world footage.

This script:
  1. Loads the trained CNN checkpoint (read-only -- the detector is NOT modified).
  2. Runs inference on real videos (and optionally fake/synthetic-positive videos)
     to collect raw logits.
  3. Reports the empirical CNN bias on real data:
       empirical_bias_real = mean(logit on real samples)
       Positive -> model over-predicts FAKE on real footage.
  4. Fits Platt scaling (one-dimensional logistic regression) to map raw logits
     to calibrated probabilities:
       P(fake | logit) = sigmoid(A * logit + B)
     where A and B are the saved calibration parameters.
  5. Writes configs/cnn_calibration.json.

Usage -- from raw videos (primary mode, as requested):
    python scripts/calibrate_cnn.py \\
        --real-dir  data/FaceForensics++_C23/original \\
        --fake-dir  data/FaceForensics++_C23/Deepfakes   # optional

Usage -- from pre-extracted face-crop images (faster, avoids re-extraction):
    python scripts/calibrate_cnn.py \\
        --real-crops data/real/frames \\
        --fake-crops data/fake/frames

Usage -- from the manifest (most thorough -- uses every extracted crop):
    python scripts/calibrate_cnn.py \\
        --manifest data/manifest.csv

All three modes write the same configs/cnn_calibration.json.

Applying the calibration later
------------------------------
    import json, math
    cfg = json.load(open("configs/cnn_calibration.json"))
    A, B = cfg["platt_A"], cfg["platt_B"]
    calibrated_prob = 1 / (1 + math.exp(-(A * raw_logit + B)))
"""

import argparse
import csv
import datetime
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from torchvision import transforms

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup -- project root is one level above scripts/
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# Add project root so we can import cnn_detector, src.*, etc.
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cnn_detector import build_model  # noqa: E402  (after sys.path setup)
from src.preprocessing.face_detector import detect_faces  # noqa: E402

# ---------------------------------------------------------------------------
# Constants (must match cnn_detector.py exactly)
# ---------------------------------------------------------------------------
_IMG_SIZE = 224
_MEAN     = [0.485, 0.456, 0.406]   # ImageNet mean
_STD      = [0.229, 0.224, 0.225]   # ImageNet std

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
_IMG_EXTS   = {".jpg", ".jpeg", ".png", ".bmp"}

_DEFAULT_MODEL_PATH  = str(_PROJECT_ROOT / "data" / "cnn_model.pth")
_DEFAULT_OUTPUT_PATH = str(_PROJECT_ROOT / "configs" / "cnn_calibration.json")


# ===========================================================================
# Model loading
# ===========================================================================

def load_cnn(model_path: str, device: torch.device) -> nn.Module:
    """
    Load the trained EfficientNet-B0 checkpoint (raw state_dict as saved
    by cnn_detector.py:  torch.save(model.state_dict(), MODEL_PATH)).

    Does NOT modify the detector -- the weights are loaded read-only and the
    model is placed in eval() mode.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"\n  [ERROR] Model checkpoint not found: {path}\n"
            "  Train the CNN first:\n"
            "      python cnn_detector.py\n"
        )

    # PyTorch >= 2.0 prefers weights_only=True for safety; fall back for older.
    try:
        state = torch.load(str(path), map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(str(path), map_location=device)

    # Support both a bare state_dict and a wrapped {"model_state_dict": ...}
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model = build_model()          # builds on cnn_detector.DEVICE
    model.load_state_dict(state)
    model.to(device)               # move to caller's device (may be a no-op)
    model.eval()
    return model


# ===========================================================================
# Transform (identical to cnn_detector.VAL_TF)
# ===========================================================================

def val_transform() -> transforms.Compose:
    """Validation transform that matches cnn_detector.py VAL_TF exactly."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((_IMG_SIZE, _IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


# ===========================================================================
# Single-image logit extraction
# ===========================================================================

@torch.no_grad()
def _bgr_to_logit(
    face_bgr: np.ndarray,
    model:    nn.Module,
    tf:       transforms.Compose,
    device:   torch.device,
) -> float:
    """
    Convert a BGR face crop (as returned by detect_faces()) to a raw CNN
    logit.  The logit is the scalar output of the final Linear layer,
    *before* sigmoid.  Positive logit -> model leans toward FAKE.
    """
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    tensor   = tf(face_rgb).unsqueeze(0).to(device)     # (1, 3, 224, 224)
    logit    = model(tensor).squeeze().item()            # scalar float
    return logit


# ===========================================================================
# Logit collectors -- three independent input modes
# ===========================================================================

def _collect_from_videos(
    video_dir:         str,
    label:             int,
    model:             nn.Module,
    tf:                transforms.Compose,
    device:            torch.device,
    max_videos:        int,
    frames_per_video:  int,
) -> tuple[list[float], list[int]]:
    """
    Walk *video_dir* recursively for video files.  For each video, sample
    *frames_per_video* frames uniformly, run Haar-cascade face detection, and
    run the CNN on every detected face.

    Returns (logits, labels) where each element corresponds to one face crop.
    """
    video_paths = sorted(
        p for p in Path(video_dir).rglob("*")
        if p.suffix.lower() in _VIDEO_EXTS
    )[:max_videos]

    if not video_paths:
        print(f"    [WARN] No video files found under: {video_dir}")
        return [], []

    label_name = "real" if label == 0 else "fake"
    print(f"    {len(video_paths):4d} {label_name} videos to process")

    logits: list[float] = []
    labels: list[int]   = []

    for vpath in video_paths:
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            print(f"      [SKIP] cannot open {vpath.name}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            cap.release()
            continue

        # Uniformly spaced frame indices
        n_sample = min(frames_per_video, total_frames)
        if total_frames <= frames_per_video:
            indices = list(range(total_frames))
        else:
            step    = total_frames // frames_per_video
            indices = [i * step for i in range(frames_per_video)]

        n_faces = 0
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            face = detect_faces(frame)
            if face is None:
                continue
            logits.append(_bgr_to_logit(face, model, tf, device))
            labels.append(label)
            n_faces += 1

        cap.release()
        # Only print per-video stats at high verbosity; suppress for long runs
        _ = n_sample   # keep linter happy

    return logits, labels


def _collect_from_crops(
    crops_dir:  str,
    label:      int,
    model:      nn.Module,
    tf:         transforms.Compose,
    device:     torch.device,
    max_crops:  Optional[int],
) -> tuple[list[float], list[int]]:
    """
    Walk *crops_dir* for JPEG/PNG face-crop images (already extracted by
    inspect_dataset.py) and run the CNN on each.

    Returns (logits, labels).
    """
    img_paths = sorted(
        p for p in Path(crops_dir).rglob("*")
        if p.suffix.lower() in _IMG_EXTS
    )
    if max_crops is not None:
        img_paths = img_paths[:max_crops]

    if not img_paths:
        print(f"    [WARN] No image files found under: {crops_dir}")
        return [], []

    label_name = "real" if label == 0 else "fake"
    print(f"    {len(img_paths):4d} {label_name} crops to process")

    logits: list[float] = []
    labels: list[int]   = []

    for ipath in img_paths:
        img = cv2.imread(str(ipath))
        if img is None:
            continue
        logits.append(_bgr_to_logit(img, model, tf, device))
        labels.append(label)

    return logits, labels


def _collect_from_manifest(
    manifest_path: str,
    model:         nn.Module,
    tf:            transforms.Compose,
    device:        torch.device,
) -> tuple[list[float], list[int]]:
    """
    Read data/manifest.csv (produced by inspect_dataset.py) and run the CNN
    on every listed face crop.  The manifest's file_path column is treated as
    relative to the project root.

    Returns (logits, labels).
    """
    mpath = Path(manifest_path)
    if not mpath.exists():
        raise FileNotFoundError(f"Manifest not found: {mpath}")

    rows: list[tuple[str, int]] = []
    with open(mpath, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append((row["file_path"], int(row["label"])))

    print(f"    {len(rows):4d} entries in manifest")

    logits: list[float] = []
    labels: list[int]   = []

    for fpath_str, label in rows:
        # Try project-root-relative path first, then as-is
        full = _PROJECT_ROOT / fpath_str
        img  = cv2.imread(str(full))
        if img is None:
            img = cv2.imread(fpath_str)
        if img is None:
            continue
        logits.append(_bgr_to_logit(img, model, tf, device))
        labels.append(label)

    return logits, labels


# ===========================================================================
# Bias statistics
# ===========================================================================

def compute_bias_stats(logits_real: list[float]) -> dict:
    """
    Empirical CNN bias computed on real (negative-class) samples.

    The *empirical_bias_real* field equals the mean raw logit on real faces.
      Positive -> model over-predicts FAKE on real data (needs correction).
      Negative -> model correctly leans toward REAL on real data.

    Returns a dict of descriptive statistics for the real-face logit
    distribution, all suitable for JSON serialisation.
    """
    arr   = np.asarray(logits_real, dtype=np.float64)
    probs = 1.0 / (1.0 + np.exp(-arr))    # sigmoid -> P(fake) per real face
    return {
        "empirical_bias_real" : float(np.mean(arr)),    # central field
        "mean_logit_real"     : float(np.mean(arr)),
        "std_logit_real"      : float(np.std(arr)),
        "median_logit_real"   : float(np.median(arr)),
        "p25_logit_real"      : float(np.percentile(arr, 25)),
        "p75_logit_real"      : float(np.percentile(arr, 75)),
        "mean_prob_real"      : float(np.mean(probs)),  # sigmoid of mean logit
    }


# ===========================================================================
# Platt scaling
# ===========================================================================

def fit_platt(
    logits: list[float],
    labels: list[int],
) -> tuple[float, float]:
    """
    Fit Platt scaling:  P(fake | logit) = sigmoid(A * logit + B).

    Trains a one-dimensional logistic regression on the raw CNN logits.
    Regularisation C=1.0 provides mild smoothing consistent with Platt (1999).

    Returns
    -------
    A : float  -- logit scale factor (~1.0 if model is already calibrated)
    B : float  -- intercept / bias correction

    Interpretation
    --------------
    A > 1   Model under-confident; calibration amplifies separation.
    A < 1   Model over-confident; calibration shrinks separation.
    B < 0   Model biased toward FAKE; negative intercept corrects it.
    B > 0   Model biased toward REAL.
    """
    X  = np.asarray(logits, dtype=np.float64).reshape(-1, 1)
    y  = np.asarray(labels, dtype=int)
    lr = LogisticRegression(
        C=1.0, fit_intercept=True,
        max_iter=2000, solver="lbfgs",
    )
    lr.fit(X, y)
    A = float(lr.coef_[0][0])
    B = float(lr.intercept_[0])
    return A, B


def bias_only_correction(logits_real: list[float]) -> tuple[float, float]:
    """
    Fallback calibration when only real (negative) samples are available.

    Sets A=1.0 (no slope change) and B = -mean_logit_real, so that the
    average real-face logit maps to calibrated P(fake) = sigmoid(0) = 0.5.
    This corrects the DC offset (bias) but cannot calibrate slope.

    Provide --fake-dir or --fake-crops for full Platt scaling.
    """
    B = float(-np.mean(logits_real))
    return 1.0, B


# ===========================================================================
# Calibration quality
# ===========================================================================

def calibration_quality(
    logits: list[float],
    labels: list[int],
    A:      float,
    B:      float,
) -> dict:
    """
    Compute quality metrics before and after Platt scaling.

    Returns a dict suitable for JSON serialisation.
    Metrics are only computed when both classes are present.
    """
    arr = np.asarray(logits, dtype=np.float64)
    y   = np.asarray(labels, dtype=int)

    if len(np.unique(y)) < 2:
        return {}   # cannot compute AUC / Brier without both classes

    raw_probs = 1.0 / (1.0 + np.exp(-arr))
    cal_probs = 1.0 / (1.0 + np.exp(-(A * arr + B)))

    return {
        "brier_score_raw"        : float(brier_score_loss(y, raw_probs)),
        "brier_score_calibrated" : float(brier_score_loss(y, cal_probs)),
        "auc_raw"                : float(roc_auc_score(y, raw_probs)),
        "auc_calibrated"         : float(roc_auc_score(y, cal_probs)),
    }


# ===========================================================================
# Convenience -- apply calibration at inference time
# ===========================================================================

def apply_calibration(raw_logit: float, A: float, B: float) -> float:
    """
    Apply Platt scaling at inference time.

        calibrated_prob = sigmoid(A * raw_logit + B)

    Import this function from other modules instead of recomputing manually.

    Parameters
    ----------
    raw_logit : float   Raw scalar output of the CNN (before any sigmoid).
    A         : float   platt_A from configs/cnn_calibration.json.
    B         : float   platt_B from configs/cnn_calibration.json.

    Returns
    -------
    float in [0, 1]  Calibrated P(fake).
    """
    return float(1.0 / (1.0 + np.exp(-(A * raw_logit + B))))


# ===========================================================================
# Output
# ===========================================================================

def save_calibration(params: dict, output_path: str) -> None:
    """Write calibration parameters to a JSON file, creating dirs as needed."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(params, fh, indent=2)
    print(f"\n  Saved -> {out}")


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="calibrate_cnn",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Input (three mutually-suggested modes) ---
    inp = p.add_argument_group(
        "Input  (choose one mode; modes may be combined for real+fake)"
    )
    inp.add_argument(
        "--real-dir", metavar="PATH",
        help="Directory of real videos (.mp4/.avi/...) -- label=0",
    )
    inp.add_argument(
        "--fake-dir", metavar="PATH",
        help="Directory of fake/deepfake videos -- label=1  (optional)",
    )
    inp.add_argument(
        "--real-crops", metavar="PATH",
        help="Directory of pre-extracted real face-crop JPEGs -- label=0",
    )
    inp.add_argument(
        "--fake-crops", metavar="PATH",
        help="Directory of pre-extracted fake face-crop JPEGs -- label=1",
    )
    inp.add_argument(
        "--manifest", metavar="PATH",
        help="Path to manifest.csv (uses every extracted crop, both classes)",
    )

    # --- Model ---
    p.add_argument(
        "--model-path", metavar="PATH", default=_DEFAULT_MODEL_PATH,
        help=f"CNN checkpoint  [default: {_DEFAULT_MODEL_PATH}]",
    )

    # --- Output ---
    p.add_argument(
        "--output", metavar="PATH", default=_DEFAULT_OUTPUT_PATH,
        help=f"Output JSON path  [default: {_DEFAULT_OUTPUT_PATH}]",
    )

    # --- Video-mode sampling ---
    vid = p.add_argument_group("Video sampling  (--real-dir / --fake-dir mode)")
    vid.add_argument(
        "--max-videos", type=int, default=100, metavar="N",
        help="Max videos per class  [default: 100]",
    )
    vid.add_argument(
        "--frames-per-video", type=int, default=4, metavar="N",
        help="Frames sampled uniformly from each video  [default: 4]",
    )

    # --- Crop-mode limit ---
    p.add_argument(
        "--max-crops", type=int, default=None, metavar="N",
        help="Max crops per class in crop mode  [default: all]",
    )

    # --- Hardware ---
    p.add_argument(
        "--device", default=None, metavar="DEVICE",
        help="Compute device: 'cpu', 'cuda', 'cuda:0'  [default: auto]",
    )

    return p.parse_args()


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args = _parse_args()

    # --- Device ---
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    banner = "=" * 62
    print(f"\n{banner}")
    print("CNN CALIBRATION  (Platt scaling)")
    print(banner)
    print(f"  Device     : {device}")
    print(f"  Model      : {args.model_path}")
    print(f"  Output     : {args.output}")
    print()

    # -----------------------------------------------------------------------
    # Step 1 -- Load model
    # -----------------------------------------------------------------------
    print("[1] Loading CNN checkpoint (read-only)...")
    model = load_cnn(args.model_path, device)
    tf    = val_transform()
    print("    OK -- EfficientNet-B0 loaded in eval() mode")
    print()

    # -----------------------------------------------------------------------
    # Step 2 -- Collect logits (choose input mode)
    # -----------------------------------------------------------------------
    real_logits: list[float] = []
    fake_logits: list[float] = []
    source_desc: str         = ""

    if args.manifest:
        # Manifest mode -- most thorough, both classes at once
        print(f"[2] Collecting logits from manifest: {args.manifest}")
        all_logits, all_raw_labels = _collect_from_manifest(
            args.manifest, model, tf, device,
        )
        real_logits = [l for l, lb in zip(all_logits, all_raw_labels) if lb == 0]
        fake_logits = [l for l, lb in zip(all_logits, all_raw_labels) if lb == 1]
        source_desc = str(args.manifest)

    elif args.real_crops or args.fake_crops:
        # Pre-extracted crops mode -- fast, no video I/O
        print("[2] Collecting logits from pre-extracted crops...")
        if args.real_crops:
            print(f"    Real crops dir: {args.real_crops}")
            real_logits, _ = _collect_from_crops(
                args.real_crops, label=0,
                model=model, tf=tf, device=device,
                max_crops=args.max_crops,
            )
        if args.fake_crops:
            print(f"    Fake crops dir: {args.fake_crops}")
            fake_logits, _ = _collect_from_crops(
                args.fake_crops, label=1,
                model=model, tf=tf, device=device,
                max_crops=args.max_crops,
            )
        parts = [str(args.real_crops or ""), str(args.fake_crops or "")]
        source_desc = " + ".join(p for p in parts if p)

    else:
        # Video mode -- primary mode per user request
        if not args.real_dir:
            print("  [ERROR] Provide at least one of: "
                  "--real-dir, --real-crops, or --manifest")
            sys.exit(1)

        print("[2] Collecting logits from videos...")
        print(f"    Real dir : {args.real_dir}")
        real_logits, _ = _collect_from_videos(
            args.real_dir, label=0,
            model=model, tf=tf, device=device,
            max_videos=args.max_videos,
            frames_per_video=args.frames_per_video,
        )
        if args.fake_dir:
            print(f"    Fake dir : {args.fake_dir}")
            fake_logits, _ = _collect_from_videos(
                args.fake_dir, label=1,
                model=model, tf=tf, device=device,
                max_videos=args.max_videos,
                frames_per_video=args.frames_per_video,
            )
        parts = [str(args.real_dir), str(args.fake_dir or "")]
        source_desc = " + ".join(p for p in parts if p)

    print()
    print(f"    Real samples : {len(real_logits)}")
    print(f"    Fake samples : {len(fake_logits)}")
    print()

    if not real_logits:
        print("[ERROR] No real samples collected.  Check your --real-dir / "
              "--real-crops / --manifest argument.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 3 -- Empirical bias on real samples
    # -----------------------------------------------------------------------
    print("[3] Empirical CNN bias on real samples...")
    bias = compute_bias_stats(real_logits)

    print(f"    Mean logit   (real) = {bias['mean_logit_real']:+.4f}")
    print(f"    Median logit (real) = {bias['median_logit_real']:+.4f}")
    print(f"    Std  logit   (real) = {bias['std_logit_real']:.4f}")
    print(f"    IQR  logit   (real) = "
          f"[{bias['p25_logit_real']:+.3f}, {bias['p75_logit_real']:+.3f}]")
    print(f"    Mean P(fake) (real) = {bias['mean_prob_real']:.4f}")

    eb = bias["empirical_bias_real"]
    if eb > 0.3:
        print(f"    [!] Large positive bias ({eb:+.4f}): "
              "model strongly over-predicts FAKE on real data.")
    elif eb > 0.05:
        print(f"    [!] Mild positive bias ({eb:+.4f}): "
              "model slightly over-predicts FAKE on real data.")
    elif eb < -0.5:
        print(f"    [ok] Strongly negative bias ({eb:+.4f}): "
              "model correctly favours REAL on real data.")
    else:
        print(f"    [ok] Bias is within acceptable range ({eb:+.4f}).")
    print()

    # -----------------------------------------------------------------------
    # Step 4 -- Platt scaling
    # -----------------------------------------------------------------------
    print("[4] Fitting Platt scaling...")

    all_logits = real_logits + fake_logits
    all_labels = [0] * len(real_logits) + [1] * len(fake_logits)

    if len(set(all_labels)) < 2:
        print("    Only one class available -- applying bias-only correction.")
        print("    Provide --fake-dir or --fake-crops for full Platt scaling.")
        platt_A, platt_B = bias_only_correction(real_logits)
        mode = "bias_only"
    else:
        platt_A, platt_B = fit_platt(all_logits, all_labels)
        mode = "platt_scaling"

    print(f"    Mode          = {mode}")
    print(f"    A  (slope)    = {platt_A:+.6f}")
    print(f"    B  (intercept)= {platt_B:+.6f}")

    # Spot-check on the mean real logit
    mean_rl    = bias["mean_logit_real"]
    cal_at_mean = apply_calibration(mean_rl, platt_A, platt_B)
    print()
    print("    Spot-check:")
    print(f"      mean real logit = {mean_rl:+.4f}")
    print(f"      -> calibrated P(fake) = {cal_at_mean:.4f}  "
          f"(ideal < 0.5 for real faces)")
    print(f"      At logit=0:   P(fake) = {apply_calibration(0.0,  platt_A, platt_B):.4f}")
    print(f"      At logit=-1:  P(fake) = {apply_calibration(-1.0, platt_A, platt_B):.4f}")
    print(f"      At logit=+1:  P(fake) = {apply_calibration(+1.0, platt_A, platt_B):.4f}")
    print()

    # -----------------------------------------------------------------------
    # Step 5 -- Calibration quality metrics (requires both classes)
    # -----------------------------------------------------------------------
    quality: dict = {}
    if len(set(all_labels)) > 1:
        print("[5] Calibration quality metrics...")
        quality = calibration_quality(all_logits, all_labels, platt_A, platt_B)
        bs_raw = quality.get("brier_score_raw", float("nan"))
        bs_cal = quality.get("brier_score_calibrated", float("nan"))
        au_raw = quality.get("auc_raw", float("nan"))
        au_cal = quality.get("auc_calibrated", float("nan"))
        print(f"    Brier score  raw        = {bs_raw:.4f}")
        direction = "better (lower)" if bs_cal < bs_raw else "worse (higher)"
        print(f"    Brier score  calibrated = {bs_cal:.4f}  ({direction})")
        print(f"    AUC          raw        = {au_raw:.4f}")
        print(f"    AUC          calibrated = {au_cal:.4f}  "
              "(AUC is rank-based -- Platt does not change it)")
        if bs_cal > bs_raw + 0.01:
            print("    [!] Calibration worsened Brier score -- "
                  "consider more samples or checking class balance.")
        print()

    # -----------------------------------------------------------------------
    # Step 6 -- Assemble and save JSON
    # -----------------------------------------------------------------------
    print("[6] Writing calibration artifact...")

    calibration_doc: dict = {
        # -- Core Platt parameters (use these at inference) ------------------
        "platt_A"               : platt_A,
        "platt_B"               : platt_B,
        "mode"                  : mode,

        # -- Real-face logit distribution (diagnostic) -----------------------
        "empirical_bias_real"   : bias["empirical_bias_real"],
        "mean_logit_real"       : bias["mean_logit_real"],
        "std_logit_real"        : bias["std_logit_real"],
        "median_logit_real"     : bias["median_logit_real"],
        "p25_logit_real"        : bias["p25_logit_real"],
        "p75_logit_real"        : bias["p75_logit_real"],
        "mean_prob_real"        : bias["mean_prob_real"],

        # -- Sample counts ----------------------------------------------------
        "n_real_samples"        : len(real_logits),
        "n_fake_samples"        : len(fake_logits),

        # -- Quality metrics (may be absent when single-class) ---------------
        **quality,

        # -- Provenance -------------------------------------------------------
        "model_path"            : str(args.model_path),
        "data_source"           : source_desc,
        "real_video_dir"        : str(args.real_dir)   if args.real_dir   else None,
        "fake_video_dir"        : str(args.fake_dir)   if args.fake_dir   else None,
        "real_crops_dir"        : str(args.real_crops) if args.real_crops else None,
        "fake_crops_dir"        : str(args.fake_crops) if args.fake_crops else None,
        "manifest"              : str(args.manifest)   if args.manifest   else None,
        "max_videos"            : args.max_videos,
        "frames_per_video"      : args.frames_per_video,
        "max_crops"             : args.max_crops,
        "calibration_date"      : datetime.datetime.now().isoformat(),

        # -- Usage note -------------------------------------------------------
        "usage": (
            "Apply at inference: "
            "from scripts.calibrate_cnn import apply_calibration; "
            "p = apply_calibration(raw_logit, platt_A, platt_B)  "
            "-- OR --  "
            "import math; p = 1/(1+math.exp(-(platt_A*raw_logit + platt_B)))"
        ),
    }

    save_calibration(calibration_doc, args.output)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print(banner)
    print("CALIBRATION COMPLETE")
    print(banner)
    print(f"  P(fake) = sigmoid({platt_A:+.4f} * raw_logit "
          f"{'+ ' if platt_B >= 0 else '- '}{abs(platt_B):.4f})")
    print(f"  Empirical bias on real data : {eb:+.4f}")
    print(f"  Real samples used           : {len(real_logits)}")
    print(f"  Fake samples used           : {len(fake_logits)}")
    if quality:
        print(f"  Brier score improvement     : "
              f"{quality['brier_score_raw']:.4f} -> "
              f"{quality['brier_score_calibrated']:.4f}")
    print(f"  Artifact                    : {args.output}")
    print()


if __name__ == "__main__":
    main()
