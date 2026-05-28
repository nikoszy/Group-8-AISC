#!/usr/bin/env python
"""
FaceSwap ensemble pipeline + ADV/OOD results generator.

Produces
--------
  data/experiments/faceswap/module3_features.csv
  data/experiments/faceswap/ensemble_model.pkl
  data/experiments/faceswap/plots/roc_curve.png
  data/experiments/faceswap/plots/precision_recall.png
  data/experiments/ml03_adv_ood_results.json

Timeout guards
--------------
  Each per-image extractor call is capped at FEATURE_TIMEOUT seconds via
  a daemon thread join.  If the thread is still alive after the cap the
  extractor returns the default value (0.0) and a warning is printed.
  This prevents a single misbehaving frame from stalling the pipeline.

Usage
-----
  cd <project-root>
  python scripts/run_faceswap_and_adv.py
"""

import csv
import json
import os
import sys
import threading
import time
import warnings
from datetime import datetime

# Ensure the project root (parent of this script's directory) is on sys.path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2
import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FS_DIR        = os.path.join("data", "experiments", "faceswap")
MANIFEST      = os.path.join(FS_DIR, "manifest.csv")
FEATURES_CSV  = os.path.join(FS_DIR, "module3_features.csv")
EAR_CSV       = os.path.join(FS_DIR, "video_ear_scores.csv")
MODEL_PKL     = os.path.join(FS_DIR, "ensemble_model.pkl")
PLOTS_DIR     = os.path.join(FS_DIR, "plots")
LOG_PATH      = os.path.join(FS_DIR, "ensemble.log")
ADV_JSON      = os.path.join("data", "experiments", "ml03_adv_ood_results.json")
VAL_SUMMARY   = os.path.join("data", "experiments", "ml03_validation_summary.json")

FEATURE_NAMES  = ["ear", "artifact", "fft", "laplacian"]
FEATURE_TIMEOUT = 5.0   # seconds per extractor call before fallback
MAX_FFR        = 0.10   # false-fake-rate constraint for threshold calibration
UNCERTAIN_BAND = 0.10

# ---------------------------------------------------------------------------
# Lazy imports (only import when feature extraction begins)
# ---------------------------------------------------------------------------
_ART = _FFT = _LAP = None

def _load_extractors():
    global _ART, _FFT, _LAP
    if _ART is None:
        from artifact_module import get_artifact_score_for_frame as _a
        from src.freq_analysis.anomaly_scorer import fft_anomaly_score as _f
        from src.freq_analysis.texture_scorer import laplacian_score as _l
        _ART, _FFT, _LAP = _a, _f, _l


# ---------------------------------------------------------------------------
# Timeout-guarded extractor
# ---------------------------------------------------------------------------

def _timed_call(fn, img, timeout=FEATURE_TIMEOUT, default=0.0):
    """
    Run fn(img) in a daemon thread.  Returns (value, error_str_or_None).
    If the thread does not finish within `timeout` seconds the function
    returns (default, "timeout").  The daemon thread continues in the
    background but will be collected at process exit.
    """
    result = [default]
    error  = [None]

    def _worker():
        try:
            result[0] = float(fn(img))
        except Exception as exc:
            error[0]  = str(exc)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return default, "timeout"
    if error[0]:
        return default, error[0]
    return result[0], None


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_manifest(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_ear_cache(path):
    cache = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                cache[row["video_id"]] = float(row["ear_score"])
    return cache


def load_features_csv(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    feats = np.array(
        [[float(r["ear_score"]), float(r["artifact_score"]),
          float(r["fft_score"]),  float(r["laplacian_score"])]
         for r in rows],
        dtype=np.float32,
    )
    labels   = np.array([int(r["label"]) for r in rows], dtype=int)
    video_ids = [r["video_id"] for r in rows]
    return feats, labels, video_ids


# ---------------------------------------------------------------------------
# Phase 1 — Feature extraction
# ---------------------------------------------------------------------------

def extract_features(manifest_rows, ear_cache, save_csv, verbose=True):
    _load_extractors()

    all_feats, all_labels, all_vids, csv_rows = [], [], [], []
    total   = len(manifest_rows)
    t_start = time.time()
    skipped = 0

    for i, row in enumerate(manifest_rows):
        path     = row["file_path"]
        label    = int(row["label"])
        video_id = row["video_id"]

        img = cv2.imread(path)
        if img is None:
            skipped += 1
            if verbose:
                print(f"  [SKIP] cannot read {os.path.basename(path)}")
            continue

        ear = float(ear_cache.get(video_id, 0.5))

        art, err = _timed_call(_ART, img)
        if err and verbose:
            print(f"  [WARN art {os.path.basename(path)}] {err}")

        fft, err = _timed_call(_FFT, img)
        if err and verbose:
            print(f"  [WARN fft {os.path.basename(path)}] {err}")

        lap, err = _timed_call(_LAP, img)
        if err and verbose:
            print(f"  [WARN lap {os.path.basename(path)}] {err}")

        all_feats.append([ear, art, fft, lap])
        all_labels.append(label)
        all_vids.append(video_id)
        csv_rows.append({
            "file_path"     : path,
            "label"         : label,
            "video_id"      : video_id,
            "ear_score"     : round(ear, 4),
            "artifact_score": round(art, 4),
            "fft_score"     : round(fft, 4),
            "laplacian_score": round(lap, 4),
        })

        if verbose and ((i + 1) % 50 == 0 or i == 0 or i == total - 1):
            elapsed = time.time() - t_start
            rate    = (i + 1) / elapsed if elapsed > 0 else 0
            eta     = (total - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1:4d}/{total}]  "
                f"ear={ear:.3f}  art={art:.4f}  fft={fft:.4f}  lap={lap:.4f}  "
                f"lbl={label}  elapsed={elapsed:.1f}s  eta={eta:.0f}s  "
                f"{os.path.basename(path)}"
            )

    if save_csv and csv_rows:
        os.makedirs(os.path.dirname(os.path.abspath(save_csv)), exist_ok=True)
        with open(save_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=[
                "file_path", "label", "video_id",
                "ear_score", "artifact_score", "fft_score", "laplacian_score",
            ])
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\n  Features saved ({len(csv_rows)} rows, {skipped} skipped): {save_csv}")

    return (
        np.array(all_feats,  dtype=np.float32),
        np.array(all_labels, dtype=int),
        all_vids,
    )


# ---------------------------------------------------------------------------
# Phase 2 — Training + calibration
# ---------------------------------------------------------------------------

def calibrate_constrained(y_true, y_scores, max_ffr=MAX_FFR):
    y_true   = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores, dtype=float)
    thresholds = np.linspace(0.01, 0.99, 197)
    candidates = []
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        real_total = tn + fp
        fpr = (fp / real_total) if real_total > 0 else 1.0
        bal = balanced_accuracy_score(y_true, y_pred)
        candidates.append({"t": float(t), "bal": float(bal), "fpr": float(fpr),
                            "ok": bool(fpr <= max_ffr)})
    valid = [c for c in candidates if c["ok"]]
    if valid:
        valid.sort(key=lambda c: (-c["bal"], c["fpr"], c["t"]))
        chosen = valid[0]
        status = "met"
    else:
        candidates.sort(key=lambda c: (c["fpr"], -c["bal"], c["t"]))
        chosen = candidates[0]
        status = "relaxed_to_lowest_false_fake"
    return chosen["t"], {
        "mode": "max_false_fake_rate",
        "max_false_fake_rate_real": max_ffr,
        "constraint_status": status,
        "threshold": round(chosen["t"], 4),
        "balanced_accuracy": round(chosen["bal"], 4),
        "false_fake_rate_real": round(chosen["fpr"], 4),
    }


def full_metrics(y_true, y_scores, threshold):
    y_pred = (y_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    real_total = tn + fp
    fake_total = fn + tp
    return {
        "accuracy"         : round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "auc"              : round(float(roc_auc_score(y_true, y_scores)), 4),
        "precision"        : round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall"           : round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1"               : round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "false_fake_rate_real": round(float(fp / real_total if real_total > 0 else 1.0), 4),
        "false_real_rate_fake": round(float(fn / fake_total if fake_total > 0 else 1.0), 4),
        "threshold"        : round(float(threshold), 4),
    }


def train_and_eval(features, labels, video_ids):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr_idx, va_idx = next(gss.split(features, labels, groups=video_ids))

    X_tr, X_va = features[tr_idx], features[va_idx]
    y_tr, y_va = labels[tr_idx],   labels[va_idx]
    v_va       = np.array(video_ids)[va_idx]

    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_va_s  = scaler.transform(X_va)

    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42,
                               class_weight="balanced")
    model.fit(X_tr_s, y_tr)

    val_scores = model.predict_proba(X_va_s)[:, 1]
    threshold, policy = calibrate_constrained(y_va, val_scores)
    metrics = full_metrics(y_va, val_scores, threshold)
    metrics.update({
        "n_val"            : int(len(y_va)),
        "val_real"         : int(np.sum(y_va == 0)),
        "val_fake"         : int(np.sum(y_va == 1)),
        "val_unique_videos": int(len(set(v_va))),
        "n_train"          : int(len(y_tr)),
    })

    # 5-fold GroupKFold cross-validation
    cv_aucs, cv_bals = [], []
    gkf = GroupKFold(n_splits=5)
    groups = np.array(video_ids)
    for fold_i, (tr, va) in enumerate(gkf.split(features, labels, groups=groups)):
        if len(np.unique(labels[va])) < 2:
            print(f"  [Fold {fold_i+1}] skipped — single class in val")
            continue
        sc = StandardScaler()
        m  = LogisticRegression(C=1.0, max_iter=1000, random_state=42,
                                class_weight="balanced")
        m.fit(sc.fit_transform(features[tr]), labels[tr])
        cv_s = m.predict_proba(sc.transform(features[va]))[:, 1]
        cv_aucs.append(roc_auc_score(labels[va], cv_s))
        # balanced accuracy at median threshold 0.5
        cv_bals.append(balanced_accuracy_score(
            labels[va], (cv_s >= 0.5).astype(int)
        ))
        n_r = int(np.sum(labels[va] == 0))
        n_f = int(np.sum(labels[va] == 1))
        print(f"  Fold {fold_i+1}/5  val={len(va)} ({n_r}r/{n_f}f)  "
              f"AUC={cv_aucs[-1]:.4f}  bal@0.5={cv_bals[-1]:.4f}")

    cv_summary = {}
    if cv_aucs:
        cv_summary = {
            "mean_auc"    : round(float(np.mean(cv_aucs)), 4),
            "std_auc"     : round(float(np.std(cv_aucs)),  4),
            "min_auc"     : round(float(np.min(cv_aucs)),  4),
            "max_auc"     : round(float(np.max(cv_aucs)),  4),
            "mean_bal_acc": round(float(np.mean(cv_bals)), 4),
            "folds"       : len(cv_aucs),
        }

    return model, scaler, threshold, policy, metrics, cv_summary, y_va, val_scores


def save_plots(y_true, y_scores, threshold, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Random (0.5)")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Curve — FaceSwap Ensemble")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.savefig(os.path.join(save_dir, "roc_curve.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    baseline = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, color="tomato", lw=2, label="Precision-Recall")
    ax.axhline(baseline, color="grey", linestyle="--", lw=1,
               label=f"Baseline = {baseline:.2f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall — FaceSwap")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    fig.savefig(os.path.join(save_dir, "precision_recall.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plots saved: {save_dir}/roc_curve.png, precision_recall.png")


# ---------------------------------------------------------------------------
# Phase 3 — ADV / OOD checks
# ---------------------------------------------------------------------------

def _score_features(feats_2d, model, scaler):
    """Return prob_fake scores for a (N×4) feature matrix."""
    return model.predict_proba(scaler.transform(feats_2d))[:, 1]


def augment_screen_recording(img):
    """Downsample -> upscale -> JPEG Q60 round-trip (screen-capture proxy)."""
    small = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
    up    = cv2.resize(small, (224, 224), interpolation=cv2.INTER_LINEAR)
    # JPEG round-trip at Q60 to simulate screen capture compression
    ok, enc = cv2.imencode(".jpg", up, [cv2.IMWRITE_JPEG_QUALITY, 60])
    if ok:
        up = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return up


def augment_cartoon_style(img):
    """Heavy bilateral filter to simulate stylised / OOD imagery."""
    cartoon = cv2.bilateralFilter(img, d=15, sigmaColor=80, sigmaSpace=80)
    return cartoon


def augment_partial_occlusion(img):
    """Black rectangle over bottom 40 % of the face crop."""
    out = img.copy()
    h, w = out.shape[:2]
    out[int(h * 0.60):, :] = 0
    return out


def _extract_feats_from_image(img, ear_value):
    """Extract [ear, art, fft, lap] from a single image with timeout guards."""
    _load_extractors()
    art, _ = _timed_call(_ART, img)
    fft, _ = _timed_call(_FFT, img)
    lap, _ = _timed_call(_LAP, img)
    return np.array([[ear_value, art, fft, lap]], dtype=np.float32)


def _adv_augmented_check(
    real_image_paths, ear_by_path,
    model, scaler, threshold,
    augment_fn, label, n_sample=80,
    rng=None,
):
    """
    Apply augment_fn to real face images, score with given model/scaler/threshold.
    Returns dict with false-fake rate and per-image scores.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    indices = rng.choice(len(real_image_paths),
                         size=min(n_sample, len(real_image_paths)),
                         replace=False)
    paths   = [real_image_paths[i] for i in indices]
    scores  = []
    skipped = 0

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            skipped += 1
            continue
        augmented  = augment_fn(img)
        ear_val    = float(ear_by_path.get(p, 1.0))  # real images -> ear~1.0
        feat_row   = _extract_feats_from_image(augmented, ear_val)
        prob_fake  = float(_score_features(feat_row, model, scaler)[0])
        scores.append(prob_fake)

    n_tested = len(scores)
    false_fake_n = sum(1 for s in scores if s >= threshold)
    false_fake_r = round(false_fake_n / n_tested, 4) if n_tested > 0 else None

    return {
        "augmentation"     : label,
        "n_real_tested"    : n_tested,
        "n_skipped"        : skipped,
        "false_fake_count" : false_fake_n,
        "false_fake_rate"  : false_fake_r,
        "threshold_used"   : round(float(threshold), 4),
        "score_mean"       : round(float(np.mean(scores)), 4) if scores else None,
        "score_std"        : round(float(np.std(scores)),  4) if scores else None,
    }


def adv_20_cross_family(families):
    """
    ADV-20: for each (model_family, target_family) pair where they differ,
    score target fakes with model_family's trained model.
    Returns list of check dicts.
    """
    checks = []
    family_names = list(families.keys())
    for src_name in family_names:
        src = families[src_name]
        model, scaler, threshold = src["model"], src["scaler"], src["threshold"]
        for tgt_name in family_names:
            if src_name == tgt_name:
                continue
            tgt      = families[tgt_name]
            feats    = tgt["features"]
            labels   = tgt["labels"]
            fake_idx = np.where(labels == 1)[0]
            if len(fake_idx) == 0:
                continue
            fake_feats  = feats[fake_idx]
            fake_scores = _score_features(fake_feats, model, scaler)
            false_real_n = int(np.sum(fake_scores < threshold))
            false_real_r = round(false_real_n / len(fake_idx), 4)
            print(f"  ADV-20  {src_name} model -> {tgt_name} fakes  "
                  f"n={len(fake_idx)}  false_real={false_real_r:.4f}  "
                  f"(threshold={threshold:.4f})")
            checks.append({
                "model_family"     : src_name,
                "test_family"      : tgt_name,
                "n_fakes_tested"   : int(len(fake_idx)),
                "false_real_count" : false_real_n,
                "false_real_rate"  : false_real_r,
                "threshold"        : round(float(threshold), 4),
            })
    return checks


# ---------------------------------------------------------------------------
# Logging: tee stdout to a log file
# ---------------------------------------------------------------------------
class _Tee:
    def __init__(self, *files):
        self._files = files

    def write(self, data):
        for f in self._files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self._files:
            f.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(FS_DIR, exist_ok=True)
    log_fh = open(LOG_PATH, "w", encoding="utf-8")
    orig_stdout = sys.stdout
    sys.stdout = _Tee(orig_stdout, log_fh)

    banner = "=" * 65
    print(banner)
    print("FACESWAP ENSEMBLE PIPELINE  +  ADV/OOD RESULTS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(banner)

    # ------------------------------------------------------------------
    # Prerequisites
    # ------------------------------------------------------------------
    for req in [MANIFEST, EAR_CSV]:
        if not os.path.exists(req):
            print(f"[ERROR] Required file not found: {req}")
            sys.exit(1)

    for exp_name in ("deepfakes", "face2face"):
        for fname in ("module3_features.csv", "ensemble_model.pkl"):
            p = os.path.join("data", "experiments", exp_name, fname)
            if not os.path.exists(p):
                print(f"[ERROR] Missing reference experiment artifact: {p}")
                print("       Run the full pipeline for deepfakes and face2face first.")
                sys.exit(1)

    # ------------------------------------------------------------------
    # PHASE 1 — Feature extraction
    # ------------------------------------------------------------------
    print("\nPHASE 1 — Feature extraction")
    print("-" * 65)

    manifest  = load_manifest(MANIFEST)
    ear_cache = load_ear_cache(EAR_CSV)
    n_real    = sum(1 for r in manifest if r["label"] == "0")
    n_fake    = sum(1 for r in manifest if r["label"] == "1")
    n_vids    = len(set(r["video_id"] for r in manifest))
    print(f"  Manifest  : {len(manifest)} rows  ({n_real} real, {n_fake} fake)")
    print(f"  EAR cache : {len(ear_cache)} entries")
    print(f"  Unique videos: {n_vids}")

    if os.path.exists(FEATURES_CSV):
        print(f"\n  Cache hit — loading features from {FEATURES_CSV}")
        features, labels, video_ids = load_features_csv(FEATURES_CSV)
        print(f"  Loaded {len(features)} rows.")
    else:
        print(f"\n  Extracting features (timeout guard: {FEATURE_TIMEOUT}s/call)...")
        t0 = time.time()
        features, labels, video_ids = extract_features(
            manifest, ear_cache, save_csv=FEATURES_CSV, verbose=True
        )
        print(f"\n  Extraction complete: {len(features)} rows in "
              f"{time.time()-t0:.1f}s")

    if len(features) == 0:
        print("[ERROR] No features extracted. Check image paths in manifest.")
        sys.exit(1)

    print(f"\n  Feature matrix: {features.shape}  cols={FEATURE_NAMES}")
    print(f"  Label vector  : {labels.shape}")

    # Feature distributions
    print("\n  FEATURE DISTRIBUTIONS (mean ± std)")
    header = f"  {'Feature':<14}  {'REAL (label=0)':>22}  {'FAKE (label=1)':>22}  {'Delta':>8}"
    print(header)
    print("  " + "-" * 72)
    for j, name in enumerate(FEATURE_NAMES):
        rv = features[labels == 0, j]
        fv = features[labels == 1, j]
        rm, rs = float(np.mean(rv)), float(np.std(rv))
        fm, fs = float(np.mean(fv)), float(np.std(fv))
        print(f"  {name:<14}  {rm:+.4f} ± {rs:.4f}          "
              f"{fm:+.4f} ± {fs:.4f}      {fm-rm:+.4f}")

    # ------------------------------------------------------------------
    # PHASE 2 — Training + evaluation
    # ------------------------------------------------------------------
    print("\nPHASE 2 — Ensemble training (video-level GroupShuffleSplit)")
    print("-" * 65)

    model, scaler, threshold, policy, metrics, cv_summary, y_va, val_scores = \
        train_and_eval(features, labels, video_ids)

    print(f"\n  Val set: {metrics['n_val']} frames "
          f"({metrics['val_real']} real, {metrics['val_fake']} fake, "
          f"{metrics['val_unique_videos']} videos)")

    coef      = model.coef_[0]
    norm_coef = np.abs(coef) / (np.abs(coef).sum() + 1e-8)
    print("\n  Coefficients (|w| normalised to sum=1):")
    for name, nc in zip(FEATURE_NAMES, norm_coef):
        bar = "#" * int(nc * 30)
        print(f"    {name:<12} = {nc:.3f}  {bar}")

    print(f"\n  CV summary: {cv_summary}")

    print(f"\n  Threshold : {threshold:.4f}  "
          f"(mode={policy['mode']}, status={policy['constraint_status']}, "
          f"ffr_real={policy['false_fake_rate_real']:.4f})")

    print("\n" + "=" * 55)
    print("EVALUATION RESULTS — FaceSwap")
    print("=" * 55)
    for k, v in metrics.items():
        print(f"  {k:<28}: {v}")
    print("=" * 55)

    # Confusion matrix
    y_pred = (val_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_va, y_pred, labels=[0, 1]).ravel()
    print("\n  CONFUSION MATRIX")
    print("                Predicted")
    print("                REAL   FAKE")
    print(f"  Actual  REAL  [{tn:4d}   {fp:4d}]   <- {fp} real wrongly flagged")
    print(f"          FAKE  [{fn:4d}   {tp:4d}]   <- {fn} fakes missed")

    # Plots
    print()
    save_plots(y_va, val_scores, threshold, PLOTS_DIR)

    # Save model bundle
    bundle = {
        "model"           : model,
        "scaler"          : scaler,
        "threshold"       : threshold,
        "threshold_mode"  : policy["mode"],
        "threshold_policy": policy,
        "uncertain_band"  : UNCERTAIN_BAND,
        "feature_names"   : FEATURE_NAMES,
    }
    os.makedirs(os.path.dirname(os.path.abspath(MODEL_PKL)), exist_ok=True)
    joblib.dump(bundle, MODEL_PKL)
    print(f"  Model bundle saved: {MODEL_PKL}")

    # ------------------------------------------------------------------
    # PHASE 3 — ADV / OOD
    # ------------------------------------------------------------------
    print("\nPHASE 3 — ADV / OOD checks")
    print("-" * 65)

    # Load all three family models and features
    families = {}
    for exp_name in ("deepfakes", "face2face", "faceswap"):
        exp_dir   = os.path.join("data", "experiments", exp_name)
        pkl_path  = os.path.join(exp_dir, "ensemble_model.pkl")
        feat_path = os.path.join(exp_dir, "module3_features.csv")
        b         = joblib.load(pkl_path)
        f, l, v   = load_features_csv(feat_path)
        families[exp_name] = {
            "model"    : b["model"],
            "scaler"   : b["scaler"],
            "threshold": float(b["threshold"]),
            "features" : f,
            "labels"   : l,
            "video_ids": v,
        }
    print(f"  Loaded models: {list(families.keys())}")

    # ADV-20: cross-family false-real
    print("\n  ADV-20 — cross-family false-real tests")
    adv_20_checks = adv_20_cross_family(families)

    # Build ear_by_path for real images (for ADV-21/22/23)
    # Use the faceswap manifest + EAR cache
    ear_by_path = {}
    for row in manifest:
        vid_id = row["video_id"]
        ear_val = float(ear_cache.get(vid_id, 1.0))  # real defaults to 1.0
        ear_by_path[row["file_path"]] = ear_val

    # Real image paths from FaceSwap
    real_paths = [
        row["file_path"]
        for row in manifest
        if row["label"] == "0"
    ]
    fs_model   = families["faceswap"]["model"]
    fs_scaler  = families["faceswap"]["scaler"]
    fs_thresh  = families["faceswap"]["threshold"]

    rng = np.random.default_rng(42)

    # ADV-21: screen-recording proxy
    print("\n  ADV-21 — screen-recording proxy (downsample+JPEG)")
    adv_21 = _adv_augmented_check(
        real_paths, ear_by_path,
        fs_model, fs_scaler, fs_thresh,
        augment_screen_recording, "screen_recording_proxy",
        n_sample=80, rng=rng,
    )
    print(f"  ADV-21  false-fake={adv_21['false_fake_rate']}  "
          f"n={adv_21['n_real_tested']}  "
          f"score_mean={adv_21['score_mean']}")

    # ADV-22: cartoon/OOD style
    print("\n  ADV-22 — cartoon / OOD style (heavy bilateral filter)")
    adv_22 = _adv_augmented_check(
        real_paths, ear_by_path,
        fs_model, fs_scaler, fs_thresh,
        augment_cartoon_style, "cartoon_ood_style",
        n_sample=80, rng=rng,
    )
    print(f"  ADV-22  false-fake={adv_22['false_fake_rate']}  "
          f"n={adv_22['n_real_tested']}  "
          f"score_mean={adv_22['score_mean']}")

    # ADV-23: partial occlusion
    print("\n  ADV-23 — partial-face occlusion (bottom 40 % blacked out)")
    adv_23 = _adv_augmented_check(
        real_paths, ear_by_path,
        fs_model, fs_scaler, fs_thresh,
        augment_partial_occlusion, "partial_face_occlusion",
        n_sample=80, rng=rng,
    )
    print(f"  ADV-23  false-fake={adv_23['false_fake_rate']}  "
          f"n={adv_23['n_real_tested']}  "
          f"score_mean={adv_23['score_mean']}")

    # ------------------------------------------------------------------
    # Assemble and save ADV JSON
    # ------------------------------------------------------------------
    adv_json = {
        "meta": {
            "generated_at"    : datetime.utcnow().isoformat() + "Z",
            "generated_by"    : "scripts/run_faceswap_and_adv.py",
            "git_head"        : _git_head(),
            "models": {
                exp: os.path.join("data", "experiments", exp, "ensemble_model.pkl")
                for exp in ("deepfakes", "face2face", "faceswap")
            },
        },
        "adv_20_family_shift": {
            "description": (
                "Cross-family false-real: each family's trained model is applied to "
                "fakes from the other two families. Low rate = model generalises. "
                "High rate = family-specific overfit."
            ),
            "checks": adv_20_checks,
        },
        "adv_21_screen_recording_proxy": {
            "description": (
                "Real face crops processed through downsample (112px) + upsample (224px) "
                "+ JPEG Q60 round-trip, simulating a screen-recording pipeline. "
                "false-fake-rate = fraction of real images flagged as fake."
            ),
            **adv_21,
        },
        "adv_22_cartoon_ood_style": {
            "description": (
                "Heavy bilateral filter (d=15, σ_color=80, σ_space=80) applied to "
                "real face crops to simulate cartoon/stylised OOD input. "
                "false-fake-rate = fraction flagged as fake."
            ),
            **adv_22,
        },
        "adv_23_partial_face_occlusion": {
            "description": (
                "Bottom 40 % of each real face crop is blacked out to simulate "
                "partial occlusion or low-frame cropping. "
                "false-fake-rate = fraction flagged as fake. "
                "Note: feature extraction operates on the cropped image directly "
                "(no re-detection), so face-detect hit-rate is not applicable here."
            ),
            **adv_23,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(ADV_JSON)), exist_ok=True)
    with open(ADV_JSON, "w", encoding="utf-8") as fh:
        json.dump(adv_json, fh, indent=2)
    print(f"\n  ADV/OOD JSON saved: {ADV_JSON}")

    # ------------------------------------------------------------------
    # Update ml03_validation_summary.json
    # ------------------------------------------------------------------
    _update_validation_summary(metrics, cv_summary, threshold)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + banner)
    print("RUN COMPLETE")
    print(banner)
    print(f"  FaceSwap features   : {FEATURES_CSV}")
    print(f"  FaceSwap model      : {MODEL_PKL}")
    print(f"  FaceSwap plots      : {PLOTS_DIR}")
    print(f"  ADV/OOD results     : {ADV_JSON}")
    print(f"  Log                 : {LOG_PATH}")
    print()
    print("  FACESWAP METRICS:")
    print(f"    AUC              = {metrics['auc']}")
    print(f"    Accuracy         = {metrics['accuracy']}")
    print(f"    Balanced Acc     = {metrics['balanced_accuracy']}")
    print(f"    F1               = {metrics['f1']}")
    print(f"    Threshold        = {metrics['threshold']}")
    print(f"    False-fake (real)= {metrics['false_fake_rate_real']}")
    if cv_summary:
        print(f"    CV AUC           = {cv_summary['mean_auc']} ± {cv_summary['std_auc']}")
    print()
    print("  ADV SUMMARY:")
    for check in adv_20_checks:
        print(f"    ADV-20  {check['model_family']:>10} -> {check['test_family']:<10}  "
              f"false_real={check['false_real_rate']}")
    print(f"    ADV-21 screen-rec  false-fake={adv_21['false_fake_rate']}")
    print(f"    ADV-22 cartoon     false-fake={adv_22['false_fake_rate']}")
    print(f"    ADV-23 occlusion   false-fake={adv_23['false_fake_rate']}")
    print()
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(banner)

    sys.stdout = orig_stdout
    log_fh.close()


def _git_head():
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _update_validation_summary(metrics, cv_summary, threshold):
    """Append FaceSwap results to ml03_validation_summary.json."""
    summary = {}
    if os.path.exists(VAL_SUMMARY):
        with open(VAL_SUMMARY, encoding="utf-8") as fh:
            summary = json.load(fh)

    summary.setdefault("tracks", {})
    summary["tracks"]["faceswap"] = {
        "artifacts": {
            "manifest"    : MANIFEST,
            "features_csv": FEATURES_CSV,
        },
        "status": "ok",
        "split": {
            "n_total"          : int(metrics["n_val"] + metrics["n_train"]),
            "n_train"          : int(metrics["n_train"]),
            "n_val"            : int(metrics["n_val"]),
            "val_real"         : int(metrics["val_real"]),
            "val_fake"         : int(metrics["val_fake"]),
            "val_unique_videos": int(metrics["val_unique_videos"]),
        },
        "metrics": {
            "auc"             : metrics["auc"],
            "accuracy"        : metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "f1"              : metrics["f1"],
            "threshold_used"  : round(float(threshold), 10),
        },
        "cv": cv_summary,
    }

    with open(VAL_SUMMARY, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Validation summary updated: {VAL_SUMMARY}")


if __name__ == "__main__":
    main()
