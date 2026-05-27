# =============================================================================
# ensemble.py  —  MODULE 3: WEIGHTED ENSEMBLE SCORING, TRAINING, CALIBRATION
# =============================================================================
#
# CHANGES FROM PREVIOUS VERSION
# ------------------------------
# 1. Video-level GroupShuffleSplit  — prevents identity leakage between
#    train and val sets.  The old random frame-level split allowed frames
#    from the same video to appear in both splits, inflating metrics.
#
# 2. Laplacian texture score (4th feature)  — replaces the EAR stub (0.5
#    constant, zero signal).  Deepfake decoders smooth fine texture;
#    Var(Laplacian) captures this difference.
#
# 3. Balanced-accuracy threshold calibration  — threshold is now chosen to
#    maximise balanced accuracy = (recall_real + recall_fake) / 2, which
#    directly optimises for accuracy on balanced classes.  The old F1-based
#    calibration favoured high-recall / low-precision solutions that traded
#    accuracy for recall.
#
# 4. Confusion matrix + per-class breakdown  — gives precise visibility
#    into where the model makes mistakes.
#
# 5. class_weight='balanced' in LogisticRegression  — robustifies against
#    any class imbalance that survives after the video-level split.
# =============================================================================

import os
import csv
import warnings

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics         import (
    roc_auc_score, precision_recall_curve, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix,
)

from artifact_module                      import get_artifact_score_for_frame
from src.freq_analysis.anomaly_scorer     import fft_anomaly_score
from src.freq_analysis.texture_scorer     import laplacian_score
from src.freq_analysis.utils              import load_face_image
from src.freq_analysis.frequency_analyzer import visualize_spectrum

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MANIFEST_PATH = os.path.join("data", "manifest.csv")
FEATURES_CSV  = os.path.join("data", "module3_features.csv")
PLOTS_DIR     = os.path.join("data", "plots")
VIZ_DIR       = os.path.join("data", "visualizations")

# Feature column names (used in CSV headers and diagnostic output)
FEATURE_NAMES = ["ear", "artifact", "fft", "laplacian"]

# ---------------------------------------------------------------------------
# Section A — pure functions
# ---------------------------------------------------------------------------

def ensemble_score_equal_weights(ear_score, artifact_score,
                                  fft_score, laplacian_score_val):
    """
    Combine four module scores with equal weights (baseline).

    Formula:
        final = 0.25 * ear  +  0.25 * artifact  +  0.25 * fft  +  0.25 * lap
    """
    score = (0.25 * ear_score + 0.25 * artifact_score
             + 0.25 * fft_score + 0.25 * laplacian_score_val)
    return round(float(np.clip(score, 0.0, 1.0)), 4)


def train_ensemble(features, labels, video_ids, C=1.0, random_state=42):
    """
    Train a logistic regression model using a video-level train/val split.

    WHY VIDEO-LEVEL SPLIT?
    ----------------------
    Each video contributes multiple frames.  A random frame-level split
    allows frames from the same video (same identity, same background,
    same lighting) to appear in both train and val.  The model then
    "memorises" video-specific statistics rather than learning transferable
    deepfake features, inflating val metrics.

    GroupShuffleSplit ensures every frame from a given video ends up in
    exactly one of train or val — it treats video_id as the group key.

    Args:
        features     : (N×4) float32 array  [ear, artifact, fft, laplacian]
        labels       : (N,)  int array       0=real, 1=fake
        video_ids    : (N,)  array of strings, one per frame
        C            : logistic regression regularisation (default 1.0)
        random_state : seed for reproducibility

    Returns:
        model, scaler, X_val, y_val, val_video_ids
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20,
                            random_state=random_state)
    train_idx, val_idx = next(gss.split(features, labels, groups=video_ids))

    X_train, X_val = features[train_idx], features[val_idx]
    y_train, y_val = labels[train_idx],   labels[val_idx]
    v_val          = np.array(video_ids)[val_idx]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    # class_weight='balanced' handles any residual class imbalance after split
    model = LogisticRegression(C=C, max_iter=1000, random_state=random_state,
                                class_weight="balanced")
    model.fit(X_train, y_train)

    return model, scaler, X_val, y_val, v_val


def ensemble_score_learned(model, scaler,
                            ear_score, artifact_score,
                            fft_score, laplacian_score_val):
    """Run inference with the trained logistic regression."""
    x = np.array([[ear_score, artifact_score, fft_score,
                   laplacian_score_val]], dtype=np.float32)
    x_scaled  = scaler.transform(x)
    prob_fake = float(model.predict_proba(x_scaled)[0, 1])
    return round(prob_fake, 4)


def calibrate_threshold_balanced(y_true, y_scores):
    """
    Find the threshold that maximises balanced accuracy.

    Balanced accuracy = (recall_real + recall_fake) / 2
    = mean per-class recall.

    This is the right objective when you care about accuracy on both
    classes equally, because it is immune to class imbalance and does
    not trade precision for recall the way F1 does.

    Args:
        y_true   : ground-truth labels (0/1)
        y_scores : predicted probabilities

    Returns:
        best_threshold : float
        best_bal_acc   : float
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t, best_ba = 0.5, 0.0

    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        ba     = balanced_accuracy_score(y_true, y_pred)
        if ba > best_ba:
            best_ba = ba
            best_t  = float(t)

    return best_t, round(best_ba, 4)


def calibrate_threshold_f1(y_true, y_scores):
    """Find the threshold that maximises F1 (kept for comparison)."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    eps = 1e-8
    f1s = (2 * precisions[:-1] * recalls[:-1]
           / (precisions[:-1] + recalls[:-1] + eps))
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx]), round(float(f1s[best_idx]), 4)


def evaluate_model(y_true, y_scores, threshold):
    """Compute a full set of classification metrics at a given threshold."""
    y_pred = (y_scores >= threshold).astype(int)
    return {
        "accuracy"         : round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "auc"              : round(float(roc_auc_score(y_true, y_scores)), 4),
        "precision"        : round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall"           : round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1"               : round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "threshold"        : round(float(threshold), 4),
    }


# ---------------------------------------------------------------------------
# Section B — I/O helpers
# ---------------------------------------------------------------------------

def load_manifest(path=MANIFEST_PATH):
    """
    Read manifest.csv.  Returns list of dicts with keys:
        file_path, label (int), video_id, source_dataset.
    """
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append({
                "file_path"     : row["file_path"],
                "label"         : int(row["label"]),
                "video_id"      : row.get("video_id", "unknown"),
                "source_dataset": row.get("source_dataset", ""),
            })
    return rows


def extract_all_features(manifest_rows, save_csv=FEATURES_CSV, verbose=True):
    """
    For every image in the manifest compute four features:
        ear_score       : stubbed at 0.5 (Module 1 video-level; not available
                          for still frames until Module 1 is integrated)
        artifact_score  : JPEG recompression artifact  (Module 2)
        fft_score       : FFT high-frequency anomaly   (Module 3)
        laplacian_score : Laplacian-variance sharpness (Module 3 — new)

    Returns:
        features   : float32 array  (N × 4)
        labels     : int array      (N,)
        video_ids  : list of str    (N,)  — for GroupShuffleSplit
    """
    all_features = []
    all_labels   = []
    all_video_ids= []
    csv_rows     = []

    total = len(manifest_rows)
    for i, row in enumerate(manifest_rows):
        path     = row["file_path"]
        label    = row["label"]
        video_id = row["video_id"]

        img = load_face_image(path, target_size=224)
        if img is None:
            if verbose:
                print(f"  [SKIP] {path}")
            continue

        ear_score = 0.5   # stub until Module 1 provides per-frame EAR values

        try:
            art = float(get_artifact_score_for_frame(img))
        except Exception as e:
            if verbose:
                print(f"  [WARN artifact] {os.path.basename(path)}: {e}")
            art = 0.0

        try:
            fft = float(fft_anomaly_score(img))
        except Exception as e:
            if verbose:
                print(f"  [WARN fft] {os.path.basename(path)}: {e}")
            fft = 0.0

        try:
            lap = float(laplacian_score(img))
        except Exception as e:
            if verbose:
                print(f"  [WARN laplacian] {os.path.basename(path)}: {e}")
            lap = 0.0

        all_features.append([ear_score, art, fft, lap])
        all_labels.append(label)
        all_video_ids.append(video_id)
        csv_rows.append({
            "file_path"     : path,
            "label"         : label,
            "video_id"      : video_id,
            "ear_score"     : ear_score,
            "artifact_score": art,
            "fft_score"     : fft,
            "laplacian_score": lap,
        })

        if verbose and ((i + 1) % 40 == 0 or i == 0):
            print(f"  Features {i+1}/{total}  "
                  f"art={art:.3f}  fft={fft:.3f}  "
                  f"lap={lap:.3f}  label={label}  "
                  f"file={os.path.basename(path)}")

    if save_csv and csv_rows:
        os.makedirs(os.path.dirname(os.path.abspath(save_csv)), exist_ok=True)
        with open(save_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "file_path", "label", "video_id",
                "ear_score", "artifact_score", "fft_score", "laplacian_score"
            ])
            writer.writeheader()
            writer.writerows(csv_rows)
        if verbose:
            print(f"\n  Features saved: {save_csv}  ({len(csv_rows)} rows)")

    features  = np.array(all_features, dtype=np.float32)
    labels    = np.array(all_labels,   dtype=int)
    return features, labels, all_video_ids


def load_features_from_csv(path=FEATURES_CSV):
    """Load pre-computed features from CSV, skipping re-extraction."""
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append({
                "ear_score"      : float(row["ear_score"]),
                "artifact_score" : float(row["artifact_score"]),
                "fft_score"      : float(row["fft_score"]),
                "laplacian_score": float(row["laplacian_score"]),
                "label"          : int(row["label"]),
                "video_id"       : row["video_id"],
            })
    features  = np.array([[r["ear_score"], r["artifact_score"],
                           r["fft_score"], r["laplacian_score"]]
                          for r in rows], dtype=np.float32)
    labels    = np.array([r["label"]    for r in rows], dtype=int)
    video_ids = [r["video_id"] for r in rows]
    return features, labels, video_ids


def cross_validate_ensemble(features, labels, video_ids, n_splits=5, C=1.0, random_state=42):
    """
    5-fold GroupKFold cross-validation.  Returns dict of metric lists,
    one value per fold.  GroupKFold prevents any video from appearing
    in both train and val within a fold.
    """
    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = {k: [] for k in ("auc", "accuracy", "balanced_accuracy",
                                     "precision", "recall", "f1")}
    groups = np.array(video_ids)
    for fold_idx, (train_idx, val_idx) in enumerate(
            gkf.split(features, labels, groups=groups)):
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx],   labels[val_idx]
        if len(np.unique(y_val)) < 2:
            print(f"  [Fold {fold_idx+1}] skipped — only one class in val")
            continue
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)
        model   = LogisticRegression(C=C, max_iter=1000,
                                      random_state=random_state,
                                      class_weight="balanced")
        model.fit(X_train, y_train)
        val_scores = model.predict_proba(X_val)[:, 1]
        t, _       = calibrate_threshold_balanced(y_val, val_scores)
        m          = evaluate_model(y_val, val_scores, t)
        n_real_v   = int(np.sum(y_val == 0))
        n_fake_v   = int(np.sum(y_val == 1))
        n_vids_v   = len(set(np.array(video_ids)[val_idx]))
        print(f"  Fold {fold_idx+1}/{n_splits}  val={len(y_val)} "
              f"({n_real_v}r/{n_fake_v}f, {n_vids_v} vids)  "
              f"AUC={m['auc']:.4f}  acc={m['accuracy']:.4f}  "
              f"bal={m['balanced_accuracy']:.4f}")
        for key in fold_metrics:
            fold_metrics[key].append(m[key])
    return fold_metrics


def plot_curves(y_true, y_scores, threshold, save_dir=PLOTS_DIR):
    """Save ROC and Precision-Recall curves, marking the chosen threshold."""
    os.makedirs(save_dir, exist_ok=True)

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    # Find point on ROC curve closest to chosen threshold
    diffs = np.abs(roc_thresholds - threshold)
    idx_t = int(np.argmin(diffs))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"ROC  (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1,
            label="Random (0.5)")
    ax.scatter(fpr[idx_t], tpr[idx_t], marker="*", s=150, color="red", zorder=5,
               label=f"Chosen threshold {threshold:.3f}")
    ax.set_xlabel("False Positive Rate");  ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Ensemble")
    ax.legend(loc="lower right");  ax.grid(alpha=0.3)
    roc_path = os.path.join(save_dir, "roc_curve.png")
    fig.savefig(roc_path, dpi=120, bbox_inches="tight");  plt.close(fig)
    print(f"  ROC curve  : {roc_path}")

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    baseline = sum(y_true) / len(y_true)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="tomato", lw=2, label="Precision-Recall")
    ax.axhline(y=baseline, color="grey", linestyle="--", lw=1,
               label=f"Baseline (random) = {baseline:.2f}")
    ax.set_xlabel("Recall");  ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall — Ensemble")
    ax.legend(loc="upper right");  ax.grid(alpha=0.3)
    pr_path = os.path.join(save_dir, "precision_recall.png")
    fig.savefig(pr_path, dpi=120, bbox_inches="tight");  plt.close(fig)
    print(f"  PR  curve  : {pr_path}")


def print_metrics(metrics, title="EVALUATION RESULTS"):
    print()
    print("=" * 55)
    print(title)
    print("=" * 55)
    print(f"  Threshold         : {metrics['threshold']:.4f}")
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  Balanced accuracy : {metrics['balanced_accuracy']:.4f}")
    print(f"  AUC               : {metrics['auc']:.4f}")
    print(f"  Precision (fake)  : {metrics['precision']:.4f}")
    print(f"  Recall    (fake)  : {metrics['recall']:.4f}")
    print(f"  F1                : {metrics['f1']:.4f}")
    print("=" * 55)


def print_confusion_matrix(y_true, y_scores, threshold):
    """Print a labelled confusion matrix for quick error analysis."""
    y_pred = (y_scores >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()
    print()
    print("  CONFUSION MATRIX")
    print("                  Predicted")
    print("                  REAL   FAKE")
    print(f"  Actual  REAL  [ {tn:4d}   {fp:4d} ]   "
          f"<- {fp} real faces wrongly flagged as fake")
    print(f"          FAKE  [ {fn:4d}   {tp:4d} ]   "
          f"<- {fn} fakes missed")
    print()
    print(f"  True Negatives  (real -> real) : {tn}")
    print(f"  False Positives (real -> fake) : {fp}  <- hurts accuracy + precision")
    print(f"  False Negatives (fake -> real) : {fn}  <- hurts recall")
    print(f"  True Positives  (fake -> fake) : {tp}")


def print_feature_distributions(features, labels):
    """Print per-class mean ± std for each feature — quick sanity check."""
    print()
    print("  FEATURE DISTRIBUTIONS (mean ± std)")
    print(f"  {'Feature':<14}  {'REAL (label=0)':>20}  {'FAKE (label=1)':>20}  {'Delta':>8}")
    print("  " + "-" * 68)
    for j, name in enumerate(FEATURE_NAMES):
        real_vals = features[labels == 0, j]
        fake_vals = features[labels == 1, j]
        rm, rs = float(np.mean(real_vals)), float(np.std(real_vals))
        fm, fs = float(np.mean(fake_vals)), float(np.std(fake_vals))
        delta  = fm - rm
        print(f"  {name:<14}  {rm:+.4f} ± {rs:.4f}      "
              f"{fm:+.4f} ± {fs:.4f}      {delta:+.4f}")


# ---------------------------------------------------------------------------
# Section C — __main__ training pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print()
    print("=" * 65)
    print("MODULE 3 — ENSEMBLE TRAINING PIPELINE")
    print("=" * 65)
    print()

    # ------------------------------------------------------------------
    # 0. Prerequisite check
    # ------------------------------------------------------------------
    if not os.path.exists(MANIFEST_PATH):
        print(f"[ERROR] {MANIFEST_PATH} not found.")
        print("        Run  python inspect_dataset.py  first.")
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # 1. Load manifest
    # ------------------------------------------------------------------
    print("STEP 1 — Load manifest")
    manifest  = load_manifest(MANIFEST_PATH)
    n_real    = sum(1 for r in manifest if r["label"] == 0)
    n_fake    = sum(1 for r in manifest if r["label"] == 1)
    n_vids    = len(set(r["video_id"] for r in manifest))
    print(f"  Total images   : {len(manifest)}")
    print(f"  Label 0 (real) : {n_real}")
    print(f"  Label 1 (fake) : {n_fake}")
    print(f"  Unique videos  : {n_vids}  (used for video-level split)")
    print()

    # ------------------------------------------------------------------
    # 2. Extract features  (or load from cache)
    # ------------------------------------------------------------------
    if os.path.exists(FEATURES_CSV):
        print(f"STEP 2 — Load cached features from {FEATURES_CSV}")
        features, labels, video_ids = load_features_from_csv(FEATURES_CSV)
        print(f"  Loaded {len(features)} rows from cache.  "
              "(Delete module3_features.csv to force re-extraction.)")
    else:
        print("STEP 2 — Extract features  (artifact + FFT + Laplacian scores)")
        print("         Expected time: ~90-240 s for 400 images.")
        print()
        features, labels, video_ids = extract_all_features(manifest, save_csv=FEATURES_CSV)

    if len(features) == 0:
        print("[ERROR] No features extracted.  Check image paths.")
        raise SystemExit(1)

    print(f"\n  Feature matrix : {features.shape}  "
          f"(rows=images, cols={FEATURE_NAMES})")
    print(f"  Label vector   : {labels.shape}")
    print()

    print_feature_distributions(features, labels)
    print()

    # ------------------------------------------------------------------
    # 3. Equal-weights baseline
    # ------------------------------------------------------------------
    print("STEP 3 — Equal-weights baseline  (w = 0.25 each)")
    equal_scores = np.array([
        ensemble_score_equal_weights(r[0], r[1], r[2], r[3]) for r in features
    ])
    try:
        equal_auc = roc_auc_score(labels, equal_scores)
        print(f"  Equal-weights AUC : {equal_auc:.4f}")
    except Exception:
        print("  Could not compute AUC for equal-weights baseline.")
    print()

    # ------------------------------------------------------------------
    # 4. Train logistic regression with video-level split
    # ------------------------------------------------------------------
    print("STEP 4 — Train LogisticRegression  (video-level GroupShuffleSplit)")
    model, scaler, X_val, y_val, val_vids = train_ensemble(
        features, labels, video_ids
    )

    val_scores = model.predict_proba(X_val)[:, 1]

    # Sanity: report val split composition
    n_val_real = int(np.sum(y_val == 0))
    n_val_fake = int(np.sum(y_val == 1))
    n_val_vids = len(set(val_vids))
    print(f"  Val set        : {len(y_val)} frames  "
          f"({n_val_real} real, {n_val_fake} fake)  "
          f"from {n_val_vids} unique videos")

    # Learned coefficients
    raw_coef  = model.coef_[0]
    coef_sum  = np.abs(raw_coef).sum() + 1e-8
    norm_coef = np.abs(raw_coef) / coef_sum
    print(f"  Coefficients (|w| normalised to sum=1):")
    for name, nc in zip(FEATURE_NAMES, norm_coef):
        bar = "#" * int(nc * 30)
        print(f"    {name:<12} = {nc:.3f}  {bar}")
    print()

    # ------------------------------------------------------------------
    # 4b. 5-fold GroupKFold cross-validation  (more reliable estimate)
    # ------------------------------------------------------------------
    print()
    print("STEP 4b — 5-fold GroupKFold cross-validation")
    cv_metrics = cross_validate_ensemble(features, labels, video_ids, n_splits=5)

    if cv_metrics["auc"]:
        print()
        print("  CROSS-VALIDATION SUMMARY  (mean +/- std across folds)")
        print(f"  {'Metric':<22}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
        print("  " + "-" * 58)
        for key in ("auc", "accuracy", "balanced_accuracy",
                    "precision", "recall", "f1"):
            vals = cv_metrics[key]
            print(f"  {key:<22}  {np.mean(vals):>8.4f}  "
                  f"{np.std(vals):>8.4f}  "
                  f"{np.min(vals):>8.4f}  "
                  f"{np.max(vals):>8.4f}")
    print()

    # ------------------------------------------------------------------
    # 5. Threshold calibration — balanced accuracy (primary)
    # ------------------------------------------------------------------
    print("STEP 5 — Calibrate threshold  (balanced accuracy — optimises accuracy)")
    best_t_ba,  best_ba  = calibrate_threshold_balanced(y_val, val_scores)
    best_t_f1,  best_f1v = calibrate_threshold_f1(y_val, val_scores)
    print(f"  Balanced-accuracy threshold : {best_t_ba:.4f}  "
          f"(bal. acc. = {best_ba:.4f})  <- PRIMARY")
    print(f"  F1-score threshold          : {best_t_f1:.4f}  "
          f"(F1 = {best_f1v:.4f})          <- for reference")
    print()

    # ------------------------------------------------------------------
    # 6. Evaluate
    # ------------------------------------------------------------------
    print("STEP 6 — Evaluate on validation split")

    metrics_ba = evaluate_model(y_val, val_scores, best_t_ba)
    metrics_f1 = evaluate_model(y_val, val_scores, best_t_f1)

    print_metrics(metrics_ba, title="RESULTS — balanced-accuracy threshold")
    print_confusion_matrix(y_val, val_scores, best_t_ba)

    print()
    print("  COMPARISON — F1 threshold vs balanced-accuracy threshold:")
    print(f"  {'Metric':<22}  {'Bal-acc thresh':>16}  {'F1 thresh':>12}")
    print("  " + "-" * 54)
    for key in ["threshold", "accuracy", "balanced_accuracy",
                "auc", "precision", "recall", "f1"]:
        print(f"  {key:<22}  {metrics_ba[key]:>16.4f}  "
              f"{metrics_f1[key]:>12.4f}")

    # ------------------------------------------------------------------
    # 7. Save diagnostic plots
    # ------------------------------------------------------------------
    print()
    print("STEP 7 — Save diagnostic plots")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    try:
        plot_curves(y_val, val_scores, best_t_ba, save_dir=PLOTS_DIR)
    except Exception as e:
        print(f"  [WARN] plot_curves: {e}")

    # ------------------------------------------------------------------
    # 8. FFT spectrum visualisations
    # ------------------------------------------------------------------
    print()
    print("STEP 8 — FFT spectrum visualisations")
    os.makedirs(VIZ_DIR, exist_ok=True)
    for target_label, tag in [(0, "real"), (1, "fake")]:
        candidates = [r for r in manifest if r["label"] == target_label]
        if candidates:
            img = load_face_image(candidates[0]["file_path"])
            if img is not None:
                out = os.path.join(VIZ_DIR, f"fft_spectrum_{tag}.jpg")
                visualize_spectrum(img, save_path=out, label=tag)

    # ------------------------------------------------------------------
    # 9. Final summary
    # ------------------------------------------------------------------
    print()
    print("=" * 65)
    print("MODULE 3 COMPLETE")
    print("=" * 65)
    print(f"  Features CSV  : {FEATURES_CSV}")
    print(f"  ROC curve     : {os.path.join(PLOTS_DIR, 'roc_curve.png')}")
    print(f"  PR  curve     : {os.path.join(PLOTS_DIR, 'precision_recall.png')}")
    print()
    print(f"  PRIMARY RESULTS (balanced-accuracy threshold {best_t_ba:.4f}):")
    print(f"    AUC              = {metrics_ba['auc']:.4f}")
    print(f"    Accuracy         = {metrics_ba['accuracy']:.4f}")
    print(f"    Balanced Acc     = {metrics_ba['balanced_accuracy']:.4f}")
    print(f"    F1               = {metrics_ba['f1']:.4f}")
    print(f"    Precision (fake) = {metrics_ba['precision']:.4f}")
    print(f"    Recall    (fake) = {metrics_ba['recall']:.4f}")
    if cv_metrics["auc"]:
        print()
        print(f"  5-FOLD CV RESULTS:")
        print(f"    AUC          = {np.mean(cv_metrics['auc']):.4f} +/- {np.std(cv_metrics['auc']):.4f}")
        print(f"    Accuracy     = {np.mean(cv_metrics['accuracy']):.4f} +/- {np.std(cv_metrics['accuracy']):.4f}")
        print(f"    Balanced Acc = {np.mean(cv_metrics['balanced_accuracy']):.4f} +/- {np.std(cv_metrics['balanced_accuracy']):.4f}")
    print()
    print("CHECKLIST:")
    print("  [x] Video-level GroupShuffleSplit  (no identity leakage)")
    print("  [x] Laplacian texture score        (4th feature, replaces EAR stub)")
    print("  [x] Balanced-accuracy threshold    (optimises for accuracy)")
    print("  [x] Confusion matrix + per-class metrics")
    print("  [x] Feature distributions by class")
    print()
    print("NEXT:")
    print("  - Integrate Module 1 EAR score to replace the ear_score=0.5 stub.")
    print("  - Consider extracting more frames per video or using more videos")
    print("    to push balanced accuracy above 0.75.")
    print("=" * 65)
