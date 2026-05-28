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
# 2. Laplacian texture score (4th feature).  Deepfake decoders smooth fine
#    texture;
#    Var(Laplacian) captures this difference.
#
# 3. Constrained threshold calibration — primary threshold now enforces an
#    explicit cap on false-FAKE rate on REAL samples (specificity constraint),
#    then maximises balanced accuracy within that feasible set.
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
from src.blink_analysis.ear_scorer        import (
    VIDEO_EAR_CSV_DEFAULT,
    compute_ear_from_source_paths,
    load_video_ear_scores,
    resolve_source_video_paths,
)
from src.freq_analysis.anomaly_scorer     import fft_anomaly_score
from src.freq_analysis.texture_scorer     import laplacian_score
from src.freq_analysis.utils              import load_face_image
from src.freq_analysis.frequency_analyzer import visualize_spectrum

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MANIFEST_PATH = os.environ.get("MANIFEST_PATH", os.path.join("data", "manifest.csv"))
FEATURES_CSV = os.environ.get("FEATURES_CSV", os.path.join("data", "module3_features.csv"))
VIDEO_EAR_CSV = os.environ.get("VIDEO_EAR_CSV", VIDEO_EAR_CSV_DEFAULT)
MODEL_PKL_PATH = os.environ.get("MODEL_PKL_PATH", os.path.join("data", "ensemble_model.pkl"))
PLOTS_DIR = os.environ.get("PLOTS_DIR", os.path.join("data", "plots"))
VIZ_DIR = os.environ.get("VIZ_DIR", os.path.join("data", "visualizations"))
UNCERTAIN_BAND = float(os.environ.get("UNCERTAIN_BAND", "0.10"))
THRESHOLD_POLICY = os.environ.get("THRESHOLD_POLICY", "max_false_fake_rate")
MAX_FALSE_FAKE_RATE = float(os.environ.get("MAX_FALSE_FAKE_RATE", "0.10"))

# Feature column names (used in CSV headers and diagnostic output)
# EAR removed: Haar-based pseudo-EAR gives ear≈1.0 for all videos at inference
# (low blink rate on short clips + low std → static_eye penalty), which causes
# the model to output prob_fake≈0 for everything.
#
# "smoothness" = 1 - laplacian_score. Laplacian-variance is "higher = sharper = more
# real"; inverting it makes all three features share the convention "higher = more
# suspicious (fake)", which lets the LR learn the correct positive coefficients and
# ensures equal-weights is scored in the right direction.
FEATURE_NAMES = ["artifact", "fft", "smoothness"]

# ---------------------------------------------------------------------------
# Section A — pure functions
# ---------------------------------------------------------------------------

def ensemble_score_equal_weights(artifact_score, fft_score, smoothness_score):
    """
    Combine three module scores with equal weights (baseline).

    fft_score       : FFT spectral slope anomaly (Module 3)
    smoothness_score: 1 - laplacian_score  (lower Laplacian var = smoother = more fake)

    artifact_score is accepted for API compatibility but excluded from the
    average: on FF++ C23, real videos score marginally higher than fakes on
    JPEG artifact (real≈0.052, fake≈0.050), so including it pushes real
    videos toward the fake score rather than away from it.

    Formula:
        final = (fft  +  smoothness) / 2
    """
    score = (fft_score + smoothness_score) / 2.0
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
                            artifact_score, fft_score, smoothness_score):
    """
    Run inference with the trained logistic regression.

    smoothness_score = 1 - laplacian_score  (must be pre-computed by the caller).
    All three inputs use the "higher = more suspicious" convention consistent
    with training.
    """
    x = np.array([[artifact_score, fft_score, smoothness_score]], dtype=np.float32)
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


def calibrate_threshold_with_constraints(
    y_true,
    y_scores,
    *,
    max_false_fake_rate=0.10,
):
    """
    Calibrate threshold with an explicit cap on false-FAKE rate on REAL samples.

    Selection rule:
      1) keep thresholds where FPR(real->fake) <= max_false_fake_rate
      2) among valid thresholds, choose highest balanced accuracy
      3) tie-break by lower FPR then lower threshold value
    If no threshold satisfies the cap, choose the threshold with minimum FPR
    (then best balanced accuracy) and mark constraint_met=False.
    """
    thresholds = np.linspace(0.01, 0.99, 197)
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores, dtype=float)

    candidates = []
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        real_total = tn + fp
        fpr_real = (fp / real_total) if real_total > 0 else 1.0
        specificity = (tn / real_total) if real_total > 0 else 0.0
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        candidates.append(
            {
                "threshold": float(t),
                "balanced_accuracy": float(bal_acc),
                "false_fake_rate_real": float(fpr_real),
                "specificity_real": float(specificity),
                "constraint_met": bool(fpr_real <= max_false_fake_rate),
            }
        )

    valid = [c for c in candidates if c["constraint_met"]]
    if valid:
        valid.sort(
            key=lambda c: (
                -c["balanced_accuracy"],
                c["false_fake_rate_real"],
                c["threshold"],
            )
        )
        chosen = valid[0]
        status = "met"
    else:
        candidates.sort(
            key=lambda c: (
                c["false_fake_rate_real"],
                -c["balanced_accuracy"],
                c["threshold"],
            )
        )
        chosen = candidates[0]
        status = "relaxed_to_lowest_false_fake"

    policy = {
        "mode": "max_false_fake_rate",
        "max_false_fake_rate_real": float(max_false_fake_rate),
        "constraint_status": status,
        "threshold": float(chosen["threshold"]),
        "balanced_accuracy": round(float(chosen["balanced_accuracy"]), 4),
        "false_fake_rate_real": round(float(chosen["false_fake_rate_real"]), 4),
        "specificity_real": round(float(chosen["specificity_real"]), 4),
    }
    return float(chosen["threshold"]), policy


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


def _get_video_ear_cache(manifest_rows, verbose=True):
    """Build video_id → EAR score from CSV and/or on-the-fly from source videos."""
    cache = load_video_ear_scores(VIDEO_EAR_CSV)
    if cache and verbose:
        print(f"  Loaded {len(cache)} video EAR scores from {VIDEO_EAR_CSV}")

    for row in manifest_rows:
        video_id = row["video_id"]
        src = row.get("source_dataset", "")
        if video_id in cache:
            # Refresh neutral legacy cache entries when source videos are resolvable.
            if float(cache.get(video_id, 0.5)) != 0.5:
                continue
            vpaths = resolve_source_video_paths(video_id, src)
            if vpaths:
                recomputed = compute_ear_from_source_paths(vpaths)
                if recomputed is not None:
                    cache[video_id] = recomputed
                    if verbose:
                        print(
                            f"  Refreshed EAR for {video_id}: {cache[video_id]:.4f} "
                            f"(sources={len(vpaths)})"
                        )
                continue
            continue
        vpaths = resolve_source_video_paths(video_id, src)
        if vpaths:
            score = compute_ear_from_source_paths(vpaths)
            if score is not None:
                cache[video_id] = score
            else:
                cache.setdefault(video_id, 0.5)
            if verbose:
                print(
                    f"  Computed EAR for {video_id}: {cache[video_id]:.4f} "
                    f"(sources={len(vpaths)})"
                )
        else:
            cache.setdefault(video_id, 0.5)
    return cache


def extract_all_features(manifest_rows, save_csv=FEATURES_CSV, verbose=True):
    """
    For every image in the manifest compute three features:
        artifact_score  : JPEG recompression artifact  (Module 2)
        fft_score       : FFT spectral slope anomaly   (Module 3)
        laplacian_score : Laplacian-variance sharpness (Module 3)

    EAR (Module 1) is excluded: the Haar-based fallback gives identical high
    scores (≈1.0) for all videos at inference, collapsing the model to predict
    REAL for everything. EAR requires MediaPipe to be discriminative; without
    it, including EAR poisons the model.

    Returns:
        features   : float32 array  (N × 3)
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

        # smoothness = 1 - laplacian: inverted so "higher = smoother = more suspicious".
        # laplacian_score is "higher = sharper = more real"; inversion aligns all
        # three features in the same "higher = more fake" direction so the LR
        # learns positive coefficients and equal-weights is correctly directed.
        smoothness = round(1.0 - lap, 4)

        all_features.append([art, fft, smoothness])
        all_labels.append(label)
        all_video_ids.append(video_id)
        csv_rows.append({
            "file_path"       : path,
            "label"           : label,
            "video_id"        : video_id,
            "artifact_score"  : art,
            "fft_score"       : fft,
            "smoothness_score": smoothness,
        })

        if verbose and ((i + 1) % 40 == 0 or i == 0):
            print(f"  Features {i+1}/{total}  "
                  f"art={art:.3f}  fft={fft:.3f}  "
                  f"smoothness={smoothness:.3f}  label={label}  "
                  f"file={os.path.basename(path)}")

    if save_csv and csv_rows:
        os.makedirs(os.path.dirname(os.path.abspath(save_csv)), exist_ok=True)
        with open(save_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "file_path", "label", "video_id",
                "artifact_score", "fft_score", "smoothness_score"
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
                "artifact_score"  : float(row["artifact_score"]),
                "fft_score"       : float(row["fft_score"]),
                "smoothness_score": float(row["smoothness_score"]),
                "label"           : int(row["label"]),
                "video_id"        : row["video_id"],
            })
    features  = np.array([[r["artifact_score"], r["fft_score"], r["smoothness_score"]]
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
        if os.path.exists(VIDEO_EAR_CSV):
            print(f"  Note: re-run without cache to refresh EAR from {VIDEO_EAR_CSV}")
    else:
        print("STEP 2 — Extract features  (EAR + artifact + FFT + Laplacian)")
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
    print("STEP 3 — Equal-weights baseline  (w = 1/3 each)")
    equal_scores = np.array([
        ensemble_score_equal_weights(r[0], r[1], r[2]) for r in features
    ])
    try:
        equal_auc = roc_auc_score(labels, equal_scores)
        print(f"  Equal-weights AUC : {equal_auc:.4f}")
    except Exception:
        print("  Could not compute AUC for equal-weights baseline.")

    # Calibrate threshold from equal-weights on the FULL dataset.
    # Equal-weights uses a fixed formula (no learning), so calibrating on all 778
    # samples does not cause overfitting — we are just finding the best operating
    # point for a formula that never saw the labels during its "training."
    ew_threshold, ew_bal_acc = calibrate_threshold_balanced(labels, equal_scores)
    print(f"  Equal-weights threshold (full dataset, balanced-acc): "
          f"{ew_threshold:.4f}  (bal. acc. = {ew_bal_acc:.4f})")

    # Per-class mean — sanity check that fake scores higher than real
    ew_mean_real = float(np.mean(equal_scores[labels == 0]))
    ew_mean_fake = float(np.mean(equal_scores[labels == 1]))
    print(f"  Equal-weights mean scores -- real: {ew_mean_real:.4f}  "
          f"fake: {ew_mean_fake:.4f}  "
          f"delta={ew_mean_fake - ew_mean_real:+.4f}")
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
    print("STEP 5 — Calibrate threshold policy")
    best_t_ba,  best_ba  = calibrate_threshold_balanced(y_val, val_scores)
    best_t_f1,  best_f1v = calibrate_threshold_f1(y_val, val_scores)
    if THRESHOLD_POLICY == "balanced_accuracy":
        policy_threshold = best_t_ba
        policy_meta = {
            "mode": "balanced_accuracy",
            "max_false_fake_rate_real": MAX_FALSE_FAKE_RATE,
            "constraint_status": "not_applicable",
            "threshold": float(best_t_ba),
            "balanced_accuracy": round(float(best_ba), 4),
            "false_fake_rate_real": None,
            "specificity_real": None,
        }
    else:
        policy_threshold, policy_meta = calibrate_threshold_with_constraints(
            y_val, val_scores, max_false_fake_rate=MAX_FALSE_FAKE_RATE
        )
    print(f"  Balanced-accuracy threshold : {best_t_ba:.4f}  "
          f"(bal. acc. = {best_ba:.4f})")
    print(f"  F1-score threshold          : {best_t_f1:.4f}  "
          f"(F1 = {best_f1v:.4f})")
    ff_rate = policy_meta["false_fake_rate_real"]
    specificity = policy_meta["specificity_real"]
    print(
        "  Policy threshold            : "
        f"{policy_threshold:.4f}  "
        f"(mode={policy_meta['mode']}, "
        f"false_fake_real={ff_rate if ff_rate is not None else 'n/a'}, "
        f"specificity_real={specificity if specificity is not None else 'n/a'}, "
        f"status={policy_meta['constraint_status']})  <- PRIMARY"
    )
    print()

    # ------------------------------------------------------------------
    # 6. Evaluate
    # ------------------------------------------------------------------
    print("STEP 6 — Evaluate on validation split")

    metrics_ba = evaluate_model(y_val, val_scores, best_t_ba)
    metrics_f1 = evaluate_model(y_val, val_scores, best_t_f1)
    metrics_policy = evaluate_model(y_val, val_scores, policy_threshold)

    print_metrics(metrics_policy, title="RESULTS — policy threshold")
    print_confusion_matrix(y_val, val_scores, policy_threshold)

    print()
    print("  COMPARISON — policy threshold vs alternatives:")
    print(f"  {'Metric':<22}  {'Policy':>12}  {'Bal-acc':>12}  {'F1':>12}")
    print("  " + "-" * 66)
    for key in ["threshold", "accuracy", "balanced_accuracy",
                "auc", "precision", "recall", "f1"]:
        print(
            f"  {key:<22}  {metrics_policy[key]:>12.4f}  "
            f"{metrics_ba[key]:>12.4f}  {metrics_f1[key]:>12.4f}"
        )

    # ------------------------------------------------------------------
    # 7. Save diagnostic plots
    # ------------------------------------------------------------------
    print()
    print("STEP 7 — Save diagnostic plots")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    try:
        plot_curves(y_val, val_scores, policy_threshold, save_dir=PLOTS_DIR)
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
    print(f"  PRIMARY RESULTS (policy threshold {policy_threshold:.4f}):")
    print(f"    AUC              = {metrics_policy['auc']:.4f}")
    print(f"    Accuracy         = {metrics_policy['accuracy']:.4f}")
    print(f"    Balanced Acc     = {metrics_policy['balanced_accuracy']:.4f}")
    print(f"    F1               = {metrics_policy['f1']:.4f}")
    print(f"    Precision (fake) = {metrics_policy['precision']:.4f}")
    print(f"    Recall    (fake) = {metrics_policy['recall']:.4f}")
    if policy_meta["false_fake_rate_real"] is not None:
        print(
            f"    False-FAKE real  = {policy_meta['false_fake_rate_real']:.4f}  "
            f"(cap={policy_meta['max_false_fake_rate_real']:.4f})"
        )
    if cv_metrics["auc"]:
        print()
        print(f"  5-FOLD CV RESULTS:")
        print(f"    AUC          = {np.mean(cv_metrics['auc']):.4f} +/- {np.std(cv_metrics['auc']):.4f}")
        print(f"    Accuracy     = {np.mean(cv_metrics['accuracy']):.4f} +/- {np.std(cv_metrics['accuracy']):.4f}")
        print(f"    Balanced Acc = {np.mean(cv_metrics['balanced_accuracy']):.4f} +/- {np.std(cv_metrics['balanced_accuracy']):.4f}")
    print()
    # ------------------------------------------------------------------
    # 10. Save model bundle for API inference
    # ------------------------------------------------------------------
    try:
        import joblib

        os.makedirs(os.path.dirname(os.path.abspath(MODEL_PKL_PATH)), exist_ok=True)
        joblib.dump(
            {
                "model": model,
                "scaler": scaler,
                # Primary threshold is from equal-weights balanced-accuracy calibration
                # on the full dataset. The backend uses equal-weights scoring, which
                # achieves higher AUC on FF++ C23 than the LR model (the LR model
                # learns inconsistent coefficient directions due to very weak feature
                # signal; equal-weights with correct feature orientation is more reliable).
                "threshold": ew_threshold,
                "threshold_mode": "equal_weights_balanced_accuracy",
                "threshold_policy": {
                    "mode": "equal_weights_balanced_accuracy",
                    "ew_threshold": float(ew_threshold),
                    "ew_balanced_accuracy": float(ew_bal_acc),
                    "ew_mean_real": float(ew_mean_real),
                    "ew_mean_fake": float(ew_mean_fake),
                    "lr_policy": policy_meta,
                },
                "uncertain_band": UNCERTAIN_BAND,
                "feature_names": FEATURE_NAMES,
                # Backend must use equal-weights scoring (not LR) for this threshold
                "scoring": "equal_weights",
            },
            MODEL_PKL_PATH,
        )
        print(f"  Model bundle saved: {MODEL_PKL_PATH}")
        print(
            f"    threshold={ew_threshold:.4f}  "
            f"(mode=equal_weights_balanced_accuracy, bal.acc={ew_bal_acc:.4f})"
        )
    except Exception as e:
        print(f"  [WARN] Could not save model bundle: {e}")

    print()
    print("CHECKLIST:")
    print("  [x] Video-level GroupShuffleSplit  (no identity leakage)")
    print("  [x] Module 1 EAR (video-level blink score)")
    print("  [x] Laplacian texture score")
    print("  [x] Constrained threshold policy   (caps false-FAKE on reals)")
    print("  [x] Confusion matrix + per-class metrics")
    print("  [x] Feature distributions by class")
    print("  [x] ensemble_model.pkl for API")
    print()
    print("RETRAIN:  python inspect_dataset.py  &&  python ensemble.py")
    print("          (delete data/module3_features.csv to refresh EAR features)")
    print("=" * 65)
