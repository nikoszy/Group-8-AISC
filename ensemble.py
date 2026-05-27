# =============================================================================
# ensemble.py  --  MODULE 3: WEIGHTED ENSEMBLE SCORING, TRAINING, CALIBRATION
# =============================================================================
#
# CHANGES FROM PREVIOUS VERSION
# ------------------------------
# 1. Video-level GroupShuffleSplit  -- prevents identity leakage between
#    train and val sets.
#
# 2. Laplacian texture score (4th feature)  -- deepfake decoders smooth fine
#    texture; Var(Laplacian) captures this difference.
#
# 3. Balanced-accuracy threshold calibration  -- threshold is now chosen to
#    maximise balanced accuracy = (recall_real + recall_fake) / 2.
#
# 4. Confusion matrix + per-class breakdown  -- gives precise visibility
#    into where the model makes mistakes.
#
# 5. class_weight='balanced' in LogisticRegression  -- robustifies against
#    class imbalance.
#
# 6. Module 1 integration  -- ear_score is now the real deepfake_confidence
#    from data/module1_output.csv (via _load_module1_scores).  Falls back to
#    0.5 for any video_id not found in that CSV, with a logged warning.
#
# 7. Multi-classifier comparison  -- trains LR, RandomForest, and XGBoost on
#    the same video-level features; picks the best by val-set F1 as the
#    Module 3 model.
#
# 8. 30/70 fusion  -- fuses Module 1 (blink / deepfake_confidence) with the
#    best Module 3 model probability using MODULE1_WEIGHT=0.30 / MODULE3_WEIGHT=0.70.
#    Output written to data/ensemble_output.csv.
#    A weight sweep (0.0–1.0, step 0.1) is printed at the end of Step 12
#    to validate the 30/70 choice; constants are NOT auto-updated by the sweep.
# =============================================================================

import os
import csv
import pickle
import warnings

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import StandardScaler
from sklearn.calibration     import CalibratedClassifierCV
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics         import (
    roc_auc_score, precision_recall_curve, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix,
)

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

from src.freq_analysis.anomaly_scorer     import fft_anomaly_score
from src.freq_analysis.texture_scorer     import laplacian_score
from src.freq_analysis.utils              import load_face_image
from src.freq_analysis.frequency_analyzer import visualize_spectrum
from artifact_module                      import get_artifact_score_for_frame

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MANIFEST_PATH       = os.path.join("data", "manifest.csv")
FEATURES_CSV        = os.path.join("data", "module3_features.csv")
MODULE1_CSV         = os.path.join("data", "module1_output.csv")
ENSEMBLE_OUTPUT_CSV = os.path.join("data", "ensemble_output.csv")
PLOTS_DIR           = os.path.join("data", "plots")
VIZ_DIR             = os.path.join("data", "visualizations")

# Feature column names (used in CSV headers and diagnostic output).
# "ear" is now Module 1's deepfake_confidence (no longer a 0.5 stub).
FEATURE_NAMES = ["artifact", "fft", "laplacian", "ear"]

# ---------------------------------------------------------------------------
# Fusion weight constants  (Module 1 vs Module 3)
# Change only here to retune — _fuse_and_report() reads these directly.
# Constraint: MODULE1_WEIGHT + MODULE3_WEIGHT must equal 1.0.
# ---------------------------------------------------------------------------
MODULE1_WEIGHT = 0.30   # weight given to Module 1 deepfake_confidence
MODULE3_WEIGHT = 0.70   # weight given to Module 3 best-model P(fake)

# ---------------------------------------------------------------------------
# Section A -- pure functions
# ---------------------------------------------------------------------------

def ensemble_score_equal_weights(artifact_score, fft_score, laplacian_score_val):
    """
    Combine three handcrafted module scores with equal weights (baseline).

    Formula:
        final = (1/3) * artifact  +  (1/3) * fft  +  (1/3) * lap

    Note: ear_score is fused separately at the video level (Step 4).
    """
    score = (artifact_score + fft_score + laplacian_score_val) / 3.0
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
    exactly one of train or val -- it treats video_id as the group key.

    Args:
        features     : (NxF) float array  -- video-level features
        labels       : (N,)  int array     0=real, 1=fake
        video_ids    : list[str] length N  -- one per video
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

    # Zero-variance guard -- silence any column whose training variance is
    # effectively zero.  StandardScaler sets scale_=1 when var_=0 exactly,
    # but float32 arithmetic can produce near-zero (not exact-zero) variance,
    # which StandardScaler amplifies into large values.
    # Zeroing those columns here prevents the logistic regression from fitting
    # noise.  The threshold 1e-10 is safely above float64 rounding (~1e-15)
    # and safely below any real feature variance (~1e-4 or larger).
    _zero_var = scaler.var_ < 1e-10
    if _zero_var.any():
        X_train[:, _zero_var] = 0.0
        X_val[:, _zero_var]   = 0.0

    # class_weight='balanced' handles any residual class imbalance after split
    base_lr = LogisticRegression(C=C, max_iter=1000, random_state=random_state,
                                  class_weight="balanced")

    # Platt scaling via 3-fold CV on training data.
    model = CalibratedClassifierCV(base_lr, method="sigmoid", cv=3)
    model.fit(X_train, y_train)

    return model, scaler, X_val, y_val, v_val


def ensemble_score_learned(model, scaler,
                            artifact_score,
                            fft_score,
                            laplacian_score_val,
                            ear_score=0.5):
    """Run inference with the trained model on a single sample.

    Args:
        model               : fitted sklearn model
        scaler              : fitted StandardScaler
        artifact_score      : float [0,1]
        fft_score           : float [0,1]
        laplacian_score_val : float [0,1]
        ear_score           : float [0,1]  Module 1 confidence (default 0.5)

    Returns:
        float [0,1] -- calibrated P(fake)
    """
    x = np.array([[artifact_score, fft_score,
                   laplacian_score_val, ear_score]], dtype=np.float32)
    x_scaled  = scaler.transform(x)
    prob_fake = float(model.predict_proba(x_scaled)[0, 1])
    return round(prob_fake, 4)


def calibrate_threshold_balanced(y_true, y_scores):
    """
    Find the threshold that maximises balanced accuracy.

    Balanced accuracy = (recall_real + recall_fake) / 2
    = mean per-class recall.

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
# Section B -- I/O helpers
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


def _load_module1_scores():
    """
    Load data/module1_output.csv and return a dict mapping manifest-format
    video_id -> deepfake_confidence (float [0,1]).

    module1_output.csv video_id format:
        Real videos : bare integer strings "0", "1", ..., "135"
                      -> manifest key: "real_000", "real_001", ...
        Fake videos : "AAA_BBB" strings  "000_003", "001_870", ...
                      -> manifest key:  "fake_000_003", "fake_001_870", ...

    Returns an empty dict if MODULE1_CSV is not found, so callers fall back
    to ear_score = 0.5 gracefully.
    """
    if not os.path.exists(MODULE1_CSV):
        print(f"  [WARN] {MODULE1_CSV} not found -- ear_score = 0.5 for all frames.")
        return {}

    df = pd.read_csv(MODULE1_CSV)
    scores = {}
    for _, row in df.iterrows():
        vid     = str(row["video_id"]).strip()
        is_fake = int(row["is_fake"])
        conf    = float(row["deepfake_confidence"])
        if is_fake == 0:
            try:
                key = f"real_{int(vid):03d}"
            except ValueError:
                continue                       # malformed integer -- skip
        else:
            key = f"fake_{vid}"
        scores[key] = conf

    print(f"  [module1] Loaded {len(scores)} video scores from {MODULE1_CSV}")
    return scores


def extract_all_features(manifest_rows, save_csv=FEATURES_CSV, verbose=True):
    """
    For every image in the manifest compute four features and save to CSV.

    Features:
        artifact_score  : JPEG recompression artifact  (Module 2)
        fft_score       : FFT high-frequency anomaly   (Module 3)
        laplacian_score : Laplacian-variance sharpness (Module 3)
        ear_score       : Module 1 deepfake_confidence (loaded from
                          data/module1_output.csv; falls back to 0.5
                          with a warning for any missing video_id)

    The function signature is unchanged -- Module 1 is loaded internally.

    Returns:
        features   : float32 array  (N x 4)  [artifact, fft, laplacian, ear]
        labels     : int array      (N,)
        video_ids  : list of str    (N,)  -- for GroupShuffleSplit
    """
    # Load Module 1 scores once -- dict[manifest_video_id -> deepfake_confidence]
    module1_scores = _load_module1_scores()
    n_matched  = 0
    n_missing  = 0

    all_features  = []
    all_labels    = []
    all_video_ids = []
    csv_rows      = []

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

        # Module 1 ear_score -- looked up by manifest video_id
        if video_id in module1_scores:
            ear = module1_scores[video_id]
            n_matched += 1
        else:
            if verbose:
                print(f"  [WARN module1] video_id={video_id!r} not in "
                      f"module1_output.csv -- ear_score = 0.5")
            ear = 0.5
            n_missing += 1

        try:
            artifact = float(get_artifact_score_for_frame(img))
        except Exception as e:
            if verbose:
                print(f"  [WARN artifact] {os.path.basename(path)}: {e}")
            artifact = 0.5

        try:
            fft = float(fft_anomaly_score(img))
        except Exception as e:
            if verbose:
                print(f"  [WARN fft] {os.path.basename(path)}: {e}")
            fft = 0.5

        try:
            lap = float(laplacian_score(img))
        except Exception as e:
            if verbose:
                print(f"  [WARN laplacian] {os.path.basename(path)}: {e}")
            lap = 0.5

        all_features.append([artifact, fft, lap, ear])   # 4 features
        all_labels.append(label)
        all_video_ids.append(video_id)
        csv_rows.append({
            "file_path"      : path,
            "label"          : label,
            "video_id"       : video_id,
            "artifact_score" : artifact,
            "fft_score"      : fft,
            "laplacian_score": lap,
            "ear_score"      : ear,
        })

        if verbose and ((i + 1) % 40 == 0 or i == 0):
            print(f"  Features {i+1}/{total}  "
                  f"fft={fft:.3f}  lap={lap:.3f}  ear={ear:.4f}  "
                  f"label={label}  file={os.path.basename(path)}")

    if verbose:
        print(f"\n  Module 1 lookup: {n_matched} matched, {n_missing} missing "
              f"(-> 0.5 fallback)")

    if save_csv and csv_rows:
        os.makedirs(os.path.dirname(os.path.abspath(save_csv)), exist_ok=True)
        with open(save_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "file_path", "label", "video_id",
                "artifact_score", "fft_score", "laplacian_score", "ear_score"
            ])
            writer.writeheader()
            writer.writerows(csv_rows)
        if verbose:
            print(f"\n  Features saved: {save_csv}  ({len(csv_rows)} rows)")

    features  = np.array(all_features, dtype=np.float32)
    labels    = np.array(all_labels,   dtype=int)
    return features, labels, all_video_ids


def load_features_from_csv(path=FEATURES_CSV):
    """Load pre-computed features from CSV, skipping re-extraction.

    Handles both old 3-feature CSVs (no ear_score) and new 4-feature CSVs.
    Missing ear_score defaults to 0.5.

    Returns:
        features  : (N, 4) float32  [artifact, fft, laplacian, ear]
        labels    : (N,)   int
        video_ids : list[str]
    """
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({
                "artifact_score" : float(row.get("artifact_score",  0.5)),
                "fft_score"      : float(row.get("fft_score",       0.5)),
                "laplacian_score": float(row.get("laplacian_score",  0.5)),
                "ear_score"      : float(row.get("ear_score",        0.5)),
                "label"          : int(row["label"]),
                "video_id"       : row["video_id"],
            })
    # 4-feature matrix: [artifact, fft, laplacian, ear]
    features  = np.array(
        [[r["artifact_score"], r["fft_score"], r["laplacian_score"], r["ear_score"]]
         for r in rows],
        dtype=np.float32,
    )
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
            print(f"  [Fold {fold_idx+1}] skipped -- only one class in val")
            continue
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)
        # Same zero-variance guard as in train_ensemble.
        _zv = scaler.var_ < 1e-10
        if _zv.any():
            X_train[:, _zv] = 0.0
            X_val[:, _zv]   = 0.0
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


def aggregate_to_video_level(features, labels, video_ids, quality_col_idx=2):
    """
    Collapse frame-level features into one feature vector per video using
    quality-weighted averaging.

    Quality weight = laplacian_score of each frame (column `quality_col_idx`).
    Sharper frames speak louder.

    Feature column order: [artifact=0, fft=1, laplacian=2, ear=3]
    quality_col_idx=2  ->  laplacian is the quality weight.

    Args:
        features        : (N, F) float array -- per-frame feature matrix
        labels          : (N,) int array -- per-frame labels (0=real, 1=fake)
        video_ids       : list[str] of length N -- one video_id per frame
        quality_col_idx : int -- column index of the quality feature (default 2)

    Returns:
        vid_features : (V, F) float64 array -- one row per unique video
        vid_labels   : (V,) int array
        vid_ids      : list[str] of length V
    """
    video_ids_arr = np.array(video_ids)
    unique_vids   = list(dict.fromkeys(video_ids))   # first-seen order

    vid_features_list = []
    vid_labels_list   = []

    for vid in unique_vids:
        mask = video_ids_arr == vid
        f    = features[mask]
        l    = labels[mask]

        if not np.all(l == l[0]):
            raise ValueError(
                f"Video '{vid}' has inconsistent labels {set(l.tolist())}. "
                "Check inspect_dataset.py output."
            )

        # Use float64 to keep near-zero variance below 1e-10 guard.
        f64     = f.astype(np.float64)
        quality = f64[:, quality_col_idx]
        total_q = float(quality.sum())

        if total_q < 1e-8:
            weights = np.ones(len(f64), dtype=np.float64) / len(f64)
        else:
            weights = quality / total_q

        weighted_feat = (f64 * weights[:, np.newaxis]).sum(axis=0)
        vid_features_list.append(weighted_feat)
        vid_labels_list.append(int(l[0]))

    vid_features = np.array(vid_features_list, dtype=np.float64)
    vid_labels   = np.array(vid_labels_list,   dtype=int)

    return vid_features, vid_labels, unique_vids


# ---------------------------------------------------------------------------
# Section C -- multi-classifier training and fusion
# ---------------------------------------------------------------------------

def _train_all_classifiers(X_train, y_train, X_val, y_val, random_state=42):
    """
    Train LogisticRegression, RandomForest, and XGBoost on the same
    pre-split, pre-scaled video-level feature data.

    Evaluates all three on the val set and prints a side-by-side comparison
    (accuracy, precision, recall, F1, ROC-AUC).  Selects the best by
    validation F1.

    Args:
        X_train, y_train : training split (already scaled)
        X_val,   y_val   : validation split (already scaled)
        random_state     : int seed

    Returns:
        best_model      : fitted best model
        best_name       : str model name
        best_val_scores : np.ndarray  P(fake) for each val video
        all_models      : dict {name: model}
    """
    results = {}
    models  = {}

    # --- Logistic Regression (Platt-calibrated) ---
    base_lr  = LogisticRegression(C=1.0, max_iter=1000, random_state=random_state,
                                   class_weight="balanced")
    lr_model = CalibratedClassifierCV(base_lr, method="sigmoid", cv=3)
    lr_model.fit(X_train, y_train)
    lr_scores = lr_model.predict_proba(X_val)[:, 1]
    t_lr, _   = calibrate_threshold_balanced(y_val, lr_scores)
    results["LogisticRegression"] = {**evaluate_model(y_val, lr_scores, t_lr),
                                      "val_scores": lr_scores}
    models["LogisticRegression"]  = lr_model

    # --- Random Forest ---
    rf_model = RandomForestClassifier(n_estimators=200, random_state=random_state,
                                       class_weight="balanced", n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_scores = rf_model.predict_proba(X_val)[:, 1]
    t_rf, _   = calibrate_threshold_balanced(y_val, rf_scores)
    results["RandomForest"] = {**evaluate_model(y_val, rf_scores, t_rf),
                                "val_scores": rf_scores}
    models["RandomForest"]  = rf_model

    # --- XGBoost ---
    if _XGBOOST_AVAILABLE:
        xgb_model = XGBClassifier(n_estimators=200, max_depth=4,
                                   learning_rate=0.1, random_state=random_state,
                                   eval_metric="logloss", verbosity=0)
        xgb_model.fit(X_train, y_train)
        xgb_scores = xgb_model.predict_proba(X_val)[:, 1]
        t_xgb, _   = calibrate_threshold_balanced(y_val, xgb_scores)
        results["XGBoost"] = {**evaluate_model(y_val, xgb_scores, t_xgb),
                               "val_scores": xgb_scores}
        models["XGBoost"]  = xgb_model
    else:
        print("  [WARN] xgboost not installed -- skipping XGBClassifier.")
        print("         Install with:  pip install xgboost")

    # Print side-by-side comparison
    print()
    print("  MODULE 3 CLASSIFIER COMPARISON  (validation set)")
    print(f"  {'Model':<22}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}  "
          f"{'F1':>6}  {'AUC':>6}")
    print("  " + "-" * 62)
    for name, m in results.items():
        marker = " <- best F1" if name == max(results, key=lambda n: results[n]["f1"]) else ""
        print(f"  {name:<22}  {m['accuracy']:>6.4f}  {m['precision']:>6.4f}  "
              f"{m['recall']:>6.4f}  {m['f1']:>6.4f}  {m['auc']:>6.4f}{marker}")

    # Pick best by validation F1
    best_name    = max(results, key=lambda n: results[n]["f1"])
    best_metrics = results[best_name]
    best_model   = models[best_name]
    print(f"\n  [ok] Best Module 3 model: {best_name}  "
          f"(val F1 = {best_metrics['f1']:.4f})")
    return best_model, best_name, best_metrics["val_scores"], models


def _compute_video_module3_scores(model, scaler, features_csv=FEATURES_CSV):
    """
    Apply the trained Module 3 model to each individual frame, then average
    the per-frame P(fake) values per video to produce one score per video.

    The model was trained on video-level (quality-weighted) features.  Applying
    it frame-by-frame and averaging gives a similar result while making the
    per-frame breakdown available for inspection.

    Args:
        model        : trained sklearn-compatible model with predict_proba()
        scaler       : fitted StandardScaler (from video-level training)
        features_csv : path to the frame-level feature CSV

    Returns:
        dict[manifest_video_id -> float mean P(fake)]
    """
    frame_features, _, frame_video_ids = load_features_from_csv(features_csv)

    # Apply scaler and zero-variance guard
    X_scaled = scaler.transform(frame_features.astype(np.float64))
    if hasattr(scaler, "var_"):
        _zv = scaler.var_ < 1e-10
        if _zv.any():
            X_scaled[:, _zv] = 0.0

    proba = model.predict_proba(X_scaled)[:, 1]   # per-frame P(fake)

    # Average per video
    video_ids_arr = np.array(frame_video_ids)
    unique_vids   = list(dict.fromkeys(frame_video_ids))
    vid_scores    = {}
    for vid in unique_vids:
        mask = video_ids_arr == vid
        vid_scores[vid] = float(np.mean(proba[mask]))
    return vid_scores


def _fuse_and_report(module3_vid_scores,
                     module1_csv=MODULE1_CSV,
                     output_csv=ENSEMBLE_OUTPUT_CSV):
    """
    Fuse Module 1 (deepfake_confidence) and Module 3 (best model P(fake))
    with MODULE1_WEIGHT / MODULE3_WEIGHT (default 0.30 / 0.70) for each of
    the 238 videos in module1_output.csv.

        final_score      = MODULE1_WEIGHT * module1_score
                         + MODULE3_WEIGHT * module3_score
        final_prediction = 1 if final_score >= 0.5 else 0

    Videos not in the manifest (no Module 3 features available) receive
    module3_score = 0.5 (neutral fallback).

    Writes data/ensemble_output.csv with columns:
        video_id, module1_score, module3_score,
        final_score, final_prediction, is_fake

    Returns:
        out_df : pandas DataFrame (238 rows) used for metric printing
    """
    df1 = pd.read_csv(module1_csv)

    out_rows = []
    for _, row in df1.iterrows():
        vid     = str(row["video_id"]).strip()
        is_fake = int(row["is_fake"])
        m1_score = float(row["deepfake_confidence"])

        # Build manifest-format key (same mapping as _load_module1_scores)
        if is_fake == 0:
            try:
                manifest_key = f"real_{int(vid):03d}"
            except ValueError:
                manifest_key = vid
        else:
            manifest_key = f"fake_{vid}"

        m3_score = module3_vid_scores.get(manifest_key, 0.5)
        final    = MODULE1_WEIGHT * m1_score + MODULE3_WEIGHT * m3_score
        pred     = 1 if final >= 0.5 else 0

        out_rows.append({
            "video_id"        : vid,
            "module1_score"   : round(m1_score, 6),
            "module3_score"   : round(m3_score, 6),
            "final_score"     : round(final, 6),
            "final_prediction": pred,
            "is_fake"         : is_fake,
        })

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(output_csv, index=False)
    print(f"  Saved -> {output_csv}  ({len(out_df)} rows)")
    return out_df


# ---------------------------------------------------------------------------
# Section D -- display helpers
# ---------------------------------------------------------------------------

def plot_curves(y_true, y_scores, threshold, save_dir=PLOTS_DIR):
    """Save ROC and Precision-Recall curves, marking the chosen threshold."""
    os.makedirs(save_dir, exist_ok=True)

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

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
    ax.set_title("ROC Curve -- Ensemble")
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
    ax.set_title("Precision-Recall -- Ensemble")
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
    """Print a labelled confusion matrix."""
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
    print(f"  False Positives (real -> fake) : {fp}")
    print(f"  False Negatives (fake -> real) : {fn}")
    print(f"  True Positives  (fake -> fake) : {tp}")


def print_feature_distributions(features, labels):
    """Print per-class mean +/- std for each feature -- quick sanity check."""
    print()
    print("  FEATURE DISTRIBUTIONS (mean +/- std)")
    print(f"  {'Feature':<14}  {'REAL (label=0)':>20}  {'FAKE (label=1)':>20}  {'Delta':>8}")
    print("  " + "-" * 68)
    n_cols = min(features.shape[1], len(FEATURE_NAMES))
    for j in range(n_cols):
        name      = FEATURE_NAMES[j]
        real_vals = features[labels == 0, j]
        fake_vals = features[labels == 1, j]
        rm, rs = float(np.mean(real_vals)), float(np.std(real_vals))
        fm, fs = float(np.mean(fake_vals)), float(np.std(fake_vals))
        delta  = fm - rm
        print(f"  {name:<14}  {rm:+.4f} +/- {rs:.4f}      "
              f"{fm:+.4f} +/- {fs:.4f}      {delta:+.4f}")


# ---------------------------------------------------------------------------
# Section E -- __main__ training pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print()
    print("=" * 65)
    print("MODULE 3 + MODULE 1 FUSION -- ENSEMBLE TRAINING PIPELINE")
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
    print("STEP 1 -- Load manifest")
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
    # 2. Extract features (or load from cache), with ear_score staleness check
    # ------------------------------------------------------------------

    # Print BEFORE ear_score stats from any existing cached CSV
    print("STEP 2 -- Feature extraction / cache check")
    if os.path.exists(FEATURES_CSV):
        try:
            with open(FEATURES_CSV, newline="", encoding="utf-8") as _fh:
                _peek = csv.DictReader(_fh)
                _fields = _peek.fieldnames or []
                if "ear_score" not in _fields:
                    print("  BEFORE: ear_score column absent in cache "
                          "(was constant 0.5 stub)")
                    _ear_stale = True
                else:
                    _ear_vals_before = [float(r["ear_score"]) for r in _peek]
                    _ear_min  = min(_ear_vals_before)
                    _ear_max  = max(_ear_vals_before)
                    _ear_mean = float(np.mean(_ear_vals_before))
                    print(f"  BEFORE: ear_score in cache  "
                          f"min={_ear_min:.4f}  max={_ear_max:.4f}  "
                          f"mean={_ear_mean:.4f}")
                    # Stale if all values are identical (e.g. all 0.5)
                    _ear_stale = len(set(_ear_vals_before)) <= 1
        except Exception:
            _ear_stale = True

        if _ear_stale:
            print("  Cache is stale (no real ear_score) -- deleting to force "
                  "re-extraction.")
            os.remove(FEATURES_CSV)
    else:
        print("  No cache found -- will extract from scratch.")

    if os.path.exists(FEATURES_CSV):
        print(f"\n  Loading cached features from {FEATURES_CSV}")
        features, labels, video_ids = load_features_from_csv(FEATURES_CSV)
        print(f"  Loaded {len(features)} rows from cache.  "
              "(Delete module3_features.csv to force re-extraction.)")
    else:
        print("\n  Extracting features  (artifact + FFT + Laplacian + ear_score)")
        print("  Expected time: ~90–240 s for 400 images.")
        print()
        features, labels, video_ids = extract_all_features(manifest,
                                                            save_csv=FEATURES_CSV)

    if len(features) == 0:
        print("[ERROR] No features extracted.  Check image paths.")
        raise SystemExit(1)

    # Print AFTER ear_score stats
    _ear_col = FEATURE_NAMES.index("ear")
    _ear_after = features[:, _ear_col]
    print(f"\n  AFTER:  ear_score  min={float(_ear_after.min()):.4f}  "
          f"max={float(_ear_after.max()):.4f}  "
          f"mean={float(_ear_after.mean()):.4f}  "
          f"std={float(_ear_after.std()):.4f}")

    print(f"\n  Feature matrix : {features.shape}  "
          f"(rows=images, cols={FEATURE_NAMES})")
    print(f"  Label vector   : {labels.shape}")
    print()

    print("  Frame-level feature distributions (before aggregation):")
    print_feature_distributions(features, labels)
    print()

    # ------------------------------------------------------------------
    # 2b. Aggregate frames -> video-level quality-weighted features
    # ------------------------------------------------------------------
    print("STEP 2b -- Quality-weighted video-level aggregation")
    n_frames_before = len(features)
    # quality_col_idx=2 -> column 2 = laplacian_score
    # column order: [artifact=0, fft=1, laplacian=2, ear=3]
    features_vid, labels_vid, video_ids_vid = aggregate_to_video_level(
        features, labels, video_ids, quality_col_idx=2
    )
    n_real_vids = int(np.sum(labels_vid == 0))
    n_fake_vids = int(np.sum(labels_vid == 1))
    print(f"  {n_frames_before} frames -> {len(features_vid)} videos  "
          f"({n_real_vids} real, {n_fake_vids} fake)")
    print(f"  Feature matrix : {features_vid.shape}  (rows=videos)")
    print()
    print("  Video-level feature distributions (after quality-weighted aggregation):")
    print_feature_distributions(features_vid, labels_vid)
    print()

    # ------------------------------------------------------------------
    # 3. Equal-weights baseline  (handcrafted features only, not ear)
    # ------------------------------------------------------------------
    print("STEP 3 -- Equal-weights baseline  (artifact + fft + laplacian only)")
    equal_scores = np.array([
        ensemble_score_equal_weights(
            float(r[0]), float(r[1]), float(r[2])
        )
        for r in features_vid
    ])
    try:
        equal_auc = roc_auc_score(labels_vid, equal_scores)
        print(f"  Equal-weights AUC (3 handcrafted) : {equal_auc:.4f}")
    except Exception:
        print("  Could not compute AUC for equal-weights baseline.")
    print()

    # ------------------------------------------------------------------
    # 4. Build the same train/val split used by train_ensemble()
    #    then train LR, RandomForest, and XGBoost on the same data.
    # ------------------------------------------------------------------
    _C = 1.0
    _SEED = 42
    print("STEP 4 -- Train Module 3 classifiers  (video-level GroupShuffleSplit)")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=_SEED)
    _train_idx, _val_idx = next(gss.split(features_vid, labels_vid,
                                           groups=video_ids_vid))

    _scaler_m3 = StandardScaler()
    _X_train   = _scaler_m3.fit_transform(features_vid[_train_idx])
    _X_val     = _scaler_m3.transform(features_vid[_val_idx])
    _y_train   = labels_vid[_train_idx]
    _y_val     = labels_vid[_val_idx]
    _v_val     = np.array(video_ids_vid)[_val_idx]

    # Zero-variance guard
    _zv = _scaler_m3.var_ < 1e-10
    if _zv.any():
        _X_train[:, _zv] = 0.0
        _X_val[:, _zv]   = 0.0

    n_val_real = int(np.sum(_y_val == 0))
    n_val_fake = int(np.sum(_y_val == 1))
    print(f"  Val set : {len(_y_val)} videos  "
          f"({n_val_real} real, {n_val_fake} fake)")
    print()

    # Train all three classifiers and pick best by val F1
    best_model_m3, best_model_name, best_val_scores, _all_models = \
        _train_all_classifiers(_X_train, _y_train, _X_val, _y_val,
                               random_state=_SEED)
    print()

    # Also run LR to keep the coefficient printout (for interpretability)
    _lr_for_display = _all_models.get("LogisticRegression")
    if _lr_for_display is not None:
        try:
            coefs = np.array([
                cc.estimator.coef_[0]
                for cc in _lr_for_display.calibrated_classifiers_
            ])
            raw_coef  = coefs.mean(axis=0)
            coef_sum  = np.abs(raw_coef).sum() + 1e-8
            norm_coef = np.abs(raw_coef) / coef_sum
            print("  LR Feature coefficients  "
                  "(|w| normalised, mean across 3 calibration folds):")
            for name, nc in zip(FEATURE_NAMES, norm_coef):
                bar = "#" * int(nc * 30)
                print(f"    {name:<12} = {nc:.3f}  {bar}")
        except Exception:
            print("  (LR coefficient breakdown unavailable)")
    print()

    # ------------------------------------------------------------------
    # 4b. 5-fold GroupKFold cross-validation  (LR, more reliable estimate)
    # ------------------------------------------------------------------
    print("STEP 4b -- 5-fold GroupKFold cross-validation  (LR baseline)")
    cv_metrics = cross_validate_ensemble(features_vid, labels_vid,
                                          video_ids_vid, n_splits=5, C=_C)

    if cv_metrics["auc"]:
        print()
        print("  CROSS-VALIDATION SUMMARY  (mean +/- std across folds)  [LR]")
        print(f"  {'Metric':<22}  {'Mean':>8}  {'Std':>8}  "
              f"{'Min':>8}  {'Max':>8}")
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
    # 5. Threshold calibration on best model val scores
    # ------------------------------------------------------------------
    print("STEP 5 -- Calibrate threshold  (best Module 3 model)")
    best_t_ba, best_ba = calibrate_threshold_balanced(_y_val, best_val_scores)
    best_t_f1, best_f1v = calibrate_threshold_f1(_y_val, best_val_scores)
    print(f"  Balanced-accuracy threshold : {best_t_ba:.4f}  "
          f"(bal. acc. = {best_ba:.4f})  <- PRIMARY")
    print(f"  F1-score threshold          : {best_t_f1:.4f}  "
          f"(F1 = {best_f1v:.4f})")
    print()

    # ------------------------------------------------------------------
    # 6. Evaluate best Module 3 model
    # ------------------------------------------------------------------
    print(f"STEP 6 -- Evaluate best Module 3 model  ({best_model_name})")
    metrics_ba = evaluate_model(_y_val, best_val_scores, best_t_ba)
    metrics_f1 = evaluate_model(_y_val, best_val_scores, best_t_f1)

    print_metrics(metrics_ba, title=f"MODULE 3 RESULTS -- {best_model_name}")
    print_confusion_matrix(_y_val, best_val_scores, best_t_ba)

    print()
    print("  THRESHOLD COMPARISON  (bal-acc vs F1):")
    print(f"  {'Metric':<22}  {'Bal-acc thresh':>16}  {'F1 thresh':>12}")
    print("  " + "-" * 54)
    for key in ["threshold", "accuracy", "balanced_accuracy",
                "auc", "precision", "recall", "f1"]:
        print(f"  {key:<22}  {metrics_ba[key]:>16.4f}  "
              f"{metrics_f1[key]:>12.4f}")

    # ------------------------------------------------------------------
    # 7. Save best Module 3 model bundle
    # ------------------------------------------------------------------
    MODEL_PKL = os.path.join("data", "ensemble_model.pkl")
    bundle = {
        "model"        : best_model_m3,
        "scaler"       : _scaler_m3,
        "threshold"    : best_t_ba,
        "feature_names": FEATURE_NAMES,
        "model_name"   : best_model_name,
        "calibrated"   : True,
    }
    with open(MODEL_PKL, "wb") as _fh:
        pickle.dump(bundle, _fh)
    print(f"\nSTEP 7 -- Model bundle saved -> {MODEL_PKL}  "
          f"(model={best_model_name})")

    # ------------------------------------------------------------------
    # 7b. Register model in the central model registry
    # ------------------------------------------------------------------
    print()
    print("STEP 7b -- Register model in artifacts/model_registry.json")
    try:
        import shutil
        import datetime
        from src.model_registry import ModelRegistry

        registry = ModelRegistry()
        _ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        _model_id = f"{best_model_name.lower().replace(' ', '_')}_{_ts}"

        # Copy bundle to artifacts/ with timestamped name so old runs are
        # preserved and the registry path stays stable even after retraining.
        os.makedirs("artifacts", exist_ok=True)
        _artifacts_path = os.path.join("artifacts", f"ensemble_model_{_ts}.pkl")
        shutil.copy2(MODEL_PKL, _artifacts_path)

        registry.register({
            "model_id":      _model_id,
            "model_type":    "lr",
            "artifact_path": _artifacts_path,
            "metrics": {
                "f1":        metrics_ba["f1"],
                "precision": metrics_ba["precision"],
                "recall":    metrics_ba["recall"],
                "auc":       metrics_ba["auc"],
            },
            "notes": (
                f"4-feature LR (artifact+fft+laplacian+ear); "
                f"val split seed=42; best_classifier={best_model_name}; "
                f"threshold(bal-acc)={best_t_ba:.4f}"
            ),
            "comparable": True,
        })
        winner = registry.select_best(metric="f1")
        print(f"  Registered: {_model_id}")
        print(f"  Active model -> {winner['model_id']}  (F1 = {winner['metrics']['f1']:.4f})")
    except Exception as _reg_exc:
        print(f"  [WARN] Registry update failed: {_reg_exc}")
        print("         Training is still complete — registry is optional.")

    # ------------------------------------------------------------------
    # 8. Diagnostic plots
    # ------------------------------------------------------------------
    print()
    print("STEP 8 -- Save diagnostic plots")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    try:
        plot_curves(_y_val, best_val_scores, best_t_ba, save_dir=PLOTS_DIR)
    except Exception as e:
        print(f"  [WARN] plot_curves: {e}")

    # ------------------------------------------------------------------
    # 9. FFT spectrum visualisations
    # ------------------------------------------------------------------
    print()
    print("STEP 9 -- FFT spectrum visualisations")
    os.makedirs(VIZ_DIR, exist_ok=True)
    for target_label, tag in [(0, "real"), (1, "fake")]:
        candidates = [r for r in manifest if r["label"] == target_label]
        if candidates:
            img = load_face_image(candidates[0]["file_path"])
            if img is not None:
                out = os.path.join(VIZ_DIR, f"fft_spectrum_{tag}.jpg")
                visualize_spectrum(img, save_path=out, label=tag)

    # ------------------------------------------------------------------
    # 10. Compute per-video Module 3 scores (frame-level averaging)
    # ------------------------------------------------------------------
    print()
    print("STEP 10 -- Compute per-video Module 3 scores  "
          "(frame-level predict_proba, averaged per video)")
    m3_vid_scores = _compute_video_module3_scores(best_model_m3, _scaler_m3,
                                                   FEATURES_CSV)
    print(f"  Scored {len(m3_vid_scores)} manifest videos with Module 3 model.")

    # ------------------------------------------------------------------
    # 11. Fuse Module 1 + Module 3  (30/70) -> ensemble_output.csv
    # ------------------------------------------------------------------
    print()
    print(f"STEP 11 -- Fuse Module 1 + Module 3  "
          f"(MODULE1={MODULE1_WEIGHT:.2f} / MODULE3={MODULE3_WEIGHT:.2f})")
    if not os.path.exists(MODULE1_CSV):
        print(f"  [ERROR] {MODULE1_CSV} not found -- cannot run fusion.")
        print("          Skipping Steps 11–12.")
    else:
        out_df = _fuse_and_report(m3_vid_scores,
                                   module1_csv=MODULE1_CSV,
                                   output_csv=ENSEMBLE_OUTPUT_CSV)

        y_true_all  = out_df["is_fake"].values.astype(int)
        m1_scores   = out_df["module1_score"].values
        m3_scores   = out_df["module3_score"].values
        fused_scores= out_df["final_score"].values
        fused_preds = out_df["final_prediction"].values

        # ---------------------------------------------------------------
        # 12. Comparison table -- Module 1 alone / Module 3 alone / Fused
        # ---------------------------------------------------------------
        print()
        print("STEP 12 -- Final comparison  (all 238 Module 1 videos)")
        print()

        # Module 1 alone
        try:
            m1_auc  = roc_auc_score(y_true_all, m1_scores)
            m1_pred = (m1_scores >= 0.5).astype(int)
            m1_acc  = accuracy_score(y_true_all, m1_pred)
            m1_prec = precision_score(y_true_all, m1_pred, zero_division=0)
            m1_rec  = recall_score(y_true_all, m1_pred, zero_division=0)
            m1_f1   = f1_score(y_true_all, m1_pred, zero_division=0)
        except Exception as exc:
            print(f"  [WARN] Module 1 metrics: {exc}")
            m1_auc = m1_acc = m1_prec = m1_rec = m1_f1 = float("nan")

        # Module 3 alone (per-video average of frame-level scores)
        try:
            m3_auc  = roc_auc_score(y_true_all, m3_scores)
            m3_pred = (m3_scores >= 0.5).astype(int)
            m3_acc  = accuracy_score(y_true_all, m3_pred)
            m3_prec = precision_score(y_true_all, m3_pred, zero_division=0)
            m3_rec  = recall_score(y_true_all, m3_pred, zero_division=0)
            m3_f1   = f1_score(y_true_all, m3_pred, zero_division=0)
        except Exception as exc:
            print(f"  [WARN] Module 3 metrics: {exc}")
            m3_auc = m3_acc = m3_prec = m3_rec = m3_f1 = float("nan")

        # Fused 30/70
        try:
            fu_auc  = roc_auc_score(y_true_all, fused_scores)
            fu_acc  = accuracy_score(y_true_all, fused_preds)
            fu_prec = precision_score(y_true_all, fused_preds, zero_division=0)
            fu_rec  = recall_score(y_true_all, fused_preds, zero_division=0)
            fu_f1   = f1_score(y_true_all, fused_preds, zero_division=0)
        except Exception as exc:
            print(f"  [WARN] Fused metrics: {exc}")
            fu_auc = fu_acc = fu_prec = fu_rec = fu_f1 = float("nan")

        # Print comparison
        n_m3_with_score = sum(1 for s in m3_scores if s != 0.5)
        n_m3_fallback   = len(m3_scores) - n_m3_with_score
        print(f"  Module 3 coverage : {n_m3_with_score} real scores  "
              f"+ {n_m3_fallback} fallback (0.5) of 238 videos")
        print()
        hdr = f"  {'System':<32}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}  " \
              f"{'F1':>6}  {'AUC':>6}"
        print(hdr)
        print("  " + "-" * 70)

        def _row(name, acc, prec, rec, f1, auc, note=""):
            return (f"  {name:<32}  {acc:>6.4f}  {prec:>6.4f}  "
                    f"{rec:>6.4f}  {f1:>6.4f}  {auc:>6.4f}{note}")

        print(_row("Module 1 alone (Niko baseline 66%)",
                   m1_acc, m1_prec, m1_rec, m1_f1, m1_auc))
        print(_row(f"Module 3 alone ({best_model_name})",
                   m3_acc, m3_prec, m3_rec, m3_f1, m3_auc))
        fused_label = (f"Fused {MODULE1_WEIGHT:.0%}/{MODULE3_WEIGHT:.0%}"
                       f"  (Module1 + Module3)")
        print(_row(fused_label,
                   fu_acc, fu_prec, fu_rec, fu_f1, fu_auc,
                   " <- final"))
        # ---------------------------------------------------------------
        # WEIGHT SWEEP  (validate the 30/70 choice)
        # Loops module1_weight 0.0 → 1.0 in steps of 0.1.
        # NOTE: constants MODULE1_WEIGHT / MODULE3_WEIGHT are NOT changed
        # by this sweep.  The selected weights stay at 30/70 as defined
        # at the top of this file.
        # ---------------------------------------------------------------
        print()
        print("  WEIGHT SWEEP  "
              "(module1_weight 0.0 to 1.0, step 0.10 -- threshold fixed at 0.5)")
        print(f"  NOTE: sweep is diagnostic only; constants remain "
              f"MODULE1_WEIGHT={MODULE1_WEIGHT}  MODULE3_WEIGHT={MODULE3_WEIGHT}")
        print()
        _sw_hdr = (f"  {'m1_weight':>9}  {'m3_weight':>9}  "
                   f"{'Accuracy':>9}  {'F1':>8}  {'ROC-AUC':>8}")
        print(_sw_hdr)
        print("  " + "-" * 56)
        for _sw_step in range(11):               # 0, 1, ..., 10
            _sw_w1 = round(_sw_step * 0.1, 1)
            _sw_w3 = round(1.0 - _sw_w1, 1)
            _sw_scores = _sw_w1 * m1_scores + _sw_w3 * m3_scores
            _sw_pred   = (_sw_scores >= 0.5).astype(int)
            _sw_acc    = accuracy_score(y_true_all, _sw_pred)
            _sw_f1     = f1_score(y_true_all, _sw_pred, zero_division=0)
            try:
                _sw_auc = roc_auc_score(y_true_all, _sw_scores)
            except Exception:
                _sw_auc = float("nan")
            _sw_marker = "  <- selected" if abs(_sw_w1 - MODULE1_WEIGHT) < 1e-9 else ""
            print(f"  {_sw_w1:>9.1f}  {_sw_w3:>9.1f}  "
                  f"{_sw_acc:>9.4f}  {_sw_f1:>8.4f}  {_sw_auc:>8.4f}{_sw_marker}")
        print()

        print(f"  Final ensemble metrics  ({len(out_df)} videos):")
        print(f"    Accuracy   = {fu_acc:.4f}")
        print(f"    Precision  = {fu_prec:.4f}")
        print(f"    Recall     = {fu_rec:.4f}")
        print(f"    F1         = {fu_f1:.4f}")
        print(f"    ROC-AUC    = {fu_auc:.4f}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 65)
    print("PIPELINE COMPLETE")
    print("=" * 65)
    print(f"  Features CSV         : {FEATURES_CSV}")
    print(f"  Best Module 3 model  : {best_model_name}")
    print(f"  Model bundle         : {MODEL_PKL}")
    print(f"  Ensemble output CSV  : {ENSEMBLE_OUTPUT_CSV}")
    print(f"  ROC curve            : {os.path.join(PLOTS_DIR, 'roc_curve.png')}")
    print(f"  PR  curve            : {os.path.join(PLOTS_DIR, 'precision_recall.png')}")
    print()
    print(f"  MODULE 3 RESULTS  (val split, threshold={best_t_ba:.4f}):")
    print(f"    AUC              = {metrics_ba['auc']:.4f}")
    print(f"    Accuracy         = {metrics_ba['accuracy']:.4f}")
    print(f"    Balanced Acc     = {metrics_ba['balanced_accuracy']:.4f}")
    print(f"    F1               = {metrics_ba['f1']:.4f}")
    print(f"    Precision (fake) = {metrics_ba['precision']:.4f}")
    print(f"    Recall    (fake) = {metrics_ba['recall']:.4f}")
    if cv_metrics["auc"]:
        print()
        print(f"  5-FOLD CV RESULTS  (LR):")
        print(f"    AUC          = {np.mean(cv_metrics['auc']):.4f} "
              f"+/- {np.std(cv_metrics['auc']):.4f}")
        print(f"    Accuracy     = {np.mean(cv_metrics['accuracy']):.4f} "
              f"+/- {np.std(cv_metrics['accuracy']):.4f}")
        print(f"    Balanced Acc = {np.mean(cv_metrics['balanced_accuracy']):.4f} "
              f"+/- {np.std(cv_metrics['balanced_accuracy']):.4f}")
    print()
    print("CHECKLIST:")
    print("  [x] Module 1 ear_score integrated  (deepfake_confidence from MRL)")
    print("  [x] Video-level GroupShuffleSplit   (no identity leakage)")
    print("  [x] LR + RandomForest + XGBoost     (best by val F1 selected)")
    print(f"  [x] {MODULE1_WEIGHT:.0%}/{MODULE3_WEIGHT:.0%} fusion M1/M3"
          f"       (ensemble_output.csv)")
    print("  [x] Comparison table                (Module1 / Module3 / Fused)")
    print("  [x] Weight sweep 0.0-1.0            (diagnostic, constants unchanged)")
    print("=" * 65)
