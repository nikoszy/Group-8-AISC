# =============================================================================
# stacking_ensemble.py  —  Learn the optimal CNN / LR blend weight
# =============================================================================
#
# WHAT THIS DOES
# --------------
# Both the CNN and the LR were trained on the same 80% training split and
# have never seen the held-out 20% val set.  This script uses those clean
# val-set predictions to find the single number alpha (0-1) that maximises
# the AUC of:
#
#     combined_score = alpha * CNN_prob + (1 - alpha) * LR_prob
#
# WHY NOT HARDCODE 65/35?
# ------------------------
# predict.py currently hardcodes 0.65 CNN + 0.35 LR.  That was a guess.
# The CNN scores 0.904 AUC and the LR scores ~0.575 AUC.  Giving 35% weight
# to a 0.575-AUC model adds noise.  Finding the optimal alpha from real
# labelled data is both more principled and more accurate.
#
# HOW LEAKAGE IS AVOIDED
# -----------------------
# The val set (20% of videos, ~34 videos) was never seen by either model
# during training.  We use it ONLY to evaluate predictions and find the
# optimal alpha — not to retrain either base model.  This is sometimes
# called "hold-out blending" and is a standard, leakage-free approach.
#
# OUTPUTS
# -------
#   data/stacking_bundle.pkl  — {"alpha": float, "cnn_auc": float,
#                                 "lr_auc": float, "combined_auc": float,
#                                 "hardcoded_auc": float}
#
# USAGE
# -----
#   python stacking_ensemble.py
#
# PREREQUISITES
# -------------
#   1. python ensemble.py     (produces data/ensemble_model.pkl + features CSV)
#   2. data/cnn_model.pth     (trained CNN checkpoint)
# =============================================================================

import os
import csv
import pickle
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")

from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold
from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import StandardScaler
from sklearn.calibration     import CalibratedClassifierCV
from sklearn.metrics         import roc_auc_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MANIFEST_PATH    = os.path.join("data", "manifest.csv")
FEATURES_CSV     = os.path.join("data", "module3_features.csv")
LR_BUNDLE_PATH   = os.path.join("data", "ensemble_model.pkl")
CNN_MODEL_PATH   = os.path.join("data", "cnn_model.pth")
STACK_BUNDLE_PATH= os.path.join("data", "stacking_bundle.pkl")

# Must match ensemble.py — same seed → same val split
_SPLIT_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_manifest(path=MANIFEST_PATH):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append({
                "file_path": row["file_path"],
                "label"    : int(row["label"]),
                "video_id" : row.get("video_id", "unknown"),
            })
    return rows


def get_val_indices(rows, seed=_SPLIT_SEED):
    """
    Reproduce the exact GroupShuffleSplit used in ensemble.py.
    Returns val_idx — the row indices that belong to the val set.
    """
    groups = np.array([r["video_id"] for r in rows])
    labels = np.array([r["label"]    for r in rows])
    gss    = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    _, val_idx = next(gss.split(np.arange(len(rows)), labels, groups=groups))
    return val_idx


def get_cnn_video_scores(val_rows):
    """
    Run CNN on every frame in val_rows.
    Returns:
        video_scores : dict { video_id -> mean CNN P(fake) over its frames }
        None if CNN is unavailable.

    Each val frame is a separate JPEG.  We average the per-frame predictions
    to get one score per video — the same aggregation that predict.py uses.
    """
    from src.cnn_runner import load_cnn, cnn_predict

    model = load_cnn(model_path=CNN_MODEL_PATH, verbose=True)
    if model is None:
        print("[stacking] CNN not available — skipping.")
        return None

    video_frame_scores = {}   # video_id -> list of frame probs
    n = len(val_rows)
    print(f"  Scoring {n} val frames with CNN …")

    for i, row in enumerate(val_rows):
        img = cv2.imread(row["file_path"])
        if img is None:
            continue

        prob = cnn_predict(model, img)
        if prob is None:
            continue

        vid = row["video_id"]
        if vid not in video_frame_scores:
            video_frame_scores[vid] = []
        video_frame_scores[vid].append(prob)

        if (i + 1) % 30 == 0 or (i + 1) == n:
            print(f"    {i+1}/{n} frames scored …")

    # Average over frames → one score per video
    video_scores = {
        vid: float(np.mean(probs))
        for vid, probs in video_frame_scores.items()
    }
    print(f"  CNN scored {len(video_scores)} unique val videos.")
    return video_scores


def get_lr_video_scores(val_rows, lr_model, lr_scaler, features_csv=FEATURES_CSV):
    """
    For each val video, load its frame-level features from the cached CSV,
    apply quality-weighted averaging (matching ensemble.py's aggregation),
    then run the LR model to get P(fake).

    Uses all 4 features: artifact, fft, laplacian, ear_score.
    (ensemble.py now trains on 4 features — this function must match.)

    Returns:
        lr_video_scores : dict { video_id -> LR P(fake) }
    """
    # Index features CSV by file_path — 4 features: artifact, fft, lap, ear
    frame_data = {}  # file_path -> {artifact, fft, lap, ear, video_id, label}
    with open(features_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            frame_data[row["file_path"]] = {
                "video_id": row["video_id"],
                "label"   : int(row["label"]),
                "artifact": float(row.get("artifact_score",  0.5)),
                "fft"     : float(row.get("fft_score",       0.5)),
                "lap"     : float(row.get("laplacian_score", 0.5)),
                "ear"     : float(row.get("ear_score",       0.5)),
            }

    # Group val frames by video_id
    video_frames  = {}  # video_id -> list of [artifact, fft, lap, ear]
    video_quality = {}  # video_id -> list of laplacian values (quality proxy)
    video_labels  = {}  # video_id -> label

    for row in val_rows:
        fp  = row["file_path"]
        vid = row["video_id"]
        if fp not in frame_data:
            continue
        fd  = frame_data[fp]
        if vid not in video_frames:
            video_frames[vid]  = []
            video_quality[vid] = []
            video_labels[vid]  = fd["label"]
        video_frames[vid].append([fd["artifact"], fd["fft"], fd["lap"], fd["ear"]])
        video_quality[vid].append(fd["lap"])

    # Quality-weighted average → LR predict_proba
    lr_video_scores = {}
    zero_var = (lr_scaler.var_ < 1e-10) if hasattr(lr_scaler, "var_") else np.zeros(4, dtype=bool)

    for vid, frames in video_frames.items():
        feats   = np.array(frames, dtype=np.float64)
        quals   = np.array(video_quality[vid], dtype=np.float64)
        total_q = quals.sum()
        weights = (quals / total_q) if total_q > 1e-8 else np.ones(len(feats)) / len(feats)
        vid_feat = (feats * weights[:, np.newaxis]).sum(axis=0).reshape(1, -1)

        scaled = lr_scaler.transform(vid_feat.astype(np.float32))
        if zero_var.any():
            scaled[:, zero_var] = 0.0

        lr_video_scores[vid] = float(lr_model.predict_proba(scaled)[0, 1])

    print(f"  LR scored {len(lr_video_scores)} unique val videos.")
    return lr_video_scores


def sweep_alpha(cnn_scores, lr_scores, labels, n_steps=201):
    """
    Sweep alpha from 0.0 to 1.0 in n_steps steps.
    combined = alpha * CNN + (1-alpha) * LR

    Returns (best_alpha, best_auc, all_alphas, all_aucs).
    Raises ValueError if labels contain fewer than two classes.
    """
    if len(np.unique(labels)) < 2:
        raise ValueError(
            f"sweep_alpha requires both classes present; "
            f"got unique labels = {np.unique(labels).tolist()}"
        )
    alphas = np.linspace(0.0, 1.0, n_steps)
    aucs   = np.zeros(n_steps)

    for i, a in enumerate(alphas):
        combined = a * cnn_scores + (1.0 - a) * lr_scores
        aucs[i]  = roc_auc_score(labels, combined)

    best_idx   = int(np.argmax(aucs))
    return float(alphas[best_idx]), float(aucs[best_idx]), alphas, aucs


# ---------------------------------------------------------------------------
# Cross-validation of alpha stability
# ---------------------------------------------------------------------------

def cross_validate_alpha(all_rows, features_csv=FEATURES_CSV,
                          n_folds=5, n_alpha_steps=101):
    """
    5-fold StratifiedKFold validation of the optimal CNN/LR blend weight.

    WHY THIS IS NEEDED
    ------------------
    The single held-out val set (34 videos) is too small to reliably estimate
    alpha.  With ~17 videos per class, a 95% CI on AUC is roughly ±0.17 —
    wide enough that the "optimal" alpha found on one split could easily be
    a fluke.

    This function repeats the alpha sweep over 5 non-overlapping folds of
    the full 167-video dataset.  Each fold uses ~134 videos for LR training
    and ~33 videos for the alpha sweep.  If the best alpha is consistently
    near the same value across all 5 folds, it is a real signal; if it
    varies wildly, we should not trust it.

    HOW IT WORKS
    ------------
    1.  Score ALL 167 videos with CNN (one pass — cached after first load).
    2.  Load frame-level features from CSV and aggregate to video level.
    3.  StratifiedKFold(5) over unique videos, stratified by label so each
        fold has a balanced class mix.
    4.  For each fold:
          a.  Re-train LR on the train-fold video features (fresh scaler).
          b.  Get LR P(fake) for each val-fold video.
          c.  Fetch pre-cached CNN P(fake) for the same val-fold videos.
          d.  Sweep alpha from 0→1 in n_alpha_steps steps; record best alpha
              and the AUC achieved at that alpha.
    5.  Return a dict with per-fold alphas, AUCs, CNN AUCs, LR AUCs.

    Args:
        all_rows      : list of manifest dicts (from load_manifest) — needed
                        to locate JPEG paths for CNN scoring.
        features_csv  : path to the per-frame features CSV (module3_features.csv).
        n_folds       : number of CV folds (default 5).
        n_alpha_steps : granularity of the alpha sweep (default 101 = 0.01 steps).

    Returns:
        dict with keys:
            "alphas"   : list[float] — best alpha per fold
            "aucs"     : list[float] — combined AUC at best alpha, per fold
            "cnn_aucs" : list[float] — CNN-alone AUC per fold (upper bound)
            "lr_aucs"  : list[float] — LR-alone AUC per fold
        Returns None if CNN is unavailable or fewer than 2 classes per fold.
    """
    # ------------------------------------------------------------------
    # 1. Pre-cache CNN scores for every video  (one full pass, ~778 frames)
    # ------------------------------------------------------------------
    print("  [CV] Pre-scoring ALL videos with CNN (this is the slow step) …")
    all_cnn_scores = get_cnn_video_scores(all_rows)   # dict: video_id -> float
    if all_cnn_scores is None:
        print("  [CV] CNN unavailable — cannot run CV alpha validation.")
        return None
    print(f"  [CV] CNN scored {len(all_cnn_scores)} unique videos.")
    print()

    # ------------------------------------------------------------------
    # 2. Load frame-level features and group by video
    # ------------------------------------------------------------------
    # frame_features : dict file_path -> {video_id, label, artifact, fft, lap, ear}
    # 4 features — matches ensemble.py's FEATURE_NAMES = [artifact, fft, laplacian, ear]
    frame_features = {}
    with open(features_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            frame_features[row["file_path"]] = {
                "video_id": row["video_id"],
                "label"   : int(row["label"]),
                "artifact": float(row.get("artifact_score",  0.5)),
                "fft"     : float(row.get("fft_score",       0.5)),
                "lap"     : float(row.get("laplacian_score", 0.5)),
                "ear"     : float(row.get("ear_score",       0.5)),
            }

    # Group into per-video containers
    vid_frame_feats = {}   # video_id -> list of [4]-element lists
    vid_frame_quals = {}   # video_id -> list of laplacian values (quality proxy)
    vid_label_map   = {}   # video_id -> int label

    for fd in frame_features.values():
        vid = fd["video_id"]
        if vid not in vid_frame_feats:
            vid_frame_feats[vid] = []
            vid_frame_quals[vid] = []
            vid_label_map[vid]   = fd["label"]
        vid_frame_feats[vid].append([fd["artifact"], fd["fft"], fd["lap"], fd["ear"]])
        vid_frame_quals[vid].append(fd["lap"])

    unique_vids = sorted(vid_label_map.keys())
    vid_labels  = np.array([vid_label_map[v] for v in unique_vids], dtype=int)

    # ------------------------------------------------------------------
    # 2b. Helper: quality-weighted video-level aggregation (mirrors
    #     ensemble.py's aggregate_to_video_level for a subset of videos)
    # ------------------------------------------------------------------
    def _aggregate(vids):
        """Return (X: np.float64, y: np.int) for a list of video_ids."""
        X, y = [], []
        for v in vids:
            feats = np.array(vid_frame_feats[v], dtype=np.float64)
            quals = np.array(vid_frame_quals[v], dtype=np.float64)
            total_q = float(quals.sum())
            weights = (quals / total_q
                       if total_q > 1e-8
                       else np.ones(len(feats), dtype=np.float64) / len(feats))
            X.append((feats * weights[:, np.newaxis]).sum(axis=0))
            y.append(vid_label_map[v])
        return np.array(X, dtype=np.float64), np.array(y, dtype=int)

    # ------------------------------------------------------------------
    # 3. 5-fold stratified split over unique videos
    # ------------------------------------------------------------------
    # StratifiedKFold keeps class proportions balanced across folds, which
    # matters here because we only have ~167 videos.
    skf     = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=_SPLIT_SEED)
    results = {"alphas": [], "aucs": [], "cnn_aucs": [], "lr_aucs": []}

    vid_idx = np.arange(len(unique_vids))   # integer indices into unique_vids

    for fold_i, (train_vi, val_vi) in enumerate(skf.split(vid_idx, vid_labels)):
        train_vids = [unique_vids[i] for i in train_vi]
        val_vids   = [unique_vids[i] for i in val_vi]

        # ---- 4a. Train LR on train-fold --------------------------------
        X_tr, y_tr = _aggregate(train_vids)
        X_va, y_va = _aggregate(val_vids)

        if len(np.unique(y_va)) < 2:
            print(f"  [CV Fold {fold_i+1}] SKIPPED — only one class in val fold")
            continue

        # Fit scaler on train ONLY (no leakage to val)
        scaler    = StandardScaler()
        X_tr_s    = scaler.fit_transform(X_tr)
        X_va_s    = scaler.transform(X_va)

        # Zero-variance guard: same logic as ensemble.py's train_ensemble()
        zv = scaler.var_ < 1e-10
        if zv.any():
            X_tr_s[:, zv] = 0.0
            X_va_s[:, zv] = 0.0

        # CalibratedClassifierCV needs ≥ cv classes per class;
        # fall back to plain LR if the fold is very small
        base_lr = LogisticRegression(C=1.0, max_iter=1000,
                                      random_state=_SPLIT_SEED,
                                      class_weight="balanced")
        min_class_count = min(int(np.sum(y_tr == 0)), int(np.sum(y_tr == 1)))
        if min_class_count >= 3:
            lr_model = CalibratedClassifierCV(base_lr, method="sigmoid", cv=3)
        else:
            lr_model = base_lr   # too few samples for Platt scaling
        lr_model.fit(X_tr_s, y_tr)

        # ---- 4b. LR scores for val-fold videos --------------------------
        lr_proba = lr_model.predict_proba(X_va_s)[:, 1]
        lr_vid_scores = {v: float(lr_proba[i]) for i, v in enumerate(val_vids)}

        # ---- 4c. CNN scores for val-fold videos -------------------------
        # Keep only videos that the CNN successfully scored
        common = [v for v in val_vids if v in all_cnn_scores]
        if len(common) < 4:
            print(f"  [CV Fold {fold_i+1}] SKIPPED — only {len(common)} videos "
                  "have CNN scores (need ≥ 4)")
            continue

        y_fold   = np.array([vid_label_map[v]    for v in common], dtype=int)
        cnn_fold = np.array([all_cnn_scores[v]   for v in common], dtype=np.float64)
        lr_fold  = np.array([lr_vid_scores[v]    for v in common], dtype=np.float64)

        if len(np.unique(y_fold)) < 2:
            print(f"  [CV Fold {fold_i+1}] SKIPPED — only one class after "
                  "CNN-score intersection")
            continue

        # ---- 4d. Alpha sweep --------------------------------------------
        try:
            best_a, best_auc, _, _ = sweep_alpha(
                cnn_fold, lr_fold, y_fold, n_steps=n_alpha_steps
            )
        except ValueError as exc:
            print(f"  [CV Fold {fold_i+1}] SKIPPED — {exc}")
            continue

        cnn_auc_f = roc_auc_score(y_fold, cnn_fold)
        lr_auc_f  = roc_auc_score(y_fold, lr_fold)

        results["alphas"].append(best_a)
        results["aucs"].append(best_auc)
        results["cnn_aucs"].append(cnn_auc_f)
        results["lr_aucs"].append(lr_auc_f)

        n_r = int(np.sum(y_fold == 0))
        n_f = int(np.sum(y_fold == 1))
        print(f"  [CV Fold {fold_i+1}/{n_folds}]  "
              f"val={len(common)} vids ({n_r}r/{n_f}f)  "
              f"CNN={cnn_auc_f:.4f}  LR={lr_auc_f:.4f}  "
              f"best_alpha={best_a:.2f}  combined={best_auc:.4f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print()
    print("=" * 65)
    print("STACKING ENSEMBLE — learn optimal CNN / LR blend weight")
    print("=" * 65)
    print()

    # ------------------------------------------------------------------
    # 0. Prerequisites
    # ------------------------------------------------------------------
    for path, name in [
        (MANIFEST_PATH,  "manifest.csv"),
        (FEATURES_CSV,   "module3_features.csv (run ensemble.py first)"),
        (LR_BUNDLE_PATH, "ensemble_model.pkl   (run ensemble.py first)"),
        (CNN_MODEL_PATH, "cnn_model.pth        (run cnn_detector.py first)"),
    ]:
        if not os.path.exists(path):
            print(f"[ERROR] Missing: {path}")
            print(f"        Need {name}")
            raise SystemExit(1)

    # ------------------------------------------------------------------
    # 1. Load manifest + reproduce val split
    # ------------------------------------------------------------------
    print("STEP 1 — Load manifest and reproduce val split")
    all_rows = load_manifest()
    val_idx  = get_val_indices(all_rows, seed=_SPLIT_SEED)
    val_rows = [all_rows[i] for i in val_idx]

    n_val_real = sum(1 for r in val_rows if r["label"] == 0)
    n_val_fake = sum(1 for r in val_rows if r["label"] == 1)
    n_val_vids = len(set(r["video_id"] for r in val_rows))
    print(f"  Val set : {len(val_rows)} frames  "
          f"({n_val_real} real, {n_val_fake} fake)  "
          f"from {n_val_vids} videos")
    print()

    # ------------------------------------------------------------------
    # 2. CNN video-level scores
    # ------------------------------------------------------------------
    print("STEP 2 — Score val frames with CNN (EfficientNet-B0)")
    cnn_video_scores = get_cnn_video_scores(val_rows)
    if cnn_video_scores is None:
        print("[ERROR] CNN not available.  Ensure data/cnn_model.pth exists "
              "and PyTorch is installed.")
        raise SystemExit(1)
    print()

    # ------------------------------------------------------------------
    # 3. LR video-level scores
    # ------------------------------------------------------------------
    print("STEP 3 — Score val videos with LR ensemble")
    with open(LR_BUNDLE_PATH, "rb") as fh:
        lr_bundle = pickle.load(fh)
    lr_model  = lr_bundle["model"]
    lr_scaler = lr_bundle["scaler"]

    lr_video_scores = get_lr_video_scores(val_rows, lr_model, lr_scaler)
    print()

    # ------------------------------------------------------------------
    # 4. Align videos — keep only videos scored by BOTH models
    # ------------------------------------------------------------------
    print("STEP 4 — Align CNN and LR scores at video level")
    common_vids = sorted(set(cnn_video_scores) & set(lr_video_scores))
    if len(common_vids) == 0:
        print("[ERROR] No videos scored by both CNN and LR.  Check paths.")
        raise SystemExit(1)

    # Build ground truth from manifest
    vid_label = {r["video_id"]: r["label"] for r in all_rows}

    cnn_arr = np.array([cnn_video_scores[v] for v in common_vids], dtype=np.float64)
    lr_arr  = np.array([lr_video_scores[v]  for v in common_vids], dtype=np.float64)
    y       = np.array([vid_label[v]         for v in common_vids], dtype=int)

    n_real = int(np.sum(y == 0))
    n_fake = int(np.sum(y == 1))
    print(f"  Aligned {len(common_vids)} videos  ({n_real} real, {n_fake} fake)")
    print()

    # ------------------------------------------------------------------
    # 5. Individual model AUCs
    # ------------------------------------------------------------------
    print("STEP 5 — Individual model AUCs")
    cnn_auc      = roc_auc_score(y, cnn_arr)
    lr_auc       = roc_auc_score(y, lr_arr)
    hardcoded_auc= roc_auc_score(y, 0.65 * cnn_arr + 0.35 * lr_arr)

    print(f"  CNN alone (EfficientNet-B0) : AUC = {cnn_auc:.4f}")
    print(f"  LR  alone (handcrafted)     : AUC = {lr_auc:.4f}")
    print(f"  Hardcoded 65/35 blend       : AUC = {hardcoded_auc:.4f}")
    print()

    # ------------------------------------------------------------------
    # 6. Alpha sweep — find the optimal blend weight
    # ------------------------------------------------------------------
    print("STEP 6 — Alpha sweep  (combined = alpha×CNN + (1-alpha)×LR)")
    best_alpha, best_auc, all_alphas, all_aucs = sweep_alpha(cnn_arr, lr_arr, y)

    print(f"  Optimal alpha     : {best_alpha:.2f}  "
          f"(meaning {best_alpha*100:.0f}% CNN + {(1-best_alpha)*100:.0f}% LR)")
    print(f"  Combined AUC      : {best_auc:.4f}")
    print(f"  Gain vs hardcoded : {best_auc - hardcoded_auc:+.4f}")
    print(f"  Gain vs CNN alone : {best_auc - cnn_auc:+.4f}")
    print()

    # Print a compact AUC-vs-alpha table around the peak
    peak_i = int(np.argmax(all_aucs))
    lo = max(0, peak_i - 4); hi = min(len(all_alphas), peak_i + 5)
    print("  AUC at alphas near the optimum:")
    for i in range(lo, hi):
        marker = " <-- best" if i == peak_i else ""
        print(f"    alpha={all_alphas[i]:.2f}  AUC={all_aucs[i]:.4f}{marker}")
    print()

    # ------------------------------------------------------------------
    # 7. Cross-validate alpha stability  (5-fold GroupKFold)
    # ------------------------------------------------------------------
    print("STEP 7 — Cross-validate alpha stability  (5-fold StratifiedKFold)")
    print("         This runs CNN over all 778 frames — may take 2-4 minutes.")
    print()
    cv_results = cross_validate_alpha(all_rows, features_csv=FEATURES_CSV,
                                       n_folds=5, n_alpha_steps=101)

    cv_alpha_reliable = False
    cv_alpha_mean     = None

    if cv_results and len(cv_results["alphas"]) >= 3:
        alpha_arr = np.array(cv_results["alphas"])
        auc_arr   = np.array(cv_results["aucs"])
        cnn_arr_cv= np.array(cv_results["cnn_aucs"])
        lr_arr_cv = np.array(cv_results["lr_aucs"])

        cv_alpha_mean = float(np.mean(alpha_arr))
        cv_alpha_std  = float(np.std(alpha_arr))
        cv_alpha_min  = float(np.min(alpha_arr))
        cv_alpha_max  = float(np.max(alpha_arr))

        print()
        print("  CV ALPHA STABILITY RESULTS:")
        print(f"  {'Fold':<8}  {'alpha':>7}  {'combined AUC':>13}  {'CNN AUC':>8}  {'LR AUC':>7}")
        print("  " + "-" * 48)
        for fi, (a, auc_f, cauc, lauc) in enumerate(
                zip(cv_results["alphas"], cv_results["aucs"],
                    cv_results["cnn_aucs"], cv_results["lr_aucs"])):
            print(f"  {fi+1:<8}  {a:>7.2f}  {auc_f:>13.4f}  {cauc:>8.4f}  {lauc:>7.4f}")
        print("  " + "-" * 48)
        print(f"  {'mean':<8}  {cv_alpha_mean:>7.2f}  {np.mean(auc_arr):>13.4f}  "
              f"{np.mean(cnn_arr_cv):>8.4f}  {np.mean(lr_arr_cv):>7.4f}")
        print(f"  {'std':<8}  {cv_alpha_std:>7.3f}  {np.std(auc_arr):>13.4f}")
        print()

        # Reliability criterion: std < 0.20 means alpha does not jump more
        # than ±0.20 across folds — acceptable given the small dataset.
        # The val-set alpha (0.24) should be within 0.15 of the CV mean.
        alpha_consistent = cv_alpha_std < 0.20
        val_close_to_cv  = abs(best_alpha - cv_alpha_mean) < 0.20

        if alpha_consistent and val_close_to_cv:
            cv_alpha_reliable = True
            print(f"  [PASS] Alpha is STABLE  (std={cv_alpha_std:.3f} < 0.20,  "
                  f"val alpha {best_alpha:.2f} ~= CV mean {cv_alpha_mean:.2f})")
            print(f"         Using CV mean alpha = {cv_alpha_mean:.2f} in the saved bundle")
            print(f"         (more robust than the single val-split alpha {best_alpha:.2f})")
        else:
            print(f"  [FAIL] Alpha is UNSTABLE  (std={cv_alpha_std:.3f}, "
                  f"val={best_alpha:.2f} vs CV mean={cv_alpha_mean:.2f})")
            print("         Saving bundle with single-split alpha but flagging as unreliable.")
            print("         Do NOT update predict.py until this is resolved.")
            print("         Hint: train CNN + LR on more data to stabilise the blend weight.")
    else:
        n_folds_run = len(cv_results["alphas"]) if cv_results else 0
        print(f"  WARNING: Only {n_folds_run} folds completed — need ≥ 3 for reliability check.")
        print("    Cannot confirm alpha stability.  Do NOT wire into predict.py.")
    print()

    # ------------------------------------------------------------------
    # 8. Save stacking bundle
    # ------------------------------------------------------------------
    print("STEP 8 — Save stacking bundle")

    # Use CV mean alpha if reliable; otherwise fall back to single-split alpha.
    final_alpha = cv_alpha_mean if cv_alpha_reliable else best_alpha

    bundle = {
        "alpha"              : final_alpha,
        "alpha_reliable"     : cv_alpha_reliable,
        "alpha_single_split" : best_alpha,          # val-set alpha (for reference)
        "alpha_cv_mean"      : cv_alpha_mean,
        "alpha_cv_std"       : (float(np.std(cv_results["alphas"]))
                                 if cv_results and cv_results["alphas"] else None),
        "cnn_auc"            : round(cnn_auc,       4),
        "lr_auc"             : round(lr_auc,        4),
        "combined_auc"       : round(best_auc,      4),
        "hardcoded_auc"      : round(hardcoded_auc, 4),
        "n_val_videos"       : len(common_vids),
        "split_seed"         : _SPLIT_SEED,
    }
    with open(STACK_BUNDLE_PATH, "wb") as fh:
        pickle.dump(bundle, fh)
    print(f"  Saved -> {STACK_BUNDLE_PATH}")
    print(f"  alpha_reliable = {cv_alpha_reliable}  |  final_alpha = {final_alpha:.2f}")

    # ------------------------------------------------------------------
    # 8b. Register stacking bundle in model registry
    # ------------------------------------------------------------------
    print()
    print("STEP 8b — Register stacking bundle in artifacts/model_registry.json")
    try:
        import shutil
        import datetime
        from src.model_registry import ModelRegistry

        registry = ModelRegistry()
        _ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        _stack_id = f"stacked_{_ts}"

        os.makedirs("artifacts", exist_ok=True)
        _artifacts_stack = os.path.join("artifacts", f"stacking_bundle_{_ts}.pkl")
        shutil.copy2(STACK_BUNDLE_PATH, _artifacts_stack)

        registry.register({
            "model_id":      _stack_id,
            "model_type":    "stacked",
            "artifact_path": _artifacts_stack,
            "metrics": {
                "f1":        None,     # stacking bundle only has AUC — not F1
                "precision": None,
                "recall":    None,
                "auc":       round(best_auc, 4),
            },
            "notes": (
                f"CNN+LR stacked; alpha={final_alpha:.2f}; "
                f"alpha_reliable={cv_alpha_reliable}; "
                f"CNN_AUC={cnn_auc:.4f}; LR_AUC={lr_auc:.4f}"
            ),
            "comparable": False,  # AUC only, not F1 → excluded from get_best(metric='f1')
        })
        # Don't call select_best() here — the stacked model has no F1.
        # The LR model registered by ensemble.py remains active.
        print(f"  Registered stacking bundle: {_stack_id}  "
              f"(AUC={best_auc:.4f}, comparable=False — won't override active LR model)")
    except Exception as _reg_exc:
        print(f"  [WARN] Registry update failed: {_reg_exc}")
        print("         Stacking complete — registry is optional.")
    print()

    # ------------------------------------------------------------------
    # 9. Summary
    # ------------------------------------------------------------------
    print("=" * 65)
    print("STACKING ENSEMBLE COMPLETE")
    print("=" * 65)
    print()
    print("  COMPARISON TABLE  (video-level, held-out val set)")
    print(f"  {'Model':<32}  {'AUC':>6}")
    print("  " + "-" * 42)
    print(f"  {'LR alone (handcrafted features)':<32}  {lr_auc:.4f}")
    print(f"  {'CNN alone (EfficientNet-B0)':<32}  {cnn_auc:.4f}")
    print(f"  {'Hardcoded 65/35 blend':<32}  {hardcoded_auc:.4f}")
    print(f"  {'Learned blend (alpha={:.2f})':<32}  {best_auc:.4f}".format(best_alpha))
    print()
    print(f"  Single-split alpha : {best_alpha:.2f}  "
          f"({best_alpha*100:.0f}% CNN + {(1-best_alpha)*100:.0f}% LR)")
    if cv_alpha_mean is not None:
        print(f"  CV mean alpha      : {cv_alpha_mean:.2f}  (5-fold, more robust)")
    print(f"  Alpha reliable?    : {cv_alpha_reliable}")
    print(f"  Final alpha used   : {final_alpha:.2f}")
    print()
    if cv_alpha_reliable:
        print("  Next steps:")
        print("    Run  python wire_stacking.py  (or manually update predict.py /")
        print("    app.py) to load stacking_bundle.pkl and use the saved alpha.")
    else:
        print("  [!] Alpha NOT reliable -- do NOT update predict.py / app.py yet.")
        print("      Options to fix:")
        print("      1.  Run inspect_dataset.py with higher TARGET_PER_CLASS to")
        print("          get more videos (the main bottleneck is face detection).")
        print("      2.  Try MediaPipe or RetinaFace instead of Haar cascade for")
        print("          better face detection recall -- more videos extracted.")
    print("=" * 65)
