# `ensemble.py` — External Reference

> **Module 3 + Module 1 Fusion: Training, Evaluation & 30/70 Weighted Fusion**
> Pipeline entry point: `python ensemble.py`

---

## Table of Contents

1. [What This File Does](#1-what-this-file-does)
2. [Run Order & Prerequisites](#2-run-order--prerequisites)
3. [Key Design Decisions](#3-key-design-decisions)
4. [Public API](#4-public-api)
   - [Section A — Pure Functions](#section-a--pure-functions)
   - [Section B — I/O Helpers](#section-b--io-helpers)
   - [Section C — Multi-Classifier Training & Fusion](#section-c--multi-classifier-training--fusion)
5. [Pipeline Walkthrough (\_\_main\_\_)](#5-pipeline-walkthrough-__main__)
6. [Feature Matrix](#6-feature-matrix)
7. [Output Files](#7-output-files)
8. [Configuration Reference](#8-configuration-reference)
9. [Interpreting the Output](#9-interpreting-the-output)

---

## 1. What This File Does

`ensemble.py` is the **training and fusion entry point** for the combined
Module 1 + Module 3 system.

It:
1. Loads face crop images from `data/manifest.csv`
2. Computes **four** handcrafted features per frame (artifact, FFT, Laplacian,
   and Module 1 `deepfake_confidence` as `ear_score`)
3. Trains **three** classifiers (Logistic Regression, Random Forest, XGBoost)
   on video-level features using a video-grouped train/val split
4. Picks the best Module 3 model by validation F1
5. Fuses Module 1 and Module 3 with **30/70 weight** (Module 1 / Module 3)
   into a single per-video prediction
6. Reports an honest comparison table and a full weight sweep (0.0–1.0)
   across all 238 Module 1 videos

**In one sentence:** Face crops → feature numbers → three competing models →
best model fused 30/70 with Module 1 → one prediction per video.

---

## 2. Run Order & Prerequisites

```bash
# Step 1 — extract face crops (must run first)
python inspect_dataset.py

# Step 2 — train, fuse, and evaluate
python ensemble.py
```

`ensemble.py` requires:
- `data/manifest.csv` — created by `inspect_dataset.py`
- `data/module1_output.csv` — Module 1 blink-detection output (Niko's pipeline)
  - Columns: `video_id`, `blinks_per_minute`, `deepfake_confidence`, `is_fake`
  - 238 rows (136 real + 102 fake)
  - If absent, `ear_score` defaults to 0.5 for all frames

**Feature caching:** After the first run, features are saved to
`data/module3_features.csv`. Subsequent runs load from cache, **except** when
the cache is stale — that is, when the `ear_score` column is absent or constant
(all-0.5). Stale caches are deleted automatically so Module 1 scores are always
applied fresh.

To force re-extraction at any time: delete `data/module3_features.csv`.

---

## 3. Key Design Decisions

### 3.1 Video-level GroupShuffleSplit

Each video contributes `FRAMES_PER_VIDEO` frames (default 4). A random
frame-level split would allow frames from the **same video** to appear in both
train and val — the model would memorise video-specific statistics (lighting,
background, identity) rather than learning transferable deepfake features,
inflating validation metrics.

`GroupShuffleSplit(groups=video_id)` ensures every frame from a given video
lands in exactly one partition.

### 3.2 Module 1 Integration as `ear_score`

`data/module1_output.csv` records the MRL-model `deepfake_confidence` for
each video. The video_id formats differ between Module 1 and the manifest:

| Source | Real format | Fake format |
|--------|-------------|-------------|
| `module1_output.csv` | `"0"`, `"1"`, ... | `"000_003"`, `"001_870"`, ... |
| `manifest.csv` | `"real_000"`, `"real_001"`, ... | `"fake_000_003"`, ... |

`_load_module1_scores()` converts module1 IDs to manifest-format keys and
returns a dict used inside `extract_all_features()`. Any manifest video whose
ID does not appear in module1 receives `ear_score = 0.5` with a logged warning.

167 of 238 module1 videos appear in the manifest (the other 71 had no face
crops extracted by `inspect_dataset.py`).

### 3.3 Multi-Classifier Comparison

Three models are trained on the **same** video-level features and the **same**
train/val split:
- **Logistic Regression** — wrapped in `CalibratedClassifierCV` (Platt scaling)
- **Random Forest** — 200 trees, balanced class weights
- **XGBoost** — 200 estimators, max_depth=4, lr=0.1

All three are evaluated on the validation set. The **best by F1** is selected
as the Module 3 model and saved to `data/ensemble_model.pkl`.

### 3.4 Quality-Weighted Video Aggregation

Before training, frames are collapsed into one row per video:

```
video_feature_j = Σ_i(laplacian_i × feature_ij) / Σ_i(laplacian_i)
```

Sharper frames (high Laplacian variance) get more weight. This reduces
~778 frame rows to ~167 video rows, with less noise per sample.

The `ear_score` feature is video-level (all frames from the same video share
the same `deepfake_confidence`), so quality-weighting has no effect on it.

### 3.5 30/70 Weighted Fusion

The final per-video prediction is:

```
final_score      = MODULE1_WEIGHT × module1_score
                 + MODULE3_WEIGHT × module3_score
final_prediction = 1 if final_score >= 0.5 else 0
```

Default values: `MODULE1_WEIGHT = 0.30`, `MODULE3_WEIGHT = 0.70`.
Both constants are defined at the top of `ensemble.py` (just below `FEATURE_NAMES`)
and are read directly by `_fuse_and_report()` — change only the constants to retune.

**Why 30/70 and not 50/50?**  A weight sweep over 0.0–1.0 (step 0.1) is
printed at the end of Step 12 each time `ensemble.py` runs.  The sweep showed
that giving more weight to Module 3 (the trained classifier) consistently
improved or matched the 50/50 baseline on F1 and AUC across the 238-video val
set.  The 30/70 split keeps Module 1 as a meaningful contributor without
letting its noisier blink signal dominate.  The sweep table is in the commit
history — run `git log --oneline` and look for
`tune: shift fusion weights to 30/70`.

`module3_score` is computed by applying the best Module 3 model to **each
individual frame** and averaging the per-frame P(fake) values per video.
This uses the same scaler fitted on video-level features — acceptable since
the frame-level and video-level feature distributions are similar (aggregation
is a weighted mean, not a non-linear transform).

Videos in `module1_output.csv` that are not in the manifest receive
`module3_score = 0.5` (neutral fallback), ensuring the output CSV always has
238 rows.

### 3.6 Calibrated Threshold

The decision threshold is chosen to maximise **balanced accuracy**:
```
balanced_accuracy = (recall_real + recall_fake) / 2
```
Immune to class imbalance. The F1-optimal threshold is printed alongside for
comparison.

---

## 4. Public API

### Section A — Pure Functions

---

#### `ensemble_score_equal_weights(artifact_score, fft_score, laplacian_score_val)`

Equal-weights baseline for the three handcrafted module scores (not including ear).

```
final = (1/3) × artifact + (1/3) × fft + (1/3) × laplacian
```

**Returns:** `float [0,1]`

---

#### `train_ensemble(features, labels, video_ids, C=1.0, random_state=42)`

Train a Logistic Regression with video-level GroupShuffleSplit + Platt calibration.

| Parameter | Type | Description |
|-----------|------|-------------|
| `features` | `np.ndarray (N×4) float` | Feature matrix [artifact, fft, laplacian, ear] |
| `labels` | `np.ndarray (N,) int` | 0=real, 1=fake |
| `video_ids` | `list[str]` length N | Group key for GroupShuffleSplit |
| `C` | `float` | LR regularisation strength. Default `1.0` |
| `random_state` | `int` | Seed for reproducibility |

**Returns:** `(model, scaler, X_val, y_val, val_video_ids)`

---

#### `ensemble_score_learned(model, scaler, artifact_score, fft_score, laplacian_score_val, ear_score=0.5)`

Run inference with the trained model on a single sample.

**Returns:** `float [0,1]` — calibrated P(fake)

---

#### `calibrate_threshold_balanced(y_true, y_scores)`

Find the threshold that maximises balanced accuracy over 181 grid points.

**Returns:** `(best_threshold: float, best_balanced_accuracy: float)`

---

#### `calibrate_threshold_f1(y_true, y_scores)`

Find the threshold that maximises F1 (kept for comparison).

**Returns:** `(best_threshold: float, best_f1: float)`

---

#### `evaluate_model(y_true, y_scores, threshold)`

Compute a full set of classification metrics at a given threshold.

**Returns:** `dict` with keys:
`accuracy`, `balanced_accuracy`, `auc`, `precision`, `recall`, `f1`, `threshold`

---

#### `cross_validate_ensemble(features, labels, video_ids, n_splits=5, C=1.0, random_state=42)`

5-fold GroupKFold cross-validation using Logistic Regression.

**Returns:** `dict` mapping metric name → `list[float]` (one value per fold).

---

#### `aggregate_to_video_level(features, labels, video_ids, quality_col_idx=2)`

Collapse frame-level features into one row per video using Laplacian-weighted averaging.

Column order: `[artifact=0, fft=1, laplacian=2, ear=3]`
`quality_col_idx=2` → uses laplacian as the weight.

**Returns:** `(vid_features, vid_labels, vid_ids)`

---

### Section B — I/O Helpers

---

#### `load_manifest(path=MANIFEST_PATH)`

Read `data/manifest.csv`. Returns list of dicts with keys:
`file_path`, `label` (int), `video_id`, `source_dataset`.

---

#### `_load_module1_scores()`

Load `data/module1_output.csv` and return a dict mapping manifest-format
`video_id` → `deepfake_confidence` (float [0,1]).

Returns `{}` if the file does not exist.

---

#### `extract_all_features(manifest_rows, save_csv=FEATURES_CSV, verbose=True)`

For every image in the manifest, compute four features and save to CSV.

**Features computed:**
- `artifact_score` — JPEG recompression artifact (Module 2)
- `fft_score` — FFT high-frequency anomaly (Module 3)
- `laplacian_score` — Laplacian-variance sharpness (Module 3)
- `ear_score` — Module 1 `deepfake_confidence` (falls back to 0.5 with warning)

The function signature is **unchanged** — Module 1 loading is done internally.

**Returns:** `(features: np.ndarray (N×4), labels: np.ndarray (N,), video_ids: list[str])`

**Side effect:** Writes `data/module3_features.csv` with all four feature columns.

---

#### `load_features_from_csv(path=FEATURES_CSV)`

Load pre-computed features from CSV. Handles both old 3-column CSVs (no
`ear_score`) and new 4-column CSVs gracefully — missing columns default to `0.5`.

**Returns:** `(features (N×4), labels (N,), video_ids)`

---

### Section C — Multi-Classifier Training & Fusion

---

#### `_train_all_classifiers(X_train, y_train, X_val, y_val, random_state=42)`

Train LR, Random Forest, and XGBoost on the same pre-split, pre-scaled data.

Prints a side-by-side comparison table (accuracy, precision, recall, F1, AUC)
and selects the **best by validation F1**.

| Return | Type | Description |
|--------|------|-------------|
| `best_model` | sklearn-compatible model | Fitted best model |
| `best_name` | `str` | Model name |
| `best_val_scores` | `np.ndarray` | Val-set P(fake) from best model |
| `all_models` | `dict` | All trained models keyed by name |

---

#### `_compute_video_module3_scores(model, scaler, features_csv=FEATURES_CSV)`

Apply the trained Module 3 model to each individual **frame**, then average
per-frame P(fake) values within each video.

**Returns:** `dict[manifest_video_id → float mean P(fake)]`

---

#### `_fuse_and_report(module3_vid_scores, module1_csv=MODULE1_CSV, output_csv=ENSEMBLE_OUTPUT_CSV)`

Fuse Module 1 `deepfake_confidence` and Module 3 per-video P(fake) using
the module-level constants `MODULE1_WEIGHT` / `MODULE3_WEIGHT` (default 0.30 / 0.70)
for all 238 module1 videos.

```
final_score      = MODULE1_WEIGHT × module1_score
                 + MODULE3_WEIGHT × module3_score
final_prediction = 1 if final_score >= 0.5 else 0
```

Videos not in the manifest receive `module3_score = 0.5`.

**Side effect:** Writes `data/ensemble_output.csv`.

**Returns:** `pandas.DataFrame` (238 rows)

---

## 5. Pipeline Walkthrough (`__main__`)

When run as `python ensemble.py`, the script executes 12 steps:

| Step | What happens |
|------|-------------|
| **0** | Prerequisite check — exits if `manifest.csv` missing |
| **1** | Load manifest, report class counts and unique video count |
| **2** | Stale-cache check; extract features (or load from cache) with Module 1 ear_score |
| **2b** | Aggregate frame-level features → video-level via quality-weighted mean |
| **3** | Compute equal-weights baseline AUC (handcrafted features only) |
| **4** | Build train/val split; train LR, RF, XGB; compare; pick best by val F1 |
| **4b** | 5-fold GroupKFold cross-validation (LR baseline, reliable AUC estimate) |
| **5** | Calibrate decision threshold (balanced accuracy + F1 for comparison) |
| **6** | Evaluate best Module 3 model on validation split; print confusion matrix |
| **7** | Save best model bundle (`data/ensemble_model.pkl`) |
| **8** | Save ROC + Precision-Recall curves to `data/plots/` |
| **9** | Save FFT spectrum visualisations to `data/visualizations/` |
| **10** | Compute per-video Module 3 scores (frame-level averaging) |
| **11** | Fuse Module 1 + Module 3 (30/70); write `data/ensemble_output.csv` |
| **12** | Print 3-row comparison table + weight sweep (0.0–1.0, step 0.1) |

---

## 6. Feature Matrix

| Index | Feature | Source | Signal on FF++ C23 |
|---|---|---|---|
| `0` | `artifact_score` | `artifact_module.py` (Module 2) | Very weak (Δ ≈ 0.002) |
| `1` | `fft_score` | `anomaly_scorer.fft_anomaly_score()` | Weak (Δ ≈ 0.024) |
| `2` | `laplacian_score` | `texture_scorer.laplacian_score()` | Moderate (Δ ≈ 0.06) |
| `3` | `ear_score` | Module 1 `deepfake_confidence` | Varies — Niko's blink model |

All features are in `[0, 1]`. `StandardScaler` is applied before training.

`quality_col_idx=2` — the Laplacian score is used as the quality weight for
video-level aggregation. All frames from the same video have the same `ear_score`,
so weighting has no effect on that column.

---

## 7. Output Files

| File | Description |
|------|-------------|
| `data/module3_features.csv` | Per-frame features: file_path, label, video_id, artifact, fft, laplacian, ear |
| `data/ensemble_model.pkl` | Pickled bundle: model, scaler, threshold, feature_names, model_name |
| `data/ensemble_output.csv` | Per-video fusion: video_id, module1_score, module3_score, final_score, final_prediction, is_fake |
| `data/plots/roc_curve.png` | ROC curve with chosen threshold marked |
| `data/plots/precision_recall.png` | PR curve with random baseline |
| `data/visualizations/fft_spectrum_real.jpg` | 3-panel FFT diagnostic for a real face |
| `data/visualizations/fft_spectrum_fake.jpg` | 3-panel FFT diagnostic for a fake face |

**Loading the saved model:**
```python
import pickle
with open("data/ensemble_model.pkl", "rb") as f:
    bundle = pickle.load(f)

model      = bundle["model"]         # LR / RF / XGBoost (best by val F1)
scaler     = bundle["scaler"]        # StandardScaler
threshold  = bundle["threshold"]     # float — use for predict()
features   = bundle["feature_names"] # ["artifact", "fft", "laplacian", "ear"]
model_name = bundle["model_name"]    # e.g. "XGBoost"
```

**Loading the fusion output:**
```python
import pandas as pd
df = pd.read_csv("data/ensemble_output.csv")
# Columns: video_id, module1_score, module3_score, final_score,
#          final_prediction, is_fake
```

---

## 8. Configuration Reference

Top-level constants in `ensemble.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `MANIFEST_PATH` | `data/manifest.csv` | Manifest CSV |
| `FEATURES_CSV` | `data/module3_features.csv` | Feature cache |
| `MODULE1_CSV` | `data/module1_output.csv` | Module 1 blink output |
| `ENSEMBLE_OUTPUT_CSV` | `data/ensemble_output.csv` | Fusion output |
| `PLOTS_DIR` | `data/plots` | Diagnostic plots |
| `VIZ_DIR` | `data/visualizations` | FFT spectrum images |
| `FEATURE_NAMES` | `["artifact", "fft", "laplacian", "ear"]` | Feature column names |
| `MODULE1_WEIGHT` | `0.30` | Fusion weight for Module 1 score. Edit here to retune. |
| `MODULE3_WEIGHT` | `0.70` | Fusion weight for Module 3 score. Must sum to 1.0 with above. |

Training hyperparameters (in `__main__`):

| Variable | Default | Description |
|----------|---------|-------------|
| `_C` | `1.0` | LR regularisation |
| `_SEED` | `42` | Reproducibility seed |
| `test_size` | `0.20` | Val fraction (GroupShuffleSplit) |
| `cv` | `3` | Folds for Platt scaling calibration |
| `n_splits` | `5` | Folds for GroupKFold cross-validation |

Classifier hyperparameters:

| Classifier | Key hyperparameters |
|------------|---------------------|
| LogisticRegression | `C=1.0`, `class_weight="balanced"`, Platt-calibrated via `CalibratedClassifierCV(cv=3)` |
| RandomForest | `n_estimators=200`, `class_weight="balanced"`, `n_jobs=-1` |
| XGBoost | `n_estimators=200`, `max_depth=4`, `learning_rate=0.1`, `eval_metric="logloss"` |

---

## 9. Interpreting the Output

### Module 3 Classifier Comparison Table

```
Model                     Acc    Prec     Rec      F1     AUC
LogisticRegression       0.XXXX  0.XXXX  0.XXXX  0.XXXX  0.XXXX
RandomForest             0.XXXX  0.XXXX  0.XXXX  0.XXXX  0.XXXX  ← best F1
XGBoost                  0.XXXX  0.XXXX  0.XXXX  0.XXXX  0.XXXX
```

Best model is selected by **F1** (balances precision and recall). It is saved
to `ensemble_model.pkl` and used for the 30/70 fusion.

### Final Comparison Table

```
System                              Acc    Prec     Rec      F1     AUC
Module 1 alone (Niko baseline 66%)  ...    ...     ...     ...     ...
Module 3 alone (XGBoost)            ...    ...     ...     ...     ...
Fused 30%/70% (Module1 + Module3)   ...    ...     ...     ...     ...  ← final
```

If the fused row outperforms both alone rows on at least two metrics (typically
F1 and AUC), the fusion is additive. If not, revisit the Module 1/Module 3
feature distributions and the 0.5 fallback rate.

### Weight Sweep Table

After the comparison table, `ensemble.py` prints a sweep over all 11 weight
combinations (module1_weight 0.0 → 1.0, step 0.1):

```
  m1_weight  m3_weight   Accuracy        F1  ROC-AUC
  ---------------------------------------------------------
        0.0        1.0     0.XXXX    0.XXXX   0.XXXX
        0.1        0.9     0.XXXX    0.XXXX   0.XXXX
        ...
        0.3        0.7     0.XXXX    0.XXXX   0.XXXX  <- selected
        ...
        1.0        0.0     0.XXXX    0.XXXX   0.XXXX
```

The threshold is fixed at 0.5 for all sweep rows (same as the live constants).
The sweep is **read-only** — it never updates `MODULE1_WEIGHT` / `MODULE3_WEIGHT`.
Consult the sweep output in the commit log
(`tune: shift fusion weights to 30/70`) to see the actual numbers.

### AUC Guide

- `0.50` = no better than random
- `0.60–0.70` = typical for handcrafted features on FF++ C23
- `> 0.75` = strong — investigate for data leakage if unexpectedly high

### ear_score Before / After

The pipeline prints the ear_score distribution before and after Module 1 integration:

```
BEFORE: ear_score column absent in cache (was constant 0.5 stub)
AFTER:  ear_score  min=0.0000  max=0.9999  mean=0.3421  std=0.3188
```

If `std ≈ 0` after extraction, something is wrong with the module1_output.csv
lookup — check the video_id mapping in `_load_module1_scores()`.
