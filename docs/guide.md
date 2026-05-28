# Group 8 AISC — Deepfake Detection System
## User Guide

> **Who this is for:** Someone new to the project who wants to understand
> what it does, how to run it, and where to find things.

---

## Table of Contents

1. [What This System Does](#1-what-this-system-does)
2. [How to Think About It (Intuition)](#2-how-to-think-about-it-intuition)
3. [Quick Start](#3-quick-start)
4. [Project Structure](#4-project-structure)
5. [The Three Detection Modules](#5-the-three-detection-modules)
6. [Step-by-Step Pipeline](#6-step-by-step-pipeline)
7. [Understanding the Outputs](#7-understanding-the-outputs)
8. [Changing the Dataset](#8-changing-the-dataset)
9. [Adding a New Feature](#9-adding-a-new-feature)
10. [Why Results Are Limited on FF++ C23](#10-why-results-are-limited-on-ff-c23)
11. [Glossary](#11-glossary)

---

## 1. What This System Does

This project detects **deepfake face videos** — videos where a real person's
face has been replaced by an AI-generated one.

The dataset is **FaceForensics++ C23**: a benchmark of 1,000 real YouTube
face videos and 3,000 fake videos produced by three different deepfake methods
(Deepfakes autoencoder, Face2Face, FaceSwap). "C23" refers to a specific H.264
compression level that makes detection harder by smoothing artifacts.

The system:
1. **Extracts** face crop images from the videos
2. **Computes** numerical "suspicion scores" from each face crop
3. **Trains** a logistic regression to combine those scores
4. **Reports** how well it separates real from fake faces

It is not a finished product — it is a learning project exploring which
handcrafted signal-processing features can detect deepfakes.

---

## 2. How to Think About It (Intuition)

Imagine you're trying to spot a forgery without a magnifying glass. You'd look for:

- **Unnaturally smooth skin** — real faces have pores and fine texture; fakes
  produced by autoencoders are slightly over-smoothed because their decoder
  applies blur-like filters.
- **Missing high-frequency texture** — in the "frequency domain" (think of it
  as a musical spectrum for images), real faces have energy across all
  frequencies. Smooth fakes have less energy at the high end.
- **Inconsistent sharpness** — compression artifacts look different on
  repeatedly-compressed fake frames versus original camera footage.

This system measures exactly those three things and combines them.

---

## 3. Quick Start

### Prerequisites

```powershell
# From the project root, activate the virtual environment
.\.venv\Scripts\activate

# Install dependencies (first time only)
pip install -r requirements.txt
```

### Run the pipeline

```powershell
# Step 1 — Extract face crops from the FF++ videos (≈5-20 min first time)
python inspect_dataset.py

# Step 2 — Extract features, train model, evaluate (≈2-4 min with cache)
python ensemble.py
```

That's it. After Step 2, you'll see AUC and balanced accuracy printed to the
terminal, and diagnostic plots saved to `data/plots/`.

### Run the CNN (optional)

If PyTorch is installed and `data/cnn_model.pth` exists:

```powershell
python cnn_detector.py
```

---

## 4. Project Structure

```
Group-8-AISC/
│
├── inspect_dataset.py    ← Step 1: extract face crops from FF++ videos
├── ensemble.py           ← Step 2: features → train → evaluate
├── cnn_detector.py       ← Optional: EfficientNet-B0 transfer learning
├── artifact_module.py    ← Module 2: JPEG recompression artifact scorer
├── main.py               ← Module 1 demo (standalone, not part of pipeline)
│
├── src/
│   ├── preprocessing/    ← Module 1 helpers
│   │   ├── face_detector.py     Haar cascade face cropper
│   │   ├── frame_extracter.py   Frame generator from video
│   │   └── video_loader.py      cv2.VideoCapture wrapper
│   │
│   ├── freq_analysis/    ← Module 3: FFT + texture
│   │   ├── utils.py             resize, load, normalise
│   │   ├── fft_extractor.py     pure FFT math (no I/O)
│   │   ├── anomaly_scorer.py    FFT scoring functions
│   │   ├── texture_scorer.py    Laplacian sharpness score
│   │   └── frequency_analyzer.py  batch scoring + visualisation
│   │
│   ├── quality_scorer.py    Frame quality weighting
│   ├── rppg_scorer.py       rPPG pulse coherence score (future)
│   ├── temporal_scorer.py   Optical-flow temporal consistency (future)
│   └── cnn_runner.py        EfficientNet-B0 inference wrapper
│
├── data/
│   ├── FaceForensics++_C23/   Source videos (read-only)
│   ├── real/frames/           Extracted real face crops (224×224 JPEGs)
│   ├── fake/frames/           Extracted fake face crops
│   ├── manifest.csv           Image list with labels
│   ├── module3_features.csv   Per-image feature scores (cache)
│   ├── plots/                 ROC + PR curves
│   └── visualizations/        FFT spectrum side-by-sides
│
└── docs/
    ├── guide.md              ← You are here
    └── api/
        ├── freq_analysis_api.md   API reference: src/freq_analysis/
        └── ensemble_reference.md  API reference: ensemble.py
```

---

## 5. The Three Detection Modules

### Module 1 — EAR Blink Detection (preprocessing helpers done; score stubbed)

**What it does:** Measures eye blink rate and naturalness using Eye Aspect Ratio (EAR).
Real people blink naturally; early deepfakes had unnatural eye movement.

**Status:** The preprocessing helpers (`face_detector.py`, `frame_extracter.py`,
`video_loader.py`) are complete. The actual EAR score computation is stubbed at
`0.5` (a constant neutral value) in `ensemble.py` until Module 1 is fully integrated.
The logistic regression correctly assigns this zero weight — it's expected behavior.

**Entry point:** `main.py` (standalone demo)

---

### Module 2 — JPEG Artifact Score (complete)

**What it does:** Re-compresses a face crop as JPEG and measures how much the
pixel values change. Original camera footage and deepfake frames respond
differently to re-compression because they have different compression histories.

**Status:** Complete. Signal on FF++ C23 is very weak (Δ ≈ 0.002).

**Entry point:** `artifact_module.py` — `get_artifact_score_for_frame(img)`

---

### Module 3 — FFT Frequency Anomaly + Laplacian Texture (complete)

**What it does:** Two complementary measurements of "smoothness":

1. **FFT anomaly score** — analyses the image in the frequency domain.
   Real photographs follow a "1/f" power law (lots of low-frequency energy,
   less at high frequencies, but in a predictable pattern). Deepfake autoencoders
   apply blur-like upsampling, making fakes smoother → less high-frequency energy.
   Measures: radial slope, HF energy ratio, spectral entropy, peak excess.

2. **Laplacian texture score** — measures fine texture using the Laplacian
   operator (detects rapid pixel changes). Sharp, textured real faces have high
   Laplacian variance; smooth fakes have less.

**Status:** Complete. Laplacian is the strongest feature (Δ ≈ 0.06);
FFT is weaker but adds independent information.

**Entry points:** `src/freq_analysis/anomaly_scorer.py` and `texture_scorer.py`

---

## 6. Step-by-Step Pipeline

### Step 1: `inspect_dataset.py`

**What it does:**
- Walks the `data/FaceForensics++_C23/original/` and `.../Deepfakes/` directories
- Samples `FRAMES_PER_VIDEO` frames from each video (default: 4)
- Detects the face in each frame using Haar cascade
- Quality-filters frames (brightness, face size thresholds)
- Saves 224×224 JPEG face crops to `data/real/frames/` and `data/fake/frames/`
- Writes `data/manifest.csv` listing every saved crop

**Key settings (at the top of `inspect_dataset.py`):**
```python
TARGET_PER_CLASS = 200   # how many face crops to collect per class
FRAMES_PER_VIDEO = 4     # frames sampled from each video
MIN_BRIGHTNESS   = 40    # reject dark frames
MIN_FACE_FRAC    = 0.04  # reject if face < 4% of frame
```

**Output:** `data/manifest.csv` with columns:
`file_path`, `label` (0=real, 1=fake), `video_id`, `source_dataset`

---

### Step 2: `ensemble.py`

**What it does:**
1. Loads `manifest.csv`
2. Computes three scores per image (artifact, FFT, Laplacian)
3. Saves scores to `data/module3_features.csv` (cached for later runs)
4. Aggregates frames → videos using quality-weighted averaging
5. Trains a logistic regression with a video-level train/val split
6. Calibrates a decision threshold
7. Reports AUC, balanced accuracy, confusion matrix
8. Saves the model, curves, and visualisations

**Expected runtime:** ~2-4 minutes with cached features, ~4-8 minutes without.

---

## 7. Understanding the Outputs

### Terminal output

```
MODULE 3 — ENSEMBLE TRAINING PIPELINE
...
STEP 4 — Train LogisticRegression
  Val set      : 40 videos (20 real, 20 fake)
  Coefficients (|w| normalised):
    artifact     = 0.09  ###
    fft          = 0.27  ########
    laplacian    = 0.64  ###################

=======================================================
RESULTS — balanced-accuracy threshold
=======================================================
  Threshold         : 0.4500
  Accuracy          : 0.6000
  Balanced accuracy : 0.6100
  AUC               : 0.6320
  Precision (fake)  : 0.6000
  Recall    (fake)  : 0.6500
  F1                : 0.6240
=======================================================

  CONFUSION MATRIX
                Predicted
                REAL   FAKE
  Actual  REAL  [  14     6 ]   <- 6 real faces wrongly flagged
          FAKE  [   7    13 ]   <- 7 fakes missed
```

### What to look at first: AUC

**AUC (Area Under the ROC Curve)** is the most reliable single number.
- `0.50` = the model is guessing randomly
- `0.63` = it's doing better than chance — 63% of the time, a randomly chosen
  fake scores higher than a randomly chosen real
- `0.70+` = strong separation for handcrafted features on FF++ C23

### Balanced Accuracy

Mean of recall-for-real and recall-for-fake. Better than raw accuracy when
the classes are imbalanced.

### Feature Coefficients

Shows which feature the model relies on most. A near-zero coefficient means
the feature carries no signal the model can use.

### The Plots

| File | What it shows |
|------|--------------|
| `data/plots/roc_curve.png` | True positive rate vs false positive rate at every threshold. The red star = chosen threshold. Area under the curve = AUC. |
| `data/plots/precision_recall.png` | How precision and recall trade off. The grey baseline = what random guessing achieves. |
| `data/visualizations/fft_spectrum_real.jpg` | 3-panel: face photo / log FFT / high-frequency only — for a real face |
| `data/visualizations/fft_spectrum_fake.jpg` | Same panels — for a fake face. Compare: the fake usually has a dimmer high-frequency panel. |

---

## 8. Changing the Dataset

### Use a different manipulation type

In `inspect_dataset.py`, change `FAKE_SRC`:

```python
# Current default:
FAKE_SRC = "data/FaceForensics++_C23/Deepfakes"

# Alternatives:
FAKE_SRC = "data/FaceForensics++_C23/Face2Face"
FAKE_SRC = "data/FaceForensics++_C23/FaceSwap"
FAKE_SRC = "data/FaceForensics++_C23/NeuralTextures"
```

Delete `data/manifest.csv` and `data/module3_features.csv` after changing,
then re-run both steps.

### Use more data

Increase `TARGET_PER_CLASS` and/or `FRAMES_PER_VIDEO` in `inspect_dataset.py`.
More data → more reliable metrics but longer extraction time.

---

## 9. Adding a New Feature

Follow these steps (from `CLAUDE.md` — the project workflow):

1. **Explain** the concept first. If unsure, ask `@concept-explainer`.
2. **Build incrementally.** One function at a time in `src/freq_analysis/`.
3. **Add the feature** to `extract_all_features()` in `ensemble.py`.
   Add the column name to `FEATURE_NAMES`.
4. **Delete** `data/module3_features.csv` to force re-extraction.
5. **Code review** via `@ml-reviewer` — check for data leakage.
6. **Measure** via `@metric-checker` — run `ensemble.py` and compare AUC
   against the baseline before declaring improvement.
7. **Sanity-check** via `@dataset-inspector` if metrics moved > 0.05 AUC.

**Hard rules:**
- Never claim improvement without `@metric-checker` numbers
- Never skip the code review step

---

## 10. Why Results Are Limited on FF++ C23

**FF++ C23 is deliberately hard.** The H.264 codec at quality level 23 applies
heavy compression that blurs the fine-detail artifacts that our features detect:

- JPEG and FFT scores measure high-frequency anomalies → compression smooths these out
- Laplacian measures texture sharpness → compression reduces sharpness for both real and fake

Expected AUC for handcrafted features on C23: **0.50–0.70**.

For comparison, the same features on the uncompressed C0 version typically reach
AUC 0.80–0.90. If you see an unexpectedly high AUC on C23 (> 0.75), check for
data leakage (frames from the same video in both train and val).

---

## 11. Evaluation

Batch-evaluate labelled videos through the same pipeline as `predict.py`:

```powershell
# One real + one fake FF++ clip (adjust paths to videos on disk)
python scripts/eval_videos.py \
  --real data/FaceForensics++_C23/original/000.mp4 \
  --fake data/FaceForensics++_C23/Deepfakes/000_003.mp4 \
  --frames 8

# Or a manifest file (path<TAB>label per line)
python scripts/eval_videos.py --manifest data/eval_manifest.txt
```

Results are written to `data/eval_results.csv` with per-video P(fake), verdict,
ear_score, and module availability flags.

### Module 1 (MRL blink) checkpoint

Live inference and training integration expect the MobileNetV2 checkpoint at
**`models/best_model.pth`**. To obtain it:

```powershell
# Train on the MRL eye dataset (requires torch + mediapipe)
python -m src.mrl.train

# Or copy a trained checkpoint:
# cp path/to/your/best_model.pth models/best_model.pth
```

Regenerate `data/module1_output.csv` for ensemble training:

```powershell
# 1. Organise extracted frames as subdirs named by video id (0, 000_003, …)
python run_inference.py --video-dir data/processed/frames --output-dir data/results_real
python run_inference.py --video-dir data/processed/frames_fake --output-dir data/results_fake

# 2. Summarise to module1_output.csv (maps real_000 / fake_000_003 ids)
python -m src.mrl.score
```

Then re-run `python ensemble.py` so `ear_score` in `module3_features.csv` is
non-constant.

---

## 12. Glossary

| Term | Plain English |
|------|--------------|
| **AUC** | Area Under the ROC Curve. 1.0 = perfect. 0.5 = random. |
| **Balanced accuracy** | Average recall across both classes. Better than accuracy when one class is more common. |
| **BGR** | Blue-Green-Red — the colour order OpenCV uses (opposite of the RGB you might expect). |
| **Calibration (threshold)** | Choosing the probability cutoff above which we predict "fake". |
| **Deepfakes autoencoder** | A type of face swap where an encoder compresses the source face and a decoder reconstructs it onto the target. Produces smoother faces than originals. |
| **EAR** | Eye Aspect Ratio — ratio of eye height to width. Low EAR = eye closed (blink). |
| **FFT** | Fast Fourier Transform — decomposes an image into sine waves at different spatial frequencies. |
| **GroupShuffleSplit** | A scikit-learn splitter that keeps all frames from one video in the same split (train or val), preventing identity leakage. |
| **Haar cascade** | A classic, fast face detector bundled with OpenCV. Works well for frontal faces. |
| **Laplacian** | An operator that measures how fast pixel values change. High Laplacian variance = lots of fine texture. |
| **Log-magnitude spectrum** | The result of taking the FFT of an image, measuring the magnitude, and taking log(1 + magnitude). The log compresses the huge dynamic range. |
| **Platt scaling** | Fitting a sigmoid curve on top of classifier scores so that the output probabilities are truly calibrated. |
| **Radial power spectrum** | The FFT energy averaged in concentric rings from the DC centre. Summarises how energy varies with spatial frequency. |
| **Spectral entropy** | How evenly spread the energy is across frequency bands. Low = concentrated in low frequencies (like a smooth face). |
| **StandardScaler** | Subtracts the mean and divides by the standard deviation so all features are on the same numerical scale before logistic regression. |
| **video_id** | An identifier like `real_042` or `fake_017` that links all frames extracted from the same source video. Used to prevent the same video from appearing in both train and val. |
