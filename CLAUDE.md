# CLAUDE.md — Group 8 AISC: Deepfake Detection System

## How to work with me

I'm learning ML as I build this — I'm not an experienced ML engineer.
When working on this project:

- Before writing code, explain in plain English what we're about to build
  and why. Use analogies where helpful.
- Build incrementally. One feature at a time. Walk me through new concepts
  (FFT internals, classifiers, calibration, etc.) line by line on the
  first pass, then I'll confirm before you continue.
- After each chunk, run it and show me the output.
- Flag jargon. First time you use a term like "azimuthal average" or
  "logistic regression," give me a one-sentence plain-English explanation.
- When you make design choices, tell me what the alternatives were and
  why you picked this one.
- Ask before installing dependencies or restructuring files.
- If I ask "why," give me the real reason, not a hand-wavy one. I'd
  rather be confused for a minute than fooled into thinking I understand.
- Don't claim something "works" until you've shown me it actually runs
  and produced sensible output. Show me the AUC/PR numbers on the
  held-out video split before claiming any change is an improvement.

---

## Workflow for new features

When implementing a new feature (especially in Module 3 / ensemble),
follow this sequence and delegate to subagents at the right steps:

1. **Explain the concept first.** If I haven't shown I understand the
   underlying idea (FFT, calibration, spectral entropy, etc.), delegate
   to `@concept-explainer`. Wait for my confirmation before coding.

2. **Build incrementally.** Write one chunk at a time. Show me the code,
   walk me through new concepts line by line on first pass, confirm
   before continuing.

3. **Code review.** After the code is written, delegate to `@ml-reviewer`
   before claiming it's done. Fix any 🚨 BLOCKER or ⚠️ CONCERN findings
   before moving on.

4. **Measure.** Delegate to `@metric-checker` to run `ensemble.py` and
   report real AUC / PR-AUC numbers on the held-out video split. Never
   claim improvement without these numbers.

5. **Sanity-check data.** If metrics moved a lot in either direction
   (|Δ AUC| > 0.05), delegate to `@dataset-inspector` to confirm the
   feature CSV and face crops still look right.

6. **Summarize.** Report what all subagents found. Only then is the
   feature "done."

Hard rules:
- Never skip steps 3 and 4.
- Never call something an "improvement" without `@metric-checker` numbers
  from a fresh run.
- If a subagent flags a blocker, stop and fix it before continuing.
- If I interrupt with "delegate that" or "run @metric-checker," do it
  immediately — don't argue.

Available subagents (in `.claude/agents/`):
- `@concept-explainer` — plain-English explanations of ML / DSP concepts
- `@ml-reviewer` — catches data leakage, scaler/test contamination, metric errors
- `@metric-checker` — runs eval, reports honest AUC/PR numbers vs baseline
- `@dataset-inspector` — read-only sanity checks on manifest, crops, features

---

## Current focus

The core detection pipeline is now substantially complete. What remains:

1. **Integrate Module 1 (MRL blink detection) into the ensemble.**
   The MRL pipeline (`src/mrl/`) is fully built — MobileNetV2 trained on the
   MRL eye dataset, inference + blink-counting done. But the `ear_score` in
   `ensemble.py` is still the `0.5` stub. The next step is wiring
   `src/mrl/inference.py` output into `extract_all_features()` so the LR
   ensemble gets a real blink-rate signal.

2. **Validate stacking blend weight.**
   `stacking_ensemble.py` learns the optimal CNN/LR alpha from the held-out
   val set. If it reports `alpha_reliable=True`, update `predict.py` and
   `app.py` to load from `data/stacking_bundle.pkl` (they already do this —
   just needs the bundle to exist on disk).

3. **End-to-end demo.**
   Run `streamlit run app.py` and `python predict.py` on a real and a fake
   video to confirm the full pipeline works top-to-bottom.

---

## Project Overview

A deepfake detection pipeline built on **FaceForensics++ C23** (face-swap videos).
The system combines handcrafted features, a fine-tuned CNN, temporal motion
analysis, and an rPPG liveness check to classify face videos as real (0) or fake (1).

Four detection modules:
- **Module 1** — MRL eye blink detection (MobileNetV2, `src/mrl/`)
- **Module 2** — JPEG compression artifact score (`artifact_module.py`)
- **Module 3** — FFT frequency anomaly + Laplacian texture + temporal + rPPG
  (`src/freq_analysis/`, `src/temporal_scorer.py`, `src/rppg_scorer.py`, `ensemble.py`)
- **CNN** — EfficientNet-B0 fine-tuned on FF++ C23 (`cnn_detector.py`, `src/cnn_runner.py`)

---

## Run Order

```bash
# (Optional) Step 0 — train the EfficientNet-B0 CNN (needs PyTorch)
python cnn_detector.py
# → writes data/cnn_model.pth, data/plots/cnn_roc.png, data/plots/cnn_training.png

# Step 1 — extract face crops from the FF++ C23 videos already on disk
python inspect_dataset.py
# → writes data/real/frames/, data/fake/frames/, data/manifest.csv

# Step 2 — extract features, train logistic regression, evaluate
python ensemble.py
# → writes data/module3_features.csv, data/ensemble_model.pkl,
#   data/plots/roc_curve.png, data/plots/precision_recall.png

# (Optional) Step 3 — learn the optimal CNN/LR blend weight
python stacking_ensemble.py
# → writes data/stacking_bundle.pkl (alpha + AUC table)

# Step 4a — CLI inference on a single video
python predict.py path/to/video.mp4
python predict.py path/to/video.mp4 --frames 16 --min-quality 0.20

# Step 4b — Streamlit web UI
streamlit run app.py
```

`main.py` is a standalone demo for Module 1 preprocessing (video loading +
face detection). It is independent of the Steps above.

---

## File Structure

```
Group-8-AISC/
│
├── .claude/
│   └── agents/              Subagent definitions (see Workflow section)
│       ├── concept-explainer.md
│       ├── ml-reviewer.md
│       ├── metric-checker.md
│       └── dataset-inspector.md
│
├── inspect_dataset.py       Step 1 — extract face crops from FF++ C23 videos
├── ensemble.py              Step 2 — feature extraction, LR training, evaluation
├── artifact_module.py       Module 2 — JPEG recompression artifact scorer
├── cnn_detector.py          CNN training — EfficientNet-B0 transfer learning
├── stacking_ensemble.py     Step 3 — learn optimal CNN/LR blend weight (alpha)
├── predict.py               CLI inference — single video → verdict + per-frame table
├── app.py                   Streamlit web UI — upload video → verdict + breakdown
├── main.py                  Module 1 demo — video loading + face detection (standalone)
├── download_data.py         Helper — Kaggle dataset downloader (kagglehub)
│
├── src/
│   ├── preprocessing/       Module 1 helpers (used by main.py)
│   │   ├── face_detector.py     Haar cascade face crop
│   │   ├── frame_extracter.py   Frame generator from cv2.VideoCapture
│   │   └── video_loader.py      cv2.VideoCapture wrapper
│   │
│   ├── freq_analysis/       Module 3 feature extractors
│   │   ├── anomaly_scorer.py    fft_anomaly_score() — 0-1 FFT score
│   │   ├── fft_extractor.py     FFT primitives (grayscale, log-mag, mask)
│   │   ├── frequency_analyzer.py  Batch scoring + visualise_spectrum()
│   │   ├── texture_scorer.py    laplacian_score() — sharpness/texture score
│   │   └── utils.py             load_face_image(), resize_to_square()
│   │
│   ├── cnn_runner.py        CNN inference wrapper — load_cnn(), cnn_predict()
│   │                        (soft import: returns None if torch not installed)
│   │
│   ├── quality_scorer.py    Per-frame quality scoring (sharpness, size,
│   │                        brightness, contrast) → quality_weighted_mean()
│   │
│   ├── temporal_scorer.py   Optical flow temporal consistency — Lucas-Kanade
│   │                        tracks feature points; scores smoothness + jitter
│   │
│   ├── rppg_scorer.py       rPPG liveness check — bandpass-filtered green
│   │                        channel; no pulse = fake signal (needs ≥ 30 frames)
│   │
│   └── mrl/                 Module 1 — MRL eye blink detection pipeline
│       ├── train.py             build_model() — MobileNetV2 architecture
│       ├── inference.py         load_model(), process_video() — per-frame
│       │                        awake/sleepy + blink counting (needs MediaPipe)
│       ├── score.py             Blink-rate → EAR-style 0-1 score
│       ├── preprocess.py        Crop eye regions from frames
│       ├── eda_mrl.py           EDA plots for MRL dataset
│       └── download_mrl.py      Download MRL dataset via kagglehub
│
├── docs/
│   ├── guide.md             User guide — what the system does, how to run it
│   ├── README.md            Docs index
│   └── api/
│       ├── freq_analysis_api.md   API reference for src/freq_analysis/
│       └── ensemble_reference.md  API reference for ensemble.py
│
├── data/
│   ├── FaceForensics++_C23/   Source videos — DO NOT MODIFY
│   │   ├── original/          1000 real YouTube face videos (.mp4)
│   │   ├── Deepfakes/         1000 autoencoder face-swap videos (.mp4)
│   │   ├── Face2Face/         1000 expression-transfer videos (.mp4)
│   │   ├── FaceSwap/          1000 geometry face-swap videos (.mp4)
│   │   └── csv/               FF++ metadata CSVs
│   ├── real/frames/           Extracted real face crops (224×224 JPEGs)
│   ├── fake/frames/           Extracted fake face crops (224×224 JPEGs)
│   ├── manifest.csv           Image list: file_path, label, video_id, source
│   ├── module3_features.csv   Per-image features: artifact, fft, laplacian
│   ├── ensemble_model.pkl     Saved LR bundle: model + scaler + threshold
│   ├── stacking_bundle.pkl    CNN/LR blend weight (alpha) + AUC comparison
│   ├── metrics_log.csv        Run history (written by @metric-checker)
│   ├── plots/                 roc_curve.png, precision_recall.png,
│   │                          cnn_roc.png, cnn_training.png
│   ├── visualizations/        FFT spectrum side-by-sides, artifact examples
│   └── cnn_model.pth          Best EfficientNet-B0 checkpoint (gitignored)
│
├── eda_outputs/               MRL EDA plots — class_counts.png, sample_grid.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Key Configuration

**`inspect_dataset.py`** — controls dataset size and quality:
```python
TARGET_PER_CLASS = 200   # face crops to extract per class
FRAMES_PER_VIDEO = 4     # frames sampled from each video
MIN_BRIGHTNESS   = 40    # reject frames darker than this
MIN_FACE_FRAC    = 0.04  # reject if face < 4% of frame area
REAL_SRC = "data/FaceForensics++_C23/original"
FAKE_SRC = "data/FaceForensics++_C23/Deepfakes"
```

To switch manipulation type, change `FAKE_SRC` to one of:
`Deepfakes` / `Face2Face` / `FaceSwap` / `NeuralTextures` / `DeepFakeDetection`

**`ensemble.py`** — controls LR training:
```python
FEATURE_NAMES = ["artifact", "fft", "laplacian"]  # EAR stub removed
# train_ensemble() uses:
#   GroupShuffleSplit(test_size=0.20, random_state=42)  — no identity leakage
#   CalibratedClassifierCV(LogisticRegression(...), method="sigmoid", cv=3)
#   class_weight="balanced"
#   Threshold chosen to maximise balanced_accuracy_score on val set
```

**`cnn_detector.py`** — controls CNN training:
```python
NUM_EPOCHS   = 30   # early stopping (PATIENCE=10) usually kicks in earlier
BATCH_SIZE   = 16
LR_HEAD      = 1e-3   # custom classifier head
LR_BACKBONE  = 1e-4   # last 2 EfficientNet-B0 blocks (layers 7-8), fine-tuned
# Layers 0-6 frozen (generic ImageNet features kept fixed)
```

**`app.py` / `predict.py`** — inference settings:
```python
# CNN blend weight (alpha) loaded from data/stacking_bundle.pkl at startup.
# If bundle missing or alpha_reliable=False, falls back to hardcoded 0.65.
# frame_prob = alpha * CNN_prob + (1 - alpha) * LR_prob
#
# Temporal signal: 15% nudge toward temporal score when available
# rPPG signal:     10% nudge toward rPPG fake score when available (≥30 face frames)
```

---

## Features Used

### Handcrafted features (LR ensemble, trained on FF++ C23 crops)

| Feature | Source | Signal on FF++ C23 |
|---|---|---|
| `artifact` | JPEG recompression delta (`artifact_module.py`) | Very weak (Δ ≈ 0.002) |
| `fft` | FFT peripheral energy (`anomaly_scorer.py`) | Weak (Δ ≈ 0.012) |
| `laplacian` | Laplacian variance / 3000 (`texture_scorer.py`) | Moderate (Δ ≈ 0.06) |

All features are in [0, 1]. `StandardScaler` applied before LR.
`CalibratedClassifierCV` (Platt scaling) converts raw logit to calibrated P(fake).

### Inference-time signals (used in `predict.py` / `app.py`, not in LR training)

| Signal | Source | Notes |
|---|---|---|
| CNN P(fake) | EfficientNet-B0 (`src/cnn_runner.py`) | AUC ≈ 0.90 on FF++ C23 (296 crops) |
| Frame quality | `src/quality_scorer.py` | Sharpness 40% + size 25% + brightness 20% + contrast 15% |
| Temporal score | `src/temporal_scorer.py` | Lucas-Kanade optical flow; smoothness + jitter + acceleration |
| rPPG fake score | `src/rppg_scorer.py` | Green-channel bandpass; needs ≥ 30 face frames |

**EAR** (Module 1 blink rate) is still a `0.5` stub — MRL pipeline built but not wired in yet.

---

## How the inference pipeline works (predict.py / app.py)

```
Video
 └─ Sample N frames evenly
     └─ For each frame:
         1. Haar cascade face detection
         2. Quality score (sharpness, size, brightness, contrast)
         3. Handcrafted features → LR → LR P(fake)
         4. CNN (EfficientNet-B0) → CNN P(fake)
         5. frame_prob = alpha×CNN + (1-alpha)×LR
     └─ Quality-weighted mean of frame_prob  → qw_prob
     └─ Dense 2-second burst from video centre → temporal score (±15% nudge)
     └─ Green-channel bandpass across all face crops → rPPG score (±10% nudge)
     └─ combined = clip(qw_prob + nudges, 0, 1)
     └─ Map to verdict band (5 tiers, 0.20 width each)
```

---

## Manifest Format

`data/manifest.csv` columns:
- `file_path` — relative path to the JPEG face crop
- `label` — `0` = real, `1` = fake
- `video_id` — e.g. `real_000`, `fake_042` — used for GroupShuffleSplit
- `source_dataset` — `FaceForensics++_C23/original` or `.../Deepfakes`

`data/module3_features.csv` columns:
- `file_path`, `label`, `video_id`, `source_dataset`
- `artifact_score`, `fft_score`, `laplacian_score`
- (Note: `ear_score` column was removed — was constant 0.5, zero signal)

---

## Known Limitations

**FF++ C23 is deliberately hard.** The C23 H.264 compression smooths out GAN
artifacts that JPEG and FFT scores are designed to catch. Expected AUC for
handcrafted LR features alone: 0.50–0.70. CNN adds significant lift (≈ 0.90).

**EAR is still stubbed.** `ear_score = 0.5` in `extract_all_features()`.
The MRL pipeline (`src/mrl/`) is complete — MobileNetV2 trained, inference
and blink-counting written. Still needs to be wired into `ensemble.py`.

**rPPG needs many frames.** The rPPG scorer requires ≥ 30 face frames (≈ 1s
at 30 fps) to produce a signal. Most short clips or low-frame-count runs
will return `available=False` and the signal is skipped.

**Temporal score uses a dense burst.** `predict.py` re-opens the video to
sample a consecutive 2-second burst from the centre for optical flow.
If the video is < 2 seconds the burst will be short and the score less reliable.

**Small val set.** The val set is ≈ 34 videos (20% of 167). AUC estimates
have wide confidence intervals. `stacking_ensemble.py` runs 5-fold CV to
check whether the optimal CNN/LR blend weight is stable — if `alpha_reliable`
is False, don't trust the learned alpha.

**Face detection fallback.** Frames where the Haar cascade finds no face are
skipped entirely (not center-cropped). Low-angle or profile shots will miss.

---

## Environment

- Python 3.13, Windows 11
- Virtual environment: `.venv/` (run `.\.venv\Scripts\activate` before any `pip` command)
- Key dependencies: `opencv-python`, `numpy`, `scikit-learn`, `matplotlib`,
  `torch`, `torchvision`, `streamlit`, `scipy`, `datasets` (HuggingFace), `kagglehub`

Install all dependencies:
```bash
pip install -r requirements.txt
```

### Environment gotchas

- Python 3.13 is new — some packages (especially older PyTorch / TF
  builds) may not have prebuilt wheels for it. Prefer recent versions and
  tell me before pinning anything old.
- Windows paths use backslashes; use `pathlib.Path` rather than string
  concatenation when writing new code so it stays portable.
- `.venv` must be activated before any pip command. Confirm activation
  in the terminal before installing anything — do not install globally.
- The `data/FaceForensics++_C23/` directory is large and read-only by
  convention. Never write into it; produce outputs in `data/real/`,
  `data/fake/`, `data/plots/`, or `data/visualizations/`.
- `streamlit run app.py` requires the `.venv` to be active and the
  `data/ensemble_model.pkl` to exist (run `ensemble.py` first).

---

## Module Integration Status

| Module / Component | Status |
|---|---|
| Module 1 — MRL blink detection (training) | ✅ Complete — `src/mrl/train.py`, `inference.py`, `score.py` |
| Module 1 — EAR wired into ensemble | ❌ Not done — still `0.5` stub in `ensemble.py` |
| Module 2 — JPEG artifact | ✅ Complete (`artifact_module.py`) |
| Module 3 — FFT + texture ensemble | ✅ Complete (`ensemble.py`) |
| Module 3 — video-level GroupShuffleSplit | ✅ Complete |
| Module 3 — LR calibration (Platt scaling) | ✅ Complete (`CalibratedClassifierCV`) |
| Module 3 — quality-weighted frame averaging | ✅ Complete (`src/quality_scorer.py`) |
| Module 3 — temporal consistency (optical flow) | ✅ Complete (`src/temporal_scorer.py`) |
| Module 3 — rPPG liveness check | ✅ Complete (`src/rppg_scorer.py`) |
| CNN — EfficientNet-B0 training | ✅ Complete (`cnn_detector.py`) |
| CNN — inference wrapper | ✅ Complete (`src/cnn_runner.py`) |
| Stacking ensemble (CNN/LR alpha) | ✅ Complete (`stacking_ensemble.py`) |
| CLI inference | ✅ Complete (`predict.py`) |
| Streamlit web UI | ✅ Complete (`app.py`) |
| Docs | ✅ Added (`docs/guide.md`, `docs/api/`) |
