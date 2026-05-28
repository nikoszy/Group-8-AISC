# AGENTS.md — Group 8 AISC: Deepfake Detection System

## Project Overview

A deepfake detection pipeline built on **FaceForensics++ C23** (face-swap videos).
The system extracts handcrafted features from face crops and trains a logistic regression ensemble to classify faces as real (0) or fake (1).

Three detection modules:
- **Module 1** — EAR blink analysis (`main.py` + `src/preprocessing/`)
- **Module 2** — JPEG compression artifact score (`artifact_module.py`)
- **Module 3** — FFT frequency anomaly + Laplacian texture score → ensemble (`src/freq_analysis/` + `ensemble.py`)

---

## Run Order

```bash
# Step 1 — extract face crops from the FF++ C23 videos already on disk
python inspect_dataset.py

# Step 2 — extract features, train logistic regression, evaluate
python ensemble.py
```

`main.py` is a standalone demo for Module 1 (video loading + face detection). It is independent of the Steps 1/2 pipeline above.

---

## File Structure

```
Group-8-AISC/
│
├── inspect_dataset.py       Step 1 — extract face crops from FF++ C23 videos
├── ensemble.py              Step 2 — feature extraction, training, evaluation
├── artifact_module.py       Module 2 — JPEG recompression artifact scorer
├── cnn_detector.py          CNN detector — EfficientNet-B0 transfer learning
├── main.py                  Module 1 demo — video loading + face detection
├── download_data.py         Helper — Kaggle dataset downloader (kagglehub)
│
├── src/
│   ├── preprocessing/       Module 1 helpers
│   │   ├── face_detector.py     Haar cascade face crop
│   │   ├── frame_extracter.py   Frame generator from cv2.VideoCapture
│   │   └── video_loader.py      cv2.VideoCapture wrapper
│   ├── blink_analysis/      Module 1 EAR / blink scoring
│   │   └── ear_scorer.py        compute_video_ear_score(), blink detection
│   │
│   └── freq_analysis/       Module 3 feature extractors
│       ├── anomaly_scorer.py    fft_anomaly_score() — 0-1 FFT score
│       ├── fft_extractor.py     FFT primitives (grayscale, log-mag, mask)
│       ├── frequency_analyzer.py  Batch scoring + visualise_spectrum()
│       ├── texture_scorer.py    laplacian_score() — sharpness/texture score
│       └── utils.py             load_face_image(), resize_to_square()
│
├── data/
│   ├── FaceForensics++_C23/   Source videos — DO NOT MODIFY
│   │   ├── original/          1000 real YouTube face videos (.mp4)
│   │   ├── Deepfakes/         1000 autoencoder face-swap videos (.mp4)
│   │   ├── Face2Face/         1000 expression-transfer videos (.mp4)
│   │   ├── FaceSwap/          1000 geometry face-swap videos (.mp4)
│   │   └── csv/               FF++ metadata CSVs
│   ├── real/frames/           Extracted real face crops (224x224 JPEGs)
│   ├── fake/frames/           Extracted fake face crops (224x224 JPEGs)
│   ├── manifest.csv           Image list: file_path, label, video_id, source
│   ├── module3_features.csv   Per-image features: ear, artifact, fft, laplacian
│   ├── video_ear_scores.csv   Per-video Module 1 scores (optional cache)
│   ├── ensemble_model.pkl     Trained LR + scaler + threshold (from ensemble.py)
│   ├── plots/                 roc_curve.png, precision_recall.png, cnn_roc.png
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

**`ensemble.py`** — controls training:
```python
# train_ensemble() uses GroupShuffleSplit(test_size=0.20)
# grouped by video_id — no identity leakage between train/val
# LogisticRegression(class_weight="balanced")
```

---

## Features Used

| Feature | Source | Convention | Signal on FF++ C23 |
|---|---|---|---|
| `artifact` | JPEG recompression delta (Module 2) | higher = more suspicious | Very weak (Δ ≈ -0.002, wrong direction vs design) |
| `fft` | FFT spectral slope anomaly (Module 3) | higher = steeper = smoother = more suspicious | Weak (Δ ≈ +0.018) |
| `smoothness` | `1 - laplacian_score` (Module 3) | higher = smoother = more suspicious | Moderate (Δ ≈ +0.057) |

**Note on `smoothness`**: `laplacian_score` returns higher values for sharper (more real) faces. It is inverted to `smoothness = 1 - laplacian_score` so all features share the "higher = more suspicious" convention. This ensures the LR learns positive coefficients and equal-weights is correctly directed.

**EAR removed from ensemble**: The Haar-based EAR fallback (without MediaPipe) gives ear_score ≈ 1.0 for all videos at inference (short clips, low blink detection → rate_deficit=1.0; consistent Haar readings → static_eye=1.0). This caused the trained model to output prob_fake ≈ 0.008 for every video (everything classified as REAL). EAR is retained in `src/blink_analysis/ear_scorer.py` for future MediaPipe integration.

**Scoring**: The backend uses `ensemble_score_equal_weights(artifact, fft, smoothness)` as the primary score (AUC ≈ 0.60–0.64 on FF++ C23). The LogisticRegression model is trained and saved in `data/ensemble_model.pkl` but its val-set AUC is lower than equal-weights on this dataset (weak feature signal causes inconsistent LR coefficient signs). The threshold in the bundle is calibrated from equal-weights balanced-accuracy on the full training set.

All features are in [0, 1]. StandardScaler + LogisticRegression are retained for diagnostics and potential future use.

---

## Manifest Format

`data/manifest.csv` columns:
- `file_path` — relative path to the JPEG face crop
- `label` — `0` = real, `1` = fake
- `video_id` — e.g. `real_000`, `fake_042` — used for GroupShuffleSplit
- `source_dataset` — `FaceForensics++_C23/original` or `.../Deepfakes`

---

## Known Limitations

**FF++ C23 is deliberately hard.** The C23 H.264 quality setting smooths out the GAN/autoencoder generation artifacts that JPEG and FFT scores are designed to detect. Expected AUC for handcrafted features on C23 is 0.50–0.70.

**EAR is video-level.** One blink/EAR suspicion score per `video_id` (from `data/video_ear_scores.csv` or computed from FF++ source `.mp4`). Still-frame crops share the parent video score. Optional `mediapipe` improves landmarks; OpenCV Haar eyes are used as fallback.

**Face detection fallback.** Frames where the Haar cascade finds no face pass the `MIN_BRIGHTNESS` and `MIN_FACE_FRAC` quality checks but may still be non-ideal crops. These are skipped (not saved) rather than using center-crop fallback.

---

## Environment

- Python 3.13, Windows 11
- Virtual environment: `.venv/` (run `.\.venv\Scripts\activate` before any `pip` command)
- Key dependencies: `opencv-python`, `numpy`, `scikit-learn`, `matplotlib`, `datasets` (HuggingFace), `kagglehub`

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## Module Integration Status

| Module | Status |
|---|---|
| Module 1 — EAR blink detection | Scorer complete (`src/blink_analysis/ear_scorer.py`); **excluded from ensemble** — Haar fallback gives identical scores for real/fake at inference, collapsing model to predict REAL for all inputs. Re-enable after MediaPipe integration. |
| Module 2 — JPEG artifact | Complete (`artifact_module.py`); weak signal on FF++ C23 (Δ ≈ -0.002) |
| Module 3 — FFT spectral slope | Complete (`ensemble.py`); weak signal (Δ ≈ +0.018) |
| Module 3 — Laplacian/smoothness | Complete (`ensemble.py`); moderate signal (Δ ≈ +0.057) |
| Module 3 — ensemble scoring | **Equal-weights primary** (AUC ≈ 0.60–0.64); LR model retained for diagnostics |
| Module 3 — video-level split | Complete (GroupShuffleSplit on video_id; no identity leakage) |
