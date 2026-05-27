# Group 8 AISC — Deepfake Detection System

Deepfake face detection pipeline using FaceForensics++ C23 videos.

## Pipeline

```bash
python inspect_dataset.py   # extract face crops from FF++ C23 videos
python ensemble.py          # train + evaluate the ensemble model
```

## Modules

| Module | File | Method |
|---|---|---|
| 1 — Blink (EAR) | `main.py` + `src/preprocessing/` | Eye Aspect Ratio over video frames |
| 2 — Artifact | `artifact_module.py` | JPEG recompression pixel delta |
| 3 — Frequency | `src/freq_analysis/` | FFT peripheral energy + Laplacian variance |
| Ensemble | `ensemble.py` | Logistic regression over Module 2+3 scores |

## Data

Requires FaceForensics++ C23 dataset at `data/FaceForensics++_C23/`.
Downloaded via Kaggle (`xdxd003/ff-c23`) using `download_data.py`.

## Setup

```bash
pip install -r requirements.txt
python inspect_dataset.py
python ensemble.py
```

See `CLAUDE.md` for full project documentation.
