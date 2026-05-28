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
| 1 — Blink (EAR) | `src/blink_analysis/ear_scorer.py` | Video-level blink / EAR suspicion |
| 2 — Artifact | `artifact_module.py` | JPEG recompression pixel delta |
| 3 — Frequency | `src/freq_analysis/` | FFT peripheral energy + Laplacian variance |
| Ensemble | `ensemble.py` | Logistic regression over all four features → `data/ensemble_model.pkl` |

## Data

Requires FaceForensics++ C23 videos at `data/FaceForensics++_C23/`.
You can download via Kaggle (`xdxd003/ff-c23`) using `download_data.py`.

## Setup

```bash
pip install -r requirements.txt
python inspect_dataset.py
python ensemble.py
```

See `CLAUDE.md` for full project documentation.

## Deployment (Vercel + Render)

### Backend on Render

This repo includes `render.yaml` for a Python web service:

- Root directory: `backend`
- Build: `pip install -r requirements.txt`
- Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`

Set these Render environment variables:

- `CORS_ORIGINS` = your Vercel frontend URL(s), comma-separated
  - Example: `https://your-app.vercel.app,https://www.yourdomain.com`
- `HANDCRAFTED_VALIDATION_AUC` (optional, default `1.0`)
- `FORCE_CNN_FALLBACK` (optional, default `0`)

Health endpoint:

- `https://your-render-service.onrender.com/health`

### Frontend on Vercel

Deploy the `frontend` directory as a Vite project.

Set Vercel environment variable:

- `VITE_API_BASE_URL` = your Render backend URL
  - Example: `https://your-render-service.onrender.com`

An example env file is included at `frontend/.env.example`.

For local development, if `VITE_API_BASE_URL` is not set, frontend requests continue to use `/api` with Vite dev proxy to `localhost:8000`.

## Reproducible ADV-21/22/23 Input Batches

Generate OOD/adversarial support inputs (data-prep only) for ADV-21/22/23:

```bash
python scripts/prepare_adv_inputs.py --max-items 120 --seed 42
```

Outputs are written under `data/experiments/adv_inputs/`:
- `ADV-21/images/`, `ADV-22/images/`, `ADV-23/images/` transformed batches
- deterministic `file_list.txt` per batch
- `metadata.json` per batch (transform parameters + per-file hashes)
- `summary.json` covering the full run

The helper reads real samples from `data/manifest.csv` and applies:
- ADV-21: screen-recording proxy (resize + double JPEG)
- ADV-22: cartoon/non-FF++ style transform
- ADV-23: partial-face/mask occlusion transform
