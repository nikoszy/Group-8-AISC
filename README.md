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

---

## How model selection works

Every time a training script (`ensemble.py`, `stacking_ensemble.py`) finishes,
it calls `ModelRegistry.register()` with its validation metrics, then
`ModelRegistry.select_best()` which marks the highest-F1 comparable model as
active. The registry is stored at `artifacts/model_registry.json`
(human-readable JSON, git-tracked). Model artifact `.pkl` / `.pth` files live
under `artifacts/` and are gitignored.

On FastAPI startup, `ModelRegistry.get_active()` is called automatically:
- **No registry file** → silent fallback to `data/ensemble_model.pkl` (backward compat)
- **Registry exists, no active model** → server fails loudly with a clear error
- **Registry exists, active model found** → that model is loaded and served

The `/analyze` response now includes `model_id`, `model_type`, and `model_f1`
so you always know exactly which model produced a given verdict.

### How to add a new candidate model

1. Train your model on the **same held-out split** (seed=42, 20% val).
2. Save the model artifact to `artifacts/` with a timestamped name.
3. Call `registry.register({model_id, model_type, artifact_path, metrics, comparable=True})`.
4. Call `registry.select_best()` — if your model has the highest val F1 among
   `comparable=True` entries, it automatically becomes active.
5. Restart the server (or call `POST /models/reload` in dev).

```python
from src.model_registry import ModelRegistry

registry = ModelRegistry()
registry.register({
    "model_id":      "my_new_model_20260601",
    "model_type":    "lr",                         # or "cnn", "stacked", etc.
    "artifact_path": "artifacts/my_new_model.pkl",
    "metrics":       {"f1": 0.81, "precision": 0.83, "recall": 0.79, "auc": 0.88},
    "notes":         "Retrained with more data",
    "comparable":    True,   # same val split seed=42
})
winner = registry.select_best()   # promotes this model if F1 > current best
print(f"Active model: {winner['model_id']}  F1={winner['metrics']['f1']}")
```

### Check what's running

```bash
# From repo root (venv active, API running):
curl http://localhost:8000/health
curl http://localhost:8000/models
python scripts/run_e2e_demo.py   # full end-to-end smoke test
```
