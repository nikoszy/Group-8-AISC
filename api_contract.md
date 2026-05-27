# API Contract

## Base URL

```
http://localhost:8000
```

## Endpoints

### POST /predict

Upload a video file and receive a deepfake verdict.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | yes | Video file (.mp4, .avi, .mov, .webm) |
| `frames` | int | no | Frames to sample (default: 16, max: 60) |
| `min_quality` | float | no | Minimum face quality threshold (default: 0.20) |

**Response** — `application/json`

```json
{
  "verdict": "LIKELY FAKE",
  "confidence": 0.83,
  "combined_score": 0.83,
  "frame_count": 16,
  "face_frames": 14,
  "module_scores": {
    "cnn": 0.91,
    "lr": 0.74,
    "temporal": 0.70,
    "rppg": null
  },
  "per_frame": [
    { "frame": 0, "cnn": 0.88, "lr": 0.71, "quality": 0.92 }
  ],
  "model_id": "randomforest_20260527_143022",
  "cnn_active": true,
  "degraded_reason": null
}
```

Field notes:
- `combined_score` — final P(fake) after quality weighting + temporal/rPPG nudges [0, 1]
- `confidence` — `abs(combined_score - 0.5) * 2` — how far from the boundary
- `module_scores.cnn` — mean CNN P(fake) across face frames; `null` if CNN not active
- `module_scores.temporal` — optical-flow temporal fake score; `null` when unavailable
- `module_scores.rppg` — rPPG fake score; `null` when fewer than ~30 face frames
- `per_frame[].quality` — frame quality score (sharpness/size/brightness) [0, 1]
- `degraded_reason` — non-null string when any module is degraded (e.g. CNN off)

---

### POST /analyze

Alternative endpoint with a richer response (used by the React frontend).

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `video` | file | yes | Video file |
| `n_frames` | int | no | Frames to sample (default: 12) |

**Response** — `application/json` — full `AnalysisResponse` schema (see `backend/models.py`).
Includes base64-encoded face crops, all per-frame intermediate scores, and registry provenance.

---

### GET /health

Liveness and model-state check.

**Response**

```json
{
  "status": "ok",
  "model_loaded": true,
  "active_model_id": "randomforest_20260527_143022",
  "active_model_type": "lr",
  "active_model_f1": 0.72,
  "cnn_loaded": true,
  "cnn_alpha": 0.65,
  "mrl_loaded": false
}
```

Field notes:
- `cnn_loaded` — `true` when EfficientNet-B0 checkpoint loaded successfully
- `cnn_alpha` — CNN blend weight in `alpha*CNN + (1-alpha)*LR`; default 0.65
- `mrl_loaded` — `true` when MRL blink-detection MobileNetV2 loaded

---

### GET /models

Return all registered models sorted by F1 descending.

**Response**

```json
{
  "models": [
    {
      "model_id": "randomforest_20260527_143022",
      "model_type": "lr",
      "artifact_path": "artifacts/ensemble_model_20260527_143022.pkl",
      "metrics": { "f1": 0.72, "precision": 0.75, "recall": 0.69, "auc": 0.78 },
      "trained_at": "2026-05-27T14:30:22+00:00",
      "notes": "4-feature LR (artifact+fft+laplacian+ear); val split seed=42",
      "is_active": true,
      "comparable": true
    }
  ],
  "active_model_id": "randomforest_20260527_143022",
  "total": 1
}
```

---

### POST /models/reload

Dev-only: re-read the registry and hot-swap the active model without restart.

**Response**

```json
{
  "reloaded": true,
  "active_model_id": "randomforest_20260527_143022",
  "active_model_type": "lr",
  "active_model_f1": 0.72
}
```

---

## Error codes

| HTTP status | Meaning |
|-------------|---------|
| 400 | Bad request (no face detected, unsupported format) |
| 422 | Validation error (missing required field, empty file) |
| 500 | Internal server error |

---

## Shared types

```typescript
type Verdict = "VERY LIKELY REAL" | "LIKELY REAL" | "UNCERTAIN"
             | "LIKELY FAKE" | "VERY LIKELY FAKE";
```

Verdict thresholds for `combined_score`:

| Range | Verdict |
|-------|---------|
| [0.00, 0.20) | VERY LIKELY REAL |
| [0.20, 0.40) | LIKELY REAL |
| [0.40, 0.60) | UNCERTAIN |
| [0.60, 0.80) | LIKELY FAKE |
| [0.80, 1.00] | VERY LIKELY FAKE |

All probabilities are floats in [0, 1] (not percentages). All booleans are `true`/`false`.
Field names are snake_case throughout.
