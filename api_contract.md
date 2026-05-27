# API Contract

> **TODO:** Fill in the actual endpoint definitions here so backend, frontend,
> and test panes share one source of truth.
>
> Suggested sections:

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
| `file` | file | yes | Video file (.mp4, .avi, .mov) |
| `frames` | int | no | Frames to sample (default: 16) |
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
  ]
}
```

### GET /health

Liveness check.

**Response**

```json
{ "status": "ok", "model_loaded": true }
```

---

## Error codes

| HTTP status | Meaning |
|-------------|---------|
| 400 | Bad request (no face detected, unsupported format) |
| 422 | Validation error (missing required field) |
| 500 | Internal server error |

---

## Shared types

```typescript
type Verdict = "VERY LIKELY REAL" | "LIKELY REAL" | "UNCERTAIN"
             | "LIKELY FAKE" | "VERY LIKELY FAKE";
```
