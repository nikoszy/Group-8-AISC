# API Contract — Group 8 Deepfake Detector

**This is the source of truth for the data shape between backend and frontend.**
**Do not change any field name without updating this file and notifying all panes.**

## Endpoint: POST /predict

**Request:** multipart/form-data with a single `file` field containing the video.

**Response (200 OK):** JSON with this exact shape and these exact field names:

```json
{
  "verdict": "REAL" | "FAKE" | "UNCERTAIN",
  "p_fake": 0.234,
  "confidence": 0.781,

  "model_id": "stacked_with_blink_v3",
  "model_type": "lr" | "stacked" | "stacked_with_blink",
  "model_f1": 0.87,

  "cnn_active": true,
  "faces_found": "9/12",
  "fps": 30.0,
  "duration_s": 7.2,
  "frames_sampled": 12,

  "signals": {
    "quality_weighted_p_fake": 0.234,
    "temporal_inconsistency": 0.113,
    "blink_score": 0.42,
    "rppg_liveness": 0.88,
    "jpeg_artifact": 0.15
  },

  "degraded_reason": null
}
```

### Field semantics
- `verdict` — final 5-tier classification, but collapsed to 3 for UI. UNCERTAIN if p_fake between 0.4–0.6.
- `p_fake` — final ensemble probability, float in [0.0, 1.0].
- `confidence` — model's confidence in its verdict, float in [0.0, 1.0]. Displayed as percentage.
- `model_id` — unique ID of the model that ran, matches an entry in `model_registry.json`.
- `model_type` — one of: `lr`, `stacked`, `stacked_with_blink`. Shown in UI as "SCORING MODEL".
- `model_f1` — validation F1 of the active model. Shown in UI under model details.
- `cnn_active` — true if CNN was actually used in the blend for this prediction. If false, `degraded_reason` must explain why.
- `faces_found` — string like "9/12" meaning 9 faces detected across 12 sampled frames.
- `signals` — per-module diagnostic outputs. Frontend renders these in the Signal Breakdown panel.
- `degraded_reason` — null when everything ran normally. String explaining the issue if `cnn_active=false` or any module failed (e.g. "CNN checkpoint not found", "Module 1 inference failed: <error>").

## Endpoint: GET /models

**Response (200 OK):** JSON array of all registered models from `artifacts/model_registry.json`:

```json
[
  {
    "model_id": "lr_baseline",
    "model_type": "lr",
    "metrics": { "f1": 0.72, "precision": 0.74, "recall": 0.70, "auc": 0.81 },
    "trained_at": "2026-05-20T14:30:00Z",
    "is_active": false,
    "comparable": true,
    "notes": "LR on handcrafted features only"
  },
  {
    "model_id": "stacked_with_blink_v3",
    "model_type": "stacked_with_blink",
    "metrics": { "f1": 0.87, "precision": 0.88, "recall": 0.86, "auc": 0.93 },
    "trained_at": "2026-05-26T09:15:00Z",
    "is_active": true,
    "comparable": true,
    "notes": "CNN + LR + Module 1 blink, stacking alpha=0.62"
  }
]
```

## Endpoint: POST /models/reload

Dev-only. Re-reads `artifacts/model_registry.json` without restarting the backend.

**Response:** `{ "reloaded": true, "active_model_id": "..." }`

## Endpoint: GET /health

**Response:** `{ "status": "ok", "cnn_loaded": true, "active_model_id": "..." }`

Used by the Playwright loop script to check the server is up.

---

## Rules
1. Field names are snake_case. Always. Never camelCase.
2. Booleans are true/false, never "YES"/"NO" — the frontend converts for display.
3. Probabilities are floats in [0.0, 1.0], never percentages — frontend multiplies by 100 for display.
4. If a field is missing from the response, the frontend should treat it as null and show "—" in the UI, not crash.
5. Any new field added to the response must be added here first and announced to all panes.
