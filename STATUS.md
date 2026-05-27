# Pane Status

Pane 1 (backend): DONE
Pane 2 (frontend): not started
Pane 3 (tests): not started

## Notes

### Pane 1 — Backend complete (2026-05-27)

All tasks done. Ready for Pane 3 to run the loop.

**What was built:**

- `src/model_registry.py` — JSON-backed registry (already existed)
- `artifacts/model_registry.json` — 2 entries: RandomForest LR (F1=0.71, active) + stacked bundle (AUC=0.78)
- CNN loading: `backend/main.py` lifespan now loads EfficientNet-B0 at startup with loud failure if checkpoint + torch present but load fails. Logs: `CNN loaded successfully from ... on device cpu`
- `backend/detector.py` — `analyze_video()` accepts `cnn_model` + `cnn_alpha`, blends CNN+LR per frame. `cnn_active: true` in responses when CNN contributes.
- **New `/predict` endpoint** — matches contract exactly: `file=` upload field, 5-tier verdict, `combined_score`, `frame_count`, `face_frames`, `module_scores`, `per_frame`
- `/health` — now includes `cnn_loaded`, `cnn_alpha`, `active_model_id`
- `/models` — returns 2 registered models
- `/models/reload` — hot-reload without restart

**Verified:**
```
GET  /health  →  {"cnn_loaded":true,"active_model_id":"randomforest_20260527_202720",...}
GET  /models  →  2 entries, highest-F1 marked active
POST /predict (IMG_9666.MOV)  →  {"cnn_active":true,"verdict":"UNCERTAIN","combined_score":0.42,...}
POST /models/reload  →  {"reloaded":true,...}
```

**Known limitation:** real-world phone videos score p_fake ≈ 0.36–0.53 (target was ≤0.35).
Root cause: MRL checkpoint (`data/best_model.pth`) is missing → ear_score defaults to 0.5 at inference.
The LR was trained on data where real videos have ear≈0.30 and fake have ear≈0.56, so ear=0.5 biases toward fake.
The CNN brings scores down significantly (e.g. CNN P(fake)≈0.14–0.34 on real frames).
Fix: train+save the MRL model to `data/best_model.pth`.

**API contract** (`api_contract.md`) updated with:
- `/predict` endpoint fully documented
- `/health` extended fields (`cnn_loaded`, `cnn_alpha`, `active_model_id`)
- `/models` and `/models/reload` documented
- 5-tier verdict thresholds table
