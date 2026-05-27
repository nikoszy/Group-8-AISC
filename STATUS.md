# Pipeline Alignment Status

Last updated: 2026-05-27

## Deliverables checklist

- [x] Shared `detect_face_crop` / `detect_face_crop_with_bbox` in `src/preprocessing/face_detector.py` used by `inspect_dataset.py`, `predict.py`, `app.py`, `backend/detector.py`
- [x] `predict.py` / `app.py` compute live MRL `ear_score` via `src/mrl/video_ear_score.py`
- [x] Checkpoint path unified: **`models/best_model.pth`** (`run_inference.py`, `backend/main.py`, `predict.py`, `app.py`, tests)
- [x] `module1_output.csv` video_id mapping fixed in `src/mrl/score.py` (`manifest_video_id`)
- [ ] `ensemble.py` + `stacking_ensemble.py` re-run — **blocked** (see Phase 0 below)
- [ ] LR / CNN / combined AUC reported — **blocked** (no `data/manifest.csv` on this machine)
- [x] Backend temporal + rPPG matches `predict.py` (`src/inference_combine.py`, `backend/detector.py`)
- [x] `scripts/eval_videos.py` + Evaluation section in `docs/guide.md`
- [x] Tests: `tests/test_face_detector.py`, `tests/test_module1_video_id.py`

## Phase 0 — Prerequisites (this workspace)

| Item | Status |
|------|--------|
| `data/manifest.csv` | **MISSING** — run `python inspect_dataset.py` |
| `data/real/frames`, `data/fake/frames` | **MISSING** |
| `data/FaceForensics++_C23/` (original + Deepfakes) | **NOT PRESENT** on disk |
| `data/cnn_model.pth` | **MISSING** — run `python cnn_detector.py` after Step 1 |
| `models/best_model.pth` (MRL) | Place trained checkpoint here (Kaggle export or local train) |
| `models/face_landmarker.task` | **OK** (3.6 MB) |
| `.venv` with torch/torchvision/mediapipe | **NOT SET UP** — `pip install -r requirements.txt` |

### Commands to unblock metrics (when FF++ data is available)

```bash
python inspect_dataset.py
python run_inference.py --video-dir ... --output-dir data/results_real   # per class
python run_inference.py --video-dir ... --output-dir data/results_fake
python -m src.mrl.score
python ensemble.py
python stacking_ensemble.py
```

## Key code changes

| Module | Change |
|--------|--------|
| `src/preprocessing/face_detector.py` | Canonical Haar crop: CONF=0.7, scale=1.1, 15% pad, INTER_AREA |
| `src/mrl/video_ear_score.py` | Shared blink → ear_score for predict/app/backend |
| `src/inference_combine.py` | Shared qw + temporal + rPPG combine formula |
| `backend/detector.py` | Full parity with predict.py; no std(probs) temporal stub |
| `api_contract.md` | `combined_score` = final prob after nudges; temporal/rPPG docs |

## Pane status

- **Backend:** aligned with predict.py (2026-05-27 alignment pass)
- **Frontend:** not started
- **Tests:** unit tests added; full MRL stress tests require `models/best_model.pth`
