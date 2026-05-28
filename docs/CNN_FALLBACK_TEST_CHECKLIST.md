# CNN Fallback Rollout Test Checklist

Target: validate backend API readiness for `cnn_fallback` rollout with current threshold/uncertain-band bundle and failure-mode coverage.

Evidence snapshot note: contract-level evidence is available in `data/experiments/api_smoke_results.json`; activation-path runtime executions (gate/force toggles) are still pending in this checklist.

## 1) Bundle + `/health` contract checks

- [ ] Start backend with current model bundle (`data/ensemble_model.pkl`) in place.
- [ ] Call `GET /health`; expect HTTP 200 and `status="ok"`.
- [ ] Verify threshold fields are present and internally consistent:
  - [ ] `threshold` exists (current observed serving value: `0.05`).
  - [ ] `uncertain_band` exists (current observed serving value: `0.1`).
  - [ ] `verdict_hi == min(1.0, threshold + uncertain_band)`.
  - [ ] `verdict_lo == max(0.0, threshold - uncertain_band)`.
- [x] Verify fallback visibility fields:
  - [x] `handcrafted_auc`
  - [x] `cnn_fallback_auc_gate` (expected `0.65`)
  - [x] `cnn_fallback_active`
  - [x] `cnn_fallback_reason`
- [x] Verify `model_used_states` includes:
  - [x] `ensemble_learned`
  - [x] `equal_weights`
  - [x] `cnn_fallback`
  - [x] `cnn_fallback_degraded`

## 2) `/analyze` happy-path checks

- [ ] Submit valid face-containing video to `POST /analyze` (`multipart/form-data` with `video` field).
- [ ] Expect HTTP 200 and response fields:
  - [ ] `verdict` in `{FAKE, REAL, UNCERTAIN}`
  - [ ] `prob_fake_mean`, `threshold`, `uncertain_band`, `verdict_hi`, `verdict_lo`
  - [ ] `model_used` in expected enum
  - [ ] `cnn_active` boolean
  - [ ] `frames_analyzed > 0`, `frames_sampled >= frames_analyzed`
  - [ ] `frames` array populated
- [ ] Validate uncertain-band verdict logic from payload:
  - [ ] If `prob_fake_mean >= verdict_hi` -> `verdict=FAKE`
  - [ ] If `prob_fake_mean <= verdict_lo` -> `verdict=REAL`
  - [ ] Otherwise -> `verdict=UNCERTAIN`

## 3) CNN fallback gate behavior (ops toggles)

- [ ] Run with `HANDCRAFTED_VALIDATION_AUC=0.50`, `FORCE_CNN_FALLBACK=0`:
  - [ ] `/health.cnn_fallback_active == true`
  - [ ] `/health.cnn_fallback_reason == "gated_auc"`
- [ ] Run with `HANDCRAFTED_VALIDATION_AUC=0.80`, `FORCE_CNN_FALLBACK=0`:
  - [ ] `/health.cnn_fallback_active == false`
  - [ ] `/health.cnn_fallback_reason == "off"`
- [ ] Run with `FORCE_CNN_FALLBACK=1`:
  - [ ] `/health.cnn_fallback_active == true`
  - [ ] `/health.cnn_fallback_reason == "forced"`

## 4) Failure-mode tests (required)

### A) Missing CNN weights / inference unavailable
- [ ] Ensure CNN infer loader returns unavailable (current placeholder path).
- [ ] Trigger fallback-active run and submit valid video to `/analyze`.
- [ ] Confirm response degrades safely:
  - [ ] HTTP 200 (no crash)
  - [ ] `model_used == "cnn_fallback_degraded"` when fallback is active and CNN inference is unavailable.

### B) Missing handcrafted bundle (`data/ensemble_model.pkl`)
- [ ] Temporarily move/remove pickle and restart API.
- [ ] Call `/health` and verify:
  - [ ] HTTP 200
  - [ ] `model_loaded == false`
  - [ ] `threshold == 0.5` and `uncertain_band == 0.1` defaults
- [ ] Submit valid video to `/analyze`; verify service still responds with `model_used == "equal_weights"` when fallback is off.

### C) Malformed uploads
- [ ] Empty file upload to `/analyze`:
  - [ ] Expect HTTP 422 with `detail` containing `Uploaded file is empty`.
- [ ] Non-video bytes uploaded as `video`:
  - [ ] Expect HTTP 422 (`Could not open video file` or equivalent user-correctable detail).

### D) No-face video
- [ ] Submit a video with no detectable face across sampled frames.
- [ ] Expect HTTP 422 and detail containing `No faces detected`.

## 5) Basic observability checks

- [ ] Confirm backend logs show request status codes for `/health` and `/analyze`.
- [ ] Confirm warnings are returned (not fatal) when partial frame issues occur.
- [ ] Record one representative smoke run in `data/experiments/api_smoke_results.json`.

## 6) Rollout go/no-go

- [ ] Go if all contract + failure-mode checks pass and no 5xx appears in smoke tests.
- [ ] No-go if any required failure-mode test returns unexpected 5xx or breaks schema contract.
