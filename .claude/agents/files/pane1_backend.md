# Pane 1: Backend + Model Registry (Python)

## Your scope
You own the Python side of the project. You do NOT touch any frontend code (no `.tsx`, `.jsx`, `.css` files). You do NOT write Playwright tests.

**Files you own:**
- `backend/` (all of it)
- `src/model_registry.py` (you'll create this)
- `src/cnn_runner.py`, `src/mrl/inference.py`, `src/freq_analysis/`, `src/temporal_scorer.py`, `src/rppg_scorer.py`, `src/quality_scorer.py`
- `ensemble.py`, `stacking_ensemble.py`, `predict.py`, `cnn_detector.py`
- `artifacts/model_registry.json` (you'll create/maintain this)

**Files you do NOT touch:**
- Anything under the `ui-migration` worktree or React app
- Anything under `tests/e2e/` or `scripts/run_loop_iteration.*`

## The contract you must follow
Read `api_contract.md` (in the repo root). The `/predict`, `/models`, `/models/reload`, and `/health` endpoints you build MUST return exactly the JSON shape defined there. Field names are snake_case. Booleans are true/false (not "YES"/"NO"). Probabilities are floats in [0,1] (not percentages).

If you need to deviate from the contract, STOP and update `api_contract.md` first, then announce the change so the other panes can react.

## Project context
- Deepfake detection system. 4 modules: MRL blink (Module 1), JPEG artifact (Module 2), FFT+Laplacian+temporal+rPPG (Module 3), EfficientNet-B0 CNN.
- Current state: CNN is failing to load (`cnn_active: false` in responses), scoring falls back to LR-only. Module 1 was just integrated into Module 3 baseline but may not be properly feeding the LR ensemble.
- Baseline from user: with Modules 1 + 3 active but CNN off, real videos score p_fake ≈ 0.20–0.30. The full ensemble should match or beat this.

## Your tasks (in order)

### 1. Build the model registry
Create `src/model_registry.py`:
- `ModelRegistry` class reading/writing `artifacts/model_registry.json`
- Methods: `register(entry)`, `get_best(metric="f1")`, `get_active()`, `set_active(model_id)`, `list_all()`
- Each entry matches the schema in `api_contract.md` under `GET /models`
- Entries with `comparable: false` are excluded from `get_best()`
- If no entry has `is_active: true`, `get_active()` returns the highest-F1 comparable entry

### 2. Fix CNN loading
Find why `cnn_active` is currently false. Check, in order:
- Does the `.pt` checkpoint exist? Print the absolute path being attempted.
- Is torch installed in the FastAPI venv? (`python -c "import torch"`)
- Is there a try/except silently swallowing the load failure? Make it loud — log full trace, re-raise on startup so the backend refuses to start with a broken CNN.
- Device mismatch? Use `torch.load(path, map_location="cpu")` if no CUDA.

The backend must log ONE clear line on startup: `CNN loaded successfully from <path> on device <cpu/cuda>` OR fail loudly with the exact reason.

### 3. Wire Module 1 into LR feature vector (if not already)
Grep for `ear_score = 0.5` anywhere in `ensemble.py` or related files. If found, replace with a real call to `src/mrl/inference.py`. If Module 1 outputs are already being computed but not fed into LR features, fix the plumbing.

### 4. Update FastAPI endpoints
In `backend/main.py`:
- On startup, load the active model via `ModelRegistry.get_active()`, cache in app state. Fail loudly if no active model.
- `POST /predict` returns the exact JSON shape in `api_contract.md`.
- `GET /models` returns the full registry as defined in the contract.
- `POST /models/reload` re-reads the registry without restart.
- `GET /health` returns the contract-defined health shape.

### 5. Retrain and register
- Run the updated `ensemble.py` (with Module 1 wired in) — it must call `registry.register(...)` at the end with its eval metrics.
- Run `stacking_ensemble.py` — same: register the stacked model with its F1.
- Confirm `artifacts/model_registry.json` now has at least 2-3 entries with real F1 numbers.
- The highest-F1 entry should be marked active.

### 6. Verify with predict.py CLI
Before declaring done, run:
```
python predict.py "C:\Users\satya\Downloads\IMG_0601.MOV"
python predict.py "C:\Users\satya\Downloads\IMG_9666.MOV"
```
Both should print CNN probabilities per frame, use the stacked model, and produce p_fake ≤ 0.35. If CLI works but frontend doesn't, the bug is in `backend/main.py` (your code) not in the modules.

## Constraints
- Don't modify Module 1 or Module 3's scoring logic — they already produce correct outputs (20–30% on real videos). Only fix the plumbing that gets their signals into the ensemble.
- Don't add silent fallbacks. If something fails, log loudly and surface it in the response's `degraded_reason` field.
- Don't change the API contract without updating `api_contract.md` first.
- Don't touch any frontend files. If you think the frontend needs a change, write it as a note for Pane 2 to pick up.

## Done means
- `python predict.py <real_video.mov>` works end-to-end with CNN active and produces p_fake ≤ 0.35
- `curl http://localhost:8000/health` returns `cnn_loaded: true` and a real `active_model_id`
- `curl http://localhost:8000/models` returns 2+ registered models
- `curl -F "file=@<video>" http://localhost:8000/predict` returns a response matching the contract exactly
- `artifacts/model_registry.json` exists with multiple entries, highest F1 marked active

When done, post a one-line summary to the shared notes for Pane 3 to know it can run the loop.
