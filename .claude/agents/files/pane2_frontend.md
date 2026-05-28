# Pane 2: Frontend (React / TypeScript)

## Your scope
You own the React frontend in the `ui-migration` worktree. You do NOT touch any Python files. You do NOT write Playwright tests.

**Files you own:**
- All `.tsx`, `.jsx`, `.ts`, `.js`, `.css` files under the `ui-migration` worktree (or wherever your React app lives)
- API client code, components, pages, types

**Files you do NOT touch:**
- `backend/`, `src/`, `*.py` files anywhere
- `tests/e2e/`, `scripts/run_loop_iteration.*`

## The contract you must follow
Read `api_contract.md` (in the repo root). The fields you display in the UI MUST come from the exact field names defined there. Don't invent your own names. Don't assume the backend will send you camelCase — it sends snake_case, you convert at the boundary if you want camelCase internally.

If you need a field that's not in the contract, STOP and request it from Pane 1 via the shared notes. Don't add it unilaterally.

## Project context
- Deepfake detection UI. The result card currently shows a verdict (FAKE/REAL/UNCERTAIN), a p_fake score (e.g. 0.721), a confidence percentage, a scoring model name (currently hardcoded as "LogReg (trained)"), and metric tiles (faces found, FPS, duration, frames sampled, CNN active).
- The current UI reads some values correctly but has hardcoded strings ("LogReg (trained)") that need to come from the API instead.
- The user wants to see which model produced the verdict and its F1 score, plus an optional "all models" leaderboard page.

## Your tasks (in order)

### 1. Update the TypeScript API types
Define types that match `api_contract.md` exactly:
- `PredictResponse` with all fields from the `POST /predict` contract
- `ModelEntry` matching `GET /models` entries
- `HealthResponse` matching `GET /health`

Put them in `src/types/api.ts` or wherever your types live.

### 2. Update the result card
Find the component that renders the analysis result (the card showing "FAKE 0.721 / SCORING MODEL: LogReg (trained)").
- `SCORING MODEL` field reads `response.model_type` (display as-is: `lr`, `stacked`, `stacked_with_blink`)
- `CNN ACTIVE` tile reads `response.cnn_active` (boolean) and displays "YES" / "NO"
- Add a small text under SCORING MODEL: `F1: {response.model_f1.toFixed(2)}` if present
- Existing fields (`verdict`, `p_fake`, `confidence`, `faces_found`, `fps`, `duration_s`, `frames_sampled`) keep reading from the API — just confirm the field names match the contract

### 3. Expand the "MODEL DETAILS" collapsible
The collapsible section currently exists but may be empty. Populate it with:
- Model ID: `response.model_id`
- Model type: `response.model_type`
- Validation F1: `response.model_f1`
- Degraded reason (if `response.degraded_reason` is not null) — render as a warning

### 4. Update the Signal Breakdown panel
The existing signal breakdown shows things like "Quality-weighted P(fake): 0.6948" and "Temporal inconsistency: 0.1132".
- Pull these from `response.signals.quality_weighted_p_fake` and `response.signals.temporal_inconsistency`
- Add rows for any other signals in the contract: `blink_score`, `rppg_liveness`, `jpeg_artifact`
- If a signal is null/missing, show "—" instead of crashing

### 5. Build a /models page (optional but nice for the demo)
- New route `/models`
- Fetches `GET /models`, renders a sortable table: model_id, model_type, F1, precision, recall, AUC, trained_at, is_active
- Highlight the active model (badge or row color)
- Sort by F1 descending by default

### 6. Handle the API contract carefully
- Probabilities come as floats in [0,1]. Multiply by 100 for percentages at the display layer, never modify the stored value.
- Booleans come as true/false. Convert to "YES"/"NO" only at the display layer.
- If any contract field is missing in the actual response, render "—" — don't crash. Log a warning to the console so we know the backend has a bug.

## Constraints
- Don't change the visual design (colors, fonts, layout) beyond what's needed to add the new fields. The existing dark theme with green/yellow/red probability bar stays.
- Don't add libraries unless absolutely necessary. Use what's already in `package.json`.
- Don't touch any Python file. If you need a backend change, write a note for Pane 1.
- Don't hardcode any text that should come from the API (no more "LogReg (trained)" strings).

## Done means
- Upload a video through the UI, response renders without crashing
- Result card shows the real `model_type` from the API (not hardcoded "LogReg (trained)")
- `CNN ACTIVE` tile reflects the real backend state
- Model F1 visible somewhere on the card
- Model Details dropdown populated with model_id, type, F1, and degraded_reason (if present)
- Signal Breakdown shows all 5 signals from the contract
- `/models` page (if built) loads and displays the registry

When done, post a one-line summary to shared notes for Pane 3 to know the UI is testable.
