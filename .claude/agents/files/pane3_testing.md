# Pane 3: Testing + Agentic Loop (Playwright)

## Your scope
You own the end-to-end testing infrastructure. You do NOT modify backend Python files or frontend React files (Panes 1 and 2 own those). Your only job is to verify their work.

**Files you own:**
- `tests/e2e/` (all of it, you create this)
- `scripts/run_loop_iteration.sh` (or `.ps1` on Windows)
- `playwright.config.ts`
- `tests/fixtures/` (test data references)

**Files you do NOT touch:**
- `backend/`, `src/`, any `.py` files
- The React app under `ui-migration` or wherever
- `api_contract.md` (read-only for you)
- `artifacts/model_registry.json` (read-only)

## The contract you reference
Read `api_contract.md`. Your Playwright test reads UI text, but you can also hit the API directly to verify the contract. If you find the frontend displaying something the contract says shouldn't be there, flag it for Pane 2. If the API returns a field that doesn't match the contract, flag it for Pane 1.

## Project context
- Two real test videos at `C:\Users\satya\Downloads\IMG_0601.MOV` and `C:\Users\satya\Downloads\IMG_9666.MOV`.
- Convert to WSL paths `/mnt/c/Users/satya/Downloads/IMG_0601.MOV` and `/mnt/c/Users/satya/Downloads/IMG_9666.MOV` if running in WSL.
- User-established baseline: Modules 1 + 3 without CNN scored these real videos at p_fake ≈ 0.20–0.30. Full ensemble should match or beat this.
- Backend runs on `http://localhost:8000` (FastAPI via uvicorn).
- Frontend runs on `http://localhost:5173` (React via Vite).

## Your tasks (in order)

### 1. Setup
Detect environment first: `uname -a` and `pwd`. Pick path format accordingly (WSL `/mnt/c/...` vs Windows `C:\...`).

Confirm both test videos exist. If missing, stop and report.

Install Playwright if not already:
```
npm install -D @playwright/test
npx playwright install chromium
```

### 2. Write the Playwright test
Create `tests/e2e/upload_and_read.spec.ts`. For EACH of the two test videos:
- Navigate to `http://localhost:5173`
- Upload the video via the file input
- Wait for the result card (look for "ANALYSIS COMPLETE" text or the verdict)
- Read these DOM values:
  - filename
  - verdict (REAL/FAKE/UNCERTAIN)
  - p_fake (the big number)
  - confidence percentage
  - scoring_model (the model name shown)
  - cnn_active (YES/NO)
  - faces_found ("X/Y")
- Click "Analyze Another" to reset for the next video

Print results as a JSON array to stdout. DOM text only — no screenshots. Exit code 0 on success, non-zero on timeout/error.

### 3. Write the loop helper script
Create `scripts/run_loop_iteration.sh` (or `.ps1` for native Windows):
- Kill any processes on ports 8000 and 5173
- Start `uvicorn backend.main:app --reload --port 8000` in background, log to `/tmp/backend.log` (or `%TEMP%\backend.log`)
- Start `npm run dev` in background (in the frontend directory), log to `/tmp/frontend.log`
- Poll up to 30s for `curl http://localhost:8000/health` to return 200 AND `curl http://localhost:5173` to return 200
- Run the Playwright test, capture JSON output
- Kill both servers cleanly
- Print the JSON output

### 4. Wait for Panes 1 and 2 to declare done
Don't run the actual iteration loop until both other panes have posted "done" in shared notes. Until then:
- Build out your test scaffolding
- Verify your Playwright test runs against a mocked response (`api_contract.md` has the expected shape — mock it)
- Verify the helper script can start/stop servers cleanly

### 5. Run the agentic loop (only after panes 1 and 2 are done)

Max 5 iterations. For each iteration:

1. Run the loop script. Capture JSON + tail last 50 lines of backend log.

2. Check success criteria — ALL must be true:
   - `cnn_active == "YES"` on BOTH videos (or `true` if reading API directly)
   - `scoring_model` is `stacked` or `stacked_with_blink` (not `lr` alone) on BOTH videos
   - `p_fake` ≤ 0.35 on BOTH videos
   - Backend log has no ERROR or CRITICAL lines

3. If success → stop and report.

4. If not success → diagnose what's broken, then **decide which pane should fix it**:
   - CNN not loading, registry empty, API returning wrong shape → Pane 1
   - UI showing wrong field, crashing on missing field, hardcoded text → Pane 2
   - Test itself broken → fix yourself
   
   Post the diagnosis to shared notes. Do NOT modify backend or frontend code yourself.

5. Re-run the loop after the appropriate pane reports a fix.

## Stop conditions
- ✅ Success: both videos meet criteria
- 🛑 5 iterations without success — report trend (did p_fake decrease? did cnn_active ever flip?)
- 🛑 CNN checkpoint genuinely missing — Pane 1's problem, escalate
- 🛑 Same error twice after a fix attempt — stop, escalate
- 🛑 Test videos missing or unuploadable — stop

## Reporting format
After every iteration, print:
```
=== ITERATION N ===
IMG_0601: verdict=X, p_fake=X.XX, cnn_active=Y/N, model=...
IMG_9666: verdict=X, p_fake=X.XX, cnn_active=Y/N, model=...
Backend log errors: <count> / <first error if any>
Pass: ✓ / ✗
Diagnosis (if fail): <what's broken, which pane should fix it>
```

Final report when stopping:
```
=== FINAL ===
Status: SUCCESS / FAILURE / BLOCKED
Iterations: N
Trend: <did p_fake decrease over iterations?>
Final scores: IMG_0601=X.XX, IMG_9666=X.XX
Baseline (Modules 1+3 without CNN): 0.20–0.30
Open issues for user: <if blocked or failed>
```

## Constraints
- Do not edit backend or frontend code. Period. If broken, escalate to the right pane.
- Do not modify the test videos.
- Do not modify the API contract.
- Do not change the success thresholds to make tests pass. 0.35 is 0.35.
- Always kill servers between iterations — zombie processes will give misleading results.
- Use DOM text reading, not screenshots. Faster, more reliable, no vision tokens burned.

## Done means
- Playwright test exists and runs cleanly
- Helper script starts/stops servers reliably
- Iteration loop completed with status SUCCESS or a clear actionable failure report
- All output saved to shared notes so user can review
