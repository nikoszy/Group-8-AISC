Agentic Debug Loop: Get full ensemble (CNN + Module 1 + Module 3 + LR) producing correct scores on real videos
Context
We are building Group 8 AISC Deepfake Detector.

Backend: FastAPI (backend/main.py, backend/detector.py) — start with uvicorn backend.main:app --reload --port 8000
Frontend: React (ui-migration branch / worktree) — start with npm run dev (Vite, usually port 5173)
4 detection modules: MRL blink (Module 1), JPEG artifact (Module 2), FFT+Laplacian+temporal+rPPG (Module 3), and EfficientNet-B0 CNN
Stacking blend: frame_prob = α × CNN + (1-α) × LR, with α from stacking_ensemble.py

The problem we are solving
When uploading real videos through the UI, the result card currently shows CNN ACTIVE: NO and the scoring model is lr — meaning we are falling back to LR-only predictions. This produces wrong scores on real videos.
Baseline established by the user: with Module 1 (blink) and Module 3 (FFT+Laplacian+temporal+rPPG) integrated, but WITHOUT the CNN active, two known real videos scored p_fake in the 20–30% range.
So Module 1 + Module 3 alone are already getting it right. The current LR-only path through the UI is producing HIGHER (worse) scores than that baseline — meaning the LR ensemble is actively making things worse than the handcrafted modules alone, because:

The CNN isn't loading (CNN ACTIVE: NO), so there's no α × CNN + (1-α) × LR blend happening
Module 1 and Module 3's signals may not be properly feeding into the LR feature vector
Or LR's weights are dominated by features that misbehave on real videos without the CNN counterweight

Goal: Get the full ensemble (CNN + Module 1 + Module 3 + LR, properly stacked) to score these real videos at ≤ 0.35. The Module 1 + 3 baseline of 20–30% is the floor — the full ensemble should match or beat it, never be worse.
Test videos
Two known REAL videos in the user's Downloads folder:

C:\Users\satya\Downloads\IMG_0601.MOV
C:\Users\satya\Downloads\IMG_9666.MOV

Environment detection (do this first): Run uname -a and pwd. If you're in WSL, convert the paths to /mnt/c/Users/satya/Downloads/IMG_0601.MOV and /mnt/c/Users/satya/Downloads/IMG_9666.MOV. If you're in native Windows / PowerShell / cmd, use the original C:\... paths. Use whichever format matches your shell — don't mix.
If either file is missing, stop immediately and report.
Your task
Run an iterative debug loop. Each iteration: make a fix → run the full stack → upload BOTH test videos via Playwright → read the results → decide what to fix next. Stop when success criteria are met OR after 5 iterations.
Setup (one time, before iteration 1)

Confirm both test videos exist at the paths above (in the format matching your environment).
Install Playwright if needed:

   npm install -D @playwright/test
   npx playwright install chromium

Write a Playwright test script at tests/e2e/upload_and_read.spec.ts that:

For EACH of the two test videos (IMG_0601.MOV and IMG_9666.MOV):

Navigates to the frontend (http://localhost:5173)
Uploads the video via the file input
Waits for the result card to appear (look for "ANALYSIS COMPLETE" or the verdict text)
Reads these DOM values:

filename
verdict (REAL/FAKE/UNCERTAIN)
p_fake (the big number)
confidence (the percentage)
scoring_model (lr / stacked / etc.)
cnn_active (YES/NO)
faces_found (e.g. "9/12")


Clicks "Analyze Another" or navigates back to reset for the next video


Prints results as a JSON array with one entry per video to stdout
Does NOT take screenshots — DOM text only, it's faster and more reliable
Exits with code 0 on success, non-zero on timeout/error


Write a helper script at scripts/run_loop_iteration.sh (or .ps1 if on native Windows) that:

Kills any existing uvicorn/vite processes on ports 8000/5173
Starts uvicorn in the background, logs to /tmp/backend.log (or %TEMP%\backend.log on Windows)
Starts npm run dev in the background, logs to /tmp/frontend.log
Waits up to 30s for both to be responsive (curl http://localhost:8000/health and http://localhost:5173)
Runs the Playwright test
Captures the JSON output
Kills both servers cleanly at the end
Returns the JSON for the agent to parse



Iteration loop (max 5 iterations)
For each iteration:

Run the loop script. Capture the JSON array of results + tail the last 50 lines of the backend log.
Check success criteria (ALL must be true to stop):

cnn_active == "YES" on BOTH videos
scoring_model is stacked or stacked_with_blink (not lr alone) on BOTH videos
p_fake <= 0.35 on BOTH test real videos (both need to pass — a fix that works on one but not the other is fragile and doesn't count)
Backend log has no ERROR or CRITICAL lines from the inference path


If success → stop and report.
If not success → diagnose and fix one thing, then loop. Use the backend log + the JSON to figure out what's wrong. Common failure modes to check, in order:

CNN checkpoint missing → check ls artifacts/ for .pt file. If missing, STOP and report — agent can't fix this, user needs to provide it.
CNN load silently failing → grep for try/except around CNN loading in backend/detector.py, src/cnn_runner.py. Make exception loud (log full trace, re-raise on startup).
Device mismatch → if log shows CUDA errors, change torch.load to use map_location="cpu".
Module 1 or Module 3 signals not reaching LR feature vector → trace from src/mrl/inference.py (Module 1) and src/freq_analysis/ + src/temporal_scorer.py + src/rppg_scorer.py (Module 3) through to where LR features are assembled. If the LR-only path is scoring HIGHER than Modules 1+3 alone, those module signals are either not being passed in or are being overridden.
Stacking alpha not loading → check if artifacts/stacking_bundle.pkl exists and alpha_reliable=True.
Registry has only lr → run stacking_ensemble.py to register the stacked model, then ensure model_registry.json marks the highest-F1 model active.
Frontend not reading new API fields → verify the React component pulls cnn_active and scoring_model from the response, not from hardcoded text.


Make ONE targeted fix per iteration. Don't shotgun changes — we need to know what worked.

Stop conditions
Stop the loop and report back if ANY of these happen:

✅ Success criteria met on BOTH videos (report what worked)
🛑 5 iterations completed without success (report trend: did p_fake decrease across iterations? did cnn_active flip to YES at any point?)
🛑 CNN checkpoint genuinely missing (report — user has to provide it)
🛑 Same error appears in 2 consecutive iterations after a fix attempt (report — you're stuck, don't burn more cycles)
🛑 Either test video file is missing or can't be uploaded by Playwright

Reporting format
After stopping (success or failure), print a summary in this exact format:
=== DEBUG LOOP REPORT ===
Iterations run: N
Final status: SUCCESS | FAILURE | BLOCKED

Iteration-by-iteration p_fake trend:
  iter 1: 
    IMG_0601: p_fake=X.XX, cnn_active=YES/NO, model=...
    IMG_9666: p_fake=X.XX, cnn_active=YES/NO, model=...
  iter 2: ...
  ...

Fixes applied:
  iter 1: <one line description>
  iter 2: ...

Current state:
  IMG_0601: cnn_active=Y/N, scoring_model=..., p_fake=X.XX
  IMG_9666: cnn_active=Y/N, scoring_model=..., p_fake=X.XX
  baseline (Module 1 + Module 3 without CNN): 0.20–0.30
  delta from baseline: ...

What I changed (files):
  - path/to/file1
  - path/to/file2

If FAILURE or BLOCKED, what I think the user needs to do:
  <specific actionable next step>
Constraints

Do not modify the test videos or the Playwright test once it's working — those are the ground truth.
Do not change Module 1 or Module 3's scoring logic to "make the number lower" — that's cheating. They already work; the goal is to make the full ensemble use them properly, not to re-tune them.
Do not skip the server cleanup between iterations — zombie processes will give misleading results.
Do not turn --reload off on uvicorn — we need it for fast iteration.
Both videos must pass. Don't claim success if only one is under 0.35.
If you hit a checkpoint/data/env problem the agent fundamentally can't solve (missing files, missing GPU, missing dataset), stop immediately and report. Don't try to work around it.

Start by

Detecting whether you're in WSL or native Windows, and picking the correct path format for the two videos.
Confirming both videos exist at those paths.
Writing the Playwright test and the loop script.
Running iteration 1 to establish a baseline of where the system currently is on BOTH videos.
Then begin diagnosis + fixes.