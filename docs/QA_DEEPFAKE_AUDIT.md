# Deepfake Detection Module — QA & Adversarial Audit Report

**Audit date (UTC):** 2026-05-28 (FaceSwap + ADV/OOD results added 2026-05-28)  
**Auditor role:** Senior QA (AI / Computer Vision)  
**Repository:** `Group-8-AISC-main` @ `9dc8bc6`  
**Raw results:** [`docs/qa_audit_results.json`](qa_audit_results.json)  
**Runner:** [`scripts/qa_audit_runner.py`](../scripts/qa_audit_runner.py)

---

## Executive summary

| Area | Result | Notes |
|------|--------|-------|
| **Module 1 (EAR)** | **PARTIAL** | Video-level EAR scorer integrated; many fake video IDs still unresolved to source files (`ear=0.5` fallback in cache) |
| **Module 2 (JPEG artifact)** | **PARTIAL** | Stable, but discriminative direction remains weak/inverted on this run (M2-02, M2-05) |
| **Module 3 (FFT + Laplacian)** | **PASS** (unit) | FFT/Laplacian unit behavior is stable |
| **Ensemble (offline)** | **PASS** (pipeline) | `ensemble.py` runs and exports pickle; validation AUC remains 1.0 on current sampled frame set |
| **API (`/analyze`)** | **PASS** (contract) | Loads `ensemble_model.pkl`; `/health` exposes `threshold`, `uncertain_band`, `verdict_hi`, `verdict_lo` |
| **Threshold alignment** | **PASS** | QA runner now uses calibrated inference bands from pickle (no hardcoded 0.6/0.4 path) |
| **E2E video (proxy)** | **FAIL** for discrimination | Proxy Haar videos still collapse to constant FAKE outcomes (50% accuracy, 100% false-FAKE on real bucket) |

### Top 5 risks (updated after FaceSwap + ADV/OOD completion)

1. **FaceSwap fundamental coverage gap:** handcrafted features have near-zero signal for geometry-only manipulations (EAR Δ≈−0.001, artifact/Laplacian ≈0). Deepfakes/Face2Face models classify 100% of FaceSwap fakes as real (ADV-20). CNN fallback gate is met but not yet activated in production.
2. **EAR-dominated over-separability on Deepfakes/Face2Face:** AUC=1.0 is driven entirely by EAR feature (fake EAR=0.5 fallback). Real discriminative power of FFT/artifact/Laplacian alone is weak (Deepfakes AUC on these three features alone is likely 0.55–0.65).
3. **OOD vulnerability — heavy smoothing:** ADV-22 bilateral filter causes 61% false-fake rate on real images with FaceSwap model. Any stylised or heavily post-processed input degrades reliability severely.
4. **Proxy E2E remains non-representative:** video protocol still uses synthetic Haar-friendly videos, not FF++ source MP4 uploads.
5. **Decision boundary mismatch across families:** Deepfakes threshold=0.015, Face2Face threshold=0.050, FaceSwap threshold=0.690 — the model cannot be deployed with a single threshold across manipulation families.

### Go / no-go

| Gate | Verdict |
|------|---------|
| **Research demo** (offline plots + honest limitations) | **GO** — with explicit caveats about extraction shortfall + proxy E2E |
| **API demo** (trained model + stable threshold contract) | **GO (conditional)** — threshold contract fixed; still communicate extreme threshold and proxy-video limitation |
| **Security / production claim** | **NO-GO** — requires real FF++ video-level protocol and broader adversarial/OOD coverage |

---

## 1. Environment

| Item | Value |
|------|-------|
| Python | 3.14.0 |
| Git | `9dc8bc6` |
| `data/FaceForensics++_C23/` | **Present** |
| `data/manifest.csv` | **Present** — 778 rows (391 real, 387 fake) |
| `data/ensemble_model.pkl` | **Present** |
| `inference_threshold` | **0.05** |
| `uncertain_band` | **0.10** |
| `verdict_hi` / `verdict_lo` | **0.15 / 0.00** |
| `TARGET_PER_CLASS` | 500 (not fully reached in this run) |

---

## 2. Code/contract status

| ID | Finding | Status |
|----|---------|--------|
| DEF-01 | EAR stub in inference path | **Partially resolved** — EAR integrated, but fallback still occurs for unresolved fake video IDs |
| DEF-02 | Missing pickle export | **Resolved** — `ensemble_model.pkl` exported and loaded by API |
| DEF-03 | Threshold mismatch (train vs API) | **Resolved** — API and QA use calibrated threshold/bands from pickle |
| DEF-04 | Scorer exception fallbacks | Open (low-medium) |
| DEF-05 | Feature semantics on C23 weak/inverted | Open |
| DEF-06 | Haar-only detection constraints | Open |
| DEF-07 | FF++ dataset absent | **Resolved** |

---

## 3. Phase 1 — Module tests (from `qa_audit_results.json`)

### 3.1 Module 2 — JPEG artifact

| ID | Pass | Evidence |
|----|------|----------|
| M2-01 | **PASS** | mean real ≈ **0.0498** |
| M2-02 | **FAIL** | mean fake **0.0489** < real **0.0498** |
| M2-03 | **PASS** | uniform gray → **0.0** |
| M2-04 | **PASS** | noise → **1.0** |
| M2-05 | **FAIL** | Q50 **0.012** < Q95 **0.055** |
| M2-06 | **PASS** | double JPEG delta ≈ **0.0** |
| M2-07 | **PASS** | native/upscale gap ≈ **0.013** |
| M2-08 | **PASS** | channel-order sensitivity bounded |

### 3.2 Module 3 — FFT/Laplacian

| ID | Pass | Evidence |
|----|------|----------|
| M3F-01..05 | **PASS** | fake FFT mean **0.8454** vs real **0.8146**, blur/sharpen checks pass |
| M3L-01..04 | **PASS** | real Laplacian mean **0.4181** vs fake **0.3653**, blur/flat checks pass |

### 3.3 Module 1 status

`module1_detection: PASS — video-level EAR scorer integrated`, but extraction logs show many fake EAR entries written as `0.5000 (no source video)` because fake `video_id` mapping is not resolvable to source path.

---

## 4. Phase 2 — Ensemble & ML QA

### 4.1 Data integrity

| ID | Pass | Evidence |
|----|------|----------|
| ML-01 | **PASS** | No `video_id` leakage |
| ML-02 | **PASS** | train fake rate **0.4724**, val fake rate **0.5926** |
| ML-06 | **WARN** | saturation **100%** |
| ML-07 | **PASS** | pickle exists |

### 4.2 Metrics (current run)

| Metric | Value |
|--------|-------|
| Samples | **778** |
| Val AUC | **1.0000** |
| Balanced acc @ `best_t_ba` | **1.0000** |
| `best_t_ba` | **0.05** |
| Accuracy @ `verdict_hi` (0.15) | **1.0000** |
| 5-fold CV AUC (mean ± std) | **1.0000 ± 0.0000** |

Independent validation cross-check (`data/experiments/ml03_validation_summary.json`):
- Deepfakes: primary log reports `AUC=1.0000`, `accuracy=1.0000`, `balanced_accuracy=1.0000`, `F1=1.0000` at threshold `0.0500`; independent recompute from `module3_features.csv` keeps `AUC=1.0000` but yields `accuracy=0.9938`, `balanced_accuracy=0.9948`, `F1=0.9948` at threshold `0.9899`.
- Face2Face: primary and independent recompute both give `AUC=1.0000`, `accuracy=1.0000`, `balanced_accuracy=1.0000`, `F1=1.0000`; selected threshold differs (`0.0500` primary vs `0.0111` validation) with no metric impact on this split.
- FaceSwap: **COMPLETED** — `AUC=0.4371`, `accuracy=0.4909`, `balanced_accuracy=0.5691`, `F1=0.2632` at threshold `0.6900`; 5-fold CV AUC `0.6056 ± 0.0858`. Features provide near-zero discrimination on this manipulation family (see §4.3 for deltas).

### 4.3 Feature deltas (fake − real) by manipulation family

| Feature | Deepfakes Δ | Face2Face Δ | FaceSwap Δ |
|---------|-------------|-------------|------------|
| ear | **−0.4988** | **−0.4988** | **−0.0011** |
| artifact | **−0.0019** | **−0.0019** | **+0.0003** |
| fft | **+0.0179** | **+0.0179** | **−0.0400** |
| laplacian | **−0.0565** | **−0.0565** | **~0.0000** |

**FaceSwap note:** EAR separation collapses entirely (Δ≈−0.001) because FaceSwap fakes also have near-natural blink dynamics (EAR≈1.0 vs real EAR=1.0). Artifact and Laplacian deltas are negligible. FFT shows a small signal (Δ=−0.040) but insufficient for reliable discrimination. This is the root cause of FaceSwap AUC=0.4371.

---

## 5. Phase 3 — Adversarial checks

### 5.1 Per-model adversarial baseline (Deepfakes/Face2Face — threshold from pickle)

Decision bands for Deepfakes model (`threshold=0.05`, `verdict_hi=0.15`):

| Check | Result |
|------|--------|
| Baseline false-REAL on fakes | **0%** |
| ADV-01 blur false-REAL | **0%** |
| ADV-02 JPEG false-REAL | **0%** |
| ADV-03 down-up false-REAL | **0%** |
| ADV-07 combo false-REAL | **0%** |
| Baseline false-FAKE on reals | **100%** |
| ADV-10 sharpen false-FAKE (>= verdict_hi) | **100%** |

Interpretation: with `verdict_hi=0.15`, this run is highly FAKE-biased; adversarial statistics are not meaningful for deployment decisions on Deepfakes/Face2Face.

### 5.2 Cross-family + OOD/ADV-20..23 (from `data/experiments/ml03_adv_ood_results.json`)

See §11.3 for the complete matrix with source JSON values.

---

## 6. Phase 4 — API and E2E

### 6.1 API checks

| ID | Result |
|----|--------|
| API-01 | PASS (empty upload → 422) |
| API-02 | PASS (corrupt bytes → 422) |
| API-04 | PASS (60-frame latency ~6.143s in runner) |
| API-05 | PASS (n=1 vs n=12 variance ~0.0005) |
| API-06 | PASS (`model_loaded: true`, threshold fields present) |
| API-07 | PASS (response size minimal in harness) |

### 6.2 Proxy E2E

| Metric | Value |
|--------|-------|
| Accuracy | **50%** |
| False-REAL on fake | **0%** |
| False-FAKE on real | **100%** |
| Typical `prob_fake_mean` | ~**0.9935** constant |

The E2E protocol is still proxy-based (drawn Haar-friendly video), not a direct FF++ MP4 upload protocol.

---

## 7. Defect log (updated)

| ID | Severity | Status | Notes |
|----|----------|--------|-------|
| DEF-01 | High | **Partial** | EAR integrated, but many fake IDs unresolved to source video → 0.5 fallback |
| DEF-02 | High | **Closed** | `ensemble_model.pkl` export + load confirmed |
| DEF-03 | Medium | **Closed** | Threshold contract aligned in API/runner (`threshold` + band) |
| DEF-04 | Low-Med | Open | Exception fallback behavior unchanged |
| DEF-05 | Medium | Open | Module 2 direction/informativeness still weak |
| DEF-06 | Medium | Open | Haar and proxy-video constraints remain |
| DEF-07 | High | **Closed** | FF++ dataset present and used (`synthetic_data_used: false`) |
| DEF-08 | Medium | Open | Proxy E2E constant-score behavior persists |

---

## 8. Risk register

| Risk | Severity | Status |
|------|----------|--------|
| FaceSwap coverage gap — EAR/artifact/Laplacian all near-zero signal | **Critical** | Open — requires CNN fallback or new feature set |
| ADV-20 cross-family false-real=1.0 for FaceSwap fakes | **Critical** | Open — handcrafted models cannot be deployed without family routing |
| ADV-22 OOD false-fake=61% on bilateral-filtered real images | High | Open |
| EAR-dominated Deepfakes/Face2Face AUC=1.0 (artificial separability) | High | Open |
| Inference threshold mismatch across families (0.015 / 0.050 / 0.690) | High | Open |
| EAR fake-video mapping fallback to 0.5 | High | Open (root cause of EAR dominance) |
| Artifact feature direction weak/inverted | Medium | Open |
| Proxy E2E not representative | High | Open |

---

## 9. Recommendations (priority)

1. **Activate CNN fallback immediately** for FaceSwap: FaceSwap CV AUC=0.6056 meets the `< 0.65` gate condition. Set `HANDCRAFTED_VALIDATION_AUC=0.6056` in backend env to trigger `cnn_fallback_active=true`.
2. **Do not serve Deepfakes/Face2Face handcrafted models as cross-family classifiers**: ADV-20 shows 100% false-real rate for FaceSwap fakes. Add a manipulation-type routing layer or treat model output as family-specific.
3. Fix fake `video_id` to source-video mapping so EAR is computed from real fake source videos (eliminate broad 0.5 fallback). This is the root cause of artificial AUC=1.0 on Deepfakes/Face2Face.
4. Address ADV-22 OOD vulnerability: the bilateral-filter false-fake rate of 61% means any heavily processed image will be wrongly flagged. Consider adding texture-invariant features or input normalisation.
5. Run a **true FF++ video-level E2E protocol** on source MP4s (not synthetic Haar proxy).
6. Recalibrate thresholds with a consistent policy across all three families; the current range (0.015–0.690) indicates family-specific overfitting rather than a generalised threshold.

---

## 10. Re-run commands

```powershell
cd "d:\New folder (2)\Group-8-AISC-main\Group-8-AISC-main"
python inspect_dataset.py
python ensemble.py
python scripts/qa_audit_runner.py
```

For API smoke:

```powershell
cd backend
pip install -r requirements.txt python-multipart
uvicorn main:app --port 8000
```

---

*End of audit report.*

---

## 11. Merge decision (consolidated evidence state)

This section replaces the pending scaffold with the currently available artifacts as of this audit.

### 11.1 Required artifact status

| Artifact | Path | Status | Notes |
|----------|------|--------|-------|
| ML-03 independent validation summary | `data/experiments/ml03_validation_summary.json` | **AVAILABLE** | All three family metrics now present |
| Deepfakes family output | `data/experiments/deepfakes/manifest.csv`, `data/experiments/deepfakes/module3_features.csv` | **AVAILABLE** | Validation recompute: AUC **1.0000**, balanced acc **0.9948**, threshold **0.9899** |
| Face2Face family output | `data/experiments/face2face/manifest.csv`, `data/experiments/face2face/module3_features.csv` | **AVAILABLE** | Validation recompute: AUC **1.0000**, balanced acc **1.0000**, threshold **0.0111** |
| FaceSwap family output | `data/experiments/faceswap/module3_features.csv`, `data/experiments/faceswap/ensemble_model.pkl` | **AVAILABLE** | AUC **0.4371** (val), CV AUC **0.6056 ± 0.0858**; handcrafted features near-useless for this family |
| ADV/OOD aggregate (`ADV-20..23`) | `data/experiments/ml03_adv_ood_results.json` | **AVAILABLE** | Generated 2026-05-28 @ git `9dc8bc6` |

### 11.2 ML-03 family results (baseline + manipulation splits)

| Track | AUC | Accuracy | Balanced Acc | F1 | CV AUC (mean±std) | Threshold | Status |
|-------|-----|----------|--------------|----|--------------------|-----------|--------|
| Deepfakes (validation recompute) | **1.0000** | **0.9938** | **0.9948** | **0.9948** | **1.0000 ± 0.0000** | **0.9899** | COMPLETE |
| Face2Face (validation recompute) | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000 ± 0.0000** | **0.0111** | COMPLETE |
| FaceSwap | **0.4371** | **0.4909** | **0.5691** | **0.2632** | **0.6056 ± 0.0858** | **0.6900** | COMPLETE — **FAIL** (AUC < 0.65 CNN gate) |

**FaceSwap interpretation:** All four handcrafted features fail to discriminate FaceSwap from real. Feature deltas are near-zero across EAR, artifact, and Laplacian; FFT provides a weak signal (Δ=−0.040) but insufficient alone. Val AUC=0.4371 is below random; 5-fold CV AUC=0.6056 confirms the signal is minimal. This manipulation family (geometry-only face swap) bypasses all three detection modules in their current form.

### 11.3 ADV-20..23 quantitative matrix (from `data/experiments/ml03_adv_ood_results.json`)

**ADV-20 — cross-family false-real rate** (model trained on row-family applied to fakes from column-family):

| Model trained on \ Tested on fakes from | Deepfakes | Face2Face | FaceSwap |
|----------------------------------------|-----------|-----------|----------|
| Deepfakes (threshold=0.015) | — | **0.00** | **1.00** ⚠ |
| Face2Face (threshold=0.050) | **0.00** | — | **1.00** ⚠ |
| FaceSwap (threshold=0.690) | **0.00** | **0.00** | — |

ADV-20 interpretation:
- Deepfakes and Face2Face models (EAR-dominated, threshold≈0.015–0.05) classify all FaceSwap fakes as real because FaceSwap fakes retain natural blink dynamics (EAR≈1.0). This is a critical cross-family coverage gap.
- FaceSwap model (threshold=0.690) correctly flags Deepfakes and Face2Face fakes as fake, but only because those fakes have EAR=0.5 which happens to score high on the noise-trained FaceSwap model — not a reliable generalisation signal.

**ADV-21 — screen-recording proxy** (downsample 112px + upsample + JPEG Q60, n=80 real images, FaceSwap model, threshold=0.69):

| Metric | Value |
|--------|-------|
| false-fake rate | **0.05** (5%) — PASS |
| score mean ± std | 0.5475 ± 0.1204 |

ADV-21 interpretation: Screen-recording compression does not significantly shift real image scores. 5% false-fake rate is acceptable.

**ADV-22 — cartoon / OOD style** (heavy bilateral filter d=15, σ=80, n=80 real images, FaceSwap model, threshold=0.69):

| Metric | Value |
|--------|-------|
| false-fake rate | **0.6125** (61%) — **FAIL** ⚠ |
| score mean ± std | 0.7066 ± 0.1230 |

ADV-22 interpretation: Heavy texture smoothing (bilateral filter) causes 61% of real face crops to cross the fake threshold. The FFT anomaly score is sensitive to smoothing; a bilateral filter reduces high-frequency content in a way the model conflates with deepfake generation artifacts. This OOD vulnerability is high-severity for any deployment context where input faces may be processed through artistic filters or heavy compression.

**ADV-23 — partial-face occlusion** (bottom 40% blacked out, n=80 real images, FaceSwap model, threshold=0.69):

| Metric | Value |
|--------|-------|
| false-fake rate | **0.2625** (26%) — ELEVATED ⚠ |
| score mean ± std | 0.5912 ± 0.1556 |

ADV-23 interpretation: Occluding the lower face region causes 26% false-fake rate. Partial occlusion alters Laplacian variance and FFT energy distribution in the image, shifting scores toward the fake region.

### 11.4 Blocking thresholds and release gates

| Gate metric | Threshold | Current evidence | Gate |
|-------------|-----------|------------------|------|
| Handcrafted validation AUC (Deepfakes) | `>= 0.65` | AUC **1.0000** | **PASS** |
| Handcrafted validation AUC (Face2Face) | `>= 0.65` | AUC **1.0000** | **PASS** |
| Handcrafted validation AUC (FaceSwap) | `>= 0.65` | CV AUC **0.6056** (below gate); val AUC **0.4371** | **FAIL — CNN fallback triggered** |
| ADV-20 cross-family coverage | false-real < 0.10 for all pairs | Deepfakes/Face2Face models: false_real=**1.00** on FaceSwap fakes | **FAIL** |
| ADV-22 OOD false-fake | < 0.20 | FaceSwap model: **0.6125** on bilateral-filtered real images | **FAIL** |
| Manipulation-family coverage | All three families complete | All three complete with artifacts | **PASS** |
| ADV/OOD artifact completeness | `ml03_adv_ood_results.json` present | Present at `data/experiments/ml03_adv_ood_results.json` | **PASS** |

### 11.5 Final GO/NO-GO decision

| Track | Decision | Evidence snapshot | Blocking risks |
|-------|----------|-------------------|----------------|
| handcrafted-only | **NO-GO** | FaceSwap AUC=0.4371 (below random on val); cross-family false-real=1.0 for FaceSwap fakes with Deepfakes/Face2Face models; ADV-22 bilateral OOD false-fake=61% | FaceSwap coverage gap is fundamental — not a calibration issue, handcrafted features lack signal for geometry-only manipulation |
| cnn_fallback rollout contract | **GO (integration-ready, conditional runtime verification)** | CNN fallback gate condition is met: FaceSwap CV AUC=0.6056 < 0.65 gate triggers `cnn_fallback_active=true`; contract implemented in backend/frontend/tests; API smoke contract artifact present | Activation-path runtime checks still required in checklist |

**Recommendation:** handcrafted-only **NO-GO**, cnn_fallback-priority **GO (conditional)**.  
**Reason:** FaceSwap results confirm the fundamental limitation — geometry-only face swaps bypass all three handcrafted detectors (EAR collapses to Δ≈−0.001, artifact/Laplacian near-zero, FFT weak). The CNN fallback gate (`handcrafted_auc < 0.65`) is met by FaceSwap's CV AUC of 0.6056. For Deepfakes and Face2Face, handcrafted AUC=1.0 but ADV-20 shows complete cross-family failure when applied to FaceSwap fakes, indicating the system cannot be deployed as a general-purpose deepfake detector without CNN fallback active.

---

## 12. Workstream B — CNN fallback contract (implemented + mapped)

### 12.1 Contract decisions

- **Gating rule (hard):** `handcrafted_auc < 0.65 -> cnn_fallback`.
- **Activation override (ops):** `FORCE_CNN_FALLBACK=1` activates fallback regardless of AUC.
- **API response `model_used` states:** `ensemble_learned`, `equal_weights`, `cnn_fallback`, `cnn_fallback_degraded`.
- **`/health` exposure additions:**
  - `handcrafted_auc`
  - `cnn_fallback_auc_gate` (fixed `0.65`)
  - `cnn_fallback_active` (boolean)
  - `cnn_fallback_reason` (`off | gated_auc | forced`)
  - `model_used_states` (enumerated contract list)

### 12.2 File-level integration map (minimal diffs)

| File | Minimal change |
|------|----------------|
| `backend/main.py` | Add fallback gate constant + resolver; read `HANDCRAFTED_VALIDATION_AUC` and `FORCE_CNN_FALLBACK`; set app-state fallback flags; expose new `/health` fields; pass `cnn_fallback_active` and optional `cnn_infer` into `analyze_video()` |
| `backend/detector.py` | Add `_resolve_model_used()` helper; accept `cnn_fallback_active` and `cnn_infer`; emit `cnn_active` and `model_used` with fallback/degraded states while preserving handcrafted scoring path |
| `backend/models.py` | Expand `model_used` contract description to include fallback states; update `cnn_active` semantics from hardcoded false to runtime-selected fallback indicator |
| `frontend/src/types.ts` | Extend `AnalysisResponse.model_used` union with `cnn_fallback` and `cnn_fallback_degraded` |
| `frontend/src/components/Verdict.tsx` | Render explicit labels for fallback states (`CNN fallback`, `CNN fallback (degraded)`) |
| `backend/cnn_detector.py` (optional extraction) | Add non-blocking loader contract (`load_cnn_infer`) returning callable-or-None so backend can ship fallback toggles before CNN artifact finalization |

### 12.2.1 Implementation verification snapshot

- Unit tests exist for fallback resolver and model-used propagation: `backend/tests/test_cnn_fallback_contract.py`.
- Backend contract fields appear in health schema and smoke artifact expectation set: `data/experiments/api_smoke_results.json`.
- Frontend enum + verdict labels include `cnn_fallback` and `cnn_fallback_degraded`.

### 12.3 Rollout / rollback plan

**Rollout (safe incremental):**
1. Deploy backend with fallback gate + health contract first; leave `HANDCRAFTED_VALIDATION_AUC=1.0`.
2. Verify `/health` includes gate fields and `cnn_fallback_active=false`.
3. Set `HANDCRAFTED_VALIDATION_AUC` from latest ML audit output.
4. If `< 0.65`, confirm fallback flips active and `model_used` returns fallback state.
5. If urgent mitigation needed, temporarily set `FORCE_CNN_FALLBACK=1`.

**Rollback (single-step):**
1. Set `FORCE_CNN_FALLBACK=0`.
2. Set `HANDCRAFTED_VALIDATION_AUC>=0.65` (or unset variable to default path).
3. Restart service and verify `/health.cnn_fallback_active=false`.
4. Confirm `/analyze` returns `model_used` in handcrafted states only.

### 12.4 Acceptance tests (activation/deactivation)

- **B-AT-01 (activate by gate):** With `HANDCRAFTED_VALIDATION_AUC=0.50`, `/health.cnn_fallback_active=true`, reason=`gated_auc`.
- **B-AT-02 (deactivate by gate):** With `HANDCRAFTED_VALIDATION_AUC=0.80`, `/health.cnn_fallback_active=false`, reason=`off`.
- **B-AT-03 (force override):** With `FORCE_CNN_FALLBACK=1`, fallback remains active regardless of AUC.
- **B-AT-04 (analyze propagation):** `/analyze` passes app fallback state into detector and response metadata (`cnn_active`, `model_used`) reflects activation path.
