# Integration Readiness Checklist (Multitask Consolidation)

Date: 2026-05-27
Scope: coordination QA across recent worker outputs (no logic edits).

## 1) Safe-to-merge groups

- Group A: CNN fallback contract surfaces are aligned.
  - `backend/main.py`
  - `backend/detector.py`
  - `backend/models.py`
  - `backend/tests/test_cnn_fallback_contract.py`
- Group B: Dataset extraction ID normalization appears internally consistent.
  - `inspect_dataset.py`
- Group C: Coordination doc update with no policy math changes.
  - `docs/CHANGES_MULTITASK.md`

## 2) Files requiring manual reconciliation

- Threshold/policy semantics and naming drift:
  - `ensemble.py`
  - `scripts/qa_audit_runner.py`
  - `docs/threshold_operating_points.md`
  - `docs/threshold_operating_points.json`
  - `docs/QA_DEEPFAKE_AUDIT.md`
  - `docs/CNN_FALLBACK_TEST_CHECKLIST.md`
  - `docs/qa_audit_results.json`
- Why reconcile:
  - Field-name mismatch: `false_fake_rate_real` (`ensemble.py`) vs `false_fake_on_real` (`threshold_policy_report` docs outputs).
  - Cap mismatch: `MAX_FALSE_FAKE_RATE=0.10` (`ensemble.py`) vs `POLICY_MAX_FALSE_FAKE_ON_REAL=0.35` (`scripts/qa_audit_runner.py`) vs `0.050` stated in `docs/threshold_operating_points.md`.
  - Threshold drift in docs: `0.015` candidate (`threshold_operating_points.*`) vs served contract `0.05` (`qa_audit_results.json`, `QA_DEEPFAKE_AUDIT.md`, checklist notes).

## 3) Recommended integration order

1. Lock canonical policy schema (single naming set and metadata keys) for:
   - threshold fields
   - uncertain band
   - policy metadata (`max_*`, `false_*`, `constraint_status`, `mode`)
2. Reconcile `ensemble.py` and `scripts/qa_audit_runner.py` to that schema and one cap source of truth.
3. Regenerate/refresh policy artifacts from code (not manual edits first):
   - `docs/threshold_operating_points.json`
   - `docs/threshold_operating_points.md`
   - `docs/qa_audit_results.json`
4. Update narrative docs to reflect regenerated artifacts:
   - `docs/QA_DEEPFAKE_AUDIT.md`
   - `docs/CNN_FALLBACK_TEST_CHECKLIST.md`
5. Keep backend contract files as-is unless schema keys change; then apply minimal key rename propagation.

## 4) Quick merge guardrails

- Do not merge a docs-only threshold value update unless it matches live `/health` contract values from current bundle.
- Prefer one canonical metric key: choose either `false_fake_rate_real` or `false_fake_on_real`, then map all producers/consumers.
- Keep one explicit policy cap source (env/config or serialized bundle), and reference it consistently in QA docs.
