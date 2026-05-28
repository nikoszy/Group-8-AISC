# Threshold Operating-Point Analysis

> Consolidation note (2026-05-28): this document records one offline sweep result.
> The currently served production/demo contract in QA artifacts is `threshold=0.05`, `uncertain_band=0.10`, `verdict_hi=0.15`, `verdict_lo=0.00` (see `docs/qa_audit_results.json` and `docs/QA_DEEPFAKE_AUDIT.md`).

- Validation ROC-AUC (score-only): `1.0000`
- Constraint: false-FAKE on real <= `0.050`
- Constraint: fake recall >= `0.950`

## Candidate operating points

| Candidate | Threshold | FFR(real->fake) | Recall(fake) | FNR(fake->real) | Balanced Acc | Accuracy | Precision(fake) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Constrained best | 0.0150 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| Lowest false-FAKE | 0.9900 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| Max recall-fake | 0.0150 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| Best balanced-acc | 0.0150 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |

## Practical threshold policy

This sweep suggests `threshold=0.0150` for its own sampled split; however, consolidated repo state currently serves `threshold=0.05` from the bundled model contract.
Guardrail: if future validation fails either constraint, pick the nearest threshold that keeps `false_fake_on_real` within cap first, then maximize `recall_fake`.

## Additional sampled thresholds

| Threshold | FFR(real->fake) | Recall(fake) | Balanced Acc | Accuracy |
|---:|---:|---:|---:|---:|
| 0.0150 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| 0.9900 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| 0.0500 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| 0.1500 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| 0.5000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| 0.9000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
