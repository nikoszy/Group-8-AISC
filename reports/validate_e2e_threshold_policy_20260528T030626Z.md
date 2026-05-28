# E2E /analyze Validation Report

- UTC timestamp: `20260528T030626Z`
- Base URL: `http://127.0.0.1:8000`
- FF++ root: `D:\New folder (2)\Group-8-AISC-main\Group-8-AISC-main\data\FaceForensics++_C23`
- Requested sample size: `20 real + 20 fake`
- n_frames per request: `12`

## Backend Policy Snapshot

- threshold: `0.05`
- uncertain_band: `0.1`
- verdict_hi: `0.15000000000000002`
- verdict_lo: `0.0`

## Outcome Metrics

- successful analyses: `40/40`
- verdict_distribution: `{'UNCERTAIN': 40}`
- prob_fake_mean overall mean/min/max: `0.00711 / 0.0047 / 0.0105`
- prob_fake_mean real_mean vs fake_mean: `0.007415 vs 0.006805`
- false-FAKE rate (real->FAKE): `0.0` (0 cases)
- false-REAL rate (fake->REAL): `0.0` (0 cases)

## Collapse Check

- collapsed_to_one_verdict: `True`
- all_probs_side_of_decision_band: `None`
- likely_cause_evidence: `['All analyzed samples map to one verdict class.', 'Threshold is very low (0.0500); with uncertain band this makes FAKE easy to trigger.']`

## Repro Commands

```bash
python scripts/validate_e2e_threshold_policy.py \
  --ffpp-root "data/FaceForensics++_C23" \
  --base-url "http://127.0.0.1:8000" \
  --limit-per-class 20 \
  --n-frames 12 \
  --seed 42
```

JSON artifact: `D:\New folder (2)\Group-8-AISC-main\reports\validate_e2e_threshold_policy_20260528T030626Z.json`