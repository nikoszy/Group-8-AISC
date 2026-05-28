# Experiment Artifact Index

This index catalogs experiment outputs currently present from this multitask run.

## Artifact Catalog

| Artifact Path | Type | Purpose | Producer script/step | Key metric fields (JSON only) |
|---|---|---|---|---|
| `docs/qa_audit_results.json` | JSON | End-to-end QA audit payload across module checks, ML validation, adversarial probes, API checks, and proxy E2E checks | `python scripts/qa_audit_runner.py` | `phase2.val_auc`, `phase2.balanced_acc_at_best_t`, `phase2.best_threshold_ba`, `phase2.cv_auc_mean`, `phase3.evasion.baseline_false_real_rate`, `phase3.spoofing.baseline_false_fake_rate`, `phase4.API-04_latency_sec`, `phase4_e2e.accuracy`, `phase4_http.API-05_http.prob_fake_mean` |
| `docs/threshold_operating_points.json` | JSON | Threshold sweep/candidate operating points and constrained selection metadata | `python scripts/threshold_policy_report.py --features-csv data/module3_features.csv --output-json docs/threshold_operating_points.json --output-md docs/threshold_operating_points.md` | `meta.val_auc`, `meta.n_val`, `meta.ffr_cap`, `meta.min_recall_fake`, `candidates[].threshold`, `candidates[].balanced_accuracy`, `candidates[].false_fake_on_real`, `named_candidates.constrained_best.threshold` |
| `docs/ear_mapping_audit.json` | JSON | EAR source-video resolution coverage for base manifest | `python scripts/qa_ear_mapping_audit.py --manifest data/manifest.csv --output-json docs/ear_mapping_audit.json` | `rows_total`, `rows_unresolved`, `rows_unresolved_rate`, `unique_video_ids_unresolved_rate`, `by_class.*.unique_video_ids_unresolved_rate` |
| `docs/ear_mapping_audit_after.json` | JSON | Re-run of base EAR mapping audit after mapping/path updates | `python scripts/qa_ear_mapping_audit.py --manifest data/manifest.csv --output-json docs/ear_mapping_audit_after.json` | `rows_total`, `rows_unresolved`, `rows_unresolved_rate`, `unique_video_ids_unresolved_rate` |
| `docs/deepfakes_ear_mapping_audit.json` | JSON | EAR source-video resolution coverage for Deepfakes experiment manifest | `python scripts/qa_ear_mapping_audit.py --manifest data/experiments/deepfakes/manifest.csv --output-json docs/deepfakes_ear_mapping_audit.json` | `rows_total`, `rows_unresolved`, `rows_unresolved_rate`, `by_source_dataset.*.unique_video_ids_unresolved_rate` |
| `docs/face2face_ear_mapping_audit.json` | JSON | EAR source-video resolution coverage for Face2Face experiment manifest | `python scripts/qa_ear_mapping_audit.py --manifest data/experiments/face2face/manifest.csv --output-json docs/face2face_ear_mapping_audit.json` | `rows_total`, `rows_unresolved`, `rows_unresolved_rate`, `by_source_dataset.*.unique_video_ids_unresolved_rate` |
| `docs/face2face_ear_mapping_audit_after.json` | JSON | Re-run Face2Face EAR mapping audit after mapping/path updates | `python scripts/qa_ear_mapping_audit.py --manifest data/experiments/face2face/manifest.csv --output-json docs/face2face_ear_mapping_audit_after.json` | `rows_total`, `rows_unresolved`, `rows_unresolved_rate` |
| `docs/faceswap_ear_mapping_audit.json` | JSON | EAR source-video resolution coverage for FaceSwap experiment manifest | `python scripts/qa_ear_mapping_audit.py --manifest data/experiments/faceswap/manifest.csv --output-json docs/faceswap_ear_mapping_audit.json` | `rows_total`, `rows_unresolved`, `rows_unresolved_rate`, `by_source_dataset.*.unique_video_ids_unresolved_rate` |
| `docs/faceswap_ear_mapping_audit_after.json` | JSON | Re-run FaceSwap EAR mapping audit after mapping/path updates | `python scripts/qa_ear_mapping_audit.py --manifest data/experiments/faceswap/manifest.csv --output-json docs/faceswap_ear_mapping_audit_after.json` | `rows_total`, `rows_unresolved`, `rows_unresolved_rate` |
| `data/experiments/ml03_validation_summary.json` | JSON | Independent recomputation summary of ML-03 metrics from produced experiment artifacts | Post-run validation step (`python -c` verifier noted inside artifact) | `tracks.deepfakes.metrics.auc`, `tracks.deepfakes.metrics.balanced_accuracy`, `tracks.deepfakes.metrics.threshold_used`, `tracks.face2face.metrics.*`, `tracks.faceswap.status`, `tracks.faceswap.reason` |
| `data/experiments/api_smoke_results.json` | JSON | Backend `/health` + `/analyze` smoke-check record and contract fields snapshot | `python scripts/api_smoke_check.py --output data/experiments/api_smoke_results.json` (or seeded from runtime observation as noted in file) | `backend_runtime_observation.observed_requests`, `contract_validation.status`, `contract_validation.health_fields_expected`, `contract_validation.analyze_fields_expected` |
| `data/experiments/deepfakes/manifest.csv` | CSV | Frame-level manifest for Deepfakes track | `python inspect_dataset.py` with env: `FAKE_SUBDIR=Deepfakes`, `MANIFEST_PATH=data/experiments/deepfakes/manifest.csv` | N/A |
| `data/experiments/deepfakes/module3_features.csv` | CSV | Feature matrix (`ear/artifact/fft/laplacian`) for Deepfakes track | `python ensemble.py` with env: `FEATURES_CSV=data/experiments/deepfakes/module3_features.csv` | N/A |
| `data/experiments/deepfakes/video_ear_scores.csv` | CSV | Video-level EAR cache for Deepfakes track | `python inspect_dataset.py` with env: `VIDEO_EAR_CSV=data/experiments/deepfakes/video_ear_scores.csv` | N/A |
| `data/experiments/deepfakes/inspect.log` | LOG | Dataset extraction/runtime log for Deepfakes track | `python inspect_dataset.py` with env overrides (logged by `scripts/run_ml03_adv_ood.py`) | N/A |
| `data/experiments/face2face/manifest.csv` | CSV | Frame-level manifest for Face2Face track | `python inspect_dataset.py` with env: `FAKE_SUBDIR=Face2Face`, `MANIFEST_PATH=data/experiments/face2face/manifest.csv` | N/A |
| `data/experiments/face2face/module3_features.csv` | CSV | Feature matrix (`ear/artifact/fft/laplacian`) for Face2Face track | `python ensemble.py` with env: `FEATURES_CSV=data/experiments/face2face/module3_features.csv` | N/A |
| `data/experiments/face2face/video_ear_scores.csv` | CSV | Video-level EAR cache for Face2Face track | `python inspect_dataset.py` with env: `VIDEO_EAR_CSV=data/experiments/face2face/video_ear_scores.csv` | N/A |
| `data/experiments/face2face/inspect.log` | LOG | Dataset extraction/runtime log for Face2Face track | `python inspect_dataset.py` with env overrides (logged by `scripts/run_ml03_adv_ood.py`) | N/A |
| `data/experiments/face2face/ensemble.log` | LOG | Training/evaluation log for Face2Face track | `python -u ensemble.py` with env overrides (logged by `scripts/run_ml03_adv_ood.py`) | N/A |
| `data/experiments/faceswap/manifest.csv` | CSV | Frame-level manifest for FaceSwap track | `python inspect_dataset.py` with env: `FAKE_SUBDIR=FaceSwap`, `MANIFEST_PATH=data/experiments/faceswap/manifest.csv` | N/A |

## Quick Re-Run Commands by Artifact Family

### 1) Full QA audit artifacts (`docs/qa_audit_results.json`)

```powershell
cd "d:\New folder (2)\Group-8-AISC-main\Group-8-AISC-main"
python scripts/qa_audit_runner.py
```

### 2) Threshold operating-point artifacts (`docs/threshold_operating_points.*`)

```powershell
cd "d:\New folder (2)\Group-8-AISC-main\Group-8-AISC-main"
python scripts/threshold_policy_report.py --features-csv data/module3_features.csv --output-json docs/threshold_operating_points.json --output-md docs/threshold_operating_points.md
```

### 3) EAR mapping audit artifacts (`docs/*ear_mapping_audit*.json`)

```powershell
cd "d:\New folder (2)\Group-8-AISC-main\Group-8-AISC-main"
python scripts/qa_ear_mapping_audit.py --manifest data/manifest.csv --output-json docs/ear_mapping_audit.json
python scripts/qa_ear_mapping_audit.py --manifest data/experiments/deepfakes/manifest.csv --output-json docs/deepfakes_ear_mapping_audit.json
python scripts/qa_ear_mapping_audit.py --manifest data/experiments/face2face/manifest.csv --output-json docs/face2face_ear_mapping_audit.json
python scripts/qa_ear_mapping_audit.py --manifest data/experiments/faceswap/manifest.csv --output-json docs/faceswap_ear_mapping_audit.json
```

### 4) API smoke artifacts (`data/experiments/api_smoke_results.json`)

```powershell
cd "d:\New folder (2)\Group-8-AISC-main\Group-8-AISC-main"
python scripts/api_smoke_check.py --base-url http://127.0.0.1:8000 --output data/experiments/api_smoke_results.json
```

### 5) ML-03 manipulation-track artifacts (`data/experiments/<track>/*`)

```powershell
cd "d:\New folder (2)\Group-8-AISC-main\Group-8-AISC-main"
python scripts/run_ml03_adv_ood.py
```

This regenerates per-track `manifest.csv`, `module3_features.csv`, `video_ear_scores.csv` (where produced), and run logs under `data/experiments/`.
