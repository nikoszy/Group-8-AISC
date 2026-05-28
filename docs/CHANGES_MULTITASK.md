## 2026-05-27 — Docs consistency sweep

- Aligned dataset path wording in `README.md` to `data/FaceForensics++_C23/` to match code defaults and QA artifacts.
- Updated `CLAUDE.md` EAR status from stubbed to integrated video-level scoring, while noting unresolved IDs may still use `0.5` fallback.
- Added `ensemble_model.pkl` export artifact to `CLAUDE.md` file structure so export status matches `ensemble.py` behavior and QA report.
- Kept threshold behavior language consistent with current QA contract (`threshold` + uncertain band served from model bundle), with no contradictory threshold claims introduced.
