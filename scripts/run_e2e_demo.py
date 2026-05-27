"""
scripts/run_e2e_demo.py — End-to-end smoke test for the deepfake detector API.

Sends one known-real and one known-fake video from the local FF++ C23 dataset
to the FastAPI endpoint and asserts:
  - HTTP 200 response
  - All required AnalysisResponse fields are present
  - New registry fields (model_id, model_type, model_f1) are present
  - verdict is one of REAL / FAKE / UNCERTAIN

Usage:
    # Start the API first:
    uvicorn backend.main:app --port 8000

    # Then run this script (from repo root, venv active):
    python scripts/run_e2e_demo.py

    # Override the API URL:
    set API_URL=http://localhost:8001
    python scripts/run_e2e_demo.py

    # Use specific video paths:
    python scripts/run_e2e_demo.py path/to/real.mp4 path/to/fake.mp4
"""

from __future__ import annotations

import os
import sys
import pathlib
import time

try:
    import requests
except ImportError:
    print("[ERROR] 'requests' not installed.  Run: pip install requests")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_URL   = os.getenv("API_URL", "http://localhost:8000")
N_FRAMES  = int(os.getenv("N_FRAMES", "8"))    # fewer frames = faster smoke test
TIMEOUT   = int(os.getenv("TIMEOUT",  "120"))  # seconds

# Fields every AnalysisResponse must have
_REQUIRED_FIELDS = {
    "video_name", "verdict", "confidence", "prob_fake_mean",
    "quality_weighted_prob_fake", "temporal_score",
    "model_used", "model_id", "model_type", "model_f1",
    "frames_analyzed", "frames_sampled", "fps", "duration_sec",
    "frames", "warnings",
}
_VALID_VERDICTS = {"REAL", "FAKE", "UNCERTAIN"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_video(directory: pathlib.Path, label: str) -> pathlib.Path | None:
    """Return first .mp4 in directory, or None."""
    vids = sorted(directory.glob("*.mp4"))
    if not vids:
        print(f"  [WARN] No .mp4 files found in {directory}")
        return None
    print(f"  Found {len(vids)} .mp4 files in {directory} — using {vids[0].name}")
    return vids[0]


def _analyze(video_path: pathlib.Path) -> dict:
    """POST video to /analyze, return parsed JSON body."""
    print(f"\n  → Uploading: {video_path.name}  ({video_path.stat().st_size // 1024} KB)")
    t0 = time.time()
    with open(video_path, "rb") as fh:
        resp = requests.post(
            f"{API_URL}/analyze",
            files={"video": (video_path.name, fh, "video/mp4")},
            data={"n_frames": N_FRAMES},
            timeout=TIMEOUT,
        )
    elapsed = time.time() - t0
    resp.raise_for_status()
    body = resp.json()
    print(f"  ← Response in {elapsed:.1f}s  (HTTP {resp.status_code})")
    return body


def _assert_response(body: dict, video_label: str) -> None:
    """Assert all required fields exist and have sensible values."""
    errors: list[str] = []

    # Required fields present
    for field in _REQUIRED_FIELDS:
        if field not in body:
            errors.append(f"  MISSING field: {field!r}")

    # Verdict is valid
    verdict = body.get("verdict", "")
    if verdict not in _VALID_VERDICTS:
        errors.append(f"  INVALID verdict: {verdict!r} (expected one of {_VALID_VERDICTS})")

    # prob_fake_mean in [0, 1]
    pfm = body.get("prob_fake_mean", -1)
    if not (0.0 <= pfm <= 1.0):
        errors.append(f"  prob_fake_mean out of range: {pfm}")

    # model_id is non-empty string
    mid = body.get("model_id", "")
    if not isinstance(mid, str) or not mid.strip():
        errors.append(f"  model_id is empty or not a string: {mid!r}")

    # model_f1 is float or None (not missing)
    if "model_f1" in body:
        mf1 = body["model_f1"]
        if mf1 is not None and not isinstance(mf1, (int, float)):
            errors.append(f"  model_f1 has unexpected type: {type(mf1)}")

    if errors:
        print(f"\n  [FAIL] Assertions failed for {video_label}:")
        for e in errors:
            print(e)
        sys.exit(1)

    # Summary
    print(f"  Verdict      : {body['verdict']}")
    print(f"  P(fake) mean : {body['prob_fake_mean']:.4f}")
    print(f"  Confidence   : {body['confidence']:.4f}")
    print(f"  Model ID     : {body.get('model_id', 'n/a')}")
    print(f"  Model type   : {body.get('model_type', 'n/a')}")
    print(f"  Model F1     : {body.get('model_f1')}")
    print(f"  Frames used  : {body['frames_analyzed']}/{body['frames_sampled']}")
    if body.get("warnings"):
        for w in body["warnings"]:
            print(f"  [WARN] {w}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str]) -> None:
    print("=" * 60)
    print("Deepfake Detector — End-to-End Demo")
    print(f"API: {API_URL}")
    print("=" * 60)

    # 1. Check API is reachable
    try:
        health = requests.get(f"{API_URL}/health", timeout=10).json()
        print(f"\n[health] status={health.get('status')}  "
              f"model_loaded={health.get('model_loaded')}  "
              f"active={health.get('active_model_id')}  "
              f"mrl={health.get('mrl_loaded')}")
    except requests.exceptions.ConnectionError:
        print(f"\n[ERROR] Cannot reach API at {API_URL}")
        print("        Start it with:  uvicorn backend.main:app --port 8000")
        sys.exit(1)

    # 2. Resolve video paths
    if len(argv) >= 3:
        real_vid = pathlib.Path(argv[1])
        fake_vid = pathlib.Path(argv[2])
        if not real_vid.exists():
            print(f"[ERROR] Real video not found: {real_vid}"); sys.exit(1)
        if not fake_vid.exists():
            print(f"[ERROR] Fake video not found: {fake_vid}"); sys.exit(1)
    else:
        data_root = pathlib.Path("data") / "FaceForensics++_C23"
        real_vid  = _find_video(data_root / "original", "real")
        fake_vid  = _find_video(data_root / "Deepfakes", "fake")
        if real_vid is None or fake_vid is None:
            print("\n[ERROR] Could not find FF++ C23 videos under data/.")
            print("        Provide paths directly:  python scripts/run_e2e_demo.py real.mp4 fake.mp4")
            sys.exit(1)

    # 3. Analyze and assert
    print("\n── Real video ──────────────────────────────────────────")
    real_body = _analyze(real_vid)
    _assert_response(real_body, "real video")

    print("\n── Fake video ──────────────────────────────────────────")
    fake_body = _analyze(fake_vid)
    _assert_response(fake_body, "fake video")

    # 4. Check /models endpoint
    print("\n── GET /models ──────────────────────────────────────────")
    models_resp = requests.get(f"{API_URL}/models", timeout=10).json()
    total = models_resp.get("total", 0)
    active_id = models_resp.get("active_model_id", "n/a")
    print(f"  Registry has {total} model(s); active = {active_id!r}")

    print()
    print("=" * 60)
    print("[PASS] All assertions passed.")
    print("=" * 60)


if __name__ == "__main__":
    main(sys.argv)
