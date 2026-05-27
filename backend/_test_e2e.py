"""
Quick end-to-end test for the FastAPI backend.
Run from the repo root with the server already up:
    .venv\\Scripts\\python backend\\_test_e2e.py
"""
import sys
import requests

BASE = "http://127.0.0.1:8000"
REAL_VIDEO = r"data\FaceForensics++_C23\original\000.mp4"
FAKE_VIDEO = r"data\FaceForensics++_C23\Deepfakes\000_003.mp4"


def section(title):
    print("\n" + "=" * 60)
    print("  " + title)
    print("=" * 60)


# -- 1. Health check -----------------------------------------------
section("1. GET /health")
r = requests.get(f"{BASE}/health", timeout=10)
print(f"  status {r.status_code}: {r.json()}")
assert r.status_code == 200
assert r.json()["status"] == "ok"
print("  [PASS] health OK")

# -- 2. Analyze a REAL video ---------------------------------------
section("2. POST /analyze  (REAL video, 6 frames)")
with open(REAL_VIDEO, "rb") as f:
    r = requests.post(
        f"{BASE}/analyze",
        files={"video": ("000_real.mp4", f, "video/mp4")},
        data={"n_frames": "6"},
        timeout=120,
    )
if r.status_code != 200:
    print(f"  ERROR {r.status_code}: {r.text[:500]}")
    sys.exit(1)

d = r.json()
print(f"  verdict               : {d['verdict']}")
print(f"  prob_fake_mean        : {d['prob_fake_mean']}")
print(f"  quality_weighted_prob : {d['quality_weighted_prob_fake']}")
print(f"  confidence            : {d['confidence']}")
print(f"  frames_analyzed       : {d['frames_analyzed']}/{d['frames_sampled']}")
print(f"  model_used            : {d['model_used']}")
print(f"  warnings              : {d.get('warnings', [])}")
detected_real = [fr for fr in d["frames"] if fr["face_detected"]]
print(f"  Per-frame probs       : {[round(fr['prob_fake'], 3) for fr in detected_real]}")

for key in ("verdict", "prob_fake_mean", "quality_weighted_prob_fake",
            "confidence", "frames_analyzed", "frames_sampled",
            "fps", "duration_sec", "frames", "warnings", "model_used"):
    assert key in d, f"Missing response field: {key}"
assert d["frames_analyzed"] > 0, "No faces detected in real video"
assert 0.0 <= d["prob_fake_mean"] <= 1.0
print("  [PASS] real video OK")

# -- 3. Analyze a FAKE video ---------------------------------------
section("3. POST /analyze  (FAKE video, 6 frames)")
with open(FAKE_VIDEO, "rb") as f:
    r = requests.post(
        f"{BASE}/analyze",
        files={"video": ("000_fake.mp4", f, "video/mp4")},
        data={"n_frames": "6"},
        timeout=120,
    )
if r.status_code != 200:
    print(f"  ERROR {r.status_code}: {r.text[:500]}")
    sys.exit(1)

d2 = r.json()
print(f"  verdict               : {d2['verdict']}")
print(f"  prob_fake_mean        : {d2['prob_fake_mean']}")
print(f"  quality_weighted_prob : {d2['quality_weighted_prob_fake']}")
print(f"  confidence            : {d2['confidence']}")
print(f"  frames_analyzed       : {d2['frames_analyzed']}/{d2['frames_sampled']}")
print(f"  model_used            : {d2['model_used']}")
detected_fake = [fr for fr in d2["frames"] if fr["face_detected"]]
print(f"  Per-frame probs       : {[round(fr['prob_fake'], 3) for fr in detected_fake]}")
assert d2["frames_analyzed"] > 0, "No faces detected in fake video"
assert 0.0 <= d2["prob_fake_mean"] <= 1.0
print("  [PASS] fake video OK")

# -- 4. Error on empty upload --------------------------------------
section("4. POST /analyze  (empty body -> 422)")
r = requests.post(f"{BASE}/analyze", data={}, timeout=10)
print(f"  status {r.status_code} (expected 422)")
assert r.status_code == 422
print("  [PASS] validation error OK")

# -- 5. Summary ----------------------------------------------------
section("SUMMARY")
print(f"  Real  prob_fake = {d['prob_fake_mean']:<6}  verdict = {d['verdict']}")
print(f"  Fake  prob_fake = {d2['prob_fake_mean']:<6}  verdict = {d2['verdict']}")
print("\n  [ALL TESTS PASSED]\n")
