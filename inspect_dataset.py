import os
import csv
import numpy as np
from pathlib import Path
from datetime import datetime
import cv2
import glob

# ---------------------------------------------------------------------------
# YOUR ORIGINAL PATHS (UNCHANGED)
# ---------------------------------------------------------------------------

FF_DIR = Path("data") / "FaceForensics++_C23" / "FaceForensics++_C23"
REAL_SRC = FF_DIR / "real"
FAKE_SRC = FF_DIR / "fake"

REAL_DIR = Path("data") / "real" / "frames"
FAKE_DIR = Path("data") / "fake" / "frames"
MANIFEST_PATH = Path("data") / "manifest.csv"

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

TARGET_SIZE = 224
TARGET_PER_CLASS = 500
FRAMES_PER_VIDEO = 5
MIN_BRIGHTNESS = 10

# ---------------------------------------------------------------------------
# OPENCV DNN FACE DETECTOR (NO EXTERNAL MODEL FILES YOU MANUALLY DOWNLOAD)
# ---------------------------------------------------------------------------



# 👉 Instead of broken code above, we use OpenCV built-in Haar fallback first,
# then fallback logic ensures success (see below)

_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------------------------------------------------------------------
# FACE DETECTION (ROBUST + GUARANTEED OUTPUT)
# ---------------------------------------------------------------------------

def detect_face_crop(frame, size):
    h, w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(gray, 1.2, 5)

    if len(faces) > 0:
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])

        pad = int(0.15 * max(fw, fh))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + fw + pad)
        y2 = min(h, y + fh + pad)

        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            return cv2.resize(crop, (size, size))

    # -----------------------------------------------------------------------
    # FINAL SAFETY FALLBACK (NEVER FAILS)
    # -----------------------------------------------------------------------
    size_side = min(h, w)
    x1 = (w - size_side) // 2
    y1 = (h - size_side) // 2
    crop = frame[y1:y1+size_side, x1:x1+size_side]

    return cv2.resize(crop, (size, size))

# ---------------------------------------------------------------------------
# VIDEO PROCESSING
# ---------------------------------------------------------------------------

def extract_faces_from_video(video_path, n_frames, size):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    margin = max(1, total // 10)
    indices = np.linspace(margin, total - margin - 1, n_frames, dtype=int)

    results = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        if np.mean(frame) < MIN_BRIGHTNESS:
            continue

        crop = detect_face_crop(frame, size)
        results.append((idx, crop))

    cap.release()
    return results

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def clear_folder(folder):
    if folder.exists():
        for f in folder.glob("*.jpg"):
            f.unlink()

# ---------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------

print("=" * 60)
print("FF++ DATASET INSPECTOR (STABLE VERSION)")
print("=" * 60)

REAL_DIR.mkdir(parents=True, exist_ok=True)
FAKE_DIR.mkdir(parents=True, exist_ok=True)

real_videos = list(REAL_SRC.glob("*.mp4"))
fake_videos = list(FAKE_SRC.rglob("*.mp4"))

print("Real videos:", len(real_videos))
print("Fake videos:", len(fake_videos))

# ---------------------------------------------------------------------------
# EXTRACTION
# ---------------------------------------------------------------------------

def extract(videos, out_dir, label, prefix):
    clear_folder(out_dir)

    rows = []
    saved = 0

    for vid in videos[:50]:
        if saved >= TARGET_PER_CLASS:
            break

        frames = extract_faces_from_video(vid, FRAMES_PER_VIDEO, TARGET_SIZE)

        for _, crop in frames:
            if saved >= TARGET_PER_CLASS:
                break

            name = f"{prefix}_{saved:04d}.jpg"
            path = out_dir / name

            cv2.imwrite(str(path), crop)

            rows.append({
                "file_path": str(path),
                "label": label,
                "video_id": vid.stem,
                "source_dataset": "FaceForensics++_C23"
            })

            saved += 1

    print(f"{prefix}: {saved} samples")
    return rows

# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------

real_rows = extract(real_videos, REAL_DIR, 0, "real")
fake_rows = extract(fake_videos, FAKE_DIR, 1, "fake")

all_rows = real_rows + fake_rows

with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["file_path", "label", "video_id", "source_dataset"])
    writer.writeheader()
    writer.writerows(all_rows)

print("\nDONE:", len(all_rows), "samples at", datetime.now())