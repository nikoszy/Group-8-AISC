# =============================================================================
# inspect_dataset.py
# =============================================================================
# Step 1 of Module 3.
#   - Extracts face frames from FaceForensics++ C23 videos already on disk.
#   - 50 videos × 4 frames per class  →  200 samples per class.
#   - Saves video_id in the manifest so ensemble.py can do video-level splits
#     and avoid identity leakage between train and val sets.
#   - Rejects low-quality frames (too dark, no face found).
#
# LABEL CONVENTION
#   0 = REAL  (data/FaceForensics++_C23/original/)
#   1 = FAKE  (data/FaceForensics++_C23/Deepfakes/)
# =============================================================================

import os
import csv
import numpy as np
from datetime import datetime

import cv2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FF_DIR    = os.path.join("data", "FaceForensics++_C23")
REAL_SRC  = os.path.join(FF_DIR, "original")
FAKE_SRC  = os.path.join(FF_DIR, "Deepfakes")

REAL_DIR      = os.path.join("data", "real", "frames")
FAKE_DIR      = os.path.join("data", "fake", "frames")
MANIFEST_PATH = os.path.join("data", "manifest.csv")

# Sequential per-video output (for Module 1 blink detection)
SEQ_REAL_DIR  = os.path.join("data", "processed", "frames", "real")
SEQ_FAKE_DIR  = os.path.join("data", "processed", "frames", "fake")
EXTRACT_FPS   = 15  # target sampling rate for sequential extraction

TARGET_SIZE      = 224   # face-crop output size (pixels)
TARGET_PER_CLASS = 500   # frames to save per class
FRAMES_PER_VIDEO = 5     # frames sampled from each video
MIN_BRIGHTNESS   = 40    # reject frames darker than this mean pixel value
MIN_FACE_FRAC    = 0.04  # reject face crop if face area < 4% of frame area

_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_face_crop(frame, target_size):
    """
    Detect the largest face and return a padded square crop resized to
    target_size.  Returns None if no face is found or the face is too small.
    """
    h, w = frame.shape[:2]
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        return None

    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])

    # Reject if face area is too small relative to the frame
    if (fw * fh) < (MIN_FACE_FRAC * h * w):
        return None

    # 15 % padding on each side
    pad = int(max(fw, fh) * 0.15)
    x1 = max(0, x - pad);  y1 = max(0, y - pad)
    x2 = min(w, x + fw + pad); y2 = min(h, y + fh + pad)
    crop = frame[y1:y2, x1:x2]

    return cv2.resize(crop, (target_size, target_size),
                      interpolation=cv2.INTER_AREA)


def extract_faces_from_video(video_path, n_frames, target_size):
    """
    Open a video, sample n_frames evenly, detect and crop faces.
    Returns list of (frame_index, face_crop) tuples for accepted frames only.
    Rejects dark frames and frames where no face passes the quality check.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    margin  = max(1, total // 10)
    indices = np.linspace(margin, total - margin - 1, n_frames, dtype=int)

    results = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Reject very dark frames (often transitions or low-light artifacts)
        if float(np.mean(frame)) < MIN_BRIGHTNESS:
            continue

        crop = detect_face_crop(frame, target_size)
        if crop is not None:
            results.append((int(idx), crop))

    cap.release()
    return results


def extract_sequential_from_video(video_path, out_dir, target_fps, target_size):
    """
    Extract face crops at *target_fps* for the full video duration and save
    them sequentially into *out_dir*.  Frames where the face detector fails
    or brightness is too low are skipped (no placeholder written).

    Returns the number of frames successfully saved.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    native_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return 0

    # Compute which native frame indices to grab so that we effectively
    # sample at *target_fps*.  E.g. for a 30-fps video with target 15 fps
    # we take every 2nd frame.
    step = max(1, round(native_fps / target_fps))
    indices = range(0, total_frames, step)

    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        if float(np.mean(frame)) < MIN_BRIGHTNESS:
            continue

        crop = detect_face_crop(frame, target_size)
        if crop is None:
            continue

        filename = f"frame_{str(saved).zfill(5)}.jpg"
        if cv2.imwrite(os.path.join(out_dir, filename), crop):
            saved += 1

    cap.release()
    return saved


def extract_class_sequential(src_dir, videos, base_out_dir, prefix, target_fps):
    """
    For every video in *videos*, extract sequential face crops at
    *target_fps* into base_out_dir/<video_id>/.

    Returns total frames saved across all videos.
    """
    total_saved = 0
    for i, vid_name in enumerate(videos):
        vid_id   = os.path.splitext(vid_name)[0]
        vid_path = os.path.join(src_dir, vid_name)
        vid_out  = os.path.join(base_out_dir, f"{prefix}_{vid_id}")

        n = extract_sequential_from_video(
            vid_path, vid_out, target_fps, TARGET_SIZE
        )
        total_saved += n

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{prefix}] video {i+1}/{len(videos)}  "
                  f"({vid_id}: {n} frames)  total={total_saved}")

    print(f"  Extracted {total_saved} sequential {prefix} frames "
          f"from {len(videos)} videos into {base_out_dir}")
    return total_saved


def count_jpgs(folder):
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])


def clear_folder(folder):
    if os.path.isdir(folder):
        for f in os.listdir(folder):
            if f.lower().endswith(".jpg"):
                os.remove(os.path.join(folder, f))


# ---------------------------------------------------------------------------
# Step 1: Greet + make folders
# ---------------------------------------------------------------------------

print("=" * 65)
print("MODULE 3 -- DATASET INSPECTION (FaceForensics++ C23)")
print("=" * 65)
print()
print("  Extracting face crops from local FF++ C23 videos.")
print(f"  Target   : {TARGET_PER_CLASS} frames per class")
print(f"  Videos   : {TARGET_PER_CLASS // FRAMES_PER_VIDEO} per class  "
      f"({FRAMES_PER_VIDEO} frames each)")
print(f"  Crop size: {TARGET_SIZE}×{TARGET_SIZE} px")
print(f"  Filters  : brightness >= {MIN_BRIGHTNESS}  |  "
      f"face area >= {int(MIN_FACE_FRAC*100)}% of frame")
print()

os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)
print(f"[OK] Folders ready: {REAL_DIR}  |  {FAKE_DIR}")
print()

# ---------------------------------------------------------------------------
# Step 2: Verify source videos
# ---------------------------------------------------------------------------

if not os.path.isdir(REAL_SRC):
    print(f"[ERROR] Not found: {REAL_SRC}")
    raise SystemExit(1)
if not os.path.isdir(FAKE_SRC):
    print(f"[ERROR] Not found: {FAKE_SRC}")
    raise SystemExit(1)

real_videos = sorted([f for f in os.listdir(REAL_SRC) if f.endswith(".mp4")])
fake_videos = sorted([f for f in os.listdir(FAKE_SRC) if f.endswith(".mp4")])
print(f"  Real videos available : {len(real_videos)}")
print(f"  Fake videos available : {len(fake_videos)}")
print()

# ---------------------------------------------------------------------------
# Helper: extract frames from a set of videos into a folder
# ---------------------------------------------------------------------------

def extract_class(src_dir, videos, out_dir, prefix, label, target_n, fps):
    """
    Extract face crops from videos into out_dir.
    Returns list of manifest rows: {file_path, label, video_id}.
    """
    clear_folder(out_dir)
    rows   = []
    saved  = 0
    needed = (target_n + fps - 1) // fps   # how many videos to open

    for vid_name in videos[:needed]:
        if saved >= target_n:
            break

        vid_id   = os.path.splitext(vid_name)[0]          # e.g. "000"
        vid_path = os.path.join(src_dir, vid_name)
        faces    = extract_faces_from_video(vid_path, fps, TARGET_SIZE)

        for _, crop in faces:
            if saved >= target_n:
                break
            filename = f"{prefix}_{str(saved).zfill(4)}.jpg"
            path     = os.path.join(out_dir, filename)
            if cv2.imwrite(path, crop):
                rows.append({
                    "file_path"     : path,
                    "label"         : label,
                    "video_id"      : f"{prefix}_{vid_id}",
                    "source_dataset": f"FaceForensics++_C23/{src_dir.split(os.sep)[-1]}",
                })
                saved += 1

        if saved % 40 == 0 and saved > 0:
            print(f"  Saved {saved}/{target_n} {prefix} frames ...")

    print(f"  Extracted {saved} {prefix} frames from "
          f"{min(needed, len(videos))} videos.")
    return rows


# ---------------------------------------------------------------------------
# Step 3: Extract REAL frames
# ---------------------------------------------------------------------------

print("-" * 65)
print(f"REAL FRAMES  <-  {REAL_SRC}")
print("-" * 65)
real_rows = extract_class(REAL_SRC, real_videos, REAL_DIR, "real", 0,
                          TARGET_PER_CLASS, FRAMES_PER_VIDEO)
print(f"  Total real on disk: {count_jpgs(REAL_DIR)}")
print()

# ---------------------------------------------------------------------------
# Step 4: Extract FAKE frames
# ---------------------------------------------------------------------------

print("-" * 65)
print(f"FAKE FRAMES  <-  {FAKE_SRC}")
print("-" * 65)
fake_rows = extract_class(FAKE_SRC, fake_videos, FAKE_DIR, "fake", 1,
                          TARGET_PER_CLASS, FRAMES_PER_VIDEO)
print(f"  Total fake on disk: {count_jpgs(FAKE_DIR)}")
print()

# ---------------------------------------------------------------------------
# Step 4b: Sequential per-video extraction at 15 FPS (Module 1 blink detection)
# ---------------------------------------------------------------------------

print("-" * 65)
print(f"SEQUENTIAL EXTRACTION  ({EXTRACT_FPS} FPS, full duration)")
print(f"  Output: {SEQ_REAL_DIR}/<video_id>/  |  {SEQ_FAKE_DIR}/<video_id>/")
print("-" * 65)

os.makedirs(SEQ_REAL_DIR, exist_ok=True)
os.makedirs(SEQ_FAKE_DIR, exist_ok=True)

seq_real_total = extract_class_sequential(
    REAL_SRC, real_videos, SEQ_REAL_DIR, "real", EXTRACT_FPS
)
seq_fake_total = extract_class_sequential(
    FAKE_SRC, fake_videos, SEQ_FAKE_DIR, "fake", EXTRACT_FPS
)

print(f"\n  Sequential real frames: {seq_real_total}")
print(f"  Sequential fake frames: {seq_fake_total}")
print()

# ---------------------------------------------------------------------------
# Step 5: Build manifest.csv
# ---------------------------------------------------------------------------

print("-" * 65)
print(f"BUILDING MANIFEST  ->  {MANIFEST_PATH}")
print("-" * 65)

all_rows = real_rows + fake_rows
with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(
        fh, fieldnames=["file_path", "label", "video_id", "source_dataset"]
    )
    writer.writeheader()
    writer.writerows(all_rows)

n_real   = sum(1 for r in all_rows if r["label"] == 0)
n_fake   = sum(1 for r in all_rows if r["label"] == 1)
n_videos = len(set(r["video_id"] for r in all_rows))
print(f"  Rows    : {len(all_rows)}  ({n_real} real, {n_fake} fake)")
print(f"  Videos  : {n_videos} unique  (enables video-level split in ensemble.py)")
print()

# ---------------------------------------------------------------------------
# Step 6: Sample inspection
# ---------------------------------------------------------------------------

print("-" * 65)
print("SAMPLE INSPECTION  (first 3 real, first 3 fake)")
print("-" * 65)
import numpy as _np
for label_val, folder, tag in [(0, REAL_DIR, "REAL"), (1, FAKE_DIR, "FAKE")]:
    files = sorted([f for f in os.listdir(folder) if f.endswith(".jpg")])[:3]
    for fname in files:
        img = cv2.imread(os.path.join(folder, fname))
        if img is not None:
            print(f"  [{tag}] {fname}  shape={img.shape}  "
                  f"mean={float(_np.mean(img)):.1f}  std={float(_np.std(img)):.1f}")
print()

# ---------------------------------------------------------------------------
# Step 7: Label distribution
# ---------------------------------------------------------------------------

print("-" * 65)
print("LABEL DISTRIBUTION")
print("-" * 65)
total = len(all_rows)
if total > 0:
    print(f"  Label 0 (REAL) : {n_real:4d}  ({100*n_real/total:.1f}%)")
    print(f"  Label 1 (FAKE) : {n_fake:4d}  ({100*n_fake/total:.1f}%)")
    print(f"  Total          : {total:4d}")
else:
    print("  No frames extracted — check source videos and face detection.")
print()

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

print("=" * 65)
print("INSPECTION COMPLETE")
print("=" * 65)
print(f"  Manifest     : {MANIFEST_PATH}  ({len(all_rows)} rows)")
print(f"  Unique videos: {n_videos}  (video-level split protects against leakage)")
print(f"  Real frames  : {n_real}  ->  {REAL_DIR}")
print(f"  Fake frames  : {n_fake}  ->  {FAKE_DIR}")
print(f"  Image size   : {TARGET_SIZE}×{TARGET_SIZE} px face crops (aligned)")
print()
print(f"  Sequential frames ({EXTRACT_FPS} FPS, per-video):")
print(f"    Real: {seq_real_total} frames  ->  {SEQ_REAL_DIR}/<video_id>/")
print(f"    Fake: {seq_fake_total} frames  ->  {SEQ_FAKE_DIR}/<video_id>/")
print()
print(f"  Date         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()
print("NEXT STEP:  python ensemble.py")
print("=" * 65)
