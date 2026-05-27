# ==============================================================================
# artifact_module.py  -  Module 2: Compression Artifact Detection
# ==============================================================================
# PIPELINE (matches flowchart):
#   Preprocessed frame
#       -> JPEG encode/decode  (cv2.imencode / cv2.imdecode)
#       -> Pixel diff score    (MSE  +  SSIM delta)
#       -> artifact_score      (0.0 = clean, 1.0 = heavily artifacted)
#
# HOW IT WORKS:
#   Deepfake generators produce pixel distributions that differ subtly from
#   real cameras.  A round-trip through lossy JPEG compression exposes this:
#   the codec discards high-frequency details it considers "noise", but GAN-
#   generated textures are NOT noise -- they carry meaningful signal.  The
#   resulting pixel-level and structural damage is measurably larger for fakes.
#
# TWO COMPLEMENTARY METRICS:
#   MSE delta   -- mean squared per-pixel error after recompression (magnitude)
#   SSIM delta  -- structural similarity drop after recompression (perceptual)
#   Combined as a weighted sum; SSIM weighted higher because it catches spatial
#   inconsistencies that pixel-level MSE averages away.
#
# PUBLIC API (used by ensemble.py):
#   get_artifact_score_for_frame(frame)   -> float  0.0-1.0
#   get_artifact_features(frame)          -> dict   {mse, ssim_delta, combined}
#   compute_artifact_score(frame_paths)   -> float  mean score across frames
#
# INTERNAL / UTILITY:
#   recompress_frame(frame, quality)
#   get_difference_map(original, recompressed)
#   compute_mse_score(orig_gray, recomp_gray)
#   compute_ssim_delta(orig_gray, recomp_gray)
#   visualize_artifacts(frame_path, save_path)
#   batch_score_folder(folder_path, label)
# ==============================================================================

import cv2
import numpy as np
import os

try:
    from skimage.metrics import structural_similarity as _ssim
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False


# --------------------------------------------------------------------------
# Tunable constants
# --------------------------------------------------------------------------

# JPEG quality used for the recompression round-trip.
# 75 is deliberately moderate: tight enough to expose GAN artifacts without
# degrading every real image above the detection threshold.
_JPEG_QUALITY = 75

# Empirical normalisation ceiling for MSE scores.
# At quality 75, real face images rarely exceed MSE ~0.003 on a [0,1] scale;
# fakes can reach 0.006-0.010.  We normalise by a ceiling of 0.01 so the
# MSE component fills the 0-1 range without clipping real images.
_MSE_CEIL = 0.01

# Weights for the combined score.  SSIM is perceptually motivated and more
# discriminative than raw pixel error, so it receives the higher weight.
_WEIGHT_MSE  = 0.35
_WEIGHT_SSIM = 0.65


# ==============================================================================
# 1. recompress_frame
# ==============================================================================

def recompress_frame(frame, quality=_JPEG_QUALITY):
    """
    One JPEG round-trip: encode the frame to JPEG bytes, then decode back.

    Inputs:
        frame   : uint8 BGR numpy array  (H x W x 3)
        quality : int 0-100  -- JPEG quality factor (lower = more compression)

    Returns:
        uint8 BGR numpy array -- same shape as input, after lossy round-trip.
        Returns the original frame unchanged if encode/decode fails.
    """
    success, jpeg_bytes = cv2.imencode(
        '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not success:
        return frame

    recompressed = cv2.imdecode(jpeg_bytes, cv2.IMREAD_COLOR)
    if recompressed is None:
        return frame

    return recompressed


# ==============================================================================
# 2. get_difference_map
# ==============================================================================

def get_difference_map(original, recompressed):
    """
    Per-pixel absolute difference between original and recompressed images.

    Returns a grayscale map where brighter pixels indicate larger changes.
    Used for visualisation and as input to MSE computation.

    Inputs:
        original     : uint8 BGR numpy array
        recompressed : uint8 BGR numpy array

    Returns:
        uint8 grayscale numpy array (H x W)
    """
    diff_color = cv2.absdiff(original, recompressed)
    diff_gray  = cv2.cvtColor(diff_color, cv2.COLOR_BGR2GRAY)
    return diff_gray


# ==============================================================================
# 3. compute_mse_score
# ==============================================================================

def compute_mse_score(orig_gray, recomp_gray):
    """
    Normalised Mean Squared Error between original and recompressed grayscale
    images.

    MSE = mean( (orig - recomp)^2 ) over all pixels, computed on float values
    in [0, 1] (pixel values divided by 255).

    Normalised to [0, 1] by dividing by _MSE_CEIL so the score is comparable
    with the SSIM delta on the same scale.

    Inputs:
        orig_gray   : uint8 grayscale array (H x W)
        recomp_gray : uint8 grayscale array (H x W)

    Returns:
        float in [0, 1]
    """
    orig_f   = orig_gray.astype(np.float32)   / 255.0
    recomp_f = recomp_gray.astype(np.float32) / 255.0

    mse = float(np.mean((orig_f - recomp_f) ** 2))

    return min(mse / _MSE_CEIL, 1.0)


# ==============================================================================
# 4. compute_ssim_delta
# ==============================================================================

def compute_ssim_delta(orig_gray, recomp_gray):
    """
    Structural similarity drop caused by JPEG recompression.

    SSIM(original, recompressed) is close to 1.0 for real images (compression
    damages them predictably); it is measurably lower for deepfakes (the codec
    destroys GAN-generated structure more aggressively).

    delta = 1 - SSIM(original, recompressed)

    Range: 0.0 (identical, no damage) to 1.0 (maximally different).

    Falls back to an RMSE-based approximation when scikit-image is unavailable.

    Inputs:
        orig_gray   : uint8 grayscale array (H x W)
        recomp_gray : uint8 grayscale array (H x W)

    Returns:
        float in [0, 1]
    """
    if _SKIMAGE_AVAILABLE:
        ssim_score, _ = _ssim(orig_gray, recomp_gray, full=True)
        # ssim_score in [-1, 1]; clamp negative end to 0 before computing delta
        ssim_score = max(float(ssim_score), 0.0)
        return 1.0 - ssim_score
    else:
        # Fallback: SSIM ~= 1 - normalised_RMSE (monotonically equivalent
        # for the small differences typical of JPEG compression)
        orig_f   = orig_gray.astype(np.float32) / 255.0
        recomp_f = recomp_gray.astype(np.float32) / 255.0
        rmse     = float(np.sqrt(np.mean((orig_f - recomp_f) ** 2)))
        return min(rmse * 4.0, 1.0)   # x4 scales RMSE into a comparable range


# ==============================================================================
# 5. get_artifact_features
# ==============================================================================

def get_artifact_features(frame, quality=_JPEG_QUALITY):
    """
    Extract all Module 2 features from a single frame.

    Returns a dict so callers can use individual metrics for analysis or
    feed them separately into a classifier.

    Inputs:
        frame   : uint8 BGR numpy array
        quality : JPEG quality for the recompression step

    Returns:
        dict with keys:
            'mse'        : float  normalised MSE score        [0, 1]
            'ssim_delta' : float  SSIM drop after compression [0, 1]
            'combined'   : float  weighted combination        [0, 1]
    """
    recompressed = recompress_frame(frame, quality)

    orig_gray   = cv2.cvtColor(frame,        cv2.COLOR_BGR2GRAY)
    recomp_gray = cv2.cvtColor(recompressed, cv2.COLOR_BGR2GRAY)

    mse        = compute_mse_score(orig_gray, recomp_gray)
    ssim_delta = compute_ssim_delta(orig_gray, recomp_gray)
    combined   = round(min(_WEIGHT_MSE * mse + _WEIGHT_SSIM * ssim_delta, 1.0), 4)

    return {
        'mse':        round(mse,        4),
        'ssim_delta': round(ssim_delta, 4),
        'combined':   combined,
    }


# ==============================================================================
# 6. get_artifact_score_for_frame  -- main ensemble API (backward-compatible)
# ==============================================================================

def get_artifact_score_for_frame(frame, quality=_JPEG_QUALITY):
    """
    Score a single frame for compression artifacts.

    0.0 = no measurable artifact damage -> likely real
    1.0 = heavy structural damage after recompression -> likely fake

    This is the function imported by ensemble.py.

    Inputs:
        frame   : uint8 BGR numpy array
        quality : JPEG quality (default _JPEG_QUALITY = 75)

    Returns:
        float in [0.0, 1.0]
    """
    features = get_artifact_features(frame, quality)
    return features['combined']


# ==============================================================================
# 7. compute_artifact_score  -- video-level scoring
# ==============================================================================

def compute_artifact_score(frame_paths, sample_n=20):
    """
    Score a video represented as a list of frame file paths.

    Evenly samples up to sample_n frames, scores each, and returns the mean.

    Inputs:
        frame_paths : list[str]  -- paths to extracted frame images
        sample_n    : int        -- max frames to analyse (default 20)

    Returns:
        float -- mean artifact score across sampled frames [0.0, 1.0]
    """
    print("[compute_artifact_score] %d paths received" % len(frame_paths))

    if len(frame_paths) > sample_n:
        step = len(frame_paths) / sample_n
        sampled = [frame_paths[int(i * step)] for i in range(sample_n)]
    else:
        sampled = frame_paths

    print("[compute_artifact_score] Sampling %d frames" % len(sampled))

    scores = []
    for path in sampled:
        try:
            frame = cv2.imread(path)
            if frame is None:
                print("  [WARN] Cannot load: %s" % path)
                continue
            scores.append(get_artifact_score_for_frame(frame))
        except Exception as exc:
            print("  [WARN] %s: %s" % (path, exc))

    if not scores:
        print("[compute_artifact_score] No frames scored -- returning 0.0")
        return 0.0

    avg = round(float(np.mean(scores)), 3)
    verdict = "SUSPICIOUS" if avg >= 0.5 else "CLEAN"
    print("[compute_artifact_score] frames=%d  avg=%.3f  %s" % (len(scores), avg, verdict))
    return avg


# ==============================================================================
# 8. visualize_artifacts
# ==============================================================================

def visualize_artifacts(frame_path, save_path=None):
    """
    Side-by-side panel: ORIGINAL | RECOMPRESSED | DIFFERENCE x10

    Bright spots in the third panel mark where JPEG compression caused
    unexpectedly large damage -- a visual fingerprint of GAN artifacts.

    Inputs:
        frame_path : str  -- path to an image file
        save_path  : str or None  -- save destination; None -> display on screen
    """
    frame = cv2.imread(frame_path)
    if frame is None:
        print("[visualize_artifacts] Cannot load: %s" % frame_path)
        return

    recompressed = recompress_frame(frame)
    diff_gray    = get_difference_map(frame, recompressed)

    panel_orig   = frame.copy()
    panel_recomp = recompressed.copy()

    diff_vis   = np.clip(diff_gray.astype(np.uint16) * 10, 0, 255).astype(np.uint8)
    panel_diff = cv2.cvtColor(diff_vis, cv2.COLOR_GRAY2BGR)

    # Uniform height
    h = min(panel_orig.shape[0], panel_recomp.shape[0], panel_diff.shape[0])
    panels = [panel_orig, panel_recomp, panel_diff]
    panels = [cv2.resize(p, (p.shape[1], h)) if p.shape[0] != h else p for p in panels]
    panel_orig, panel_recomp, panel_diff = panels

    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    for panel, title in zip(
        [panel_orig, panel_recomp, panel_diff],
        ["ORIGINAL", "RECOMPRESSED (q=75)", "DIFF x10"]
    ):
        cv2.putText(panel, title, (8, 28), font, 0.65, (255, 255, 255), 2)

    combined_img = np.hstack([panel_orig, panel_recomp, panel_diff])

    # Score overlay
    features = get_artifact_features(frame)
    label = "MSE=%.4f  SSIM_delta=%.4f  score=%.4f" % (
        features['mse'], features['ssim_delta'], features['combined']
    )
    cv2.putText(combined_img, label,
                (8, combined_img.shape[0] - 10),
                font, 0.55, (0, 255, 255), 1)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        cv2.imwrite(save_path, combined_img)
        print("[visualize_artifacts] Saved: %s" % save_path)
    else:
        cv2.imshow("Module 2 - Artifact Visualisation", combined_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ==============================================================================
# 9. batch_score_folder
# ==============================================================================

def batch_score_folder(folder_path, label):
    """
    Score every JPEG in a folder and return a list of result dicts.

    Each dict: {"file": path, "mse": float, "ssim_delta": float,
                "combined": float, "label": label}

    Inputs:
        folder_path : str  -- directory containing .jpg files
        label       : str  -- "real" or "fake"

    Returns:
        list[dict]
    """
    print("\n[batch_score_folder] %s  label=%s" % (folder_path, label))

    try:
        all_files = os.listdir(folder_path)
    except FileNotFoundError:
        print("[batch_score_folder] Folder not found: %s" % folder_path)
        return []

    jpg_files = sorted(f for f in all_files if f.lower().endswith('.jpg'))
    print("[batch_score_folder] %d JPEG files found" % len(jpg_files))

    results = []
    for i, filename in enumerate(jpg_files):
        path = os.path.join(folder_path, filename)
        try:
            frame = cv2.imread(path)
            if frame is None:
                continue
            feats = get_artifact_features(frame)
            results.append({
                'file':       path,
                'mse':        feats['mse'],
                'ssim_delta': feats['ssim_delta'],
                'combined':   feats['combined'],
                'label':      label,
            })
            if (i + 1) % 10 == 0:
                print("  %d/%d" % (i + 1, len(jpg_files)))
        except Exception as exc:
            print("  [ERROR] %s: %s" % (filename, exc))

    if results:
        avg = round(float(np.mean([r['combined'] for r in results])), 3)
        print("[batch_score_folder] avg_combined=%.3f" % avg)

    return results


# ==============================================================================
# Self-test  (python artifact_module.py)
# ==============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("MODULE 2 - Compression Artifact Detection - self-test")
    print("scikit-image SSIM: %s" % ("available" if _SKIMAGE_AVAILABLE else "NOT available (using fallback)"))
    print("=" * 60)

    # ------------------------------------------------------------------
    # Synthetic image test
    # ------------------------------------------------------------------
    print("\n[1] Synthetic image test")
    test_frame = np.full((224, 224, 3), 128, dtype=np.uint8)
    rng = np.random.default_rng(42)
    test_frame = np.clip(
        test_frame.astype(np.int16) + rng.integers(-20, 20, test_frame.shape, dtype=np.int16),
        0, 255
    ).astype(np.uint8)

    feats = get_artifact_features(test_frame)
    print("  MSE score  : %.4f" % feats['mse'])
    print("  SSIM delta : %.4f" % feats['ssim_delta'])
    print("  Combined   : %.4f" % feats['combined'])
    score = get_artifact_score_for_frame(test_frame)
    print("  Score (ensemble API) : %.4f" % score)

    # ------------------------------------------------------------------
    # Quality sweep
    # ------------------------------------------------------------------
    print("\n[2] Quality sweep (lower quality -> higher artifact score)")
    for q in [95, 85, 75, 60, 40]:
        s = get_artifact_score_for_frame(test_frame, quality=q)
        print("  quality=%3d  score=%.4f" % (q, s))

    # ------------------------------------------------------------------
    # Folder test (only if data directories exist)
    # ------------------------------------------------------------------
    real_dir = 'data/real/frames/'
    fake_dir = 'data/fake/frames/'

    if os.path.isdir(real_dir) and os.path.isdir(fake_dir):
        print("\n[3] Real vs fake folder comparison")
        real_results = batch_score_folder(real_dir, 'real')
        fake_results = batch_score_folder(fake_dir, 'fake')

        avg_real = round(np.mean([r['combined'] for r in real_results]), 3) if real_results else 0.0
        avg_fake = round(np.mean([r['combined'] for r in fake_results]), 3) if fake_results else 0.0
        print("\n  avg real score : %.3f" % avg_real)
        print("  avg fake score : %.3f" % avg_fake)
        print("  fake > real    : %s" % ("YES (expected)" if avg_fake > avg_real else "NO -- check data"))
    else:
        print("\n[3] Skipping folder test -- %s or %s not found" % (real_dir, fake_dir))

    print("\n" + "=" * 60)
    print("Self-test complete.")
    print("=" * 60)
