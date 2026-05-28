# =============================================================================
# predict.py  —  Deepfake inference on a single video file
#
# Usage:
#   python predict.py path/to/video.mp4
#   python predict.py path/to/video.mp4 --frames 16
#   python predict.py path/to/video.mp4 --min-quality 0.20
#   python predict.py path/to/video.mp4 --retrain
#
# Pipeline per video:
#   1. Sample N frames evenly (or densely around scene changes)
#   2. Detect face in each frame (Haar cascade)
#   3. Score quality of each face frame
#   4. Extract handcrafted features (artifact, FFT slope, Laplacian)
#   5. Run CNN (EfficientNet-B0) if available
#   6. Ensemble CNN + LR scores with quality-aware weighting
#   7. Run temporal consistency analysis (MediaPipe, if installed)
#   8. Run rPPG pulse check (scipy bandpass, needs >= 30 face frames)
#   9. Combine all signals into a final calibrated verdict
# =============================================================================

import argparse
import os
import sys
import pickle
import warnings
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore")

from src.preprocessing.face_detector import detect_face_crop_with_bbox
from src.inference_combine import (
    load_cnn_alpha,
    sample_temporal_burst,
    run_rppg,
    combine_video_score,
)
from src.mrl.video_ear_score import video_ear_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FEATURES_CSV      = os.path.join("data", "module3_features.csv")
MODEL_PKL         = os.path.join("data", "ensemble_model.pkl")
STACK_BUNDLE_PATH = os.path.join("data", "stacking_bundle.pkl")

# ---------------------------------------------------------------------------
# CNN blend weight  (loaded from stacking_bundle.pkl if available)
# ---------------------------------------------------------------------------
# stacking_ensemble.py learns the optimal CNN / LR blend weight from real
# labelled val-set data.  We load it here so predict.py uses a principled
# weight instead of the hardcoded 0.65 guess.
#
# Fallback: if the bundle is missing or marked unreliable (alpha_reliable=False),
# we keep the old 0.65 / 0.35 split rather than crashing.

def _load_cnn_alpha(bundle_path=STACK_BUNDLE_PATH, fallback=0.65):
    """Return CNN blend weight; print status when run as CLI."""
    alpha = load_cnn_alpha(bundle_path, fallback)
    if not os.path.exists(bundle_path):
        return alpha
    try:
        with open(bundle_path, "rb") as fh:
            sb = pickle.load(fh)
        if sb.get("alpha_reliable", False):
            print(f"[predict] Stacking bundle loaded: "
                  f"alpha={alpha:.2f}  "
                  f"({alpha*100:.0f}% CNN + {(1-alpha)*100:.0f}% LR)  "
                  f"combined-AUC={sb.get('combined_auc', '?')}")
        else:
            print(f"[predict] Stacking bundle present but alpha_reliable=False "
                  f"— using fallback alpha={fallback:.2f}")
    except Exception as exc:
        print(f"[predict] Could not load stacking bundle ({exc}) "
              f"— using fallback alpha={fallback:.2f}")
    return alpha


_MRL_CKPT = Path("models") / "best_model.pth"


def _load_mrl_bundle():
    """Soft-load MRL checkpoint; returns (model, img_size, idx_to_label, device) or Nones."""
    if not _MRL_CKPT.exists():
        return None, 84, {}, None
    try:
        from src.mrl.inference import load_model, resolve_device
        device = resolve_device(None)
        model, img_size, idx_to_label = load_model(_MRL_CKPT, device=device)
        return model, img_size, idx_to_label, device
    except Exception as exc:
        print(f"[predict] MRL model not loaded ({exc}) — ear_score will be 0.5")
        return None, 84, {}, None


# Module-level constant: loaded once when predict.py is imported / run.
_CNN_ALPHA = _load_cnn_alpha()

# ---------------------------------------------------------------------------
# Verdict bands
# ---------------------------------------------------------------------------
VERDICT_BANDS = [
    (0.00, 0.20, "Authentic (high confidence)",  "REAL"),
    (0.20, 0.40, "Likely authentic",              "REAL"),
    (0.40, 0.60, "Inconclusive",                  "UNCERTAIN"),
    (0.60, 0.80, "Likely manipulated",            "FAKE"),
    (0.80, 1.00, "Manipulated (high confidence)", "FAKE"),
]


def verdict_band(prob):
    """Return (label, category) for a probability score."""
    for lo, hi, label, cat in VERDICT_BANDS:
        if lo <= prob < hi:
            return label, cat
    return "Manipulated (high confidence)", "FAKE"


# ---------------------------------------------------------------------------
# LR model loading / training
# ---------------------------------------------------------------------------
def _train_and_save(features_csv=FEATURES_CSV, model_pkl=MODEL_PKL):
    import csv
    from sklearn.linear_model    import LogisticRegression
    from sklearn.preprocessing   import StandardScaler
    from sklearn.calibration     import CalibratedClassifierCV
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics         import balanced_accuracy_score

    print(f"[predict] Training LR ensemble from {features_csv} ...")
    if not os.path.exists(features_csv):
        sys.exit(f"[ERROR] {features_csv} not found. Run ensemble.py first.")

    rows = []
    with open(features_csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({
                "artifact" : float(row.get("artifact_score",  0.5)),
                "fft"      : float(row.get("fft_score",       0.5)),
                "laplacian": float(row.get("laplacian_score",  0.5)),
                "ear"      : float(row.get("ear_score",        0.5)),
                "label"    : int(row["label"]),
                "video_id" : row["video_id"],
            })

    # 4 features — ear_score = 0.5 at inference (Module 1 not run live)
    features  = np.array([[r["artifact"], r["fft"], r["laplacian"], r["ear"]]
                           for r in rows], dtype=np.float32)
    labels    = np.array([r["label"]    for r in rows], dtype=int)
    video_ids = np.array([r["video_id"] for r in rows])

    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(gss.split(features, labels, groups=video_ids))

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(features[train_idx])
    y_train = labels[train_idx]

    base_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42,
                                  class_weight="balanced")
    model = CalibratedClassifierCV(base_lr, method="sigmoid", cv=3)
    model.fit(X_train, y_train)

    X_val   = scaler.transform(features[val_idx])
    y_val   = labels[val_idx]
    val_scores = model.predict_proba(X_val)[:, 1]
    best_t, best_ba = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 181):
        y_pred = (val_scores >= t).astype(int)
        ba = balanced_accuracy_score(y_val, y_pred)
        if ba > best_ba:
            best_ba, best_t = ba, float(t)

    bundle = {"model": model, "scaler": scaler, "threshold": best_t,
              "feature_names": ["artifact", "fft", "laplacian", "ear"],
              "calibrated": True}
    with open(model_pkl, "wb") as fh:
        pickle.dump(bundle, fh)
    print(f"[predict] Model saved -> {model_pkl}  "
          f"(threshold={best_t:.4f}, val-bal-acc={best_ba:.4f})")
    return bundle


def load_lr_model(model_pkl=MODEL_PKL):
    """Load LR bundle, or train + save if missing/stale."""
    if os.path.exists(model_pkl):
        try:
            with open(model_pkl, "rb") as fh:
                bundle = pickle.load(fh)
            if all(k in bundle for k in ("model", "scaler", "threshold")):
                print(f"[predict] LR model loaded from {model_pkl}  "
                      f"(threshold={bundle['threshold']:.4f})")
                return bundle
        except Exception as e:
            print(f"[predict] Could not load {model_pkl} ({e}), retraining ...")
    return _train_and_save()


# ---------------------------------------------------------------------------
# Handcrafted feature extraction  (mirrors ensemble.py)
# ---------------------------------------------------------------------------
def extract_features(crop_bgr, ear_score: float = 0.5):
    """Return [artifact, fft, laplacian, ear] for a 224×224 BGR crop."""
    from artifact_module import get_artifact_score_for_frame
    from src.freq_analysis.anomaly_scorer import fft_anomaly_score
    from src.freq_analysis.texture_scorer import laplacian_score

    try:
        artifact = float(get_artifact_score_for_frame(crop_bgr))
    except Exception:
        artifact = 0.5
    try:
        fft = float(fft_anomaly_score(crop_bgr))
    except Exception:
        fft = 0.5
    try:
        lap = float(laplacian_score(crop_bgr))
    except Exception:
        lap = 0.5
    return [artifact, fft, lap, float(ear_score)]


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------
def sample_frames(video_path, n_frames=8):
    """Sample n_frames evenly spaced frames. Returns (frames, fps, total, dur)."""
    cap   = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dur   = total / fps

    print(f"[predict] {os.path.basename(video_path)}"
          f"  |  {total} frames  {fps:.1f} fps  {dur:.1f}s")

    indices = np.linspace(0, max(total - 1, 0), n_frames, dtype=int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames, fps, total, dur


# ---------------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------------
def predict_video(video_path, lr_bundle, cnn_model=None,
                  n_frames=8, min_quality=0.10, threshold=None,
                  mrl_model=None, mrl_img_size=84,
                  mrl_idx_to_label=None, mrl_device=None):
    """
    Run full inference pipeline on a video file.

    Returns:
        frame_results : list of per-frame dicts
        summary       : dict with video-level aggregate scores
    """
    from src.quality_scorer import compute_frame_quality, quality_weighted_mean
    from src.cnn_runner      import cnn_predict

    lr_model  = lr_bundle["model"]
    lr_scaler = lr_bundle["scaler"]
    threshold = threshold if threshold is not None else lr_bundle["threshold"]

    frames, fps, total_frames, dur = sample_frames(video_path, n_frames)
    if not frames:
        sys.exit("[ERROR] No frames read from video.")

    if mrl_idx_to_label is None:
        mrl_idx_to_label = {}

    ear_score_video = video_ear_score(
        frames, dur,
        mrl_model=mrl_model,
        mrl_img_size=mrl_img_size,
        mrl_idx_to_label=mrl_idx_to_label,
        mrl_device=mrl_device,
    )

    frame_results = []
    face_crops    = []

    for i, frame in enumerate(frames):
        crop, bbox = detect_face_crop_with_bbox(frame)
        if crop is None:
            frame_results.append({"frame_idx": i, "face": False})
            face_crops.append(None)
            continue

        # Quality
        quality = compute_frame_quality(crop, bbox, frame.shape)

        # Handcrafted features → LR probability
        feats  = extract_features(crop, ear_score=ear_score_video)
        x_sc   = lr_scaler.transform(np.array([feats], dtype=np.float32))
        lr_prob = float(lr_model.predict_proba(x_sc)[0, 1])

        # CNN probability (if available)
        cnn_prob = cnn_predict(cnn_model, crop) if cnn_model is not None else None

        # Ensemble  (alpha loaded from stacking_bundle.pkl at startup)
        if cnn_prob is not None:
            frame_prob = _CNN_ALPHA * cnn_prob + (1.0 - _CNN_ALPHA) * lr_prob
            agreement  = abs(cnn_prob - lr_prob)
        else:
            frame_prob = lr_prob
            agreement  = None

        frame_results.append({
            "frame_idx": i,
            "face"     : True,
            "quality"  : round(quality, 4),
            "lr_prob"  : round(lr_prob, 4),
            "cnn_prob" : round(cnn_prob, 4) if cnn_prob is not None else None,
            "frame_prob": round(frame_prob, 4),
            "agreement": round(agreement, 4) if agreement is not None else None,
            "artifact" : round(feats[0], 4),
            "fft"      : round(feats[1], 4),
            "laplacian": round(feats[2], 4),
            "ear"      : round(feats[3], 4),
        })
        face_crops.append(crop)

    # Quality-weighted mean P(fake)
    face_results = [r for r in frame_results if r["face"]]
    if not face_results:
        summary = {
            "verdict": "UNKNOWN", "band": "No faces detected",
            "prob": 0.5, "quality_weighted_prob": 0.5,
            "temporal": None, "rppg": None,
            "n_face_frames": 0, "n_frames": len(frames),
            "fps": fps, "duration": dur, "cnn_active": cnn_model is not None,
        }
        return frame_results, summary

    probs     = [r["frame_prob"]  for r in face_results]
    qualities = [r["quality"]     for r in face_results]

    qw_prob   = quality_weighted_mean(probs, qualities, min_quality=min_quality)
    raw_prob  = float(np.mean(probs))

    temporal_result = sample_temporal_burst(video_path, fps)
    rppg_result = run_rppg(face_crops, fps=fps)
    combined, signal_count = combine_video_score(
        qw_prob, temporal_result, rppg_result
    )
    band_label, verdict_cat = verdict_band(combined)

    summary = {
        "verdict"              : verdict_cat,
        "band"                 : band_label,
        "prob"                 : round(combined, 4),
        "quality_weighted_prob": round(qw_prob, 4),
        "raw_mean_prob"        : round(raw_prob, 4),
        "ear_score"            : round(ear_score_video, 4),
        "threshold"            : threshold,
        "temporal"             : temporal_result,
        "rppg"                 : rppg_result,
        "n_face_frames"        : len(face_results),
        "n_frames"             : len(frames),
        "fps"                  : round(fps, 1),
        "duration"             : round(dur, 1),
        "cnn_active"           : cnn_model is not None,
        "signal_count"         : signal_count,
    }
    return frame_results, summary


# ---------------------------------------------------------------------------
# CLI output
# ---------------------------------------------------------------------------
def print_report(frame_results, summary, threshold):
    face_results = [r for r in frame_results if r["face"]]

    print()
    print("=" * 70)
    print("  FRAME-BY-FRAME RESULTS")
    print("=" * 70)

    has_cnn = any(r.get("cnn_prob") is not None for r in face_results)
    if has_cnn:
        print(f"  {'Frame':<6}  {'Quality':<9}  {'LR P(f)':>8}  "
              f"{'CNN P(f)':>9}  {'Ensemble':>9}  {'Agree':>6}")
        print("  " + "-" * 58)
    else:
        print(f"  {'Frame':<6}  {'Quality':<9}  {'LR P(f)':>8}  "
              f"{'FFT':>6}  {'Lap':>6}")
        print("  " + "-" * 50)

    for r in frame_results:
        if not r["face"]:
            print(f"  {r['frame_idx']:<6}  {'-- no face --'}")
            continue
        q_label = "Hi" if r["quality"] >= 0.70 else ("Med" if r["quality"] >= 0.40 else "Low")
        if has_cnn and r.get("cnn_prob") is not None:
            agree = f"{r['agreement']:.3f}" if r.get("agreement") is not None else "  n/a"
            print(f"  {r['frame_idx']:<6}  {r['quality']:.3f} {q_label:<4}  "
                  f"{r['lr_prob']:>8.4f}  {r['cnn_prob']:>9.4f}  "
                  f"{r['frame_prob']:>9.4f}  {agree:>6}")
        else:
            print(f"  {r['frame_idx']:<6}  {r['quality']:.3f} {q_label:<4}  "
                  f"{r['lr_prob']:>8.4f}  {r['fft']:>6.4f}  {r['laplacian']:>6.4f}")

    print()
    print("=" * 70)
    print("  AGGREGATE SCORES")
    print("=" * 70)
    print(f"  Frames analysed  : {summary['n_face_frames']} / {summary['n_frames']}")
    print(f"  CNN active       : {'Yes (EfficientNet-B0)' if summary['cnn_active'] else 'No (LR only)'}")
    print(f"  Raw mean P(fake) : {summary['raw_mean_prob']:.4f}")
    print(f"  Quality-weighted : {summary['quality_weighted_prob']:.4f}")

    t = summary.get("temporal")
    if t and t.get("available"):
        print(f"  Temporal score   : {t['score']:.4f}  "
              f"(smooth={t['smoothness']:.3f}  jitter={t['jitter']:.3f}  "
              f"frames={t['n_frames']})")
    else:
        note = t.get("note", "not available") if t else "module error"
        print(f"  Temporal score   : -- ({note})")

    r = summary.get("rppg")
    if r and r.get("available"):
        print(f"  rPPG pulse       : coherence={r['coherence']:.3f}  "
              f"SNR={r['snr_db']:.1f} dB  fake_score={r['fake_score']:.4f}")
    else:
        note = r.get("note", "not available") if r else "module error"
        print(f"  rPPG pulse       : -- ({note})")

    print(f"  FINAL P(fake)    : {summary['prob']:.4f}  "
          f"({summary['signal_count']} signal(s) combined)")
    print()

    # Verdict banner
    prob    = summary["prob"]
    verdict = summary["verdict"]
    band    = summary["band"]

    if verdict == "FAKE":
        print(f"  [!!] VERDICT: {band.upper()}")
        print(f"       P(fake) = {prob*100:.1f}%  |  P(real) = {(1-prob)*100:.1f}%")
    elif verdict == "REAL":
        print(f"  [OK] VERDICT: {band.upper()}")
        print(f"       P(fake) = {prob*100:.1f}%  |  P(real) = {(1-prob)*100:.1f}%")
    else:
        print(f"  [??] VERDICT: {band.upper()}")
        print(f"       P(fake) = {prob*100:.1f}%  — result is inconclusive")

    print("=" * 70)
    print()
    print("Confidence bands:")
    for lo, hi, label, _ in VERDICT_BANDS:
        marker = " <-- YOU ARE HERE" if lo <= prob < hi else ""
        print(f"  {lo:.2f}-{hi:.2f}  {label}{marker}")
    print()
    print("NOTE: Handcrafted features (LR) are weak on FF++ C23.")
    if not summary["cnn_active"]:
        print("      Install PyTorch for CNN ensemble: "
              "pip install torch torchvision --index-url "
              "https://download.pytorch.org/whl/cpu")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Deepfake detection inference — single video file."
    )
    parser.add_argument("video", help="Path to video (.mp4, .avi, .mov, …)")
    parser.add_argument("--frames", type=int, default=8, metavar="N",
                        help="Frames to sample (default: 8)")
    parser.add_argument("--min-quality", type=float, default=0.10, metavar="Q",
                        help="Skip frames below this quality (default: 0.10)")
    parser.add_argument("--threshold", type=float, default=None, metavar="T",
                        help="Override classification threshold")
    parser.add_argument("--retrain", action="store_true",
                        help="Force retrain LR model")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        sys.exit(f"[ERROR] Video not found: {args.video}")

    if args.retrain and os.path.exists(MODEL_PKL):
        os.remove(MODEL_PKL)

    # Load LR model
    lr_bundle = load_lr_model()

    # Load CNN (soft — None if torch not installed or checkpoint missing)
    from src.cnn_runner import load_cnn
    cnn_model = load_cnn(verbose=True)

    mrl_model, mrl_img_size, mrl_idx_to_label, mrl_device = _load_mrl_bundle()

    threshold = args.threshold if args.threshold is not None else lr_bundle["threshold"]

    frame_results, summary = predict_video(
        args.video, lr_bundle, cnn_model=cnn_model,
        n_frames=args.frames, min_quality=args.min_quality,
        threshold=threshold,
        mrl_model=mrl_model,
        mrl_img_size=mrl_img_size,
        mrl_idx_to_label=mrl_idx_to_label,
        mrl_device=mrl_device,
    )

    print_report(frame_results, summary, threshold)


if __name__ == "__main__":
    main()
