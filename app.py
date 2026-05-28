# =============================================================================
# app.py  —  Group 8 AISC: Deepfake Detection Web UI  (Streamlit)
#
# Run:
#   streamlit run app.py
# =============================================================================

import os
import sys
import pickle
import tempfile
import warnings
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

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
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Deepfake Detector — Group 8 AISC",
    page_icon="🎭",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FEATURES_CSV      = os.path.join("data", "module3_features.csv")
MODEL_PKL         = os.path.join("data", "ensemble_model.pkl")
STACK_BUNDLE_PATH = os.path.join("data", "stacking_bundle.pkl")

# ---------------------------------------------------------------------------
# CNN blend weight  (loaded from stacking_bundle.pkl if available)
# ---------------------------------------------------------------------------

def _load_cnn_alpha(bundle_path=STACK_BUNDLE_PATH, fallback=0.65):
    return load_cnn_alpha(bundle_path, fallback)


_MRL_CKPT = Path("models") / "best_model.pth"


@st.cache_resource(show_spinner="Loading MRL blink model...")
def get_mrl_model():
    if not _MRL_CKPT.exists():
        return None, 84, {}, None
    try:
        from src.mrl.inference import load_model, resolve_device
        device = resolve_device(None)
        model, img_size, idx_to_label = load_model(_MRL_CKPT, device=device)
        return model, img_size, idx_to_label, device
    except Exception:
        return None, 84, {}, None


# Module-level constant — resolved once when app.py is first imported.
_CNN_ALPHA = _load_cnn_alpha()

# ---------------------------------------------------------------------------
# Verdict bands
# ---------------------------------------------------------------------------
VERDICT_BANDS = [
    (0.00, 0.20, "Authentic (high confidence)",  "REAL",      "green",   "#1a9641"),
    (0.20, 0.40, "Likely authentic",              "REAL",      "success", "#a6d96a"),
    (0.40, 0.60, "Inconclusive",                  "UNCERTAIN", "warning", "#fdae61"),
    (0.60, 0.80, "Likely manipulated",            "FAKE",      "error",   "#d7191c"),
    (0.80, 1.00, "Manipulated (high confidence)", "FAKE",      "error",   "#bd0026"),
]


def verdict_band(prob):
    for lo, hi, label, cat, st_type, colour in VERDICT_BANDS:
        if lo <= prob < hi:
            return label, cat, st_type, colour
    return VERDICT_BANDS[-1][2], VERDICT_BANDS[-1][3], "error", "#bd0026"


# ---------------------------------------------------------------------------
# Model loading (cached once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading LR model...")
def get_lr_model():
    import csv
    from sklearn.linear_model    import LogisticRegression
    from sklearn.preprocessing   import StandardScaler
    from sklearn.calibration     import CalibratedClassifierCV
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics         import balanced_accuracy_score

    if os.path.exists(MODEL_PKL):
        try:
            with open(MODEL_PKL, "rb") as fh:
                bundle = pickle.load(fh)
            if all(k in bundle for k in ("model", "scaler", "threshold")):
                return bundle
        except Exception:
            pass

    if not os.path.exists(FEATURES_CSV):
        st.error(f"No features CSV found at {FEATURES_CSV}. Run ensemble.py first.")
        st.stop()

    rows = []
    with open(FEATURES_CSV, newline="", encoding="utf-8") as fh:
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

    X_val  = scaler.transform(features[val_idx])
    y_val  = labels[val_idx]
    val_s  = model.predict_proba(X_val)[:, 1]
    best_t, best_ba = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 181):
        ba = balanced_accuracy_score(y_val, (val_s >= t).astype(int))
        if ba > best_ba:
            best_ba, best_t = ba, float(t)

    bundle = {"model": model, "scaler": scaler, "threshold": best_t,
              "feature_names": ["artifact", "fft", "laplacian", "ear"],
              "calibrated": True}
    with open(MODEL_PKL, "wb") as fh:
        pickle.dump(bundle, fh)
    return bundle


@st.cache_resource(show_spinner="Loading CNN model (EfficientNet-B0)...")
def get_cnn_model():
    from src.cnn_runner import load_cnn
    return load_cnn(verbose=False)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_features(crop_bgr, ear_score: float = 0.5):
    """Return (artifact, fft, lap, ear) — 4 features matching the saved model."""
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
    return artifact, fft, lap, float(ear_score)


# ---------------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------------
def process_video(video_path, n_frames, lr_bundle, cnn_model, min_quality,
                  mrl_model=None, mrl_img_size=84,
                  mrl_idx_to_label=None, mrl_device=None):
    from src.quality_scorer import compute_frame_quality, quality_weighted_mean
    from src.cnn_runner      import cnn_predict

    lr_model  = lr_bundle["model"]
    lr_scaler = lr_bundle["scaler"]

    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dur   = total / fps

    indices   = np.linspace(0, max(total - 1, 0), n_frames, dtype=int)
    pending   = []
    thumbnails = []
    raw_frames = []

    if mrl_idx_to_label is None:
        mrl_idx_to_label = {}

    prog = st.progress(0, text="Analysing frames...")

    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        prog.progress((i + 1) / n_frames, text=f"Frame {i+1} / {n_frames}")

        if not ret:
            pending.append({"frame_idx": i, "face": False})
            thumbnails.append(None)
            raw_frames.append(None)
            continue

        raw_frames.append(frame)
        crop, bbox = detect_face_crop_with_bbox(frame)

        thumb = cv2.resize(frame, (320, 180))
        if bbox is not None:
            sx = 320 / frame.shape[1]
            sy = 180 / frame.shape[0]
            x, y, w, h = bbox
            cv2.rectangle(thumb,
                          (int(x * sx), int(y * sy)),
                          (int((x + w) * sx), int((y + h) * sy)),
                          (0, 255, 0), 2)
        thumbnails.append(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))

        if crop is None:
            pending.append({"frame_idx": i, "face": False})
            continue

        pending.append({
            "frame_idx": i, "face": True,
            "crop": crop, "bbox": bbox, "frame": frame,
        })

    cap.release()
    prog.empty()

    ear_score_video = video_ear_score(
        raw_frames, dur,
        mrl_model=mrl_model,
        mrl_img_size=mrl_img_size,
        mrl_idx_to_label=mrl_idx_to_label,
        mrl_device=mrl_device,
    )

    frame_results = []
    face_crops = []

    for item in pending:
        if not item["face"]:
            frame_results.append({"frame_idx": item["frame_idx"], "face": False})
            face_crops.append(None)
            continue

        crop = item["crop"]
        bbox = item["bbox"]
        frame = item["frame"]
        quality = compute_frame_quality(crop, bbox, frame.shape)
        artifact, fft, lap, ear = extract_features(crop, ear_score=ear_score_video)
        x_sc = lr_scaler.transform(np.array([[artifact, fft, lap, ear]], dtype=np.float32))
        lr_prob = float(lr_model.predict_proba(x_sc)[0, 1])

        cnn_prob = cnn_predict(cnn_model, crop) if cnn_model is not None else None

        if cnn_prob is not None:
            frame_prob = _CNN_ALPHA * cnn_prob + (1.0 - _CNN_ALPHA) * lr_prob
            agreement = abs(cnn_prob - lr_prob)
        else:
            frame_prob = lr_prob
            agreement = None

        frame_results.append({
            "frame_idx": item["frame_idx"],
            "face": True,
            "quality": round(quality, 4),
            "lr_prob": round(lr_prob, 4),
            "cnn_prob": round(cnn_prob, 4) if cnn_prob is not None else None,
            "frame_prob": round(frame_prob, 4),
            "agreement": round(agreement, 4) if agreement is not None else None,
            "fft": round(fft, 4),
            "laplacian": round(lap, 4),
            "artifact": round(artifact, 4),
        })
        face_crops.append(crop)

    face_results = [r for r in frame_results if r["face"]]
    if not face_results:
        return frame_results, thumbnails, {
            "verdict": "UNKNOWN", "band": "No faces detected",
            "prob": 0.5, "quality_weighted_prob": 0.5,
            "temporal": None, "rppg": None,
            "n_face_frames": 0, "n_frames": n_frames,
            "fps": round(fps, 1), "duration": round(dur, 1),
            "cnn_active": cnn_model is not None,
        }

    probs     = [r["frame_prob"] for r in face_results]
    qualities = [r["quality"]    for r in face_results]
    qw_prob   = quality_weighted_mean(probs, qualities, min_quality=min_quality)

    temporal_result = sample_temporal_burst(video_path, fps)
    rppg_result = run_rppg(face_crops, fps=fps)
    combined, _ = combine_video_score(qw_prob, temporal_result, rppg_result)

    band_label, verdict_cat, _, _ = verdict_band(combined)
    summary = {
        "verdict"              : verdict_cat,
        "band"                 : band_label,
        "prob"                 : round(combined, 4),
        "quality_weighted_prob": round(qw_prob, 4),
        "raw_mean_prob"        : round(float(np.mean(probs)), 4),
        "ear_score"            : round(ear_score_video, 4),
        "temporal"             : temporal_result,
        "rppg"                 : rppg_result,
        "n_face_frames"        : len(face_results),
        "n_frames"             : n_frames,
        "fps"                  : round(fps, 1),
        "duration"             : round(dur, 1),
        "cnn_active"           : cnn_model is not None,
    }
    return frame_results, thumbnails, summary


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("Deepfake Detector")
st.caption("Group 8 AISC  |  FF++ C23  |  LR ensemble + EfficientNet-B0 + temporal + rPPG")

# Load models
lr_bundle = get_lr_model()
cnn_model = get_cnn_model()
mrl_model, mrl_img_size, mrl_idx_to_label, mrl_device = get_mrl_model()

# Sidebar
with st.sidebar:
    st.header("Settings")
    n_frames    = st.slider("Frames to sample", 4, 64, 12, step=2)
    min_quality = st.slider("Min frame quality", 0.05, 0.50, 0.10, step=0.05,
                            help="Frames below this quality score are excluded from the weighted mean.")
    st.divider()

    st.markdown("**Model status**")
    st.markdown(f"- LR ensemble: {'OK' if lr_bundle else 'ERROR'}")
    _alpha_pct = int(round(_CNN_ALPHA * 100))
    _alpha_pct = int(round(_CNN_ALPHA * 100))
    st.markdown(f"- CNN (EfficientNet-B0): "
                f"{'OK  (' + str(_alpha_pct) + '% weight per frame)' if cnn_model else 'Not loaded (LR only)'}")
    st.markdown(f"- MRL blink (Module 1): "
                f"{'OK' if mrl_model is not None else 'Not loaded (ear=0.5)'}")

    try:
        from src.temporal_scorer import mediapipe_available
        mp_ok = mediapipe_available()
    except Exception:
        mp_ok = False
    st.markdown(f"- MediaPipe temporal: {'OK' if mp_ok else 'Not installed'}")

    try:
        from src.rppg_scorer import rppg_available
        rppg_ok = rppg_available()
    except Exception:
        rppg_ok = False
    st.markdown(f"- rPPG pulse check: {'OK (needs 30+ frames)' if rppg_ok else 'scipy missing'}")

    st.divider()
    st.markdown("""
**Verdict bands**
| P(fake) | Verdict |
|---|---|
| 0.00–0.20 | Authentic (high conf.) |
| 0.20–0.40 | Likely authentic |
| 0.40–0.60 | Inconclusive |
| 0.60–0.80 | Likely manipulated |
| 0.80–1.00 | Manipulated (high conf.) |
""")
    st.divider()
    st.warning(
        "FF++ C23 is hard. Handcrafted features alone are ~65% accurate. "
        "CNN adds significant lift. Results on your own videos may vary."
    )

# Upload
uploaded = st.file_uploader(
    "Upload a video to analyse",
    type=["mp4", "avi", "mov", "mkv", "webm"],
    help="Longer videos are fine — only the sampled frames are processed."
)

if uploaded is not None:
    suffix = os.path.splitext(uploaded.name)[-1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.video(uploaded)

    with st.spinner("Running inference..."):
        results, thumbs, summary = process_video(
            tmp_path, n_frames, lr_bundle, cnn_model, min_quality,
            mrl_model=mrl_model,
            mrl_img_size=mrl_img_size,
            mrl_idx_to_label=mrl_idx_to_label,
            mrl_device=mrl_device,
        )
    os.unlink(tmp_path)

    # ---------- Verdict banner ----------
    st.divider()
    prob    = summary["prob"]
    verdict = summary["verdict"]
    band    = summary["band"]
    _, _, st_type, _ = verdict_band(prob)

    msg = (f"## {band}\n"
           f"**P(fake) = {prob*100:.1f}%**  |  "
           f"{summary['n_face_frames']} face frames analysed")

    if st_type == "error":
        st.error(msg)
    elif st_type == "success":
        st.success(msg)
    elif st_type == "warning":
        st.warning(msg)
    else:
        st.info(msg)

    # ---------- Confidence bar ----------
    st.markdown("**Confidence bands:**")
    cols_band = st.columns(5)
    labels_only = ["< 0.20\nAuthentic", "0.20-0.40\nLikely real",
                   "0.40-0.60\nInconclusive", "0.60-0.80\nLikely fake",
                   "> 0.80\nFake"]
    for j, (cb, lbl) in enumerate(zip(cols_band, labels_only)):
        lo = j * 0.20
        hi = lo + 0.20
        if lo <= prob < hi:
            cb.metric(lbl, f"{prob:.3f}", delta=None)
        else:
            cb.metric(lbl, "")

    # ---------- Metadata ----------
    st.divider()
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Total frames", summary["n_frames"])
    mc2.metric("FPS", summary["fps"])
    mc3.metric("Duration (s)", summary["duration"])
    mc4.metric("Faces found", f"{summary['n_face_frames']} / {summary['n_frames']}")
    mc5.metric("CNN active", "Yes" if summary["cnn_active"] else "No")

    # ---------- Signal breakdown ----------
    st.subheader("Signal breakdown")
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Quality-weighted P(fake)", f"{summary['quality_weighted_prob']:.4f}",
               delta=f"{(summary['quality_weighted_prob'] - summary['raw_mean_prob']):.4f} vs raw",
               delta_color="off")
    t = summary.get("temporal")
    if t and t.get("available"):
        sc2.metric("Temporal score", f"{t['score']:.4f}",
                   help=f"smooth={t['smoothness']:.3f}  jitter={t['jitter']:.3f}  "
                        f"frames={t['n_frames']}")
    else:
        note = t.get("note", "n/a") if t else "module error"
        sc2.metric("Temporal score", "n/a", help=note)

    r = summary.get("rppg")
    if r and r.get("available"):
        sc3.metric("rPPG fake score", f"{r['fake_score']:.4f}",
                   help=f"coherence={r['coherence']:.3f}  SNR={r['snr_db']:.1f} dB  "
                        f"frames={r['n_frames']}")
    else:
        note = r.get("note", "n/a") if r else "n/a"
        sc3.metric("rPPG fake score", "n/a", help=note)

    # ---------- Sampled frames ----------
    st.divider()
    st.subheader("Sampled frames")
    face_results = [r for r in results if r["face"]]
    n_cols = min(len(results), 4)
    if n_cols > 0:
        cols = st.columns(n_cols)
        for i, (r, thumb) in enumerate(zip(results, thumbs)):
            col = cols[i % n_cols]
            if thumb is not None:
                col.image(thumb, use_container_width=True)
            if not r["face"]:
                col.caption(f"Frame {r['frame_idx']}: no face")
            else:
                q = r["quality"]
                q_label = "Hi" if q >= 0.70 else ("Med" if q >= 0.40 else "Low")
                prob_here = r["frame_prob"]
                icon = "🔴" if prob_here >= 0.60 else ("🟡" if prob_here >= 0.40 else "🟢")
                cap_parts = [f"Frame {r['frame_idx']}  {icon}  P(f)={prob_here:.3f}",
                             f"Quality={q:.3f} ({q_label})"]
                if r.get("cnn_prob") is not None:
                    cap_parts.append(f"CNN={r['cnn_prob']:.3f}  LR={r['lr_prob']:.3f}")
                col.caption("\n".join(cap_parts))

    # ---------- Per-frame table ----------
    st.divider()
    st.subheader("Per-frame scores")
    if face_results:
        import pandas as pd
        cols_to_show = ["frame_idx", "quality", "lr_prob", "frame_prob",
                        "fft", "laplacian", "artifact"]
        if any(r.get("cnn_prob") is not None for r in face_results):
            cols_to_show.insert(3, "cnn_prob")
            cols_to_show.insert(4, "agreement")
        df = pd.DataFrame(face_results)[cols_to_show].copy()
        df.columns = [c.replace("_", " ").title() for c in cols_to_show]
        st.dataframe(
            df.style.background_gradient(subset=["Frame Prob"], cmap="RdYlGn_r"),
            use_container_width=True,
            hide_index=True,
        )

        # Chart
        st.subheader("P(fake) per frame (with quality overlay)")
        chart_df = pd.DataFrame({
            "Frame": [str(r["frame_idx"]) for r in face_results],
            "P(fake)": [r["frame_prob"] for r in face_results],
            "Quality": [r["quality"] for r in face_results],
        }).set_index("Frame")
        st.bar_chart(chart_df[["P(fake)"]])

else:
    st.info("Upload a video above to get started.")
    st.markdown(f"""
### Detection pipeline

1. **Face detection** — Haar cascade locates the face in each sampled frame
2. **Quality scoring** — each frame scored for sharpness, size, brightness; low-quality frames get low weight
3. **Handcrafted features** — FFT spectral slope, Laplacian texture variance, JPEG artifact score
4. **CNN** — EfficientNet-B0 (trained on FF++ C23) runs on each face crop
5. **Ensemble** — learned blend (CNN {int(round(_CNN_ALPHA*100))}% + LR {int(round((1-_CNN_ALPHA)*100))}%), then quality-weighted mean across frames
6. **Temporal** — optical flow (Lucas-Kanade) tracks feature points; unnatural smoothness or jitter = deepfake signal
7. **rPPG** — bandpass-filtered green-channel signal checks for a heartbeat (needs 30+ frames)
8. **Verdict** — calibrated P(fake) mapped to a 5-band confidence scale
""")
