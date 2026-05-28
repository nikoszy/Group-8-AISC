#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifact_module import get_artifact_score_for_frame
from src.freq_analysis.anomaly_scorer import fft_anomaly_score
from src.freq_analysis.texture_scorer import laplacian_score
from src.preprocessing.face_detector import detect_faces

EXP_ROOT = ROOT / "data" / "experiments"
INSPECT_SCRIPT = ROOT / "inspect_dataset.py"
ENSEMBLE_SCRIPT = ROOT / "ensemble.py"

MANIP_MAP = {
    "deepfakes": "Deepfakes",
    "face2face": "Face2Face",
    "faceswap": "FaceSwap",
}


@dataclass
class RunArtifacts:
    name: str
    base_dir: Path
    manifest: Path
    features_csv: Path
    model_pkl: Path
    plots_dir: Path
    viz_dir: Path
    log_file: Path


def _run(cmd: list[str], env: dict[str, str], log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def _parse_ensemble_metrics(log_file: Path) -> dict[str, float | None]:
    text = log_file.read_text(encoding="utf-8", errors="ignore")
    out: dict[str, float | None] = {
        "auc": None,
        "accuracy": None,
        "balanced_accuracy": None,
        "f1": None,
        "precision": None,
        "recall": None,
        "threshold": None,
    }
    patterns = {
        "auc": r"AUC\s*=\s*([0-9.]+)",
        "accuracy": r"Accuracy\s*=\s*([0-9.]+)",
        "balanced_accuracy": r"Balanced Acc\s*=\s*([0-9.]+)",
        "f1": r"F1\s*=\s*([0-9.]+)",
        "precision": r"Precision \(fake\)\s*=\s*([0-9.]+)",
        "recall": r"Recall\s+\(fake\)\s*=\s*([0-9.]+)",
        "threshold": r"threshold=([0-9.]+)\s+\(balanced-accuracy on val split\)",
    }
    for key, pat in patterns.items():
        m = re.findall(pat, text)
        if m:
            out[key] = float(m[-1])
    return out


def _build_env(name: str, fake_subdir: str) -> tuple[dict[str, str], RunArtifacts]:
    base = EXP_ROOT / name
    env = os.environ.copy()
    env["FF_DIR"] = str(ROOT / "data" / "FaceForensics++_C23")
    env["REAL_SUBDIR"] = "original"
    env["FAKE_SUBDIR"] = fake_subdir
    env["REAL_DIR"] = str(base / "real_frames")
    env["FAKE_DIR"] = str(base / "fake_frames")
    env["MANIFEST_PATH"] = str(base / "manifest.csv")
    env["VIDEO_EAR_CSV"] = str(base / "video_ear_scores.csv")
    env["FEATURES_CSV"] = str(base / "module3_features.csv")
    env["MODEL_PKL_PATH"] = str(base / "ensemble_model.pkl")
    env["PLOTS_DIR"] = str(base / "plots")
    env["VIZ_DIR"] = str(base / "visualizations")
    art = RunArtifacts(
        name=name,
        base_dir=base,
        manifest=base / "manifest.csv",
        features_csv=base / "module3_features.csv",
        model_pkl=base / "ensemble_model.pkl",
        plots_dir=base / "plots",
        viz_dir=base / "visualizations",
        log_file=base / "ensemble.log",
    )
    return env, art


def _load_feature_rows(features_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    import csv

    rows = []
    labels = []
    with features_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(
                [
                    float(row["ear_score"]),
                    float(row["artifact_score"]),
                    float(row["fft_score"]),
                    float(row["laplacian_score"]),
                ]
            )
            labels.append(int(row["label"]))
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=int)


def _predict_scores(bundle: dict[str, Any], features: np.ndarray) -> np.ndarray:
    model = bundle["model"]
    scaler = bundle["scaler"]
    x_scaled = scaler.transform(features)
    return model.predict_proba(x_scaled)[:, 1]


def _adv_frame_prob(bundle: dict[str, Any], frame: np.ndarray, ear_score: float = 0.5) -> float:
    x = np.array(
        [[ear_score, get_artifact_score_for_frame(frame), fft_anomaly_score(frame), laplacian_score(frame)]],
        dtype=np.float32,
    )
    x_scaled = bundle["scaler"].transform(x)
    return float(bundle["model"].predict_proba(x_scaled)[0, 1])


def _sample_images(folder: Path, n: int) -> list[Path]:
    imgs = [p for p in folder.glob("*.jpg")]
    if not imgs:
        return []
    random.shuffle(imgs)
    return imgs[: min(n, len(imgs))]


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    EXP_ROOT.mkdir(parents=True, exist_ok=True)

    run_outputs: dict[str, Any] = {"tracks": {}, "adv": {}, "notes": []}

    # ML-03 tracks: deepfakes baseline + Face2Face + FaceSwap
    artifacts: dict[str, RunArtifacts] = {}
    for key, fake_subdir in MANIP_MAP.items():
        env, art = _build_env(key, fake_subdir)
        artifacts[key] = art
        print(f"[run] inspect_dataset.py ({key})")
        _run(["python", str(INSPECT_SCRIPT)], env=env, log_file=art.base_dir / "inspect.log")
        print(f"[run] ensemble.py ({key})")
        _run(["python", "-u", str(ENSEMBLE_SCRIPT)], env=env, log_file=art.log_file)
        run_outputs["tracks"][key] = {
            "fake_subdir": fake_subdir,
            "artifacts": {
                "manifest": str(art.manifest.relative_to(ROOT)),
                "features_csv": str(art.features_csv.relative_to(ROOT)),
                "model_pkl": str(art.model_pkl.relative_to(ROOT)),
                "log": str(art.log_file.relative_to(ROOT)),
            },
            "metrics": _parse_ensemble_metrics(art.log_file),
        }

    # ADV-20: evaluate Deepfakes-trained model on Face2Face/FaceSwap features (OOD family shift)
    deep_bundle = joblib.load(artifacts["deepfakes"].model_pkl)
    threshold = float(deep_bundle.get("threshold", 0.5))
    run_outputs["adv"]["ADV-20"] = {}
    for target in ("face2face", "faceswap"):
        x, y = _load_feature_rows(artifacts[target].features_csv)
        scores = _predict_scores(deep_bundle, x)
        y_pred = (scores >= threshold).astype(int)
        run_outputs["adv"]["ADV-20"][target] = {
            "n": int(len(y)),
            "auc": float(roc_auc_score(y, scores)),
            "accuracy": float(accuracy_score(y, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
            "false_real_rate_on_fake": float(np.mean(scores[y == 1] < 0.4)) if np.any(y == 1) else None,
        }

    # ADV-21: double-compress real frames (proxy for screen-recorded real call)
    real_folder = artifacts["deepfakes"].base_dir / "real_frames"
    real_samples = _sample_images(real_folder, 120)
    adv21_scores = []
    for p in real_samples:
        img = cv2.imread(str(p))
        if img is None:
            continue
        ok, b1 = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            continue
        img1 = cv2.imdecode(b1, cv2.IMREAD_COLOR)
        ok, b2 = cv2.imencode(".jpg", img1, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue
        img2 = cv2.imdecode(b2, cv2.IMREAD_COLOR)
        adv21_scores.append(_adv_frame_prob(deep_bundle, img2, ear_score=0.5))
    run_outputs["adv"]["ADV-21"] = {
        "n": len(adv21_scores),
        "mean_prob_fake": float(np.mean(adv21_scores)) if adv21_scores else None,
        "false_fake_rate_at_threshold": float(np.mean(np.array(adv21_scores) >= threshold)) if adv21_scores else None,
    }

    # ADV-22: cartoonized/non-FF++ style transformation reliability check
    adv22_scores = []
    for p in real_samples[:100]:
        img = cv2.imread(str(p))
        if img is None:
            continue
        cartoon = cv2.stylization(img, sigma_s=60, sigma_r=0.45)
        adv22_scores.append(_adv_frame_prob(deep_bundle, cartoon, ear_score=0.5))
    run_outputs["adv"]["ADV-22"] = {
        "n": len(adv22_scores),
        "mean_prob_fake": float(np.mean(adv22_scores)) if adv22_scores else None,
        "false_fake_rate_at_threshold": float(np.mean(np.array(adv22_scores) >= threshold)) if adv22_scores else None,
        "note": "OOD synthetic style images; interpret as reliability stress, not calibrated classification.",
    }

    # ADV-23: partial-face/mask robustness + face detector hit-rate impact
    adv23_scores = []
    detect_ok_orig = 0
    detect_ok_occ = 0
    for p in real_samples[:120]:
        img = cv2.imread(str(p))
        if img is None:
            continue
        if detect_faces(img) is not None:
            detect_ok_orig += 1
        occ = img.copy()
        h, w = occ.shape[:2]
        # lower-face mask and side occlusion
        cv2.rectangle(occ, (0, int(0.58 * h)), (w, h), (0, 0, 0), -1)
        cv2.rectangle(occ, (0, int(0.25 * h)), (int(0.22 * w), int(0.78 * h)), (10, 10, 10), -1)
        if detect_faces(occ) is not None:
            detect_ok_occ += 1
        adv23_scores.append(_adv_frame_prob(deep_bundle, occ, ear_score=0.5))
    n_adv23 = len(adv23_scores)
    run_outputs["adv"]["ADV-23"] = {
        "n": n_adv23,
        "mean_prob_fake": float(np.mean(adv23_scores)) if adv23_scores else None,
        "false_fake_rate_at_threshold": float(np.mean(np.array(adv23_scores) >= threshold)) if adv23_scores else None,
        "face_detect_rate_original": float(detect_ok_orig / n_adv23) if n_adv23 else None,
        "face_detect_rate_occluded": float(detect_ok_occ / n_adv23) if n_adv23 else None,
    }

    # CNN gating recommendation inputs
    deep_auc = run_outputs["tracks"]["deepfakes"]["metrics"].get("auc")
    run_outputs["cnn_gating"] = {
        "handcrafted_auc_reference": deep_auc,
        "gating_threshold_auc": 0.65,
        "recommend_cnn_fallback_now": bool(deep_auc is not None and deep_auc < 0.65),
        "model_used_states": ["ensemble_learned", "equal_weights", "cnn_fallback"],
    }

    out_json = EXP_ROOT / "ml03_adv_ood_results.json"
    out_json.write_text(json.dumps(run_outputs, indent=2), encoding="utf-8")
    print(f"[ok] wrote {out_json}")


if __name__ == "__main__":
    main()
