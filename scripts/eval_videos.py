#!/usr/bin/env python3
"""Batch video evaluation CLI — runs predict.py pipeline on labelled FF++ clips.

Usage::

    python scripts/eval_videos.py \\
        --real  data/FaceForensics++_C23/original/000.mp4 \\
        --fake  data/FaceForensics++_C23/Deepfakes/000_003.mp4

    python scripts/eval_videos.py --manifest eval_list.txt

Each line in ``--manifest`` is: ``path\\tlabel`` where label is real or fake.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predict import (  # noqa: E402
    _load_mrl_bundle,
    load_lr_model,
    predict_video,
    verdict_band,
)


def _parse_manifest(path: Path) -> list[tuple[Path, str]]:
    rows: list[tuple[Path, str]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) == 1:
                parts = line.split(",")
            p = Path(parts[0].strip())
            label = parts[1].strip().lower() if len(parts) > 1 else "unknown"
            rows.append((p, label))
    return rows


def _evaluate_one(video_path: Path, label: str, lr_bundle, cnn_model,
                  mrl_model, mrl_img_size, mrl_idx_to_label, mrl_device,
                  n_frames: int, min_quality: float) -> dict:
    if not video_path.exists():
        return {"path": str(video_path), "label": label, "error": "file not found"}

    _, summary = predict_video(
        str(video_path), lr_bundle, cnn_model=cnn_model,
        n_frames=n_frames, min_quality=min_quality,
        mrl_model=mrl_model, mrl_img_size=mrl_img_size,
        mrl_idx_to_label=mrl_idx_to_label, mrl_device=mrl_device,
    )
    band_label, verdict_cat = verdict_band(summary["prob"])
    return {
        "path": str(video_path),
        "label": label,
        "prob_fake": summary["prob"],
        "quality_weighted_prob": summary["quality_weighted_prob"],
        "ear_score": summary.get("ear_score"),
        "verdict": verdict_cat,
        "band": band_label,
        "n_face_frames": summary["n_face_frames"],
        "cnn_active": summary["cnn_active"],
        "temporal_available": bool(
            summary.get("temporal") and summary["temporal"].get("available")
        ),
        "rppg_available": bool(
            summary.get("rppg") and summary["rppg"].get("available")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate videos through predict.py pipeline")
    parser.add_argument("--real", type=Path, help="Path to a known-real video")
    parser.add_argument("--fake", type=Path, help="Path to a known-fake video")
    parser.add_argument("--manifest", type=Path, help="TSV/CSV manifest: path, label")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--min-quality", type=float, default=0.10)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval_results.csv",
    )
    args = parser.parse_args()

    jobs: list[tuple[Path, str]] = []
    if args.manifest:
        jobs.extend(_parse_manifest(args.manifest))
    if args.real:
        jobs.append((args.real, "real"))
    if args.fake:
        jobs.append((args.fake, "fake"))

    if not jobs:
        parser.error("Provide --real/--fake and/or --manifest")

    lr_bundle = load_lr_model()
    from src.cnn_runner import load_cnn
    cnn_model = load_cnn(verbose=False)
    mrl_model, mrl_img_size, mrl_idx_to_label, mrl_device = _load_mrl_bundle()

    results = []
    for path, label in jobs:
        print(f"Evaluating {path} ({label}) ...")
        row = _evaluate_one(
            path, label, lr_bundle, cnn_model,
            mrl_model, mrl_img_size, mrl_idx_to_label, mrl_device,
            args.frames, args.min_quality,
        )
        results.append(row)
        if "error" in row:
            print(f"  ERROR: {row['error']}")
        else:
            print(
                f"  P(fake)={row['prob_fake']:.4f}  verdict={row['verdict']}  "
                f"faces={row['n_face_frames']}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path", "label", "prob_fake", "quality_weighted_prob", "ear_score",
        "verdict", "band", "n_face_frames", "cnn_active",
        "temporal_available", "rppg_available",
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nWrote {len(results)} rows -> {args.output}")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()
