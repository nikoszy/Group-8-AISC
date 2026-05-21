#!/usr/bin/env python3
"""Run MRL eye-state inference on directories of extracted video frames.

Each subdirectory of ``--video-dir`` is treated as one video.  Inside each
subdirectory, every ``.jpg`` / ``.jpeg`` file is a frame.

Example::

    python run_inference.py \\
        --video-dir  data/processed/frames \\
        --output-dir data/inference_results

Produces one CSV per video in ``--output-dir`` with columns:
    video_id, frame_id, timestamp, eye_state, blink_count, blinks_per_minute
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mrl.inference import load_model, process_video  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch eye-state inference + blink counting",
    )
    p.add_argument(
        "--video-dir",
        type=Path,
        required=True,
        help="Directory whose subdirectories each contain JPEG frames for one video",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write per-video CSV results",
    )
    p.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "models" / "best_model.pth",
        help="Path to trained checkpoint (default: models/best_model.pth)",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Assumed frame rate for timestamp / BPM calculation (default: 30)",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device (e.g. 'cpu', 'cuda').  Auto-detected if omitted.",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    import torch
    device = (
        torch.device(args.device) if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    logging.info("Loading model from %s", args.model_path)
    model, img_size, class_to_label = load_model(args.model_path, device)
    logging.info(
        "Model ready — img_size=%d  classes=%s  device=%s",
        img_size, class_to_label, device,
    )

    # ------------------------------------------------------------------
    # Discover video subdirectories
    # ------------------------------------------------------------------
    video_dir = Path(args.video_dir)
    if not video_dir.is_dir():
        sys.exit(f"ERROR: --video-dir does not exist: {video_dir}")

    video_dirs = sorted(
        d for d in video_dir.iterdir() if d.is_dir()
    )

    if not video_dirs:
        sys.exit(f"ERROR: no subdirectories found in {video_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Process each video
    # ------------------------------------------------------------------
    total_frames = 0
    total_blinks = 0

    for vdir in video_dirs:
        logging.info("Processing %s ...", vdir.name)
        df = process_video(
            vdir, model, img_size, class_to_label, device, fps=args.fps,
        )

        if df.empty:
            logging.warning("  Skipped %s (no frames)", vdir.name)
            continue

        csv_path = output_dir / f"{vdir.name}.csv"
        df.to_csv(csv_path, index=False)

        n_frames = len(df)
        n_blinks = int(df["blink_count"].iloc[-1])
        total_frames += n_frames
        total_blinks += n_blinks
        logging.info(
            "  %s — %d frames, %d blinks  → %s",
            vdir.name, n_frames, n_blinks, csv_path,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 55)
    print("INFERENCE COMPLETE")
    print("=" * 55)
    print(f"  Videos processed : {len(video_dirs)}")
    print(f"  Total frames     : {total_frames}")
    print(f"  Total blinks     : {total_blinks}")
    print(f"  Results in       : {output_dir}")
    print("=" * 55)


if __name__ == "__main__":
    main()
