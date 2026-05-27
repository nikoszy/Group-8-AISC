"""Module 1 scoring: map blinks-per-minute to deepfake confidence.

Reads per-video inference CSVs (from ``run_inference.py``), takes the final
frame row for each video, and applies a sigmoid over BPM.

Usage::

    python -m src.mrl.score
    python src/mrl/score.py
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def bpm_to_confidence(bpm: float) -> float:
    """Map blinks-per-minute to a 0–1 deepfake confidence score.

    Uses ``1 / (1 + exp((bpm - 15) / 10))`` so that low BPM (e.g. 0) → ~1
    (likely fake) and high BPM → ~0 (likely real).
    """
    return 1.0 / (1.0 + math.exp((bpm - 15.0) / 10.0))


def _summarize_results_dir(results_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []

    for csv_path in sorted(results_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        last = df.iloc[-1]
        bpm = float(last["blinks_per_minute"])

        if "is_fake" in last.index:
            is_fake = int(last["is_fake"])
        else:
            is_fake = -1

        video_id = last["video_id"] if "video_id" in last.index else csv_path.stem

        rows.append({
            "video_id": video_id,
            "blinks_per_minute": bpm,
            "deepfake_confidence": bpm_to_confidence(bpm),
            "is_fake": is_fake,
        })

    return pd.DataFrame(
        rows,
        columns=["video_id", "blinks_per_minute", "deepfake_confidence", "is_fake"],
    )


def generate_summary_csv(results_dir: str, output_path: str) -> None:
    """Build a per-video summary CSV from all inference results in *results_dir*."""
    summary = _summarize_results_dir(Path(results_dir))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)


def main() -> None:
    data_dir = _PROJECT_ROOT / "data"
    combined_parts: list[pd.DataFrame] = []

    for name in ("results_real", "results_fake"):
        results_path = data_dir / name
        if not results_path.is_dir():
            continue
        combined_parts.append(_summarize_results_dir(results_path))

    if not combined_parts:
        raise FileNotFoundError(
            "No results directories found under data/ "
            "(expected results_real and/or results_fake)."
        )

    combined = pd.concat(combined_parts, ignore_index=True)
    output_path = data_dir / "module1_output.csv"
    combined.to_csv(output_path, index=False)
    print(f"Wrote {len(combined)} videos -> {output_path}")


if __name__ == "__main__":
    main()
