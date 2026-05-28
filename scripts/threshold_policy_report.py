#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class OperatingPoint:
    threshold: float
    accuracy: float
    balanced_accuracy: float
    precision_fake: float
    recall_fake: float
    false_fake_on_real: float
    false_real_on_fake: float
    support_real: int
    support_fake: int
    feasible: bool


def load_features_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    labels: list[int] = []
    video_ids: list[str] = []
    with path.open(newline="", encoding="utf-8") as fh:
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
            video_ids.append(row["video_id"])
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
        np.asarray(video_ids),
    )


def collect_validation_scores(
    features: np.ndarray, labels: np.ndarray, video_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    from ensemble import train_ensemble

    model, scaler, x_val, y_val, _ = train_ensemble(
        features, labels, video_ids, random_state=42
    )
    val_scores = model.predict_proba(x_val)[:, 1]
    return y_val.astype(int), val_scores.astype(float)


def evaluate_point(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> OperatingPoint:
    y_pred = (scores >= threshold).astype(int)

    real_mask = y_true == 0
    fake_mask = y_true == 1
    fp = int(np.sum((y_pred == 1) & real_mask))
    fn = int(np.sum((y_pred == 0) & fake_mask))
    n_real = int(np.sum(real_mask))
    n_fake = int(np.sum(fake_mask))

    false_fake_on_real = float(fp / n_real) if n_real else 0.0
    false_real_on_fake = float(fn / n_fake) if n_fake else 0.0

    return OperatingPoint(
        threshold=float(threshold),
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        precision_fake=float(precision_score(y_true, y_pred, zero_division=0)),
        recall_fake=float(recall_score(y_true, y_pred, zero_division=0)),
        false_fake_on_real=false_fake_on_real,
        false_real_on_fake=false_real_on_fake,
        support_real=n_real,
        support_fake=n_fake,
        feasible=False,
    )


def make_grid(min_t: float, max_t: float, n_steps: int) -> np.ndarray:
    return np.linspace(min_t, max_t, n_steps)


def select_candidates(
    points: list[OperatingPoint],
    ffr_cap: float,
    min_recall_fake: float,
    probe_thresholds: list[float],
) -> tuple[list[OperatingPoint], dict[str, OperatingPoint | None]]:
    feasible = []
    for p in points:
        ok = (p.false_fake_on_real <= ffr_cap) and (p.recall_fake >= min_recall_fake)
        p.feasible = bool(ok)
        if ok:
            feasible.append(p)

    constrained_best = None
    if feasible:
        constrained_best = max(
            feasible,
            key=lambda p: (p.recall_fake, p.balanced_accuracy, p.accuracy, -p.threshold),
        )

    min_false_fake = min(
        points, key=lambda p: (p.false_fake_on_real, p.false_real_on_fake, -p.threshold)
    )
    max_recall = max(points, key=lambda p: (p.recall_fake, -p.false_fake_on_real, -p.threshold))
    bal_best = max(points, key=lambda p: (p.balanced_accuracy, p.recall_fake, -p.threshold))

    keys = {}
    for name, point in {
        "constrained_best": constrained_best,
        "lowest_false_fake": min_false_fake,
        "max_recall_fake": max_recall,
        "best_balanced_accuracy": bal_best,
    }.items():
        if point is None:
            keys[name] = None
        else:
            keys[name] = point

    probes = []
    for t in probe_thresholds:
        nearest = min(points, key=lambda p: abs(p.threshold - t))
        probes.append(nearest)

    unique: list[OperatingPoint] = []
    seen = set()
    for p in [
        keys["constrained_best"],
        keys["lowest_false_fake"],
        keys["max_recall_fake"],
        keys["best_balanced_accuracy"],
        *probes,
    ]:
        if p is None:
            continue
        stamp = round(p.threshold, 6)
        if stamp not in seen:
            seen.add(stamp)
            unique.append(p)
    return unique, keys


def write_markdown_report(
    output_path: Path,
    auc: float,
    ffr_cap: float,
    min_recall_fake: float,
    points: list[OperatingPoint],
    keys: dict[str, OperatingPoint | None],
) -> None:
    def fmt(v: float) -> str:
        return f"{v:.4f}"

    lines = []
    lines.append("# Threshold Operating-Point Analysis")
    lines.append("")
    lines.append(f"- Validation ROC-AUC (score-only): `{auc:.4f}`")
    lines.append(f"- Constraint: false-FAKE on real <= `{ffr_cap:.3f}`")
    lines.append(f"- Constraint: fake recall >= `{min_recall_fake:.3f}`")
    lines.append("")
    lines.append("## Candidate operating points")
    lines.append("")
    lines.append("| Candidate | Threshold | FFR(real->fake) | Recall(fake) | FNR(fake->real) | Balanced Acc | Accuracy | Precision(fake) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    by_obj = {
        "constrained_best": "Constrained best",
        "lowest_false_fake": "Lowest false-FAKE",
        "max_recall_fake": "Max recall-fake",
        "best_balanced_accuracy": "Best balanced-acc",
    }
    for key, label in by_obj.items():
        p = keys.get(key)
        if p is None:
            continue
        lines.append(
            f"| {label} | {fmt(p.threshold)} | {fmt(p.false_fake_on_real)} | {fmt(p.recall_fake)} | "
            f"{fmt(p.false_real_on_fake)} | {fmt(p.balanced_accuracy)} | {fmt(p.accuracy)} | {fmt(p.precision_fake)} |"
        )

    lines.append("")
    lines.append("## Practical threshold policy")
    lines.append("")
    if keys.get("constrained_best") is None:
        lines.append(
            "No threshold in the scanned grid satisfies both constraints. "
            "Relax one constraint or inspect calibration quality."
        )
    else:
        p = keys["constrained_best"]
        assert p is not None
        lines.append(
            f"Use `threshold={fmt(p.threshold)}` as the primary operating point, because it satisfies the "
            f"false-FAKE cap (`{fmt(p.false_fake_on_real)}` <= `{ffr_cap:.3f}`) while preserving fake recall "
            f"(`{fmt(p.recall_fake)}` >= `{min_recall_fake:.3f}`)."
        )
        lines.append(
            "Guardrail: if future validation fails either constraint, pick the nearest threshold that keeps "
            "`false_fake_on_real` within cap first, then maximize `recall_fake`."
        )

    if points:
        lines.append("")
        lines.append("## Additional sampled thresholds")
        lines.append("")
        lines.append("| Threshold | FFR(real->fake) | Recall(fake) | Balanced Acc | Accuracy |")
        lines.append("|---:|---:|---:|---:|---:|")
        for p in points[:6]:
            lines.append(
                f"| {fmt(p.threshold)} | {fmt(p.false_fake_on_real)} | {fmt(p.recall_fake)} | "
                f"{fmt(p.balanced_accuracy)} | {fmt(p.accuracy)} |"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Constrained threshold operating-point analysis")
    parser.add_argument("--features-csv", default=str(ROOT / "data" / "module3_features.csv"))
    parser.add_argument("--output-json", default=str(ROOT / "docs" / "threshold_operating_points.json"))
    parser.add_argument("--output-md", default=str(ROOT / "docs" / "threshold_operating_points.md"))
    parser.add_argument("--ffr-cap", type=float, default=0.05, help="Max false-FAKE on real")
    parser.add_argument("--min-recall-fake", type=float, default=0.95, help="Min recall on fake")
    parser.add_argument("--min-threshold", type=float, default=0.01)
    parser.add_argument("--max-threshold", type=float, default=0.99)
    parser.add_argument("--n-steps", type=int, default=197)
    parser.add_argument(
        "--probe-thresholds",
        nargs="*",
        type=float,
        default=[0.05, 0.15, 0.5, 0.9],
        help="Always include nearest candidates to these thresholds",
    )
    args = parser.parse_args()

    features_csv = Path(args.features_csv)
    y_true, y_scores = None, None
    features, labels, video_ids = load_features_csv(features_csv)
    y_true, y_scores = collect_validation_scores(features, labels, video_ids)

    auc = float(roc_auc_score(y_true, y_scores))
    thresholds = make_grid(args.min_threshold, args.max_threshold, args.n_steps)
    points = [evaluate_point(y_true, y_scores, t) for t in thresholds]
    candidates, keys = select_candidates(
        points, args.ffr_cap, args.min_recall_fake, args.probe_thresholds
    )

    out = {
        "meta": {
            "features_csv": str(features_csv),
            "ffr_cap": float(args.ffr_cap),
            "min_recall_fake": float(args.min_recall_fake),
            "threshold_grid": {
                "min_threshold": float(args.min_threshold),
                "max_threshold": float(args.max_threshold),
                "n_steps": int(args.n_steps),
            },
            "val_auc": auc,
            "n_val": int(len(y_true)),
        },
        "candidates": [asdict(c) for c in candidates],
        "named_candidates": {k: (asdict(v) if v is not None else None) for k, v in keys.items()},
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown_report(output_md, auc, args.ffr_cap, args.min_recall_fake, candidates, keys)

    print(f"[ok] wrote {output_json}")
    print(f"[ok] wrote {output_md}")


if __name__ == "__main__":
    main()
