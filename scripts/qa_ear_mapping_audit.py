#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.blink_analysis.ear_scorer import resolve_source_video_paths


def _label_name(label_raw: str) -> str:
    try:
        return "real" if int(label_raw) == 0 else "fake"
    except Exception:
        return f"unknown({label_raw})"


def _init_bucket() -> dict:
    return {
        "rows_total": 0,
        "rows_unresolved": 0,
        "unique_video_ids_total": 0,
        "unique_video_ids_unresolved": 0,
    }


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def audit_manifest(manifest_path: Path) -> dict:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows = list(csv.DictReader(manifest_path.open("r", newline="", encoding="utf-8")))

    per_class = defaultdict(_init_bucket)
    per_source = defaultdict(_init_bucket)
    per_pair = defaultdict(_init_bucket)

    unique_keys = set()
    unresolved_unique_keys = set()
    unresolved_examples = []

    for row in rows:
        video_id = (row.get("video_id") or "").strip()
        source_dataset = (row.get("source_dataset") or "").strip()
        label_name = _label_name(row.get("label", ""))
        key = (video_id, source_dataset, label_name)
        unique_keys.add(key)

        resolved_paths = resolve_source_video_paths(video_id, source_dataset)
        unresolved = len(resolved_paths) == 0
        if unresolved:
            unresolved_unique_keys.add(key)
            if len(unresolved_examples) < 50:
                unresolved_examples.append(
                    {
                        "video_id": video_id,
                        "source_dataset": source_dataset,
                        "class": label_name,
                    }
                )

        for bucket in (per_class[label_name], per_source[source_dataset], per_pair[(label_name, source_dataset)]):
            bucket["rows_total"] += 1
            if unresolved:
                bucket["rows_unresolved"] += 1

    for video_id, source_dataset, label_name in unique_keys:
        per_class[label_name]["unique_video_ids_total"] += 1
        per_source[source_dataset]["unique_video_ids_total"] += 1
        per_pair[(label_name, source_dataset)]["unique_video_ids_total"] += 1
    for video_id, source_dataset, label_name in unresolved_unique_keys:
        per_class[label_name]["unique_video_ids_unresolved"] += 1
        per_source[source_dataset]["unique_video_ids_unresolved"] += 1
        per_pair[(label_name, source_dataset)]["unique_video_ids_unresolved"] += 1

    def finalize(stats: dict) -> dict:
        out = {}
        for k, v in sorted(stats.items(), key=lambda kv: str(kv[0])):
            out_key = f"{k[0]} | {k[1]}" if isinstance(k, tuple) else str(k)
            out[out_key] = {
                **v,
                "rows_unresolved_rate": _rate(v["rows_unresolved"], v["rows_total"]),
                "unique_video_ids_unresolved_rate": _rate(
                    v["unique_video_ids_unresolved"], v["unique_video_ids_total"]
                ),
            }
        return out

    total_rows = len(rows)
    total_unresolved_rows = sum(1 for row in rows if not resolve_source_video_paths(
        (row.get("video_id") or "").strip(), (row.get("source_dataset") or "").strip()
    ))
    total_unique = len(unique_keys)
    total_unique_unresolved = len(unresolved_unique_keys)

    return {
        "manifest_path": str(manifest_path),
        "rows_total": total_rows,
        "rows_unresolved": total_unresolved_rows,
        "rows_unresolved_rate": _rate(total_unresolved_rows, total_rows),
        "unique_video_ids_total": total_unique,
        "unique_video_ids_unresolved": total_unique_unresolved,
        "unique_video_ids_unresolved_rate": _rate(total_unique_unresolved, total_unique),
        "by_class": finalize(per_class),
        "by_source_dataset": finalize(per_source),
        "by_class_and_source_dataset": finalize(per_pair),
        "unresolved_examples": unresolved_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit EAR source-video resolution coverage.")
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "data" / "manifest.csv"),
        help="Path to manifest CSV (default: data/manifest.csv)",
    )
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "docs" / "ear_mapping_audit.json"),
        help="Where to write machine-readable audit report",
    )
    args = parser.parse_args()

    report = audit_manifest(Path(args.manifest))
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[ok] wrote {out_path}")
    print(
        "overall:"
        f" rows_unresolved={report['rows_unresolved']}/{report['rows_total']}"
        f" ({report['rows_unresolved_rate']:.2%}),"
        f" unique_unresolved={report['unique_video_ids_unresolved']}/{report['unique_video_ids_total']}"
        f" ({report['unique_video_ids_unresolved_rate']:.2%})"
    )
    print("by_class:")
    for key, stats in report["by_class"].items():
        print(
            f"  {key}: unique_unresolved={stats['unique_video_ids_unresolved']}/{stats['unique_video_ids_total']}"
            f" ({stats['unique_video_ids_unresolved_rate']:.2%})"
        )
    print("by_source_dataset:")
    for key, stats in report["by_source_dataset"].items():
        print(
            f"  {key}: unique_unresolved={stats['unique_video_ids_unresolved']}/{stats['unique_video_ids_total']}"
            f" ({stats['unique_video_ids_unresolved_rate']:.2%})"
        )


if __name__ == "__main__":
    main()
