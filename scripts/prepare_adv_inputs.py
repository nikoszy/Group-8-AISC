#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifest.csv"
DEFAULT_OUT_ROOT = REPO_ROOT / "data" / "experiments" / "adv_inputs"


@dataclass(frozen=True)
class ManifestRow:
    source_path: Path
    label: int
    video_id: str
    source_dataset: str


def _repo_rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest_rows(manifest_path: Path, max_items: int | None) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                label = int(row.get("label", "1"))
            except ValueError:
                continue
            if label != 0:
                continue
            raw = (row.get("file_path") or "").strip()
            if not raw:
                continue
            normalized = raw.replace("\\", "/")
            path = Path(normalized)
            full_path = path if path.is_absolute() else (REPO_ROOT / path)
            if not full_path.exists():
                continue
            rows.append(
                ManifestRow(
                    source_path=full_path.resolve(),
                    label=label,
                    video_id=(row.get("video_id") or ""),
                    source_dataset=(row.get("source_dataset") or ""),
                )
            )

    rows.sort(key=lambda r: (_repo_rel(r.source_path), r.video_id))
    if max_items is not None:
        return rows[:max_items]
    return rows


def _adv21_screen_recording_proxy(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    down = cv2.resize(img, (max(64, int(w * 0.85)), max(64, int(h * 0.85))), interpolation=cv2.INTER_AREA)
    up = cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)
    ok1, b1 = cv2.imencode(".jpg", up, [cv2.IMWRITE_JPEG_QUALITY, 84])
    if not ok1:
        raise RuntimeError("ADV-21 first JPEG encode failed")
    img1 = cv2.imdecode(b1, cv2.IMREAD_COLOR)
    ok2, b2 = cv2.imencode(".jpg", img1, [cv2.IMWRITE_JPEG_QUALITY, 76])
    if not ok2:
        raise RuntimeError("ADV-21 second JPEG encode failed")
    img2 = cv2.imdecode(b2, cv2.IMREAD_COLOR)
    if img2 is None:
        raise RuntimeError("ADV-21 decode failed")
    return img2


def _adv22_cartoon_non_ff_style(img: np.ndarray) -> np.ndarray:
    return cv2.stylization(img, sigma_s=62, sigma_r=0.45)


def _adv23_partial_face_occlusion(img: np.ndarray) -> np.ndarray:
    occ = img.copy()
    h, w = occ.shape[:2]
    cv2.rectangle(occ, (0, int(0.58 * h)), (w, h), (0, 0, 0), -1)
    cv2.rectangle(occ, (0, int(0.24 * h)), (int(0.24 * w), int(0.80 * h)), (18, 18, 18), -1)
    return occ


def _write_batch(
    batch_name: str,
    rows: list[ManifestRow],
    transform_name: str,
    transform,
    out_root: Path,
    transform_parameters: dict[str, object],
    seed: int,
) -> dict[str, object]:
    batch_root = out_root / batch_name
    image_dir = batch_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    for idx, row in enumerate(rows):
        img = cv2.imread(str(row.source_path))
        if img is None:
            continue
        out_img = transform(img)
        out_name = f"{batch_name.lower()}_{idx:05d}.jpg"
        out_path = image_dir / out_name
        ok = cv2.imwrite(str(out_path), out_img)
        if not ok:
            continue
        records.append(
            {
                "index": idx,
                "source_path": _repo_rel(row.source_path),
                "output_path": _repo_rel(out_path),
                "video_id": row.video_id,
                "label": row.label,
                "source_dataset": row.source_dataset,
                "sha256": _sha256_file(out_path),
            }
        )

    records.sort(key=lambda r: (str(r["source_path"]), str(r["output_path"])))

    file_list_path = batch_root / "file_list.txt"
    file_list_path.write_text(
        "\n".join(str(r["output_path"]) for r in records) + ("\n" if records else ""),
        encoding="utf-8",
    )

    metadata = {
        "batch": batch_name,
        "transform_name": transform_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "n_inputs": len(rows),
        "n_outputs": len(records),
        "manifest_source": _repo_rel(DEFAULT_MANIFEST),
        "transform_parameters": transform_parameters,
        "file_list_path": _repo_rel(file_list_path),
        "records": records,
    }
    metadata_path = batch_root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"batch": batch_name, "metadata_path": _repo_rel(metadata_path), "n_outputs": len(records)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic ADV-21/22/23 OOD/adversarial input batches."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--max-items",
        type=int,
        default=120,
        help="Maximum number of real samples from manifest to transform.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Recorded reproducibility seed.")
    args = parser.parse_args()

    np.random.seed(args.seed)

    manifest_path = args.manifest if args.manifest.is_absolute() else (REPO_ROOT / args.manifest)
    out_root = args.out_root if args.out_root.is_absolute() else (REPO_ROOT / args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    rows = _load_manifest_rows(manifest_path, max_items=args.max_items)
    if not rows:
        raise SystemExit("No readable real-image manifest rows found for ADV input generation.")

    results = [
        _write_batch(
            batch_name="ADV-21",
            rows=rows,
            transform_name="screen_recording_proxy",
            transform=_adv21_screen_recording_proxy,
            out_root=out_root,
            transform_parameters={"resize_ratio": 0.85, "jpeg_qualities": [84, 76]},
            seed=args.seed,
        ),
        _write_batch(
            batch_name="ADV-22",
            rows=rows,
            transform_name="cartoon_non_ff_style",
            transform=_adv22_cartoon_non_ff_style,
            out_root=out_root,
            transform_parameters={"stylization_sigma_s": 62, "stylization_sigma_r": 0.45},
            seed=args.seed,
        ),
        _write_batch(
            batch_name="ADV-23",
            rows=rows,
            transform_name="partial_face_mask_occlusion",
            transform=_adv23_partial_face_occlusion,
            out_root=out_root,
            transform_parameters={
                "lower_mask_y_ratio": 0.58,
                "side_mask_rect": [0.0, 0.24, 0.24, 0.80],
            },
            seed=args.seed,
        ),
    ]

    summary_path = out_root / "summary.json"
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_source": _repo_rel(manifest_path),
        "out_root": _repo_rel(out_root),
        "seed": args.seed,
        "max_items": args.max_items,
        "batches": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[ok] wrote {summary_path}")
    for item in results:
        print(f"[ok] {item['batch']} outputs={item['n_outputs']} metadata={item['metadata_path']}")


if __name__ == "__main__":
    main()
