"""Exploratory Data Analysis for the MRL Eye Dataset.

Expects the dataset layout produced by `download_mrl.py`:

    data/mrl/
        train/
            awake/   *.png|*.jpg|...
            sleepy/  *.png|*.jpg|...
        val/         (also accepts "valid" or "validation")
            awake/
            sleepy/
        test/
            awake/
            sleepy/

For each split/class the script:
  1. Counts and prints the number of images.
  2. Plots a grid of sample images (awake vs sleepy) and saves it to
     `eda_outputs/sample_grid.png`.
  3. Checks that all image dimensions are consistent and flags any that aren't.
  4. Confirms images are grayscale rather than RGB.
  5. Scans for corrupted image files using PIL.

Run:
    python eda_mrl.py [--data-dir data/mrl] [--out-dir eda_outputs]
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
CLASSES = ("awake", "sleepy")
SPLIT_ALIASES = {
    "train": ["train", "training"],
    "val": ["val", "valid", "validation"],
    "test": ["test", "testing"],
}


def _resolve_split_dirs(data_dir: Path) -> Dict[str, Path]:
    """Map canonical split name -> actual directory path on disk."""
    resolved: Dict[str, Path] = {}
    for canonical, aliases in SPLIT_ALIASES.items():
        for alias in aliases:
            candidate = data_dir / alias
            if candidate.is_dir():
                resolved[canonical] = candidate
                break
    return resolved


def _iter_images(directory: Path) -> Iterable[Path]:
    if not directory.is_dir():
        return []
    return (p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def count_images(split_dirs: Dict[str, Path]) -> Dict[str, Dict[str, int]]:
    """Return counts[split][class] = number of images."""
    counts: Dict[str, Dict[str, int]] = {}
    for split, split_path in split_dirs.items():
        counts[split] = {}
        for cls in CLASSES:
            cls_path = split_path / cls
            counts[split][cls] = sum(1 for _ in _iter_images(cls_path))
    return counts


def print_counts(counts: Dict[str, Dict[str, int]]) -> None:
    print("\n=== (1) Image counts per split / class ===")
    header = f"{'split':<8} {'awake':>10} {'sleepy':>10} {'total':>10}"
    print(header)
    print("-" * len(header))
    grand_total = 0
    for split in ("train", "val", "test"):
        if split not in counts:
            continue
        row = counts[split]
        a = row.get("awake", 0)
        s = row.get("sleepy", 0)
        total = a + s
        grand_total += total
        print(f"{split:<8} {a:>10,} {s:>10,} {total:>10,}")
    print("-" * len(header))
    print(f"{'ALL':<8} {'':>10} {'':>10} {grand_total:>10,}")


def plot_sample_grid(
    split_dirs: Dict[str, Path],
    out_path: Path,
    samples_per_class: int = 8,
    preferred_split: str = "train",
    seed: int = 0,
) -> None:
    """Plot a grid of sample images, one row per class."""
    print("\n=== (2) Plotting sample image grid ===")
    rng = random.Random(seed)

    split_order = [preferred_split] + [s for s in ("train", "val", "test")
                                       if s != preferred_split]
    chosen: Dict[str, List[Path]] = {}
    for cls in CLASSES:
        picks: List[Path] = []
        for split in split_order:
            if split not in split_dirs:
                continue
            files = list(_iter_images(split_dirs[split] / cls))
            if files:
                rng.shuffle(files)
                picks = files[:samples_per_class]
                break
        chosen[cls] = picks

    if not any(chosen.values()):
        print("  No images found to plot. Skipping grid.")
        return

    cols = max(samples_per_class, 1)
    rows = len(CLASSES)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.8))
    if rows == 1:
        axes = np.array([axes])
    if cols == 1:
        axes = axes.reshape(rows, 1)

    for r, cls in enumerate(CLASSES):
        files = chosen[cls]
        for c in range(cols):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            if c < len(files):
                try:
                    with Image.open(files[c]) as img:
                        ax.imshow(np.asarray(img), cmap="gray")
                except (UnidentifiedImageError, OSError) as e:
                    ax.text(0.5, 0.5, f"err\n{type(e).__name__}",
                            ha="center", va="center", fontsize=7)
            else:
                ax.axis("off")
            if c == 0:
                ax.set_ylabel(cls, fontsize=11)

    fig.suptitle("MRL Eye Dataset – sample images (awake vs sleepy)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _safe_open_size_mode(path: Path) -> Tuple[Tuple[int, int] | None,
                                              str | None, str | None]:
    """Return (size, mode, error) for an image file."""
    try:
        with Image.open(path) as img:
            img.load()
            return img.size, img.mode, None
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as e:
        return None, None, f"{type(e).__name__}: {e}"


def analyze_images(
    split_dirs: Dict[str, Path],
    out_dir: Path,
    max_dim_mismatches_to_show: int = 20,
    max_color_mismatches_to_show: int = 20,
    max_corrupt_to_show: int = 50,
) -> None:
    """One pass to compute dimension/mode/corruption stats."""
    size_counter: Counter = Counter()
    size_examples: Dict[Tuple[int, int], Path] = {}
    mode_counter: Counter = Counter()
    per_split_modes: Dict[str, Counter] = defaultdict(Counter)
    per_split_sizes: Dict[str, Counter] = defaultdict(Counter)
    rgb_examples: List[Tuple[Path, str]] = []
    corrupt: List[Tuple[Path, str]] = []
    total = 0

    for split, split_path in split_dirs.items():
        for cls in CLASSES:
            for p in _iter_images(split_path / cls):
                total += 1
                size, mode, err = _safe_open_size_mode(p)
                if err is not None:
                    corrupt.append((p, err))
                    continue
                size_counter[size] += 1
                per_split_sizes[split][size] += 1
                size_examples.setdefault(size, p)
                mode_counter[mode] += 1
                per_split_modes[split][mode] += 1
                if mode not in ("L", "1", "I", "F", "LA", "I;16"):
                    if len(rgb_examples) < max_color_mismatches_to_show:
                        rgb_examples.append((p, mode))

    print(f"\nScanned {total:,} image files in total.")

    # (3) dimension consistency
    print("\n=== (3) Image dimension consistency ===")
    if not size_counter:
        print("  No readable images found.")
    else:
        print(f"  Unique image sizes: {len(size_counter)}")
        for size, count in size_counter.most_common(10):
            pct = 100.0 * count / max(total - len(corrupt), 1)
            example = size_examples[size]
            print(f"    {size[0]}x{size[1]}: {count:,} files "
                  f"({pct:5.2f}%)  e.g. {example.name}")
        if len(size_counter) == 1:
            print("  All readable images share the same dimensions. OK.")
        else:
            dominant_size, dominant_count = size_counter.most_common(1)[0]
            mismatches = [
                (size, count) for size, count in size_counter.items()
                if size != dominant_size
            ]
            print(f"  WARNING: {sum(c for _, c in mismatches):,} files have "
                  f"dimensions different from the dominant "
                  f"{dominant_size[0]}x{dominant_size[1]}.")
            print("  Per-split size breakdown:")
            for split, sizes in per_split_sizes.items():
                summary = ", ".join(
                    f"{w}x{h}: {c}" for (w, h), c in sizes.most_common(5)
                )
                print(f"    {split}: {summary}")
            print(f"  Showing up to {max_dim_mismatches_to_show} mismatched "
                  "example files:")
            shown = 0
            for split, split_path in split_dirs.items():
                for cls in CLASSES:
                    for p in _iter_images(split_path / cls):
                        if shown >= max_dim_mismatches_to_show:
                            break
                        size, _, err = _safe_open_size_mode(p)
                        if err is None and size != dominant_size:
                            print(f"    {p}  -> {size[0]}x{size[1]}")
                            shown += 1
                    if shown >= max_dim_mismatches_to_show:
                        break
                if shown >= max_dim_mismatches_to_show:
                    break

    # (4) grayscale check
    print("\n=== (4) Color mode (grayscale check) ===")
    if not mode_counter:
        print("  No readable images found.")
    else:
        print("  Pillow mode distribution:")
        for mode, count in mode_counter.most_common():
            print(f"    {mode}: {count:,}")
        non_gray = sum(c for m, c in mode_counter.items()
                       if m not in ("L", "1", "I", "F", "LA", "I;16"))
        if non_gray == 0:
            print("  All readable images are grayscale (mode L or similar). "
                  "OK.")
        else:
            print(f"  WARNING: {non_gray:,} image(s) appear to be NOT "
                  "grayscale (RGB/RGBA/P/etc.).")
            for path, mode in rgb_examples:
                print(f"    {path}  -> mode={mode}")
        print("  Per-split mode breakdown:")
        for split, modes in per_split_modes.items():
            summary = ", ".join(f"{m}: {c}" for m, c in modes.most_common())
            print(f"    {split}: {summary}")

    # (5) corrupted files
    print("\n=== (5) Corrupted file scan (PIL) ===")
    if not corrupt:
        print("  No corrupted files detected. OK.")
    else:
        print(f"  WARNING: {len(corrupt):,} corrupted/unreadable file(s) "
              "detected.")
        for path, err in corrupt[:max_corrupt_to_show]:
            print(f"    {path}  -> {err}")
        report_path = out_dir / "corrupted_files.txt"
        out_dir.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            for path, err in corrupt:
                f.write(f"{path}\t{err}\n")
        print(f"  Full list written to: {report_path}")

    # Bonus plot: per-split class counts.
    counts_plot_path = out_dir / "class_counts.png"
    _plot_class_counts(split_dirs, counts_plot_path)


def _plot_class_counts(split_dirs: Dict[str, Path], out_path: Path) -> None:
    counts = count_images(split_dirs)
    splits = [s for s in ("train", "val", "test") if s in counts]
    if not splits:
        return
    awake = [counts[s].get("awake", 0) for s in splits]
    sleepy = [counts[s].get("sleepy", 0) for s in splits]
    x = np.arange(len(splits))
    width = 0.38

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, awake, width, label="awake", color="#4C9AFF")
    ax.bar(x + width / 2, sleepy, width, label="sleepy", color="#FF8B4C")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("Number of images")
    ax.set_title("MRL Eye Dataset – images per split/class")
    ax.legend()
    for i, (a, s) in enumerate(zip(awake, sleepy)):
        ax.text(i - width / 2, a, f"{a:,}", ha="center", va="bottom",
                fontsize=8)
        ax.text(i + width / 2, s, f"{s:,}", ha="center", va="bottom",
                fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved class-counts plot: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EDA for the MRL Eye Dataset.")
    parser.add_argument("--data-dir",
                        default=str(PROJECT_ROOT / "data" / "mrl"),
                        help="Root of the MRL dataset (default: data/mrl).")
    parser.add_argument("--out-dir",
                        default=str(PROJECT_ROOT / "eda_outputs"),
                        help="Directory to save plots and reports.")
    parser.add_argument("--samples", type=int, default=8,
                        help="Samples per class for the grid plot.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data dir: {data_dir}")
    print(f"Output dir: {out_dir}")

    if not data_dir.is_dir():
        # Fallback: maybe the archive nested everything under another folder.
        candidates = [p for p in data_dir.parent.glob("**/awake") if p.is_dir()]
        hint = ""
        if candidates:
            hint = "\nFound 'awake' folders elsewhere:\n  " + "\n  ".join(
                str(c.parent) for c in candidates[:5])
        raise SystemExit(
            f"ERROR: data directory not found: {data_dir}{hint}")

    split_dirs = _resolve_split_dirs(data_dir)
    if not split_dirs:
        # Try one level deeper (some Kaggle archives nest content).
        for sub in data_dir.iterdir():
            if sub.is_dir():
                resolved = _resolve_split_dirs(sub)
                if resolved:
                    print(f"NOTE: using nested data dir: {sub}")
                    split_dirs = resolved
                    data_dir = sub
                    break
    if not split_dirs:
        raise SystemExit(
            f"ERROR: no train/val/test folders found under {data_dir}. "
            f"Expected subfolders: {sorted(SPLIT_ALIASES['train'] + SPLIT_ALIASES['val'] + SPLIT_ALIASES['test'])}"
        )

    print("Resolved split directories:")
    for canonical, path in split_dirs.items():
        print(f"  {canonical:<5} -> {path}")

    counts = count_images(split_dirs)
    print_counts(counts)

    plot_sample_grid(
        split_dirs,
        out_dir / "sample_grid.png",
        samples_per_class=args.samples,
        seed=args.seed,
    )

    analyze_images(split_dirs, out_dir)

    print("\nEDA complete.")


if __name__ == "__main__":
    main()
