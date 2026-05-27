"""MRL Eye preprocessing: load, resize, normalize [0, 1], PyTorch DataLoaders.

Dataset layout (after `download_mrl.py`):

    data/mrl/data/train/{awake,sleepy}/...
    data/mrl/data/val/{awake,sleepy}/...
    data/mrl/data/test/{awake,sleepy}/...

If `train/` is not directly under `--data-dir`, the parent folder is resolved
the same way as `eda_mrl.py` (one nested level such as `data/mrl/data`).

Typical MRL crops vary in size; this pipeline resizes to a fixed square.
Default image size is 84 (common for this dataset); pass `--img-size 32` for
32×32 inputs.

Usage:
    python preprocess.py                          # verify loader + 5-epoch dummy train
    python preprocess.py --img-size 32 --limit 1000
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Match eda_mrl layout discovery
SPLIT_ALIASES = {
    "train": ["train", "training"],
    "val": ["val", "valid", "validation"],
    "test": ["test", "testing"],
}


def _resolve_split_dirs(data_dir: Path) -> Dict[str, Path]:
    resolved: Dict[str, Path] = {}
    for canonical, aliases in SPLIT_ALIASES.items():
        for alias in aliases:
            candidate = data_dir / alias
            if candidate.is_dir():
                resolved[canonical] = candidate
                break
    return resolved


def resolve_mrl_root(data_dir: Path) -> Tuple[Path, Dict[str, Path]]:
    """Return (root_folder_containing_splits, split_name -> path)."""
    data_dir = Path(data_dir).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    split_dirs = _resolve_split_dirs(data_dir)
    if split_dirs:
        return data_dir, split_dirs

    for sub in sorted(data_dir.iterdir()):
        if sub.is_dir():
            inner = _resolve_split_dirs(sub)
            if inner:
                return sub, inner

    raise FileNotFoundError(
        f"No train/val/test splits under {data_dir}. "
        "Expected e.g. data/mrl/data with train/, val/, test/."
    )


def default_img_size() -> int:
    """Default resize edge length (MRL crops are often ~84×84; sizes vary)."""
    return 84


def build_transform(img_size: int) -> transforms.Compose:
    """Resize to square, grayscale single channel, float tensor in [0, 1]."""
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # scales to [0, 1]
        ]
    )


def make_image_folder(split_path: Path, img_size: int) -> datasets.ImageFolder:
    if not split_path.is_dir():
        raise FileNotFoundError(f"Split folder not found: {split_path}")
    tfm = build_transform(img_size)
    return datasets.ImageFolder(str(split_path), transform=tfm)


def make_dataloader(
    split_path: Path,
    *,
    batch_size: int = 32,
    shuffle: bool = True,
    img_size: int | None = None,
    num_workers: int = 0,
    subset_indices: Optional[list[int]] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[DataLoader, datasets.ImageFolder]:
    """Build a DataLoader for one split (awake/sleepy subfolders)."""
    size = img_size if img_size is not None else default_img_size()
    ds = make_image_folder(split_path, size)
    target = ds
    if subset_indices is not None:
        target = Subset(ds, subset_indices)

    loader = DataLoader(
        target,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    return loader, ds


def verify_dataloader(
    train_split_path: Path,
    *,
    batch_size: int = 32,
    img_size: int,
    out_path: Path,
) -> None:
    """Load one batch, print shapes, save a visualization grid."""
    loader, ds = make_dataloader(
        train_split_path,
        batch_size=batch_size,
        shuffle=True,
        img_size=img_size,
    )
    images, labels = next(iter(loader))
    print("\n=== DataLoader verification ===")
    print(f"Dataset length: {len(ds)}")
    print(f"Classes ({ds.classes}): {ds.class_to_idx}")
    print(f"Batch images shape: {tuple(images.shape)}  (N, C, H, W)")
    print(f"Batch labels shape: {tuple(labels.shape)}")
    print(
        f"Pixel range: min={images.min().item():.4f}, "
        f"max={images.max().item():.4f} (expect [0, 1])"
    )

    n_show = min(batch_size, 16)
    cols = min(8, n_show)
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.5))
    axes = np.atleast_2d(np.array(axes))
    for i in range(n_show):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        img = images[i, 0].numpy()
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(ds.classes[labels[i]], fontsize=8)
        ax.axis("off")
    for j in range(n_show, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")
    fig.suptitle("One batch (normalized grayscale)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved batch preview: {out_path}")


class TinyCNN(nn.Module):
    """Small CNN; works for any img_size >= 8 via adaptive pooling."""

    def __init__(self, num_classes: int = 2, in_ch: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x)


def train_dummy_run(
    train_split_path: Path,
    *,
    img_size: int,
    limit: int,
    epochs: int,
    batch_size: int,
    seed: int,
    lr: float,
) -> list[float]:
    """Train on `limit` random training images for `epochs`; return epoch losses."""
    full_ds = make_image_folder(train_split_path, img_size)
    n = len(full_ds)
    if n == 0:
        raise RuntimeError("Training set is empty.")
    take = min(limit, n)

    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    indices = perm[:take]

    loader, _ = make_dataloader(
        train_split_path,
        batch_size=batch_size,
        shuffle=True,
        img_size=img_size,
        subset_indices=indices,
        generator=g,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyCNN(num_classes=2, in_ch=1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch_losses: list[float] = []

    print("\n=== Dummy training run ===")
    print(f"Device: {device}")
    print(f"Training on {take} images, batch_size={batch_size}, epochs={epochs}")

    model.train()
    for epoch in range(epochs):
        running = 0.0
        seen = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)
            seen += images.size(0)
        avg = running / max(seen, 1)
        epoch_losses.append(avg)
        print(f"  Epoch {epoch + 1}/{epochs}  train_loss={avg:.6f}")

    return epoch_losses


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MRL preprocessing + loader check")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "mrl",
        help="Dataset root (default: <project>/data/mrl)",
    )
    p.add_argument(
        "--img-size",
        type=int,
        default=None,
        help=f"Square resize (default: {default_img_size()})",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int, default=1000, help="Dummy train images")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "eda_outputs",
        help="Where to save batch preview image",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    img_size = args.img_size if args.img_size is not None else default_img_size()

    root, splits = resolve_mrl_root(args.data_dir)
    train_path = splits["train"]
    print(f"Resolved data root: {root}")
    print(f"Train split path: {train_path}")

    preview_path = args.out_dir / "preprocess_batch_preview.png"
    verify_dataloader(
        train_path,
        batch_size=args.batch_size,
        img_size=img_size,
        out_path=preview_path,
    )

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    losses = train_dummy_run(
        train_path,
        img_size=img_size,
        limit=args.limit,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        lr=args.lr,
    )

    print("\n=== Loss trend ===")
    if losses[0] > losses[-1]:
        print(
            f"Train loss decreased: {losses[0]:.6f} -> {losses[-1]:.6f} (OK)"
        )
    else:
        print(
            f"WARNING: train loss did not decrease "
            f"({losses[0]:.6f} -> {losses[-1]:.6f}). "
            "Try lower lr, more epochs, or different seed."
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
