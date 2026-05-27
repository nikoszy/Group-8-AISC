# =============================================================================
# cnn_detector.py  --  CNN deepfake detector (EfficientNet-B0, transfer learning)
#
# Replaces the handcrafted feature ensemble with a fine-tuned CNN.
# On FF++ C23 the CNN can learn subtle spatial patterns that JPEG/FFT
# features miss, typically reaching AUC 0.65-0.85 with 296 face crops.
#
# Architecture:
#   EfficientNet-B0 pretrained on ImageNet
#   Backbone layers 0-6  frozen      (generic low/mid-level features)
#   Backbone layers 7-8  trainable   (high-level features, fine-tuned)
#   Custom classifier head: 1280 -> 256 -> 1 (sigmoid)
#
# Usage:
#   py cnn_detector.py
#
# Outputs:
#   data/cnn_model.pth          best checkpoint (highest val AUC)
#   data/plots/cnn_roc.png      ROC curve
#   data/plots/cnn_training.png training loss/AUC curves
# =============================================================================

import os
import csv
import warnings

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, balanced_accuracy_score,
    confusion_matrix, roc_curve,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MANIFEST_PATH = os.path.join("data", "manifest.csv")
PLOTS_DIR     = os.path.join("data", "plots")
MODEL_PATH    = os.path.join("data", "cnn_model.pth")

IMG_SIZE            = 224
BATCH_SIZE          = 16
NUM_EPOCHS          = 50   # main training run
CV_EPOCHS           = 25   # epochs per fold in cross-validation
PATIENCE            = 12   # early stopping (val AUC does not improve)
LR_HEAD             = 1e-3
LR_BACKBONE_LATE    = 1e-4   # features[7-8] — last MBConv + head conv
LR_BACKBONE_EARLY   = 5e-5   # features[5-6] — wider unfreeze
WEIGHT_DECAY        = 1e-4
MIXUP_ALPHA         = 0.4    # Beta distribution param for MixUp
MIXUP_PROB          = 0.5    # probability of applying MixUp per batch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet normalisation
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
TRAIN_TF = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])

VAL_TF = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class FaceDataset(Dataset):
    def __init__(self, rows, transform=None):
        self.rows      = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = cv2.imread(row["file_path"])
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(row["label"], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model():
    """
    EfficientNet-B0 pretrained on ImageNet.
    Freeze layers 0-4, unfreeze layers 5-8 (4 blocks + head conv).
    Three LR groups: early backbone < late backbone < classifier head.
    Replace classifier with a 2-layer head that outputs a single logit.
    """
    try:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model   = models.efficientnet_b0(weights=weights)
    except AttributeError:
        model   = models.efficientnet_b0(pretrained=True)

    for param in model.features.parameters():
        param.requires_grad = False
    # Unfreeze last 4 feature blocks (5, 6, 7, 8)
    for idx in [5, 6, 7, 8]:
        for param in model.features[idx].parameters():
            param.requires_grad = True

    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, 1),
    )
    return model.to(DEVICE)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def make_loader(rows, transform, batch_size=BATCH_SIZE,
                shuffle=False, balanced=False):
    ds = FaceDataset(rows, transform)
    if balanced:
        labels       = [r["label"] for r in rows]
        class_counts = np.bincount(labels)
        weights      = [1.0 / class_counts[l] for l in labels]
        sampler      = WeightedRandomSampler(weights, len(weights),
                                             replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          num_workers=0)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0)


# ---------------------------------------------------------------------------
# MixUp augmentation
# ---------------------------------------------------------------------------
def mixup_data(x, y, alpha=MIXUP_ALPHA):
    """Mix two random samples within a batch."""
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx   = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1.0 - lam) * x[idx]
    return mixed, y, y[idx], lam


def mixup_loss(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        if np.random.random() < MIXUP_PROB:
            imgs, y_a, y_b, lam = mixup_data(imgs, labels)
            logits = model(imgs).squeeze(1)
            loss   = mixup_loss(criterion, logits, y_a, y_b, lam)
            # Accuracy against dominant label for display only
            preds        = (torch.sigmoid(logits) >= 0.5).long()
            mixed_labels = ((lam * y_a + (1.0 - lam) * y_b) >= 0.5).long()
            correct += (preds == mixed_labels).sum().item()
        else:
            logits = model(imgs).squeeze(1)
            loss   = criterion(logits, labels)
            preds   = (torch.sigmoid(logits) >= 0.5).long()
            correct += (preds == labels.long()).sum().item()

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        total      += len(labels)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    """Returns loss, accuracy, AUC, y_true (np), y_scores (np)."""
    model.eval()
    all_logits, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        all_logits.append(model(imgs).squeeze(1).cpu())
        all_labels.append(labels)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    loss   = float(nn.BCEWithLogitsLoss()(logits, labels))
    y_true = labels.numpy().astype(int)
    scores = torch.sigmoid(logits).numpy()
    preds  = (scores >= 0.5).astype(int)
    acc    = accuracy_score(y_true, preds)
    auc    = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.5
    return loss, acc, auc, y_true, scores


def train_model(train_rows, val_rows, num_epochs=NUM_EPOCHS, verbose=True):
    """
    Train EfficientNet-B0 on train_rows and validate on val_rows.
    Returns (model, history_dict, best_val_auc).
    """
    model = build_model()

    early_params      = (list(model.features[5].parameters()) +
                         list(model.features[6].parameters()))
    late_params       = (list(model.features[7].parameters()) +
                         list(model.features[8].parameters()))
    classifier_params = list(model.classifier.parameters())

    optimizer = torch.optim.AdamW([
        {"params": early_params,      "lr": LR_BACKBONE_EARLY, "weight_decay": WEIGHT_DECAY},
        {"params": late_params,       "lr": LR_BACKBONE_LATE,  "weight_decay": WEIGHT_DECAY},
        {"params": classifier_params, "lr": LR_HEAD,           "weight_decay": WEIGHT_DECAY},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )

    pos  = sum(r["label"] for r in train_rows)
    neg  = len(train_rows) - pos
    pw   = torch.tensor([neg / max(pos, 1)], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    train_loader = make_loader(train_rows, TRAIN_TF, balanced=True)
    val_loader   = make_loader(val_rows,   VAL_TF,   shuffle=False)

    best_auc, best_state, patience_count = 0.0, None, 0
    history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_acc": []}

    for epoch in range(1, num_epochs + 1):
        tr_loss, _            = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_auc, _, _ = evaluate(model, val_loader)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["val_acc"].append(val_acc)

        improved = val_auc > best_auc
        if improved:
            best_auc   = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if verbose:
            print(f"  Epoch {epoch:3d}/{num_epochs}  "
                  f"tr_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"val_acc={val_acc:.4f}  val_auc={val_auc:.4f}"
                  + ("  *best*" if improved else ""))

        if patience_count >= PATIENCE:
            if verbose:
                print(f"  Early stop at epoch {epoch}  "
                      f"(best AUC={best_auc:.4f})")
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, history, best_auc


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_manifest(path=MANIFEST_PATH):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append({
                "file_path": row["file_path"],
                "label"    : int(row["label"]),
                "video_id" : row.get("video_id", "unknown"),
            })
    return rows


def plot_training(history, save_dir=PLOTS_DIR):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train loss")
    ax1.plot(epochs, history["val_loss"],   label="Val loss")
    ax1.set_xlabel("Epoch"); ax1.set_title("Loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["val_auc"],
             color="steelblue", label="Val AUC")
    ax2.plot(epochs, history["val_acc"],
             color="tomato", label="Val Accuracy")
    ax2.set_xlabel("Epoch"); ax2.set_title("Val AUC / Accuracy")
    ax2.legend(); ax2.grid(alpha=0.3)

    path = os.path.join(save_dir, "cnn_training.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training curves: {path}")


def plot_roc(y_true, y_scores, auc_val, save_dir=PLOTS_DIR):
    os.makedirs(save_dir, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"EfficientNet-B0  AUC={auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "grey", linestyle="--", lw=1,
            label="Random (0.50)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("CNN ROC Curve -- Deepfake Detection")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    path = os.path.join(save_dir, "cnn_roc.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ROC curve      : {path}")


def print_confusion(y_true, y_scores, threshold=0.5):
    y_pred        = (y_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print()
    print("  CONFUSION MATRIX  (threshold=0.50)")
    print("                  Predicted")
    print("                  REAL   FAKE")
    print(f"  Actual  REAL  [ {tn:4d}   {fp:4d} ]  <- {fp} real wrongly flagged")
    print(f"          FAKE  [ {fn:4d}   {tp:4d} ]  <- {fn} fakes missed")


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
def cross_validate_cnn(all_rows, n_splits=5):
    gkf    = GroupKFold(n_splits=n_splits)
    groups = np.array([r["video_id"] for r in all_rows])
    labels = np.array([r["label"]    for r in all_rows])
    rows   = np.array(all_rows, dtype=object)

    fold_aucs, fold_accs, fold_bals = [], [], []

    for fold_idx, (train_idx, val_idx) in enumerate(
            gkf.split(rows, labels, groups=groups)):
        train_rows = rows[train_idx].tolist()
        val_rows   = rows[val_idx].tolist()
        n_r  = sum(1 for r in val_rows if r["label"] == 0)
        n_f  = sum(1 for r in val_rows if r["label"] == 1)
        n_v  = len(set(r["video_id"] for r in val_rows))
        print(f"\n  --- Fold {fold_idx+1}/{n_splits}  "
              f"val={len(val_rows)} ({n_r}r/{n_f}f, {n_v} vids) ---")

        if len(np.unique([r["label"] for r in val_rows])) < 2:
            print("  Skipped: single class in val set")
            continue

        model, _, _ = train_model(
            train_rows, val_rows,
            num_epochs=CV_EPOCHS, verbose=False
        )
        _, val_acc, val_auc, y_true, y_scores = evaluate(
            model, make_loader(val_rows, VAL_TF, shuffle=False)
        )
        bal = balanced_accuracy_score(y_true, (y_scores >= 0.5).astype(int))

        fold_aucs.append(val_auc)
        fold_accs.append(val_acc)
        fold_bals.append(bal)
        print(f"  AUC={val_auc:.4f}  acc={val_acc:.4f}  "
              f"balanced_acc={bal:.4f}")

    return fold_aucs, fold_accs, fold_bals


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print()
    print("=" * 65)
    print("CNN DEEPFAKE DETECTOR  (EfficientNet-B0 transfer learning)")
    print("=" * 65)
    print(f"  Device        : {DEVICE}")
    print(f"  Model         : EfficientNet-B0 (ImageNet pretrained)")
    print(f"  Frozen layers : features[0..4]  (generic texture/edge features)")
    print(f"  Trained layers: features[5..8] + classifier head  (3 LR groups)")
    print(f"  Epochs        : up to {NUM_EPOCHS}  (early stop patience={PATIENCE})")
    print(f"  Batch size    : {BATCH_SIZE}")
    print()

    if not os.path.exists(MANIFEST_PATH):
        print(f"[ERROR] {MANIFEST_PATH} not found.")
        print("        Run  py inspect_dataset.py  first.")
        raise SystemExit(1)

    # ---------------------------------------------------------------
    # 1. Load manifest
    # ---------------------------------------------------------------
    print("STEP 1 -- Load manifest")
    all_rows = load_manifest()
    n_real   = sum(1 for r in all_rows if r["label"] == 0)
    n_fake   = sum(1 for r in all_rows if r["label"] == 1)
    n_vids   = len(set(r["video_id"] for r in all_rows))
    print(f"  Total : {len(all_rows)} images  "
          f"({n_real} real, {n_fake} fake, {n_vids} unique videos)")
    print()

    # ---------------------------------------------------------------
    # 2. Video-level train / val split  (80 / 20)
    # ---------------------------------------------------------------
    print("STEP 2 -- Video-level 80/20 split  (GroupShuffleSplit)")
    gss        = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    groups     = np.array([r["video_id"] for r in all_rows])
    labels_all = np.array([r["label"]    for r in all_rows])
    rows_arr   = np.array(all_rows, dtype=object)
    train_idx, val_idx = next(gss.split(rows_arr, labels_all, groups=groups))
    train_rows = rows_arr[train_idx].tolist()
    val_rows   = rows_arr[val_idx].tolist()

    n_tr_r = sum(1 for r in train_rows if r["label"] == 0)
    n_tr_f = sum(1 for r in train_rows if r["label"] == 1)
    n_vl_r = sum(1 for r in val_rows   if r["label"] == 0)
    n_vl_f = sum(1 for r in val_rows   if r["label"] == 1)
    print(f"  Train : {len(train_rows)} frames  ({n_tr_r} real, {n_tr_f} fake)")
    print(f"  Val   : {len(val_rows)}  frames  ({n_vl_r} real, {n_vl_f} fake)")
    print()

    # ---------------------------------------------------------------
    # 3. Train
    # ---------------------------------------------------------------
    print("STEP 3 -- Train EfficientNet-B0")
    print("  * marks epochs where val AUC improved (saved as best checkpoint)")
    print()
    model, history, best_train_auc = train_model(
        train_rows, val_rows, num_epochs=NUM_EPOCHS, verbose=True
    )
    print()

    # ---------------------------------------------------------------
    # 4. Final evaluation on val set (best checkpoint)
    # ---------------------------------------------------------------
    print("STEP 4 -- Evaluate best checkpoint on val set")
    _, val_acc, val_auc, y_true, y_scores = evaluate(
        model, make_loader(val_rows, VAL_TF, shuffle=False)
    )
    y_pred = (y_scores >= 0.5).astype(int)
    bal    = balanced_accuracy_score(y_true, y_pred)

    print()
    print("=" * 55)
    print("SINGLE-SPLIT RESULTS")
    print("=" * 55)
    print(f"  AUC              = {val_auc:.4f}")
    print(f"  Accuracy         = {val_acc:.4f}")
    print(f"  Balanced Acc     = {bal:.4f}")
    print("=" * 55)

    print_confusion(y_true, y_scores)
    print()

    # ---------------------------------------------------------------
    # 5. Save plots + model
    # ---------------------------------------------------------------
    print("STEP 5 -- Save plots and checkpoint")
    plot_training(history)
    plot_roc(y_true, y_scores, val_auc)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"  Model checkpoint: {MODEL_PATH}")
    print()

    # ---------------------------------------------------------------
    # 6. 5-fold GroupKFold cross-validation
    # ---------------------------------------------------------------
    print("STEP 6 -- 5-fold GroupKFold cross-validation")
    print(f"  Each fold trains from scratch for up to {CV_EPOCHS} epochs.")
    print(f"  This gives a reliable mean +/- std estimate.")
    fold_aucs, fold_accs, fold_bals = cross_validate_cnn(all_rows, n_splits=5)

    print()
    print("=" * 55)
    print("CROSS-VALIDATION SUMMARY  (mean +/- std)")
    print("=" * 55)
    if fold_aucs:
        print(f"  AUC          = {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}")
        print(f"  Accuracy     = {np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f}")
        print(f"  Balanced Acc = {np.mean(fold_bals):.4f} +/- {np.std(fold_bals):.4f}")
    else:
        print("  No folds completed.")
    print("=" * 55)

    # ---------------------------------------------------------------
    # 7. Final summary
    # ---------------------------------------------------------------
    print()
    print("=" * 65)
    print("CNN DETECTOR COMPLETE")
    print("=" * 65)
    print(f"  Single-split  AUC      = {val_auc:.4f}")
    print(f"  Single-split  Accuracy = {val_acc:.4f}")
    if fold_aucs:
        print(f"  5-fold CV     AUC      = "
              f"{np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}")
        print(f"  5-fold CV     Accuracy = "
              f"{np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f}")
    print()
    print(f"  Model   : {MODEL_PATH}")
    print(f"  ROC     : {os.path.join(PLOTS_DIR, 'cnn_roc.png')}")
    print(f"  Curves  : {os.path.join(PLOTS_DIR, 'cnn_training.png')}")
    print()
    print("  Comparison vs handcrafted features (ensemble.py):")
    print("    Handcrafted  CV AUC = 0.383  (FFT + artifact + Laplacian)")
    print(f"    CNN          CV AUC = "
          + (f"{np.mean(fold_aucs):.3f}" if fold_aucs else "see above")
          + "  (EfficientNet-B0 fine-tuned)")
    print("=" * 65)
