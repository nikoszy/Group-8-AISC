# =============================================================================
# src/cnn_runner.py  —  EfficientNet-B0 inference wrapper
#
# Loads the checkpoint trained by cnn_detector.py and exposes a simple
# predict() function.  Soft-imports PyTorch so the rest of the pipeline
# degrades gracefully if torch is not installed.
#
# Architecture must EXACTLY match cnn_detector.build_model():
#   EfficientNet-B0 backbone (layers 0-4 frozen, 5-8 trainable)
#   Custom head: Dropout(0.4) → Linear(1280,256) → ReLU → Dropout(0.3) → Linear(256,1)
#   Output: raw logit → sigmoid → P(fake)
# =============================================================================

import os
import numpy as np

# ImageNet normalisation (must match cnn_detector.VAL_TF)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_IMG_SIZE = 224
_MODEL_PATH = os.path.join("data", "cnn_model.pth")

# Module-level cache so the model is loaded only once per process
_cached_model = None
_torch_available = None


def _check_torch():
    global _torch_available
    if _torch_available is None:
        try:
            import torch          # noqa: F401
            import torchvision    # noqa: F401
            _torch_available = True
        except ImportError:
            _torch_available = False
    return _torch_available


def _build_architecture():
    """Reconstruct the model architecture — must match cnn_detector.build_model()."""
    import torch.nn as nn
    from torchvision import models

    try:
        # torchvision >= 0.13
        model = models.efficientnet_b0(weights=None)
    except TypeError:
        model = models.efficientnet_b0(pretrained=False)

    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, 1),
    )
    return model


def load_cnn(model_path=_MODEL_PATH, verbose=True):
    """
    Load the EfficientNet-B0 checkpoint from disk.

    Returns:
        model (torch.nn.Module in eval mode) if torch + checkpoint found,
        None otherwise (caller should fall back to LR-only inference).
    """
    global _cached_model

    if _cached_model is not None:
        return _cached_model

    if not _check_torch():
        if verbose:
            print("[cnn_runner] PyTorch not installed — CNN disabled.")
        return None

    if not os.path.exists(model_path):
        if verbose:
            print(f"[cnn_runner] Checkpoint not found at {model_path} — CNN disabled.")
        return None

    import torch

    try:
        model = _build_architecture()
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _cached_model = model
        if verbose:
            print(f"[cnn_runner] EfficientNet-B0 loaded from {model_path}")
        return model
    except Exception as e:
        if verbose:
            print(f"[cnn_runner] Failed to load checkpoint: {e} — CNN disabled.")
        return None


def _preprocess(face_crop_bgr):
    """
    Apply the same transforms as cnn_detector.VAL_TF:
      BGR → RGB → resize 224×224 → float32 /255 → normalize ImageNet → NCHW tensor
    Returns: torch.Tensor of shape (1, 3, 224, 224)
    """
    import torch
    import cv2

    img = cv2.resize(face_crop_bgr, (_IMG_SIZE, _IMG_SIZE),
                     interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD                    # (H, W, C) normalised
    img = np.transpose(img, (2, 0, 1))            # (C, H, W)
    tensor = torch.from_numpy(img).unsqueeze(0)   # (1, C, H, W)
    return tensor


def cnn_predict(model, face_crop_bgr):
    """
    Run CNN inference on a single 224×224 BGR face crop.

    Args:
        model        : torch.nn.Module returned by load_cnn()
        face_crop_bgr: numpy array (H × W × 3, BGR)

    Returns:
        float in [0.0, 1.0] — P(fake), or None if inference fails.
    """
    if model is None:
        return None

    import torch

    try:
        tensor = _preprocess(face_crop_bgr)
        with torch.no_grad():
            logit = model(tensor).squeeze()
            prob  = float(torch.sigmoid(logit).item())
        return round(prob, 4)
    except Exception as e:
        return None


def cnn_available():
    """Return True if torch is installed and the checkpoint exists."""
    return _check_torch() and os.path.exists(_MODEL_PATH)
