"""MRL Eye training: MobileNetV2 fine-tuned for awake/sleepy classification.

Provides ``build_model()`` used by both training and inference pipelines.

The full training loop is maintained on Kaggle; this module exposes the
**exact** model architecture so ``inference.py`` can reconstruct the network
and load the checkpoint weights.

Checkpoint format (saved by the Kaggle training notebook)::

    {
        "epoch":            int,
        "model_state_dict": OrderedDict,
        "val_acc":          float,
        "img_size":         84,
        "class_to_label":   {"awake": 1, "sleepy": 0},   # name → index
    }

Label convention
    ImageFolder discovers  awake=0, sleepy=1  (alphabetical).
    The training loop remaps via ``1 - labels`` so the model learns
    **awake=1, sleepy=0**.  ``class_to_label`` records this remapping.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def build_model() -> nn.Module:
    """MobileNetV2 pretrained on ImageNet, adapted for 1-ch grayscale input.

    Modifications from the ImageNet default:
      - ``features[0][0]``: 3→1 input channel; weights initialised by
        averaging the original RGB filters (``sum(dim=1) / 3``).
      - ``classifier[1]``: output changed to 2 classes.
    """
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

    old_conv = model.features[0][0]
    new_conv = nn.Conv2d(
        1,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    with torch.no_grad():
        new_conv.weight.copy_(old_conv.weight.sum(dim=1, keepdim=True) / 3.0)
    model.features[0][0] = new_conv

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    return model
