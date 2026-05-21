"""MRL Eye training: MobileNetV2 fine-tuned for awake/sleepy classification.

Provides ``build_model()`` used by both training and inference pipelines.

The full training loop is maintained on Kaggle; this module exposes the model
architecture so ``inference.py`` can reconstruct the network and load weights.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 2, in_channels: int = 1) -> nn.Module:
    """MobileNetV2 adapted for single-channel grayscale eye crops.

    Modifications from the ImageNet default:
      - ``features[0][0]``: Conv2d input channels changed from 3 → *in_channels*
      - ``classifier[1]``: final Linear output changed to *num_classes*
    """
    model = models.mobilenet_v2(weights=None)

    first_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        in_channels,
        first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=False,
    )

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model
