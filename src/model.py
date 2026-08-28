"""Model definitions for the CIFAR-10 image classifier."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18


def get_model(architecture: str = "resnet18", num_classes: int = 10) -> nn.Module:
    """
    Create and return the requested image classification model.

    Args:
        architecture: Model architecture name.
        num_classes: Number of output classes.

    Returns:
        A PyTorch classification model.

    Raises:
        ValueError: If the requested architecture is unsupported.
    """
    architecture = architecture.lower().strip()

    if architecture != "resnet18":
        raise ValueError(
            f"Unsupported architecture: {architecture}. "
            "Supported architecture: resnet18."
        )

    model = resnet18(weights=None)

    # CIFAR-10 images are 32x32. The standard ImageNet ResNet-18
    # uses a larger initial kernel and max-pooling layer, so we adapt
    # the stem for small images.
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()

    # Replace ImageNet's 1000-class classifier with CIFAR-10 output.
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model