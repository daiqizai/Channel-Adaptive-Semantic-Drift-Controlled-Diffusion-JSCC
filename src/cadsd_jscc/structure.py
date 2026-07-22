from __future__ import annotations

import torch
import torch.nn.functional as F


def structural_feature_maps(images: torch.Tensor) -> torch.Tensor:
    """Return Sobel magnitude and absolute Laplacian in the refiner's [0, 1] scale."""

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected BCHW RGB images, got shape {tuple(images.shape)}")
    if not images.is_floating_point():
        raise TypeError(f"Expected floating-point images, got {images.dtype}")
    weights = images.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    luma = (images * weights).sum(dim=1, keepdim=True)
    sobel_x = images.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3)
    sobel_y = images.new_tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
    ).view(1, 1, 3, 3)
    laplacian = images.new_tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
    ).view(1, 1, 3, 3)
    gx = F.conv2d(luma, sobel_x, padding=1)
    gy = F.conv2d(luma, sobel_y, padding=1)
    sobel = torch.sqrt(gx.square() + gy.square() + 1e-6).div(4.0).clamp(0.0, 1.0)
    lap = F.conv2d(luma, laplacian, padding=1).abs().div(4.0).clamp(0.0, 1.0)
    return torch.cat([sobel, lap], dim=1)


def structure_rgb(images: torch.Tensor, third_channel: str = "maximum") -> torch.Tensor:
    """Pack two structural maps into an RGB tensor for the unmodified DeepJSCC backbone."""

    features = structural_feature_maps(images)
    if third_channel == "maximum":
        third = features.max(dim=1, keepdim=True).values
    elif third_channel == "mean":
        third = features.mean(dim=1, keepdim=True)
    elif third_channel == "zeros":
        third = torch.zeros_like(features[:, :1])
    else:
        raise ValueError(f"Unsupported structure RGB third channel: {third_channel!r}")
    return torch.cat([features, third], dim=1)
