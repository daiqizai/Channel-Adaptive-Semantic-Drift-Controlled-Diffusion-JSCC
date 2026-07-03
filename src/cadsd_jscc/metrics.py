from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim


def mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x, y, reduction="mean")


def psnr(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0) -> float:
    value = mse(x, y).detach().cpu().item()
    if value <= 0:
        return float("inf")
    return 10.0 * math.log10((max_val * max_val) / value)


def psnr_per_sample(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    values = F.mse_loss(x, y, reduction="none").flatten(start_dim=1).mean(dim=1)
    return 10.0 * torch.log10((max_val * max_val) / values.clamp_min(1e-12))


def ssim_per_sample(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    return ssim(x, y, data_range=max_val, size_average=False)


def ms_ssim_per_sample(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    return ms_ssim(x, y, data_range=max_val, size_average=False)
