"""Capacity-matched pixel-domain fusion of B0 and a controlled diffusion observation."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.net(inputs)


class DualInputResidualRefiner(nn.Module):
    """B1-shaped refiner with one extra RGB observation and exactly nine inputs."""

    def __init__(self, base_channels: int = 64, num_blocks: int = 6) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(9, base_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)
        self.register_buffer(
            "luma_weights",
            torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "sobel_x",
            torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(
                1, 1, 3, 3
            ),
            persistent=False,
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(
                1, 1, 3, 3
            ),
            persistent=False,
        )
        self.register_buffer(
            "laplacian",
            torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(
                1, 1, 3, 3
            ),
            persistent=False,
        )
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def structural_conditions(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        luma = (
            image * self.luma_weights.to(dtype=image.dtype, device=image.device)
        ).sum(dim=1, keepdim=True)
        gx = F.conv2d(
            luma, self.sobel_x.to(dtype=image.dtype, device=image.device), padding=1
        )
        gy = F.conv2d(
            luma, self.sobel_y.to(dtype=image.dtype, device=image.device), padding=1
        )
        sobel = torch.sqrt(gx.square() + gy.square() + 1e-6).div(4.0).clamp(0.0, 1.0)
        laplacian = F.conv2d(
            luma, self.laplacian.to(dtype=image.dtype, device=image.device), padding=1
        ).abs().div(4.0).clamp(0.0, 1.0)
        return sobel, laplacian

    def forward(
        self,
        b0: torch.Tensor,
        auxiliary: torch.Tensor,
        snr_norm: torch.Tensor,
        residual_gate: torch.Tensor,
    ) -> torch.Tensor:
        if b0.shape != auxiliary.shape:
            raise ValueError("B0 and auxiliary images must have identical shapes")
        batch, channels, height, width = b0.shape
        if channels != 3:
            raise ValueError("fusion refiner expects RGB observations")
        snr_map = snr_norm.view(batch, 1, 1, 1).expand(batch, 1, height, width)
        sobel, laplacian = self.structural_conditions(b0)
        features = self.head(torch.cat([b0, auxiliary, snr_map, sobel, laplacian], dim=1))
        residual = torch.tanh(self.tail(self.body(features)))
        return (b0 + residual_gate.view(batch, 1, 1, 1) * residual).clamp(0.0, 1.0)


def expand_b1_state_dict(
    b1_state: Mapping[str, torch.Tensor], model: DualInputResidualRefiner
) -> dict[str, torch.Tensor]:
    """Expand B1's six-channel head to nine channels, initially ignoring auxiliary RGB."""

    old_head = b1_state.get("head.0.weight")
    if old_head is None or tuple(old_head.shape[1:]) != (6, 3, 3):
        raise ValueError("B1 checkpoint does not have the expected six-channel head")
    expanded = {key: value.detach().clone() for key, value in model.state_dict().items()}
    unknown = sorted(set(b1_state) - set(expanded))
    if unknown:
        raise ValueError(f"B1 checkpoint has unexpected parameters: {unknown}")
    for key, value in b1_state.items():
        if key == "head.0.weight":
            continue
        if key not in expanded or expanded[key].shape != value.shape:
            raise ValueError(f"B1 parameter shape mismatch: {key}")
        expanded[key] = value.detach().clone()
    new_head = torch.zeros_like(expanded["head.0.weight"])
    new_head[:, 0:3] = old_head[:, 0:3]
    new_head[:, 6:9] = old_head[:, 3:6]
    expanded["head.0.weight"] = new_head
    return expanded


def residual_gate_tensor(
    snr_db: torch.Tensor, gates: Mapping[str, float], device: torch.device
) -> torch.Tensor:
    values: list[float] = []
    for value in snr_db.detach().cpu().tolist():
        key = str(int(value)) if float(value).is_integer() else str(float(value))
        if key not in gates:
            raise KeyError(f"missing residual gate for SNR {value}")
        values.append(float(gates[key]))
    return torch.tensor(values, dtype=torch.float32, device=device)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
