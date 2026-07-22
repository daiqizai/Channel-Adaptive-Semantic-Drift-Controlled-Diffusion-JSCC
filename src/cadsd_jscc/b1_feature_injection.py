"""Minimal zero-conv injection of matched-diffusion differences into frozen B1 features."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


class FrozenB1FeatureInjection(nn.Module):
    def __init__(self, frozen_b1: nn.Module, feature_channels: int = 64) -> None:
        super().__init__()
        self.b1 = frozen_b1.eval().requires_grad_(False)
        self.aux_projection = nn.Conv2d(
            3, feature_channels, kernel_size=3, padding=1, bias=False
        )
        nn.init.zeros_(self.aux_projection.weight)

    def train(self, mode: bool = True) -> "FrozenB1FeatureInjection":
        """Train only the projection while keeping the complete B1 anchor in eval mode."""

        super().train(mode)
        self.b1.eval()
        return self

    def forward(
        self,
        b0: torch.Tensor,
        auxiliary: torch.Tensor,
        snr_norm: torch.Tensor,
        b1_residual_gate: torch.Tensor,
        auxiliary_envelope: torch.Tensor,
    ) -> torch.Tensor:
        if b0.shape != auxiliary.shape:
            raise ValueError("B0 and auxiliary must have identical shapes")
        batch, channels, height, width = b0.shape
        if channels != 3:
            raise ValueError("feature injection expects RGB inputs")
        if snr_norm.shape != (batch,):
            raise ValueError("snr_norm must have shape [batch]")
        if b1_residual_gate.shape != (batch,) or auxiliary_envelope.shape != (batch,):
            raise ValueError("gate and envelope must have shape [batch]")
        snr_map = snr_norm.view(batch, 1, 1, 1).expand(batch, 1, height, width)
        conditions = self.b1.structural_conditions(b0)
        base_features = self.b1.head(torch.cat([b0, snr_map, *conditions], dim=1))
        difference = auxiliary - b0
        injected = base_features + auxiliary_envelope.view(batch, 1, 1, 1) * self.aux_projection(
            difference
        )
        residual = torch.tanh(self.b1.tail(self.b1.body(injected)))
        return (
            b0 + b1_residual_gate.view(batch, 1, 1, 1) * residual
        ).clamp(0.0, 1.0)


def envelope_tensor(
    snr_db: torch.Tensor, envelope: Mapping[str, float], device: torch.device
) -> torch.Tensor:
    values = []
    for value in snr_db.detach().cpu().tolist():
        key = str(int(value)) if float(value).is_integer() else str(float(value))
        if key not in envelope:
            raise KeyError(f"missing feature-injection envelope for SNR {value}")
        values.append(float(envelope[key]))
    return torch.tensor(values, dtype=torch.float32, device=device)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
