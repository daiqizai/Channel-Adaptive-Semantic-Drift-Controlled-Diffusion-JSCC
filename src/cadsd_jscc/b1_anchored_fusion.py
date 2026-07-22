"""B1-anchored, spatially gated injection of an auxiliary restoration."""

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


class B1AnchoredGatedAdapter(nn.Module):
    """Inject a bounded auxiliary residual while keeping B1 as an exact fixed point."""

    def __init__(
        self,
        base_channels: int = 64,
        num_blocks: int = 6,
        spatial_gate_mode: str = "learned_sigmoid",
    ) -> None:
        super().__init__()
        if spatial_gate_mode not in {"learned_sigmoid", "fixed_one"}:
            raise ValueError(f"unsupported spatial_gate_mode: {spatial_gate_mode}")
        self.spatial_gate_mode = spatial_gate_mode
        self.head = nn.Sequential(
            nn.Conv2d(12, base_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(num_blocks)])
        self.residual_head = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)
        self.gate_head = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
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
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)

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
        anchor: torch.Tensor,
        auxiliary: torch.Tensor,
        snr_norm: torch.Tensor,
        max_injection: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if anchor.shape != auxiliary.shape:
            raise ValueError("anchor and auxiliary images must have identical shapes")
        batch, channels, height, width = anchor.shape
        if channels != 3:
            raise ValueError("B1-anchored adapter expects RGB images")
        if snr_norm.shape != (batch,) or max_injection.shape != (batch,):
            raise ValueError("snr_norm and max_injection must have shape [batch]")
        snr_map = snr_norm.view(batch, 1, 1, 1).expand(batch, 1, height, width)
        sobel, laplacian = self.structural_conditions(anchor)
        disagreement = (auxiliary - anchor).abs()
        features = self.head(
            torch.cat(
                [anchor, auxiliary, disagreement, snr_map, sobel, laplacian], dim=1
            )
        )
        features = self.body(features)
        residual = torch.tanh(self.residual_head(features))
        spatial_gate = (
            torch.sigmoid(self.gate_head(features))
            if self.spatial_gate_mode == "learned_sigmoid"
            else torch.ones(
                (batch, 1, height, width), dtype=features.dtype, device=features.device
            )
        )
        scale = max_injection.view(batch, 1, 1, 1)
        injection = scale * spatial_gate * residual
        output = (anchor + injection).clamp(0.0, 1.0)
        if not return_diagnostics:
            return output
        return output, {
            "spatial_gate": spatial_gate,
            "residual": residual,
            "injection": injection,
            "disagreement": disagreement,
        }


def injection_gate_tensor(
    snr_db: torch.Tensor, gates: Mapping[str, float], device: torch.device
) -> torch.Tensor:
    values: list[float] = []
    for value in snr_db.detach().cpu().tolist():
        key = str(int(value)) if float(value).is_integer() else str(float(value))
        if key not in gates:
            raise KeyError(f"missing max injection gate for SNR {value}")
        values.append(float(gates[key]))
    return torch.tensor(values, dtype=torch.float32, device=device)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
