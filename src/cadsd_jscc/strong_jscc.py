"""Clean-room, rate-native, SNR-conditioned residual JSCC backbone.

This module intentionally does not import third-party DiffJSCC code.  It keeps
the physical link contract used by this project while replacing the historical
tiny DeepJSCC sanity model with a substantially wider residual architecture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def _group_count(channels: int, maximum: int = 32) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class SNRConditionEmbedding(nn.Module):
    """Embed a scalar SNR using fixed Fourier features and a small MLP."""

    def __init__(self, embedding_dim: int = 128, fourier_bands: int = 16) -> None:
        super().__init__()
        if embedding_dim <= 0 or fourier_bands <= 0:
            raise ValueError("embedding dimensions must be positive")
        self.embedding_dim = int(embedding_dim)
        self.fourier_bands = int(fourier_bands)
        frequencies = 2.0 ** torch.arange(self.fourier_bands, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)
        input_dim = 1 + 2 * self.fourier_bands
        self.network = nn.Sequential(
            nn.Linear(input_dim, self.embedding_dim),
            nn.SiLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.SiLU(),
        )

    def forward(self, snr_db: torch.Tensor) -> torch.Tensor:
        snr = snr_db.reshape(-1, 1).float() / 20.0
        angles = math.pi * snr * self.frequencies.reshape(1, -1)
        features = torch.cat((snr, torch.sin(angles), torch.cos(angles)), dim=1)
        return self.network(features)


class ConditionalResidualBlock(nn.Module):
    """Pre-activation residual block with SNR-dependent affine modulation."""

    def __init__(self, channels: int, condition_dim: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.condition = nn.Sequential(
            nn.SiLU(), nn.Linear(condition_dim, 2 * channels)
        )
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

        # Start each residual branch near identity without making it a dead path.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.condition[-1].weight)
        nn.init.zeros_(self.condition[-1].bias)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(value)))
        hidden = self.norm2(hidden)
        scale, shift = self.condition(condition).chunk(2, dim=1)
        hidden = hidden * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv2(F.silu(hidden))
        return value + hidden


class DownsampleStage(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, blocks: int, condition_dim: int
    ) -> None:
        super().__init__()
        self.downsample = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )
        self.blocks = nn.ModuleList(
            ConditionalResidualBlock(out_channels, condition_dim) for _ in range(blocks)
        )

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        value = self.downsample(value)
        for block in self.blocks:
            value = block(value, condition)
        return value


class UpsampleStage(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, blocks: int, condition_dim: int
    ) -> None:
        super().__init__()
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            ConditionalResidualBlock(out_channels, condition_dim) for _ in range(blocks)
        )

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(value, scale_factor=2.0, mode="bilinear", align_corners=False)
        value = self.projection(value)
        for block in self.blocks:
            value = block(value, condition)
        return value


class StrongJSCCEncoder(nn.Module):
    def __init__(
        self,
        *,
        latent_channels: int,
        stage_channels: tuple[int, int, int, int],
        stage_blocks: tuple[int, int, int, int],
        condition_dim: int,
    ) -> None:
        super().__init__()
        if len(stage_channels) != 4 or len(stage_blocks) != 4:
            raise ValueError("strong JSCC requires exactly four encoder stages")
        self.stem = nn.Conv2d(3, stage_channels[0], kernel_size=7, padding=3)
        input_channels = stage_channels[0]
        stages = []
        for output_channels, blocks in zip(stage_channels, stage_blocks):
            stages.append(
                DownsampleStage(input_channels, output_channels, blocks, condition_dim)
            )
            input_channels = output_channels
        self.stages = nn.ModuleList(stages)
        self.output_norm = nn.GroupNorm(_group_count(input_channels), input_channels)
        self.output = nn.Conv2d(input_channels, latent_channels, kernel_size=3, padding=1)

    def forward(self, image: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        value = self.stem(image)
        for stage in self.stages:
            value = stage(value, condition)
        return self.output(F.silu(self.output_norm(value)))


class StrongJSCCDecoder(nn.Module):
    def __init__(
        self,
        *,
        latent_channels: int,
        stage_channels: tuple[int, int, int, int],
        stage_blocks: tuple[int, int, int, int],
        condition_dim: int,
    ) -> None:
        super().__init__()
        if len(stage_channels) != 4 or len(stage_blocks) != 4:
            raise ValueError("strong JSCC requires exactly four decoder stages")
        reversed_channels = tuple(reversed(stage_channels))
        reversed_blocks = tuple(reversed(stage_blocks))
        self.input = nn.Conv2d(
            latent_channels, reversed_channels[0], kernel_size=3, padding=1
        )
        self.bottleneck = nn.ModuleList(
            ConditionalResidualBlock(reversed_channels[0], condition_dim)
            for _ in range(reversed_blocks[0])
        )
        stages = []
        input_channels = reversed_channels[0]
        for output_channels, blocks in zip(
            reversed_channels[1:] + (stage_channels[0],),
            reversed_blocks[1:] + (stage_blocks[0],),
        ):
            stages.append(
                UpsampleStage(input_channels, output_channels, blocks, condition_dim)
            )
            input_channels = output_channels
        self.stages = nn.ModuleList(stages)
        self.output_norm = nn.GroupNorm(_group_count(input_channels), input_channels)
        self.output = nn.Conv2d(input_channels, 3, kernel_size=7, padding=3)

    def forward(self, latent: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        value = self.input(latent)
        for block in self.bottleneck:
            value = block(value, condition)
        for stage in self.stages:
            value = stage(value, condition)
        return torch.sigmoid(self.output(F.silu(self.output_norm(value))))


@dataclass(frozen=True)
class StrongJSCCObservation:
    transmitted: torch.Tensor
    received: torch.Tensor
    normalized_power: torch.Tensor
    snr_db: torch.Tensor


class StrongJSCC(nn.Module):
    """A native exact-rate residual JSCC model for 256x256 RGB images.

    With four spatial downsampling stages and ``latent_channels=77``, the
    channel tensor has exactly ``77 * 16 * 16 = 19,712`` real coordinates.
    Consecutive real coordinates are interpreted as the I/Q components of a
    complex channel use.  AWGN therefore follows the project convention
    ``var(real noise) = P / (2 * 10**(SNR/10))``.
    """

    def __init__(
        self,
        *,
        image_size: int = 256,
        latent_channels: int = 77,
        stage_channels: tuple[int, int, int, int] = (64, 128, 256, 384),
        stage_blocks: tuple[int, int, int, int] = (1, 1, 2, 4),
        condition_dim: int = 128,
    ) -> None:
        super().__init__()
        if image_size % 16:
            raise ValueError("image_size must be divisible by 16")
        self.image_size = int(image_size)
        self.latent_channels = int(latent_channels)
        self.real_symbols = self.latent_channels * (self.image_size // 16) ** 2
        if self.real_symbols % 2:
            raise ValueError("real-symbol count must be even for paired-real AWGN")
        self.condition_embedding = SNRConditionEmbedding(condition_dim)
        self.encoder = StrongJSCCEncoder(
            latent_channels=self.latent_channels,
            stage_channels=stage_channels,
            stage_blocks=stage_blocks,
            condition_dim=condition_dim,
        )
        self.decoder = StrongJSCCDecoder(
            latent_channels=self.latent_channels,
            stage_channels=stage_channels,
            stage_blocks=stage_blocks,
            condition_dim=condition_dim,
        )

    def _snr_tensor(
        self, snr_db: torch.Tensor | float, batch: int, device: torch.device
    ) -> torch.Tensor:
        value = torch.as_tensor(snr_db, device=device, dtype=torch.float32).reshape(-1)
        if value.numel() == 1:
            value = value.expand(batch)
        if value.numel() != batch:
            raise ValueError(f"expected one SNR per image, got {value.numel()} for {batch}")
        return value

    def encode(self, image: torch.Tensor, snr_db: torch.Tensor | float) -> torch.Tensor:
        if tuple(image.shape[-2:]) != (self.image_size, self.image_size):
            raise ValueError(
                f"expected {self.image_size}x{self.image_size}, got {tuple(image.shape[-2:])}"
            )
        snr = self._snr_tensor(snr_db, image.shape[0], image.device)
        return self.encoder(image, self.condition_embedding(snr))

    @staticmethod
    def normalize_channel_input(latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = latent.flatten(start_dim=1)
        power = flat.float().square().mean(dim=1, keepdim=True).clamp_min(1e-12)
        normalized = flat / torch.sqrt(power).to(flat.dtype)
        return normalized.reshape_as(latent), power.reshape(-1)

    def transmit(
        self,
        transmitted: torch.Tensor,
        snr_db: torch.Tensor | float,
        standard_normal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        snr = self._snr_tensor(snr_db, transmitted.shape[0], transmitted.device)
        if standard_normal is None:
            noise = torch.randn_like(transmitted)
        else:
            noise = standard_normal.to(device=transmitted.device, dtype=transmitted.dtype)
            if noise.shape != transmitted.shape:
                if noise.numel() != transmitted.numel():
                    raise ValueError("standard-normal tensor does not match channel tensor")
                noise = noise.reshape_as(transmitted)
        power = transmitted.float().flatten(start_dim=1).square().mean(dim=1)
        gamma = torch.pow(10.0, snr / 10.0)
        sigma = torch.sqrt(power / (2.0 * gamma)).to(transmitted.dtype)
        return transmitted + noise * sigma[:, None, None, None]

    def decode(self, received: torch.Tensor, snr_db: torch.Tensor | float) -> torch.Tensor:
        snr = self._snr_tensor(snr_db, received.shape[0], received.device)
        return self.decoder(received, self.condition_embedding(snr))

    def forward_with_observation(
        self,
        image: torch.Tensor,
        snr_db: torch.Tensor | float,
        standard_normal: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, StrongJSCCObservation]:
        latent = self.encode(image, snr_db)
        transmitted, _unnormalized_power = self.normalize_channel_input(latent)
        received = self.transmit(transmitted, snr_db, standard_normal)
        reconstruction = self.decode(received, snr_db)
        normalized_power = transmitted.float().flatten(start_dim=1).square().mean(dim=1)
        snr = self._snr_tensor(snr_db, image.shape[0], image.device)
        observation = StrongJSCCObservation(
            transmitted=transmitted,
            received=received,
            normalized_power=normalized_power,
            snr_db=snr,
        )
        return reconstruction, observation

    def forward(
        self,
        image: torch.Tensor,
        snr_db: torch.Tensor | float,
        standard_normal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_with_observation(image, snr_db, standard_normal)[0]


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
