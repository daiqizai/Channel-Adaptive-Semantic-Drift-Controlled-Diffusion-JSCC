from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def channel_alpha(
    snr_db: float | torch.Tensor,
    noise_variance_factor_per_real: float = 0.5,
) -> float | torch.Tensor:
    """Map the frozen AWGN convention to a diffusion cumulative alpha.

    The project channel uses ``sigma^2 = factor * P / gamma`` per real
    coordinate.  After normalizing the received coordinate by its total
    standard deviation, ``alpha = 1 / (1 + factor / gamma)``.
    """

    factor = float(noise_variance_factor_per_real)
    if factor <= 0:
        raise ValueError("noise_variance_factor_per_real must be positive")
    if isinstance(snr_db, torch.Tensor):
        gamma = torch.pow(snr_db.new_tensor(10.0), snr_db / 10.0)
        return 1.0 / (1.0 + factor / gamma)
    gamma = 10.0 ** (float(snr_db) / 10.0)
    return 1.0 / (1.0 + factor / gamma)


def alpha_to_logsnr(alpha: torch.Tensor) -> torch.Tensor:
    alpha = alpha.clamp(1e-6, 1.0 - 1e-6)
    return torch.log(alpha) - torch.log1p(-alpha)


def normalize_channel_observation(
    received: torch.Tensor, alpha: float | torch.Tensor
) -> torch.Tensor:
    """Convert an unscaled unit-power AWGN observation into diffusion state x_t."""

    alpha_tensor = torch.as_tensor(alpha, device=received.device, dtype=received.dtype)
    while alpha_tensor.ndim < received.ndim:
        alpha_tensor = alpha_tensor.unsqueeze(-1)
    return received * alpha_tensor.sqrt()


def reverse_alpha_schedule(
    alpha_start: float,
    sampling_steps: int,
    alpha_max: float = 0.999,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return monotonically increasing alphas, spaced uniformly in log-SNR."""

    alpha_start = float(alpha_start)
    alpha_max = float(alpha_max)
    if sampling_steps <= 0:
        raise ValueError("sampling_steps must be positive")
    if not 0.0 < alpha_start < alpha_max < 1.0:
        raise ValueError("require 0 < alpha_start < alpha_max < 1")
    start = math.log(alpha_start) - math.log1p(-alpha_start)
    end = math.log(alpha_max) - math.log1p(-alpha_max)
    logsnr = torch.linspace(start, end, sampling_steps, device=device, dtype=dtype)
    return torch.sigmoid(logsnr)


def expand_valid_mask(valid_mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    mask = torch.as_tensor(valid_mask, device=reference.device, dtype=torch.bool)
    if tuple(mask.shape) == tuple(reference.shape[1:]):
        mask = mask.unsqueeze(0).expand(reference.shape[0], -1, -1, -1)
    elif tuple(mask.shape) == tuple(reference.shape):
        pass
    else:
        raise ValueError(
            f"valid mask shape {tuple(mask.shape)} is incompatible with "
            f"reference shape {tuple(reference.shape)}"
        )
    return mask


def masked_mse_per_sample(
    prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    mask = expand_valid_mask(valid_mask, prediction)
    count = mask.flatten(1).sum(dim=1)
    if bool((count == 0).any()):
        raise ValueError("valid mask must retain coordinates")
    squared = (prediction - target).square() * mask
    return squared.flatten(1).sum(dim=1) / count.to(prediction.dtype)


class SinusoidalLogSNREmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension < 4 or dimension % 2:
            raise ValueError("time embedding dimension must be even and at least four")
        self.dimension = int(dimension)

    def forward(self, logsnr: torch.Tensor) -> torch.Tensor:
        if logsnr.ndim != 1:
            raise ValueError("logsnr must be a batch vector")
        half = self.dimension // 2
        exponent = -math.log(10000.0) * torch.arange(
            half, device=logsnr.device, dtype=logsnr.dtype
        ) / max(half - 1, 1)
        phase = logsnr[:, None] * exponent.exp()[None, :]
        return torch.cat([phase.sin(), phase.cos()], dim=1)


class TimeConditionedResidualBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int, groups: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time = nn.Linear(time_dim, channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, inputs: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(inputs)))
        hidden = hidden + self.time(F.silu(time_embedding))[:, :, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return inputs + hidden


class ChannelMatchedLatentDenoiser(nn.Module):
    """Small masked epsilon predictor for the frozen DeepJSCC codeword space."""

    def __init__(
        self,
        latent_channels: int,
        base_channels: int = 48,
        num_blocks: int = 6,
        time_embedding_dim: int = 96,
        group_norm_groups: int = 8,
    ) -> None:
        super().__init__()
        if latent_channels <= 0 or base_channels <= 0 or num_blocks <= 0:
            raise ValueError("model dimensions must be positive")
        if base_channels % group_norm_groups:
            raise ValueError("base channels must be divisible by group norm groups")
        self.latent_channels = int(latent_channels)
        self.time_embedding = SinusoidalLogSNREmbedding(time_embedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embedding_dim * 2, time_embedding_dim),
        )
        self.input = nn.Conv2d(2 * latent_channels, base_channels, 3, padding=1)
        self.blocks = nn.ModuleList(
            [
                TimeConditionedResidualBlock(
                    base_channels, time_embedding_dim, group_norm_groups
                )
                for _ in range(num_blocks)
            ]
        )
        self.output_norm = nn.GroupNorm(group_norm_groups, base_channels)
        self.output = nn.Conv2d(base_channels, latent_channels, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, noisy_latent: torch.Tensor, alpha: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        if noisy_latent.ndim != 4 or noisy_latent.shape[1] != self.latent_channels:
            raise ValueError("noisy latent has the wrong BCHW shape")
        if alpha.ndim == 0:
            alpha = alpha.expand(noisy_latent.shape[0])
        if alpha.shape != (noisy_latent.shape[0],):
            raise ValueError("alpha must be scalar or a batch vector")
        mask = expand_valid_mask(valid_mask, noisy_latent).to(noisy_latent.dtype)
        time = self.time_mlp(self.time_embedding(alpha_to_logsnr(alpha)))
        hidden = self.input(torch.cat([noisy_latent * mask, mask], dim=1))
        for block in self.blocks:
            hidden = block(hidden, time)
        epsilon = self.output(F.silu(self.output_norm(hidden)))
        return epsilon * mask


def predict_x0_from_epsilon(
    x_t: torch.Tensor, epsilon: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
    alpha_view = alpha
    while alpha_view.ndim < x_t.ndim:
        alpha_view = alpha_view.unsqueeze(-1)
    return (x_t - (1.0 - alpha_view).sqrt() * epsilon) / alpha_view.sqrt()


@torch.no_grad()
def deterministic_ddim(
    model: nn.Module,
    x_t: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    alpha_start: float,
    sampling_steps: int,
    alpha_max: float = 0.999,
    measurement: torch.Tensor | None = None,
    measurement_blend: float = 0.0,
) -> torch.Tensor:
    """Run a deterministic masked DDIM path and return the final x0 estimate."""

    if not 0.0 <= measurement_blend < 1.0:
        raise ValueError("measurement_blend must lie in [0, 1)")
    if measurement_blend > 0 and measurement is None:
        raise ValueError("measurement is required when measurement_blend is nonzero")
    mask = expand_valid_mask(valid_mask, x_t).to(x_t.dtype)
    state = x_t * mask
    schedule = reverse_alpha_schedule(
        alpha_start,
        sampling_steps,
        alpha_max,
        device=x_t.device,
        dtype=x_t.dtype,
    )
    final_x0: torch.Tensor | None = None
    for index, alpha_scalar in enumerate(schedule):
        alpha = alpha_scalar.expand(state.shape[0])
        epsilon = model(state, alpha, mask)
        x0 = predict_x0_from_epsilon(state, epsilon, alpha) * mask
        if measurement_blend:
            assert measurement is not None
            x0 = (
                (1.0 - measurement_blend) * x0
                + measurement_blend * measurement * mask
            )
        final_x0 = x0
        if index + 1 < len(schedule):
            next_alpha = schedule[index + 1]
            state = (
                next_alpha.sqrt() * x0
                + (1.0 - next_alpha).sqrt() * epsilon
            ) * mask
    if final_x0 is None:
        raise RuntimeError("empty DDIM schedule")
    return final_x0
