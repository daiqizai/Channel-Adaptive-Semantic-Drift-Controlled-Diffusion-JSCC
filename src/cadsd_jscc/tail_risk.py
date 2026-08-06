"""Conditional tail-risk utilities and block-fading Rayleigh channel.

This module supports the CVaR candidate direction (`候选二`).  It deliberately
keeps two concerns separate:

1.  Tail statistics (empirical CVaR / low-tail means) that are shared by the
    diagnostic stage and any later risk-sensitive training stage.
2.  A minimal, reproducible block-fading Rayleigh channel that reuses the
    project's existing AWGN conventions.

Channel conventions inherited from ``cadsd_jscc.external_common`` and
``cadsd_jscc.strong_jscc``:

*   ``snr_db`` is Es/N0 per **complex** channel use.
*   Signal power ``P`` is measured per sample over all real coordinates.
*   Every real coordinate receives noise variance ``P / (2 * SNR_linear)``.

Adjacent real coordinates in flattened per-sample order are paired into one
complex symbol: real index ``2k`` is the in-phase part and ``2k + 1`` the
quadrature part of complex symbol ``k``.  This pairing is a *definition* of
this module and is asserted by the unit tests.
"""

from __future__ import annotations

import hashlib
import math

import torch


# --------------------------------------------------------------------------- #
# Tail statistics
# --------------------------------------------------------------------------- #


def tail_count(count: int, tail_fraction: float) -> int:
    """Number of samples in the empirical tail, at least one."""

    if count <= 0:
        raise ValueError("count must be positive")
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1]")
    return max(1, math.ceil(count * tail_fraction))


def empirical_upper_cvar(
    distortion: torch.Tensor,
    tail_fraction: float,
    dim: int = -1,
) -> torch.Tensor:
    """Empirical upper-tail CVaR, where larger ``distortion`` is worse.

    ``tail_fraction=0.1`` returns the mean of the worst 10% of values along
    ``dim``.  The tail size is ``max(1, ceil(n * tail_fraction))``, so a small
    ``n`` can degenerate to the single worst value; callers that need CVaR-10%
    and CVaR-20% to differ must supply enough realizations.
    """

    if distortion.numel() == 0:
        raise ValueError("distortion must not be empty")
    count = distortion.shape[dim]
    worst = torch.topk(
        distortion,
        k=tail_count(count, tail_fraction),
        dim=dim,
        largest=True,
        sorted=False,
    ).values
    return worst.mean(dim=dim)


def empirical_lower_tail_mean(
    quality: torch.Tensor,
    tail_fraction: float,
    dim: int = -1,
) -> torch.Tensor:
    """Mean of the worst values when larger ``quality`` is better (e.g. PSNR)."""

    return -empirical_upper_cvar(-quality, tail_fraction=tail_fraction, dim=dim)


def conditional_cvar_objective(
    distortion: torch.Tensor,
    tail_fraction: float,
    risk_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Mixed mean/CVaR objective over a ``[B, M]`` distortion matrix.

    ``B`` indexes distinct source images and ``M`` independent channel
    realizations of the *same* image.  CVaR is taken per image and then
    averaged over images, so image-content difficulty cannot leak into the
    channel tail.
    """

    if distortion.ndim != 2:
        raise ValueError(
            f"Expected distortion with shape [B, M], got {tuple(distortion.shape)}"
        )
    if not 0.0 <= risk_weight <= 1.0:
        raise ValueError("risk_weight must be in [0, 1]")

    mean_loss = distortion.mean()
    per_image_cvar = empirical_upper_cvar(distortion, tail_fraction, dim=1)
    cvar_loss = per_image_cvar.mean()
    total_loss = (1.0 - risk_weight) * mean_loss + risk_weight * cvar_loss
    stats = {
        "loss_total": total_loss.detach(),
        "loss_mean": mean_loss.detach(),
        "loss_cvar": cvar_loss.detach(),
    }
    return total_loss, stats


# --------------------------------------------------------------------------- #
# Reproducible block-fading Rayleigh channel
# --------------------------------------------------------------------------- #


def fading_seed(base_seed: int, sample_id: str, snr_db: float, realization: int) -> int:
    """Derive one framework-stable CPU RNG seed for a fading realization."""

    material = (
        f"cvar-tail-risk-fading-v1|{int(base_seed)}|{sample_id}"
        f"|{float(snr_db):.6f}|{int(realization)}"
    )
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def block_fading_coefficient(
    base_seed: int,
    sample_id: str,
    snr_db: float,
    realization: int,
) -> tuple[float, float]:
    """One complex Rayleigh coefficient ``h ~ CN(0, 1)`` for a whole image.

    Returns ``(h_real, h_imag)`` with ``E[|h|^2] = 1``, i.e. both parts are
    ``N(0, 1/2)``.  Generation is on CPU and depends only on the arguments, so
    a realization can be replayed exactly on any device.
    """

    generator = torch.Generator(device="cpu")
    generator.manual_seed(fading_seed(base_seed, sample_id, snr_db, realization))
    parts = torch.randn(2, generator=generator, dtype=torch.float64) * (0.5**0.5)
    return float(parts[0]), float(parts[1])


def effective_snr_db(snr_db: float, h_power: float) -> float:
    """Post-equalization Es/N0 for zero-forcing with perfect CSI."""

    if h_power <= 0.0:
        raise ValueError("h_power must be positive")
    return float(snr_db) + 10.0 * math.log10(float(h_power))


def apply_block_fading_channel(
    transmitted: torch.Tensor,
    standard_normal: torch.Tensor,
    snr_db: float | torch.Tensor,
    h_real: torch.Tensor,
    h_imag: torch.Tensor,
    *,
    epsilon: float = 0.0,
) -> torch.Tensor:
    """Transmit through ``y = h x + n`` and zero-forcing equalize with known ``h``.

    ``transmitted`` is ``[B, ...]`` of real coordinates with an even count per
    sample.  ``h_real``/``h_imag`` are ``[B]`` tensors holding one complex
    coefficient per sample (block fading: constant across the whole image).
    ``standard_normal`` supplies unit-variance noise with the same shape as
    ``transmitted``.  ``snr_db`` is either one scalar shared by the batch or a
    ``[B]`` tensor of per-sample nominal SNRs (needed when training samples a
    different SNR per image).

    The returned tensor is the equalized observation
    ``x + conj(h) / (|h|^2 + epsilon) * n``.  With ``epsilon == 0`` this is
    exactly AWGN at ``effective_snr_db(snr_db, |h|^2)``, which is why the
    diagnostic must record ``|h|^2`` alongside every metric.
    """

    if transmitted.ndim < 2:
        raise ValueError("transmitted tensor must include batch and symbol dimensions")
    batch = transmitted.shape[0]
    symbols_per_sample = transmitted[0].numel()
    if symbols_per_sample % 2:
        raise ValueError("paired-real complex symbols require an even coordinate count")
    if standard_normal.numel() != transmitted.numel():
        raise ValueError(
            f"noise has {standard_normal.numel()} values for {transmitted.numel()} symbols"
        )
    if h_real.shape != (batch,) or h_imag.shape != (batch,):
        raise ValueError("h_real and h_imag must both have shape [batch]")

    flat = transmitted.flatten(start_dim=1)
    power = flat.float().square().mean(dim=1, keepdim=True)
    if isinstance(snr_db, torch.Tensor):
        if snr_db.reshape(-1).shape != (batch,):
            raise ValueError("tensor snr_db must have shape [batch]")
        gamma = torch.pow(
            10.0, snr_db.to(flat.device).float().reshape(batch, 1) / 10.0
        )
    else:
        gamma = 10.0 ** (float(snr_db) / 10.0)
    sigma = torch.sqrt(power / (2.0 * gamma))
    noise = standard_normal.to(device=flat.device, dtype=flat.dtype).reshape(
        batch, symbols_per_sample
    )

    signal = torch.view_as_complex(
        flat.float().reshape(batch, symbols_per_sample // 2, 2).contiguous()
    )
    scaled_noise = (noise.float() * sigma.float()).reshape(
        batch, symbols_per_sample // 2, 2
    )
    noise_complex = torch.view_as_complex(scaled_noise.contiguous())

    h = torch.complex(h_real.to(flat.device).float(), h_imag.to(flat.device).float())
    h = h.reshape(batch, 1)
    h_power = (h.real.square() + h.imag.square()).clamp_min(1e-30)

    received = h * signal + noise_complex
    equalized = received * torch.conj(h) / (h_power + float(epsilon))
    out = torch.view_as_real(equalized).reshape(batch, symbols_per_sample)
    return out.to(transmitted.dtype).view_as(transmitted)
