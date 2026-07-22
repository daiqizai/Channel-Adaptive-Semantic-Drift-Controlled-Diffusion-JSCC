from __future__ import annotations

import hashlib
import math

import torch


def canonical_noise_seed(base_seed: int, sample_id: str, snr_db: float) -> int:
    """Derive one framework-stable CPU RNG seed for an image/SNR condition."""

    material = f"external-common-v1|{int(base_seed)}|{sample_id}|{float(snr_db):.6f}"
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def canonical_standard_normal(
    base_seed: int,
    sample_id: str,
    snr_db: float,
    real_symbols: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate the frozen CPU standard-normal vector shared by all methods."""

    if real_symbols <= 0:
        raise ValueError("real_symbols must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(canonical_noise_seed(base_seed, sample_id, snr_db))
    return torch.randn(real_symbols, generator=generator, dtype=dtype)


def canonical_noise_sha256(noise: torch.Tensor) -> str:
    value = noise.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def complex_awgn_from_standard_normal(
    transmitted: torch.Tensor,
    standard_normal: torch.Tensor,
    snr_db: float,
) -> torch.Tensor:
    """Apply complex AWGN represented by paired real coordinates.

    ``snr_db`` is Es/N0 per complex channel use.  Each real coordinate gets
    variance ``P/(2*SNR_linear)``.  Signal power is measured independently per
    sample over all real coordinates, matching the project's DeepJSCC channel.
    """

    if transmitted.ndim < 2:
        raise ValueError("transmitted tensor must include batch and symbol dimensions")
    if standard_normal.numel() != transmitted.numel():
        raise ValueError(
            f"noise has {standard_normal.numel()} values for {transmitted.numel()} symbols"
        )
    batch = transmitted.shape[0]
    symbols_per_sample = transmitted[0].numel()
    if any(item.numel() != symbols_per_sample for item in transmitted):
        raise ValueError("all transmitted samples must have the same symbol count")
    flat = transmitted.flatten(start_dim=1)
    power = flat.square().mean(dim=1, keepdim=True)
    sigma = torch.sqrt(power / (2.0 * (10.0 ** (float(snr_db) / 10.0))))
    z = standard_normal.to(device=transmitted.device, dtype=transmitted.dtype).reshape(
        batch, symbols_per_sample
    )
    return (flat + z * sigma).view_as(transmitted)


def complex_cbr(total_real_symbols: int, source_real_dimensions: int) -> float:
    if total_real_symbols <= 0 or source_real_dimensions <= 0:
        raise ValueError("symbol and source dimensions must be positive")
    if total_real_symbols % 2:
        raise ValueError("paired-real complex channel uses require an even symbol count")
    return (total_real_symbols / 2.0) / source_real_dimensions


def repetition_majority_bit_error_probability(raw_bit_error: float, repetitions: int) -> float:
    """Exact odd-repetition hard-decision BER, used only for protocol checks."""

    if not 0.0 <= raw_bit_error <= 1.0:
        raise ValueError("raw_bit_error must lie in [0, 1]")
    if repetitions <= 0 or repetitions % 2 != 1:
        raise ValueError("repetitions must be a positive odd integer")
    threshold = repetitions // 2 + 1
    return sum(
        math.comb(repetitions, errors)
        * raw_bit_error**errors
        * (1.0 - raw_bit_error) ** (repetitions - errors)
        for errors in range(threshold, repetitions + 1)
    )
