from __future__ import annotations

import math

import torch


def fixed_rademacher_projection(
    source_dim: int,
    sketch_dim: int,
    seed: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a deterministic dense +/-1/sqrt(source_dim) projection matrix."""

    if source_dim <= 0 or sketch_dim <= 0:
        raise ValueError("source_dim and sketch_dim must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    signs = torch.randint(0, 2, (sketch_dim, source_dim), generator=generator, dtype=torch.int8)
    matrix = signs.to(torch.float32).mul_(2.0).sub_(1.0).div_(math.sqrt(source_dim))
    return matrix.to(device=device, dtype=dtype)


def normalize_sketch(sketch: torch.Tensor) -> torch.Tensor:
    if sketch.ndim != 2:
        raise ValueError(f"Expected BxD sketch, got {tuple(sketch.shape)}")
    return sketch * math.sqrt(sketch.shape[1]) / sketch.norm(dim=1, keepdim=True).clamp_min(1e-8)


def probabilities_to_sketch(probabilities: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    """Project a probability vector and normalize it to unit mean symbol power."""

    if probabilities.ndim != 2 or projection.ndim != 2:
        raise ValueError("probabilities and projection must both be matrices")
    if probabilities.shape[1] != projection.shape[1]:
        raise ValueError(
            f"Source dimension mismatch: {probabilities.shape[1]} vs {projection.shape[1]}"
        )
    if not torch.isfinite(probabilities).all():
        raise ValueError("Non-finite semantic probabilities")
    if (probabilities < 0).any():
        raise ValueError("Semantic probabilities must be non-negative")
    return normalize_sketch(probabilities @ projection.t())


def probabilities_to_simplex_sketch(probabilities: torch.Tensor) -> torch.Tensor:
    """Encode a non-negative class distribution as a unit-power analog sketch."""

    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must have shape [batch, classes>=2]")
    if not torch.isfinite(probabilities).all():
        raise ValueError("Non-finite semantic probabilities")
    if bool((probabilities < 0).any()):
        raise ValueError("Semantic probabilities must be non-negative")
    totals = probabilities.sum(dim=1, keepdim=True)
    if bool((totals <= 0).any()):
        raise ValueError("Semantic probabilities must have positive row sums")
    return normalize_sketch(probabilities / totals)


def simplex_sketch_to_probabilities(sketch: torch.Tensor) -> torch.Tensor:
    """Recover a simplex distribution from a noisy non-negative analog sketch."""

    if sketch.ndim != 2 or sketch.shape[1] < 2:
        raise ValueError("sketch must have shape [batch, classes>=2]")
    if not torch.isfinite(sketch).all():
        raise ValueError("Non-finite semantic sketch")
    positive = sketch.clamp_min(0.0)
    totals = positive.sum(dim=1, keepdim=True)
    normalized = positive / totals.clamp_min(1e-12)
    uniform = torch.full_like(normalized, 1.0 / sketch.shape[1])
    return torch.where(totals > 1e-12, normalized, uniform)


def quantize_probabilities_uniform(
    probabilities: torch.Tensor, bits_per_class: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize each class probability uniformly and renormalize the row."""

    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must have shape [batch, classes>=2]")
    if bits_per_class <= 0 or bits_per_class > 16:
        raise ValueError("bits_per_class must lie in [1, 16]")
    if not torch.isfinite(probabilities).all() or bool((probabilities < 0).any()):
        raise ValueError("probabilities must be finite and non-negative")
    levels = (1 << bits_per_class) - 1
    codes = torch.round(probabilities * levels).to(torch.int64).clamp(0, levels)
    decoded = codes.to(probabilities.dtype)
    totals = decoded.sum(dim=1, keepdim=True)
    uniform = torch.full_like(decoded, 1.0 / probabilities.shape[1])
    decoded = torch.where(totals > 0, decoded / totals.clamp_min(1.0), uniform)
    return codes, decoded


def integer_codes_to_bits(codes: torch.Tensor, bits_per_code: int) -> torch.Tensor:
    """Serialize non-negative integer codes MSB-first."""

    if codes.ndim != 2 or bits_per_code <= 0 or bits_per_code > 16:
        raise ValueError("codes must be BxD and bits_per_code must lie in [1, 16]")
    if bool((codes < 0).any()) or bool((codes >= (1 << bits_per_code)).any()):
        raise ValueError("integer code exceeds the configured bit width")
    shifts = torch.arange(
        bits_per_code - 1, -1, -1, device=codes.device, dtype=torch.int64
    )
    return ((codes.to(torch.int64).unsqueeze(-1) >> shifts) & 1).reshape(codes.shape[0], -1)


def bits_to_integer_codes(
    bits: torch.Tensor, code_count: int, bits_per_code: int
) -> torch.Tensor:
    """Deserialize MSB-first binary rows into integer codes."""

    if bits.ndim != 2 or code_count <= 0 or bits_per_code <= 0:
        raise ValueError("invalid serialized bit matrix dimensions")
    if bits.shape[1] != code_count * bits_per_code:
        raise ValueError("serialized bit count does not match code shape")
    if bool(((bits != 0) & (bits != 1)).any()):
        raise ValueError("serialized values must be binary")
    shifts = torch.arange(
        bits_per_code - 1, -1, -1, device=bits.device, dtype=torch.int64
    )
    weights = (1 << shifts).reshape(1, 1, -1)
    return (
        bits.to(torch.int64).reshape(bits.shape[0], code_count, bits_per_code) * weights
    ).sum(dim=2)


def reserved_symbol_indices(
    total_real_symbols: int, payload_real_symbols: int, *, device: torch.device | str | None = None
) -> torch.Tensor:
    """Choose deterministic, evenly spread latent positions for the semantic payload."""

    if total_real_symbols <= 0:
        raise ValueError("total_real_symbols must be positive")
    if payload_real_symbols <= 0 or payload_real_symbols >= total_real_symbols:
        raise ValueError("payload_real_symbols must lie strictly inside the latent budget")
    indices = torch.div(
        torch.arange(payload_real_symbols, dtype=torch.int64) * total_real_symbols,
        payload_real_symbols,
        rounding_mode="floor",
    )
    if torch.unique(indices).numel() != payload_real_symbols:
        raise RuntimeError("Reserved semantic payload indices are not unique")
    return indices.to(device=device)


def embed_repeated_sketch(
    latent: torch.Tensor, sketch: torch.Tensor, repetitions: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Overwrite a fixed subset of a BCHW latent with a repeated analog sketch."""

    if latent.ndim != 4:
        raise ValueError(f"Expected BCHW latent, got {tuple(latent.shape)}")
    if sketch.ndim != 2 or sketch.shape[0] != latent.shape[0]:
        raise ValueError("Sketch batch does not match latent batch")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    payload_symbols = sketch.shape[1] * repetitions
    symbols_per_sample = latent[0].numel()
    indices = reserved_symbol_indices(symbols_per_sample, payload_symbols, device=latent.device)
    flat = latent.flatten(start_dim=1).clone()
    payload = sketch.unsqueeze(-1).expand(-1, -1, repetitions).reshape(latent.shape[0], -1)
    flat[:, indices] = payload.to(dtype=flat.dtype)
    flat = flat * math.sqrt(symbols_per_sample) / flat.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return flat.view_as(latent), indices


def recover_repeated_sketch_and_erase(
    received_latent: torch.Tensor,
    sketch_dim: int,
    repetitions: int,
    indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average the analog repetition code and erase reserved locations before decoding structure."""

    if received_latent.ndim != 4:
        raise ValueError(f"Expected BCHW latent, got {tuple(received_latent.shape)}")
    if sketch_dim <= 0 or repetitions <= 0:
        raise ValueError("sketch_dim and repetitions must be positive")
    payload_symbols = sketch_dim * repetitions
    symbols_per_sample = received_latent[0].numel()
    if indices is None:
        indices = reserved_symbol_indices(
            symbols_per_sample, payload_symbols, device=received_latent.device
        )
    if indices.numel() != payload_symbols:
        raise ValueError("Reserved index count does not match semantic payload")
    flat = received_latent.flatten(start_dim=1)
    recovered = flat[:, indices].reshape(received_latent.shape[0], sketch_dim, repetitions).mean(dim=2)
    recovered = normalize_sketch(recovered)
    erased = flat.clone()
    erased[:, indices] = 0.0
    return recovered, erased.view_as(received_latent)


def semantic_payload_accounting(
    inner_channel: int, image_size: int, sketch_dim: int, repetitions: int
) -> dict[str, float | int]:
    if inner_channel <= 0 or image_size <= 0:
        raise ValueError("inner_channel and image_size must be positive")
    latent_side = image_size // 4
    total_real = 2 * inner_channel * latent_side * latent_side
    payload_real = sketch_dim * repetitions
    if payload_real >= total_real:
        raise ValueError("Semantic payload exhausts the structure latent")
    return {
        "total_real_symbols": total_real,
        "payload_real_symbols": payload_real,
        "structure_real_symbols_after_reservation": total_real - payload_real,
        "payload_fraction_of_structure": payload_real / total_real,
        "structure_fraction_after_reservation": (total_real - payload_real) / total_real,
    }
