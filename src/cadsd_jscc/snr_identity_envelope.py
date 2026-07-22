"""SNR-conditioned identity envelopes for frozen latent-diffusion corrections."""

from __future__ import annotations

import math
from typing import Any

import torch

from .channel_matched_latent_diffusion import channel_alpha


def envelope_strength(
    snr_db: float,
    specification: dict[str, Any],
    *,
    noise_variance_factor_per_real: float = 0.5,
    reference_snr_db: float = 1.0,
) -> float:
    """Return a bounded correction strength for one frozen envelope specification."""

    kind = str(specification["kind"])
    if kind == "smooth_power":
        exponent = float(specification["exponent"])
        if exponent <= 0.0:
            raise ValueError("smooth envelope exponent must be positive")
        alpha = channel_alpha(float(snr_db), noise_variance_factor_per_real)
        alpha_reference = channel_alpha(
            float(reference_snr_db), noise_variance_factor_per_real
        )
        ratio = (1.0 - alpha) / (1.0 - alpha_reference)
        return float(min(1.0, max(0.0, ratio)) ** exponent)
    if kind == "hard_cutoff":
        full_through = float(specification["full_through_snr_db"])
        identity_from = float(specification["identity_from_snr_db"])
        if identity_from <= full_through:
            raise ValueError("hard cutoff identity point must exceed its full-strength point")
        if float(snr_db) <= full_through:
            return 1.0
        if float(snr_db) >= identity_from:
            return 0.0
        return float((identity_from - float(snr_db)) / (identity_from - full_through))
    raise ValueError(f"unknown envelope kind: {kind}")


def apply_correction_envelope(
    received: torch.Tensor, diffusion: torch.Tensor, strength: float
) -> torch.Tensor:
    if received.shape != diffusion.shape:
        raise ValueError("received and diffusion codewords must have identical shapes")
    if not math.isfinite(float(strength)) or not 0.0 <= float(strength) <= 1.0:
        raise ValueError("correction strength must lie in [0, 1]")
    return received + float(strength) * (diffusion - received)


def select_envelope_policy(
    summaries: list[dict[str, Any]], *, nonnegative_tolerance_db: float
) -> dict[str, Any]:
    """Apply the preregistered lexicographic policy selector."""

    if not summaries:
        raise ValueError("cannot select from an empty envelope family")
    ranked: list[tuple[tuple[float, float, float, int], dict[str, Any]]] = []
    for index, item in enumerate(summaries):
        per_snr = item.get("per_snr")
        if not isinstance(per_snr, list) or not per_snr:
            raise ValueError("each candidate must provide per-SNR summaries")
        nonnegative = sum(
            float(row["psnr_delta_vs_b0"]) >= -float(nonnegative_tolerance_db)
            for row in per_snr
        )
        enriched = {**item, "nonnegative_snr_count": nonnegative}
        key = (
            -float(nonnegative),
            -float(item["mean_psnr_delta_vs_b0"]),
            float(item["mean_lpips_delta_vs_b0"]),
            index,
        )
        ranked.append((key, enriched))
    ranked.sort(key=lambda value: value[0])
    return ranked[0][1]
