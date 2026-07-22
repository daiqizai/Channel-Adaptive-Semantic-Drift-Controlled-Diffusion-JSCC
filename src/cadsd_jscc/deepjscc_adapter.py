from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch


def deepjscc_encode(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Encode images into the normalized channel-input latent."""
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise TypeError("DeepJSCC model has no encoder")
    return encoder(images)


def deepjscc_transmit(model: torch.nn.Module, latent: torch.Tensor) -> torch.Tensor:
    """Apply the configured channel and return the receiver-visible latent."""
    channel = getattr(model, "channel", None)
    return latent if channel is None else channel(latent)


def deepjscc_decode(model: torch.nn.Module, received: torch.Tensor) -> torch.Tensor:
    """Decode a receiver-visible channel latent."""
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        raise TypeError("DeepJSCC model has no decoder")
    return decoder(received)


def deepjscc_forward_with_latents(
    model: torch.nn.Module, images: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return reconstruction, transmitted latent, and received latent."""
    transmitted = deepjscc_encode(model, images)
    received = deepjscc_transmit(model, transmitted)
    reconstruction = deepjscc_decode(model, received)
    return reconstruction, transmitted, received


def received_latent_consistency_loss(
    model: torch.nn.Module,
    candidate: torch.Tensor,
    received: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalized channel-latent data term for posterior-consistent restoration."""
    return received_latent_consistency_per_sample(
        model, candidate, received, valid_mask=valid_mask
    ).mean()


def received_latent_consistency_per_sample(
    model: torch.nn.Module,
    candidate: torch.Tensor,
    received: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-image normalized channel-latent data term.

    ``valid_mask`` excludes channel positions reserved for an in-budget side
    payload.  It may be a flat per-sample mask, a CHW mask, or a BCHW mask.
    The default keeps the historical full-latent definition unchanged.
    """
    encoded = deepjscc_encode(model, candidate)
    if encoded.shape != received.shape:
        raise ValueError(
            f"Encoded candidate shape {tuple(encoded.shape)} != received {tuple(received.shape)}"
        )
    received_flat = received.detach().flatten(start_dim=1)
    error_flat = (encoded - received).square().flatten(start_dim=1)
    if valid_mask is None:
        scale = received_flat.square().mean(dim=1).clamp_min(1e-8)
        error = error_flat.mean(dim=1)
        return error / scale

    mask = torch.as_tensor(valid_mask, device=received.device, dtype=torch.bool)
    symbols_per_sample = received_flat.shape[1]
    if mask.ndim == 1 and mask.numel() == symbols_per_sample:
        mask_flat = mask.reshape(1, -1).expand(received.shape[0], -1)
    elif tuple(mask.shape) == tuple(received.shape[1:]):
        mask_flat = mask.reshape(1, -1).expand(received.shape[0], -1)
    elif tuple(mask.shape) == tuple(received.shape):
        mask_flat = mask.flatten(start_dim=1)
    else:
        raise ValueError(
            f"valid_mask shape {tuple(mask.shape)} is incompatible with "
            f"received shape {tuple(received.shape)}"
        )
    counts = mask_flat.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("valid_mask must retain at least one symbol per sample")
    counts_float = counts.to(dtype=received.dtype)
    scale = (
        (received_flat.square() * mask_flat).sum(dim=1) / counts_float
    ).clamp_min(1e-8)
    error = (error_flat * mask_flat).sum(dim=1) / counts_float
    return error / scale


def extract_deepjscc_state_dict(checkpoint: object) -> dict:
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model_state = checkpoint["model"]
        if isinstance(model_state, dict):
            return model_state
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported DeepJSCC checkpoint type: {type(checkpoint)!r}")


def build_deepjscc_model(
    repo_root: str | Path,
    inner_channel: int,
    channel: str,
    snr: float,
) -> torch.nn.Module:
    repo_root = Path(repo_root).resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"DeepJSCC repo not found: {repo_root}")

    sys.path.insert(0, str(repo_root))
    try:
        model_module = importlib.import_module("model")
        return model_module.DeepJSCC(c=inner_channel, channel_type=channel, snr=snr)
    finally:
        try:
            sys.path.remove(str(repo_root))
        except ValueError:
            pass


def load_deepjscc_model(
    repo_root: str | Path,
    checkpoint_path: str | Path,
    inner_channel: int,
    channel: str,
    snr: float,
    device: torch.device,
) -> torch.nn.Module:
    repo_root = Path(repo_root).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"DeepJSCC repo not found: {repo_root}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"DeepJSCC checkpoint not found: {checkpoint_path}")

    model = build_deepjscc_model(
        repo_root=repo_root,
        inner_channel=inner_channel,
        channel=channel,
        snr=snr,
    )
    state_dict = extract_deepjscc_state_dict(torch.load(checkpoint_path, map_location=device))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
