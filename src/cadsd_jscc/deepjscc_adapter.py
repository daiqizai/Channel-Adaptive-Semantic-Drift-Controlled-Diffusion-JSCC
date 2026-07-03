from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch


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
