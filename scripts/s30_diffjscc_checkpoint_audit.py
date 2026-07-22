#!/usr/bin/env python3
"""Inspect the official S30 checkpoint before constructing a model instance."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(value: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(resolve(value).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("config root must be a mapping")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/s30_diffjscc_external_comparison.yaml")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = load_config(config_path)
    checkpoint = resolve(config["assets"]["checkpoint_file"])
    expected_size = int(config["assets"]["checkpoint_expected_bytes"])
    if not checkpoint.is_file() or checkpoint.stat().st_size != expected_size:
        raise RuntimeError("official DiffJSCC checkpoint is absent or incomplete")
    observed_sha = sha256_file(checkpoint)
    if observed_sha != str(config["assets"]["checkpoint_sha256"]):
        raise RuntimeError("official DiffJSCC checkpoint hash mismatch")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict) or not state:
        raise RuntimeError("checkpoint has no non-empty state dict")
    tensors = {key: value for key, value in state.items() if isinstance(value, torch.Tensor)}
    prefix_specs = {
        "blip_model": "blip_model.",
        "openclip_text": "cond_stage_model.",
        "deepjscc": "preprocess_model.",
        "controlnet": "control_model.",
        "diffusion_unet": "model.diffusion_model.",
        "vae": "first_stage_model.",
        "spatial_condition_encoder": "cond_encoder.",
    }
    prefix_summary: dict[str, Any] = {}
    for name, prefix in prefix_specs.items():
        selected = {key: value for key, value in tensors.items() if key.startswith(prefix)}
        prefix_summary[name] = {
            "prefix": prefix,
            "tensor_keys": len(selected),
            "numel": sum(value.numel() for value in selected.values()),
            "bytes": sum(value.numel() * value.element_size() for value in selected.values()),
        }
    missing_critical = [
        name
        for name in (
            "openclip_text",
            "deepjscc",
            "controlnet",
            "diffusion_unet",
            "vae",
            "spatial_condition_encoder",
        )
        if prefix_summary[name]["tensor_keys"] == 0
    ]
    blip_excluded_as_authored = prefix_summary["blip_model"]["tensor_keys"] == 0
    if not blip_excluded_as_authored:
        missing_critical.append("unexpected_embedded_blip_model")
    dtype_counts = Counter(str(value.dtype) for value in tensors.values())
    dtype_bytes = Counter()
    for value in tensors.values():
        dtype_bytes[str(value.dtype)] += value.numel() * value.element_size()
    summary = {
        "analysis_id": "ANALYSIS-S30-DIFFJSCC-CHECKPOINT-AUDIT-001",
        "status": "PASS" if not missing_critical else "FAIL",
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "bytes": checkpoint.stat().st_size,
            "sha256": observed_sha,
        },
        "state_dict": {
            "tensor_keys": len(tensors),
            "total_numel": sum(value.numel() for value in tensors.values()),
            "tensor_bytes": sum(
                value.numel() * value.element_size() for value in tensors.values()
            ),
            "dtype_tensor_counts": dict(dtype_counts),
            "dtype_bytes": dict(dtype_bytes),
            "critical_prefixes": prefix_summary,
            "missing_critical_prefixes": missing_critical,
        },
        "decision": {
            "author_blip_exclusion_confirmed": blip_excluded_as_authored,
            "exact_external_author_blip_weights_required": True,
            "external_caption_weight_substitution_allowed": False,
            "external_openclip_weight_substitution_allowed": False,
        },
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if not args.no_write:
        output = resolve(config["outputs"]["checkpoint_audit"])
        output.mkdir(parents=True, exist_ok=False)
        (output / "summary.json").write_text(rendered + "\n", encoding="utf-8")
        (output / "config_snapshot.yaml").write_text(
            config_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
    if missing_critical:
        raise RuntimeError(f"checkpoint misses critical prefixes: {missing_critical}")


if __name__ == "__main__":
    main()
