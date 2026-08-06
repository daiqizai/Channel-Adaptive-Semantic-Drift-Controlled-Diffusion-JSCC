#!/usr/bin/env python3
"""Measure frozen S33 inference cost under the S34D contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from cadsd_jscc.external_common import canonical_standard_normal  # noqa: E402
from cadsd_jscc.strong_jscc import trainable_parameter_count  # noqa: E402
from s30_diffjscc_preflight import sha256_file  # noqa: E402
from s32_strong_jscc_external_comparison import build_model, load_population  # noqa: E402


SCRIPT = Path(__file__).resolve()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_yaml(value: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(resolve(value).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(value)
    return payload


def require_sha(value: str | Path, expected: str) -> Path:
    path = resolve(value)
    if not path.is_file() or sha256_file(path) != str(expected):
        raise RuntimeError(f"missing or hash-mismatched input: {path}")
    return path


def gpu_inventory() -> dict[str, str]:
    fields = "uuid,name,driver_version"
    line = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
        text=True,
    ).strip().splitlines()[0]
    uuid, name, driver = (part.strip() for part in line.split(",", 2))
    return {"uuid": uuid, "name": name, "driver_version": driver}


def sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def timed(device: torch.device, fn: Callable[[], Any]) -> tuple[Any, float]:
    sync(device)
    started = time.perf_counter()
    value = fn()
    sync(device)
    return value, (time.perf_counter() - started) * 1000.0


def profile_flops(fn: Callable[[], Any]) -> int:
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_flops=True,
        record_shapes=True,
    ) as prof:
        fn()
        torch.cuda.synchronize()
    return int(sum(int(event.flops or 0) for event in prof.key_averages()))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "std": float(array.std(ddof=1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional non-existing supplementary output below the S34D root.",
    )
    args = parser.parse_args()
    config_path = resolve("configs/s34d_generative_inference_cost.yaml")
    config = load_yaml(config_path)
    if config["status"] != "preregistered_and_authorized_before_measurement":
        raise RuntimeError("S34D is not authorized")
    if any(
        (
            config["protocol"]["new_training"],
            config["protocol"]["network_download"],
            config["protocol"]["official_imagenette_validation_accessed"],
        )
    ):
        raise RuntimeError("measurement-only contract changed")
    inventory = gpu_inventory()
    if inventory["uuid"] != config["hardware"]["required_gpu_uuid"]:
        raise RuntimeError(f"wrong GPU: {inventory}")
    device = torch.device(config["hardware"]["device"])
    torch.backends.cudnn.benchmark = bool(config["hardware"]["cudnn_benchmark"])

    output = (
        resolve(args.output_dir)
        if args.output_dir is not None
        else resolve(config["outputs"]["s33"])
    )
    root_output = resolve(config["outputs"]["root"])
    if root_output not in output.parents:
        raise RuntimeError("supplementary output must remain below the S34D root")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    shutil.copy2(config_path, output / "config_snapshot.yaml")
    shutil.copy2(SCRIPT, output / SCRIPT.name)

    checkpoint_path = require_sha(
        config["inputs"]["s33_checkpoint"], config["inputs"]["s33_checkpoint_sha256"]
    )
    s33_config_path = require_sha(
        config["inputs"]["s33_config"], config["inputs"]["s33_config_sha256"]
    )
    s33_config = load_yaml(s33_config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint).to(device).eval().requires_grad_(False)
    if model.real_symbols != int(config["fairness_boundaries"]["s33_real_symbols"]):
        raise RuntimeError("S33 rate changed")
    samples, _classes = load_population(s33_config)
    transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()]
    )
    host_images = [Image.open(item["path"]).convert("RGB") for item in samples]
    snrs = list(map(float, config["latency_contract"]["timed_snrs_db"]))
    timed_count = int(config["latency_contract"]["timed_source_images"])
    latency_indices = list(range(timed_count))
    latent_shape = (1, model.latent_channels, 16, 16)

    @torch.inference_mode()
    def run_once(index: int, snr: float, collect: bool) -> dict[str, Any]:
        item = samples[index]
        full_noise = canonical_standard_normal(
            int(config["quality_curve"]["channel_seeds"][0]),
            str(item["sample_id"]),
            snr,
            19712,
        )
        noise = full_noise[: model.real_symbols].reshape(latent_shape)
        sync(device)
        wall_started = time.perf_counter()
        image, preprocess_ms = timed(
            device,
            lambda: transform(host_images[index]).unsqueeze(0).to(device),
        )
        latent, encoder_ms = timed(device, lambda: model.encode(image, snr))

        def channel_forward() -> tuple[torch.Tensor, torch.Tensor]:
            transmitted, power = model.normalize_channel_input(latent)
            received = model.transmit(transmitted, snr, noise)
            return received, power

        (received, _power), channel_ms = timed(device, channel_forward)
        reconstruction, decoder_ms = timed(device, lambda: model.decode(received, snr))
        host_output, postprocess_ms = timed(
            device,
            lambda: reconstruction.clamp(0, 1).mul(255).to(torch.uint8).cpu(),
        )
        wall_ms = (time.perf_counter() - wall_started) * 1000.0
        if tuple(host_output.shape) != (1, 3, 256, 256):
            raise RuntimeError("S33 output shape changed")
        if not collect:
            return {}
        component_sum = (
            preprocess_ms + encoder_ms + channel_ms + decoder_ms + postprocess_ms
        )
        return {
            "sample_id": item["sample_id"],
            "snr_db": snr,
            "batch_size_source_images": 1,
            "receiver_wall_ms": wall_ms,
            "preprocess_h2d_ms": preprocess_ms,
            "encoder_ms": encoder_ms,
            "channel_normalize_awgn_ms": channel_ms,
            "decoder_ms": decoder_ms,
            "postprocess_d2h_ms": postprocess_ms,
            "component_sum_ms": component_sum,
            "wall_minus_component_sum_ms": wall_ms - component_sum,
        }

    warmup_keys: list[tuple[int, float]] = []
    for index in range(int(config["latency_contract"]["warmup_source_keys"])):
        warmup_keys.append((index, snrs[index % len(snrs)]))
    for index, snr in warmup_keys:
        run_once(index, snr, False)

    rows = [
        run_once(index, snr, True)
        for snr in snrs
        for index in latency_indices
    ]
    write_csv(output / "latency_rows.csv", rows)

    # Profile one representative forward after warmup. Channel elementwise FLOPs are
    # mostly unsupported by torch.profiler and therefore intentionally remain zero/lower-bound.
    rep_index = 0
    rep_snr = float(config["flops_contract"]["representative_snr_db"])
    rep_image = transform(host_images[rep_index]).unsqueeze(0).to(device)
    rep_noise = canonical_standard_normal(
        int(config["quality_curve"]["channel_seeds"][0]),
        str(samples[rep_index]["sample_id"]),
        rep_snr,
        19712,
    )[: model.real_symbols].reshape(latent_shape)
    with torch.inference_mode():
        rep_latent = model.encode(rep_image, rep_snr)
        rep_transmitted, _ = model.normalize_channel_input(rep_latent)
        rep_received = model.transmit(rep_transmitted, rep_snr, rep_noise)
        encoder_flops = profile_flops(lambda: model.encode(rep_image, rep_snr))
        channel_flops = profile_flops(
            lambda: model.transmit(
                model.normalize_channel_input(rep_latent)[0], rep_snr, rep_noise
            )
        )
        decoder_flops = profile_flops(lambda: model.decode(rep_received, rep_snr))

    component_fields = [
        "preprocess_h2d_ms",
        "encoder_ms",
        "channel_normalize_awgn_ms",
        "decoder_ms",
        "postprocess_d2h_ms",
        "receiver_wall_ms",
    ]
    summary = {
        "status": "PASS",
        "method": "S33 strong",
        "hardware": inventory,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "batch_size_source_images": 1,
        "latency_rows": len(rows),
        "latency": {
            field: summarize([float(row[field]) for row in rows])
            for field in component_fields
        },
        "parameters": {
            "unique_live_parameters": int(
                sum(parameter.numel() for parameter in model.parameters())
            ),
            "trainable_parameters_in_checkpoint_role": int(
                sum(parameter.numel() for parameter in model.parameters())
            ),
            "encoder": int(sum(p.numel() for p in model.encoder.parameters())),
            "decoder": int(sum(p.numel() for p in model.decoder.parameters())),
            "shared_snr_embedding": int(
                sum(p.numel() for p in model.condition_embedding.parameters())
            ),
        },
        "profiled_flops_lower_bound": {
            "encoder": encoder_flops,
            "channel": channel_flops,
            "decoder": decoder_flops,
            "total": encoder_flops + channel_flops + decoder_flops,
            "coverage": config["flops_contract"]["definition"],
        },
        "inputs": {
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "config_sha256": sha256_file(config_path),
            "script_sha256": sha256_file(SCRIPT),
        },
        "new_training": False,
        "network_download": False,
        "official_imagenette_validation_accessed": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "STATE.json").write_text(
        json.dumps({"status": "complete", "rows": len(rows)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
