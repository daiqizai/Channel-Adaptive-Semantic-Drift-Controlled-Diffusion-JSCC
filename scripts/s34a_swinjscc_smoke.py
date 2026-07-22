#!/usr/bin/env python3
"""Run the preregistered one-microbatch SwinJSCC S34A timing smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cadsd_jscc.datasets import FlatImageDataset  # noqa: E402
from cadsd_jscc.swinjscc_adapter import (  # noqa: E402
    OFFICIAL_SOURCE,
    OfficialSwinJSCCSA,
    trainable_parameter_count,
)


SCRIPT = Path(__file__).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/s34a_swinjscc_equal_rate_comparison.yaml"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            result[str(path.relative_to(root))] = sha256_file(path)
    return result


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_input(config: dict[str, Any], microbatch: int) -> torch.Tensor:
    size = int(config["rate"]["image_shape"][1])
    scale = tuple(float(value) for value in config["data"]["train_crop_scale"])
    operations: list[Any] = [
        transforms.RandomResizedCrop(
            size, scale=scale, ratio=(0.75, 1.3333333333333333)
        )
    ]
    if bool(config["data"]["random_horizontal_flip"]):
        operations.append(transforms.RandomHorizontalFlip())
    operations.append(transforms.ToTensor())
    dataset = FlatImageDataset(
        resolve(config["data"]["train_root"]), transforms.Compose(operations)
    )
    loader = DataLoader(dataset, batch_size=microbatch, shuffle=False, num_workers=0)
    images, _ = next(iter(loader))
    if len(images) != microbatch:
        raise RuntimeError("smoke batch is smaller than the frozen microbatch")
    return images


def build_model(arm: dict[str, Any]) -> OfficialSwinJSCCSA:
    return OfficialSwinJSCCSA(
        image_size=256,
        latent_channels=64,
        encoder_depths=tuple(int(value) for value in arm["encoder_depths"]),
        decoder_depths=tuple(int(value) for value in arm["decoder_depths"]),
    )


def smoke_arm(
    *,
    name: str,
    arm: dict[str, Any],
    images_cpu: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
    output: Path,
) -> dict[str, Any]:
    seed_everything(int(config["seed"]))
    build_started = time.perf_counter()
    model = build_model(arm)
    build_seconds = time.perf_counter() - build_started
    parameters = trainable_parameter_count(model)
    expected_parameters = int(arm["trainable_parameters_static_audit"])
    if parameters != expected_parameters:
        raise RuntimeError(f"{name} parameter mismatch: {parameters} != {expected_parameters}")
    if model.real_symbols != int(config["rate"]["native_real_symbols"]):
        raise RuntimeError(f"{name} exact-rate contract failed")

    model.to(device)
    images = images_cpu.to(device)
    snr_choices = torch.tensor(config["channel"]["train_snrs_db"], dtype=torch.float32)
    snr = snr_choices[
        torch.arange(len(images), dtype=torch.long) % len(snr_choices)
    ].to(device)
    generator = torch.Generator(device="cpu").manual_seed(int(config["seed"]) + 991)
    noise = torch.randn(
        (len(images), 256, 64), generator=generator, dtype=torch.float32
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["equal_budget_training"]["main"]["learning_rate"]),
        weight_decay=float(config["equal_budget_training"]["main"]["weight_decay"]),
    )
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    reconstruction, observation = model.forward_with_observation(images, snr, noise)
    forward_finished = time.perf_counter()
    loss = F.mse_loss(reconstruction, images)
    if not torch.isfinite(loss):
        raise RuntimeError(f"{name} produced non-finite loss")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), float(config["equal_budget_training"]["grad_clip_norm"])
    )
    if not torch.isfinite(gradient_norm):
        raise RuntimeError(f"{name} produced non-finite gradients")
    backward_finished = time.perf_counter()
    optimizer.step()
    torch.cuda.synchronize(device)
    finished = time.perf_counter()

    expected_latent = (len(images), 256, 64)
    if tuple(observation.latent.shape) != expected_latent:
        raise RuntimeError(
            f"{name} latent shape mismatch: {tuple(observation.latent.shape)} != {expected_latent}"
        )
    if tuple(reconstruction.shape) != tuple(images.shape):
        raise RuntimeError(f"{name} reconstruction shape mismatch")
    power_error = float((observation.normalized_power - 1.0).abs().max().detach().cpu())
    if power_error > float(config["smoke"]["normalized_power_abs_error_max"]):
        raise RuntimeError(f"{name} normalized-power gate failed: {power_error}")

    checkpoint = output / f"{name}_roundtrip.pt"
    torch.save({"model": model.state_dict()}, checkpoint)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model"], strict=True)
    checkpoint_sha = sha256_file(checkpoint)
    checkpoint.unlink()

    microbatch = len(images)
    accumulation = math.ceil(
        int(config["equal_budget_training"]["effective_batch_size"]) / microbatch
    )
    timed_seconds = finished - started
    return {
        "arm": name,
        "status": "PASS",
        "parameters": parameters,
        "real_symbols_per_image": model.real_symbols,
        "latent_shape": list(observation.latent.shape),
        "output_shape": list(reconstruction.shape),
        "microbatch_size": microbatch,
        "gradient_accumulation_for_effective_batch_32": accumulation,
        "loss_mse": float(loss.detach().cpu()),
        "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
        "normalized_power_max_abs_error": power_error,
        "build_seconds": build_seconds,
        "forward_seconds": forward_finished - started,
        "backward_seconds": backward_finished - forward_finished,
        "optimizer_step_and_sync_seconds": finished - backward_finished,
        "train_microbatch_seconds": timed_seconds,
        "estimated_optimizer_step_seconds_linear_accumulation": timed_seconds * accumulation,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
        "checkpoint_roundtrip_sha256_before_delete": checkpoint_sha,
    }


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["status"] != "confirmed_smoke_only":
        raise RuntimeError("S34A config is not in confirmed smoke-only state")
    if config["smoke"]["run_allowed"] is not True:
        raise RuntimeError("S34A smoke is not authorized")
    if config["formal_training"]["run_allowed"] is not False:
        raise RuntimeError("formal training must remain gated until the smoke report")
    if config["sealed"]["official_imagenette_validation"] is not True:
        raise RuntimeError("official Imagenette validation must remain sealed")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the timing smoke requires CUDA")

    output = resolve(args.output_dir or config["smoke"]["output_dir"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    shutil.copy2(config_path, output / "config_snapshot.yaml")
    shutil.copy2(SCRIPT, output / SCRIPT.name)

    microbatch = int(config["smoke"]["microbatch_size"])
    seed_everything(int(config["seed"]))
    images = build_input(config, microbatch)
    results = []
    for name in config["smoke"]["arms"]:
        results.append(
            smoke_arm(
                name=name,
                arm=config["arms_confirmed"][name],
                images_cpu=images,
                config=config,
                device=device,
                output=output,
            )
        )
        torch.cuda.empty_cache()

    payload = {
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": relative(config_path),
        "config_sha256": sha256_file(config_path),
        "script": relative(SCRIPT),
        "script_sha256": sha256_file(SCRIPT),
        "official_source": relative(OFFICIAL_SOURCE),
        "official_commit_declared": config["source"]["audited_commit"],
        "official_source_manifest": source_manifest(OFFICIAL_SOURCE),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "official_imagenette_validation_accessed": False,
        "results": results,
    }
    write_json(output / "smoke_result.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
