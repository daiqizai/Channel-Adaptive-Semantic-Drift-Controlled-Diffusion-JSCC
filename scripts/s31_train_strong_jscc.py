#!/usr/bin/env python3
"""Train the preregistered S31 native exact-rate strong JSCC backbone."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
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
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import FakeData
from torchvision.utils import save_image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cadsd_jscc.datasets import FlatImageDataset  # noqa: E402
from cadsd_jscc.metrics import ms_ssim_per_sample, psnr_per_sample  # noqa: E402
from cadsd_jscc.strong_jscc import StrongJSCC, trainable_parameter_count  # noqa: E402


SCRIPT = Path(__file__).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/s31_strong_jscc_coco256_awgn.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
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


def require_sha(path: Path, expected: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != str(expected):
        raise RuntimeError(f"SHA mismatch for {path}: {actual} != {expected}")
    return path


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_transform(config: dict[str, Any], train: bool):
    size = int(config["image_size"])
    if train:
        scale = tuple(float(value) for value in config["data"]["train_crop_scale"])
        operations: list[Any] = [
            transforms.RandomResizedCrop(
                size, scale=scale, ratio=(0.75, 1.3333333333333333)
            )
        ]
        if bool(config["data"]["random_horizontal_flip"]):
            operations.append(transforms.RandomHorizontalFlip())
        operations.append(transforms.ToTensor())
        return transforms.Compose(operations)
    return transforms.Compose(
        [transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor()]
    )


def fixed_subset(dataset, size: int | None, seed: int):
    if size is None:
        return dataset
    count = min(int(size), len(dataset))
    indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(seed)
    )[:count].tolist()
    return Subset(dataset, indices)


def build_loaders(
    config: dict[str, Any], device: torch.device, dry_run: bool
) -> tuple[DataLoader, DataLoader]:
    seed = int(config["seed"])
    size = int(config["image_size"])
    if dry_run:
        train_set = FakeData(
            size=8,
            image_size=(3, size, size),
            num_classes=1,
            transform=transforms.ToTensor(),
            random_offset=seed,
        )
        val_set = FakeData(
            size=4,
            image_size=(3, size, size),
            num_classes=1,
            transform=transforms.ToTensor(),
            random_offset=seed + 1,
        )
        workers = 0
    else:
        train_set = FlatImageDataset(
            resolve(config["data"]["train_root"]), make_transform(config, True)
        )
        val_set = FlatImageDataset(
            resolve(config["data"]["val_root"]), make_transform(config, False)
        )
        train_set = fixed_subset(train_set, config["data"].get("train_subset"), seed)
        val_set = fixed_subset(val_set, config["data"].get("val_subset"), seed + 1)
        workers = int(config["training"]["num_workers"])
    common = dict(
        batch_size=int(config["training"]["batch_size"]),
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        **common,
    )
    val_loader = DataLoader(val_set, shuffle=False, **common)
    return train_loader, val_loader


def build_model(config: dict[str, Any]) -> StrongJSCC:
    model = config["model"]
    return StrongJSCC(
        image_size=int(config["image_size"]),
        latent_channels=int(model["latent_channels"]),
        stage_channels=tuple(int(value) for value in model["stage_channels"]),
        stage_blocks=tuple(int(value) for value in model["stage_blocks"]),
        condition_dim=int(model["condition_dim"]),
    )


def validate_contract(config: dict[str, Any], model: StrongJSCC) -> None:
    if config["status"] != "preregistered_before_training_output":
        raise RuntimeError("S31 config must be preregistered before output")
    if config["official_imagenette_validation_accessed"] is not False:
        raise RuntimeError("official Imagenette validation must remain sealed")
    if config["channel"]["type"] != "AWGN":
        raise RuntimeError("S31 first closure is AWGN only")
    rate = config["rate"]
    if model.real_symbols != int(rate["native_real_symbols"]):
        raise RuntimeError("model does not natively emit the frozen real-symbol count")
    expected_shape = [model.latent_channels, model.image_size // 16, model.image_size // 16]
    if expected_shape != list(map(int, rate["latent_shape"])):
        raise RuntimeError("latent shape contract mismatch")
    if rate["mask_or_padding"] != "none":
        raise RuntimeError("S31 forbids latent masking and padding")
    if int(rate["complex_channel_uses"]) != model.real_symbols // 2:
        raise RuntimeError("complex-use count mismatch")
    exact_cbr = (model.real_symbols / 2) / int(rate["source_real_dimensions"])
    if not math.isclose(exact_cbr, float(rate["exact_cbr"]), abs_tol=1e-15):
        raise RuntimeError("CBR mismatch")
    parameters = trainable_parameter_count(model)
    if not (
        int(config["model"]["minimum_trainable_parameters"])
        <= parameters
        <= int(config["model"]["maximum_trainable_parameters"])
    ):
        raise RuntimeError(f"trainable parameter contract failed: {parameters}")
    if config["training"]["loss"] != "mse":
        raise RuntimeError("S31 selection contract requires MSE-only training")


def deterministic_noise(
    shape: torch.Size, seed: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(
        device=device, dtype=dtype, non_blocking=True
    )


@torch.no_grad()
def evaluate(
    model: StrongJSCC,
    loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    sample_dir: Path,
    tag: str,
    max_batches: int | None,
) -> dict[str, Any]:
    model.eval()
    per_snr: list[dict[str, float]] = []
    base_seed = int(config["channel"]["validation_noise_seed"])
    latent_shape = (
        int(config["model"]["latent_channels"]),
        int(config["image_size"]) // 16,
        int(config["image_size"]) // 16,
    )
    for snr_index, snr in enumerate(config["channel"]["validation_snrs_db"]):
        mses: list[float] = []
        psnrs: list[float] = []
        similarities: list[float] = []
        powers: list[float] = []
        generator = torch.Generator(device="cpu").manual_seed(base_seed + 1009 * snr_index)
        for batch_index, (images_cpu, _labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images_cpu.to(device, non_blocking=True)
            noise = torch.randn(
                (len(images), *latent_shape), generator=generator, dtype=torch.float32
            ).to(device=device, dtype=images.dtype, non_blocking=True)
            output, observation = model.forward_with_observation(images, float(snr), noise)
            output = output.clamp(0.0, 1.0)
            sample_mse = (
                F.mse_loss(output, images, reduction="none")
                .flatten(start_dim=1)
                .mean(dim=1)
            )
            mses.extend(sample_mse.cpu().tolist())
            psnrs.extend(psnr_per_sample(output, images).cpu().tolist())
            similarities.extend(ms_ssim_per_sample(output, images).cpu().tolist())
            powers.extend(observation.normalized_power.cpu().tolist())
            if batch_index == 0:
                count = min(4, len(images))
                save_image(
                    torch.cat((images[:count].cpu(), output[:count].cpu())),
                    sample_dir / f"{tag}_snr_{int(snr):02d}.png",
                    nrow=count,
                )
        if not mses:
            raise RuntimeError("validation produced no samples")
        per_snr.append(
            {
                "snr_db": float(snr),
                "samples": len(mses),
                "mse": float(sum(mses) / len(mses)),
                "psnr_db": float(sum(psnrs) / len(psnrs)),
                "ms_ssim": float(sum(similarities) / len(similarities)),
                "normalized_power_mean": float(sum(powers) / len(powers)),
                "normalized_power_max_abs_error": float(
                    max(abs(value - 1.0) for value in powers)
                ),
            }
        )
    return {
        "per_snr": per_snr,
        "aggregate": {
            "mse": float(sum(row["mse"] for row in per_snr) / len(per_snr)),
            "psnr_db": float(
                sum(row["psnr_db"] for row in per_snr) / len(per_snr)
            ),
            "ms_ssim": float(
                sum(row["ms_ssim"] for row in per_snr) / len(per_snr)
            ),
            "normalized_power_max_abs_error": float(
                max(row["normalized_power_max_abs_error"] for row in per_snr)
            ),
        },
    }


def learning_rate_multiplier(
    step: int, total_steps: int, warmup_steps: int, minimum_ratio: float
) -> float:
    if step < warmup_steps:
        return max(1e-8, (step + 1) / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def train_epoch(
    model: StrongJSCC,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    config: dict[str, Any],
    device: torch.device,
    max_batches: int | None,
    snr_generator: torch.Generator,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    use_amp = bool(config["training"]["amp"] and device.type == "cuda")
    choices = torch.tensor(config["channel"]["train_snrs_db"], dtype=torch.float32)
    losses: list[float] = []
    optimizer_steps = 0
    started = time.perf_counter()
    expected_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    for batch_index, (images_cpu, _labels) in enumerate(loader):
        if batch_index >= expected_batches:
            break
        images = images_cpu.to(device, non_blocking=True)
        indices = torch.randint(
            len(choices), (len(images),), generator=snr_generator, device="cpu"
        )
        snr = choices[indices].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            output = model(images, snr)
            raw_loss = F.mse_loss(output, images)
            loss = raw_loss / accumulation
        if not torch.isfinite(raw_loss):
            raise RuntimeError(f"non-finite training loss at batch {batch_index}")
        scaler.scale(loss).backward()
        losses.append(float(raw_loss.detach().cpu()))
        should_step = (batch_index + 1) % accumulation == 0 or (
            batch_index + 1 == expected_batches
        )
        if should_step:
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["training"]["grad_clip_norm"])
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError(f"non-finite gradient norm at batch {batch_index}")
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            optimizer_steps += 1
    if not losses:
        raise RuntimeError("training produced no batches")
    return {
        "train_mse": float(sum(losses) / len(losses)),
        "train_batches": len(losses),
        "optimizer_steps": optimizer_steps,
        "epoch_seconds": time.perf_counter() - started,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }


def checkpoint_payload(
    *,
    model: StrongJSCC,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_optimizer_steps: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
    config_sha256: str,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "epoch": epoch,
        "global_optimizer_steps": global_optimizer_steps,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "metrics": metrics,
        "config": config,
        "config_sha256": config_sha256,
        "rng": {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config_sha = sha256_file(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_everything(int(config["seed"]))
    model = build_model(config)
    validate_contract(config, model)

    initialization_audit: dict[str, Any] | None = None
    initialization = config.get("initialization")
    if initialization is not None:
        if args.resume:
            raise RuntimeError("model-only initialization and --resume are mutually exclusive")
        if initialization.get("type") != "model_only_from_frozen_checkpoint":
            raise RuntimeError("unsupported S31 initialization contract")
        initialization_path = require_sha(
            resolve(initialization["checkpoint"]), initialization["checkpoint_sha256"]
        )
        initialization_checkpoint = torch.load(
            initialization_path, map_location="cpu", weights_only=False
        )
        if int(initialization_checkpoint["epoch"]) != int(
            initialization["expected_epoch"]
        ):
            raise RuntimeError("initialization checkpoint epoch mismatch")
        model.load_state_dict(initialization_checkpoint["model"], strict=True)
        initialization_audit = {
            "type": initialization["type"],
            "checkpoint": relative(initialization_path),
            "checkpoint_sha256": sha256_file(initialization_path),
            "source_experiment": initialization["source_experiment"],
            "source_epoch": int(initialization_checkpoint["epoch"]),
            "model_only": True,
            "optimizer_or_scheduler_loaded": False,
        }

    output = resolve(
        args.output_dir
        or config["outputs"]["smoke_dir" if args.dry_run else "train_dir"]
    )
    if output.exists() and not args.resume:
        raise FileExistsError(output)
    if not output.exists():
        output.mkdir(parents=True)
        (output / "checkpoints").mkdir()
        (output / "samples").mkdir()
        shutil.copy2(config_path, output / "config_snapshot.yaml")
        shutil.copy2(SCRIPT, output / SCRIPT.name)
    elif sha256_file(output / "config_snapshot.yaml") != config_sha:
        raise RuntimeError("resume config does not match output snapshot")

    model.to(device)
    training = config["training"]
    learning_rate = float(training["learning_rate"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(training["weight_decay"]),
    )
    train_loader, val_loader = build_loaders(config, device, args.dry_run)
    accumulation = int(training["gradient_accumulation_steps"])
    batches_per_epoch = len(train_loader)
    optimizer_steps_per_epoch = math.ceil(batches_per_epoch / accumulation)
    epochs = 1 if args.dry_run else int(training["epochs"])
    total_optimizer_steps = max(1, optimizer_steps_per_epoch * epochs)
    minimum_ratio = float(training["minimum_learning_rate"]) / learning_rate
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: learning_rate_multiplier(
            step,
            total_optimizer_steps,
            int(training["warmup_optimizer_steps"]),
            minimum_ratio,
        ),
    )
    use_amp = bool(training["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    snr_generator = torch.Generator(device="cpu").manual_seed(int(config["seed"]) + 17)

    start_epoch = 0
    global_optimizer_steps = 0
    best_psnr = -math.inf
    best_ms_ssim = -math.inf
    latest = output / "checkpoints" / "latest.pt"
    best = output / "checkpoints" / "best.pt"
    if args.resume:
        if not latest.is_file():
            raise FileNotFoundError(latest)
        checkpoint = torch.load(latest, map_location=device, weights_only=False)
        if checkpoint["config_sha256"] != config_sha:
            raise RuntimeError("resume checkpoint config mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_optimizer_steps = int(checkpoint["global_optimizer_steps"])
        if best.is_file():
            best_checkpoint = torch.load(best, map_location="cpu", weights_only=False)
            best_psnr = float(best_checkpoint["metrics"]["aggregate"]["psnr_db"])
            best_ms_ssim = float(best_checkpoint["metrics"]["aggregate"]["ms_ssim"])

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": relative(config_path),
        "config_sha256": config_sha,
        "script": relative(SCRIPT),
        "script_sha256": sha256_file(SCRIPT),
        "command": " ".join(sys.argv),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "trainable_parameters": trainable_parameter_count(model),
        "native_real_symbols": model.real_symbols,
        "complex_channel_uses": model.real_symbols // 2,
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "planned_optimizer_steps": total_optimizer_steps,
        "dry_run": bool(args.dry_run),
        "initialization": initialization_audit,
        "official_imagenette_validation_accessed": False,
    }
    write_json(output / "metadata.json", metadata)
    write_json(output / "STATE.json", {"status": "running", "next_epoch": start_epoch})

    history_path = output / "history.csv"
    history_fields = [
        "epoch",
        "train_mse",
        "aggregate_psnr_db",
        "aggregate_ms_ssim",
        "optimizer_steps",
        "global_optimizer_steps",
        "learning_rate",
        "epoch_seconds",
    ]
    new_history = not history_path.exists()
    history_handle = history_path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(history_handle, fieldnames=history_fields)
    if new_history:
        writer.writeheader()

    try:
        for epoch in range(start_epoch, epochs):
            train_metrics = train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                config,
                device,
                args.max_train_batches,
                snr_generator,
            )
            global_optimizer_steps += int(train_metrics["optimizer_steps"])
            validation = evaluate(
                model,
                val_loader,
                config,
                device,
                output / "samples",
                f"epoch_{epoch:03d}",
                args.max_val_batches,
            )
            aggregate = validation["aggregate"]
            if not all(
                math.isfinite(float(value))
                for value in (
                    train_metrics["train_mse"],
                    aggregate["mse"],
                    aggregate["psnr_db"],
                    aggregate["ms_ssim"],
                )
            ):
                raise RuntimeError(f"non-finite epoch metrics at epoch {epoch}")
            power_limit = float(
                config["stage_gates"]["smoke"]["normalized_power_abs_error_max"]
            )
            if aggregate["normalized_power_max_abs_error"] > power_limit:
                raise RuntimeError("channel normalization power gate failed")
            row = {
                "epoch": epoch,
                "train_mse": train_metrics["train_mse"],
                "aggregate_psnr_db": aggregate["psnr_db"],
                "aggregate_ms_ssim": aggregate["ms_ssim"],
                "optimizer_steps": train_metrics["optimizer_steps"],
                "global_optimizer_steps": global_optimizer_steps,
                "learning_rate": train_metrics["learning_rate"],
                "epoch_seconds": train_metrics["epoch_seconds"],
            }
            writer.writerow(row)
            history_handle.flush()
            payload = checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_optimizer_steps=global_optimizer_steps,
                metrics=validation,
                config=config,
                config_sha256=config_sha,
            )
            torch.save(payload, latest)
            candidate = (float(aggregate["psnr_db"]), float(aggregate["ms_ssim"]))
            if candidate > (best_psnr, best_ms_ssim):
                best_psnr, best_ms_ssim = candidate
                shutil.copy2(latest, best)
            write_json(
                output / "STATE.json",
                {
                    "status": "running",
                    "completed_epoch": epoch,
                    "next_epoch": epoch + 1,
                    "best_psnr_db": best_psnr,
                    "best_ms_ssim": best_ms_ssim,
                },
            )
            print(json.dumps({"epoch": epoch, **row, "validation": validation}, indent=2))
    except BaseException as error:
        write_json(
            output / "STATE.json",
            {
                "status": "failed_or_interrupted",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    finally:
        history_handle.close()

    if not best.is_file():
        raise RuntimeError("training completed without a best checkpoint")
    best_checkpoint = torch.load(best, map_location="cpu", weights_only=False)
    summary = {
        "status": "complete",
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_metrics": best_checkpoint["metrics"],
        "best_checkpoint": relative(best),
        "best_checkpoint_sha256": sha256_file(best),
        "metadata": metadata,
    }
    write_json(output / "summary.json", summary)
    write_json(
        output / "STATE.json",
        {"status": "complete", "best_epoch": summary["best_epoch"]},
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
