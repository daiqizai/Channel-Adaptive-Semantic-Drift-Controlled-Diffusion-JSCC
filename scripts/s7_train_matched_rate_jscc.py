#!/usr/bin/env python3
"""Warm-start and fine-tune matched-rate RGB-main or structural DeepJSCC arms."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cadsd_jscc.datasets import FlatImageDataset  # noqa: E402
from cadsd_jscc.deepjscc_adapter import build_deepjscc_model, extract_deepjscc_state_dict  # noqa: E402
from cadsd_jscc.metrics import psnr_per_sample, ssim_per_sample  # noqa: E402
from cadsd_jscc.structure import structure_rgb  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
ARMS = {"main", "structure"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/s7_matched_rate_jscc_pilot_coco256_awgn.yaml")
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {value}")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_transforms(image_size: int, train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size, scale=(0.6, 1.0), ratio=(0.75, 1.3333333333)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )


def maybe_subset(dataset, size: int | None, seed: int):
    if size is None:
        return dataset
    size = min(int(size), len(dataset))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
    return Subset(dataset, indices)


def build_loaders(
    config: dict[str, Any], args: argparse.Namespace, device: torch.device
) -> tuple[DataLoader, DataLoader]:
    image_size = int(config["image_size"])
    seed = int(config["seed"])
    if args.dry_run:
        train_dataset = FakeData(
            size=16,
            image_size=(3, image_size, image_size),
            num_classes=1,
            transform=transforms.ToTensor(),
            random_offset=seed,
        )
        val_dataset = FakeData(
            size=8,
            image_size=(3, image_size, image_size),
            num_classes=1,
            transform=transforms.ToTensor(),
            random_offset=seed + 1,
        )
    else:
        train_dataset = FlatImageDataset(
            resolve(config["data"]["train_root"]),
            transform=make_transforms(image_size, train=True),
        )
        val_dataset = FlatImageDataset(
            resolve(config["data"]["val_root"]),
            transform=make_transforms(image_size, train=False),
        )
        train_dataset = maybe_subset(train_dataset, config["data"].get("train_subset"), seed)
        val_dataset = maybe_subset(val_dataset, config["data"].get("val_subset"), seed + 1)
    batch_size = int(config["training"]["batch_size"])
    workers = 0 if args.dry_run else int(config["training"]["num_workers"])
    generator = torch.Generator().manual_seed(seed + (0 if args.arm == "main" else 1000))
    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    return (
        DataLoader(train_dataset, shuffle=True, generator=generator, **common),
        DataLoader(val_dataset, shuffle=False, **common),
    )


def representation(images: torch.Tensor, arm: str, config: dict[str, Any]) -> torch.Tensor:
    if arm == "main":
        return images
    if arm == "structure":
        return structure_rgb(
            images, third_channel=str(config["representation"]["third_channel"])
        )
    raise ValueError(f"Unknown arm: {arm}")


def validate_rate_contract(config: dict[str, Any]) -> dict[str, Any]:
    denominator = int(config["protocol"]["cbr_denominator"])
    rate = config["rate"]
    main_c = int(rate["main_inner_channel"])
    structure_c = int(rate["structure_inner_channel"])
    total_c = int(rate["total_inner_channel"])
    reference_c = int(rate["reference_inner_channel"])
    if main_c + structure_c != total_c or total_c != reference_c:
        raise RuntimeError(
            f"Invalid channel budget: {main_c}+{structure_c}!={total_c}!={reference_c}"
        )
    expected = {
        "main_cbr": main_c / denominator,
        "structure_cbr": structure_c / denominator,
        "total_cbr": total_c / denominator,
        "reference_cbr": reference_c / denominator,
    }
    for key, value in expected.items():
        if not math.isclose(float(rate[key]), value, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"Rate contract mismatch for {key}: config={rate[key]}, exact={value}")
    return {"denominator": denominator, "channels": {"main": main_c, "structure": structure_c, "total": total_c}, **expected}


def latent_importance(state: dict[str, torch.Tensor]) -> torch.Tensor:
    encoder = state["encoder.conv5.conv.weight"].float().flatten(1).norm(dim=1)
    decoder = state["decoder.tconv1.transconv.weight"].float().flatten(1).norm(dim=1)
    if encoder.shape != decoder.shape:
        raise RuntimeError(f"Encoder/decoder latent shapes differ: {encoder.shape} vs {decoder.shape}")
    return torch.sqrt(encoder * decoder)


def select_latent_channels(
    state: dict[str, torch.Tensor], target_inner_channel: int
) -> tuple[list[int], list[float]]:
    importance = latent_importance(state)
    count = 2 * int(target_inner_channel)
    if count <= 0 or count > len(importance):
        raise ValueError(f"Cannot select {count} latent real channels from {len(importance)}")
    indices = sorted(int(value) for value in torch.topk(importance, k=count).indices.tolist())
    return indices, [float(importance[index].item()) for index in indices]


def warm_start_pruned_model(
    model: torch.nn.Module,
    source_checkpoint: Path,
    target_inner_channel: int,
) -> dict[str, Any]:
    checkpoint = torch.load(source_checkpoint, map_location="cpu")
    source_state = extract_deepjscc_state_dict(checkpoint)
    indices, selected_importance = select_latent_channels(source_state, target_inner_channel)
    target_state = model.state_dict()
    sliced_keys = {
        "encoder.conv5.conv.weight": 0,
        "encoder.conv5.conv.bias": 0,
        "decoder.tconv1.transconv.weight": 0,
    }
    loaded: dict[str, torch.Tensor] = {}
    for key, target_value in target_state.items():
        if key not in source_state:
            raise RuntimeError(f"Warm-start source is missing key: {key}")
        source_value = source_state[key]
        if key in sliced_keys:
            source_value = source_value[indices]
        if source_value.shape != target_value.shape:
            raise RuntimeError(
                f"Warm-start shape mismatch for {key}: source={tuple(source_value.shape)}, "
                f"target={tuple(target_value.shape)}"
            )
        loaded[key] = source_value
    model.load_state_dict(loaded, strict=True)
    source_config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    return {
        "source_checkpoint": relative(source_checkpoint),
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
        "source_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "source_inner_channel": None
        if not isinstance(source_config, dict)
        else source_config.get("inner_channel"),
        "selection_method": "encoder_decoder_geometric_l2",
        "selected_real_latent_channel_indices": indices,
        "selected_importance": selected_importance,
        "num_selected_real_channels": len(indices),
    }


def finite_metrics(metrics: dict[str, Any]) -> bool:
    return all(math.isfinite(float(value)) for value in metrics.values())


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    arm: str,
    config: dict[str, Any],
    device: torch.device,
    max_batches: int | None,
    sample_path: Path | None,
) -> dict[str, float]:
    model.eval()
    mse_values: list[float] = []
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    first = True
    for batch_index, (images_cpu, _labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images_cpu.to(device, non_blocking=True)
        target = representation(images, arm, config)
        output = model(target).clamp(0.0, 1.0)
        mse = F.mse_loss(output, target, reduction="none").flatten(start_dim=1).mean(dim=1)
        mse_values.extend(float(value) for value in mse.cpu().tolist())
        psnr_values.extend(float(value) for value in psnr_per_sample(output, target).cpu().tolist())
        ssim_values.extend(float(value) for value in ssim_per_sample(output, target).cpu().tolist())
        if first and sample_path is not None:
            count = min(4, len(target))
            save_image(
                torch.cat([target[:count].cpu(), output[:count].cpu()], dim=0),
                sample_path,
                nrow=count,
            )
            first = False
    if not mse_values:
        raise RuntimeError("Validation produced no samples")
    return {
        "mse": float(sum(mse_values) / len(mse_values)),
        "psnr_db": float(sum(psnr_values) / len(psnr_values)),
        "ssim": float(sum(ssim_values) / len(ssim_values)),
    }


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    arm: str,
    config: dict[str, Any],
    device: torch.device,
    max_batches: int | None,
) -> dict[str, float]:
    model.train()
    use_amp = bool(config["training"]["amp"] and device.type == "cuda")
    losses: list[float] = []
    started = time.perf_counter()
    for batch_index, (images_cpu, _labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images_cpu.to(device, non_blocking=True)
        target = representation(images, arm, config)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            output = model(target).clamp(0.0, 1.0)
            loss = F.mse_loss(output, target)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at batch {batch_index}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["training"]["grad_clip_norm"])
        )
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu().item()))
    if not losses:
        raise RuntimeError("Training produced no batches")
    return {
        "train_mse": float(sum(losses) / len(losses)),
        "epoch_seconds": float(time.perf_counter() - started),
        "train_batches": len(losses),
    }


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    config: dict[str, Any],
    arm: str,
    inner_channel: int,
    metrics: dict[str, Any],
    warm_start: dict[str, Any],
    rate_contract: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "config": config,
        "arm": arm,
        "representation": config["representation"][arm],
        "inner_channel": inner_channel,
        "actual_cbr": inner_channel / int(config["protocol"]["cbr_denominator"]),
        "metrics": metrics,
        "warm_start": warm_start,
        "rate_contract": rate_contract,
        "official_val_accessed": False,
    }


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Config root must be a mapping: {config_path}")
    rate_contract = validate_rate_contract(config)
    device = resolve_device(args.device)
    seed_everything(int(config["seed"]) + (0 if args.arm == "main" else 1000))
    train_loader, val_loader = build_loaders(config, args, device)
    inner_channel = int(config["rate"][f"{args.arm}_inner_channel"])
    model = build_deepjscc_model(
        repo_root=resolve(config["baseline"]["repo"]),
        inner_channel=inner_channel,
        channel=str(config["channel"]),
        snr=float(config["snr_db"]),
    ).to(device)
    warm_start = warm_start_pruned_model(
        model, resolve(config["protocol"]["source_checkpoint"]), inner_channel
    )
    plan = {
        "analysis": config["experiment_id"],
        "arm": args.arm,
        "representation": config["representation"][args.arm],
        "inner_channel": inner_channel,
        "actual_cbr": inner_channel / int(config["protocol"]["cbr_denominator"]),
        "rate_contract": rate_contract,
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
        "device": str(device),
        "warm_start": warm_start,
        "official_val_accessed": False,
    }
    if args.dry_run:
        images = next(iter(train_loader))[0].to(device)
        target = representation(images, args.arm, config)
        with torch.no_grad():
            output = model(target)
        plan.update(
            {
                "dry_run": True,
                "input_shape": list(images.shape),
                "target_shape": list(target.shape),
                "output_shape": list(output.shape),
                "finite": bool(torch.isfinite(output).all().item()),
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            }
        )
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return

    output_dir = resolve(
        args.output_dir or config["outputs"][f"{args.arm}_train_dir"]
    )
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Output exists; use --resume only for this exact run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    sample_dir = output_dir / "samples"
    checkpoint_dir.mkdir(exist_ok=True)
    sample_dir.mkdir(exist_ok=True)
    if not args.resume:
        shutil.copy2(config_path, output_dir / "config.yaml")
        shutil.copy2(SCRIPT_PATH, output_dir / SCRIPT_PATH.name)
        save_json(output_dir / "run_plan.json", plan)

    arm_training = config["training"][args.arm]
    epochs = int(args.epochs if args.epochs is not None else arm_training["epochs"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(arm_training["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(enabled=bool(config["training"]["amp"] and device.type == "cuda"))
    latest_path = checkpoint_dir / "latest.pt"
    best_path = checkpoint_dir / "best.pt"
    start_epoch = 0
    best_mse = math.inf
    history_path = output_dir / "history.csv"
    if args.resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint is missing: {latest_path}")
        resumed = torch.load(latest_path, map_location=device)
        if resumed.get("arm") != args.arm or int(resumed.get("inner_channel", -1)) != inner_channel:
            raise RuntimeError("Resume checkpoint arm/rate mismatch")
        model.load_state_dict(resumed["model"], strict=True)
        optimizer.load_state_dict(resumed["optimizer"])
        scaler.load_state_dict(resumed["scaler"])
        start_epoch = int(resumed["epoch"]) + 1
        if best_path.is_file():
            best_mse = float(torch.load(best_path, map_location="cpu")["metrics"]["val_mse"])

    fieldnames = [
        "epoch",
        "train_mse",
        "val_mse",
        "val_psnr_db",
        "val_ssim",
        "epoch_seconds",
        "train_batches",
    ]
    history_exists = history_path.exists() and history_path.stat().st_size > 0
    mode = "a" if args.resume else "w"
    with history_path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not history_exists:
            writer.writeheader()
        if not args.resume:
            initial = evaluate(
                model,
                val_loader,
                args.arm,
                config,
                device,
                args.max_val_batches,
                sample_dir / "initial.png",
            )
            if not finite_metrics(initial):
                raise RuntimeError(f"Warm-start validation is non-finite: {initial}")
            initial_row = {
                "epoch": -1,
                "train_mse": "",
                "val_mse": initial["mse"],
                "val_psnr_db": initial["psnr_db"],
                "val_ssim": initial["ssim"],
                "epoch_seconds": 0.0,
                "train_batches": 0,
            }
            writer.writerow(initial_row)
            handle.flush()
            best_mse = float(initial["mse"])
            torch.save(
                checkpoint_payload(
                    model,
                    optimizer,
                    scaler,
                    -1,
                    config,
                    args.arm,
                    inner_channel,
                    {"val_mse": initial["mse"], "val_psnr_db": initial["psnr_db"], "val_ssim": initial["ssim"]},
                    warm_start,
                    rate_contract,
                ),
                best_path,
            )
            print(json.dumps(initial_row, indent=2))
        for epoch in range(start_epoch, epochs):
            train_metrics = train_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                args.arm,
                config,
                device,
                args.max_train_batches,
            )
            validation = evaluate(
                model,
                val_loader,
                args.arm,
                config,
                device,
                args.max_val_batches,
                sample_dir / f"epoch_{epoch:03d}.png",
            )
            row = {
                "epoch": epoch,
                **train_metrics,
                "val_mse": validation["mse"],
                "val_psnr_db": validation["psnr_db"],
                "val_ssim": validation["ssim"],
            }
            if not finite_metrics({key: value for key, value in row.items() if key != "epoch"}):
                raise RuntimeError(f"Non-finite epoch metrics: {row}")
            writer.writerow(row)
            handle.flush()
            payload = checkpoint_payload(
                model,
                optimizer,
                scaler,
                epoch,
                config,
                args.arm,
                inner_channel,
                {"val_mse": validation["mse"], "val_psnr_db": validation["psnr_db"], "val_ssim": validation["ssim"]},
                warm_start,
                rate_contract,
            )
            torch.save(payload, latest_path)
            if float(validation["mse"]) < best_mse:
                best_mse = float(validation["mse"])
                shutil.copy2(latest_path, best_path)
            print(json.dumps(row, indent=2))

    best = torch.load(best_path, map_location=device)
    model.load_state_dict(best["model"], strict=True)
    final_metrics = evaluate(
        model,
        val_loader,
        args.arm,
        config,
        device,
        args.max_val_batches,
        sample_dir / "best.png",
    )
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_commit": os.popen(f"git -C {PROJECT_ROOT} rev-parse HEAD").read().strip(),
        "config": relative(config_path),
        "config_sha256": sha256_file(config_path),
        "script": relative(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "arm": args.arm,
        "inner_channel": inner_channel,
        "actual_cbr": inner_channel / int(config["protocol"]["cbr_denominator"]),
        "rate_contract": rate_contract,
        "warm_start": warm_start,
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
        "best_epoch": int(best["epoch"]),
        "best_metrics": final_metrics,
        "best_checkpoint": relative(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "proxy_environment_present": sorted(key for key in os.environ if "proxy" in key.lower()),
        "official_val_accessed": False,
    }
    save_json(output_dir / "metadata.json", metadata)
    save_json(
        output_dir / "STATE.json",
        {
            "state": "COMPLETE",
            "arm": args.arm,
            "best_epoch": int(best["epoch"]),
            "best_checkpoint_sha256": metadata["best_checkpoint_sha256"],
            "official_val_accessed": False,
        },
    )
    print(json.dumps({"output_dir": relative(output_dir), **metadata}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
