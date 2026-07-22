#!/usr/bin/env python3
"""Train the preregistered exact-19,712-real-symbol DeepJSCC baseline."""

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
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from cadsd_jscc.datasets import FlatImageDataset  # noqa: E402
from cadsd_jscc.deepjscc_adapter import build_deepjscc_model  # noqa: E402
from cadsd_jscc.external_rate_alignment import ExactRateMaskedDeepJSCC  # noqa: E402
from cadsd_jscc.metrics import psnr_per_sample, ssim_per_sample  # noqa: E402
from s7_train_matched_rate_jscc import warm_start_pruned_model  # noqa: E402


SCRIPT = Path(__file__).resolve()


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


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


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_transform(image_size: int, train: bool):
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
        [transforms.Resize(image_size), transforms.CenterCrop(image_size), transforms.ToTensor()]
    )


def subset(dataset, size: int, seed: int):
    count = min(int(size), len(dataset))
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[
        :count
    ].tolist()
    return Subset(dataset, indices)


def build_loaders(config: dict[str, Any], device: torch.device, dry_run: bool):
    image_size = int(config["image_size"])
    seed = int(config["seed"])
    if dry_run:
        train_set = FakeData(
            size=16,
            image_size=(3, image_size, image_size),
            num_classes=1,
            transform=transforms.ToTensor(),
            random_offset=seed,
        )
        val_set = FakeData(
            size=8,
            image_size=(3, image_size, image_size),
            num_classes=1,
            transform=transforms.ToTensor(),
            random_offset=seed + 1,
        )
    else:
        train_set = FlatImageDataset(
            resolve(config["data"]["train_root"]), make_transform(image_size, True)
        )
        val_set = FlatImageDataset(
            resolve(config["data"]["val_root"]), make_transform(image_size, False)
        )
        if config["data"].get("train_subset") is not None:
            train_set = subset(train_set, int(config["data"]["train_subset"]), seed)
        if config["data"].get("val_subset") is not None:
            val_set = subset(val_set, int(config["data"]["val_subset"]), seed + 1)
    workers = 0 if dry_run else int(config["training"]["num_workers"])
    common = dict(
        batch_size=int(config["training"]["batch_size"]),
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    return (
        DataLoader(
            train_set,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            **common,
        ),
        DataLoader(val_set, shuffle=False, **common),
    )


def validate_contract(config: dict[str, Any]) -> None:
    if config.get("status") not in {
        "preregistered_before_training_output",
        "registered_after_numeric_failure_before_stable_run_output",
        "registered_after_pilot_curve_before_fullcoco_continuation_output",
    }:
        raise RuntimeError("training config is not preregistered")
    if config.get("official_val_accessed") is not False:
        raise RuntimeError("official validation must remain sealed")
    rate = config["rate"]
    dense = int(rate["dense_real_symbols"])
    active = int(rate["active_real_symbols"])
    if dense != 2 * int(rate["inner_channel"]) * 64 * 64:
        raise RuntimeError("dense latent count does not match c and spatial size")
    if active != 19712 or int(rate["complex_channel_uses"]) != active // 2:
        raise RuntimeError("author-rate active symbol contract changed")
    exact_cbr = (active / 2) / int(rate["source_real_dimensions"])
    if not math.isclose(exact_cbr, float(rate["exact_cbr"]), abs_tol=1e-15):
        raise RuntimeError("exact CBR mismatch")


@torch.no_grad()
def evaluate(
    model: ExactRateMaskedDeepJSCC,
    loader: DataLoader,
    device: torch.device,
    sample_path: Path | None,
    max_batches: int | None,
) -> dict[str, float]:
    model.eval()
    mses: list[float] = []
    psnrs: list[float] = []
    ssims: list[float] = []
    for batch_index, (images_cpu, _labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images_cpu.to(device, non_blocking=True)
        output = model(images).clamp(0.0, 1.0)
        mses.extend(
            F.mse_loss(output, images, reduction="none")
            .flatten(start_dim=1)
            .mean(dim=1)
            .cpu()
            .tolist()
        )
        psnrs.extend(psnr_per_sample(output, images).cpu().tolist())
        ssims.extend(ssim_per_sample(output, images).cpu().tolist())
        if batch_index == 0 and sample_path is not None:
            count = min(4, len(images))
            save_image(
                torch.cat([images[:count].cpu(), output[:count].cpu()]),
                sample_path,
                nrow=count,
            )
    if not mses:
        raise RuntimeError("validation produced no batches")
    return {
        "mse": float(sum(mses) / len(mses)),
        "psnr_db": float(sum(psnrs) / len(psnrs)),
        "ssim": float(sum(ssims) / len(ssims)),
    }


def train_epoch(
    model: ExactRateMaskedDeepJSCC,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
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
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            output = model(images).clamp(0.0, 1.0)
            loss = F.mse_loss(output, images)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at batch {batch_index}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["training"]["grad_clip_norm"])
        )
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    if not losses:
        raise RuntimeError("training produced no batches")
    return {
        "train_mse": sum(losses) / len(losses),
        "train_batches": len(losses),
        "epoch_seconds": time.perf_counter() - started,
    }


def checkpoint_payload(
    model: ExactRateMaskedDeepJSCC,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    metrics: dict[str, float],
    config: dict[str, Any],
    warm_start: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "config": config,
        "warm_start": warm_start,
        "rate_contract": config["rate"],
        "official_val_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/external_author_rate_deepjscc_train.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    args = parser.parse_args()

    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_contract(config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed_everything(int(config["seed"]))
    train_loader, val_loader = build_loaders(config, device, args.dry_run)
    base = build_deepjscc_model(
        resolve(config["baseline"]["repo"]),
        int(config["rate"]["inner_channel"]),
        str(config["channel"]["type"]),
        float(config["channel"]["train_snr_db"]),
    )
    warm_start = warm_start_pruned_model(
        base,
        resolve(config["protocol"]["source_checkpoint"]),
        int(config["rate"]["inner_channel"]),
    )
    model = ExactRateMaskedDeepJSCC(
        base,
        dense_symbols=int(config["rate"]["dense_real_symbols"]),
        active_symbols=int(config["rate"]["active_real_symbols"]),
        snr_db=float(config["channel"]["train_snr_db"]),
    ).to(device)
    exact_resume = config["protocol"].get("exact_resume_checkpoint")
    if exact_resume is not None:
        exact_resume_path = resolve(exact_resume)
        observed_sha = sha256_file(exact_resume_path)
        expected_sha = str(config["protocol"]["exact_resume_checkpoint_sha256"])
        if observed_sha != expected_sha:
            raise RuntimeError("exact-rate continuation checkpoint SHA-256 mismatch")
        resumed = torch.load(exact_resume_path, map_location=device)
        model.load_state_dict(resumed["model"], strict=True)
        warm_start = {
            **warm_start,
            "exact_resume_checkpoint": relative(exact_resume_path),
            "exact_resume_checkpoint_sha256": observed_sha,
            "exact_resume_epoch": int(resumed["epoch"]),
        }

    plan = {
        "experiment_id": config["experiment_id"],
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
        "device": str(device),
        "rate": config["rate"],
        "warm_start": warm_start,
        "official_val_accessed": False,
    }
    if args.dry_run:
        images = next(iter(train_loader))[0].to(device)
        with torch.no_grad():
            output = model(images)
        print(
            json.dumps(
                {
                    **plan,
                    "input_shape": list(images.shape),
                    "output_shape": list(output.shape),
                    "finite": bool(torch.isfinite(output).all()),
                    "active_index_count": int(model.active_indices.numel()),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    output_dir = resolve(config["output_dir"])
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = output_dir / "checkpoints"
    sample_dir = output_dir / "samples"
    checkpoint_dir.mkdir()
    sample_dir.mkdir()
    shutil.copy2(config_path, output_dir / "config.yaml")
    shutil.copy2(SCRIPT, output_dir / SCRIPT.name)
    save_json(output_dir / "run_plan.json", plan)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(
        enabled=bool(config["training"]["amp"] and device.type == "cuda")
    )
    history_path = output_dir / "history.csv"
    best_path = checkpoint_dir / "best.pt"
    latest_path = checkpoint_dir / "latest.pt"
    best_mse = math.inf
    fields = [
        "epoch",
        "train_mse",
        "val_mse",
        "val_psnr_db",
        "val_ssim",
        "epoch_seconds",
        "train_batches",
    ]
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        initial = evaluate(model, val_loader, device, sample_dir / "initial.png", args.max_val_batches)
        writer.writerow(
            {
                "epoch": -1,
                "train_mse": "",
                "val_mse": initial["mse"],
                "val_psnr_db": initial["psnr_db"],
                "val_ssim": initial["ssim"],
                "epoch_seconds": 0.0,
                "train_batches": 0,
            }
        )
        handle.flush()
        torch.save(
            checkpoint_payload(model, optimizer, scaler, -1, initial, config, warm_start),
            best_path,
        )
        best_mse = initial["mse"]
        print(json.dumps({"epoch": -1, **initial}, ensure_ascii=False))
        for epoch in range(int(config["training"]["epochs"])):
            train = train_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                config,
                device,
                args.max_train_batches,
            )
            validation = evaluate(
                model,
                val_loader,
                device,
                sample_dir / f"epoch_{epoch:03d}.png",
                args.max_val_batches,
            )
            row = {
                "epoch": epoch,
                **train,
                "val_mse": validation["mse"],
                "val_psnr_db": validation["psnr_db"],
                "val_ssim": validation["ssim"],
            }
            writer.writerow(row)
            handle.flush()
            payload = checkpoint_payload(
                model, optimizer, scaler, epoch, validation, config, warm_start
            )
            torch.save(payload, latest_path)
            if validation["mse"] < best_mse:
                best_mse = validation["mse"]
                shutil.copy2(latest_path, best_path)
            print(json.dumps(row, ensure_ascii=False))

    best = torch.load(best_path, map_location=device)
    model.load_state_dict(best["model"], strict=True)
    final_metrics = evaluate(model, val_loader, device, sample_dir / "best.png", None)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": relative(config_path),
        "config_sha256": sha256_file(config_path),
        "script": relative(SCRIPT),
        "script_sha256": sha256_file(SCRIPT),
        "best_epoch": int(best["epoch"]),
        "best_metrics": final_metrics,
        "best_checkpoint": relative(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "rate": config["rate"],
        "warm_start": warm_start,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "official_val_accessed": False,
    }
    save_json(output_dir / "metadata.json", metadata)
    save_json(
        output_dir / "STATE.json",
        {
            "state": "COMPLETE",
            "best_epoch": int(best["epoch"]),
            "best_checkpoint_sha256": metadata["best_checkpoint_sha256"],
            "official_val_accessed": False,
        },
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
