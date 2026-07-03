from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import FakeData
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cadsd_jscc.datasets import FlatImageDataset
from cadsd_jscc.deepjscc_adapter import build_deepjscc_model
from cadsd_jscc.metrics import psnr_per_sample, ssim_per_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a high-resolution DeepJSCC checkpoint.")
    parser.add_argument("--config", default="configs/s2_deepjscc_coco256_awgn.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--train-root", default=None, help="Override config data.train_root.")
    parser.add_argument("--val-root", default=None, help="Override config data.val_root.")
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic 256x256 images for a quick code-path check.")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def all_finite(metrics: dict) -> bool:
    return all(math.isfinite(float(value)) for value in metrics.values())


def cbr_to_inner_channel(cbr: float) -> int:
    return int(round(48.0 * cbr))


def make_transforms(image_size: int, train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0), ratio=(0.75, 1.3333333333)),
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


def build_loaders(config: dict, args: argparse.Namespace):
    image_size = int(config["image_size"])
    train_transform = make_transforms(image_size=image_size, train=True)
    val_transform = make_transforms(image_size=image_size, train=False)
    seed = int(config["seed"])

    if args.dry_run:
        train_dataset = FakeData(size=8, image_size=(3, image_size, image_size), num_classes=1, transform=transforms.ToTensor())
        val_dataset = FakeData(size=4, image_size=(3, image_size, image_size), num_classes=1, transform=transforms.ToTensor())
    else:
        train_root = PROJECT_ROOT / (args.train_root or config["data"]["train_root"])
        val_root = PROJECT_ROOT / (args.val_root or config["data"]["val_root"])
        train_dataset = FlatImageDataset(train_root, transform=train_transform)
        val_dataset = FlatImageDataset(val_root, transform=val_transform)
        train_dataset = maybe_subset(train_dataset, config["data"].get("train_subset"), seed=seed)
        val_dataset = maybe_subset(val_dataset, config["data"].get("val_subset"), seed=seed + 1)

    training = config["training"]
    batch_size = int(args.batch_size or training["batch_size"])
    num_workers = int(args.num_workers if args.num_workers is not None else training["num_workers"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, max_batches: int | None, sample_path: Path | None = None) -> dict:
    model.eval()
    losses: list[float] = []
    psnrs: list[float] = []
    ssims: list[float] = []
    sample_saved = False
    with torch.no_grad():
        for batch_idx, (images, _labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            outputs = model(images).clamp(0, 1)
            losses.append(F.mse_loss(outputs, images).item())
            psnrs.extend(psnr_per_sample(outputs, images).detach().cpu().tolist())
            ssims.extend(ssim_per_sample(outputs, images).detach().cpu().tolist())
            if sample_path is not None and not sample_saved:
                count = min(4, images.size(0))
                save_image(torch.cat([images[:count], outputs[:count]], dim=0), sample_path, nrow=count)
                sample_saved = True
    return {"mse": mean(losses), "psnr_db": mean(psnrs), "ssim": mean(ssims)}


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    grad_clip_norm: float | None,
    max_batches: int | None,
) -> dict:
    model.train()
    losses: list[float] = []
    for batch_idx, (images, _labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images).clamp(0, 1)
            loss = F.mse_loss(outputs, images)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at batch {batch_idx}: {loss.detach().cpu().item()}")
        scaler.scale(loss).backward()
        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.detach().cpu().item())
    return {"mse": mean(losses)}


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    config_path = PROJECT_ROOT / args.config
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    torch.manual_seed(int(config["seed"]))
    device = resolve_device(args.device)
    training = config["training"]
    epochs = int(args.epochs or training["epochs"])
    use_amp = bool(training.get("amp", False) and device.type == "cuda")
    max_train_batches = args.max_train_batches if args.max_train_batches is not None else training.get("max_train_batches")
    max_val_batches = args.max_val_batches if args.max_val_batches is not None else training.get("max_val_batches")

    output_dir = PROJECT_ROOT / (args.output_dir or (config["outputs"]["smoke_dir"] if args.dry_run else config["outputs"]["train_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "samples").mkdir(exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")

    train_loader, val_loader = build_loaders(config, args)
    inner_channel = int(config.get("inner_channel") or cbr_to_inner_channel(float(config["cbr"])))
    model = build_deepjscc_model(
        repo_root=PROJECT_ROOT / config["baseline"]["repo"],
        inner_channel=inner_channel,
        channel=str(config["channel"]),
        snr=float(config["snr_db"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(enabled=use_amp)

    metadata = {
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "run_command": " ".join(sys.argv),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "dry_run": bool(args.dry_run),
        "inner_channel": inner_channel,
        "snr_db": float(config["snr_db"]),
        "cbr": float(config["cbr"]),
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
        "train_root": args.train_root or config["data"]["train_root"],
        "val_root": args.val_root or config["data"]["val_root"],
    }
    write_json(output_dir / "metadata.json", metadata)

    history_path = output_dir / "history.csv"
    best_val = math.inf
    best_path = output_dir / "checkpoints" / "best.pt"
    stopped_nonfinite = False
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_mse", "val_mse", "val_psnr_db", "val_ssim"])
        writer.writeheader()
        for epoch in range(epochs):
            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                use_amp=use_amp,
                grad_clip_norm=training.get("grad_clip_norm"),
                max_batches=max_train_batches,
            )
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                max_batches=max_val_batches,
                sample_path=output_dir / "samples" / f"epoch_{epoch:04d}.png",
            )
            row = {
                "epoch": epoch,
                "train_mse": train_metrics["mse"],
                "val_mse": val_metrics["mse"],
                "val_psnr_db": val_metrics["psnr_db"],
                "val_ssim": val_metrics["ssim"],
            }
            writer.writerow(row)
            handle.flush()
            print(json.dumps(row, indent=2))

            if not all_finite(row):
                print(f"Stopping early due to non-finite metrics at epoch {epoch}.", file=sys.stderr)
                stopped_nonfinite = True
                break

            latest_path = output_dir / "checkpoints" / "latest.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": config, "metrics": row}, latest_path)
            if val_metrics["mse"] < best_val:
                best_val = val_metrics["mse"]
                shutil.copy2(latest_path, best_path)

    final_checkpoint = None
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        final_checkpoint = str(best_path.relative_to(PROJECT_ROOT))
    elif stopped_nonfinite:
        raise RuntimeError("Training stopped on non-finite metrics before a finite checkpoint was saved.")

    final_metrics = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        max_batches=max_val_batches,
        sample_path=output_dir / "samples" / "final.png",
    )
    write_json(
        output_dir / "metrics.json",
        {
            "metadata": metadata,
            "final": final_metrics,
            "best_val_mse": best_val,
            "final_checkpoint": final_checkpoint,
            "stopped_nonfinite": stopped_nonfinite,
        },
    )
    print(json.dumps({"final": final_metrics, "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
