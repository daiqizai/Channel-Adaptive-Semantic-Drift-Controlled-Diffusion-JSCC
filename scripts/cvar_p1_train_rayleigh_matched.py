#!/usr/bin/env python3
"""P1 attribution closure: Rayleigh-matched MEAN training of the strong JSCC backbone.

Purpose is narrow and pre-registered: the P0 diagnostic showed a large
conditional tail under Rayleigh block fading, but also showed that the frozen
S33B backbone had never seen fading and that its SNR conditioning could not
represent the effective SNR a deep fade produces.  This script removes both
mismatches by training the *mean* objective on the matched channel:

*   encoder is conditioned on the **nominal** SNR only (block fading without a
    feedback link means the transmitter cannot know ``h``);
*   the channel is the same block-fading + zero-forcing path used by the
    diagnostic (``cadsd_jscc.tail_risk.apply_block_fading_channel``);
*   the decoder is conditioned on the true **effective** SNR
    ``nominal + 10*log10(|h|^2)``, unclamped, so the conditioning embedding is
    trained over the full range that fading actually produces.

The loss is plain MSE.  This is deliberately NOT a CVaR model: it is the
`Repeated-fading mean control` that the task book requires any CVaR claim to
beat, and it is the model whose residual tail decides whether CVaR is worth
testing at all.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from cadsd_jscc.metrics import ms_ssim_per_sample, psnr_per_sample  # noqa: E402
from cadsd_jscc.strong_jscc import StrongJSCC  # noqa: E402
from cadsd_jscc.tail_risk import apply_block_fading_channel  # noqa: E402
from s31_train_strong_jscc import (  # noqa: E402
    build_loaders,
    build_model,
    learning_rate_multiplier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/cvar_p1_rayleigh_matched_mean_training.yaml"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def sample_fading(
    count: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """``h ~ CN(0, 1)`` per image, i.e. both parts ``N(0, 1/2)``."""

    parts = torch.randn(count, 2, generator=generator, dtype=torch.float32) * (0.5**0.5)
    return parts[:, 0].contiguous(), parts[:, 1].contiguous()


def matched_forward(
    model: StrongJSCC,
    images: torch.Tensor,
    nominal_snr: torch.Tensor,
    h_real: torch.Tensor,
    h_imag: torch.Tensor,
    noise: torch.Tensor | None,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encoder sees nominal SNR, decoder sees the true post-equalization SNR."""

    latent = model.encode(images, nominal_snr)
    transmitted, _ = model.normalize_channel_input(latent)
    if noise is None:
        noise = torch.randn_like(transmitted)
    received = apply_block_fading_channel(
        transmitted, noise, nominal_snr, h_real, h_imag, epsilon=epsilon
    )
    h_power = (h_real.square() + h_imag.square()).clamp_min(1e-30)
    effective = nominal_snr + 10.0 * torch.log10(h_power.to(nominal_snr.device))
    return model.decode(received, effective), effective


def evaluate(
    model: StrongJSCC,
    loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    max_batches: int | None,
) -> dict[str, Any]:
    """Validation on the matched channel with frozen per-condition fading seeds."""

    model.eval()
    channel = config["channel"]
    epsilon = float(channel["equalization_epsilon"])
    seed = int(channel["validation_seed"])
    realizations = int(config["validation"]["realizations_per_image"])
    per_snr: list[dict[str, Any]] = []
    for snr in [float(value) for value in channel["validation_snrs_db"]]:
        psnrs: list[float] = []
        ms_ssims: list[float] = []
        mses: list[float] = []
        for realization in range(realizations):
            generator = torch.Generator().manual_seed(
                seed + realization * 1000 + int(round(snr * 10))
            )
            for batch_index, (images_cpu, _labels) in enumerate(loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                images = images_cpu.to(device, non_blocking=True)
                count = images.shape[0]
                h_real, h_imag = sample_fading(count, generator)
                noise = torch.randn(
                    count,
                    model.latent_channels,
                    model.image_size // 16,
                    model.image_size // 16,
                    generator=generator,
                    dtype=torch.float32,
                ).to(device)
                nominal = torch.full((count,), snr, dtype=torch.float32, device=device)
                with torch.inference_mode():
                    output, _ = matched_forward(
                        model,
                        images,
                        nominal,
                        h_real.to(device),
                        h_imag.to(device),
                        noise,
                        epsilon,
                    )
                    output = torch.floor(output.clamp(0.0, 1.0) * 255.0) / 255.0
                    psnrs += psnr_per_sample(output, images).tolist()
                    ms_ssims += ms_ssim_per_sample(output, images).tolist()
                    mses += (
                        (output - images).square().flatten(start_dim=1).mean(dim=1).tolist()
                    )
        per_snr.append(
            {
                "snr_db": snr,
                "samples": len(psnrs),
                "mse": sum(mses) / len(mses),
                "psnr_db": sum(psnrs) / len(psnrs),
                "ms_ssim": sum(ms_ssims) / len(ms_ssims),
            }
        )
    aggregate = {
        "psnr_db": sum(row["psnr_db"] for row in per_snr) / len(per_snr),
        "ms_ssim": sum(row["ms_ssim"] for row in per_snr) / len(per_snr),
        "mse": sum(row["mse"] for row in per_snr) / len(per_snr),
    }
    return {"per_snr": per_snr, "aggregate": aggregate}


def train_epoch(
    model: StrongJSCC,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    config: dict[str, Any],
    device: torch.device,
    max_batches: int | None,
    snr_generator: torch.Generator,
    fading_generator: torch.Generator,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    channel = config["channel"]
    epsilon = float(channel["equalization_epsilon"])
    choices = torch.tensor(channel["train_snrs_db"], dtype=torch.float32)
    losses: list[float] = []
    optimizer_steps = 0
    started = time.perf_counter()
    expected = len(loader) if max_batches is None else min(len(loader), max_batches)
    for batch_index, (images_cpu, _labels) in enumerate(loader):
        if batch_index >= expected:
            break
        images = images_cpu.to(device, non_blocking=True)
        count = images.shape[0]
        indices = torch.randint(
            len(choices), (count,), generator=snr_generator, device="cpu"
        )
        nominal = choices[indices].to(device, non_blocking=True)
        h_real, h_imag = sample_fading(count, fading_generator)
        output, _ = matched_forward(
            model, images, nominal, h_real.to(device), h_imag.to(device), None, epsilon
        )
        loss = F.mse_loss(output, images)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss at batch {batch_index}")
        loss.backward()
        losses.append(float(loss.detach().cpu()))
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["training"]["grad_clip_norm"])
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError(f"non-finite gradient norm at batch {batch_index}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        optimizer_steps += 1
        if batch_index % 200 == 0:
            print(
                f"    batch {batch_index}/{expected} loss={loss.item():.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.3e} "
                f"elapsed={time.perf_counter() - started:.0f}s",
                flush=True,
            )
    if not losses:
        raise RuntimeError("training produced no batches")
    return {
        "train_mse": sum(losses) / len(losses),
        "train_batches": len(losses),
        "optimizer_steps": optimizer_steps,
        "epoch_seconds": time.perf_counter() - started,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["official_imagenette_validation_accessed"] is not False:
        raise RuntimeError("official validation must remain sealed")

    output = resolve(
        config["outputs"]["dry_run_directory"]
        if args.dry_run
        else config["outputs"]["directory"]
    )
    if output.exists() and bool(config["outputs"]["overwrite_forbidden"]):
        raise FileExistsError(f"refusing to overwrite {output}")
    (output / "checkpoints").mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device or config["training"]["device"])

    source = resolve(config["initialization"]["checkpoint"])
    source_sha = sha256_file(source)
    expected_sha = str(config["initialization"]["checkpoint_sha256"])
    if source_sha != expected_sha:
        raise RuntimeError(f"init checkpoint SHA mismatch: {source_sha} != {expected_sha}")
    source_checkpoint = torch.load(source, map_location="cpu", weights_only=False)

    model = build_model(config).to(device)
    model.load_state_dict(source_checkpoint["model"], strict=True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"initialized from {source.name} (epoch {source_checkpoint['epoch']}), "
          f"{trainable} trainable parameters", flush=True)

    train_loader, val_loader = build_loaders(config, device, args.dry_run)
    epochs = int(config["training"]["epochs"])
    max_batches = int(config["training"]["dry_run_batches"]) if args.dry_run else None
    steps_per_epoch = len(train_loader) if max_batches is None else max_batches
    total_steps = max(1, epochs * steps_per_epoch)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    minimum_ratio = float(config["training"]["minimum_learning_rate"]) / float(
        config["training"]["learning_rate"]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: learning_rate_multiplier(
            step, total_steps, int(config["training"]["warmup_optimizer_steps"]), minimum_ratio
        ),
    )
    snr_generator = torch.Generator().manual_seed(seed + 11)
    fading_generator = torch.Generator().manual_seed(seed + 22)

    log_path = output / "training_log.csv"
    log_fields = [
        "epoch",
        "global_step",
        "learning_rate",
        "train_mse",
        "epoch_seconds",
        "val_aggregate_psnr_db",
        "val_aggregate_ms_ssim",
        "val_aggregate_mse",
        "gpu_memory_allocated_mb",
        "elapsed_time_s",
    ]
    log_handle = log_path.open("w", newline="", encoding="utf-8")
    log_writer = csv.DictWriter(log_handle, fieldnames=log_fields)
    log_writer.writeheader()

    config_sha = sha256_file(config_path)
    started = time.time()
    global_step = 0
    best_psnr = -math.inf
    history: list[dict[str, Any]] = []
    val_batches = int(config["training"]["dry_run_batches"]) if args.dry_run else None

    for epoch in range(epochs):
        print(f"epoch {epoch}", flush=True)
        stats = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            config,
            device,
            max_batches,
            snr_generator,
            fading_generator,
        )
        global_step += int(stats["optimizer_steps"])
        metrics = evaluate(model, val_loader, config, device, val_batches)
        aggregate = metrics["aggregate"]
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "learning_rate": stats["learning_rate"],
            "train_mse": stats["train_mse"],
            "epoch_seconds": stats["epoch_seconds"],
            "val_aggregate_psnr_db": aggregate["psnr_db"],
            "val_aggregate_ms_ssim": aggregate["ms_ssim"],
            "val_aggregate_mse": aggregate["mse"],
            "gpu_memory_allocated_mb": (
                torch.cuda.max_memory_allocated(device) / (1 << 20)
                if device.type == "cuda"
                else 0.0
            ),
            "elapsed_time_s": time.time() - started,
        }
        log_writer.writerow(row)
        log_handle.flush()
        history.append({**row, "per_snr": metrics["per_snr"]})
        print(
            f"  epoch {epoch}: train_mse={stats['train_mse']:.6f} "
            f"val_psnr={aggregate['psnr_db']:.4f} dB "
            f"({stats['epoch_seconds']:.0f}s)",
            flush=True,
        )
        for entry in metrics["per_snr"]:
            print(
                f"    snr={entry['snr_db']:5.1f} psnr={entry['psnr_db']:.4f} "
                f"ms_ssim={entry['ms_ssim']:.5f}",
                flush=True,
            )

        payload = {
            "format_version": 1,
            "epoch": epoch,
            "global_optimizer_steps": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": metrics,
            "config": config,
            "config_sha256": config_sha,
        }
        torch.save(payload, output / "checkpoints" / "latest.pt")
        # Pre-registered selection rule: highest aggregate validation PSNR on the
        # matched Rayleigh channel.  Same rule S31/S33 used, only the channel差.
        if aggregate["psnr_db"] > best_psnr:
            best_psnr = aggregate["psnr_db"]
            torch.save(payload, output / "checkpoints" / "best.pt")
            print(f"  new best: {best_psnr:.4f} dB", flush=True)

    log_handle.close()
    best_path = output / "checkpoints" / "best.pt"
    metadata = {
        "experiment_id": config["experiment_id"],
        "dry_run": args.dry_run,
        "git_commit": git_commit(),
        "config_sha256": config_sha,
        "init_checkpoint": str(source.relative_to(ROOT)),
        "init_checkpoint_sha256": source_sha,
        "torch_version": torch.__version__,
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "trainable_parameters": trainable,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "global_optimizer_steps": global_step,
        "best_aggregate_psnr_db": best_psnr,
        "best_checkpoint_sha256": sha256_file(best_path),
        "total_seconds": time.time() - started,
        "history": history,
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (output / "config_copy.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    print(f"best aggregate PSNR {best_psnr:.4f} dB")
    print(f"best checkpoint SHA256 {metadata['best_checkpoint_sha256']}")


if __name__ == "__main__":
    main()
