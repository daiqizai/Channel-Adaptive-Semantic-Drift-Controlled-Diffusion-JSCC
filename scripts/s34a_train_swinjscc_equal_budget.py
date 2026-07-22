#!/usr/bin/env python3
"""Train one authorized S34A SwinJSCC equal-budget arm (hard cap: 12 epochs)."""

from __future__ import annotations

import argparse
import csv
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
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.utils import save_image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cadsd_jscc.datasets import FlatImageDataset  # noqa: E402
from cadsd_jscc.metrics import ms_ssim_per_sample, psnr_per_sample  # noqa: E402
from cadsd_jscc.swinjscc_adapter import (  # noqa: E402
    OFFICIAL_SOURCE,
    OfficialSwinJSCCSA,
    trainable_parameter_count,
)


SCRIPT = Path(__file__).resolve()
AUTHORIZED_STATUS = "equal_budget_dual_arm_authorized_extension_forbidden"
AUTHORIZED_TOTAL_EPOCHS = 12
AUTHORIZED_ARMS = {"official_base_sa", "capacity_matched_sa"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/s34a_swinjscc_equal_rate_comparison.yaml"
    )
    parser.add_argument("--arm", required=True, choices=sorted(AUTHORIZED_ARMS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Audit the frozen contract without creating a formal output directory.",
    )
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(arm: dict[str, Any]) -> OfficialSwinJSCCSA:
    return OfficialSwinJSCCSA(
        image_size=256,
        latent_channels=64,
        encoder_depths=tuple(int(value) for value in arm["encoder_depths"]),
        decoder_depths=tuple(int(value) for value in arm["decoder_depths"]),
    )


def make_transform(config: dict[str, Any], train: bool):
    size = int(config["rate"]["image_shape"][1])
    if train:
        operations: list[Any] = [
            transforms.RandomResizedCrop(
                size,
                scale=tuple(float(value) for value in config["data"]["train_crop_scale"]),
                ratio=(0.75, 1.3333333333333333),
            )
        ]
        if bool(config["data"]["random_horizontal_flip"]):
            operations.append(transforms.RandomHorizontalFlip())
        operations.append(transforms.ToTensor())
        return transforms.Compose(operations)
    return transforms.Compose(
        [transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor()]
    )


def fixed_subset(dataset, size: int, seed: int):
    count = min(int(size), len(dataset))
    indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(int(seed))
    )[:count].tolist()
    return Subset(dataset, indices)


def build_loaders(
    config: dict[str, Any], device: torch.device
) -> tuple[DataLoader, DataLoader, torch.Generator]:
    train_set = FlatImageDataset(
        resolve(config["data"]["train_root"]), make_transform(config, True)
    )
    val_full = FlatImageDataset(
        resolve(config["data"]["val_root"]), make_transform(config, False)
    )
    val_set = fixed_subset(
        val_full,
        int(config["data"]["val_subset"]),
        int(config["data"]["val_subset_seed"]),
    )
    training = config["formal_training"]
    workers = int(training["num_workers"])
    shuffle_generator = torch.Generator().manual_seed(int(config["seed"]))
    common = dict(
        batch_size=int(training["microbatch_size"]),
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    train_loader = DataLoader(
        train_set, shuffle=True, generator=shuffle_generator, **common
    )
    val_loader = DataLoader(val_set, shuffle=False, **common)
    return train_loader, val_loader, shuffle_generator


def validate_contract(
    config: dict[str, Any], arm_name: str, model: OfficialSwinJSCCSA
) -> dict[str, Any]:
    if config["status"] != AUTHORIZED_STATUS:
        raise RuntimeError(f"formal S34A status must be {AUTHORIZED_STATUS!r}")
    formal = config["formal_training"]
    if formal["run_allowed"] is not True or formal["scope"] != "equal_budget_only":
        raise RuntimeError("equal-budget formal training is not authorized")
    if formal["extension_run_allowed"] is not False:
        raise RuntimeError("extension must remain forbidden")
    if config["convergence_assessment"]["fully_converged_extension"]["run_allowed"] is not False:
        raise RuntimeError("convergence extension must remain forbidden")
    if int(formal["maximum_total_epochs_per_arm"]) != AUTHORIZED_TOTAL_EPOCHS:
        raise RuntimeError("authorized hard epoch cap must be exactly 12")
    if int(config["equal_budget_training"]["total_epochs"]) != AUTHORIZED_TOTAL_EPOCHS:
        raise RuntimeError("equal-budget contract must total exactly 12 epochs")
    if arm_name not in formal["allowed_arms"] or arm_name not in AUTHORIZED_ARMS:
        raise RuntimeError(f"arm is not authorized: {arm_name}")
    if config["sealed"]["official_imagenette_validation"] is not True:
        raise RuntimeError("official Imagenette validation must remain sealed")
    if config["official_imagenette_validation_accessed"] is not False:
        raise RuntimeError("official Imagenette validation access flag changed")
    if config["channel"]["type"] != "AWGN":
        raise RuntimeError("S34A is AWGN-only")
    if config["channel"]["train_snr_sampling"] != "discrete_uniform_per_image":
        raise RuntimeError("training SNR must be discrete uniform per image")
    if config["equal_budget_training"]["precision"] != "FP32":
        raise RuntimeError("S34A equal-budget training is FP32-only")
    if config["equal_budget_training"]["loss"] != "mse":
        raise RuntimeError("S34A equal-budget training is MSE-only")
    if int(config["equal_budget_training"]["main"]["epochs"]) != 4:
        raise RuntimeError("main phase must be exactly four epochs")
    if int(config["equal_budget_training"]["continuation"]["epochs"]) != 8:
        raise RuntimeError("continuation phase must be exactly eight epochs")
    if int(formal["microbatch_size"]) * int(formal["gradient_accumulation_steps"]) != int(
        formal["effective_batch_size"]
    ):
        raise RuntimeError("microbatch and accumulation do not preserve effective batch 32")
    rate = config["rate"]
    if model.real_symbols != int(rate["native_real_symbols"]):
        raise RuntimeError("model does not natively emit 16,384 real symbols")
    if list(rate["latent_shape"]) != [64, 16, 16]:
        raise RuntimeError("frozen latent shape changed")
    if rate["mask_or_padding"] != "none" or int(rate["side_information_real_symbols"]) != 0:
        raise RuntimeError("mask, padding, and side information are forbidden")
    if int(rate["complex_channel_uses"]) != model.real_symbols // 2:
        raise RuntimeError("complex channel-use ledger mismatch")
    exact_cbr = (model.real_symbols / 2) / int(rate["source_real_dimensions"])
    if not math.isclose(exact_cbr, float(rate["exact_cbr"]), abs_tol=1e-15):
        raise RuntimeError("CBR ledger mismatch")
    expected_parameters = int(config["arms_confirmed"][arm_name]["trainable_parameters_static_audit"])
    parameters = trainable_parameter_count(model)
    if parameters != expected_parameters:
        raise RuntimeError(f"parameter mismatch: {parameters} != {expected_parameters}")

    smoke_path = resolve(config["smoke"]["result"])
    if sha256_file(smoke_path) != str(config["smoke"]["result_sha256"]):
        raise RuntimeError("frozen smoke result SHA mismatch")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    current_source_manifest = source_manifest(OFFICIAL_SOURCE)
    if current_source_manifest != smoke["official_source_manifest"]:
        raise RuntimeError("pinned official SwinJSCC source changed after smoke")
    smoke_arm = next(row for row in smoke["results"] if row["arm"] == arm_name)
    if smoke_arm["status"] != "PASS" or int(smoke_arm["parameters"]) != parameters:
        raise RuntimeError("authorized arm did not pass the frozen smoke")
    return {
        "trainable_parameters": parameters,
        "official_source_manifest": current_source_manifest,
        "smoke_result": relative(smoke_path),
        "smoke_result_sha256": sha256_file(smoke_path),
    }


def learning_rate_multiplier(
    step: int, total_steps: int, warmup_steps: int, minimum_ratio: float
) -> float:
    if step < warmup_steps:
        return max(1e-8, (step + 1) / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def make_optimizer_scheduler(
    model: OfficialSwinJSCCSA,
    phase_config: dict[str, Any],
    optimizer_steps_per_epoch: int,
    phase_epochs: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    learning_rate = float(phase_config["learning_rate"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(phase_config.get("weight_decay", 0.0001)),
    )
    total_steps = optimizer_steps_per_epoch * phase_epochs
    warmup = int(phase_config.get("warmup_optimizer_steps", 0))
    minimum_ratio = float(phase_config["minimum_learning_rate"]) / learning_rate
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: learning_rate_multiplier(step, total_steps, warmup, minimum_ratio),
    )
    return optimizer, scheduler


def deterministic_noise(
    shape: tuple[int, ...], generator: torch.Generator, device: torch.device
) -> torch.Tensor:
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(
        device=device, non_blocking=True
    )


@torch.no_grad()
def evaluate(
    model: OfficialSwinJSCCSA,
    loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    sample_dir: Path,
    epoch_number: int,
) -> dict[str, Any]:
    model.eval()
    per_snr: list[dict[str, float]] = []
    latent_shape = (256, 64)
    base_seed = int(config["channel"]["validation_noise_seed"])
    for snr_index, snr_db in enumerate(config["channel"]["validation_snrs_db"]):
        generator = torch.Generator(device="cpu").manual_seed(base_seed + 1009 * snr_index)
        mse_sum = psnr_sum = similarity_sum = power_sum = 0.0
        samples = 0
        maximum_power_error = 0.0
        for batch_index, (images_cpu, _labels) in enumerate(loader):
            images = images_cpu.to(device, non_blocking=True)
            noise = deterministic_noise((len(images), *latent_shape), generator, device)
            reconstruction, observation = model.forward_with_observation(
                images, float(snr_db), noise
            )
            reconstruction = reconstruction.clamp(0.0, 1.0)
            sample_mse = F.mse_loss(
                reconstruction, images, reduction="none"
            ).flatten(start_dim=1).mean(dim=1)
            sample_psnr = psnr_per_sample(reconstruction, images)
            sample_ms_ssim = ms_ssim_per_sample(reconstruction, images)
            batch_size = len(images)
            samples += batch_size
            mse_sum += float(sample_mse.sum().cpu())
            psnr_sum += float(sample_psnr.sum().cpu())
            similarity_sum += float(sample_ms_ssim.sum().cpu())
            power = observation.normalized_power.detach()
            power_sum += float(power.sum().cpu())
            maximum_power_error = max(
                maximum_power_error, float((power - 1.0).abs().max().cpu())
            )
            if batch_index == 0:
                count = min(4, batch_size)
                save_image(
                    torch.cat((images[:count].cpu(), reconstruction[:count].cpu())),
                    sample_dir / f"epoch_{epoch_number:02d}_snr_{int(snr_db):02d}.png",
                    nrow=count,
                )
        if samples != len(loader.dataset):
            raise RuntimeError(f"validation sample count mismatch: {samples}")
        per_snr.append(
            {
                "snr_db": float(snr_db),
                "samples": samples,
                "mse": mse_sum / samples,
                "psnr_db": psnr_sum / samples,
                "ms_ssim": similarity_sum / samples,
                "normalized_power_mean": power_sum / samples,
                "normalized_power_max_abs_error": maximum_power_error,
            }
        )
    return {
        "per_snr": per_snr,
        "aggregate": {
            "mse": sum(row["mse"] for row in per_snr) / len(per_snr),
            "psnr_db": sum(row["psnr_db"] for row in per_snr) / len(per_snr),
            "ms_ssim": sum(row["ms_ssim"] for row in per_snr) / len(per_snr),
            "normalized_power_max_abs_error": max(
                row["normalized_power_max_abs_error"] for row in per_snr
            ),
        },
    }


def train_epoch(
    *,
    model: OfficialSwinJSCCSA,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    config: dict[str, Any],
    device: torch.device,
    snr_generator: torch.Generator,
    epoch_number: int,
    progress_path: Path,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    formal = config["formal_training"]
    accumulation = int(formal["gradient_accumulation_steps"])
    microbatch = int(formal["microbatch_size"])
    choices = torch.tensor(config["channel"]["train_snrs_db"], dtype=torch.float32)
    expected_batches = len(loader)
    dataset_samples = len(loader.dataset)
    pixel_values = 3 * 256 * 256
    loss_sum = 0.0
    seen_samples = 0
    optimizer_steps = 0
    started = time.perf_counter()
    last_report = started
    for batch_index, (images_cpu, _labels) in enumerate(loader):
        images = images_cpu.to(device, non_blocking=True)
        batch_size = len(images)
        indices = torch.randint(
            len(choices), (batch_size,), generator=snr_generator, device="cpu"
        )
        snr = choices[indices].to(device, non_blocking=True)
        reconstruction = model(images, snr)
        squared_error_sum = F.mse_loss(reconstruction, images, reduction="sum")
        raw_loss = squared_error_sum / (batch_size * pixel_values)
        if not torch.isfinite(raw_loss):
            raise RuntimeError(
                f"non-finite training loss at epoch {epoch_number}, batch {batch_index}"
            )

        group_start = (batch_index // accumulation) * accumulation
        group_end = min(group_start + accumulation, expected_batches)
        group_samples = sum(
            min(microbatch, dataset_samples - index * microbatch)
            for index in range(group_start, group_end)
        )
        (squared_error_sum / (group_samples * pixel_values)).backward()
        loss_sum += float(squared_error_sum.detach().cpu()) / pixel_values
        seen_samples += batch_size

        should_step = (batch_index + 1) % accumulation == 0 or (
            batch_index + 1 == expected_batches
        )
        if should_step:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["equal_budget_training"]["grad_clip_norm"])
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError(
                    f"non-finite gradient at epoch {epoch_number}, batch {batch_index}"
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            optimizer_steps += 1
            if optimizer_steps % 250 == 0 or batch_index + 1 == expected_batches:
                now = time.perf_counter()
                progress = {
                    "status": "training_epoch",
                    "epoch": epoch_number,
                    "microbatches_completed": batch_index + 1,
                    "microbatches_total": expected_batches,
                    "optimizer_steps_completed": optimizer_steps,
                    "running_train_mse": loss_sum / seen_samples,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "elapsed_seconds": now - started,
                }
                write_json(progress_path, progress)
                if now - last_report >= 300 or batch_index + 1 == expected_batches:
                    print(json.dumps(progress), flush=True)
                    last_report = now
    if seen_samples != dataset_samples:
        raise RuntimeError(f"training sample count mismatch: {seen_samples} != {dataset_samples}")
    return {
        "train_mse": loss_sum / seen_samples,
        "train_samples": seen_samples,
        "train_microbatches": expected_batches,
        "optimizer_steps": optimizer_steps,
        "epoch_seconds": time.perf_counter() - started,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }


def checkpoint_payload(
    *,
    model: OfficialSwinJSCCSA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch_number: int,
    phase: str,
    phase_optimizer_steps: int,
    global_optimizer_steps: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
    config_sha256: str,
    snr_generator: torch.Generator,
    shuffle_generator: torch.Generator,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "epoch_number_1based": epoch_number,
        "phase": phase,
        "phase_optimizer_steps": phase_optimizer_steps,
        "global_optimizer_steps": global_optimizer_steps,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metrics": metrics,
        "config": config,
        "config_sha256": config_sha256,
        "rng": {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "snr_generator": snr_generator.get_state(),
            "shuffle_generator": shuffle_generator.get_state(),
        },
    }


def restore_rng(checkpoint: dict[str, Any], snr_generator, shuffle_generator) -> None:
    rng = checkpoint["rng"]
    random.setstate(rng["python"])
    # ``torch.load(..., map_location=cuda)`` also moves serialized CPU RNG
    # byte tensors.  PyTorch's CPU/default and explicit CPU generators require
    # CPU ByteTensor state, so move only RNG bookkeeping back before restore.
    torch.set_rng_state(rng["torch_cpu"].cpu())
    if torch.cuda.is_available() and rng["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in rng["torch_cuda"]])
    snr_generator.set_state(rng["snr_generator"].cpu())
    shuffle_generator.set_state(rng["shuffle_generator"].cpu())


def convergence_assessment(history_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    with history_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if [int(row["epoch"]) for row in rows] != list(range(1, 13)):
        raise RuntimeError("convergence assessment requires exactly epochs 1--12")
    tail = rows[8:12]
    xs = [float(row["epoch"]) for row in tail]
    ys = [float(row["aggregate_psnr_db"]) for row in tail]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / sum(
        (x - x_mean) ** 2 for x in xs
    )
    best_row = max(
        rows,
        key=lambda row: (
            float(row["aggregate_psnr_db"]), float(row["aggregate_ms_ssim"])
        ),
    )
    best_epoch = int(best_row["epoch"])
    delta_12_minus_9 = ys[-1] - ys[0]
    gate = config["convergence_assessment"]["clearly_not_converged_at_epoch_12_if_all"]
    best_is_late = best_epoch in (11, 12)
    slope_pass = slope >= float(gate[1]["ols_psnr_slope_over_epochs_9_to_12_gte_db_per_epoch"])
    delta_pass = delta_12_minus_9 >= float(gate[2]["validation_psnr_epoch_12_minus_epoch_9_gte_db"])
    triggered = best_is_late and slope_pass and delta_pass
    return {
        "epochs_used": [9, 10, 11, 12],
        "aggregate_psnr_db": ys,
        "best_epoch_across_1_to_12": best_epoch,
        "best_in_epoch_11_or_12": best_is_late,
        "ols_slope_db_per_epoch": slope,
        "slope_gate_pass": slope_pass,
        "epoch12_minus_epoch9_db": delta_12_minus_9,
        "delta_gate_pass": delta_pass,
        "clearly_not_converged_triggered": triggered,
        "interpretation": (
            "extension_triggered_but_not_authorized"
            if triggered
            else "no_clear_evidence_of_nonconvergence_not_proof_of_global_convergence"
        ),
        "extension_executed": False,
        "extension_authorized": False,
    }


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config_sha = sha256_file(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal S34A training requires CUDA")

    seed_everything(int(config["seed"]))
    arm_config = config["arms_confirmed"][args.arm]
    model = build_model(arm_config)
    audit = validate_contract(config, args.arm, model)
    output = resolve(config["formal_training"]["output_directories"][args.arm])
    preflight = {
        "status": "PASS",
        "arm": args.arm,
        "output": relative(output),
        "output_exists": output.exists(),
        "resume_requested": bool(args.resume),
        "authorized_epochs": AUTHORIZED_TOTAL_EPOCHS,
        "extension_run_allowed": False,
        "official_imagenette_validation_accessed": False,
        **audit,
    }
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return

    if output.exists() and not args.resume:
        raise FileExistsError(output)
    if not output.exists() and args.resume:
        raise FileNotFoundError(output)
    if not output.exists():
        output.mkdir(parents=True)
        (output / "checkpoints").mkdir()
        (output / "samples").mkdir()
        shutil.copy2(config_path, output / "config_snapshot.yaml")
        shutil.copy2(SCRIPT, output / SCRIPT.name)
        shutil.copy2(ROOT / "src/cadsd_jscc/swinjscc_adapter.py", output / "swinjscc_adapter.py")
        write_json(output / "official_source_manifest.json", audit["official_source_manifest"])
    elif sha256_file(output / "config_snapshot.yaml") != config_sha:
        raise RuntimeError("resume config does not match output snapshot")

    model.to(device)
    train_loader, val_loader, shuffle_generator = build_loaders(config, device)
    formal = config["formal_training"]
    accumulation = int(formal["gradient_accumulation_steps"])
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / accumulation)
    main_epochs = int(config["equal_budget_training"]["main"]["epochs"])
    continuation_epochs = int(config["equal_budget_training"]["continuation"]["epochs"])
    if main_epochs + continuation_epochs != AUTHORIZED_TOTAL_EPOCHS:
        raise RuntimeError("phase epoch counts exceed authorized hard cap")
    snr_generator = torch.Generator(device="cpu").manual_seed(int(config["seed"]) + 17)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arm": args.arm,
        "config": relative(config_path),
        "config_sha256": config_sha,
        "script": relative(SCRIPT),
        "script_sha256": sha256_file(SCRIPT),
        "adapter": "src/cadsd_jscc/swinjscc_adapter.py",
        "adapter_sha256": sha256_file(ROOT / "src/cadsd_jscc/swinjscc_adapter.py"),
        "official_source": relative(OFFICIAL_SOURCE),
        "official_commit_declared": config["source"]["audited_commit"],
        "official_source_manifest": audit["official_source_manifest"],
        "command": " ".join(sys.argv),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "trainable_parameters": trainable_parameter_count(model),
        "native_real_symbols": model.real_symbols,
        "complex_channel_uses": model.real_symbols // 2,
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "microbatch_size": int(formal["microbatch_size"]),
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": int(formal["effective_batch_size"]),
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "planned_optimizer_steps": optimizer_steps_per_epoch * AUTHORIZED_TOTAL_EPOCHS,
        "maximum_total_epochs": AUTHORIZED_TOTAL_EPOCHS,
        "extension_run_allowed": False,
        "official_imagenette_validation_accessed": False,
        "resume_note": "epoch-boundary model/optimizer/RNG state is saved; persistent worker augmentation RNG may differ after interruption",
    }
    if not args.resume:
        write_json(output / "metadata.json", metadata)

    history_path = output / "history.csv"
    fields = [
        "epoch",
        "phase",
        "train_mse",
        "aggregate_psnr_db",
        "aggregate_ms_ssim",
        "optimizer_steps",
        "phase_optimizer_steps",
        "global_optimizer_steps",
        "learning_rate",
        "epoch_seconds",
    ]
    latest = output / "checkpoints/latest.pt"
    main_best = output / "checkpoints/main_best.pt"
    continuation_best = output / "checkpoints/continuation_best.pt"
    final_best = output / "checkpoints/best.pt"

    start_epoch = 1
    global_optimizer_steps = 0
    phase_optimizer_steps = 0
    best_main = (-math.inf, -math.inf)
    best_continuation = (-math.inf, -math.inf)
    resume_checkpoint: dict[str, Any] | None = None
    if args.resume:
        if not latest.is_file():
            raise FileNotFoundError(latest)
        resume_checkpoint = torch.load(latest, map_location=device, weights_only=False)
        if resume_checkpoint["config_sha256"] != config_sha:
            raise RuntimeError("resume checkpoint config mismatch")
        completed_epoch = int(resume_checkpoint["epoch_number_1based"])
        if completed_epoch >= AUTHORIZED_TOTAL_EPOCHS:
            raise RuntimeError("authorized 12 epochs are already complete; extension is forbidden")
        start_epoch = completed_epoch + 1
        global_optimizer_steps = int(resume_checkpoint["global_optimizer_steps"])
        phase_optimizer_steps = int(resume_checkpoint["phase_optimizer_steps"])
        if main_best.is_file():
            candidate = torch.load(main_best, map_location="cpu", weights_only=False)
            best_main = (
                float(candidate["metrics"]["aggregate"]["psnr_db"]),
                float(candidate["metrics"]["aggregate"]["ms_ssim"]),
            )
        if continuation_best.is_file():
            candidate = torch.load(continuation_best, map_location="cpu", weights_only=False)
            best_continuation = (
                float(candidate["metrics"]["aggregate"]["psnr_db"]),
                float(candidate["metrics"]["aggregate"]["ms_ssim"]),
            )

    initial_phase = "main" if start_epoch <= main_epochs else "continuation"
    phase_config = config["equal_budget_training"][initial_phase]
    phase_epochs = main_epochs if initial_phase == "main" else continuation_epochs
    optimizer, scheduler = make_optimizer_scheduler(
        model, phase_config, optimizer_steps_per_epoch, phase_epochs
    )
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model"], strict=True)
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        scheduler.load_state_dict(resume_checkpoint["scheduler"])
        restore_rng(resume_checkpoint, snr_generator, shuffle_generator)
        write_json(
            output / f"resume_event_before_epoch_{start_epoch:02d}.json",
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "resume_from_checkpoint": relative(latest),
                "resume_from_checkpoint_sha256": sha256_file(latest),
                "next_epoch": start_epoch,
                "script": relative(SCRIPT),
                "resume_script_sha256": sha256_file(SCRIPT),
                "initial_script_snapshot": relative(output / SCRIPT.name),
                "initial_script_snapshot_sha256": sha256_file(output / SCRIPT.name),
                "change_scope": "RNG state device restoration only; model/optimizer/training math unchanged",
                "extension_run_allowed": False,
                "official_imagenette_validation_accessed": False,
            },
        )

    new_history = not history_path.exists()
    history_handle = history_path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(history_handle, fieldnames=fields)
    if new_history:
        writer.writeheader()
        history_handle.flush()
    write_json(
        output / "STATE.json",
        {
            "status": "running_equal_budget_only",
            "arm": args.arm,
            "next_epoch": start_epoch,
            "maximum_authorized_epoch": AUTHORIZED_TOTAL_EPOCHS,
            "extension_run_allowed": False,
        },
    )

    try:
        for epoch_number in range(start_epoch, AUTHORIZED_TOTAL_EPOCHS + 1):
            phase = "main" if epoch_number <= main_epochs else "continuation"
            if epoch_number == main_epochs + 1:
                if not main_best.is_file():
                    raise RuntimeError("continuation cannot start without frozen main best")
                frozen_main = torch.load(main_best, map_location=device, weights_only=False)
                model.load_state_dict(frozen_main["model"], strict=True)
                optimizer, scheduler = make_optimizer_scheduler(
                    model,
                    config["equal_budget_training"]["continuation"],
                    optimizer_steps_per_epoch,
                    continuation_epochs,
                )
                phase_optimizer_steps = 0
                write_json(
                    output / "phase_transition.json",
                    {
                        "from": "main",
                        "to": "continuation",
                        "source_checkpoint": relative(main_best),
                        "source_checkpoint_sha256": sha256_file(main_best),
                        "model_only": True,
                        "fresh_optimizer_scheduler": True,
                        "next_epoch": epoch_number,
                    },
                )

            train_metrics = train_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                device=device,
                snr_generator=snr_generator,
                epoch_number=epoch_number,
                progress_path=output / "live_progress.json",
            )
            phase_optimizer_steps += int(train_metrics["optimizer_steps"])
            global_optimizer_steps += int(train_metrics["optimizer_steps"])
            validation = evaluate(
                model,
                val_loader,
                config,
                device,
                output / "samples",
                epoch_number,
            )
            aggregate = validation["aggregate"]
            finite_values = [
                train_metrics["train_mse"],
                aggregate["mse"],
                aggregate["psnr_db"],
                aggregate["ms_ssim"],
            ]
            if not all(math.isfinite(float(value)) for value in finite_values):
                raise RuntimeError(f"non-finite epoch metrics at epoch {epoch_number}")
            if aggregate["normalized_power_max_abs_error"] > float(
                config["smoke"]["normalized_power_abs_error_max"]
            ):
                raise RuntimeError("channel normalization power gate failed")

            row = {
                "epoch": epoch_number,
                "phase": phase,
                "train_mse": train_metrics["train_mse"],
                "aggregate_psnr_db": aggregate["psnr_db"],
                "aggregate_ms_ssim": aggregate["ms_ssim"],
                "optimizer_steps": train_metrics["optimizer_steps"],
                "phase_optimizer_steps": phase_optimizer_steps,
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
                epoch_number=epoch_number,
                phase=phase,
                phase_optimizer_steps=phase_optimizer_steps,
                global_optimizer_steps=global_optimizer_steps,
                metrics=validation,
                config=config,
                config_sha256=config_sha,
                snr_generator=snr_generator,
                shuffle_generator=shuffle_generator,
            )
            torch.save(payload, latest)
            candidate = (float(aggregate["psnr_db"]), float(aggregate["ms_ssim"]))
            if phase == "main" and candidate > best_main:
                best_main = candidate
                shutil.copy2(latest, main_best)
            if phase == "continuation" and candidate > best_continuation:
                best_continuation = candidate
                shutil.copy2(latest, continuation_best)
                shutil.copy2(latest, final_best)
            state = {
                "status": "running_equal_budget_only",
                "arm": args.arm,
                "completed_epoch": epoch_number,
                "next_epoch": epoch_number + 1,
                "maximum_authorized_epoch": AUTHORIZED_TOTAL_EPOCHS,
                "extension_run_allowed": False,
                "latest_validation": validation,
                "best_main": best_main,
                "best_continuation": best_continuation,
            }
            write_json(output / "STATE.json", state)
            print(json.dumps({"epoch_complete": epoch_number, **row, "validation": validation}), flush=True)
    except BaseException as error:
        write_json(
            output / "STATE.json",
            {
                "status": "failed_or_interrupted",
                "arm": args.arm,
                "error_type": type(error).__name__,
                "error": str(error),
                "maximum_authorized_epoch": AUTHORIZED_TOTAL_EPOCHS,
                "extension_run_allowed": False,
            },
        )
        raise
    finally:
        history_handle.close()

    if not final_best.is_file():
        raise RuntimeError("equal-budget training completed without a continuation best")
    convergence = convergence_assessment(history_path, config)
    best_checkpoint = torch.load(final_best, map_location="cpu", weights_only=False)
    summary = {
        "status": "complete_equal_budget_only",
        "arm": args.arm,
        "epochs_completed": AUTHORIZED_TOTAL_EPOCHS,
        "extension_executed": False,
        "extension_authorized": False,
        "best_epoch": int(best_checkpoint["epoch_number_1based"]),
        "best_metrics": best_checkpoint["metrics"],
        "best_checkpoint": relative(final_best),
        "best_checkpoint_sha256": sha256_file(final_best),
        "main_best_checkpoint": relative(main_best),
        "main_best_checkpoint_sha256": sha256_file(main_best),
        "convergence_assessment": convergence,
        "metadata": metadata,
    }
    write_json(output / "summary.json", summary)
    write_json(
        output / "STATE.json",
        {
            "status": "complete_equal_budget_only_waiting_for_user_extension_decision",
            "arm": args.arm,
            "epochs_completed": AUTHORIZED_TOTAL_EPOCHS,
            "best_epoch": summary["best_epoch"],
            "convergence_assessment": convergence,
            "extension_executed": False,
            "extension_authorized": False,
        },
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
