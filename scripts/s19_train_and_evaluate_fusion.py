#!/usr/bin/env python3
"""Train the paired S19 control/fusion models or evaluate frozen holdout checkpoints."""

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
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.utils import save_image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from cadsd_jscc.diffusion_fusion import (  # noqa: E402
    DualInputResidualRefiner,
    expand_b1_state_dict,
    parameter_count,
    residual_gate_tensor,
)
from cadsd_jscc.metrics import ms_ssim_per_sample, psnr_per_sample  # noqa: E402
from s17_channel_matched_latent_diffusion import classifier_model, classify  # noqa: E402
from s5_residual_refiner_pilot import build_model, try_load_lpips  # noqa: E402


STAGES = ("b0", "diffusion", "b1", "control", "fusion")


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


def snr_name(snr: float) -> str:
    return f"snr_{int(snr):02d}db" if float(snr).is_integer() else f"snr_{snr:g}db"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        return transforms.functional.to_tensor(image.convert("RGB"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class FusionPairDataset(Dataset):
    def __init__(self, config: dict[str, Any], role: str, train: bool) -> None:
        root = resolve(config["inputs"]["population_root"])
        with resolve(config["inputs"]["source_manifest"]).open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            records = [row for row in csv.DictReader(handle) if row["role"] == role]
        expected = int(config["population"]["roles"][role])
        if len(records) != expected:
            raise RuntimeError(f"{role} manifest count changed: {len(records)} != {expected}")
        self.items = [
            (str(row["sample"]), float(snr))
            for snr in config["channel"]["snrs_db"]
            for row in records
        ]
        self.root = root
        self.diffusion_snrs = {
            float(value) for value in config["diffusion"]["cache_diffusion_snrs_db"]
        }
        self.snr_norm_max = float(config["fusion_model"]["snr_norm_max"])
        self.crop_size = int(config["training"]["crop_size"]) if train else 0
        self.random_flip = bool(config["training"]["random_flip"]) if train else False

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        name, snr = self.items[index]
        stem = snr_name(snr)
        b0 = load_rgb(self.root / "exports" / stem / "reconstruction" / name)
        target = load_rgb(self.root / "exports" / "original" / name)
        auxiliary = (
            load_rgb(self.root / "exports" / stem / "identity_diffusion" / name)
            if snr in self.diffusion_snrs
            else b0.clone()
        )
        if self.crop_size:
            _, height, width = b0.shape
            if height < self.crop_size or width < self.crop_size:
                raise RuntimeError("cached image is smaller than training crop")
            top = random.randint(0, height - self.crop_size)
            left = random.randint(0, width - self.crop_size)
            slices = (slice(None), slice(top, top + self.crop_size), slice(left, left + self.crop_size))
            b0, auxiliary, target = b0[slices], auxiliary[slices], target[slices]
        if self.random_flip and random.random() < 0.5:
            b0, auxiliary, target = (value.flip(-1) for value in (b0, auxiliary, target))
        return {
            "b0": b0,
            "auxiliary": auxiliary,
            "target": target,
            "snr_db": torch.tensor(snr, dtype=torch.float32),
            "snr_norm": torch.tensor(snr / self.snr_norm_max, dtype=torch.float32),
            "sample": name,
        }


def validate(config: dict[str, Any], mode: str) -> None:
    expected_status = {
        "train": "cache_frozen_before_training_output",
        "holdout": "models_frozen_before_holdout_output",
    }[mode]
    if config["protocol"]["status"] != expected_status:
        raise RuntimeError(f"S19 config status is not executable for {mode}")
    if config["protocol"].get("official_imagenette_accessed") is not False:
        raise RuntimeError("official Imagenette validation must remain sealed")
    pairs = [
        ("source_manifest", "source_manifest_sha256"),
        ("cache_manifest", "cache_manifest_sha256"),
        ("b1_config", "b1_config_sha256"),
        ("b1_checkpoint", "b1_checkpoint_sha256"),
    ]
    if mode == "holdout":
        pairs.extend(
            [
                ("control_checkpoint", "control_checkpoint_sha256"),
                ("fusion_checkpoint", "fusion_checkpoint_sha256"),
            ]
        )
    for key, hash_key in pairs:
        path = resolve(config["inputs"][key])
        expected = str(config["inputs"][hash_key])
        if expected.startswith("PENDING_") or not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"input hash mismatch: {key}")
    if int(config["fusion_model"]["input_channels"]) != 9:
        raise RuntimeError("fusion input contract changed")


def build_initial_models(
    config: dict[str, Any], device: torch.device
) -> tuple[DualInputResidualRefiner, DualInputResidualRefiner, torch.nn.Module, dict[str, Any]]:
    model_cfg = config["fusion_model"]
    b1_config = yaml.safe_load(resolve(config["inputs"]["b1_config"]).read_text(encoding="utf-8"))
    b1_checkpoint = torch.load(resolve(config["inputs"]["b1_checkpoint"]), map_location="cpu")
    b1_state = b1_checkpoint["model_state_dict"]
    template = DualInputResidualRefiner(
        int(model_cfg["base_channels"]), int(model_cfg["num_blocks"])
    )
    expanded = expand_b1_state_dict(b1_state, template)
    control = DualInputResidualRefiner(
        int(model_cfg["base_channels"]), int(model_cfg["num_blocks"])
    )
    fusion = DualInputResidualRefiner(
        int(model_cfg["base_channels"]), int(model_cfg["num_blocks"])
    )
    control.load_state_dict(expanded, strict=True)
    fusion.load_state_dict(expanded, strict=True)
    if parameter_count(control) != parameter_count(fusion):
        raise RuntimeError("control/fusion parameter counts differ")
    for left, right in zip(control.state_dict().values(), fusion.state_dict().values()):
        if not torch.equal(left, right):
            raise RuntimeError("control/fusion initial states differ")
    b1 = build_model(b1_config)
    b1.load_state_dict(b1_state, strict=True)
    return control.to(device), fusion.to(device), b1.to(device).eval().requires_grad_(False), b1_config


def gates(config: dict[str, Any], snr: torch.Tensor, device: torch.device) -> torch.Tensor:
    return residual_gate_tensor(snr, config["fusion_model"]["residual_gates"], device)


@torch.no_grad()
def selection_evaluate(
    model: DualInputResidualRefiner,
    loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    use_diffusion: bool,
) -> dict[str, Any]:
    model.eval()
    total_psnr = 0.0
    total_mse = 0.0
    count = 0
    per_snr_psnr: defaultdict[float, float] = defaultdict(float)
    per_snr_count: defaultdict[float, int] = defaultdict(int)
    for batch in loader:
        b0 = batch["b0"].to(device, non_blocking=True)
        auxiliary = batch["auxiliary"].to(device, non_blocking=True) if use_diffusion else b0
        target = batch["target"].to(device, non_blocking=True)
        snr = batch["snr_db"].to(device, non_blocking=True)
        output = model(
            b0,
            auxiliary,
            batch["snr_norm"].to(device, non_blocking=True),
            gates(config, snr, device),
        )
        psnr = psnr_per_sample(output, target)
        mse = F.mse_loss(output, target, reduction="none").flatten(1).mean(1)
        total_psnr += float(psnr.sum().cpu())
        total_mse += float(mse.sum().cpu())
        count += output.shape[0]
        for value, score in zip(snr.detach().cpu().tolist(), psnr.detach().cpu().tolist()):
            per_snr_psnr[float(value)] += float(score)
            per_snr_count[float(value)] += 1
    return {
        "mean_psnr": total_psnr / count,
        "mean_mse": total_mse / count,
        "rows": count,
        "per_snr_psnr": {
            str(int(snr)): per_snr_psnr[snr] / per_snr_count[snr]
            for snr in sorted(per_snr_count)
        },
    }


def train_epoch(
    control: DualInputResidualRefiner,
    fusion: DualInputResidualRefiner,
    loader: DataLoader,
    control_optimizer: torch.optim.Optimizer,
    fusion_optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    control.train()
    fusion.train()
    totals: defaultdict[str, float] = defaultdict(float)
    count = 0
    for batch in loader:
        b0 = batch["b0"].to(device, non_blocking=True)
        auxiliary = batch["auxiliary"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        snr = batch["snr_db"].to(device, non_blocking=True)
        snr_norm = batch["snr_norm"].to(device, non_blocking=True)
        gate = gates(config, snr, device)
        batch_size = b0.shape[0]
        for name, model, optimizer, second_input in (
            ("control", control, control_optimizer, b0),
            ("fusion", fusion, fusion_optimizer, auxiliary),
        ):
            optimizer.zero_grad(set_to_none=True)
            output = model(b0, second_input, snr_norm, gate)
            mse = F.mse_loss(output, target)
            l1 = F.l1_loss(output, target)
            loss = (
                float(config["training"]["mse_weight"]) * mse
                + float(config["training"]["l1_weight"]) * l1
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["training"]["grad_clip_norm"])
            )
            optimizer.step()
            totals[f"{name}_loss"] += float(loss.detach().cpu()) * batch_size
            totals[f"{name}_mse"] += float(mse.detach().cpu()) * batch_size
            totals[f"{name}_l1"] += float(l1.detach().cpu()) * batch_size
        count += batch_size
    return {key: value / count for key, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: DualInputResidualRefiner,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    selection: dict[str, Any],
    config: dict[str, Any],
    branch: str,
) -> None:
    torch.save(
        {
            "format_version": 1,
            "experiment_id": config["experiment_id"],
            "branch": branch,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "selection": selection,
            "parameter_count": parameter_count(model),
            "official_imagenette_accessed": False,
            "config": config,
        },
        path,
    )


def initial_equivalence(
    control: DualInputResidualRefiner,
    fusion: DualInputResidualRefiner,
    b1: torch.nn.Module,
    loader: DataLoader,
    config: dict[str, Any],
    b1_config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    batch = next(iter(loader))
    b0 = batch["b0"].to(device)
    auxiliary = batch["auxiliary"].to(device)
    snr = batch["snr_db"].to(device)
    snr_norm = batch["snr_norm"].to(device)
    gate = gates(config, snr, device)
    with torch.no_grad():
        control_output = control(b0, b0, snr_norm, gate)
        fusion_output = fusion(b0, auxiliary, snr_norm, gate)
        b1_output = b1(b0, snr_norm, gate)
    return {
        "control_vs_fusion_max_abs": float(
            (control_output - fusion_output).abs().max().detach().cpu()
        ),
        "control_vs_b1_max_abs": float(
            (control_output - b1_output).abs().max().detach().cpu()
        ),
        "fusion_auxiliary_initial_invariance_max_abs": float(
            (fusion_output - fusion(b0, torch.zeros_like(auxiliary), snr_norm, gate))
            .abs()
            .max()
            .detach()
            .cpu()
        ),
    }


def run_train(config: dict[str, Any], config_path: Path, device: torch.device) -> None:
    validate(config, "train")
    output = resolve(config["outputs"]["training_dir"])
    selection_output = resolve(config["outputs"]["selection_dir"])
    if output.exists() or selection_output.exists():
        raise FileExistsError("S19 training or selection output already exists")
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    selection_output.mkdir(parents=True)
    shutil.copy2(config_path, output / "config_before_training.yaml")
    shutil.copy2(SCRIPT, output / SCRIPT.name)
    seed_everything(int(config["training"]["seed"]))
    train_dataset = FusionPairDataset(config, "train", train=True)
    selection_dataset = FusionPairDataset(config, "selection", train=False)
    generator = torch.Generator().manual_seed(int(config["training"]["seed"]))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(config["training"]["num_workers"]) > 0,
    )
    selection_loader = DataLoader(
        selection_dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["evaluation"]["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(config["evaluation"]["num_workers"]) > 0,
    )
    control, fusion, b1, b1_config = build_initial_models(config, device)
    control_optimizer = torch.optim.Adam(
        control.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    fusion_optimizer = torch.optim.Adam(
        fusion.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    equivalence = initial_equivalence(
        control, fusion, b1, selection_loader, config, b1_config, device
    )
    if equivalence["control_vs_fusion_max_abs"] > 1e-6 or equivalence[
        "control_vs_b1_max_abs"
    ] > 1e-6:
        raise RuntimeError(f"B1 expansion equivalence failed: {equivalence}")
    initial_control = selection_evaluate(control, selection_loader, config, device, False)
    initial_fusion = selection_evaluate(fusion, selection_loader, config, device, True)
    best = {
        "control": {"epoch": 0, **initial_control},
        "fusion": {"epoch": 0, **initial_fusion},
    }
    save_checkpoint(
        checkpoint_dir / "control_best.pt",
        control,
        control_optimizer,
        0,
        initial_control,
        config,
        "control",
    )
    save_checkpoint(
        checkpoint_dir / "fusion_best.pt",
        fusion,
        fusion_optimizer,
        0,
        initial_fusion,
        config,
        "fusion",
    )
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "control_selection_psnr": initial_control["mean_psnr"],
            "fusion_selection_psnr": initial_fusion["mean_psnr"],
            "control_selection_mse": initial_control["mean_mse"],
            "fusion_selection_mse": initial_fusion["mean_mse"],
        }
    ]
    start_time = time.time()
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_stats = train_epoch(
            control,
            fusion,
            train_loader,
            control_optimizer,
            fusion_optimizer,
            config,
            device,
        )
        control_eval = selection_evaluate(control, selection_loader, config, device, False)
        fusion_eval = selection_evaluate(fusion, selection_loader, config, device, True)
        row = {
            "epoch": epoch,
            **train_stats,
            "control_selection_psnr": control_eval["mean_psnr"],
            "fusion_selection_psnr": fusion_eval["mean_psnr"],
            "control_selection_mse": control_eval["mean_mse"],
            "fusion_selection_mse": fusion_eval["mean_mse"],
        }
        history.append(row)
        for branch, model, optimizer, stats in (
            ("control", control, control_optimizer, control_eval),
            ("fusion", fusion, fusion_optimizer, fusion_eval),
        ):
            if float(stats["mean_psnr"]) > float(best[branch]["mean_psnr"]):
                best[branch] = {"epoch": epoch, **stats}
                save_checkpoint(
                    checkpoint_dir / f"{branch}_best.pt",
                    model,
                    optimizer,
                    epoch,
                    stats,
                    config,
                    branch,
                )
        print(json.dumps(row), flush=True)
    write_csv(output / "train_history.csv", history)
    hashes = {
        branch: sha256_file(checkpoint_dir / f"{branch}_best.pt")
        for branch in ("control", "fusion")
    }
    summary = {
        "experiment_id": config["experiment_id"],
        "selection_analysis_id": config["selection_analysis_id"],
        "train_rows_per_epoch": len(train_dataset),
        "selection_rows": len(selection_dataset),
        "parameter_count_control": parameter_count(control),
        "parameter_count_fusion": parameter_count(fusion),
        "initial_equivalence": equivalence,
        "best": best,
        "checkpoint_sha256": hashes,
        "elapsed_seconds": time.time() - start_time,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "official_imagenette_accessed": False,
        "holdout_accessed": False,
    }
    save_json(output / "training_summary.json", summary)
    save_json(selection_output / "selection_summary.json", summary)
    shutil.copy2(config_path, selection_output / "config_before_checkpoint_freeze.yaml")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def load_frozen_model(config: dict[str, Any], branch: str, device: torch.device):
    model_cfg = config["fusion_model"]
    model = DualInputResidualRefiner(
        int(model_cfg["base_channels"]), int(model_cfg["num_blocks"])
    ).to(device)
    checkpoint = torch.load(resolve(config["inputs"][f"{branch}_checkpoint"]), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if int(checkpoint["parameter_count"]) != parameter_count(model):
        raise RuntimeError(f"{branch} checkpoint parameter count changed")
    return model.eval().requires_grad_(False), checkpoint


def summarize_holdout(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "analysis_id": config["holdout_analysis_id"],
        "rows": len(rows),
        "images": len({row["sample"] for row in rows}),
        "snrs_db": sorted({float(row["snr_db"]) for row in rows}),
        "per_snr": [],
    }
    for snr in summary["snrs_db"]:
        subset = [row for row in rows if float(row["snr_db"]) == snr]
        item: dict[str, Any] = {"snr_db": snr, "rows": len(subset)}
        for stage in STAGES:
            for metric in ("psnr", "ms_ssim", "lpips"):
                item[f"mean_{stage}_{metric}"] = sum(
                    float(row[f"{stage}_{metric}"]) for row in subset
                ) / len(subset)
        for stage in STAGES[1:]:
            item[f"{stage}_minus_b0_psnr"] = (
                item[f"mean_{stage}_psnr"] - item["mean_b0_psnr"]
            )
        item["fusion_minus_control_psnr"] = (
            item["mean_fusion_psnr"] - item["mean_control_psnr"]
        )
        item["fusion_minus_b1_psnr"] = item["mean_fusion_psnr"] - item["mean_b1_psnr"]
        summary["per_snr"].append(item)
    for stage in STAGES:
        for metric in ("psnr", "ms_ssim", "lpips"):
            summary[f"mean_{stage}_{metric}"] = sum(
                float(row[f"{stage}_{metric}"]) for row in rows
            ) / len(rows)
    for left, right in (("fusion", "control"), ("fusion", "b1"), ("fusion", "diffusion")):
        for metric in ("psnr", "lpips"):
            summary[f"{left}_minus_{right}_{metric}"] = (
                summary[f"mean_{left}_{metric}"] - summary[f"mean_{right}_{metric}"]
            )
    threshold = float(config["evaluation"]["pseudo_original_confidence_min"])
    eligible = [row for row in rows if float(row["alexnet_original_confidence"]) >= threshold]
    summary["alexnet_eligible_rows"] = len(eligible)
    for stage in STAGES:
        summary[f"alexnet_{stage}_failure"] = sum(
            int(row[f"alexnet_{stage}_prediction"]) != int(row["alexnet_original_prediction"])
            for row in eligible
        )
        summary[f"alexnet_{stage}_new"] = sum(
            int(row["alexnet_b0_prediction"]) == int(row["alexnet_original_prediction"])
            and int(row[f"alexnet_{stage}_prediction"]) != int(row["alexnet_original_prediction"])
            for row in eligible
        )
        summary[f"alexnet_{stage}_repair"] = sum(
            int(row["alexnet_b0_prediction"]) != int(row["alexnet_original_prediction"])
            and int(row[f"alexnet_{stage}_prediction"]) == int(row["alexnet_original_prediction"])
            for row in eligible
        )
        stage_correct = [
            sum(
                int(row[f"{classifier}_{stage}_prediction"])
                == int(row[f"{classifier}_original_prediction"])
                for classifier in ("alexnet", "resnet18", "mobilenet_v3_small")
            )
            >= 2
            for row in rows
        ]
        b0_correct = [
            sum(
                int(row[f"{classifier}_b0_prediction"])
                == int(row[f"{classifier}_original_prediction"])
                for classifier in ("alexnet", "resnet18", "mobilenet_v3_small")
            )
            >= 2
            for row in rows
        ]
        summary[f"majority_{stage}_failure"] = sum(not value for value in stage_correct)
        summary[f"majority_{stage}_new"] = sum(
            base and not candidate for base, candidate in zip(b0_correct, stage_correct)
        )
        summary[f"majority_{stage}_repair"] = sum(
            not base and candidate for base, candidate in zip(b0_correct, stage_correct)
        )
    return summary


@torch.no_grad()
def run_holdout(config: dict[str, Any], config_path: Path, device: torch.device) -> None:
    validate(config, "holdout")
    output = resolve(config["outputs"]["holdout_dir"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    shutil.copy2(config_path, output / "config_before_holdout_access.yaml")
    shutil.copy2(SCRIPT, output / SCRIPT.name)
    seed_everything(int(config["seed"]))
    dataset = FusionPairDataset(config, "holdout", train=False)
    loader = DataLoader(
        dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["evaluation"]["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(config["evaluation"]["num_workers"]) > 0,
    )
    control, control_checkpoint = load_frozen_model(config, "control", device)
    fusion, fusion_checkpoint = load_frozen_model(config, "fusion", device)
    _unused_control, _unused_fusion, b1, _b1_config = build_initial_models(config, device)
    lpips_model, lpips_error = try_load_lpips(device, resolve("outputs/cache"))
    if lpips_model is None:
        raise RuntimeError(lpips_error)
    classifiers = {
        name: classifier_model(name, resolve(config["classifiers"][name]), device)
        for name in ("alexnet", "resnet18", "mobilenet_v3_small")
    }
    mean_tensor = torch.tensor(config["classifiers"]["imagenet_mean"], device=device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(config["classifiers"]["imagenet_std"], device=device).view(1, 3, 1, 1)
    rows: list[dict[str, Any]] = []
    saved_snrs: set[float] = set()
    for batch in loader:
        b0 = batch["b0"].to(device, non_blocking=True)
        diffusion = batch["auxiliary"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        snr = batch["snr_db"].to(device, non_blocking=True)
        snr_norm = batch["snr_norm"].to(device, non_blocking=True)
        gate = gates(config, snr, device)
        candidates = {
            "b0": b0,
            "diffusion": diffusion,
            "b1": b1(b0, snr_norm, gate),
            "control": control(b0, b0, snr_norm, gate),
            "fusion": fusion(b0, diffusion, snr_norm, gate),
        }
        quality = {
            stage: {
                "psnr": psnr_per_sample(image, target),
                "ms_ssim": ms_ssim_per_sample(image, target),
                "lpips": lpips_model(image * 2.0 - 1.0, target * 2.0 - 1.0).flatten(),
            }
            for stage, image in candidates.items()
        }
        predictions: dict[str, dict[str, torch.Tensor]] = {}
        confidences: dict[str, dict[str, torch.Tensor]] = {}
        for name, classifier in classifiers.items():
            predictions[name] = {}
            confidences[name] = {}
            original_prediction, original_confidence = classify(
                classifier, target, mean_tensor, std_tensor
            )
            predictions[name]["original"] = original_prediction
            confidences[name]["original"] = original_confidence
            for stage, image in candidates.items():
                prediction, confidence = classify(classifier, image, mean_tensor, std_tensor)
                predictions[name][stage] = prediction
                confidences[name][stage] = confidence
        batch_snr = float(snr[0].cpu())
        if batch_snr not in saved_snrs:
            count = min(int(config["evaluation"]["sample_grid_count"]), b0.shape[0])
            save_image(
                torch.cat([target[:count], *[candidates[stage][:count] for stage in STAGES]]).cpu(),
                output / f"snr_{int(batch_snr):02d}_fusion_grid.png",
                nrow=count,
            )
            saved_snrs.add(batch_snr)
        for index, sample in enumerate(batch["sample"]):
            row: dict[str, Any] = {
                "analysis_id": config["holdout_analysis_id"],
                "sample": sample,
                "snr_db": float(snr[index].cpu()),
            }
            for stage in STAGES:
                for metric in ("psnr", "ms_ssim", "lpips"):
                    row[f"{stage}_{metric}"] = float(quality[stage][metric][index].cpu())
            for classifier in classifiers:
                for stage in ("original", *STAGES):
                    row[f"{classifier}_{stage}_prediction"] = int(
                        predictions[classifier][stage][index].cpu()
                    )
                    row[f"{classifier}_{stage}_confidence"] = float(
                        confidences[classifier][stage][index].cpu()
                    )
            rows.append(row)
    write_csv(output / "per_sample.csv", rows)
    summary = summarize_holdout(rows, config)
    summary.update(
        {
            "control_epoch": int(control_checkpoint["epoch"]),
            "fusion_epoch": int(fusion_checkpoint["epoch"]),
            "parameter_count_control": parameter_count(control),
            "parameter_count_fusion": parameter_count(fusion),
            "official_imagenette_accessed": False,
        }
    )
    save_json(output / "summary.json", summary)
    save_json(output / "STATE.json", {"state": "HOLDOUT_COMPLETE", **summary})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/s19_diffusion_fusion_ablation.yaml")
    parser.add_argument("--mode", choices=("train", "holdout"), required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    os.environ.setdefault("TORCH_HOME", str(resolve("outputs/cache/torch")))
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if args.mode == "train":
        run_train(config, config_path, device)
    else:
        run_holdout(config, config_path, device)


if __name__ == "__main__":
    main()
