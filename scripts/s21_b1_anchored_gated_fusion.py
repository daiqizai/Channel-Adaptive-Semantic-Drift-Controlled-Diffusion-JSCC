#!/usr/bin/env python3
"""Train, audit, and bootstrap the preregistered S21 B1-anchored fusion."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import save_image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from cadsd_jscc.b1_anchored_fusion import (  # noqa: E402
    B1AnchoredGatedAdapter,
    injection_gate_tensor,
    parameter_count,
)
from cadsd_jscc.metrics import ms_ssim_per_sample, psnr_per_sample  # noqa: E402
from s17_channel_matched_latent_diffusion import classifier_model, classify  # noqa: E402
from s19_train_and_evaluate_fusion import FusionPairDataset  # noqa: E402
from s5_residual_refiner_pilot import build_model, gate_tensor, try_load_lpips  # noqa: E402


STAGES = ("b0", "diffusion", "b1", "control", "fusion")


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("S21 config root must be a mapping")
    parent = payload.pop("extends", None)
    if parent is None:
        return payload
    return deep_merge(load_config(resolve(parent)), payload)


def validate(config: dict[str, Any], mode: str) -> None:
    expected_status = {
        "train": "cache_frozen_before_training_output",
        "holdout": "models_frozen_before_holdout_output",
        "bootstrap": "models_frozen_before_holdout_output",
    }[mode]
    if config["protocol"]["status"] != expected_status:
        raise RuntimeError(f"S21 config status is not executable for {mode}")
    if config["protocol"].get("official_imagenette_accessed") is not False:
        raise RuntimeError("official Imagenette validation must remain sealed")
    hash_pairs = [
        ("source_manifest", "source_manifest_sha256"),
        ("cache_manifest", "cache_manifest_sha256"),
        ("b1_config", "b1_config_sha256"),
        ("b1_checkpoint", "b1_checkpoint_sha256"),
    ]
    if mode in {"holdout", "bootstrap"}:
        hash_pairs.extend(
            [
                ("control_checkpoint", "control_checkpoint_sha256"),
                ("fusion_checkpoint", "fusion_checkpoint_sha256"),
            ]
        )
    for key, hash_key in hash_pairs:
        path = resolve(config["inputs"][key])
        expected = str(config["inputs"][hash_key])
        if expected.startswith("PENDING_") or not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"input hash mismatch: {key}")
    model_cfg = config["fusion_model"]
    if int(model_cfg["input_channels"]) != 12:
        raise RuntimeError("S21 adapter input contract changed")
    exact_snrs = {float(value) for value in model_cfg["exact_anchor_snrs_db"]}
    for value in exact_snrs:
        key = str(int(value)) if value.is_integer() else str(value)
        if float(model_cfg["max_injection_gates"][key]) != 0.0:
            raise RuntimeError("exact-anchor SNR has a nonzero injection gate")
    if int(config["rate"]["active_real_symbols"]) != 19712:
        raise RuntimeError("strict total-rate contract changed")
    if int(config["rate"]["fusion_side_information_real_symbols"]) != 0:
        raise RuntimeError("fusion introduced unmetered side information")


def build_b1(config: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    b1_config = yaml.safe_load(resolve(config["inputs"]["b1_config"]).read_text(encoding="utf-8"))
    checkpoint = torch.load(resolve(config["inputs"]["b1_checkpoint"]), map_location="cpu")
    model = build_model(b1_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval().requires_grad_(False), b1_config


def build_adapter(config: dict[str, Any], device: torch.device) -> B1AnchoredGatedAdapter:
    spec = config["fusion_model"]
    gate_mode = (
        "fixed_one"
        if str(spec.get("spatial_gate_activation", "sigmoid")) == "fixed_one"
        else "learned_sigmoid"
    )
    return B1AnchoredGatedAdapter(
        base_channels=int(spec["base_channels"]),
        num_blocks=int(spec["num_blocks"]),
        spatial_gate_mode=gate_mode,
    ).to(device)


def anchor_output(
    b1: torch.nn.Module,
    b1_config: dict[str, Any],
    b0: torch.Tensor,
    snr: torch.Tensor,
    snr_norm: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    return b1(b0, snr_norm, gate_tensor(b1_config, snr, device))


def max_injection(config: dict[str, Any], snr: torch.Tensor, device: torch.device) -> torch.Tensor:
    return injection_gate_tensor(
        snr, config["fusion_model"]["max_injection_gates"], device
    )


def paired_initial_models(
    config: dict[str, Any], device: torch.device
) -> tuple[B1AnchoredGatedAdapter, B1AnchoredGatedAdapter]:
    control = build_adapter(config, device)
    fusion = build_adapter(config, device)
    fusion.load_state_dict(control.state_dict(), strict=True)
    if parameter_count(control) != parameter_count(fusion):
        raise RuntimeError("control and fusion parameter counts differ")
    for left, right in zip(control.state_dict().values(), fusion.state_dict().values()):
        if not torch.equal(left, right):
            raise RuntimeError("control and fusion initial states differ")
    return control, fusion


@torch.no_grad()
def evaluate_selection(
    model: B1AnchoredGatedAdapter,
    b1: torch.nn.Module,
    b1_config: dict[str, Any],
    loader: DataLoader,
    lpips_model: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device,
    use_diffusion: bool,
) -> dict[str, Any]:
    model.eval()
    totals: defaultdict[str, float] = defaultdict(float)
    per_snr: defaultdict[float, defaultdict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    count = 0
    for batch in loader:
        b0 = batch["b0"].to(device, non_blocking=True)
        diffusion = batch["auxiliary"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        snr = batch["snr_db"].to(device, non_blocking=True)
        snr_norm = batch["snr_norm"].to(device, non_blocking=True)
        anchor = anchor_output(b1, b1_config, b0, snr, snr_norm, device)
        auxiliary = diffusion if use_diffusion else b0
        output, diagnostics = model(
            anchor,
            auxiliary,
            snr_norm,
            max_injection(config, snr, device),
            return_diagnostics=True,
        )
        psnr = psnr_per_sample(output, target)
        lpips = lpips_model(output * 2.0 - 1.0, target * 2.0 - 1.0).flatten()
        mse = F.mse_loss(output, target, reduction="none").flatten(1).mean(1)
        injection = diagnostics["injection"].abs().flatten(1).mean(1)
        gate = diagnostics["spatial_gate"].flatten(1).mean(1)
        exact = (output - anchor).abs().flatten(1).max(1).values
        batch_size = output.shape[0]
        totals["psnr"] += float(psnr.sum().cpu())
        totals["lpips"] += float(lpips.sum().cpu())
        totals["mse"] += float(mse.sum().cpu())
        totals["gate"] += float(gate.sum().cpu())
        totals["injection"] += float(injection.sum().cpu())
        totals["max_anchor_difference"] = max(
            totals["max_anchor_difference"], float(exact.max().cpu())
        )
        count += batch_size
        for index, value in enumerate(snr.detach().cpu().tolist()):
            bucket = per_snr[float(value)]
            bucket["count"] += 1
            bucket["psnr"] += float(psnr[index].cpu())
            bucket["lpips"] += float(lpips[index].cpu())
            bucket["gate"] += float(gate[index].cpu())
            bucket["injection"] += float(injection[index].cpu())
            bucket["max_anchor_difference"] = max(
                bucket["max_anchor_difference"], float(exact[index].cpu())
            )
    return {
        "rows": count,
        "mean_psnr": totals["psnr"] / count,
        "mean_lpips": totals["lpips"] / count,
        "mean_mse": totals["mse"] / count,
        "mean_spatial_gate": totals["gate"] / count,
        "mean_abs_injection": totals["injection"] / count,
        "max_anchor_difference": totals["max_anchor_difference"],
        "per_snr": {
            str(int(snr)): {
                "mean_psnr": values["psnr"] / values["count"],
                "mean_lpips": values["lpips"] / values["count"],
                "mean_spatial_gate": values["gate"] / values["count"],
                "mean_abs_injection": values["injection"] / values["count"],
                "max_anchor_difference": values["max_anchor_difference"],
            }
            for snr, values in sorted(per_snr.items())
        },
    }


def train_epoch(
    control: B1AnchoredGatedAdapter,
    fusion: B1AnchoredGatedAdapter,
    b1: torch.nn.Module,
    b1_config: dict[str, Any],
    loader: DataLoader,
    control_optimizer: torch.optim.Optimizer,
    fusion_optimizer: torch.optim.Optimizer,
    lpips_model: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    control.train()
    fusion.train()
    totals: defaultdict[str, float] = defaultdict(float)
    count = 0
    for batch in loader:
        b0 = batch["b0"].to(device, non_blocking=True)
        diffusion = batch["auxiliary"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        snr = batch["snr_db"].to(device, non_blocking=True)
        snr_norm = batch["snr_norm"].to(device, non_blocking=True)
        with torch.no_grad():
            anchor = anchor_output(b1, b1_config, b0, snr, snr_norm, device)
        injection_limit = max_injection(config, snr, device)
        batch_size = b0.shape[0]
        for name, model, optimizer, auxiliary in (
            ("control", control, control_optimizer, b0),
            ("fusion", fusion, fusion_optimizer, diffusion),
        ):
            optimizer.zero_grad(set_to_none=True)
            output, diagnostics = model(
                anchor,
                auxiliary,
                snr_norm,
                injection_limit,
                return_diagnostics=True,
            )
            mse = F.mse_loss(output, target)
            l1 = F.l1_loss(output, target)
            perceptual = lpips_model(output * 2.0 - 1.0, target * 2.0 - 1.0).mean()
            active = injection_limit > 0
            gate_mean = (
                diagnostics["spatial_gate"][active].mean()
                if bool(active.any())
                else diagnostics["spatial_gate"].mean() * 0.0
            )
            loss = (
                float(config["training"]["mse_weight"]) * mse
                + float(config["training"]["l1_weight"]) * l1
                + float(config["training"]["lpips_weight"]) * perceptual
                + float(config["training"]["spatial_gate_mean_weight"]) * gate_mean
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["training"]["grad_clip_norm"])
            )
            optimizer.step()
            for key, value in (
                ("loss", loss),
                ("mse", mse),
                ("l1", l1),
                ("lpips", perceptual),
                ("gate", gate_mean),
            ):
                totals[f"{name}_{key}"] += float(value.detach().cpu()) * batch_size
        count += batch_size
    return {key: value / count for key, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: B1AnchoredGatedAdapter,
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


def run_train(config: dict[str, Any], config_path: Path, device: torch.device) -> None:
    validate(config, "train")
    output = resolve(config["outputs"]["training_dir"])
    selection_output = resolve(config["outputs"]["selection_dir"])
    if output.exists() or selection_output.exists():
        raise FileExistsError("S21 training or selection output already exists")
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
    b1, b1_config = build_b1(config, device)
    control, fusion = paired_initial_models(config, device)
    control_optimizer = torch.optim.Adam(
        control.parameters(), lr=float(config["training"]["lr"]), weight_decay=0.0
    )
    fusion_optimizer = torch.optim.Adam(
        fusion.parameters(), lr=float(config["training"]["lr"]), weight_decay=0.0
    )
    lpips_model, lpips_error = try_load_lpips(device, resolve("outputs/cache"))
    if lpips_model is None:
        raise RuntimeError(lpips_error)
    lpips_model.eval().requires_grad_(False)
    initial = {
        "control": evaluate_selection(
            control, b1, b1_config, selection_loader, lpips_model, config, device, False
        ),
        "fusion": evaluate_selection(
            fusion, b1, b1_config, selection_loader, lpips_model, config, device, True
        ),
    }
    tolerance = float(config["success_criteria"]["initial_control_equals_fusion_and_b1_max_abs"])
    if initial["control"]["max_anchor_difference"] > tolerance or initial["fusion"][
        "max_anchor_difference"
    ] > tolerance:
        raise RuntimeError(f"initial exact-B1 gate failed: {initial}")
    best = {
        branch: {"epoch": 0, **stats}
        for branch, stats in initial.items()
    }
    for branch, model, optimizer in (
        ("control", control, control_optimizer),
        ("fusion", fusion, fusion_optimizer),
    ):
        save_checkpoint(
            checkpoint_dir / f"{branch}_best.pt",
            model,
            optimizer,
            0,
            initial[branch],
            config,
            branch,
        )
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            **{
                f"{branch}_selection_{metric}": initial[branch][f"mean_{metric}"]
                for branch in ("control", "fusion")
                for metric in ("psnr", "lpips", "mse")
            },
        }
    ]
    start = time.time()
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        training = train_epoch(
            control,
            fusion,
            b1,
            b1_config,
            train_loader,
            control_optimizer,
            fusion_optimizer,
            lpips_model,
            config,
            device,
        )
        evaluations = {
            "control": evaluate_selection(
                control, b1, b1_config, selection_loader, lpips_model, config, device, False
            ),
            "fusion": evaluate_selection(
                fusion, b1, b1_config, selection_loader, lpips_model, config, device, True
            ),
        }
        row: dict[str, Any] = {"epoch": epoch, **training}
        for branch in ("control", "fusion"):
            for metric in ("psnr", "lpips", "mse", "spatial_gate", "abs_injection"):
                row[f"{branch}_selection_{metric}"] = evaluations[branch][f"mean_{metric}"]
            qualifies = evaluations[branch]["mean_lpips"] <= initial[branch]["mean_lpips"] + 1e-12
            if qualifies and evaluations[branch]["mean_psnr"] > best[branch]["mean_psnr"]:
                best[branch] = {"epoch": epoch, **evaluations[branch]}
                model = control if branch == "control" else fusion
                optimizer = control_optimizer if branch == "control" else fusion_optimizer
                save_checkpoint(
                    checkpoint_dir / f"{branch}_best.pt",
                    model,
                    optimizer,
                    epoch,
                    evaluations[branch],
                    config,
                    branch,
                )
        history.append(row)
        print(json.dumps(row), flush=True)
    write_csv(output / "train_history.csv", history)
    hashes = {
        branch: sha256_file(checkpoint_dir / f"{branch}_best.pt")
        for branch in ("control", "fusion")
    }
    summary = {
        "experiment_id": config["experiment_id"],
        "train_rows_per_epoch": len(train_dataset),
        "selection_rows": len(selection_dataset),
        "parameter_count_control": parameter_count(control),
        "parameter_count_fusion": parameter_count(fusion),
        "initial": initial,
        "best": best,
        "checkpoint_sha256": hashes,
        "elapsed_seconds": time.time() - start,
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


def load_adapter(
    config: dict[str, Any], branch: str, device: torch.device
) -> tuple[B1AnchoredGatedAdapter, dict[str, Any]]:
    model = build_adapter(config, device)
    checkpoint = torch.load(resolve(config["inputs"][f"{branch}_checkpoint"]), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if int(checkpoint["parameter_count"]) != parameter_count(model):
        raise RuntimeError(f"{branch} parameter count changed")
    return model.eval().requires_grad_(False), checkpoint


def semantic_summary(rows: list[dict[str, Any]], summary: dict[str, Any], config: dict[str, Any]) -> None:
    threshold = float(config["evaluation"]["pseudo_original_confidence_min"])
    eligible = [row for row in rows if float(row["alexnet_original_confidence"]) >= threshold]
    summary["alexnet_eligible_rows"] = len(eligible)
    for stage in STAGES:
        summary[f"alexnet_{stage}_failure"] = sum(
            int(row[f"alexnet_{stage}_prediction"]) != int(row["alexnet_original_prediction"])
            for row in eligible
        )
        summary[f"alexnet_{stage}_new_vs_b1"] = sum(
            int(row["alexnet_b1_prediction"]) == int(row["alexnet_original_prediction"])
            and int(row[f"alexnet_{stage}_prediction"]) != int(row["alexnet_original_prediction"])
            for row in eligible
        )
        summary[f"alexnet_{stage}_repair_vs_b1"] = sum(
            int(row["alexnet_b1_prediction"]) != int(row["alexnet_original_prediction"])
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
        b1_correct = [
            sum(
                int(row[f"{classifier}_b1_prediction"])
                == int(row[f"{classifier}_original_prediction"])
                for classifier in ("alexnet", "resnet18", "mobilenet_v3_small")
            )
            >= 2
            for row in rows
        ]
        summary[f"majority_{stage}_failure"] = sum(not value for value in stage_correct)
        summary[f"majority_{stage}_new_vs_b1"] = sum(
            anchor and not candidate for anchor, candidate in zip(b1_correct, stage_correct)
        )
        summary[f"majority_{stage}_repair_vs_b1"] = sum(
            not anchor and candidate for anchor, candidate in zip(b1_correct, stage_correct)
        )


def summarize_holdout(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "analysis_id": config["holdout_analysis_id"],
        "rows": len(rows),
        "images": len({str(row["sample"]) for row in rows}),
        "snrs_db": sorted({float(row["snr_db"]) for row in rows}),
        "per_snr": [],
    }
    for snr in summary["snrs_db"]:
        subset = [row for row in rows if float(row["snr_db"]) == snr]
        item: dict[str, Any] = {"snr_db": snr, "rows": len(subset)}
        for stage in STAGES:
            for metric in ("psnr", "ms_ssim", "lpips"):
                item[f"mean_{stage}_{metric}"] = float(
                    np.mean([float(row[f"{stage}_{metric}"]) for row in subset])
                )
        for branch in ("control", "fusion"):
            for metric in ("spatial_gate", "abs_injection", "abs_disagreement"):
                item[f"mean_{branch}_{metric}"] = float(
                    np.mean([float(row[f"{branch}_{metric}"]) for row in subset])
                )
            item[f"max_{branch}_anchor_difference"] = max(
                float(row[f"{branch}_anchor_difference"]) for row in subset
            )
        item["fusion_minus_control_psnr"] = item["mean_fusion_psnr"] - item["mean_control_psnr"]
        item["fusion_minus_control_lpips"] = item["mean_fusion_lpips"] - item["mean_control_lpips"]
        item["fusion_minus_b1_psnr"] = item["mean_fusion_psnr"] - item["mean_b1_psnr"]
        item["fusion_minus_b1_lpips"] = item["mean_fusion_lpips"] - item["mean_b1_lpips"]
        summary["per_snr"].append(item)
    for stage in STAGES:
        for metric in ("psnr", "ms_ssim", "lpips"):
            summary[f"mean_{stage}_{metric}"] = float(
                np.mean([float(row[f"{stage}_{metric}"]) for row in rows])
            )
    for left, right in (("fusion", "control"), ("fusion", "b1"), ("control", "b1")):
        for metric in ("psnr", "lpips"):
            summary[f"{left}_minus_{right}_{metric}"] = (
                summary[f"mean_{left}_{metric}"] - summary[f"mean_{right}_{metric}"]
            )
    semantic_summary(rows, summary, config)
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
    b1, b1_config = build_b1(config, device)
    control, control_checkpoint = load_adapter(config, "control", device)
    fusion, fusion_checkpoint = load_adapter(config, "fusion", device)
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
        anchor = anchor_output(b1, b1_config, b0, snr, snr_norm, device)
        limit = max_injection(config, snr, device)
        control_output, control_diag = control(
            anchor, b0, snr_norm, limit, return_diagnostics=True
        )
        fusion_output, fusion_diag = fusion(
            anchor, diffusion, snr_norm, limit, return_diagnostics=True
        )
        candidates = {
            "b0": b0,
            "diffusion": diffusion,
            "b1": anchor,
            "control": control_output,
            "fusion": fusion_output,
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
            prediction, confidence = classify(classifier, target, mean_tensor, std_tensor)
            predictions[name]["original"] = prediction
            confidences[name]["original"] = confidence
            for stage, image in candidates.items():
                prediction, confidence = classify(classifier, image, mean_tensor, std_tensor)
                predictions[name][stage] = prediction
                confidences[name][stage] = confidence
        batch_snr = float(snr[0].cpu())
        if batch_snr not in saved_snrs:
            count = min(int(config["evaluation"]["sample_grid_count"]), b0.shape[0])
            save_image(
                torch.cat([target[:count], *[candidates[stage][:count] for stage in STAGES]]).cpu(),
                output / f"snr_{int(batch_snr):02d}_b1_anchored_fusion_grid.png",
                nrow=count,
            )
            saved_snrs.add(batch_snr)
        diagnostics = {"control": control_diag, "fusion": fusion_diag}
        for index, sample in enumerate(batch["sample"]):
            row: dict[str, Any] = {
                "analysis_id": config["holdout_analysis_id"],
                "sample": sample,
                "snr_db": float(snr[index].cpu()),
            }
            for stage in STAGES:
                for metric in ("psnr", "ms_ssim", "lpips"):
                    row[f"{stage}_{metric}"] = float(quality[stage][metric][index].cpu())
            for branch, diag in diagnostics.items():
                row[f"{branch}_spatial_gate"] = float(diag["spatial_gate"][index].mean().cpu())
                row[f"{branch}_abs_injection"] = float(diag["injection"][index].abs().mean().cpu())
                row[f"{branch}_abs_disagreement"] = float(diag["disagreement"][index].mean().cpu())
                row[f"{branch}_anchor_difference"] = float(
                    (candidates[branch][index] - anchor[index]).abs().max().cpu()
                )
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


def run_bootstrap(config: dict[str, Any], config_path: Path) -> None:
    validate(config, "bootstrap")
    holdout = resolve(config["outputs"]["holdout_dir"])
    state = json.loads((holdout / "STATE.json").read_text(encoding="utf-8"))
    if state.get("state") != "HOLDOUT_COMPLETE":
        raise RuntimeError("S21 holdout is incomplete")
    with (holdout / "per_sample.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sample"])].append(row)
    names = sorted(grouped)
    expected_images = int(config["population"]["roles"]["holdout"])
    expected_snrs = len(config["channel"]["snrs_db"])
    if len(names) != expected_images or any(len(grouped[name]) != expected_snrs for name in names):
        raise RuntimeError("holdout cluster structure changed")
    comparisons = (("fusion", "control"), ("fusion", "b1"), ("control", "b1"))
    cluster_values: dict[str, dict[str, float]] = {}
    for name in names:
        cluster_values[name] = {
            f"{left}_minus_{right}_{metric}": float(
                np.mean(
                    [
                        float(row[f"{left}_{metric}"]) - float(row[f"{right}_{metric}"])
                        for row in grouped[name]
                    ]
                )
            )
            for left, right in comparisons
            for metric in ("psnr", "lpips")
        }
    rng = np.random.default_rng(int(config["evaluation"]["bootstrap_seed"]))
    replicates = int(config["evaluation"]["bootstrap_replicates"])
    matrix = np.asarray(
        [[cluster_values[name][key] for key in cluster_values[name]] for name in names],
        dtype=np.float64,
    )
    keys = list(cluster_values[names[0]])
    sampled = rng.integers(0, len(names), size=(replicates, len(names)))
    distribution = matrix[sampled].mean(axis=1)
    intervals = {
        key: {
            "mean": float(matrix[:, index].mean()),
            "ci_low": float(np.quantile(distribution[:, index], 0.025)),
            "ci_high": float(np.quantile(distribution[:, index], 0.975)),
        }
        for index, key in enumerate(keys)
    }
    summary = json.loads((holdout / "summary.json").read_text(encoding="utf-8"))
    criteria = config["success_criteria"]
    low_snrs = {1.0, 4.0, 7.0}
    exact_snrs = {13.0, 19.0}
    checks = {
        "primary_fusion_minus_control_psnr_ci_low": intervals[
            "fusion_minus_control_psnr"
        ]["ci_low"]
        > float(criteria["primary_fusion_minus_control_psnr_ci_low_min_db"]),
        "primary_fusion_minus_control_lpips_ci_high": intervals[
            "fusion_minus_control_lpips"
        ]["ci_high"]
        < float(criteria["primary_fusion_minus_control_lpips_ci_high_max"]),
        "fusion_minus_control_nonnegative_low_snr_count": sum(
            float(item["fusion_minus_control_psnr"]) >= 0
            for item in summary["per_snr"]
            if float(item["snr_db"]) in low_snrs
        )
        >= int(criteria["fusion_minus_control_nonnegative_low_snr_count_min"]),
        "fusion_minus_b1_mean_psnr": float(summary["fusion_minus_b1_psnr"])
        > float(criteria["fusion_minus_b1_mean_psnr_min_db"]),
        "fusion_minus_b1_nonnegative_all_snr_count": sum(
            float(item["fusion_minus_b1_psnr"]) >= -1e-12 for item in summary["per_snr"]
        )
        >= int(criteria["fusion_minus_b1_nonnegative_all_snr_count_min"]),
        "high_snr_exact_b1": max(
            max(
                float(item["max_control_anchor_difference"]),
                float(item["max_fusion_anchor_difference"]),
            )
            for item in summary["per_snr"]
            if float(item["snr_db"]) in exact_snrs
        )
        <= float(criteria["high_snr_control_and_fusion_equal_b1_max_abs"]),
        "fusion_majority_new_not_greater_than_repair": int(
            summary["majority_fusion_new_vs_b1"]
        )
        <= int(summary["majority_fusion_repair_vs_b1"]),
        "fusion_majority_new_not_greater_than_control": int(
            summary["majority_fusion_new_vs_b1"]
        )
        <= int(summary["majority_control_new_vs_b1"]),
    }
    result = {
        "analysis_id": config["bootstrap_analysis_id"],
        "holdout_per_sample_sha256": sha256_file(holdout / "per_sample.csv"),
        "bootstrap_unit": "source_image_cluster_across_five_snrs",
        "clusters": len(names),
        "replicates": replicates,
        "seed": int(config["evaluation"]["bootstrap_seed"]),
        "intervals": intervals,
        "checks": checks,
        "pass_count": sum(checks.values()),
        "check_count": len(checks),
        "primary_quality_fusion_demonstrated": checks[
            "primary_fusion_minus_control_psnr_ci_low"
        ]
        and checks["primary_fusion_minus_control_lpips_ci_high"],
        "official_imagenette_accessed": False,
    }
    output = resolve(config["outputs"]["bootstrap_dir"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    shutil.copy2(config_path, output / "config.yaml")
    shutil.copy2(SCRIPT, output / SCRIPT.name)
    save_json(output / "bootstrap_summary.json", result)
    save_json(output / "STATE.json", {"state": "COMPLETE", **result})
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/s21_b1_anchored_gated_fusion.yaml")
    parser.add_argument("--mode", choices=("train", "holdout", "bootstrap"), required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    os.environ.setdefault("TORCH_HOME", str(resolve("outputs/cache/torch")))
    config_path = resolve(args.config)
    config = load_config(config_path)
    if args.mode == "train":
        run_train(config, config_path, torch.device(args.device))
    elif args.mode == "holdout":
        run_holdout(config, config_path, torch.device(args.device))
    else:
        run_bootstrap(config, config_path)


if __name__ == "__main__":
    main()
