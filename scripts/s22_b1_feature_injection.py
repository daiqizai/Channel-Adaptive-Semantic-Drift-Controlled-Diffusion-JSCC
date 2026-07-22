#!/usr/bin/env python3
"""Train and audit the preregistered S22 frozen-B1 feature injection."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from cadsd_jscc.b1_feature_injection import (  # noqa: E402
    FrozenB1FeatureInjection,
    envelope_tensor,
    trainable_parameter_count,
)
from cadsd_jscc.metrics import ms_ssim_per_sample, psnr_per_sample  # noqa: E402
from s17_channel_matched_latent_diffusion import classifier_model, classify  # noqa: E402
from s19_train_and_evaluate_fusion import FusionPairDataset  # noqa: E402
from s21_b1_anchored_gated_fusion import (  # noqa: E402
    anchor_output,
    build_b1,
    load_config,
    resolve,
    save_json,
    seed_everything,
    sha256_file,
    write_csv,
)
from s5_residual_refiner_pilot import try_load_lpips  # noqa: E402


STAGES = ("b0", "diffusion", "b1", "control", "fusion")


def validate(config: dict[str, Any], mode: str) -> None:
    expected_status = {
        "smoke": "cache_frozen_before_training_output",
        "train": "cache_frozen_before_training_output",
        "holdout": "models_frozen_before_holdout_output",
        "bootstrap": "models_frozen_before_holdout_output",
    }[mode]
    if config["protocol"]["status"] != expected_status:
        raise RuntimeError(f"S22 config status is not executable for {mode}")
    if config["protocol"].get("official_imagenette_accessed") is not False:
        raise RuntimeError("official Imagenette validation must remain sealed")
    if config["protocol"].get("b1_backbone_frozen") is not True:
        raise RuntimeError("S22 requires a frozen B1 backbone")
    hash_pairs = [
        ("source_manifest", "source_manifest_sha256"),
        ("cache_manifest", "cache_manifest_sha256"),
        ("b1_config", "b1_config_sha256"),
        ("b1_checkpoint", "b1_checkpoint_sha256"),
    ]
    if mode in {"holdout", "bootstrap"}:
        hash_pairs.append(
            ("feature_injection_checkpoint", "feature_injection_checkpoint_sha256")
        )
    for key, hash_key in hash_pairs:
        path = resolve(config["inputs"][key])
        expected = str(config["inputs"][hash_key])
        if expected.startswith("PENDING_") or not path.is_file():
            raise RuntimeError(f"missing frozen S22 input: {key}")
        if sha256_file(path) != expected:
            raise RuntimeError(f"input hash mismatch: {key}")
    spec = config["feature_injection"]
    if int(spec["auxiliary_channels"]) != 3 or int(spec["target_feature_channels"]) != 64:
        raise RuntimeError("S22 projection contract changed")
    if int(spec["trainable_parameters_expected"]) != 1728:
        raise RuntimeError("S22 parameter budget changed")
    for snr in (13, 19):
        if float(spec["envelope"][str(snr)]) != 0.0:
            raise RuntimeError("high-SNR exact-B1 envelope changed")
    if int(config["rate"]["active_real_symbols"]) != 19712:
        raise RuntimeError("strict total-rate contract changed")
    if int(config["rate"]["fusion_side_information_real_symbols"]) != 0:
        raise RuntimeError("feature injection introduced unmetered side information")


def build_feature_model(
    config: dict[str, Any], device: torch.device
) -> tuple[FrozenB1FeatureInjection, dict[str, Any]]:
    b1, b1_config = build_b1(config, device)
    model = FrozenB1FeatureInjection(
        b1, feature_channels=int(config["feature_injection"]["target_feature_channels"])
    ).to(device)
    expected = int(config["feature_injection"]["trainable_parameters_expected"])
    if trainable_parameter_count(model) != expected:
        raise RuntimeError("unexpected S22 trainable parameter count")
    if int(model.b1.head[0].out_channels) != int(
        config["feature_injection"]["target_feature_channels"]
    ):
        raise RuntimeError("B1 head width changed")
    if any(parameter.requires_grad for parameter in model.b1.parameters()):
        raise RuntimeError("B1 was not fully frozen")
    return model, b1_config


def envelopes(config: dict[str, Any], snr: torch.Tensor, device: torch.device) -> torch.Tensor:
    return envelope_tensor(snr, config["feature_injection"]["envelope"], device)


@torch.no_grad()
def evaluate_selection(
    model: FrozenB1FeatureInjection,
    b1_config: dict[str, Any],
    loader: DataLoader,
    lpips_model: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    totals: defaultdict[str, float] = defaultdict(float)
    buckets: defaultdict[float, defaultdict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    count = 0
    for batch in loader:
        b0 = batch["b0"].to(device, non_blocking=True)
        diffusion = batch["auxiliary"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        snr = batch["snr_db"].to(device, non_blocking=True)
        snr_norm = batch["snr_norm"].to(device, non_blocking=True)
        gate = anchor_output(model.b1, b1_config, b0, snr, snr_norm, device)
        output = model(
            b0,
            diffusion,
            snr_norm,
            _b1_gate(b1_config, snr, device),
            envelopes(config, snr, device),
        )
        projected = envelopes(config, snr, device).view(-1, 1, 1, 1) * model.aux_projection(
            diffusion - b0
        )
        metrics = {}
        for name, image in (("b1", gate), ("fusion", output)):
            metrics[name] = {
                "psnr": psnr_per_sample(image, target),
                "lpips": lpips_model(image * 2.0 - 1.0, target * 2.0 - 1.0).flatten(),
                "mse": F.mse_loss(image, target, reduction="none").flatten(1).mean(1),
            }
        exact = (output - gate).abs().flatten(1).max(1).values
        injection = projected.abs().flatten(1).mean(1)
        batch_size = b0.shape[0]
        count += batch_size
        for name in ("b1", "fusion"):
            for metric in ("psnr", "lpips", "mse"):
                totals[f"{name}_{metric}"] += float(metrics[name][metric].sum().cpu())
        totals["abs_injection"] += float(injection.sum().cpu())
        totals["max_b1_difference"] = max(
            totals["max_b1_difference"], float(exact.max().cpu())
        )
        for index, value in enumerate(snr.detach().cpu().tolist()):
            bucket = buckets[float(value)]
            bucket["count"] += 1
            for name in ("b1", "fusion"):
                for metric in ("psnr", "lpips"):
                    bucket[f"{name}_{metric}"] += float(metrics[name][metric][index].cpu())
            bucket["abs_injection"] += float(injection[index].cpu())
            bucket["max_b1_difference"] = max(
                bucket["max_b1_difference"], float(exact[index].cpu())
            )
    result: dict[str, Any] = {"rows": count}
    for name in ("b1", "fusion"):
        for metric in ("psnr", "lpips", "mse"):
            result[f"mean_{name}_{metric}"] = totals[f"{name}_{metric}"] / count
    result["fusion_minus_b1_psnr"] = result["mean_fusion_psnr"] - result["mean_b1_psnr"]
    result["fusion_minus_b1_lpips"] = result["mean_fusion_lpips"] - result["mean_b1_lpips"]
    result["mean_abs_feature_injection"] = totals["abs_injection"] / count
    result["max_b1_difference"] = totals["max_b1_difference"]
    result["per_snr"] = {}
    for snr, values in sorted(buckets.items()):
        rows = values["count"]
        item = {
            f"mean_{name}_{metric}": values[f"{name}_{metric}"] / rows
            for name in ("b1", "fusion")
            for metric in ("psnr", "lpips")
        }
        item["fusion_minus_b1_psnr"] = item["mean_fusion_psnr"] - item["mean_b1_psnr"]
        item["fusion_minus_b1_lpips"] = item["mean_fusion_lpips"] - item["mean_b1_lpips"]
        item["mean_abs_feature_injection"] = values["abs_injection"] / rows
        item["max_b1_difference"] = values["max_b1_difference"]
        result["per_snr"][str(int(snr))] = item
    return result


def _b1_gate(
    b1_config: dict[str, Any], snr: torch.Tensor, device: torch.device
) -> torch.Tensor:
    # Kept local so S22 has one explicit forward contract in both train and audit.
    from s5_residual_refiner_pilot import gate_tensor

    return gate_tensor(b1_config, snr, device)


def train_epoch(
    model: FrozenB1FeatureInjection,
    b1_config: dict[str, Any],
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lpips_model: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals: defaultdict[str, float] = defaultdict(float)
    count = 0
    for batch in loader:
        b0 = batch["b0"].to(device, non_blocking=True)
        diffusion = batch["auxiliary"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        snr = batch["snr_db"].to(device, non_blocking=True)
        snr_norm = batch["snr_norm"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        output = model(
            b0,
            diffusion,
            snr_norm,
            _b1_gate(b1_config, snr, device),
            envelopes(config, snr, device),
        )
        mse = F.mse_loss(output, target)
        l1 = F.l1_loss(output, target)
        perceptual = lpips_model(output * 2.0 - 1.0, target * 2.0 - 1.0).mean()
        loss = (
            float(config["training"]["mse_weight"]) * mse
            + float(config["training"]["l1_weight"]) * l1
            + float(config["training"]["lpips_weight"]) * perceptual
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [model.aux_projection.weight], float(config["training"]["grad_clip_norm"])
        )
        optimizer.step()
        batch_size = b0.shape[0]
        count += batch_size
        for key, value in (("loss", loss), ("mse", mse), ("l1", l1), ("lpips", perceptual)):
            totals[key] += float(value.detach().cpu()) * batch_size
    return {key: value / count for key, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: FrozenB1FeatureInjection,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    selection: dict[str, Any],
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "format_version": 1,
            "experiment_id": config["experiment_id"],
            "epoch": epoch,
            "projection_state_dict": model.aux_projection.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "selection": selection,
            "trainable_parameter_count": trainable_parameter_count(model),
            "b1_checkpoint_sha256": config["inputs"]["b1_checkpoint_sha256"],
            "official_imagenette_accessed": False,
            "config": config,
        },
        path,
    )


def make_loaders(
    config: dict[str, Any], device: torch.device
) -> tuple[FusionPairDataset, FusionPairDataset, DataLoader, DataLoader]:
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
    return train_dataset, selection_dataset, train_loader, selection_loader


def run_smoke(config: dict[str, Any], device: torch.device) -> None:
    validate(config, "smoke")
    seed_everything(int(config["training"]["seed"]))
    dataset = FusionPairDataset(config, "train", train=True)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    model, b1_config = build_feature_model(config, device)
    lpips_model, lpips_error = try_load_lpips(device, resolve("outputs/cache"))
    if lpips_model is None:
        raise RuntimeError(lpips_error)
    lpips_model.eval().requires_grad_(False)
    batch = next(iter(loader))
    b0 = batch["b0"].to(device)
    diffusion = batch["auxiliary"].to(device)
    target = batch["target"].to(device)
    snr = batch["snr_db"].to(device)
    snr_norm = batch["snr_norm"].to(device)
    b1 = anchor_output(model.b1, b1_config, b0, snr, snr_norm, device)
    output = model(
        b0,
        diffusion,
        snr_norm,
        _b1_gate(b1_config, snr, device),
        envelopes(config, snr, device),
    )
    loss = F.mse_loss(output, target) + 0.01 * lpips_model(
        output * 2.0 - 1.0, target * 2.0 - 1.0
    ).mean()
    loss.backward()
    gradient = model.aux_projection.weight.grad
    result = {
        "rows": b0.shape[0],
        "initial_max_b1_difference": float((output - b1).abs().max().detach().cpu()),
        "projection_gradient_finite": gradient is not None and bool(torch.isfinite(gradient).all()),
        "projection_gradient_l1": float(gradient.abs().sum().cpu()) if gradient is not None else 0.0,
        "trainable_parameter_count": trainable_parameter_count(model),
    }
    if result["initial_max_b1_difference"] > float(
        config["success_criteria"]["initial_fusion_equals_b1_max_abs"]
    ):
        raise RuntimeError("S22 initial exact-B1 smoke check failed")
    if not result["projection_gradient_finite"] or result["projection_gradient_l1"] <= 0:
        raise RuntimeError("S22 projection did not receive a usable gradient")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_train(config: dict[str, Any], config_path: Path, device: torch.device) -> None:
    validate(config, "train")
    output = resolve(config["outputs"]["training_dir"])
    selection_output = resolve(config["outputs"]["selection_dir"])
    if output.exists() or selection_output.exists():
        raise FileExistsError("S22 training or selection output already exists")
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    selection_output.mkdir(parents=True)
    shutil.copy2(config_path, output / "config_before_training.yaml")
    shutil.copy2(SCRIPT, output / SCRIPT.name)
    seed_everything(int(config["training"]["seed"]))
    train_dataset, selection_dataset, train_loader, selection_loader = make_loaders(config, device)
    model, b1_config = build_feature_model(config, device)
    optimizer = torch.optim.Adam(
        [model.aux_projection.weight],
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    lpips_model, lpips_error = try_load_lpips(device, resolve("outputs/cache"))
    if lpips_model is None:
        raise RuntimeError(lpips_error)
    lpips_model.eval().requires_grad_(False)
    initial = evaluate_selection(
        model, b1_config, selection_loader, lpips_model, config, device
    )
    if initial["max_b1_difference"] > float(
        config["success_criteria"]["initial_fusion_equals_b1_max_abs"]
    ):
        raise RuntimeError(f"initial S22 exact-B1 check failed: {initial}")
    best = {"epoch": 0, **initial}
    checkpoint_path = checkpoint_dir / "best.pt"
    save_checkpoint(checkpoint_path, model, optimizer, 0, initial, config)
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            **{key: value for key, value in initial.items() if key != "per_snr"},
        }
    ]
    start = time.time()
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        training = train_epoch(
            model, b1_config, train_loader, optimizer, lpips_model, config, device
        )
        selection = evaluate_selection(
            model, b1_config, selection_loader, lpips_model, config, device
        )
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in training.items()},
            **{key: value for key, value in selection.items() if key != "per_snr"},
        }
        history.append(row)
        qualifies = selection["mean_fusion_lpips"] <= initial["mean_b1_lpips"] + 1e-12
        if qualifies and selection["mean_fusion_psnr"] > best["mean_fusion_psnr"]:
            best = {"epoch": epoch, **selection}
            save_checkpoint(checkpoint_path, model, optimizer, epoch, selection, config)
        print(json.dumps(row), flush=True)
    write_csv(output / "train_history.csv", history)
    checkpoint_hash = sha256_file(checkpoint_path)
    summary = {
        "experiment_id": config["experiment_id"],
        "train_rows_per_epoch": len(train_dataset),
        "selection_rows": len(selection_dataset),
        "trainable_parameter_count": trainable_parameter_count(model),
        "initial": initial,
        "best": best,
        "selected_nonzero_training_epoch": int(best["epoch"]) > 0,
        "checkpoint_sha256": checkpoint_hash,
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


def load_feature_model(
    config: dict[str, Any], device: torch.device
) -> tuple[FrozenB1FeatureInjection, dict[str, Any], dict[str, Any]]:
    model, b1_config = build_feature_model(config, device)
    checkpoint = torch.load(
        resolve(config["inputs"]["feature_injection_checkpoint"]), map_location=device
    )
    if int(checkpoint["epoch"]) <= 0:
        raise RuntimeError("S22 holdout cannot use the epoch-zero no-op checkpoint")
    if int(checkpoint["trainable_parameter_count"]) != trainable_parameter_count(model):
        raise RuntimeError("S22 checkpoint parameter budget changed")
    if checkpoint["b1_checkpoint_sha256"] != config["inputs"]["b1_checkpoint_sha256"]:
        raise RuntimeError("S22 checkpoint was trained on another B1 anchor")
    model.aux_projection.load_state_dict(checkpoint["projection_state_dict"], strict=True)
    return model.eval().requires_grad_(False), b1_config, checkpoint


def semantic_summary(
    rows: list[dict[str, Any]], summary: dict[str, Any], config: dict[str, Any]
) -> None:
    classifiers = ("alexnet", "resnet18", "mobilenet_v3_small")
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
        candidate_correct = [
            sum(
                int(row[f"{classifier}_{stage}_prediction"])
                == int(row[f"{classifier}_original_prediction"])
                for classifier in classifiers
            )
            >= 2
            for row in rows
        ]
        b1_correct = [
            sum(
                int(row[f"{classifier}_b1_prediction"])
                == int(row[f"{classifier}_original_prediction"])
                for classifier in classifiers
            )
            >= 2
            for row in rows
        ]
        summary[f"majority_{stage}_failure"] = sum(not value for value in candidate_correct)
        summary[f"majority_{stage}_new_vs_b1"] = sum(
            anchor and not candidate
            for anchor, candidate in zip(b1_correct, candidate_correct)
        )
        summary[f"majority_{stage}_repair_vs_b1"] = sum(
            not anchor and candidate
            for anchor, candidate in zip(b1_correct, candidate_correct)
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
        item["mean_abs_feature_injection"] = float(
            np.mean([float(row["abs_feature_injection"]) for row in subset])
        )
        item["max_fusion_b1_difference"] = max(
            float(row["fusion_b1_difference"]) for row in subset
        )
        item["max_control_b1_difference"] = max(
            float(row["control_b1_difference"]) for row in subset
        )
        item["fusion_minus_b1_psnr"] = item["mean_fusion_psnr"] - item["mean_b1_psnr"]
        item["fusion_minus_b1_lpips"] = item["mean_fusion_lpips"] - item["mean_b1_lpips"]
        summary["per_snr"].append(item)
    for stage in STAGES:
        for metric in ("psnr", "ms_ssim", "lpips"):
            summary[f"mean_{stage}_{metric}"] = float(
                np.mean([float(row[f"{stage}_{metric}"]) for row in rows])
            )
    for left, right in (("fusion", "b1"), ("control", "b1"), ("fusion", "control")):
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
    model, b1_config, checkpoint = load_feature_model(config, device)
    lpips_model, lpips_error = try_load_lpips(device, resolve("outputs/cache"))
    if lpips_model is None:
        raise RuntimeError(lpips_error)
    lpips_model.eval().requires_grad_(False)
    classifiers = {
        name: classifier_model(name, resolve(config["classifiers"][name]), device)
        for name in ("alexnet", "resnet18", "mobilenet_v3_small")
    }
    mean_tensor = torch.tensor(config["classifiers"]["imagenet_mean"], device=device).view(
        1, 3, 1, 1
    )
    std_tensor = torch.tensor(config["classifiers"]["imagenet_std"], device=device).view(
        1, 3, 1, 1
    )
    rows: list[dict[str, Any]] = []
    saved_snrs: set[float] = set()
    for batch in loader:
        b0 = batch["b0"].to(device, non_blocking=True)
        diffusion = batch["auxiliary"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        snr = batch["snr_db"].to(device, non_blocking=True)
        snr_norm = batch["snr_norm"].to(device, non_blocking=True)
        gate = _b1_gate(b1_config, snr, device)
        anchor = anchor_output(model.b1, b1_config, b0, snr, snr_norm, device)
        envelope = envelopes(config, snr, device)
        control = model(b0, b0, snr_norm, gate, envelope)
        fusion = model(b0, diffusion, snr_norm, gate, envelope)
        projected = envelope.view(-1, 1, 1, 1) * model.aux_projection(diffusion - b0)
        candidates = {
            "b0": b0,
            "diffusion": diffusion,
            "b1": anchor,
            "control": control,
            "fusion": fusion,
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
                torch.cat(
                    [target[:count], *[candidates[stage][:count] for stage in STAGES]]
                ).cpu(),
                output / f"snr_{int(batch_snr):02d}_b1_feature_injection_grid.png",
                nrow=count,
            )
            saved_snrs.add(batch_snr)
        for index, sample in enumerate(batch["sample"]):
            row: dict[str, Any] = {
                "analysis_id": config["holdout_analysis_id"],
                "sample": sample,
                "snr_db": float(snr[index].cpu()),
                "abs_feature_injection": float(projected[index].abs().mean().cpu()),
                "fusion_b1_difference": float((fusion[index] - anchor[index]).abs().max().cpu()),
                "control_b1_difference": float((control[index] - anchor[index]).abs().max().cpu()),
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
            "selected_epoch": int(checkpoint["epoch"]),
            "trainable_parameter_count": int(checkpoint["trainable_parameter_count"]),
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
        raise RuntimeError("S22 holdout is incomplete")
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
    keys = ("fusion_minus_b1_psnr", "fusion_minus_b1_lpips")
    matrix = np.asarray(
        [
            [
                np.mean(
                    [
                        float(row[f"fusion_{metric}"]) - float(row[f"b1_{metric}"])
                        for row in grouped[name]
                    ]
                )
                for metric in ("psnr", "lpips")
            ]
            for name in names
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(int(config["evaluation"]["bootstrap_seed"]))
    replicates = int(config["evaluation"]["bootstrap_replicates"])
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
        "fusion_minus_b1_psnr_ci_low": intervals["fusion_minus_b1_psnr"]["ci_low"]
        > float(criteria["fusion_minus_b1_psnr_ci_low_min_db"]),
        "fusion_minus_b1_lpips_ci_high": intervals["fusion_minus_b1_lpips"]["ci_high"]
        < float(criteria["fusion_minus_b1_lpips_ci_high_max"]),
        "fusion_minus_b1_nonnegative_low_snr_count": sum(
            float(item["fusion_minus_b1_psnr"]) >= 0
            for item in summary["per_snr"]
            if float(item["snr_db"]) in low_snrs
        )
        >= int(criteria["fusion_minus_b1_nonnegative_low_snr_count_min"]),
        "high_snr_exact_b1": max(
            max(
                float(item["max_fusion_b1_difference"]),
                float(item["max_control_b1_difference"]),
            )
            for item in summary["per_snr"]
            if float(item["snr_db"]) in exact_snrs
        )
        <= float(criteria["high_snr_fusion_equals_b1_max_abs"]),
        "fusion_majority_new_not_greater_than_repair": int(
            summary["majority_fusion_new_vs_b1"]
        )
        <= int(summary["majority_fusion_repair_vs_b1"]),
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
        "all_preregistered_checks_passed": all(checks.values()),
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
    parser.add_argument("--config", default="configs/s22_b1_feature_injection.yaml")
    parser.add_argument(
        "--mode", choices=("smoke", "train", "holdout", "bootstrap"), required=True
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    os.environ.setdefault("TORCH_HOME", str(resolve("outputs/cache/torch")))
    config_path = resolve(args.config)
    config = load_config(config_path)
    if args.mode == "smoke":
        run_smoke(config, torch.device(args.device))
    elif args.mode == "train":
        run_train(config, config_path, torch.device(args.device))
    elif args.mode == "holdout":
        run_holdout(config, config_path, torch.device(args.device))
    else:
        run_bootstrap(config, config_path)


if __name__ == "__main__":
    main()
