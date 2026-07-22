#!/usr/bin/env python3
"""Train/evaluate a short-chain residual-shift diffusion from a frozen CNN anchor."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cadsd_jscc.metrics import psnr_per_sample  # noqa: E402
from cadsd_jscc.structure import structural_feature_maps  # noqa: E402
from s5_residual_refiner_pilot import (  # noqa: E402
    ResidualBlock,
    ResidualPairDataset,
    aggregate_results,
    build_model as build_anchor_model,
    evaluate_snr,
    gate_tensor,
    get_project_version,
    load_classifier,
    mean,
    parse_snrs,
    project_relative,
    refine_and_save_snr,
    resolve_device,
    resolve_project_path,
    save_json,
    semantic_distillation_loss,
    try_load_lpips,
    validate_inputs,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/s10_short_chain_residual_shift_diffusion_pilot.yaml"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--train-count", type=int, default=None)
    parser.add_argument("--eval-count", type=int, default=None)
    parser.add_argument("--snrs", default=None, help="Comma-separated override for smoke runs")
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


class ResidualShiftDenoiser(nn.Module):
    """Predict a bounded target correction from a stochastic anchor-to-target bridge state."""

    def __init__(self, input_channels: int, base_channels: int, num_blocks: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(
        self,
        bridge_state: torch.Tensor,
        anchor: torch.Tensor,
        structure: torch.Tensor,
        snr_norm: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, height, width = anchor.shape
        if (
            structure.ndim != 4
            or structure.shape[0] != anchor.shape[0]
            or structure.shape[1] < 2
            or structure.shape[-2:] != anchor.shape[-2:]
        ):
            raise ValueError(
                "Structural condition must match anchor batch/spatial shape and have >=2 channels"
            )
        snr_map = snr_norm.view(batch, 1, 1, 1).expand(batch, 1, height, width)
        tau_map = tau.view(batch, 1, 1, 1).expand(batch, 1, height, width)
        model_input = torch.cat(
            [bridge_state, anchor, snr_map, tau_map, structure[:, :2]], dim=1
        )
        return torch.tanh(self.tail(self.body(self.head(model_input))))


class ShortChainResidualShiftDiffusion(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        diffusion = config["diffusion"]
        expected_channels = 3 + 3 + 1 + 1 + 2
        if int(model["input_channels"]) != expected_channels:
            raise ValueError(
                f"model.input_channels must be {expected_channels}, got {model['input_channels']}"
            )
        self.train_timesteps = int(diffusion["train_timesteps"])
        self.sampling_steps = int(diffusion["sampling_steps"])
        if self.train_timesteps < 2 or self.sampling_steps < 1:
            raise ValueError("Diffusion train_timesteps>=2 and sampling_steps>=1 are required")
        self.bridge_noise_sigma = float(diffusion["bridge_noise_sigma"])
        self.condition_source = str(
            model.get(
                "diffusion_condition_source",
                model.get("condition_source", "decoded_structure_rgb"),
            )
        )
        supported = {"decoded_structure_rgb", "receiver_anchor_structural_maps"}
        if self.condition_source not in supported:
            raise ValueError(
                f"Unsupported diffusion condition source {self.condition_source!r}; "
                f"supported={sorted(supported)}"
            )
        self.denoiser = ResidualShiftDenoiser(
            expected_channels,
            int(model["base_channels"]),
            int(model["num_blocks"]),
        )

    @staticmethod
    def bridge_state(
        target: torch.Tensor,
        anchor: torch.Tensor,
        tau: torch.Tensor,
        noise: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        weight = tau.view(-1, 1, 1, 1)
        stochastic_scale = float(sigma) * torch.sqrt((weight * (1.0 - weight)).clamp_min(0.0))
        return (1.0 - weight) * target + weight * anchor + stochastic_scale * noise

    def predict_target(
        self,
        bridge_state: torch.Tensor,
        anchor: torch.Tensor,
        structure: torch.Tensor,
        snr_norm: torch.Tensor,
        tau: torch.Tensor,
        correction_gate: torch.Tensor,
    ) -> torch.Tensor:
        correction = self.denoiser(bridge_state, anchor, structure, snr_norm, tau)
        return (anchor + correction_gate.view(-1, 1, 1, 1) * correction).clamp(0.0, 1.0)

    def forward(
        self,
        m0: torch.Tensor,
        snr_norm: torch.Tensor,
        residual_gate_value: torch.Tensor,
        condition_image: torch.Tensor | None = None,
        semantic_sketch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del semantic_sketch
        anchor = m0
        if self.condition_source == "receiver_anchor_structural_maps":
            if condition_image is not None:
                raise ValueError(
                    "receiver_anchor_structural_maps must not receive an external condition image"
                )
            condition_image = structural_feature_maps(anchor)
        elif condition_image is None:
            raise ValueError("decoded_structure_rgb diffusion requires condition_image")
        current = anchor
        taus = torch.linspace(
            1.0, 0.0, self.sampling_steps + 1, device=anchor.device, dtype=anchor.dtype
        )
        for index in range(self.sampling_steps):
            tau = torch.full(
                (anchor.shape[0],), float(taus[index]), device=anchor.device, dtype=anchor.dtype
            )
            predicted_target = self.predict_target(
                current,
                anchor,
                condition_image,
                snr_norm,
                tau,
                residual_gate_value,
            )
            next_tau = taus[index + 1]
            current = next_tau * anchor + (1.0 - next_tau) * predicted_target
        return current.clamp(0.0, 1.0)


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return payload


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.epochs is not None:
        config["training"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        config["training"]["batch_size"] = int(args.batch_size)
    if args.train_count is not None:
        config["split"]["train_sample_count"] = int(args.train_count)
    if args.eval_count is not None:
        config["split"]["eval_sample_count"] = int(args.eval_count)
    if args.snrs:
        config["snrs"] = [float(item.strip()) for item in args.snrs.split(",") if item.strip()]


def batch_condition(
    config: dict[str, Any], anchor: torch.Tensor, batch: dict[str, Any]
) -> torch.Tensor:
    source = str(
        config["model"].get(
            "diffusion_condition_source",
            config["model"].get("condition_source", "decoded_structure_rgb"),
        )
    )
    if source == "decoded_structure_rgb":
        if "condition_image" not in batch:
            raise ValueError("decoded_structure_rgb batch is missing condition_image")
        return batch["condition_image"].to(anchor.device, non_blocking=True)
    if source == "receiver_anchor_structural_maps":
        return structural_feature_maps(anchor)
    raise ValueError(f"Unsupported diffusion condition source: {source!r}")


def load_training_semantic_teacher(
    config: dict[str, Any], device: torch.device
) -> nn.Module | None:
    weight = float(config["training"].get("semantic_kl_weight", 0.0))
    if weight <= 0:
        return None
    teacher = config.get("training_semantic_teacher")
    if not isinstance(teacher, dict):
        raise ValueError("training.semantic_kl_weight>0 requires training_semantic_teacher")
    if str(teacher.get("model_name")) != "resnet18":
        raise ValueError("Only the preregistered local resnet18 training teacher is supported")
    weights_path = resolve_project_path(teacher["weights_file"])
    if not weights_path.is_file():
        raise FileNotFoundError(f"Local training teacher weights missing: {weights_path}")
    import torchvision.models as models

    model = models.resnet18(weights=None)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
    return model.to(device).eval().requires_grad_(False)


@torch.no_grad()
def materialize_anchor_cache(
    config: dict[str, Any], names: list[str], snrs: list[float], output: Path, device: torch.device
) -> dict[str, Any]:
    anchor_config_path = resolve_project_path(config["inputs"]["anchor_config"])
    anchor_config = load_yaml(anchor_config_path)
    anchor_config["training"]["batch_size"] = int(config["training"]["batch_size"])
    anchor_model = build_anchor_model(anchor_config).to(device)
    checkpoint_path = resolve_project_path(config["inputs"]["anchor_checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    anchor_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    anchor_model.eval().requires_grad_(False)
    output.mkdir(parents=True, exist_ok=False)
    elapsed = 0.0
    for snr in snrs:
        _, current_elapsed = refine_and_save_snr(
            anchor_model, anchor_config, snr, names, output, device
        )
        elapsed += current_elapsed
    return {
        "anchor_config": project_relative(anchor_config_path),
        "anchor_checkpoint": project_relative(checkpoint_path),
        "num_images": len(names),
        "snrs": snrs,
        "elapsed_seconds": elapsed,
    }


def train_one_epoch(
    model: ShortChainResidualShiftDiffusion,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    device: torch.device,
    semantic_teacher: nn.Module | None = None,
) -> dict[str, float]:
    model.train()
    losses: list[float] = []
    mse_losses: list[float] = []
    l1_losses: list[float] = []
    edge_losses: list[float] = []
    semantic_losses: list[float] = []
    for batch in loader:
        anchor = batch["m0"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        structure = batch_condition(config, anchor, batch)
        snr_db = batch["snr_db"].to(device, non_blocking=True)
        snr_norm = batch["snr_norm"].to(device, non_blocking=True)
        correction_gate = gate_tensor(config, snr_db, device)
        step = torch.randint(1, model.train_timesteps + 1, (anchor.shape[0],), device=device)
        tau = step.float() / float(model.train_timesteps)
        state = model.bridge_state(
            target,
            anchor,
            tau,
            torch.randn_like(target),
            model.bridge_noise_sigma,
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = model.predict_target(
            state, anchor, structure, snr_norm, tau, correction_gate
        )
        mse_loss = F.mse_loss(prediction, target)
        l1_loss = F.l1_loss(prediction, target)
        edge_loss = F.l1_loss(
            structural_feature_maps(prediction), structural_feature_maps(target)
        )
        semantic_weight = float(config["training"].get("semantic_kl_weight", 0.0))
        if semantic_weight > 0:
            if semantic_teacher is None:
                raise RuntimeError("semantic_kl_weight requires a frozen training teacher")
            semantic_loss = semantic_distillation_loss(
                semantic_teacher,
                prediction,
                target,
                float(config["training"].get("semantic_kl_temperature", 1.0)),
            )
        else:
            semantic_loss = mse_loss.new_zeros(())
        loss = (
            float(config["training"]["mse_weight"]) * mse_loss
            + float(config["training"]["l1_weight"]) * l1_loss
            + float(config["training"].get("edge_l1_weight", 0.0)) * edge_loss
            + semantic_weight * semantic_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["training"]["grad_clip_norm"])
        )
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        mse_losses.append(float(mse_loss.detach().cpu()))
        l1_losses.append(float(l1_loss.detach().cpu()))
        edge_losses.append(float(edge_loss.detach().cpu()))
        semantic_losses.append(float(semantic_loss.detach().cpu()))
    return {
        "loss": float(mean(losses) or 0.0),
        "mse_loss": float(mean(mse_losses) or 0.0),
        "l1_loss": float(mean(l1_losses) or 0.0),
        "edge_l1_loss": float(mean(edge_losses) or 0.0),
        "semantic_kl_loss": float(mean(semantic_losses) or 0.0),
    }


@torch.no_grad()
def quick_eval(
    model: ShortChainResidualShiftDiffusion,
    loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    anchor_psnr: list[float] = []
    diffusion_psnr: list[float] = []
    for batch in loader:
        anchor = batch["m0"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        structure = batch_condition(config, anchor, batch)
        snr_db = batch["snr_db"].to(device, non_blocking=True)
        snr_norm = batch["snr_norm"].to(device, non_blocking=True)
        correction_gate = gate_tensor(config, snr_db, device)
        condition = None if model.condition_source == "receiver_anchor_structural_maps" else structure
        refined = model(anchor, snr_norm, correction_gate, condition_image=condition)
        anchor_psnr.extend(psnr_per_sample(anchor, target).cpu().tolist())
        diffusion_psnr.extend(psnr_per_sample(refined, target).cpu().tolist())
    anchor_mean = float(mean(anchor_psnr) or 0.0)
    diffusion_mean = float(mean(diffusion_psnr) or 0.0)
    return {
        "eval_anchor_psnr_db": anchor_mean,
        "eval_diffusion_psnr_db": diffusion_mean,
        "eval_delta_psnr_vs_anchor_db": diffusion_mean - anchor_mean,
    }


def incremental_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    new_errors = 0
    repairs = 0
    for result in results:
        for row in result["per_sample"]:
            anchor_correct = bool(row["m0_matches_original_top1"])
            diffusion_correct = bool(row["refined_matches_original_top1"])
            new_errors += int(anchor_correct and not diffusion_correct)
            repairs += int((not anchor_correct) and diffusion_correct)
    return {"raw_new_error_rows": new_errors, "raw_repair_rows": repairs}


def decision(summary: list[dict[str, Any]], counts: dict[str, int], config: dict[str, Any]) -> dict[str, Any]:
    psnr_deltas = [float(row["refined_delta_psnr_vs_m0_db"]) for row in summary]
    lpips_deltas = [
        float(row["refined_delta_lpips_vs_m0"])
        for row in summary
        if row["refined_delta_lpips_vs_m0"] is not None
    ]
    criteria = config["success_criteria"]
    checks = {
        "mean_lpips_improves": bool(lpips_deltas)
        and sum(lpips_deltas) / len(lpips_deltas)
        < float(criteria["mean_raw_minus_anchor_lpips_max"]),
        "mean_psnr_within_budget": sum(psnr_deltas) / len(psnr_deltas)
        >= float(criteria["mean_raw_minus_anchor_psnr_min_db"]),
        "lpips_improves_enough_snrs": sum(value < 0 for value in lpips_deltas)
        >= int(criteria["minimum_lpips_improved_snr_count"]),
        "new_error_not_greater_than_repair": counts["raw_new_error_rows"]
        <= counts["raw_repair_rows"],
        "sampling_steps_within_limit": int(config["diffusion"]["sampling_steps"])
        <= int(criteria["maximum_sampling_steps"]),
    }
    return {"checks": checks, "all_pass": all(checks.values())}


def write_report(
    path: Path,
    summary: list[dict[str, Any]],
    counts: dict[str, int],
    pilot_decision: dict[str, Any],
    config: dict[str, Any],
) -> None:
    def format_metric(value: Any, *, signed: bool = False) -> str:
        if value is None:
            return "N/A"
        return f"{float(value):+.4f}" if signed else f"{float(value):.4f}"

    lines = [
        "# Short-Chain Conditional Residual-Shift Diffusion Pilot",
        "",
        f"Pilot decision: **{'PROMISING' if pilot_decision['all_pass'] else 'NEGATIVE'}**.",
        "",
        str(
            config.get("protocol", {}).get(
                "anchor_description",
                "The inherited `m0` columns denote the frozen deterministic anchor.",
            )
        ),
        "",
        "| SNR | Anchor PSNR | Diffusion PSNR | ΔPSNR | Anchor LPIPS | Diffusion LPIPS | ΔLPIPS | Final ΔPSNR |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {snr:g} | {a} | {d} | {dp} | {al} | {dl} | {dpl} | {fp} |".format(
                snr=float(row["snr_db"]),
                a=format_metric(row["m0_psnr_db"]),
                d=format_metric(row["refined_psnr_db"]),
                dp=format_metric(row["refined_delta_psnr_vs_m0_db"], signed=True),
                al=format_metric(row["m0_lpips"]),
                dl=format_metric(row["refined_lpips"]),
                dpl=format_metric(row["refined_delta_lpips_vs_m0"], signed=True),
                fp=format_metric(row["m3_delta_psnr_vs_m0_db"], signed=True),
            )
        )
    lines.extend(
        [
            "",
            "## Incremental pseudo-semantic events",
            "",
            f"- Raw new-error rows relative to anchor: `{counts['raw_new_error_rows']}`",
            f"- Raw repair rows relative to anchor: `{counts['raw_repair_rows']}`",
            "- These are frozen AlexNet diagnostics on COCO, not supervised safety evidence.",
            "",
            "## Preregistered checks",
            "",
        ]
    )
    for name, passed in pilot_decision["checks"].items():
        lines.append(f"- `{name}`: **{'PASS' if passed else 'FAIL'}**")
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            f"- Sampling steps: `{int(config['diffusion']['sampling_steps'])}`.",
            "- No Stable Diffusion, VAE, prompt, Imagenette policy-dev, or official validation was used.",
            "- Passing only authorizes a later controlled/supervised follow-up; it does not establish final M3 safety.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    config = load_yaml(config_path)
    apply_overrides(config, args)
    snrs = parse_snrs(config)
    source_manifest = validate_inputs(config, snrs)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "experiment_id": config["experiment_id"],
                    "snrs": snrs,
                    "train_count": len(source_manifest["train_names"]),
                    "eval_count": len(source_manifest["eval_names"]),
                    "sampling_steps": int(config["diffusion"]["sampling_steps"]),
                    "anchor_checkpoint": config["inputs"]["anchor_checkpoint"],
                },
                indent=2,
            )
        )
        return

    output_dir = resolve_project_path(args.output_dir or config["outputs"]["output_dir"])
    if output_dir.exists():
        raise FileExistsError(f"Output exists, refusing to overwrite: {output_dir}")
    output_dir.mkdir(parents=True)
    shutil.copy2(config_path, output_dir / "config.yaml")
    device = resolve_device(args.device)
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    all_names = list(dict.fromkeys(source_manifest["train_names"] + source_manifest["eval_names"]))
    anchor_root = output_dir / "anchor_cache"
    anchor_manifest = materialize_anchor_cache(config, all_names, snrs, anchor_root, device)

    runtime = copy.deepcopy(config)
    runtime["inputs"]["m0_export_dir"] = project_relative(anchor_root)
    runtime["inputs"]["m0_reconstruction_subdir"] = "refined"
    runtime_manifest = validate_inputs(runtime, snrs)
    save_json(output_dir / "source_manifest.json", {"source": source_manifest, "anchor": anchor_manifest})

    train_dataset = ResidualPairDataset(runtime, runtime_manifest["train_names"], snrs, train=True)
    eval_dataset = ResidualPairDataset(runtime, runtime_manifest["eval_names"], snrs, train=False)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(runtime["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(runtime["training"]["num_workers"]),
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(runtime["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(runtime["training"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    model = ShortChainResidualShiftDiffusion(runtime).to(device)
    semantic_teacher = load_training_semantic_teacher(runtime, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(runtime["training"]["lr"]),
        weight_decay=float(runtime["training"]["weight_decay"]),
    )
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir()
    best_path = checkpoint_dir / "best.pt"
    latest_path = checkpoint_dir / "latest.pt"
    history: list[dict[str, Any]] = []
    best_psnr = float("-inf")
    epochs = int(runtime["training"]["epochs"])
    validate_every = int(runtime["training"]["validation_every_epochs"])
    for epoch in range(epochs):
        stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            runtime,
            device,
            semantic_teacher=semantic_teacher,
        )
        row: dict[str, Any] = {"epoch": epoch, **stats}
        if (epoch + 1) % validate_every == 0 or epoch == epochs - 1:
            evaluation = quick_eval(model, eval_loader, runtime, device)
            row.update(evaluation)
            if evaluation["eval_diffusion_psnr_db"] > best_psnr:
                best_psnr = evaluation["eval_diffusion_psnr_db"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": runtime,
                        "eval_stats": evaluation,
                    },
                    best_path,
                )
        history.append(row)
        print(json.dumps(row, indent=2))
    torch.save(
        {"epoch": epochs - 1, "model_state_dict": model.state_dict(), "config": runtime},
        latest_path,
    )
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    write_csv(output_dir / "train_history.csv", history)

    cache_root = resolve_project_path("outputs/cache")
    classifier, preprocess, categories = load_classifier(runtime, device)
    lpips_model, lpips_error = (
        (None, "Skipped by --skip-lpips")
        if args.skip_lpips
        else try_load_lpips(device, cache_root)
    )
    results: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []
    for snr in snrs:
        result = evaluate_snr(
            model,
            runtime,
            snr,
            runtime_manifest["eval_names"],
            output_dir,
            classifier,
            preprocess,
            categories,
            lpips_model,
            device,
        )
        results.append(result)
        per_sample.extend(result["per_sample"])
    summary = aggregate_results(results)
    counts = incremental_counts(results)
    pilot_decision = decision(summary, counts, runtime)
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "per_sample.csv", per_sample)
    write_report(output_dir / "REPORT.md", summary, counts, pilot_decision, runtime)

    import importlib.metadata as md
    import torchvision

    metadata = {
        "project_version": get_project_version(),
        "run_command": " ".join(sys.argv),
        "config": project_relative(config_path),
        "device": str(device),
        "seed": seed,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "sampling_steps": int(runtime["diffusion"]["sampling_steps"]),
        "diffusion_condition_source": model.condition_source,
        "training_semantic_teacher": runtime.get("training_semantic_teacher"),
        "lpips_error": lpips_error,
        "official_imagenette_accessed": False,
        "python_version": platform.python_version(),
        "package_versions": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "pillow": md.version("pillow"),
        },
        "proxy_environment_present": sorted(key for key in os.environ if "proxy" in key.lower()),
        "download_note": "No download; existing local data/checkpoints/weights only.",
    }
    save_json(
        output_dir / "metrics.json",
        {
            "metadata": metadata,
            "anchor_manifest": anchor_manifest,
            "train_history": history,
            "results": results,
            "summary_rows": summary,
            "incremental_counts": counts,
            "pilot_decision": pilot_decision,
        },
    )
    print((output_dir / "REPORT.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
