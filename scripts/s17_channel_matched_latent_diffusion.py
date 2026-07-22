#!/usr/bin/env python3
"""Train and audit the preregistered exact-rate channel-matched latent diffusion."""

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
from datetime import datetime, timezone
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

from cadsd_jscc.channel_matched_latent_diffusion import (  # noqa: E402
    ChannelMatchedLatentDenoiser,
    channel_alpha,
    deterministic_ddim,
    expand_valid_mask,
    masked_mse_per_sample,
    normalize_channel_observation,
    predict_x0_from_epsilon,
)
from cadsd_jscc.deepjscc_adapter import build_deepjscc_model  # noqa: E402
from cadsd_jscc.external_common import (  # noqa: E402
    canonical_noise_sha256,
    canonical_standard_normal,
)
from cadsd_jscc.external_rate_alignment import ExactRateMaskedDeepJSCC  # noqa: E402
from cadsd_jscc.metrics import ms_ssim_per_sample, psnr_per_sample  # noqa: E402
from cadsd_jscc.semantic_sketch import reserved_symbol_indices  # noqa: E402
from s5_residual_refiner_pilot import build_model, gate_tensor, try_load_lpips  # noqa: E402


LEGACY_STAGES = (
    "b0",
    "scalar_lmmse",
    "fixed_step_ddim",
    "matched_one_step",
    "matched_ddim",
    "b1",
    "matched_ddim_b1",
)
LEGACY_LATENT_STAGES = (
    "raw",
    "scalar_lmmse",
    "fixed_step_ddim",
    "matched_one_step",
    "matched_ddim",
)


def evaluation_stages(config: dict[str, Any]) -> tuple[str, ...]:
    configured = tuple(str(value) for value in config["evaluation"].get("candidates", ()))
    return configured or LEGACY_STAGES


def latent_stages(config: dict[str, Any]) -> tuple[str, ...]:
    available = {
        "raw",
        "scalar_lmmse",
        "fixed_step_ddim",
        "parent_matched_ddim",
        "control_matched_ddim",
        "matched_one_step",
        "matched_ddim",
    }
    ordered = ["raw", *evaluation_stages(config)]
    return tuple(stage for stage in ordered if stage in available)


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


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CachedOriginalDataset(Dataset):
    def __init__(
        self,
        cache_root: Path,
        manifest_path: Path,
        role: str,
        *,
        start: int = 0,
        count: int | None = None,
        random_flip: bool = False,
    ) -> None:
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            records = [row for row in csv.DictReader(handle) if row["role"] == role]
        if start < 0 or start >= len(records):
            raise ValueError(f"invalid dataset start {start} for {len(records)} records")
        stop = len(records) if count is None else start + int(count)
        self.records = records[start:stop]
        if count is not None and len(self.records) != int(count):
            raise RuntimeError("dataset slice is shorter than its frozen count")
        self.root = cache_root / "exports" / "original"
        self.to_tensor = transforms.ToTensor()
        self.random_flip = bool(random_flip)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        record = self.records[index]
        with Image.open(self.root / record["sample"]) as image:
            tensor = self.to_tensor(image.convert("RGB"))
        if self.random_flip and torch.rand(()) < 0.5:
            tensor = tensor.flip(-1)
        return tensor, str(record["sample"])


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("config root must be a mapping")
    return config


def validate_contract(config: dict[str, Any], mode: str = "train") -> None:
    protocol = config["protocol"]
    if protocol["status"] not in {
        "preregistered_before_training_output",
        "registered_after_numeric_failure_before_stable_run_output",
        "preregistered_decoder_aware_followup_before_fresh_population_output",
    }:
        raise RuntimeError("S17 config is not preregistered")
    if protocol.get("official_imagenette_accessed") is not False:
        raise RuntimeError("official Imagenette validation must remain sealed")
    rate = config["rate"]
    if int(rate["active_real_symbols"]) != 19712:
        raise RuntimeError("active-symbol contract changed")
    if int(rate["payload_real_symbols"]) != 80:
        raise RuntimeError("payload reservation changed")
    if int(rate["image_active_real_symbols"]) != 19632:
        raise RuntimeError("image active-symbol contract changed")
    if int(rate["dense_real_symbols"]) != 24576:
        raise RuntimeError("dense latent contract changed")
    if int(rate["complex_channel_uses"]) != int(rate["active_real_symbols"]) // 2:
        raise RuntimeError("complex-use ledger changed")
    selection_end = int(config["split"]["selection_start"]) + int(
        config["split"]["selection_count"]
    )
    holdout_start = int(config["split"]["holdout_start"])
    if selection_end > holdout_start:
        raise RuntimeError("selection and holdout overlap")
    for key, expected_key in (
        ("source_manifest", "source_manifest_sha256"),
        ("deepjscc_checkpoint", "deepjscc_checkpoint_sha256"),
        ("b1_checkpoint", "b1_checkpoint_sha256"),
    ):
        path = resolve(config["inputs"][key])
        if not path.exists() or sha256_file(path) != str(
            config["inputs"].get(expected_key, protocol.get(expected_key))
        ):
            raise RuntimeError(f"input hash mismatch: {key}")
    for key, expected_key in (
        (
            "parent_latent_diffusion_checkpoint",
            "parent_latent_diffusion_checkpoint_sha256",
        ),
        (
            "control_latent_diffusion_checkpoint",
            "control_latent_diffusion_checkpoint_sha256",
        ),
    ):
        if key not in config["inputs"]:
            continue
        path = resolve(config["inputs"][key])
        expected = str(config["inputs"].get(expected_key, ""))
        if expected.startswith("PENDING_") and mode != "holdout":
            continue
        if not path.exists() or sha256_file(path) != expected:
            raise RuntimeError(f"input hash mismatch: {key}")
    if resolve(config["inputs"]["deepjscc_checkpoint"]).resolve() == resolve(
        config["inputs"]["forbidden_deepjscc_checkpoint"]
    ).resolve():
        raise RuntimeError("forbidden DeepJSCC latest checkpoint selected")


def build_jscc(config: dict[str, Any], device: torch.device) -> ExactRateMaskedDeepJSCC:
    rate = config["rate"]
    base = build_deepjscc_model(
        resolve(config["baseline"]["repo"]),
        int(rate["inner_channel"]),
        str(config["channel"]["type"]),
        float(config["channel"]["snrs_db"][0]),
    )
    model = ExactRateMaskedDeepJSCC(
        base,
        dense_symbols=int(rate["dense_real_symbols"]),
        active_symbols=int(rate["active_real_symbols"]),
        snr_db=float(config["channel"]["snrs_db"][0]),
    ).to(device)
    checkpoint = torch.load(resolve(config["inputs"]["deepjscc_checkpoint"]), map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval().requires_grad_(False)


def build_denoiser(config: dict[str, Any], device: torch.device) -> ChannelMatchedLatentDenoiser:
    model = config["model"]
    return ChannelMatchedLatentDenoiser(
        latent_channels=int(model["latent_channels"]),
        base_channels=int(model["base_channels"]),
        num_blocks=int(model["num_blocks"]),
        time_embedding_dim=int(model["time_embedding_dim"]),
        group_norm_groups=int(model["group_norm_groups"]),
    ).to(device)


def load_denoiser_checkpoint(
    model: ChannelMatchedLatentDenoiser,
    path: Path,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return checkpoint


def build_checkpoint_denoiser(
    config: dict[str, Any], key: str, device: torch.device
) -> ChannelMatchedLatentDenoiser:
    model = build_denoiser(config, device)
    load_denoiser_checkpoint(model, resolve(config["inputs"][key]), device)
    return model.eval().requires_grad_(False)


def coordinate_contract(
    jscc: ExactRateMaskedDeepJSCC, config: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reserved = reserved_symbol_indices(
        jscc.active_symbols, int(config["rate"]["payload_real_symbols"]), device=device
    )
    valid_active = torch.ones(jscc.active_symbols, dtype=torch.bool, device=device)
    valid_active[reserved] = False
    valid_dense_flat = torch.zeros(jscc.dense_symbols, dtype=torch.bool, device=device)
    valid_dense_flat[jscc.active_indices[valid_active]] = True
    dense_shape = (
        int(config["model"]["latent_channels"]),
        int(config["model"]["latent_height"]),
        int(config["model"]["latent_width"]),
    )
    valid_dense = valid_dense_flat.reshape(dense_shape)
    if int(valid_dense.sum()) != int(config["rate"]["image_active_real_symbols"]):
        raise RuntimeError("runtime valid-coordinate count changed")
    return reserved, valid_active, valid_dense


def balanced_payload(batch: int, symbols: int, device: torch.device, dtype: torch.dtype):
    values = torch.ones(symbols, device=device, dtype=dtype)
    values[1::2] = -1
    return values.unsqueeze(0).expand(batch, -1)


@torch.no_grad()
def clean_transmitted_active(
    jscc: ExactRateMaskedDeepJSCC,
    images: torch.Tensor,
    reserved: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    active, dense_shape = jscc.encode_active(images)
    transmitted = active.clone()
    transmitted[:, reserved] = balanced_payload(
        images.shape[0], reserved.numel(), images.device, active.dtype
    )
    norm = transmitted.float().square().sum(dim=1, keepdim=True).clamp_min(1e-12).sqrt()
    transmitted = transmitted * math.sqrt(jscc.active_symbols) / norm.to(transmitted.dtype)
    clean_erased = transmitted.clone()
    clean_erased[:, reserved] = 0.0
    return transmitted, clean_erased, dense_shape


def active_to_dense(
    jscc: ExactRateMaskedDeepJSCC, active: torch.Tensor, dense_shape: tuple[int, ...]
) -> torch.Tensor:
    dense = active.new_zeros((active.shape[0], jscc.dense_symbols))
    dense.index_copy_(1, jscc.active_indices, active)
    return dense.reshape(active.shape[0], *dense_shape)


def dense_to_active(jscc: ExactRateMaskedDeepJSCC, dense: torch.Tensor) -> torch.Tensor:
    return dense.flatten(1).index_select(1, jscc.active_indices)


def canonical_batch_noise(
    sample_ids: list[str] | tuple[str, ...], snr: float, seed: int, symbols: int
) -> tuple[torch.Tensor, list[str]]:
    rows = torch.stack(
        [canonical_standard_normal(seed, sample_id, snr, symbols) for sample_id in sample_ids]
    )
    return rows, [canonical_noise_sha256(row) for row in rows]


def sample_training_alpha(config: dict[str, Any], batch: int, device: torch.device) -> torch.Tensor:
    factor = float(config["channel"]["noise_variance_factor_per_real"])
    alpha_min = float(channel_alpha(min(config["channel"]["snrs_db"]), factor))
    alpha_max = float(config["diffusion"]["train_alpha_max"])
    log_min = math.log(alpha_min) - math.log1p(-alpha_min)
    log_max = math.log(alpha_max) - math.log1p(-alpha_max)
    return torch.sigmoid(torch.empty(batch, device=device).uniform_(log_min, log_max))


@torch.no_grad()
def diagnose_loss_scale(
    denoiser: ChannelMatchedLatentDenoiser,
    jscc: ExactRateMaskedDeepJSCC,
    loader: DataLoader,
    reserved: torch.Tensor,
    valid_dense: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Choose the decoder-loss weight from preregistered scale-only statistics."""

    denoiser.eval()
    totals: defaultdict[str, float] = defaultdict(float)
    count = 0
    batch_limit = int(config["loss_diagnostic"]["batches"])
    for batch_index, (images_cpu, _sample_ids) in enumerate(loader):
        if batch_index >= batch_limit:
            break
        images = images_cpu.to(device, non_blocking=True)
        _transmitted, clean_active, dense_shape = clean_transmitted_active(
            jscc, images, reserved
        )
        clean = active_to_dense(jscc, clean_active, dense_shape).float()
        alpha = sample_training_alpha(config, images.shape[0], device)
        mask = expand_valid_mask(valid_dense, clean).to(clean.dtype)
        epsilon = torch.randn_like(clean) * mask
        alpha_view = alpha[:, None, None, None]
        noisy = alpha_view.sqrt() * clean + (1.0 - alpha_view).sqrt() * epsilon
        prediction = denoiser(noisy, alpha, valid_dense)
        epsilon_loss = masked_mse_per_sample(prediction, epsilon, valid_dense).mean()
        x0_prediction = predict_x0_from_epsilon(noisy, prediction, alpha)
        x0_loss = masked_mse_per_sample(x0_prediction, clean, valid_dense).mean()
        predicted_active = dense_to_active(jscc, x0_prediction)
        decoded_prediction = jscc.decode_active(predicted_active, dense_shape)
        image_loss = F.mse_loss(decoded_prediction, images)
        base_loss = (
            float(config["training"]["epsilon_mse_weight"]) * epsilon_loss
            + float(config["training"]["x0_mse_weight"]) * x0_loss
        )
        totals["epsilon_mse"] += float(epsilon_loss)
        totals["x0_mse"] += float(x0_loss)
        totals["base_loss"] += float(base_loss)
        totals["decoder_image_mse"] += float(image_loss)
        count += 1
    if count != batch_limit:
        raise RuntimeError(f"loss diagnostic expected {batch_limit} batches, got {count}")
    means = {key: value / count for key, value in totals.items()}
    target_ratio = float(config["loss_diagnostic"]["target_decoder_to_base_loss_ratio"])
    continuous_weight = target_ratio * means["base_loss"] / means["decoder_image_mse"]
    allowed = [
        float(value)
        for value in config["loss_diagnostic"]["allowed_decoder_image_mse_weights"]
    ]
    selected = min(allowed, key=lambda value: (abs(math.log(value / continuous_weight)), value))
    return {
        "experiment_id": config["experiment_id"],
        "seed": int(config["loss_diagnostic"]["seed"]),
        "batches": count,
        **means,
        "target_decoder_to_base_loss_ratio": target_ratio,
        "continuous_scale_weight": continuous_weight,
        "allowed_decoder_image_mse_weights": allowed,
        "selected_decoder_image_mse_weight": selected,
        "selected_weighted_decoder_to_base_ratio": selected
        * means["decoder_image_mse"]
        / means["base_loss"],
        "selection_metrics_accessed": False,
        "holdout_metrics_accessed": False,
    }


def train_epoch(
    denoiser: ChannelMatchedLatentDenoiser,
    jscc: ExactRateMaskedDeepJSCC,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    reserved: torch.Tensor,
    valid_dense: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
    max_batches: int | None,
) -> dict[str, float]:
    denoiser.train()
    use_amp = bool(config["training"]["amp"] and device.type == "cuda")
    totals: defaultdict[str, float] = defaultdict(float)
    count = 0
    started = time.perf_counter()
    for batch_index, (images_cpu, _sample_ids) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images_cpu.to(device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
            _transmitted, clean_active, dense_shape = clean_transmitted_active(
                jscc, images, reserved
            )
            clean = active_to_dense(jscc, clean_active, dense_shape)
        clean = clean.float()
        alpha = sample_training_alpha(config, images.shape[0], device)
        mask = expand_valid_mask(valid_dense, clean).to(clean.dtype)
        epsilon = torch.randn_like(clean) * mask
        alpha_view = alpha[:, None, None, None]
        noisy = alpha_view.sqrt() * clean + (1.0 - alpha_view).sqrt() * epsilon
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            prediction = denoiser(noisy, alpha, valid_dense)
            epsilon_loss = masked_mse_per_sample(prediction, epsilon, valid_dense).mean()
            x0_prediction = predict_x0_from_epsilon(noisy, prediction, alpha)
            x0_loss = masked_mse_per_sample(x0_prediction, clean, valid_dense).mean()
            loss = (
                float(config["training"]["epsilon_mse_weight"]) * epsilon_loss
                + float(config["training"]["x0_mse_weight"]) * x0_loss
            )
            decoder_weight = float(config["training"].get("decoder_image_mse_weight", 0.0))
            if decoder_weight > 0.0:
                predicted_active = dense_to_active(jscc, x0_prediction)
                decoded_prediction = jscc.decode_active(predicted_active, dense_shape)
                decoder_image_loss = F.mse_loss(decoded_prediction, images)
                loss = loss + decoder_weight * decoder_image_loss
            else:
                decoder_image_loss = loss.new_zeros(())
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss at batch {batch_index}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            denoiser.parameters(), float(config["training"]["grad_clip_norm"])
        )
        scaler.step(optimizer)
        scaler.update()
        totals["loss"] += float(loss.detach())
        totals["epsilon_mse"] += float(epsilon_loss.detach())
        totals["x0_mse"] += float(x0_loss.detach())
        totals["decoder_image_mse"] += float(decoder_image_loss.detach())
        totals["weighted_decoder_image_mse"] += float(
            decoder_weight * decoder_image_loss.detach()
        )
        count += 1
    if count == 0:
        raise RuntimeError("training epoch produced no batches")
    return {
        **{key: value / count for key, value in totals.items()},
        "batches": count,
        "seconds": time.perf_counter() - started,
    }


@torch.no_grad()
def selection_evaluate(
    denoiser: ChannelMatchedLatentDenoiser,
    jscc: ExactRateMaskedDeepJSCC,
    loader: DataLoader,
    reserved: torch.Tensor,
    valid_dense: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
    max_batches: int | None,
) -> dict[str, Any]:
    denoiser.eval()
    factor = float(config["channel"]["noise_variance_factor_per_real"])
    rows: list[dict[str, float]] = []
    for snr in [float(value) for value in config["channel"]["snrs_db"]]:
        alpha = float(channel_alpha(snr, factor))
        for batch_index, (images_cpu, sample_ids) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images_cpu.to(device, non_blocking=True)
            transmitted, clean_active, dense_shape = clean_transmitted_active(jscc, images, reserved)
            noise_cpu, _hashes = canonical_batch_noise(
                list(sample_ids),
                snr,
                int(config["channel"]["selection_base_seed"]),
                jscc.active_symbols,
            )
            jscc.snr_db = snr
            received = jscc.transmit_active(transmitted, noise_cpu.to(device))
            received[:, reserved] = 0.0
            b0 = jscc.decode_active(received, dense_shape).clamp(0.0, 1.0)
            state = active_to_dense(
                jscc, normalize_channel_observation(received, alpha), dense_shape
            )
            denoised_dense = deterministic_ddim(
                denoiser,
                state,
                valid_dense,
                alpha_start=alpha,
                sampling_steps=int(config["diffusion"]["sampling_steps"]),
                alpha_max=float(config["diffusion"]["train_alpha_max"]),
            )
            denoised_active = dense_to_active(jscc, denoised_dense)
            denoised_active[:, reserved] = 0.0
            decoded = jscc.decode_active(denoised_active, dense_shape).clamp(0.0, 1.0)
            raw_mse = ((received - clean_active).square()[:, :]).sum(dim=1)
            valid_count = int(config["rate"]["image_active_real_symbols"])
            raw_mse = raw_mse / valid_count
            matched_mse = ((denoised_active - clean_active).square()).sum(dim=1) / valid_count
            b0_psnr = psnr_per_sample(b0, images)
            matched_psnr = psnr_per_sample(decoded, images)
            for index in range(images.shape[0]):
                rows.append(
                    {
                        "snr_db": snr,
                        "raw_latent_mse": float(raw_mse[index]),
                        "matched_latent_mse": float(matched_mse[index]),
                        "b0_psnr": float(b0_psnr[index]),
                        "matched_psnr": float(matched_psnr[index]),
                    }
                )
    summaries = []
    for snr in sorted({row["snr_db"] for row in rows}):
        subset = [row for row in rows if row["snr_db"] == snr]
        summaries.append(
            {
                "snr_db": snr,
                "rows": len(subset),
                "raw_latent_mse": sum(row["raw_latent_mse"] for row in subset) / len(subset),
                "matched_latent_mse": sum(row["matched_latent_mse"] for row in subset)
                / len(subset),
                "b0_psnr": sum(row["b0_psnr"] for row in subset) / len(subset),
                "matched_psnr": sum(row["matched_psnr"] for row in subset) / len(subset),
            }
        )
        summaries[-1]["matched_minus_b0_psnr"] = (
            summaries[-1]["matched_psnr"] - summaries[-1]["b0_psnr"]
        )
    return {
        "rows": len(rows),
        "per_snr": summaries,
        "mean_matched_psnr": sum(row["matched_psnr"] for row in rows) / len(rows),
        "mean_b0_psnr": sum(row["b0_psnr"] for row in rows) / len(rows),
        "mean_matched_minus_b0_psnr": sum(
            row["matched_psnr"] - row["b0_psnr"] for row in rows
        )
        / len(rows),
    }


def classifier_model(name: str, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    builders = {
        "alexnet": models.alexnet,
        "resnet18": models.resnet18,
        "mobilenet_v3_small": models.mobilenet_v3_small,
    }
    model = builders[name](weights=None)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


@torch.no_grad()
def classify(
    model: torch.nn.Module,
    images: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    resized = F.interpolate(
        images, size=(224, 224), mode="bilinear", align_corners=False, antialias=True
    )
    probabilities = torch.softmax(model((resized - mean) / std), dim=1)
    confidence, prediction = probabilities.max(dim=1)
    return prediction, confidence


def build_b1(config: dict[str, Any], device: torch.device):
    b1_config = yaml.safe_load(resolve(config["inputs"]["b1_config"]).read_text(encoding="utf-8"))
    model = build_model(b1_config).to(device)
    checkpoint = torch.load(resolve(config["inputs"]["b1_checkpoint"]), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.eval().requires_grad_(False), b1_config


def mean(rows: list[dict[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows]
    return sum(values) / len(values)


def holdout_summary(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    stages = evaluation_stages(config)
    latent_stage_names = latent_stages(config)
    summary: dict[str, Any] = {
        "analysis_id": config["analysis_id"],
        "rows": len(rows),
        "images": len({row["sample_id"] for row in rows}),
        "snrs_db": sorted({float(row["snr_db"]) for row in rows}),
        "per_snr": [],
    }
    for snr in summary["snrs_db"]:
        subset = [row for row in rows if float(row["snr_db"]) == snr]
        item: dict[str, Any] = {"snr_db": snr, "rows": len(subset)}
        for stage in stages:
            for metric in ("psnr", "ms_ssim", "lpips"):
                item[f"mean_{stage}_{metric}"] = mean(subset, f"{stage}_{metric}")
        for stage in latent_stage_names:
            item[f"mean_{stage}_latent_mse"] = mean(subset, f"{stage}_latent_mse")
        for stage in stages[1:]:
            item[f"{stage}_minus_b0_psnr"] = (
                item[f"mean_{stage}_psnr"] - item["mean_b0_psnr"]
            )
            item[f"{stage}_minus_b0_lpips"] = (
                item[f"mean_{stage}_lpips"] - item["mean_b0_lpips"]
            )
        summary["per_snr"].append(item)
    for stage in stages:
        summary[f"mean_{stage}_psnr"] = mean(rows, f"{stage}_psnr")
        summary[f"mean_{stage}_lpips"] = mean(rows, f"{stage}_lpips")
    for stage in latent_stage_names:
        summary[f"mean_{stage}_latent_mse"] = mean(rows, f"{stage}_latent_mse")
    threshold = float(config["evaluation"]["pseudo_original_confidence_min"])
    eligible = [row for row in rows if float(row["alexnet_original_confidence"]) >= threshold]
    summary["alexnet_pseudo_eligible_rows"] = len(eligible)
    for stage in stages:
        summary[f"alexnet_{stage}_failure"] = sum(
            int(row[f"alexnet_{stage}_prediction"]) != int(row["alexnet_original_prediction"])
            for row in eligible
        )
        summary[f"alexnet_{stage}_new"] = sum(
            int(row["alexnet_b0_prediction"]) == int(row["alexnet_original_prediction"])
            and int(row[f"alexnet_{stage}_prediction"])
            != int(row["alexnet_original_prediction"])
            for row in eligible
        )
        summary[f"alexnet_{stage}_repair"] = sum(
            int(row["alexnet_b0_prediction"]) != int(row["alexnet_original_prediction"])
            and int(row[f"alexnet_{stage}_prediction"])
            == int(row["alexnet_original_prediction"])
            for row in eligible
        )
        summary[f"majority_{stage}_new"] = sum(bool(row[f"majority_{stage}_new"]) for row in rows)
        summary[f"majority_{stage}_repair"] = sum(
            bool(row[f"majority_{stage}_repair"]) for row in rows
        )
    criteria = config["success_criteria"]
    semantic_stage = "matched_ddim" if "control_matched_ddim" in stages else "matched_ddim_b1"
    checks: dict[str, bool] = {
        "matched_latent_mse_better_all_five": all(
            item["mean_matched_ddim_latent_mse"] < item["mean_raw_latent_mse"]
            for item in summary["per_snr"]
        ),
        "matched_ddim_mean_psnr_positive": summary["mean_matched_ddim_psnr"]
        - summary["mean_b0_psnr"]
        >= float(criteria["matched_ddim_mean_psnr_minus_b0_min_db"]),
        "matched_ddim_psnr_positive_four_of_five": sum(
            item["matched_ddim_minus_b0_psnr"] > 0 for item in summary["per_snr"]
        )
        >= int(criteria["matched_ddim_psnr_improved_snr_count_min"]),
        "matched_ddim_lpips_nonworse": summary["mean_matched_ddim_lpips"]
        - summary["mean_b0_lpips"]
        <= float(criteria["matched_ddim_mean_lpips_minus_b0_max"]),
        "alexnet_new_not_greater_than_repair": summary[
            f"alexnet_{semantic_stage}_new"
        ]
        <= summary[f"alexnet_{semantic_stage}_repair"],
        "majority_new_not_greater_than_repair": summary[
            f"majority_{semantic_stage}_new"
        ]
        <= summary[f"majority_{semantic_stage}_repair"],
    }
    if "control_matched_ddim" in stages:
        checks.update(
            {
                "decoder_aware_mean_psnr_better_than_control": summary[
                    "mean_matched_ddim_psnr"
                ]
                - summary["mean_control_matched_ddim_psnr"]
                >= float(criteria["matched_ddim_mean_psnr_minus_control_min_db"]),
                "decoder_aware_psnr_better_than_control_three_of_five": sum(
                    item["mean_matched_ddim_psnr"]
                    > item["mean_control_matched_ddim_psnr"]
                    for item in summary["per_snr"]
                )
                >= int(criteria["matched_ddim_psnr_improved_vs_control_snr_count_min"]),
            }
        )
    else:
        checks.update(
            {
                "matched_beats_fixed_step_psnr": summary["mean_matched_ddim_psnr"]
                >= summary["mean_fixed_step_ddim_psnr"],
                "matched_b1_beats_b1_psnr": summary["mean_matched_ddim_b1_psnr"]
                - summary["mean_b1_psnr"]
                >= float(criteria["matched_ddim_b1_mean_psnr_minus_b1_min_db"]),
            }
        )
    summary["checks"] = checks
    summary["verdict"] = "PASS" if all(checks.values()) else "NEGATIVE_OR_PARTIAL"
    return summary


@torch.no_grad()
def run_holdout(
    denoiser: ChannelMatchedLatentDenoiser,
    jscc: ExactRateMaskedDeepJSCC,
    loader: DataLoader,
    reserved: torch.Tensor,
    valid_dense: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
    output: Path,
    max_batches: int | None,
) -> dict[str, Any]:
    denoiser.eval()
    stages = evaluation_stages(config)
    b1, b1_config = build_b1(config, device)
    parent_denoiser = (
        build_checkpoint_denoiser(
            config, "parent_latent_diffusion_checkpoint", device
        )
        if "parent_matched_ddim" in stages
        else None
    )
    control_denoiser = (
        build_checkpoint_denoiser(
            config, "control_latent_diffusion_checkpoint", device
        )
        if "control_matched_ddim" in stages
        else None
    )
    lpips_model, lpips_error = try_load_lpips(device, resolve("outputs/cache"))
    if lpips_model is None:
        raise RuntimeError(lpips_error)
    classifiers = {
        name: classifier_model(name, resolve(path), device)
        for name, path in config["classifiers"].items()
        if name in {"alexnet", "resnet18", "mobilenet_v3_small"}
    }
    mean_tensor = torch.tensor(
        config["classifiers"]["imagenet_mean"], device=device
    ).reshape(1, 3, 1, 1)
    std_tensor = torch.tensor(
        config["classifiers"]["imagenet_std"], device=device
    ).reshape(1, 3, 1, 1)
    factor = float(config["channel"]["noise_variance_factor_per_real"])
    fixed_snr = float(config["diffusion"]["fixed_step_ablation_snr_db"])
    fixed_alpha = float(channel_alpha(fixed_snr, factor))
    rows: list[dict[str, Any]] = []
    sample_saved: set[float] = set()
    for snr in [float(value) for value in config["channel"]["snrs_db"]]:
        alpha = float(channel_alpha(snr, factor))
        for batch_index, (images_cpu, sample_ids) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images_cpu.to(device, non_blocking=True)
            transmitted, clean_active, dense_shape = clean_transmitted_active(jscc, images, reserved)
            noise_cpu, noise_hashes = canonical_batch_noise(
                list(sample_ids),
                snr,
                int(config["channel"]["holdout_base_seed"]),
                jscc.active_symbols,
            )
            jscc.snr_db = snr
            received = jscc.transmit_active(transmitted, noise_cpu.to(device))
            received[:, reserved] = 0.0
            b0 = jscc.decode_active(received, dense_shape).clamp(0.0, 1.0)

            scalar_active = received * alpha
            scalar_active[:, reserved] = 0.0
            scalar = jscc.decode_active(scalar_active, dense_shape).clamp(0.0, 1.0)

            matched_state = active_to_dense(
                jscc, normalize_channel_observation(received, alpha), dense_shape
            )
            alpha_batch = torch.full(
                (images.shape[0],), alpha, device=device, dtype=matched_state.dtype
            )
            one_epsilon = denoiser(matched_state, alpha_batch, valid_dense)
            one_dense = predict_x0_from_epsilon(matched_state, one_epsilon, alpha_batch)
            one_active = dense_to_active(jscc, one_dense)
            one_active[:, reserved] = 0.0
            one = jscc.decode_active(one_active, dense_shape).clamp(0.0, 1.0)

            matched_dense = deterministic_ddim(
                denoiser,
                matched_state,
                valid_dense,
                alpha_start=alpha,
                sampling_steps=int(config["diffusion"]["sampling_steps"]),
                alpha_max=float(config["diffusion"]["train_alpha_max"]),
            )
            matched_active = dense_to_active(jscc, matched_dense)
            matched_active[:, reserved] = 0.0
            matched = jscc.decode_active(matched_active, dense_shape).clamp(0.0, 1.0)

            comparison_candidates: dict[str, torch.Tensor] = {}
            comparison_latents: dict[str, torch.Tensor] = {}
            for stage, comparison_model in (
                ("parent_matched_ddim", parent_denoiser),
                ("control_matched_ddim", control_denoiser),
            ):
                if comparison_model is None:
                    continue
                comparison_dense = deterministic_ddim(
                    comparison_model,
                    matched_state,
                    valid_dense,
                    alpha_start=alpha,
                    sampling_steps=int(config["diffusion"]["sampling_steps"]),
                    alpha_max=float(config["diffusion"]["train_alpha_max"]),
                )
                comparison_active = dense_to_active(jscc, comparison_dense)
                comparison_active[:, reserved] = 0.0
                comparison_candidates[stage] = jscc.decode_active(
                    comparison_active, dense_shape
                ).clamp(0.0, 1.0)
                comparison_latents[stage] = comparison_active

            fixed_state = active_to_dense(
                jscc, normalize_channel_observation(received, fixed_alpha), dense_shape
            )
            fixed_dense = deterministic_ddim(
                denoiser,
                fixed_state,
                valid_dense,
                alpha_start=fixed_alpha,
                sampling_steps=int(config["diffusion"]["sampling_steps"]),
                alpha_max=float(config["diffusion"]["train_alpha_max"]),
            )
            fixed_active = dense_to_active(jscc, fixed_dense)
            fixed_active[:, reserved] = 0.0
            fixed = jscc.decode_active(fixed_active, dense_shape).clamp(0.0, 1.0)

            snr_tensor = torch.full((images.shape[0],), snr, device=device)
            snr_norm = snr_tensor / float(b1_config["model"]["snr_norm_max"])
            b1_image = b1(b0, snr_norm, gate_tensor(b1_config, snr_tensor, device))
            matched_b1 = b1(
                matched, snr_norm, gate_tensor(b1_config, snr_tensor, device)
            )
            candidates = {
                "b0": b0,
                "scalar_lmmse": scalar,
                "fixed_step_ddim": fixed,
                **comparison_candidates,
                "matched_one_step": one,
                "matched_ddim": matched,
                "b1": b1_image,
                "matched_ddim_b1": matched_b1,
            }
            latent_candidates = {
                "raw": received,
                "scalar_lmmse": scalar_active,
                "fixed_step_ddim": fixed_active,
                **comparison_latents,
                "matched_one_step": one_active,
                "matched_ddim": matched_active,
            }
            missing = set(stages) - set(candidates)
            if missing:
                raise RuntimeError(f"configured holdout candidates are missing: {sorted(missing)}")
            quality: dict[str, dict[str, torch.Tensor]] = {}
            for stage, candidate in candidates.items():
                quality[stage] = {
                    "psnr": psnr_per_sample(candidate, images),
                    "ms_ssim": ms_ssim_per_sample(candidate, images),
                    "lpips": lpips_model(
                        candidate * 2.0 - 1.0, images * 2.0 - 1.0
                    ).flatten(),
                }
            latent_mse = {
                stage: ((candidate - clean_active).square()).sum(dim=1)
                / int(config["rate"]["image_active_real_symbols"])
                for stage, candidate in latent_candidates.items()
            }
            predictions: dict[str, dict[str, torch.Tensor]] = {}
            original_classification: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            for name, classifier in classifiers.items():
                original_classification[name] = classify(
                    classifier, images, mean_tensor, std_tensor
                )
                predictions[name] = {
                    stage: classify(classifier, candidate, mean_tensor, std_tensor)[0]
                    for stage, candidate in candidates.items()
                }
            if snr not in sample_saved:
                count = min(int(config["evaluation"]["sample_grid_count"]), images.shape[0])
                save_image(
                    torch.cat(
                        [images[:count], *[candidates[stage][:count] for stage in stages]]
                    ).cpu(),
                    output / f"snr_{int(snr):02d}_candidate_grid.png",
                    nrow=count,
                )
                sample_saved.add(snr)
            for index, sample_id in enumerate(sample_ids):
                row: dict[str, Any] = {
                    "analysis_id": config["analysis_id"],
                    "sample_id": sample_id,
                    "snr_db": snr,
                    "alpha_channel": alpha,
                    "fixed_step_alpha": fixed_alpha,
                    "canonical_noise_sha256": noise_hashes[index],
                    "total_real_symbols": int(config["rate"]["active_real_symbols"]),
                    "image_active_real_symbols": int(
                        config["rate"]["image_active_real_symbols"]
                    ),
                }
                for stage in stages:
                    for metric in ("psnr", "ms_ssim", "lpips"):
                        row[f"{stage}_{metric}"] = float(quality[stage][metric][index])
                for stage in latent_stages(config):
                    row[f"{stage}_latent_mse"] = float(latent_mse[stage][index])
                for name in classifiers:
                    original_prediction, original_confidence = original_classification[name]
                    row[f"{name}_original_prediction"] = int(original_prediction[index])
                    row[f"{name}_original_confidence"] = float(original_confidence[index])
                    for stage in stages:
                        row[f"{name}_{stage}_prediction"] = int(
                            predictions[name][stage][index]
                        )
                for stage in stages:
                    new_votes = 0
                    repair_votes = 0
                    for name in classifiers:
                        original = int(row[f"{name}_original_prediction"])
                        baseline = int(row[f"{name}_b0_prediction"])
                        candidate = int(row[f"{name}_{stage}_prediction"])
                        new_votes += int(baseline == original and candidate != original)
                        repair_votes += int(baseline != original and candidate == original)
                    row[f"majority_{stage}_new"] = new_votes >= 2
                    row[f"majority_{stage}_repair"] = repair_votes >= 2
                rows.append(row)
        print(json.dumps({"holdout_snr_complete": snr, "rows": len(rows)}))
    write_csv(output / "per_sample.csv", rows)
    summary = holdout_summary(rows, config)
    save_json(output / "summary.json", summary)
    return summary


def make_loaders(config: dict[str, Any], device: torch.device, dry_run: bool):
    cache_root = resolve(config["inputs"]["cache_root"])
    manifest = resolve(config["inputs"]["source_manifest"])
    split = config["split"]
    train_count = 24 if dry_run else int(split["train_count"])
    selection_count = 16 if dry_run else int(split["selection_count"])
    holdout_count = 16 if dry_run else int(split["holdout_count"])
    train = CachedOriginalDataset(
        cache_root,
        manifest,
        str(split["train_role"]),
        start=0,
        count=train_count,
        random_flip=bool(config["training"]["random_horizontal_flip"]),
    )
    selection = CachedOriginalDataset(
        cache_root,
        manifest,
        str(split["selection_role"]),
        start=int(split["selection_start"]),
        count=selection_count,
    )
    holdout = CachedOriginalDataset(
        cache_root,
        manifest,
        str(split["holdout_role"]),
        start=int(split["holdout_start"]),
        count=holdout_count,
    )
    workers = 0 if dry_run else int(config["training"]["num_workers"])
    train_loader = DataLoader(
        train,
        batch_size=min(int(config["training"]["batch_size"]), len(train)),
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(int(config["seed"])),
    )
    eval_workers = 0 if dry_run else int(config["evaluation"]["num_workers"])
    common = dict(
        batch_size=min(int(config["evaluation"]["batch_size"]), selection_count),
        shuffle=False,
        num_workers=eval_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=eval_workers > 0,
    )
    return train_loader, DataLoader(selection, **common), DataLoader(holdout, **common)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/s17_channel_matched_latent_diffusion.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--mode",
        choices=("train", "holdout", "loss-diagnostic", "dry-run"),
        default="train",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--holdout-dir", default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    args = parser.parse_args()

    config_path = resolve(args.config)
    config = load_config(config_path)
    validate_contract(config, args.mode)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed_everything(int(config["seed"]))
    train_loader, selection_loader, holdout_loader = make_loaders(
        config, device, args.mode == "dry-run"
    )
    jscc = build_jscc(config, device)
    denoiser = build_denoiser(config, device)
    warm_start_path: Path | None = None
    if "parent_latent_diffusion_checkpoint" in config["inputs"]:
        warm_start_path = resolve(config["inputs"]["parent_latent_diffusion_checkpoint"])
        load_denoiser_checkpoint(denoiser, warm_start_path, device)
    reserved, _valid_active, valid_dense = coordinate_contract(jscc, config, device)
    plan = {
        "experiment_id": config["experiment_id"],
        "analysis_id": config["analysis_id"],
        "device": str(device),
        "train_images": len(train_loader.dataset),
        "selection_images": len(selection_loader.dataset),
        "holdout_images": len(holdout_loader.dataset),
        "valid_dense_coordinates": int(valid_dense.sum()),
        "reserved_coordinates": int(reserved.numel()),
        "model_parameters": sum(parameter.numel() for parameter in denoiser.parameters()),
        "warm_start_checkpoint": relative(warm_start_path) if warm_start_path else None,
        "warm_start_checkpoint_sha256": sha256_file(warm_start_path)
        if warm_start_path
        else None,
        "official_imagenette_accessed": False,
    }
    if args.mode == "loss-diagnostic":
        if "loss_diagnostic" not in config:
            raise RuntimeError("config does not define a loss-diagnostic contract")
        output = resolve(args.output_dir or config["outputs"]["loss_diagnostic_dir"])
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
        seed_everything(int(config["loss_diagnostic"]["seed"]))
        diagnostic = diagnose_loss_scale(
            denoiser,
            jscc,
            train_loader,
            reserved,
            valid_dense,
            config,
            device,
        )
        shutil.copy2(config_path, output / "config_before_weight_freeze.yaml")
        shutil.copy2(SCRIPT, output / SCRIPT.name)
        save_json(output / "run_plan.json", plan)
        save_json(output / "summary.json", diagnostic)
        save_json(
            output / "STATE.json",
            {
                "state": "COMPLETE",
                "selected_decoder_image_mse_weight": diagnostic[
                    "selected_decoder_image_mse_weight"
                ],
                "selection_metrics_accessed": False,
                "holdout_metrics_accessed": False,
            },
        )
        print(json.dumps(diagnostic, ensure_ascii=False, indent=2))
        return
    if args.mode == "dry-run":
        train = train_epoch(
            denoiser,
            jscc,
            train_loader,
            torch.optim.AdamW(denoiser.parameters(), lr=1e-4),
            torch.amp.GradScaler(enabled=device.type == "cuda"),
            reserved,
            valid_dense,
            config,
            device,
            1,
        )
        selection = selection_evaluate(
            denoiser,
            jscc,
            selection_loader,
            reserved,
            valid_dense,
            config,
            device,
            1,
        )
        print(json.dumps({"dry_run": True, "plan": plan, "train": train, "selection": selection}, indent=2))
        return

    if args.mode == "train":
        output = resolve(args.output_dir or config["outputs"]["train_dir"])
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
        (output / "checkpoints").mkdir()
        shutil.copy2(config_path, output / "config.yaml")
        shutil.copy2(SCRIPT, output / SCRIPT.name)
        save_json(output / "run_plan.json", plan)
        optimizer = torch.optim.AdamW(
            denoiser.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        scaler = torch.amp.GradScaler(
            enabled=bool(config["training"]["amp"] and device.type == "cuda")
        )
        history: list[dict[str, Any]] = []
        best_psnr = -math.inf
        best_path = output / "checkpoints" / "best.pt"
        for epoch in range(int(config["training"]["epochs"])):
            train = train_epoch(
                denoiser,
                jscc,
                train_loader,
                optimizer,
                scaler,
                reserved,
                valid_dense,
                config,
                device,
                args.max_train_batches,
            )
            selection = selection_evaluate(
                denoiser,
                jscc,
                selection_loader,
                reserved,
                valid_dense,
                config,
                device,
                args.max_eval_batches,
            )
            row = {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train.items()},
                "selection_mean_b0_psnr": selection["mean_b0_psnr"],
                "selection_mean_matched_psnr": selection["mean_matched_psnr"],
                "selection_mean_matched_minus_b0_psnr": selection[
                    "mean_matched_minus_b0_psnr"
                ],
            }
            history.append(row)
            payload = {
                "format_version": 1,
                "epoch": epoch,
                "model_state_dict": denoiser.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "selection": selection,
                "config": config,
            }
            torch.save(payload, output / "checkpoints" / "latest.pt")
            if selection["mean_matched_psnr"] > best_psnr:
                best_psnr = selection["mean_matched_psnr"]
                shutil.copy2(output / "checkpoints" / "latest.pt", best_path)
            write_csv(output / "history.csv", history)
            save_json(output / f"selection_epoch_{epoch:02d}.json", selection)
            print(json.dumps(row, ensure_ascii=False))
        metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": relative(config_path),
            "config_sha256": sha256_file(config_path),
            "script": relative(SCRIPT),
            "script_sha256": sha256_file(SCRIPT),
            "best_checkpoint": relative(best_path),
            "best_checkpoint_sha256": sha256_file(best_path),
            "best_selection_psnr": best_psnr,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "official_imagenette_accessed": False,
            "download_note": "No download; local COCO/checkpoints only.",
        }
        save_json(output / "metadata.json", metadata)
        save_json(output / "STATE.json", {"state": "COMPLETE", **metadata})
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return

    checkpoint_path = resolve(
        args.checkpoint
        or Path(config["outputs"]["train_dir"]) / "checkpoints" / "best.pt"
    )
    expected_checkpoint_sha256 = config["protocol"].get("trained_checkpoint_sha256")
    if expected_checkpoint_sha256 and sha256_file(checkpoint_path) != str(
        expected_checkpoint_sha256
    ):
        raise RuntimeError("trained decoder-aware checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    denoiser.load_state_dict(checkpoint["model_state_dict"], strict=True)
    output = resolve(args.holdout_dir or config["outputs"]["holdout_dir"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    shutil.copy2(config_path, output / "config.yaml")
    shutil.copy2(SCRIPT, output / SCRIPT.name)
    save_json(
        output / "run_plan.json",
        {
            **plan,
            "checkpoint": relative(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        },
    )
    summary = run_holdout(
        denoiser,
        jscc,
        holdout_loader,
        reserved,
        valid_dense,
        config,
        device,
        output,
        args.max_eval_batches,
    )
    save_json(
        output / "STATE.json",
        {
            "state": "COMPLETE",
            "verdict": summary["verdict"],
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "official_imagenette_accessed": False,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
