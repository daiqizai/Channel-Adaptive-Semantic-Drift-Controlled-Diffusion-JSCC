#!/usr/bin/env python3
"""Run official DiffJSCC on the frozen S20/S28 comparison population."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
import types
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
AUTHOR_ROOT = ROOT / "third_party" / "DiffJSCC"
OPENCLIP_ROOT = ROOT / "third_party" / "open_clip_2_24" / "src"
TRANSFORMERS_RUNTIME = AUTHOR_ROOT / "runtime_env" / "transformers_4_51_1"
sys.path[:0] = [
    str(TRANSFORMERS_RUNTIME),
    str(OPENCLIP_ROOT),
    str(ROOT / "src"),
    str(ROOT / "scripts"),
    str(AUTHOR_ROOT),
]
# Ubuntu's cv2 is usable with the venv NumPy, but lives in dist-packages.
sys.path.append("/usr/lib/python3/dist-packages")

# DiffJSCC uses a Lightning 1.x import path.  This is an API-only shim.
from pytorch_lightning.utilities.rank_zero import rank_zero_only  # noqa: E402

pl_distributed = types.ModuleType("pytorch_lightning.utilities.distributed")
pl_distributed.rank_zero_only = rank_zero_only
sys.modules.setdefault("pytorch_lightning.utilities.distributed", pl_distributed)

import einops  # noqa: E402
import open_clip  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from transformers import (  # noqa: E402
    Blip2ForConditionalGeneration,
    Blip2Processor,
)

from cadsd_jscc.external_common import (  # noqa: E402
    canonical_noise_sha256,
    canonical_standard_normal,
)
from cadsd_jscc.metrics import ms_ssim_per_sample, psnr_per_sample  # noqa: E402
from model.spaced_sampler import SpacedSampler  # noqa: E402
from pc_imagenette_supervised_audit import (  # noqa: E402
    evaluate_probabilities,
    load_scratch_classifier,
)
from s30_diffjscc_preflight import metadata_tree_sha256, sha256_file  # noqa: E402
from s5_residual_refiner_pilot import try_load_lpips  # noqa: E402
from utils.common import instantiate_from_config  # noqa: E402
from utils.image import auto_resize, pad  # noqa: E402


SCRIPT = Path(__file__).resolve()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_yaml(value: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(resolve(value).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"YAML root must be a mapping: {value}")
    return payload


def require_sha(value: str | Path, expected: str) -> Path:
    path = resolve(value)
    if not path.is_file() or sha256_file(path) != str(expected):
        raise RuntimeError(f"missing or hash-mismatched frozen input: {path}")
    return path


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sampler_seed(base_seed: int, sample_id: str, snr_db: float) -> int:
    material = (
        f"diffjscc-sampler-v1|{int(base_seed)}|{sample_id}|{float(snr_db):.6f}"
    )
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def evaluator_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "imagenette": {
            "normalization_mean": config["evaluator"]["normalization_mean"],
            "normalization_std": config["evaluator"]["normalization_std"],
        }
    }


def load_population(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    reference_path = require_sha(
        config["inputs"]["population_reference"],
        config["inputs"]["population_reference_sha256"],
    )
    reference = load_yaml(reference_path)
    population = reference["population"]
    manifest_path = require_sha(
        config["inputs"]["split_manifest"], config["inputs"]["split_manifest_sha256"]
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    classes = [str(value) for value in manifest["classes"]]
    by_id = {
        str(item["sample_id"]): item
        for item in manifest["samples"]
        if str(item["split"]) == str(population["required_split"])
    }
    source_root = resolve(manifest["source_train_root"])
    samples: list[dict[str, Any]] = []
    for frozen in population["samples"]:
        sample_id = str(frozen["sample_id"])
        item = dict(by_id[sample_id])
        image = source_root / str(item["relative_path"])
        require_sha(image, str(frozen["content_sha256"]))
        if int(item["class_idx"]) != int(frozen["class_idx"]):
            raise RuntimeError(f"class mismatch: {sample_id}")
        item["path"] = image
        samples.append(item)
    expected = int(config["population"]["expected_sample_count"])
    if len(samples) != expected or len({item["sample_id"] for item in samples}) != expected:
        raise RuntimeError("frozen population size/uniqueness changed")
    return samples, classes


def checkpoint_state(config: dict[str, Any]) -> dict[str, torch.Tensor]:
    checkpoint = require_sha(
        config["assets"]["checkpoint_file"], config["assets"]["checkpoint_sha256"]
    )
    if checkpoint.stat().st_size != int(config["assets"]["checkpoint_expected_bytes"]):
        raise RuntimeError("checkpoint byte count changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict) or not state:
        raise RuntimeError("checkpoint has no state dict")
    required = (
        "cond_stage_model.",
        "preprocess_model.",
        "control_model.",
        "model.diffusion_model.",
        "first_stage_model.",
        "cond_encoder.",
    )
    absent = [prefix for prefix in required if not any(key.startswith(prefix) for key in state)]
    if absent:
        raise RuntimeError(f"checkpoint omits critical model weights: {absent}")
    if any(key.startswith("blip_model.") for key in state):
        raise RuntimeError("checkpoint unexpectedly embeds BLIP despite frozen author contract")
    return state


def instantiate_official_model(
    config: dict[str, Any], state: dict[str, torch.Tensor]
) -> tuple[torch.nn.Module, dict[str, Any]]:
    metadata = resolve(config["assets"]["blip_metadata_directory"])
    if metadata_tree_sha256(metadata) != str(config["assets"]["blip_metadata_tree_sha256"]):
        raise RuntimeError("BLIP2 metadata changed")
    model_config = OmegaConf.load(
        require_sha(config["assets"]["model_config"], config["assets"]["model_config_sha256"])
    )

    original_openclip = open_clip.create_model_and_transforms
    original_processor = Blip2Processor.from_pretrained
    original_blip = Blip2ForConditionalGeneration.from_pretrained
    blip_weights = resolve(config["assets"]["blip_weights_directory"])
    for item in config["assets"]["blip_weight_files"]:
        path = blip_weights / str(item["name"])
        if not path.is_file() or path.stat().st_size != int(item["bytes"]):
            raise RuntimeError(f"missing exact author BLIP2 weight: {path}")
        if sha256_file(path) != str(item["sha256"]):
            raise RuntimeError(f"BLIP2 weight hash mismatch: {path}")

    def empty_openclip(*args: Any, **kwargs: Any):
        kwargs["pretrained"] = None
        return original_openclip(*args, **kwargs)

    def exact_processor(*_args: Any, **_kwargs: Any):
        return original_processor(str(metadata), local_files_only=True, use_fast=False)

    def exact_blip(*_args: Any, **kwargs: Any):
        return original_blip(
            str(blip_weights),
            local_files_only=True,
            torch_dtype=kwargs.get("torch_dtype", torch.float16),
            max_length=int(kwargs.get("max_length", 32)),
        )

    with ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(open_clip, "create_model_and_transforms", side_effect=empty_openclip)
        )
        stack.enter_context(
            mock.patch.object(Blip2Processor, "from_pretrained", side_effect=exact_processor)
        )
        stack.enter_context(
            mock.patch.object(
                Blip2ForConditionalGeneration, "from_pretrained", side_effect=exact_blip
            )
        )
        model = instantiate_from_config(model_config)

    incompatible = model.load_state_dict(state, strict=False, assign=True)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    disallowed_missing = [key for key in missing if not key.startswith("blip_model.")]
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "official checkpoint is not strict under the compatibility loader: "
            f"missing={disallowed_missing[:20]}, unexpected={unexpected[:20]}"
        )
    model.eval().requires_grad_(False)
    return model, {
        "allowed_missing_blip_keys": len(missing) - len(disallowed_missing),
        "disallowed_missing_keys": disallowed_missing,
        "unexpected_keys": unexpected,
        "blip_metadata_revision": config["assets"]["blip_metadata_revision"],
        "exact_external_author_blip_weights": True,
        "remote_weight_substitution": False,
    }


@torch.no_grad()
def fixed_deepjscc_forward(self: torch.nn.Module, x: torch.Tensor):
    snr = torch.full(
        (x.shape[0], 1), float(self._s30_snr_db), device=x.device, dtype=x.dtype
    )
    csi = snr
    latent = self.E(x, csi)
    n, channels, height, width = latent.shape
    if channels % 2:
        raise RuntimeError("DiffJSCC latent channel count is not even")
    expected = int(self._s30_expected_real_symbols)
    if latent[0].numel() != expected:
        raise RuntimeError(f"runtime latent has {latent[0].numel()} real symbols != {expected}")
    latent_complex = latent[:, : channels // 2] + 1j * latent[:, channels // 2 :]
    latent_power = latent_complex.abs().square().mean((-3, -2, -1), keepdim=True)
    latent_tx = latent_complex / torch.sqrt(latent_power)
    standard = self._s30_standard_normal.to(device=x.device, dtype=x.dtype).reshape_as(latent)
    sigma = 10 ** (-snr / 20)
    noise_real = sigma.view(n, 1, 1, 1) / math.sqrt(2.0) * standard
    noise = noise_real[:, : channels // 2] + 1j * noise_real[:, channels // 2 :]
    latent_rx = latent_tx + noise
    latent_equalized = torch.cat((latent_rx.real, latent_rx.imag), dim=1)
    reconstruction = self.G(latent_equalized, csi)
    self._s30_last_observation = {
        "latent_shape": [int(channels), int(height), int(width)],
        "real_symbols": int(latent[0].numel()),
        "complex_channel_uses": int(latent_complex[0].numel()),
        "normalized_complex_power": float(latent_tx.abs().square().mean()),
    }
    return reconstruction, snr, torch.ones_like(snr).view(n, 1, 1, 1)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def author_process(
    model: torch.nn.Module,
    control_img: np.ndarray,
    steps: int,
    sampler_rng_seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, str, dict[str, float]]:
    started = time.perf_counter()
    sampler = SpacedSampler(model, var_type="fixed_small")
    img_t = torch.tensor(
        np.stack([control_img]) / 255.0, dtype=torch.float32, device=device
    ).clamp_(0, 1)
    img_t = einops.rearrange(img_t, "n h w c -> n c h w").contiguous()
    sync(device)
    jscc_started = time.perf_counter()
    img_init, cond_snr, _ = model.preprocess_model(img_t)
    sync(device)
    jscc_ms = (time.perf_counter() - jscc_started) * 1000.0

    input_img = [model.transform_to_pil(img_init[0])]
    sync(device)
    caption_started = time.perf_counter()
    inputs = model.processor(images=input_img, return_tensors="pt", max_length=32).to(
        device, torch.float16
    )
    generated_ids = model.blip_model.generate(**inputs)
    generated_text = model.processor.batch_decode(generated_ids, skip_special_tokens=True)
    cond_text = model.get_learned_conditioning(generated_text)
    sync(device)
    caption_ms = (time.perf_counter() - caption_started) * 1000.0

    height, width = img_t.shape[-2:]
    shape = (1, 4, height // 8, width // 8)
    torch.manual_seed(int(sampler_rng_seed))
    torch.cuda.manual_seed_all(int(sampler_rng_seed))
    sync(device)
    diffusion_started = time.perf_counter()
    samples = sampler.sample(
        steps=steps,
        shape=shape,
        cond_img=img_init,
        cond_snr=cond_snr,
        cond_text=cond_text,
        positive_prompt="",
        negative_prompt="",
        x_T=None,
        cfg_scale=1.0,
        cond_fn=None,
        color_fix_type="wavelet",
    ).clamp(0, 1)
    sync(device)
    diffusion_ms = (time.perf_counter() - diffusion_started) * 1000.0
    total_ms = (time.perf_counter() - started) * 1000.0
    pred = (
        einops.rearrange(samples, "b c h w -> b h w c")[0]
        .mul(255)
        .cpu()
        .numpy()
        .clip(0, 255)
        .astype(np.uint8)
    )
    jscc = (
        einops.rearrange(img_init, "b c h w -> b h w c")[0]
        .mul(255)
        .cpu()
        .numpy()
        .clip(0, 255)
        .astype(np.uint8)
    )
    return pred, jscc, str(generated_text[0]), {
        "jscc_ms": jscc_ms,
        "caption_ms": caption_ms,
        "diffusion_ms": diffusion_ms,
        "total_ms": total_ms,
    }


def metric_triplet(
    target: torch.Tensor, candidate: torch.Tensor, lpips_model: torch.nn.Module
) -> tuple[float, float, float]:
    with torch.no_grad():
        return (
            float(psnr_per_sample(candidate, target)[0]),
            float(ms_ssim_per_sample(candidate, target)[0]),
            float(
                lpips_model(candidate * 2.0 - 1.0, target * 2.0 - 1.0)
                .flatten()[0]
            ),
        )


def cluster_ci(
    rows: list[dict[str, Any]], field: str, replicates: int, seed: int
) -> list[float]:
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sample_id"])].append(float(row[field]))
    values = np.asarray([np.mean(grouped[key]) for key in sorted(grouped)], dtype=np.float64)
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(replicates, len(values)))
    boot = values[indices].mean(axis=1)
    return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def summarize(config: dict[str, Any], stage: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    mean_fields = [
        "author_jscc_psnr",
        "author_jscc_ms_ssim",
        "author_jscc_lpips",
        "diffjscc_psnr",
        "diffjscc_ms_ssim",
        "diffjscc_lpips",
        "current_psnr",
        "current_ms_ssim",
        "current_lpips",
        "b1_psnr",
        "b1_ms_ssim",
        "b1_lpips",
        "diffjscc_minus_author_jscc_psnr",
        "diffjscc_minus_author_jscc_ms_ssim",
        "diffjscc_minus_author_jscc_lpips",
        "current_minus_diffjscc_psnr",
        "current_minus_diffjscc_ms_ssim",
        "current_minus_diffjscc_lpips",
        "b1_minus_diffjscc_psnr",
        "b1_minus_diffjscc_ms_ssim",
        "b1_minus_diffjscc_lpips",
    ]
    means = {field: float(np.mean([float(row[field]) for row in rows])) for field in mean_fields}
    replicates = int(config["metrics"]["bootstrap_replicates"])
    bootstrap_seed = int(config["metrics"]["bootstrap_seed"])
    ci_fields = [
        "diffjscc_minus_author_jscc_psnr",
        "diffjscc_minus_author_jscc_lpips",
        "current_minus_diffjscc_psnr",
        "current_minus_diffjscc_lpips",
        "current_minus_diffjscc_failure",
        "b1_minus_diffjscc_psnr",
        "b1_minus_diffjscc_lpips",
        "b1_minus_diffjscc_failure",
    ]
    cis = {
        field: cluster_ci(rows, field, replicates, bootstrap_seed + index)
        for index, field in enumerate(ci_fields)
    }
    by_snr: dict[str, Any] = {}
    for snr in sorted({float(row["snr_db"]) for row in rows}):
        selected = [row for row in rows if float(row["snr_db"]) == snr]
        by_snr[str(int(snr))] = {
            "rows": len(selected),
            "diffjscc_psnr": float(np.mean([float(row["diffjscc_psnr"]) for row in selected])),
            "diffjscc_lpips": float(np.mean([float(row["diffjscc_lpips"]) for row in selected])),
            "current_minus_diffjscc_psnr": float(
                np.mean([float(row["current_minus_diffjscc_psnr"]) for row in selected])
            ),
            "current_minus_diffjscc_lpips": float(
                np.mean([float(row["current_minus_diffjscc_lpips"]) for row in selected])
            ),
            "b1_minus_diffjscc_psnr": float(
                np.mean([float(row["b1_minus_diffjscc_psnr"]) for row in selected])
            ),
            "b1_minus_diffjscc_lpips": float(
                np.mean([float(row["b1_minus_diffjscc_lpips"]) for row in selected])
            ),
            "diffjscc_failures": sum(as_bool(row["diffjscc_failure"]) for row in selected),
            "current_failures": sum(as_bool(row["current_failure"]) for row in selected),
            "b1_failures": sum(as_bool(row["b1_failure"]) for row in selected),
            "outside_author_training_snr": snr
            in set(map(float, config["channel"]["extrapolation_snrs_db"])),
        }
    current_dominates = (
        cis["current_minus_diffjscc_psnr"][0] > 0
        and cis["current_minus_diffjscc_lpips"][1] < 0
        and cis["current_minus_diffjscc_failure"][1] <= 0
    )
    diffjscc_dominates = (
        cis["current_minus_diffjscc_psnr"][1] < 0
        and cis["current_minus_diffjscc_lpips"][0] > 0
        and cis["current_minus_diffjscc_failure"][0] >= 0
    )
    verdict = (
        "CURRENT_STRICTLY_DOMINATES"
        if current_dominates
        else "DIFFJSCC_STRICTLY_DOMINATES"
        if diffjscc_dominates
        else "PARETO_OR_INCONCLUSIVE"
    )
    if stage == "smoke":
        verdict = "SMOKE_ONLY_NO_COMPARATIVE_CLAIM"
    return {
        "analysis_id": f"ANALYSIS-S30-DIFFJSCC-{stage.upper()}-001",
        "status": "PASS",
        "stage": stage,
        "rows": len(rows),
        "unique_samples": len({row["sample_id"] for row in rows}),
        "means": means,
        "cluster_bootstrap_95ci": cis,
        "failures": {
            "author_jscc": sum(as_bool(row["author_jscc_failure"]) for row in rows),
            "diffjscc": sum(as_bool(row["diffjscc_failure"]) for row in rows),
            "current": sum(as_bool(row["current_failure"]) for row in rows),
            "b1": sum(as_bool(row["b1_failure"]) for row in rows),
            "diffjscc_new_vs_author_jscc": sum(
                as_bool(row["diffjscc_new_error_vs_author_jscc"]) for row in rows
            ),
            "diffjscc_repair_vs_author_jscc": sum(
                as_bool(row["diffjscc_repair_vs_author_jscc"]) for row in rows
            ),
        },
        "by_snr": by_snr,
        "rate": config["rate"],
        "verdict": verdict,
        "claim_scope": config["protocol"]["claim_scope"],
    }


def stage_contract(config: dict[str, Any], stage: str) -> tuple[list[int], list[float], int, Path]:
    if stage == "smoke":
        spec = config["population"]["smoke"]
        return (
            list(map(int, spec["channel_seeds"])),
            list(map(float, spec["snrs_db"])),
            int(spec["sample_count"]),
            resolve(config["outputs"]["smoke"]),
        )
    if stage == "first-seed":
        spec = config["population"]["first_seed_stage"]
        return (
            list(map(int, spec["channel_seeds"])),
            list(map(float, spec["snrs_db"])),
            int(spec["sample_count"]),
            resolve(config["outputs"]["first_seed_stage"]),
        )
    if stage == "full":
        return (
            list(map(int, config["population"]["channel_seeds"])),
            list(map(float, config["population"]["snrs_db"])),
            int(config["population"]["expected_sample_count"]),
            resolve(config["outputs"]["full"]),
        )
    raise ValueError(stage)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/s30_diffjscc_external_comparison.yaml")
    parser.add_argument("--stage", choices=["preload", "smoke", "first-seed", "full"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = load_yaml(config_path)
    if config["protocol"]["status"] != "preregistered_before_any_diffjscc_reconstruction":
        raise RuntimeError("S30 config is no longer preregistered")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("official 512px DiffJSCC comparison requires CUDA")
    torch.cuda.set_device(device)

    state = checkpoint_state(config)
    model, load_audit = instantiate_official_model(config, state)
    del state
    model.preprocess_model.forward = types.MethodType(
        fixed_deepjscc_forward, model.preprocess_model
    )
    model.preprocess_model._s30_expected_real_symbols = int(
        config["rate"]["diffjscc_real_symbols"]
    )
    model.to(device)
    model.eval().requires_grad_(False)
    if args.stage == "preload":
        print(json.dumps({"status": "PASS", "load_audit": load_audit}, indent=2))
        return

    seeds, snrs, sample_count, output = stage_contract(config, args.stage)
    samples, classes = load_population(config)
    samples = samples[:sample_count]
    current_rows = read_csv(
        require_sha(
            config["inputs"]["current_per_sample"],
            config["inputs"]["current_per_sample_sha256"],
        )
    )
    current_by_key = {
        (str(row["sample_id"]), int(row["base_seed"]), float(row["snr_db"])): row
        for row in current_rows
    }
    b1_rows = read_csv(
        require_sha(
            config["inputs"]["b1_per_sample"],
            config["inputs"]["b1_per_sample_sha256"],
        )
    )
    b1_by_key = {
        (str(row["sample_id"]), int(row["base_seed"]), float(row["snr_db"])): row
        for row in b1_rows
    }
    if set(current_by_key) != set(b1_by_key):
        raise RuntimeError("frozen S28 current/B1 key sets differ")

    evaluator, evaluator_temperature = load_scratch_classifier(
        str(
            require_sha(
                config["inputs"]["t_cls_checkpoint"],
                config["inputs"]["t_cls_checkpoint_sha256"],
            )
        ),
        classes,
        device,
        str(config["evaluator"]["expected_role"]),
    )
    lpips_model, lpips_error = try_load_lpips(device, resolve("outputs/cache"))
    if lpips_model is None:
        raise RuntimeError(lpips_error)
    lpips_model.eval().requires_grad_(False)
    eval_cfg = evaluator_config(config)
    source_transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()]
    )
    to_tensor = transforms.ToTensor()
    targets_cpu = [
        source_transform(Image.open(item["path"]).convert("RGB")) for item in samples
    ]

    if output.exists() and not args.resume:
        raise FileExistsError(output)
    if not output.exists():
        output.mkdir(parents=True)
        (output / "images").mkdir()
        shutil.copy2(config_path, output / "config_snapshot.yaml")
        shutil.copy2(SCRIPT, output / SCRIPT.name)
        (output / "load_audit.json").write_text(
            json.dumps(load_audit, indent=2) + "\n", encoding="utf-8"
        )
    elif not (output / "config_snapshot.yaml").is_file():
        raise RuntimeError("resume output has no config snapshot")
    partial = output / "rows.partial.jsonl"
    rows: list[dict[str, Any]] = []
    if partial.is_file():
        rows = [json.loads(line) for line in partial.read_text(encoding="utf-8").splitlines()]
    completed = {
        (str(row["sample_id"]), int(row["base_seed"]), float(row["snr_db"]))
        for row in rows
    }

    for base_seed in seeds:
        for snr in snrs:
            for item, target_cpu in zip(samples, targets_cpu):
                sample_id = str(item["sample_id"])
                key = (sample_id, base_seed, snr)
                if key in completed:
                    continue
                current = current_by_key[key]
                b1 = b1_by_key[key]
                target = target_cpu.unsqueeze(0).to(device)
                label = int(item["class_idx"])
                with torch.no_grad():
                    source_prob = evaluate_probabilities(
                        evaluator, evaluator_temperature, target, eval_cfg
                    )
                source_conf, source_pred = source_prob.max(dim=1)
                if int(source_pred[0]) != label or float(source_conf[0]) < float(
                    config["evaluator"]["clean_confidence_threshold"]
                ):
                    raise RuntimeError(f"source is no longer T_cls clean-correct: {sample_id}")

                source_pil = transforms.ToPILImage()(target_cpu)
                resized = auto_resize(source_pil, 512)
                control = pad(np.array(resized), scale=64)
                full_noise = canonical_standard_normal(
                    base_seed,
                    sample_id,
                    snr,
                    int(config["channel"]["canonical_noise_real_symbols"]),
                )
                prefix = full_noise[: int(config["rate"]["diffjscc_real_symbols"])]
                if canonical_noise_sha256(full_noise) != current["canonical_noise_sha256"]:
                    raise RuntimeError(f"canonical noise mismatch with S28: {key}")
                model.preprocess_model._s30_snr_db = snr
                model.preprocess_model._s30_standard_normal = prefix.unsqueeze(0)
                rng_seed = sampler_seed(base_seed, sample_id, snr)
                torch.cuda.reset_peak_memory_stats(device)
                pred_np, jscc_np, caption, timing = author_process(
                    model,
                    control,
                    int(config["author_inference"]["sampling_steps"]),
                    rng_seed,
                    device,
                )
                pred_np = pred_np[: resized.height, : resized.width]
                jscc_np = jscc_np[: resized.height, : resized.width]
                pred_np = np.array(
                    Image.fromarray(pred_np).resize(source_pil.size, Image.Resampling.LANCZOS)
                )
                jscc_np = np.array(
                    Image.fromarray(jscc_np).resize(source_pil.size, Image.Resampling.LANCZOS)
                )
                pred = to_tensor(pred_np).unsqueeze(0).to(device)
                jscc = to_tensor(jscc_np).unsqueeze(0).to(device)
                jscc_metrics = metric_triplet(target, jscc, lpips_model)
                diff_metrics = metric_triplet(target, pred, lpips_model)
                with torch.no_grad():
                    candidates = torch.cat((jscc, pred), dim=0)
                    probability = evaluate_probabilities(
                        evaluator, evaluator_temperature, candidates, eval_cfg
                    )
                    prediction = probability.argmax(dim=1)
                jscc_correct = int(prediction[0]) == label
                diff_correct = int(prediction[1]) == label
                current_correct = int(current["final_prediction"]) == label
                b1_correct = int(b1["final_prediction"]) == label
                observation = dict(model.preprocess_model._s30_last_observation)
                row = {
                    "sample_id": sample_id,
                    "wnid": item["wnid"],
                    "class_idx": label,
                    "base_seed": base_seed,
                    "snr_db": snr,
                    "outside_author_training_snr": snr
                    in set(map(float, config["channel"]["extrapolation_snrs_db"])),
                    "canonical_noise_sha256": canonical_noise_sha256(full_noise),
                    "diffjscc_noise_prefix_sha256": canonical_noise_sha256(prefix),
                    "sampler_seed": rng_seed,
                    "caption": caption,
                    "source_prediction": int(source_pred[0]),
                    "source_confidence": float(source_conf[0]),
                    "author_jscc_prediction": int(prediction[0]),
                    "author_jscc_correct": jscc_correct,
                    "author_jscc_failure": not jscc_correct,
                    "diffjscc_prediction": int(prediction[1]),
                    "diffjscc_correct": diff_correct,
                    "diffjscc_failure": not diff_correct,
                    "diffjscc_new_error_vs_author_jscc": jscc_correct and not diff_correct,
                    "diffjscc_repair_vs_author_jscc": (not jscc_correct) and diff_correct,
                    "current_prediction": int(current["final_prediction"]),
                    "current_correct": current_correct,
                    "current_failure": as_bool(current["final_failure"]),
                    "current_minus_diffjscc_failure": float(as_bool(current["final_failure"]))
                    - float(not diff_correct),
                    "b1_prediction": int(b1["final_prediction"]),
                    "b1_correct": b1_correct,
                    "b1_failure": as_bool(b1["final_failure"]),
                    "b1_minus_diffjscc_failure": float(as_bool(b1["final_failure"]))
                    - float(not diff_correct),
                    "author_jscc_psnr": jscc_metrics[0],
                    "author_jscc_ms_ssim": jscc_metrics[1],
                    "author_jscc_lpips": jscc_metrics[2],
                    "diffjscc_psnr": diff_metrics[0],
                    "diffjscc_ms_ssim": diff_metrics[1],
                    "diffjscc_lpips": diff_metrics[2],
                    "current_psnr": float(current["final_psnr"]),
                    "current_ms_ssim": float(current["final_ms_ssim"]),
                    "current_lpips": float(current["final_lpips"]),
                    "b1_psnr": float(b1["final_psnr"]),
                    "b1_ms_ssim": float(b1["final_ms_ssim"]),
                    "b1_lpips": float(b1["final_lpips"]),
                    "diffjscc_minus_author_jscc_psnr": diff_metrics[0] - jscc_metrics[0],
                    "diffjscc_minus_author_jscc_ms_ssim": diff_metrics[1] - jscc_metrics[1],
                    "diffjscc_minus_author_jscc_lpips": diff_metrics[2] - jscc_metrics[2],
                    "current_minus_diffjscc_psnr": float(current["final_psnr"])
                    - diff_metrics[0],
                    "current_minus_diffjscc_ms_ssim": float(current["final_ms_ssim"])
                    - diff_metrics[1],
                    "current_minus_diffjscc_lpips": float(current["final_lpips"])
                    - diff_metrics[2],
                    "b1_minus_diffjscc_psnr": float(b1["final_psnr"])
                    - diff_metrics[0],
                    "b1_minus_diffjscc_ms_ssim": float(b1["final_ms_ssim"])
                    - diff_metrics[1],
                    "b1_minus_diffjscc_lpips": float(b1["final_lpips"])
                    - diff_metrics[2],
                    "diffjscc_real_symbols": observation["real_symbols"],
                    "diffjscc_complex_channel_uses": observation["complex_channel_uses"],
                    "diffjscc_effective_cbr_256": float(
                        config["rate"]["effective_cbr_against_256_source"]
                    ),
                    "project_budget_fraction_used": float(
                        config["rate"]["project_budget_fraction_used"]
                    ),
                    "normalized_complex_power": observation["normalized_complex_power"],
                    "author_jscc_runtime_ms": timing["jscc_ms"],
                    "caption_runtime_ms": timing["caption_ms"],
                    "diffusion_runtime_ms": timing["diffusion_ms"],
                    "total_runtime_ms": timing["total_ms"],
                    "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
                }
                image_name = hashlib.sha256(
                    f"{sample_id}|{base_seed}|{snr}".encode("utf-8")
                ).hexdigest()[:16]
                sheet = np.concatenate(
                    (np.array(source_pil), jscc_np, pred_np), axis=1
                )
                Image.fromarray(sheet).save(output / "images" / f"{image_name}.png")
                with partial.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                rows.append(row)
                completed.add(key)
                print(
                    json.dumps(
                        {
                            "completed": len(rows),
                            "sample_id": sample_id,
                            "seed": base_seed,
                            "snr": snr,
                            "psnr": diff_metrics[0],
                            "lpips": diff_metrics[2],
                            "caption": caption,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    expected_rows = len(seeds) * len(snrs) * sample_count
    if len(rows) != expected_rows:
        raise RuntimeError(f"row count incomplete: {len(rows)} != {expected_rows}")
    write_csv(output / "per_sample.csv", rows)
    summary = summarize(config, args.stage, rows)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "STATE.json").write_text(
        json.dumps({"status": "complete", "rows": len(rows)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
