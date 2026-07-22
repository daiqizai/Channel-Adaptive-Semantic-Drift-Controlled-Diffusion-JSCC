#!/usr/bin/env python3
"""Compare a frozen strong-JSCC checkpoint on the frozen S30 population."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from cadsd_jscc.external_common import (  # noqa: E402
    canonical_noise_sha256,
    canonical_standard_normal,
)
from cadsd_jscc.metrics import ms_ssim_per_sample, psnr_per_sample  # noqa: E402
from cadsd_jscc.strong_jscc import StrongJSCC, trainable_parameter_count  # noqa: E402
from pc_imagenette_supervised_audit import (  # noqa: E402
    evaluate_probabilities,
    load_scratch_classifier,
)
from s5_residual_refiner_pilot import try_load_lpips  # noqa: E402


SCRIPT = Path(__file__).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/s32_strong_jscc_external_comparison.yaml"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


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


def require_sha(value: str | Path, expected: str) -> Path:
    path = resolve(value)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != str(expected):
        raise RuntimeError(f"SHA mismatch for {path}: {actual} != {expected}")
    return path


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return value


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


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def load_population(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    reference_path = require_sha(
        config["inputs"]["population_reference"],
        config["inputs"]["population_reference_sha256"],
    )
    reference = load_yaml(reference_path)
    population = reference["population"]
    manifest_path = require_sha(
        config["inputs"]["split_manifest"],
        config["inputs"]["split_manifest_sha256"],
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
        path = source_root / str(item["relative_path"])
        require_sha(path, str(frozen["content_sha256"]))
        if int(item["class_idx"]) != int(frozen["class_idx"]):
            raise RuntimeError(f"class mismatch: {sample_id}")
        item["path"] = path
        samples.append(item)
    expected = int(config["population"]["expected_sample_count"])
    if len(samples) != expected or len({row["sample_id"] for row in samples}) != expected:
        raise RuntimeError("frozen population size or uniqueness changed")
    return samples, classes


def build_model(checkpoint: dict[str, Any]) -> StrongJSCC:
    train_config = checkpoint["config"]
    model_config = train_config["model"]
    model = StrongJSCC(
        image_size=int(train_config["image_size"]),
        latent_channels=int(model_config["latent_channels"]),
        stage_channels=tuple(int(value) for value in model_config["stage_channels"]),
        stage_blocks=tuple(int(value) for value in model_config["stage_blocks"]),
        condition_dim=int(model_config["condition_dim"]),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def evaluator_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "imagenette": {
            "normalization_mean": config["evaluator"]["normalization_mean"],
            "normalization_std": config["evaluator"]["normalization_std"],
        }
    }


def cluster_ci(
    rows: list[dict[str, Any]], field: str, replicates: int, seed: int
) -> list[float]:
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sample_id"])].append(float(row[field]))
    values = np.asarray(
        [np.mean(grouped[key]) for key in sorted(grouped)], dtype=np.float64
    )
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(replicates, len(values)))
    boot = values[indices].mean(axis=1)
    return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def summarize(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    methods = ["strong", "author_jscc", "diffjscc", "current", "b1"]
    metrics = ["psnr", "ms_ssim", "lpips"]
    means = {
        method: {
            metric: float(np.mean([float(row[f"{method}_{metric}"]) for row in rows]))
            for metric in metrics
        }
        | {
            "failures": int(sum(as_bool(row[f"{method}_failure"]) for row in rows)),
            "failure_rate": float(
                np.mean([float(as_bool(row[f"{method}_failure"])) for row in rows])
            ),
        }
        for method in methods
    }
    references = ["author_jscc", "diffjscc", "current", "b1"]
    replicates = int(config["metrics"]["bootstrap_replicates"])
    seed = int(config["metrics"]["bootstrap_seed"])
    intervals: dict[str, list[float]] = {}
    deltas: dict[str, float] = {}
    for reference in references:
        for metric in metrics + ["failure"]:
            field = f"strong_minus_{reference}_{metric}"
            deltas[field] = float(np.mean([float(row[field]) for row in rows]))
            intervals[field] = cluster_ci(rows, field, replicates, seed)
    by_snr: dict[str, Any] = {}
    for snr in map(float, config["population"]["snrs_db"]):
        selected = [row for row in rows if float(row["snr_db"]) == snr]
        by_snr[str(int(snr))] = {
            "rows": len(selected),
            "strong": {
                metric: float(
                    np.mean([float(row[f"strong_{metric}"]) for row in selected])
                )
                for metric in metrics
            }
            | {
                "failures": int(
                    sum(as_bool(row["strong_failure"]) for row in selected)
                )
            },
            "author_jscc": {
                metric: float(
                    np.mean([float(row[f"author_jscc_{metric}"]) for row in selected])
                )
                for metric in metrics
            }
            | {
                "failures": int(
                    sum(as_bool(row["author_jscc_failure"]) for row in selected)
                )
            },
            "strong_minus_author_jscc_psnr": float(
                np.mean(
                    [float(row["strong_minus_author_jscc_psnr"]) for row in selected]
                )
            ),
            "strong_minus_author_jscc_ms_ssim": float(
                np.mean(
                    [
                        float(row["strong_minus_author_jscc_ms_ssim"])
                        for row in selected
                    ]
                )
            ),
            "strong_minus_author_jscc_lpips": float(
                np.mean(
                    [float(row["strong_minus_author_jscc_lpips"]) for row in selected]
                )
            ),
            "strong_minus_current_psnr": float(
                np.mean([float(row["strong_minus_current_psnr"]) for row in selected])
            ),
        }
    author_psnr_ci = intervals["strong_minus_author_jscc_psnr"]
    author_lpips_ci = intervals["strong_minus_author_jscc_lpips"]
    if author_psnr_ci[0] > 0 and author_lpips_ci[1] < 0:
        author_relation = "STRONG_QUALITY_DOMINATES_AUTHOR_JSCC"
    elif author_psnr_ci[1] < 0 and author_lpips_ci[0] > 0:
        author_relation = "AUTHOR_JSCC_QUALITY_DOMINATES_STRONG"
    else:
        author_relation = "PARETO_OR_INCONCLUSIVE"
    margin = float(config.get("claim_rule", {}).get("noninferiority_margin_db", 0.0))
    if author_psnr_ci[0] > 0.0:
        author_psnr_verdict = "SIGNIFICANTLY_SUPERIOR"
    elif margin > 0.0 and author_psnr_ci[0] > -margin:
        author_psnr_verdict = "NONINFERIOR_WITHIN_PREREGISTERED_MARGIN"
    elif margin > 0.0 and author_psnr_ci[0] < -margin:
        author_psnr_verdict = "INFERIOR_UNDER_PREREGISTERED_MARGIN"
    elif margin > 0.0:
        author_psnr_verdict = "EXACTLY_ON_PREREGISTERED_MARGIN_BOUNDARY"
    else:
        author_psnr_verdict = "NO_NONINFERIORITY_MARGIN_REGISTERED"
    author_new = sum(
        as_bool(row["strong_failure"]) and not as_bool(row["author_jscc_failure"])
        for row in rows
    )
    author_repair = sum(
        not as_bool(row["strong_failure"]) and as_bool(row["author_jscc_failure"])
        for row in rows
    )
    author_new_clusters = len(
        {
            str(row["sample_id"])
            for row in rows
            if as_bool(row["strong_failure"])
            and not as_bool(row["author_jscc_failure"])
        }
    )
    author_repair_clusters = len(
        {
            str(row["sample_id"])
            for row in rows
            if not as_bool(row["strong_failure"])
            and as_bool(row["author_jscc_failure"])
        }
    )
    float_gap = {
        metric: float(
            np.mean([float(row[f"strong_float_minus_uint8_{metric}"]) for row in rows])
        )
        for metric in metrics
    }
    return {
        "analysis_id": config["analysis_id"],
        "status": "PASS",
        "rows": len(rows),
        "unique_samples": len({row["sample_id"] for row in rows}),
        "means_primary_uint8": means,
        "strong_minus_references": deltas,
        "source_image_cluster_95ci": intervals,
        "by_snr": by_snr,
        "float_minus_uint8_sensitivity": float_gap,
        "author_relation": author_relation,
        "author_psnr_verdict": author_psnr_verdict,
        "noninferiority_margin_db": margin,
        "strong_vs_author_semantic_transitions": {
            "new_error_rows": int(author_new),
            "repair_rows": int(author_repair),
            "new_error_source_clusters": int(author_new_clusters),
            "repair_source_clusters": int(author_repair_clusters),
        },
        "claim_boundary": str(
            config["protocol"].get(
                "claim_scope",
                "known policy-dev external positioning; not an independent final test",
            )
        ),
    }


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config = load_yaml(config_path)
    if config["protocol"]["status"] not in {
        "frozen_after_s31_checkpoint_selection",
        "frozen_after_s33_checkpoint_selection",
    }:
        raise RuntimeError("strong comparison requires a frozen checkpoint contract")
    if config["protocol"]["official_imagenette_validation_accessed"] is not False:
        raise RuntimeError("official Imagenette validation must remain sealed")
    if config["metrics"]["primary_quantization"] != "floor_uint8":
        raise RuntimeError("S32 primary quantization contract changed")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    checkpoint_path = require_sha(
        config["inputs"]["strong_checkpoint"],
        config["inputs"]["strong_checkpoint_sha256"],
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint["epoch"]) != int(config["inputs"]["strong_best_epoch"]):
        raise RuntimeError("strong checkpoint epoch differs from frozen contract")
    model = build_model(checkpoint)
    parameter_count = trainable_parameter_count(model)
    model = model.to(device).eval().requires_grad_(False)
    if model.real_symbols != int(config["rate"]["strong_real_symbols"]):
        raise RuntimeError("strong model rate contract changed")
    if parameter_count != int(config["model"]["trainable_parameters"]):
        raise RuntimeError("strong model parameter count changed")
    canonical_reference_symbols = int(
        config["rate"].get("canonical_noise_reference_real_symbols", model.real_symbols)
    )
    if canonical_reference_symbols < model.real_symbols:
        raise RuntimeError("canonical noise reference cannot be shorter than model rate")

    s30_path = require_sha(
        config["inputs"]["s30_per_sample"],
        config["inputs"]["s30_per_sample_sha256"],
    )
    s30_rows = read_csv(s30_path)
    s30_by_key = {
        (str(row["sample_id"]), int(row["base_seed"]), float(row["snr_db"])): row
        for row in s30_rows
    }
    expected_rows = (
        int(config["population"]["expected_sample_count"])
        * len(config["population"]["channel_seeds"])
        * len(config["population"]["snrs_db"])
    )
    if len(s30_by_key) != expected_rows:
        raise RuntimeError("S30 key count differs from frozen S32 population")

    samples, classes = load_population(config)
    evaluator, temperature = load_scratch_classifier(
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
    eval_config = evaluator_config(config)
    transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()]
    )
    targets_cpu = [
        transform(Image.open(item["path"]).convert("RGB")) for item in samples
    ]
    targets_all = torch.stack(targets_cpu).to(device)
    with torch.inference_mode():
        source_prob = evaluate_probabilities(
            evaluator, temperature, targets_all, eval_config
        )
    source_conf, source_pred = source_prob.max(dim=1)
    threshold = float(config["evaluator"]["clean_confidence_threshold"])
    for index, item in enumerate(samples):
        if int(source_pred[index]) != int(item["class_idx"]) or float(
            source_conf[index]
        ) < threshold:
            raise RuntimeError(f"source no longer T_cls clean-correct: {item['sample_id']}")

    if args.preflight:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "checkpoint_epoch": int(checkpoint["epoch"]),
                    "trainable_parameters": parameter_count,
                    "strong_real_symbols": model.real_symbols,
                    "s30_rows": len(s30_rows),
                    "population_samples": len(samples),
                    "official_imagenette_validation_accessed": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    output = resolve(config["outputs"]["directory"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / "images").mkdir()
    shutil.copy2(config_path, output / "config_snapshot.yaml")
    shutil.copy2(SCRIPT, output / SCRIPT.name)

    rows: list[dict[str, Any]] = []
    batch_size = int(config["runtime"]["batch_size"])
    latent_shape = (model.latent_channels, model.image_size // 16, model.image_size // 16)
    for base_seed in map(int, config["population"]["channel_seeds"]):
        for snr in map(float, config["population"]["snrs_db"]):
            for start in range(0, len(samples), batch_size):
                end = min(start + batch_size, len(samples))
                batch_items = samples[start:end]
                target = targets_all[start:end]
                noises = []
                old_rows = []
                for item in batch_items:
                    sample_id = str(item["sample_id"])
                    old = s30_by_key[(sample_id, base_seed, snr)]
                    reference_noise = canonical_standard_normal(
                        base_seed, sample_id, snr, canonical_reference_symbols
                    )
                    if (
                        canonical_noise_sha256(reference_noise)
                        != old["canonical_noise_sha256"]
                    ):
                        raise RuntimeError(f"canonical noise mismatch: {sample_id}/{snr}")
                    noise = reference_noise[: model.real_symbols].contiguous()
                    noises.append(noise.reshape(latent_shape))
                    old_rows.append(old)
                noise_batch = torch.stack(noises).to(device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                with torch.inference_mode():
                    strong_float, observation = model.forward_with_observation(
                        target, snr, noise_batch
                    )
                    strong_float = strong_float.clamp(0.0, 1.0)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                runtime_ms = (time.perf_counter() - started) * 1000.0 / len(target)
                # Match the S30 author arm's uint8 truncation before primary metrics.
                strong = torch.floor(strong_float * 255.0).clamp(0.0, 255.0) / 255.0
                with torch.inference_mode():
                    psnr = psnr_per_sample(strong, target)
                    msssim = ms_ssim_per_sample(strong, target)
                    lpips = lpips_model(strong * 2.0 - 1.0, target * 2.0 - 1.0).flatten()
                    float_psnr = psnr_per_sample(strong_float, target)
                    float_msssim = ms_ssim_per_sample(strong_float, target)
                    float_lpips = lpips_model(
                        strong_float * 2.0 - 1.0, target * 2.0 - 1.0
                    ).flatten()
                    probabilities = evaluate_probabilities(
                        evaluator, temperature, strong, eval_config
                    )
                    prediction = probabilities.argmax(dim=1)
                for offset, (item, old) in enumerate(zip(batch_items, old_rows)):
                    label = int(item["class_idx"])
                    strong_failure = int(prediction[offset]) != label
                    row: dict[str, Any] = {
                        "sample_id": str(item["sample_id"]),
                        "wnid": item["wnid"],
                        "class_idx": label,
                        "base_seed": base_seed,
                        "snr_db": snr,
                        "canonical_noise_sha256": old["canonical_noise_sha256"],
                        "strong_noise_prefix_sha256": canonical_noise_sha256(
                            noises[offset].reshape(-1)
                        ),
                        "strong_prediction": int(prediction[offset]),
                        "strong_failure": strong_failure,
                        "strong_psnr": float(psnr[offset]),
                        "strong_ms_ssim": float(msssim[offset]),
                        "strong_lpips": float(lpips[offset]),
                        "strong_float_psnr": float(float_psnr[offset]),
                        "strong_float_ms_ssim": float(float_msssim[offset]),
                        "strong_float_lpips": float(float_lpips[offset]),
                        "strong_float_minus_uint8_psnr": float(
                            float_psnr[offset] - psnr[offset]
                        ),
                        "strong_float_minus_uint8_ms_ssim": float(
                            float_msssim[offset] - msssim[offset]
                        ),
                        "strong_float_minus_uint8_lpips": float(
                            float_lpips[offset] - lpips[offset]
                        ),
                        "strong_normalized_power": float(
                            observation.normalized_power[offset]
                        ),
                        "strong_runtime_ms": runtime_ms,
                    }
                    for method in ["author_jscc", "diffjscc", "current", "b1"]:
                        row[f"{method}_prediction"] = int(old[f"{method}_prediction"])
                        row[f"{method}_failure"] = as_bool(old[f"{method}_failure"])
                        for metric in ["psnr", "ms_ssim", "lpips"]:
                            row[f"{method}_{metric}"] = float(old[f"{method}_{metric}"])
                            row[f"strong_minus_{method}_{metric}"] = float(
                                row[f"strong_{metric}"] - row[f"{method}_{metric}"]
                            )
                        row[f"strong_minus_{method}_failure"] = float(strong_failure) - float(
                            row[f"{method}_failure"]
                        )
                    rows.append(row)
                if start == 0:
                    save_image(
                        torch.cat((target[:4].cpu(), strong[:4].cpu())),
                        output
                        / "images"
                        / f"seed_{base_seed}_snr_{int(snr):02d}.png",
                        nrow=min(4, len(target)),
                    )

    if len(rows) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} rows, got {len(rows)}")
    power_error = max(abs(float(row["strong_normalized_power"]) - 1.0) for row in rows)
    if power_error > float(config["rate"]["normalized_power_abs_error_max"]):
        raise RuntimeError("strong normalized-power audit failed")
    for row in rows:
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise RuntimeError(f"non-finite {key}")
    write_csv(output / "per_sample.csv", rows)
    summary = summarize(config, rows)
    summary["audit"] = {
        "config": relative(config_path),
        "config_sha256": sha256_file(config_path),
        "script": relative(SCRIPT),
        "script_sha256": sha256_file(SCRIPT),
        "strong_checkpoint": relative(checkpoint_path),
        "strong_checkpoint_sha256": sha256_file(checkpoint_path),
        "strong_best_epoch": int(checkpoint["epoch"]),
        "trainable_parameters": parameter_count,
        "strong_real_symbols": model.real_symbols,
        "strong_complex_channel_uses": model.real_symbols // 2,
        "canonical_noise_reference_real_symbols": canonical_reference_symbols,
        "normalized_power_max_abs_error": power_error,
        "official_imagenette_validation_accessed": False,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
