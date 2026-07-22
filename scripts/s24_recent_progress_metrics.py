#!/usr/bin/env python3
"""Reaggregate recent frozen results into comparable quality, semantic, and cost tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from s19_train_and_evaluate_fusion import FusionPairDataset  # noqa: E402
from s21_b1_anchored_gated_fusion import (  # noqa: E402
    anchor_output,
    load_config,
    resolve,
    seed_everything,
)
from s22_b1_feature_injection import (  # noqa: E402
    _b1_gate,
    envelopes,
    load_feature_model,
)


CLASSIFIERS = ("alexnet", "resnet18", "mobilenet_v3_small")
METRICS = ("psnr", "ms_ssim", "lpips")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate(config: dict[str, Any]) -> None:
    if config["protocol"]["status"] != "preregistered_before_derived_aggregation_output":
        raise RuntimeError("S24 derived aggregation is not executable")
    if config["protocol"]["official_imagenette_accessed"] is not False:
        raise RuntimeError("official Imagenette validation must remain sealed")
    for key in ("s19_per_sample", "s19_bootstrap", "s20_summary", "s23_per_sample", "s23_bootstrap", "s23_config"):
        path = resolve(config["inputs"][key])
        expected = config["inputs"][f"{key}_sha256"]
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"frozen input hash mismatch: {key}")
    if int(config["bootstrap"]["replicates"]) != 10000:
        raise RuntimeError("bootstrap replicate count changed")


def majority_correct(row: dict[str, str], stage: str) -> bool:
    return sum(
        int(row[f"{name}_{stage}_prediction"])
        == int(row[f"{name}_original_prediction"])
        for name in CLASSIFIERS
    ) >= 2


def stage_summary(
    dataset: str,
    rows: list[dict[str, str]],
    stage: str,
    threshold: float,
    snr: float | None,
) -> dict[str, Any]:
    subset = rows if snr is None else [row for row in rows if float(row["snr_db"]) == snr]
    eligible = [row for row in subset if float(row["alexnet_original_confidence"]) >= threshold]
    majority_failures = sum(not majority_correct(row, stage) for row in subset)
    alexnet_failures = sum(
        int(row[f"alexnet_{stage}_prediction"]) != int(row["alexnet_original_prediction"])
        for row in eligible
    )
    result: dict[str, Any] = {
        "dataset": dataset,
        "scope": "aggregate" if snr is None else "per_snr",
        "snr_db": "all" if snr is None else snr,
        "stage": stage,
        "rows": len(subset),
        "images": len({row["sample"] for row in subset}),
    }
    for metric in METRICS:
        result[f"mean_{metric}"] = float(
            np.mean([float(row[f"{stage}_{metric}"]) for row in subset])
        )
    result.update(
        {
            "majority_failure_count": majority_failures,
            "majority_failure_rate": majority_failures / len(subset),
            "alexnet_eligible_rows": len(eligible),
            "alexnet_failure_count": alexnet_failures,
            "alexnet_failure_rate": alexnet_failures / len(eligible),
        }
    )
    return result


def quantile_interval(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
    }


def paired_comparison(
    dataset: str,
    rows: list[dict[str, str]],
    left: str,
    right: str,
    replicates: int,
    seed: int,
    threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["sample"]].append(row)
    names = sorted(grouped)
    quality_keys = [f"{metric}_delta" for metric in METRICS]
    quality = np.asarray(
        [
            [
                np.mean(
                    [
                        float(row[f"{left}_{metric}"]) - float(row[f"{right}_{metric}"])
                        for row in grouped[name]
                    ]
                )
                for metric in METRICS
            ]
            for name in names
        ],
        dtype=np.float64,
    )
    semantic = np.asarray(
        [
            np.mean(
                [
                    float(not majority_correct(row, left))
                    - float(not majority_correct(row, right))
                    for row in grouped[name]
                ]
            )
            for name in names
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(names), size=(replicates, len(names)))
    quality_distribution = quality[indices].mean(axis=1)
    semantic_distribution = semantic[indices].mean(axis=1)
    quality_result: dict[str, Any] = {
        "dataset": dataset,
        "left": left,
        "right": right,
        "clusters": len(names),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }
    for index, key in enumerate(quality_keys):
        quality_result[key] = quantile_interval(quality_distribution[:, index])
    left_correct = [majority_correct(row, left) for row in rows]
    right_correct = [majority_correct(row, right) for row in rows]
    new_count = sum(reference and not candidate for candidate, reference in zip(left_correct, right_correct))
    repair_count = sum(not reference and candidate for candidate, reference in zip(left_correct, right_correct))
    eligible = [row for row in rows if float(row["alexnet_original_confidence"]) >= threshold]
    alex_left = [
        int(row[f"alexnet_{left}_prediction"]) == int(row["alexnet_original_prediction"])
        for row in eligible
    ]
    alex_right = [
        int(row[f"alexnet_{right}_prediction"]) == int(row["alexnet_original_prediction"])
        for row in eligible
    ]
    semantic_result = {
        "dataset": dataset,
        "left": left,
        "right": right,
        "majority_failure_rate_delta": quantile_interval(semantic_distribution),
        "majority_new_error_count": new_count,
        "majority_repair_count": repair_count,
        "majority_net_failure_change": new_count - repair_count,
        "alexnet_eligible_rows": len(eligible),
        "alexnet_new_error_count": sum(
            reference and not candidate for candidate, reference in zip(alex_left, alex_right)
        ),
        "alexnet_repair_count": sum(
            not reference and candidate for candidate, reference in zip(alex_left, alex_right)
        ),
    }
    return quality_result, semantic_result


def benchmark_cuda(
    function: Callable[[], torch.Tensor], warmup: int, iterations: int, batch_size: int
) -> dict[str, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            function()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            function()
        end.record()
        torch.cuda.synchronize()
    total_ms = float(start.elapsed_time(end))
    return {
        "total_ms": total_ms,
        "iterations": iterations,
        "batch_size": batch_size,
        "mean_ms_per_batch": total_ms / iterations,
        "mean_ms_per_image": total_ms / (iterations * batch_size),
    }


def latency_audit(config: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"status": "skipped_cuda_unavailable"}
    device = torch.device(config["latency"]["device"])
    s23 = load_config(resolve(config["inputs"]["s23_config"]))
    seed_everything(int(s23["seed"]))
    dataset = FusionPairDataset(s23, "holdout", train=False)
    loader = DataLoader(
        dataset,
        batch_size=int(config["latency"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(loader))
    b0 = batch["b0"].to(device)
    diffusion = batch["auxiliary"].to(device)
    snr = batch["snr_db"].to(device)
    snr_norm = batch["snr_norm"].to(device)
    model, b1_config, checkpoint = load_feature_model(s23, device)
    gate = _b1_gate(b1_config, snr, device)
    envelope = envelopes(s23, snr, device)
    warmup = int(config["latency"]["warmup_iterations"])
    iterations = int(config["latency"]["timed_iterations"])
    batch_size = b0.shape[0]
    b1_time = benchmark_cuda(
        lambda: anchor_output(model.b1, b1_config, b0, snr, snr_norm, device),
        warmup,
        iterations,
        batch_size,
    )
    fusion_time = benchmark_cuda(
        lambda: model(b0, diffusion, snr_norm, gate, envelope),
        warmup,
        iterations,
        batch_size,
    )
    b1_parameters = sum(parameter.numel() for parameter in model.b1.parameters())
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    return {
        "status": "complete",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "scope": config["latency"]["scope"],
        "excludes": config["latency"]["excludes"],
        "b1": b1_time,
        "s23_fusion": fusion_time,
        "s23_minus_b1_ms_per_image": fusion_time["mean_ms_per_image"]
        - b1_time["mean_ms_per_image"],
        "s23_over_b1_latency_ratio": fusion_time["mean_ms_per_image"]
        / b1_time["mean_ms_per_image"],
        "b1_parameters": b1_parameters,
        "feature_projection_parameters": int(checkpoint["trainable_parameter_count"]),
        "s23_total_parameters": total_parameters,
    }


def make_figures(output: Path, stage_rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    aggregate = {
        row["stage"]: row
        for row in stage_rows
        if row["dataset"] == "S23" and row["scope"] == "aggregate"
    }
    stages = ["b0", "diffusion", "b1", "fusion"]
    labels = ["B0 JSCC", "Matched diffusion", "B1 anchor", "S23 fusion"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    specs = (("mean_psnr", "PSNR ↑"), ("mean_ms_ssim", "MS-SSIM ↑"), ("mean_lpips", "LPIPS ↓"))
    colors = ["#9ca3af", "#60a5fa", "#34d399", "#f59e0b"]
    for axis, (key, title) in zip(axes, specs):
        values = [aggregate[stage][key] for stage in stages]
        axis.bar(labels, values, color=colors)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("S23 same independent holdout: receiver outputs")
    fig.tight_layout()
    fig.savefig(output / "s23_same_holdout_quality_readable.png", dpi=180)
    plt.close(fig)

    per_snr = summary["s23_fusion_minus_b1_per_snr"]
    snrs = [item["snr_db"] for item in per_snr]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(snrs, [item["psnr_delta"] for item in per_snr], color="#f59e0b")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("S23 fusion minus B1: PSNR")
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("dB")
    axes[1].bar(snrs, [item["lpips_delta"] for item in per_snr], color="#8b5cf6")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("S23 fusion minus B1: LPIPS")
    axes[1].set_xlabel("SNR (dB)")
    fig.tight_layout()
    fig.savefig(output / "s23_per_snr_deltas_readable.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    failure = [aggregate[stage]["majority_failure_rate"] for stage in stages]
    axis.bar(labels, failure, color=colors)
    axis.set_ylabel("3-classifier majority failure rate (lower is better)")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    axis.set_title("S23 same independent holdout: semantic diagnostic")
    fig.tight_layout()
    fig.savefig(output / "s23_semantic_failure_readable.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/s24_recent_progress_metrics.yaml")
    args = parser.parse_args()
    os.environ.setdefault("TORCH_HOME", str(resolve("outputs/cache/torch")))
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate(config)
    output = resolve(config["outputs"]["directory"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    shutil.copy2(config_path, output / "config.yaml")
    shutil.copy2(SCRIPT, output / SCRIPT.name)

    threshold = float(config["metrics"]["alexnet_original_confidence_min"])
    inputs = {
        "S19": read_rows(resolve(config["inputs"]["s19_per_sample"])),
        "S23": read_rows(resolve(config["inputs"]["s23_per_sample"])),
    }
    stage_rows: list[dict[str, Any]] = []
    for dataset, rows in inputs.items():
        stages = ("b0", "diffusion", "b1", "control", "fusion")
        for stage in stages:
            stage_rows.append(stage_summary(dataset, rows, stage, threshold, None))
            for snr in (1.0, 4.0, 7.0, 13.0, 19.0):
                stage_rows.append(stage_summary(dataset, rows, stage, threshold, snr))
    write_csv(output / "same_population_method_metrics.csv", stage_rows)

    pairs = {
        "S19": (("fusion", "control"), ("fusion", "b1"), ("b1", "b0")),
        "S23": (("fusion", "b1"), ("fusion", "diffusion"), ("fusion", "b0"), ("b1", "b0")),
    }
    quality_comparisons: list[dict[str, Any]] = []
    semantic_comparisons: list[dict[str, Any]] = []
    for dataset, comparisons in pairs.items():
        for offset, (left, right) in enumerate(comparisons):
            quality, semantic = paired_comparison(
                dataset,
                inputs[dataset],
                left,
                right,
                int(config["bootstrap"]["replicates"]),
                int(config["bootstrap"]["seed"]) + offset,
                threshold,
            )
            quality_comparisons.append(quality)
            semantic_comparisons.append(semantic)
    write_json(output / "paired_quality_comparisons.json", quality_comparisons)
    write_json(output / "paired_semantic_comparisons.json", semantic_comparisons)

    s20 = json.loads(resolve(config["inputs"]["s20_summary"]).read_text(encoding="utf-8"))
    latency = latency_audit(config)
    s23_aggregate = {
        row["stage"]: row
        for row in stage_rows
        if row["dataset"] == "S23" and row["scope"] == "aggregate"
    }
    s23_per_snr = [
        {
            "snr_db": snr,
            "psnr_delta": next(
                row["mean_psnr"]
                for row in stage_rows
                if row["dataset"] == "S23"
                and row["scope"] == "per_snr"
                and row["stage"] == "fusion"
                and row["snr_db"] == snr
            )
            - next(
                row["mean_psnr"]
                for row in stage_rows
                if row["dataset"] == "S23"
                and row["scope"] == "per_snr"
                and row["stage"] == "b1"
                and row["snr_db"] == snr
            ),
            "lpips_delta": next(
                row["mean_lpips"]
                for row in stage_rows
                if row["dataset"] == "S23"
                and row["scope"] == "per_snr"
                and row["stage"] == "fusion"
                and row["snr_db"] == snr
            )
            - next(
                row["mean_lpips"]
                for row in stage_rows
                if row["dataset"] == "S23"
                and row["scope"] == "per_snr"
                and row["stage"] == "b1"
                and row["snr_db"] == snr
            ),
        }
        for snr in (1.0, 4.0, 7.0, 13.0, 19.0)
    ]
    summary = {
        "analysis_id": config["analysis_id"],
        "claim_scope": "derived_summary_of_known_frozen_outcomes",
        "s23_same_holdout": s23_aggregate,
        "s23_fusion_minus_b1_per_snr": s23_per_snr,
        "paired_quality_comparisons": quality_comparisons,
        "paired_semantic_comparisons": semantic_comparisons,
        "s20_external_comparison": s20,
        "latency_and_parameters": latency,
        "rate_contract": {
            "active_real_symbols": 19712,
            "complex_channel_uses": 9856,
            "s19_and_s23_extra_side_information_real_symbols": 0,
            "sgd_released_main_plus_edge_real_symbols": s20["strict_rate_audit"]["released_main_plus_edge_real_symbols"],
            "sgd_minimum_caption_real_symbols": s20["strict_rate_audit"]["minimum_unprotected_caption_real_symbols"],
            "sgd_minimum_total_with_caption": s20["strict_rate_audit"]["minimum_total_with_caption"],
        },
        "comparison_warning": "S19, S23, and S20 use different frozen populations; compare absolute values only within a dataset and use paired deltas across protocols.",
        "official_imagenette_accessed": False,
        "downloaded": False,
    }
    make_figures(output, stage_rows, summary)
    write_json(output / "summary.json", summary)
    write_json(output / "STATE.json", {"state": "COMPLETE", **summary})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
