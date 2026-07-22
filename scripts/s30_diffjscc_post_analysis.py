#!/usr/bin/env python3
"""Derive range-separated statistics from a completed S30 DiffJSCC run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("input CSV is empty")
    return rows


def clustered_ci(
    rows: list[dict[str, Any]], field: str, replicates: int, seed: int
) -> list[float]:
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sample_id"])].append(float(row[field]))
    values = np.asarray(
        [np.mean(grouped[sample_id]) for sample_id in sorted(grouped)],
        dtype=np.float64,
    )
    if len(values) < 2:
        return [float(values.mean()), float(values.mean())]
    generator = np.random.default_rng(seed)
    sample = generator.integers(0, len(values), size=(replicates, len(values)))
    means = values[sample].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def mean(rows: list[dict[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def failure_count(rows: list[dict[str, Any]], field: str) -> int:
    return sum(as_bool(row[field]) for row in rows)


def method_summary(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    failures = failure_count(rows, f"{prefix}_failure")
    return {
        "psnr": mean(rows, f"{prefix}_psnr"),
        "ms_ssim": mean(rows, f"{prefix}_ms_ssim"),
        "lpips": mean(rows, f"{prefix}_lpips"),
        "failures": failures,
        "failure_rate": failures / len(rows),
    }


def add_derived_author_jscc_deltas(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["diffjscc_minus_author_jscc_failure"] = float(
            as_bool(row["diffjscc_failure"])
        ) - float(as_bool(row["author_jscc_failure"]))
        for method in ("current", "b1"):
            for metric in ("psnr", "ms_ssim", "lpips"):
                row[f"{method}_minus_author_jscc_{metric}"] = (
                    float(row[f"{method}_{metric}"])
                    - float(row[f"author_jscc_{metric}"])
                )
            row[f"{method}_minus_author_jscc_failure"] = float(
                as_bool(row[f"{method}_failure"])
            ) - float(as_bool(row["author_jscc_failure"]))


def range_summary(
    rows: list[dict[str, Any]], name: str, replicates: int, seed: int
) -> dict[str, Any]:
    fields = [
        "diffjscc_minus_author_jscc_psnr",
        "diffjscc_minus_author_jscc_ms_ssim",
        "diffjscc_minus_author_jscc_lpips",
        "diffjscc_minus_author_jscc_failure",
        "current_minus_diffjscc_psnr",
        "current_minus_diffjscc_ms_ssim",
        "current_minus_diffjscc_lpips",
        "current_minus_diffjscc_failure",
        "b1_minus_diffjscc_psnr",
        "b1_minus_diffjscc_ms_ssim",
        "b1_minus_diffjscc_lpips",
        "b1_minus_diffjscc_failure",
        "current_minus_author_jscc_psnr",
        "current_minus_author_jscc_ms_ssim",
        "current_minus_author_jscc_lpips",
        "current_minus_author_jscc_failure",
        "b1_minus_author_jscc_psnr",
        "b1_minus_author_jscc_ms_ssim",
        "b1_minus_author_jscc_lpips",
        "b1_minus_author_jscc_failure",
    ]
    deltas = {
        field: {
            "mean": mean(rows, field),
            "source_image_cluster_95ci": clustered_ci(
                rows, field, replicates, seed + index
            ),
        }
        for index, field in enumerate(fields)
    }
    return {
        "name": name,
        "rows": len(rows),
        "unique_samples": len({str(row["sample_id"]) for row in rows}),
        "snrs_db": sorted({float(row["snr_db"]) for row in rows}),
        "methods": {
            "author_jscc": method_summary(rows, "author_jscc"),
            "diffjscc": method_summary(rows, "diffjscc"),
            "current": method_summary(rows, "current"),
            "b1": method_summary(rows, "b1"),
        },
        "deltas": deltas,
        "diffjscc_semantic_events_vs_author_jscc": {
            "new_error_rows": failure_count(
                rows, "diffjscc_new_error_vs_author_jscc"
            ),
            "repair_rows": failure_count(rows, "diffjscc_repair_vs_author_jscc"),
            "new_error_source_clusters": len(
                {
                    str(row["sample_id"])
                    for row in rows
                    if as_bool(row["diffjscc_new_error_vs_author_jscc"])
                }
            ),
            "repair_source_clusters": len(
                {
                    str(row["sample_id"])
                    for row in rows
                    if as_bool(row["diffjscc_repair_vs_author_jscc"])
                }
            ),
        },
    }


def semantic_event_rows(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    keep = (
        "sample_id",
        "base_seed",
        "snr_db",
        "class_idx",
        "caption",
        "author_jscc_prediction",
        "diffjscc_prediction",
        "current_prediction",
        "author_jscc_psnr",
        "diffjscc_psnr",
        "current_psnr",
        "author_jscc_lpips",
        "diffjscc_lpips",
        "current_lpips",
    )
    return [
        {key: row[key] for key in keep}
        for row in rows
        if as_bool(row[field])
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=(
            "outputs/external_baselines/"
            "ANALYSIS-S30-DIFFJSCC-COMPARISON-001/per_sample.csv"
        ),
    )
    parser.add_argument(
        "--config", default="configs/s30_diffjscc_external_comparison.yaml"
    )
    parser.add_argument(
        "--output",
        default=(
            "outputs/external_baselines/"
            "ANALYSIS-S30-DIFFJSCC-COMPARISON-001/post_analysis_v3.json"
        ),
    )
    args = parser.parse_args()

    input_path = resolve(args.input)
    config_path = resolve(args.config)
    output_path = resolve(args.output)
    if output_path.exists():
        raise FileExistsError(output_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rows = load_rows(input_path)
    add_derived_author_jscc_deltas(rows)

    expected_rows = int(config["population"]["expected_rows_full"])
    expected_snrs = sorted(map(float, config["population"]["snrs_db"]))
    expected_seeds = sorted(map(int, config["population"]["channel_seeds"]))
    keys = {
        (str(row["sample_id"]), int(row["base_seed"]), float(row["snr_db"]))
        for row in rows
    }
    if len(rows) != expected_rows or len(keys) != expected_rows:
        raise RuntimeError(
            f"row/key contract failed: rows={len(rows)}, keys={len(keys)}, "
            f"expected={expected_rows}"
        )
    if sorted({float(row["snr_db"]) for row in rows}) != expected_snrs:
        raise RuntimeError("SNR set changed")
    if sorted({int(row["base_seed"]) for row in rows}) != expected_seeds:
        raise RuntimeError("channel seed set changed")
    numeric_fields = [
        "author_jscc_psnr",
        "author_jscc_ms_ssim",
        "author_jscc_lpips",
        "diffjscc_psnr",
        "diffjscc_ms_ssim",
        "diffjscc_lpips",
        "current_psnr",
        "current_ms_ssim",
        "current_lpips",
        "normalized_complex_power",
    ]
    if any(
        not math.isfinite(float(row[field]))
        for row in rows
        for field in numeric_fields
    ):
        raise RuntimeError("non-finite metric detected")

    extrapolation = set(map(float, config["channel"]["extrapolation_snrs_db"]))
    in_range = [row for row in rows if float(row["snr_db"]) not in extrapolation]
    out_range = [row for row in rows if float(row["snr_db"]) in extrapolation]
    replicates = int(config["metrics"]["bootstrap_replicates"])
    bootstrap_seed = int(config["metrics"]["bootstrap_seed"])
    ranges = {
        "all_preregistered_snrs": range_summary(
            rows, "all_preregistered_snrs", replicates, bootstrap_seed
        ),
        "within_author_training_snr": range_summary(
            in_range,
            "within_author_training_snr",
            replicates,
            bootstrap_seed + 100,
        ),
        "author_snr_extrapolation": range_summary(
            out_range,
            "author_snr_extrapolation",
            replicates,
            bootstrap_seed + 200,
        ),
    }
    all_delta = ranges["all_preregistered_snrs"]["deltas"]
    current_psnr_ci = all_delta["current_minus_diffjscc_psnr"][
        "source_image_cluster_95ci"
    ]
    current_lpips_ci = all_delta["current_minus_diffjscc_lpips"][
        "source_image_cluster_95ci"
    ]
    current_failure_ci = all_delta["current_minus_diffjscc_failure"][
        "source_image_cluster_95ci"
    ]
    if (
        current_psnr_ci[0] > 0
        and current_lpips_ci[1] < 0
        and current_failure_ci[1] <= 0
    ):
        verdict = "CURRENT_STRICTLY_DOMINATES"
    elif (
        current_psnr_ci[1] < 0
        and current_lpips_ci[0] > 0
        and current_failure_ci[0] >= 0
    ):
        verdict = "DIFFJSCC_STRICTLY_DOMINATES"
    else:
        verdict = "PARETO_OR_INCONCLUSIVE"

    by_snr = {
        str(int(snr)): range_summary(
            [row for row in rows if float(row["snr_db"]) == snr],
            f"snr_{int(snr)}_db",
            replicates,
            bootstrap_seed + 300 + index * 20,
        )
        for index, snr in enumerate(expected_snrs)
    }
    result = {
        "analysis_id": "ANALYSIS-S30-DIFFJSCC-POST-003",
        "status": "PASS",
        "input": {
            "per_sample_csv": str(input_path.relative_to(ROOT)),
            "per_sample_sha256": sha256_file(input_path),
            "config": str(config_path.relative_to(ROOT)),
            "config_sha256": sha256_file(config_path),
        },
        "contract": {
            "rows": len(rows),
            "unique_keys": len(keys),
            "unique_samples": len({str(row["sample_id"]) for row in rows}),
            "channel_seeds": expected_seeds,
            "snrs_db": expected_snrs,
            "all_metrics_finite": True,
            "normalized_complex_power_min": min(
                float(row["normalized_complex_power"]) for row in rows
            ),
            "normalized_complex_power_max": max(
                float(row["normalized_complex_power"]) for row in rows
            ),
        },
        "ranges": ranges,
        "by_snr": by_snr,
        "semantic_event_rows": {
            "diffjscc_new_error_vs_author_jscc": semantic_event_rows(
                rows, "diffjscc_new_error_vs_author_jscc"
            ),
            "diffjscc_repair_vs_author_jscc": semantic_event_rows(
                rows, "diffjscc_repair_vs_author_jscc"
            ),
        },
        "systems": {
            "mean_author_jscc_runtime_ms": mean(rows, "author_jscc_runtime_ms"),
            "mean_caption_runtime_ms": mean(rows, "caption_runtime_ms"),
            "mean_diffusion_runtime_ms": mean(rows, "diffusion_runtime_ms"),
            "mean_total_runtime_ms": mean(rows, "total_runtime_ms"),
            "peak_gpu_memory_mib": max(
                float(row["peak_gpu_memory_mib"]) for row in rows
            ),
        },
        "rate": config["rate"],
        "verdict": verdict,
        "claim_boundary": (
            "External positioning on a frozen policy-development population; "
            "not a claim of beating all published methods."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
