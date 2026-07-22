#!/usr/bin/env python3
"""Cluster-paired bootstrap for the frozen S19 fusion holdout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def interval(values: list[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values),
        "ci_low": quantile(values, 0.025),
        "ci_high": quantile(values, 0.975),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/s19_diffusion_fusion_ablation.yaml")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["protocol"]["status"] != "models_frozen_before_holdout_output":
        raise RuntimeError("S19 bootstrap requires frozen model status")
    holdout = resolve(config["outputs"]["holdout_dir"])
    state = json.loads((holdout / "STATE.json").read_text(encoding="utf-8"))
    if state.get("state") != "HOLDOUT_COMPLETE":
        raise RuntimeError("S19 holdout is incomplete")
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
    comparisons = (
        ("fusion", "control"),
        ("fusion", "b1"),
        ("fusion", "diffusion"),
        ("fusion", "b0"),
    )
    cluster_values: dict[str, dict[str, float]] = {}
    for name in names:
        values: dict[str, float] = {}
        for left, right in comparisons:
            for metric in ("psnr", "lpips"):
                key = f"{left}_minus_{right}_{metric}"
                values[key] = sum(
                    float(row[f"{left}_{metric}"]) - float(row[f"{right}_{metric}"])
                    for row in grouped[name]
                ) / expected_snrs
        cluster_values[name] = values
    replicates = int(config["evaluation"]["bootstrap_replicates"])
    rng = random.Random(int(config["evaluation"]["bootstrap_seed"]))
    distributions: defaultdict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        sampled = [names[rng.randrange(len(names))] for _ in names]
        for key in next(iter(cluster_values.values())):
            distributions[key].append(
                sum(cluster_values[name][key] for name in sampled) / len(sampled)
            )
    intervals = {key: interval(values) for key, values in distributions.items()}
    summary = json.loads((holdout / "summary.json").read_text(encoding="utf-8"))
    criteria = config["success_criteria"]
    checks = {
        "primary_fusion_minus_control_psnr_ci_low": intervals[
            "fusion_minus_control_psnr"
        ]["ci_low"]
        > float(criteria["primary_fusion_minus_control_psnr_ci_low_min_db"]),
        "fusion_minus_control_lpips": float(summary["fusion_minus_control_lpips"])
        <= float(criteria["fusion_minus_control_mean_lpips_max"]),
        "fusion_minus_control_nonnegative_snr_count": sum(
            float(item["fusion_minus_control_psnr"]) >= 0 for item in summary["per_snr"]
        )
        >= int(criteria["fusion_minus_control_nonnegative_snr_count_min"]),
        "fusion_minus_b1_psnr": float(summary["fusion_minus_b1_psnr"])
        > float(criteria["fusion_minus_b1_mean_psnr_min_db"]),
        "fusion_alexnet_new_not_greater_than_repair": int(summary["alexnet_fusion_new"])
        <= int(summary["alexnet_fusion_repair"]),
        "fusion_majority_new_not_greater_than_repair": int(summary["majority_fusion_new"])
        <= int(summary["majority_fusion_repair"]),
        "fusion_majority_new_not_greater_than_control": int(summary["majority_fusion_new"])
        <= int(summary["majority_control_new"]),
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
        "primary_complementary_information_demonstrated": checks[
            "primary_fusion_minus_control_psnr_ci_low"
        ],
        "official_imagenette_accessed": False,
    }
    output = resolve(config["outputs"]["bootstrap_dir"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    shutil.copy2(config_path, output / "config.yaml")
    shutil.copy2(SCRIPT, output / SCRIPT.name)
    (output / "bootstrap_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "STATE.json").write_text(
        json.dumps({"state": "COMPLETE", **result}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
