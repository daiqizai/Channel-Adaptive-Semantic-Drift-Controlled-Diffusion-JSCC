#!/usr/bin/env python3
"""Frozen image-cluster bootstrap and final gate for S18 identity control."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_INPUT_SHA256 = "b488fc0205fe3535ea5d128a99f7808ce0c701d379ed242c08e458e1502f95ea"
EXPECTED_HOLDOUT_SUMMARY_SHA256 = "593bf32ca2700ed8cd735371317682ee49e66fa270bdb92b4eb478a553e66671"
PAIRS = {
    "selected_minus_full_psnr": ("hard_identity_7db_psnr", "full_psnr"),
    "selected_minus_full_lpips": ("hard_identity_7db_lpips", "full_lpips"),
    "selected_minus_full_latent_mse": (
        "hard_identity_7db_latent_mse",
        "full_latent_mse",
    ),
    "selected_minus_b0_psnr": ("hard_identity_7db_psnr", "b0_psnr"),
    "selected_minus_b0_lpips": ("hard_identity_7db_lpips", "b0_lpips"),
    "full_minus_b0_psnr": ("full_psnr", "b0_psnr"),
    "smooth_p0p5_minus_selected_psnr": (
        "smooth_p0p5_psnr",
        "hard_identity_7db_psnr",
    ),
    "b1_minus_selected_psnr": ("b1_psnr", "hard_identity_7db_psnr"),
}


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def interval(values: list[float], replicates: int, seed: int) -> dict[str, float]:
    if len(values) < 2:
        raise ValueError("bootstrap requires at least two image clusters")
    rng = random.Random(seed)
    estimates = [
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(replicates)
    ]
    estimates.sort()
    return {
        "mean": sum(values) / len(values),
        "ci95_low": estimates[int(0.025 * replicates)],
        "ci95_high": estimates[min(replicates - 1, int(0.975 * replicates))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/s18_snr_identity_envelope.yaml")
    parser.add_argument(
        "--input",
        default="outputs/analysis/ANALYSIS-S18-IDENTITY-HOLDOUT-001/per_sample.csv",
    )
    parser.add_argument(
        "--holdout-summary",
        default="outputs/analysis/ANALYSIS-S18-IDENTITY-HOLDOUT-001/summary.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/analysis/ANALYSIS-S18-IDENTITY-BOOTSTRAP-001",
    )
    args = parser.parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    source = resolve(args.input)
    holdout_summary_path = resolve(args.holdout_summary)
    if sha256_file(source) != EXPECTED_INPUT_SHA256:
        raise RuntimeError("frozen S18 holdout CSV hash mismatch")
    if sha256_file(holdout_summary_path) != EXPECTED_HOLDOUT_SUMMARY_SHA256:
        raise RuntimeError("frozen S18 holdout summary hash mismatch")
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1280 or len({row["sample_id"] for row in rows}) != 256:
        raise RuntimeError("frozen S18 holdout dimensions changed")
    cluster_counts: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        cluster_counts[row["sample_id"]] += 1
    if set(cluster_counts.values()) != {5}:
        raise RuntimeError("each S18 image cluster must retain all five SNR rows")
    replicates = int(config["evaluation"]["bootstrap_replicates"])
    seed = int(config["evaluation"]["bootstrap_seed"])
    output = resolve(args.output_dir)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    payload: dict[str, Any] = {
        "analysis_id": "ANALYSIS-S18-IDENTITY-BOOTSTRAP-001",
        "input": str(source.relative_to(ROOT)),
        "input_sha256": sha256_file(source),
        "holdout_summary_sha256": sha256_file(holdout_summary_path),
        "replicates": replicates,
        "seed": seed,
        "cluster": "sample_id retaining all five SNR rows",
        "overall": {},
        "per_snr": {},
    }
    for pair_index, (name, (left, right)) in enumerate(PAIRS.items()):
        clusters: defaultdict[str, list[float]] = defaultdict(list)
        for row in rows:
            clusters[row["sample_id"]].append(float(row[left]) - float(row[right]))
        values = [sum(items) / len(items) for items in clusters.values()]
        payload["overall"][name] = interval(values, replicates, seed + pair_index)
    for snr_index, snr in enumerate(sorted({float(row["snr_db"]) for row in rows})):
        subset = [row for row in rows if float(row["snr_db"]) == snr]
        payload["per_snr"][str(snr)] = {}
        for pair_index, (name, (left, right)) in enumerate(PAIRS.items()):
            values = [float(row[left]) - float(row[right]) for row in subset]
            payload["per_snr"][str(snr)][name] = interval(
                values, replicates, seed + 100 + 10 * snr_index + pair_index
            )
    holdout = json.loads(holdout_summary_path.read_text(encoding="utf-8"))
    ci_low = payload["overall"]["selected_minus_full_psnr"]["ci95_low"]
    bootstrap_pass = ci_low > float(
        config["success_criteria"]["selected_minus_full_psnr_bootstrap_ci_low_min_db"]
    )
    prechecks = {str(key): bool(value) for key, value in holdout["checks_before_bootstrap"].items()}
    checks = {**prechecks, "selected_minus_full_psnr_ci_low_positive": bootstrap_pass}
    payload["checks"] = checks
    payload["verdict"] = "PASS" if all(checks.values()) else "NEGATIVE_OR_PARTIAL"
    (output / "bootstrap.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "STATE.json").write_text(
        json.dumps(
            {"state": "COMPLETE", "verdict": payload["verdict"], "checks": checks},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
