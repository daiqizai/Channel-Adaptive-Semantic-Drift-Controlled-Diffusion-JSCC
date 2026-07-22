#!/usr/bin/env python3
"""Paired image-cluster bootstrap for the frozen decoder-aware S17 holdout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_INPUT_SHA256 = "9a42ce71c05036f6401a0509b4aa6cde200b660a5acad603b0ce0293926baf92"
PAIRS = {
    "decoder_minus_control_psnr": ("matched_ddim_psnr", "control_matched_ddim_psnr"),
    "decoder_minus_control_lpips": ("matched_ddim_lpips", "control_matched_ddim_lpips"),
    "decoder_minus_control_latent_mse": (
        "matched_ddim_latent_mse",
        "control_matched_ddim_latent_mse",
    ),
    "decoder_minus_parent_psnr": ("matched_ddim_psnr", "parent_matched_ddim_psnr"),
    "decoder_minus_parent_lpips": ("matched_ddim_lpips", "parent_matched_ddim_lpips"),
    "decoder_minus_parent_latent_mse": (
        "matched_ddim_latent_mse",
        "parent_matched_ddim_latent_mse",
    ),
    "decoder_minus_b0_psnr": ("matched_ddim_psnr", "b0_psnr"),
    "decoder_minus_b0_lpips": ("matched_ddim_lpips", "b0_lpips"),
    "decoder_b1_minus_b1_psnr": ("matched_ddim_b1_psnr", "b1_psnr"),
    "decoder_b1_minus_b1_lpips": ("matched_ddim_b1_lpips", "b1_lpips"),
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
    parser.add_argument(
        "--input",
        default="outputs/analysis/ANALYSIS-S17-LATDIFF-HOLDOUT-003/per_sample.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/analysis/ANALYSIS-S17-LATDIFF-BOOTSTRAP-002",
    )
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260737)
    args = parser.parse_args()
    source = resolve(args.input)
    observed = sha256_file(source)
    if observed != EXPECTED_INPUT_SHA256:
        raise RuntimeError(f"frozen input SHA mismatch: {observed}")
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    sample_ids = {row["sample_id"] for row in rows}
    if len(rows) != 1160 or len(sample_ids) != 232:
        raise RuntimeError("frozen decoder-aware holdout dimensions changed")
    counts: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["sample_id"]] += 1
    if set(counts.values()) != {5}:
        raise RuntimeError("each image cluster must retain exactly five SNR rows")
    output = resolve(args.output_dir)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    payload: dict[str, Any] = {
        "analysis_id": "ANALYSIS-S17-LATDIFF-BOOTSTRAP-002",
        "input": str(source.relative_to(ROOT)),
        "input_sha256": observed,
        "replicates": args.replicates,
        "seed": args.seed,
        "cluster": "sample_id retaining all five SNR rows",
        "overall": {},
        "per_snr": {},
    }
    for pair_index, (name, (left, right)) in enumerate(PAIRS.items()):
        clusters: defaultdict[str, list[float]] = defaultdict(list)
        for row in rows:
            clusters[row["sample_id"]].append(float(row[left]) - float(row[right]))
        values = [sum(items) / len(items) for items in clusters.values()]
        payload["overall"][name] = interval(
            values, args.replicates, args.seed + pair_index
        )
    for snr_index, snr in enumerate(sorted({float(row["snr_db"]) for row in rows})):
        subset = [row for row in rows if float(row["snr_db"]) == snr]
        payload["per_snr"][str(snr)] = {}
        for pair_index, (name, (left, right)) in enumerate(PAIRS.items()):
            values = [float(row[left]) - float(row[right]) for row in subset]
            payload["per_snr"][str(snr)][name] = interval(
                values,
                args.replicates,
                args.seed + 100 + 10 * snr_index + pair_index,
            )
    (output / "bootstrap.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
