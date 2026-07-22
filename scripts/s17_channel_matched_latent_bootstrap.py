#!/usr/bin/env python3
"""Image-cluster paired bootstrap for the frozen S17 latent-diffusion holdout."""

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
EXPECTED_INPUT_SHA256 = "a13bfe7ffa827d421c8f64c28226546ca79d98d2cfbf7f783672cba2236e1363"
PAIRS = {
    "matched_ddim_minus_b0_psnr": ("matched_ddim_psnr", "b0_psnr"),
    "matched_ddim_minus_b0_lpips": ("matched_ddim_lpips", "b0_lpips"),
    "matched_ddim_minus_fixed_psnr": ("matched_ddim_psnr", "fixed_step_ddim_psnr"),
    "matched_ddim_b1_minus_b1_psnr": ("matched_ddim_b1_psnr", "b1_psnr"),
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
    estimates = []
    for _ in range(replicates):
        estimates.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    estimates.sort()
    low = estimates[int(0.025 * replicates)]
    high = estimates[min(replicates - 1, int(0.975 * replicates))]
    return {"mean": sum(values) / len(values), "ci95_low": low, "ci95_high": high}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="outputs/analysis/ANALYSIS-S17-LATDIFF-HOLDOUT-002/per_sample.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/analysis/ANALYSIS-S17-LATDIFF-BOOTSTRAP-001",
    )
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260734)
    args = parser.parse_args()
    source = resolve(args.input)
    observed = sha256_file(source)
    if observed != EXPECTED_INPUT_SHA256:
        raise RuntimeError(f"frozen input SHA mismatch: {observed}")
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1280 or len({row["sample_id"] for row in rows}) != 256:
        raise RuntimeError("frozen S17 holdout dimensions changed")
    output = resolve(args.output_dir)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    payload: dict[str, Any] = {
        "analysis_id": "ANALYSIS-S17-LATDIFF-BOOTSTRAP-001",
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
                values, args.replicates, args.seed + 100 + 10 * snr_index + pair_index
            )
    (output / "bootstrap.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
