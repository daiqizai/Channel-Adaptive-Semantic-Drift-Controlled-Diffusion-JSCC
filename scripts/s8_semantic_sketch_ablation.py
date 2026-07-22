#!/usr/bin/env python3
"""Causal received/zero/shuffled semantic-sketch ablation for a frozen refiner."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from s5_residual_refiner_pilot import (  # noqa: E402
    build_model,
    load_rgb_tensor,
    refine_and_save_snr,
    snr_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/s8_semantic_sketch_causal_ablation.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def psnr(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    mse = float((reference - candidate).square().mean())
    return 99.0 if mse <= 0 else 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def bootstrap(values: dict[str, list[float]], replicates: int, seed: int) -> dict[str, float | int]:
    names = sorted(values)
    cluster_means = [sum(values[name]) / len(values[name]) for name in names]
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        draws.append(sum(cluster_means[rng.randrange(len(names))] for _ in names) / len(names))
    draws.sort()
    return {
        "estimate": sum(cluster_means) / len(cluster_means),
        "ci_low": draws[int(0.025 * replicates)],
        "ci_high": draws[min(replicates - 1, int(0.975 * replicates))],
        "replicates": replicates,
        "num_image_clusters": len(names),
    }


def main() -> None:
    args = parse_args()
    analysis_config_path = resolve(args.config)
    analysis = yaml.safe_load(analysis_config_path.read_text(encoding="utf-8"))
    source_config_path = resolve(analysis["source_config"])
    source = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    split_for_name: dict[str, str] = {}
    if "sample_ranges" in analysis:
        names = []
        for item in analysis["sample_ranges"]:
            split_name = str(item["name"])
            split_names = [
                f"sample_{index:06d}.png"
                for index in range(int(item["start"]), int(item["start"]) + int(item["count"]))
            ]
            names.extend(split_names)
            split_for_name.update({name: split_name for name in split_names})
    else:
        names = [
            f"sample_{index:06d}.png"
            for index in range(int(analysis["sample_start"]), int(analysis["sample_start"]) + int(analysis["sample_count"]))
        ]
        split_for_name = {name: "validation" for name in names}
    snrs = [float(item) for item in analysis["snrs"]]
    plan = {
        "analysis_id": analysis["analysis_id"],
        "checkpoint": analysis["checkpoint"],
        "num_images": len(names),
        "snrs": snrs,
        "modes": analysis["modes"],
        "official_val_accessed": False,
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, "plan": plan}, indent=2))
        return
    output_dir = resolve(analysis["outputs"]["output_dir"])
    if output_dir.exists():
        raise FileExistsError(f"Output exists: {output_dir}")
    output_dir.mkdir(parents=True)
    shutil.copy2(analysis_config_path, output_dir / "config.yaml")
    device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available() else args.device)
    model = build_model(source).to(device)
    checkpoint = torch.load(resolve(analysis["checkpoint"]), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    generated_roots: dict[str, Path] = {}
    modes_to_generate = ["received", "zeros", "shuffled"] if analysis.get("generate_received", False) else ["zeros", "shuffled"]
    if not analysis.get("generate_received", False):
        generated_roots["received"] = resolve(analysis["received_outputs"])
    for mode in modes_to_generate:
        mode_config = json.loads(json.dumps(source))
        mode_config.setdefault("evaluation", {})["semantic_sketch_mode"] = mode
        mode_root = output_dir / mode
        mode_root.mkdir()
        for snr in snrs:
            refine_and_save_snr(model, mode_config, snr, names, mode_root, device)
        generated_roots[mode] = mode_root
    original_dir = resolve(source["inputs"]["original_dir"])
    rows: list[dict[str, Any]] = []
    cluster_deltas: dict[str, dict[str, list[float]]] = {
        "received_minus_zeros": defaultdict(list),
        "received_minus_shuffled": defaultdict(list),
    }
    by_snr: dict[str, dict[str, float]] = {}
    for snr in snrs:
        sums = defaultdict(float)
        for name in names:
            original = load_rgb_tensor(original_dir / name)
            values = {}
            for mode, root in generated_roots.items():
                path = root / "exports" / snr_name(snr) / "refined" / name
                values[mode] = psnr(original, load_rgb_tensor(path))
                sums[mode] += values[mode]
            zero_delta = values["received"] - values["zeros"]
            shuffled_delta = values["received"] - values["shuffled"]
            cluster_deltas["received_minus_zeros"][name].append(zero_delta)
            cluster_deltas["received_minus_shuffled"][name].append(shuffled_delta)
            rows.append(
                {
                    "sample": name,
                    "split": split_for_name[name],
                    "snr_db": snr,
                    "received_psnr_db": values["received"],
                    "zeros_psnr_db": values["zeros"],
                    "shuffled_psnr_db": values["shuffled"],
                    "received_minus_zeros_psnr_db": zero_delta,
                    "received_minus_shuffled_psnr_db": shuffled_delta,
                }
            )
        by_snr[str(snr)] = {
            "received_minus_zeros_psnr_db": (sums["received"] - sums["zeros"]) / len(names),
            "received_minus_shuffled_psnr_db": (sums["received"] - sums["shuffled"]) / len(names),
        }
    replicates = int(analysis["bootstrap_replicates"])
    result = {
        "bootstrap": {
            key: bootstrap(value, replicates, int(analysis["seed"]) + index)
            for index, (key, value) in enumerate(cluster_deltas.items())
        },
        "by_snr": by_snr,
        "official_val_accessed": False,
    }
    if "sample_ranges" in analysis:
        by_split_snr: dict[str, dict[str, dict[str, float]]] = {}
        for split_name in sorted(set(split_for_name.values())):
            by_split_snr[split_name] = {}
            for snr in snrs:
                subset = [
                    row for row in rows if row["split"] == split_name and float(row["snr_db"]) == snr
                ]
                by_split_snr[split_name][str(snr)] = {
                    "received_minus_zeros_psnr_db": sum(float(row["received_minus_zeros_psnr_db"]) for row in subset) / len(subset),
                    "received_minus_shuffled_psnr_db": sum(float(row["received_minus_shuffled_psnr_db"]) for row in subset) / len(subset),
                }
        result["by_split_snr"] = by_split_snr
    zero_pass = result["bootstrap"]["received_minus_zeros"]["ci_low"] > 0
    shuffle_pass = result["bootstrap"]["received_minus_shuffled"]["ci_low"] > 0
    point_source = [
        item for split in result.get("by_split_snr", {"all": by_snr}).values() for item in split.values()
    ] if "by_split_snr" in result else list(by_snr.values())
    points_pass = all(
        row["received_minus_zeros_psnr_db"] >= 0 and row["received_minus_shuffled_psnr_db"] >= 0
        for row in point_source
    )
    result["success"] = {
        "received_beats_zeros": zero_pass,
        "received_beats_shuffled": shuffle_pass,
        "nonnegative_each_snr": points_pass,
        "all_pass": zero_pass and shuffle_pass and points_pass,
    }
    with (output_dir / "per_sample.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# Semantic Sketch Causal Ablation",
        "",
        f"Decision: **{'PASS' if result['success']['all_pass'] else 'FAIL'}**.",
        "",
        "| Comparison | Estimate | 95% CI |",
        "|---|---:|---:|",
    ]
    for key, value in result["bootstrap"].items():
        lines.append(f"| {key} | {value['estimate']:+.4f} dB | [{value['ci_low']:+.4f},{value['ci_high']:+.4f}] |")
    lines.extend(["", "Official Imagenette validation was not accessed."])
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((output_dir / "REPORT.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
