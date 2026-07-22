#!/usr/bin/env python3
"""Paired P0 comparison of c8+same-refiner B1 against decoded-structure B3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.transforms import functional as TF


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/s11_p0_b1_b3_paired_comparison.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def snr_name(snr: float) -> str:
    return f"snr_{int(snr):02d}db" if float(snr).is_integer() else f"snr_{str(snr).replace('.', 'p')}db"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def image(path: Path) -> torch.Tensor:
    with Image.open(path) as handle:
        return TF.to_tensor(handle.convert("RGB"))


def psnr(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    mse = float(torch.mean((reference - candidate).square()).item())
    return -10.0 * math.log10(max(mse, 1e-12))


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def semantic_map(path: Path) -> dict[tuple[str, float], dict[str, str]]:
    output: dict[tuple[str, float], dict[str, str]] = {}
    for row in read_csv(path):
        key = (str(row["sample"]), float(row["snr_db"]))
        if key in output:
            raise RuntimeError(f"Duplicate semantic row: {path}: {key}")
        output[key] = row
    return output


def bootstrap(values: np.ndarray, replicates: int, seed: int) -> dict[str, float | int]:
    if values.ndim != 1 or not len(values):
        raise ValueError("Bootstrap requires a non-empty 1-D vector")
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 256):
        count = min(256, replicates - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        estimates[start : start + count] = values[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "estimate": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "num_clusters": int(len(values)),
        "replicates": int(replicates),
    }


def validate_rate(config: dict[str, Any]) -> None:
    rate = config["rate"]
    if int(rate["b0_b1_inner_channel"]) != int(rate["total_inner_channel"]):
        raise RuntimeError("B0/B1 rate does not equal total rate")
    if int(rate["b3_main_inner_channel"]) + int(rate["b3_structure_inner_channel"]) != int(
        rate["total_inner_channel"]
    ):
        raise RuntimeError("B3 rate does not equal total rate")


def validate_refiner_contract(b1: dict[str, Any], b3: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "seed": (b1["seed"], b3["seed"]),
        "split": (b1["split"], b3["split"]),
        "base_channels": (b1["model"]["base_channels"], b3["model"]["base_channels"]),
        "num_blocks": (b1["model"]["num_blocks"], b3["model"]["num_blocks"]),
        "input_channels": (b1["model"]["input_channels"], b3["model"]["input_channels"]),
        "residual_gates": (b1["model"]["residual_gates"], b3["model"]["residual_gates"]),
    }
    for key in (
        "epochs",
        "batch_size",
        "num_workers",
        "crop_size",
        "random_flip",
        "lr",
        "weight_decay",
        "grad_clip_norm",
        "mse_weight",
        "l1_weight",
        "validation_every_epochs",
    ):
        checks[f"training.{key}"] = (b1["training"][key], b3["training"][key])
    mismatches = {key: values for key, values in checks.items() if values[0] != values[1]}
    if mismatches:
        raise RuntimeError(f"B1/B3 refiner contract mismatch: {mismatches}")
    return {key: values[0] for key, values in checks.items()}


def checkpoint_parameters(path: Path) -> tuple[int, int]:
    payload = torch.load(path, map_location="cpu")
    state = payload["model_state_dict"]
    return int(sum(value.numel() for value in state.values())), int(payload["epoch"])


def summarize_events(rows: list[dict[str, Any]], prefix: str) -> dict[str, int]:
    return {
        "new_error_rows": int(sum(bool(row[f"{prefix}_input_correct"]) and not bool(row[f"{prefix}_raw_correct"]) for row in rows)),
        "repair_rows": int(sum(not bool(row[f"{prefix}_input_correct"]) and bool(row[f"{prefix}_raw_correct"]) for row in rows)),
    }


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_rate(config)
    with resolve(config["inputs"]["b1_config"]).open("r", encoding="utf-8") as handle:
        b1_config = yaml.safe_load(handle)
    with resolve(config["inputs"]["b3_config"]).open("r", encoding="utf-8") as handle:
        b3_config = yaml.safe_load(handle)
    matched_contract = validate_refiner_contract(b1_config, b3_config)
    evaluation = config["evaluation"]
    names = [
        f"sample_{index:06d}.png"
        for index in range(int(evaluation["sample_start"]), int(evaluation["sample_start"]) + int(evaluation["sample_count"]))
    ]
    snrs = [float(value) for value in evaluation["snrs"]]
    plan = {
        "analysis_id": config["analysis_id"],
        "names": names,
        "snrs": snrs,
        "matched_refiner_contract": matched_contract,
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, "plan": plan}, indent=2, ensure_ascii=False))
        return

    output_dir = resolve(args.output_dir or config["outputs"]["output_dir"])
    if output_dir.exists():
        raise FileExistsError(f"Output exists: {output_dir}")
    output_dir.mkdir(parents=True)
    shutil.copy2(config_path, output_dir / "config.yaml")
    shutil.copy2(SCRIPT, output_dir / SCRIPT.name)
    save_json(output_dir / "run_plan.json", plan)

    original_b1 = resolve(config["inputs"]["original_b1_dir"])
    original_b3 = resolve(config["inputs"]["original_b3_dir"])
    b0_export = resolve(config["inputs"]["b0_export_dir"])
    b1_output = resolve(config["inputs"]["b1_output_dir"])
    b3_output = resolve(config["inputs"]["b3_output_dir"])
    b1_semantic = semantic_map(b1_output / "per_sample.csv")
    b3_semantic = semantic_map(b3_output / "per_sample.csv")
    expected = {(name, snr) for name in names for snr in snrs}
    if set(b1_semantic) != expected or set(b3_semantic) != expected:
        raise RuntimeError("B1/B3 semantic grids do not match the frozen evaluation grid")

    manifest_rows: list[dict[str, Any]] = []
    for name in names:
        left = original_b1 / name
        right = original_b3 / name
        left_hash, right_hash = sha256(left), sha256(right)
        if left_hash != right_hash:
            raise RuntimeError(f"Original PNG mismatch: {name}")
        manifest_rows.append({"sample": name, "b1_original": relative(left), "b3_original": relative(right), "sha256": left_hash})

    rows: list[dict[str, Any]] = []
    for snr in snrs:
        folder = snr_name(snr)
        for name in names:
            key = (name, snr)
            b1_sem, b3_sem = b1_semantic[key], b3_semantic[key]
            if int(b1_sem["original_top1_index"]) != int(b3_sem["original_top1_index"]):
                raise RuntimeError(f"Original classifier prediction mismatch: {key}")
            paths = {
                "original": original_b1 / name,
                "b0": b0_export / "exports" / folder / str(config["inputs"]["b0_reconstruction_subdir"]) / name,
                "b1_raw": b1_output / "exports" / folder / "refined" / name,
                "b1_final": b1_output / "exports" / folder / "final" / name,
                "b3_raw": b3_output / "exports" / folder / "refined" / name,
                "b3_final": b3_output / "exports" / folder / "final" / name,
            }
            for path in paths.values():
                if not path.is_file():
                    raise FileNotFoundError(path)
            tensors = {role: image(path) for role, path in paths.items()}
            quality = {role: psnr(tensors["original"], tensor) for role, tensor in tensors.items() if role != "original"}
            rows.append(
                {
                    "sample": name,
                    "image_id": name.removesuffix(".png"),
                    "snr_db": snr,
                    **{f"{role}_psnr_db": value for role, value in quality.items()},
                    "b1_raw_minus_b0_psnr_db": quality["b1_raw"] - quality["b0"],
                    "b3_raw_minus_b0_psnr_db": quality["b3_raw"] - quality["b0"],
                    "b3_minus_b1_raw_psnr_db": quality["b3_raw"] - quality["b1_raw"],
                    "b3_minus_b1_final_psnr_db": quality["b3_final"] - quality["b1_final"],
                    "b1_input_correct": as_bool(b1_sem["m0_matches_original_top1"]),
                    "b1_raw_correct": as_bool(b1_sem["refined_matches_original_top1"]),
                    "b3_input_correct": as_bool(b3_sem["m0_matches_original_top1"]),
                    "b3_raw_correct": as_bool(b3_sem["refined_matches_original_top1"]),
                }
            )

    replicates = int(evaluation["bootstrap_replicates"])
    seed = int(config["seed"])
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["image_id"])].append(float(row["b3_minus_b1_raw_psnr_db"]))
    cluster_values = np.array([np.mean(grouped[name]) for name in sorted(grouped)], dtype=np.float64)
    primary = bootstrap(cluster_values, replicates, seed)
    per_snr: list[dict[str, Any]] = []
    for index, snr in enumerate(snrs):
        values = np.array([float(row["b3_minus_b1_raw_psnr_db"]) for row in rows if float(row["snr_db"]) == snr])
        result = bootstrap(values, replicates, seed + 1 + index)
        per_snr.append({"snr_db": snr, **result})
    events = {"b1": summarize_events(rows, "b1"), "b3": summarize_events(rows, "b3")}

    b1_summary = read_csv(b1_output / "summary.csv")
    b3_summary = read_csv(b3_output / "summary.csv")
    mean_b1_latency = float(np.mean([float(row["refiner_time_ms_per_image"]) for row in b1_summary]))
    mean_b3_latency = float(np.mean([float(row["refiner_time_ms_per_image"]) for row in b3_summary]))
    mean_b1_lpips = float(np.mean([float(row["refined_lpips"]) for row in b1_summary]))
    mean_b3_lpips = float(np.mean([float(row["refined_lpips"]) for row in b3_summary]))
    b1_params, b1_epoch = checkpoint_parameters(b1_output / "checkpoints" / "best.pt")
    b3_params, b3_epoch = checkpoint_parameters(b3_output / "checkpoints" / "best.pt")
    if b1_params != b3_params:
        raise RuntimeError(f"Refiner parameter mismatch: {b1_params} vs {b3_params}")

    checks = {
        "primary_ci_lower_gt_zero": float(primary["ci_low"]) > 0.0,
        "positive_snr_count": sum(float(row["estimate"]) > 0.0 for row in per_snr)
        >= int(evaluation["required_positive_snr_count"]),
        "b3_new_error_not_greater_than_b1": events["b3"]["new_error_rows"]
        <= events["b1"]["new_error_rows"],
    }
    decision = {
        "b3_passes_first_structure_increment_gate": all(checks.values()),
        "checks": checks,
        "primary": primary,
        "positive_snr_count": int(sum(float(row["estimate"]) > 0.0 for row in per_snr)),
        "events": events,
    }
    summary_rows = [
        {
            "comparison": "B3_raw_minus_B1_raw",
            **primary,
            "mean_b1_raw_minus_b0_psnr_db": float(np.mean([row["b1_raw_minus_b0_psnr_db"] for row in rows])),
            "mean_b3_raw_minus_b0_psnr_db": float(np.mean([row["b3_raw_minus_b0_psnr_db"] for row in rows])),
            "mean_b3_minus_b1_final_psnr_db": float(np.mean([row["b3_minus_b1_final_psnr_db"] for row in rows])),
            "b1_mean_raw_lpips": mean_b1_lpips,
            "b3_mean_raw_lpips": mean_b3_lpips,
            "b3_minus_b1_raw_lpips": mean_b3_lpips - mean_b1_lpips,
            "b1_refiner_params": b1_params,
            "b3_refiner_params": b3_params,
            "b1_best_epoch": b1_epoch,
            "b3_best_epoch": b3_epoch,
            "b1_latency_ms_per_image": mean_b1_latency,
            "b3_latency_ms_per_image": mean_b3_latency,
            "decision": "PASS" if decision["b3_passes_first_structure_increment_gate"] else "FAIL",
        }
    ]
    write_csv(output_dir / "per_sample.csv", rows)
    write_csv(output_dir / "per_snr.csv", per_snr)
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "original_manifest.csv", manifest_rows)
    save_json(output_dir / "decision.json", decision)
    report = [
        "# P0 B1 vs B3 Paired Fairness Comparison",
        "",
        f"Decision: **{'PASS' if decision['b3_passes_first_structure_increment_gate'] else 'FAIL'}**.",
        "",
        "Primary endpoint is raw PSNR of B3 (`c6+c2 decoded structure + refiner`) minus B1 (`c8 + same-capacity receiver-only refiner`).",
        "",
        f"- Mean paired delta: `{primary['estimate']:+.4f} dB`.",
        f"- Image-cluster bootstrap 95% CI: `[{primary['ci_low']:+.4f}, {primary['ci_high']:+.4f}] dB`.",
        f"- Positive SNR points: `{decision['positive_snr_count']}/{len(snrs)}`.",
        f"- B1 raw new errors / repairs: `{events['b1']['new_error_rows']}/{events['b1']['repair_rows']}`.",
        f"- B3 raw new errors / repairs: `{events['b3']['new_error_rows']}/{events['b3']['repair_rows']}`.",
        f"- Refiner parameters: `{b1_params}` for both arms.",
        f"- Mean refiner latency B1/B3: `{mean_b1_latency:.4f}/{mean_b3_latency:.4f} ms/image`.",
        "",
        "| SNR | B3 raw - B1 raw PSNR | 95% CI |",
        "|---:|---:|---:|",
    ]
    for row in per_snr:
        report.append(f"| {row['snr_db']:g} | {row['estimate']:+.4f} | [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] |")
    report.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This comparison controls refiner capacity, split, seed, loss, epochs, gates, and total channel-use ratio. It does not make the c8 and c6/c2 encoder training histories identical, and AlexNet events remain pseudo-semantic diagnostics. A failure blocks attribution of the current gain to decoded structure; it does not claim that all structure representations or all diffusion backends are ineffective.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": relative(output_dir), "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
