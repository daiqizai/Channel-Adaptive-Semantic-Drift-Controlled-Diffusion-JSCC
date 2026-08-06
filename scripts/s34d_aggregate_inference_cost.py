#!/usr/bin/env python3
"""Aggregate completed S34D method-specific measurements without model inference."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(path)
    return payload


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(summary: dict[str, Any], field: str) -> float:
    return float(summary["latency"][field]["mean"])


def main() -> None:
    config_path = resolve("configs/s34d_generative_inference_cost.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = resolve(config["outputs"]["root"])
    final_state = root / "STATE.json"
    if final_state.exists():
        raise FileExistsError(final_state)
    s33 = load_json(resolve(config["outputs"]["s33"]) / "summary.json")
    diff = load_json(resolve(config["outputs"]["diffjscc"]) / "summary.json")
    sgd = load_json(resolve(config["outputs"]["sgd"]) / "summary.json")
    if any(item["status"] != "PASS" for item in (s33, diff, sgd)):
        raise RuntimeError("one method-specific measurement is incomplete")

    s33_wall = float(s33["latency"]["receiver_wall_ms"]["mean"])
    s33_flops = int(s33["profiled_flops_lower_bound"]["total"])
    latency_rows: list[dict[str, Any]] = [
        {
            "method": "S33 strong",
            "sampling_steps": 0,
            "receiver_wall_mean_ms": s33_wall,
            "receiver_wall_median_ms": s33["latency"]["receiver_wall_ms"]["median"],
            "slowdown_vs_S33_mean": 1.0,
            "unique_live_parameters": int(
                s33["parameters"]["unique_live_parameters"]
            ),
            "parameter_ratio_vs_S33": 1.0,
            "profiled_FLOPs_lower_bound": s33_flops,
            "profiled_FLOPs_ratio_vs_S33": 1.0,
            "quality_role": "exact-rate_fidelity_reference",
        }
    ]
    diff_params = int(diff["parameters"]["unique_live_parameters"])
    diff_flops = diff["profiled_flops_lower_bound"]["totals_by_steps"]
    for row in diff["quality_curve"]:
        steps = int(row["steps"])
        wall = float(row["receiver_wall_ms"]["mean"])
        latency_rows.append(
            {
                "method": "DiffJSCC",
                "sampling_steps": steps,
                "receiver_wall_mean_ms": wall,
                "receiver_wall_median_ms": row["receiver_wall_ms"]["median"],
                "slowdown_vs_S33_mean": wall / s33_wall,
                "unique_live_parameters": diff_params,
                "parameter_ratio_vs_S33": diff_params
                / int(s33["parameters"]["unique_live_parameters"]),
                "profiled_FLOPs_lower_bound": int(diff_flops[str(steps)]),
                "profiled_FLOPs_ratio_vs_S33": int(diff_flops[str(steps)])
                / s33_flops,
                "quality_role": (
                    "retains_significant_LPIPS_advantage"
                    if row["retains_significant_lpips_advantage_vs_s33"]
                    else "does_not_retain_significant_LPIPS_advantage"
                ),
            }
        )
    sgd_wall = float(sgd["latency"]["receiver_wall_ms"]["mean"])
    sgd_params = int(sgd["parameters"]["unique_live_parameters"])
    sgd_flops = int(sgd["profiled_flops_lower_bound"]["total"])
    latency_rows.append(
        {
            "method": "SGD paper upper",
            "sampling_steps": 50,
            "receiver_wall_mean_ms": sgd_wall,
            "receiver_wall_median_ms": sgd["latency"]["receiver_wall_ms"]["median"],
            "slowdown_vs_S33_mean": sgd_wall / s33_wall,
            "unique_live_parameters": sgd_params,
            "parameter_ratio_vs_S33": sgd_params
            / int(s33["parameters"]["unique_live_parameters"]),
            "profiled_FLOPs_lower_bound": sgd_flops,
            "profiled_FLOPs_ratio_vs_S33": sgd_flops / s33_flops,
            "quality_role": "cross-contract_non-ranking_paper_upper",
        }
    )
    write_csv(root / "latency_parameter_flops_comparison.csv", latency_rows)
    write_csv(root / "diffjscc_latency_quality_curve.csv", diff["quality_curve"])

    passing = [
        row
        for row in diff["quality_curve"]
        if row["retains_significant_lpips_advantage_vs_s33"]
    ]
    minimum = min(passing, key=lambda row: int(row["steps"])) if passing else None
    aggregate = {
        "status": "PASS",
        "analysis_id": config["analysis_id"],
        "primary_receiver_entry": config["latency_contract"]["start"],
        "primary_receiver_exit": config["latency_contract"]["end"],
        "model_loading_included": False,
        "batch_size_source_images": 1,
        "hardware": s33["hardware"],
        "s33_receiver_wall_mean_ms": s33_wall,
        "diffjscc_minimum_passing_point": minimum,
        "diffjscc_minimum_passing_slowdown_vs_S33": (
            float(minimum["receiver_wall_ms"]["mean"]) / s33_wall
            if minimum
            else None
        ),
        "diffjscc_100step_slowdown_vs_S33": float(
            next(
                row["receiver_wall_ms"]["mean"]
                for row in diff["quality_curve"]
                if int(row["steps"]) == 100
            )
        )
        / s33_wall,
        "sgd_receiver_wall_mean_ms": sgd_wall,
        "sgd_slowdown_vs_S33": sgd_wall / s33_wall,
        "parameter_counts": {
            "S33": int(s33["parameters"]["unique_live_parameters"]),
            "DiffJSCC": diff_params,
            "SGD": sgd_params,
        },
        "profiled_flops_are_lower_bounds": True,
        "software_stack_boundary": {
            "S33_torch": s33["torch_version"],
            "DiffJSCC_torch": diff["torch_version"],
            "SGD_torch": sgd["torch_version"],
            "interpretation": "same physical GPU and entry contract; frozen method runtimes require different PyTorch stacks, so FLOPs lower bounds are reported alongside ms",
        },
        "fixed_and_optimizable_cost_interpretation": {
            "diffjscc_current_pipeline_fixed_per_image": [
                "JSCC frontend",
                "receiver BLIP2 caption",
                "OpenCLIP text conditioning",
                "spatial condition encoder",
                "one VAE decode",
                "wavelet color correction",
            ],
            "diffjscc_linear_with_steps": "ControlNet+UNet denoiser evaluations",
            "not_diffusion_inherent_but_current_method_specific": [
                "BLIP2 caption",
                "OpenCLIP text conditioner",
                "512x512 internal working resolution",
                "wavelet color correction",
            ],
            "latent_diffusion_minimum_structural_cost": [
                "at least one/few denoiser evaluations",
                "latent-to-pixel VAE decode",
            ],
            "sgd_removable_without_using_its_return_value": "pipeline internal VAE decode whose pixel output is discarded before a second final VAE decode",
        },
        "claim_boundary": [
            "quality population is known policy-development, not official validation",
            "SGD remains a non-rate-matched paper upper and is not quality-ranked",
            "profiled FLOPs count supported aten conv/linear/mm/bmm ops only",
            "4 steps is the smallest preregistered candidate, not a proof that fewer steps cannot work",
        ],
        "new_training": False,
        "network_download": False,
        "official_imagenette_validation_accessed": False,
    }
    (root / "aggregate_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(config_path, root / "final_config_snapshot.yaml")
    shutil.copy2(SCRIPT, root / SCRIPT.name)
    final_state.write_text(
        json.dumps({"status": "complete"}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
