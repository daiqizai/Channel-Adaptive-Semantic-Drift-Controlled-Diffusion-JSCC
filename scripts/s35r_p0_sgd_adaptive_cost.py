#!/usr/bin/env python3
"""Derive the preregistered SGD-JSCC latency-SNR and step-count audit."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
CONFIG = ROOT / "configs/s35r_p0_sgd_adaptive_cost.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "std": float(array.std(ddof=1)),
    }


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def sigmoid_schedule_inverse(
    output: float,
    start: float = 0.0,
    end: float = 3.0,
    tau: float = 0.7,
) -> float:
    """Exact scalar translation of the released DiffusionGenerator helper."""
    v_start = sigmoid(start / tau)
    v_end = sigmoid(end / tau)
    adjusted = output * (v_end - v_start) + v_start
    value = (-start - tau * math.log(1.0 / adjusted - 1.0)) / (end - start)
    return float(np.clip(value, start, end))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config["status"] != "preregistered_and_authorized_for_measurement":
        raise RuntimeError("P0 is not authorized for measurement")

    source_csv = resolve(config["input_measurement"]["source"])
    if sha256(source_csv) != config["input_measurement"]["source_sha256"]:
        raise RuntimeError("S34D latency rows changed")

    inference_path = resolve(
        config["step_matching_audit"]["author_source"]["inference"]
    )
    sampler_path = resolve(config["step_matching_audit"]["author_source"]["sampler"])
    if sha256(inference_path) != config["step_matching_audit"]["author_source"][
        "inference_sha256"
    ]:
        raise RuntimeError("SGD inference source changed")
    if sha256(sampler_path) != config["step_matching_audit"]["author_source"][
        "sampler_sha256"
    ]:
        raise RuntimeError("SGD sampler source changed")

    inference_source = inference_path.read_text(encoding="utf-8")
    sampler_source = sampler_path.read_text(encoding="utf-8")
    required_inference_fragments = [
        "if step_style == 'continuous':",
        "cur_step = 1 - predicted_signal_scale",
        "diffusion_step=diffusion_step,step_style=step_style",
    ]
    required_sampler_fragments = [
        "timesteps = np.linspace(0.001,curr_timestep,diffusion_step).tolist()",
        "for i in tqdm(range(len(noise_levels) - 1)):",
        "x0_pred = self.pred_image(x_t, labels, next_noise",
    ]
    for fragment in required_inference_fragments:
        if fragment not in inference_source:
            raise RuntimeError(f"missing inference fragment: {fragment}")
    for fragment in required_sampler_fragments:
        if fragment not in sampler_source:
            raise RuntimeError(f"missing sampler fragment: {fragment}")

    with source_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != int(config["input_measurement"]["rows"]):
        raise RuntimeError("unexpected S34D row count")

    expected_snrs = [float(v) for v in config["input_measurement"]["snrs_db"]]
    per_snr_count = int(config["input_measurement"]["source_images_per_snr"])
    fields = [
        "receiver_wall_ms",
        "blip2_caption_ms",
        "edge_extractor_ms",
        "diffusion_solver_ms",
        "core_model_ms",
        "main_vae_encode_ms",
        "edge_jscc_ms",
        "edge_vae_encode_ms",
        "clip_text_conditioning_ms",
        "redundant_pipeline_vae_decode_ms",
        "final_vae_decode_ms",
        "preprocess_h2d_patch_split_ms",
        "postprocess_merge_d2h_ms",
    ]

    curve_rows: list[dict[str, Any]] = []
    detailed: dict[str, Any] = {}
    for snr in expected_snrs:
        subset = [row for row in rows if float(row["snr_db"]) == snr]
        if len(subset) != per_snr_count:
            raise RuntimeError(f"unexpected row count at {snr} dB")
        observed_counts = {
            int(float(row["executed_denoiser_evaluations"])) for row in subset
        }
        if observed_counts != {50}:
            raise RuntimeError(f"unexpected recorded denoiser count: {observed_counts}")

        gamma = 10.0 ** (snr / 10.0)
        alpha_bar = 2.0 * gamma / (2.0 * gamma + 1.0)
        one_minus_alpha = 1.0 - alpha_bar
        endpoint = sigmoid_schedule_inverse(one_minus_alpha)
        component = {
            field: summary([float(row[field]) for row in subset]) for field in fields
        }
        floor_values = [
            float(row["blip2_caption_ms"]) + float(row["edge_extractor_ms"])
            for row in subset
        ]
        component["blip2_plus_muge_floor_ms"] = summary(floor_values)
        detailed[str(int(snr))] = component
        curve_rows.append(
            {
                "snr_db": int(snr),
                "gamma_linear": gamma,
                "alpha_bar_channel": alpha_bar,
                "one_minus_alpha_bar_channel": one_minus_alpha,
                "continuous_schedule_endpoint": endpoint,
                "scheduled_continuous_points": 50,
                "loop_denoiser_evaluations": 49,
                "final_denoiser_evaluations": 1,
                "actual_denoiser_evaluations": 50,
                "receiver_wall_mean_ms": component["receiver_wall_ms"]["mean"],
                "receiver_wall_median_ms": component["receiver_wall_ms"]["median"],
                "receiver_wall_p05_ms": component["receiver_wall_ms"]["p05"],
                "receiver_wall_p95_ms": component["receiver_wall_ms"]["p95"],
                "blip2_mean_ms": component["blip2_caption_ms"]["mean"],
                "muge_mean_ms": component["edge_extractor_ms"]["mean"],
                "blip2_plus_muge_floor_mean_ms": component[
                    "blip2_plus_muge_floor_ms"
                ]["mean"],
                "diffusion_solver_mean_ms": component["diffusion_solver_ms"]["mean"],
            }
        )

    floor_means = [
        float(row["blip2_plus_muge_floor_mean_ms"]) for row in curve_rows
    ]
    blip_means = [float(row["blip2_mean_ms"]) for row in curve_rows]
    muge_means = [float(row["muge_mean_ms"]) for row in curve_rows]
    wall_means = [float(row["receiver_wall_mean_ms"]) for row in curve_rows]
    solver_means = [float(row["diffusion_solver_mean_ms"]) for row in curve_rows]

    output = resolve(config["outputs"]["directory"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    shutil.copy2(CONFIG, output / "config_snapshot.yaml")
    shutil.copy2(SCRIPT, output / SCRIPT.name)
    write_csv(output / "latency_snr_curve.csv", curve_rows)

    result = {
        "status": "PASS",
        "analysis_id": config["analysis_id"],
        "method": config["scope"]["method_label"],
        "measurement_reused_without_new_inference": True,
        "new_training": False,
        "network_download": False,
        "official_imagenette_validation_accessed": False,
        "hardware": {
            "name": config["input_measurement"]["hardware"],
            "gpu_uuid": config["input_measurement"]["gpu_uuid"],
            "torch_runtime": config["input_measurement"]["torch_runtime"],
            "batch_size_source_images": 1,
        },
        "latency_contract": config["latency_contract"],
        "step_audit": {
            "formula": "alpha_bar_channel=2*gamma/(2*gamma+1)",
            "step_style": "continuous",
            "diffusion_step_argument": 50,
            "scheduled_points": 50,
            "loop_predictions": 49,
            "final_prediction": 1,
            "actual_denoiser_evaluations_at_every_snr": 50,
            "interpretation": (
                "step matching changes the continuous trajectory endpoint; "
                "the released working point does not reduce solver evaluations with SNR"
            ),
            "source_fragments_verified": True,
        },
        "latency_snr_curve": curve_rows,
        "per_snr_component_statistics": detailed,
        "fixed_floor_audit": {
            "blip2_structurally_once_before_step_matching": True,
            "muge_structurally_once_before_step_matching": True,
            "both_independent_of_diffusion_step_count_in_released_pipeline": True,
            "combined_mean_across_five_snr_means_ms": float(np.mean(floor_means)),
            "combined_max_minus_min_mean_across_snr_ms": float(
                max(floor_means) - min(floor_means)
            ),
            "blip2_mean_range_ms": [float(min(blip_means)), float(max(blip_means))],
            "muge_mean_range_ms": [float(min(muge_means)), float(max(muge_means))],
            "interpretation": (
                "fixed means structurally unavoidable for this released pipeline, "
                "not numerically constant and not inherent to all generative JSCC"
            ),
        },
        "curve_range": {
            "receiver_wall_mean_max_minus_min_ms": float(
                max(wall_means) - min(wall_means)
            ),
            "diffusion_solver_mean_max_minus_min_ms": float(
                max(solver_means) - min(solver_means)
            ),
        },
        "fairness_boundary": config["fairness_boundary"],
        "inputs": {
            "config_sha256": sha256(CONFIG),
            "script_sha256": sha256(SCRIPT),
            "source_latency_rows_sha256": sha256(source_csv),
            "inference_source_sha256": sha256(inference_path),
            "sampler_source_sha256": sha256(sampler_path),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "STATE.json").write_text(
        json.dumps({"status": "complete", "rows": len(rows)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
