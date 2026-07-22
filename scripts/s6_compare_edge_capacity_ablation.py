from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a reproducible paired 2x2 edge-by-capacity validation ablation."
    )
    parser.add_argument(
        "--config",
        default="configs/s6_edge_capacity_ablation_exp_s4_006_008_009_010.yaml",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def project_relative(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def serialize_csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_csv_value(row.get(key)) for key in fieldnames})


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return {
        "path": project_relative(path),
        "bytes": size,
        "sha256": digest.hexdigest(),
    }


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return "N/A (not a project git repo)"


def git_dirty_state() -> str:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        )
        return "dirty" if output.strip() else "clean"
    except Exception:  # noqa: BLE001
        return "unknown"


def snr_name(snr: float) -> str:
    return str(int(snr)) if float(snr).is_integer() else str(snr)


def row_key(row: dict[str, Any]) -> tuple[float, str]:
    return float(row["snr_db"]), str(row["sample"])


def remove_nested(mapping: dict[str, Any], path: tuple[str, ...]) -> None:
    current: Any = mapping
    for key in path[:-1]:
        if not isinstance(current, dict) or key not in current:
            return
        current = current[key]
    if isinstance(current, dict):
        current.pop(path[-1], None)


ARM_METADATA_PATHS = [
    ("experiment_id",),
    ("method",),
    ("notes",),
    ("outputs",),
]

EDGE_FACTOR_PATHS = [
    ("model", "name"),
    ("model", "input_channels"),
    ("model", "condition_features"),
]

CAPACITY_FACTOR_PATHS = [
    ("model", "base_channels"),
    ("model", "num_blocks"),
    ("training", "epochs"),
]


def normalized_config(
    config: dict[str, Any], *, drop_edge: bool, drop_capacity: bool
) -> dict[str, Any]:
    output = copy.deepcopy(config)
    for path in ARM_METADATA_PATHS:
        remove_nested(output, path)
    if drop_edge:
        for path in EDGE_FACTOR_PATHS:
            remove_nested(output, path)
    if drop_capacity:
        for path in CAPACITY_FACTOR_PATHS:
            remove_nested(output, path)
    return output


def value_differences(left: Any, right: Any, path: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else str(key)
            if key not in left:
                differences.append(f"{child}: <missing> != {right[key]!r}")
            elif key not in right:
                differences.append(f"{child}: {left[key]!r} != <missing>")
            else:
                differences.extend(value_differences(left[key], right[key], child))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if left == right:
            return []
        return [f"{path}: {left!r} != {right!r}"]
    if left != right:
        return [f"{path}: {left!r} != {right!r}"]
    return []


@dataclass
class ArmData:
    key: str
    experiment_id: str
    capacity: str
    edge_conditioned: bool
    config_path: Path
    checkpoint_path: Path
    per_sample_path: Path
    summary_path: Path
    source_config: dict[str, Any]
    checkpoint: dict[str, Any]
    rows: list[dict[str, str]]
    rows_by_key: dict[tuple[float, str], dict[str, str]]
    summary_rows: list[dict[str, str]]
    parameter_count: int
    best_epoch: int
    input_files: dict[str, dict[str, Any]]


def validate_arm(
    arm_cfg: dict[str, Any], analysis_config: dict[str, Any], expected_snrs: list[float]
) -> ArmData:
    required_paths = {
        "config": resolve_project_path(arm_cfg["config"]),
        "checkpoint": resolve_project_path(arm_cfg["checkpoint"]),
        "per_sample_csv": resolve_project_path(arm_cfg["per_sample_csv"]),
        "summary_csv": resolve_project_path(arm_cfg["summary_csv"]),
    }
    missing = [f"{name}: {path}" for name, path in required_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing arm inputs:\n" + "\n".join(missing))

    with required_paths["config"].open("r", encoding="utf-8") as handle:
        source_config = yaml.safe_load(handle)
    if not isinstance(source_config, dict):
        raise TypeError(f"Source config is not a mapping: {required_paths['config']}")

    checkpoint = torch.load(required_paths["checkpoint"], map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(f"Checkpoint lacks model_state_dict: {required_paths['checkpoint']}")
    if bool(analysis_config["validation"].get("require_checkpoint_config_match", True)):
        checkpoint_config = checkpoint.get("config")
        if checkpoint_config != source_config:
            differences = value_differences(source_config, checkpoint_config)
            raise RuntimeError(
                f"Checkpoint config mismatch for {arm_cfg['key']}:\n" + "\n".join(differences[:30])
            )

    if str(source_config.get("experiment_id")) != str(arm_cfg["experiment_id"]):
        raise RuntimeError(
            f"Experiment ID mismatch for {arm_cfg['key']}: "
            f"analysis={arm_cfg['experiment_id']} source={source_config.get('experiment_id')}"
        )
    if [float(item) for item in source_config["snrs"]] != expected_snrs:
        raise RuntimeError(f"Unexpected SNR list for {arm_cfg['key']}: {source_config['snrs']}")

    capacity = str(arm_cfg["capacity"])
    expected_capacity = analysis_config["design"]["capacity_levels"][capacity]
    actual_capacity = {
        "base_channels": int(source_config["model"]["base_channels"]),
        "num_blocks": int(source_config["model"]["num_blocks"]),
        "epochs": int(source_config["training"]["epochs"]),
    }
    if actual_capacity != {key: int(value) for key, value in expected_capacity.items()}:
        raise RuntimeError(
            f"Capacity signature mismatch for {arm_cfg['key']}: "
            f"actual={actual_capacity} expected={expected_capacity}"
        )

    expected_features = [str(item) for item in analysis_config["design"]["edge_features"]]
    features = source_config["model"].get("condition_features", [])
    if isinstance(features, str):
        features = [item.strip() for item in features.split(",") if item.strip()]
    features = [str(item) for item in features]
    edge_conditioned = bool(arm_cfg["edge_conditioned"])
    expected_arm_features = expected_features if edge_conditioned else []
    if features != expected_arm_features:
        raise RuntimeError(
            f"Edge feature mismatch for {arm_cfg['key']}: actual={features} expected={expected_arm_features}"
        )
    expected_channels = 4 + len(features)
    if int(source_config["model"].get("input_channels", expected_channels)) != expected_channels:
        raise RuntimeError(
            f"Input channel mismatch for {arm_cfg['key']}: "
            f"actual={source_config['model'].get('input_channels')} expected={expected_channels}"
        )

    rows = read_csv(required_paths["per_sample_csv"])
    rows_by_key: dict[tuple[float, str], dict[str, str]] = {}
    for row in rows:
        key = row_key(row)
        if key in rows_by_key:
            raise RuntimeError(f"Duplicate per-sample key for {arm_cfg['key']}: {key}")
        rows_by_key[key] = row
        for path_field in ["original", "m0_reconstruction", "refined", "m3_final"]:
            image_path = resolve_project_path(row[path_field])
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Missing {path_field} image for {arm_cfg['key']} {key}: {image_path}"
                )

    observed_snrs = sorted({float(key[0]) for key in rows_by_key})
    if observed_snrs != sorted(expected_snrs):
        raise RuntimeError(f"Per-sample SNR mismatch for {arm_cfg['key']}: {observed_snrs}")
    samples = sorted({key[1] for key in rows_by_key})
    for sample in samples:
        sample_snrs = sorted(snr for snr, name in rows_by_key if name == sample)
        if sample_snrs != sorted(expected_snrs):
            raise RuntimeError(
                f"Cluster {sample} in {arm_cfg['key']} does not contain all SNRs: {sample_snrs}"
            )

    summary_rows = read_csv(required_paths["summary_csv"])
    summary_snrs = sorted(float(row["snr_db"]) for row in summary_rows)
    if summary_snrs != sorted(expected_snrs):
        raise RuntimeError(f"Summary SNR mismatch for {arm_cfg['key']}: {summary_snrs}")

    state = checkpoint["model_state_dict"]
    parameter_count = int(sum(value.numel() for value in state.values() if torch.is_tensor(value)))
    best_epoch = int(checkpoint.get("epoch", -1))
    input_files = {name: file_fingerprint(path) for name, path in required_paths.items()}
    return ArmData(
        key=str(arm_cfg["key"]),
        experiment_id=str(arm_cfg["experiment_id"]),
        capacity=capacity,
        edge_conditioned=edge_conditioned,
        config_path=required_paths["config"],
        checkpoint_path=required_paths["checkpoint"],
        per_sample_path=required_paths["per_sample_csv"],
        summary_path=required_paths["summary_csv"],
        source_config=source_config,
        checkpoint=checkpoint,
        rows=rows,
        rows_by_key=rows_by_key,
        summary_rows=summary_rows,
        parameter_count=parameter_count,
        best_epoch=best_epoch,
        input_files=input_files,
    )


def validate_design(arms: dict[str, ArmData]) -> dict[str, Any]:
    required = {"small_no_edge", "small_edge", "large_no_edge", "large_edge"}
    if set(arms) != required:
        raise RuntimeError(f"Expected exactly four factorial arms {sorted(required)}, got {sorted(arms)}")

    matched_pairs = [
        ("small_no_edge", "small_edge"),
        ("large_no_edge", "large_edge"),
    ]
    pair_records: list[dict[str, Any]] = []
    for control_key, edge_key in matched_pairs:
        control = arms[control_key]
        edge = arms[edge_key]
        left = normalized_config(control.source_config, drop_edge=True, drop_capacity=False)
        right = normalized_config(edge.source_config, drop_edge=True, drop_capacity=False)
        differences = value_differences(left, right)
        if differences:
            raise RuntimeError(
                f"Matched edge pair {control_key}/{edge_key} differs outside edge inputs:\n"
                + "\n".join(differences[:30])
            )
        pair_records.append(
            {
                "control": control_key,
                "edge": edge_key,
                "capacity": control.capacity,
                "matched_outside_edge_inputs": True,
                "control_parameters": control.parameter_count,
                "edge_parameters": edge.parameter_count,
                "parameter_difference_from_two_input_channels": edge.parameter_count - control.parameter_count,
            }
        )

    common_reference = normalized_config(
        arms["small_no_edge"].source_config, drop_edge=True, drop_capacity=True
    )
    for key, arm in arms.items():
        candidate = normalized_config(arm.source_config, drop_edge=True, drop_capacity=True)
        differences = value_differences(common_reference, candidate)
        if differences:
            raise RuntimeError(
                f"Factorial arm {key} differs outside declared edge/capacity factors:\n"
                + "\n".join(differences[:30])
            )

    reference_keys = set(arms["small_no_edge"].rows_by_key)
    reference_samples = sorted({sample for _snr, sample in reference_keys})
    for key, arm in arms.items():
        if set(arm.rows_by_key) != reference_keys:
            missing = sorted(reference_keys - set(arm.rows_by_key))[:10]
            extra = sorted(set(arm.rows_by_key) - reference_keys)[:10]
            raise RuntimeError(f"Per-sample key mismatch for {key}: missing={missing} extra={extra}")
    for key_tuple in sorted(reference_keys):
        original_paths = {
            project_relative(resolve_project_path(arm.rows_by_key[key_tuple]["original"])) for arm in arms.values()
        }
        m0_paths = {
            project_relative(resolve_project_path(arm.rows_by_key[key_tuple]["m0_reconstruction"]))
            for arm in arms.values()
        }
        if len(original_paths) != 1 or len(m0_paths) != 1:
            raise RuntimeError(
                f"Original/M0 alignment mismatch at {key_tuple}: originals={original_paths}, m0={m0_paths}"
            )

    return {
        "matched_edge_pairs": pair_records,
        "all_nonfactor_fields_identical": True,
        "identical_sample_snr_keys": True,
        "identical_original_and_m0_paths": True,
        "num_samples": len(reference_samples),
        "num_rows_per_arm": len(reference_keys),
    }


def load_rgb_with_fingerprint(
    path: Path, png_registry: dict[str, dict[str, Any]]
) -> torch.Tensor:
    relative = project_relative(path)
    data = path.read_bytes()
    fingerprint = {
        "path": relative,
        "bytes": len(data),
        "sha256": sha256_bytes(data),
    }
    previous = png_registry.get(relative)
    if previous is not None and previous != fingerprint:
        raise RuntimeError(f"Input PNG changed during analysis: {relative}")
    png_registry[relative] = fingerprint
    with Image.open(io.BytesIO(data)) as image:
        array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).to(torch.float32).div_(255.0)


def psnr(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    if reference.shape != candidate.shape:
        raise RuntimeError(f"Image shape mismatch: reference={reference.shape} candidate={candidate.shape}")
    mse = torch.mean((candidate - reference) ** 2).item()
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def semantic_fields(source: dict[str, str]) -> dict[str, Any]:
    m0_ok = parse_bool(source["m0_matches_original_top1"])
    raw_ok = parse_bool(source["refined_matches_original_top1"])
    raw_matches_m0 = parse_bool(source["refined_matches_m0_top1"])
    m3_ok = parse_bool(source["m3_matches_original_top1"])
    accepted = parse_bool(source["detector_accept_refined"])
    return {
        "m0_matches_original_top1": m0_ok,
        "raw_refined_matches_original_top1": raw_ok,
        "raw_refined_matches_m0_top1": raw_matches_m0,
        "m3_matches_original_top1": m3_ok,
        "m3_accept_refined": accepted,
        "raw_new_error": m0_ok and not raw_ok,
        "raw_repair": (not m0_ok) and raw_ok,
        "m3_new_error": m0_ok and not m3_ok,
        "m3_repair": (not m0_ok) and m3_ok,
    }


def compute_per_sample(
    arms: dict[str, ArmData], snrs: list[float]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, set[str]]]:
    ordered_arms = ["small_no_edge", "small_edge", "large_no_edge", "large_edge"]
    samples = sorted({sample for _snr, sample in arms["small_no_edge"].rows_by_key})
    png_registry: dict[str, dict[str, Any]] = {}
    arm_png_paths: dict[str, set[str]] = {key: set() for key in ordered_arms}
    output_rows: list[dict[str, Any]] = []

    for sample in samples:
        for snr in snrs:
            key = (float(snr), sample)
            source_rows = {arm_key: arms[arm_key].rows_by_key[key] for arm_key in ordered_arms}
            original_path = resolve_project_path(source_rows[ordered_arms[0]]["original"])
            m0_path = resolve_project_path(source_rows[ordered_arms[0]]["m0_reconstruction"])
            original = load_rgb_with_fingerprint(original_path, png_registry)
            m0 = load_rgb_with_fingerprint(m0_path, png_registry)
            m0_psnr = psnr(original, m0)
            for arm_key in ordered_arms:
                arm = arms[arm_key]
                source = source_rows[arm_key]
                refined_path = resolve_project_path(source["refined"])
                m3_path = resolve_project_path(source["m3_final"])
                refined = load_rgb_with_fingerprint(refined_path, png_registry)
                m3 = load_rgb_with_fingerprint(m3_path, png_registry)
                refined_psnr = psnr(original, refined)
                m3_psnr = psnr(original, m3)
                paths = [original_path, m0_path, refined_path, m3_path]
                arm_png_paths[arm_key].update(project_relative(path) for path in paths)
                fingerprints = [png_registry[project_relative(path)] for path in paths]
                output_rows.append(
                    {
                        "arm": arm_key,
                        "experiment_id": arm.experiment_id,
                        "capacity": arm.capacity,
                        "edge_conditioned": arm.edge_conditioned,
                        "base_channels": int(arm.source_config["model"]["base_channels"]),
                        "num_blocks": int(arm.source_config["model"]["num_blocks"]),
                        "epochs": int(arm.source_config["training"]["epochs"]),
                        "parameter_count": arm.parameter_count,
                        "best_epoch": arm.best_epoch,
                        "snr_db": float(snr),
                        "sample": sample,
                        "m0_psnr_db": m0_psnr,
                        "raw_refined_psnr_db": refined_psnr,
                        "m3_psnr_db": m3_psnr,
                        "raw_refined_delta_vs_m0_db": refined_psnr - m0_psnr,
                        "m3_delta_vs_m0_db": m3_psnr - m0_psnr,
                        **semantic_fields(source),
                        "original": project_relative(original_path),
                        "m0_reconstruction": project_relative(m0_path),
                        "raw_refined": project_relative(refined_path),
                        "m3_final": project_relative(m3_path),
                        "original_sha256": fingerprints[0]["sha256"],
                        "m0_sha256": fingerprints[1]["sha256"],
                        "raw_refined_sha256": fingerprints[2]["sha256"],
                        "m3_sha256": fingerprints[3]["sha256"],
                    }
                )
    return output_rows, png_registry, arm_png_paths


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot compute mean of empty values")
    return float(sum(values) / len(values))


def summarize_arm_rows(
    rows: list[dict[str, Any]], arm: ArmData, level: str, snr: float | str
) -> dict[str, Any]:
    m0 = mean([float(row["m0_psnr_db"]) for row in rows])
    raw = mean([float(row["raw_refined_psnr_db"]) for row in rows])
    m3 = mean([float(row["m3_psnr_db"]) for row in rows])
    count = len(rows)
    return {
        "level": level,
        "arm": arm.key,
        "experiment_id": arm.experiment_id,
        "capacity": arm.capacity,
        "edge_conditioned": arm.edge_conditioned,
        "base_channels": int(arm.source_config["model"]["base_channels"]),
        "num_blocks": int(arm.source_config["model"]["num_blocks"]),
        "epochs": int(arm.source_config["training"]["epochs"]),
        "parameter_count": arm.parameter_count,
        "best_epoch": arm.best_epoch,
        "snr_db": snr,
        "num_clusters": len({str(row["sample"]) for row in rows}),
        "num_rows": count,
        "m0_psnr_db": m0,
        "raw_refined_psnr_db": raw,
        "m3_psnr_db": m3,
        "raw_refined_delta_vs_m0_db": raw - m0,
        "m3_delta_vs_m0_db": m3 - m0,
        "m0_failure_rate": mean([not bool(row["m0_matches_original_top1"]) for row in rows]),
        "raw_refined_failure_rate": mean(
            [not bool(row["raw_refined_matches_original_top1"]) for row in rows]
        ),
        "m3_failure_rate": mean([not bool(row["m3_matches_original_top1"]) for row in rows]),
        "raw_refinement_drift_rate": mean(
            [not bool(row["raw_refined_matches_m0_top1"]) for row in rows]
        ),
        "m3_accept_rate": mean([bool(row["m3_accept_refined"]) for row in rows]),
        "raw_new_error_count": sum(bool(row["raw_new_error"]) for row in rows),
        "raw_repair_count": sum(bool(row["raw_repair"]) for row in rows),
        "m3_new_error_count": sum(bool(row["m3_new_error"]) for row in rows),
        "m3_repair_count": sum(bool(row["m3_repair"]) for row in rows),
    }


def build_arm_summary(
    per_sample_rows: list[dict[str, Any]], arms: dict[str, ArmData], snrs: list[float]
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for arm_key in ["small_no_edge", "small_edge", "large_no_edge", "large_edge"]:
        arm_rows = [row for row in per_sample_rows if row["arm"] == arm_key]
        summaries.append(summarize_arm_rows(arm_rows, arms[arm_key], "all", "all"))
        for snr in snrs:
            subset = [row for row in arm_rows if float(row["snr_db"]) == float(snr)]
            summaries.append(summarize_arm_rows(subset, arms[arm_key], "snr", float(snr)))
    return summaries


def validate_source_summaries(
    per_sample_rows: list[dict[str, Any]], arms: dict[str, ArmData], tolerance: float
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    max_abs_difference = 0.0
    for arm_key, arm in arms.items():
        source_by_snr = {float(row["snr_db"]): row for row in arm.summary_rows}
        for snr, source in sorted(source_by_snr.items()):
            rows = [
                row
                for row in per_sample_rows
                if row["arm"] == arm_key and float(row["snr_db"]) == float(snr)
            ]
            values = {
                "m0_psnr_db": mean([float(row["m0_psnr_db"]) for row in rows]),
                "refined_psnr_db": mean([float(row["raw_refined_psnr_db"]) for row in rows]),
                "m3_psnr_db": mean([float(row["m3_psnr_db"]) for row in rows]),
            }
            for field, recomputed in values.items():
                recorded = float(source[field])
                difference = recomputed - recorded
                max_abs_difference = max(max_abs_difference, abs(difference))
                checks.append(
                    {
                        "arm": arm_key,
                        "snr_db": snr,
                        "field": field,
                        "recorded": recorded,
                        "recomputed": recomputed,
                        "difference": difference,
                    }
                )
    if max_abs_difference > tolerance:
        worst = max(checks, key=lambda row: abs(float(row["difference"])))
        raise RuntimeError(
            f"PNG PSNR does not reproduce source summary within {tolerance} dB; "
            f"worst={worst}"
        )
    return {
        "tolerance_db": tolerance,
        "max_abs_difference_db": max_abs_difference,
        "num_checks": len(checks),
        "passed": True,
    }


def factorial_contrasts() -> list[dict[str, Any]]:
    return [
        {
            "name": "edge_effect_small",
            "kind": "simple_edge_effect",
            "control": "small_no_edge",
            "treatment": "small_edge",
            "coefficients": {"small_no_edge": -1.0, "small_edge": 1.0},
            "positive_means": "edge improves PSNR at small capacity",
        },
        {
            "name": "edge_effect_large",
            "kind": "simple_edge_effect",
            "control": "large_no_edge",
            "treatment": "large_edge",
            "coefficients": {"large_no_edge": -1.0, "large_edge": 1.0},
            "positive_means": "edge improves PSNR at large capacity",
        },
        {
            "name": "capacity_budget_effect_no_edge",
            "kind": "simple_capacity_budget_effect",
            "control": "small_no_edge",
            "treatment": "large_no_edge",
            "coefficients": {"small_no_edge": -1.0, "large_no_edge": 1.0},
            "positive_means": "large capacity/training budget improves PSNR without edge",
        },
        {
            "name": "capacity_budget_effect_edge",
            "kind": "simple_capacity_budget_effect",
            "control": "small_edge",
            "treatment": "large_edge",
            "coefficients": {"small_edge": -1.0, "large_edge": 1.0},
            "positive_means": "large capacity/training budget improves PSNR with edge",
        },
        {
            "name": "average_edge_main_effect",
            "kind": "factorial_main_effect",
            "control": "average(no_edge)",
            "treatment": "average(edge)",
            "coefficients": {
                "small_no_edge": -0.5,
                "large_no_edge": -0.5,
                "small_edge": 0.5,
                "large_edge": 0.5,
            },
            "positive_means": "edge improves PSNR averaged over capacity levels",
        },
        {
            "name": "average_capacity_budget_main_effect",
            "kind": "factorial_main_effect",
            "control": "average(small)",
            "treatment": "average(large)",
            "coefficients": {
                "small_no_edge": -0.5,
                "small_edge": -0.5,
                "large_no_edge": 0.5,
                "large_edge": 0.5,
            },
            "positive_means": "large capacity/training budget improves PSNR averaged over edge levels",
        },
        {
            "name": "edge_by_capacity_interaction",
            "kind": "factorial_interaction",
            "control": "edge_effect_small",
            "treatment": "edge_effect_large",
            "coefficients": {
                "small_no_edge": 1.0,
                "small_edge": -1.0,
                "large_no_edge": -1.0,
                "large_edge": 1.0,
            },
            "positive_means": "edge benefit is larger at large capacity/training budget",
        },
    ]


def build_paired_effects(
    per_sample_rows: list[dict[str, Any]],
    snrs: list[float],
    replicates: int,
    seed: int,
    confidence_level: float,
) -> list[dict[str, Any]]:
    if replicates <= 0:
        raise ValueError("Bootstrap replicates must be positive")
    samples = sorted({str(row["sample"]) for row in per_sample_rows})
    arms = ["small_no_edge", "small_edge", "large_no_edge", "large_edge"]
    outcomes = ["raw_refined_psnr_db", "m3_psnr_db"]
    by_key = {
        (str(row["arm"]), str(row["sample"]), float(row["snr_db"])): row for row in per_sample_rows
    }
    arrays: dict[str, dict[str, np.ndarray]] = {outcome: {} for outcome in outcomes}
    for outcome in outcomes:
        for arm in arms:
            arrays[outcome][arm] = np.asarray(
                [
                    [float(by_key[(arm, sample, float(snr))][outcome]) for snr in snrs]
                    for sample in samples
                ],
                dtype=np.float64,
            )

    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(0, len(samples), size=(replicates, len(samples)), endpoint=False)
    alpha = (1.0 - confidence_level) / 2.0
    output: list[dict[str, Any]] = []
    subsets: list[tuple[str, str | float, list[int]]] = [("all", "all", list(range(len(snrs))))]
    subsets.extend(("snr", float(snr), [idx]) for idx, snr in enumerate(snrs))

    for contrast in factorial_contrasts():
        coefficients = contrast["coefficients"]
        for outcome in outcomes:
            contrast_array = sum(
                float(weight) * arrays[outcome][arm] for arm, weight in coefficients.items()
            )
            for level, snr_value, snr_indices in subsets:
                cluster_values = contrast_array[:, snr_indices].mean(axis=1)
                replicate_values = cluster_values[bootstrap_indices].mean(axis=1)
                estimate = float(cluster_values.mean())
                ci_low, ci_high = np.quantile(replicate_values, [alpha, 1.0 - alpha]).tolist()
                probability_positive = float(np.mean(replicate_values > 0.0))
                probability_negative = float(np.mean(replicate_values < 0.0))
                p_two_sided = float(min(1.0, 2.0 * min(probability_positive, probability_negative)))
                output.append(
                    {
                        "level": level,
                        "snr_db": snr_value,
                        "contrast": contrast["name"],
                        "contrast_kind": contrast["kind"],
                        "outcome": outcome,
                        "control": contrast["control"],
                        "treatment": contrast["treatment"],
                        "coefficients": coefficients,
                        "estimate_db": estimate,
                        "ci_low_db": float(ci_low),
                        "ci_high_db": float(ci_high),
                        "bootstrap_standard_error_db": float(np.std(replicate_values, ddof=1)),
                        "cluster_standard_deviation_db": float(np.std(cluster_values, ddof=1)),
                        "probability_effect_gt_zero": probability_positive,
                        "bootstrap_two_sided_p": p_two_sided,
                        "ci_excludes_zero": bool(ci_low > 0.0 or ci_high < 0.0),
                        "num_clusters": len(samples),
                        "snrs_per_cluster": len(snr_indices),
                        "bootstrap_replicates": replicates,
                        "bootstrap_seed": seed,
                        "confidence_level": confidence_level,
                        "positive_means": contrast["positive_means"],
                    }
                )
    return output


def png_manifest(
    paths: set[str], png_registry: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    digest = hashlib.sha256()
    total_bytes = 0
    for path in sorted(paths):
        item = png_registry[path]
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item["sha256"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(item["bytes"]).encode("ascii"))
        digest.update(b"\n")
        total_bytes += int(item["bytes"])
    return {
        "num_unique_pngs": len(paths),
        "total_bytes": total_bytes,
        "manifest_sha256": digest.hexdigest(),
        "definition": "SHA256 over sorted path\\0file_sha256\\0bytes\\n records",
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{float(value):.{digits}f}"
    return str(value)


def signed(value: Any, digits: int = 4) -> str:
    return f"{float(value):+.{digits}f}"


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    lines = [
        "| " + " | ".join(label for _key, label in columns) + " |",
        "|" + "|".join("---" for _key, _label in columns) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key)) for key, _label in columns) + " |")
    return lines


def make_report(
    config: dict[str, Any],
    arm_summary: list[dict[str, Any]],
    paired_effects: list[dict[str, Any]],
    design_validation: dict[str, Any],
    summary_validation: dict[str, Any],
) -> str:
    all_arms = [row for row in arm_summary if row["level"] == "all"]
    all_effects = [
        row
        for row in paired_effects
        if row["level"] == "all"
        and row["contrast"]
        in {
            "edge_effect_small",
            "edge_effect_large",
            "average_edge_main_effect",
            "average_capacity_budget_main_effect",
            "edge_by_capacity_interaction",
        }
    ]
    small_edge_raw = next(
        row
        for row in all_effects
        if row["contrast"] == "edge_effect_small" and row["outcome"] == "raw_refined_psnr_db"
    )
    large_edge_raw = next(
        row
        for row in all_effects
        if row["contrast"] == "edge_effect_large" and row["outcome"] == "raw_refined_psnr_db"
    )
    small_edge_m3 = next(
        row
        for row in all_effects
        if row["contrast"] == "edge_effect_small" and row["outcome"] == "m3_psnr_db"
    )
    large_edge_m3 = next(
        row
        for row in all_effects
        if row["contrast"] == "edge_effect_large" and row["outcome"] == "m3_psnr_db"
    )
    arms_by_key = {str(row["arm"]): row for row in all_arms}
    small_raw_failure_delta = (
        float(arms_by_key["small_edge"]["raw_refined_failure_rate"])
        - float(arms_by_key["small_no_edge"]["raw_refined_failure_rate"])
    )
    large_raw_failure_delta = (
        float(arms_by_key["large_edge"]["raw_refined_failure_rate"])
        - float(arms_by_key["large_no_edge"]["raw_refined_failure_rate"])
    )
    lines = [
        "# 2x2 Edge × Capacity/Training-Budget Validation Ablation",
        "",
        "This report recomputes per-sample PSNR directly from saved PNG files for four aligned validation arms. "
        "Paired uncertainty uses a sample-cluster bootstrap: each sampled cluster retains all five SNR rows and the same draw is used across all arms.",
        "",
        "## Bottom Line",
        "",
        f"- At small capacity, edge changes raw refined PSNR by `{signed(small_edge_raw['estimate_db'])}` dB "
        f"(95% CI `{signed(small_edge_raw['ci_low_db'])}`, `{signed(small_edge_raw['ci_high_db'])}`) and M3 PSNR by "
        f"`{signed(small_edge_m3['estimate_db'])}` dB (95% CI `{signed(small_edge_m3['ci_low_db'])}`, `{signed(small_edge_m3['ci_high_db'])}`).",
        f"- At large capacity, edge changes raw refined PSNR by `{signed(large_edge_raw['estimate_db'])}` dB "
        f"(95% CI `{signed(large_edge_raw['ci_low_db'])}`, `{signed(large_edge_raw['ci_high_db'])}`) and M3 PSNR by "
        f"`{signed(large_edge_m3['estimate_db'])}` dB (95% CI `{signed(large_edge_m3['ci_low_db'])}`, `{signed(large_edge_m3['ci_high_db'])}`).",
        f"- Edge is not a same-classifier semantic improvement: raw pseudo failure changes by "
        f"`{signed(small_raw_failure_delta)}` at small capacity and `{signed(large_raw_failure_delta)}` at large capacity. "
        f"Large-edge raw new errors/repairs are `{arms_by_key['large_edge']['raw_new_error_count']}/{arms_by_key['large_edge']['raw_repair_count']}` "
        f"versus `{arms_by_key['large_no_edge']['raw_new_error_count']}/{arms_by_key['large_no_edge']['raw_repair_count']}` without edge.",
        "- The capacity factor is intentionally labeled capacity/training-budget: width, depth, and epochs change together, so it is not a pure parameter-count effect.",
        "- Source-AlexNet M3 top-1 fallback preserves its own M0 top-1 by construction; raw refined semantic counts remain the informative same-classifier risk diagnostic.",
        "",
        "## Arm Summary",
        "",
    ]
    lines += markdown_table(
        all_arms,
        [
            ("arm", "Arm"),
            ("experiment_id", "Experiment"),
            ("capacity", "Capacity"),
            ("edge_conditioned", "Edge"),
            ("parameter_count", "Parameters"),
            ("best_epoch", "Best Epoch"),
            ("raw_refined_delta_vs_m0_db", "Raw ΔPSNR"),
            ("m3_delta_vs_m0_db", "M3 ΔPSNR"),
            ("raw_refined_failure_rate", "Raw Failure"),
            ("raw_new_error_count", "Raw New Error"),
            ("raw_repair_count", "Raw Repair"),
        ],
    )
    lines.extend(["", "## Paired Factorial Effects", ""])
    lines += markdown_table(
        all_effects,
        [
            ("contrast", "Contrast"),
            ("outcome", "Outcome"),
            ("estimate_db", "Estimate dB"),
            ("ci_low_db", "CI Low"),
            ("ci_high_db", "CI High"),
            ("ci_excludes_zero", "CI Excludes 0"),
            ("probability_effect_gt_zero", "P(effect>0)"),
        ],
    )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Matched edge pairs validated: `{len(design_validation['matched_edge_pairs'])}`.",
            f"- Aligned clusters/rows: `{design_validation['num_samples']}` samples × `5` SNR = `{design_validation['num_rows_per_arm']}` rows per arm.",
            f"- Maximum PNG-vs-source-summary PSNR discrepancy: `{summary_validation['max_abs_difference_db']:.8f}` dB.",
            f"- Bootstrap: `{config['bootstrap']['replicates']}` paired replicates, seed `{config['bootstrap']['seed']}`, percentile `{int(float(config['bootstrap']['confidence_level']) * 100)}`% CI.",
            "",
            "## Output Files",
            "",
            "- `arm_summary.csv`: all-SNR and per-SNR arm summaries.",
            "- `paired_effects.csv`: simple effects, factorial main effects, and interaction with cluster-bootstrap intervals.",
            "- `per_sample.csv`: recomputed PNG-level PSNR, semantics, paths, and SHA256 values.",
            "- `metadata.json`: design checks, complete primary input fingerprints, PNG manifest fingerprints, checkpoint metadata, and output fingerprints.",
            "",
            "## Scope",
            "",
            "This is a validation-split factorial ablation, not a held-out or test-like claim. It isolates edge inputs within each matched capacity/training-budget level. "
            "It does not establish supervised semantic correctness on COCO pseudo-labels.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Analysis config is not a mapping: {config_path}")
    if args.bootstrap_replicates is not None:
        config["bootstrap"]["replicates"] = int(args.bootstrap_replicates)
    torch.set_num_threads(int(config.get("runtime", {}).get("torch_num_threads", 1)))
    snrs = [float(item) for item in config["snrs"]]
    arms: dict[str, ArmData] = {}
    for arm_cfg in config["arms"]:
        arm = validate_arm(arm_cfg, config, snrs)
        if arm.key in arms:
            raise RuntimeError(f"Duplicate arm key: {arm.key}")
        arms[arm.key] = arm
    design_validation = validate_design(arms)

    dry_run_payload = {
        "status": "ok",
        "config": file_fingerprint(config_path),
        "analysis_id": config["analysis_id"],
        "snrs": snrs,
        "bootstrap": config["bootstrap"],
        "design_validation": design_validation,
        "arms": {
            key: {
                "experiment_id": arm.experiment_id,
                "capacity": arm.capacity,
                "edge_conditioned": arm.edge_conditioned,
                "parameter_count": arm.parameter_count,
                "best_epoch": arm.best_epoch,
                "num_rows": len(arm.rows),
                "input_files": arm.input_files,
            }
            for key, arm in arms.items()
        },
    }
    if args.dry_run:
        print(json.dumps(dry_run_payload, indent=2, ensure_ascii=False, sort_keys=True))
        return

    output_dir = resolve_project_path(args.output_dir or config["outputs"]["output_dir"])
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory exists; use --overwrite to replace: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    per_sample_rows, png_registry, arm_png_paths = compute_per_sample(arms, snrs)
    summary_validation = validate_source_summaries(
        per_sample_rows,
        arms,
        float(config["validation"]["summary_psnr_tolerance_db"]),
    )
    arm_summary = build_arm_summary(per_sample_rows, arms, snrs)
    paired_effects = build_paired_effects(
        per_sample_rows,
        snrs,
        int(config["bootstrap"]["replicates"]),
        int(config["bootstrap"]["seed"]),
        float(config["bootstrap"]["confidence_level"]),
    )

    arm_summary_path = output_dir / "arm_summary.csv"
    paired_effects_path = output_dir / "paired_effects.csv"
    per_sample_path = output_dir / "per_sample.csv"
    report_path = output_dir / "REPORT.md"
    metadata_path = output_dir / "metadata.json"
    write_csv(arm_summary_path, arm_summary)
    write_csv(paired_effects_path, paired_effects)
    write_csv(per_sample_path, per_sample_rows)
    report_path.write_text(
        make_report(config, arm_summary, paired_effects, design_validation, summary_validation),
        encoding="utf-8",
    )

    arm_metadata: dict[str, Any] = {}
    for key, arm in arms.items():
        arm_metadata[key] = {
            "experiment_id": arm.experiment_id,
            "capacity": arm.capacity,
            "edge_conditioned": arm.edge_conditioned,
            "base_channels": int(arm.source_config["model"]["base_channels"]),
            "num_blocks": int(arm.source_config["model"]["num_blocks"]),
            "epochs": int(arm.source_config["training"]["epochs"]),
            "parameter_count": arm.parameter_count,
            "best_epoch": arm.best_epoch,
            "checkpoint_eval_stats": arm.checkpoint.get("eval_stats", {}),
            "input_files": arm.input_files,
            "png_inputs": png_manifest(arm_png_paths[key], png_registry),
        }

    output_fingerprints = {
        "arm_summary_csv": file_fingerprint(arm_summary_path),
        "paired_effects_csv": file_fingerprint(paired_effects_path),
        "per_sample_csv": file_fingerprint(per_sample_path),
        "report_md": file_fingerprint(report_path),
    }
    metadata = {
        "analysis_id": config["analysis_id"],
        "method": config["method"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_commit": git_commit(),
        "git_dirty_state": git_dirty_state(),
        "repository_url": config.get("repository_url"),
        "run_command": " ".join(sys.argv),
        "script": file_fingerprint(Path(__file__)),
        "config": file_fingerprint(config_path),
        "dataset": config["dataset"],
        "image_size": int(config["image_size"]),
        "channel": config["channel"],
        "snrs": snrs,
        "cbr": float(config["cbr"]),
        "seed": int(config["seed"]),
        "design": config["design"],
        "design_validation": design_validation,
        "source_summary_validation": summary_validation,
        "bootstrap": {
            **config["bootstrap"],
            "implementation": "paired percentile bootstrap over sample IDs; every draw retains all configured SNR rows and all four arms",
        },
        "arms": arm_metadata,
        "all_unique_png_inputs": png_manifest(set(png_registry), png_registry),
        "outputs": output_fingerprints,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "pillow": Image.__version__ if hasattr(Image, "__version__") else "unknown",
            "platform": platform.platform(),
            "proxy_environment_present": sorted(key for key in os.environ if "proxy" in key.lower()),
        },
        "notes": [
            "PSNR is recomputed from 8-bit saved PNG files using float32 tensors in [0,1].",
            "The capacity factor bundles width/depth and training epochs; interpret it as capacity/training-budget, not pure capacity.",
            "No model inference, classifier inference, diffusion, model download, or data download is performed.",
        ],
    }
    save_json(metadata_path, metadata)
    print(
        json.dumps(
            {
                "output_dir": project_relative(output_dir),
                "num_per_sample_rows": len(per_sample_rows),
                "num_arm_summary_rows": len(arm_summary),
                "num_paired_effect_rows": len(paired_effects),
                "bootstrap_replicates": int(config["bootstrap"]["replicates"]),
                "max_source_summary_difference_db": summary_validation["max_abs_difference_db"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
