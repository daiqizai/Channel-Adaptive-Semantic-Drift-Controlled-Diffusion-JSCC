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
        description="Compare matched large-capacity edge/no-edge raw refinements across validation and holdout splits."
    )
    parser.add_argument(
        "--config",
        default="configs/s6_matched_edge_holdout_audit_exp_s4_008_009.yaml",
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
    if isinstance(value, (list, dict, tuple)):
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
    return {"path": project_relative(path), "bytes": size, "sha256": digest.hexdigest()}


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


def remove_nested(mapping: dict[str, Any], path: tuple[str, ...]) -> None:
    current: Any = mapping
    for key in path[:-1]:
        if not isinstance(current, dict) or key not in current:
            return
        current = current[key]
    if isinstance(current, dict):
        current.pop(path[-1], None)


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
    if left != right:
        return [f"{path}: {left!r} != {right!r}"]
    return []


BASE_METADATA_PATHS = [
    ("experiment_id",),
    ("method",),
    ("notes",),
    ("outputs",),
]

EDGE_PATHS = [
    ("model", "name"),
    ("model", "input_channels"),
    ("model", "condition_features"),
]

GATE_METADATA_PATHS = [
    ("analysis_id",),
    ("source_experiment",),
    ("method",),
    ("notes",),
    ("outputs",),
    ("evaluation", "warning"),
    ("inputs", "refiner_checkpoint"),
    ("inputs", "source_config"),
]


def normalized_config(
    config: dict[str, Any], metadata_paths: list[tuple[str, ...]]
) -> dict[str, Any]:
    output = copy.deepcopy(config)
    for path in [*metadata_paths, *EDGE_PATHS]:
        remove_nested(output, path)
    return output


def row_key(row: dict[str, Any]) -> tuple[float, str]:
    return float(row["snr_db"]), str(row["sample"])


@dataclass
class BaseArm:
    name: str
    experiment_id: str
    config_path: Path
    checkpoint_path: Path
    source_config: dict[str, Any]
    checkpoint: dict[str, Any]
    parameter_count: int
    best_epoch: int
    inputs: dict[str, dict[str, Any]]


@dataclass
class SplitInput:
    split: str
    expected_sample_count: int
    no_edge_config: dict[str, Any]
    edge_config: dict[str, Any]
    no_edge_rows: dict[tuple[float, str], dict[str, str]]
    edge_rows: dict[tuple[float, str], dict[str, str]]
    no_edge_summary: list[dict[str, str]]
    edge_summary: list[dict[str, str]]
    input_files: dict[str, dict[str, Any]]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"YAML root is not a mapping: {path}")
    return payload


def load_base_arm(name: str, cfg: dict[str, Any]) -> BaseArm:
    config_path = resolve_project_path(cfg["config"])
    checkpoint_path = resolve_project_path(cfg["checkpoint"])
    for path in [config_path, checkpoint_path]:
        if not path.is_file():
            raise FileNotFoundError(path)
    source_config = load_yaml(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("config") != source_config:
        differences = value_differences(source_config, checkpoint.get("config"))
        raise RuntimeError(f"Checkpoint config mismatch for {name}:\n" + "\n".join(differences[:30]))
    if str(source_config.get("experiment_id")) != str(cfg["experiment_id"]):
        raise RuntimeError(f"Experiment ID mismatch for {name}")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint lacks model_state_dict: {checkpoint_path}")
    parameter_count = int(sum(value.numel() for value in state.values() if torch.is_tensor(value)))
    return BaseArm(
        name=name,
        experiment_id=str(cfg["experiment_id"]),
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        source_config=source_config,
        checkpoint=checkpoint,
        parameter_count=parameter_count,
        best_epoch=int(checkpoint.get("epoch", -1)),
        inputs={
            "config": file_fingerprint(config_path),
            "checkpoint": file_fingerprint(checkpoint_path),
        },
    )


def validate_base_pair(no_edge: BaseArm, edge: BaseArm, expected_features: list[str]) -> dict[str, Any]:
    left = normalized_config(no_edge.source_config, BASE_METADATA_PATHS)
    right = normalized_config(edge.source_config, BASE_METADATA_PATHS)
    differences = value_differences(left, right)
    if differences:
        raise RuntimeError("Large matched base arms differ outside edge inputs:\n" + "\n".join(differences[:30]))
    no_edge_features = no_edge.source_config["model"].get("condition_features", [])
    edge_features = edge.source_config["model"].get("condition_features", [])
    if no_edge_features != [] or edge_features != expected_features:
        raise RuntimeError(
            f"Unexpected edge features: no_edge={no_edge_features}, edge={edge_features}, expected={expected_features}"
        )
    if int(no_edge.source_config["model"]["input_channels"]) != 4:
        raise RuntimeError("No-edge base arm must have four inputs")
    if int(edge.source_config["model"]["input_channels"]) != 4 + len(expected_features):
        raise RuntimeError("Edge base arm input channel count does not match edge features")
    return {
        "matched_outside_edge_inputs": True,
        "no_edge_parameters": no_edge.parameter_count,
        "edge_parameters": edge.parameter_count,
        "parameter_difference": edge.parameter_count - no_edge.parameter_count,
        "no_edge_best_epoch": no_edge.best_epoch,
        "edge_best_epoch": edge.best_epoch,
        "base_channels": int(edge.source_config["model"]["base_channels"]),
        "num_blocks": int(edge.source_config["model"]["num_blocks"]),
        "epochs": int(edge.source_config["training"]["epochs"]),
    }


def index_rows(path: Path) -> dict[tuple[float, str], dict[str, str]]:
    output: dict[tuple[float, str], dict[str, str]] = {}
    for row in read_csv(path):
        key = row_key(row)
        if key in output:
            raise RuntimeError(f"Duplicate row key in {path}: {key}")
        output[key] = row
    return output


def validate_split_pair_configs(
    split: str, no_edge_config: dict[str, Any], edge_config: dict[str, Any]
) -> None:
    metadata_paths = BASE_METADATA_PATHS if split == "validation" else GATE_METADATA_PATHS
    left = normalized_config(no_edge_config, metadata_paths)
    right = normalized_config(edge_config, metadata_paths)
    differences = value_differences(left, right)
    if differences:
        raise RuntimeError(
            f"Split {split} edge/no-edge configs differ outside allowed arm metadata:\n"
            + "\n".join(differences[:30])
        )
    if split != "validation":
        if str(no_edge_config.get("split_name")) != split or str(edge_config.get("split_name")) != split:
            raise RuntimeError(f"Gate config split name mismatch for {split}")


def load_split(
    split_cfg: dict[str, Any], expected_snrs: list[float]
) -> SplitInput:
    split = str(split_cfg["name"])
    paths: dict[str, Path] = {}
    for arm in ["no_edge", "edge"]:
        for field in ["config", "per_sample_csv", "summary_csv"]:
            path = resolve_project_path(split_cfg[arm][field])
            if not path.is_file():
                raise FileNotFoundError(path)
            paths[f"{arm}_{field}"] = path
    no_edge_config = load_yaml(paths["no_edge_config"])
    edge_config = load_yaml(paths["edge_config"])
    validate_split_pair_configs(split, no_edge_config, edge_config)
    no_edge_rows = index_rows(paths["no_edge_per_sample_csv"])
    edge_rows = index_rows(paths["edge_per_sample_csv"])
    if set(no_edge_rows) != set(edge_rows):
        missing = sorted(set(no_edge_rows) - set(edge_rows))[:10]
        extra = sorted(set(edge_rows) - set(no_edge_rows))[:10]
        raise RuntimeError(f"Split {split} paired key mismatch: edge_missing={missing}, edge_extra={extra}")
    observed_snrs = sorted({snr for snr, _sample in no_edge_rows})
    if observed_snrs != sorted(expected_snrs):
        raise RuntimeError(f"Split {split} SNR mismatch: {observed_snrs}")
    samples = sorted({sample for _snr, sample in no_edge_rows})
    if len(samples) != int(split_cfg["expected_sample_count"]):
        raise RuntimeError(
            f"Split {split} sample count mismatch: actual={len(samples)} expected={split_cfg['expected_sample_count']}"
        )
    for sample in samples:
        sample_snrs = sorted(snr for snr, name in no_edge_rows if name == sample)
        if sample_snrs != sorted(expected_snrs):
            raise RuntimeError(f"Split {split} cluster {sample} does not retain all SNRs: {sample_snrs}")
    for key in sorted(no_edge_rows):
        no_edge_row = no_edge_rows[key]
        edge_row = edge_rows[key]
        for path_field in ["original", "m0_reconstruction"]:
            if project_relative(resolve_project_path(no_edge_row[path_field])) != project_relative(
                resolve_project_path(edge_row[path_field])
            ):
                raise RuntimeError(f"Split {split} {path_field} mismatch at {key}")
        for prediction_field in ["original_top1_index", "m0_top1_index", "m0_matches_original_top1"]:
            if str(no_edge_row[prediction_field]) != str(edge_row[prediction_field]):
                raise RuntimeError(f"Split {split} AlexNet source prediction mismatch at {key}: {prediction_field}")
        for arm_name, row in [("no_edge", no_edge_row), ("edge", edge_row)]:
            for path_field in ["original", "m0_reconstruction", "refined"]:
                path = resolve_project_path(row[path_field])
                if not path.is_file():
                    raise FileNotFoundError(f"Missing {arm_name} {path_field} at {key}: {path}")
    return SplitInput(
        split=split,
        expected_sample_count=int(split_cfg["expected_sample_count"]),
        no_edge_config=no_edge_config,
        edge_config=edge_config,
        no_edge_rows=no_edge_rows,
        edge_rows=edge_rows,
        no_edge_summary=read_csv(paths["no_edge_summary_csv"]),
        edge_summary=read_csv(paths["edge_summary_csv"]),
        input_files={name: file_fingerprint(path) for name, path in paths.items()},
    )


def load_rgb_with_fingerprint(path: Path, registry: dict[str, dict[str, Any]]) -> torch.Tensor:
    relative = project_relative(path)
    data = path.read_bytes()
    item = {"path": relative, "bytes": len(data), "sha256": sha256_bytes(data)}
    previous = registry.get(relative)
    if previous is not None and previous != item:
        raise RuntimeError(f"PNG changed during analysis: {relative}")
    registry[relative] = item
    with Image.open(io.BytesIO(data)) as image:
        array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).to(torch.float32).div_(255.0)


def psnr(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    if reference.shape != candidate.shape:
        raise RuntimeError(f"Image shape mismatch: {reference.shape} != {candidate.shape}")
    mse = torch.mean((candidate - reference) ** 2).item()
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def semantic_values(row: dict[str, str]) -> dict[str, bool]:
    m0_ok = parse_bool(row["m0_matches_original_top1"])
    refined_ok = parse_bool(row["refined_matches_original_top1"])
    refined_matches_m0 = parse_bool(row["refined_matches_m0_top1"])
    return {
        "m0_ok": m0_ok,
        "raw_ok": refined_ok,
        "raw_matches_m0": refined_matches_m0,
        "raw_failure": not refined_ok,
        "raw_new_error": m0_ok and not refined_ok,
        "raw_repair": (not m0_ok) and refined_ok,
    }


def compute_rows(
    split_inputs: list[SplitInput], snrs: list[float]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, set[str]], dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    paths_by_split: dict[str, set[str]] = {item.split: set() for item in split_inputs}
    output: list[dict[str, Any]] = []
    row_psnr_checks: list[float] = []
    for split_input in split_inputs:
        samples = sorted({sample for _snr, sample in split_input.no_edge_rows})
        for sample in samples:
            for snr in snrs:
                key = (float(snr), sample)
                no_edge_source = split_input.no_edge_rows[key]
                edge_source = split_input.edge_rows[key]
                original_path = resolve_project_path(no_edge_source["original"])
                m0_path = resolve_project_path(no_edge_source["m0_reconstruction"])
                no_edge_path = resolve_project_path(no_edge_source["refined"])
                edge_path = resolve_project_path(edge_source["refined"])
                original = load_rgb_with_fingerprint(original_path, registry)
                m0 = load_rgb_with_fingerprint(m0_path, registry)
                no_edge_image = load_rgb_with_fingerprint(no_edge_path, registry)
                edge_image = load_rgb_with_fingerprint(edge_path, registry)
                m0_psnr = psnr(original, m0)
                no_edge_psnr = psnr(original, no_edge_image)
                edge_psnr = psnr(original, edge_image)
                for source, computed in [(no_edge_source, no_edge_psnr), (edge_source, edge_psnr)]:
                    recorded = source.get("refined_psnr_db", "")
                    if recorded not in ("", None):
                        row_psnr_checks.append(computed - float(recorded))
                no_edge_semantic = semantic_values(no_edge_source)
                edge_semantic = semantic_values(edge_source)
                paths = [original_path, m0_path, no_edge_path, edge_path]
                paths_by_split[split_input.split].update(project_relative(path) for path in paths)
                fingerprints = [registry[project_relative(path)] for path in paths]
                output.append(
                    {
                        "split": split_input.split,
                        "snr_db": float(snr),
                        "sample": sample,
                        "m0_psnr_db": m0_psnr,
                        "no_edge_raw_psnr_db": no_edge_psnr,
                        "edge_raw_psnr_db": edge_psnr,
                        "edge_minus_no_edge_raw_psnr_db": edge_psnr - no_edge_psnr,
                        "no_edge_raw_delta_vs_m0_db": no_edge_psnr - m0_psnr,
                        "edge_raw_delta_vs_m0_db": edge_psnr - m0_psnr,
                        "m0_matches_original_top1": no_edge_semantic["m0_ok"],
                        "no_edge_raw_matches_original_top1": no_edge_semantic["raw_ok"],
                        "edge_raw_matches_original_top1": edge_semantic["raw_ok"],
                        "no_edge_raw_matches_m0_top1": no_edge_semantic["raw_matches_m0"],
                        "edge_raw_matches_m0_top1": edge_semantic["raw_matches_m0"],
                        "no_edge_raw_failure": no_edge_semantic["raw_failure"],
                        "edge_raw_failure": edge_semantic["raw_failure"],
                        "edge_minus_no_edge_raw_failure_indicator": int(edge_semantic["raw_failure"])
                        - int(no_edge_semantic["raw_failure"]),
                        "no_edge_raw_new_error": no_edge_semantic["raw_new_error"],
                        "edge_raw_new_error": edge_semantic["raw_new_error"],
                        "edge_minus_no_edge_raw_new_error_indicator": int(edge_semantic["raw_new_error"])
                        - int(no_edge_semantic["raw_new_error"]),
                        "no_edge_raw_repair": no_edge_semantic["raw_repair"],
                        "edge_raw_repair": edge_semantic["raw_repair"],
                        "edge_minus_no_edge_raw_repair_indicator": int(edge_semantic["raw_repair"])
                        - int(no_edge_semantic["raw_repair"]),
                        "original": project_relative(original_path),
                        "m0_reconstruction": project_relative(m0_path),
                        "no_edge_raw_refined": project_relative(no_edge_path),
                        "edge_raw_refined": project_relative(edge_path),
                        "original_sha256": fingerprints[0]["sha256"],
                        "m0_sha256": fingerprints[1]["sha256"],
                        "no_edge_raw_sha256": fingerprints[2]["sha256"],
                        "edge_raw_sha256": fingerprints[3]["sha256"],
                    }
                )
    return output, registry, paths_by_split, {
        "num_row_level_checks": len(row_psnr_checks),
        "max_abs_row_psnr_difference_db": max((abs(value) for value in row_psnr_checks), default=0.0),
    }


def mean(values: list[float | bool | int]) -> float:
    if not values:
        raise ValueError("Cannot average empty values")
    return float(sum(values) / len(values))


def source_summary_by_snr(rows: list[dict[str, str]]) -> dict[float, dict[str, str]]:
    output: dict[float, dict[str, str]] = {}
    for row in rows:
        value = row.get("snr_db", "")
        if value in ("", None, "all"):
            continue
        output[float(value)] = row
    return output


def validate_source_summaries(
    per_sample: list[dict[str, Any]], split_inputs: list[SplitInput], tolerance: float, row_check: dict[str, Any]
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for split_input in split_inputs:
        for arm, source_rows, field in [
            ("no_edge", split_input.no_edge_summary, "no_edge_raw_psnr_db"),
            ("edge", split_input.edge_summary, "edge_raw_psnr_db"),
        ]:
            source = source_summary_by_snr(source_rows)
            for snr, source_row in sorted(source.items()):
                rows = [
                    row
                    for row in per_sample
                    if row["split"] == split_input.split and float(row["snr_db"]) == float(snr)
                ]
                recomputed = mean([float(row[field]) for row in rows])
                recorded = float(source_row["refined_psnr_db"])
                checks.append(
                    {
                        "split": split_input.split,
                        "arm": arm,
                        "snr_db": snr,
                        "recorded": recorded,
                        "recomputed": recomputed,
                        "difference": recomputed - recorded,
                    }
                )
    max_summary = max(abs(float(row["difference"])) for row in checks)
    max_row = float(row_check["max_abs_row_psnr_difference_db"])
    if max(max_summary, max_row) > tolerance:
        raise RuntimeError(
            f"PNG PSNR reproduction exceeds tolerance {tolerance}: summary={max_summary}, row={max_row}"
        )
    return {
        "tolerance_db": tolerance,
        "num_summary_checks": len(checks),
        "max_abs_summary_psnr_difference_db": max_summary,
        **row_check,
        "passed": True,
    }


def summarize_semantics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    no_edge_failure = mean([bool(row["no_edge_raw_failure"]) for row in rows])
    edge_failure = mean([bool(row["edge_raw_failure"]) for row in rows])
    no_edge_new = sum(bool(row["no_edge_raw_new_error"]) for row in rows)
    edge_new = sum(bool(row["edge_raw_new_error"]) for row in rows)
    no_edge_repair = sum(bool(row["no_edge_raw_repair"]) for row in rows)
    edge_repair = sum(bool(row["edge_raw_repair"]) for row in rows)
    return {
        "pseudo_no_edge_raw_failure_rate": no_edge_failure,
        "pseudo_edge_raw_failure_rate": edge_failure,
        "pseudo_edge_minus_no_edge_failure_rate": edge_failure - no_edge_failure,
        "pseudo_no_edge_raw_new_error_count": no_edge_new,
        "pseudo_edge_raw_new_error_count": edge_new,
        "pseudo_edge_minus_no_edge_new_error_count": edge_new - no_edge_new,
        "pseudo_no_edge_raw_new_error_rate": no_edge_new / count,
        "pseudo_edge_raw_new_error_rate": edge_new / count,
        "pseudo_edge_minus_no_edge_new_error_rate": (edge_new - no_edge_new) / count,
        "pseudo_no_edge_raw_repair_count": no_edge_repair,
        "pseudo_edge_raw_repair_count": edge_repair,
        "pseudo_edge_minus_no_edge_repair_count": edge_repair - no_edge_repair,
        "pseudo_no_edge_raw_repair_rate": no_edge_repair / count,
        "pseudo_edge_raw_repair_rate": edge_repair / count,
        "pseudo_edge_minus_no_edge_repair_rate": (edge_repair - no_edge_repair) / count,
        "semantic_reference": "source_alexnet_pseudo_top1",
    }


def build_split_summary(
    per_sample: list[dict[str, Any]],
    split_inputs: list[SplitInput],
    snrs: list[float],
    replicates: int,
    seed: int,
    confidence_level: float,
) -> list[dict[str, Any]]:
    alpha = (1.0 - confidence_level) / 2.0
    output: list[dict[str, Any]] = []
    for split_input in split_inputs:
        split_rows = [row for row in per_sample if row["split"] == split_input.split]
        samples = sorted({str(row["sample"]) for row in split_rows})
        by_key = {(str(row["sample"]), float(row["snr_db"])): row for row in split_rows}
        effects = np.asarray(
            [
                [float(by_key[(sample, float(snr))]["edge_minus_no_edge_raw_psnr_db"]) for snr in snrs]
                for sample in samples
            ],
            dtype=np.float64,
        )
        rng = np.random.default_rng(seed)
        bootstrap_indices = rng.integers(0, len(samples), size=(replicates, len(samples)), endpoint=False)
        per_snr_estimates = effects.mean(axis=0)
        positive_snr_count = int(np.sum(per_snr_estimates > 0.0))
        negative_snr_count = int(np.sum(per_snr_estimates < 0.0))
        zero_snr_count = int(len(snrs) - positive_snr_count - negative_snr_count)
        same_direction = "positive" if positive_snr_count == len(snrs) else (
            "negative" if negative_snr_count == len(snrs) else "mixed"
        )
        subsets: list[tuple[str, float | str, list[int]]] = [("all", "all", list(range(len(snrs))))]
        subsets.extend(("snr", float(snr), [index]) for index, snr in enumerate(snrs))
        for level, snr_value, snr_indices in subsets:
            rows = split_rows if level == "all" else [
                row for row in split_rows if float(row["snr_db"]) == float(snr_value)
            ]
            cluster_values = effects[:, snr_indices].mean(axis=1)
            bootstrap_values = cluster_values[bootstrap_indices].mean(axis=1)
            ci_low, ci_high = np.quantile(bootstrap_values, [alpha, 1.0 - alpha]).tolist()
            probability_positive = float(np.mean(bootstrap_values > 0.0))
            probability_negative = float(np.mean(bootstrap_values < 0.0))
            no_edge_psnr = mean([float(row["no_edge_raw_psnr_db"]) for row in rows])
            edge_psnr = mean([float(row["edge_raw_psnr_db"]) for row in rows])
            output.append(
                {
                    "split": split_input.split,
                    "level": level,
                    "snr_db": snr_value,
                    "num_clusters": len(samples),
                    "num_rows": len(rows),
                    "no_edge_raw_psnr_db": no_edge_psnr,
                    "edge_raw_psnr_db": edge_psnr,
                    "edge_minus_no_edge_raw_psnr_db": edge_psnr - no_edge_psnr,
                    "ci_low_db": float(ci_low),
                    "ci_high_db": float(ci_high),
                    "bootstrap_standard_error_db": float(np.std(bootstrap_values, ddof=1)),
                    "cluster_standard_deviation_db": float(np.std(cluster_values, ddof=1)),
                    "probability_effect_gt_zero": probability_positive,
                    "bootstrap_two_sided_p": float(
                        min(1.0, 2.0 * min(probability_positive, probability_negative))
                    ),
                    "ci_excludes_zero": bool(ci_low > 0.0 or ci_high < 0.0),
                    "positive_snr_count": positive_snr_count if level == "all" else "",
                    "negative_snr_count": negative_snr_count if level == "all" else "",
                    "zero_snr_count": zero_snr_count if level == "all" else "",
                    "all_five_snrs_same_direction": same_direction != "mixed" if level == "all" else "",
                    "five_snr_direction": same_direction if level == "all" else "",
                    "bootstrap_replicates": replicates,
                    "bootstrap_seed": seed,
                    "confidence_level": confidence_level,
                    **summarize_semantics(rows),
                }
            )
    return output


def png_manifest(paths: set[str], registry: dict[str, dict[str, Any]]) -> dict[str, Any]:
    digest = hashlib.sha256()
    total_bytes = 0
    for path in sorted(paths):
        item = registry[path]
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
        return f"{value:.{digits}f}"
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
    config: dict[str, Any], summary: list[dict[str, Any]], validation: dict[str, Any]
) -> str:
    overall = [row for row in summary if row["level"] == "all"]
    per_snr = [row for row in summary if row["level"] == "snr"]
    lines = [
        "# Matched Large Edge Cross-Split Paired Audit",
        "",
        "This audit compares the capacity/training-budget matched large edge (`EXP-S4-008`) and no-edge (`EXP-S4-009`) refiners. "
        "PSNR is recomputed from original/refined PNG pairs. Each split uses a paired bootstrap over sample IDs, retaining all five SNR rows per sampled cluster.",
        "",
        "## Bottom Line",
        "",
    ]
    for row in overall:
        lines.append(
            f"- {row['split']}: edge − no-edge raw PSNR `{signed(row['edge_minus_no_edge_raw_psnr_db'])}` dB "
            f"(95% CI `{signed(row['ci_low_db'])}`, `{signed(row['ci_high_db'])}`); "
            f"SNR directions `{row['positive_snr_count']} positive / {row['negative_snr_count']} negative / {row['zero_snr_count']} zero`. "
            f"AlexNet-pseudo failure delta `{signed(row['pseudo_edge_minus_no_edge_failure_rate'])}`, "
            f"new-error delta `{row['pseudo_edge_minus_no_edge_new_error_count']}`, repair delta `{row['pseudo_edge_minus_no_edge_repair_count']}`."
        )
    lines.extend(["", "## Split Summary", ""])
    lines += markdown_table(
        overall,
        [
            ("split", "Split"),
            ("num_clusters", "Samples"),
            ("edge_minus_no_edge_raw_psnr_db", "Edge−NoEdge dB"),
            ("ci_low_db", "CI Low"),
            ("ci_high_db", "CI High"),
            ("positive_snr_count", "+SNR"),
            ("negative_snr_count", "−SNR"),
            ("five_snr_direction", "5-SNR Direction"),
            ("pseudo_edge_minus_no_edge_failure_rate", "Pseudo Failure Δ"),
            ("pseudo_edge_minus_no_edge_new_error_count", "Pseudo New Error Δ"),
            ("pseudo_edge_minus_no_edge_repair_count", "Pseudo Repair Δ"),
        ],
    )
    lines.extend(["", "## Per-SNR Paired Effects", ""])
    lines += markdown_table(
        per_snr,
        [
            ("split", "Split"),
            ("snr_db", "SNR"),
            ("edge_minus_no_edge_raw_psnr_db", "Edge−NoEdge dB"),
            ("ci_low_db", "CI Low"),
            ("ci_high_db", "CI High"),
            ("ci_excludes_zero", "CI Excludes 0"),
            ("pseudo_edge_minus_no_edge_failure_rate", "Pseudo Failure Δ"),
        ],
    )
    lines.extend(
        [
            "",
            "## Validation and Scope",
            "",
            "- The two base refiners are identical in split, seed, width, depth, epochs, loss, channel settings, and residual gates; they differ only in the two receiver-visible structural input channels and resulting first-layer parameters.",
            f"- PNG recomputation maximum source-summary discrepancy: `{validation['max_abs_summary_psnr_difference_db']:.8f}` dB; maximum available row-level discrepancy: `{validation['max_abs_row_psnr_difference_db']:.8f}` dB.",
            f"- Bootstrap uses `{config['bootstrap']['replicates_per_split']}` replicates per split with seed `{config['bootstrap']['seed']}`. Validation/test-like/holdout results are not pooled.",
            "- Failure/new-error/repair statistics use frozen source AlexNet top-1 on COCO and are pseudo-label diagnostics, not supervised semantic truth.",
            "- These splits belong to the same COCO val export family. The audit tests downstream sample transfer, not dataset-level external validity.",
            "",
            "## Files",
            "",
            "- `per_sample.csv`: paired PNG-level PSNR, pseudo semantic indicators, paths, and SHA256 values.",
            "- `split_summary.csv`: all-SNR and per-SNR paired effects with sample-cluster bootstrap intervals.",
            "- `metadata.json`: source/config/checkpoint/PNG/output fingerprints and strict matching checks.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    config = load_yaml(config_path)
    if args.bootstrap_replicates is not None:
        config["bootstrap"]["replicates_per_split"] = int(args.bootstrap_replicates)
    torch.set_num_threads(int(config.get("runtime", {}).get("torch_num_threads", 1)))
    snrs = [float(item) for item in config["snrs"]]
    no_edge = load_base_arm("no_edge", config["matched_arms"]["no_edge"])
    edge = load_base_arm("edge", config["matched_arms"]["edge"])
    base_validation = validate_base_pair(
        no_edge, edge, [str(item) for item in config.get("edge_features", ["sobel_magnitude", "laplacian_abs"])]
    )
    split_inputs = [load_split(split_cfg, snrs) for split_cfg in config["splits"]]
    split_names = [item.split for item in split_inputs]
    if split_names != ["validation", "held-out", "test-like", "fresh-holdout"]:
        raise RuntimeError(f"Unexpected split order/names: {split_names}")

    dry_payload = {
        "status": "ok",
        "analysis_id": config["analysis_id"],
        "config": file_fingerprint(config_path),
        "base_pair_validation": base_validation,
        "bootstrap": config["bootstrap"],
        "splits": {
            item.split: {
                "samples": item.expected_sample_count,
                "rows": len(item.no_edge_rows),
                "input_files": item.input_files,
            }
            for item in split_inputs
        },
    }
    if args.dry_run:
        print(json.dumps(dry_payload, indent=2, ensure_ascii=False, sort_keys=True))
        return

    output_dir = resolve_project_path(args.output_dir or config["outputs"]["output_dir"])
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory exists; use --overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    per_sample, png_registry, paths_by_split, row_check = compute_rows(split_inputs, snrs)
    source_validation = validate_source_summaries(
        per_sample,
        split_inputs,
        float(config["validation"]["summary_psnr_tolerance_db"]),
        row_check,
    )
    summary = build_split_summary(
        per_sample,
        split_inputs,
        snrs,
        int(config["bootstrap"]["replicates_per_split"]),
        int(config["bootstrap"]["seed"]),
        float(config["bootstrap"]["confidence_level"]),
    )

    per_sample_path = output_dir / "per_sample.csv"
    summary_path = output_dir / "split_summary.csv"
    report_path = output_dir / "REPORT.md"
    metadata_path = output_dir / "metadata.json"
    write_csv(per_sample_path, per_sample)
    write_csv(summary_path, summary)
    report_path.write_text(make_report(config, summary, source_validation), encoding="utf-8")

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
        "base_pair_validation": base_validation,
        "base_arms": {
            "no_edge": {
                "experiment_id": no_edge.experiment_id,
                "parameter_count": no_edge.parameter_count,
                "best_epoch": no_edge.best_epoch,
                "checkpoint_eval_stats": no_edge.checkpoint.get("eval_stats", {}),
                "inputs": no_edge.inputs,
            },
            "edge": {
                "experiment_id": edge.experiment_id,
                "parameter_count": edge.parameter_count,
                "best_epoch": edge.best_epoch,
                "checkpoint_eval_stats": edge.checkpoint.get("eval_stats", {}),
                "inputs": edge.inputs,
            },
        },
        "splits": {
            item.split: {
                "expected_sample_count": item.expected_sample_count,
                "num_rows": len(item.no_edge_rows),
                "matched_config_outside_edge_metadata": True,
                "identical_sample_snr_keys": True,
                "identical_original_m0_paths": True,
                "identical_original_m0_alexnet_predictions": True,
                "input_files": item.input_files,
                "png_inputs": png_manifest(paths_by_split[item.split], png_registry),
            }
            for item in split_inputs
        },
        "source_psnr_validation": source_validation,
        "bootstrap": {
            **config["bootstrap"],
            "implementation": "independent per-split paired percentile bootstrap over sample IDs; each sampled cluster retains all five SNRs",
        },
        "all_unique_png_inputs": png_manifest(set(png_registry), png_registry),
        "outputs": {
            "per_sample_csv": file_fingerprint(per_sample_path),
            "split_summary_csv": file_fingerprint(summary_path),
            "report_md": file_fingerprint(report_path),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "pillow": getattr(Image, "__version__", "unknown"),
            "platform": platform.platform(),
            "proxy_environment_present": sorted(key for key in os.environ if "proxy" in key.lower()),
        },
        "notes": [
            "PSNR is recomputed from 8-bit saved PNGs with float32 tensors in [0,1].",
            "AlexNet failure/new-error/repair fields are pseudo-label diagnostics on COCO, not supervised semantic truth.",
            "No model inference, classifier inference, training, diffusion, or download is performed.",
        ],
    }
    save_json(metadata_path, metadata)
    print(
        json.dumps(
            {
                "output_dir": project_relative(output_dir),
                "per_sample_rows": len(per_sample),
                "split_summary_rows": len(summary),
                "bootstrap_replicates_per_split": int(config["bootstrap"]["replicates_per_split"]),
                "max_summary_psnr_difference_db": source_validation[
                    "max_abs_summary_psnr_difference_db"
                ],
                "max_row_psnr_difference_db": source_validation["max_abs_row_psnr_difference_db"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
