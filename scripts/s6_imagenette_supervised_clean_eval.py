from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import tarfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cadsd_jscc.deepjscc_adapter import build_deepjscc_model, extract_deepjscc_state_dict  # noqa: E402
from cadsd_jscc.metrics import ms_ssim_per_sample, psnr_per_sample, ssim_per_sample  # noqa: E402
from s5_residual_refiner_pilot import build_model as build_refiner  # noqa: E402
from s5_residual_refiner_pilot import residual_gate  # noqa: E402


REQUIRED_ARMS = (
    "M0",
    "no_edge_scheduled",
    "M2_edge_scheduled",
    "M3_scratch_gate_fallback",
)
FINAL_SPLIT_ALIASES = {"official_val", "final", "val"}
TRAIN_SPLIT_ALIASES = {"policy_dev", "dev", "policy"}
SHA256_HEX_LENGTH = 64
GIT_COMMIT_HEX_LENGTH = 40
ANALYSIS_OUTPUT_ROOT = (PROJECT_ROOT / "outputs" / "analysis").resolve()
EVALUATOR_SCRIPT = Path(__file__).resolve()
DEEPJSCC_ADAPTER_SOURCE = (PROJECT_ROOT / "src" / "cadsd_jscc" / "deepjscc_adapter.py").resolve()
METRICS_SOURCE = (PROJECT_ROOT / "src" / "cadsd_jscc" / "metrics.py").resolve()
REFINER_SOURCE = (PROJECT_ROOT / "scripts" / "s5_residual_refiner_pilot.py").resolve()

PREREGISTERED_CONSTANTS: dict[str, Any] = {
    "analysis_id": "imagenette_supervised_clean_correct_20260710",
    "method": "ScratchClassifierSupervisedCleanCorrectAudit",
    "dataset": "Imagenette2-320",
    "seed": 20260710,
    "archive_size_bytes": 341663724,
    "archive_md5": "3df6f0d01a2c9592104656642f5e78a3",
    "train_image_count": 9469,
    "val_image_count": 3925,
    "image_size": 256,
    "classes": [
        "n01440764",
        "n02102040",
        "n02979186",
        "n03000684",
        "n03028079",
        "n03394916",
        "n03417042",
        "n03425413",
        "n03445777",
        "n03888257",
    ],
    "split_method": "stratified_sha256_relative_path",
    "split_seed": 20260710,
    "split_ratios": {"cls_train": 0.70, "cls_cal": 0.10, "policy_dev": 0.20},
    "manifest_algorithm": "per_wnid_sort_sha256_seed_colon_relative_path_then_largest_remainder_v1",
    "channel_type": "AWGN",
    "snrs": [1.0, 4.0, 7.0, 13.0, 19.0],
    "policy_dev_seeds": [20260710],
    "final_seeds": [20260711, 20260712, 20260713],
    "primary_clean_threshold": 0.50,
    "clean_thresholds": [0.0, 0.5, 0.7],
    "primary_snrs": [1.0, 4.0, 7.0],
    "bootstrap_replicates": 10000,
    "bootstrap_seed": 161803,
    "schedule_alphas": {1.0: 0.75, 4.0: 0.75, 7.0: 0.75, 13.0: 1.0, 19.0: 0.75},
    "classifiers": {
        "G_gate": {
            "role": "receiver_gate",
            "architecture": "mobilenet_v3_small",
            "seed": 271828,
            "min_cal_macro_top1": 0.80,
        },
        "T_cls": {
            "role": "primary_independent_evaluator",
            "architecture": "resnet18",
            "seed": 314159,
            "min_cal_macro_top1": 0.85,
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the frozen DeepJSCC + residual-refiner pipeline on Imagenette with independent "
            "scratch-trained semantic classifiers and true 10-way labels."
        )
    )
    parser.add_argument("--config", default="configs/s6_imagenette_supervised_clean_eval.yaml")
    parser.add_argument("--split", default="policy_dev", help="policy_dev or official_val (aliases: dev/final/val)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--unlock-final", action="store_true")
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def require_analysis_output_path(path: Path, label: str) -> Path:
    """Resolve an output path and reject paths outside the project's analysis tree."""

    resolved = path.expanduser().resolve()
    if resolved == ANALYSIS_OUTPUT_ROOT or ANALYSIS_OUTPUT_ROOT not in resolved.parents:
        raise ValueError(
            f"{label} must be a strict descendant of {ANALYSIS_OUTPUT_ROOT}, got {resolved}"
        )
    return resolved


def logical_output_path(path: Path, physical_root: Path, logical_root: Path) -> Path:
    relative = path.resolve().relative_to(physical_root.resolve())
    return logical_root / relative


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    return device


def normalize_split_name(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in TRAIN_SPLIT_ALIASES:
        return "policy_dev"
    if normalized in FINAL_SPLIT_ALIASES:
        return "official_val"
    raise ValueError(f"Unsupported split {value!r}; expected policy_dev or official_val")


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_stream(handle: Any, chunk_size: int = 4 * 1024 * 1024) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while True:
        block = handle.read(chunk_size)
        if not block:
            break
        digest.update(block)
        size += len(block)
    return digest.hexdigest(), size


def md5_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.md5()  # noqa: S324 - required solely to verify the dataset publisher's MD5.
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def protocol_sha256(config: dict[str, Any]) -> str:
    """Hash immutable protocol fields while allowing the final_lock block itself to be populated."""

    payload = copy.deepcopy(config)
    payload.pop("final_lock", None)
    return canonical_json_sha256(payload)


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)


def serialize_csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
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


def mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    if not materialized:
        return None
    return float(sum(materialized) / len(materialized))


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return f"{value:.{digits}f}"
    return str(value)


def get_git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL, text=True
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except Exception as exc:  # noqa: BLE001
        return {"commit": "N/A (not a project git repo)", "dirty": None, "error": str(exc)}


def package_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import torchvision

        versions["torchvision"] = torchvision.__version__
    except Exception as exc:  # noqa: BLE001
        versions["torchvision"] = f"unavailable: {exc}"
    try:
        import pytorch_msssim

        versions["pytorch_msssim"] = getattr(pytorch_msssim, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        versions["pytorch_msssim"] = f"unavailable: {exc}"
    return versions


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derived_channel_seed(channel_seed: int, snr: float, batch_start: int) -> int:
    material = f"imagenette-supervised-awgn|{channel_seed}|{snr:.8f}|{batch_start}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(material).digest()[:8], byteorder="big")
    return value % (2**31 - 1)


def torch_load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint must be a mapping: {path} ({type(payload)!r})")
    return payload


def state_dict_from_checkpoint(checkpoint: dict[str, Any], path: Path) -> dict[str, torch.Tensor]:
    for key in ("model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and value and all(torch.is_tensor(item) for item in value.values()):
            state = dict(value)
            if state and all(name.startswith("module.") for name in state):
                state = {name.removeprefix("module."): tensor for name, tensor in state.items()}
            return state
    raise KeyError(f"No model_state_dict/state_dict/model tensor mapping in checkpoint: {path}")


def nested_checkpoint_value(checkpoint: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in checkpoint:
        return checkpoint[key]
    metadata = checkpoint.get("metadata")
    if isinstance(metadata, dict) and key in metadata:
        return metadata[key]
    return default


def normalize_class_mapping(mapping: Any) -> dict[str, int]:
    if not isinstance(mapping, dict):
        raise TypeError(f"class_to_idx must be a mapping, got {type(mapping)!r}")
    return {str(key): int(value) for key, value in mapping.items()}


def load_scratch_classifier(
    checkpoint_path: Path,
    role_name: str,
    role_config: dict[str, Any],
    expected_classes: list[str],
    manifest_sha256: str,
    expected_protocol_sha256: str,
    device: torch.device,
) -> tuple[torch.nn.Module, float, dict[str, Any]]:
    checkpoint = torch_load_checkpoint(checkpoint_path)
    exact_contract = {
        "format_version": 1,
        "role": role_name,
        "role_description": str(role_config["role"]),
        "architecture": str(role_config["architecture"]),
        "num_classes": len(expected_classes),
        "seed": int(role_config["seed"]),
        "weights": None,
        "pretrained": False,
        "random_initialization": True,
        "training_split": "cls_train",
        "selection_split": "cls_cal",
        "temperature_scaling_split": "cls_cal",
        "policy_dev_manifest_hash_only": True,
        "policy_dev_used_for_training_selection_or_calibration": False,
        "official_val_accessed": False,
    }
    contract_mismatches = {
        key: {"checkpoint": checkpoint.get(key), "expected": expected}
        for key, expected in exact_contract.items()
        if checkpoint.get(key) != expected
    }
    if contract_mismatches:
        raise RuntimeError(
            f"{role_name} checkpoint violates the scratch/split contract: {contract_mismatches}"
        )
    if Path(str(checkpoint.get("split_manifest_path", ""))) != Path(
        project_relative(resolve_project_path("outputs/analysis/imagenette_scratch_classifiers/split_manifest.json"))
    ):
        raise RuntimeError(f"{role_name} checkpoint records a non-canonical split manifest path")
    checkpoint_training_script = resolve_project_path(str(checkpoint.get("training_script", "")))
    checkpoint_training_script_sha = str(checkpoint.get("training_script_sha256", ""))
    if (
        len(checkpoint_training_script_sha) != SHA256_HEX_LENGTH
        or not checkpoint_training_script.is_file()
        or sha256_file(checkpoint_training_script) != checkpoint_training_script_sha
    ):
        raise RuntimeError(f"{role_name} checkpoint training-script snapshot/hash is invalid")
    checkpoint_config_sha = str(checkpoint.get("config_sha256", ""))
    if (
        len(checkpoint_config_sha) != SHA256_HEX_LENGTH
        or checkpoint.get("config_hash") != checkpoint_config_sha
    ):
        raise RuntimeError(f"{role_name} checkpoint config hash contract is invalid")
    architecture = str(nested_checkpoint_value(checkpoint, "architecture", ""))
    expected_architecture = str(role_config["architecture"])
    if architecture != expected_architecture:
        raise RuntimeError(
            f"{role_name} architecture mismatch: checkpoint={architecture!r}, config={expected_architecture!r}"
        )
    checkpoint_role = str(nested_checkpoint_value(checkpoint, "role", ""))
    if checkpoint_role != role_name:
        raise RuntimeError(
            f"{role_name} checkpoint role mismatch: checkpoint={checkpoint_role!r}, expected={role_name!r}"
        )
    expected_description = str(role_config.get("role", ""))
    role_description = str(nested_checkpoint_value(checkpoint, "role_description", ""))
    if expected_description and role_description != expected_description:
        raise RuntimeError(
            f"{role_name} role_description mismatch: checkpoint={role_description!r}, "
            f"config={expected_description!r}"
        )
    if "pretrained" not in checkpoint or checkpoint["pretrained"] is not False:
        raise RuntimeError(f"{role_name} checkpoint must explicitly assert pretrained=false")
    if "weights" not in checkpoint or checkpoint["weights"] is not None:
        raise RuntimeError(f"{role_name} checkpoint must explicitly assert weights=null")
    pretrained = bool(checkpoint["pretrained"])
    weights = checkpoint["weights"]
    if pretrained or weights is not None:
        raise RuntimeError(f"{role_name} must be scratch-trained (pretrained=False, weights=None)")
    if nested_checkpoint_value(checkpoint, "quality_gate_passed", None) is not True:
        raise RuntimeError(f"{role_name} classifier quality gate did not pass")
    if nested_checkpoint_value(checkpoint, "official_val_accessed", None) is not False:
        raise RuntimeError(f"{role_name} checkpoint does not prove official_val_accessed=false")
    checkpoint_protocol_sha = str(nested_checkpoint_value(checkpoint, "protocol_sha256", ""))
    if checkpoint_protocol_sha != expected_protocol_sha256:
        raise RuntimeError(
            f"{role_name} protocol hash mismatch: checkpoint={checkpoint_protocol_sha}, "
            f"current={expected_protocol_sha256}"
        )
    num_classes = int(nested_checkpoint_value(checkpoint, "num_classes", len(expected_classes)))
    if num_classes != len(expected_classes):
        raise RuntimeError(f"{role_name} num_classes={num_classes}, expected {len(expected_classes)}")
    class_to_idx = normalize_class_mapping(nested_checkpoint_value(checkpoint, "class_to_idx", {}))
    expected_mapping = {wnid: index for index, wnid in enumerate(expected_classes)}
    if class_to_idx != expected_mapping:
        raise RuntimeError(
            f"{role_name} class mapping mismatch: checkpoint={class_to_idx}, expected={expected_mapping}"
        )
    if [str(item) for item in checkpoint.get("idx_to_class", [])] != expected_classes:
        raise RuntimeError(f"{role_name} idx_to_class does not match the preregistered WNID order")
    checkpoint_manifest_sha = str(nested_checkpoint_value(checkpoint, "split_manifest_sha256", ""))
    if checkpoint_manifest_sha != manifest_sha256:
        raise RuntimeError(
            f"{role_name} manifest hash mismatch: checkpoint={checkpoint_manifest_sha}, actual={manifest_sha256}"
        )
    temperature = float(nested_checkpoint_value(checkpoint, "temperature", 0.0))
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise RuntimeError(f"{role_name} has invalid calibrated temperature: {temperature}")
    cal_macro_top1 = float(nested_checkpoint_value(checkpoint, "best_cls_cal_macro_top1", -1.0))
    minimum = float(role_config.get("min_cal_macro_top1", 0.0))
    recorded_minimum = float(nested_checkpoint_value(checkpoint, "min_cal_macro_top1", -1.0))
    if not math.isclose(recorded_minimum, minimum, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            f"{role_name} checkpoint minimum quality gate changed: "
            f"checkpoint={recorded_minimum}, preregistered={minimum}"
        )
    if cal_macro_top1 < minimum:
        raise RuntimeError(
            f"{role_name} cls_cal macro top-1 {cal_macro_top1:.4f} is below required {minimum:.4f}"
        )

    import torchvision.models as models

    constructors: dict[str, Callable[..., torch.nn.Module]] = {
        "mobilenet_v3_small": models.mobilenet_v3_small,
        "resnet18": models.resnet18,
    }
    if architecture not in constructors:
        raise ValueError(f"Unsupported scratch classifier architecture: {architecture}")
    model = constructors[architecture](weights=None, num_classes=num_classes)
    model.load_state_dict(state_dict_from_checkpoint(checkpoint, checkpoint_path), strict=True)
    model.to(device).eval().requires_grad_(False)
    metadata = {
        "role_name": role_name,
        "role": checkpoint_role,
        "role_description": role_description,
        "architecture": architecture,
        "num_classes": num_classes,
        "temperature": temperature,
        "best_cls_cal_macro_top1": cal_macro_top1,
        "seed": nested_checkpoint_value(checkpoint, "seed"),
        "best_epoch": nested_checkpoint_value(checkpoint, "best_epoch"),
        "checkpoint": project_relative(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "split_manifest_sha256": checkpoint_manifest_sha,
        "protocol_sha256": checkpoint_protocol_sha,
        "quality_gate_passed": True,
        "official_val_accessed": False,
        "pretrained": pretrained,
        "weights": weights,
        "random_initialization": True,
        "training_split": "cls_train",
        "selection_split": "cls_cal",
        "temperature_scaling_split": "cls_cal",
        "policy_dev_used_for_training_selection_or_calibration": False,
        "training_script": project_relative(checkpoint_training_script),
        "training_script_sha256": checkpoint_training_script_sha,
    }
    return model, temperature, metadata


def critical_model_config(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get("model")
    if not isinstance(model, dict):
        raise KeyError("Refiner config is missing model mapping")
    keys = (
        "input_channels",
        "condition_features",
        "base_channels",
        "num_blocks",
        "snr_norm_max",
        "residual_gates",
    )
    return {key: model.get(key) for key in keys}


def load_refiner_model(
    checkpoint_path: Path,
    source_config_path: Path,
    expected_edge_conditioned: bool,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    checkpoint = torch_load_checkpoint(checkpoint_path)
    embedded_config = checkpoint.get("config")
    if not isinstance(embedded_config, dict):
        raise RuntimeError(f"Refiner checkpoint has no embedded config: {checkpoint_path}")
    with source_config_path.open("r", encoding="utf-8") as handle:
        source_config = yaml.safe_load(handle)
    if critical_model_config(embedded_config) != critical_model_config(source_config):
        raise RuntimeError(
            f"Refiner embedded config does not match source config: {checkpoint_path} vs {source_config_path}"
        )
    features = list(embedded_config["model"].get("condition_features", []))
    if expected_edge_conditioned and not features:
        raise RuntimeError(f"Edge refiner has no structural condition features: {checkpoint_path}")
    if not expected_edge_conditioned and features:
        raise RuntimeError(f"No-edge control unexpectedly has condition features: {features}")
    model = build_refiner(embedded_config)
    model.load_state_dict(state_dict_from_checkpoint(checkpoint, checkpoint_path), strict=True)
    model.to(device).eval().requires_grad_(False)
    metadata = {
        "checkpoint": project_relative(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "source_config": project_relative(source_config_path),
        "source_config_sha256": sha256_file(source_config_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "model": embedded_config["model"],
    }
    return model, embedded_config, metadata


def load_deepjscc(
    config_path: Path,
    checkpoint_path: Path,
    channel_type: str,
    initial_snr: float,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    with config_path.open("r", encoding="utf-8") as handle:
        model_config = yaml.safe_load(handle)
    checkpoint = torch_load_checkpoint(checkpoint_path)
    embedded_config = checkpoint.get("config")
    if isinstance(embedded_config, dict):
        for key in ("inner_channel", "channel", "cbr", "image_size"):
            if key in embedded_config and key in model_config and embedded_config[key] != model_config[key]:
                raise RuntimeError(
                    f"DeepJSCC config mismatch for {key}: checkpoint={embedded_config[key]!r}, "
                    f"source={model_config[key]!r}"
                )
    configured_channel = str(model_config["channel"])
    if configured_channel.lower() != channel_type.lower():
        raise RuntimeError(
            f"DeepJSCC channel mismatch: source config={configured_channel}, evaluation={channel_type}"
        )
    model = build_deepjscc_model(
        repo_root=resolve_project_path(model_config["baseline"]["repo"]),
        inner_channel=int(model_config["inner_channel"]),
        channel=configured_channel,
        snr=float(initial_snr),
    )
    model.load_state_dict(extract_deepjscc_state_dict(checkpoint), strict=True)
    model.to(device).eval().requires_grad_(False)
    metadata = {
        "checkpoint": project_relative(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "source_config": project_relative(config_path),
        "source_config_sha256": sha256_file(config_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "inner_channel": int(model_config["inner_channel"]),
        "channel": configured_channel,
        "cbr": float(model_config["cbr"]),
        "third_party_repo": project_relative(resolve_project_path(model_config["baseline"]["repo"])),
        "third_party_commit": model_config["baseline"].get("commit"),
    }
    return model, model_config, metadata


def snr_key(snr: float) -> str:
    return str(int(snr)) if float(snr).is_integer() else str(float(snr))


def lookup_numeric_key(mapping: dict[str, Any], snr: float) -> Any:
    candidates = (str(float(snr)), snr_key(snr), f"{float(snr):.1f}")
    for key in candidates:
        if key in mapping:
            return mapping[key]
    raise KeyError(f"No value for SNR {snr}; tried {candidates}")


def load_frozen_schedule(
    schedule_path: Path,
    schedule_key: str,
    snrs: list[float],
    edge_config: dict[str, Any],
    no_edge_config: dict[str, Any],
) -> tuple[dict[float, float], dict[str, Any], dict[str, Any]]:
    with schedule_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    values = payload.get(schedule_key)
    if not isinstance(values, dict):
        raise KeyError(f"Schedule key {schedule_key!r} missing from {schedule_path}")
    alphas: dict[float, float] = {}
    effective_strengths: list[tuple[float, float]] = []
    for snr in snrs:
        alpha = float(lookup_numeric_key(values, snr))
        if not 0.0 <= alpha <= 1.0:
            raise RuntimeError(f"Frozen alpha outside [0,1] at {snr} dB: {alpha}")
        edge_gate = float(residual_gate(edge_config, snr))
        no_edge_gate = float(residual_gate(no_edge_config, snr))
        if not math.isclose(edge_gate, no_edge_gate, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(
                f"Matched edge/no-edge residual gates differ at {snr} dB: {edge_gate} vs {no_edge_gate}"
            )
        schedule_gates = payload.get("residual_gates")
        if isinstance(schedule_gates, dict):
            recorded = float(lookup_numeric_key(schedule_gates, snr))
            if not math.isclose(recorded, edge_gate, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(
                    f"Schedule residual gate mismatch at {snr} dB: schedule={recorded}, checkpoint={edge_gate}"
                )
        alphas[float(snr)] = alpha
        effective_strengths.append((float(snr), alpha * edge_gate))
    expected_alphas = PREREGISTERED_CONSTANTS["schedule_alphas"]
    if alphas != expected_alphas:
        raise RuntimeError(f"Frozen schedule differs from preregistration: {alphas} vs {expected_alphas}")
    ordered = sorted(effective_strengths)
    for (left_snr, left), (right_snr, right) in zip(ordered, ordered[1:]):
        if left + 1e-12 < right:
            raise RuntimeError(
                "Frozen effective strength must be non-increasing with SNR: "
                f"{left_snr} dB={left}, {right_snr} dB={right}"
            )
    metadata = {
        "path": project_relative(schedule_path),
        "sha256": sha256_file(schedule_path),
        "key": schedule_key,
        "alphas_by_snr": {snr_key(key): value for key, value in alphas.items()},
        "effective_strength_by_snr": {
            snr_key(snr): strength for snr, strength in effective_strengths
        },
    }
    return alphas, payload, metadata


def candidate_record_lists(manifest: dict[str, Any], split_name: str) -> list[dict[str, Any]]:
    aliases = [split_name]
    if split_name == "official_val":
        aliases += ["val", "final", "official_validation"]
    records: list[dict[str, Any]] = []
    samples = manifest.get("samples")
    if isinstance(samples, list):
        records.extend(item for item in samples if isinstance(item, dict))
    for key in ("official_val_samples", "val_samples", "final_samples"):
        values = manifest.get(key)
        if isinstance(values, list):
            for item in values:
                if isinstance(item, dict):
                    enriched = dict(item)
                    enriched.setdefault("split", "official_val")
                    records.append(enriched)
    splits = manifest.get("splits")
    if isinstance(splits, dict):
        for alias in aliases:
            values = splits.get(alias)
            if isinstance(values, list):
                for item in values:
                    if isinstance(item, dict):
                        enriched = dict(item)
                        enriched.setdefault("split", split_name)
                        records.append(enriched)
    for alias in aliases:
        values = manifest.get(alias)
        if isinstance(values, list):
            for item in values:
                if isinstance(item, dict):
                    enriched = dict(item)
                    enriched.setdefault("split", split_name)
                    records.append(enriched)
    selected = []
    for item in records:
        item_split = str(item.get("split", ""))
        if item_split in aliases or (split_name == "official_val" and item_split in FINAL_SPLIT_ALIASES):
            selected.append(item)
    unique: dict[str, dict[str, Any]] = {}
    for item in selected:
        identity = str(
            item.get("sample_id")
            or item.get("image_id")
            or item.get("relative_path")
            or item.get("path")
            or ""
        )
        if not identity:
            raise RuntimeError(f"Manifest sample has no identity field: {item}")
        if identity in unique and unique[identity] != item:
            raise RuntimeError(f"Conflicting duplicate manifest record: {identity}")
        unique[identity] = item
    return list(unique.values())


def resolve_record_path(record: dict[str, Any], config: dict[str, Any], split_name: str) -> Path:
    raw = record.get("path") or record.get("image_path") or record.get("relative_path")
    if raw is None:
        raise KeyError(f"Manifest record has no path: {record}")
    path = Path(str(raw))
    if path.is_absolute():
        return path
    root = resolve_project_path(config["data"]["root"])
    split_root = resolve_project_path(
        config["data"]["val_dir"] if split_name == "official_val" else config["data"]["train_dir"]
    )
    candidates = [PROJECT_ROOT / path, root / path, split_root / path]
    if path.parts and path.parts[0] in {"train", "val"}:
        candidates.append(root / Path(*path.parts[1:]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[-1]


def validate_training_manifest_contract(manifest: dict[str, Any], config: dict[str, Any]) -> None:
    expected = PREREGISTERED_CONSTANTS
    if manifest.get("format_version") != 1:
        raise RuntimeError(f"Unsupported split-manifest format_version: {manifest.get('format_version')!r}")
    if manifest.get("algorithm") != expected["manifest_algorithm"]:
        raise RuntimeError(f"Split-manifest algorithm changed: {manifest.get('algorithm')!r}")
    if manifest.get("official_val_accessed") is not False:
        raise RuntimeError("Training split manifest does not assert official_val_accessed=false")
    if int(manifest.get("split_seed", -1)) != int(expected["split_seed"]):
        raise RuntimeError("Training split-manifest seed changed")
    ratios = {str(key): float(value) for key, value in dict(manifest.get("ratios", {})).items()}
    if ratios != expected["split_ratios"]:
        raise RuntimeError(f"Training split-manifest ratios changed: {ratios}")
    classes = [str(item) for item in manifest.get("classes", [])]
    if classes != expected["classes"] or classes != [str(item) for item in config["data"]["classes"]]:
        raise RuntimeError("Training split-manifest WNID order changed")
    expected_mapping = {wnid: index for index, wnid in enumerate(classes)}
    if normalize_class_mapping(manifest.get("class_to_idx", {})) != expected_mapping:
        raise RuntimeError("Training split-manifest class_to_idx changed")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != int(expected["train_image_count"]):
        raise RuntimeError("Training split-manifest sample count changed")
    allowed_splits = set(expected["split_ratios"])
    actual_counts: dict[str, int] = defaultdict(int)
    identities: set[str] = set()
    relative_paths: set[str] = set()
    content_hashes: set[str] = set()
    for row in samples:
        if not isinstance(row, dict):
            raise RuntimeError("Training split manifest contains a non-mapping sample")
        split = str(row.get("split", ""))
        if split not in allowed_splits:
            raise RuntimeError(f"Training split manifest contains forbidden split {split!r}")
        actual_counts[split] += 1
        identity = str(row.get("sample_id", ""))
        relative_path = str(row.get("relative_path", ""))
        content_sha = str(row.get("content_sha256", ""))
        if not identity or identity in identities:
            raise RuntimeError(f"Duplicate/empty sample_id in training manifest: {identity!r}")
        if not relative_path or relative_path in relative_paths:
            raise RuntimeError(f"Duplicate/empty relative_path in training manifest: {relative_path!r}")
        if len(content_sha) != SHA256_HEX_LENGTH or content_sha in content_hashes:
            raise RuntimeError(f"Duplicate/invalid content SHA in training manifest: {identity}")
        identities.add(identity)
        relative_paths.add(relative_path)
        content_hashes.add(content_sha)
    recorded_counts = {str(key): int(value) for key, value in dict(manifest.get("split_counts", {})).items()}
    if recorded_counts != dict(actual_counts):
        raise RuntimeError(
            f"Training split-manifest recorded/actual counts differ: {recorded_counts} vs {dict(actual_counts)}"
        )


def load_manifest_records(
    manifest_path: Path,
    config: dict[str, Any],
    split_name: str,
    verify_content: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_training_manifest_contract(manifest, config)
    samples = candidate_record_lists(manifest, split_name)
    if not samples:
        raise RuntimeError(f"Manifest contains no {split_name} samples: {manifest_path}")
    classes = [str(item) for item in config["data"]["classes"]]
    class_to_idx = {wnid: index for index, wnid in enumerate(classes)}
    manifest_mapping = manifest.get("class_to_idx")
    if manifest_mapping is None and isinstance(manifest.get("classes"), dict):
        manifest_mapping = manifest["classes"].get("class_to_idx")
    if manifest_mapping is not None and normalize_class_mapping(manifest_mapping) != class_to_idx:
        raise RuntimeError("Manifest class_to_idx does not match the preregistered WNID order")

    records: list[dict[str, Any]] = []
    identities: set[str] = set()
    selected_content_hashes: set[str] = set()
    for raw in samples:
        path = resolve_record_path(raw, config, split_name)
        if not path.is_file():
            raise FileNotFoundError(f"Manifest image is missing: {path}")
        wnid = str(raw.get("wnid") or raw.get("class") or path.parent.name)
        if wnid not in class_to_idx:
            raise RuntimeError(f"Unknown Imagenette WNID {wnid!r} in {path}")
        label = int(raw.get("class_idx", raw.get("label", class_to_idx[wnid])))
        if label != class_to_idx[wnid]:
            raise RuntimeError(f"Manifest label/WNID mismatch for {path}: label={label}, wnid={wnid}")
        image_id = str(raw.get("sample_id") or raw.get("image_id") or raw.get("relative_path") or path.name)
        if image_id in identities:
            raise RuntimeError(f"Duplicate image_id in selected manifest split: {image_id}")
        identities.add(image_id)
        expected_content_sha = str(raw.get("content_sha256", ""))
        if bool(config["split"].get("require_content_sha256", False)) and len(expected_content_sha) != 64:
            raise RuntimeError(f"Missing required content_sha256 for {image_id}")
        actual_content_sha = expected_content_sha
        if verify_content:
            actual_content_sha = sha256_file(path)
            if expected_content_sha and actual_content_sha != expected_content_sha:
                raise RuntimeError(
                    f"Content SHA256 mismatch for {image_id}: manifest={expected_content_sha}, actual={actual_content_sha}"
                )
        if actual_content_sha:
            if actual_content_sha in selected_content_hashes:
                raise RuntimeError(f"Exact duplicate image content inside {split_name}: {image_id}")
            selected_content_hashes.add(actual_content_sha)
        records.append(
            {
                "image_id": image_id,
                "relative_path": str(raw.get("relative_path") or project_relative(path)),
                "path": path,
                "wnid": wnid,
                "true_label": label,
                "content_sha256": actual_content_sha,
                "size_bytes": int(raw.get("size_bytes", path.stat().st_size)),
            }
        )

    if bool(config["split"].get("reject_exact_content_duplicates", False)):
        all_records: list[dict[str, Any]] = []
        manifest_samples = manifest.get("samples")
        if isinstance(manifest_samples, list):
            all_records.extend(item for item in manifest_samples if isinstance(item, dict))
        for key in ("official_val_samples", "val_samples", "final_samples"):
            values = manifest.get(key)
            if isinstance(values, list):
                all_records.extend(item for item in values if isinstance(item, dict))
        owners: dict[str, set[str]] = defaultdict(set)
        for raw in all_records:
            content_sha = str(raw.get("content_sha256", ""))
            item_split = str(raw.get("split", "official_val" if raw in manifest.get("official_val_samples", []) else ""))
            if content_sha:
                owners[content_sha].add(item_split)
        cross_split = {digest: values for digest, values in owners.items() if len(values) > 1}
        if cross_split:
            preview = list(cross_split.items())[:5]
            raise RuntimeError(f"Exact image content appears across manifest splits: {preview}")

    records.sort(key=lambda row: row["image_id"])
    manifest_metadata = {
        "path": project_relative(manifest_path),
        "sha256": sha256_file(manifest_path),
        "format_version": manifest.get("format_version"),
        "algorithm": manifest.get("algorithm"),
        "selected_split": split_name,
        "num_selected_images": len(records),
        "content_hashes_verified": verify_content,
    }
    return records, manifest, manifest_metadata


def build_sealed_official_val_manifest(
    training_manifest_path: Path,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Scan official val only after the explicit final lock has been validated.

    The classifier-training manifest intentionally excludes official val.  This
    separate manifest is therefore constructed by the final evaluator and is
    never consumed by classifier training, calibration, checkpoint selection,
    gate selection, or alpha selection.
    """

    archive = resolve_project_path(config["data"]["archive"])
    if not archive.is_file():
        raise FileNotFoundError(f"Imagenette archive is missing: {archive}")
    expected_size = int(config["data"]["archive_size_bytes"])
    if archive.stat().st_size != expected_size:
        raise RuntimeError(
            f"Imagenette archive size mismatch: actual={archive.stat().st_size}, expected={expected_size}"
        )
    actual_md5 = md5_file(archive)
    expected_md5 = str(config["data"]["archive_md5"]).lower()
    if actual_md5.lower() != expected_md5:
        raise RuntimeError(f"Imagenette archive MD5 mismatch: actual={actual_md5}, expected={expected_md5}")

    with training_manifest_path.open("r", encoding="utf-8") as handle:
        training_manifest = json.load(handle)
    validate_training_manifest_contract(training_manifest, config)
    train_hashes = {
        str(item.get("content_sha256", ""))
        for item in training_manifest.get("samples", [])
        if isinstance(item, dict) and item.get("content_sha256")
    }
    classes = [str(item) for item in config["data"]["classes"]]
    class_to_idx = {wnid: index for index, wnid in enumerate(classes)}
    val_root = resolve_project_path(config["data"]["val_dir"])
    if not val_root.is_dir():
        raise FileNotFoundError(f"Official Imagenette val directory is missing: {val_root}")
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    archive_root_name = Path(str(config["data"]["root"])).name
    archive_members: dict[str, dict[str, Any]] = {}
    with tarfile.open(archive, mode="r:gz") as tar_handle:
        for member in tar_handle:
            parts = tuple(part for part in Path(member.name).parts if part not in {"", "."})
            if ".." in parts or Path(member.name).is_absolute():
                raise RuntimeError(f"Unsafe member path in verified Imagenette archive: {member.name!r}")
            if len(parts) < 4 or parts[0:2] != (archive_root_name, "val"):
                continue
            wnid = parts[2]
            relative_path = Path(*parts[2:]).as_posix()
            if wnid not in class_to_idx or Path(relative_path).suffix.lower() not in extensions:
                continue
            if not member.isfile():
                raise RuntimeError(f"Official val archive image member is not a regular file: {member.name}")
            if relative_path in archive_members:
                raise RuntimeError(f"Duplicate official val member in verified archive: {relative_path}")
            extracted = tar_handle.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Could not read official val archive member: {member.name}")
            member_sha256, member_size = sha256_stream(extracted)
            if member_size != int(member.size):
                raise RuntimeError(
                    f"Official val archive member size changed while reading: {member.name}"
                )
            archive_members[relative_path] = {
                "archive_member": member.name,
                "content_sha256": member_sha256,
                "size_bytes": member_size,
            }
    expected_count = int(config["data"]["val_image_count"])
    if len(archive_members) != expected_count:
        raise RuntimeError(
            f"Official val archive-member count mismatch: actual={len(archive_members)}, expected={expected_count}"
        )

    samples: list[dict[str, Any]] = []
    seen_hashes: dict[str, str] = {}
    per_class_counts: dict[str, int] = {}
    for wnid in classes:
        class_dir = val_root / wnid
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Official val WNID directory is missing: {class_dir}")
        paths = sorted(
            path for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in extensions
        )
        per_class_counts[wnid] = len(paths)
        for path in paths:
            relative_path = path.relative_to(val_root).as_posix()
            content_sha = sha256_file(path)
            archive_member = archive_members.get(relative_path)
            if archive_member is None:
                raise RuntimeError(
                    f"Extracted official val image has no corresponding member in verified archive: {relative_path}"
                )
            if (
                int(archive_member["size_bytes"]) != path.stat().st_size
                or str(archive_member["content_sha256"]) != content_sha
            ):
                raise RuntimeError(
                    f"Extracted official val bytes differ from verified archive member: {relative_path}"
                )
            if content_sha in train_hashes:
                raise RuntimeError(f"Official val has an exact-content duplicate in official train: {relative_path}")
            if content_sha in seen_hashes:
                raise RuntimeError(
                    f"Exact duplicate inside official val: {relative_path} and {seen_hashes[content_sha]}"
                )
            seen_hashes[content_sha] = relative_path
            samples.append(
                {
                    "sample_id": f"official_val/{relative_path}",
                    "relative_path": relative_path,
                    "wnid": wnid,
                    "class_idx": class_to_idx[wnid],
                    "split": "official_val",
                    "path_sha256": hashlib.sha256(relative_path.encode("utf-8")).hexdigest(),
                    "content_sha256": content_sha,
                    "size_bytes": path.stat().st_size,
                    "archive_member": archive_member["archive_member"],
                    "archive_member_bytes_verified": True,
                }
            )
    samples.sort(key=lambda row: row["sample_id"])
    if len(samples) != expected_count:
        raise RuntimeError(f"Official val image count mismatch: actual={len(samples)}, expected={expected_count}")
    extracted_paths = {str(row["relative_path"]) for row in samples}
    if extracted_paths != set(archive_members):
        missing = sorted(set(archive_members) - extracted_paths)[:5]
        extra = sorted(extracted_paths - set(archive_members))[:5]
        raise RuntimeError(
            f"Extracted/archive official val membership differs: missing={missing}, extra={extra}"
        )
    payload = {
        "format_version": 1,
        "role": "sealed_official_val_final_test",
        "created_by": project_relative(Path(__file__)),
        "source_val_root": project_relative(val_root),
        "training_split_manifest": project_relative(training_manifest_path),
        "training_split_manifest_sha256": sha256_file(training_manifest_path),
        "classifier_training_official_val_accessed": False,
        "archive": project_relative(archive),
        "archive_size_bytes": archive.stat().st_size,
        "archive_md5": actual_md5,
        "archive_member_bytes_verified": True,
        "archive_val_member_count": len(archive_members),
        "archive_val_member_manifest_sha256": canonical_json_sha256(archive_members),
        "classes": classes,
        "class_to_idx": class_to_idx,
        "sample_count": len(samples),
        "per_class_counts": per_class_counts,
        "cross_train_exact_content_duplicates": 0,
        "within_val_exact_content_duplicates": 0,
        "samples": samples,
    }
    payload_sha = canonical_json_sha256(payload)
    records = [
        {
            "image_id": str(item["sample_id"]),
            "relative_path": str(item["relative_path"]),
            "path": val_root / str(item["relative_path"]),
            "wnid": str(item["wnid"]),
            "true_label": int(item["class_idx"]),
            "content_sha256": str(item["content_sha256"]),
            "size_bytes": int(item["size_bytes"]),
        }
        for item in samples
    ]
    metadata = {
        "role": payload["role"],
        "canonical_payload_sha256": payload_sha,
        "training_split_manifest": project_relative(training_manifest_path),
        "training_split_manifest_sha256": payload["training_split_manifest_sha256"],
        "num_selected_images": len(records),
        "per_class_counts": per_class_counts,
        "content_hashes_verified": True,
        "archive_md5_verified": True,
        "archive_member_bytes_verified": True,
        "archive_val_member_count": len(archive_members),
        "archive_val_member_manifest_sha256": payload["archive_val_member_manifest_sha256"],
    }
    return records, payload, metadata


class ManifestImageDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], image_size: int) -> None:
        self.records = records
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path = self.records[index]["path"]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, index


def make_loader(
    dataset: Dataset,
    config: dict[str, Any],
    device: torch.device,
) -> DataLoader:
    evaluation = config["evaluation"]
    return DataLoader(
        dataset,
        batch_size=int(evaluation["batch_size"]),
        shuffle=False,
        num_workers=int(evaluation["num_workers"]),
        pin_memory=bool(evaluation.get("pin_memory", True)) and device.type == "cuda",
        drop_last=False,
        persistent_workers=int(evaluation["num_workers"]) > 0,
    )


def normalize_classifier_input(images: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    training = config.get("training", {})
    mean_values = training.get("normalization_mean", [0.485, 0.456, 0.406])
    std_values = training.get("normalization_std", [0.229, 0.224, 0.225])
    mean_tensor = torch.tensor(mean_values, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std_values, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return (images - mean_tensor) / std_tensor


@torch.no_grad()
def predict_calibrated(
    model: torch.nn.Module,
    images: torch.Tensor,
    temperature: float,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = model(normalize_classifier_input(images, config))
    probabilities = torch.softmax(logits.float() / temperature, dim=1)
    confidence, prediction = probabilities.max(dim=1)
    return prediction, confidence


def quantize_png_tensor(images: torch.Tensor, enabled: bool) -> torch.Tensor:
    images = images.clamp(0.0, 1.0)
    if not enabled:
        return images
    return torch.round(images * 255.0) / 255.0


def gate_tensor_for_snr(
    refiner_config: dict[str, Any], snr: float, batch_size: int, device: torch.device
) -> torch.Tensor:
    value = float(residual_gate(refiner_config, snr))
    return torch.full((batch_size,), value, dtype=torch.float32, device=device)


def try_load_lpips(config: dict[str, Any], device: torch.device, skip: bool) -> tuple[Any, str | None]:
    required = bool(config["success_criteria"].get("require_negative_lpips_delta", False))
    if skip:
        if required:
            raise RuntimeError("LPIPS is a jointly required endpoint; --skip-lpips is fail-closed")
        return None, "disabled by --skip-lpips"
    if not bool(config["evaluation"].get("lpips", True)):
        if required:
            raise RuntimeError("LPIPS is a jointly required endpoint; evaluation.lpips=false is forbidden")
        return None, "disabled by evaluation.lpips=false"
    cache_dir = resolve_project_path(config["evaluation"].get("lpips_cache_dir", "outputs/cache/torch"))
    os.environ.setdefault("TORCH_HOME", str(cache_dir))
    try:
        import lpips

        model = lpips.LPIPS(net=str(config["evaluation"].get("lpips_net", "alex")), verbose=False)
        model.to(device).eval().requires_grad_(False)
        return model, None
    except Exception as exc:  # noqa: BLE001
        message = f"{type(exc).__name__}: {exc}"
        if required:
            raise RuntimeError(f"Required LPIPS endpoint could not be loaded: {message}") from exc
        return None, message


@torch.no_grad()
def quality_per_sample(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    lpips_model: Any,
) -> dict[str, torch.Tensor | None]:
    output: dict[str, torch.Tensor | None] = {
        "psnr_db": psnr_per_sample(candidate, reference).float(),
        "ssim": ssim_per_sample(candidate, reference).float(),
        "ms_ssim": ms_ssim_per_sample(candidate, reference).float(),
        "lpips": None,
    }
    if lpips_model is not None:
        values = lpips_model(candidate * 2.0 - 1.0, reference * 2.0 - 1.0)
        output["lpips"] = values.flatten().float()
    return output


@torch.no_grad()
def classify_originals(
    loader: DataLoader,
    evaluator: torch.nn.Module,
    evaluator_temperature: float,
    config: dict[str, Any],
    device: torch.device,
) -> dict[int, dict[str, float | int]]:
    predictions: dict[int, dict[str, float | int]] = {}
    for images_cpu, indices in loader:
        images = images_cpu.to(device, non_blocking=True)
        pred, confidence = predict_calibrated(evaluator, images, evaluator_temperature, config)
        for local, dataset_index in enumerate(indices.tolist()):
            predictions[int(dataset_index)] = {
                "prediction": int(pred[local].item()),
                "confidence": float(confidence[local].item()),
            }
    if len(predictions) != len(loader.dataset):
        raise RuntimeError(
            f"Original classification count mismatch: got {len(predictions)}, expected {len(loader.dataset)}"
        )
    return predictions


def row_clean_at(row: dict[str, Any], threshold: float) -> bool:
    return bool(row["original_correct"]) and float(row["original_tcls_confidence"]) >= float(threshold)


def sanitize_filename(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    return "".join(character if character in allowed else "_" for character in value)


def maybe_save_event_galleries(
    output_dir: Path,
    logical_output_dir: Path,
    gallery_counts: dict[str, int],
    gallery_index: list[dict[str, Any]],
    limit: int,
    event_flags: dict[str, bool],
    row: dict[str, Any],
    original: torch.Tensor,
    m0: torch.Tensor,
    candidate: torch.Tensor,
    final: torch.Tensor,
) -> None:
    if limit <= 0 or not bool(row["clean_primary"]):
        return
    for event, active in event_flags.items():
        if not active or gallery_counts[event] >= limit:
            continue
        event_dir = output_dir / "gallery" / event
        event_dir.mkdir(parents=True, exist_ok=True)
        stem = sanitize_filename(
            f"{row['image_id']}__seed_{row['channel_seed']}__snr_{float(row['snr_db']):g}db"
        )
        path = event_dir / f"{stem}.png"
        grid = torch.stack([original, m0, candidate, final]).detach().cpu()
        save_image(grid, path, nrow=4, padding=2)
        gallery_counts[event] += 1
        gallery_index.append(
            {
                "event": event,
                "path": project_relative(logical_output_path(path, output_dir, logical_output_dir)),
                "image_id": row["image_id"],
                "wnid": row["wnid"],
                "channel_seed": row["channel_seed"],
                "snr_db": row["snr_db"],
                "columns": ["original", "M0", "M2_edge_scheduled", "M3_scratch_gate_fallback"],
            }
        )


@torch.no_grad()
def evaluate_pipeline(
    config: dict[str, Any],
    split_name: str,
    records: list[dict[str, Any]],
    loader: DataLoader,
    original_predictions: dict[int, dict[str, float | int]],
    deepjscc: torch.nn.Module,
    deepjscc_config: dict[str, Any],
    edge_refiner: torch.nn.Module,
    edge_config: dict[str, Any],
    no_edge_refiner: torch.nn.Module,
    no_edge_config: dict[str, Any],
    gate_model: torch.nn.Module,
    gate_temperature: float,
    evaluator: torch.nn.Module,
    evaluator_temperature: float,
    alphas: dict[float, float],
    channel_seeds: list[int],
    snrs: list[float],
    lpips_model: Any,
    output_dir: Path,
    logical_output_dir: Path,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    quantize = bool(config["evaluation"].get("quantize_png", True))
    thresholds = [float(item) for item in config["evaluation"]["clean_thresholds"]]
    primary_threshold = float(config["evaluation"]["primary_clean_threshold"])
    gallery_limit = int(config["evaluation"].get("save_gallery_per_event", 0))
    gallery_counts: dict[str, int] = defaultdict(int)
    gallery_index: list[dict[str, Any]] = []
    (output_dir / "gallery").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    timings: dict[str, float] = defaultdict(float)
    channel_type = str(config["channel"]["type"])

    per_sample_path = output_dir / "per_sample.csv"
    csv_handle = per_sample_path.open("w", encoding="utf-8", newline="")
    csv_writer: csv.DictWriter | None = None
    try:
        for channel_seed in channel_seeds:
            for snr in snrs:
                deepjscc.change_channel(str(deepjscc_config["channel"]), float(snr))
                alpha = float(alphas[float(snr)])
                batch_start = 0
                for images_cpu, indices in loader:
                    images = images_cpu.to(device, non_blocking=True)
                    batch_size = int(images.shape[0])
                    call_seed = derived_channel_seed(int(channel_seed), float(snr), batch_start)
                    seed_everything(call_seed)

                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    m0 = quantize_png_tensor(deepjscc(images), quantize)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    timings["deepjscc_seconds"] += time.perf_counter() - started

                    snr_norm_edge = torch.full(
                        (batch_size,),
                        float(snr) / float(edge_config["model"]["snr_norm_max"]),
                        dtype=torch.float32,
                        device=device,
                    )
                    edge_gate = gate_tensor_for_snr(edge_config, snr, batch_size, device)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    edge_raw = quantize_png_tensor(edge_refiner(m0, snr_norm_edge, edge_gate), quantize)
                    edge_candidate = quantize_png_tensor(m0 + alpha * (edge_raw - m0), quantize)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    timings["edge_refiner_seconds"] += time.perf_counter() - started

                    snr_norm_no_edge = torch.full(
                        (batch_size,),
                        float(snr) / float(no_edge_config["model"]["snr_norm_max"]),
                        dtype=torch.float32,
                        device=device,
                    )
                    no_edge_gate = gate_tensor_for_snr(no_edge_config, snr, batch_size, device)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    no_edge_raw = quantize_png_tensor(
                        no_edge_refiner(m0, snr_norm_no_edge, no_edge_gate), quantize
                    )
                    no_edge_candidate = quantize_png_tensor(m0 + alpha * (no_edge_raw - m0), quantize)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    timings["no_edge_refiner_seconds"] += time.perf_counter() - started

                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    gate_m0_pred, gate_m0_conf = predict_calibrated(
                        gate_model, m0, gate_temperature, config
                    )
                    gate_candidate_pred, gate_candidate_conf = predict_calibrated(
                        gate_model, edge_candidate, gate_temperature, config
                    )
                    gate_accept = gate_m0_pred.eq(gate_candidate_pred)
                    final = torch.where(gate_accept.view(-1, 1, 1, 1), edge_candidate, m0)
                    final = quantize_png_tensor(final, quantize)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    timings["gate_seconds"] += time.perf_counter() - started

                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    tcls_m0_pred, tcls_m0_conf = predict_calibrated(
                        evaluator, m0, evaluator_temperature, config
                    )
                    tcls_no_edge_pred, tcls_no_edge_conf = predict_calibrated(
                        evaluator, no_edge_candidate, evaluator_temperature, config
                    )
                    tcls_edge_pred, tcls_edge_conf = predict_calibrated(
                        evaluator, edge_candidate, evaluator_temperature, config
                    )
                    tcls_final_pred = torch.where(gate_accept, tcls_edge_pred, tcls_m0_pred)
                    tcls_final_conf = torch.where(gate_accept, tcls_edge_conf, tcls_m0_conf)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    timings["evaluator_seconds"] += time.perf_counter() - started

                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    metrics = {
                        "M0": quality_per_sample(images, m0, lpips_model),
                        "no_edge_scheduled": quality_per_sample(images, no_edge_candidate, lpips_model),
                        "M2_edge_scheduled": quality_per_sample(images, edge_candidate, lpips_model),
                        "M3_scratch_gate_fallback": quality_per_sample(images, final, lpips_model),
                    }
                    edge_raw_psnr = psnr_per_sample(edge_raw, images).float()
                    no_edge_raw_psnr = psnr_per_sample(no_edge_raw, images).float()
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    timings["quality_metrics_seconds"] += time.perf_counter() - started

                    for local_index, dataset_index_raw in enumerate(indices.tolist()):
                        dataset_index = int(dataset_index_raw)
                        record = records[dataset_index]
                        original = original_predictions[dataset_index]
                        true_label = int(record["true_label"])
                        original_pred = int(original["prediction"])
                        original_conf = float(original["confidence"])
                        original_correct = original_pred == true_label
                        m0_pred = int(tcls_m0_pred[local_index].item())
                        no_edge_pred = int(tcls_no_edge_pred[local_index].item())
                        edge_pred = int(tcls_edge_pred[local_index].item())
                        final_pred = int(tcls_final_pred[local_index].item())
                        accept = bool(gate_accept[local_index].item())
                        m0_correct = m0_pred == true_label
                        edge_correct = edge_pred == true_label
                        accepted_new_error = accept and m0_correct and not edge_correct
                        accepted_repair = accept and (not m0_correct) and edge_correct
                        protective_reject = (not accept) and m0_correct and (not edge_correct)
                        missed_repair = (not accept) and (not m0_correct) and edge_correct
                        row: dict[str, Any] = {
                            "split": split_name,
                            "image_id": record["image_id"],
                            "relative_path": record["relative_path"],
                            "wnid": record["wnid"],
                            "true_label": true_label,
                            "channel_type": channel_type,
                            "channel_seed": int(channel_seed),
                            "channel_call_seed": call_seed,
                            "snr_db": float(snr),
                            "alpha": alpha,
                            "edge_residual_gate": float(residual_gate(edge_config, snr)),
                            "effective_edge_strength": alpha * float(residual_gate(edge_config, snr)),
                            "original_tcls_prediction": original_pred,
                            "original_tcls_confidence": original_conf,
                            "original_correct": original_correct,
                            "clean_primary": original_correct and original_conf >= primary_threshold,
                            "gate_m0_prediction": int(gate_m0_pred[local_index].item()),
                            "gate_m0_confidence": float(gate_m0_conf[local_index].item()),
                            "gate_candidate_prediction": int(gate_candidate_pred[local_index].item()),
                            "gate_candidate_confidence": float(gate_candidate_conf[local_index].item()),
                            "gate_accept": accept,
                            "M0_tcls_prediction": m0_pred,
                            "M0_tcls_confidence": float(tcls_m0_conf[local_index].item()),
                            "M0_correct": m0_correct,
                            "M0_failure": not m0_correct,
                            "no_edge_scheduled_tcls_prediction": no_edge_pred,
                            "no_edge_scheduled_tcls_confidence": float(
                                tcls_no_edge_conf[local_index].item()
                            ),
                            "no_edge_scheduled_correct": no_edge_pred == true_label,
                            "no_edge_scheduled_failure": no_edge_pred != true_label,
                            "M2_edge_scheduled_tcls_prediction": edge_pred,
                            "M2_edge_scheduled_tcls_confidence": float(tcls_edge_conf[local_index].item()),
                            "M2_edge_scheduled_correct": edge_correct,
                            "M2_edge_scheduled_failure": not edge_correct,
                            "M3_scratch_gate_fallback_tcls_prediction": final_pred,
                            "M3_scratch_gate_fallback_tcls_confidence": float(
                                tcls_final_conf[local_index].item()
                            ),
                            "M3_scratch_gate_fallback_correct": final_pred == true_label,
                            "M3_scratch_gate_fallback_failure": final_pred != true_label,
                            "M2_new_error_vs_M0": m0_correct and not edge_correct,
                            "M2_repair_vs_M0": (not m0_correct) and edge_correct,
                            "M3_accepted_new_error": accepted_new_error,
                            "M3_accepted_repair": accepted_repair,
                            "M3_protective_reject": protective_reject,
                            "M3_missed_repair": missed_repair,
                            "edge_raw_psnr_db": float(edge_raw_psnr[local_index].item()),
                            "no_edge_raw_psnr_db": float(no_edge_raw_psnr[local_index].item()),
                        }
                        for threshold in thresholds:
                            key = str(threshold).replace(".", "p")
                            row[f"clean_tau_{key}"] = original_correct and original_conf >= threshold
                        for arm in REQUIRED_ARMS:
                            for metric_name, values in metrics[arm].items():
                                row[f"{arm}_{metric_name}"] = (
                                    None if values is None else float(values[local_index].item())
                                )
                        event_flags = {
                            "accepted_new_error": accepted_new_error,
                            "accepted_repair": accepted_repair,
                            "protective_reject": protective_reject,
                            "missed_repair": missed_repair,
                        }
                        maybe_save_event_galleries(
                            output_dir=output_dir,
                            logical_output_dir=logical_output_dir,
                            gallery_counts=gallery_counts,
                            gallery_index=gallery_index,
                            limit=gallery_limit,
                            event_flags=event_flags,
                            row=row,
                            original=images[local_index],
                            m0=m0[local_index],
                            candidate=edge_candidate[local_index],
                            final=final[local_index],
                        )
                        if csv_writer is None:
                            csv_writer = csv.DictWriter(csv_handle, fieldnames=list(row.keys()))
                            csv_writer.writeheader()
                        csv_writer.writerow(
                            {key: serialize_csv_value(value) for key, value in row.items()}
                        )
                        rows.append(row)
                    batch_start += batch_size
    finally:
        csv_handle.close()
    save_json(output_dir / "gallery" / "index.json", gallery_index)
    return rows, {
        "timings_seconds": dict(timings),
        "gallery_counts": dict(gallery_counts),
        "gallery_index": project_relative(logical_output_dir / "gallery" / "index.json"),
        "per_sample_csv": project_relative(logical_output_dir / "per_sample.csv"),
        "num_rows": len(rows),
    }


def assert_complete_unique_pipeline_rows(
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    snrs: list[float],
    channel_seeds: list[int],
    require_lpips: bool,
) -> dict[str, Any]:
    image_ids = {str(record["image_id"]) for record in records}
    expected = {
        (image_id, float(snr), int(seed))
        for image_id in image_ids
        for snr in snrs
        for seed in channel_seeds
    }
    observed: set[tuple[str, float, int]] = set()
    duplicates: list[tuple[str, float, int]] = []
    for row in rows:
        key = (str(row["image_id"]), float(row["snr_db"]), int(row["channel_seed"]))
        if key in observed:
            duplicates.append(key)
        observed.add(key)
        if require_lpips:
            missing_lpips = [arm for arm in REQUIRED_ARMS if row.get(f"{arm}_lpips") is None]
            if missing_lpips:
                raise RuntimeError(f"Required LPIPS values are missing for {key}: {missing_lpips}")
    missing = expected - observed
    extra = observed - expected
    if duplicates or missing or extra or len(rows) != len(expected):
        raise RuntimeError(
            "Pipeline row grid is incomplete or non-unique: "
            f"rows={len(rows)}, expected={len(expected)}, duplicates={duplicates[:5]}, "
            f"missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}"
        )
    return {
        "assertion": "exactly one row per image_id x SNR x channel_seed",
        "passed": True,
        "num_unique_images": len(image_ids),
        "num_snrs": len(snrs),
        "num_channel_seeds": len(channel_seeds),
        "expected_rows": len(expected),
        "observed_rows": len(rows),
        "lpips_complete_for_all_arms": require_lpips,
    }


def clean_image_ids(
    records: list[dict[str, Any]],
    original_predictions: dict[int, dict[str, float | int]],
    threshold: float,
) -> set[str]:
    selected: set[str] = set()
    for index, record in enumerate(records):
        prediction = original_predictions[index]
        if (
            int(prediction["prediction"]) == int(record["true_label"])
            and float(prediction["confidence"]) >= threshold
        ):
            selected.add(str(record["image_id"]))
    return selected


def build_clean_coverage_rows(
    records: list[dict[str, Any]],
    original_predictions: dict[int, dict[str, float | int]],
    thresholds: list[float],
    classes: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        total_by_class: dict[int, int] = defaultdict(int)
        clean_by_class: dict[int, int] = defaultdict(int)
        for index, record in enumerate(records):
            label = int(record["true_label"])
            total_by_class[label] += 1
            prediction = original_predictions[index]
            if int(prediction["prediction"]) == label and float(prediction["confidence"]) >= threshold:
                clean_by_class[label] += 1
        total = sum(total_by_class.values())
        clean = sum(clean_by_class.values())
        rows.append(
            {
                "row_type": "clean_coverage",
                "clean_threshold": threshold,
                "class_index": "all",
                "wnid": "all",
                "num_images": total,
                "clean_count": clean,
                "clean_coverage": clean / total if total else 0.0,
            }
        )
        for class_index, wnid in enumerate(classes):
            class_total = total_by_class[class_index]
            class_clean = clean_by_class[class_index]
            rows.append(
                {
                    "row_type": "clean_coverage",
                    "clean_threshold": threshold,
                    "class_index": class_index,
                    "wnid": wnid,
                    "num_images": class_total,
                    "clean_count": class_clean,
                    "clean_coverage": class_clean / class_total if class_total else 0.0,
                }
            )
    return rows


def scoped_rows(
    rows: list[dict[str, Any]], image_ids: set[str], snrs: set[float]
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if str(row["image_id"]) in image_ids and float(row["snr_db"]) in snrs
    ]


def arm_summary_row(
    selected: list[dict[str, Any]],
    arm: str,
    population: str,
    clean_threshold: float | None,
    snr_scope: str,
) -> dict[str, Any]:
    if not selected:
        return {
            "row_type": "arm_summary",
            "population": population,
            "clean_threshold": clean_threshold,
            "snr_scope": snr_scope,
            "arm": arm,
            "num_rows": 0,
            "num_image_clusters": 0,
        }
    m0_failure = [float(bool(row["M0_failure"])) for row in selected]
    m2_failure = [float(bool(row["M2_edge_scheduled_failure"])) for row in selected]
    arm_failure = [float(bool(row[f"{arm}_failure"])) for row in selected]
    psnr_values = [float(row[f"{arm}_psnr_db"]) for row in selected]
    m0_psnr = [float(row["M0_psnr_db"]) for row in selected]
    m2_psnr = [float(row["M2_edge_scheduled_psnr_db"]) for row in selected]
    ssim_values = [float(row[f"{arm}_ssim"]) for row in selected]
    ms_ssim_values = [float(row[f"{arm}_ms_ssim"]) for row in selected]
    lpips_values = [row[f"{arm}_lpips"] for row in selected]
    valid_lpips = [float(item) for item in lpips_values if item is not None]
    m0_lpips = [row["M0_lpips"] for row in selected]
    paired_lpips_delta = [
        float(value) - float(baseline)
        for value, baseline in zip(lpips_values, m0_lpips)
        if value is not None and baseline is not None
    ]
    summary = {
        "row_type": "arm_summary",
        "population": population,
        "clean_threshold": clean_threshold,
        "snr_scope": snr_scope,
        "arm": arm,
        "num_rows": len(selected),
        "num_image_clusters": len({str(row["image_id"]) for row in selected}),
        "num_channel_seeds": len({int(row["channel_seed"]) for row in selected}),
        "num_snrs": len({float(row["snr_db"]) for row in selected}),
        "failure_rate": mean(arm_failure),
        "delta_failure_vs_M0": mean(value - baseline for value, baseline in zip(arm_failure, m0_failure)),
        "delta_failure_vs_M2": mean(value - baseline for value, baseline in zip(arm_failure, m2_failure)),
        "psnr_db": mean(psnr_values),
        "delta_psnr_vs_M0_db": mean(value - baseline for value, baseline in zip(psnr_values, m0_psnr)),
        "delta_psnr_vs_M2_db": mean(value - baseline for value, baseline in zip(psnr_values, m2_psnr)),
        "ssim": mean(ssim_values),
        "ms_ssim": mean(ms_ssim_values),
        "lpips": mean(valid_lpips),
        "delta_lpips_vs_M0": mean(paired_lpips_delta),
        "accept_rate": mean(float(bool(row["gate_accept"])) for row in selected)
        if arm == "M3_scratch_gate_fallback"
        else (1.0 if arm in {"no_edge_scheduled", "M2_edge_scheduled"} else 0.0),
    }
    if arm == "M2_edge_scheduled":
        summary.update(
            {
                "new_error_count": sum(bool(row["M2_new_error_vs_M0"]) for row in selected),
                "repair_count": sum(bool(row["M2_repair_vs_M0"]) for row in selected),
                "protective_reject_count": 0,
                "missed_repair_count": 0,
            }
        )
    elif arm == "M3_scratch_gate_fallback":
        summary.update(
            {
                "new_error_count": sum(bool(row["M3_accepted_new_error"]) for row in selected),
                "repair_count": sum(bool(row["M3_accepted_repair"]) for row in selected),
                "protective_reject_count": sum(bool(row["M3_protective_reject"]) for row in selected),
                "missed_repair_count": sum(bool(row["M3_missed_repair"]) for row in selected),
            }
        )
    else:
        summary.update(
            {
                "new_error_count": None,
                "repair_count": None,
                "protective_reject_count": None,
                "missed_repair_count": None,
            }
        )
    for key in ("new_error", "repair", "protective_reject", "missed_repair"):
        count = summary[f"{key}_count"]
        summary[f"{key}_rate"] = None if count is None else float(count) / len(selected)
    return summary


def build_arm_summaries(
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    original_predictions: dict[int, dict[str, float | int]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    snrs = [float(item) for item in config["channel"]["snrs"]]
    primary_snrs = [float(item) for item in config["evaluation"]["primary_snrs"]]
    populations: list[tuple[str, float | None, set[str]]] = [
        ("all_images", None, {str(record["image_id"]) for record in records})
    ]
    for threshold in [float(item) for item in config["evaluation"]["clean_thresholds"]]:
        populations.append(
            (
                f"clean_correct_tau_{str(threshold).replace('.', 'p')}",
                threshold,
                clean_image_ids(records, original_predictions, threshold),
            )
        )
    scopes: list[tuple[str, set[float]]] = [(f"snr_{snr_key(snr)}", {snr}) for snr in snrs]
    scopes.extend([("primary_snrs", set(primary_snrs)), ("all_snrs", set(snrs))])
    summaries: list[dict[str, Any]] = []
    for population, threshold, image_ids in populations:
        for scope_name, scope_snrs in scopes:
            selected = scoped_rows(rows, image_ids, scope_snrs)
            for arm in REQUIRED_ARMS:
                summaries.append(
                    arm_summary_row(selected, arm, population, threshold, scope_name)
                )
    return summaries


def build_per_class_rows(
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    original_predictions: dict[int, dict[str, float | int]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    threshold = float(config["evaluation"]["primary_clean_threshold"])
    clean_ids = clean_image_ids(records, original_predictions, threshold)
    primary_snrs = {float(item) for item in config["evaluation"]["primary_snrs"]}
    classes = [str(item) for item in config["data"]["classes"]]
    result: list[dict[str, Any]] = []
    for class_index, wnid in enumerate(classes):
        class_ids = {
            str(record["image_id"])
            for record in records
            if int(record["true_label"]) == class_index and str(record["image_id"]) in clean_ids
        }
        selected = scoped_rows(rows, class_ids, primary_snrs)
        for arm in REQUIRED_ARMS:
            summary = arm_summary_row(
                selected,
                arm,
                "primary_clean_correct_per_class",
                threshold,
                "primary_snrs",
            )
            summary["row_type"] = "per_class_arm_summary"
            summary["class_index"] = class_index
            summary["wnid"] = wnid
            result.append(summary)
    return result


def cluster_values(
    rows: list[dict[str, Any]],
    image_ids: set[str],
    snrs: set[float],
    value_function: Callable[[dict[str, Any]], float | None],
) -> tuple[list[str], np.ndarray]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        image_id = str(row["image_id"])
        if image_id not in image_ids or float(row["snr_db"]) not in snrs:
            continue
        value = value_function(row)
        if value is not None and math.isfinite(float(value)):
            grouped[image_id].append(float(value))
    keys = sorted(grouped)
    values = np.asarray([np.mean(grouped[key], dtype=np.float64) for key in keys], dtype=np.float64)
    return keys, values


def bootstrap_mean_ci(
    values: np.ndarray,
    replicates: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    if values.ndim != 1 or len(values) == 0:
        return {
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
            "confidence": confidence,
            "replicates": replicates,
            "num_clusters": 0,
        }
    rng = np.random.default_rng(seed)
    sampled_means = np.empty(replicates, dtype=np.float64)
    chunk_size = 256
    for start in range(0, replicates, chunk_size):
        count = min(chunk_size, replicates - start)
        indices = rng.integers(0, len(values), size=(count, len(values)), endpoint=False)
        sampled_means[start : start + count] = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(sampled_means, [alpha, 1.0 - alpha])
    return {
        "estimate": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence": confidence,
        "replicates": int(replicates),
        "num_clusters": int(len(values)),
        "cluster_unit": "image_id",
    }


def bootstrap_ratio_ci(
    numerator: np.ndarray,
    denominator: np.ndarray,
    replicates: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    if numerator.shape != denominator.shape or numerator.ndim != 1 or len(numerator) == 0:
        raise ValueError("Ratio bootstrap arrays must be aligned, non-empty 1-D arrays")
    denominator_mean = float(denominator.mean())
    estimate = float(numerator.mean() / denominator_mean) if abs(denominator_mean) > 1e-12 else None
    rng = np.random.default_rng(seed)
    sampled = np.full(replicates, np.nan, dtype=np.float64)
    chunk_size = 256
    for start in range(0, replicates, chunk_size):
        count = min(chunk_size, replicates - start)
        indices = rng.integers(0, len(numerator), size=(count, len(numerator)), endpoint=False)
        num = numerator[indices].mean(axis=1)
        den = denominator[indices].mean(axis=1)
        valid = np.abs(den) > 1e-12
        sampled[start : start + count][valid] = num[valid] / den[valid]
    valid_sampled = sampled[np.isfinite(sampled)]
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(valid_sampled, [alpha, 1.0 - alpha]) if len(valid_sampled) else (None, None)
    return {
        "estimate": estimate,
        "ci_low": None if low is None else float(low),
        "ci_high": None if high is None else float(high),
        "confidence": confidence,
        "replicates": int(replicates),
        "valid_replicates": int(len(valid_sampled)),
        "num_clusters": int(len(numerator)),
        "cluster_unit": "image_id",
    }


def bootstrap_clustered_conditional_rate(
    rows: list[dict[str, Any]],
    image_ids: set[str],
    snrs: set[float],
    numerator_function: Callable[[dict[str, Any]], bool],
    denominator_function: Callable[[dict[str, Any]], bool],
    replicates: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """One-sided clustered upper bound plus image-level any-event exact bound."""

    if not math.isclose(confidence, 0.95, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("The preregistered accepted-new-error endpoint requires q=0.95")
    clustered: dict[str, list[int]] = {}
    for row in rows:
        image_id = str(row["image_id"])
        if image_id not in image_ids or float(row["snr_db"]) not in snrs:
            continue
        denominator = bool(denominator_function(row))
        numerator = bool(numerator_function(row))
        if numerator and not denominator:
            raise RuntimeError(
                f"Conditional endpoint numerator occurred outside its denominator for image {image_id}"
            )
        if denominator:
            counts_for_image = clustered.setdefault(image_id, [0, 0])
            counts_for_image[1] += 1
            if numerator:
                counts_for_image[0] += 1
    keys = sorted(clustered)
    counts = np.asarray([clustered[key] for key in keys], dtype=np.float64)
    numerator_total = float(counts[:, 0].sum()) if len(counts) else 0.0
    denominator_total = float(counts[:, 1].sum()) if len(counts) else 0.0
    estimate = numerator_total / denominator_total if denominator_total > 0.0 else None
    event_image_clusters = int(np.count_nonzero(counts[:, 0] > 0.0)) if len(counts) else 0
    eligible_image_clusters = int(len(counts))
    exact_any_event_upper = clopper_pearson_upper_95(
        event_image_clusters, eligible_image_clusters
    )
    if len(counts) == 0:
        return {
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
            "confidence": confidence,
            "replicates": int(replicates),
            "valid_replicates": 0,
            "num_clusters": 0,
            "numerator_events": 0,
            "denominator_rows": 0,
            "eligible_image_clusters": 0,
            "event_image_clusters": 0,
            "clustered_bootstrap_upper_q95": None,
            "image_cluster_any_event_rate": None,
            "image_cluster_any_event_clopper_pearson_upper_95": None,
            "conservative_upper_95": None,
            "cluster_unit": "image_id",
            "estimand": "row event rate conditional on M0-correct rows",
            "eligibility_definition": "image has at least one M0-correct row in the selected SNR scope",
            "event_image_definition": "eligible image has at least one accepted-new-error row",
        }
    rng = np.random.default_rng(seed)
    sampled_rates = np.full(replicates, np.nan, dtype=np.float64)
    chunk_size = 256
    for start in range(0, replicates, chunk_size):
        count = min(chunk_size, replicates - start)
        indices = rng.integers(0, len(counts), size=(count, len(counts)), endpoint=False)
        sampled_num = counts[indices, 0].sum(axis=1)
        sampled_den = counts[indices, 1].sum(axis=1)
        valid = sampled_den > 0.0
        sampled_rates[start : start + count][valid] = sampled_num[valid] / sampled_den[valid]
    valid_rates = sampled_rates[np.isfinite(sampled_rates)]
    bootstrap_upper = float(np.quantile(valid_rates, confidence)) if len(valid_rates) else None
    conservative_upper = (
        max(bootstrap_upper, exact_any_event_upper)
        if bootstrap_upper is not None and exact_any_event_upper is not None
        else None
    )
    return {
        "estimate": None if estimate is None else float(estimate),
        "ci_low": None,
        "ci_high": conservative_upper,
        "confidence": confidence,
        "one_sided_quantile": confidence,
        "replicates": int(replicates),
        "valid_replicates": int(len(valid_rates)),
        "num_clusters": int(len(counts)),
        "numerator_events": int(numerator_total),
        "denominator_rows": int(denominator_total),
        "eligible_image_clusters": eligible_image_clusters,
        "event_image_clusters": event_image_clusters,
        "clustered_bootstrap_upper_q95": bootstrap_upper,
        "image_cluster_any_event_rate": event_image_clusters / eligible_image_clusters,
        "image_cluster_any_event_clopper_pearson_upper_95": exact_any_event_upper,
        "conservative_upper_95": conservative_upper,
        "cluster_unit": "image_id",
        "estimand": "sum(accepted-new-error rows) / sum(M0-correct rows)",
        "eligibility_definition": "image has at least one M0-correct row in the selected SNR scope",
        "event_image_definition": "eligible image has at least one accepted-new-error row",
        "hard_gate_upper": (
            "max(clustered bootstrap one-sided q=0.95 upper, "
            "image-cluster any-event Clopper-Pearson one-sided 95% upper)"
        ),
    }


def clopper_pearson_upper_95(successes: int, trials: int) -> float | None:
    if trials <= 0:
        return None
    if successes < 0 or successes > trials:
        raise ValueError(f"Invalid binomial counts: successes={successes}, trials={trials}")
    from scipy.stats import beta

    if successes >= trials:
        return 1.0
    value = float(beta.ppf(0.95, successes + 1, trials - successes))
    if not math.isfinite(value):
        raise RuntimeError(
            f"Clopper-Pearson upper bound is non-finite for {successes}/{trials}"
        )
    return value


def exact_binomial_upper_95(successes: int, trials: int) -> float | None:
    return clopper_pearson_upper_95(successes, trials)


def build_bootstrap_results(
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    original_predictions: dict[int, dict[str, float | int]],
    config: dict[str, Any],
    lpips_available: bool,
) -> dict[str, Any]:
    replicates = int(config["evaluation"]["bootstrap_replicates"])
    base_seed = int(config["evaluation"]["bootstrap_seed"])
    primary_snrs = {float(item) for item in config["evaluation"]["primary_snrs"]}
    all_snrs = {float(item) for item in config["channel"]["snrs"]}
    all_ids = {str(record["image_id"]) for record in records}
    output: dict[str, Any] = {
        "method": "paired nonparametric cluster bootstrap over image_id; each cluster retains all SNR/seed rows",
        "replicates": replicates,
        "base_seed": base_seed,
        "primary_snrs": sorted(primary_snrs),
        "sensitivity_by_clean_threshold": {},
    }
    for threshold_index, threshold in enumerate(
        float(item) for item in config["evaluation"]["clean_thresholds"]
    ):
        clean_ids = clean_image_ids(records, original_predictions, threshold)
        keys_eff, efficacy_values = cluster_values(
            rows,
            clean_ids,
            primary_snrs,
            lambda row: float(bool(row["M3_scratch_gate_fallback_failure"]))
            - float(bool(row["M2_edge_scheduled_failure"])),
        )
        keys_safe, safety_values = cluster_values(
            rows,
            clean_ids,
            primary_snrs,
            lambda row: float(bool(row["M3_scratch_gate_fallback_failure"]))
            - float(bool(row["M0_failure"])),
        )
        if keys_eff != keys_safe:
            raise RuntimeError("Semantic bootstrap cluster alignment failure")
        output["sensitivity_by_clean_threshold"][str(threshold)] = {
            "clean_count": len(clean_ids),
            "gate_efficacy_M3_minus_M2_failure": bootstrap_mean_ci(
                efficacy_values, replicates, base_seed + 10 * threshold_index + 1
            ),
            "safety_M3_minus_M0_failure": bootstrap_mean_ci(
                safety_values, replicates, base_seed + 10 * threshold_index + 2
            ),
            "accepted_new_error_rate_conditional_on_M0_correct": bootstrap_clustered_conditional_rate(
                rows,
                clean_ids,
                primary_snrs,
                lambda row: bool(row["M3_accepted_new_error"]),
                lambda row: bool(row["M0_correct"]),
                replicates,
                base_seed + 10 * threshold_index + 3,
            ),
        }

    primary_threshold = float(config["evaluation"]["primary_clean_threshold"])
    primary_clean_ids = clean_image_ids(records, original_predictions, primary_threshold)
    primary_semantic = output["sensitivity_by_clean_threshold"][str(primary_threshold)]
    safety_by_snr: dict[str, Any] = {}
    for snr_index, snr in enumerate(sorted(primary_snrs)):
        _keys, values = cluster_values(
            rows,
            primary_clean_ids,
            {snr},
            lambda row: float(bool(row["M3_scratch_gate_fallback_failure"]))
            - float(bool(row["M0_failure"])),
        )
        safety_by_snr[snr_key(snr)] = bootstrap_mean_ci(
            values, replicates, base_seed + 500 + snr_index
        )
    primary_semantic["safety_M3_minus_M0_failure_by_primary_snr"] = safety_by_snr

    class_safety: dict[str, Any] = {}
    for class_index, wnid in enumerate(str(item) for item in config["data"]["classes"]):
        class_ids = {
            str(record["image_id"])
            for record in records
            if int(record["true_label"]) == class_index
            and str(record["image_id"]) in primary_clean_ids
        }
        _keys, values = cluster_values(
            rows,
            class_ids,
            primary_snrs,
            lambda row: float(bool(row["M3_scratch_gate_fallback_failure"]))
            - float(bool(row["M0_failure"])),
        )
        class_safety[wnid] = {
            "estimate": float(values.mean()) if len(values) else None,
            "num_clusters": int(len(values)),
        }
    valid_classes = {
        wnid: value for wnid, value in class_safety.items() if value["estimate"] is not None
    }
    if not valid_classes:
        raise RuntimeError("Primary clean-correct population has no class with evaluable safety rows")
    worst_wnid, worst_value = max(valid_classes.items(), key=lambda item: float(item[1]["estimate"]))
    primary_semantic["safety_M3_minus_M0_failure_by_class"] = class_safety
    primary_semantic["worst_class_safety_point_estimate"] = {
        "wnid": worst_wnid,
        **worst_value,
    }

    keys_m3, psnr_m3_values = cluster_values(
        rows,
        all_ids,
        all_snrs,
        lambda row: float(row["M3_scratch_gate_fallback_psnr_db"]) - float(row["M0_psnr_db"]),
    )
    keys_m2, psnr_m2_values = cluster_values(
        rows,
        all_ids,
        all_snrs,
        lambda row: float(row["M2_edge_scheduled_psnr_db"]) - float(row["M0_psnr_db"]),
    )
    if keys_m3 != keys_m2:
        raise RuntimeError("Quality bootstrap cluster alignment failure")
    quality: dict[str, Any] = {
        "population": "all_images_all_snrs",
        "M3_minus_M0_psnr_db": bootstrap_mean_ci(
            psnr_m3_values, replicates, base_seed + 1001
        ),
        "M2_minus_M0_psnr_db": bootstrap_mean_ci(
            psnr_m2_values, replicates, base_seed + 1002
        ),
        "fraction_M2_psnr_gain_retained_by_M3": bootstrap_ratio_ci(
            psnr_m3_values, psnr_m2_values, replicates, base_seed + 1003
        ),
    }
    if lpips_available:
        _keys_lpips, lpips_values = cluster_values(
            rows,
            all_ids,
            all_snrs,
            lambda row: float(row["M3_scratch_gate_fallback_lpips"]) - float(row["M0_lpips"]),
        )
        quality["M3_minus_M0_lpips"] = bootstrap_mean_ci(
            lpips_values, replicates, base_seed + 1004
        )
    else:
        quality["M3_minus_M0_lpips"] = {
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
            "reason": "LPIPS unavailable or skipped",
        }
    quality["M3_minus_M0_psnr_by_snr"] = {}
    for snr_index, snr in enumerate(sorted(all_snrs)):
        _keys, values = cluster_values(
            rows,
            all_ids,
            {snr},
            lambda row: float(row["M3_scratch_gate_fallback_psnr_db"]) - float(row["M0_psnr_db"]),
        )
        quality["M3_minus_M0_psnr_by_snr"][snr_key(snr)] = bootstrap_mean_ci(
            values, replicates, base_seed + 2000 + snr_index
        )
    output["quality"] = quality
    return output


def success_gate_row(name: str, estimate: Any, ci_low: Any, ci_high: Any, criterion: str, passed: Any) -> dict[str, Any]:
    return {
        "row_type": "success_gate",
        "gate": name,
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "criterion": criterion,
        "passed": passed,
    }


def evaluate_success_criteria(
    split_name: str,
    config: dict[str, Any],
    coverage_rows: list[dict[str, Any]],
    bootstrap: dict[str, Any],
    evaluator_metadata: dict[str, Any],
    lpips_available: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    criteria = config["success_criteria"]
    primary_threshold = float(config["evaluation"]["primary_clean_threshold"])
    semantic = bootstrap["sensitivity_by_clean_threshold"][str(primary_threshold)]
    efficacy = semantic["gate_efficacy_M3_minus_M2_failure"]
    safety = semantic["safety_M3_minus_M0_failure"]
    conditional_new_error = semantic["accepted_new_error_rate_conditional_on_M0_correct"]
    safety_by_snr = semantic["safety_M3_minus_M0_failure_by_primary_snr"]
    worst_class = semantic["worst_class_safety_point_estimate"]
    quality = bootstrap["quality"]
    psnr = quality["M3_minus_M0_psnr_db"]
    retention = quality["fraction_M2_psnr_gain_retained_by_M3"]
    lpips = quality["M3_minus_M0_lpips"]

    coverage_primary = [
        row
        for row in coverage_rows
        if float(row["clean_threshold"]) == primary_threshold and row["class_index"] == "all"
    ][0]
    per_class_primary = [
        row
        for row in coverage_rows
        if float(row["clean_threshold"]) == primary_threshold and row["class_index"] != "all"
    ]
    final_only = split_name == "official_val"
    min_clean = int(criteria["min_final_clean_count"])
    min_per_class = int(criteria["min_final_clean_count_per_class"])
    coverage_pass: bool | None = (
        int(coverage_primary["clean_count"]) >= min_clean if final_only else None
    )
    per_class_pass: bool | None = (
        min(int(row["clean_count"]) for row in per_class_primary) >= min_per_class if final_only else None
    )
    efficacy_pass = efficacy["ci_high"] is not None and float(efficacy["ci_high"]) < 0.0
    safety_limit = float(criteria["safety_vs_m0_ci_upper_max_absolute"])
    safety_pass = safety["ci_high"] is not None and float(safety["ci_high"]) <= safety_limit
    conditional_new_error_limit = float(
        criteria["accepted_new_error_rate_ci_upper_max_conditional_on_m0_correct"]
    )
    conditional_new_error_pass = (
        conditional_new_error["conservative_upper_95"] is not None
        and float(conditional_new_error["conservative_upper_95"]) <= conditional_new_error_limit
    )
    per_snr_safety_limit = float(criteria["safety_vs_m0_ci_upper_max_absolute_each_primary_snr"])
    per_snr_safety_pass = {
        snr: endpoint["ci_high"] is not None and float(endpoint["ci_high"]) <= per_snr_safety_limit
        for snr, endpoint in safety_by_snr.items()
    }
    worst_class_limit = float(criteria["max_worst_class_m3_minus_m0_failure_point_estimate"])
    worst_class_pass = (
        worst_class["estimate"] is not None and float(worst_class["estimate"]) <= worst_class_limit
    )
    psnr_pass = psnr["ci_low"] is not None and float(psnr["ci_low"]) > 0.0
    by_snr = quality["M3_minus_M0_psnr_by_snr"]
    each_snr_pass = all(
        value["estimate"] is not None and float(value["estimate"]) > 0.0 for value in by_snr.values()
    )
    retention_min = float(criteria["min_fraction_m2_psnr_gain_retained"])
    retention_pass = retention["estimate"] is not None and float(retention["estimate"]) >= retention_min
    lpips_pass = bool(
        lpips_available and lpips["estimate"] is not None and float(lpips["estimate"]) < 0.0
    )
    classifier_min = float(criteria["min_t_cls_cal_macro_top1"])
    classifier_pass = float(evaluator_metadata["best_cls_cal_macro_top1"]) >= classifier_min

    rows = [
        success_gate_row(
            "clean_coverage_count",
            coverage_primary["clean_count"],
            None,
            None,
            f">= {min_clean} on official_val",
            coverage_pass,
        ),
        success_gate_row(
            "clean_coverage_min_per_class",
            min(int(row["clean_count"]) for row in per_class_primary),
            None,
            None,
            f">= {min_per_class} on official_val",
            per_class_pass,
        ),
        success_gate_row(
            "T_cls_cls_cal_macro_top1",
            evaluator_metadata["best_cls_cal_macro_top1"],
            None,
            None,
            f">= {classifier_min}",
            classifier_pass,
        ),
        success_gate_row(
            "gate_efficacy_M3_minus_M2_failure",
            efficacy["estimate"],
            efficacy["ci_low"],
            efficacy["ci_high"],
            "paired cluster-bootstrap 95% CI upper < 0",
            efficacy_pass,
        ),
        success_gate_row(
            "safety_M3_minus_M0_failure",
            safety["estimate"],
            safety["ci_low"],
            safety["ci_high"],
            f"paired cluster-bootstrap 95% CI upper <= {safety_limit}",
            safety_pass,
        ),
        success_gate_row(
            "accepted_new_error_rate_conditional_on_M0_correct",
            conditional_new_error["estimate"],
            conditional_new_error["ci_low"],
            conditional_new_error["conservative_upper_95"],
            (
                "max(one-sided clustered q=.95 upper, image-any-event Clopper-Pearson upper) "
                f"<= {conditional_new_error_limit}; repairs cannot offset errors"
            ),
            conditional_new_error_pass,
        ),
        *[
            success_gate_row(
                f"safety_M3_minus_M0_failure_snr_{snr}",
                safety_by_snr[snr]["estimate"],
                safety_by_snr[snr]["ci_low"],
                safety_by_snr[snr]["ci_high"],
                f"per-SNR clustered 95% CI upper <= {per_snr_safety_limit}",
                per_snr_safety_pass[snr],
            )
            for snr in sorted(safety_by_snr, key=float)
        ],
        success_gate_row(
            "worst_class_M3_minus_M0_failure",
            worst_class["estimate"],
            None,
            None,
            f"worst-class point estimate <= {worst_class_limit} (worst WNID: {worst_class['wnid']})",
            worst_class_pass,
        ),
        success_gate_row(
            "M3_minus_M0_psnr_db",
            psnr["estimate"],
            psnr["ci_low"],
            psnr["ci_high"],
            "paired cluster-bootstrap 95% CI lower > 0",
            psnr_pass,
        ),
        success_gate_row(
            "positive_M3_psnr_gain_each_snr",
            min(float(value["estimate"]) for value in by_snr.values()),
            None,
            None,
            "point estimate > 0 at every preregistered SNR",
            each_snr_pass,
        ),
        success_gate_row(
            "fraction_M2_psnr_gain_retained_by_M3",
            retention["estimate"],
            retention["ci_low"],
            retention["ci_high"],
            f">= {retention_min}",
            retention_pass,
        ),
        success_gate_row(
            "M3_minus_M0_lpips",
            lpips["estimate"],
            lpips.get("ci_low"),
            lpips.get("ci_high"),
            "point estimate < 0",
            lpips_pass,
        ),
    ]
    evaluated = [bool(row["passed"]) for row in rows if row["passed"] is not None]
    result = {
        "split": split_name,
        "all_evaluated_gates_pass": bool(evaluated) and all(evaluated),
        "final_coverage_gates_applicable": final_only,
        "lpips_gate_evaluated": lpips_available,
        "lpips_jointly_required_fail_closed": True,
        "num_pass": sum(evaluated),
        "num_fail": len(evaluated) - sum(evaluated),
        "num_not_applicable": sum(row["passed"] is None for row in rows),
        "rows": rows,
    }
    return rows, result


def find_summary_row(
    rows: list[dict[str, Any]], population: str, scope: str, arm: str
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row.get("row_type") == "arm_summary"
        and row.get("population") == population
        and row.get("snr_scope") == scope
        and row.get("arm") == arm
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one summary row for {(population, scope, arm)}, got {len(matches)}")
    return matches[0]


def make_report(
    config: dict[str, Any],
    split_name: str,
    records: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    arm_rows: list[dict[str, Any]],
    bootstrap: dict[str, Any],
    success: dict[str, Any],
    metadata: dict[str, Any],
) -> str:
    primary_threshold = float(config["evaluation"]["primary_clean_threshold"])
    primary_population = f"clean_correct_tau_{str(primary_threshold).replace('.', 'p')}"
    coverage = [
        row
        for row in coverage_rows
        if float(row["clean_threshold"]) == primary_threshold and row["class_index"] == "all"
    ][0]
    primary_rows = [
        find_summary_row(arm_rows, primary_population, "primary_snrs", arm) for arm in REQUIRED_ARMS
    ]
    semantic = bootstrap["sensitivity_by_clean_threshold"][str(primary_threshold)]
    efficacy = semantic["gate_efficacy_M3_minus_M2_failure"]
    safety = semantic["safety_M3_minus_M0_failure"]
    conditional_new_error = semantic["accepted_new_error_rate_conditional_on_M0_correct"]
    quality = bootstrap["quality"]
    lines = [
        f"# Imagenette Supervised Clean-Correct Audit: {split_name}",
        "",
        "## Bottom line",
        "",
        f"- Evaluated `{len(records)}` unique images with true 10-way WNID labels.",
        f"- Primary clean-correct coverage at calibrated `T_cls` confidence >= `{primary_threshold}` is "
        f"`{coverage['clean_count']}/{coverage['num_images']}` (`{fmt(coverage['clean_coverage'])}`).",
        f"- Gate efficacy `M3-M2` failure delta is `{fmt(efficacy['estimate'])}` with clustered 95% CI "
        f"`[{fmt(efficacy['ci_low'])}, {fmt(efficacy['ci_high'])}]`.",
        f"- Safety `M3-M0` failure delta is `{fmt(safety['estimate'])}` with clustered 95% CI "
        f"`[{fmt(safety['ci_low'])}, {fmt(safety['ci_high'])}]`.",
        f"- Accepted-new-error row rate conditional on M0-correct is "
        f"`{fmt(conditional_new_error['estimate'])}`; the conservative one-sided 95% upper is "
        f"`{fmt(conditional_new_error['conservative_upper_95'])}` (cluster bootstrap "
        f"`{fmt(conditional_new_error['clustered_bootstrap_upper_q95'])}`, image-any-event exact "
        f"`{fmt(conditional_new_error['image_cluster_any_event_clopper_pearson_upper_95'])}`; "
        f"`{conditional_new_error['event_image_clusters']}/{conditional_new_error['eligible_image_clusters']}` "
        "event/eligible images). Repairs do not offset this endpoint.",
        f"- Full-population `M3-M0` PSNR delta is `{fmt(quality['M3_minus_M0_psnr_db']['estimate'])}` dB "
        f"with clustered 95% CI `[{fmt(quality['M3_minus_M0_psnr_db']['ci_low'])}, "
        f"{fmt(quality['M3_minus_M0_psnr_db']['ci_high'])}]`.",
        f"- Preregistered evaluated gates: **{'PASS' if success['all_evaluated_gates_pass'] else 'FAIL'}** "
        f"(`{success['num_pass']}` pass, `{success['num_fail']}` fail, "
        f"`{success['num_not_applicable']}` not applicable).",
        "",
        "## Primary semantic population (SNR 1/4/7 dB)",
        "",
        "| Arm | Failure | Delta vs M0 | Delta vs M2 | Accept | New error | Repair | Protective reject | Missed repair |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary_rows:
        lines.append(
            f"| {row['arm']} | {fmt(row.get('failure_rate'))} | {fmt(row.get('delta_failure_vs_M0'))} | "
            f"{fmt(row.get('delta_failure_vs_M2'))} | {fmt(row.get('accept_rate'))} | "
            f"{fmt(row.get('new_error_count'))} | {fmt(row.get('repair_count'))} | "
            f"{fmt(row.get('protective_reject_count'))} | {fmt(row.get('missed_repair_count'))} |"
        )
    lines.extend(
        [
            "",
            "## Preregistered success gates",
            "",
            "| Gate | Estimate | 95% CI | Criterion | Status |",
            "|---|---:|---:|---|---|",
        ]
    )
    for row in success["rows"]:
        status = "N/A" if row["passed"] is None else ("PASS" if row["passed"] else "FAIL")
        interval = (
            "N/A"
            if row["ci_low"] is None
            else f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}]"
        )
        lines.append(
            f"| {row['gate']} | {fmt(row['estimate'])} | {interval} | {row['criterion']} | {status} |"
        )
    lines.extend(
        [
            "",
            "## PSNR gain by SNR (all images)",
            "",
            "| SNR (dB) | M3-M0 PSNR (dB) | Clustered 95% CI |",
            "|---:|---:|---:|",
        ]
    )
    for snr, endpoint in quality["M3_minus_M0_psnr_by_snr"].items():
        lines.append(
            f"| {snr} | {fmt(endpoint['estimate'])} | [{fmt(endpoint['ci_low'])}, {fmt(endpoint['ci_high'])}] |"
        )
    lines.extend(
        [
            "",
            "## Clean-correct coverage by class",
            "",
            "| WNID | Clean / total | Coverage |",
            "|---|---:|---:|",
        ]
    )
    class_coverage = [
        row
        for row in coverage_rows
        if float(row["clean_threshold"]) == primary_threshold and row["class_index"] != "all"
    ]
    for row in class_coverage:
        lines.append(
            f"| {row['wnid']} | {row['clean_count']} / {row['num_images']} | {fmt(row['clean_coverage'])} |"
        )
    lines.extend(
        [
            "",
            "## Integrity and interpretation",
            "",
            f"- Protocol SHA256: `{metadata['hashes']['protocol_sha256']}`",
            f"- Split manifest SHA256: `{metadata['hashes']['split_manifest_sha256']}`",
            f"- Gate checkpoint SHA256: `{metadata['hashes']['gate_checkpoint_sha256']}`",
            f"- Evaluator checkpoint SHA256: `{metadata['hashes']['evaluator_checkpoint_sha256']}`",
            "- `G_gate` sees only receiver-visible M0 and the edge candidate. It never sees the original or the true label.",
            "- `T_cls` is an independent scratch-trained evaluator and does not participate in gating, alpha selection, "
            "refiner training, or DeepJSCC training.",
            "- M0, raw-refiner outputs, scheduled candidates, and M3 finals are quantized as "
            "`round(255*x)/255`; no full-image export is retained.",
            "- Bootstrap clusters are original image IDs and preserve every selected SNR/channel-seed row.",
            "- On clean-correct images, Drift-Origin and Drift-GT are numerically identical; they are not counted as "
            "independent evidence.",
            "",
            "## Artifacts",
            "",
            f"- Per-sample CSV: `{metadata['artifacts']['per_sample_csv']}`",
            f"- Summary CSV: `{metadata['artifacts']['summary_csv']}`",
            f"- Summary JSON: `{metadata['artifacts']['summary_json']}`",
            f"- Metadata JSON: `{metadata['artifacts']['metadata_json']}`",
            f"- Critical galleries only: `{metadata['artifacts']['gallery_index']}`",
            "",
        ]
    )
    return "\n".join(lines)


def artifact_paths(config: dict[str, Any]) -> dict[str, Path]:
    inputs = config["inputs"]
    return {
        "split_manifest": resolve_project_path(config["outputs"]["split_manifest"]),
        "gate_checkpoint": resolve_project_path(inputs["gate_checkpoint"]),
        "evaluator_checkpoint": resolve_project_path(inputs["evaluator_checkpoint"]),
        "deepjscc_config": resolve_project_path(inputs["deepjscc_config"]),
        "deepjscc_checkpoint": resolve_project_path(inputs["deepjscc_checkpoint"]),
        "forbidden_deepjscc_checkpoint": resolve_project_path(inputs["forbidden_deepjscc_checkpoint"]),
        "edge_config": resolve_project_path(inputs["edge_config"]),
        "edge_checkpoint": resolve_project_path(inputs["edge_checkpoint"]),
        "no_edge_config": resolve_project_path(inputs["no_edge_config"]),
        "no_edge_checkpoint": resolve_project_path(inputs["no_edge_checkpoint"]),
        "schedule": resolve_project_path(inputs["monotonic_schedule"]),
        "preregistration": resolve_project_path(config["protocol"]["preregistration"]),
        "evaluator_script": EVALUATOR_SCRIPT,
        "deepjscc_adapter_source": DEEPJSCC_ADAPTER_SOURCE,
        "metrics_source": METRICS_SOURCE,
        "refiner_source": REFINER_SOURCE,
    }


def assert_preregistered_constants(config: dict[str, Any]) -> None:
    expected = PREREGISTERED_CONSTANTS

    def require(label: str, actual: Any, target: Any) -> None:
        if actual != target:
            raise RuntimeError(f"Preregistered constant changed: {label}={actual!r}, expected={target!r}")

    require("analysis_id", config.get("analysis_id"), expected["analysis_id"])
    require("method", config.get("method"), expected["method"])
    require("dataset", config.get("dataset"), expected["dataset"])
    require("seed", int(config.get("seed", -1)), expected["seed"])
    protocol = config["protocol"]
    require(
        "protocol.preregistration",
        str(protocol.get("preregistration")),
        "reports/imagenette_supervised_preregistration_2026-07-10.md",
    )
    require("protocol.official_val_is_final_only", protocol.get("official_val_is_final_only"), True)
    require("protocol.pretrained_imagenet_models_are_primary", protocol.get("pretrained_imagenet_models_are_primary"), False)

    data = config["data"]
    for key in ("archive_size_bytes", "train_image_count", "val_image_count", "image_size"):
        require(f"data.{key}", int(data[key]), int(expected[key]))
    require("data.archive_md5", str(data["archive_md5"]).lower(), expected["archive_md5"])
    require("data.classes", [str(item) for item in data["classes"]], expected["classes"])

    split = config["split"]
    require("split.method", str(split.get("method")), expected["split_method"])
    require("split.seed", int(split.get("seed", -1)), expected["split_seed"])
    require(
        "split.ratios",
        {str(key): float(value) for key, value in dict(split.get("ratios", {})).items()},
        expected["split_ratios"],
    )
    require("split.official_val_role", str(split.get("official_val_role")), "sealed_final_test")
    require("split.require_content_sha256", split.get("require_content_sha256"), True)
    require("split.reject_exact_content_duplicates", split.get("reject_exact_content_duplicates"), True)

    training = config["training"]
    require("training.checkpoint_selection", training.get("checkpoint_selection"), "cls_cal_macro_top1")
    require("training.temperature_scaling_split", training.get("temperature_scaling_split"), "cls_cal")
    require("training.no_tta", training.get("no_tta"), True)
    for role, role_expected in expected["classifiers"].items():
        role_config = config["scratch_classifiers"][role]
        for key, target in role_expected.items():
            actual = (
                float(role_config[key]) if key == "min_cal_macro_top1" else role_config[key]
            )
            require(f"scratch_classifiers.{role}.{key}", actual, target)
        require(f"scratch_classifiers.{role}.weights", role_config.get("weights"), None)
        require(f"scratch_classifiers.{role}.pretrained", role_config.get("pretrained", False), False)

    channel = config["channel"]
    require("channel.type", str(channel.get("type")), expected["channel_type"])
    require("channel.snrs", [float(item) for item in channel["snrs"]], expected["snrs"])
    require(
        "channel.policy_dev_seeds",
        [int(item) for item in channel["policy_dev_seeds"]],
        expected["policy_dev_seeds"],
    )
    require(
        "channel.final_seeds",
        [int(item) for item in channel["final_seeds"]],
        expected["final_seeds"],
    )

    require("arms.names", [str(item) for item in config["arms"]["names"]], list(REQUIRED_ARMS))
    require(
        "arms.candidate_formula",
        config["arms"].get("candidate_formula"),
        "clamp(M0 + alpha_snr * (raw_refined - M0), 0, 1)",
    )
    require(
        "arms.gate_accept_rule",
        config["arms"].get("gate_accept_rule"),
        "G_gate(candidate).top1 == G_gate(M0).top1",
    )

    evaluation = config["evaluation"]
    require(
        "evaluation.primary_clean_threshold",
        float(evaluation["primary_clean_threshold"]),
        expected["primary_clean_threshold"],
    )
    require(
        "evaluation.clean_thresholds",
        [float(item) for item in evaluation["clean_thresholds"]],
        expected["clean_thresholds"],
    )
    require(
        "evaluation.primary_snrs",
        [float(item) for item in evaluation["primary_snrs"]],
        expected["primary_snrs"],
    )
    require("evaluation.bootstrap_replicates", int(evaluation["bootstrap_replicates"]), expected["bootstrap_replicates"])
    require("evaluation.bootstrap_seed", int(evaluation["bootstrap_seed"]), expected["bootstrap_seed"])
    require("evaluation.bootstrap_cluster", evaluation.get("bootstrap_cluster"), "image_id")
    require("evaluation.quantize_png", evaluation.get("quantize_png"), True)
    require("evaluation.lpips", evaluation.get("lpips"), True)
    require("evaluation.lpips_net", evaluation.get("lpips_net"), "alex")
    require("evaluation.full_test_quality_population", evaluation.get("full_test_quality_population"), "all_official_val_images")
    require("evaluation.main_semantic_population", evaluation.get("main_semantic_population"), "T_cls_clean_correct_at_primary_threshold")

    criteria = config["success_criteria"]
    expected_criteria = {
        "min_final_clean_count": 2500,
        "min_final_clean_count_per_class": 150,
        "min_t_cls_cal_macro_top1": 0.85,
        "gate_efficacy_ci_upper_strictly_below_zero": True,
        "safety_vs_m0_ci_upper_max_absolute": 0.005,
        "accepted_new_error_rate_ci_upper_max_conditional_on_m0_correct": 0.005,
        "safety_vs_m0_ci_upper_max_absolute_each_primary_snr": 0.005,
        "max_worst_class_m3_minus_m0_failure_point_estimate": 0.02,
        "m3_psnr_gain_ci_lower_strictly_above_zero": True,
        "require_positive_m3_psnr_point_estimate_each_snr": True,
        "min_fraction_m2_psnr_gain_retained": 0.50,
        "require_negative_lpips_delta": True,
    }
    for key, target in expected_criteria.items():
        actual = criteria.get(key)
        if isinstance(target, float):
            actual = float(actual)
        require(f"success_criteria.{key}", actual, target)


def validate_static_config(config: dict[str, Any], split_name: str, args: argparse.Namespace) -> None:
    assert_preregistered_constants(config)
    configured_arms = tuple(str(item) for item in config["arms"]["names"])
    missing_arms = [arm for arm in REQUIRED_ARMS if arm not in configured_arms]
    if missing_arms:
        raise RuntimeError(f"Config is missing required arms: {missing_arms}")
    if not bool(config["evaluation"].get("quantize_png", False)):
        raise RuntimeError("This audit requires evaluation.quantize_png=true")
    stages = {str(item) for item in config["arms"].get("quantize_png_stages", [])}
    required_stages = {"M0", "raw_refined", "candidate", "final"}
    if not required_stages.issubset(stages):
        raise RuntimeError(f"Missing required PNG-simulation stages: {sorted(required_stages - stages)}")
    thresholds = [float(item) for item in config["evaluation"]["clean_thresholds"]]
    primary_threshold = float(config["evaluation"]["primary_clean_threshold"])
    if primary_threshold not in thresholds or 0.0 not in thresholds or 0.7 not in thresholds:
        raise RuntimeError("clean_thresholds must contain the primary threshold plus 0.0 and 0.7")
    snrs = [float(item) for item in config["channel"]["snrs"]]
    primary_snrs = [float(item) for item in config["evaluation"]["primary_snrs"]]
    if not set(primary_snrs).issubset(snrs):
        raise RuntimeError("evaluation.primary_snrs must be a subset of channel.snrs")
    if int(config["evaluation"]["bootstrap_replicates"]) != 10000:
        raise RuntimeError("The preregistered audit requires exactly 10,000 bootstrap replicates")
    if args.skip_lpips:
        raise RuntimeError("--skip-lpips is forbidden because LPIPS is a jointly required endpoint")
    policy_seeds = [int(item) for item in config["channel"]["policy_dev_seeds"]]
    final_seeds = [int(item) for item in config["channel"]["final_seeds"]]
    if len(policy_seeds) != 1:
        raise RuntimeError(f"policy_dev requires exactly one channel seed, got {policy_seeds}")
    if len(final_seeds) != 3 or len(set(final_seeds)) != 3:
        raise RuntimeError(f"official_val requires exactly three distinct channel seeds, got {final_seeds}")
    if split_name == "policy_dev" and args.unlock_final:
        raise RuntimeError("--unlock-final is only valid with --split official_val")
    if split_name == "official_val" and not bool(config["protocol"].get("official_val_is_final_only", False)):
        raise RuntimeError("Config does not mark official_val as final-only")
    if split_name == "official_val":
        forbidden = []
        if args.dry_run:
            forbidden.append("--dry-run")
        if args.overwrite:
            forbidden.append("--overwrite")
        if args.output_dir is not None:
            forbidden.append("--output-dir")
        if forbidden:
            raise RuntimeError(
                "Official final forbids noncanonical or repeatable execution flags: "
                + ", ".join(forbidden)
            )


def deepjscc_repo_provenance(deepjscc_config_path: Path) -> dict[str, Any]:
    with deepjscc_config_path.open("r", encoding="utf-8") as handle:
        deepjscc_config = yaml.safe_load(handle)
    repo_root = resolve_project_path(deepjscc_config["baseline"]["repo"]).resolve()
    configured_commit = str(deepjscc_config["baseline"].get("commit", ""))
    if len(configured_commit) != GIT_COMMIT_HEX_LENGTH:
        raise RuntimeError(f"DeepJSCC config has invalid baseline commit: {configured_commit!r}")
    try:
        actual_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.STDOUT, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root, stderr=subprocess.STDOUT, text=True
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Could not verify DeepJSCC repository at {repo_root}: {exc.output}") from exc
    if actual_head != configured_commit:
        raise RuntimeError(
            f"DeepJSCC actual HEAD differs from source config: actual={actual_head}, configured={configured_commit}"
        )
    if status:
        raise RuntimeError(f"DeepJSCC repository must be clean for the locked audit: {status}")
    model_path = (repo_root / "model.py").resolve()
    channel_path = (repo_root / "channel.py").resolve()
    for label, path in (("model.py", model_path), ("channel.py", channel_path)):
        if not path.is_file() or repo_root not in path.parents:
            raise RuntimeError(f"DeepJSCC {label} is missing or escaped the verified repo: {path}")
    return {
        "repo_root": repo_root,
        "configured_commit": configured_commit,
        "actual_head": actual_head,
        "clean": True,
        "model_path": model_path,
        "model_sha256": sha256_file(model_path),
        "channel_path": channel_path,
        "channel_sha256": sha256_file(channel_path),
    }


def compute_hashes(
    config_path: Path,
    paths: dict[str, Path],
    config: dict[str, Any],
    deepjscc_repo: dict[str, Any],
) -> dict[str, str]:
    return {
        "config_file_sha256": sha256_file(config_path),
        "protocol_sha256": protocol_sha256(config),
        "preregistration_sha256": sha256_file(paths["preregistration"]),
        "evaluator_script_sha256": sha256_file(paths["evaluator_script"]),
        "deepjscc_adapter_source_sha256": sha256_file(paths["deepjscc_adapter_source"]),
        "metrics_source_sha256": sha256_file(paths["metrics_source"]),
        "refiner_source_sha256": sha256_file(paths["refiner_source"]),
        "split_manifest_sha256": sha256_file(paths["split_manifest"]),
        "gate_checkpoint_sha256": sha256_file(paths["gate_checkpoint"]),
        "evaluator_checkpoint_sha256": sha256_file(paths["evaluator_checkpoint"]),
        "deepjscc_config_sha256": sha256_file(paths["deepjscc_config"]),
        "deepjscc_checkpoint_sha256": sha256_file(paths["deepjscc_checkpoint"]),
        "edge_config_sha256": sha256_file(paths["edge_config"]),
        "edge_checkpoint_sha256": sha256_file(paths["edge_checkpoint"]),
        "no_edge_config_sha256": sha256_file(paths["no_edge_config"]),
        "no_edge_checkpoint_sha256": sha256_file(paths["no_edge_checkpoint"]),
        "schedule_sha256": sha256_file(paths["schedule"]),
        "deepjscc_repo_head": str(deepjscc_repo["actual_head"]),
        "deepjscc_model_py_sha256": str(deepjscc_repo["model_sha256"]),
        "deepjscc_channel_py_sha256": str(deepjscc_repo["channel_sha256"]),
    }


def validate_final_lock(
    config: dict[str, Any], split_name: str, unlock_argument: bool, hashes: dict[str, str]
) -> dict[str, Any]:
    lock = config.get("final_lock")
    if not isinstance(lock, dict):
        raise RuntimeError("Config is missing final_lock mapping")
    if split_name != "official_val":
        return {"enforced": False, "reason": "policy_dev does not consume the final lock"}
    if not unlock_argument:
        raise RuntimeError("official_val is sealed; pass --unlock-final for the one-shot final invocation")
    if not bool(lock.get("unlocked", False)):
        raise RuntimeError("official_val is sealed; final_lock.unlocked must be explicitly set to true")
    if not lock.get("locked_at_utc"):
        raise RuntimeError("final_lock.locked_at_utc must be populated before official_val")
    required = (
        "protocol_sha256",
        "preregistration_sha256",
        "evaluator_script_sha256",
        "split_manifest_sha256",
        "gate_checkpoint_sha256",
        "evaluator_checkpoint_sha256",
        "deepjscc_config_sha256",
        "deepjscc_checkpoint_sha256",
        "edge_config_sha256",
        "edge_checkpoint_sha256",
        "no_edge_config_sha256",
        "no_edge_checkpoint_sha256",
        "schedule_sha256",
        "deepjscc_adapter_source_sha256",
        "metrics_source_sha256",
        "refiner_source_sha256",
        "deepjscc_model_py_sha256",
        "deepjscc_channel_py_sha256",
        "policy_dev_summary_sha256",
        "policy_dev_metadata_sha256",
    )
    verified: dict[str, str] = {}
    for key in required:
        expected = lock.get(key)
        if not isinstance(expected, str) or len(expected) != SHA256_HEX_LENGTH:
            raise RuntimeError(f"Final lock hash is unpopulated or invalid: final_lock.{key}")
        if key.startswith("policy_dev_"):
            actual = expected
        else:
            actual = hashes[key]
        if expected.lower() != actual.lower():
            raise RuntimeError(f"Final lock mismatch for {key}: expected={expected}, actual={actual}")
        verified[key] = actual
    expected_head = lock.get("deepjscc_repo_head")
    if not isinstance(expected_head, str) or len(expected_head) != GIT_COMMIT_HEX_LENGTH:
        raise RuntimeError("Final lock has invalid final_lock.deepjscc_repo_head")
    if expected_head.lower() != hashes["deepjscc_repo_head"].lower():
        raise RuntimeError(
            f"Final lock mismatch for deepjscc_repo_head: expected={expected_head}, "
            f"actual={hashes['deepjscc_repo_head']}"
        )
    verified["deepjscc_repo_head"] = hashes["deepjscc_repo_head"]
    return {
        "enforced": True,
        "unlocked_argument": True,
        "config_unlocked": True,
        "locked_at_utc": lock["locked_at_utc"],
        "verified_hashes": verified,
    }


def validate_locked_policy_dev_artifacts(
    config: dict[str, Any], lock: dict[str, Any], hashes: dict[str, str]
) -> dict[str, Any]:
    policy_dir = require_analysis_output_path(
        resolve_project_path(config["outputs"]["policy_dev_dir"]), "canonical policy_dev_dir"
    )
    summary_path = policy_dir / "summary.json"
    metadata_path = policy_dir / "metadata.json"
    state_path = policy_dir / "STATE.json"
    for label, path in (("summary", summary_path), ("metadata", metadata_path), ("state", state_path)):
        if not path.is_file():
            raise FileNotFoundError(f"Canonical policy-dev {label} artifact is missing: {path}")
    expected_summary_sha = str(lock["policy_dev_summary_sha256"]).lower()
    expected_metadata_sha = str(lock["policy_dev_metadata_sha256"]).lower()
    actual_summary_sha = sha256_file(summary_path)
    actual_metadata_sha = sha256_file(metadata_path)
    if actual_summary_sha != expected_summary_sha or actual_metadata_sha != expected_metadata_sha:
        raise RuntimeError(
            "Canonical policy-dev artifact hash mismatch: "
            f"summary={actual_summary_sha}, metadata={actual_metadata_sha}"
        )
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    with state_path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    if summary.get("split") != "policy_dev" or metadata.get("split") != "policy_dev":
        raise RuntimeError("Locked policy-dev artifacts do not identify the policy_dev split")
    if summary.get("success_criteria", {}).get("all_evaluated_gates_pass") is not True:
        raise RuntimeError("Canonical policy-dev summary did not PASS every jointly required gate")
    if state.get("state") != "COMPLETE" or state.get("split") != "policy_dev":
        raise RuntimeError("Canonical policy-dev output has no COMPLETE state")
    immutable_keys = (
        "protocol_sha256",
        "preregistration_sha256",
        "evaluator_script_sha256",
        "split_manifest_sha256",
        "gate_checkpoint_sha256",
        "evaluator_checkpoint_sha256",
        "deepjscc_config_sha256",
        "deepjscc_checkpoint_sha256",
        "edge_config_sha256",
        "edge_checkpoint_sha256",
        "no_edge_config_sha256",
        "no_edge_checkpoint_sha256",
        "schedule_sha256",
        "deepjscc_adapter_source_sha256",
        "metrics_source_sha256",
        "refiner_source_sha256",
        "deepjscc_repo_head",
        "deepjscc_model_py_sha256",
        "deepjscc_channel_py_sha256",
    )
    metadata_hashes = metadata.get("hashes", {})
    mismatches = {
        key: {"policy_dev": metadata_hashes.get(key), "current": hashes.get(key)}
        for key in immutable_keys
        if metadata_hashes.get(key) != hashes.get(key)
    }
    if mismatches:
        raise RuntimeError(f"Canonical policy-dev provenance is stale: {mismatches}")
    if metadata.get("evaluation", {}).get("row_grid", {}).get("passed") is not True:
        raise RuntimeError("Canonical policy-dev metadata does not prove a complete unique row grid")
    return {
        "directory": project_relative(policy_dir),
        "summary": project_relative(summary_path),
        "summary_sha256": actual_summary_sha,
        "metadata": project_relative(metadata_path),
        "metadata_sha256": actual_metadata_sha,
        "all_evaluated_gates_pass": True,
        "state": "COMPLETE",
    }


def validate_paths(paths: dict[str, Path]) -> None:
    missing = [f"{key}: {path}" for key, path in paths.items() if key != "forbidden_deepjscc_checkpoint" and not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))
    if paths["deepjscc_checkpoint"].resolve() == paths["forbidden_deepjscc_checkpoint"].resolve():
        raise RuntimeError("Config points to the forbidden DeepJSCC latest.pt checkpoint")


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    path = require_analysis_output_path(path, "output directory")
    if path.exists():
        if path.is_symlink():
            raise RuntimeError(f"Refusing to replace symlink output directory: {path}")
        if any(path.iterdir()):
            if not overwrite:
                raise FileExistsError(f"Output directory is non-empty; use --overwrite explicitly: {path}")
            require_analysis_output_path(path, "overwrite target")
            shutil.rmtree(path)
        else:
            path.rmdir()
    path.mkdir(parents=True, exist_ok=False)


def official_final_paths(final_dir: Path) -> tuple[Path, Path]:
    final_dir = require_analysis_output_path(final_dir, "canonical final_dir")
    staging_dir = require_analysis_output_path(
        final_dir.with_name(f".{final_dir.name}.staging"), "official-final staging directory"
    )
    consumed_marker = require_analysis_output_path(
        final_dir.with_name(f".{final_dir.name}.OFFICIAL_VAL_CONSUMED.json"),
        "official-val consumed marker",
    )
    return staging_dir, consumed_marker


def create_consumed_marker_exclusive(
    marker_path: Path, config: dict[str, Any], hashes: dict[str, str]
) -> dict[str, Any]:
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state": "OFFICIAL_VAL_CONSUMED_BEFORE_ACCESS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_id": config["analysis_id"],
        "protocol_sha256": hashes["protocol_sha256"],
        "evaluator_script_sha256": hashes["evaluator_script_sha256"],
        "note": "Created with O_CREAT|O_EXCL before archive member, extracted val, or val image access.",
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(marker_path, flags, 0o444)
    try:
        encoded = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {
        "path": project_relative(marker_path),
        "sha256": sha256_file(marker_path),
        **payload,
    }


def write_completion_state(
    output_dir: Path,
    logical_output_dir: Path,
    split_name: str,
    state: str,
    artifact_hashes: dict[str, str] | None = None,
) -> None:
    save_json(
        output_dir / "STATE.json",
        {
            "state": state,
            "split": split_name,
            "logical_output_dir": project_relative(logical_output_dir),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_hashes": artifact_hashes or {},
        },
    )


def main() -> None:
    args = parse_args()
    split_name = normalize_split_name(args.split)
    config_path = resolve_project_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Config must be a YAML mapping: {config_path}")
    validate_static_config(config, split_name, args)
    paths = artifact_paths(config)
    validate_paths(paths)
    deepjscc_repo = deepjscc_repo_provenance(paths["deepjscc_config"])
    hashes = compute_hashes(config_path, paths, config, deepjscc_repo)
    final_lock = validate_final_lock(config, split_name, args.unlock_final, hashes)
    device = resolve_device(args.device)
    channel_seeds = [
        int(item)
        for item in config["channel"][
            "final_seeds" if split_name == "official_val" else "policy_dev_seeds"
        ]
    ]
    snrs = [float(item) for item in config["channel"]["snrs"]]

    canonical_output_dir = require_analysis_output_path(
        resolve_project_path(config["outputs"]["final_dir" if split_name == "official_val" else "policy_dev_dir"]),
        "canonical evaluation output",
    )
    logical_output_dir = canonical_output_dir
    if split_name == "policy_dev" and args.output_dir is not None:
        logical_output_dir = require_analysis_output_path(
            resolve_project_path(args.output_dir), "custom policy-dev output"
        )
    output_dir = logical_output_dir
    consumed_marker_metadata: dict[str, Any] | None = None
    locked_policy_dev: dict[str, Any] | None = None
    if split_name == "official_val":
        locked_policy_dev = validate_locked_policy_dev_artifacts(
            config, config["final_lock"], hashes
        )
        if canonical_output_dir.exists():
            raise FileExistsError(
                f"Canonical official final directory must be absent: {canonical_output_dir}"
            )
        staging_dir, consumed_marker = official_final_paths(canonical_output_dir)
        if staging_dir.exists():
            raise FileExistsError(f"Official-final staging directory already exists: {staging_dir}")
        consumed_marker_metadata = create_consumed_marker_exclusive(
            consumed_marker, config, hashes
        )
        staging_dir.mkdir(parents=True, exist_ok=False)
        output_dir = staging_dir
        write_completion_state(output_dir, logical_output_dir, split_name, "IN_PROGRESS")

    official_val_manifest: dict[str, Any] | None = None
    if split_name == "official_val":
        records, official_val_manifest, manifest_metadata = build_sealed_official_val_manifest(
            paths["split_manifest"], config
        )
        if manifest_metadata["training_split_manifest_sha256"] != hashes["split_manifest_sha256"]:
            raise RuntimeError("Training split manifest changed while official val was being sealed")
    else:
        records, _manifest, manifest_metadata = load_manifest_records(
            paths["split_manifest"], config, split_name, verify_content=True
        )
        if manifest_metadata["sha256"] != hashes["split_manifest_sha256"]:
            raise RuntimeError("Manifest changed while inputs were being validated")

    edge_checkpoint = torch_load_checkpoint(paths["edge_checkpoint"])
    no_edge_checkpoint = torch_load_checkpoint(paths["no_edge_checkpoint"])
    edge_embedded_config = edge_checkpoint.get("config")
    no_edge_embedded_config = no_edge_checkpoint.get("config")
    if not isinstance(edge_embedded_config, dict) or not isinstance(no_edge_embedded_config, dict):
        raise RuntimeError("Both refiner checkpoints must contain embedded configs")
    alphas, _schedule_payload, schedule_metadata = load_frozen_schedule(
        paths["schedule"],
        str(config["inputs"]["schedule_key"]),
        snrs,
        edge_embedded_config,
        no_edge_embedded_config,
    )

    plan = {
        "config": project_relative(config_path),
        "split": split_name,
        "num_images": len(records),
        "channel_seeds": channel_seeds,
        "snrs": snrs,
        "num_pipeline_rows": len(records) * len(channel_seeds) * len(snrs),
        "device": str(device),
        "output_dir": project_relative(logical_output_dir),
        "skip_lpips": args.skip_lpips,
        "hashes": hashes,
        "final_lock": final_lock,
        "schedule": schedule_metadata,
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, "plan": plan}, indent=2, ensure_ascii=False))
        return

    if split_name == "policy_dev":
        prepare_output_dir(output_dir, args.overwrite)
        write_completion_state(output_dir, logical_output_dir, split_name, "IN_PROGRESS")
    shutil.copy2(config_path, output_dir / "config.yaml")
    evaluator_snapshot = output_dir / EVALUATOR_SCRIPT.name
    shutil.copy2(EVALUATOR_SCRIPT, evaluator_snapshot)
    if sha256_file(evaluator_snapshot) != hashes["evaluator_script_sha256"]:
        raise RuntimeError("Evaluator script changed while its immutable snapshot was being copied")
    if official_val_manifest is not None:
        official_manifest_path = output_dir / "official_val_manifest.json"
        save_json(official_manifest_path, official_val_manifest)
        manifest_metadata["path"] = project_relative(
            logical_output_path(official_manifest_path, output_dir, logical_output_dir)
        )
        manifest_metadata["file_sha256"] = sha256_file(official_manifest_path)
    save_json(output_dir / "run_plan.json", plan)
    seed_everything(int(config["seed"]))

    gate_model, gate_temperature, gate_metadata = load_scratch_classifier(
        paths["gate_checkpoint"],
        "G_gate",
        config["scratch_classifiers"]["G_gate"],
        [str(item) for item in config["data"]["classes"]],
        hashes["split_manifest_sha256"],
        hashes["protocol_sha256"],
        device,
    )
    evaluator, evaluator_temperature, evaluator_metadata = load_scratch_classifier(
        paths["evaluator_checkpoint"],
        "T_cls",
        config["scratch_classifiers"]["T_cls"],
        [str(item) for item in config["data"]["classes"]],
        hashes["split_manifest_sha256"],
        hashes["protocol_sha256"],
        device,
    )
    deepjscc, deepjscc_config, deepjscc_metadata = load_deepjscc(
        paths["deepjscc_config"],
        paths["deepjscc_checkpoint"],
        str(config["channel"]["type"]),
        snrs[0],
        device,
    )
    loaded_model_module = sys.modules.get("model")
    loaded_model_path = Path(str(getattr(loaded_model_module, "__file__", ""))).resolve()
    if loaded_model_path != deepjscc_repo["model_path"]:
        raise RuntimeError(
            f"Imported DeepJSCC model module is not the verified model.py: {loaded_model_path}"
        )
    deepjscc_metadata["verified_repo"] = {
        "repo_root": project_relative(deepjscc_repo["repo_root"]),
        "actual_head": deepjscc_repo["actual_head"],
        "configured_commit": deepjscc_repo["configured_commit"],
        "clean": deepjscc_repo["clean"],
        "model_py": project_relative(deepjscc_repo["model_path"]),
        "model_py_sha256": deepjscc_repo["model_sha256"],
        "channel_py": project_relative(deepjscc_repo["channel_path"]),
        "channel_py_sha256": deepjscc_repo["channel_sha256"],
        "imported_model_module_path": project_relative(loaded_model_path),
    }
    edge_refiner, edge_config, edge_metadata = load_refiner_model(
        paths["edge_checkpoint"], paths["edge_config"], True, device
    )
    no_edge_refiner, no_edge_config, no_edge_metadata = load_refiner_model(
        paths["no_edge_checkpoint"], paths["no_edge_config"], False, device
    )
    lpips_model, lpips_error = try_load_lpips(config, device, args.skip_lpips)

    dataset = ManifestImageDataset(records, int(config["data"]["image_size"]))
    loader = make_loader(dataset, config, device)
    run_started = time.perf_counter()
    original_predictions = classify_originals(
        loader, evaluator, evaluator_temperature, config, device
    )
    original_classification_seconds = time.perf_counter() - run_started
    rows, pipeline_metadata = evaluate_pipeline(
        config=config,
        split_name=split_name,
        records=records,
        loader=loader,
        original_predictions=original_predictions,
        deepjscc=deepjscc,
        deepjscc_config=deepjscc_config,
        edge_refiner=edge_refiner,
        edge_config=edge_config,
        no_edge_refiner=no_edge_refiner,
        no_edge_config=no_edge_config,
        gate_model=gate_model,
        gate_temperature=gate_temperature,
        evaluator=evaluator,
        evaluator_temperature=evaluator_temperature,
        alphas=alphas,
        channel_seeds=channel_seeds,
        snrs=snrs,
        lpips_model=lpips_model,
        output_dir=output_dir,
        logical_output_dir=logical_output_dir,
        device=device,
    )
    inference_seconds = time.perf_counter() - run_started
    row_grid_metadata = assert_complete_unique_pipeline_rows(
        rows,
        records,
        snrs,
        channel_seeds,
        require_lpips=bool(config["success_criteria"]["require_negative_lpips_delta"]),
    )

    thresholds = [float(item) for item in config["evaluation"]["clean_thresholds"]]
    classes = [str(item) for item in config["data"]["classes"]]
    coverage_rows = build_clean_coverage_rows(records, original_predictions, thresholds, classes)
    arm_rows = build_arm_summaries(rows, records, original_predictions, config)
    per_class_rows = build_per_class_rows(rows, records, original_predictions, config)
    bootstrap = build_bootstrap_results(
        rows, records, original_predictions, config, lpips_available=lpips_model is not None
    )
    success_rows, success = evaluate_success_criteria(
        split_name,
        config,
        coverage_rows,
        bootstrap,
        evaluator_metadata,
        lpips_available=lpips_model is not None,
    )
    summary_rows = coverage_rows + arm_rows + per_class_rows + success_rows
    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"
    write_csv(summary_csv, summary_rows)

    primary_threshold = float(config["evaluation"]["primary_clean_threshold"])
    clean_ids = clean_image_ids(records, original_predictions, primary_threshold)
    primary_snrs = {float(item) for item in config["evaluation"]["primary_snrs"]}
    primary_rows = scoped_rows(rows, clean_ids, primary_snrs)
    event_counts = {
        "M2_new_error": sum(bool(row["M2_new_error_vs_M0"]) for row in primary_rows),
        "M2_repair": sum(bool(row["M2_repair_vs_M0"]) for row in primary_rows),
        "M3_accepted_new_error": sum(bool(row["M3_accepted_new_error"]) for row in primary_rows),
        "M3_accepted_repair": sum(bool(row["M3_accepted_repair"]) for row in primary_rows),
        "M3_protective_reject": sum(bool(row["M3_protective_reject"]) for row in primary_rows),
        "M3_missed_repair": sum(bool(row["M3_missed_repair"]) for row in primary_rows),
    }
    event_denominator_rows = {
        "M2_new_error": sum(bool(row["M0_correct"]) for row in primary_rows),
        "M2_repair": sum(not bool(row["M0_correct"]) for row in primary_rows),
        "M3_accepted_new_error": sum(bool(row["M0_correct"]) for row in primary_rows),
        "M3_accepted_repair": sum(not bool(row["M0_correct"]) for row in primary_rows),
        "M3_protective_reject": sum(bool(row["M0_correct"]) for row in primary_rows),
        "M3_missed_repair": sum(not bool(row["M0_correct"]) for row in primary_rows),
    }
    event_exact_upper_95 = {
        key: exact_binomial_upper_95(value, event_denominator_rows[key])
        for key, value in event_counts.items()
    }
    summary_payload = {
        "analysis_id": config["analysis_id"],
        "split": split_name,
        "primary_clean_threshold": primary_threshold,
        "primary_snrs": sorted(primary_snrs),
        "clean_coverage": coverage_rows,
        "arm_summaries": arm_rows,
        "per_class_summaries": per_class_rows,
        "bootstrap": bootstrap,
        "success_criteria": success,
        "primary_event_counts": event_counts,
        "primary_event_denominator_rows": event_denominator_rows,
        "primary_event_row_level_clopper_pearson_upper_95": event_exact_upper_95,
        "binomial_interval_method": "one-sided 95% Clopper-Pearson via scipy.stats.beta",
    }
    save_json(summary_json, summary_payload)

    artifacts = {
        "per_sample_csv": project_relative(logical_output_dir / "per_sample.csv"),
        "summary_csv": project_relative(logical_output_dir / "summary.csv"),
        "summary_json": project_relative(logical_output_dir / "summary.json"),
        "metadata_json": project_relative(logical_output_dir / "metadata.json"),
        "report": project_relative(logical_output_dir / "REPORT.md"),
        "gallery_index": pipeline_metadata["gallery_index"],
        "config_snapshot": project_relative(logical_output_dir / "config.yaml"),
        "evaluator_script_snapshot": project_relative(logical_output_dir / EVALUATOR_SCRIPT.name),
        "evaluator_script_snapshot_sha256": hashes["evaluator_script_sha256"],
        "run_plan": project_relative(logical_output_dir / "run_plan.json"),
        "state": project_relative(logical_output_dir / "STATE.json"),
    }
    metadata = {
        "analysis_id": config["analysis_id"],
        "split": split_name,
        "run_command": " ".join(sys.argv),
        "project_root": str(PROJECT_ROOT),
        "device": str(device),
        "git": get_git_metadata(),
        "versions": package_versions(),
        "hashes": hashes,
        "final_lock": final_lock,
        "locked_policy_dev": locked_policy_dev,
        "official_val_consumed_marker": consumed_marker_metadata,
        "manifest": manifest_metadata,
        "schedule": schedule_metadata,
        "classifiers": {"G_gate": gate_metadata, "T_cls": evaluator_metadata},
        "models": {
            "DeepJSCC": deepjscc_metadata,
            "edge_refiner": edge_metadata,
            "no_edge_refiner": no_edge_metadata,
        },
        "channel": {
            "type": config["channel"]["type"],
            "seeds": channel_seeds,
            "snrs": snrs,
            "deterministic_batch_seed_derivation": "sha256(channel_seed, snr, batch_start)",
        },
        "evaluation": {
            "num_unique_images": len(records),
            "num_rows": len(rows),
            "batch_size": int(config["evaluation"]["batch_size"]),
            "clean_thresholds": thresholds,
            "primary_clean_threshold": primary_threshold,
            "primary_snrs": sorted(primary_snrs),
            "quantize_png": bool(config["evaluation"]["quantize_png"]),
            "quantize_formula": "round(255*x)/255 after M0, raw refiner, candidate, and final",
            "lpips_available": lpips_model is not None,
            "lpips_error": lpips_error,
            "bootstrap_replicates": int(config["evaluation"]["bootstrap_replicates"]),
            "bootstrap_cluster": "image_id",
            "row_grid": row_grid_metadata,
            "original_classification_seconds": original_classification_seconds,
            "total_inference_and_original_classification_seconds": inference_seconds,
            **pipeline_metadata,
        },
        "receiver_visibility": {
            "G_gate_inputs": ["M0", "M2_edge_scheduled candidate"],
            "G_gate_uses_original": False,
            "G_gate_uses_true_label": False,
            "T_cls_participates_in_gate": False,
        },
        "artifacts": artifacts,
    }
    save_json(output_dir / "metadata.json", metadata)
    report = make_report(
        config,
        split_name,
        records,
        coverage_rows,
        arm_rows,
        bootstrap,
        success,
        metadata,
    )
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    completion_hashes = {
        name: sha256_file(output_dir / filename)
        for name, filename in {
            "per_sample_csv": "per_sample.csv",
            "summary_csv": "summary.csv",
            "summary_json": "summary.json",
            "metadata_json": "metadata.json",
            "report": "REPORT.md",
            "config_snapshot": "config.yaml",
            "evaluator_script_snapshot": EVALUATOR_SCRIPT.name,
            "run_plan": "run_plan.json",
        }.items()
    }
    write_completion_state(
        output_dir,
        logical_output_dir,
        split_name,
        "COMPLETE",
        artifact_hashes=completion_hashes,
    )
    if split_name == "official_val":
        if logical_output_dir.exists():
            raise FileExistsError(
                f"Canonical final directory appeared before atomic publication: {logical_output_dir}"
            )
        output_dir.rename(logical_output_dir)
    print(
        json.dumps(
            {
                "output_dir": project_relative(logical_output_dir),
                "split": split_name,
                "num_images": len(records),
                "num_rows": len(rows),
                "all_evaluated_gates_pass": success["all_evaluated_gates_pass"],
                "report": artifacts["report"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
