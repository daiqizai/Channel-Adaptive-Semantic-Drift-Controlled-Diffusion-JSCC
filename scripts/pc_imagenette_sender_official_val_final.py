#!/usr/bin/env python3
"""One-shot outcome-sealed official Imagenette audit for the frozen sender M3."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

SHA256_LENGTH = 64
CANONICAL_OUTPUT = "outputs/analysis/imagenette_supervised_final"
CANONICAL_STAGING = "outputs/analysis/.imagenette_supervised_final.staging"
CANONICAL_MARKER = "outputs/analysis/.imagenette_supervised_final.OFFICIAL_VAL_CONSUMED.json"
CANONICAL_CAPSULE_PREP = "outputs/analysis/.imagenette_supervised_final.capsule.preflight"
EXPECTED_SEEDS = [20260711, 20260712, 20260713]
EXPECTED_SNRS = [1, 4, 7, 13, 19]
MIN_FREE_BYTES = 8 * 1024**3
LPIPS_ROOT_SENTINEL = "__DISCOVERED_LPIPS_PACKAGE_ROOT__"


# This is deliberately an exact schema, not a user-extensible list.  The final
# lock must contain precisely these labels and each label must resolve to the
# path declared here.  The final YAML itself is intentionally absent: putting
# its own byte hash in final_lock.artifacts would create a self-reference.  Its
# immutable fields are covered by protocol_sha256 instead.
REQUIRED_ARTIFACT_PATHS: dict[str, str] = {
    "official_runner": "scripts/pc_imagenette_sender_official_val_final.py",
    "sender_audit": "scripts/pc_imagenette_sender_inbudget_awgn_audit.py",
    "supervised_audit_helper": "scripts/pc_imagenette_supervised_audit.py",
    "posterior_replication_helper": "scripts/pc_posterior_consistency_replication.py",
    "residual_refiner_helper": "scripts/s5_residual_refiner_pilot.py",
    "short_chain_diffusion_helper": "scripts/s10_short_chain_residual_shift_diffusion.py",
    "scaleup_seed_helper": "scripts/s13_export_coco_train2017_c8_scaleup.py",
    "official_manifest_builder": "scripts/s6_imagenette_supervised_clean_eval.py",
    "source_description_import": "scripts/s6_imagenette_source_semantic_description_eval.py",
    "coco_clip_import": "scripts/s5_coco_object_clip_clean_eval.py",
    "paired_replay_verifier": "scripts/pc_verify_sender_paired_replay.py",
    "deepjscc_adapter": "src/cadsd_jscc/deepjscc_adapter.py",
    "metrics_source": "src/cadsd_jscc/metrics.py",
    "semantic_sketch_source": "src/cadsd_jscc/semantic_sketch.py",
    "structure_source": "src/cadsd_jscc/structure.py",
    "cadsd_package_init": "src/cadsd_jscc/__init__.py",
    "deepjscc_model_source": "third_party/Deep-JSCC-PyTorch/model.py",
    "deepjscc_channel_source": "third_party/Deep-JSCC-PyTorch/channel.py",
    "audit_template": "configs/pc_imagenette_sender_crossmodel_triplet_official_val_template.yaml",
    "official_val_source_config": "configs/s6_imagenette_supervised_clean_eval.yaml",
    "source_config": "configs/s13_coco_train2017_c8_scaleup_export.yaml",
    "b1_config": "configs/s13_scaleup_b1_anchor_train.yaml",
    "diffusion_config": "configs/s14_scaleup_b1_anchored_diffusion.yaml",
    "deepjscc_checkpoint": (
        "outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt"
    ),
    "b1_checkpoint": "outputs/EXP-S13-001/checkpoints/best.pt",
    "diffusion_checkpoint": "outputs/EXP-S14-001/checkpoints/best.pt",
    "g_aux_checkpoint": (
        "outputs/analysis/imagenette_scratch_risk_classifier/G_aux/checkpoints/best.pt"
    ),
    "g_gate_checkpoint": (
        "outputs/analysis/imagenette_scratch_classifiers/G_gate/checkpoints/best.pt"
    ),
    "t_cls_checkpoint": (
        "outputs/analysis/imagenette_scratch_classifiers/T_cls/checkpoints/best.pt"
    ),
    "training_split_manifest": (
        "outputs/analysis/imagenette_scratch_classifiers/split_manifest.json"
    ),
    "lpips_alexnet_trunk": "outputs/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth",
    "lpips_calibration": f"{LPIPS_ROOT_SENTINEL}/weights/v0.1/alex.pth",
    "lpips_package_init": f"{LPIPS_ROOT_SENTINEL}/__init__.py",
    "lpips_package_metric": f"{LPIPS_ROOT_SENTINEL}/lpips.py",
    "lpips_package_backbone": f"{LPIPS_ROOT_SENTINEL}/pretrained_networks.py",
    "lpips_package_trainer": f"{LPIPS_ROOT_SENTINEL}/trainer.py",
    "imagenette_archive": "data/imagenette/imagenette2-320.tgz",
    "paired_replay_verification": (
        "outputs/analysis/pc_imagenette_sender_crossmodel_triplet_seed20260727_"
        "paired_replay_verification/verification.json"
    ),
    "preregistration": (
        "reports/posterior_sender_crossmodel_triplet_official_val_preregistration_2026-07-14.md"
    ),
}


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def protocol_sha256(config: dict[str, Any]) -> str:
    return canonical_json_sha256({key: value for key, value in config.items() if key != "final_lock"})


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return payload


def project_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def discover_lpips_package_root() -> Path:
    spec = importlib.util.find_spec("lpips")
    if spec is None or spec.origin is None:
        raise RuntimeError("the locked LPIPS package is not importable")
    root = Path(spec.origin).resolve().parent
    if not (root / "weights" / "v0.1" / "alex.pth").is_file():
        raise RuntimeError(f"LPIPS v0.1 Alex calibration is missing below {root}")
    return root


def required_artifact_paths() -> tuple[dict[str, Path], Path]:
    lpips_root = discover_lpips_package_root()
    resolved: dict[str, Path] = {}
    prefix = f"{LPIPS_ROOT_SENTINEL}/"
    for label, declared in REQUIRED_ARTIFACT_PATHS.items():
        if declared.startswith(prefix):
            path = lpips_root / declared[len(prefix) :]
        else:
            path = resolve(declared)
        resolved[label] = path.resolve()
    return resolved, lpips_root


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        try:
            os.fchmod(descriptor, mode)
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:  # pragma: no cover - defensive short-write guard
                    raise OSError(f"short write while creating {path}")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload: Any, mode: int = 0o644) -> None:
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    atomic_write_bytes(path, encoded, mode=mode)


def atomic_write_text(path: Path, payload: str, mode: int = 0o644) -> None:
    atomic_write_bytes(path, payload.encode("utf-8"), mode=mode)


def copy_verified_file(source: Path, destination: Path, expected_digest: str | None) -> str:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.copy.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        shutil.copy2(source, temporary)
        copied_digest = sha256_file(temporary)
        if expected_digest is not None and copied_digest != expected_digest:
            raise RuntimeError(f"artifact changed while copying: {source}")
        os.replace(temporary, destination)
        return copied_digest
    finally:
        temporary.unlink(missing_ok=True)


def capsule_relative_path(source: Path, lpips_root: Path) -> Path:
    source = source.resolve()
    try:
        return source.relative_to(ROOT.resolve())
    except ValueError:
        try:
            return Path("vendor/lpips") / source.relative_to(lpips_root.resolve())
        except ValueError as exc:
            raise RuntimeError(f"locked artifact has no capsule mapping: {source}") from exc


def capsule_artifact_path(
    capsule: Path, source: Path, lpips_root: Path
) -> Path:
    return capsule / capsule_relative_path(source, lpips_root)


def iter_python_sources(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.py")
        if path.is_file() and "__pycache__" not in path.parts
    )


def build_run_capsule(
    capsule: Path,
    required_paths: dict[str, Path],
    verified: dict[str, str],
    lpips_root: Path,
) -> dict[str, Any]:
    if capsule.exists():
        raise FileExistsError(capsule)
    temporary = capsule.with_name(f".{capsule.name}.building.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        # Preserve the project-relative import layout.  Every Python dependency
        # is snapshotted; method-critical files are additionally required by the
        # exact final-lock schema below.
        for tree in (ROOT / "scripts", ROOT / "src"):
            for source in iter_python_sources(tree):
                destination = temporary / source.relative_to(ROOT)
                digest = copy_verified_file(source, destination, None)
                critical = [
                    label
                    for label, required in required_paths.items()
                    if required == source.resolve()
                ]
                for label in critical:
                    if digest != verified[label]:
                        raise RuntimeError(f"capsule copy differs from locked {label}")

        # Copy the complete LPIPS package so the metric implementation and its
        # calibration travel together.  Bytecode/cache files are excluded.
        for source in sorted(lpips_root.rglob("*")):
            if (
                not source.is_file()
                or "__pycache__" in source.parts
                or source.suffix in {".pyc", ".pyo"}
            ):
                continue
            destination = temporary / "vendor/lpips" / source.relative_to(lpips_root)
            copy_verified_file(source, destination, None)

        for label, source in required_paths.items():
            destination = capsule_artifact_path(temporary, source, lpips_root)
            if destination.is_file():
                digest = sha256_file(destination)
                if digest != verified[label]:
                    raise RuntimeError(f"capsule artifact differs from locked {label}")
            else:
                copy_verified_file(source, destination, verified[label])

        capsule_files = {
            path.relative_to(temporary).as_posix(): sha256_file(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "format_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "locked_artifacts": {
                label: {
                    "source_path": str(required_paths[label]),
                    "capsule_path": capsule_relative_path(required_paths[label], lpips_root).as_posix(),
                    "sha256": verified[label],
                }
                for label in sorted(required_paths)
            },
            "all_capsule_files": capsule_files,
        }
        atomic_write_json(temporary / "CAPSULE_MANIFEST.json", manifest)
        for path in sorted(temporary.rglob("*"), reverse=True):
            if path.is_file():
                path.chmod(0o444)
            elif path.is_dir():
                path.chmod(0o555)
        temporary.chmod(0o555)
        fsync_tree(temporary)
        os.replace(temporary, capsule)
        fsync_directory(capsule.parent)
        return manifest
    except BaseException:
        if temporary.exists():
            for path in sorted(temporary.rglob("*"), reverse=True):
                try:
                    path.chmod(0o755 if path.is_dir() else 0o644)
                except OSError:
                    pass
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def child_environment(capsule: Path, marker_digest: str | None = None) -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
    ):
        environment.pop(name, None)
    environment["TORCH_HOME"] = str(capsule / "outputs/cache/torch")
    environment["PYTHONPATH"] = str(capsule / "vendor")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    if marker_digest is not None:
        environment["CADSD_OFFICIAL_VAL_AUTHORIZATION"] = marker_digest
    return environment


def run_environment_preflight(
    capsule: Path,
    required_paths: dict[str, Path],
    lpips_root: Path,
    output_parent: Path,
) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("cuda:0 is unavailable for the one-shot official audit")
    probe = torch.empty((1,), device="cuda:0")
    probe.add_(1)
    torch.cuda.synchronize(0)
    del probe

    checkpoint_labels = (
        "deepjscc_checkpoint",
        "b1_checkpoint",
        "diffusion_checkpoint",
        "g_aux_checkpoint",
        "g_gate_checkpoint",
        "t_cls_checkpoint",
    )
    checkpoint_metadata: dict[str, Any] = {}
    for label in checkpoint_labels:
        path = capsule_artifact_path(capsule, required_paths[label], lpips_root)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint_metadata[label] = {
            "path": str(path),
            "type": type(payload).__name__,
            "size_bytes": path.stat().st_size,
        }
        del payload

    calibration = capsule_artifact_path(
        capsule, required_paths["lpips_calibration"], lpips_root
    )
    lpips_probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, pathlib, torch, lpips; "
                f"root=pathlib.Path({str(capsule / 'vendor')!r}).resolve(); "
                "module=pathlib.Path(lpips.__file__).resolve(); "
                "assert root in module.parents, (root, module); "
                f"model=lpips.LPIPS(net='alex', version='0.1', model_path={str(calibration)!r}, verbose=False).to('cuda:0'); "
                "torch.cuda.synchronize(0); "
                "print(json.dumps({'lpips_module':str(module),'device':'cuda:0'}))"
            ),
        ],
        cwd=ROOT,
        env=child_environment(capsule),
        text=True,
        capture_output=True,
        check=False,
    )
    if lpips_probe.returncode != 0:
        raise RuntimeError(
            "fixed-cache LPIPS CUDA preflight failed: "
            + (lpips_probe.stderr.strip() or lpips_probe.stdout.strip())
        )

    output_parent.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(output_parent)
    if disk.free < MIN_FREE_BYTES:
        raise RuntimeError(
            f"official output filesystem has only {disk.free} free bytes; "
            f"requires at least {MIN_FREE_BYTES}"
        )
    probe_source = output_parent / f".official_val_rename_probe.{os.getpid()}.source"
    probe_target = output_parent / f".official_val_rename_probe.{os.getpid()}.target"
    if probe_source.exists() or probe_target.exists():
        raise FileExistsError(probe_source if probe_source.exists() else probe_target)
    try:
        probe_source.mkdir()
        atomic_write_text(probe_source / "probe.txt", "official-val preflight only\n")
        fsync_directory(probe_source)
        os.replace(probe_source, probe_target)
        fsync_directory(output_parent)
    finally:
        shutil.rmtree(probe_source, ignore_errors=True)
        shutil.rmtree(probe_target, ignore_errors=True)
        fsync_directory(output_parent)

    return {
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "free_bytes_after_capsule": disk.free,
        "minimum_required_free_bytes": MIN_FREE_BYTES,
        "checkpoint_loads": checkpoint_metadata,
        "lpips_probe": json.loads(lpips_probe.stdout.strip().splitlines()[-1]),
        "torch_home": str(capsule / "outputs/cache/torch"),
    }


def load_capsule_manifest_builder(capsule: Path):
    path = capsule / REQUIRED_ARTIFACT_PATHS["official_manifest_builder"]
    module_name = "cadsd_official_capsule_manifest_builder"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load capsule manifest builder: {path}")
    old_path = list(sys.path)
    old_dont_write = sys.dont_write_bytecode
    try:
        sys.path[:0] = [str(capsule / "scripts"), str(capsule / "src")]
        sys.dont_write_bytecode = True
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_dont_write
    return module.build_sealed_official_val_manifest


def fsync_tree(root: Path) -> None:
    files = [path for path in root.rglob("*") if path.is_file()]
    for path in files:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        fsync_directory(path)
    fsync_directory(root)


def remove_tree_force(path: Path) -> None:
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        try:
            child.chmod(0o755 if child.is_dir() else 0o644)
        except OSError:
            pass
    path.chmod(0o755)
    shutil.rmtree(path)


def validate_audit_template(template: dict[str, Any]) -> None:
    if template.get("analysis_id") != "ANALYSIS-PC-SENDER-CROSSMODEL-OFFICIAL-VAL-001":
        raise RuntimeError("official template analysis_id mismatch")
    exact = {
        "channel_seeds": EXPECTED_SEEDS,
        "snrs": EXPECTED_SNRS,
        "primary_snrs": [1, 4, 7],
        "batch_size": 8,
        "proximal_steps": 3,
        "normalized_step_size": 0.001,
        "clean_confidence_threshold": 0.5,
        "minimum_clean_images": 2500,
        "reference_mode": "paired_unpunctured_same_noise",
    }
    for key, expected in exact.items():
        if template.get(key) != expected:
            raise RuntimeError(f"official template changed frozen field {key}")
    imagenette = template["imagenette"]
    if (
        imagenette.get("required_split") != "official_val"
        or imagenette.get("official_val_accessed") is not True
        or int(imagenette.get("official_val_expected_count", -1)) != 3925
        or imagenette.get("verify_official_val_content_sha256") is not True
        or imagenette.get("official_val_manifest") != "__RUNTIME_OFFICIAL_VAL_MANIFEST__"
        or imagenette.get("official_val_manifest_sha256")
        != "__RUNTIME_OFFICIAL_VAL_MANIFEST_SHA256__"
        or imagenette.get("outcome_consumed_marker")
        != "__RUNTIME_OUTCOME_CONSUMED_MARKER__"
        or imagenette.get("outcome_consumed_marker_sha256")
        != "__RUNTIME_OUTCOME_CONSUMED_MARKER_SHA256__"
        or imagenette.get("official_val_staging_root")
        != "__RUNTIME_OFFICIAL_VAL_STAGING_ROOT__"
    ):
        raise RuntimeError("official template Imagenette lock fields changed")
    if template.get("output_dir") != "__RUNTIME_AUDIT_OUTPUT__":
        raise RuntimeError("official template output sentinel changed")
    if template["success_criteria"].get("minimum_clean_images_per_class") != 150:
        raise RuntimeError("official template per-class clean minimum changed")
    if template.get("statistical_inference") != {
        "cluster_unit": "sample_id",
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 161803,
        "retain_all_seed_snr_rows_per_resampled_image": True,
    }:
        raise RuntimeError("official template paired cluster-bootstrap contract changed")


def validate_lock(
    config: dict[str, Any], config_path: Path
) -> tuple[dict[str, str], dict[str, Path], Path]:
    if config.get("analysis_id") != "ANALYSIS-PC-SENDER-CROSSMODEL-OFFICIAL-VAL-001":
        raise RuntimeError("final config analysis_id mismatch")
    expected_paths = {
        "canonical_output_dir": CANONICAL_OUTPUT,
        "staging_output_dir": CANONICAL_STAGING,
        "consumed_marker": CANONICAL_MARKER,
    }
    for key, expected in expected_paths.items():
        if str(config.get(key)) != expected:
            raise RuntimeError(f"final config changed canonical {key}")
    declared_inputs = {
        "audit_template": REQUIRED_ARTIFACT_PATHS["audit_template"],
        "official_val_source_config": REQUIRED_ARTIFACT_PATHS["official_val_source_config"],
        "training_split_manifest": REQUIRED_ARTIFACT_PATHS["training_split_manifest"],
        "audit_script": REQUIRED_ARTIFACT_PATHS["sender_audit"],
        "preregistration": REQUIRED_ARTIFACT_PATHS["preregistration"],
    }
    for key, expected_path in declared_inputs.items():
        if str(config.get(key)) != expected_path:
            raise RuntimeError(f"final config changed frozen input path {key}")
    if config.get("device") != "cuda:0":
        raise RuntimeError("official audit device must remain cuda:0")
    expected = config.get("expected", {})
    if (
        int(expected.get("images", -1)) != 3925
        or list(expected.get("channel_seeds", [])) != EXPECTED_SEEDS
        or list(expected.get("snrs", [])) != EXPECTED_SNRS
        or int(expected.get("rows", -1)) != 58875
        or expected.get("official_val_atime_probe_incident_recorded") is not True
    ):
        raise RuntimeError("official expected population/grid lock changed")
    lock = config.get("final_lock")
    if not isinstance(lock, dict) or lock.get("unlocked") is not True:
        raise RuntimeError("official final_lock is not explicitly unlocked")
    if not lock.get("locked_at_utc"):
        raise RuntimeError("official final_lock lacks locked_at_utc")
    actual_protocol = protocol_sha256(config)
    if str(lock.get("protocol_sha256")) != actual_protocol:
        raise RuntimeError("official protocol SHA-256 mismatch")
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise RuntimeError("official final_lock artifact map is empty")
    required_paths, lpips_root = required_artifact_paths()
    if set(artifacts) != set(required_paths):
        missing = sorted(set(required_paths) - set(artifacts))
        extra = sorted(set(artifacts) - set(required_paths))
        raise RuntimeError(
            f"official final_lock artifact schema mismatch: missing={missing}, extra={extra}"
        )
    verified: dict[str, str] = {}
    for label in sorted(required_paths):
        entry = artifacts[label]
        if not isinstance(entry, dict):
            raise RuntimeError(f"invalid artifact lock entry: {label}")
        path = resolve(str(entry.get("path", "")))
        expected_path = required_paths[label]
        if path.resolve() != expected_path:
            raise RuntimeError(
                f"locked artifact path mismatch for {label}: {path} != {expected_path}"
            )
        expected_digest = str(entry.get("sha256", ""))
        if len(expected_digest) != SHA256_LENGTH or not path.is_file():
            raise RuntimeError(f"invalid or missing locked artifact: {label}")
        actual_digest = sha256_file(path)
        if actual_digest != expected_digest:
            raise RuntimeError(f"locked artifact changed: {label}")
        verified[label] = actual_digest
    template_path = required_paths["audit_template"]
    validate_audit_template(load_yaml(template_path))
    repo = resolve("third_party/Deep-JSCC-PyTorch")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    ).strip()
    if head != str(lock.get("deepjscc_repo_head")) or dirty:
        raise RuntimeError("locked DeepJSCC repository provenance changed")
    if sha256_file(config_path) == "":  # pragma: no cover - forces explicit config readability
        raise RuntimeError("unreadable final config")
    return verified, required_paths, lpips_root


def create_marker(path: Path, config: dict[str, Any], verified: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state": "OFFICIAL_VAL_OUTCOME_CONSUMED_BEFORE_MODEL_INFERENCE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_id": config["analysis_id"],
        "protocol_sha256": config["final_lock"]["protocol_sha256"],
        "locked_artifact_sha256": verified,
        "byte_access_incident": (
            "Official-val method outcomes were sealed, but prior archive integrity checks and a "
            "2026-07-14 read-only ripgrep binary probe had already opened image bytes."
        ),
        "note": "Failure or interruption consumes the one-shot outcome audit; rerun is forbidden.",
    }
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        encoded = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:  # pragma: no cover - defensive short-write guard
                raise OSError(f"short write while creating consumed marker {path}")
            offset += written
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    fsync_directory(path.parent)


def write_json(path: Path, payload: Any) -> None:
    atomic_write_json(path, payload)


def artifact_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"STATE.json", "artifact_hashes.json"}
    }


def validate_completed_audit(audit_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    metrics_path = audit_dir / "metrics.json"
    per_sample_path = audit_dir / "per_sample.csv"
    seed_summary_path = audit_dir / "seed_snr_summary.csv"
    for path in (metrics_path, per_sample_path, seed_summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    aggregate = metrics["aggregate"]
    expected = config["expected"]
    if (
        aggregate.get("image_population") != "official_val"
        or aggregate.get("official_val_accessed") is not True
        or int(aggregate.get("image_population_images", -1)) != int(expected["images"])
        or int(aggregate.get("rows", -1)) != int(expected["rows"])
        or list(aggregate.get("channel_seeds", [])) != list(expected["channel_seeds"])
        or aggregate.get("reference_mode") != "paired_unpunctured_same_noise"
    ):
        raise RuntimeError("completed official audit metadata/grid mismatch")
    with per_sample_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        keys = {
            (int(row["channel_seed"]), float(row["snr_db"]), row["sample_id"])
            for row in reader
        }
    if len(keys) != int(expected["rows"]):
        raise RuntimeError("official per-sample CSV lacks the complete unique grid")
    with seed_summary_path.open(encoding="utf-8", newline="") as handle:
        if sum(1 for _ in csv.DictReader(handle)) != len(EXPECTED_SEEDS) * len(EXPECTED_SNRS):
            raise RuntimeError("official seed-by-SNR summary is incomplete")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/pc_imagenette_sender_crossmodel_triplet_official_val_final.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.device != "cuda:0":
        raise RuntimeError("official outcome audit is frozen to cuda:0")
    config_path = resolve(args.config)
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    if not isinstance(config, dict):
        raise TypeError(f"YAML root must be a mapping: {config_path}")
    verified, required_paths, lpips_root = validate_lock(config, config_path)

    canonical = resolve(config["canonical_output_dir"])
    staging = resolve(config["staging_output_dir"])
    marker = resolve(config["consumed_marker"])
    capsule_preflight = resolve(CANONICAL_CAPSULE_PREP)
    for path in (canonical, staging, marker, capsule_preflight):
        if path.exists():
            raise FileExistsError(path)

    try:
        capsule_manifest = build_run_capsule(
            capsule_preflight, required_paths, verified, lpips_root
        )
        preflight = run_environment_preflight(
            capsule_preflight,
            required_paths,
            lpips_root,
            canonical.parent,
        )

        # The capsule and every non-data prerequisite are complete before the
        # irrevocable marker.  Merely preparing this staging directory does not
        # consume the outcome; official bytes are still untouched by this run.
        staging.mkdir(parents=True, exist_ok=False)
        run_capsule = staging / "run_capsule"
        os.replace(capsule_preflight, run_capsule)
        fsync_directory(staging)
        atomic_write_json(
            staging / "STATE.json",
            {
                "state": "PREPARED_NO_OFFICIAL_OUTCOME_ACCESSED",
                "analysis_id": config["analysis_id"],
                "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
                "official_val_outcome_consumed": False,
            },
        )
        create_marker(marker, config, verified)
        atomic_write_json(
            staging / "STATE.json",
            {
                "state": "IN_PROGRESS",
                "analysis_id": config["analysis_id"],
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "official_val_outcome_consumed": True,
                "run_capsule_manifest_sha256": sha256_file(
                    run_capsule / "CAPSULE_MANIFEST.json"
                ),
            },
        )

        # Load the manifest builder itself from the immutable capsule.  Only
        # after the consumed marker exists do we point it at the extracted
        # official validation population.
        source_config = load_yaml(
            capsule_artifact_path(
                run_capsule, required_paths["official_val_source_config"], lpips_root
            )
        )
        source_config = copy.deepcopy(source_config)
        source_config["data"]["archive"] = str(
            capsule_artifact_path(
                run_capsule, required_paths["imagenette_archive"], lpips_root
            )
        )
        for key in ("root", "train_dir", "val_dir"):
            source_config["data"][key] = str(
                resolve(str(source_config["data"][key])).resolve()
            )
        build_sealed_official_val_manifest = load_capsule_manifest_builder(run_capsule)
        training_manifest = capsule_artifact_path(
            run_capsule, required_paths["training_split_manifest"], lpips_root
        )
        records, official_manifest, manifest_metadata = build_sealed_official_val_manifest(
            training_manifest, source_config
        )
        if len(records) != int(config["expected"]["images"]):
            raise RuntimeError("official manifest population count mismatch")
        manifest_path = staging / str(config["official_val_manifest_name"])
        atomic_write_json(manifest_path, official_manifest)
        manifest_digest = sha256_file(manifest_path)

        template = load_yaml(
            capsule_artifact_path(run_capsule, required_paths["audit_template"], lpips_root)
        )
        validate_audit_template(template)
        template["source_config"] = str(
            capsule_artifact_path(run_capsule, required_paths["source_config"], lpips_root)
        )
        template["b1_config"] = str(
            capsule_artifact_path(run_capsule, required_paths["b1_config"], lpips_root)
        )
        template["b1_checkpoint"] = str(
            capsule_artifact_path(run_capsule, required_paths["b1_checkpoint"], lpips_root)
        )
        template["diffusion_config"] = str(
            capsule_artifact_path(run_capsule, required_paths["diffusion_config"], lpips_root)
        )
        template["diffusion_checkpoint"] = str(
            capsule_artifact_path(
                run_capsule, required_paths["diffusion_checkpoint"], lpips_root
            )
        )
        template["deepjscc_checkpoint"] = str(
            capsule_artifact_path(
                run_capsule, required_paths["deepjscc_checkpoint"], lpips_root
            )
        )
        template["imagenette"]["split_manifest"] = str(training_manifest)
        template["imagenette"]["evaluator_checkpoint"] = str(
            capsule_artifact_path(run_capsule, required_paths["t_cls_checkpoint"], lpips_root)
        )
        template["controller"]["checkpoint"] = str(
            capsule_artifact_path(run_capsule, required_paths["g_aux_checkpoint"], lpips_root)
        )
        template["controller"]["receiver_guard"]["checkpoint"] = str(
            capsule_artifact_path(run_capsule, required_paths["g_gate_checkpoint"], lpips_root)
        )
        audit_dir = staging / "audit"
        template["imagenette"]["official_val_manifest"] = str(manifest_path)
        template["imagenette"]["official_val_manifest_sha256"] = manifest_digest
        marker_digest = sha256_file(marker)
        template["imagenette"]["outcome_consumed_marker"] = str(marker)
        template["imagenette"]["outcome_consumed_marker_sha256"] = marker_digest
        template["imagenette"]["official_val_staging_root"] = str(staging)
        template["output_dir"] = str(audit_dir)
        resolved_config_path = staging / str(config["resolved_audit_config_name"])
        atomic_write_text(
            resolved_config_path,
            yaml.safe_dump(template, sort_keys=False, allow_unicode=True),
        )
        atomic_write_json(staging / "official_val_manifest_metadata.json", manifest_metadata)
        atomic_write_bytes(staging / "final_lock_config.yaml", config_bytes)

        environment = child_environment(run_capsule, marker_digest)
        run_log = staging / "run.log"
        with run_log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(
                        capsule_artifact_path(
                            run_capsule, required_paths["sender_audit"], lpips_root
                        )
                    ),
                    "--config",
                    str(resolved_config_path),
                    "--device",
                    args.device,
                ],
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"official audit child failed with code {completed.returncode}")
        metrics = validate_completed_audit(audit_dir, config)
        atomic_write_json(
            staging / "environment.json",
            {
                "python": sys.version,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": args.device,
                "proxy_variables_cleared_for_child": True,
                "torch_home_for_child": environment["TORCH_HOME"],
                "lpips_version": importlib.metadata.version("lpips"),
                "torchvision_version": importlib.metadata.version("torchvision"),
                "pillow_version": importlib.metadata.version("Pillow"),
                "scipy_version": importlib.metadata.version("scipy"),
                "pyyaml_version": importlib.metadata.version("PyYAML"),
                "preflight": preflight,
                "capsule_manifest_file_count": len(capsule_manifest["all_capsule_files"]),
            },
        )
        relocation = {
            "format_version": 1,
            "note": (
                "executed paths are historical staging paths; published paths are obtained "
                "by the exact prefix mappings below"
            ),
            "prefix_mappings": [
                {
                    "executed_prefix": str(staging),
                    "published_prefix": str(canonical),
                },
                {
                    "preflight_prefix": str(capsule_preflight),
                    "published_prefix": str(canonical / "run_capsule"),
                },
            ],
            "published_locations": {
                "audit": str(canonical / "audit"),
                "official_val_manifest": str(
                    canonical / str(config["official_val_manifest_name"])
                ),
                "resolved_executed_config": str(
                    canonical / str(config["resolved_audit_config_name"])
                ),
                "run_capsule": str(canonical / "run_capsule"),
            },
        }
        atomic_write_json(staging / "PATH_RELOCATION.json", relocation)
        hashes = artifact_hashes(staging)
        atomic_write_json(staging / "artifact_hashes.json", hashes)
        atomic_write_json(
            staging / "STATE.json",
            {
                "state": "COMPLETE",
                "analysis_id": config["analysis_id"],
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "official_val_outcome_consumed": True,
                "verdict": metrics["verdict"],
                "manifest_sha256": manifest_digest,
                "artifact_hashes_file": "artifact_hashes.json",
                "path_relocation_file": "PATH_RELOCATION.json",
            },
        )
        fsync_tree(staging)
        os.replace(staging, canonical)
        fsync_directory(canonical.parent)
        print(
            json.dumps(
                {
                    "state": "COMPLETE",
                    "output_dir": project_relative(canonical),
                    "verdict": metrics["verdict"],
                },
                ensure_ascii=False,
            )
        )
    except BaseException as exc:
        outcome_consumed = marker.exists()
        if outcome_consumed and staging.is_dir():
            atomic_write_json(
                staging / "STATE.json",
                {
                    "state": "FAILED_CONSUMED_NO_RERUN",
                    "analysis_id": config["analysis_id"],
                    "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "official_val_outcome_consumed": True,
                },
            )
        elif not outcome_consumed:
            # No official outcome was opened, so preflight artifacts are safe to
            # remove and do not create a false consumed state.
            if staging.exists():
                remove_tree_force(staging)
            if capsule_preflight.exists():
                remove_tree_force(capsule_preflight)
        raise


if __name__ == "__main__":
    main()
