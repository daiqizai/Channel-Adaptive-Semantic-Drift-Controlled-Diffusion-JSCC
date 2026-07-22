#!/usr/bin/env python3
"""Fail-closed asset, population, channel-noise, and rate audit for S30."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cadsd_jscc.external_common import (  # noqa: E402
    canonical_noise_sha256,
    canonical_standard_normal,
    complex_cbr,
)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_yaml(value: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(resolve(value).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"YAML root must be a mapping: {value}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.iterdir()):
        if item.is_file():
            digest.update(item.name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def recursive_tree_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for item in sorted(path.rglob("*")):
        if item.is_file():
            digest.update(item.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(bytes.fromhex(sha256_file(item)))
            count += 1
    return digest.hexdigest(), count


def require_sha(value: str | Path, expected: str) -> Path:
    path = resolve(value)
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != str(expected):
        raise RuntimeError(f"SHA-256 mismatch: {path}: {observed} != {expected}")
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def validate_population(config: dict[str, Any]) -> tuple[list[dict[str, Any]], Path]:
    reference_path = require_sha(
        config["inputs"]["population_reference"],
        config["inputs"]["population_reference_sha256"],
    )
    reference = load_yaml(reference_path)
    population = reference["population"]
    if bool(reference.get("official_val_accessed")):
        raise RuntimeError("frozen population records official validation access")
    if int(population["expected_sample_count"]) != int(
        config["population"]["expected_sample_count"]
    ):
        raise RuntimeError("population count changed")
    manifest_path = require_sha(
        config["inputs"]["split_manifest"], config["inputs"]["split_manifest_sha256"]
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bool(manifest.get("official_val_accessed")):
        raise RuntimeError("split manifest records official validation access")
    source_root = resolve(manifest["source_train_root"])
    by_id = {
        str(item["sample_id"]): item
        for item in manifest["samples"]
        if str(item["split"]) == str(population["required_split"])
    }
    samples: list[dict[str, Any]] = []
    for frozen in population["samples"]:
        sample_id = str(frozen["sample_id"])
        if sample_id not in by_id:
            raise RuntimeError(f"missing frozen sample: {sample_id}")
        item = dict(by_id[sample_id])
        if int(item["class_idx"]) != int(frozen["class_idx"]):
            raise RuntimeError(f"class mismatch: {sample_id}")
        image = source_root / str(item["relative_path"])
        if sha256_file(image) != str(frozen["content_sha256"]):
            raise RuntimeError(f"content hash mismatch: {sample_id}")
        item["path"] = str(image)
        samples.append(item)
    expected = int(config["population"]["expected_sample_count"])
    if len(samples) != expected or len({row["sample_id"] for row in samples}) != expected:
        raise RuntimeError("population size/uniqueness gate failed")
    return samples, reference_path


def validate_current_noise(config: dict[str, Any]) -> dict[str, Any]:
    current_path = require_sha(
        config["inputs"]["current_per_sample"],
        config["inputs"]["current_per_sample_sha256"],
    )
    rows = read_csv(current_path)
    b1_path = require_sha(
        config["inputs"]["b1_per_sample"],
        config["inputs"]["b1_per_sample_sha256"],
    )
    b1_rows = read_csv(b1_path)
    require_sha(config["inputs"]["s28_summary"], config["inputs"]["s28_summary_sha256"])
    require_sha(
        config["inputs"]["t_cls_checkpoint"],
        config["inputs"]["t_cls_checkpoint_sha256"],
    )
    expected = int(config["population"]["expected_rows_full"])
    if len(rows) != expected or len(b1_rows) != expected:
        raise RuntimeError(
            f"S28 current/B1 row counts changed: {len(rows)}/{len(b1_rows)} != {expected}"
        )
    full_symbols = int(config["channel"]["canonical_noise_real_symbols"])
    used_symbols = int(config["channel"]["diffjscc_uses_prefix_real_symbols"])
    mismatches = 0
    first_prefix_sha = ""
    seen: set[tuple[str, int, float]] = set()
    current_noise_by_key: dict[tuple[str, int, float], str] = {}
    for row in rows:
        key = (str(row["sample_id"]), int(row["base_seed"]), float(row["snr_db"]))
        if key in seen:
            raise RuntimeError(f"duplicate S28 key: {key}")
        seen.add(key)
        current_noise_by_key[key] = str(row["canonical_noise_sha256"])
        noise = canonical_standard_normal(key[1], key[0], key[2], full_symbols)
        if canonical_noise_sha256(noise) != row["canonical_noise_sha256"]:
            mismatches += 1
        if not first_prefix_sha:
            first_prefix_sha = canonical_noise_sha256(noise[:used_symbols])
    b1_by_key = {
        (str(row["sample_id"]), int(row["base_seed"]), float(row["snr_db"])): row
        for row in b1_rows
    }
    if set(b1_by_key) != seen:
        raise RuntimeError("S28 current/B1 key sets changed")
    if any(
        b1_by_key[key]["canonical_noise_sha256"] != current_noise_by_key[key]
        for key in seen
    ):
        raise RuntimeError("S28 current/B1 canonical-noise hashes differ")
    if mismatches:
        raise RuntimeError(f"canonical full-noise SHA mismatches: {mismatches}")
    return {
        "rows": len(rows),
        "unique_keys": len(seen),
        "b1_rows": len(b1_rows),
        "b1_key_set_exact": True,
        "b1_noise_sha_matches_current": True,
        "full_noise_sha_mismatches": mismatches,
        "first_diffjscc_prefix_sha256": first_prefix_sha,
    }


def checkpoint_state(config: dict[str, Any], allow_incomplete: bool) -> dict[str, Any]:
    spec = config["assets"]
    path = resolve(spec["checkpoint_file"])
    expected_bytes = int(spec["checkpoint_expected_bytes"])
    if path.is_file():
        size = path.stat().st_size
        if size != expected_bytes:
            raise RuntimeError(f"checkpoint size mismatch: {size} != {expected_bytes}")
        observed_sha = sha256_file(path)
        if observed_sha != str(spec["checkpoint_sha256"]):
            raise RuntimeError("checkpoint SHA-256 mismatch")
        return {"status": "complete", "bytes": size, "sha256": observed_sha}
    download_dir = resolve(spec["checkpoint_directory"]) / ".cache" / "huggingface" / "download"
    partials = sorted(download_dir.glob("*.incomplete")) if download_dir.is_dir() else []
    durable_partial = resolve(spec["checkpoint_directory"]) / "model.ckpt.durable.part"
    if durable_partial.is_file():
        partials.append(durable_partial)
    # Xet and standard HTTP resumptions may leave alternative partial formats
    # for the same logical file.  They are not additive; report the largest.
    partial_bytes = max((item.stat().st_size for item in partials), default=0)
    if not allow_incomplete:
        raise FileNotFoundError(path)
    return {
        "status": "downloading",
        "bytes": partial_bytes,
        "expected_bytes": expected_bytes,
        "fraction": partial_bytes / expected_bytes,
        "partial_files": [item.name for item in partials],
    }


def blip_weight_state(config: dict[str, Any], allow_incomplete: bool) -> dict[str, Any]:
    spec = config["assets"]
    directory = resolve(spec["blip_weights_directory"])
    cache = directory / ".cache" / "huggingface" / "download"
    files: list[dict[str, Any]] = []
    all_complete = True
    for expected in spec["blip_weight_files"]:
        name = str(expected["name"])
        expected_bytes = int(expected["bytes"])
        expected_sha = str(expected["sha256"])
        path = directory / name
        if path.is_file():
            size = path.stat().st_size
            if size != expected_bytes:
                raise RuntimeError(
                    f"BLIP2 weight size mismatch: {name}: {size} != {expected_bytes}"
                )
            observed_sha = sha256_file(path)
            if observed_sha != expected_sha:
                raise RuntimeError(f"BLIP2 weight SHA-256 mismatch: {name}")
            files.append(
                {"name": name, "status": "complete", "bytes": size, "sha256": observed_sha}
            )
            continue

        all_complete = False
        # Standard HTTP names partials with the LFS SHA, while Xet may use a
        # different resume name.  Restrict by the expected SHA where possible
        # and never add alternative partials for one logical file together.
        matching = sorted(cache.glob(f"*.{expected_sha}.*.incomplete")) if cache.is_dir() else []
        durable_wget_partial = directory / f"{name}.wget.part"
        if durable_wget_partial.is_file():
            matching.append(durable_wget_partial)
        durable_curl_partial = directory / f"{name}.curl.part"
        if durable_curl_partial.is_file():
            matching.append(durable_curl_partial)
        range_parts = sorted(directory.glob(f"{name}.range*.part"))
        segmented_bytes = sum(item.stat().st_size for item in range_parts)
        partial_bytes = max(
            [item.stat().st_size for item in matching] + [segmented_bytes],
            default=0,
        )
        if partial_bytes > expected_bytes:
            raise RuntimeError(f"BLIP2 partial byte accounting exceeds expected size: {name}")
        files.append(
            {
                "name": name,
                "status": "downloading" if partial_bytes else "absent",
                "bytes": partial_bytes,
                "expected_bytes": expected_bytes,
                "fraction": partial_bytes / expected_bytes,
                "partial_files": [item.name for item in matching + range_parts],
                "segmented_partial_bytes": segmented_bytes,
            }
        )
    if not all_complete and not allow_incomplete:
        missing = [item["name"] for item in files if item["status"] != "complete"]
        raise FileNotFoundError(f"incomplete exact BLIP2 weights: {missing}")
    expected_total = sum(int(item["bytes"]) for item in spec["blip_weight_files"])
    observed_total = sum(int(item["bytes"]) for item in files)
    return {
        "status": "complete" if all_complete else "downloading",
        "bytes": observed_total,
        "expected_bytes": expected_total,
        "fraction": observed_total / expected_total,
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/s30_diffjscc_external_comparison.yaml")
    parser.add_argument("--allow-incomplete-checkpoint", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    config_path = resolve(args.config)
    config = load_yaml(config_path)
    if config["protocol"]["status"] != "preregistered_before_any_diffjscc_reconstruction":
        raise RuntimeError("S30 protocol is not in the preregistered state")
    if config["protocol"]["official_imagenette_validation_accessed"] is not False:
        raise RuntimeError("official validation must stay sealed")

    repo = resolve(config["assets"]["repository"])
    observed_commit = git(repo, "rev-parse", "HEAD")
    if observed_commit != str(config["assets"]["repository_commit"]):
        raise RuntimeError("DiffJSCC source commit changed")
    tracked_status = git(repo, "status", "--short", "--untracked-files=no")
    tracked_lines = [line for line in tracked_status.splitlines() if line.strip()]
    # The author repository unfortunately tracks generated ``__pycache__`` files;
    # importing the untouched source can refresh those bytecode blobs.  Audit
    # algorithm/config changes strictly while recording bytecode-only churn.
    algorithm_changes = [
        line
        for line in tracked_lines
        if "__pycache__/" not in line and not line.rstrip().endswith(".pyc")
    ]
    if algorithm_changes:
        raise RuntimeError(f"tracked author algorithm source is modified: {algorithm_changes}")

    model_config_path = require_sha(
        config["assets"]["model_config"], config["assets"]["model_config_sha256"]
    )
    require_sha(repo / "README.md", config["assets"]["author_readme_sha256"])
    blip_metadata = resolve(config["assets"]["blip_metadata_directory"])
    if metadata_tree_sha256(blip_metadata) != str(
        config["assets"]["blip_metadata_tree_sha256"]
    ):
        raise RuntimeError("BLIP2 metadata tree hash mismatch")
    open_clip_source = resolve(config["assets"]["open_clip_source"])
    open_clip_tree, open_clip_files = recursive_tree_sha256(open_clip_source)
    if open_clip_tree != str(config["assets"]["open_clip_tree_sha256"]):
        raise RuntimeError("OpenCLIP 2.24 source tree hash mismatch")
    if open_clip_files != int(config["assets"]["open_clip_tree_files"]):
        raise RuntimeError("OpenCLIP source file count mismatch")
    require_sha(open_clip_source / "LICENSE", config["assets"]["open_clip_license_sha256"])
    runtime = resolve(config["assets"]["transformers_runtime"])
    require_sha(
        runtime / "transformers-4.51.1.dist-info" / "METADATA",
        config["assets"]["transformers_metadata_sha256"],
    )
    require_sha(
        runtime / "tokenizers-0.21.4.dist-info" / "METADATA",
        config["assets"]["tokenizers_metadata_sha256"],
    )
    require_sha(
        runtime / "huggingface_hub-0.30.2.dist-info" / "METADATA",
        config["assets"]["huggingface_hub_metadata_sha256"],
    )
    execution_site = (
        resolve(config["assets"]["execution_venv"])
        / "lib"
        / "python3.10"
        / "site-packages"
    )
    runtime_metadata = (
        ("torch-2.1.0.dist-info", "torch_metadata_sha256"),
        ("torchvision-0.16.0.dist-info", "torchvision_metadata_sha256"),
        ("pytorch_lightning-2.4.0.dist-info", "pytorch_lightning_metadata_sha256"),
        ("xformers-0.0.22.post7.dist-info", "xformers_metadata_sha256"),
    )
    for directory, hash_key in runtime_metadata:
        require_sha(execution_site / directory / "METADATA", config["assets"][hash_key])
    model_config = load_yaml(model_config_path)
    preprocess = model_config["params"]["preprocess_config"]["params"]
    c_channel = int(preprocess["C_channel"])
    n_downsample = int(preprocess["n_downsample"])
    side = int(config["rate"]["author_input_short_edge_after_resize"]) // (2**n_downsample)
    latent_shape = [c_channel, side, side]
    real_symbols = c_channel * side * side
    if latent_shape != list(config["rate"]["author_latent_shape_at_square_input"]):
        raise RuntimeError(f"latent shape contract changed: {latent_shape}")
    if real_symbols != int(config["rate"]["diffjscc_real_symbols"]):
        raise RuntimeError("DiffJSCC symbol ledger changed")
    observed_cbr = complex_cbr(real_symbols, int(config["rate"]["source_real_dimensions_at_comparison_resolution"]))
    if abs(observed_cbr - float(config["rate"]["effective_cbr_against_256_source"])) > 1e-15:
        raise RuntimeError("effective CBR contract changed")

    samples, reference_path = validate_population(config)
    noise = validate_current_noise(config)
    checkpoint = checkpoint_state(config, args.allow_incomplete_checkpoint)
    blip_weights = blip_weight_state(config, args.allow_incomplete_checkpoint)
    assets_complete = checkpoint["status"] == "complete" and blip_weights["status"] == "complete"
    summary = {
        "analysis_id": "ANALYSIS-S30-DIFFJSCC-PREFLIGHT-001",
        "status": "PASS" if assets_complete else "WAITING_FOR_EXACT_WEIGHTS",
        "config": str(config_path.relative_to(ROOT)),
        "source": {
            "repository": str(repo.relative_to(ROOT)),
            "commit": observed_commit,
            "algorithm_source_clean": True,
            "tracked_bytecode_churn": tracked_lines,
        },
        "checkpoint": checkpoint,
        "external_blip_weights": blip_weights,
        "population": {
            "reference": str(reference_path.relative_to(ROOT)),
            "samples": len(samples),
            "official_validation_accessed": False,
        },
        "noise": noise,
        "rate": {
            "latent_shape": latent_shape,
            "real_symbols": real_symbols,
            "complex_channel_uses": real_symbols // 2,
            "effective_cbr_against_256_source": observed_cbr,
            "project_budget_fraction_used": real_symbols
            / int(config["rate"]["project_budget_real_symbols"]),
            "unused_project_budget_real_symbols": int(
                config["rate"]["project_budget_real_symbols"]
            )
            - real_symbols,
        },
        "checks": {
            "source_commit_exact": True,
            "tracked_algorithm_source_clean": True,
            "model_config_hash_exact": True,
            "blip_metadata_hash_exact": True,
            "open_clip_2_24_source_hash_exact": True,
            "transformers_runtime_metadata_exact": True,
            "execution_venv_metadata_exact": True,
            "population_hashes_exact": True,
            "s28_canonical_noise_exact": True,
            "symbol_ledger_exact": True,
            "checkpoint_complete_and_exact": checkpoint["status"] == "complete",
            "blip_weights_complete_and_exact": blip_weights["status"] == "complete",
        },
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if not args.no_write:
        output = resolve(config["outputs"]["preflight"])
        output.mkdir(parents=True, exist_ok=False)
        (output / "summary.json").write_text(rendered + "\n", encoding="utf-8")
        (output / "config_snapshot.yaml").write_text(
            config_path.read_text(encoding="utf-8"), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
