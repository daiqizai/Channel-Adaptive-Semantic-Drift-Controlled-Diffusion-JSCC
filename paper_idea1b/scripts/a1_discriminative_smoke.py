#!/usr/bin/env python3
"""Preflight and native-large-image smoke for paper idea1b A1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cadsd_jscc.external_common import (  # noqa: E402
    canonical_noise_sha256,
    canonical_standard_normal,
)
from cadsd_jscc.strong_jscc import (  # noqa: E402
    StrongJSCC,
    trainable_parameter_count as strong_parameter_count,
)
from cadsd_jscc.swinjscc_adapter import (  # noqa: E402
    OfficialSwinJSCCSA,
    trainable_parameter_count as swin_parameter_count,
)


SCRIPT = Path(__file__).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="paper_idea1b/configs/a1_discriminative_benchmark.yaml"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage", choices=("preflight", "smoke"), required=True)
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(entry: dict[str, Any] | str, expected: str | None = None) -> Path:
    if isinstance(entry, dict):
        path = resolve(entry["path"] if "path" in entry else entry["checkpoint"])
        expected_value = str(entry["sha256"])
    else:
        path = resolve(entry)
        if expected is None:
            raise ValueError("expected SHA is required")
        expected_value = str(expected)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected_value:
        raise RuntimeError(f"SHA mismatch: {path}: {actual} != {expected_value}")
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def gpu_inventory() -> dict[str, str]:
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "name": properties.name,
        "total_memory_bytes": str(properties.total_memory),
        "torch": torch.__version__,
        "cuda_runtime": str(torch.version.cuda),
    }


def load_inputs(config: dict[str, Any]) -> tuple[
    dict[str, dict[str, str]], dict[str, list[dict[str, str]]]
]:
    source_path = require_sha(config["inputs"]["source_manifest"])
    tile_path = require_sha(config["inputs"]["tile_manifest"])
    sources = {row["sample_id"]: row for row in read_csv(source_path)}
    grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(tile_path):
        grouped[row["sample_id"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["tile_index"]))
    if set(sources) != set(grouped):
        raise RuntimeError("source/tile manifest key mismatch")
    return sources, dict(grouped)


def build_s33(checkpoint_path: Path) -> tuple[StrongJSCC, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_config = checkpoint["config"]
    model_config = train_config["model"]
    model = StrongJSCC(
        image_size=int(train_config["image_size"]),
        latent_channels=int(model_config["latent_channels"]),
        stage_channels=tuple(int(value) for value in model_config["stage_channels"]),
        stage_blocks=tuple(int(value) for value in model_config["stage_blocks"]),
        condition_dim=int(model_config["condition_dim"]),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, {
        "epoch": int(checkpoint["epoch"]),
        "phase": str(checkpoint.get("phase", "")),
    }


def build_swin(
    checkpoint_path: Path, expected_arm: str
) -> tuple[OfficialSwinJSCCSA, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint["epoch_number_1based"]) > 12:
        raise RuntimeError("A1 must not use an extension checkpoint")
    checkpoint_arm = checkpoint.get("arm")
    if checkpoint_arm is not None and str(checkpoint_arm) != expected_arm:
        raise RuntimeError(f"checkpoint arm mismatch: {checkpoint_arm} != {expected_arm}")
    arm = checkpoint["config"]["arms_confirmed"][expected_arm]
    model = OfficialSwinJSCCSA(
        image_size=256,
        latent_channels=64,
        encoder_depths=tuple(int(value) for value in arm["encoder_depths"]),
        decoder_depths=tuple(int(value) for value in arm["decoder_depths"]),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, {
        "epoch": int(checkpoint["epoch_number_1based"]),
        "phase": str(checkpoint["phase"]),
    }


def load_model(
    arm: str, config: dict[str, Any]
) -> tuple[torch.nn.Module, int, dict[str, Any]]:
    if arm == "s33_strong":
        path = require_sha(config["inputs"]["s33"])
        model, checkpoint_meta = build_s33(path)
        parameters = strong_parameter_count(model)
    else:
        expected_arm = arm.removeprefix("swin_")
        entry = config["inputs"][arm]
        path = require_sha(entry)
        summary_path = require_sha(
            str(entry["training_summary"]), str(entry["training_summary_sha256"])
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary["status"] != "complete_equal_budget_only"
            or summary["arm"] != expected_arm
            or int(summary["epochs_completed"]) != 12
            or summary["extension_executed"] is not False
            or str(summary["best_checkpoint_sha256"]) != str(entry["sha256"])
        ):
            raise RuntimeError(f"frozen training summary mismatch for {arm}")
        model, checkpoint_meta = build_swin(path, expected_arm)
        parameters = swin_parameter_count(model)
    expected_parameters = int(
        config["models"]["expected_trainable_parameters"][arm]
    )
    if parameters != expected_parameters:
        raise RuntimeError(f"parameter mismatch for {arm}: {parameters}")
    if int(model.real_symbols) != int(config["models"]["expected_real_symbols_per_tile"]):
        raise RuntimeError(f"real-symbol mismatch for {arm}")
    return model, parameters, {
        **checkpoint_meta,
        "checkpoint": relative(path),
        "checkpoint_sha256": sha256_file(path),
    }


def preflight(config: dict[str, Any]) -> dict[str, Any]:
    if config["protocol"]["diffusion_allowed"] is not False:
        raise RuntimeError("A1 discriminative gate must forbid diffusion")
    if config["protocol"]["sgd_allowed"] is not False:
        raise RuntimeError("A1 discriminative gate must forbid SGD")
    if config["protocol"]["official_imagenette_validation_accessed"] is not False:
        raise RuntimeError("official validation must remain sealed")
    require_sha(config["inputs"]["source_manifest"])
    require_sha(config["inputs"]["tile_manifest"])
    for entry in config["inputs"]["source_code"].values():
        require_sha(entry)
    sources, tiles = load_inputs(config)
    expected_counts = {"kodak": 24, "clic2020_test": 428}
    actual_counts: dict[str, int] = defaultdict(int)
    for row in sources.values():
        actual_counts[row["dataset"]] += 1
    if dict(actual_counts) != expected_counts:
        raise RuntimeError(f"source counts changed: {dict(actual_counts)}")

    models: dict[str, Any] = {}
    for arm in config["models"]["order"]:
        model, parameters, checkpoint_meta = load_model(str(arm), config)
        models[str(arm)] = {
            "parameters": parameters,
            "real_symbols_per_tile": int(model.real_symbols),
            **checkpoint_meta,
        }
        del model
    torch.cuda.empty_cache()
    return {
        "status": "PASS",
        "analysis_id": config["analysis"]["id"],
        "source_counts": dict(actual_counts),
        "tile_rows": sum(len(value) for value in tiles.values()),
        "models": models,
        "diffusion_loaded": False,
        "sgd_loaded": False,
        "official_imagenette_validation_accessed": False,
    }


def pad_tile(tile: np.ndarray, bottom: int, right: int) -> np.ndarray:
    value = tile
    if bottom:
        mode = "reflect" if value.shape[0] > 1 else "edge"
        value = np.pad(value, ((0, bottom), (0, 0), (0, 0)), mode=mode)
    if right:
        mode = "reflect" if value.shape[1] > 1 else "edge"
        value = np.pad(value, ((0, 0), (0, right), (0, 0)), mode=mode)
    if value.shape != (256, 256, 3):
        raise RuntimeError(f"unexpected padded tile shape: {value.shape}")
    return value


def extract_tiles(
    source: np.ndarray, tile_rows: list[dict[str, str]]
) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for row in tile_rows:
        top, left = int(row["top"]), int(row["left"])
        height, width = int(row["valid_height"]), int(row["valid_width"])
        tile = source[top : top + height, left : left + width]
        tile = pad_tile(tile, int(row["pad_bottom"]), int(row["pad_right"]))
        values.append(torch.from_numpy(np.ascontiguousarray(tile)).permute(2, 0, 1))
    return torch.stack(values).float().div_(255.0)


def stitch_tiles(
    reconstructed: list[np.ndarray],
    tile_rows: list[dict[str, str]],
    height: int,
    width: int,
) -> np.ndarray:
    output = np.zeros((height, width, 3), dtype=np.uint8)
    if len(reconstructed) != len(tile_rows):
        raise RuntimeError("tile reconstruction count mismatch")
    for tile, row in zip(reconstructed, tile_rows):
        top, left = int(row["top"]), int(row["left"])
        valid_h, valid_w = int(row["valid_height"]), int(row["valid_width"])
        output[top : top + valid_h, left : left + valid_w] = tile[:valid_h, :valid_w]
    return output


@torch.inference_mode()
def warmup(
    model: torch.nn.Module,
    first_tile: torch.Tensor,
    first_noise: torch.Tensor,
    snr: float,
    device: torch.device,
) -> None:
    model.forward_with_observation(
        first_tile.unsqueeze(0).to(device),
        snr,
        first_noise.unsqueeze(0).to(device),
    )
    torch.cuda.synchronize(device)


@torch.inference_mode()
def reconstruct_image(
    model: torch.nn.Module,
    arm: str,
    source_row: dict[str, str],
    tile_rows: list[dict[str, str]],
    *,
    seed: int,
    snr: float,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    path = resolve(source_row["path"])
    with Image.open(path) as image:
        source = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = source.shape[:2]
    total_real = len(tile_rows) * int(model.real_symbols)
    noise = canonical_standard_normal(
        seed, str(source_row["sample_id"]), snr, total_real
    )
    noise_sha = canonical_noise_sha256(noise)
    started = time.perf_counter()
    tiles = extract_tiles(source, tile_rows)
    reconstructed: list[np.ndarray] = []
    gpu_model_ms = 0.0
    max_power_error = 0.0
    for start in range(0, len(tile_rows), batch_size):
        end = min(start + batch_size, len(tile_rows))
        tile_batch = tiles[start:end].to(device)
        noise_batch = noise[
            start * int(model.real_symbols) : end * int(model.real_symbols)
        ].reshape(end - start, int(model.real_symbols)).to(device)
        torch.cuda.synchronize(device)
        gpu_started = time.perf_counter()
        reconstruction, observation = model.forward_with_observation(
            tile_batch, snr, noise_batch
        )
        reconstruction = reconstruction.clamp(0.0, 1.0)
        torch.cuda.synchronize(device)
        gpu_model_ms += (time.perf_counter() - gpu_started) * 1000.0
        max_power_error = max(
            max_power_error,
            float((observation.normalized_power - 1.0).abs().max().item()),
        )
        quantized = (
            torch.floor(reconstruction * 255.0)
            .clamp(0.0, 255.0)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )
        reconstructed.extend(list(quantized))
        del tile_batch, noise_batch, reconstruction, observation, quantized
    output = stitch_tiles(reconstructed, tile_rows, height, width)
    wall_ms = (time.perf_counter() - started) * 1000.0
    complex_uses = total_real // 2
    actual_cbr = complex_uses / (3 * height * width)
    return output, {
        "arm": arm,
        "dataset": source_row["dataset"],
        "sample_id": source_row["sample_id"],
        "width": width,
        "height": height,
        "tile_count": len(tile_rows),
        "real_symbols": total_real,
        "complex_channel_uses": complex_uses,
        "actual_cbr": actual_cbr,
        "seed": seed,
        "snr_db": snr,
        "canonical_noise_sha256": noise_sha,
        "gpu_model_ms": gpu_model_ms,
        "end_to_end_wall_ms": wall_ms,
        "max_normalized_power_abs_error": max_power_error,
        "output_shape": list(output.shape),
    }


def run_smoke(
    config: dict[str, Any],
    config_path: Path,
    device: torch.device,
    preflight_summary: dict[str, Any],
) -> dict[str, Any]:
    output = resolve(config["analysis"]["smoke_output"])
    if output.exists():
        raise FileExistsError(f"smoke output exists: {output}")
    output.mkdir(parents=True)
    (output / "images").mkdir()
    shutil.copy2(config_path, output / "config_snapshot.yaml")
    shutil.copy2(SCRIPT, output / SCRIPT.name)
    sources, tiles = load_inputs(config)
    sample_ids = list(map(str, config["smoke"]["samples"]))
    for sample_id in sample_ids:
        if sample_id not in sources:
            raise KeyError(f"smoke sample missing: {sample_id}")
    seed = int(config["smoke"]["seed"])
    snr = float(config["smoke"]["snr_db"])
    batch_size = int(config["native_processing"]["tile_batch_size"])
    rows: list[dict[str, Any]] = []

    write_json(
        output / "STATE.json",
        {
            "status": "running",
            "diffusion_loaded": False,
            "sgd_loaded": False,
            "official_imagenette_validation_accessed": False,
        },
    )
    try:
        for arm in map(str, config["models"]["order"]):
            model, parameters, checkpoint_meta = load_model(arm, config)
            model = model.to(device).eval()
            model.requires_grad_(False)
            first_source = sources[sample_ids[0]]
            with Image.open(resolve(first_source["path"])) as image:
                first_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
            first_tiles = extract_tiles(first_array, tiles[sample_ids[0]])
            first_noise = canonical_standard_normal(
                seed,
                sample_ids[0],
                snr,
                len(tiles[sample_ids[0]]) * int(model.real_symbols),
            )[: int(model.real_symbols)]
            warmup(model, first_tiles[0], first_noise, snr, device)
            del first_tiles, first_noise

            torch.cuda.reset_peak_memory_stats(device)
            arm_started = time.perf_counter()
            for sample_id in sample_ids:
                reconstruction, row = reconstruct_image(
                    model,
                    arm,
                    sources[sample_id],
                    tiles[sample_id],
                    seed=seed,
                    snr=snr,
                    batch_size=batch_size,
                    device=device,
                )
                row["parameters"] = parameters
                row.update(checkpoint_meta)
                filename = (
                    f"{arm}__{sources[sample_id]['dataset']}__"
                    f"{Path(sample_id).stem}.png"
                )
                Image.fromarray(reconstruction).save(output / "images" / filename)
                row["output_path"] = relative(output / "images" / filename)
                rows.append(row)
            torch.cuda.synchronize(device)
            arm_wall_ms = (time.perf_counter() - arm_started) * 1000.0
            peak_allocated = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)
            for row in rows:
                if row["arm"] == arm:
                    row["arm_total_smoke_wall_ms"] = arm_wall_ms
                    row["peak_allocated_bytes"] = peak_allocated
                    row["peak_reserved_bytes"] = peak_reserved
            del model
            torch.cuda.empty_cache()

        expected_rows = len(sample_ids) * len(config["models"]["order"])
        if len(rows) != expected_rows:
            raise RuntimeError(f"smoke row count mismatch: {len(rows)}")
        power_limit = float(config["channel"]["normalized_power_abs_error_max"])
        if max(float(row["max_normalized_power_abs_error"]) for row in rows) > power_limit:
            raise RuntimeError("normalized-power smoke audit failed")
        for sample_id in sample_ids:
            selected = [row for row in rows if row["sample_id"] == sample_id]
            rates = {
                (
                    int(row["real_symbols"]),
                    int(row["complex_channel_uses"]),
                    float(row["actual_cbr"]),
                )
                for row in selected
            }
            noise_shas = {row["canonical_noise_sha256"] for row in selected}
            if len(rates) != 1 or len(noise_shas) != 1:
                raise RuntimeError(f"paired rate/noise mismatch for {sample_id}")

        summary = {
            "status": "PASS",
            "smoke_id": config["analysis"]["smoke_id"],
            "analysis_id": config["analysis"]["id"],
            "device": str(device),
            "platform": platform.platform(),
            "gpu": gpu_inventory(),
            "preflight": preflight_summary,
            "rows": rows,
            "paired_actual_rate_and_noise_equal_across_arms": True,
            "diffusion_loaded": False,
            "sgd_loaded": False,
            "official_imagenette_validation_accessed": False,
            "formal_run_started": False,
            "script_sha256": sha256_file(SCRIPT),
            "config_sha256": sha256_file(config_path),
        }
        write_json(output / "summary.json", summary)
        write_json(
            output / "STATE.json",
            {
                "status": "complete",
                "formal_run_started": False,
                "official_imagenette_validation_accessed": False,
                "summary_sha256": sha256_file(output / "summary.json"),
            },
        )
        return summary
    except Exception:
        write_json(
            output / "STATE.json",
            {
                "status": "failed",
                "formal_run_started": False,
                "official_imagenette_validation_accessed": False,
                "traceback": traceback.format_exc(),
            },
        )
        raise


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["protocol"]["status"] != "preregistered_before_a1_discriminative_smoke":
        raise RuntimeError("A1 preregistration status changed")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("A1 smoke requires CUDA")
    torch.cuda.set_device(device)
    preflight_summary = preflight(config)
    if args.stage == "preflight":
        print(json.dumps(preflight_summary, ensure_ascii=False, indent=2))
        return
    summary = run_smoke(config, config_path, device, preflight_summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
