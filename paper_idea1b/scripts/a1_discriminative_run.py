#!/usr/bin/env python3
"""Run frozen S33/Swin native-tile reconstructions for paper idea1b A1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "paper_idea1b" / "scripts")]

from a1_discriminative_smoke import (  # noqa: E402
    load_inputs,
    load_model,
    preflight,
    reconstruct_image,
    relative,
    resolve,
    sha256_file,
)
from cadsd_jscc.metrics import ms_ssim_per_sample, psnr_per_sample  # noqa: E402


SCRIPT = Path(__file__).resolve()
SMOKE_SCRIPT = ROOT / "paper_idea1b" / "scripts" / "a1_discriminative_smoke.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="paper_idea1b/configs/a1_discriminative_benchmark.yaml"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset", choices=("kodak", "clic2020_test"), required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        f"invalid JSONL at {path}:{line_number}"
                    ) from error
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("refuse to write empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def output_filename(sample_id: str) -> str:
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:16]
    return f"{digest}__{Path(sample_id).stem}.png"


def tensor_from_uint8(image, device: torch.device) -> torch.Tensor:
    value = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0)
    return value.unsqueeze(0).to(device)


def setup_output(
    output: Path,
    config_path: Path,
    *,
    resume: bool,
) -> None:
    if not output.exists():
        if resume:
            raise FileNotFoundError(output)
        output.mkdir(parents=True)
        (output / "reconstructions").mkdir()
        shutil.copy2(config_path, output / "config_snapshot.yaml")
        shutil.copy2(SCRIPT, output / SCRIPT.name)
        shutil.copy2(SMOKE_SCRIPT, output / SMOKE_SCRIPT.name)
    else:
        if not resume:
            raise FileExistsError(output)
        if sha256_file(output / "config_snapshot.yaml") != sha256_file(config_path):
            raise RuntimeError("resume config differs from frozen output config")
        for source, frozen in (
            (SCRIPT, output / SCRIPT.name),
            (SMOKE_SCRIPT, output / SMOKE_SCRIPT.name),
        ):
            if sha256_file(source) != sha256_file(frozen):
                raise RuntimeError(f"resume script differs from frozen snapshot: {source}")


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["protocol"]["status"] != "preregistered_before_a1_discriminative_smoke":
        raise RuntimeError("A1 preregistration status changed")
    smoke_state_path = resolve(config["analysis"]["smoke_output"]) / "STATE.json"
    if not smoke_state_path.is_file():
        raise FileNotFoundError("A1 smoke has not completed")
    smoke_state = json.loads(smoke_state_path.read_text(encoding="utf-8"))
    if smoke_state.get("status") != "complete" or smoke_state.get(
        "formal_run_started"
    ) is not False:
        raise RuntimeError(f"unexpected smoke state: {smoke_state}")
    if smoke_state.get("official_imagenette_validation_accessed") is not False:
        raise RuntimeError("official validation boundary changed")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("A1 formal reconstruction requires CUDA")
    torch.cuda.set_device(device)
    preflight_summary = preflight(config)
    output = resolve(config["analysis"]["output"])
    setup_output(output, config_path, resume=args.resume)
    sources, tiles = load_inputs(config)
    dataset = args.dataset
    dataset_config = config["formal"][dataset]
    selected = sorted(
        (row for row in sources.values() if row["dataset"] == dataset),
        key=lambda row: row["sample_id"],
    )
    if len(selected) != int(dataset_config["sample_count"]):
        raise RuntimeError(f"{dataset} source count changed")
    seeds = list(map(int, dataset_config["seeds"]))
    snrs = list(map(float, dataset_config["snrs_db"]))
    arms = list(map(str, config["models"]["order"]))
    expected_rows = len(selected) * len(seeds) * len(snrs) * len(arms)
    jsonl = output / f"per_sample_{dataset}.jsonl"
    existing = read_jsonl(jsonl)
    completed = {
        (
            str(row["arm"]),
            str(row["sample_id"]),
            int(row["seed"]),
            float(row["snr_db"]),
        )
        for row in existing
    }
    if len(completed) != len(existing):
        raise RuntimeError("duplicate keys in existing JSONL")
    if existing and not args.resume:
        raise RuntimeError("existing rows require --resume")

    write_json(
        output / "STATE.json",
        {
            "status": "running",
            "current_dataset": dataset,
            "completed_rows_this_dataset": len(existing),
            "expected_rows_this_dataset": expected_rows,
            "diffusion_loaded": False,
            "sgd_loaded": False,
            "official_imagenette_validation_accessed": False,
        },
    )
    started = time.perf_counter()
    try:
        for arm in arms:
            model, parameters, checkpoint_meta = load_model(arm, config)
            model = model.to(device).eval()
            model.requires_grad_(False)
            for seed in seeds:
                for snr in snrs:
                    target_dir = (
                        output
                        / "reconstructions"
                        / dataset
                        / arm
                        / f"seed_{seed}"
                        / f"snr_{int(snr):02d}"
                    )
                    target_dir.mkdir(parents=True, exist_ok=True)
                    for source_row in selected:
                        key = (arm, source_row["sample_id"], seed, snr)
                        if key in completed:
                            continue
                        reconstruction, row = reconstruct_image(
                            model,
                            arm,
                            source_row,
                            tiles[source_row["sample_id"]],
                            seed=seed,
                            snr=snr,
                            batch_size=int(
                                config["native_processing"]["tile_batch_size"]
                            ),
                            device=device,
                        )
                        with Image.open(resolve(source_row["path"])) as image:
                            source_uint8 = np.asarray(
                                image.convert("RGB"), dtype=np.uint8
                            )
                        target_tensor = tensor_from_uint8(source_uint8, device)
                        reconstruction_tensor = tensor_from_uint8(
                            reconstruction, device
                        )
                        with torch.inference_mode():
                            psnr = float(
                                psnr_per_sample(
                                    reconstruction_tensor, target_tensor
                                ).item()
                            )
                            ms_ssim = float(
                                ms_ssim_per_sample(
                                    reconstruction_tensor, target_tensor
                                ).item()
                            )
                        filename = output_filename(source_row["sample_id"])
                        reconstruction_path = target_dir / filename
                        Image.fromarray(reconstruction).save(
                            reconstruction_path, optimize=False
                        )
                        row.update(
                            {
                                "source_path": source_row["path"],
                                "source_sha256": source_row["content_sha256"],
                                "reconstruction_path": relative(
                                    reconstruction_path
                                ),
                                "reconstruction_sha256": sha256_file(
                                    reconstruction_path
                                ),
                                "psnr": psnr,
                                "ms_ssim": ms_ssim,
                                "parameters": parameters,
                                **checkpoint_meta,
                            }
                        )
                        for key_name, value in row.items():
                            if isinstance(value, float) and not math.isfinite(value):
                                raise RuntimeError(f"non-finite {key_name}")
                        append_jsonl(jsonl, row)
                        existing.append(row)
                        completed.add(key)
                        if len(existing) % 25 == 0:
                            elapsed = time.perf_counter() - started
                            write_json(
                                output / "STATE.json",
                                {
                                    "status": "running",
                                    "current_dataset": dataset,
                                    "current_arm": arm,
                                    "current_seed": seed,
                                    "current_snr_db": snr,
                                    "completed_rows_this_dataset": len(existing),
                                    "expected_rows_this_dataset": expected_rows,
                                    "elapsed_seconds_this_invocation": elapsed,
                                    "diffusion_loaded": False,
                                    "sgd_loaded": False,
                                    "official_imagenette_validation_accessed": False,
                                },
                            )
                        del target_tensor, reconstruction_tensor
            del model
            torch.cuda.empty_cache()

        if len(existing) != expected_rows:
            raise RuntimeError(
                f"{dataset} row count mismatch: {len(existing)} != {expected_rows}"
            )
        by_source_rate: dict[
            tuple[str, int, float], set[tuple[int, float, str]]
        ] = {}
        for row in existing:
            key = (str(row["sample_id"]), int(row["seed"]), float(row["snr_db"]))
            by_source_rate.setdefault(key, set()).add(
                (
                    int(row["real_symbols"]),
                    float(row["actual_cbr"]),
                    str(row["canonical_noise_sha256"]),
                )
            )
        if any(len(values) != 1 for values in by_source_rate.values()):
            raise RuntimeError("paired actual-rate/noise equality failed")
        csv_path = output / f"per_sample_{dataset}.csv"
        write_csv(csv_path, existing)
        dataset_summary = {
            "status": "reconstruction_complete",
            "analysis_id": config["analysis"]["id"],
            "dataset": dataset,
            "rows": len(existing),
            "sources": len(selected),
            "arms": arms,
            "seeds": seeds,
            "snrs_db": snrs,
            "paired_actual_rate_and_noise_equal_across_arms": True,
            "elapsed_seconds_this_invocation": time.perf_counter() - started,
            "per_sample_csv": relative(csv_path),
            "per_sample_csv_sha256": sha256_file(csv_path),
            "preflight": preflight_summary,
            "diffusion_loaded": False,
            "sgd_loaded": False,
            "official_imagenette_validation_accessed": False,
            "metrics_complete": False,
            "script_sha256": sha256_file(SCRIPT),
            "smoke_dependency_sha256": sha256_file(SMOKE_SCRIPT),
            "config_sha256": sha256_file(config_path),
        }
        write_json(output / f"summary_{dataset}_reconstruction.json", dataset_summary)
        write_json(
            output / "STATE.json",
            {
                "status": f"{dataset}_reconstruction_complete",
                "metrics_complete": False,
                "diffusion_loaded": False,
                "sgd_loaded": False,
                "official_imagenette_validation_accessed": False,
            },
        )
        print(json.dumps(dataset_summary, ensure_ascii=False, indent=2))
    except Exception:
        write_json(
            output / "STATE.json",
            {
                "status": "failed",
                "current_dataset": dataset,
                "completed_rows_this_dataset": len(existing),
                "expected_rows_this_dataset": expected_rows,
                "diffusion_loaded": False,
                "sgd_loaded": False,
                "official_imagenette_validation_accessed": False,
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
