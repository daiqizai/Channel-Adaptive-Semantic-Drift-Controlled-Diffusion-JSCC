#!/usr/bin/env python3
"""Paired comparison of receiver-M0 structural guidance and sender source-edge oracle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.utils import save_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/s6_source_edge_oracle_comparison_exp_s4_008_011.yaml"
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


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


def snr_dir(snr: float) -> str:
    return f"snr_{int(snr):02d}db" if float(snr).is_integer() else f"snr_{str(snr).replace('.', 'p')}db"


def image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        return TF.to_tensor(image.convert("RGB"))


def psnr(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    mse = float(torch.mean((reference - candidate).square()).item())
    return float(-10.0 * math.log10(max(mse, 1e-12)))


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def read_semantic_rows(path: Path) -> dict[tuple[float, str], dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output: dict[tuple[float, str], dict[str, str]] = {}
    for row in rows:
        key = (float(row["snr_db"]), str(row["sample"]))
        if key in output:
            raise RuntimeError(f"Duplicate semantic row in {path}: {key}")
        output[key] = row
    return output


def bootstrap_mean(values: np.ndarray, replicates: int, seed: int) -> dict[str, Any]:
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Bootstrap values must be a non-empty vector")
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 256):
        count = min(256, replicates - start)
        indices = rng.integers(0, len(values), size=(count, len(values)), endpoint=False)
        samples[start : start + count] = values[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "estimate": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "replicates": replicates,
        "num_clusters": int(len(values)),
        "cluster": "sample",
    }


def load_summary_lpips(path: Path) -> dict[float, float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output: dict[float, float] = {}
    for row in rows:
        value = row.get("refined_delta_lpips_vs_m0")
        if value not in (None, ""):
            output[float(row["snr_db"])] = float(value)
    return output


def optional_delta(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else float(left - right)


def optional_mean(values: list[Any]) -> float | None:
    available = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(available)) if available else None


def format_optional(value: Any, digits: int = 4) -> str:
    if value is None or not math.isfinite(float(value)):
        return "N/A"
    return f"{float(value):+.{digits}f}"


def validate_matched_configs(receiver: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    equal_paths = [
        ("dataset",),
        ("image_size",),
        ("channel",),
        ("snrs",),
        ("cbr",),
        ("seed",),
        ("split",),
        ("model", "input_channels"),
        ("model", "condition_features"),
        ("model", "base_channels"),
        ("model", "num_blocks"),
        ("model", "snr_norm_max"),
        ("model", "residual_gates"),
        ("training",),
    ]

    def get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
        value: Any = payload
        for key in path:
            value = value[key]
        return value

    mismatches = {
        ".".join(path): {"receiver": get(receiver, path), "source": get(source, path)}
        for path in equal_paths
        if get(receiver, path) != get(source, path)
    }
    if mismatches:
        raise RuntimeError(f"Source/receiver edge experiments are not matched: {mismatches}")
    receiver_source = str(receiver["model"].get("condition_source", "receiver_m0"))
    source_source = str(source["model"].get("condition_source", "receiver_m0"))
    if receiver_source != "receiver_m0" or source_source != "sender_original_oracle":
        raise RuntimeError(
            f"Unexpected condition-source contrast: receiver={receiver_source}, source={source_source}"
        )
    return {
        "matched_fields": [".".join(path) for path in equal_paths],
        "receiver_condition_source": receiver_source,
        "source_condition_source": source_source,
        "sole_intended_difference": "model.condition_source",
    }


def save_comparison_sheet(
    output_dir: Path,
    original_dir: Path,
    m0_root: Path,
    receiver_dir: Path,
    source_dir: Path,
    snr: float,
    names: list[str],
) -> Path:
    folder = snr_dir(snr)
    selected = names[: min(4, len(names))]
    rows = []
    for role in ("original", "m0", "receiver_raw", "source_raw", "receiver_final", "source_final"):
        tensors = []
        for name in selected:
            paths = {
                "original": original_dir / name,
                "m0": m0_root / "exports" / folder / "reconstruction" / name,
                "receiver_raw": receiver_dir / "exports" / folder / "refined" / name,
                "source_raw": source_dir / "exports" / folder / "refined" / name,
                "receiver_final": receiver_dir / "exports" / folder / "final" / name,
                "source_final": source_dir / "exports" / folder / "final" / name,
            }
            tensors.append(image_tensor(paths[role]))
        rows.extend(tensors)
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    path = sample_dir / f"{folder}_original_m0_receiverraw_sourceraw_receiverfinal_sourcefinal.png"
    save_image(torch.stack(rows), path, nrow=len(selected), padding=2)
    return path


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    receiver_dir = resolve(config["inputs"]["receiver_edge_dir"])
    source_dir = resolve(config["inputs"]["source_edge_oracle_dir"])
    receiver_config_path = resolve(config["inputs"]["receiver_edge_config"])
    source_config_path = resolve(config["inputs"]["source_edge_oracle_config"])
    with receiver_config_path.open("r", encoding="utf-8") as handle:
        receiver_config = yaml.safe_load(handle)
    with source_config_path.open("r", encoding="utf-8") as handle:
        source_config = yaml.safe_load(handle)
    matched = validate_matched_configs(receiver_config, source_config)
    snrs = [float(value) for value in config["channel"]["snrs"]]
    names = [
        f"sample_{index:06d}.png"
        for index in range(
            int(config["split"]["sample_start"]),
            int(config["split"]["sample_start"]) + int(config["split"]["sample_count"]),
        )
    ]
    output_dir = resolve(args.output_dir or config["outputs"]["output_dir"])
    plan = {
        "analysis_id": config["analysis_id"],
        "num_images": len(names),
        "snrs": snrs,
        "matched_contract": matched,
        "source_edge_channel": config["channel"]["source_edge_channel"],
        "output_dir": relative(output_dir),
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, "plan": plan}, indent=2, ensure_ascii=False))
        return
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}")
        if output_dir == PROJECT_ROOT or (PROJECT_ROOT / "outputs" / "analysis") not in output_dir.parents:
            raise RuntimeError(f"Unsafe overwrite target: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    shutil.copy2(config_path, output_dir / "config.yaml")
    shutil.copy2(SCRIPT_PATH, output_dir / SCRIPT_PATH.name)
    save_json(output_dir / "run_plan.json", plan)

    original_dir = resolve(config["inputs"]["original_dir"])
    m0_root = resolve(config["inputs"]["m0_export_dir"])
    receiver_semantic = read_semantic_rows(receiver_dir / "per_sample.csv")
    source_semantic = read_semantic_rows(source_dir / "per_sample.csv")
    expected_keys = {(snr, name) for snr in snrs for name in names}
    if set(receiver_semantic) != expected_keys or set(source_semantic) != expected_keys:
        raise RuntimeError("Semantic CSV row grid does not match the configured sample/SNR grid")

    rows: list[dict[str, Any]] = []
    input_manifest: list[dict[str, str]] = []
    for snr in snrs:
        folder = snr_dir(snr)
        for name in names:
            paths = {
                "original": original_dir / name,
                "m0": m0_root / "exports" / folder / "reconstruction" / name,
                "receiver_raw": receiver_dir / "exports" / folder / "refined" / name,
                "source_raw": source_dir / "exports" / folder / "refined" / name,
                "receiver_final": receiver_dir / "exports" / folder / "final" / name,
                "source_final": source_dir / "exports" / folder / "final" / name,
            }
            for role, path in paths.items():
                if not path.is_file():
                    raise FileNotFoundError(path)
                input_manifest.append(
                    {"snr_db": str(snr), "sample": name, "role": role, "path": relative(path), "sha256": sha256_file(path)}
                )
            tensors = {key: image_tensor(path) for key, path in paths.items()}
            shape = tensors["original"].shape
            if any(tensor.shape != shape for tensor in tensors.values()):
                raise RuntimeError(f"Image shape mismatch at {snr} dB/{name}")
            m0_psnr = psnr(tensors["original"], tensors["m0"])
            receiver_raw_psnr = psnr(tensors["original"], tensors["receiver_raw"])
            source_raw_psnr = psnr(tensors["original"], tensors["source_raw"])
            receiver_final_psnr = psnr(tensors["original"], tensors["receiver_final"])
            source_final_psnr = psnr(tensors["original"], tensors["source_final"])
            receiver_sem = receiver_semantic[(snr, name)]
            source_sem = source_semantic[(snr, name)]
            for semantic_row in (receiver_sem, source_sem):
                if float(semantic_row["snr_db"]) != snr or semantic_row["sample"] != name:
                    raise RuntimeError("Semantic row identity mismatch")
            rows.append(
                {
                    "snr_db": snr,
                    "sample": name,
                    "m0_psnr_db": m0_psnr,
                    "receiver_raw_psnr_db": receiver_raw_psnr,
                    "source_raw_psnr_db": source_raw_psnr,
                    "receiver_final_psnr_db": receiver_final_psnr,
                    "source_final_psnr_db": source_final_psnr,
                    "receiver_raw_delta_psnr_db": receiver_raw_psnr - m0_psnr,
                    "source_raw_delta_psnr_db": source_raw_psnr - m0_psnr,
                    "source_minus_receiver_raw_psnr_db": source_raw_psnr - receiver_raw_psnr,
                    "receiver_final_delta_psnr_db": receiver_final_psnr - m0_psnr,
                    "source_final_delta_psnr_db": source_final_psnr - m0_psnr,
                    "source_minus_receiver_final_psnr_db": source_final_psnr - receiver_final_psnr,
                    "receiver_raw_new_error": bool_value(receiver_sem["m0_matches_original_top1"])
                    and not bool_value(receiver_sem["refined_matches_original_top1"]),
                    "source_raw_new_error": bool_value(source_sem["m0_matches_original_top1"])
                    and not bool_value(source_sem["refined_matches_original_top1"]),
                    "receiver_raw_repair": not bool_value(receiver_sem["m0_matches_original_top1"])
                    and bool_value(receiver_sem["refined_matches_original_top1"]),
                    "source_raw_repair": not bool_value(source_sem["m0_matches_original_top1"])
                    and bool_value(source_sem["refined_matches_original_top1"]),
                    "receiver_raw_failure": not bool_value(receiver_sem["refined_matches_original_top1"]),
                    "source_raw_failure": not bool_value(source_sem["refined_matches_original_top1"]),
                }
            )
    write_csv(output_dir / "per_sample.csv", rows)
    write_csv(output_dir / "input_manifest.csv", input_manifest)
    comparison_sheet = save_comparison_sheet(
        output_dir,
        original_dir,
        m0_root,
        receiver_dir,
        source_dir,
        7.0 if 7.0 in snrs else snrs[0],
        names,
    )

    receiver_lpips = load_summary_lpips(receiver_dir / "summary.csv")
    source_lpips = load_summary_lpips(source_dir / "summary.csv")
    summary_rows: list[dict[str, Any]] = []
    for snr in snrs:
        selected = [row for row in rows if float(row["snr_db"]) == snr]
        receiver_lpips_value = receiver_lpips.get(snr)
        source_lpips_value = source_lpips.get(snr)
        summary_rows.append(
            {
                "scope": f"snr_{snr:g}",
                "num_rows": len(selected),
                "receiver_raw_delta_psnr_db": float(np.mean([row["receiver_raw_delta_psnr_db"] for row in selected])),
                "source_raw_delta_psnr_db": float(np.mean([row["source_raw_delta_psnr_db"] for row in selected])),
                "source_minus_receiver_raw_psnr_db": float(np.mean([row["source_minus_receiver_raw_psnr_db"] for row in selected])),
                "receiver_final_delta_psnr_db": float(np.mean([row["receiver_final_delta_psnr_db"] for row in selected])),
                "source_final_delta_psnr_db": float(np.mean([row["source_final_delta_psnr_db"] for row in selected])),
                "source_minus_receiver_final_psnr_db": float(np.mean([row["source_minus_receiver_final_psnr_db"] for row in selected])),
                "receiver_raw_lpips_delta": receiver_lpips_value,
                "source_raw_lpips_delta": source_lpips_value,
                "source_minus_receiver_raw_lpips": optional_delta(
                    source_lpips_value, receiver_lpips_value
                ),
                "receiver_raw_failure_rate": float(np.mean([row["receiver_raw_failure"] for row in selected])),
                "source_raw_failure_rate": float(np.mean([row["source_raw_failure"] for row in selected])),
                "receiver_raw_new_error_count": int(sum(row["receiver_raw_new_error"] for row in selected)),
                "source_raw_new_error_count": int(sum(row["source_raw_new_error"] for row in selected)),
                "receiver_raw_repair_count": int(sum(row["receiver_raw_repair"] for row in selected)),
                "source_raw_repair_count": int(sum(row["source_raw_repair"] for row in selected)),
            }
        )
    by_sample: dict[str, list[float]] = defaultdict(list)
    by_sample_final: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_sample[str(row["sample"])].append(float(row["source_minus_receiver_raw_psnr_db"]))
        by_sample_final[str(row["sample"])].append(float(row["source_minus_receiver_final_psnr_db"]))
    raw_cluster = np.asarray([np.mean(by_sample[name]) for name in sorted(by_sample)], dtype=np.float64)
    final_cluster = np.asarray([np.mean(by_sample_final[name]) for name in sorted(by_sample_final)], dtype=np.float64)
    replicates = int(config["statistics"]["bootstrap_replicates"])
    seed = int(config["statistics"]["bootstrap_seed"])
    intervals = {
        "source_minus_receiver_raw_psnr_db": bootstrap_mean(raw_cluster, replicates, seed),
        "source_minus_receiver_final_psnr_db": bootstrap_mean(final_cluster, replicates, seed + 1),
        "source_minus_receiver_raw_psnr_by_snr": {
            str(snr): bootstrap_mean(
                np.asarray(
                    [row["source_minus_receiver_raw_psnr_db"] for row in rows if float(row["snr_db"]) == snr],
                    dtype=np.float64,
                ),
                replicates,
                seed + 10 + index,
            )
            for index, snr in enumerate(snrs)
        },
    }
    all_snr_positive = all(
        row["source_minus_receiver_raw_psnr_db"] > 0.0 for row in summary_rows
    )
    success = {
        "source_minus_receiver_raw_psnr_ci_lower_strictly_positive": intervals[
            "source_minus_receiver_raw_psnr_db"
        ]["ci_low"]
        > 0.0,
        "source_minus_receiver_raw_psnr_point_positive_each_snr": all_snr_positive,
    }
    success["all_pass"] = all(success.values())
    mean_summary = {
        key: optional_mean([row[key] for row in summary_rows])
        for key in (
            "receiver_raw_delta_psnr_db",
            "source_raw_delta_psnr_db",
            "source_minus_receiver_raw_psnr_db",
            "receiver_final_delta_psnr_db",
            "source_final_delta_psnr_db",
            "source_minus_receiver_final_psnr_db",
            "receiver_raw_lpips_delta",
            "source_raw_lpips_delta",
            "source_minus_receiver_raw_lpips",
            "receiver_raw_failure_rate",
            "source_raw_failure_rate",
        )
    }
    mean_summary.update(
        {
            "receiver_raw_new_error_count": int(sum(row["receiver_raw_new_error_count"] for row in summary_rows)),
            "source_raw_new_error_count": int(sum(row["source_raw_new_error_count"] for row in summary_rows)),
            "receiver_raw_repair_count": int(sum(row["receiver_raw_repair_count"] for row in summary_rows)),
            "source_raw_repair_count": int(sum(row["source_raw_repair_count"] for row in summary_rows)),
        }
    )
    summary_rows.append({"scope": "mean_or_total", "num_rows": len(rows), **mean_summary})
    write_csv(output_dir / "summary.csv", summary_rows)
    payload = {
        "analysis_id": config["analysis_id"],
        "matched_contract": matched,
        "summary": mean_summary,
        "per_snr": summary_rows[:-1],
        "bootstrap": intervals,
        "success": success,
        "communication_guardrail": {
            "main_image_cbr": config["channel"]["main_image_cbr"],
            "source_edge_channel": config["channel"]["source_edge_channel"],
            "total_cbr_defined": False,
            "deployable_claim_allowed": False,
        },
    }
    save_json(output_dir / "summary.json", payload)
    raw_ci = intervals["source_minus_receiver_raw_psnr_db"]
    lines = [
        "# Source-Edge Oracle vs Receiver-Edge",
        "",
        f"Feasibility decision: **{'PASS' if success['all_pass'] else 'FAIL'}**.",
        "",
        "| SNR | Receiver-edge ΔPSNR | Source-edge ΔPSNR | Source − receiver | Source − receiver ΔLPIPS | Receiver/Source raw failure |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows[:-1]:
        lines.append(
            f"| {row['scope'].replace('snr_', '')} | {row['receiver_raw_delta_psnr_db']:+.4f} | "
            f"{row['source_raw_delta_psnr_db']:+.4f} | {row['source_minus_receiver_raw_psnr_db']:+.4f} | "
            f"{format_optional(row['source_minus_receiver_raw_lpips'])} | "
            f"{row['receiver_raw_failure_rate']:.4f}/{row['source_raw_failure_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Across SNRs, source-edge improves raw PSNR over receiver-edge by `{raw_ci['estimate']:+.4f}` dB "
            f"(paired image-cluster 95% CI `[{raw_ci['ci_low']:+.4f}, {raw_ci['ci_high']:+.4f}]`).",
            f"All five per-SNR point estimates are positive: `{all_snr_positive}`.",
            f"Raw pseudo failure changes from `{mean_summary['receiver_raw_failure_rate']:.4f}` to "
            f"`{mean_summary['source_raw_failure_rate']:.4f}`; this is auxiliary, not supervised semantic evidence.",
            "",
            "## Guardrail",
            "",
            "The source structural maps are perfectly available and have no coded rate or channel errors. "
            "This is an architecture feasibility upper bound, not a matched-CBR communication result. "
            "The next valid step is a separately transmitted lossy edge representation with a fixed total CBR.",
            f"Visual sheet: `{relative(comparison_sheet)}`; rows are original, M0, receiver raw, source raw, receiver final, source final.",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": relative(config_path),
        "script": relative(SCRIPT_PATH),
        "hashes": {
            "config_sha256": sha256_file(config_path),
            "script_sha256": sha256_file(SCRIPT_PATH),
            "training_script_sha256": sha256_file(
                PROJECT_ROOT / "scripts" / "s5_residual_refiner_pilot.py"
            ),
            "receiver_config_sha256": sha256_file(receiver_config_path),
            "source_config_sha256": sha256_file(source_config_path),
            "receiver_checkpoint_sha256": sha256_file(
                receiver_dir / "checkpoints" / "best.pt"
            ),
            "source_checkpoint_sha256": sha256_file(
                source_dir / "checkpoints" / "best.pt"
            ),
            "preregistration_sha256": sha256_file(resolve(config["inputs"]["preregistration"])),
            "input_manifest_sha256": sha256_file(output_dir / "input_manifest.csv"),
        },
        "num_unique_pngs": len({row["sha256"] for row in input_manifest}),
        "num_manifest_rows": len(input_manifest),
        "source_edge_channel": config["channel"]["source_edge_channel"],
        "comparison_sheet": relative(comparison_sheet),
    }
    save_json(output_dir / "metadata.json", metadata)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
