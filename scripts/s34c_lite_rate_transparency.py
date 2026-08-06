#!/usr/bin/env python3
"""Read-only rate/prior transparency aggregation for frozen S34C-Lite inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "s34c_lite_rate_transparency_preregistration.yaml"
METRICS = ("psnr", "ms_ssim", "lpips", "failure")
METHODS = ("S33 strong", "DiffJSCC", "SGD paper upper")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def truth(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered not in {"true", "false"}:
        raise ValueError(f"invalid boolean: {value}")
    return lowered == "true"


def key(row: dict[str, str]) -> tuple[str, int, int]:
    return (
        row["sample_id"],
        int(row["base_seed"]),
        int(round(float(row["snr_db"]))),
    )


def require_unique(rows: list[dict[str, str]], label: str) -> dict[tuple[str, int, int], dict[str, str]]:
    indexed: dict[tuple[str, int, int], dict[str, str]] = {}
    for row in rows:
        row_key = key(row)
        if row_key in indexed:
            raise RuntimeError(f"duplicate {label} key: {row_key}")
        indexed[row_key] = row
    return indexed


def normalize_rows(
    s33: dict[tuple[str, int, int], dict[str, str]],
    diff: dict[tuple[str, int, int], dict[str, str]],
    sgd: dict[tuple[str, int, int], dict[str, str]],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {name: [] for name in METHODS}
    for row_key in sorted(s33):
        s33_row = s33[row_key]
        diff_row = diff[row_key]
        sgd_row = sgd[row_key]
        shared = {
            "sample_id": row_key[0],
            "base_seed": row_key[1],
            "snr_db": row_key[2],
            "source_cluster": row_key[0],
        }
        result["S33 strong"].append(
            {
                **shared,
                "psnr": float(s33_row["strong_psnr"]),
                "ms_ssim": float(s33_row["strong_ms_ssim"]),
                "lpips": float(s33_row["strong_lpips"]),
                "failure": float(truth(s33_row["strong_failure"])),
                "runtime_ms": float(s33_row["strong_runtime_ms"]),
                "peak_gpu_memory_mib": None,
            }
        )
        result["DiffJSCC"].append(
            {
                **shared,
                "psnr": float(diff_row["diffjscc_psnr"]),
                "ms_ssim": float(diff_row["diffjscc_ms_ssim"]),
                "lpips": float(diff_row["diffjscc_lpips"]),
                "failure": float(truth(diff_row["diffjscc_failure"])),
                "runtime_ms": float(diff_row["total_runtime_ms"]),
                "peak_gpu_memory_mib": float(diff_row["peak_gpu_memory_mib"]),
            }
        )
        result["SGD paper upper"].append(
            {
                **shared,
                "psnr": float(sgd_row["final_psnr"]),
                "ms_ssim": float(sgd_row["final_ms_ssim"]),
                "lpips": float(sgd_row["final_lpips"]),
                "failure": float(truth(sgd_row["final_failure"])),
                "runtime_ms": float(sgd_row["runtime_ms_per_image"]),
                "peak_gpu_memory_mib": float(sgd_row["peak_gpu_memory_mib"]),
            }
        )
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "unique_sources": len({row["source_cluster"] for row in rows}),
        "mean_psnr": float(np.mean([row["psnr"] for row in rows])),
        "mean_ms_ssim": float(np.mean([row["ms_ssim"] for row in rows])),
        "mean_lpips": float(np.mean([row["lpips"] for row in rows])),
        "failures": int(sum(row["failure"] for row in rows)),
        "failure_rate": float(np.mean([row["failure"] for row in rows])),
        "mean_runtime_ms_per_image": float(np.mean([row["runtime_ms"] for row in rows])),
        "peak_gpu_memory_mib": (
            None
            if all(row["peak_gpu_memory_mib"] is None for row in rows)
            else float(max(row["peak_gpu_memory_mib"] for row in rows if row["peak_gpu_memory_mib"] is not None))
        ),
    }


def cluster_delta(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    metric: str,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    left_by_key = {(row["sample_id"], row["base_seed"], row["snr_db"]): row for row in left}
    right_by_key = {(row["sample_id"], row["base_seed"], row["snr_db"]): row for row in right}
    if set(left_by_key) != set(right_by_key):
        raise RuntimeError("pairwise metric keys differ")
    per_source: dict[str, list[float]] = defaultdict(list)
    for row_key in sorted(left_by_key):
        per_source[row_key[0]].append(
            float(left_by_key[row_key][metric]) - float(right_by_key[row_key][metric])
        )
    sources = sorted(per_source)
    cluster_values = np.asarray([np.mean(per_source[source]) for source in sources], dtype=np.float64)
    mean = float(cluster_values.mean())
    indices = rng.integers(0, len(sources), size=(replicates, len(sources)))
    bootstrap = cluster_values[indices].mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return mean, float(low), float(high)


def pairwise_rows(
    normalized: dict[str, list[dict[str, Any]]], replicates: int
) -> list[dict[str, Any]]:
    pairs = (
        ("S33 strong", "DiffJSCC", "exact-rate_descriptive_Pareto"),
        ("S33 strong", "SGD paper upper", "cross-contract_non-ranking"),
        ("DiffJSCC", "SGD paper upper", "cross-contract_non-ranking"),
    )
    output: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260723)
    for scope_snr in (None, 1, 4, 7, 13, 19):
        scope = "aggregate" if scope_snr is None else f"snr_{scope_snr}"
        for left_name, right_name, claim_scope in pairs:
            left = normalized[left_name]
            right = normalized[right_name]
            if scope_snr is not None:
                left = [row for row in left if row["snr_db"] == scope_snr]
                right = [row for row in right if row["snr_db"] == scope_snr]
            for metric in METRICS:
                mean, low, high = cluster_delta(left, right, metric, replicates, rng)
                output.append(
                    {
                        "scope": scope,
                        "left_method": left_name,
                        "right_method": right_name,
                        "claim_scope": claim_scope,
                        "metric": metric,
                        "left_minus_right": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "lower_is_better": metric in {"lpips", "failure"},
                    }
                )
    return output


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["status"] != "authorized_read_only_analysis":
        raise RuntimeError("S34C-Lite is not authorized")
    if not config["formal_output_creation_authorized"]:
        raise RuntimeError("formal output creation is not authorized")
    if config["new_training"] or config["new_model_inference"] or config["network_download"]:
        raise RuntimeError("S34C-Lite must stay read-only")

    output = resolve(config["planned_outputs"]["directory"])
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    s33_path = resolve(config["inputs"]["s33"]["per_sample"])
    diff_path = resolve(config["inputs"]["diffjscc"]["per_sample"])
    sgd_paths = [resolve(path) for path in config["inputs"]["sgdjscc"]["per_sample_by_seed"]]
    all_inputs = [s33_path, diff_path, *sgd_paths]
    for path in all_inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    s33_rows = read_csv(s33_path)
    diff_rows = read_csv(diff_path)
    sgd_rows = [row for path in sgd_paths for row in read_csv(path)]
    s33 = require_unique(s33_rows, "S33")
    diff = require_unique(diff_rows, "DiffJSCC")
    sgd = require_unique(sgd_rows, "SGD")
    expected_rows = int(config["population"]["rows_per_method"])
    if not (len(s33) == len(diff) == len(sgd) == expected_rows):
        raise RuntimeError(f"row count mismatch: {len(s33)}/{len(diff)}/{len(sgd)}")
    if not (set(s33) == set(diff) == set(sgd)):
        raise RuntimeError("960-key populations do not match")

    canonical_mismatches = 0
    class_mismatches = 0
    embedded_diff_failure_mismatches = 0
    embedded_diff_max_abs = {metric: 0.0 for metric in ("psnr", "ms_ssim", "lpips")}
    for row_key in sorted(s33):
        s33_row, diff_row, sgd_row = s33[row_key], diff[row_key], sgd[row_key]
        if len({s33_row["canonical_noise_sha256"], diff_row["canonical_noise_sha256"], sgd_row["canonical_noise_sha256"]}) != 1:
            canonical_mismatches += 1
        if len({s33_row["wnid"], diff_row["wnid"], sgd_row["wnid"]}) != 1 or len({s33_row["class_idx"], diff_row["class_idx"], sgd_row["class_idx"]}) != 1:
            class_mismatches += 1
        if truth(s33_row["diffjscc_failure"]) != truth(diff_row["diffjscc_failure"]):
            embedded_diff_failure_mismatches += 1
        for metric in embedded_diff_max_abs:
            embedded_diff_max_abs[metric] = max(
                embedded_diff_max_abs[metric],
                abs(float(s33_row[f"diffjscc_{metric}"]) - float(diff_row[f"diffjscc_{metric}"])),
            )
    if canonical_mismatches or class_mismatches or embedded_diff_failure_mismatches or any(embedded_diff_max_abs.values()):
        raise RuntimeError("cross-run audit failed before output creation")

    normalized = normalize_rows(s33, diff, sgd)
    summaries = {method: summarize(rows) for method, rows in normalized.items()}
    per_snr: list[dict[str, Any]] = []
    for snr in config["population"]["snrs_db"]:
        for method in METHODS:
            summary = summarize([row for row in normalized[method] if row["snr_db"] == int(snr)])
            per_snr.append({"snr_db": int(snr), "method": method, **summary})

    rate = config["rate_ledger"]
    method_rows = [
        {
            "method": "S33 strong",
            "ranking_role": "exact-rate_pure-JSCC_reference",
            "main_real_symbols": rate["s33"]["main_real_symbols"],
            "edge_real_symbols": 0,
            "minimum_caption_real_symbols": 0,
            "executed_channel_real_symbols": 16384,
            "minimum_physical_total_real_symbols": 16384,
            "overrun_vs_16384_real_symbols": 0,
            "overrun_vs_16384_percent": 0.0,
            "sender_side_information": "none",
            "receiver_or_external_prior": "none; random-init COCO task training",
            "task_training_contract": "COCO; discrete [1,4,7,13,19] dB",
            "metric_quantization_path": "floor_uint8 primary",
            "fid_available": False,
            "kid_available": False,
            **summaries["S33 strong"],
        },
        {
            "method": "DiffJSCC",
            "ranking_role": "exact-rate_generative_external-positioning",
            "main_real_symbols": rate["diffjscc"]["main_real_symbols"],
            "edge_real_symbols": 0,
            "minimum_caption_real_symbols": 0,
            "executed_channel_real_symbols": 16384,
            "minimum_physical_total_real_symbols": 16384,
            "overrun_vs_16384_real_symbols": 0,
            "overrun_vs_16384_percent": 0.0,
            "sender_side_information": "none; caption generated at receiver from noisy initial reconstruction",
            "receiver_or_external_prior": "SD2.1 + BLIP2 + OpenCLIP; author OpenImage weights",
            "task_training_contract": "OpenImage; continuous [0,14] dB (19 dB out of range)",
            "metric_quantization_path": "author output uint8 truncation",
            "fid_available": False,
            "kid_available": False,
            **summaries["DiffJSCC"],
        },
        {
            "method": "SGD paper upper",
            "ranking_role": "cross-contract_non-ranking_paper-upper",
            "main_real_symbols": rate["sgd_paper_upper"]["main_real_symbols"],
            "edge_real_symbols": rate["sgd_paper_upper"]["active_edge_real_symbols"],
            "minimum_caption_real_symbols": rate["sgd_paper_upper"]["minimum_caption_real_symbols"],
            "executed_channel_real_symbols": 19712,
            "minimum_physical_total_real_symbols": rate["sgd_paper_upper"]["minimum_total_real_symbols"],
            "overrun_vs_16384_real_symbols": rate["sgd_paper_upper"]["overrun_vs_16384_real_symbols"],
            "overrun_vs_16384_percent": 100.0 * rate["sgd_paper_upper"]["overrun_fraction_vs_16384"],
            "sender_side_information": "active edge + four perfect unmetered captions",
            "receiver_or_external_prior": "released diffusion/ControlNet + BLIP2 + CLIP",
            "task_training_contract": "released weights; JSCC ImageNet fixed 10 dB; other pretraining external",
            "metric_quantization_path": "float reconstruction tensor",
            "fid_available": False,
            "kid_available": False,
            **summaries["SGD paper upper"],
        },
    ]
    pairs = pairwise_rows(normalized, int(config["metrics"]["bootstrap_replicates"]))

    audit = {
        "status": "PASS",
        "analysis_id": config["planned_outputs"]["analysis_id"],
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256_file(config_path),
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "input_files": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "rows": len(read_csv(path))}
            for path in all_inputs
        ],
        "unique_keys_each_method": expected_rows,
        "unique_sources": len({row_key[0] for row_key in s33}),
        "seeds": sorted({row_key[1] for row_key in s33}),
        "snrs_db": sorted({row_key[2] for row_key in s33}),
        "canonical_noise_sha_mismatches": canonical_mismatches,
        "class_or_wnid_mismatches": class_mismatches,
        "s33_embedded_diffjscc_failure_mismatches_vs_s30": embedded_diff_failure_mismatches,
        "s33_embedded_diffjscc_metric_max_abs_vs_s30": embedded_diff_max_abs,
        "metric_path_warning": "S33/DiffJSCC primary metrics use uint8 truncation; SGD frozen metrics use float tensors. Cross-contract deltas are descriptive and not a rank.",
        "new_training": False,
        "new_model_inference": False,
        "network_download": False,
        "official_imagenette_validation_accessed": False,
    }
    ledger = {
        "communication_budget_reference_real_symbols": 16384,
        "methods": {
            "S33 strong": rate["s33"],
            "DiffJSCC": rate["diffjscc"],
            "SGD paper upper": {
                **rate["sgd_paper_upper"],
                "executed_channel_real_symbols_excluding_perfect_caption": 19712,
                "minimum_caption_accounting_note": "4 captions x 536 raw bits, one unprotected BPSK real coordinate per bit; robust protection would cost more",
            },
        },
        "interpretation": {
            "external_prior_is_channel_rate": False,
            "S33_vs_DiffJSCC_exact_rate": True,
            "SGD_direct_ranking_allowed": False,
            "FID_available": False,
            "KID_available": False,
        },
    }
    summary = {
        "analysis_id": config["planned_outputs"]["analysis_id"],
        "status": "PASS",
        "claim_scope": "rate-and-prior transparency; no global ranking",
        "population": config["population"],
        "method_means": summaries,
        "rate": ledger,
        "aggregate_pairwise_descriptive_deltas_with_ci": [row for row in pairs if row["scope"] == "aggregate"],
        "limitations": [
            "Known policy-development population, not independent final test.",
            "No common FID/KID is available in frozen results.",
            "Training data, model capacity, external pretraining, and compute are not aligned.",
            "SGD paper-upper captions are perfect and unmetered; 21,856 real is only the minimum unprotected physical accounting.",
            "S33/DiffJSCC and SGD use different frozen metric quantization paths; cross-contract deltas are descriptive.",
        ],
        "verdict": "EXACT_RATE_S33_DIFFJSCC_FIDELITY_PERCEPTION_PARETO__SGD_NONRANKING_UPPER",
    }

    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "unified_method_table.csv", method_rows)
    write_csv(output / "per_snr_table.csv", per_snr)
    write_csv(output / "pairwise_descriptive_deltas_with_ci.csv", pairs)
    write_json(output / "rate_and_prior_ledger.json", ledger)
    write_json(output / "input_audit.json", audit)
    write_json(output / "summary.json", summary)
    shutil.copy2(config_path, output / "config_snapshot.yaml")
    shutil.copy2(Path(__file__).resolve(), output / Path(__file__).name)
    print(json.dumps({"status": "PASS", "output": str(output.relative_to(ROOT)), "method_means": summaries}, indent=2))


if __name__ == "__main__":
    main()
