#!/usr/bin/env python3
"""Validate and aggregate the frozen external common-comparison pilot."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"invalid boolean value: {value!r}")
    return normalized == "true"


def mean(rows: list[dict[str, str]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)


def method_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "mean_psnr": mean(rows, "final_psnr"),
        "mean_ms_ssim": mean(rows, "final_ms_ssim"),
        "mean_lpips": mean(rows, "final_lpips"),
        "final_failures": sum(as_bool(row["final_failure"]) for row in rows),
        "new_errors_vs_deepjscc": sum(
            as_bool(row["new_error_vs_deepjscc"]) for row in rows
        ),
        "repairs_vs_deepjscc": sum(as_bool(row["repair_vs_deepjscc"]) for row in rows),
        "mean_runtime_ms_per_image": mean(rows, "runtime_ms_per_image"),
        "peak_gpu_memory_mib": max(float(row["peak_gpu_memory_mib"]) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/external_common_comparison_pilot.yaml")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    methods = {
        "ours_m3": resolve(config["outputs"]["ours"]),
        "sgd_jscc_common": resolve(config["outputs"]["sgd_jscc"]),
        "sing_zero_style": resolve(config["outputs"]["sing_zero_style"]),
    }
    rows_by_method = {
        name: read_csv(path / "per_sample.csv") for name, path in methods.items()
    }
    expected_keys = {
        (str(item["sample_id"]), float(snr))
        for item in config["population"]["samples"]
        for snr in config["channel"]["snrs_db"]
    }
    keyed: dict[str, dict[tuple[str, float], dict[str, str]]] = {}
    for method, rows in rows_by_method.items():
        if len(rows) != 40:
            raise RuntimeError(f"{method} has {len(rows)} rows instead of 40")
        mapping = {(row["sample_id"], float(row["snr_db"])): row for row in rows}
        if set(mapping) != expected_keys or len(mapping) != 40:
            raise RuntimeError(f"{method} has a sample/SNR key mismatch")
        for row in rows:
            if int(row["total_real_symbols"]) != 65536:
                raise RuntimeError(f"{method} violates the real-symbol budget")
            if int(row["total_complex_channel_uses"]) != 32768:
                raise RuntimeError(f"{method} violates the complex-use budget")
            if abs(float(row["cbr"]) - 1.0 / 6.0) > 1e-15:
                raise RuntimeError(f"{method} violates CBR=1/6")
            if row["noise_variance_convention"] != "complex_awgn_per_real_half_variance":
                raise RuntimeError(f"{method} uses another AWGN convention")
        keyed[method] = mapping
    for key in sorted(expected_keys):
        noise_hashes = {
            keyed[method][key]["canonical_noise_sha256"] for method in keyed
        }
        if len(noise_hashes) != 1:
            raise RuntimeError(f"canonical channel noise differs for {key}: {noise_hashes}")
        reference_fields = (
            "deepjscc_prediction",
            "deepjscc_correct",
            "deepjscc_psnr",
            "deepjscc_ms_ssim",
            "deepjscc_lpips",
        )
        reference_values = {
            tuple(keyed[method][key][field] for field in reference_fields)
            for method in keyed
        }
        if len(reference_values) != 1:
            raise RuntimeError(f"DeepJSCC reference differs for {key}")

    aggregate = resolve(config["outputs"]["aggregate"])
    if aggregate.exists():
        raise FileExistsError(aggregate)
    aggregate.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, aggregate / "config_snapshot.yaml")
    summaries = {
        method: method_summary(rows) for method, rows in rows_by_method.items()
    }
    ours_rows = rows_by_method["ours_m3"]
    deepjscc_summary = {
        "rows": len(ours_rows),
        "mean_psnr": mean(ours_rows, "deepjscc_psnr"),
        "mean_ms_ssim": mean(ours_rows, "deepjscc_ms_ssim"),
        "mean_lpips": mean(ours_rows, "deepjscc_lpips"),
        "failures": sum(not as_bool(row["deepjscc_correct"]) for row in ours_rows),
    }

    pairs = [
        ("ours_m3", "sgd_jscc_common"),
        ("ours_m3", "sing_zero_style"),
        ("sgd_jscc_common", "sing_zero_style"),
    ]
    paired_rows: list[dict[str, Any]] = []
    paired_summary: dict[str, Any] = {}
    for first, second in pairs:
        name = f"{first}_minus_{second}"
        deltas = []
        for key in sorted(expected_keys, key=lambda value: (value[1], value[0])):
            left = keyed[first][key]
            right = keyed[second][key]
            item = {
                "comparison": name,
                "sample_id": key[0],
                "snr_db": key[1],
                "psnr_delta": float(left["final_psnr"]) - float(right["final_psnr"]),
                "ms_ssim_delta": float(left["final_ms_ssim"])
                - float(right["final_ms_ssim"]),
                "lpips_delta": float(left["final_lpips"]) - float(right["final_lpips"]),
                "first_correct": as_bool(left["final_correct"]),
                "second_correct": as_bool(right["final_correct"]),
            }
            deltas.append(item)
            paired_rows.append(item)
        paired_summary[name] = {
            "mean_psnr_delta": sum(row["psnr_delta"] for row in deltas) / len(deltas),
            "mean_ms_ssim_delta": sum(row["ms_ssim_delta"] for row in deltas)
            / len(deltas),
            "mean_lpips_delta": sum(row["lpips_delta"] for row in deltas) / len(deltas),
            "psnr_first_wins": sum(row["psnr_delta"] > 0 for row in deltas),
            "ms_ssim_first_wins": sum(row["ms_ssim_delta"] > 0 for row in deltas),
            "lpips_first_wins": sum(row["lpips_delta"] < 0 for row in deltas),
            "semantic_first_only_correct": sum(
                row["first_correct"] and not row["second_correct"] for row in deltas
            ),
            "semantic_second_only_correct": sum(
                row["second_correct"] and not row["first_correct"] for row in deltas
            ),
        }

    by_snr: dict[str, Any] = {}
    for snr in map(float, config["channel"]["snrs_db"]):
        by_snr[str(int(snr))] = {}
        for method, rows in rows_by_method.items():
            subset = [row for row in rows if float(row["snr_db"]) == snr]
            by_snr[str(int(snr))][method] = method_summary(subset)
        ref = [row for row in ours_rows if float(row["snr_db"]) == snr]
        by_snr[str(int(snr))]["deepjscc_reference"] = {
            "mean_psnr": mean(ref, "deepjscc_psnr"),
            "mean_ms_ssim": mean(ref, "deepjscc_ms_ssim"),
            "mean_lpips": mean(ref, "deepjscc_lpips"),
            "failures": sum(not as_bool(row["deepjscc_correct"]) for row in ref),
        }

    result = {
        "analysis_id": config["analysis_id"],
        "status": "PASS",
        "validation": {
            "methods": list(methods),
            "rows_per_method": 40,
            "total_rows": 120,
            "sample_snr_keys_identical": True,
            "canonical_noise_sha256_identical_each_key": True,
            "deepjscc_reference_identical_each_key": True,
            "exact_total_real_symbols_each_row": 65536,
            "exact_total_complex_channel_uses_each_row": 32768,
            "exact_cbr_each_row": 1.0 / 6.0,
            "awgn_convention": "complex_awgn_per_real_half_variance",
        },
        "claim_scope": {
            "pilot_only": True,
            "outcome_claims_allowed": False,
            "reason": "eight policy-development images are sufficient for integration and directional diagnostics only",
            "sing_label_caveat": "mechanism-level final-only projection, not an exact SING/DDNM reproduction",
            "sgd_label_caveat": "project common-contract adapter, not author-native rate accounting",
        },
        "deepjscc_reference": deepjscc_summary,
        "methods": summaries,
        "paired": paired_summary,
        "by_snr": by_snr,
    }
    (aggregate / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (aggregate / "paired_differences.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
