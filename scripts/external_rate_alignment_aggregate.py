#!/usr/bin/env python3
"""Validate and aggregate both preregistered external rate alignments."""

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


def read_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(resolve(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("YAML root must be a mapping")
    return value


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with resolve(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def keyed(rows: list[dict[str, str]]) -> dict[tuple[str, float], dict[str, str]]:
    result = {(row["sample_id"], float(row["snr_db"])): row for row in rows}
    if len(rows) != 40 or len(result) != 40:
        raise RuntimeError("every pilot method must have 40 unique sample/SNR rows")
    return result


def as_bool(value: str) -> bool:
    return str(value).lower() == "true"


def summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "mean_psnr": sum(float(row["final_psnr"]) for row in rows) / len(rows),
        "mean_ms_ssim": sum(float(row["final_ms_ssim"]) for row in rows) / len(rows),
        "mean_lpips": sum(float(row["final_lpips"]) for row in rows) / len(rows),
        "failures": sum(as_bool(row["final_failure"]) for row in rows),
        "mean_runtime_ms_per_image": sum(float(row["runtime_ms_per_image"]) for row in rows)
        / len(rows),
        "peak_gpu_memory_mib": max(float(row["peak_gpu_memory_mib"]) for row in rows),
    }


def paired(
    left: dict[tuple[str, float], dict[str, str]],
    right: dict[tuple[str, float], dict[str, str]],
) -> dict[str, Any]:
    if set(left) != set(right):
        raise RuntimeError("paired methods use different sample/SNR keys")
    deltas = [
        (
            float(left[key]["final_psnr"]) - float(right[key]["final_psnr"]),
            float(left[key]["final_ms_ssim"]) - float(right[key]["final_ms_ssim"]),
            float(left[key]["final_lpips"]) - float(right[key]["final_lpips"]),
        )
        for key in sorted(left, key=lambda item: (item[1], item[0]))
    ]
    return {
        "mean_psnr_delta": sum(item[0] for item in deltas) / len(deltas),
        "mean_ms_ssim_delta": sum(item[1] for item in deltas) / len(deltas),
        "mean_lpips_delta": sum(item[2] for item in deltas) / len(deltas),
        "psnr_left_wins": sum(item[0] > 0 for item in deltas),
        "ms_ssim_left_wins": sum(item[1] > 0 for item in deltas),
        "lpips_left_wins": sum(item[2] < 0 for item in deltas),
        "left_only_correct": sum(
            as_bool(left[key]["final_correct"]) and not as_bool(right[key]["final_correct"])
            for key in left
        ),
        "right_only_correct": sum(
            as_bool(right[key]["final_correct"]) and not as_bool(left[key]["final_correct"])
            for key in left
        ),
    }


def require_same_noise(*maps: dict[tuple[str, float], dict[str, str]]) -> None:
    keys = set(maps[0])
    if any(set(value) != keys for value in maps[1:]):
        raise RuntimeError("sample/SNR keys differ")
    for key in keys:
        if len({value[key]["canonical_noise_sha256"] for value in maps}) != 1:
            raise RuntimeError(f"canonical noise differs at {key}")


def write_result(output: Path, config_paths: list[Path], result: dict[str, Any]) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    for path in config_paths:
        shutil.copy2(path, output / path.name)
    (output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--author-config", default="configs/external_author_rate_alignment_pilot.yaml"
    )
    parser.add_argument(
        "--project-config",
        default="configs/external_project_rate_sgd_reallocation_pilot.yaml",
    )
    parser.add_argument("--author-only", action="store_true")
    args = parser.parse_args()
    author_path = resolve(args.author_config)
    project_path = resolve(args.project_config)
    common_path = resolve("configs/external_common_comparison_pilot.yaml")
    author = read_yaml(author_path)
    project = read_yaml(project_path)
    common = read_yaml(common_path)

    author_deep_rows = read_rows(Path(author["outputs"]["deepjscc"]) / "per_sample.csv")
    author_sgd_rows = read_rows(Path(author["outputs"]["sgd_jscc"]) / "per_sample.csv")
    author_deep = keyed(author_deep_rows)
    author_sgd = keyed(author_sgd_rows)
    require_same_noise(author_deep, author_sgd)
    for rows in (author_deep_rows, author_sgd_rows):
        if any(int(row["total_real_symbols"]) != 19712 for row in rows):
            raise RuntimeError("author-rate image branch is not exactly 19,712 real symbols")
    author_result = {
        "analysis_id": author["analysis_id"],
        "status": "PASS",
        "validation": {
            "rows_per_method": 40,
            "same_sample_snr_keys": True,
            "same_canonical_noise_each_key": True,
            "image_branch_real_symbols": 19712,
            "complex_channel_uses": 9856,
            "exact_cbr": author["rate"]["exact_cbr"],
        },
        "methods": {
            "deepjscc_exact_rate": summary(author_deep_rows),
            "sgd_jscc_paper_protocol": summary(author_sgd_rows),
        },
        "paired_sgd_minus_deepjscc": paired(author_sgd, author_deep),
        "caveat": (
            "SGD-JSCC uses perfect unmetered captions under the paper protocol, so this is "
            "not a strict end-to-end physical-rate match."
        ),
        "claim_scope": "eight-image_directional_pilot_only",
    }
    write_result(
        resolve(author["outputs"]["aggregate"]), [author_path], author_result
    )

    if args.author_only:
        print(json.dumps({"author_rate": author_result}, ensure_ascii=False, indent=2))
        return

    realloc_rows = read_rows(Path(project["output_dir"]) / "per_sample.csv")
    old_sgd_rows = read_rows(Path(common["outputs"]["sgd_jscc"]) / "per_sample.csv")
    ours_rows = read_rows(Path(common["outputs"]["ours"]) / "per_sample.csv")
    realloc = keyed(realloc_rows)
    old_sgd = keyed(old_sgd_rows)
    ours = keyed(ours_rows)
    require_same_noise(realloc, old_sgd, ours)
    for rows in (realloc_rows, old_sgd_rows, ours_rows):
        if any(int(row["total_real_symbols"]) != 65536 for row in rows):
            raise RuntimeError("project-rate method violates 65,536-real-symbol budget")
    project_result = {
        "analysis_id": project["analysis_id"],
        "status": "PASS",
        "validation": {
            "rows_per_method": 40,
            "same_sample_snr_keys": True,
            "same_canonical_noise_each_key": True,
            "total_real_symbols": 65536,
            "complex_channel_uses": 32768,
            "cbr": 1.0 / 6.0,
        },
        "methods": {
            "sgd_jscc_r1_text_r21": summary(old_sgd_rows),
            "sgd_jscc_main_r2_text_r13": summary(realloc_rows),
            "ours_m3": summary(ours_rows),
        },
        "paired_reallocated_minus_old_sgd": paired(realloc, old_sgd),
        "paired_ours_minus_reallocated_sgd": paired(ours, realloc),
        "interpretation": (
            "released-weight allocation sensitivity only; repetition increases robustness, "
            "not SGD-JSCC representation capacity"
        ),
        "claim_scope": "eight-image_directional_pilot_only",
    }
    write_result(
        resolve(Path(project["output_dir"]).parent / "aggregate"),
        [project_path, common_path],
        project_result,
    )
    print(json.dumps({"author_rate": author_result, "project_rate": project_result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
