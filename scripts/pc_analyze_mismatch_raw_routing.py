#!/usr/bin/env python3
"""Recompute the frozen source-anchor-mismatch three-way route from stored sender rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pc_imagenette_sender_inbudget_awgn_audit import (  # noqa: E402
    image_cluster_any_event_endpoint,
    paired_image_cluster_inference,
)


BOOL_FIELDS = {
    "clean_correct",
    "accepted",
    "cross_model_source_anchor_accepted",
    "anchor_correct",
    "raw_correct",
    "posterior_correct",
    "reference_raw_correct",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--primary-snrs", default="1,4,7")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=271828)
    parser.add_argument("--max-system-new-cluster-upper", type=float, default=0.005)
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"invalid boolean: {value!r}")
    return normalized == "true"


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "channel_seed",
        "snr_db",
        "sample_id",
        *BOOL_FIELDS,
        "anchor_psnr",
        "raw_psnr",
        "posterior_psnr",
        "anchor_lpips",
        "raw_lpips",
        "posterior_lpips",
        "reference_raw_psnr",
        "reference_raw_lpips",
    }
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise RuntimeError(f"input CSV missing fields: {sorted(missing)}")
    for row in rows:
        for field in BOOL_FIELDS:
            row[field] = parse_bool(str(row[field]))
        row["channel_seed"] = int(row["channel_seed"])
        row["snr_db"] = float(row["snr_db"])
        for field in (
            "anchor_psnr",
            "raw_psnr",
            "posterior_psnr",
            "anchor_lpips",
            "raw_lpips",
            "posterior_lpips",
            "reference_raw_psnr",
            "reference_raw_lpips",
        ):
            row[field] = float(row[field])
    return rows


def route(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        if bool(row["accepted"]):
            selected = "posterior"
        elif not bool(row["cross_model_source_anchor_accepted"]):
            selected = "raw"
        else:
            selected = "anchor"
        routed = dict(row)
        routed["selected_candidate"] = selected
        routed["final_correct"] = bool(row[f"{selected}_correct"])
        routed["final_psnr"] = float(row[f"{selected}_psnr"])
        routed["final_lpips"] = float(row[f"{selected}_lpips"])
        output.append(routed)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def main() -> None:
    args = parse_args()
    input_path = resolve(args.input_csv)
    output_dir = resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {output_dir}")
    if args.bootstrap_replicates <= 0:
        raise ValueError("bootstrap-replicates must be positive")
    primary_snrs = {float(item) for item in args.primary_snrs.split(",")}
    rows = route(load_rows(input_path))
    channel_seeds = sorted({int(row["channel_seed"]) for row in rows})
    snrs = sorted({float(row["snr_db"]) for row in rows})
    sample_ids = {str(row["sample_id"]) for row in rows}
    clean_sample_ids = {
        str(row["sample_id"]) for row in rows if bool(row["clean_correct"])
    }
    expected_rows = len(channel_seeds) * len(snrs) * len(sample_ids)
    if len(rows) != expected_rows:
        raise RuntimeError(f"row grid incomplete: {len(rows)} != {expected_rows}")

    summary: list[dict[str, Any]] = []
    for snr in snrs:
        subset = [row for row in rows if float(row["snr_db"]) == snr]
        clean = [row for row in subset if bool(row["clean_correct"])]
        summary.append(
            {
                "snr_db": snr,
                "rows": len(subset),
                "clean_rows": len(clean),
                "reference_raw_failure": sum(
                    not bool(row["reference_raw_correct"]) for row in clean
                ),
                "final_failure": sum(not bool(row["final_correct"]) for row in clean),
                "final_minus_reference_raw_psnr": sum(
                    float(row["final_psnr"]) - float(row["reference_raw_psnr"])
                    for row in subset
                )
                / len(subset),
                "final_minus_reference_raw_lpips": sum(
                    float(row["final_lpips"]) - float(row["reference_raw_lpips"])
                    for row in subset
                )
                / len(subset),
            }
        )

    primary = [
        row
        for row in rows
        if bool(row["clean_correct"]) and float(row["snr_db"]) in primary_snrs
    ]
    new_endpoint = image_cluster_any_event_endpoint(
        primary,
        lambda row: bool(row["reference_raw_correct"]),
        lambda row: bool(row["reference_raw_correct"])
        and not bool(row["final_correct"]),
    )
    repair_rows = [
        row
        for row in primary
        if not bool(row["reference_raw_correct"]) and bool(row["final_correct"])
    ]
    repair_ids = {str(row["sample_id"]) for row in repair_rows}
    inference = paired_image_cluster_inference(
        rows,
        primary_snrs=primary_snrs,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
        all_sample_ids=sample_ids,
        clean_sample_ids=clean_sample_ids,
        expected_all_rows_per_sample=len(channel_seeds) * len(snrs),
        expected_primary_rows_per_sample=len(channel_seeds) * len(primary_snrs),
    )
    failure_ci = inference["primary_failure_rate_delta_final_minus_reference_raw"]
    psnr_ci = inference["all_snr_psnr_delta_final_minus_reference_raw"]
    lpips_ci = inference["all_snr_lpips_delta_final_minus_reference_raw"]
    gates = {
        "aggregate_failure_not_above_reference": sum(
            not bool(row["final_correct"]) for row in primary
        )
        <= sum(not bool(row["reference_raw_correct"]) for row in primary),
        "each_snr_failure_not_above_reference": all(
            int(row["final_failure"]) <= int(row["reference_raw_failure"])
            for row in summary
            if float(row["snr_db"]) in primary_snrs
        ),
        "each_snr_psnr_positive": all(
            float(row["final_minus_reference_raw_psnr"]) > 0 for row in summary
        ),
        "system_new_cluster_upper_within_limit": float(
            new_endpoint["image_cluster_any_event_clopper_pearson_upper_95"]
        )
        <= args.max_system_new_cluster_upper,
        "failure_ci_upper_strictly_below_zero": float(failure_ci["ci95_upper"]) < 0,
        "psnr_ci_lower_strictly_above_zero": float(psnr_ci["ci95_lower"]) > 0,
        "lpips_ci_upper_nonpositive": float(lpips_ci["ci95_upper"]) <= 0,
    }
    metrics = {
        "source_csv": str(input_path.relative_to(PROJECT_ROOT)),
        "source_csv_sha256": sha256(input_path),
        "official_imagenette_accessed": False,
        "routing_rule": (
            "accepted->posterior; rejected and recovered-source/anchor mismatch->raw; "
            "other rejected->anchor"
        ),
        "channel_seeds": channel_seeds,
        "snrs_db": snrs,
        "primary_snrs_db": sorted(primary_snrs),
        "rows": len(rows),
        "images": len(sample_ids),
        "primary_reference_failure_rows": sum(
            not bool(row["reference_raw_correct"]) for row in primary
        ),
        "primary_final_failure_rows": sum(
            not bool(row["final_correct"]) for row in primary
        ),
        "primary_system_new_rows": int(new_endpoint["event_rows"]),
        "primary_system_new_image_clusters": int(new_endpoint["event_image_clusters"]),
        "primary_system_new_eligible_image_clusters": int(
            new_endpoint["eligible_image_clusters"]
        ),
        "primary_system_new_cluster_upper_95": float(
            new_endpoint["image_cluster_any_event_clopper_pearson_upper_95"]
        ),
        "primary_system_repair_rows": len(repair_rows),
        "primary_system_repair_image_clusters": len(repair_ids),
        "summary": summary,
        "paired_image_cluster_inference": inference,
        "gates": gates,
        "verdict": "POSITIVE" if all(gates.values()) else "NEGATIVE",
    }
    decision_fields = [
        "channel_seed",
        "snr_db",
        "sample_id",
        "clean_correct",
        "accepted",
        "cross_model_source_anchor_accepted",
        "selected_candidate",
        "anchor_correct",
        "raw_correct",
        "posterior_correct",
        "reference_raw_correct",
        "final_correct",
        "reference_raw_psnr",
        "final_psnr",
        "reference_raw_lpips",
        "final_lpips",
    ]
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "routing_decisions.csv", rows, decision_fields)
    write_csv(output_dir / "summary.csv", summary, list(summary[0]))
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps({"verdict": metrics["verdict"], "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
