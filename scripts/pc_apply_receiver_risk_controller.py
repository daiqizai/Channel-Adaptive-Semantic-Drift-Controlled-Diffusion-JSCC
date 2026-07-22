#!/usr/bin/env python3
"""Apply a frozen receiver-risk percentile controller to a preregistered audit table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts")]

from pc_fit_receiver_risk_controller import (  # noqa: E402
    parse_bool,
    read_csv,
    row_key,
    score_with_references,
    sha256_file,
    simulate,
    verify_hash,
)
from pc_imagenette_supervised_audit import RECEIVER_RISK_FEATURE_COLUMNS  # noqa: E402
from pc_posterior_consistency_replication import load_yaml, resolve, write_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    frozen = config["frozen_controller"]
    controller_path = resolve(frozen["controller_json"])
    cdf_path = resolve(frozen["empirical_cdfs"])
    verify_hash(controller_path, str(frozen["controller_sha256"]))
    verify_hash(cdf_path, str(frozen["empirical_cdfs_sha256"]))
    controller = json.loads(controller_path.read_text(encoding="utf-8"))
    if bool(controller.get("official_val_accessed")):
        raise RuntimeError("frozen controller records official validation access")
    if bool(controller.get("teacher_fields_in_controller_inputs")):
        raise RuntimeError("frozen controller includes teacher inputs")
    components = [dict(item) for item in controller["components"]]
    input_features = [str(item["feature"]) for item in components]
    if input_features != list(controller["receiver_input_features"]):
        raise RuntimeError("controller component/input feature order differs")
    if any(name not in RECEIVER_RISK_FEATURE_COLUMNS for name in input_features):
        raise RuntimeError("controller contains a non-receiver feature")
    source = config["audit_source"]
    extraction_config_path = resolve(source["extraction_config"])
    verify_hash(extraction_config_path, str(source["extraction_config_sha256"]))
    extraction_config = load_yaml(extraction_config_path)
    if extraction_config["channel_seeds"] != [int(source["expected_channel_seed"])]:
        raise RuntimeError("extraction config channel seed differs from frozen audit seed")
    if bool(extraction_config["imagenette"].get("official_val_accessed")):
        raise RuntimeError("extraction config permits official validation access")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "analysis_id": config["analysis_id"],
                    "channel_seed": source["expected_channel_seed"],
                    "controller_sha256": frozen["controller_sha256"],
                    "empirical_cdfs_sha256": frozen["empirical_cdfs_sha256"],
                    "threshold": controller["threshold"],
                    "receiver_input_features": input_features,
                    "teacher_fields_in_controller_inputs": False,
                    "official_val_accessed": False,
                },
                indent=2,
            )
        )
        return

    risk_path = resolve(source["risk_features_csv"])
    audit_path = resolve(source["audit_csv"])
    schema_path = resolve(source["schema_json"])
    risk_rows = read_csv(risk_path)
    audit_rows = read_csv(audit_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    expected_rows = int(source["expected_rows"])
    if len(risk_rows) != expected_rows or len(audit_rows) != expected_rows:
        raise RuntimeError("frozen audit input row count mismatch")
    risk_keys = [row_key(row) for row in risk_rows]
    audit_by_key = {row_key(row): row for row in audit_rows}
    if len(set(risk_keys)) != expected_rows or set(risk_keys) != set(audit_by_key):
        raise RuntimeError("frozen audit keys are not unique and aligned")
    expected_seed = int(source["expected_channel_seed"])
    if {key[1] for key in risk_keys} != {expected_seed}:
        raise RuntimeError("frozen audit table contains an unexpected channel seed")
    if bool(schema.get("official_val_accessed")):
        raise RuntimeError("frozen audit schema records official validation access")
    if str(schema.get("feature_version")) != str(controller["feature_version"]):
        raise RuntimeError("audit/controller feature versions differ")
    with np.load(cdf_path, allow_pickle=False) as artifact:
        references = {name: np.asarray(artifact[name]) for name in input_features}
    scores = score_with_references(risk_rows, components, references)
    if not np.isfinite(scores).all():
        raise RuntimeError("frozen controller produced non-finite scores")
    reference_mask = np.asarray(
        [
            parse_bool(row["teacher_clean_correct"])
            and parse_bool(row["teacher_anchor_correct"])
            for row in risk_rows
        ],
        dtype=bool,
    )
    posterior_quality = {
        "mean_psnr_gain": float(
            np.mean(
                [
                    float(row["posterior_psnr"]) - float(row["raw_psnr"])
                    for row in audit_rows
                ]
            )
        ),
        "mean_lpips_gain": float(
            np.mean(
                [
                    float(row["posterior_lpips"]) - float(row["raw_lpips"])
                    for row in audit_rows
                ]
            )
        ),
    }
    result = simulate(
        risk_rows,
        audit_by_key,
        scores,
        float(controller["threshold"]),
        reference_mask,
        {float(item) for item in config["evaluation"]["primary_snrs"]},
        posterior_quality,
        dict(config["evaluation"]["gates"]),
    )
    decisions = result.pop("decisions")
    primary_snrs = {float(item) for item in config["evaluation"]["primary_snrs"]}
    primary_audit = [
        row
        for row in audit_rows
        if parse_bool(row["clean_correct"]) and float(row["snr_db"]) in primary_snrs
    ]
    scratch_failure = sum(not parse_bool(row["final_correct"]) for row in primary_audit)
    scratch_new = sum(
        parse_bool(row["anchor_correct"]) and not parse_bool(row["final_correct"])
        for row in primary_audit
    )
    quality_by_snr: dict[str, dict[str, float]] = {}
    for snr in sorted({float(row["snr_db"]) for row in decisions}):
        subset = [row for row in decisions if float(row["snr_db"]) == snr]
        quality_by_snr[str(int(snr))] = {
            "mean_final_minus_raw_psnr": float(
                np.mean([float(row["final_psnr"]) - float(row["raw_psnr"]) for row in subset])
            ),
            "mean_final_minus_raw_lpips": float(
                np.mean(
                    [float(row["final_lpips"]) - float(row["raw_lpips"]) for row in subset]
                )
            ),
        }
    unique_clean_images = len(
        {row["sample_id"] for row in audit_rows if parse_bool(row["clean_correct"])}
    )
    extra_gates = {
        "minimum_clean_images": unique_clean_images
        >= int(config["evaluation"]["minimum_clean_images"]),
        "reference_reject_rate_within_frozen_range": float(
            config["evaluation"]["minimum_reference_reject_rate"]
        )
        <= float(result["reference_reject_rate"])
        <= float(config["evaluation"]["maximum_reference_reject_rate"]),
        "primary_failure_each_snr_not_above_raw": all(
            item["final_failure"] <= item["raw_failure"]
            for item in result["per_snr"].values()
        ),
        "quality_each_snr": all(
            item["mean_final_minus_raw_psnr"] > 0
            and item["mean_final_minus_raw_lpips"] <= 0
            for item in quality_by_snr.values()
        ),
    }
    all_gates = {**result["gates"], **extra_gates}
    verdict = "POSITIVE" if all(all_gates.values()) else "NEGATIVE"
    output = resolve(config["output_dir"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    write_csv(output / "decisions.csv", decisions)
    payload: dict[str, Any] = {
        "config": config,
        "controller": controller,
        "input_artifacts": {
            "extraction_config_sha256": sha256_file(extraction_config_path),
            "risk_features_sha256": sha256_file(risk_path),
            "audit_sha256": sha256_file(audit_path),
            "schema_sha256": sha256_file(schema_path),
            "controller_sha256": sha256_file(controller_path),
            "empirical_cdfs_sha256": sha256_file(cdf_path),
        },
        "posterior_quality": posterior_quality,
        "result": result,
        "quality_by_snr": quality_by_snr,
        "existing_scratch_gate_primary": {
            "final_failure": scratch_failure,
            "final_new": scratch_new,
        },
        "unique_clean_images": unique_clean_images,
        "gates": all_gates,
        "verdict": verdict,
        "official_val_accessed": False,
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "result": result,
                "quality_by_snr": quality_by_snr,
                "existing_scratch_gate_primary": payload["existing_scratch_gate_primary"],
                "gates": all_gates,
                "verdict": verdict,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
