#!/usr/bin/env python3
"""Fit the frozen low-complexity receiver-risk percentile controller on policy-dev."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts")]

from pc_imagenette_supervised_audit import (  # noqa: E402
    RECEIVER_RISK_FEATURE_COLUMNS,
    clopper_pearson_upper_95,
)
from pc_posterior_consistency_replication import load_yaml, resolve, write_csv  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path}: expected={expected}, actual={actual}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, Any]) -> tuple[str, int, float]:
    return str(row["sample_id"]), int(row["channel_seed"]), float(row["snr_db"])


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if str(value) == "True":
        return True
    if str(value) == "False":
        return False
    raise ValueError(f"not a serialized boolean: {value!r}")


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    if reference.ndim != 1 or len(reference) == 0:
        raise ValueError("empirical CDF reference must be a non-empty vector")
    if not np.isfinite(reference).all() or not np.isfinite(values).all():
        raise ValueError("empirical CDF inputs must be finite")
    ordered = np.sort(reference.astype(np.float64, copy=False))
    return np.searchsorted(ordered, values, side="right") / len(ordered)


def build_score(
    rows: list[dict[str, str]],
    reference_mask: np.ndarray,
    components: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if reference_mask.shape != (len(rows),) or not reference_mask.any():
        raise ValueError("invalid percentile reference mask")
    receiver_whitelist = set(RECEIVER_RISK_FEATURE_COLUMNS)
    oriented_references: dict[str, np.ndarray] = {}
    percentiles: list[np.ndarray] = []
    for component in components:
        name = str(component["feature"])
        direction = int(component["direction"])
        if name not in receiver_whitelist:
            raise ValueError(f"controller input is not receiver-whitelisted: {name!r}")
        if direction not in {-1, 1}:
            raise ValueError(f"feature direction must be -1 or 1: {component!r}")
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64) * direction
        reference = np.sort(values[reference_mask])
        oriented_references[name] = reference
        percentiles.append(empirical_percentile(reference, values))
    return np.mean(np.stack(percentiles, axis=1), axis=1), oriented_references


def score_with_references(
    rows: list[dict[str, str]],
    components: list[dict[str, Any]],
    oriented_references: dict[str, np.ndarray],
) -> np.ndarray:
    percentiles = []
    for component in components:
        name = str(component["feature"])
        direction = int(component["direction"])
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64) * direction
        percentiles.append(empirical_percentile(oriented_references[name], values))
    return np.mean(np.stack(percentiles, axis=1), axis=1)


def simulate(
    risk_rows: list[dict[str, str]],
    audit_by_key: dict[tuple[str, int, float], dict[str, str]],
    scores: np.ndarray,
    threshold: float,
    reference_mask: np.ndarray,
    primary_snrs: set[float],
    posterior_quality: dict[str, float],
    gates_config: dict[str, Any],
) -> dict[str, Any]:
    reject = scores >= threshold
    primary_records: list[dict[str, Any]] = []
    psnr_delta: list[float] = []
    lpips_delta: list[float] = []
    quality_by_seed: dict[int, dict[str, list[float]]] = {}
    decisions: list[dict[str, Any]] = []
    for index, risk_row in enumerate(risk_rows):
        audit = audit_by_key[row_key(risk_row)]
        channel_seed = int(audit["channel_seed"])
        rejected = bool(reject[index])
        final_correct = (
            parse_bool(audit["anchor_correct"])
            if rejected
            else parse_bool(audit["posterior_correct"])
        )
        final_psnr = float(audit["anchor_psnr"] if rejected else audit["posterior_psnr"])
        final_lpips = float(audit["anchor_lpips"] if rejected else audit["posterior_lpips"])
        row_psnr_delta = final_psnr - float(audit["raw_psnr"])
        row_lpips_delta = final_lpips - float(audit["raw_lpips"])
        psnr_delta.append(row_psnr_delta)
        lpips_delta.append(row_lpips_delta)
        seed_quality = quality_by_seed.setdefault(channel_seed, {"psnr": [], "lpips": []})
        seed_quality["psnr"].append(row_psnr_delta)
        seed_quality["lpips"].append(row_lpips_delta)
        if parse_bool(audit["clean_correct"]) and float(audit["snr_db"]) in primary_snrs:
            primary_records.append(
                {
                    "sample_id": audit["sample_id"],
                    "channel_seed": channel_seed,
                    "snr_db": float(audit["snr_db"]),
                    "anchor_correct": parse_bool(audit["anchor_correct"]),
                    "raw_correct": parse_bool(audit["raw_correct"]),
                    "posterior_correct": parse_bool(audit["posterior_correct"]),
                    "final_correct": final_correct,
                }
            )
        decisions.append(
            {
                "channel_seed": channel_seed,
                "snr_db": float(audit["snr_db"]),
                "sample_id": audit["sample_id"],
                "risk_score": float(scores[index]),
                "threshold": float(threshold),
                "rejected": rejected,
                "final_source": "anchor" if rejected else "posterior",
                "clean_correct": parse_bool(audit["clean_correct"]),
                "anchor_correct": parse_bool(audit["anchor_correct"]),
                "raw_correct": parse_bool(audit["raw_correct"]),
                "posterior_correct": parse_bool(audit["posterior_correct"]),
                "final_correct": final_correct,
                "raw_psnr": float(audit["raw_psnr"]),
                "final_psnr": final_psnr,
                "raw_lpips": float(audit["raw_lpips"]),
                "final_lpips": final_lpips,
            }
        )

    def counts(subset: list[dict[str, Any]]) -> dict[str, int]:
        eligible = [row for row in subset if bool(row["anchor_correct"])]
        return {
            "rows": len(subset),
            "raw_failure": sum(not bool(row["raw_correct"]) for row in subset),
            "posterior_failure": sum(not bool(row["posterior_correct"]) for row in subset),
            "final_failure": sum(not bool(row["final_correct"]) for row in subset),
            "raw_new": sum(not bool(row["raw_correct"]) for row in eligible),
            "posterior_new": sum(not bool(row["posterior_correct"]) for row in eligible),
            "final_new": sum(not bool(row["final_correct"]) for row in eligible),
        }

    primary_counts = counts(primary_records)
    per_snr = {
        str(int(snr)): counts([row for row in primary_records if row["snr_db"] == snr])
        for snr in sorted(primary_snrs)
    }
    seeds = sorted({int(row["channel_seed"]) for row in primary_records})
    per_seed = {
        str(seed): counts(
            [row for row in primary_records if int(row["channel_seed"]) == seed]
        )
        for seed in seeds
    }
    eligible_image_ids = {
        str(row["sample_id"]) for row in primary_records if bool(row["anchor_correct"])
    }
    final_new_image_ids = {
        str(row["sample_id"])
        for row in primary_records
        if bool(row["anchor_correct"]) and not bool(row["final_correct"])
    }
    cluster_upper = clopper_pearson_upper_95(
        len(final_new_image_ids), len(eligible_image_ids)
    )
    mean_psnr = float(np.mean(psnr_delta))
    mean_lpips = float(np.mean(lpips_delta))
    psnr_retained = mean_psnr / float(posterior_quality["mean_psnr_gain"])
    quality_seed_summary = {
        str(seed): {
            "mean_final_minus_raw_psnr": float(np.mean(values["psnr"])),
            "mean_final_minus_raw_lpips": float(np.mean(values["lpips"])),
        }
        for seed, values in quality_by_seed.items()
    }
    gates = {
        "primary_new_total_not_above_raw": primary_counts["final_new"]
        <= primary_counts["raw_new"],
        "primary_new_each_snr_not_above_raw": all(
            item["final_new"] <= item["raw_new"] for item in per_snr.values()
        ),
        "primary_new_each_seed_not_above_raw": all(
            item["final_new"] <= item["raw_new"] for item in per_seed.values()
        ),
        "primary_failure_total_not_above_raw": primary_counts["final_failure"]
        <= primary_counts["raw_failure"],
        "primary_failure_each_seed_not_above_raw": all(
            item["final_failure"] <= item["raw_failure"] for item in per_seed.values()
        ),
        "new_error_cluster_upper_within_limit": cluster_upper
        <= float(gates_config["max_primary_final_new_image_cluster_upper"]),
        "mean_psnr_gain_positive": mean_psnr > 0,
        "mean_lpips_gain_nonpositive": mean_lpips <= 0,
        "quality_each_seed": all(
            item["mean_final_minus_raw_psnr"] > 0
            and item["mean_final_minus_raw_lpips"] <= 0
            for item in quality_seed_summary.values()
        ),
        "minimum_posterior_psnr_gain_retained": psnr_retained
        >= float(gates_config["minimum_posterior_psnr_gain_retained_fraction"]),
    }
    return {
        "threshold": float(threshold),
        "reference_reject_rate": float(reject[reference_mask].mean()),
        "all_row_reject_rate": float(reject.mean()),
        "primary": primary_counts,
        "per_snr": per_snr,
        "per_seed": per_seed,
        "primary_eligible_image_clusters": len(eligible_image_ids),
        "primary_final_new_image_clusters": len(final_new_image_ids),
        "primary_final_new_image_cluster_upper95": cluster_upper,
        "mean_final_minus_raw_psnr": mean_psnr,
        "mean_final_minus_raw_lpips": mean_lpips,
        "posterior_psnr_gain_retained_fraction": psnr_retained,
        "quality_by_seed": quality_seed_summary,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "decisions": decisions,
    }


def select_first_passing(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    for candidate in candidates:
        if bool(candidate["all_gates_pass"]):
            return candidate
    raise RuntimeError("no preregistered risk threshold candidate passed all development gates")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    source = config["feature_source"]
    risk_path = resolve(source["risk_features_csv"])
    audit_path = resolve(source["audit_csv"])
    schema_path = resolve(source["schema_json"])
    verify_hash(risk_path, str(source["risk_features_sha256"]))
    verify_hash(audit_path, str(source["audit_sha256"]))
    verify_hash(schema_path, str(source["schema_sha256"]))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if bool(schema.get("official_val_accessed")):
        raise RuntimeError("risk feature schema records official validation access")
    components = [dict(item) for item in config["score"]["components"]]
    input_features = [str(item["feature"]) for item in components]
    if len(input_features) != len(set(input_features)):
        raise ValueError("risk score components must be unique")
    if any(name not in RECEIVER_RISK_FEATURE_COLUMNS for name in input_features):
        raise ValueError("risk score contains a non-receiver input")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "analysis_id": config["analysis_id"],
                    "score_name": config["score"]["name"],
                    "receiver_input_features": input_features,
                    "threshold_reject_rate_grid": config["selection"][
                        "reference_reject_rate_grid"
                    ],
                    "teacher_fields_in_controller_inputs": False,
                    "official_val_accessed": False,
                },
                indent=2,
            )
        )
        return

    output = resolve(config["output_dir"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    risk_rows = read_csv(risk_path)
    audit_rows = read_csv(audit_path)
    expected_rows = int(source["expected_rows"])
    if len(risk_rows) != expected_rows or len(audit_rows) != expected_rows:
        raise RuntimeError("feature/audit row count differs from frozen expectation")
    risk_keys = [row_key(row) for row in risk_rows]
    audit_by_key = {row_key(row): row for row in audit_rows}
    if len(set(risk_keys)) != expected_rows or set(risk_keys) != set(audit_by_key):
        raise RuntimeError("feature/audit keys are not unique and exactly aligned")
    reference_mask = np.asarray(
        [
            parse_bool(row["teacher_clean_correct"])
            and parse_bool(row["teacher_anchor_correct"])
            for row in risk_rows
        ],
        dtype=bool,
    )
    scores, references = build_score(risk_rows, reference_mask, components)
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
    candidate_results = []
    for target_rate in map(float, config["selection"]["reference_reject_rate_grid"]):
        threshold = float(
            np.quantile(scores[reference_mask], 1.0 - target_rate, method="higher")
        )
        result = simulate(
            risk_rows,
            audit_by_key,
            scores,
            threshold,
            reference_mask,
            {float(item) for item in config["selection"]["primary_snrs"]},
            posterior_quality,
            dict(config["selection"]["gates"]),
        )
        result["target_reference_reject_rate"] = target_rate
        candidate_results.append(result)
    selected = select_first_passing(candidate_results)
    write_csv(output / "selected_decisions.csv", selected.pop("decisions"))
    for candidate in candidate_results:
        candidate.pop("decisions", None)
    cdf_path = output / "empirical_cdfs.npz"
    np.savez_compressed(cdf_path, **references)
    controller = {
        "analysis_id": config["analysis_id"],
        "feature_version": schema["feature_version"],
        "score_name": config["score"]["name"],
        "aggregation": "arithmetic_mean_empirical_percentiles",
        "empirical_cdf_side": "right",
        "components": components,
        "receiver_input_features": input_features,
        "teacher_fields_in_controller_inputs": False,
        "source_or_ground_truth_in_controller_inputs": False,
        "threshold": selected["threshold"],
        "target_reference_reject_rate": selected["target_reference_reject_rate"],
        "development_reference_rows": int(reference_mask.sum()),
        "cdf_artifact": str(cdf_path),
        "cdf_artifact_sha256": sha256_file(cdf_path),
        "feature_source_sha256": str(source["risk_features_sha256"]),
        "feature_schema_sha256": str(source["schema_sha256"]),
        "official_val_accessed": False,
    }
    controller_path = output / "controller.json"
    controller_path.write_text(json.dumps(controller, indent=2), encoding="utf-8")
    payload = {
        "config": config,
        "posterior_quality": posterior_quality,
        "candidate_results": candidate_results,
        "selected": selected,
        "controller": controller,
        "controller_sha256": sha256_file(controller_path),
        "official_val_accessed": False,
        "verdict": "DEVELOPMENT_PASS",
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected, "controller": controller}, indent=2))


if __name__ == "__main__":
    main()
