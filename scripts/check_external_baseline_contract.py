#!/usr/bin/env python3
"""Fail-closed validation and no-download dry-run for external JSCC baselines."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SNRS = [1, 4, 7, 13, 19]
EXPECTED_CHANNEL = "AWGN"
EXPECTED_CBR = 1.0 / 6.0
EXPECTED_SYMBOLS = 65536
REQUIRED_METRIC_GROUPS = {
    "image_quality",
    "perceptual_quality",
    "semantic_reliability",
    "systems",
}
REQUIRED_METHOD_ORDER = [
    "SGD_JSCC_AUTHOR",
    "SING_ZERO_STYLE",
    "DIFFJSCC_AUTHOR",
    "DIT_JSCC_AUTHOR",
]


class ContractError(RuntimeError):
    """Raised when a comparison contract would permit an unfair claim."""


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_contract(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError("contract root must be a mapping")
    return payload


def git_head(path: Path) -> str | None:
    if not (path / ".git").is_dir():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_contract(payload: dict[str, Any], *, check_paths: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if payload.get("status") != "protocol_only_no_outcomes":
        errors.append("status must remain protocol_only_no_outcomes before experiments")
    if payload.get("official_val_accessed") is not False:
        errors.append("official_val_accessed must be false")
    if payload.get("outcome_claims_allowed") is not False:
        errors.append("outcome_claims_allowed must be false in the protocol-only contract")

    boundary = payload.get("research_boundary", {})
    for field in (
        "project_mainline_unchanged",
        "awgn_minimal_closure_preserved",
        "semantic_drift_is_primary_failure_mode",
        "no_new_large_model_training",
    ):
        if boundary.get(field) is not True:
            errors.append(f"research_boundary.{field} must be true")

    tracks = payload.get("comparison_tracks", {})
    if tracks.get("author_native", {}).get("direct_ranking_against_ours_allowed") is not False:
        errors.append("author_native track must not directly rank against ours")
    common_track = tracks.get("common_contract", {})
    if common_track.get("require_total_rate_accounting") is not True:
        errors.append("common track must require total rate accounting")

    common = payload.get("common_contract", {})
    if common.get("channel") != EXPECTED_CHANNEL:
        errors.append(f"common channel must be {EXPECTED_CHANNEL}")
    if common.get("snrs_db") != EXPECTED_SNRS:
        errors.append(f"common SNR grid must be {EXPECTED_SNRS}")
    if abs(float(common.get("target_cbr", -1.0)) - EXPECTED_CBR) > 1e-12:
        errors.append("common target CBR must equal 1/6")
    if common.get("target_total_real_symbols") != EXPECTED_SYMBOLS:
        errors.append(f"common symbol budget must equal {EXPECTED_SYMBOLS}")
    if common.get("real_dimensions_per_complex_channel_use") != 2:
        errors.append("common CBR must pair two real coordinates per complex channel use")
    expected_from_symbols = EXPECTED_SYMBOLS / 2 / (3 * 256 * 256)
    if abs(expected_from_symbols - EXPECTED_CBR) > 1e-12:
        errors.append("common real-symbol budget is inconsistent with complex-use CBR")
    if common.get("official_imagenette_validation_forbidden") is not True:
        errors.append("official Imagenette validation must remain forbidden")
    if common.get("final_claim_requires_fresh_frozen_population") is not True:
        errors.append("a final claim must require a fresh frozen population")

    metrics = payload.get("metrics", {})
    missing_metric_groups = sorted(REQUIRED_METRIC_GROUPS - set(metrics))
    if missing_metric_groups:
        errors.append(f"missing metric groups: {missing_metric_groups}")
    semantic_metrics = set(metrics.get("semantic_reliability", []))
    for metric in ("T_cls_clean_correct_final_failure", "T_cls_new_error"):
        if metric not in semantic_metrics:
            errors.append(f"missing semantic metric: {metric}")

    rate = payload.get("rate_accounting", {})
    for field in (
        "count_main_latent",
        "count_edge_or_structure_branch",
        "count_text_or_semantic_payload",
        "count_pilots_and_csi_payload",
        "free_oracle_side_information_forbidden_in_common_contract",
        "unknown_rate_blocks_direct_ranking",
    ):
        if rate.get(field) is not True:
            errors.append(f"rate_accounting.{field} must be true")

    methods = payload.get("methods", [])
    method_ids = [method.get("id") for method in methods]
    if method_ids != REQUIRED_METHOD_ORDER:
        errors.append(f"method order must be {REQUIRED_METHOD_ORDER}")
    method_map = {method.get("id"): method for method in methods}

    sgd = method_map.get("SGD_JSCC_AUTHOR", {})
    if sgd.get("source_read_only") is not True:
        errors.append("SGD-JSCC source must be read-only")
    if sgd.get("direct_ranking_status") != (
        "author_native_blocked_common_adapter_rate_gate_passed_stage_metrics_pending"
    ):
        errors.append("SGD-JSCC author-native and common-adapter ranking states must stay separate")
    inventory = sgd.get("checkpoint_inventory", {})
    if inventory.get("download_authorized_by_user") is not True:
        errors.append("completed SGD-JSCC downloads require recorded user authorization")
    if inventory.get("download_completed") is not True:
        errors.append("SGD-JSCC asset state must record the completed download")
    text_status = (
        sgd.get("semantic_side_information", {})
        .get("text_caption", {})
        .get("common_contract_status")
    )
    if text_status != "explicit_fixed_utf8_crc16_bpsk_r21_smoke_passed":
        errors.append("SGD-JSCC common adapter must retain its explicit text transport")
    common_adapter = sgd.get("common_adapter", {})
    if common_adapter.get("total_real_symbols") != EXPECTED_SYMBOLS:
        errors.append("SGD-JSCC common adapter must close the 65,536-real budget")
    if common_adapter.get("total_complex_channel_uses") != EXPECTED_SYMBOLS // 2:
        errors.append("SGD-JSCC common adapter complex-use count is inconsistent")
    if common_adapter.get("rate_gate_status") != "passed_one_image_runtime_validation":
        errors.append("SGD-JSCC common adapter rate gate must reflect the completed smoke")

    sing = method_map.get("SING_ZERO_STYLE", {})
    if sing.get("reproduction_label") != "mechanism_level_not_exact_paper_reproduction":
        errors.append("SING fallback must be labeled mechanism-level, not exact reproduction")

    dit = method_map.get("DIT_JSCC_AUTHOR", {})
    if dit.get("current_status") != "watch_only_not_runnable":
        errors.append("DiT-JSCC must remain watch-only while its repository is not runnable")

    source_checks: dict[str, Any] = {}
    if check_paths:
        for field in ("source_manifest", "source_dir"):
            path = resolve(common.get(field, ""))
            exists = path.exists()
            source_checks[field] = {"path": str(path), "exists": exists}
            if not exists:
                errors.append(f"common_contract.{field} does not exist: {path}")

        source_path = resolve(sgd.get("source_path", ""))
        observed_head = git_head(source_path)
        expected_head = sgd.get("source_commit")
        source_checks["sgd_jscc"] = {
            "path": str(source_path),
            "exists": source_path.exists(),
            "expected_commit": expected_head,
            "observed_commit": observed_head,
        }
        if observed_head != expected_head:
            errors.append(
                f"SGD-JSCC commit mismatch: expected {expected_head}, observed {observed_head}"
            )
        for required_file in ("README.md", "inference_one.py", "configs/inference.yaml"):
            if not (source_path / required_file).is_file():
                errors.append(f"missing SGD-JSCC source file: {required_file}")

        checkpoint_dir = source_path / "checkpoint"
        downloaded = []
        if checkpoint_dir.is_dir():
            downloaded = sorted(
                str(path.relative_to(source_path))
                for path in checkpoint_dir.rglob("*")
                if path.is_file()
            )
        source_checks["sgd_jscc"]["downloaded_checkpoint_files"] = downloaded
        if downloaded and inventory.get("download_completed") is not True:
            warnings.append(
                "checkpoint files exist without a completed-download record"
            )

    schedule = payload.get("schedule", [])
    if not schedule or schedule[0].get("milestone") != "EXT0_source_and_contract_audit":
        errors.append("schedule must begin with EXT0 source/contract audit")
    if not any(item.get("milestone") == "EXT3_SING_zero_style_common_contract" for item in schedule):
        errors.append("schedule must include the SING-Zero-style common-contract baseline")

    output = payload.get("outputs", {})
    if output.get("overwrite_forbidden") is not True:
        errors.append("external baseline outputs must be non-overwriting")

    if errors:
        raise ContractError("\n".join(errors))

    blockers = {
        "SGD_JSCC_AUTHOR": [
            "author-native track remains non-comparable because text is perfect/free",
            "common adapter still needs 8-image and 64-image semantic metric stages",
            "single smoke shows a qualitative patch-boundary/hallucination risk that needs counting",
        ],
        "SING_ZERO_STYLE": [
            "author implementation not located; must keep mechanism-level label",
        ],
        "DIFFJSCC_AUTHOR": [
            "deferred until SGD-JSCC and SING-style tracks close",
        ],
        "DIT_JSCC_AUTHOR": [
            "official repository is README-only in the 2026-07-14 audit",
        ],
    }
    return {
        "status": "PASS",
        "mode": "no_download_dry_run",
        "analysis_id": payload.get("analysis_id"),
        "official_val_accessed": False,
        "outcome_claims_allowed": False,
        "method_order": method_ids,
        "common_contract": {
            "channel": EXPECTED_CHANNEL,
            "snrs_db": EXPECTED_SNRS,
            "target_cbr": EXPECTED_CBR,
            "target_total_real_symbols": EXPECTED_SYMBOLS,
        },
        "source_checks": source_checks,
        "warnings": warnings,
        "blockers": blockers,
        "next_milestone": "EXT3_SING_zero_style_common_contract",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/external_baseline_comparison_contract.yaml",
    )
    parser.add_argument(
        "--skip-path-checks",
        action="store_true",
        help="Validate only the declarative fairness contract.",
    )
    args = parser.parse_args()
    config_path = resolve(args.config)
    result = validate_contract(
        load_contract(config_path), check_paths=not args.skip_path_checks
    )
    result["config"] = str(config_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
