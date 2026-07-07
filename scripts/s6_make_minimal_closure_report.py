from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a minimal paper-closure report from existing outputs.")
    parser.add_argument("--config", default="configs/s6_minimal_closure_report.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def serialize_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_value(row.get(key, "")) for key in fieldnames})


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def to_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def to_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(float(value))


def mean(values: list[float]) -> float:
    clean = [float(value) for value in values if value is not None]
    return float(sum(clean) / len(clean)) if clean else 0.0


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "N/A"


def git_dirty_state() -> str:
    try:
        output = subprocess.check_output(["git", "status", "--short"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"
    return "dirty" if output else "clean"


def proxy_environment_present() -> list[str]:
    keys = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "NO_PROXY", "no_proxy"]
    return [key for key in keys if os.environ.get(key)]


def validate_inputs(config: dict[str, Any]) -> dict[str, str]:
    paths = {key: resolve_project_path(value) for key, value in config["inputs"].items()}
    missing = [f"{key}: {path}" for key, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))
    return {key: project_relative(path) for key, path in paths.items()}


def snr_key(row: dict[str, Any]) -> float:
    return float(row["snr_db"])


def build_formal_m0_rows(m0_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in m0_metrics["results"]:
        rows.append(
            {
                "split": "formal_coco512",
                "method": "M0-DeepJSCC-HR",
                "snr_db": float(item["snr_db"]),
                "num_images": int(item["num_images"]),
                "psnr_db": float(item["psnr_db"]),
                "ssim": float(item["ssim"]),
                "ms_ssim": float(item["ms_ssim"]),
                "inference_time_ms_per_image": float(item["inference_time_ms_per_image"]),
            }
        )
    return rows


def build_m1_rows(m1_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in m1_metrics["results"]:
        m0 = item["m0_reconstruction_vs_original"]
        m1 = item["m1_refined_vs_original"]
        rows.append(
            {
                "split": "exp_s2_002_16img_per_snr",
                "method": "M1-BlindDiffusion-SDImg2Img",
                "snr_db": float(item["snr_db"]),
                "num_images": int(item["num_images"]),
                "m0_psnr_db": float(m0["psnr_db"]),
                "m1_psnr_db": float(m1["psnr_db"]),
                "delta_psnr_vs_m0_db": float(m1["psnr_db"]) - float(m0["psnr_db"]),
                "m0_lpips": float(m0["lpips"]),
                "m1_lpips": float(m1["lpips"]),
                "delta_lpips_vs_m0": float(m1["lpips"]) - float(m0["lpips"]),
                "m0_ms_ssim": float(m0["ms_ssim"]),
                "m1_ms_ssim": float(m1["ms_ssim"]),
                "diffusion_time_ms_per_image": float(item["diffusion_time_ms_per_image"]),
            }
        )
    return rows


def build_residual_rows(summary_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for item in sorted(summary_rows, key=snr_key):
        snr = float(item["snr_db"])
        num_images = to_int(item["num_images"])
        rows.extend(
            [
                {
                    "split": "exp_s4_006_eval",
                    "method": "M0-DeepJSCC-HR",
                    "snr_db": snr,
                    "num_images": num_images,
                    "psnr_db": to_float(item["m0_psnr_db"]),
                    "delta_psnr_vs_m0_db": 0.0,
                    "lpips": to_float(item["m0_lpips"]),
                    "delta_lpips_vs_m0": 0.0,
                    "semantic_failure": to_float(item["m0_final_failure"]),
                    "accept_rate": "",
                    "time_ms_per_image": "",
                },
                {
                    "split": "exp_s4_006_eval",
                    "method": "M2-SNRConditionedPixelResidualRestoration",
                    "snr_db": snr,
                    "num_images": num_images,
                    "psnr_db": to_float(item["refined_psnr_db"]),
                    "delta_psnr_vs_m0_db": to_float(item["refined_delta_psnr_vs_m0_db"]),
                    "lpips": to_float(item["refined_lpips"]),
                    "delta_lpips_vs_m0": to_float(item["refined_delta_lpips_vs_m0"]),
                    "semantic_failure": to_float(item["refined_failure"]),
                    "refinement_drift": to_float(item["refined_refinement_drift"]),
                    "accept_rate": "",
                    "time_ms_per_image": to_float(item["refiner_time_ms_per_image"]),
                },
                {
                    "split": "exp_s4_006_eval",
                    "method": "M3-ResidualRestorationTop1Fallback",
                    "snr_db": snr,
                    "num_images": num_images,
                    "psnr_db": to_float(item["m3_psnr_db"]),
                    "delta_psnr_vs_m0_db": to_float(item["m3_delta_psnr_vs_m0_db"]),
                    "lpips": to_float(item["m3_lpips"]),
                    "delta_lpips_vs_m0": to_float(item["m3_delta_lpips_vs_m0"]),
                    "semantic_failure": to_float(item["m3_final_failure"]),
                    "accept_rate": to_float(item["accept_rate"]),
                    "false_accept_rate": to_float(item["false_accept_rate"]),
                    "false_reject_rate": to_float(item["false_reject_rate"]),
                    "time_ms_per_image": to_float(item["refiner_time_ms_per_image"]),
                },
            ]
        )
    return rows


def row_for_policy(rows: list[dict[str, str]], policy: str, subset: str | None = None) -> dict[str, str]:
    for row in rows:
        if row.get("policy") != policy:
            continue
        if subset is not None and row.get("subset") != subset:
            continue
        if str(row.get("snr_db")) == "all":
            return row
    raise KeyError(f"Policy not found: {policy}, subset={subset}")


def build_method_summary(
    formal_m0: list[dict[str, Any]],
    m1_rows: list[dict[str, Any]],
    residual_rows: list[dict[str, Any]],
    shrink_validation_rows: list[dict[str, str]],
    heldout_shrink_rows: list[dict[str, str]],
    testlike_shrink_rows: list[dict[str, str]],
    adaptive_rows: list[dict[str, str]],
    testlike_rows: list[dict[str, str]],
    clean_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    residual_by_method: dict[str, list[dict[str, Any]]] = {}
    for row in residual_rows:
        residual_by_method.setdefault(row["method"], []).append(row)

    top1 = row_for_policy(testlike_rows, "top1_equal")
    selected = row_for_policy(testlike_rows, "selected_risk_rule")
    selected_plus = row_for_policy(testlike_rows, "selected_risk_rule_plus_ensemble_veto")
    validation_shrink = row_for_policy(shrink_validation_rows, "selected_top1_fallback_shrink_schedule")
    heldout_top1_full = row_for_policy(heldout_shrink_rows, "top1_full_strength")
    heldout_shrink = row_for_policy(heldout_shrink_rows, "validation_top1_shrink_schedule")
    testlike_top1_full = row_for_policy(testlike_shrink_rows, "top1_full_strength")
    testlike_shrink = row_for_policy(testlike_shrink_rows, "validation_top1_shrink_schedule")
    adaptive_validation = next(
        row
        for row in adaptive_rows
        if row.get("split") == "validation"
        and row.get("policy") == "adaptive_max_top1_consistent_alpha"
        and row.get("snr_db") == "all"
    )
    adaptive_heldout = next(
        row
        for row in adaptive_rows
        if row.get("split") == "held-out"
        and row.get("policy") == "adaptive_max_top1_consistent_alpha"
        and row.get("snr_db") == "all"
    )
    adaptive_testlike = next(
        row
        for row in adaptive_rows
        if row.get("split") == "test-like"
        and row.get("policy") == "adaptive_max_top1_consistent_alpha"
        and row.get("snr_db") == "all"
    )
    clean_selected = row_for_policy(clean_rows, "selected_risk_rule", subset="clean_correct")
    clean_selected_plus = row_for_policy(clean_rows, "selected_risk_rule_plus_ensemble_veto", subset="clean_correct")

    m2 = residual_by_method["M2-SNRConditionedPixelResidualRestoration"]
    m3 = residual_by_method["M3-ResidualRestorationTop1Fallback"]
    m0_eval = residual_by_method["M0-DeepJSCC-HR"]

    return [
        {
            "method": "M0-DeepJSCC-HR",
            "role": "baseline",
            "split": "formal_coco512",
            "snrs": "[1,4,7,13,19]",
            "mean_psnr_db": mean([row["psnr_db"] for row in formal_m0]),
            "mean_ms_ssim": mean([row["ms_ssim"] for row in formal_m0]),
            "mean_semantic_failure": "",
            "mean_delta_psnr_vs_m0_db": 0.0,
            "mean_delta_lpips_vs_m0": "",
            "status": "usable M0 baseline",
        },
        {
            "method": "M1-BlindDiffusion-SDImg2Img",
            "role": "negative reference",
            "split": "exp_s2_002_16img_per_snr",
            "snrs": "[1,7,19]",
            "mean_psnr_db": mean([row["m1_psnr_db"] for row in m1_rows]),
            "mean_ms_ssim": mean([row["m1_ms_ssim"] for row in m1_rows]),
            "mean_semantic_failure": "",
            "mean_delta_psnr_vs_m0_db": mean([row["delta_psnr_vs_m0_db"] for row in m1_rows]),
            "mean_delta_lpips_vs_m0": mean([row["delta_lpips_vs_m0"] for row in m1_rows]),
            "status": "failed due quality and semantic drift",
        },
        {
            "method": "M2-SNRConditionedPixelResidualRestoration",
            "role": "positive restoration anchor",
            "split": "exp_s4_006_eval",
            "snrs": "[1,4,7,13,19]",
            "mean_psnr_db": mean([row["psnr_db"] for row in m2]),
            "mean_ms_ssim": "",
            "mean_semantic_failure": mean([row["semantic_failure"] for row in m2]),
            "mean_delta_psnr_vs_m0_db": mean([row["delta_psnr_vs_m0_db"] for row in m2]),
            "mean_delta_lpips_vs_m0": mean([row["delta_lpips_vs_m0"] for row in m2]),
            "status": "positive quality, needs semantic handling",
        },
        {
            "method": "M3-ResidualRestorationTop1Fallback",
            "role": "conservative first M3",
            "split": "exp_s4_006_eval",
            "snrs": "[1,4,7,13,19]",
            "mean_psnr_db": mean([row["psnr_db"] for row in m3]),
            "mean_ms_ssim": "",
            "mean_semantic_failure": mean([row["semantic_failure"] for row in m3]),
            "mean_delta_failure_vs_m0": mean([row["semantic_failure"] for row in m3])
            - mean([row["semantic_failure"] for row in m0_eval]),
            "mean_delta_psnr_vs_m0_db": mean([row["delta_psnr_vs_m0_db"] for row in m3]),
            "mean_delta_lpips_vs_m0": mean([row["delta_lpips_vs_m0"] for row in m3]),
            "status": "safe conservative closure on pseudo-label metric",
        },
        {
            "method": "M3-ResidualRestorationTop1ShrinkFallback",
            "role": "fixed-schedule conservative candidate",
            "split": "validation_selected_frozen_testlike",
            "snrs": "[1,4,7,13,19]",
            "mean_psnr_db": to_float(validation_shrink["final_psnr_db"]),
            "mean_ms_ssim": to_float(validation_shrink["final_ms_ssim"]),
            "mean_semantic_failure": to_float(validation_shrink["final_failure_rate"]),
            "mean_delta_failure_vs_m0": to_float(validation_shrink["delta_final_failure_vs_m0"]),
            "mean_delta_psnr_vs_m0_db": to_float(validation_shrink["delta_psnr_vs_m0_db"]),
            "mean_delta_lpips_vs_m0": to_float(validation_shrink["delta_lpips_vs_m0"]),
            "testlike_psnr_db": to_float(testlike_shrink["final_psnr_db"]),
            "heldout_delta_psnr_vs_m0_db": to_float(heldout_shrink["delta_psnr_vs_m0_db"]),
            "heldout_delta_psnr_vs_full_top1_db": to_float(heldout_shrink["delta_psnr_vs_m0_db"])
            - to_float(heldout_top1_full["delta_psnr_vs_m0_db"]),
            "heldout_delta_lpips_vs_m0": to_float(heldout_shrink["delta_lpips_vs_m0"]),
            "heldout_final_failure_rate": to_float(heldout_shrink["final_failure_rate"]),
            "heldout_accepted_new_error_count": to_int(heldout_shrink["accepted_new_error_count"]),
            "testlike_delta_psnr_vs_m0_db": to_float(testlike_shrink["delta_psnr_vs_m0_db"]),
            "testlike_delta_psnr_vs_full_top1_db": to_float(testlike_shrink["delta_psnr_vs_m0_db"])
            - to_float(testlike_top1_full["delta_psnr_vs_m0_db"]),
            "testlike_delta_lpips_vs_m0": to_float(testlike_shrink["delta_lpips_vs_m0"]),
            "testlike_final_failure_rate": to_float(testlike_shrink["final_failure_rate"]),
            "testlike_accepted_new_error_count": to_int(testlike_shrink["accepted_new_error_count"]),
            "status": "conservative fixed-schedule candidate; superseded by adaptive alpha on PSNR",
        },
        {
            "method": "M3-AdaptiveResidualAlphaTop1Fallback",
            "role": "strongest conservative M3 candidate",
            "split": "validation_heldout_testlike_adaptive_policy",
            "snrs": "[1,4,7,13,19]",
            "mean_psnr_db": to_float(adaptive_validation["final_psnr_db"]),
            "mean_ms_ssim": to_float(adaptive_validation["final_ms_ssim"]),
            "mean_semantic_failure": to_float(adaptive_validation["final_failure_rate"]),
            "mean_delta_failure_vs_m0": to_float(adaptive_validation["delta_final_failure_vs_m0"]),
            "mean_delta_psnr_vs_m0_db": to_float(adaptive_validation["delta_psnr_vs_m0_db"]),
            "mean_delta_lpips_vs_m0": to_float(adaptive_validation["delta_lpips_vs_m0"]),
            "mean_selected_alpha_accepted": to_float(adaptive_validation["mean_selected_alpha_accepted"]),
            "accepted_new_error_count": to_int(adaptive_validation["accepted_new_error_count"]),
            "missed_repair_count": to_int(adaptive_validation["missed_repair_count"]),
            "heldout_delta_psnr_vs_m0_db": to_float(adaptive_heldout["delta_psnr_vs_m0_db"]),
            "heldout_delta_lpips_vs_m0": to_float(adaptive_heldout["delta_lpips_vs_m0"]),
            "heldout_final_failure_rate": to_float(adaptive_heldout["final_failure_rate"]),
            "heldout_accepted_new_error_count": to_int(adaptive_heldout["accepted_new_error_count"]),
            "heldout_missed_repair_count": to_int(adaptive_heldout["missed_repair_count"]),
            "testlike_delta_psnr_vs_m0_db": to_float(adaptive_testlike["delta_psnr_vs_m0_db"]),
            "testlike_delta_lpips_vs_m0": to_float(adaptive_testlike["delta_lpips_vs_m0"]),
            "testlike_final_failure_rate": to_float(adaptive_testlike["final_failure_rate"]),
            "testlike_accepted_new_error_count": to_int(adaptive_testlike["accepted_new_error_count"]),
            "testlike_missed_repair_count": to_int(adaptive_testlike["missed_repair_count"]),
            "status": "strongest conservative quality gain; still no semantic repair",
        },
        {
            "method": "M3-SelectedRiskRuleCandidate",
            "role": "test-like candidate gate",
            "split": "testlike_policy",
            "snrs": "[1,4,7,13,19]",
            "mean_psnr_db": to_float(selected["final_psnr_db"]),
            "mean_semantic_failure": to_float(selected["final_failure_rate"]),
            "delta_psnr_vs_top1_db": to_float(selected["delta_final_psnr_vs_top1_equal_db"]),
            "delta_failure_vs_top1": to_float(selected["delta_final_failure_vs_top1_equal"]),
            "accepted_repair_count": to_int(selected["accepted_repair_count"]),
            "accepted_new_error_count": to_int(selected["accepted_new_error_count"]),
            "clean_correct_new_error_count": to_int(clean_selected["accepted_new_error_gt_count"]),
            "status": "not final; leaves AlexNet/GT-like risk",
        },
        {
            "method": "M3-EnsembleVetoSafetyBound",
            "role": "safety upper-bound",
            "split": "testlike_policy_and_clean_correct",
            "snrs": "[1,4,7,13,19]",
            "mean_psnr_db": to_float(selected_plus["final_psnr_db"]),
            "mean_semantic_failure": to_float(selected_plus["final_failure_rate"]),
            "delta_psnr_vs_selected_db": to_float(selected_plus["delta_final_psnr_vs_selected_risk_rule_db"]),
            "accepted_repair_count": to_int(selected_plus["accepted_repair_count"]),
            "accepted_new_error_count": to_int(selected_plus["accepted_new_error_count"]),
            "clean_correct_new_error_count": to_int(clean_selected_plus["accepted_new_error_gt_count"]),
            "clean_correct_delta_psnr_vs_top1_db": to_float(clean_selected_plus["delta_final_psnr_vs_top1_equal_db"]),
            "status": "too conservative for first M3",
        },
    ]


def build_testlike_policy_export(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if row.get("snr_db") != "all":
            continue
        output.append(
            {
                "policy": row["policy"],
                "num_images": to_int(row["num_images"]),
                "accept_rate": to_float(row["accept_rate"]),
                "final_failure_rate": to_float(row["final_failure_rate"]),
                "final_psnr_db": to_float(row["final_psnr_db"]),
                "delta_final_failure_vs_top1": to_float(row["delta_final_failure_vs_top1_equal"]),
                "delta_final_psnr_vs_top1_db": to_float(row["delta_final_psnr_vs_top1_equal_db"]),
                "accepted_repair_count": to_int(row["accepted_repair_count"]),
                "accepted_new_error_count": to_int(row["accepted_new_error_count"]),
            }
        )
    return output


def build_clean_policy_export(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if row.get("subset") != "clean_correct" or row.get("snr_db") != "all":
            continue
        output.append(
            {
                "policy": row["policy"],
                "num_rows": to_int(row["num_rows"]),
                "accept_rate": to_float(row["accept_rate"]),
                "final_failure_gt": to_float(row["final_failure_gt"]),
                "final_psnr_db": to_float(row["final_psnr_db"]),
                "delta_final_failure_gt_vs_top1": to_float(row["delta_final_failure_gt_vs_top1_equal"]),
                "delta_final_psnr_vs_top1_db": to_float(row["delta_final_psnr_vs_top1_equal_db"]),
                "accepted_repair_gt_count": to_int(row["accepted_repair_gt_count"]),
                "accepted_new_error_gt_count": to_int(row["accepted_new_error_gt_count"]),
            }
        )
    return output


def build_shrink_policy_export(
    validation_rows: list[dict[str, str]],
    heldout_rows: list[dict[str, str]],
    testlike_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    output = []
    for split, rows in [
        ("validation_exp_s4_006", validation_rows),
        ("frozen_heldout", heldout_rows),
        ("frozen_testlike", testlike_rows),
    ]:
        for row in rows:
            if row.get("snr_db") != "all":
                continue
            alpha = row.get("alpha", "")
            policy_label = f"{row['policy']}@{alpha}" if alpha not in ("", None) else row["policy"]
            output.append(
                {
                    "split": split,
                    "policy": row["policy"],
                    "policy_label": policy_label,
                    "alpha": alpha,
                    "num_images": to_int(row["num_images"]),
                    "accept_rate": to_float(row["accept_rate"]),
                    "final_failure_rate": to_float(row["final_failure_rate"]),
                    "delta_failure_vs_m0": to_float(row["delta_final_failure_vs_m0"]),
                    "final_psnr_db": to_float(row["final_psnr_db"]),
                    "delta_psnr_vs_m0_db": to_float(row["delta_psnr_vs_m0_db"]),
                    "final_lpips": to_float(row["final_lpips"]),
                    "delta_lpips_vs_m0": to_float(row["delta_lpips_vs_m0"]),
                    "repair_count": to_int(row["repair_count"]),
                    "accepted_new_error_count": to_int(row["accepted_new_error_count"]),
                    "rejected_good_count": to_int(row["rejected_good_count"]),
                }
            )
    return output


def build_adaptive_policy_export(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if row.get("snr_db") != "all":
            continue
        output.append(
            {
                "split": row["split"],
                "policy": row["policy"],
                "num_images": to_int(row["num_images"]),
                "accept_rate": to_float(row["accept_rate"]),
                "fallback_rate": to_float(row["fallback_rate"]),
                "mean_selected_alpha_accepted": to_float(row["mean_selected_alpha_accepted"]),
                "final_failure_rate": to_float(row["final_failure_rate"]),
                "delta_failure_vs_m0": to_float(row["delta_final_failure_vs_m0"]),
                "final_psnr_db": to_float(row["final_psnr_db"]),
                "delta_psnr_vs_m0_db": to_float(row["delta_psnr_vs_m0_db"]),
                "final_lpips": to_float(row["final_lpips"]),
                "delta_lpips_vs_m0": to_float(row["delta_lpips_vs_m0"]),
                "repair_count": to_int(row["repair_count"]),
                "accepted_new_error_count": to_int(row["accepted_new_error_count"]),
                "missed_repair_count": to_int(row["missed_repair_count"]),
                "available_repair_count": to_int(row["available_repair_count"]),
            }
        )
    return output


def style_axes(ax, title: str, xlabel: str | None = None, ylabel: str | None = None) -> None:
    ax.set_title(title, fontsize=12, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_residual_quality(residual_rows: list[dict[str, Any]], path: Path) -> None:
    methods = [
        ("M0-DeepJSCC-HR", "M0", "#4a7c8c"),
        ("M2-SNRConditionedPixelResidualRestoration", "M2 refined", "#6b8f3a"),
        ("M3-ResidualRestorationTop1Fallback", "M3 final", "#b57b2a"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for method, label, color in methods:
        rows = sorted([row for row in residual_rows if row["method"] == method], key=snr_key)
        axes[0].plot([row["snr_db"] for row in rows], [row["psnr_db"] for row in rows], marker="o", label=label, color=color)
        axes[1].plot([row["snr_db"] for row in rows], [row["lpips"] for row in rows], marker="o", label=label, color=color)
    style_axes(axes[0], "EXP-S4-006 Quality: PSNR", "SNR (dB)", "PSNR (dB)")
    style_axes(axes[1], "EXP-S4-006 Quality: LPIPS", "SNR (dB)", "LPIPS")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_residual_semantics(residual_rows: list[dict[str, Any]], path: Path) -> None:
    methods = [
        ("M0-DeepJSCC-HR", "M0", "#4a7c8c"),
        ("M2-SNRConditionedPixelResidualRestoration", "M2 refined", "#6b8f3a"),
        ("M3-ResidualRestorationTop1Fallback", "M3 final", "#b57b2a"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for method, label, color in methods:
        rows = sorted([row for row in residual_rows if row["method"] == method], key=snr_key)
        axes[0].plot(
            [row["snr_db"] for row in rows],
            [row["semantic_failure"] for row in rows],
            marker="o",
            label=label,
            color=color,
        )
    m3_rows = sorted([row for row in residual_rows if row["method"] == "M3-ResidualRestorationTop1Fallback"], key=snr_key)
    axes[1].plot([row["snr_db"] for row in m3_rows], [row["accept_rate"] for row in m3_rows], marker="o", color="#7a5a9e")
    style_axes(axes[0], "EXP-S4-006 Pseudo Semantic Failure", "SNR (dB)", "failure rate")
    style_axes(axes[1], "M3 Top-1 Fallback Accept Rate", "SNR (dB)", "accept rate")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_testlike_tradeoff(policy_rows: list[dict[str, Any]], clean_rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for row in policy_rows:
        axes[0].scatter(row["delta_final_psnr_vs_top1_db"], row["accepted_new_error_count"], s=70)
        axes[0].annotate(row["policy"], (row["delta_final_psnr_vs_top1_db"], row["accepted_new_error_count"]), fontsize=8)
    for row in clean_rows:
        axes[1].scatter(row["delta_final_psnr_vs_top1_db"], row["accepted_new_error_gt_count"], s=70)
        axes[1].annotate(row["policy"], (row["delta_final_psnr_vs_top1_db"], row["accepted_new_error_gt_count"]), fontsize=8)
    style_axes(axes[0], "Test-Like AlexNet Policy Tradeoff", "PSNR delta vs top-1 (dB)", "accepted new errors")
    style_axes(axes[1], "COCO-Object Clean-Correct Tradeoff", "PSNR delta vs top-1 (dB)", "GT-like new errors")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_shrink_tradeoff(shrink_rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    splits = [
        ("validation_exp_s4_006", "Validation Shrink Tradeoff"),
        ("frozen_heldout", "Frozen Held-Out Shrink Tradeoff"),
        ("frozen_testlike", "Frozen Test-Like Shrink Tradeoff"),
    ]
    for ax, (split, title) in zip(axes, splits):
        rows = [row for row in shrink_rows if row["split"] == split and row["policy"] != "m0"]
        for row in rows:
            ax.scatter(row["delta_psnr_vs_m0_db"], row["accepted_new_error_count"], s=70)
            ax.annotate(row["policy_label"], (row["delta_psnr_vs_m0_db"], row["accepted_new_error_count"]), fontsize=8)
        style_axes(ax, title, "PSNR delta vs M0 (dB)", "accepted new errors")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_adaptive_tradeoff(adaptive_rows: list[dict[str, Any]], path: Path) -> None:
    label_map = {
        "top1_full_strength": "top1 full",
        "fixed_validation_top1_shrink_schedule": "fixed shrink",
        "adaptive_max_top1_consistent_alpha": "adaptive alpha",
        "always_full_strength": "always full",
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    splits = [
        ("validation", "Validation Adaptive Alpha"),
        ("held-out", "Held-Out Adaptive Alpha"),
        ("test-like", "Test-Like Adaptive Alpha"),
    ]
    for ax, (split, title) in zip(axes, splits):
        rows = [row for row in adaptive_rows if row["split"] == split and row["policy"] != "m0"]
        for row in rows:
            ax.scatter(row["delta_psnr_vs_m0_db"], row["accepted_new_error_count"], s=70)
            ax.annotate(label_map.get(row["policy"], row["policy"]), (row["delta_psnr_vs_m0_db"], row["accepted_new_error_count"]), fontsize=8)
        style_axes(ax, title, "PSNR delta vs M0 (dB)", "accepted new errors")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    lines = []
    headers = [label for _key, label in columns]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---" for _ in headers]) + "|")
    for row in rows:
        values = []
        for key, _label in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = fmt(value)
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def make_report(
    config: dict[str, Any],
    method_summary: list[dict[str, Any]],
    residual_rows: list[dict[str, Any]],
    shrink_rows: list[dict[str, Any]],
    adaptive_rows: list[dict[str, Any]],
    testlike_rows: list[dict[str, Any]],
    clean_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> str:
    m1 = next(row for row in method_summary if row["method"] == "M1-BlindDiffusion-SDImg2Img")
    m2 = next(row for row in method_summary if row["method"] == "M2-SNRConditionedPixelResidualRestoration")
    m3 = next(row for row in method_summary if row["method"] == "M3-ResidualRestorationTop1Fallback")
    m3_shrink = next(row for row in method_summary if row["method"] == "M3-ResidualRestorationTop1ShrinkFallback")
    m3_adaptive = next(row for row in method_summary if row["method"] == "M3-AdaptiveResidualAlphaTop1Fallback")
    selected = next(row for row in method_summary if row["method"] == "M3-SelectedRiskRuleCandidate")
    safety = next(row for row in method_summary if row["method"] == "M3-EnsembleVetoSafetyBound")

    lines = [
        "# Minimal Closure Report",
        "",
        "This derived report gathers the current COCO-256 AWGN evidence into one first-paper closure view.",
        "It does not run models or download data; it only reads existing metrics and analysis CSV files.",
        "",
        "## Bottom Line",
        "",
        f"- `M1-BlindDiffusion-SDImg2Img` remains a negative reference: mean PSNR delta vs its M0 input is `{signed(float(m1['mean_delta_psnr_vs_m0_db']))}` dB and mean LPIPS delta is `{signed(float(m1['mean_delta_lpips_vs_m0']))}`.",
        f"- `M2-SNRConditionedPixelResidualRestoration` is the positive restoration anchor: mean PSNR delta vs M0 is `{signed(float(m2['mean_delta_psnr_vs_m0_db']))}` dB and mean LPIPS delta is `{signed(float(m2['mean_delta_lpips_vs_m0']))}` on `EXP-S4-006`.",
        f"- `M3-ResidualRestorationTop1Fallback` is the conservative first closure: mean PSNR delta vs M0 is `{signed(float(m3['mean_delta_psnr_vs_m0_db']))}` dB, mean LPIPS delta is `{signed(float(m3['mean_delta_lpips_vs_m0']))}`, and pseudo semantic failure does not increase vs M0.",
        f"- `M3-ResidualRestorationTop1ShrinkFallback` is the fixed-schedule conservative candidate: validation PSNR delta is `{signed(float(m3_shrink['mean_delta_psnr_vs_m0_db']))}` dB, frozen held-out/test-like PSNR deltas are `{signed(float(m3_shrink['heldout_delta_psnr_vs_m0_db']))}`/`{signed(float(m3_shrink['testlike_delta_psnr_vs_m0_db']))}` dB, and held-out/test-like accepted new errors are `{m3_shrink['heldout_accepted_new_error_count']}`/`{m3_shrink['testlike_accepted_new_error_count']}`.",
        f"- `M3-AdaptiveResidualAlphaTop1Fallback` is the strongest conservative candidate so far: validation PSNR delta is `{signed(float(m3_adaptive['mean_delta_psnr_vs_m0_db']))}` dB, held-out/test-like PSNR deltas are `{signed(float(m3_adaptive['heldout_delta_psnr_vs_m0_db']))}`/`{signed(float(m3_adaptive['testlike_delta_psnr_vs_m0_db']))}` dB, accepted new errors are `{m3_adaptive['accepted_new_error_count']}`/`{m3_adaptive['heldout_accepted_new_error_count']}`/`{m3_adaptive['testlike_accepted_new_error_count']}`, but repair remains `0` and missed repair remains high.",
        f"- `selected_risk_rule` is still only a candidate: on test-like it gives `{signed(float(selected['delta_psnr_vs_top1_db']))}` dB vs top-1 and `accepted_new_error_count={selected['accepted_new_error_count']}`; on COCO-object clean-correct it still has `clean_correct_new_error_count={selected['clean_correct_new_error_count']}`.",
        f"- The ensemble-veto safety bound removes COCO-object clean-correct new errors, but its clean-correct PSNR delta vs top-1 is `{signed(float(safety['clean_correct_delta_psnr_vs_top1_db']))}` dB, so it is too conservative as the main M3.",
        "",
        "## Method Naming",
        "",
        "| Label | Current concrete implementation | Status |",
        "|---|---|---|",
        "| M0 | DeepJSCC-HR `best.pt` reconstruction | Baseline |",
        "| M1 | Stable Diffusion img2img blind refinement, empty prompt | Negative reference |",
        "| M2 | SNR-conditioned pixel residual CNN refinement | Positive restoration anchor |",
        "| M3 | M2 plus top-1 semantic fallback | Conservative first closure |",
        "| M3 fixed-schedule candidate | M2 with validation-selected residual shrink plus top-1 fallback | Conservative schedule ablation / backup |",
        "| M3 adaptive-alpha candidate | M2 with per-sample max top-1-consistent residual alpha | Strongest conservative pseudo-safe candidate |",
        "| M3 candidate | M2 plus selected risk rule | Higher PSNR candidate, not final-safe |",
        "",
        "## Closure Summary",
        "",
    ]
    lines += markdown_table(
        method_summary,
        [
            ("method", "Method"),
            ("role", "Role"),
            ("split", "Split"),
            ("mean_delta_psnr_vs_m0_db", "Mean Delta PSNR"),
            ("mean_delta_lpips_vs_m0", "Mean Delta LPIPS"),
            ("mean_semantic_failure", "Mean Failure"),
            ("status", "Status"),
        ],
    )
    lines.extend(["", "## EXP-S4-006 Residual Per-SNR", ""])
    residual_export = [
        row
        for row in residual_rows
        if row["method"] in ["M0-DeepJSCC-HR", "M2-SNRConditionedPixelResidualRestoration", "M3-ResidualRestorationTop1Fallback"]
    ]
    lines += markdown_table(
        residual_export,
        [
            ("method", "Method"),
            ("snr_db", "SNR"),
            ("psnr_db", "PSNR"),
            ("delta_psnr_vs_m0_db", "Delta PSNR"),
            ("lpips", "LPIPS"),
            ("semantic_failure", "Failure"),
            ("accept_rate", "Accept"),
        ],
    )
    lines.extend(["", "## Residual Shrink Schedule Check", ""])
    shrink_focus = [
        row
        for row in shrink_rows
        if row["policy"]
        in [
            "top1_fallback_alpha",
            "selected_top1_fallback_shrink_schedule",
            "top1_full_strength",
            "validation_top1_shrink_schedule",
            "always_full_strength",
            "validation_always_m0_failure_constrained_schedule",
        ]
    ]
    lines += markdown_table(
        shrink_focus,
        [
            ("split", "Split"),
            ("policy_label", "Policy"),
            ("delta_psnr_vs_m0_db", "Delta PSNR"),
            ("delta_lpips_vs_m0", "Delta LPIPS"),
            ("final_failure_rate", "Failure"),
            ("accept_rate", "Accept"),
            ("accepted_new_error_count", "New Error"),
        ],
    )
    lines.extend(["", "## Adaptive Residual Alpha Policy", ""])
    adaptive_focus = [
        row
        for row in adaptive_rows
        if row["policy"]
        in [
            "top1_full_strength",
            "fixed_validation_top1_shrink_schedule",
            "adaptive_max_top1_consistent_alpha",
            "always_full_strength",
        ]
    ]
    lines += markdown_table(
        adaptive_focus,
        [
            ("split", "Split"),
            ("policy", "Policy"),
            ("delta_psnr_vs_m0_db", "Delta PSNR"),
            ("delta_lpips_vs_m0", "Delta LPIPS"),
            ("final_failure_rate", "Failure"),
            ("accept_rate", "Accept"),
            ("mean_selected_alpha_accepted", "Mean Alpha"),
            ("accepted_new_error_count", "New Error"),
            ("missed_repair_count", "Missed Repair"),
        ],
    )
    lines.extend(["", "## Test-Like Gate Risk", ""])
    lines += markdown_table(
        testlike_rows,
        [
            ("policy", "Policy"),
            ("final_failure_rate", "Failure"),
            ("delta_final_psnr_vs_top1_db", "Delta PSNR vs Top1"),
            ("accepted_repair_count", "Repair"),
            ("accepted_new_error_count", "New Error"),
        ],
    )
    lines.extend(["", "## COCO-Object Clean-Correct Risk", ""])
    lines += markdown_table(
        clean_rows,
        [
            ("policy", "Policy"),
            ("final_failure_gt", "GT-like Failure"),
            ("delta_final_psnr_vs_top1_db", "Delta PSNR vs Top1"),
            ("accepted_repair_gt_count", "GT-like Repair"),
            ("accepted_new_error_gt_count", "GT-like New Error"),
        ],
    )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            f"- Residual quality: `{metadata['figures']['residual_quality']}`",
            f"- Residual semantics: `{metadata['figures']['residual_semantics']}`",
            f"- Residual shrink tradeoff: `{metadata['figures']['shrink_tradeoff']}`",
            f"- Adaptive alpha tradeoff: `{metadata['figures']['adaptive_tradeoff']}`",
            f"- Test-like tradeoff: `{metadata['figures']['testlike_tradeoff']}`",
            "",
            "## Caveats",
            "",
            "- M1 and EXP-S4-006 use different sample counts and splits; M1 is included as a negative reference, not a same-split comparison.",
            "- Residual shrink was selected on validation and frozen on held-out/test-like splits; it is a fixed-schedule conservative candidate, not supervised-label proof.",
            "- Adaptive residual alpha is receiver-side and does not use original images, but it is still a post-hoc policy over existing alpha candidates rather than a retrained residual model.",
            "- COCO pseudo-label, classifier ensemble, caption CLIP, and COCO-object CLIP are auxiliary semantic diagnostics.",
            "- The first closure should not claim that the selected risk rule is cross-model or supervised-label safe.",
            "",
            "## Files",
            "",
            f"- Method summary: `{metadata['method_summary_csv']}`",
            f"- Residual per-SNR table: `{metadata['residual_per_snr_csv']}`",
            f"- Blind diffusion negative table: `{metadata['blind_diffusion_csv']}`",
            f"- Residual shrink table: `{metadata['shrink_policy_csv']}`",
            f"- Adaptive residual alpha table: `{metadata['adaptive_policy_csv']}`",
            f"- Test-like policy table: `{metadata['testlike_policy_csv']}`",
            f"- COCO-object clean-correct table: `{metadata['clean_correct_policy_csv']}`",
            f"- Metadata: `{metadata['metadata_json']}`",
            "",
            "## Inputs",
            "",
        ]
    )
    for key, value in config["inputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    input_manifest = validate_inputs(config)
    output_dir = resolve_project_path(args.output_dir or config["outputs"]["output_dir"])
    dry_payload = {
        "status": "ok",
        "config": project_relative(config_path),
        "inputs": input_manifest,
        "output_dir": project_relative(output_dir),
        "proxy_environment_present": proxy_environment_present(),
        "notes": config.get("notes", []),
    }
    if args.dry_run:
        print(json.dumps(dry_payload, indent=2, ensure_ascii=False))
        return

    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Use --overwrite to replace it.")
    output_dir.mkdir(parents=True, exist_ok=False)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")

    m0_metrics = load_json(resolve_project_path(config["inputs"]["m0_formal_metrics"]))
    m1_metrics = load_json(resolve_project_path(config["inputs"]["m1_blind_diffusion_metrics"]))
    residual_summary = read_csv(resolve_project_path(config["inputs"]["residual_validation_summary"]))
    shrink_validation_summary = read_csv(resolve_project_path(config["inputs"]["residual_shrink_validation_summary"]))
    heldout_shrink_summary = read_csv(resolve_project_path(config["inputs"]["heldout_residual_shrink_summary"]))
    testlike_shrink_summary = read_csv(resolve_project_path(config["inputs"]["testlike_residual_shrink_summary"]))
    adaptive_summary = read_csv(resolve_project_path(config["inputs"]["adaptive_residual_alpha_summary"]))
    testlike_policy = read_csv(resolve_project_path(config["inputs"]["testlike_policy_summary"]))
    clean_summary = read_csv(resolve_project_path(config["inputs"]["testlike_clean_correct_summary"]))

    formal_m0_rows = build_formal_m0_rows(m0_metrics)
    m1_rows = build_m1_rows(m1_metrics)
    residual_rows = build_residual_rows(residual_summary)
    shrink_rows = build_shrink_policy_export(shrink_validation_summary, heldout_shrink_summary, testlike_shrink_summary)
    adaptive_rows = build_adaptive_policy_export(adaptive_summary)
    testlike_rows = build_testlike_policy_export(testlike_policy)
    clean_rows = build_clean_policy_export(clean_summary)
    method_summary = build_method_summary(
        formal_m0_rows,
        m1_rows,
        residual_rows,
        shrink_validation_summary,
        heldout_shrink_summary,
        testlike_shrink_summary,
        adaptive_summary,
        testlike_policy,
        clean_summary,
    )

    method_summary_csv = output_dir / "method_closure_summary.csv"
    residual_csv = output_dir / "residual_per_snr_quality_semantics.csv"
    blind_csv = output_dir / "blind_diffusion_negative_reference.csv"
    shrink_csv = output_dir / "residual_shrink_policy_tradeoff.csv"
    adaptive_csv = output_dir / "adaptive_residual_alpha_policy_tradeoff.csv"
    testlike_csv = output_dir / "testlike_policy_tradeoff.csv"
    clean_csv = output_dir / "coco_object_clean_correct_tradeoff.csv"
    metadata_json = output_dir / "metadata.json"
    report_md = output_dir / "REPORT.md"

    write_csv(method_summary_csv, method_summary)
    write_csv(residual_csv, residual_rows)
    write_csv(blind_csv, m1_rows)
    write_csv(shrink_csv, shrink_rows)
    write_csv(adaptive_csv, adaptive_rows)
    write_csv(testlike_csv, testlike_rows)
    write_csv(clean_csv, clean_rows)

    residual_quality = figures_dir / "residual_quality_vs_snr.png"
    residual_semantics = figures_dir / "residual_semantics_vs_snr.png"
    shrink_tradeoff = figures_dir / "residual_shrink_policy_tradeoff.png"
    adaptive_tradeoff = figures_dir / "adaptive_residual_alpha_policy_tradeoff.png"
    testlike_tradeoff = figures_dir / "testlike_policy_tradeoff.png"
    plot_residual_quality(residual_rows, residual_quality)
    plot_residual_semantics(residual_rows, residual_semantics)
    plot_shrink_tradeoff(shrink_rows, shrink_tradeoff)
    plot_adaptive_tradeoff(adaptive_rows, adaptive_tradeoff)
    plot_testlike_tradeoff(testlike_rows, clean_rows, testlike_tradeoff)

    metadata = {
        "analysis_id": config["analysis_id"],
        "config": project_relative(config_path),
        "copied_config": project_relative(output_dir / "config.yaml"),
        "output_dir": project_relative(output_dir),
        "inputs": input_manifest,
        "method_summary_csv": project_relative(method_summary_csv),
        "residual_per_snr_csv": project_relative(residual_csv),
        "blind_diffusion_csv": project_relative(blind_csv),
        "shrink_policy_csv": project_relative(shrink_csv),
        "adaptive_policy_csv": project_relative(adaptive_csv),
        "testlike_policy_csv": project_relative(testlike_csv),
        "clean_correct_policy_csv": project_relative(clean_csv),
        "metadata_json": project_relative(metadata_json),
        "report_md": project_relative(report_md),
        "figures": {
            "residual_quality": project_relative(residual_quality),
            "residual_semantics": project_relative(residual_semantics),
            "shrink_tradeoff": project_relative(shrink_tradeoff),
            "adaptive_tradeoff": project_relative(adaptive_tradeoff),
            "testlike_tradeoff": project_relative(testlike_tradeoff),
        },
        "git_commit": git_commit(),
        "git_dirty_state": git_dirty_state(),
        "python": sys.version,
        "platform": platform.platform(),
        "proxy_environment_present": proxy_environment_present(),
        "download_note": "No download is required; this report only reads existing local outputs.",
    }
    save_json(metadata_json, metadata)
    report_md.write_text(
        make_report(config, method_summary, residual_rows, shrink_rows, adaptive_rows, testlike_rows, clean_rows, metadata),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": project_relative(output_dir), "report": project_relative(report_md)}, indent=2))


if __name__ == "__main__":
    main()
