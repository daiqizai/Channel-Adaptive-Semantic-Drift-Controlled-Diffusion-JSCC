from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from s6_residual_shrink_selection import (
    aggregate_all_summary,
    build_candidate_images,
    classify_paths,
    compute_pair_metrics,
    fmt,
    git_commit,
    git_dirty_state,
    load_classifier,
    make_policy_rows_for_snr,
    make_sample_grids,
    proxy_environment_present,
    read_csv,
    resolve_device,
    resolve_project_path,
    save_json,
    signed,
    snr_name,
    summarize_policy,
    tensors_for_policy,
    try_load_lpips,
    write_csv,
    attach_metrics,
    project_relative,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply frozen residual-shrink schedules to held-out/test-like outputs.")
    parser.add_argument("--config", default="configs/s6_testlike_residual_shrink_schedule_check_exp_s4_006.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def input_value(config: dict[str, Any], key: str, legacy_key: str | None = None) -> str:
    inputs = config["inputs"]
    if key in inputs:
        return str(inputs[key])
    if legacy_key is not None and legacy_key in inputs:
        return str(inputs[legacy_key])
    raise KeyError(f"Missing input key: {key}")


def validate_inputs(config: dict[str, Any], snrs: list[float]) -> dict[str, str]:
    paths = {
        "validation_shrink_schedule": resolve_project_path(config["inputs"]["validation_shrink_schedule"]),
        "validation_shrink_summary": resolve_project_path(config["inputs"]["validation_shrink_summary"]),
        "source_config": resolve_project_path(input_value(config, "source_config", "testlike_source_config")),
        "per_sample_csv": resolve_project_path(input_value(config, "per_sample_csv", "testlike_per_sample_csv")),
        "summary_csv": resolve_project_path(input_value(config, "summary_csv", "testlike_summary_csv")),
        "original_dir": resolve_project_path(config["inputs"]["original_dir"]),
        "m0_export_dir": resolve_project_path(config["inputs"]["m0_export_dir"]),
        "refined_export_dir": resolve_project_path(config["inputs"]["refined_export_dir"]),
        "jscc_checkpoint": resolve_project_path(config["inputs"]["jscc_checkpoint"]),
        "forbidden_checkpoint": resolve_project_path(config["inputs"]["forbidden_checkpoint"]),
        "classifier_weights": resolve_project_path(config["classifier"]["weights_file"]),
    }
    missing = [f"{key}: {path}" for key, path in paths.items() if key != "forbidden_checkpoint" and not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))
    if paths["jscc_checkpoint"] == paths["forbidden_checkpoint"]:
        raise RuntimeError("Config points to forbidden latest.pt checkpoint.")
    if not paths["classifier_weights"].is_file() or paths["classifier_weights"].stat().st_size < 10 * 1024 * 1024:
        raise RuntimeError(f"Classifier weights missing from local cache: {paths['classifier_weights']}")
    for snr in snrs:
        refined_dir = paths["refined_export_dir"] / snr_name(snr) / "refined"
        if not refined_dir.exists():
            raise FileNotFoundError(f"Refined directory missing: {refined_dir}")
    return {key: project_relative(path) for key, path in paths.items()}


def snr_key_from_config(snr: float) -> str:
    if float(snr).is_integer():
        return str(int(snr))
    return str(snr)


def schedule_alpha(schedule: dict[str, Any], snr: float) -> float | None:
    values = schedule["alphas_by_snr"]
    key = snr_key_from_config(snr)
    if key not in values:
        raise KeyError(f"Schedule {schedule['name']} missing SNR {key}")
    value = values[key]
    if value in (None, "null", ""):
        return None
    return float(value)


def validate_frozen_schedule_json(config: dict[str, Any]) -> dict[str, Any]:
    path = resolve_project_path(config["inputs"]["validation_shrink_schedule"])
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    mismatches: list[str] = []
    for schedule in config["frozen_schedules"]:
        source = schedule.get("schedule_source", "")
        if "::" not in source:
            continue
        source_key = source.split("::", 1)[1]
        if source_key not in payload:
            mismatches.append(f"{schedule['name']}: missing source key {source_key}")
            continue
        for snr, alpha in schedule["alphas_by_snr"].items():
            source_alpha = payload[source_key].get(f"{float(snr):.1f}", payload[source_key].get(str(snr)))
            if source_alpha != alpha:
                mismatches.append(f"{schedule['name']} SNR {snr}: config={alpha}, source={source_alpha}")
    if mismatches:
        raise RuntimeError("Frozen schedule config does not match validation schedule:\n" + "\n".join(mismatches))
    return payload


def needed_alphas(config: dict[str, Any], snrs: list[float]) -> list[float]:
    values: set[float] = set()
    for schedule in config["frozen_schedules"]:
        for snr in snrs:
            alpha = schedule_alpha(schedule, snr)
            if alpha is not None:
                values.add(float(alpha))
    return sorted(values)


def materialize_final_sources(rows: list[dict[str, Any]], output_dir: Path, schedule_name: str) -> None:
    for row in rows:
        snr = float(row["snr_db"])
        out_dir = output_dir / "exports" / snr_name(snr) / schedule_name
        out_dir.mkdir(parents=True, exist_ok=True)
        final_path = out_dir / row["sample"]
        shutil.copy2(resolve_project_path(row["final_source"]), final_path)
        row["final_source"] = project_relative(final_path)


def summarize_rows_for_snr(
    source_rows: list[dict[str, str]],
    policy_rows: list[dict[str, Any]],
    policy_name: str,
    alpha: float | None,
    snr: float,
    m0_metrics: dict[str, float | None],
    lpips_model,
    device: torch.device,
    image_batch_size: int,
) -> dict[str, Any]:
    references, _m0_tensor, final_tensor = tensors_for_policy(source_rows, policy_rows)
    metrics = compute_pair_metrics(references, final_tensor, lpips_model, device, image_batch_size)
    summary = summarize_policy(policy_rows, policy_name, alpha, snr)
    attach_metrics(summary, metrics, m0_metrics)
    return summary


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    lines = []
    headers = [label for _key, label in columns]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---" for _ in headers]) + "|")
    for row in rows:
        values = [fmt(row.get(key, "")) for key, _label in columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def make_report(
    config: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> str:
    split_name = str(config.get("split_name", "test-like"))
    all_rows = {row["policy"]: row for row in summary_rows if row["snr_db"] == "all"}
    top1_full = all_rows["top1_full_strength"]
    top1_shrink = all_rows["validation_top1_shrink_schedule"]
    always_full = all_rows["always_full_strength"]
    always_constrained = all_rows["validation_always_m0_failure_constrained_schedule"]
    delta_shrink_vs_full = float(top1_shrink["delta_psnr_vs_m0_db"]) - float(top1_full["delta_psnr_vs_m0_db"])
    lines = [
        f"# Frozen Residual Shrink Schedule Check: {split_name}",
        "",
        f"This report applies residual-shrink schedules selected on EXP-S4-006 validation outputs to the {split_name} split.",
        f"It does not tune alpha on {split_name} samples.",
        "",
        "## Bottom Line",
        "",
        f"- Full-strength top-1 fallback gives PSNR delta `{signed(top1_full['delta_psnr_vs_m0_db'])}` dB vs M0 with final failure delta `{signed(top1_full['delta_final_failure_vs_m0'])}`.",
        f"- Frozen validation top-1 shrink schedule gives PSNR delta `{signed(top1_shrink['delta_psnr_vs_m0_db'])}` dB vs M0 with final failure delta `{signed(top1_shrink['delta_final_failure_vs_m0'])}`.",
        f"- Frozen shrink changes PSNR by `{signed(delta_shrink_vs_full)}` dB vs full-strength top-1 fallback on {split_name}.",
        f"- Always-accept full strength has `{always_full['accepted_new_error_count']}` accepted new errors; validation always-constrained schedule still has `{always_constrained['accepted_new_error_count']}`, so neither is a safe M3.",
        "",
        "## All-Policy Summary",
        "",
    ]
    all_export = [row for row in summary_rows if row["snr_db"] == "all"]
    lines += markdown_table(
        all_export,
        [
            ("policy", "Policy"),
            ("delta_psnr_vs_m0_db", "Delta PSNR"),
            ("delta_lpips_vs_m0", "Delta LPIPS"),
            ("final_failure_rate", "Failure"),
            ("delta_final_failure_vs_m0", "Delta Failure"),
            ("accept_rate", "Accept"),
            ("repair_count", "Repair"),
            ("accepted_new_error_count", "New Error"),
        ],
    )
    lines.extend(["", "## Per-SNR Frozen Top-1 Shrink", ""])
    shrink_rows = [
        row
        for row in summary_rows
        if row["policy"] == "validation_top1_shrink_schedule" and row["snr_db"] != "all"
    ]
    lines += markdown_table(
        shrink_rows,
        [
            ("snr_db", "SNR"),
            ("alpha", "Alpha"),
            ("delta_psnr_vs_m0_db", "Delta PSNR"),
            ("final_failure_rate", "Failure"),
            ("accept_rate", "Accept"),
            ("repair_count", "Repair"),
            ("accepted_new_error_count", "New Error"),
        ],
    )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Summary CSV: `{metadata['summary_csv']}`",
            f"- Per-sample CSV: `{metadata['per_sample_csv']}`",
            f"- Metadata: `{metadata['metadata_json']}`",
            f"- Sample grids: `{metadata['sample_dir']}`",
            "",
            "## Caveats",
            "",
            "- This is still a COCO pseudo-label/AlexNet auxiliary semantic metric.",
            f"- The schedule is frozen from validation, but this is still a {split_name} split from the same COCO val export family.",
            "- A smaller residual strength may improve validation but must be verified before becoming final M3.",
            "",
            "## Frozen Schedules",
            "",
        ]
    )
    for schedule in config["frozen_schedules"]:
        lines.append(f"- `{schedule['name']}`: `{schedule['alphas_by_snr']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    snrs = [float(item) for item in config["snrs"]]
    input_manifest = validate_inputs(config, snrs)
    validation_schedule = validate_frozen_schedule_json(config)
    output_dir = resolve_project_path(args.output_dir or config["outputs"]["output_dir"])
    dry_payload = {
        "status": "ok",
        "config": project_relative(config_path),
        "inputs": input_manifest,
        "output_dir": project_relative(output_dir),
        "snrs": snrs,
        "needed_alphas": needed_alphas(config, snrs),
        "validation_schedule": {
            "top1_fallback_alpha": validation_schedule.get("top1_fallback_alpha", {}),
            "always_alpha": validation_schedule.get("always_alpha", {}),
        },
        "proxy_environment_present": proxy_environment_present(),
        "notes": config.get("notes", []),
    }
    if args.dry_run:
        print(json.dumps(dry_payload, indent=2, ensure_ascii=False))
        return
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output_dir / "config.yaml")

    device = resolve_device(args.device)
    image_batch_size = int(config["evaluation"]["image_batch_size"])
    rows = read_csv(resolve_project_path(input_value(config, "per_sample_csv", "testlike_per_sample_csv")))
    rows = sorted(rows, key=lambda row: (float(row["snr_db"]), row["sample"]))
    grouped: dict[float, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(float(row["snr_db"]), []).append(row)

    classifier_model, classifier_preprocess, categories = load_classifier(config, device)
    lpips_model = None
    lpips_error = None
    if not args.skip_lpips:
        lpips_model, lpips_error = try_load_lpips(device, output_dir / "cache")

    alpha_paths: dict[float, dict[tuple[float, str], Path]] = {}
    alpha_preds: dict[float, dict[float, list[dict[str, Any]]]] = {}
    classification_times: dict[str, float] = {}
    for alpha in needed_alphas(config, snrs):
        alpha_paths[alpha] = build_candidate_images(rows, alpha, output_dir / "candidates")
        alpha_preds[alpha] = {}
        for snr in snrs:
            snr_rows = grouped[snr]
            paths = [alpha_paths[alpha][(snr, row["sample"])] for row in snr_rows]
            preds, elapsed = classify_paths(
                classifier_model,
                classifier_preprocess,
                paths,
                int(config["classifier"]["batch_size"]),
                int(config["classifier"]["topk"]),
                device,
            )
            alpha_preds[alpha][snr] = preds
            classification_times[f"alpha_{alpha}_{snr_name(snr)}"] = elapsed

    policy_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    m0_metrics_by_snr: dict[float, dict[str, float | None]] = {}

    for snr in snrs:
        snr_rows = grouped[snr]
        m0_rows = make_policy_rows_for_snr(snr_rows, None, None, None, "m0", categories)
        references, m0_tensor, _final_tensor = tensors_for_policy(snr_rows, m0_rows)
        m0_metrics = compute_pair_metrics(references, m0_tensor, lpips_model, device, image_batch_size)
        m0_metrics_by_snr[snr] = m0_metrics
        materialize_final_sources(m0_rows, output_dir, "m0")
        m0_summary = summarize_rows_for_snr(
            snr_rows,
            m0_rows,
            "m0",
            None,
            snr,
            m0_metrics,
            lpips_model,
            device,
            image_batch_size,
        )
        policy_rows.extend(m0_rows)
        summary_rows.append(m0_summary)

    for schedule in config["frozen_schedules"]:
        schedule_name = str(schedule["name"])
        source_policy = str(schedule["source_policy"])
        for snr in snrs:
            snr_rows = grouped[snr]
            alpha = schedule_alpha(schedule, snr)
            if alpha is None:
                rows_for_policy = make_policy_rows_for_snr(snr_rows, None, None, None, "m0", categories)
            else:
                candidate_paths = [alpha_paths[alpha][(snr, row["sample"])] for row in snr_rows]
                candidate_preds = alpha_preds[alpha][snr]
                rows_for_policy = make_policy_rows_for_snr(
                    snr_rows,
                    alpha,
                    candidate_preds,
                    candidate_paths,
                    source_policy,
                    categories,
                )
            for row in rows_for_policy:
                row["policy"] = schedule_name
                row["source_policy"] = source_policy
                row["alpha"] = "" if alpha is None else float(alpha)
            materialize_final_sources(rows_for_policy, output_dir, schedule_name)
            summary = summarize_rows_for_snr(
                snr_rows,
                rows_for_policy,
                schedule_name,
                alpha,
                snr,
                m0_metrics_by_snr[snr],
                lpips_model,
                device,
                image_batch_size,
            )
            policy_rows.extend(rows_for_policy)
            summary_rows.append(summary)

    policies = ["m0"] + [str(schedule["name"]) for schedule in config["frozen_schedules"]]
    for policy in policies:
        rows_for_policy = [row for row in policy_rows if row["policy"] == policy]
        aggregate_all_summary(rows_for_policy, summary_rows, lpips_model, device, image_batch_size)

    sample_dir = output_dir / "samples"
    sample_grids = make_sample_grids(
        [row for row in policy_rows if row["policy"] == "validation_top1_shrink_schedule"],
        output_dir,
        int(config["evaluation"]["sample_grid_count"]),
    )
    summary_csv = output_dir / "summary.csv"
    per_sample_csv = output_dir / "per_sample.csv"
    metadata_json = output_dir / "metadata.json"
    report_path = output_dir / "REPORT.md"
    write_csv(summary_csv, summary_rows)
    write_csv(per_sample_csv, policy_rows)
    metadata = {
        "analysis_id": config["analysis_id"],
        "method": config["method"],
        "split_name": config.get("split_name", "test-like"),
        "project_commit": git_commit(),
        "git_dirty_state": git_dirty_state(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "config": project_relative(config_path),
        "input_manifest": input_manifest,
        "summary_csv": project_relative(summary_csv),
        "per_sample_csv": project_relative(per_sample_csv),
        "metadata_json": project_relative(metadata_json),
        "sample_dir": project_relative(sample_dir),
        "sample_grids": sample_grids,
        "classification_times_sec": classification_times,
        "lpips_error": lpips_error,
        "proxy_environment_present": proxy_environment_present(),
        "notes": config.get("notes", []),
    }
    save_json(metadata_json, metadata)
    report_path.write_text(make_report(config, summary_rows, metadata), encoding="utf-8")
    print(json.dumps({"output_dir": project_relative(output_dir), "report": project_relative(report_path)}, indent=2))


if __name__ == "__main__":
    main()
