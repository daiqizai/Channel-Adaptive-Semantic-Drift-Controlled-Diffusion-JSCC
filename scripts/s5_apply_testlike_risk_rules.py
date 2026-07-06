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

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from s5_sweep_conf_gain_clip_veto import (  # noqa: E402
    add_clip_similarity,
    encode_paths,
    load_clip_model,
    normalize_rows,
    resolve_device,
)
from s5_sweep_conf_gain_risk_rules import (  # noqa: E402
    make_sheet,
    parse_float_list,
    parse_list_field,
    rule_shadow_veto,
    top1_rank_in_topk,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply frozen EXP-S4-006 selected risk rules on the test-like split."
    )
    parser.add_argument("--config", default="configs/s5_testlike_risk_rule_check_exp_s4_006.yaml")
    parser.add_argument("--device", default="auto")
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


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def rate(flags: list[bool]) -> float:
    return float(sum(flags) / len(flags)) if flags else 0.0


def snr_name(snr: float) -> str:
    if float(snr).is_integer():
        return f"snr_{int(snr):02d}db"
    return f"snr_{str(snr).replace('.', 'p')}db"


def serialize_value(value: Any) -> Any:
    if isinstance(value, bool):
        return bool_text(value)
    if isinstance(value, (list, dict)):
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_value(row.get(key, "")) for key in fieldnames})


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


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
    inputs = config["inputs"]
    paths = {
        "testlike_csv": resolve_project_path(inputs["testlike_csv"]),
        "selected_risk_rule_json": resolve_project_path(inputs["selected_risk_rule_json"]),
        "conservative_veto_rule_json": resolve_project_path(inputs["conservative_veto_rule_json"]),
        "source_risk_rule_config": resolve_project_path(inputs["source_risk_rule_config"]),
        "source_conservative_veto_config": resolve_project_path(inputs["source_conservative_veto_config"]),
        "checkpoint": resolve_project_path(inputs["checkpoint"]),
        "forbidden_checkpoint": resolve_project_path(inputs["forbidden_checkpoint"]),
        "clip_checkpoint": resolve_project_path(config["clip"]["pretrained_path"]),
    }
    for key, path in paths.items():
        if key == "forbidden_checkpoint":
            continue
        if not path.exists():
            raise FileNotFoundError(f"Required input missing: {key}: {path}")
    if paths["checkpoint"] == paths["forbidden_checkpoint"]:
        raise RuntimeError("Config points to forbidden latest.pt checkpoint.")
    if paths["clip_checkpoint"].stat().st_size < 100 * 1024 * 1024:
        raise RuntimeError(f"CLIP checkpoint is missing or too small: {paths['clip_checkpoint']}")
    return {key: project_relative(path) for key, path in paths.items()}


def check_image_paths(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        for key in ["original", "m0_reconstruction", "refined"]:
            path = resolve_project_path(row[key])
            if not path.exists():
                raise FileNotFoundError(f"Image path from CSV not found: {path}")


def enrich_receiver_features(base_rows: list[dict[str, Any]], raw_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    raw_by_key = {(float(row["snr_db"]), row["sample"]): row for row in raw_rows}
    output: list[dict[str, Any]] = []
    for row in base_rows:
        key = (float(row["snr_db"]), str(row["sample"]))
        if key not in raw_by_key:
            raise KeyError(f"Missing raw top-k row for {key}")
        raw = raw_by_key[key]
        m0_indices = parse_list_field(raw["m0_top_indices"])
        refined_indices = parse_list_field(raw["refined_top_indices"])
        m0_probs = parse_float_list(raw["m0_top_probs"])
        refined_probs = parse_float_list(raw["refined_top_probs"])
        enriched = dict(row)
        enriched["original_top1_label"] = raw.get("original_top1_label", "")
        enriched["m0_refined_top5_overlap"] = len(set(m0_indices) & set(refined_indices))
        enriched["m0_top1_rank_in_refined_top5"] = top1_rank_in_topk(m0_indices[0], refined_indices) if m0_indices else 99
        enriched["refined_top1_rank_in_m0_top5"] = top1_rank_in_topk(refined_indices[0], m0_indices) if refined_indices else 99
        enriched["m0_top1_margin"] = (m0_probs[0] - m0_probs[1]) if len(m0_probs) > 1 else (m0_probs[0] if m0_probs else 0.0)
        enriched["refined_top1_margin"] = (
            refined_probs[0] - refined_probs[1]
            if len(refined_probs) > 1
            else (refined_probs[0] if refined_probs else 0.0)
        )
        enriched["m0_top_indices"] = "|".join(m0_indices)
        enriched["refined_top_indices"] = "|".join(refined_indices)
        enriched["m0_top_probs"] = "|".join(f"{value:.8f}" for value in m0_probs)
        enriched["refined_top_probs"] = "|".join(f"{value:.8f}" for value in refined_probs)
        output.append(enriched)
    return output


def selected_risk_rule_accept(row: dict[str, Any], rule: dict[str, Any]) -> bool:
    if bool(row["baseline_accept_refined"]):
        return True
    if not bool(row["candidate_accept_refined"]):
        return False
    if float(row["clip_sim_m0_refined"]) < float(rule["clip_min"]):
        return False
    if int(row["m0_refined_top5_overlap"]) < int(rule["top5_overlap_min"]):
        return False
    if rule_shadow_veto(row, rule):
        return False
    return True


def conservative_veto(row: dict[str, Any], selected_accept: bool, rule: dict[str, Any]) -> bool:
    if not selected_accept:
        return False
    baseline_accept = bool(row["baseline_accept_refined"])
    new_accept = selected_accept and not baseline_accept
    if new_accept and float(row["refined_top1_margin"]) <= float(rule["new_accept_refined_margin_max"]):
        return True
    if (
        baseline_accept
        and float(row["refined_conf_gain_vs_m0"]) <= float(rule["top1_refined_conf_gain_max"])
        and float(row["m0_top1_margin"]) >= float(rule["top1_m0_margin_min"])
    ):
        return True
    return False


def policy_accept(row: dict[str, Any], policy: str, selected_rule: dict[str, Any], conservative_rule: dict[str, Any]) -> bool:
    if policy == "top1_equal":
        return bool(row["baseline_accept_refined"])
    if policy == "raw_conf_gain":
        return bool(row["candidate_accept_refined"])
    if policy == "fixed_clip_ge_0p98":
        return bool(row["baseline_accept_refined"]) or (
            bool(row["candidate_accept_refined"]) and float(row["clip_sim_m0_refined"]) >= 0.98
        )
    if policy == "selected_risk_rule":
        return selected_risk_rule_accept(row, selected_rule)
    if policy == "selected_risk_rule_plus_ensemble_veto":
        selected_accept = selected_risk_rule_accept(row, selected_rule)
        return selected_accept and not conservative_veto(row, selected_accept, conservative_rule)
    raise ValueError(f"Unknown policy: {policy}")


def make_decisions(
    rows: list[dict[str, Any]],
    policies: list[str],
    selected_rule: dict[str, Any],
    conservative_rule: dict[str, Any],
) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for row in rows:
        selected_accept = selected_risk_rule_accept(row, selected_rule)
        selected_shadow_veto = bool(
            bool(row["candidate_accept_refined"])
            and (not bool(row["baseline_accept_refined"]))
            and rule_shadow_veto(row, selected_rule)
        )
        conservative_veto_flag = conservative_veto(row, selected_accept, conservative_rule)
        for policy in policies:
            accept = policy_accept(row, policy, selected_rule, conservative_rule)
            m0_ok = bool(row["m0_matches_original_top1"])
            refined_ok = bool(row["refined_matches_original_top1"])
            baseline_accept = bool(row["baseline_accept_refined"])
            candidate_accept = bool(row["candidate_accept_refined"])
            final_key = "refined" if accept else "m0_reconstruction"
            decisions.append(
                {
                    "split": row["split"],
                    "policy": policy,
                    "snr_db": row["snr_db"],
                    "sample": row["sample"],
                    "accept_refined": accept,
                    "baseline_accept_refined": baseline_accept,
                    "candidate_accept_refined": candidate_accept,
                    "new_accept_vs_top1": accept and not baseline_accept,
                    "vetoed_raw_candidate_accept": candidate_accept and not accept,
                    "selected_risk_rule_shadow_veto": selected_shadow_veto,
                    "conservative_veto": conservative_veto_flag if policy == "selected_risk_rule_plus_ensemble_veto" else False,
                    "final_matches_original_top1": refined_ok if accept else m0_ok,
                    "m0_matches_original_top1": m0_ok,
                    "refined_matches_original_top1": refined_ok,
                    "accepted_repair": accept and (not m0_ok) and refined_ok,
                    "missed_repair": (not accept) and (not m0_ok) and refined_ok,
                    "accepted_new_error": accept and m0_ok and (not refined_ok),
                    "protective_reject": (not accept) and m0_ok and (not refined_ok),
                    "vetoed_raw_candidate_repair": candidate_accept and (not accept) and (not m0_ok) and refined_ok,
                    "vetoed_raw_candidate_new_error": candidate_accept and (not accept) and m0_ok and (not refined_ok),
                    "final_psnr_db": float(row["refined_psnr_db"] if accept else row["m0_psnr_db"]),
                    "m0_psnr_db": row["m0_psnr_db"],
                    "refined_psnr_db": row["refined_psnr_db"],
                    "clip_sim_m0_refined": row["clip_sim_m0_refined"],
                    "clip_distance_m0_refined": row["clip_distance_m0_refined"],
                    "m0_refined_top5_overlap": row["m0_refined_top5_overlap"],
                    "m0_top1_rank_in_refined_top5": row["m0_top1_rank_in_refined_top5"],
                    "refined_top1_rank_in_m0_top5": row["refined_top1_rank_in_m0_top5"],
                    "m0_top1_margin": row["m0_top1_margin"],
                    "refined_top1_margin": row["refined_top1_margin"],
                    "m0_top1_label": row["m0_top1_label"],
                    "refined_top1_label": row["refined_top1_label"],
                    "original_top1_label": row.get("original_top1_label", ""),
                    "m0_top1_prob": row["m0_top1_prob"],
                    "refined_top1_prob": row["refined_top1_prob"],
                    "refined_conf_gain_vs_m0": row["refined_conf_gain_vs_m0"],
                    "original": row["original"],
                    "m0_reconstruction": row["m0_reconstruction"],
                    "refined": row["refined"],
                    "final_source": row[final_key],
                }
            )
    return decisions


def summarize(rows: list[dict[str, Any]], policy: str, snr: float | None = None) -> dict[str, Any]:
    subset = [row for row in rows if row["policy"] == policy and (snr is None or float(row["snr_db"]) == float(snr))]
    return {
        "split": "testlike",
        "policy": policy,
        "snr_db": "all" if snr is None else float(snr),
        "num_images": len(subset),
        "accept_count": sum(parse_bool(row["accept_refined"]) for row in subset),
        "accept_rate": rate([parse_bool(row["accept_refined"]) for row in subset]),
        "new_accept_vs_top1_count": sum(parse_bool(row["new_accept_vs_top1"]) for row in subset),
        "vetoed_raw_candidate_accept_count": sum(parse_bool(row["vetoed_raw_candidate_accept"]) for row in subset),
        "vetoed_raw_candidate_repair_count": sum(parse_bool(row["vetoed_raw_candidate_repair"]) for row in subset),
        "vetoed_raw_candidate_new_error_count": sum(parse_bool(row["vetoed_raw_candidate_new_error"]) for row in subset),
        "selected_risk_rule_shadow_veto_count": sum(parse_bool(row["selected_risk_rule_shadow_veto"]) for row in subset),
        "conservative_veto_count": sum(parse_bool(row["conservative_veto"]) for row in subset),
        "m0_failure_rate": 1.0 - rate([parse_bool(row["m0_matches_original_top1"]) for row in subset])
        if subset and "m0_matches_original_top1" in subset[0]
        else "",
        "refined_failure_rate": 1.0 - rate([parse_bool(row["refined_matches_original_top1"]) for row in subset])
        if subset and "refined_matches_original_top1" in subset[0]
        else "",
        "final_failure_rate": 1.0 - rate([parse_bool(row["final_matches_original_top1"]) for row in subset]),
        "accepted_repair_count": sum(parse_bool(row["accepted_repair"]) for row in subset),
        "accepted_new_error_count": sum(parse_bool(row["accepted_new_error"]) for row in subset),
        "missed_repair_count": sum(parse_bool(row["missed_repair"]) for row in subset),
        "protective_reject_count": sum(parse_bool(row["protective_reject"]) for row in subset),
        "final_psnr_db": mean([float(row["final_psnr_db"]) for row in subset]),
        "m0_psnr_db": mean([float(row["m0_psnr_db"]) for row in subset]),
        "refined_psnr_db": mean([float(row["refined_psnr_db"]) for row in subset]),
        "final_delta_psnr_vs_m0_db": mean([float(row["final_psnr_db"]) for row in subset])
        - mean([float(row["m0_psnr_db"]) for row in subset]),
        "new_accept_clip_sim_m0_refined_min": min(
            [float(row["clip_sim_m0_refined"]) for row in subset if parse_bool(row["new_accept_vs_top1"])],
            default="",
        ),
        "new_accept_clip_sim_m0_refined_mean": mean(
            [float(row["clip_sim_m0_refined"]) for row in subset if parse_bool(row["new_accept_vs_top1"])]
        ),
    }


def add_summary_deltas(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed = {(str(row["policy"]), str(row["snr_db"])): row for row in summary_rows}
    output: list[dict[str, Any]] = []
    for row in summary_rows:
        enriched = dict(row)
        for baseline in ["top1_equal", "raw_conf_gain", "selected_risk_rule"]:
            base = keyed.get((baseline, str(row["snr_db"])))
            if not base:
                continue
            suffix = baseline
            enriched[f"delta_final_failure_vs_{suffix}"] = (
                float(row["final_failure_rate"]) - float(base["final_failure_rate"])
            )
            enriched[f"delta_final_psnr_vs_{suffix}_db"] = float(row["final_psnr_db"]) - float(base["final_psnr_db"])
            enriched[f"delta_accepted_repair_vs_{suffix}"] = (
                int(row["accepted_repair_count"]) - int(base["accepted_repair_count"])
            )
            enriched[f"delta_accepted_new_error_vs_{suffix}"] = (
                int(row["accepted_new_error_count"]) - int(base["accepted_new_error_count"])
            )
        output.append(enriched)
    return output


def materialize(decisions: list[dict[str, Any]], policies: list[str], output_dir: Path) -> None:
    for row in decisions:
        if str(row["policy"]) not in policies:
            continue
        source_path = resolve_project_path(row["final_source"])
        final_dir = output_dir / "exports" / str(row["policy"]) / snr_name(float(row["snr_db"])) / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / str(row["sample"])
        shutil.copy2(source_path, final_path)
        row["materialized_final"] = project_relative(final_path)
        row["materialized_source"] = project_relative(source_path)


def write_galleries(decisions: list[dict[str, Any]], output_dir: Path, gallery_rows: int) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for policy in ["selected_risk_rule", "selected_risk_rule_plus_ensemble_veto"]:
        rows = [row for row in decisions if str(row["policy"]) == policy]
        groups = [
            ("accepted_new_errors", [row for row in rows if parse_bool(row["accepted_new_error"])]),
            ("accepted_repairs", [row for row in rows if parse_bool(row["accepted_repair"])]),
            ("vetoed_raw_candidate_new_errors", [row for row in rows if parse_bool(row["vetoed_raw_candidate_new_error"])]),
            ("vetoed_raw_candidate_repairs", [row for row in rows if parse_bool(row["vetoed_raw_candidate_repair"])]),
        ]
        for name, group_rows in groups:
            selected = sorted(group_rows, key=lambda row: float(row["clip_sim_m0_refined"]))[:gallery_rows]
            path = output_dir / "galleries" / policy / f"{name}.png"
            make_sheet(selected, path, f"{policy} testlike {name}")
            if path.exists():
                manifest[f"{policy}_{name}"] = project_relative(path)
    return manifest


def make_report(summary_rows: list[dict[str, Any]], metadata: dict[str, Any]) -> str:
    global_rows = [row for row in summary_rows if str(row["snr_db"]) == "all"]
    by_policy = {str(row["policy"]): row for row in global_rows}
    selected = by_policy["selected_risk_rule"]
    conservative = by_policy["selected_risk_rule_plus_ensemble_veto"]
    lines = [
        "# EXP-S4-006 Test-Like Frozen Risk Rule Check",
        "",
        "This analysis applies frozen receiver-side rules to the test-like split; it does not tune thresholds on these samples.",
        "",
        "## Overall",
        "",
        "| Policy | Final Failure | Delta vs Top1 | Final PSNR | Delta PSNR vs Top1 | Repair | New Error | New Accept | Vetoed Raw New Error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in global_rows:
        lines.append(
            "| {policy} | {fail:.4f} | {dfail:+.4f} | {psnr:.4f} | {dpsnr:+.4f} | {repair} | {new_error} | {new_accept} | {vnew} |".format(
                policy=row["policy"],
                fail=float(row["final_failure_rate"]),
                dfail=float(row.get("delta_final_failure_vs_top1_equal", 0.0)),
                psnr=float(row["final_psnr_db"]),
                dpsnr=float(row.get("delta_final_psnr_vs_top1_equal_db", 0.0)),
                repair=int(row["accepted_repair_count"]),
                new_error=int(row["accepted_new_error_count"]),
                new_accept=int(row["new_accept_vs_top1_count"]),
                vnew=int(row["vetoed_raw_candidate_new_error_count"]),
            )
        )
    lines.extend(
        [
            "",
            "## Frozen Rule Comparison",
            "",
            "- `selected_risk_rule` is the validation-tuned shadow-margin rule from the previous sweep.",
            "- `selected_risk_rule_plus_ensemble_veto` adds the frozen conservative veto selected from classifier-ensemble risk analysis.",
            "- Both policies use receiver-side features only; original pseudo-labels are used here only for offline evaluation.",
            "",
            "## Key Readout",
            "",
            f"- Selected risk rule: repair `{int(selected['accepted_repair_count'])}`, new error `{int(selected['accepted_new_error_count'])}`, PSNR delta vs top-1 `{float(selected['delta_final_psnr_vs_top1_equal_db']):+.4f}` dB.",
            f"- Conservative veto: repair `{int(conservative['accepted_repair_count'])}`, new error `{int(conservative['accepted_new_error_count'])}`, PSNR delta vs selected `{float(conservative['delta_final_psnr_vs_selected_risk_rule_db']):+.4f}` dB.",
            "",
            "## Output Files",
            "",
            f"- `per_sample_with_clip.csv`: `{metadata['per_sample_with_clip_csv']}`",
            f"- `policy_decisions.csv`: `{metadata['policy_decisions_csv']}`",
            f"- `policy_summary.csv`: `{metadata['policy_summary_csv']}`",
            f"- `policy_by_snr.csv`: `{metadata['policy_by_snr_csv']}`",
            f"- `metadata.json`: `{metadata['metadata_json']}`",
        ]
    )
    if metadata["galleries"]:
        lines.extend(["", "## Galleries", ""])
        for key, path in sorted(metadata["galleries"].items()):
            lines.append(f"- `{key}`: `{path}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    config = load_yaml(config_path)
    paths = validate_inputs(config)
    split_name = str(config["policy"]["split_name"])
    raw_rows = read_csv(resolve_project_path(config["inputs"]["testlike_csv"]))
    rows = normalize_rows(raw_rows, split_name, float(config["policy"]["refined_conf_gain_margin"]))
    rows = enrich_receiver_features(rows, raw_rows)
    check_image_paths(rows)

    selected_rule = load_json(resolve_project_path(config["inputs"]["selected_risk_rule_json"]))
    conservative_rule = load_json(resolve_project_path(config["inputs"]["conservative_veto_rule_json"]))
    output_dir = resolve_project_path(args.output_dir or config["outputs"]["output_dir"])
    policies = [
        "top1_equal",
        "raw_conf_gain",
        "fixed_clip_ge_0p98",
        "selected_risk_rule",
        "selected_risk_rule_plus_ensemble_veto",
    ]
    unique_clip_paths = sorted({resolve_project_path(row[key]) for row in rows for key in ["m0_reconstruction", "refined"]})

    if args.dry_run:
        print(json.dumps(
            {
                "status": "ok",
                "config": project_relative(config_path),
                "rows": len(rows),
                "split_name": split_name,
                "unique_clip_image_paths": len(unique_clip_paths),
                "policies": policies,
                "output_dir": project_relative(output_dir),
                "proxy_environment_present": proxy_environment_present(),
            },
            indent=2,
        ))
        return

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output_dir / "config.yaml")

    device = resolve_device(args.device)
    model, preprocess = load_clip_model(config, device)
    image_features, clip_elapsed = encode_paths(
        model=model,
        preprocess=preprocess,
        paths=unique_clip_paths,
        batch_size=int(config["clip"]["batch_size"]),
        device=device,
    )
    rows = add_clip_similarity(rows, image_features)
    per_sample_with_clip_path = output_dir / "per_sample_with_clip.csv"
    write_csv(per_sample_with_clip_path, rows)

    decisions = make_decisions(rows, policies, selected_rule, conservative_rule)
    materialize(decisions, [str(item) for item in config["policy"]["materialize_policies"]], output_dir)
    summary_rows = [summarize(decisions, policy, None) for policy in policies]
    summary_rows += [
        summarize(decisions, policy, snr)
        for policy in policies
        for snr in sorted({float(row["snr_db"]) for row in rows})
    ]
    summary_rows = add_summary_deltas(summary_rows)
    gallery_manifest = write_galleries(decisions, output_dir, int(config["evaluation"]["gallery_rows"]))

    decisions_path = output_dir / "policy_decisions.csv"
    summary_path = output_dir / "policy_summary.csv"
    by_snr_path = output_dir / "policy_by_snr.csv"
    metadata_path = output_dir / "metadata.json"
    report_path = output_dir / "REPORT.md"
    write_csv(decisions_path, decisions)
    write_csv(summary_path, [row for row in summary_rows if str(row["snr_db"]) == "all"])
    write_csv(by_snr_path, [row for row in summary_rows if str(row["snr_db"]) != "all"])

    metadata = {
        "analysis_id": config["analysis_id"],
        "source_experiment": config["source_experiment"],
        "config": project_relative(config_path),
        "copied_config": project_relative(output_dir / "config.yaml"),
        "inputs": paths,
        "selected_rule": selected_rule,
        "conservative_rule": conservative_rule,
        "output_dir": project_relative(output_dir),
        "per_sample_with_clip_csv": project_relative(per_sample_with_clip_path),
        "policy_decisions_csv": project_relative(decisions_path),
        "policy_summary_csv": project_relative(summary_path),
        "policy_by_snr_csv": project_relative(by_snr_path),
        "metadata_json": project_relative(metadata_path),
        "report": project_relative(report_path),
        "galleries": gallery_manifest,
        "policies": policies,
        "materialized_policies": [str(item) for item in config["policy"]["materialize_policies"]],
        "clip_elapsed_seconds": clip_elapsed,
        "clip_num_unique_images": len(unique_clip_paths),
        "device": str(device),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "git_commit": git_commit(),
        "git_dirty_state": git_dirty_state(),
        "proxy_environment_present": proxy_environment_present(),
        "download_note": "No model or data download is required; CLIP checkpoint is loaded from local cache.",
    }
    save_json(metadata_path, metadata)
    report_path.write_text(make_report(summary_rows, metadata), encoding="utf-8")
    print(json.dumps({"output_dir": project_relative(output_dir), "report": project_relative(report_path)}, indent=2))


if __name__ == "__main__":
    main()
