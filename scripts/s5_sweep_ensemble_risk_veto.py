from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep receiver-side ensemble-risk veto rules on top of selected_risk_rule."
    )
    parser.add_argument("--config", default="configs/s5_ensemble_risk_veto_sweep_exp_s4_006.yaml")
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


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def parse_float(value: Any) -> float:
    if value in ("", None):
        return 0.0
    return float(value)


def snr_name(snr: float) -> str:
    if float(snr).is_integer():
        return f"snr_{int(snr):02d}db"
    return f"snr_{str(snr).replace('.', 'p')}db"


def safe_value(value: float | None) -> str:
    if value is None:
        return "none"
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def rate(flags: list[bool]) -> float:
    return float(sum(flags) / len(flags)) if flags else 0.0


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_selected_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    bool_keys = [
        "accept_refined",
        "baseline_accept_refined",
        "candidate_accept_refined",
        "new_accept_vs_top1",
        "vetoed_candidate_accept",
        "shadow_veto",
        "final_matches_original_top1",
        "accepted_repair",
        "missed_repair",
        "accepted_new_error",
        "protective_reject",
        "vetoed_candidate_repair",
        "vetoed_candidate_new_error",
        "top1_equal_accept_refined",
        "top1_equal_final_matches_original_top1",
    ]
    float_keys = [
        "snr_db",
        "final_psnr_db",
        "clip_sim_m0_refined",
        "m0_refined_top5_overlap",
        "m0_top1_rank_in_refined_top5",
        "refined_top1_rank_in_m0_top5",
        "m0_top1_margin",
        "refined_top1_margin",
        "m0_top1_prob",
        "refined_top1_prob",
        "refined_conf_gain_vs_m0",
        "top1_equal_final_psnr_db",
        "delta_final_psnr_vs_top1_equal_db",
    ]
    for row in rows:
        out = dict(row)
        for key in bool_keys:
            if key in out:
                out[key] = parse_bool(out[key])
        for key in float_keys:
            if key in out:
                out[key] = parse_float(out[key])
        output.append(out)
    return output


def normalize_vote_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        out["snr_db"] = float(out["snr_db"])
        out["selected_accept_refined"] = parse_bool(out["selected_accept_refined"])
        out["selected_new_accept_vs_alexnet_top1"] = parse_bool(out["selected_new_accept_vs_alexnet_top1"])
        out["selected_shadow_veto"] = parse_bool(out["selected_shadow_veto"])
        for key in [
            "classifier_count",
            "selected_accepted_new_error_vote_count",
            "selected_accepted_repair_vote_count",
            "selected_missed_repair_vote_count",
            "selected_protective_reject_vote_count",
        ]:
            out[key] = int(out[key])
        output.append(out)
    return output


def normalize_model_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        out["snr_db"] = float(out["snr_db"])
        for key in [
            "selected_accept_refined",
            "selected_new_accept_vs_alexnet_top1",
            "selected_shadow_veto",
            "m0_matches_original_top1",
            "refined_matches_original_top1",
            "refined_matches_m0_top1",
            "selected_final_matches_original_top1",
            "selected_accepted_repair",
            "selected_accepted_new_error",
            "selected_accepted_both_wrong",
            "selected_missed_repair",
            "selected_protective_reject",
        ]:
            if key in out:
                out[key] = parse_bool(out[key])
        output.append(out)
    return output


def sample_key(row: dict[str, Any]) -> tuple[str, float, str]:
    return (str(row["split"]), float(row["snr_db"]), str(row["sample"]))


def validate_inputs(config: dict[str, Any]) -> dict[str, str]:
    paths = {
        "selected_policy_per_sample_csv": resolve_project_path(
            config["inputs"]["selected_policy_per_sample_csv"]
        ),
        "ensemble_votes_csv": resolve_project_path(config["inputs"]["ensemble_votes_csv"]),
        "ensemble_per_model_csv": resolve_project_path(config["inputs"]["ensemble_per_model_csv"]),
        "source_risk_rule_config": resolve_project_path(config["inputs"]["source_risk_rule_config"]),
        "source_ensemble_audit_config": resolve_project_path(config["inputs"]["source_ensemble_audit_config"]),
        "checkpoint": resolve_project_path(config["inputs"]["checkpoint"]),
        "forbidden_checkpoint": resolve_project_path(config["inputs"]["forbidden_checkpoint"]),
    }
    for key, path in paths.items():
        if key == "forbidden_checkpoint":
            continue
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {key}: {path}")
    if paths["checkpoint"] == paths["forbidden_checkpoint"]:
        raise RuntimeError("Config points to forbidden latest.pt checkpoint.")
    return {key: project_relative(path) for key, path in paths.items()}


def validate_join(
    selected_rows: list[dict[str, Any]],
    vote_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
) -> None:
    selected_keys = {sample_key(row) for row in selected_rows}
    vote_keys = {sample_key(row) for row in vote_rows}
    model_keys = {sample_key(row) for row in model_rows}
    if selected_keys != vote_keys:
        raise RuntimeError("Selected rows and ensemble vote rows do not have identical sample keys.")
    if not model_keys.issubset(selected_keys):
        raise RuntimeError("Model rows contain unknown sample keys.")


def psnr(reference_path: str, candidate_path: str) -> float:
    ref = np.asarray(Image.open(resolve_project_path(reference_path)).convert("RGB"), dtype=np.float32) / 255.0
    cand = np.asarray(Image.open(resolve_project_path(candidate_path)).convert("RGB"), dtype=np.float32) / 255.0
    mse = float(np.mean((ref - cand) ** 2))
    if mse <= 0.0:
        return 99.0
    return float(10.0 * math.log10(1.0 / mse))


def add_psnr_columns(rows: list[dict[str, Any]]) -> None:
    cache: dict[tuple[str, str], float] = {}
    for row in rows:
        original = str(row["original"])
        m0 = str(row["m0_reconstruction"])
        refined = str(row["refined"])
        for key, path in [("m0_psnr_db", m0), ("refined_psnr_db", refined)]:
            cache_key = (original, path)
            if cache_key not in cache:
                cache[cache_key] = psnr(original, path)
            row[key] = cache[cache_key]


def veto_reason(
    row: dict[str, Any],
    new_accept_refined_margin_max: float | None,
    top1_refined_conf_gain_max: float | None,
    top1_m0_margin_min: float,
) -> str:
    if not bool(row["accept_refined"]):
        return ""
    if bool(row["new_accept_vs_top1"]):
        if new_accept_refined_margin_max is not None and float(row["refined_top1_margin"]) <= new_accept_refined_margin_max:
            return "new_accept_low_refined_margin"
        return ""
    if (
        top1_refined_conf_gain_max is not None
        and float(row["refined_conf_gain_vs_m0"]) <= top1_refined_conf_gain_max
        and float(row["m0_top1_margin"]) >= top1_m0_margin_min
    ):
        return "top1_equal_weak_refined_gain_high_m0_margin"
    return ""


def rule_name(new_margin: float | None, gain_max: float | None, m0_margin_min: float) -> str:
    return (
        "ensemble_veto_new_margin_le_"
        f"{safe_value(new_margin)}_top1_gain_le_{safe_value(gain_max)}_m0margin_ge_{safe_value(m0_margin_min)}"
    )


def build_candidate_rows(
    selected_rows: list[dict[str, Any]],
    vote_index: dict[tuple[str, float, str], dict[str, Any]],
    new_margin: float | None,
    gain_max: float | None,
    m0_margin_min: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    policy = rule_name(new_margin, gain_max, m0_margin_min)
    for row in selected_rows:
        vote = vote_index[sample_key(row)]
        reason = veto_reason(row, new_margin, gain_max, m0_margin_min)
        extra_veto = bool(reason)
        final_accept = bool(row["accept_refined"]) and not extra_veto
        final_psnr = float(row["refined_psnr_db"] if final_accept else row["m0_psnr_db"])
        out = dict(row)
        out.update(
            {
                "policy": policy,
                "extra_veto": extra_veto,
                "extra_veto_reason": reason,
                "final_accept_refined_after_extra_veto": final_accept,
                "final_output_kind_after_extra_veto": "accepted_refined" if final_accept else "fallback_m0",
                "final_psnr_after_extra_veto_db": final_psnr,
                "delta_psnr_vs_selected_db": final_psnr - float(row["final_psnr_db"]),
                "selected_accepted_new_error_vote_count": int(vote["selected_accepted_new_error_vote_count"]),
                "selected_accepted_repair_vote_count": int(vote["selected_accepted_repair_vote_count"]),
                "selected_missed_repair_vote_count": int(vote["selected_missed_repair_vote_count"]),
                "selected_protective_reject_vote_count": int(vote["selected_protective_reject_vote_count"]),
                "classifier_count": int(vote["classifier_count"]),
                "majority_new_error_under_selected": int(vote["selected_accepted_new_error_vote_count"])
                > int(vote["classifier_count"]) / 2,
                "any_new_error_under_selected": int(vote["selected_accepted_new_error_vote_count"]) >= 1,
                "majority_repair_under_selected": int(vote["selected_accepted_repair_vote_count"])
                > int(vote["classifier_count"]) / 2,
                "any_repair_under_selected": int(vote["selected_accepted_repair_vote_count"]) >= 1,
            }
        )
        rows.append(out)
    return rows


def model_final_match(row: dict[str, Any], final_accept: bool) -> bool:
    return bool(row["refined_matches_original_top1"] if final_accept else row["m0_matches_original_top1"])


def summarize_policy_rows(rows: list[dict[str, Any]], subset: str) -> dict[str, Any]:
    final_accept_flags = [bool(row["final_accept_refined_after_extra_veto"]) for row in rows]
    extra_veto_flags = [bool(row["extra_veto"]) for row in rows]
    majority_new_remaining = [
        bool(row["majority_new_error_under_selected"]) and not bool(row["extra_veto"]) for row in rows
    ]
    any_new_remaining = [bool(row["any_new_error_under_selected"]) and not bool(row["extra_veto"]) for row in rows]
    majority_repair_remaining = [
        bool(row["majority_repair_under_selected"]) and not bool(row["extra_veto"]) for row in rows
    ]
    any_repair_remaining = [bool(row["any_repair_under_selected"]) and not bool(row["extra_veto"]) for row in rows]
    majority_new_vetoed = [bool(row["majority_new_error_under_selected"]) and bool(row["extra_veto"]) for row in rows]
    any_new_vetoed = [bool(row["any_new_error_under_selected"]) and bool(row["extra_veto"]) for row in rows]
    majority_repair_vetoed = [bool(row["majority_repair_under_selected"]) and bool(row["extra_veto"]) for row in rows]
    any_repair_vetoed = [bool(row["any_repair_under_selected"]) and bool(row["extra_veto"]) for row in rows]
    return {
        "subset": subset,
        "num_images": len(rows),
        "accept_count": sum(final_accept_flags),
        "accept_rate": rate(final_accept_flags),
        "extra_veto_count": sum(extra_veto_flags),
        "extra_veto_rate": rate(extra_veto_flags),
        "remaining_majority_new_error_count": sum(majority_new_remaining),
        "remaining_any_new_error_count": sum(any_new_remaining),
        "remaining_majority_repair_count": sum(majority_repair_remaining),
        "remaining_any_repair_count": sum(any_repair_remaining),
        "vetoed_majority_new_error_count": sum(majority_new_vetoed),
        "vetoed_any_new_error_count": sum(any_new_vetoed),
        "vetoed_majority_repair_count": sum(majority_repair_vetoed),
        "vetoed_any_repair_count": sum(any_repair_vetoed),
        "final_psnr_db": mean([float(row["final_psnr_after_extra_veto_db"]) for row in rows]),
        "selected_psnr_db": mean([float(row["final_psnr_db"]) for row in rows]),
        "delta_psnr_vs_selected_db": mean([float(row["delta_psnr_vs_selected_db"]) for row in rows]),
    }


def summarize_model_rows(
    model_rows: list[dict[str, Any]],
    decision_index: dict[tuple[str, float, str], dict[str, Any]],
    subset: str,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for classifier in sorted({str(row["classifier"]) for row in model_rows}):
        cls_rows = [row for row in model_rows if str(row["classifier"]) == classifier]
        matches: list[bool] = []
        selected_matches: list[bool] = []
        repairs = 0
        new_errors = 0
        for row in cls_rows:
            decision = decision_index[sample_key(row)]
            final_accept = bool(decision["final_accept_refined_after_extra_veto"])
            final_match = model_final_match(row, final_accept)
            selected_match = bool(row["selected_final_matches_original_top1"])
            matches.append(final_match)
            selected_matches.append(selected_match)
            if final_accept and (not bool(row["m0_matches_original_top1"])) and bool(row["refined_matches_original_top1"]):
                repairs += 1
            if final_accept and bool(row["m0_matches_original_top1"]) and (not bool(row["refined_matches_original_top1"])):
                new_errors += 1
        summaries.append(
            {
                "subset": subset,
                "classifier": classifier,
                "num_images": len(cls_rows),
                "candidate_final_failure_rate": 1.0 - rate(matches),
                "selected_final_failure_rate": 1.0 - rate(selected_matches),
                "delta_failure_vs_selected": (1.0 - rate(matches)) - (1.0 - rate(selected_matches)),
                "candidate_accepted_repair_count": repairs,
                "candidate_accepted_new_error_count": new_errors,
            }
        )
    return summaries


def summarize_all(
    candidate_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    policy_rows: list[dict[str, Any]] = []
    model_summary_rows: list[dict[str, Any]] = []
    for split in ["validation", "heldout"]:
        split_rows = [row for row in candidate_rows if str(row["split"]) == split]
        policy_rows.append(summarize_policy_rows(split_rows, split))
        decision_index = {sample_key(row): row for row in split_rows}
        split_model_rows = [row for row in model_rows if str(row["split"]) == split]
        model_summary_rows.extend(summarize_model_rows(split_model_rows, decision_index, split))
        for snr in sorted({float(row["snr_db"]) for row in split_rows}):
            subset = [row for row in split_rows if float(row["snr_db"]) == snr]
            policy_rows.append(summarize_policy_rows(subset, f"{split}_{snr_name(snr)}"))
    return policy_rows, model_summary_rows


def candidate_score(summary_rows: list[dict[str, Any]], config: dict[str, Any]) -> tuple[Any, ...]:
    validation = next(row for row in summary_rows if row["subset"] == "validation")
    budget = int(config["risk_veto_search"]["majority_new_error_budget"])
    remaining_new = int(validation["remaining_majority_new_error_count"])
    budget_miss = max(0, remaining_new - budget)
    return (
        budget_miss,
        remaining_new,
        -int(validation["remaining_majority_repair_count"]),
        -int(validation["remaining_any_repair_count"]),
        int(validation["extra_veto_count"]),
        -float(validation["final_psnr_db"]),
    )


def scan_rules(
    selected_rows: list[dict[str, Any]],
    vote_index: dict[tuple[str, float, str], dict[str, Any]],
    model_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    best_rule: dict[str, Any] | None = None
    best_rows: list[dict[str, Any]] = []
    best_policy_summary: list[dict[str, Any]] = []
    best_model_summary: list[dict[str, Any]] = []
    search_cfg = config["risk_veto_search"]
    for new_margin in search_cfg["new_accept_refined_margin_max_grid"]:
        for gain_max in search_cfg["top1_refined_conf_gain_max_grid"]:
            for m0_margin_min in search_cfg["top1_m0_margin_min_grid"]:
                candidate_rows = build_candidate_rows(
                    selected_rows,
                    vote_index,
                    None if new_margin is None else float(new_margin),
                    None if gain_max is None else float(gain_max),
                    float(m0_margin_min),
                )
                policy_summary, model_summary = summarize_all(candidate_rows, model_rows)
                validation = next(row for row in policy_summary if row["subset"] == "validation")
                heldout = next(row for row in policy_summary if row["subset"] == "heldout")
                row = {
                    "rule_name": rule_name(
                        None if new_margin is None else float(new_margin),
                        None if gain_max is None else float(gain_max),
                        float(m0_margin_min),
                    ),
                    "new_accept_refined_margin_max": "" if new_margin is None else float(new_margin),
                    "top1_refined_conf_gain_max": "" if gain_max is None else float(gain_max),
                    "top1_m0_margin_min": float(m0_margin_min),
                }
                for prefix, summary in [("validation", validation), ("heldout", heldout)]:
                    for key, value in summary.items():
                        if key != "subset":
                            row[f"{prefix}_{key}"] = value
                candidates.append(row)
                score = candidate_score(policy_summary, config)
                if best_rule is None or score < best_rule["score"]:
                    best_rule = {
                        "rule_name": row["rule_name"],
                        "new_accept_refined_margin_max": row["new_accept_refined_margin_max"],
                        "top1_refined_conf_gain_max": row["top1_refined_conf_gain_max"],
                        "top1_m0_margin_min": row["top1_m0_margin_min"],
                        "score": score,
                    }
                    best_rows = candidate_rows
                    best_policy_summary = policy_summary
                    best_model_summary = model_summary
    if best_rule is None:
        raise RuntimeError("No risk veto candidates were generated.")
    best_rule["score"] = list(best_rule["score"])
    return candidates, best_rule, best_rows, best_policy_summary + best_model_summary


def serialize_value(value: Any) -> Any:
    if isinstance(value, bool):
        return bool_text(value)
    if isinstance(value, (dict, list, tuple)):
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


def load_grid_image(path: str, size: int) -> Image.Image:
    return Image.open(resolve_project_path(path)).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)


def make_grid(rows: list[dict[str, Any]], output_path: Path, count: int) -> None:
    if not rows:
        return
    rows = rows[:count]
    tile = 160
    label_height = 50
    cols = 4
    canvas = Image.new("RGB", (tile * cols, (tile + label_height) * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for row_index, row in enumerate(rows):
        y = row_index * (tile + label_height)
        images = [
            ("original", row["original"]),
            ("m0", row["m0_reconstruction"]),
            ("refined", row["refined"]),
            ("candidate", row["refined"] if bool(row["final_accept_refined_after_extra_veto"]) else row["m0_reconstruction"]),
        ]
        for col, (label, path) in enumerate(images):
            x = col * tile
            canvas.paste(load_grid_image(path, tile), (x, y + label_height))
            draw.text((x + 4, y + 4), label, fill=(0, 0, 0), font=font)
        detail = (
            f"{row['split']} {row['sample']} {snr_name(float(row['snr_db']))} "
            f"veto={bool_text(bool(row['extra_veto']))} reason={row['extra_veto_reason']}"
        )
        votes = (
            f"maj_new={bool_text(bool(row['majority_new_error_under_selected']))} "
            f"any_new={bool_text(bool(row['any_new_error_under_selected']))} "
            f"maj_rep={bool_text(bool(row['majority_repair_under_selected']))}"
        )
        draw.text((4, y + 18), detail[:120], fill=(0, 0, 0), font=font)
        draw.text((4, y + 34), votes[:120], fill=(0, 0, 0), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def write_galleries(rows: list[dict[str, Any]], config: dict[str, Any], output_dir: Path) -> dict[str, str]:
    count = int(config["evaluation"]["gallery_rows"])
    gallery_dir = output_dir / "galleries"
    groups = {
        "vetoed_majority_new_errors": [
            row for row in rows if bool(row["extra_veto"]) and bool(row["majority_new_error_under_selected"])
        ],
        "remaining_any_new_errors": [
            row for row in rows if (not bool(row["extra_veto"])) and bool(row["any_new_error_under_selected"])
        ],
        "vetoed_any_repairs": [
            row for row in rows if bool(row["extra_veto"]) and bool(row["any_repair_under_selected"])
        ],
        "remaining_majority_repairs": [
            row for row in rows if (not bool(row["extra_veto"])) and bool(row["majority_repair_under_selected"])
        ],
    }
    manifest: dict[str, str] = {}
    for name, subset in groups.items():
        subset = sorted(
            subset,
            key=lambda row: (
                str(row["split"]),
                float(row["snr_db"]),
                str(row["sample"]),
            ),
        )
        path = gallery_dir / f"{name}.png"
        make_grid(subset, path, count)
        if subset:
            manifest[name] = project_relative(path)
    return manifest


def get_project_version() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return "N/A"


def get_git_dirty_state() -> str:
    try:
        output = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return "dirty" if output else "clean"
    except Exception:  # noqa: BLE001
        return "unknown"


def make_report(
    best_rule: dict[str, Any],
    policy_summary_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> str:
    policy_rows = [row for row in policy_summary_rows if "classifier" not in row]
    validation = next(row for row in policy_rows if row["subset"] == "validation")
    heldout = next(row for row in policy_rows if row["subset"] == "heldout")
    model_rows = [row for row in policy_summary_rows if "classifier" in row and row["subset"] in {"validation", "heldout"}]
    lines = [
        "# EXP-S4-006 Ensemble-Risk Veto Sweep",
        "",
        "This derived analysis searches a transparent receiver-side veto on top of `selected_risk_rule`.",
        "",
        "The search labels come from the offline classifier ensemble audit, but the veto features are receiver-side AlexNet/CLIP/top-k features already present in the selected-risk-rule CSV.",
        "",
        "## Selected Rule",
        "",
        f"- Rule name: `{best_rule['rule_name']}`",
        f"- New-accept refined margin max: `{best_rule['new_accept_refined_margin_max']}`",
        f"- Top-1-equal refined confidence gain max: `{best_rule['top1_refined_conf_gain_max']}`",
        f"- Top-1-equal M0 margin min: `{best_rule['top1_m0_margin_min']}`",
        "",
        "## Vote Summary",
        "",
        "| Split | Extra Veto | Remaining Majority New Error | Remaining Any New Error | Remaining Majority Repair | Remaining Any Repair | Delta PSNR vs Selected |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in [validation, heldout]:
        lines.append(
            "| "
            f"{row['subset']} | {int(row['extra_veto_count'])} | "
            f"{int(row['remaining_majority_new_error_count'])} | "
            f"{int(row['remaining_any_new_error_count'])} | "
            f"{int(row['remaining_majority_repair_count'])} | "
            f"{int(row['remaining_any_repair_count'])} | "
            f"{float(row['delta_psnr_vs_selected_db']):+.4f} dB |"
        )
    lines.extend(
        [
            "",
            "## Per-Classifier Final Failure",
            "",
            "| Split | Classifier | Candidate Failure | Delta vs Selected | Repair | New Error |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in model_rows:
        lines.append(
            "| "
            f"{row['subset']} | {row['classifier']} | "
            f"{float(row['candidate_final_failure_rate']):.4f} | "
            f"{float(row['delta_failure_vs_selected']):+.4f} | "
            f"{int(row['candidate_accepted_repair_count'])} | "
            f"{int(row['candidate_accepted_new_error_count'])} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Rule candidates: `{metadata['rule_candidates_csv']}`",
            f"- Policy decisions: `{metadata['policy_decisions_csv']}`",
            f"- Policy summary: `{metadata['policy_summary_csv']}`",
            f"- Selected rule: `{metadata['selected_rule_json']}`",
            f"- Galleries: `{metadata['gallery_dir']}`",
            "",
            "## Caveat",
            "",
            "This is still pseudo-label validation tuning. The result is useful for narrowing the gate, not for declaring final M3.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    manifest = validate_inputs(config)
    selected_rows = normalize_selected_rows(read_csv(resolve_project_path(config["inputs"]["selected_policy_per_sample_csv"])))
    vote_rows = normalize_vote_rows(read_csv(resolve_project_path(config["inputs"]["ensemble_votes_csv"])))
    model_rows = normalize_model_rows(read_csv(resolve_project_path(config["inputs"]["ensemble_per_model_csv"])))
    validate_join(selected_rows, vote_rows, model_rows)
    dry_run_payload = {
        "status": "ok",
        "num_selected_rows": len(selected_rows),
        "num_vote_rows": len(vote_rows),
        "num_model_rows": len(model_rows),
        "splits": {
            split: sum(1 for row in selected_rows if str(row["split"]) == split)
            for split in sorted({str(row["split"]) for row in selected_rows})
        },
        "candidate_grid_size": len(config["risk_veto_search"]["new_accept_refined_margin_max_grid"])
        * len(config["risk_veto_search"]["top1_refined_conf_gain_max_grid"])
        * len(config["risk_veto_search"]["top1_m0_margin_min_grid"]),
        "manifest": manifest,
    }
    if args.dry_run:
        print(json.dumps(dry_run_payload, indent=2, ensure_ascii=False))
        return

    output_dir = resolve_project_path(args.output_dir or config["outputs"]["output_dir"])
    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory already exists and is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")

    add_psnr_columns(selected_rows)
    vote_index = {sample_key(row): row for row in vote_rows}
    rule_candidates, best_rule, best_rows, policy_summary_rows = scan_rules(
        selected_rows, vote_index, model_rows, config
    )
    galleries = write_galleries(best_rows, config, output_dir)

    rule_candidates_csv = output_dir / "rule_candidates.csv"
    policy_decisions_csv = output_dir / "policy_decisions.csv"
    policy_summary_csv = output_dir / "policy_summary.csv"
    selected_rule_json = output_dir / "selected_rule.json"
    metadata_json = output_dir / "metadata.json"
    report_md = output_dir / "REPORT.md"
    write_csv(rule_candidates_csv, rule_candidates)
    write_csv(policy_decisions_csv, best_rows)
    write_csv(policy_summary_csv, policy_summary_rows)
    save_json(selected_rule_json, best_rule)
    metadata = {
        "project_version": get_project_version(),
        "git_dirty_state": get_git_dirty_state(),
        "config": project_relative(config_path),
        "output_dir": project_relative(output_dir),
        "rule_candidates_csv": project_relative(rule_candidates_csv),
        "policy_decisions_csv": project_relative(policy_decisions_csv),
        "policy_summary_csv": project_relative(policy_summary_csv),
        "selected_rule_json": project_relative(selected_rule_json),
        "metadata_json": project_relative(metadata_json),
        "report_md": project_relative(report_md),
        "gallery_dir": project_relative(output_dir / "galleries"),
        "galleries": galleries,
        "source_inputs": manifest,
        "dry_run_payload": dry_run_payload,
        "policy": config["policy"],
        "python_version": platform.python_version(),
        "proxy_environment_present": sorted(key for key in os.environ if "proxy" in key.lower()),
        "download_note": "No model or data download is required; this script only reads existing CSVs and PNGs.",
    }
    save_json(metadata_json, metadata)
    report_md.write_text(make_report(best_rule, policy_summary_rows, metadata), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": project_relative(output_dir),
                "selected_rule": best_rule["rule_name"],
                "rule_candidates": len(rule_candidates),
                "report_md": project_relative(report_md),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
