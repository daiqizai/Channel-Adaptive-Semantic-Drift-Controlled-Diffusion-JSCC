from __future__ import annotations

import argparse
import ast
import csv
import itertools
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import yaml
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep receiver-side risk rules for EXP-S4-006 confidence-gain gate."
    )
    parser.add_argument("--config", default="configs/s5_conf_gain_risk_rule_sweep_exp_s4_006.yaml")
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


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


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


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_list_field(value: str) -> list[str]:
    value = str(value).strip()
    if not value:
        return []
    if "|" in value:
        return [item.strip() for item in value.split("|") if item.strip()]
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except Exception:
        pass
    return [item.strip() for item in re.split(r"[\[\],|]+", value) if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in parse_list_field(value)]


def normalize_clip_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        output.append(
            {
                "split": row["split"],
                "snr_db": float(row["snr_db"]),
                "sample": row["sample"],
                "original": row["original"],
                "m0_reconstruction": row["m0_reconstruction"],
                "refined": row["refined"],
                "m0_top1_label": row.get("m0_top1_label", ""),
                "refined_top1_label": row.get("refined_top1_label", ""),
                "m0_top1_prob": float(row["m0_top1_prob"]),
                "refined_top1_prob": float(row["refined_top1_prob"]),
                "refined_conf_gain_vs_m0": float(row["refined_conf_gain_vs_m0"]),
                "m0_matches_original_top1": parse_bool(row["m0_matches_original_top1"]),
                "refined_matches_original_top1": parse_bool(row["refined_matches_original_top1"]),
                "baseline_accept_refined": parse_bool(row["baseline_accept_refined"]),
                "candidate_accept_refined": parse_bool(row["candidate_accept_refined"]),
                "newly_accepted_by_candidate": parse_bool(row["newly_accepted_by_candidate"]),
                "candidate_accepted_repair": parse_bool(row["candidate_accepted_repair"]),
                "candidate_accepted_new_error": parse_bool(row["candidate_accepted_new_error"]),
                "m0_psnr_db": float(row["m0_psnr_db"]),
                "refined_psnr_db": float(row["refined_psnr_db"]),
                "clip_sim_m0_refined": float(row["clip_sim_m0_refined"]),
            }
        )
    return output


def load_topk_map(validation_csv: Path, heldout_csv: Path) -> dict[tuple[str, float, str], dict[str, str]]:
    mapping: dict[tuple[str, float, str], dict[str, str]] = {}
    for split, path in [("validation", validation_csv), ("heldout", heldout_csv)]:
        for row in read_csv(path):
            mapping[(split, float(row["snr_db"]), row["sample"])] = row
    return mapping


def top1_rank_in_topk(top1: str, topk: list[str]) -> int:
    if top1 in topk:
        return topk.index(top1) + 1
    return 99


def enrich_with_receiver_features(
    rows: list[dict[str, Any]],
    topk_map: dict[tuple[str, float, str], dict[str, str]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        key = (row["split"], float(row["snr_db"]), row["sample"])
        if key not in topk_map:
            raise KeyError(f"Missing top-k predictions for {key}")
        topk = topk_map[key]
        m0_indices = parse_list_field(topk["m0_top_indices"])
        refined_indices = parse_list_field(topk["refined_top_indices"])
        m0_probs = parse_float_list(topk["m0_top_probs"])
        refined_probs = parse_float_list(topk["refined_top_probs"])
        enriched = dict(row)
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


def check_inputs(config: dict[str, Any]) -> dict[str, str]:
    paths = {
        "per_sample_with_clip_csv": resolve_project_path(config["inputs"]["per_sample_with_clip_csv"]),
        "validation_topk_csv": resolve_project_path(config["inputs"]["validation_topk_csv"]),
        "heldout_topk_csv": resolve_project_path(config["inputs"]["heldout_topk_csv"]),
    }
    for key, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Required input missing: {key}: {path}")
    return {key: project_relative(path) for key, path in paths.items()}


def check_image_paths(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        for key in ["original", "m0_reconstruction", "refined"]:
            path = resolve_project_path(row[key])
            if not path.exists():
                raise FileNotFoundError(f"Image path from CSV not found: {path}")


def rule_shadow_veto(row: dict[str, Any], rule: dict[str, Any]) -> bool:
    return (
        int(row["m0_top1_rank_in_refined_top5"]) <= int(rule["shadow_m0_rank_max"])
        and float(row["m0_top1_margin"]) <= float(rule["shadow_m0_margin_max"])
        and float(row["refined_top1_margin"]) >= float(rule["shadow_ref_margin_min"])
        and float(row["clip_sim_m0_refined"]) < float(rule["shadow_clip_max"])
    )


def rule_accepts_candidate(row: dict[str, Any], rule: dict[str, Any]) -> bool:
    if float(row["clip_sim_m0_refined"]) < float(rule["clip_min"]):
        return False
    if int(row["m0_refined_top5_overlap"]) < int(rule["top5_overlap_min"]):
        return False
    if rule_shadow_veto(row, rule):
        return False
    return True


def accept_for_policy(row: dict[str, Any], policy_name: str, rule: dict[str, Any] | None = None) -> bool:
    if policy_name == "top1_equal":
        return bool(row["baseline_accept_refined"])
    if policy_name == "raw_conf_gain":
        return bool(row["candidate_accept_refined"])
    if policy_name == "fixed_clip_ge_0p98":
        return bool(row["baseline_accept_refined"]) or (
            bool(row["candidate_accept_refined"]) and float(row["clip_sim_m0_refined"]) >= 0.98
        )
    if policy_name == "selected_risk_rule":
        if rule is None:
            raise ValueError("selected_risk_rule requires a rule")
        return bool(row["baseline_accept_refined"]) or (
            bool(row["candidate_accept_refined"]) and rule_accepts_candidate(row, rule)
        )
    raise ValueError(f"Unknown policy: {policy_name}")


def evaluate_rows(
    rows: list[dict[str, Any]],
    split: str,
    policy_name: str,
    rule: dict[str, Any] | None = None,
    snr: float | None = None,
) -> dict[str, Any]:
    subset = [
        row
        for row in rows
        if row["split"] == split and (snr is None or float(row["snr_db"]) == float(snr))
    ]
    total = len(subset)
    accepted = 0
    final_correct = 0
    m0_correct = 0
    refined_correct = 0
    accepted_repair = 0
    missed_repair = 0
    accepted_new_error = 0
    protective_reject = 0
    false_accept = 0
    false_reject = 0
    new_accept = 0
    vetoed_candidate_accept = 0
    vetoed_candidate_repair = 0
    vetoed_candidate_new_error = 0
    shadow_veto_count = 0
    final_psnrs: list[float] = []
    m0_psnrs: list[float] = []
    refined_psnrs: list[float] = []
    new_accept_clip: list[float] = []

    for row in subset:
        accept = accept_for_policy(row, policy_name, rule)
        baseline_accept = bool(row["baseline_accept_refined"])
        candidate_accept = bool(row["candidate_accept_refined"])
        shadow_veto = bool(rule and candidate_accept and (not baseline_accept) and rule_shadow_veto(row, rule))
        m0_ok = bool(row["m0_matches_original_top1"])
        refined_ok = bool(row["refined_matches_original_top1"])
        final_ok = refined_ok if accept else m0_ok
        final_psnr = float(row["refined_psnr_db"] if accept else row["m0_psnr_db"])

        accepted += int(accept)
        final_correct += int(final_ok)
        m0_correct += int(m0_ok)
        refined_correct += int(refined_ok)
        accepted_repair += int(accept and (not m0_ok) and refined_ok)
        missed_repair += int((not accept) and (not m0_ok) and refined_ok)
        accepted_new_error += int(accept and m0_ok and (not refined_ok))
        protective_reject += int((not accept) and m0_ok and (not refined_ok))
        false_accept += int(accept and not refined_ok)
        false_reject += int((not accept) and refined_ok)
        new_accept += int(accept and not baseline_accept)
        vetoed_candidate_accept += int(candidate_accept and not accept)
        vetoed_candidate_repair += int((candidate_accept and not accept) and (not m0_ok) and refined_ok)
        vetoed_candidate_new_error += int((candidate_accept and not accept) and m0_ok and not refined_ok)
        shadow_veto_count += int(shadow_veto)
        final_psnrs.append(final_psnr)
        m0_psnrs.append(float(row["m0_psnr_db"]))
        refined_psnrs.append(float(row["refined_psnr_db"]))
        if accept and not baseline_accept:
            new_accept_clip.append(float(row["clip_sim_m0_refined"]))

    return {
        "split": split,
        "policy": policy_name,
        "snr_db": "all" if snr is None else float(snr),
        "num_images": total,
        "accept_count": accepted,
        "accept_rate": rate(accepted, total),
        "reject_count": total - accepted,
        "reject_rate": rate(total - accepted, total),
        "new_accept_vs_top1_count": new_accept,
        "vetoed_candidate_accept_count": vetoed_candidate_accept,
        "vetoed_candidate_repair_count": vetoed_candidate_repair,
        "vetoed_candidate_new_error_count": vetoed_candidate_new_error,
        "shadow_veto_count": shadow_veto_count,
        "m0_failure_rate": 1.0 - rate(m0_correct, total),
        "refined_failure_rate": 1.0 - rate(refined_correct, total),
        "final_failure_rate": 1.0 - rate(final_correct, total),
        "false_accept_count": false_accept,
        "false_accept_rate": rate(false_accept, total),
        "false_reject_count": false_reject,
        "false_reject_rate": rate(false_reject, total),
        "accepted_repair_count": accepted_repair,
        "accepted_repair_rate": rate(accepted_repair, total),
        "missed_repair_count": missed_repair,
        "missed_repair_rate": rate(missed_repair, total),
        "protective_reject_count": protective_reject,
        "protective_reject_rate": rate(protective_reject, total),
        "accepted_new_error_count": accepted_new_error,
        "accepted_new_error_rate": rate(accepted_new_error, total),
        "m0_psnr_db": mean(m0_psnrs),
        "refined_psnr_db": mean(refined_psnrs),
        "final_psnr_db": mean(final_psnrs),
        "final_delta_psnr_vs_m0_db": mean(final_psnrs) - mean(m0_psnrs),
        "new_accept_clip_sim_m0_refined_mean": mean(new_accept_clip),
        "new_accept_clip_sim_m0_refined_min": min(new_accept_clip) if new_accept_clip else "",
    }


def add_policy_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed = {(str(row["split"]), str(row["snr_db"]), str(row["policy"])): row for row in rows}
    output: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        for baseline_name in ["top1_equal", "raw_conf_gain", "fixed_clip_ge_0p98"]:
            baseline = keyed.get((str(row["split"]), str(row["snr_db"]), baseline_name))
            if not baseline:
                continue
            suffix = baseline_name
            enriched[f"delta_final_failure_vs_{suffix}"] = (
                float(row["final_failure_rate"]) - float(baseline["final_failure_rate"])
            )
            enriched[f"delta_final_psnr_vs_{suffix}_db"] = (
                float(row["final_psnr_db"]) - float(baseline["final_psnr_db"])
            )
            enriched[f"delta_accepted_repair_vs_{suffix}"] = (
                int(row["accepted_repair_count"]) - int(baseline["accepted_repair_count"])
            )
            enriched[f"delta_accepted_new_error_vs_{suffix}"] = (
                int(row["accepted_new_error_count"]) - int(baseline["accepted_new_error_count"])
            )
        output.append(enriched)
    return output


def rule_to_name(rule: dict[str, Any]) -> str:
    shadow_clip = "no_clip_escape" if float(rule["shadow_clip_max"]) > 1.0 else f"shadowclip_lt_{rule['shadow_clip_max']:g}"
    return (
        "clip_ge_{clip_min:g}__overlap_ge_{top5_overlap_min}__shadow_rank_le_{shadow_m0_rank_max}"
        "__m0margin_le_{shadow_m0_margin_max:g}__refmargin_ge_{shadow_ref_margin_min:g}"
        f"__{shadow_clip}"
    ).format(**rule).replace(".", "p")


def make_rule_from_values(
    clip_min: float,
    top5_overlap_min: int,
    shadow_clip_max: float,
    shadow_m0_rank_max: int,
    shadow_m0_margin_max: float,
    shadow_ref_margin_min: float,
) -> dict[str, Any]:
    return {
        "clip_min": float(clip_min),
        "top5_overlap_min": int(top5_overlap_min),
        "shadow_clip_max": float(shadow_clip_max),
        "shadow_m0_rank_max": int(shadow_m0_rank_max),
        "shadow_m0_margin_max": float(shadow_m0_margin_max),
        "shadow_ref_margin_min": float(shadow_ref_margin_min),
    }


def rule_sort_key(item: tuple[dict[str, Any], dict[str, Any]]) -> tuple[Any, ...]:
    metrics, rule = item
    # The last terms are a predeclared conservative tie-breaker: for equal validation
    # utility, prefer a broader shadow veto rather than a brittle narrow exception.
    return (
        int(metrics["accepted_repair_count"]),
        float(metrics["final_psnr_db"]),
        -float(metrics["final_failure_rate"]),
        -int(metrics["accepted_new_error_count"]),
        float(rule["shadow_clip_max"]),
        int(rule["shadow_m0_rank_max"]),
        float(rule["shadow_m0_margin_max"]),
        -float(rule["shadow_ref_margin_min"]),
        float(rule["clip_min"]),
        int(rule["top5_overlap_min"]),
    )


def sweep_rules(rows: list[dict[str, Any]], config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    search = config["risk_rule_search"]
    split = search["calibration_split"]
    budget = int(search["accepted_new_error_budget"])
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    candidate_rows: list[dict[str, Any]] = []

    grids = [
        search["clip_min_grid"],
        search["top5_overlap_min_grid"],
        search["shadow_clip_max_grid"],
        search["shadow_m0_rank_max_grid"],
        search["shadow_m0_margin_max_grid"],
        search["shadow_ref_margin_min_grid"],
    ]
    for values in itertools.product(*grids):
        rule = make_rule_from_values(*values)
        metrics = evaluate_rows(rows, split, "selected_risk_rule", rule)
        row = dict(metrics)
        row.update(rule)
        row["rule_name"] = rule_to_name(rule)
        candidate_rows.append(row)
        if int(metrics["accepted_new_error_count"]) <= budget:
            candidates.append((metrics, rule))

    if not candidates:
        raise RuntimeError("No feasible validation rule found under accepted-new-error budget")

    selected_metrics, selected_rule = sorted(candidates, key=rule_sort_key, reverse=True)[0]
    selected_name = rule_to_name(selected_rule)
    for row in candidate_rows:
        row["selected"] = row["rule_name"] == selected_name
    return selected_rule, candidate_rows


def make_decisions(
    rows: list[dict[str, Any]],
    policies: list[str],
    selected_rule: dict[str, Any],
) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for row in rows:
        for policy_name in policies:
            rule = selected_rule if policy_name == "selected_risk_rule" else None
            accept = accept_for_policy(row, policy_name, rule)
            baseline_accept = bool(row["baseline_accept_refined"])
            candidate_accept = bool(row["candidate_accept_refined"])
            m0_ok = bool(row["m0_matches_original_top1"])
            refined_ok = bool(row["refined_matches_original_top1"])
            shadow_veto = bool(rule and candidate_accept and (not baseline_accept) and rule_shadow_veto(row, rule))
            decisions.append(
                {
                    "split": row["split"],
                    "policy": policy_name,
                    "snr_db": row["snr_db"],
                    "sample": row["sample"],
                    "accept_refined": accept,
                    "baseline_accept_refined": baseline_accept,
                    "candidate_accept_refined": candidate_accept,
                    "new_accept_vs_top1": accept and not baseline_accept,
                    "vetoed_candidate_accept": candidate_accept and not accept,
                    "shadow_veto": shadow_veto,
                    "final_matches_original_top1": refined_ok if accept else m0_ok,
                    "accepted_repair": accept and (not m0_ok) and refined_ok,
                    "missed_repair": (not accept) and (not m0_ok) and refined_ok,
                    "accepted_new_error": accept and m0_ok and (not refined_ok),
                    "protective_reject": (not accept) and m0_ok and (not refined_ok),
                    "vetoed_candidate_repair": candidate_accept and (not accept) and (not m0_ok) and refined_ok,
                    "vetoed_candidate_new_error": candidate_accept and (not accept) and m0_ok and (not refined_ok),
                    "final_psnr_db": row["refined_psnr_db"] if accept else row["m0_psnr_db"],
                    "clip_sim_m0_refined": row["clip_sim_m0_refined"],
                    "m0_refined_top5_overlap": row["m0_refined_top5_overlap"],
                    "m0_top1_rank_in_refined_top5": row["m0_top1_rank_in_refined_top5"],
                    "refined_top1_rank_in_m0_top5": row["refined_top1_rank_in_m0_top5"],
                    "m0_top1_margin": row["m0_top1_margin"],
                    "refined_top1_margin": row["refined_top1_margin"],
                    "m0_top1_label": row["m0_top1_label"],
                    "refined_top1_label": row["refined_top1_label"],
                    "m0_top1_prob": row["m0_top1_prob"],
                    "refined_top1_prob": row["refined_top1_prob"],
                    "refined_conf_gain_vs_m0": row["refined_conf_gain_vs_m0"],
                    "original": row["original"],
                    "m0_reconstruction": row["m0_reconstruction"],
                    "refined": row["refined"],
                }
            )
    return decisions


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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_value(row.get(key, "")) for key in fieldnames})


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_gallery_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB").resize((192, 192), Image.Resampling.BICUBIC)


def make_quad(row: dict[str, Any], output_path: Path) -> None:
    final_path = row["refined"] if parse_bool(row["accept_refined"]) else row["m0_reconstruction"]
    specs = [
        ("original", resolve_project_path(row["original"])),
        ("m0", resolve_project_path(row["m0_reconstruction"])),
        ("refined", resolve_project_path(row["refined"])),
        ("final", resolve_project_path(final_path)),
    ]
    images = [load_gallery_image(path) for _label, path in specs]
    label_height = 58
    canvas = Image.new("RGB", (192 * len(images), 192 + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for idx, ((label, _path), image) in enumerate(zip(specs, images)):
        x = idx * 192
        canvas.paste(image, (x, label_height))
        draw.text((x + 4, 4), label, fill=(0, 0, 0), font=font)
    detail = (
        f"{row['split']} {row['policy']} {float(row['snr_db']):g}dB {row['sample']} "
        f"clip={float(row['clip_sim_m0_refined']):.4f} ov={row['m0_refined_top5_overlap']} "
        f"m0rank={row['m0_top1_rank_in_refined_top5']} "
        f"m0mar={float(row['m0_top1_margin']):.3f} refmar={float(row['refined_top1_margin']):.3f}"
    )
    draw.text((4, 24), detail[:150], fill=(0, 0, 0), font=font)
    label_detail = f"{row['m0_top1_label']} -> {row['refined_top1_label']}"
    draw.text((4, 42), label_detail[:150], fill=(0, 0, 0), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def make_sheet(rows: list[dict[str, Any]], output_path: Path, title: str) -> None:
    if not rows:
        return
    tmp_dir = output_path.parent / "_tmp_quads"
    images: list[Image.Image] = []
    for idx, row in enumerate(rows):
        quad_path = tmp_dir / f"{idx:03d}.png"
        make_quad(row, quad_path)
        images.append(Image.open(quad_path).convert("RGB"))
    header_height = 28
    width = max(image.width for image in images)
    height = header_height + sum(image.height for image in images)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 8), title[:160], fill=(0, 0, 0), font=ImageFont.load_default())
    y = header_height
    for image in images:
        sheet.paste(image, (0, y))
        y += image.height
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def write_galleries(decisions: list[dict[str, Any]], output_dir: Path, gallery_rows: int) -> dict[str, str]:
    manifest: dict[str, str] = {}
    selected = [row for row in decisions if row["policy"] == "selected_risk_rule"]
    for split in ["validation", "heldout"]:
        split_rows = [row for row in selected if row["split"] == split]
        groups: list[tuple[str, Callable[[dict[str, Any]], bool], Callable[[dict[str, Any]], Any]]] = [
            ("accepted_new_errors", lambda row: parse_bool(row["accepted_new_error"]), lambda row: row["clip_sim_m0_refined"]),
            ("accepted_repairs", lambda row: parse_bool(row["accepted_repair"]), lambda row: row["clip_sim_m0_refined"]),
            (
                "vetoed_candidate_new_errors",
                lambda row: parse_bool(row["vetoed_candidate_new_error"]),
                lambda row: row["clip_sim_m0_refined"],
            ),
            (
                "vetoed_candidate_repairs",
                lambda row: parse_bool(row["vetoed_candidate_repair"]),
                lambda row: row["clip_sim_m0_refined"],
            ),
            ("shadow_vetoes", lambda row: parse_bool(row["shadow_veto"]), lambda row: row["clip_sim_m0_refined"]),
        ]
        for name, predicate, sort_key in groups:
            rows_subset = sorted([row for row in split_rows if predicate(row)], key=sort_key)[:gallery_rows]
            path = output_dir / "galleries" / "selected_risk_rule" / f"{split}_{name}.png"
            make_sheet(rows_subset, path, f"selected_risk_rule {split} {name}")
            if path.exists():
                manifest[f"selected_risk_rule_{split}_{name}"] = project_relative(path)
    return manifest


def make_report(
    summary_rows: list[dict[str, Any]],
    selected_rule: dict[str, Any],
    metadata: dict[str, Any],
) -> str:
    global_rows = [row for row in summary_rows if str(row["snr_db"]) == "all"]
    keyed = {(row["split"], row["policy"]): row for row in global_rows}
    policies = ["top1_equal", "raw_conf_gain", "fixed_clip_ge_0p98", "selected_risk_rule"]
    overlap_line = (
        "- no minimum top-5 overlap requirement"
        if int(selected_rule["top5_overlap_min"]) <= 0
        else f"- `m0_refined_top5_overlap >= {selected_rule['top5_overlap_min']}`"
    )
    shadow_clip_line = (
        "- shadow veto has no CLIP escape once the rank/margin pattern matches"
        if float(selected_rule["shadow_clip_max"]) > 1.0
        else f"- shadow veto if `clip_sim_m0_refined < {selected_rule['shadow_clip_max']}`"
    )
    lines = [
        "# EXP-S4-006 Confidence-Gain Risk Rule Sweep",
        "",
        "This derived analysis searches transparent receiver-side risk rules on validation only, then evaluates the selected rule on held-out rows.",
        "",
        "## Selected Rule",
        "",
        "Accept top-1 agreement outputs as before. For new confidence-gain accepts, require the global pass conditions and reject a shadow-risk pattern:",
        "",
        f"- `clip_sim_m0_refined >= {selected_rule['clip_min']}`",
        overlap_line,
        f"- shadow veto if `m0_top1_rank_in_refined_top5 <= {selected_rule['shadow_m0_rank_max']}`",
        f"- shadow veto if `m0_top1_margin <= {selected_rule['shadow_m0_margin_max']}`",
        f"- shadow veto if `refined_top1_margin >= {selected_rule['shadow_ref_margin_min']}`",
        shadow_clip_line,
        "",
        "The intuition is that a refined prediction is risky when the M0 top-1 label is still a close runner-up after refinement, M0 itself had a weak top-1 margin, and the refined classifier becomes more decisive. The selected validation rule treats that shadow pattern as risky regardless of CLIP escape.",
        "",
        "## Global Results",
        "",
        "| Split | Policy | Final Failure | Delta Failure vs Top1 | Final PSNR | Delta PSNR vs Top1 | Repair | New Error | Vetoed Candidate New Error |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ["validation", "heldout"]:
        for policy in policies:
            row = keyed[(split, policy)]
            lines.append(
                "| {split} | {policy} | {fail:.4f} | {dfail:+.4f} | {psnr:.4f} | {dpsnr:+.4f} | {repair} | {new_error} | {veto_new} |".format(
                    split=split,
                    policy=policy,
                    fail=float(row["final_failure_rate"]),
                    dfail=float(row.get("delta_final_failure_vs_top1_equal", 0.0)),
                    psnr=float(row["final_psnr_db"]),
                    dpsnr=float(row.get("delta_final_psnr_vs_top1_equal_db", 0.0)),
                    repair=int(row["accepted_repair_count"]),
                    new_error=int(row["accepted_new_error_count"]),
                    veto_new=int(row["vetoed_candidate_new_error_count"]),
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The selected rule is calibrated only on validation pseudo-labels; held-out accepted new errors remain the key risk check.",
            "- The rule is still a pseudo-label detector, not a final clean-label semantic guarantee.",
            "- A positive held-out result should be treated as a stronger candidate for M3 gating, not as a final paper conclusion until the method is frozen and tested on a proper split.",
            "",
            "## Output Files",
            "",
        ]
    )
    for key, value in metadata["outputs"].items():
        if key != "galleries":
            lines.append(f"- `{key}`: `{value}`")
    if metadata["outputs"].get("galleries"):
        lines.extend(["", "## Galleries", ""])
        for key, value in metadata["outputs"]["galleries"].items():
            lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    config = load_config(config_path)
    input_paths = check_inputs(config)

    output_dir = resolve_project_path(args.output_dir or config["outputs"]["output_dir"])
    clip_rows = normalize_clip_rows(read_csv(resolve_project_path(config["inputs"]["per_sample_with_clip_csv"])))
    topk_map = load_topk_map(
        resolve_project_path(config["inputs"]["validation_topk_csv"]),
        resolve_project_path(config["inputs"]["heldout_topk_csv"]),
    )
    rows = enrich_with_receiver_features(clip_rows, topk_map)
    check_image_paths(rows)

    if args.dry_run:
        print(f"Config: {project_relative(config_path)}")
        print(f"Input rows: {len(rows)}")
        print(f"Splits: {sorted({row['split'] for row in rows})}")
        print(f"SNRs: {sorted({row['snr_db'] for row in rows})}")
        print(
            "Search grid size: "
            f"{len(config['risk_rule_search']['clip_min_grid']) * len(config['risk_rule_search']['top5_overlap_min_grid']) * len(config['risk_rule_search']['shadow_clip_max_grid']) * len(config['risk_rule_search']['shadow_m0_rank_max_grid']) * len(config['risk_rule_search']['shadow_m0_margin_max_grid']) * len(config['risk_rule_search']['shadow_ref_margin_min_grid'])}"
        )
        print(f"Output dir: {project_relative(output_dir)}")
        print(f"Proxy env present: {proxy_environment_present()}")
        return 0

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Use --overwrite to replace.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    selected_rule, candidate_rows = sweep_rules(rows, config)
    policies = ["top1_equal", "raw_conf_gain", "fixed_clip_ge_0p98", "selected_risk_rule"]
    summary_rows: list[dict[str, Any]] = []
    by_snr_rows: list[dict[str, Any]] = []
    for split in config["risk_rule_search"]["evaluation_splits"]:
        for policy in policies:
            rule = selected_rule if policy == "selected_risk_rule" else None
            summary_rows.append(evaluate_rows(rows, split, policy, rule))
            for snr in config["snrs"]:
                by_snr_rows.append(evaluate_rows(rows, split, policy, rule, float(snr)))
    summary_rows = add_policy_deltas(summary_rows)
    by_snr_rows = add_policy_deltas(by_snr_rows)
    decisions = make_decisions(rows, policies, selected_rule)

    write_csv(output_dir / "rule_candidates.csv", candidate_rows)
    write_csv(output_dir / "policy_summary.csv", summary_rows)
    write_csv(output_dir / "policy_by_snr.csv", by_snr_rows)
    write_csv(output_dir / "policy_decisions.csv", decisions)
    save_json(output_dir / "selected_rule.json", selected_rule)

    metadata = {
        "analysis_id": config["analysis_id"],
        "config": project_relative(config_path),
        "inputs": input_paths,
        "output_dir": project_relative(output_dir),
        "selected_rule": selected_rule,
        "selected_rule_name": rule_to_name(selected_rule),
        "num_rows": len(rows),
        "num_rule_candidates": len(candidate_rows),
        "git_commit": git_commit(),
        "git_dirty_state": git_dirty_state(),
        "python": sys.version,
        "platform": platform.platform(),
        "proxy_environment_present": proxy_environment_present(),
        "download_note": "No model or data download is required; this analysis reuses existing CSVs and PNGs.",
        "outputs": {
            "rule_candidates_csv": project_relative(output_dir / "rule_candidates.csv"),
            "policy_summary_csv": project_relative(output_dir / "policy_summary.csv"),
            "policy_by_snr_csv": project_relative(output_dir / "policy_by_snr.csv"),
            "policy_decisions_csv": project_relative(output_dir / "policy_decisions.csv"),
            "selected_rule_json": project_relative(output_dir / "selected_rule.json"),
            "report_md": project_relative(output_dir / "REPORT.md"),
        },
    }
    galleries = write_galleries(decisions, output_dir, int(config["evaluation"]["gallery_rows"]))
    metadata["outputs"]["galleries"] = galleries
    save_json(output_dir / "metadata.json", metadata)
    (output_dir / "REPORT.md").write_text(make_report(summary_rows, selected_rule, metadata), encoding="utf-8")

    selected_validation = [
        row for row in summary_rows if row["split"] == "validation" and row["policy"] == "selected_risk_rule"
    ][0]
    selected_heldout = [
        row for row in summary_rows if row["split"] == "heldout" and row["policy"] == "selected_risk_rule"
    ][0]
    print(f"Wrote {project_relative(output_dir)}")
    print(f"Selected rule: {rule_to_name(selected_rule)}")
    print(
        "Validation selected: "
        f"failure={selected_validation['final_failure_rate']:.4f}, "
        f"repair={selected_validation['accepted_repair_count']}, "
        f"new_error={selected_validation['accepted_new_error_count']}, "
        f"delta_psnr_vs_top1={selected_validation['delta_final_psnr_vs_top1_equal_db']:+.4f} dB"
    )
    print(
        "Held-out selected: "
        f"failure={selected_heldout['final_failure_rate']:.4f}, "
        f"repair={selected_heldout['accepted_repair_count']}, "
        f"new_error={selected_heldout['accepted_new_error_count']}, "
        f"delta_psnr_vs_top1={selected_heldout['delta_final_psnr_vs_top1_equal_db']:+.4f} dB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
