from __future__ import annotations

import argparse
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
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate per-SNR receiver-side CLIP veto thresholds for EXP-S4-006 confidence-gain gate."
    )
    parser.add_argument("--config", default="configs/s5_conf_gain_clip_veto_snr_calibration_exp_s4_006.yaml")
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


def safe_float(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def threshold_display(value: float) -> str:
    if value <= 0.0:
        return "no_veto"
    if value > 1.0:
        return "top1_only"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def schedule_name(prefix: str, schedule: dict[float, float]) -> str:
    parts = [f"{int(snr)}db_{safe_float(threshold)}" for snr, threshold in sorted(schedule.items())]
    return f"{prefix}__" + "__".join(parts)


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


def normalize_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
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


def check_inputs(config: dict[str, Any]) -> dict[str, str]:
    paths = {
        "per_sample_with_clip_csv": resolve_project_path(config["inputs"]["per_sample_with_clip_csv"]),
        "parent_report": resolve_project_path(config["inputs"]["parent_report"]),
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


def accept_with_threshold(row: dict[str, Any], threshold: float) -> bool:
    if bool(row["baseline_accept_refined"]):
        return True
    if not bool(row["candidate_accept_refined"]):
        return False
    return float(row["clip_sim_m0_refined"]) >= threshold


def accept_for_policy(row: dict[str, Any], policy_name: str, schedule: dict[float, float] | None) -> bool:
    if policy_name == "top1_equal":
        return bool(row["baseline_accept_refined"])
    if policy_name == "raw_conf_gain":
        return bool(row["candidate_accept_refined"])
    if schedule is None:
        raise ValueError(f"Schedule required for policy: {policy_name}")
    threshold = schedule[float(row["snr_db"])]
    return accept_with_threshold(row, threshold)


def evaluate_rows(
    rows: list[dict[str, Any]],
    policy_name: str,
    split: str,
    schedule: dict[float, float] | None,
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
    final_psnrs: list[float] = []
    m0_psnrs: list[float] = []
    refined_psnrs: list[float] = []
    accepted_clip: list[float] = []
    new_accept_clip: list[float] = []

    for row in subset:
        accept = accept_for_policy(row, policy_name, schedule)
        baseline_accept = bool(row["baseline_accept_refined"])
        candidate_accept = bool(row["candidate_accept_refined"])
        m0_ok = bool(row["m0_matches_original_top1"])
        refined_ok = bool(row["refined_matches_original_top1"])
        final_ok = refined_ok if accept else m0_ok
        final_psnr = float(row["refined_psnr_db"] if accept else row["m0_psnr_db"])
        clip_sim = float(row["clip_sim_m0_refined"])

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
        final_psnrs.append(final_psnr)
        m0_psnrs.append(float(row["m0_psnr_db"]))
        refined_psnrs.append(float(row["refined_psnr_db"]))
        if accept:
            accepted_clip.append(clip_sim)
        if accept and not baseline_accept:
            new_accept_clip.append(clip_sim)

    threshold_value = ""
    if schedule is not None and snr is not None:
        threshold_value = schedule[float(snr)]
    return {
        "split": split,
        "policy": policy_name,
        "snr_db": "all" if snr is None else float(snr),
        "threshold": threshold_value,
        "threshold_display": "" if threshold_value == "" else threshold_display(float(threshold_value)),
        "num_images": total,
        "accept_count": accepted,
        "accept_rate": rate(accepted, total),
        "reject_count": total - accepted,
        "reject_rate": rate(total - accepted, total),
        "new_accept_vs_top1_count": new_accept,
        "vetoed_candidate_accept_count": vetoed_candidate_accept,
        "vetoed_candidate_repair_count": vetoed_candidate_repair,
        "vetoed_candidate_new_error_count": vetoed_candidate_new_error,
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
        "accepted_clip_sim_m0_refined_mean": mean(accepted_clip),
        "new_accept_clip_sim_m0_refined_mean": mean(new_accept_clip),
        "new_accept_clip_sim_m0_refined_min": min(new_accept_clip) if new_accept_clip else "",
    }


def add_policy_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed = {(str(row["split"]), str(row["snr_db"]), str(row["policy"])): row for row in rows}
    output: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        for baseline_name in ["top1_equal", "raw_conf_gain"]:
            baseline = keyed.get((str(row["split"]), str(row["snr_db"]), baseline_name))
            if not baseline:
                continue
            suffix = "top1_equal" if baseline_name == "top1_equal" else "raw_conf_gain"
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


def select_independent_schedule(
    rows: list[dict[str, Any]],
    snrs: list[float],
    thresholds: list[float],
    split: str,
    budget: int,
) -> tuple[dict[float, float], list[dict[str, Any]]]:
    schedule: dict[float, float] = {}
    candidates: list[dict[str, Any]] = []
    for snr in snrs:
        rows_for_snr: list[dict[str, Any]] = []
        for threshold in thresholds:
            candidate_schedule = {item: 1.01 for item in snrs}
            candidate_schedule[snr] = threshold
            metrics = evaluate_rows(rows, f"candidate_threshold_{safe_float(threshold)}", split, candidate_schedule, snr)
            metrics["candidate_threshold"] = threshold
            rows_for_snr.append(metrics)
        feasible = [row for row in rows_for_snr if int(row["accepted_new_error_count"]) <= budget]
        if not feasible:
            feasible = [row for row in rows_for_snr if float(row["threshold"]) > 1.0]
        selected = sorted(
            feasible,
            key=lambda row: (
                int(row["accepted_repair_count"]),
                float(row["final_psnr_db"]),
                -float(row["candidate_threshold"]),
            ),
            reverse=True,
        )[0]
        schedule[snr] = float(selected["candidate_threshold"])
        selected = dict(selected)
        selected["selected"] = True
        for row in rows_for_snr:
            row_out = dict(row)
            row_out["selected"] = float(row_out["candidate_threshold"]) == schedule[snr]
            candidates.append(row_out)
    return schedule, candidates


def is_monotonic_low_snr_not_weaker(snrs: list[float], schedule: dict[float, float]) -> bool:
    ordered = [schedule[snr] for snr in snrs]
    return all(left >= right for left, right in zip(ordered, ordered[1:]))


def select_monotonic_schedule(
    rows: list[dict[str, Any]],
    snrs: list[float],
    thresholds: list[float],
    split: str,
    budget: int,
) -> tuple[dict[float, float], list[dict[str, Any]]]:
    candidates: list[tuple[dict[str, Any], dict[float, float]]] = []
    for combo in itertools.product(thresholds, repeat=len(snrs)):
        schedule = dict(zip(snrs, combo))
        if not is_monotonic_low_snr_not_weaker(snrs, schedule):
            continue
        metrics = evaluate_rows(rows, "monotonic_candidate", split, schedule, None)
        if int(metrics["accepted_new_error_count"]) <= budget:
            candidates.append((metrics, schedule))
    if not candidates:
        top1 = {snr: 1.01 for snr in snrs}
        return top1, []
    selected_metrics, selected_schedule = sorted(
        candidates,
        key=lambda item: (
            int(item[0]["accepted_repair_count"]),
            float(item[0]["final_psnr_db"]),
            -sum(item[1].values()),
        ),
        reverse=True,
    )[0]
    rows_out: list[dict[str, Any]] = []
    for metrics, schedule in candidates:
        row = dict(metrics)
        row["schedule_json"] = json.dumps({str(int(k)): v for k, v in sorted(schedule.items())})
        row["selected"] = schedule == selected_schedule
        rows_out.append(row)
    return selected_schedule, rows_out


def make_schedule_rows(schedules: dict[str, dict[float, float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy, schedule in schedules.items():
        for snr, threshold in sorted(schedule.items()):
            rows.append(
                {
                    "policy": policy,
                    "snr_db": snr,
                    "threshold": threshold,
                    "threshold_display": threshold_display(threshold),
                }
            )
    return rows


def make_decisions(
    rows: list[dict[str, Any]],
    policies: list[tuple[str, dict[float, float] | None]],
) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for row in rows:
        for policy_name, schedule in policies:
            accept = accept_for_policy(row, policy_name, schedule)
            baseline_accept = bool(row["baseline_accept_refined"])
            candidate_accept = bool(row["candidate_accept_refined"])
            m0_ok = bool(row["m0_matches_original_top1"])
            refined_ok = bool(row["refined_matches_original_top1"])
            decisions.append(
                {
                    "split": row["split"],
                    "policy": policy_name,
                    "snr_db": row["snr_db"],
                    "sample": row["sample"],
                    "threshold": "" if schedule is None else schedule[float(row["snr_db"])],
                    "threshold_display": "" if schedule is None else threshold_display(schedule[float(row["snr_db"])]),
                    "accept_refined": accept,
                    "baseline_accept_refined": baseline_accept,
                    "candidate_accept_refined": candidate_accept,
                    "new_accept_vs_top1": accept and not baseline_accept,
                    "vetoed_candidate_accept": candidate_accept and not accept,
                    "final_matches_original_top1": refined_ok if accept else m0_ok,
                    "accepted_repair": accept and (not m0_ok) and refined_ok,
                    "missed_repair": (not accept) and (not m0_ok) and refined_ok,
                    "accepted_new_error": accept and m0_ok and (not refined_ok),
                    "protective_reject": (not accept) and m0_ok and (not refined_ok),
                    "final_psnr_db": row["refined_psnr_db"] if accept else row["m0_psnr_db"],
                    "clip_sim_m0_refined": row["clip_sim_m0_refined"],
                    "m0_top1_label": row["m0_top1_label"],
                    "refined_top1_label": row["refined_top1_label"],
                    "m0_top1_prob": row["m0_top1_prob"],
                    "refined_top1_prob": row["refined_top1_prob"],
                    "original": row["original"],
                    "m0_reconstruction": row["m0_reconstruction"],
                    "refined": row["refined"],
                }
            )
    return decisions


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
    label_height = 50
    canvas = Image.new("RGB", (192 * len(images), 192 + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for idx, ((label, _path), image) in enumerate(zip(specs, images)):
        x = idx * 192
        canvas.paste(image, (x, label_height))
        draw.text((x + 4, 4), label, fill=(0, 0, 0), font=font)
    detail = (
        f"{row['split']} {row['policy']} {float(row['snr_db']):g}dB {row['sample']} "
        f"clip={float(row['clip_sim_m0_refined']):.4f} thr={row['threshold_display']}"
    )
    draw.text((4, 24), detail[:150], fill=(0, 0, 0), font=font)
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


def write_galleries(
    decisions: list[dict[str, Any]],
    output_dir: Path,
    policies: list[str],
    gallery_rows: int,
) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for policy in policies:
        safe_policy = re.sub(r"[^A-Za-z0-9_.-]+", "_", policy)
        policy_rows = [row for row in decisions if row["policy"] == policy]
        for split in ["validation", "heldout"]:
            split_rows = [row for row in policy_rows if row["split"] == split]
            new_errors = sorted(
                [row for row in split_rows if parse_bool(row["accepted_new_error"])],
                key=lambda row: float(row["clip_sim_m0_refined"]),
            )[:gallery_rows]
            new_accepts = sorted(
                [row for row in split_rows if parse_bool(row["new_accept_vs_top1"])],
                key=lambda row: float(row["clip_sim_m0_refined"]),
            )[:gallery_rows]
            for name, rows_subset in [
                ("accepted_new_errors", new_errors),
                ("new_accepts_lowest_clip", new_accepts),
            ]:
                path = output_dir / "galleries" / safe_policy / f"{split}_{name}.png"
                make_sheet(rows_subset, path, f"{policy} {split} {name}")
                if path.exists():
                    manifest[f"{safe_policy}_{split}_{name}"] = project_relative(path)
    return manifest


def make_report(
    summary_rows: list[dict[str, Any]],
    schedules: dict[str, dict[float, float]],
    metadata: dict[str, Any],
) -> str:
    global_rows = [row for row in summary_rows if str(row["snr_db"]) == "all"]
    keyed = {(row["split"], row["policy"]): row for row in global_rows}
    selected_policies = [
        "top1_equal",
        "raw_conf_gain",
        "fixed_clip_ge_0p98",
        "snr_independent_calibrated",
        "snr_monotonic_calibrated",
    ]
    lines = [
        "# EXP-S4-006 SNR-Calibrated CLIP Veto",
        "",
        "This derived analysis calibrates receiver-side `CLIP(M0, refined)` veto thresholds on validation only, then evaluates the selected schedules on held-out rows.",
        "",
        "## Calibrated Schedules",
        "",
        "| Policy | 1 dB | 4 dB | 7 dB | 13 dB | 19 dB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy, schedule in schedules.items():
        values = [threshold_display(schedule[snr]) for snr in sorted(schedule)]
        lines.append("| {policy} | {values} |".format(policy=policy, values=" | ".join(values)))
    lines.extend(
        [
            "",
            "## Global Results",
            "",
            "| Split | Policy | Final Failure | Delta Failure vs Top1 | Final PSNR | Delta PSNR vs Top1 | Accepted Repair | Accepted New Error |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split in ["validation", "heldout"]:
        for policy in selected_policies:
            row = keyed[(split, policy)]
            lines.append(
                "| {split} | {policy} | {fail:.4f} | {dfail:+.4f} | {psnr:.4f} | {dpsnr:+.4f} | {repair} | {new_error} |".format(
                    split=split,
                    policy=policy,
                    fail=float(row["final_failure_rate"]),
                    dfail=float(row.get("delta_final_failure_vs_top1_equal", 0.0)),
                    psnr=float(row["final_psnr_db"]),
                    dpsnr=float(row.get("delta_final_psnr_vs_top1_equal_db", 0.0)),
                    repair=int(row["accepted_repair_count"]),
                    new_error=int(row["accepted_new_error_count"]),
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `snr_independent_calibrated` is selected per SNR on validation and may violate the low-SNR-not-weaker monotonic constraint.",
            "- `snr_monotonic_calibrated` enforces threshold(1 dB) >= threshold(4 dB) >= threshold(7 dB) >= threshold(13 dB) >= threshold(19 dB), matching the current SNR-aware semantic-control discipline.",
            "- Held-out accepted new errors remain the deciding risk signal; a schedule that improves validation but leaks new errors on held-out must stay diagnostic.",
            "",
            "## Output Files",
            "",
            f"- `policy_summary.csv`: `{metadata['policy_summary_csv']}`",
            f"- `policy_by_snr.csv`: `{metadata['policy_by_snr_csv']}`",
            f"- `policy_decisions.csv`: `{metadata['policy_decisions_csv']}`",
            f"- `calibrated_schedules.csv`: `{metadata['calibrated_schedules_csv']}`",
            f"- `metadata.json`: `{metadata['metadata_json']}`",
        ]
    )
    if metadata.get("galleries"):
        lines.extend(["", "## Galleries", ""])
        for key, value in sorted(metadata["galleries"].items()):
            lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    config = load_config(config_path)
    input_paths = check_inputs(config)
    rows = normalize_rows(read_csv(resolve_project_path(config["inputs"]["per_sample_with_clip_csv"])))
    check_image_paths(rows)

    snrs = [float(item) for item in config["snrs"]]
    thresholds = sorted({float(item) for item in config["policy"]["threshold_grid"]})
    calibration_split = str(config["calibration"]["calibration_split"])
    eval_splits = [str(item) for item in config["calibration"]["evaluation_splits"]]
    budget = int(config["calibration"]["accepted_new_error_budget"])
    fixed_threshold = float(config["policy"]["fixed_reference_threshold"])
    output_dir = resolve_project_path(args.output_dir or config["outputs"]["output_dir"])

    if args.dry_run:
        print(f"Config: {project_relative(config_path)}")
        print(f"Input rows: {len(rows)}")
        print(f"Splits: {sorted({row['split'] for row in rows})}")
        print(f"SNRs: {snrs}")
        print(f"Thresholds: {thresholds}")
        print(f"Output dir: {project_relative(output_dir)}")
        print(f"Proxy env present: {proxy_environment_present()}")
        return

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output_dir / "config.yaml")

    independent_schedule, independent_candidates = select_independent_schedule(
        rows=rows,
        snrs=snrs,
        thresholds=thresholds,
        split=calibration_split,
        budget=budget,
    )
    monotonic_schedule, monotonic_candidates = select_monotonic_schedule(
        rows=rows,
        snrs=snrs,
        thresholds=thresholds,
        split=calibration_split,
        budget=budget,
    )
    fixed_schedule = {snr: fixed_threshold for snr in snrs}
    schedules = {
        "fixed_clip_ge_0p98": fixed_schedule,
        "snr_independent_calibrated": independent_schedule,
        "snr_monotonic_calibrated": monotonic_schedule,
    }
    policies: list[tuple[str, dict[float, float] | None]] = [
        ("top1_equal", None),
        ("raw_conf_gain", None),
        ("fixed_clip_ge_0p98", fixed_schedule),
        ("snr_independent_calibrated", independent_schedule),
        ("snr_monotonic_calibrated", monotonic_schedule),
    ]
    summary_rows = [
        evaluate_rows(rows, policy_name, split, schedule, None)
        for split in eval_splits
        for policy_name, schedule in policies
    ]
    summary_rows = add_policy_deltas(summary_rows)
    by_snr_rows = [
        evaluate_rows(rows, policy_name, split, schedule, snr)
        for split in eval_splits
        for policy_name, schedule in policies
        for snr in snrs
    ]
    by_snr_rows = add_policy_deltas(by_snr_rows)
    decisions = make_decisions(rows, policies)
    gallery_manifest = write_galleries(
        decisions=decisions,
        output_dir=output_dir,
        policies=["raw_conf_gain", "snr_independent_calibrated", "snr_monotonic_calibrated"],
        gallery_rows=int(config["evaluation"]["gallery_rows"]),
    )

    policy_summary_path = output_dir / "policy_summary.csv"
    policy_by_snr_path = output_dir / "policy_by_snr.csv"
    decisions_path = output_dir / "policy_decisions.csv"
    schedules_path = output_dir / "calibrated_schedules.csv"
    independent_candidates_path = output_dir / "independent_threshold_candidates.csv"
    monotonic_candidates_path = output_dir / "monotonic_schedule_candidates.csv"
    metadata_path = output_dir / "metadata.json"
    report_path = output_dir / "REPORT.md"

    write_csv(policy_summary_path, summary_rows)
    write_csv(policy_by_snr_path, by_snr_rows)
    write_csv(decisions_path, decisions)
    write_csv(schedules_path, make_schedule_rows(schedules))
    write_csv(independent_candidates_path, independent_candidates)
    write_csv(monotonic_candidates_path, monotonic_candidates)

    metadata = {
        "analysis_id": config["analysis_id"],
        "source_experiment": config["source_experiment"],
        "source_analysis": config["source_analysis"],
        "config": project_relative(config_path),
        "copied_config": project_relative(output_dir / "config.yaml"),
        "inputs": input_paths,
        "output_dir": project_relative(output_dir),
        "policy_summary_csv": project_relative(policy_summary_path),
        "policy_by_snr_csv": project_relative(policy_by_snr_path),
        "policy_decisions_csv": project_relative(decisions_path),
        "calibrated_schedules_csv": project_relative(schedules_path),
        "independent_threshold_candidates_csv": project_relative(independent_candidates_path),
        "monotonic_schedule_candidates_csv": project_relative(monotonic_candidates_path),
        "metadata_json": project_relative(metadata_path),
        "report": project_relative(report_path),
        "galleries": gallery_manifest,
        "calibration_split": calibration_split,
        "evaluation_splits": eval_splits,
        "threshold_grid": thresholds,
        "accepted_new_error_budget": budget,
        "schedules": {name: {str(int(snr)): value for snr, value in sorted(schedule.items())} for name, schedule in schedules.items()},
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "git_dirty_state": git_dirty_state(),
        "proxy_environment_present": proxy_environment_present(),
        "download_note": "No model or data download is required; this analysis reuses cached CLIP similarities.",
    }
    save_json(metadata_path, metadata)
    report_path.write_text(make_report(summary_rows, schedules, metadata), encoding="utf-8")
    print(f"Wrote {project_relative(report_path)}")
    print(f"Independent schedule: {metadata['schedules']['snr_independent_calibrated']}")
    print(f"Monotonic schedule: {metadata['schedules']['snr_monotonic_calibrated']}")


if __name__ == "__main__":
    main()
