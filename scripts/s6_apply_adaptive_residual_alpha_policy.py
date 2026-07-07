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

import yaml
import torch
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from s6_residual_shrink_selection import (  # noqa: E402
    alpha_name,
    classify_paths,
    compute_pair_metrics,
    fmt,
    load_classifier,
    load_rgb_tensor,
    resolve_device,
    snr_name,
    try_load_lpips,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply receiver-side adaptive residual-alpha policies to existing shrink candidates."
    )
    parser.add_argument("--config", default="configs/s6_adaptive_residual_alpha_policy_exp_s4_006.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-lpips", action="store_true")
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def serialize_value(value: Any) -> Any:
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


def bool_from_csv(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def rate(values: list[bool]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def signed(value: Any, digits: int = 4) -> str:
    if value in ("", None):
        return ""
    return f"{float(value):+.{digits}f}"


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


def snr_config_key(snr: float) -> str:
    if float(snr).is_integer():
        return str(int(snr))
    return str(snr)


def schedule_alpha(config: dict[str, Any], snr: float) -> float | None:
    value = config["fixed_validation_top1_shrink_schedule"][snr_config_key(snr)]
    if value in (None, "", "null"):
        return None
    return float(value)


def validate_inputs(config: dict[str, Any], snrs: list[float], alphas: list[float]) -> dict[str, Any]:
    paths: dict[str, Path] = {
        "jscc_checkpoint": resolve_project_path(config["inputs"]["jscc_checkpoint"]),
        "forbidden_checkpoint": resolve_project_path(config["inputs"]["forbidden_checkpoint"]),
        "classifier_weights": resolve_project_path(config["classifier"]["weights_file"]),
    }
    split_counts: dict[str, int] = {}
    for split in config["splits"]:
        split_name = str(split["name"])
        csv_path = resolve_project_path(split["base_per_sample_csv"])
        candidate_root = resolve_project_path(split["candidate_root"])
        paths[f"{split_name}_base_per_sample_csv"] = csv_path
        paths[f"{split_name}_candidate_root"] = candidate_root
        if csv_path.exists():
            base_rows = [row for row in read_csv(csv_path) if row.get("policy") == str(split.get("base_policy", "m0"))]
            split_counts[split_name] = len(base_rows)
            for row in base_rows:
                for key in ["original", "m0_reconstruction"]:
                    path = resolve_project_path(row[key])
                    if not path.exists():
                        raise FileNotFoundError(f"Missing {key} image for {split_name}: {path}")
            for snr in snrs:
                snr_rows = [row for row in base_rows if abs(float(row["snr_db"]) - float(snr)) < 1e-9]
                if not snr_rows:
                    raise RuntimeError(f"No base rows for {split_name} SNR {snr}")
                for alpha in alphas:
                    for row in snr_rows:
                        candidate = candidate_root / alpha_name(alpha) / snr_name(snr) / row["sample"]
                        if not candidate.exists():
                            raise FileNotFoundError(f"Missing alpha candidate: {candidate}")
    missing = [f"{key}: {path}" for key, path in paths.items() if key != "forbidden_checkpoint" and not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))
    if paths["jscc_checkpoint"] == paths["forbidden_checkpoint"]:
        raise RuntimeError("Config points to forbidden latest.pt checkpoint.")
    if not paths["classifier_weights"].is_file() or paths["classifier_weights"].stat().st_size < 10 * 1024 * 1024:
        raise RuntimeError(f"Classifier weights missing from local cache: {paths['classifier_weights']}")
    return {
        "paths": {key: project_relative(path) for key, path in paths.items()},
        "split_counts": split_counts,
    }


def load_base_rows(split: dict[str, Any]) -> list[dict[str, str]]:
    rows = read_csv(resolve_project_path(split["base_per_sample_csv"]))
    base_policy = str(split.get("base_policy", "m0"))
    base_rows = [row for row in rows if row.get("policy") == base_policy]
    return sorted(base_rows, key=lambda row: (float(row["snr_db"]), row["sample"]))


def candidate_path(split: dict[str, Any], alpha: float, snr: float, sample: str) -> Path:
    return resolve_project_path(split["candidate_root"]) / alpha_name(alpha) / snr_name(snr) / sample


def classify_split_candidates(
    split: dict[str, Any],
    base_rows: list[dict[str, str]],
    alphas: list[float],
    classifier_model,
    classifier_preprocess,
    config: dict[str, Any],
    device,
) -> tuple[dict[tuple[float, float, str], dict[str, Any]], dict[str, float]]:
    preds: dict[tuple[float, float, str], dict[str, Any]] = {}
    times: dict[str, float] = {}
    batch_size = int(config["classifier"]["batch_size"])
    topk = int(config["classifier"]["topk"])
    for alpha in alphas:
        for snr in sorted({float(row["snr_db"]) for row in base_rows}):
            snr_rows = [row for row in base_rows if abs(float(row["snr_db"]) - float(snr)) < 1e-9]
            paths = [candidate_path(split, alpha, snr, row["sample"]) for row in snr_rows]
            alpha_preds, elapsed = classify_paths(
                classifier_model,
                classifier_preprocess,
                paths,
                batch_size,
                topk,
                device,
            )
            times[f"{split['name']}_alpha_{alpha}_{snr_name(snr)}"] = elapsed
            for row, pred, path in zip(snr_rows, alpha_preds, paths):
                preds[(float(alpha), float(snr), row["sample"])] = {
                    "path": path,
                    "top1_index": int(pred["top_indices"][0]),
                    "top1_prob": float(pred["top_probs"][0]),
                }
    return preds, times


def candidate_for(
    preds: dict[tuple[float, float, str], dict[str, Any]],
    alpha: float,
    snr: float,
    sample: str,
) -> dict[str, Any]:
    return preds[(float(alpha), float(snr), sample)]


def select_policy(
    policy: str,
    base: dict[str, str],
    split: dict[str, Any],
    config: dict[str, Any],
    preds: dict[tuple[float, float, str], dict[str, Any]],
    alphas: list[float],
) -> tuple[float | None, bool, dict[str, Any] | None, str]:
    snr = float(base["snr_db"])
    sample = base["sample"]
    m0_top1 = int(base["m0_top1_index"])
    if policy == "m0":
        return None, False, None, "m0_baseline"
    if policy == "always_full_strength":
        cand = candidate_for(preds, 1.0, snr, sample)
        return 1.0, True, cand, "always_accept_alpha_1"
    if policy == "top1_full_strength":
        cand = candidate_for(preds, 1.0, snr, sample)
        return 1.0, cand["top1_index"] == m0_top1, cand, "alpha_1_if_top1_consistent"
    if policy == "fixed_validation_top1_shrink_schedule":
        alpha = schedule_alpha(config, snr)
        if alpha is None:
            return None, False, None, "fixed_schedule_fallback_m0"
        cand = candidate_for(preds, alpha, snr, sample)
        return alpha, cand["top1_index"] == m0_top1, cand, "validation_fixed_alpha_if_top1_consistent"
    if policy == "adaptive_max_top1_consistent_alpha":
        for alpha in sorted(alphas, reverse=True):
            cand = candidate_for(preds, alpha, snr, sample)
            if cand["top1_index"] == m0_top1:
                return alpha, True, cand, "largest_alpha_with_candidate_top1_equal_m0"
        return None, False, None, "no_top1_consistent_alpha_fallback_m0"
    raise KeyError(f"Unknown policy: {policy}")


def make_policy_row(
    split_name: str,
    policy: str,
    base: dict[str, str],
    selected_alpha: float | None,
    accept: bool,
    selected_candidate: dict[str, Any] | None,
    decision_reason: str,
    preds: dict[tuple[float, float, str], dict[str, Any]],
    alphas: list[float],
) -> dict[str, Any]:
    snr = float(base["snr_db"])
    sample = base["sample"]
    original_top1 = int(base["original_top1_index"])
    m0_top1 = int(base["m0_top1_index"])
    m0_prob = float(base["m0_top1_prob"])
    m0_matches_origin = bool_from_csv(base["m0_matches_original_top1"])
    if selected_candidate is None:
        candidate_top1 = m0_top1
        candidate_prob = m0_prob
        candidate_source = resolve_project_path(base["m0_reconstruction"])
        candidate_matches_origin = m0_matches_origin
    else:
        candidate_top1 = int(selected_candidate["top1_index"])
        candidate_prob = float(selected_candidate["top1_prob"])
        candidate_source = selected_candidate["path"]
        candidate_matches_origin = candidate_top1 == original_top1
    final_source = candidate_source if accept else resolve_project_path(base["m0_reconstruction"])
    final_top1 = candidate_top1 if accept else m0_top1
    final_prob = candidate_prob if accept else m0_prob

    all_candidate_matches_origin = []
    all_candidate_top1_equal_m0 = []
    for alpha in alphas:
        cand = candidate_for(preds, alpha, snr, sample)
        all_candidate_matches_origin.append(int(cand["top1_index"]) == original_top1)
        all_candidate_top1_equal_m0.append(int(cand["top1_index"]) == m0_top1)
    any_candidate_matches_origin = any(all_candidate_matches_origin)
    max_top1_consistent_alpha = None
    for alpha in sorted(alphas, reverse=True):
        cand = candidate_for(preds, alpha, snr, sample)
        if int(cand["top1_index"]) == m0_top1:
            max_top1_consistent_alpha = alpha
            break

    return {
        "split": split_name,
        "policy": policy,
        "snr_db": snr,
        "sample": sample,
        "selected_alpha": "" if selected_alpha is None else float(selected_alpha),
        "accept_candidate": accept,
        "decision_reason": decision_reason,
        "original": base["original"],
        "m0_reconstruction": base["m0_reconstruction"],
        "candidate": project_relative(candidate_source),
        "final_source": project_relative(final_source),
        "original_top1_index": original_top1,
        "original_top1_label": base["original_top1_label"],
        "original_top1_prob": float(base["original_top1_prob"]),
        "m0_top1_index": m0_top1,
        "m0_top1_label": base["m0_top1_label"],
        "m0_top1_prob": m0_prob,
        "candidate_top1_index": candidate_top1,
        "candidate_top1_prob": candidate_prob,
        "final_top1_index": final_top1,
        "final_top1_prob": final_prob,
        "m0_matches_original_top1": m0_matches_origin,
        "candidate_matches_original_top1": candidate_matches_origin,
        "candidate_matches_m0_top1": candidate_top1 == m0_top1,
        "final_matches_original_top1": final_top1 == original_top1,
        "final_matches_m0_top1": final_top1 == m0_top1,
        "accepted_repair": accept and (not m0_matches_origin) and final_top1 == original_top1,
        "accepted_new_error": accept and m0_matches_origin and final_top1 != original_top1,
        "any_candidate_matches_original_top1": any_candidate_matches_origin,
        "missed_repair": (not m0_matches_origin) and final_top1 != original_top1 and any_candidate_matches_origin,
        "protective_reject": (not accept) and m0_matches_origin and any(not item for item in all_candidate_matches_origin),
        "max_top1_consistent_alpha": "" if max_top1_consistent_alpha is None else float(max_top1_consistent_alpha),
    }


def stack_paths(paths: list[Path]) -> Any:
    return torch.stack([load_rgb_tensor(path) for path in paths])


def summarize_policy(
    split: str,
    policy: str,
    snr: float | str,
    rows: list[dict[str, Any]],
    m0_metrics: dict[str, float | None],
    lpips_model,
    device,
    batch_size: int,
) -> dict[str, Any]:
    if not rows:
        return {}
    references = stack_paths([resolve_project_path(row["original"]) for row in rows])
    final = stack_paths([resolve_project_path(row["final_source"]) for row in rows])
    metrics = compute_pair_metrics(references, final, lpips_model, device, batch_size)
    m0_failure = 1.0 - rate([bool(row["m0_matches_original_top1"]) for row in rows])
    final_failure = 1.0 - rate([bool(row["final_matches_original_top1"]) for row in rows])
    selected_alpha_values = [row["selected_alpha"] for row in rows if row["selected_alpha"] != ""]
    out = {
        "split": split,
        "policy": policy,
        "snr_db": snr,
        "num_images": len(rows),
        "accept_rate": rate([bool(row["accept_candidate"]) for row in rows]),
        "fallback_rate": 1.0 - rate([bool(row["accept_candidate"]) for row in rows]),
        "mean_selected_alpha_accepted": mean([float(row["selected_alpha"]) for row in rows if row["accept_candidate"] and row["selected_alpha"] != ""]),
        "mean_selected_alpha_all_nonfallback": mean([float(value) for value in selected_alpha_values]),
        "m0_failure_rate": m0_failure,
        "final_failure_rate": final_failure,
        "delta_final_failure_vs_m0": final_failure - m0_failure,
        "repair_count": int(sum(bool(row["accepted_repair"]) for row in rows)),
        "accepted_new_error_count": int(sum(bool(row["accepted_new_error"]) for row in rows)),
        "missed_repair_count": int(sum(bool(row["missed_repair"]) for row in rows)),
        "available_repair_count": int(
            sum((not bool(row["m0_matches_original_top1"])) and bool(row["any_candidate_matches_original_top1"]) for row in rows)
        ),
        "protective_reject_count": int(sum(bool(row["protective_reject"]) for row in rows)),
        "final_mse": metrics["mse"],
        "final_psnr_db": metrics["psnr_db"],
        "final_ssim": metrics["ssim"],
        "final_ms_ssim": metrics["ms_ssim"],
        "final_lpips": metrics["lpips"],
        "delta_psnr_vs_m0_db": None
        if metrics["psnr_db"] is None or m0_metrics["psnr_db"] is None
        else metrics["psnr_db"] - m0_metrics["psnr_db"],
        "delta_lpips_vs_m0": None
        if metrics["lpips"] is None or m0_metrics["lpips"] is None
        else metrics["lpips"] - m0_metrics["lpips"],
        "delta_ms_ssim_vs_m0": None
        if metrics["ms_ssim"] is None or m0_metrics["ms_ssim"] is None
        else metrics["ms_ssim"] - m0_metrics["ms_ssim"],
    }
    return out


def compute_m0_metrics(rows: list[dict[str, str]], lpips_model, device, batch_size: int) -> dict[str, float | None]:
    references = stack_paths([resolve_project_path(row["original"]) for row in rows])
    m0 = stack_paths([resolve_project_path(row["m0_reconstruction"]) for row in rows])
    return compute_pair_metrics(references, m0, lpips_model, device, batch_size)


def make_sample_grids(policy_rows: list[dict[str, Any]], output_dir: Path, sample_grid_count: int) -> list[str]:
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in policy_rows:
        if row["policy"] != "adaptive_max_top1_consistent_alpha":
            continue
        groups.setdefault((row["split"], float(row["snr_db"])), []).append(row)
    for (split, snr), rows in sorted(groups.items()):
        selected = rows[:sample_grid_count]
        if not selected:
            continue
        tensors = []
        tensors.extend([load_rgb_tensor(resolve_project_path(row["original"])) for row in selected])
        tensors.extend([load_rgb_tensor(resolve_project_path(row["m0_reconstruction"])) for row in selected])
        tensors.extend([load_rgb_tensor(resolve_project_path(row["candidate"])) for row in selected])
        tensors.extend([load_rgb_tensor(resolve_project_path(row["final_source"])) for row in selected])
        path = sample_dir / f"{split}_{snr_name(snr)}_original_m0_selected_final.png"
        save_image(torch.stack(tensors), path, nrow=len(selected))
        paths.append(project_relative(path))
    return paths


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    lines = []
    lines.append("| " + " | ".join(label for _key, label in columns) + " |")
    lines.append("|" + "|".join(["---" for _ in columns]) + "|")
    for row in rows:
        values = [fmt(row.get(key, "")) for key, _label in columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def all_row(summary_rows: list[dict[str, Any]], split: str, policy: str) -> dict[str, Any]:
    for row in summary_rows:
        if row["split"] == split and row["policy"] == policy and str(row["snr_db"]) == "all":
            return row
    raise KeyError(f"Missing all summary row: {split}/{policy}")


def make_report(config: dict[str, Any], summary_rows: list[dict[str, Any]], metadata: dict[str, Any]) -> str:
    splits = [str(item["name"]) for item in config["splits"]]
    compact = [
        row
        for row in summary_rows
        if str(row["snr_db"]) == "all"
        and row["policy"]
        in {
            "m0",
            "top1_full_strength",
            "fixed_validation_top1_shrink_schedule",
            "adaptive_max_top1_consistent_alpha",
            "always_full_strength",
        }
    ]
    adaptive_psnr = "/".join(signed(all_row(summary_rows, split, "adaptive_max_top1_consistent_alpha")["delta_psnr_vs_m0_db"]) for split in splits)
    adaptive_new_errors = "/".join(
        str(all_row(summary_rows, split, "adaptive_max_top1_consistent_alpha")["accepted_new_error_count"])
        for split in splits
    )
    fixed_psnr = "/".join(
        signed(all_row(summary_rows, split, "fixed_validation_top1_shrink_schedule")["delta_psnr_vs_m0_db"])
        for split in splits
    )
    full_psnr = "/".join(signed(all_row(summary_rows, split, "top1_full_strength")["delta_psnr_vs_m0_db"]) for split in splits)
    lines = [
        "# Adaptive Residual Alpha Policy",
        "",
        "This derived analysis evaluates whether residual strength can be selected per sample with receiver-side top-1 consistency.",
        "It reads existing alpha candidate PNG files and does not train, run diffusion, regenerate residuals, download data, or tune on held-out/test-like splits.",
        "",
        "## Bottom Line",
        "",
        f"- Fixed validation shrink schedule PSNR deltas on validation/held-out/test-like: `{fixed_psnr}` dB.",
        f"- Full-strength top-1 fallback PSNR deltas on validation/held-out/test-like: `{full_psnr}` dB.",
        f"- Adaptive max top-1-consistent alpha PSNR deltas on validation/held-out/test-like: `{adaptive_psnr}` dB.",
        f"- Adaptive max top-1-consistent alpha accepted new errors on validation/held-out/test-like: `{adaptive_new_errors}`.",
        "- Interpretation: per-sample alpha selection is a stronger candidate for moving residual strength control into the M3 method than a fixed per-SNR alpha schedule.",
        "",
        "## All-Split Policy Summary",
        "",
    ]
    display_rows = []
    for row in compact:
        display_rows.append(
            {
                "split": row["split"],
                "policy": row["policy"],
                "delta_psnr": signed(row["delta_psnr_vs_m0_db"]),
                "delta_lpips": signed(row["delta_lpips_vs_m0"]),
                "failure_delta": signed(row["delta_final_failure_vs_m0"]),
                "accept": fmt(row["accept_rate"]),
                "mean_alpha": fmt(row["mean_selected_alpha_accepted"]),
                "repair": row["repair_count"],
                "new_error": row["accepted_new_error_count"],
                "missed_repair": row["missed_repair_count"],
            }
        )
    lines += markdown_table(
        display_rows,
        [
            ("split", "Split"),
            ("policy", "Policy"),
            ("delta_psnr", "Delta PSNR"),
            ("delta_lpips", "Delta LPIPS"),
            ("failure_delta", "Failure Delta"),
            ("accept", "Accept"),
            ("mean_alpha", "Mean Alpha"),
            ("repair", "Repair"),
            ("new_error", "New Error"),
            ("missed_repair", "Missed Repair"),
        ],
    )
    lines.extend(["", "## Per-SNR Adaptive Policy", ""])
    adaptive_snr = [
        row for row in summary_rows if row["policy"] == "adaptive_max_top1_consistent_alpha" and str(row["snr_db"]) != "all"
    ]
    lines += markdown_table(
        adaptive_snr,
        [
            ("split", "Split"),
            ("snr_db", "SNR"),
            ("delta_psnr_vs_m0_db", "Delta PSNR"),
            ("delta_lpips_vs_m0", "Delta LPIPS"),
            ("accept_rate", "Accept"),
            ("mean_selected_alpha_accepted", "Mean Alpha"),
            ("accepted_new_error_count", "New Error"),
            ("missed_repair_count", "Missed Repair"),
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
            "- The semantic signal is still frozen AlexNet pseudo-label agreement on COCO crops.",
            "- The adaptive rule is receiver-side and does not use original images, but it is still a post-hoc policy over existing alpha candidates.",
            "- This is not yet a retrained residual CNN with learned semantic-risk-aware amplitude.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    snrs = [float(item) for item in config["snrs"]]
    alphas = [float(item) for item in config["alphas"]]
    input_manifest = validate_inputs(config, snrs, alphas)
    output_dir = resolve_project_path(args.output_dir or config["outputs"]["output_dir"])
    dry_payload = {
        "status": "ok",
        "config": project_relative(config_path),
        "inputs": input_manifest,
        "output_dir": project_relative(output_dir),
        "snrs": snrs,
        "alphas": alphas,
        "policies": config["policies"],
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
    classifier_model, classifier_preprocess, _categories = load_classifier(config, device)
    lpips_model = None
    lpips_error = None
    if not args.skip_lpips:
        lpips_model, lpips_error = try_load_lpips(device, output_dir / "cache")

    policy_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    classification_times: dict[str, float] = {}
    image_batch_size = int(config["evaluation"]["image_batch_size"])

    for split in config["splits"]:
        split_name = str(split["name"])
        base_rows = load_base_rows(split)
        preds, times = classify_split_candidates(
            split,
            base_rows,
            alphas,
            classifier_model,
            classifier_preprocess,
            config,
            device,
        )
        classification_times.update(times)

        m0_metrics_by_snr: dict[float, dict[str, float | None]] = {}
        for snr in snrs:
            snr_base = [row for row in base_rows if abs(float(row["snr_db"]) - float(snr)) < 1e-9]
            m0_metrics_by_snr[float(snr)] = compute_m0_metrics(snr_base, lpips_model, device, image_batch_size)

        for policy in config["policies"]:
            for snr in snrs:
                snr_base = [row for row in base_rows if abs(float(row["snr_db"]) - float(snr)) < 1e-9]
                rows_for_policy = []
                for base in snr_base:
                    selected_alpha, accept, selected_candidate, reason = select_policy(
                        str(policy),
                        base,
                        split,
                        config,
                        preds,
                        alphas,
                    )
                    rows_for_policy.append(
                        make_policy_row(
                            split_name,
                            str(policy),
                            base,
                            selected_alpha,
                            accept,
                            selected_candidate,
                            reason,
                            preds,
                            alphas,
                        )
                    )
                policy_rows.extend(rows_for_policy)
                summary_rows.append(
                    summarize_policy(
                        split_name,
                        str(policy),
                        float(snr),
                        rows_for_policy,
                        m0_metrics_by_snr[float(snr)],
                        lpips_model,
                        device,
                        image_batch_size,
                    )
                )

            all_rows = [row for row in policy_rows if row["split"] == split_name and row["policy"] == str(policy)]
            all_m0_metrics = compute_m0_metrics(base_rows, lpips_model, device, image_batch_size)
            summary_rows.append(
                summarize_policy(
                    split_name,
                    str(policy),
                    "all",
                    all_rows,
                    all_m0_metrics,
                    lpips_model,
                    device,
                    image_batch_size,
                )
            )

    sample_grids = make_sample_grids(policy_rows, output_dir, int(config["evaluation"]["sample_grid_count"]))
    summary_csv = output_dir / "summary.csv"
    per_sample_csv = output_dir / "per_sample.csv"
    metadata_json = output_dir / "metadata.json"
    report_path = output_dir / "REPORT.md"
    write_csv(summary_csv, summary_rows)
    write_csv(per_sample_csv, policy_rows)
    metadata = {
        "analysis_id": config["analysis_id"],
        "method": config["method"],
        "project_commit": git_commit(),
        "git_dirty_state": git_dirty_state(),
        "python": sys.version,
        "platform": platform.platform(),
        "config": project_relative(config_path),
        "input_manifest": input_manifest,
        "summary_csv": project_relative(summary_csv),
        "per_sample_csv": project_relative(per_sample_csv),
        "metadata_json": project_relative(metadata_json),
        "sample_dir": project_relative(output_dir / "samples"),
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
