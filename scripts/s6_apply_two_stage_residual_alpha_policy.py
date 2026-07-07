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
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from s6_residual_shrink_selection import compute_pair_metrics, load_rgb_tensor, resolve_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose a two-stage residual-alpha policy from an existing adaptive-alpha per-sample table."
    )
    parser.add_argument("--config", default="configs/s6_two_stage_residual_alpha_policy_exp_s4_006.yaml")
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


def to_float(value: Any, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    return float(value)


def to_int(value: Any, default: int = 0) -> int:
    if value in ("", None):
        return default
    return int(float(value))


def fmt(value: Any, digits: int = 4) -> str:
    if value in ("", None):
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


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


def row_key(row: dict[str, Any]) -> tuple[str, float, str]:
    return str(row["split"]), float(row["snr_db"]), str(row["sample"])


def path_exists_from_row(row: dict[str, str], key: str) -> bool:
    value = row.get(key, "")
    return bool(value) and resolve_project_path(value).exists()


def validate_inputs(config: dict[str, Any]) -> dict[str, Any]:
    paths = {key: resolve_project_path(value) for key, value in config["inputs"].items()}
    missing = [f"{key}: {path}" for key, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    rows = read_csv(paths["source_per_sample_csv"])
    required_policies = {
        "m0",
        str(config["two_stage_policy"]["first_stage_policy"]),
        str(config["two_stage_policy"]["fallback_policy"]),
    }
    counts: dict[str, int] = {}
    missing_paths: list[str] = []
    for policy in sorted(required_policies):
        policy_rows = [row for row in rows if row.get("policy") == policy]
        counts[policy] = len(policy_rows)
        for row in policy_rows:
            for key in ["original", "m0_reconstruction", "final_source"]:
                if not path_exists_from_row(row, key):
                    missing_paths.append(f"{policy}/{row.get('split')}/{row.get('snr_db')}/{row.get('sample')}:{key}")
                    if len(missing_paths) >= 10:
                        break
            if len(missing_paths) >= 10:
                break
        if len(missing_paths) >= 10:
            break
    if missing_paths:
        raise FileNotFoundError("Missing image paths in source per-sample table:\n" + "\n".join(missing_paths))

    by_split_policy: dict[str, dict[str, int]] = {}
    for row in rows:
        split = str(row.get("split", ""))
        policy = str(row.get("policy", ""))
        by_split_policy.setdefault(split, {}).setdefault(policy, 0)
        by_split_policy[split][policy] += 1

    return {
        "paths": {key: project_relative(path) for key, path in paths.items()},
        "policy_counts": counts,
        "split_policy_counts": by_split_policy,
    }


def index_rows(rows: list[dict[str, str]]) -> dict[str, dict[tuple[str, float, str], dict[str, str]]]:
    indexed: dict[str, dict[tuple[str, float, str], dict[str, str]]] = {}
    for row in rows:
        indexed.setdefault(row["policy"], {})[row_key(row)] = row
    return indexed


def build_two_stage_rows(config: dict[str, Any], source_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    policy_name = str(config["two_stage_policy"]["name"])
    first_policy = str(config["two_stage_policy"]["first_stage_policy"])
    fallback_policy = str(config["two_stage_policy"]["fallback_policy"])
    indexed = index_rows(source_rows)
    split_order = {str(split): idx for idx, split in enumerate(config["splits"])}
    keys = sorted(indexed["m0"], key=lambda item: (split_order.get(item[0], 999), item[1], item[2]))
    rows: list[dict[str, Any]] = []
    for key in keys:
        first = indexed[first_policy][key]
        fallback = indexed[fallback_policy][key]
        if bool_from_csv(first["accept_candidate"]):
            selected = first
            decision = "first_stage_full_strength_top1_consistent"
            fallback_used = False
            fallback_accept: bool | str = ""
        else:
            selected = fallback
            fallback_used = True
            fallback_accept = bool_from_csv(fallback["accept_candidate"])
            if fallback_accept:
                decision = "full_strength_rejected_then_fixed_schedule_accepted"
            else:
                decision = "full_strength_rejected_then_fixed_schedule_rejected_m0"

        out = {**selected}
        out["policy"] = policy_name
        out["decision_reason"] = decision
        out["source_policy_for_final"] = selected["policy"]
        out["first_stage_policy"] = first_policy
        out["fallback_policy"] = fallback_policy
        out["first_stage_accept_candidate"] = bool_from_csv(first["accept_candidate"])
        out["fallback_used"] = fallback_used
        out["fallback_accept_candidate"] = fallback_accept
        out["full_strength_candidate"] = first["candidate"]
        out["fixed_schedule_candidate"] = fallback["candidate"]
        rows.append(out)
    return rows


def stack_paths(paths: list[Path]) -> torch.Tensor:
    return torch.stack([load_rgb_tensor(path) for path in paths])


def summarize_rows(
    split: str,
    policy: str,
    snr: float | str,
    rows: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    references = stack_paths([resolve_project_path(row["original"]) for row in rows])
    m0 = stack_paths([resolve_project_path(row["m0_reconstruction"]) for row in rows])
    final = stack_paths([resolve_project_path(row["final_source"]) for row in rows])
    m0_metrics = compute_pair_metrics(references, m0, None, device, batch_size)
    final_metrics = compute_pair_metrics(references, final, None, device, batch_size)

    m0_failure = 1.0 - rate([bool_from_csv(row["m0_matches_original_top1"]) for row in rows])
    final_failure = 1.0 - rate([bool_from_csv(row["final_matches_original_top1"]) for row in rows])
    selected_alpha_values = [row["selected_alpha"] for row in rows if row.get("selected_alpha") not in ("", None)]
    fallback_used = [bool_from_csv(row.get("fallback_used", False)) for row in rows]
    fallback_accepts = [
        bool_from_csv(row.get("fallback_accept_candidate", False))
        for row in rows
        if bool_from_csv(row.get("fallback_used", False))
    ]
    return {
        "split": split,
        "policy": policy,
        "snr_db": snr,
        "num_images": len(rows),
        "accept_rate": rate([bool_from_csv(row["accept_candidate"]) for row in rows]),
        "fallback_rate": 1.0 - rate([bool_from_csv(row["accept_candidate"]) for row in rows]),
        "first_stage_accept_rate": rate([bool_from_csv(row.get("first_stage_accept_candidate", False)) for row in rows]),
        "fallback_stage_rate": rate(fallback_used),
        "fallback_accept_rate_when_used": rate(fallback_accepts),
        "mean_selected_alpha_accepted": mean(
            [
                float(row["selected_alpha"])
                for row in rows
                if bool_from_csv(row["accept_candidate"]) and row.get("selected_alpha") not in ("", None)
            ]
        ),
        "mean_selected_alpha_all_nonfallback": mean([float(value) for value in selected_alpha_values]),
        "m0_failure_rate": m0_failure,
        "final_failure_rate": final_failure,
        "delta_final_failure_vs_m0": final_failure - m0_failure,
        "repair_count": int(sum(bool_from_csv(row["accepted_repair"]) for row in rows)),
        "accepted_new_error_count": int(sum(bool_from_csv(row["accepted_new_error"]) for row in rows)),
        "missed_repair_count": int(sum(bool_from_csv(row["missed_repair"]) for row in rows)),
        "available_repair_count": int(
            sum(
                (not bool_from_csv(row["m0_matches_original_top1"]))
                and bool_from_csv(row["any_candidate_matches_original_top1"])
                for row in rows
            )
        ),
        "protective_reject_count": int(sum(bool_from_csv(row["protective_reject"]) for row in rows)),
        "final_mse": final_metrics["mse"],
        "final_psnr_db": final_metrics["psnr_db"],
        "final_ssim": final_metrics["ssim"],
        "final_ms_ssim": final_metrics["ms_ssim"],
        "final_lpips": "",
        "delta_psnr_vs_m0_db": None
        if final_metrics["psnr_db"] is None or m0_metrics["psnr_db"] is None
        else final_metrics["psnr_db"] - m0_metrics["psnr_db"],
        "delta_lpips_vs_m0": "",
        "delta_ms_ssim_vs_m0": None
        if final_metrics["ms_ssim"] is None or m0_metrics["ms_ssim"] is None
        else final_metrics["ms_ssim"] - m0_metrics["ms_ssim"],
    }


def build_two_stage_summary(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    policy = str(config["two_stage_policy"]["name"])
    summary_rows: list[dict[str, Any]] = []
    for split in config["splits"]:
        split_rows = [row for row in rows if row["split"] == split]
        for snr in config["snrs"]:
            snr_rows = [row for row in split_rows if abs(float(row["snr_db"]) - float(snr)) < 1e-9]
            summary_rows.append(summarize_rows(str(split), policy, float(snr), snr_rows, device, batch_size))
        summary_rows.append(summarize_rows(str(split), policy, "all", split_rows, device, batch_size))
    return summary_rows


def all_row(summary_rows: list[dict[str, Any]], split: str, policy: str) -> dict[str, Any]:
    for row in summary_rows:
        if row.get("split") == split and row.get("policy") == policy and str(row.get("snr_db")) == "all":
            return row
    raise KeyError(f"Missing all-row: {split}/{policy}")


def plot_tradeoff(config: dict[str, Any], summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    splits = [str(item) for item in config["splits"]]
    policies = [str(item) for item in config["comparison_policies"] if str(item) != "m0"]
    labels = {
        "top1_full_strength": "full",
        "fixed_validation_top1_shrink_schedule": "fixed",
        "full_then_fixed_schedule": "2-stage",
        "adaptive_max_top1_consistent_alpha": "adaptive",
        "always_full_strength": "always",
    }
    colors = ["#5875a4", "#cc8963", "#5f9e6e", "#b55d60", "#8172b3"]
    width = 0.16
    x_positions = list(range(len(splits)))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    for idx, policy in enumerate(policies):
        offsets = [x + (idx - (len(policies) - 1) / 2) * width for x in x_positions]
        rows = [all_row(summary_rows, split, policy) for split in splits]
        axes[0].bar(
            offsets,
            [to_float(row["delta_psnr_vs_m0_db"]) for row in rows],
            width=width,
            color=colors[idx % len(colors)],
            label=labels.get(policy, policy),
        )
        axes[1].bar(
            offsets,
            [to_int(row["accepted_new_error_count"]) for row in rows],
            width=width,
            color=colors[idx % len(colors)],
        )
    for ax, title, ylabel in [
        (axes[0], "Quality Gain", "PSNR delta vs M0 (dB)"),
        (axes[1], "Accepted New Errors", "count"),
    ]:
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(splits)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, ncols=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    lines = ["| " + " | ".join(label for _key, label in columns) + " |"]
    lines.append("|" + "|".join(["---" for _ in columns]) + "|")
    for row in rows:
        values = [fmt(row.get(key, "")) for key, _label in columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def make_report(config: dict[str, Any], summary_rows: list[dict[str, Any]], metadata: dict[str, Any]) -> str:
    splits = [str(item) for item in config["splits"]]
    policy = str(config["two_stage_policy"]["name"])
    reference = str(config["two_stage_policy"]["reference_policy"])
    first = str(config["two_stage_policy"]["first_stage_policy"])
    fallback = str(config["two_stage_policy"]["fallback_policy"])
    two_stage_psnr = "/".join(signed(all_row(summary_rows, split, policy)["delta_psnr_vs_m0_db"]) for split in splits)
    two_stage_new_errors = "/".join(str(all_row(summary_rows, split, policy)["accepted_new_error_count"]) for split in splits)
    reference_psnr = "/".join(signed(all_row(summary_rows, split, reference)["delta_psnr_vs_m0_db"]) for split in splits)
    fallback_psnr = "/".join(signed(all_row(summary_rows, split, fallback)["delta_psnr_vs_m0_db"]) for split in splits)
    full_psnr = "/".join(signed(all_row(summary_rows, split, first)["delta_psnr_vs_m0_db"]) for split in splits)

    display_rows: list[dict[str, Any]] = []
    for split in splits:
        for item in config["comparison_policies"]:
            row = all_row(summary_rows, split, str(item))
            display_rows.append(
                {
                    "split": split,
                    "policy": item,
                    "delta_psnr": signed(row["delta_psnr_vs_m0_db"]),
                    "failure_delta": signed(row["delta_final_failure_vs_m0"]),
                    "accept": fmt(row["accept_rate"]),
                    "first_accept": fmt(row.get("first_stage_accept_rate", "")),
                    "fallback_stage": fmt(row.get("fallback_stage_rate", "")),
                    "new_error": row["accepted_new_error_count"],
                    "missed_repair": row["missed_repair_count"],
                }
            )

    per_snr_rows = [
        {
            "split": row["split"],
            "snr_db": row["snr_db"],
            "delta_psnr": signed(row["delta_psnr_vs_m0_db"]),
            "failure_delta": signed(row["delta_final_failure_vs_m0"]),
            "accept": fmt(row["accept_rate"]),
            "first_accept": fmt(row["first_stage_accept_rate"]),
            "fallback_stage": fmt(row["fallback_stage_rate"]),
            "fallback_accept": fmt(row["fallback_accept_rate_when_used"]),
            "new_error": row["accepted_new_error_count"],
            "missed_repair": row["missed_repair_count"],
        }
        for row in summary_rows
        if row.get("policy") == policy and str(row.get("snr_db")) != "all"
    ]

    lines = [
        "# Two-Stage Residual Alpha Policy",
        "",
        "This derived analysis compresses the adaptive alpha evidence into a simpler receiver-side policy.",
        f"It tries `{first}` first; if full strength is not top-1-consistent with M0, it falls back to `{fallback}`.",
        "It reads existing per-sample decisions and image paths, does not rerun a classifier, and intentionally omits LPIPS to avoid loading external weights.",
        "",
        "## Bottom Line",
        "",
        f"- Two-stage PSNR deltas on validation/held-out/test-like: `{two_stage_psnr}` dB.",
        f"- Two-stage accepted new errors on validation/held-out/test-like: `{two_stage_new_errors}`.",
        f"- Full-strength top-1 fallback PSNR deltas: `{full_psnr}` dB.",
        f"- Fixed validation shrink schedule PSNR deltas: `{fallback_psnr}` dB.",
        f"- Exhaustive adaptive max top-1-consistent alpha PSNR deltas: `{reference_psnr}` dB.",
        "- Interpretation: the two-stage policy is semantically safe under the same pseudo-label gate, but it does not recover the PSNR gain of exhaustive adaptive alpha; it is a deployability ablation, not the new strongest candidate.",
        "",
        "## All-Split Policy Summary",
        "",
    ]
    lines += markdown_table(
        display_rows,
        [
            ("split", "Split"),
            ("policy", "Policy"),
            ("delta_psnr", "Delta PSNR"),
            ("failure_delta", "Failure Delta"),
            ("accept", "Accept"),
            ("first_accept", "Full Accept"),
            ("fallback_stage", "Fallback Stage"),
            ("new_error", "New Error"),
            ("missed_repair", "Missed Repair"),
        ],
    )
    lines.extend(["", "## Two-Stage Per-SNR Detail", ""])
    lines += markdown_table(
        per_snr_rows,
        [
            ("split", "Split"),
            ("snr_db", "SNR"),
            ("delta_psnr", "Delta PSNR"),
            ("failure_delta", "Failure Delta"),
            ("accept", "Accept"),
            ("first_accept", "Full Accept"),
            ("fallback_stage", "Fallback Stage"),
            ("fallback_accept", "Fallback Accept"),
            ("new_error", "New Error"),
            ("missed_repair", "Missed Repair"),
        ],
    )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Summary CSV: `{metadata['summary_csv']}`",
            f"- Per-sample CSV: `{metadata['per_sample_csv']}`",
            f"- Policy tradeoff figure: `{metadata['tradeoff_figure']}`",
            f"- Metadata: `{metadata['metadata_json']}`",
            "",
            "## Caveats",
            "",
            "- The semantic signal is still frozen AlexNet pseudo-label agreement on COCO crops.",
            "- This policy is derived from existing alpha-candidate decisions; it is not a learned alpha predictor yet.",
            "- Since LPIPS is omitted in this run, compare PSNR/SSIM/MS-SSIM and semantic counts only for the two-stage row.",
            "",
        ]
    )
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
        "two_stage_policy": config["two_stage_policy"],
        "comparison_policies": config["comparison_policies"],
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

    source_summary = read_csv(resolve_project_path(config["inputs"]["source_summary_csv"]))
    source_rows = read_csv(resolve_project_path(config["inputs"]["source_per_sample_csv"]))
    two_stage_rows = build_two_stage_rows(config, source_rows)

    device = resolve_device(args.device)
    batch_size = int(config["evaluation"]["image_batch_size"])
    two_stage_summary = build_two_stage_summary(config, two_stage_rows, device, batch_size)
    summary_rows: list[dict[str, Any]] = [*source_summary, *two_stage_summary]

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tradeoff_figure = figures_dir / "two_stage_policy_tradeoff.png"
    plot_tradeoff(config, summary_rows, tradeoff_figure)

    summary_csv = output_dir / "summary.csv"
    per_sample_csv = output_dir / "per_sample.csv"
    metadata_json = output_dir / "metadata.json"
    report_path = output_dir / "REPORT.md"
    write_csv(summary_csv, summary_rows)
    write_csv(per_sample_csv, two_stage_rows)
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
        "tradeoff_figure": project_relative(tradeoff_figure),
        "metadata_json": project_relative(metadata_json),
        "device": str(device),
        "lpips": "omitted",
        "proxy_environment_present": proxy_environment_present(),
        "notes": config.get("notes", []),
    }
    save_json(metadata_json, metadata)
    report_path.write_text(make_report(config, summary_rows, metadata), encoding="utf-8")
    print(json.dumps({"output_dir": project_relative(output_dir), "report": project_relative(report_path)}, indent=2))


if __name__ == "__main__":
    main()
