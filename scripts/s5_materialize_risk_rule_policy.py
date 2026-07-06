from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize final PNGs for a selected receiver-side risk-rule policy "
            "from EXP-S4-006 risk-rule sweep decisions."
        )
    )
    parser.add_argument("--config", default="configs/s5_materialize_risk_rule_gate_exp_s4_006.yaml")
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


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def rate(flags: list[bool]) -> float:
    return float(sum(flags) / len(flags)) if flags else 0.0


def decision_key(row: dict[str, Any]) -> tuple[str, float, str]:
    return (str(row["split"]), float(row["snr_db"]), str(row["sample"]))


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_decision_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in [
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
    ]:
        if key in out:
            out[key] = parse_bool(out[key])
    for key in [
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
    ]:
        if key in out:
            out[key] = parse_float(out[key])
    return out


def validate_inputs(config: dict[str, Any]) -> dict[str, str]:
    paths = {
        "policy_decisions_csv": resolve_project_path(config["inputs"]["policy_decisions_csv"]),
        "policy_summary_csv": resolve_project_path(config["inputs"]["policy_summary_csv"]),
        "source_risk_rule_config": resolve_project_path(config["inputs"]["source_risk_rule_config"]),
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


def read_decisions(path: Path) -> list[dict[str, Any]]:
    return [normalize_decision_row(row) for row in read_csv(path)]


def build_policy_index(rows: list[dict[str, Any]], policy: str) -> dict[tuple[str, float, str], dict[str, Any]]:
    index: dict[tuple[str, float, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("policy") != policy:
            continue
        key = decision_key(row)
        if key in index:
            raise RuntimeError(f"Duplicate decision row for {policy}: {key}")
        index[key] = row
    return index


def check_source_images(rows: list[dict[str, Any]]) -> list[str]:
    missing: list[str] = []
    for row in rows:
        source_key = "refined" if bool(row["accept_refined"]) else "m0_reconstruction"
        for key in ["original", "m0_reconstruction", "refined", source_key]:
            path = resolve_project_path(row[key])
            if not path.exists():
                missing.append(project_relative(path))
    return sorted(set(missing))


def materialize_rows(
    selected_rows: list[dict[str, Any]],
    top1_index: dict[tuple[str, float, str], dict[str, Any]],
    config: dict[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    per_sample: list[dict[str, Any]] = []
    for row in selected_rows:
        key = decision_key(row)
        if key not in top1_index:
            raise RuntimeError(f"Missing top1_equal reference row for {key}")
        top1_row = top1_index[key]

        source_key = "refined" if bool(row["accept_refined"]) else "m0_reconstruction"
        source_path = resolve_project_path(row[source_key])
        if not source_path.exists():
            raise FileNotFoundError(f"Final source image not found: {source_path}")

        final_dir = output_dir / "exports" / str(row["split"]) / snr_name(float(row["snr_db"])) / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / str(row["sample"])
        shutil.copy2(source_path, final_path)

        selected_failure = not bool(row["final_matches_original_top1"])
        top1_failure = not bool(top1_row["final_matches_original_top1"])
        out = dict(row)
        out.update(
            {
                "materialized_policy": str(config["policy"]["name"]),
                "materialized_output_kind": "accepted_refined" if bool(row["accept_refined"]) else "fallback_m0",
                "materialized_final": project_relative(final_path),
                "materialized_source": project_relative(source_path),
                "top1_equal_accept_refined": bool(top1_row["accept_refined"]),
                "top1_equal_final_matches_original_top1": bool(top1_row["final_matches_original_top1"]),
                "top1_equal_final_psnr_db": float(top1_row["final_psnr_db"]),
                "delta_final_psnr_vs_top1_equal_db": float(row["final_psnr_db"]) - float(top1_row["final_psnr_db"]),
                "delta_final_failure_vs_top1_equal": int(selected_failure) - int(top1_failure),
            }
        )
        per_sample.append(out)
    return per_sample


def summarize(rows: list[dict[str, Any]], level: str, split: str, snr: str) -> dict[str, Any]:
    final_matches = [bool(row["final_matches_original_top1"]) for row in rows]
    top1_matches = [bool(row["top1_equal_final_matches_original_top1"]) for row in rows]
    accept_flags = [bool(row["accept_refined"]) for row in rows]
    top1_accept_flags = [bool(row["top1_equal_accept_refined"]) for row in rows]
    return {
        "level": level,
        "split": split,
        "snr_db": snr,
        "num_images": len(rows),
        "accept_count": sum(accept_flags),
        "accept_rate": rate(accept_flags),
        "top1_equal_accept_count": sum(top1_accept_flags),
        "top1_equal_accept_rate": rate(top1_accept_flags),
        "new_accept_vs_top1_count": sum(bool(row["new_accept_vs_top1"]) for row in rows),
        "vetoed_candidate_accept_count": sum(bool(row["vetoed_candidate_accept"]) for row in rows),
        "shadow_veto_count": sum(bool(row["shadow_veto"]) for row in rows),
        "vetoed_candidate_repair_count": sum(bool(row["vetoed_candidate_repair"]) for row in rows),
        "vetoed_candidate_new_error_count": sum(bool(row["vetoed_candidate_new_error"]) for row in rows),
        "final_failure_rate": 1.0 - rate(final_matches),
        "top1_equal_final_failure_rate": 1.0 - rate(top1_matches),
        "delta_final_failure_vs_top1_equal": (1.0 - rate(final_matches)) - (1.0 - rate(top1_matches)),
        "accepted_repair_count": sum(bool(row["accepted_repair"]) for row in rows),
        "accepted_repair_rate": rate([bool(row["accepted_repair"]) for row in rows]),
        "accepted_new_error_count": sum(bool(row["accepted_new_error"]) for row in rows),
        "accepted_new_error_rate": rate([bool(row["accepted_new_error"]) for row in rows]),
        "missed_repair_count": sum(bool(row["missed_repair"]) for row in rows),
        "protective_reject_count": sum(bool(row["protective_reject"]) for row in rows),
        "final_psnr_db": mean([float(row["final_psnr_db"]) for row in rows]),
        "top1_equal_final_psnr_db": mean([float(row["top1_equal_final_psnr_db"]) for row in rows]),
        "delta_final_psnr_vs_top1_equal_db": mean(
            [float(row["delta_final_psnr_vs_top1_equal_db"]) for row in rows]
        ),
        "new_accept_clip_sim_m0_refined_mean": mean(
            [float(row["clip_sim_m0_refined"]) for row in rows if bool(row["new_accept_vs_top1"])]
        ),
        "new_accept_clip_sim_m0_refined_min": min(
            [float(row["clip_sim_m0_refined"]) for row in rows if bool(row["new_accept_vs_top1"])],
            default="",
        ),
    }


def make_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = [summarize(rows, "all", "all", "all")]
    for split in sorted({str(row["split"]) for row in rows}):
        split_rows = [row for row in rows if str(row["split"]) == split]
        output.append(summarize(split_rows, "split", split, "all"))
        for snr in sorted({float(row["snr_db"]) for row in split_rows}):
            subset = [row for row in split_rows if float(row["snr_db"]) == snr]
            output.append(summarize(subset, "split_snr", split, snr_name(snr)))
    return output


def serialize_value(value: Any) -> Any:
    if isinstance(value, bool):
        return bool_text(value)
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


def load_rgb(path: Path, size: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)


def make_grid(rows: list[dict[str, Any]], output_path: Path, count: int) -> None:
    if not rows:
        return
    rows = rows[:count]
    tile = 160
    label_height = 44
    cols = 4
    canvas = Image.new("RGB", (tile * cols, (tile + label_height) * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for row_index, row in enumerate(rows):
        images = [
            ("original", resolve_project_path(row["original"])),
            ("m0", resolve_project_path(row["m0_reconstruction"])),
            ("refined", resolve_project_path(row["refined"])),
            ("final", resolve_project_path(row["materialized_final"])),
        ]
        y = row_index * (tile + label_height)
        for col, (label, path) in enumerate(images):
            x = col * tile
            canvas.paste(load_rgb(path, tile), (x, y + label_height))
            draw.text((x + 4, y + 4), label, fill=(0, 0, 0), font=font)
        detail = (
            f"{row['split']} {row['sample']} {snr_name(float(row['snr_db']))} "
            f"accept={bool_text(bool(row['accept_refined']))} "
            f"repair={bool_text(bool(row['accepted_repair']))} "
            f"newerr={bool_text(bool(row['accepted_new_error']))} "
            f"shadow={bool_text(bool(row['shadow_veto']))}"
        )
        draw.text((4, y + 18), detail[:115], fill=(0, 0, 0), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (str(row["split"]), float(row["snr_db"]), str(row["sample"])))


def write_grids(rows: list[dict[str, Any]], config: dict[str, Any], output_dir: Path) -> dict[str, str]:
    count = int(config["evaluation"]["sample_grid_count"])
    sample_dir = output_dir / "samples"
    manifest: dict[str, str] = {}
    for split in sorted({str(row["split"]) for row in rows}):
        split_rows = sort_rows([row for row in rows if str(row["split"]) == split])
        groups = {
            "overview": split_rows,
            "accepted_repairs": [row for row in split_rows if bool(row["accepted_repair"])],
            "vetoed_candidate_new_errors": [
                row for row in split_rows if bool(row["vetoed_candidate_new_error"])
            ],
            "shadow_vetoes": [row for row in split_rows if bool(row["shadow_veto"])],
        }
        for name, subset in groups.items():
            path = sample_dir / f"{split}_{name}.png"
            make_grid(subset, path, count)
            if subset:
                manifest[f"{split}_{name}"] = project_relative(path)
    return manifest


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


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


def make_report(summary_rows: list[dict[str, Any]], metadata: dict[str, Any]) -> str:
    split_rows = [row for row in summary_rows if row["level"] == "split"]
    lines = [
        "# EXP-S4-006 Selected Risk-Rule Gate Candidate Outputs",
        "",
        "This derived artifact materializes final PNGs for `selected_risk_rule` from the risk-rule sweep decisions.",
        "",
        "No model is trained and no semantic model is recomputed here. The script only copies existing M0/refined PNGs according to saved receiver-side decisions.",
        "",
        "The policy remains a pseudo-label validation/held-out candidate, not final M3.",
        "",
        "## Split Summary",
        "",
        "| Split | Images | Failure | Delta failure vs top-1 | PSNR | Delta PSNR vs top-1 | Repairs | New errors | Shadow veto |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in split_rows:
        lines.append(
            "| "
            f"{row['split']} | "
            f"{int(row['num_images'])} | "
            f"{float(row['final_failure_rate']):.4f} | "
            f"{float(row['delta_final_failure_vs_top1_equal']):+.4f} | "
            f"{float(row['final_psnr_db']):.4f} dB | "
            f"{float(row['delta_final_psnr_vs_top1_equal_db']):+.4f} dB | "
            f"{int(row['accepted_repair_count'])} | "
            f"{int(row['accepted_new_error_count'])} | "
            f"{int(row['shadow_veto_count'])} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Per-sample CSV: `{metadata['per_sample_csv']}`",
            f"- Summary CSV: `{metadata['summary_csv']}`",
            f"- Final image root: `{metadata['final_root']}`",
            f"- Metadata: `{metadata['metadata_json']}`",
            f"- Sample grids: `{metadata['sample_dir']}`",
            "",
            "## Policy",
            "",
            "- Baseline top-1 agreement accepts refined.",
            "- Confidence-gain new accepts require `CLIP(M0, refined) >= 0.90`.",
            "- Shadow-margin veto falls back to M0 when M0 top-1 remains rank <= 2 in refined top-5, M0 margin <= 0.07, and refined top-1 margin >= 0.05.",
            "",
        ]
    )
    return "\n".join(lines)


def count_by_policy(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["policy"])] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    manifest = validate_inputs(config)
    decisions_path = resolve_project_path(config["inputs"]["policy_decisions_csv"])
    all_rows = read_decisions(decisions_path)
    policy_name = str(config["policy"]["name"])
    baseline_name = str(config["policy"]["baseline"])
    selected_rows = sort_rows([row for row in all_rows if row.get("policy") == policy_name])
    top1_index = build_policy_index(all_rows, baseline_name)
    if not selected_rows:
        raise RuntimeError(f"No rows found for policy: {policy_name}")
    missing = check_source_images(selected_rows)
    if missing:
        raise FileNotFoundError("Missing source images:\n" + "\n".join(missing[:20]))

    output_dir = resolve_project_path(args.output_dir or config["outputs"]["output_dir"])
    dry_run_payload = {
        "status": "ok",
        "policy": policy_name,
        "num_selected_rows": len(selected_rows),
        "policy_row_counts": count_by_policy(all_rows),
        "splits": {
            split: sum(1 for row in selected_rows if str(row["split"]) == split)
            for split in sorted({str(row["split"]) for row in selected_rows})
        },
        "output_dir": project_relative(output_dir),
        "manifest": manifest,
    }
    if args.dry_run:
        print(json.dumps(dry_run_payload, indent=2, ensure_ascii=False))
        return

    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory already exists and is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")

    per_sample = materialize_rows(selected_rows, top1_index, config, output_dir)
    summary_rows = make_summary(per_sample)
    sample_grids = write_grids(per_sample, config, output_dir)
    per_sample_csv = output_dir / "per_sample.csv"
    summary_csv = output_dir / "summary.csv"
    metadata_json = output_dir / "metadata.json"
    report_md = output_dir / "REPORT.md"
    write_csv(per_sample_csv, per_sample)
    write_csv(summary_csv, summary_rows)
    metadata = {
        "project_version": get_project_version(),
        "config": project_relative(config_path),
        "source_inputs": manifest,
        "output_dir": project_relative(output_dir),
        "per_sample_csv": project_relative(per_sample_csv),
        "summary_csv": project_relative(summary_csv),
        "metadata_json": project_relative(metadata_json),
        "report_md": project_relative(report_md),
        "final_root": project_relative(output_dir / "exports"),
        "sample_dir": project_relative(output_dir / "samples"),
        "sample_grids": sample_grids,
        "run_command": " ".join(sys.argv),
        "policy": config["policy"],
        "dry_run_payload": dry_run_payload,
        "num_rows": len(per_sample),
        "python_version": platform.python_version(),
        "proxy_environment_present": sorted(key for key in os.environ if "proxy" in key.lower()),
        "download_note": "No model or data download is required; this script only copies existing PNGs and evaluates saved policy decisions.",
    }
    save_json(metadata_json, metadata)
    report_md.write_text(make_report(summary_rows, metadata), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": project_relative(output_dir),
                "num_rows": len(per_sample),
                "summary_csv": project_relative(summary_csv),
                "report_md": project_relative(report_md),
                "final_root": metadata["final_root"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
