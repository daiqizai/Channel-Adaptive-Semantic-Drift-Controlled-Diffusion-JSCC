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
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize receiver-side residual gate candidate outputs from EXP-S4-006 top-k predictions."
    )
    parser.add_argument("--config", default="configs/s5_materialize_conf_gain_gate_exp_s4_006.yaml")
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


def snr_name(snr: float) -> str:
    if float(snr).is_integer():
        return f"snr_{int(snr):02d}db"
    return f"snr_{str(snr).replace('.', 'p')}db"


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def rate(flags: list[bool]) -> float:
    return float(sum(flags) / len(flags)) if flags else 0.0


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["snr_db"] = float(row["snr_db"])
        row["m0_psnr_db"] = float(row["m0_psnr_db"])
        row["refined_psnr_db"] = float(row["refined_psnr_db"])
        row["m0_top1_index"] = int(row["m0_top1_index"])
        row["refined_top1_index"] = int(row["refined_top1_index"])
        row["original_top1_index"] = int(row["original_top1_index"])
        row["m0_top1_prob"] = float(row["m0_top1_prob"])
        row["refined_top1_prob"] = float(row["refined_top1_prob"])
        for key in [
            "m0_matches_original_top1",
            "refined_matches_original_top1",
            "refined_matches_m0_top1",
        ]:
            row[key] = parse_bool(row[key])
    return rows


def validate_inputs(config: dict[str, Any]) -> dict[str, str]:
    paths = {
        "gate_predictions_csv": resolve_project_path(config["inputs"]["gate_predictions_csv"]),
        "aux_audit_summary_csv": resolve_project_path(config["inputs"]["aux_audit_summary_csv"]),
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


def candidate_accept(row: dict[str, Any], margin: float) -> bool:
    return row["refined_top1_index"] == row["m0_top1_index"] or (
        row["refined_top1_prob"] >= row["m0_top1_prob"] + margin
    )


def materialize_rows(rows: list[dict[str, Any]], config: dict[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    margin = float(config["policy"]["refined_conf_gain_margin"])
    per_sample: list[dict[str, Any]] = []
    for row in rows:
        accept = candidate_accept(row, margin)
        source_key = "refined" if accept else "m0_reconstruction"
        source_path = resolve_project_path(row[source_key])
        if not source_path.exists():
            raise FileNotFoundError(f"Final source image not found: {source_path}")
        final_dir = output_dir / "exports" / snr_name(float(row["snr_db"])) / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / str(row["sample"])
        shutil.copy2(source_path, final_path)

        baseline_accept = row["refined_top1_index"] == row["m0_top1_index"]
        m0_ok = bool(row["m0_matches_original_top1"])
        refined_ok = bool(row["refined_matches_original_top1"])
        baseline_final_ok = refined_ok if baseline_accept else m0_ok
        candidate_final_ok = refined_ok if accept else m0_ok
        baseline_psnr = float(row["refined_psnr_db"] if baseline_accept else row["m0_psnr_db"])
        candidate_psnr = float(row["refined_psnr_db"] if accept else row["m0_psnr_db"])
        new_accept = accept and not baseline_accept

        out = dict(row)
        out.update(
            {
                "policy": str(config["policy"]["name"]),
                "baseline_accept_refined": baseline_accept,
                "candidate_accept_refined": accept,
                "newly_accepted_by_candidate": new_accept,
                "candidate_output_kind": "accepted_refined" if accept else "fallback_m0",
                "candidate_final": project_relative(final_path),
                "candidate_final_matches_original_top1": candidate_final_ok,
                "baseline_final_matches_original_top1": baseline_final_ok,
                "candidate_accepted_repair": accept and (not m0_ok) and refined_ok,
                "candidate_accepted_new_error": accept and m0_ok and (not refined_ok),
                "baseline_final_psnr_db": baseline_psnr,
                "candidate_final_psnr_db": candidate_psnr,
                "candidate_delta_psnr_vs_baseline_db": candidate_psnr - baseline_psnr,
                "candidate_delta_psnr_vs_m0_db": candidate_psnr - float(row["m0_psnr_db"]),
            }
        )
        per_sample.append(out)
    return per_sample


def summarize(rows: list[dict[str, Any]], subset: str) -> dict[str, Any]:
    return {
        "subset": subset,
        "num_images": len(rows),
        "candidate_accept_count": sum(bool(row["candidate_accept_refined"]) for row in rows),
        "candidate_accept_rate": rate([bool(row["candidate_accept_refined"]) for row in rows]),
        "new_accept_count": sum(bool(row["newly_accepted_by_candidate"]) for row in rows),
        "candidate_final_failure_rate": 1.0
        - rate([bool(row["candidate_final_matches_original_top1"]) for row in rows]),
        "baseline_final_failure_rate": 1.0
        - rate([bool(row["baseline_final_matches_original_top1"]) for row in rows]),
        "m0_failure_rate": 1.0 - rate([bool(row["m0_matches_original_top1"]) for row in rows]),
        "refined_failure_rate": 1.0 - rate([bool(row["refined_matches_original_top1"]) for row in rows]),
        "accepted_repair_count": sum(bool(row["candidate_accepted_repair"]) for row in rows),
        "accepted_new_error_count": sum(bool(row["candidate_accepted_new_error"]) for row in rows),
        "m0_psnr_db": mean([float(row["m0_psnr_db"]) for row in rows]),
        "refined_psnr_db": mean([float(row["refined_psnr_db"]) for row in rows]),
        "baseline_final_psnr_db": mean([float(row["baseline_final_psnr_db"]) for row in rows]),
        "candidate_final_psnr_db": mean([float(row["candidate_final_psnr_db"]) for row in rows]),
        "candidate_delta_psnr_vs_baseline_db": mean(
            [float(row["candidate_delta_psnr_vs_baseline_db"]) for row in rows]
        ),
        "candidate_delta_psnr_vs_m0_db": mean([float(row["candidate_delta_psnr_vs_m0_db"]) for row in rows]),
    }


def make_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = [summarize(rows, "all")]
    for snr in sorted({float(row["snr_db"]) for row in rows}):
        output.append(summarize([row for row in rows if float(row["snr_db"]) == snr], snr_name(snr)))
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
    label_height = 34
    cols = 4
    canvas = Image.new("RGB", (tile * cols, (tile + label_height) * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for row_index, row in enumerate(rows):
        images = [
            ("original", resolve_project_path(row["original"])),
            ("m0", resolve_project_path(row["m0_reconstruction"])),
            ("refined", resolve_project_path(row["refined"])),
            ("candidate", resolve_project_path(row["candidate_final"])),
        ]
        y = row_index * (tile + label_height)
        for col, (label, path) in enumerate(images):
            x = col * tile
            canvas.paste(load_rgb(path, tile), (x, y + label_height))
            draw.text((x + 4, y + 4), label, fill=(0, 0, 0), font=font)
        detail = f"{row['sample']} snr={float(row['snr_db']):g} accept={bool(row['candidate_accept_refined'])}"
        draw.text((4, y + 18), detail[:110], fill=(0, 0, 0), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def write_grids(rows: list[dict[str, Any]], config: dict[str, Any], output_dir: Path) -> dict[str, str]:
    count = int(config["evaluation"]["sample_grid_count"])
    manifest: dict[str, str] = {}
    sample_dir = output_dir / "samples"
    for snr in sorted({float(row["snr_db"]) for row in rows}):
        subset = [row for row in rows if float(row["snr_db"]) == snr]
        path = sample_dir / f"{snr_name(snr)}_original_m0_refined_candidate.png"
        make_grid(subset, path, count)
        manifest[snr_name(snr)] = project_relative(path)
    risk_rows = [row for row in rows if bool(row["candidate_accepted_new_error"])]
    risk_path = sample_dir / "accepted_new_error_quads.png"
    make_grid(risk_rows, risk_path, max(1, len(risk_rows)))
    if risk_rows:
        manifest["accepted_new_error_quads"] = project_relative(risk_path)
    return manifest


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


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
    all_row = summary_rows[0]
    lines = [
        "# EXP-S4-006 Confidence-Gain Gate Candidate Outputs",
        "",
        "This derived artifact materializes final PNGs for `top1_equal_or_refined_conf_gain_ge_0p05` from existing M0/refined images.",
        "",
        "The policy is validation-tuned and remains a candidate, not final M3.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Candidate accept rate | {float(all_row['candidate_accept_rate']):.4f} |",
        f"| Newly accepted by candidate | {int(all_row['new_accept_count'])} |",
        f"| Candidate final failure | {float(all_row['candidate_final_failure_rate']):.4f} |",
        f"| Baseline final failure | {float(all_row['baseline_final_failure_rate']):.4f} |",
        f"| Candidate final PSNR | {float(all_row['candidate_final_psnr_db']):.4f} dB |",
        f"| Delta PSNR vs baseline | {float(all_row['candidate_delta_psnr_vs_baseline_db']):+.4f} dB |",
        f"| Delta PSNR vs M0 | {float(all_row['candidate_delta_psnr_vs_m0_db']):+.4f} dB |",
        f"| Accepted repairs | {int(all_row['accepted_repair_count'])} |",
        f"| Accepted new errors | {int(all_row['accepted_new_error_count'])} |",
        "",
        "## Files",
        "",
        f"- Per-sample CSV: `{metadata['per_sample_csv']}`",
        f"- Summary CSV: `{metadata['summary_csv']}`",
        f"- Final image root: `{metadata['final_root']}`",
        f"- Metadata: `{metadata['metadata_json']}`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    manifest = validate_inputs(config)
    rows = read_rows(resolve_project_path(config["inputs"]["gate_predictions_csv"]))
    if args.dry_run:
        print(json.dumps({"status": "ok", "num_rows": len(rows), "manifest": manifest}, indent=2))
        return

    output_dir = resolve_project_path(args.output_dir or config["outputs"]["output_dir"])
    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory already exists and is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")

    per_sample = materialize_rows(rows, config, output_dir)
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
        "output_dir": project_relative(output_dir),
        "per_sample_csv": project_relative(per_sample_csv),
        "summary_csv": project_relative(summary_csv),
        "metadata_json": project_relative(metadata_json),
        "report_md": project_relative(report_md),
        "final_root": project_relative(output_dir / "exports"),
        "sample_grids": sample_grids,
        "run_command": " ".join(sys.argv),
        "policy": config["policy"],
        "num_rows": len(per_sample),
        "python_version": platform.python_version(),
        "proxy_environment_present": sorted(key for key in os.environ if "proxy" in key.lower()),
        "download_note": "No model or data download is required; this script only copies existing PNGs and evaluates saved predictions.",
    }
    save_json(metadata_json, metadata)
    report_md.write_text(make_report(summary_rows, metadata), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": project_relative(output_dir),
                "num_rows": len(per_sample),
                "final_root": metadata["final_root"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
