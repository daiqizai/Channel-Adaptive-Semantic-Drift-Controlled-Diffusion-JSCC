from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize M1 blind diffusion negative results across image and semantic metrics.")
    parser.add_argument("--m1-metrics", default="outputs/EXP-S2-002/metrics.json")
    parser.add_argument("--clip-metrics", default="outputs/EXP-S3-001/metrics.json")
    parser.add_argument("--classifier-metrics", default="outputs/EXP-S3-002/metrics.json")
    parser.add_argument("--output-dir", default="outputs/analysis/m1_negative_result_summary")
    parser.add_argument("--overwrite", action="store_true")
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


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def by_snr(payload: dict[str, Any]) -> dict[float, dict[str, Any]]:
    return {float(row["snr_db"]): row for row in payload["results"]}


def metric(row: dict[str, Any], group: str, key: str) -> float:
    return float(row[group][key])


def summary_rows(
    m1_payload: dict[str, Any],
    clip_payload: dict[str, Any],
    classifier_payload: dict[str, Any],
) -> list[dict[str, float]]:
    m1_by_snr = by_snr(m1_payload)
    clip_by_snr = by_snr(clip_payload)
    cls_by_snr = by_snr(classifier_payload)
    snrs = sorted(set(m1_by_snr) & set(clip_by_snr) & set(cls_by_snr))
    rows: list[dict[str, float]] = []
    for snr in snrs:
        m1 = m1_by_snr[snr]
        clip = clip_by_snr[snr]
        cls = cls_by_snr[snr]
        m0_img = m1["m0_reconstruction_vs_original"]
        m1_img = m1["m1_refined_vs_original"]
        clip_summary = clip["summary"]
        cls_all = cls["pseudo_label_consistency"]["all"]
        cls_conf03 = cls["pseudo_label_consistency"]["original_conf_ge_0p3"]
        rows.append(
            {
                "snr_db": float(snr),
                "num_images": float(m1["num_images"]),
                "m0_psnr_db": float(m0_img["psnr_db"]),
                "m1_psnr_db": float(m1_img["psnr_db"]),
                "delta_psnr_m1_minus_m0": float(m1_img["psnr_db"]) - float(m0_img["psnr_db"]),
                "m0_ssim": float(m0_img["ssim"]),
                "m1_ssim": float(m1_img["ssim"]),
                "delta_ssim_m1_minus_m0": float(m1_img["ssim"]) - float(m0_img["ssim"]),
                "m0_ms_ssim": float(m0_img["ms_ssim"]),
                "m1_ms_ssim": float(m1_img["ms_ssim"]),
                "delta_ms_ssim_m1_minus_m0": float(m1_img["ms_ssim"]) - float(m0_img["ms_ssim"]),
                "m0_lpips": float(m0_img["lpips"]),
                "m1_lpips": float(m1_img["lpips"]),
                "delta_lpips_m1_minus_m0": float(m1_img["lpips"]) - float(m0_img["lpips"]),
                "clip_sim_original_m0": float(clip_summary["clip_sim_original_m0"]["mean"]),
                "clip_sim_original_m1": float(clip_summary["clip_sim_original_m1"]["mean"]),
                "clip_drop_m0_minus_m1": float(clip_summary["clip_drop_m0_minus_m1"]["mean"]),
                "clip_m1_lower_rate": float(clip["diagnostic_rates"]["m1_less_similar_than_m0"]),
                "clip_drop_ge_0p10_rate": float(clip["diagnostic_rates"]["drop_ge_0p1"]),
                "cls_all_m0_match_original_top1": float(cls_all["m0_matches_original_top1"]),
                "cls_all_m1_match_original_top1": float(cls_all["m1_matches_original_top1"]),
                "cls_all_m1_pseudo_drift_origin": float(cls_all["m1_pseudo_drift_origin"]),
                "cls_all_m1_refinement_drift": float(cls_all["m1_refinement_drift"]),
                "cls_conf03_num_images": float(cls_conf03["num_images"]),
                "cls_conf03_m0_match_original_top1": float(cls_conf03["m0_matches_original_top1"]),
                "cls_conf03_m1_match_original_top1": float(cls_conf03["m1_matches_original_top1"]),
                "cls_conf03_m1_pseudo_drift_origin": float(cls_conf03["m1_pseudo_drift_origin"]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = [key for key in rows[0] if key not in {"snr_db"}] if rows else []
    return {key: mean([float(row[key]) for row in rows]) for key in keys}


def fmt(value: float) -> str:
    return f"{value:.4f}"


def make_report(rows: list[dict[str, float]], aggregate_row: dict[str, float], sources: dict[str, str]) -> str:
    lines = [
        "# M1 Blind Diffusion Negative Result Summary",
        "",
        "This report is a derived summary over existing experiments. It does not add new model runs.",
        "",
        "## Sources",
        "",
        f"- M1 image metrics: `{sources['m1_metrics']}`",
        f"- CLIP diagnostic: `{sources['clip_metrics']}`",
        f"- Frozen classifier diagnostic: `{sources['classifier_metrics']}`",
        "",
        "## Main Table",
        "",
        "| SNR(dB) | M0 PSNR | M1 PSNR | Delta PSNR | M0 LPIPS | M1 LPIPS | CLIP M0 | CLIP M1 | CLIP drop | M1 cls match | M1 cls drift |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {snr:g} | {m0_psnr} | {m1_psnr} | {dpsnr} | {m0_lpips} | {m1_lpips} | {clip_m0} | {clip_m1} | {clip_drop} | {cls_match} | {cls_drift} |".format(
                snr=row["snr_db"],
                m0_psnr=fmt(row["m0_psnr_db"]),
                m1_psnr=fmt(row["m1_psnr_db"]),
                dpsnr=fmt(row["delta_psnr_m1_minus_m0"]),
                m0_lpips=fmt(row["m0_lpips"]),
                m1_lpips=fmt(row["m1_lpips"]),
                clip_m0=fmt(row["clip_sim_original_m0"]),
                clip_m1=fmt(row["clip_sim_original_m1"]),
                clip_drop=fmt(row["clip_drop_m0_minus_m1"]),
                cls_match=fmt(row["cls_all_m1_match_original_top1"]),
                cls_drift=fmt(row["cls_all_m1_pseudo_drift_origin"]),
            )
        )
    lines.extend(
        [
            "",
            "## Aggregate Over SNRs",
            "",
            f"- Mean PSNR delta M1-M0: `{fmt(aggregate_row['delta_psnr_m1_minus_m0'])}` dB",
            f"- Mean LPIPS delta M1-M0: `{fmt(aggregate_row['delta_lpips_m1_minus_m0'])}`",
            f"- Mean CLIP drop M0-M1: `{fmt(aggregate_row['clip_drop_m0_minus_m1'])}`",
            f"- Mean classifier all-subset M1 match original top-1: `{fmt(aggregate_row['cls_all_m1_match_original_top1'])}`",
            f"- Mean classifier all-subset M1 pseudo drift-origin: `{fmt(aggregate_row['cls_all_m1_pseudo_drift_origin'])}`",
            "",
            "## Interpretation",
            "",
            "- Fixed-strength blind SD img2img is a negative result on this sample set.",
            "- The degradation is consistent across low-level metrics, LPIPS, CLIP image consistency, and frozen-classifier pseudo-label consistency.",
            "- This report should be used as the baseline failure mode motivating SNR-adaptive strength and semantic failure handling.",
            "- CLIP and pseudo-label classifier diagnostics are auxiliary. They do not replace the final clean-correct semantic drift metric required by `MILESTONES.md`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    m1_path = resolve_project_path(args.m1_metrics)
    clip_path = resolve_project_path(args.clip_metrics)
    classifier_path = resolve_project_path(args.classifier_metrics)
    output_dir = resolve_project_path(args.output_dir)

    for path in [m1_path, clip_path, classifier_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required metrics file not found: {path}")
    if output_dir.exists():
        if not args.overwrite and any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory already exists and is non-empty: {output_dir}")
        if args.overwrite:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    m1_payload = load_json(m1_path)
    clip_payload = load_json(clip_path)
    classifier_payload = load_json(classifier_path)
    rows = summary_rows(m1_payload, clip_payload, classifier_payload)
    if not rows:
        raise RuntimeError("No overlapping SNR rows found across metrics files.")
    aggregate_row = aggregate(rows)

    sources = {
        "m1_metrics": project_relative(m1_path),
        "clip_metrics": project_relative(clip_path),
        "classifier_metrics": project_relative(classifier_path),
    }
    payload = {
        "metadata": {
            "note": "Derived summary only; no new model inference or metric computation beyond aggregating existing outputs.",
            "sources": sources,
            "key_sources": ["scripts/s4_summarize_m1_negative_result.py"],
        },
        "rows": rows,
        "aggregate_over_snrs": aggregate_row,
    }
    save_json(output_dir / "summary.json", payload)
    write_csv(output_dir / "summary.csv", rows)
    (output_dir / "REPORT.md").write_text(make_report(rows, aggregate_row, sources), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": project_relative(output_dir),
                "rows": len(rows),
                "report": project_relative(output_dir / "REPORT.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
