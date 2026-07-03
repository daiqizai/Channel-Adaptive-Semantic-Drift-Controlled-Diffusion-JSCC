from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a visual project progress summary from existing outputs.")
    parser.add_argument("--output-dir", default="outputs/analysis/project_progress_visual_summary")
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


def load_json(path: str | Path) -> Any:
    with resolve_project_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite and any(path.iterdir()):
            raise FileExistsError(f"Output directory already exists and is non-empty: {path}")
        if overwrite:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def result_by_snr(payload: dict[str, Any]) -> dict[float, dict[str, Any]]:
    return {float(item["snr_db"]): item for item in payload["results"]}


def build_stage_rows() -> list[dict[str, Any]]:
    return [
        {"stage": "S1", "label": "CIFAR sanity", "status": "done", "progress": 1.0},
        {"stage": "S2-HR", "label": "COCO DeepJSCC", "status": "done-best-pt", "progress": 1.0},
        {"stage": "S3", "label": "Blind diffusion", "status": "done-negative", "progress": 1.0},
        {"stage": "S4", "label": "Semantic metrics", "status": "partial-auxiliary", "progress": 0.72},
        {"stage": "S5", "label": "Adaptive control", "status": "not-started", "progress": 0.0},
        {"stage": "S6", "label": "Full experiment", "status": "not-started", "progress": 0.0},
    ]


def extract_metrics() -> dict[str, Any]:
    s1 = load_json("outputs/EXP-S1-001/metrics.json")
    m0 = load_json("outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export/metrics.json")
    m1 = load_json("outputs/EXP-S2-002/metrics.json")
    clip = load_json("outputs/EXP-S3-001/metrics.json")
    classifier = load_json("outputs/EXP-S3-002/metrics.json")
    caption = load_json("outputs/EXP-S3-003/metrics.json")
    negative = load_json("outputs/analysis/m1_negative_result_summary/summary.json")
    return {
        "s1": s1,
        "m0": m0,
        "m1": m1,
        "clip": clip,
        "classifier": classifier,
        "caption": caption,
        "negative": negative,
    }


def build_tables(payloads: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    s1_rows = [
        {
            "dataset": "CIFAR-10",
            "method": "M0-DeepJSCC",
            "snr_db": float(row["snr_db"]),
            "psnr_db": float(row["psnr_db"]),
            "ssim": float(row["ssim"]),
        }
        for row in payloads["s1"]["rows"]
    ]
    m0_rows = [
        {
            "dataset": "COCO2017-val",
            "method": "M0-DeepJSCC-HR",
            "snr_db": float(row["snr_db"]),
            "psnr_db": float(row["psnr_db"]),
            "ssim": float(row["ssim"]),
            "ms_ssim": float(row["ms_ssim"]),
            "inference_time_ms_per_image": float(row["inference_time_ms_per_image"]),
        }
        for row in payloads["m0"]["results"]
    ]

    m1_rows = []
    clip_by = result_by_snr(payloads["clip"])
    cls_by = result_by_snr(payloads["classifier"])
    cap_by = result_by_snr(payloads["caption"])
    for row in payloads["m1"]["results"]:
        snr = float(row["snr_db"])
        m0_img = row["m0_reconstruction_vs_original"]
        m1_img = row["m1_refined_vs_original"]
        clip_row = clip_by[snr]
        cls_row = cls_by[snr]
        cap_row = cap_by[snr]
        cls_all = cls_row["pseudo_label_consistency"]["all"]
        cls_conf03 = cls_row["pseudo_label_consistency"]["original_conf_ge_0p3"]
        m1_rows.append(
            {
                "snr_db": snr,
                "num_images": int(row["num_images"]),
                "m0_psnr_db": float(m0_img["psnr_db"]),
                "m1_psnr_db": float(m1_img["psnr_db"]),
                "delta_psnr_m1_minus_m0": float(m1_img["psnr_db"]) - float(m0_img["psnr_db"]),
                "m0_ssim": float(m0_img["ssim"]),
                "m1_ssim": float(m1_img["ssim"]),
                "m0_ms_ssim": float(m0_img["ms_ssim"]),
                "m1_ms_ssim": float(m1_img["ms_ssim"]),
                "m0_lpips": float(m0_img["lpips"]),
                "m1_lpips": float(m1_img["lpips"]),
                "delta_lpips_m1_minus_m0": float(m1_img["lpips"]) - float(m0_img["lpips"]),
                "clip_sim_original_m0": float(clip_row["summary"]["clip_sim_original_m0"]["mean"]),
                "clip_sim_original_m1": float(clip_row["summary"]["clip_sim_original_m1"]["mean"]),
                "clip_drop_m0_minus_m1": float(clip_row["summary"]["clip_drop_m0_minus_m1"]["mean"]),
                "classifier_all_m0_match_original": float(cls_all["m0_matches_original_top1"]),
                "classifier_all_m1_match_original": float(cls_all["m1_matches_original_top1"]),
                "classifier_all_m1_drift_origin": float(cls_all["m1_pseudo_drift_origin"]),
                "classifier_conf03_n": int(cls_conf03["num_images"]),
                "classifier_conf03_m0_match_original": float(cls_conf03["m0_matches_original_top1"]),
                "classifier_conf03_m1_match_original": float(cls_conf03["m1_matches_original_top1"]),
                "caption_max_original": float(cap_row["summary"]["clip_text_sim_caption_max_original"]["mean"]),
                "caption_max_m0": float(cap_row["summary"]["clip_text_sim_caption_max_m0"]["mean"]),
                "caption_max_m1": float(cap_row["summary"]["clip_text_sim_caption_max_m1"]["mean"]),
                "caption_drop_m0_minus_m1": float(cap_row["summary"]["clip_text_drop_max_m0_minus_m1"]["mean"]),
            }
        )
    return {"stage": build_stage_rows(), "s1": s1_rows, "m0": m0_rows, "m1": m1_rows}


def style_axes(ax, title: str, ylabel: str | None = None, xlabel: str | None = None) -> None:
    ax.set_title(title, fontsize=13, pad=10)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(True, linestyle="--", alpha=0.32)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_stage_progress(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [f"{row['stage']}\n{row['label']}" for row in rows]
    values = [float(row["progress"]) for row in rows]
    colors = ["#2f7f67", "#2f7f67", "#b4554a", "#d08a2e", "#b8b8b8", "#b8b8b8"]
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.bar(labels, values, color=colors, width=0.62)
    ax.set_ylim(0, 1.08)
    style_axes(ax, "Project Stage Progress", "completion proxy")
    for idx, row in enumerate(rows):
        ax.text(idx, values[idx] + 0.035, str(row["status"]), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_snr_curves(tables: dict[str, list[dict[str, Any]]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    s1 = sorted(tables["s1"], key=lambda row: row["snr_db"])
    m0 = sorted(tables["m0"], key=lambda row: row["snr_db"])
    axes[0].plot([r["snr_db"] for r in s1], [r["psnr_db"] for r in s1], marker="o", label="CIFAR-10 M0")
    axes[0].plot([r["snr_db"] for r in m0], [r["psnr_db"] for r in m0], marker="o", label="COCO-256 M0")
    style_axes(axes[0], "M0 PSNR vs SNR", "PSNR (dB)", "SNR (dB)")
    axes[0].legend(frameon=False)
    axes[1].plot([r["snr_db"] for r in s1], [r["ssim"] for r in s1], marker="o", label="CIFAR-10 M0")
    axes[1].plot([r["snr_db"] for r in m0], [r["ssim"] for r in m0], marker="o", label="COCO-256 M0")
    style_axes(axes[1], "M0 SSIM vs SNR", "SSIM", "SNR (dB)")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_m1_quality(rows: list[dict[str, Any]], path: Path) -> None:
    rows = sorted(rows, key=lambda row: row["snr_db"])
    x = np.arange(len(rows))
    width = 0.36
    labels = [f"{row['snr_db']:g} dB" for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    metrics = [
        ("psnr_db", "PSNR (dB)", "Higher is better", "m0_psnr_db", "m1_psnr_db"),
        ("ms_ssim", "MS-SSIM", "Higher is better", "m0_ms_ssim", "m1_ms_ssim"),
        ("lpips", "LPIPS", "Lower is better", "m0_lpips", "m1_lpips"),
        ("ssim", "SSIM", "Higher is better", "m0_ssim", "m1_ssim"),
    ]
    for ax, (_metric, ylabel, title_suffix, m0_key, m1_key) in zip(axes.flat, metrics):
        ax.bar(x - width / 2, [row[m0_key] for row in rows], width, label="M0", color="#2f7f67")
        ax.bar(x + width / 2, [row[m1_key] for row in rows], width, label="M1", color="#b4554a")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        style_axes(ax, f"M0 vs M1 {ylabel} ({title_suffix})", ylabel, "SNR")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_semantic_diagnostics(rows: list[dict[str, Any]], path: Path) -> None:
    rows = sorted(rows, key=lambda row: row["snr_db"])
    x = np.arange(len(rows))
    width = 0.36
    labels = [f"{row['snr_db']:g} dB" for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))

    axes[0, 0].bar(x - width / 2, [row["clip_sim_original_m0"] for row in rows], width, label="M0", color="#2f7f67")
    axes[0, 0].bar(x + width / 2, [row["clip_sim_original_m1"] for row in rows], width, label="M1", color="#b4554a")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels)
    axes[0, 0].legend(frameon=False)
    style_axes(axes[0, 0], "CLIP Image-Image Similarity to Original", "cosine", "SNR")

    axes[0, 1].bar(x - width / 2, [row["caption_max_m0"] for row in rows], width, label="M0", color="#2f7f67")
    axes[0, 1].bar(x + width / 2, [row["caption_max_m1"] for row in rows], width, label="M1", color="#b4554a")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels)
    axes[0, 1].legend(frameon=False)
    style_axes(axes[0, 1], "COCO Caption CLIP Image-Text Similarity", "caption-max cosine", "SNR")

    axes[1, 0].bar(x - width / 2, [row["classifier_all_m0_match_original"] for row in rows], width, label="M0", color="#2f7f67")
    axes[1, 0].bar(x + width / 2, [row["classifier_all_m1_match_original"] for row in rows], width, label="M1", color="#b4554a")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels)
    axes[1, 0].legend(frameon=False)
    style_axes(axes[1, 0], "Frozen Classifier Pseudo-Label Match", "match rate", "SNR")

    axes[1, 1].bar(x, [row["classifier_all_m1_drift_origin"] for row in rows], width=0.55, color="#b4554a", label="M1")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels)
    axes[1, 1].set_ylim(0, 1.05)
    style_axes(axes[1, 1], "M1 Pseudo-Label Drift-Origin", "drift rate", "SNR")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_negative_deltas(rows: list[dict[str, Any]], path: Path) -> None:
    rows = sorted(rows, key=lambda row: row["snr_db"])
    labels = [f"{row['snr_db']:g} dB" for row in rows]
    x = np.arange(len(rows))
    fig, ax1 = plt.subplots(figsize=(9.5, 4.8))
    ax1.bar(x - 0.18, [row["delta_psnr_m1_minus_m0"] for row in rows], width=0.36, color="#b4554a", label="Delta PSNR")
    ax1.set_ylabel("Delta PSNR M1-M0 (dB)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.axhline(0, color="#333333", linewidth=1)
    ax2 = ax1.twinx()
    ax2.plot(x + 0.18, [row["delta_lpips_m1_minus_m0"] for row in rows], marker="o", color="#4c78a8", label="Delta LPIPS")
    ax2.set_ylabel("Delta LPIPS M1-M0")
    style_axes(ax1, "Blind Diffusion Degradation Deltas", None, "SNR")
    lines, labels_a = ax1.get_legend_handles_labels()
    lines2, labels_b = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels_a + labels_b, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def copy_representative_assets(output_dir: Path) -> list[dict[str, str]]:
    assets = [
        ("m1_snr01_grid", "outputs/EXP-S2-002/samples/snr_01db_original_reconstruction_refined.png"),
        ("m1_snr07_grid", "outputs/EXP-S2-002/samples/snr_07db_original_reconstruction_refined.png"),
        ("m1_snr19_grid", "outputs/EXP-S2-002/samples/snr_19db_original_reconstruction_refined.png"),
        ("clip_global_failures", "outputs/EXP-S3-001/failure_cases/sheets/global_top_clip_drop.png"),
        ("classifier_global_failures", "outputs/EXP-S3-002/failure_cases/sheets/global_top_classifier_drift.png"),
        ("caption_global_failures", "outputs/EXP-S3-003/failure_cases/sheets/global_top_caption_clip_drop.png"),
    ]
    copied = []
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    for name, src in assets:
        src_path = resolve_project_path(src)
        if not src_path.exists():
            continue
        dst = assets_dir / f"{name}{src_path.suffix}"
        shutil.copy2(src_path, dst)
        copied.append({"name": name, "source": project_relative(src_path), "copy": project_relative(dst)})
    return copied


def make_thumbnail(path: Path, width: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    ratio = width / image.width
    height = max(1, int(image.height * ratio))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def make_overview_sheet(copied_assets: list[dict[str, str]], output_path: Path) -> None:
    tile_w = 420
    margin = 22
    gap = 18
    header_h = 64
    title_font = load_font(22)
    label_font = load_font(14)
    thumbs: list[tuple[str, Image.Image]] = []
    for item in copied_assets:
        image = make_thumbnail(resolve_project_path(item["copy"]), tile_w)
        if image.height > 780:
            image = image.crop((0, 0, image.width, 780))
        thumbs.append((item["name"], image))
    if not thumbs:
        return
    cols = 2
    rows = int(np.ceil(len(thumbs) / cols))
    row_heights = []
    for row in range(rows):
        row_items = thumbs[row * cols : (row + 1) * cols]
        row_heights.append(max(image.height for _name, image in row_items) + 34)
    width = margin * 2 + cols * tile_w + (cols - 1) * gap
    height = header_h + sum(row_heights) + gap * (rows - 1) + margin
    canvas = Image.new("RGB", (width, height), (242, 242, 242))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 18), "Project Visual Summary: representative outputs", font=title_font, fill=(20, 20, 20))
    y = header_h
    for row in range(rows):
        row_items = thumbs[row * cols : (row + 1) * cols]
        for col, (name, image) in enumerate(row_items):
            x = margin + col * (tile_w + gap)
            draw.text((x, y), name, font=label_font, fill=(35, 35, 35))
            canvas.paste(image, (x, y + 24))
            draw.rectangle((x, y + 24, x + image.width - 1, y + 24 + image.height - 1), outline=(205, 205, 205))
        y += row_heights[row] + gap
    canvas.save(output_path)


def fmt(value: float) -> str:
    return f"{value:.4f}"


def make_report(tables: dict[str, list[dict[str, Any]]], copied_assets: list[dict[str, str]]) -> str:
    m1_rows = sorted(tables["m1"], key=lambda row: row["snr_db"])
    m0_rows = sorted(tables["m0"], key=lambda row: row["snr_db"])
    lines = [
        "# Project Progress Visual Summary",
        "",
        "This is a derived analysis report built from existing metrics and images. It does not run training, diffusion, or new model inference.",
        "",
        "## Current Status",
        "",
        "- S1 CIFAR-10 sanity baseline: complete.",
        "- S2-HR COCO-256 DeepJSCC: complete with `best.pt`; `latest.pt` is NaN and must not be used.",
        "- S3 M1-BlindDiffusion: complete on 1/7/19 dB, negative result.",
        "- S4 semantic diagnostics: three auxiliary diagnostics complete: CLIP image-image, ImageNet pseudo-label classifier, COCO caption CLIP image-text.",
        "- S5/S6 adaptive control and full experiment: not started.",
        "",
        "## Generated Figures",
        "",
        "- `figures/stage_progress.png`",
        "- `figures/m0_snr_curves.png`",
        "- `figures/m1_quality_metrics.png`",
        "- `figures/m1_semantic_diagnostics.png`",
        "- `figures/m1_negative_deltas.png`",
        "- `figures/representative_visual_outputs.png`",
        "",
        "## Formal M0 COCO-256 Baseline",
        "",
        "| SNR(dB) | PSNR | SSIM | MS-SSIM |",
        "|---:|---:|---:|---:|",
    ]
    for row in m0_rows:
        lines.append(
            f"| {row['snr_db']:g} | {fmt(row['psnr_db'])} | {fmt(row['ssim'])} | {fmt(row['ms_ssim'])} |"
        )
    lines.extend(
        [
            "",
            "## M1 Blind Diffusion Negative Result",
            "",
            "| SNR(dB) | M0 PSNR | M1 PSNR | Delta PSNR | M0 LPIPS | M1 LPIPS | CLIP drop | Caption drop | M1 pseudo drift |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in m1_rows:
        lines.append(
            "| {snr:g} | {m0_psnr} | {m1_psnr} | {delta_psnr} | {m0_lpips} | {m1_lpips} | {clip_drop} | {caption_drop} | {drift} |".format(
                snr=row["snr_db"],
                m0_psnr=fmt(row["m0_psnr_db"]),
                m1_psnr=fmt(row["m1_psnr_db"]),
                delta_psnr=fmt(row["delta_psnr_m1_minus_m0"]),
                m0_lpips=fmt(row["m0_lpips"]),
                m1_lpips=fmt(row["m1_lpips"]),
                clip_drop=fmt(row["clip_drop_m0_minus_m1"]),
                caption_drop=fmt(row["caption_drop_m0_minus_m1"]),
                drift=fmt(row["classifier_all_m1_drift_origin"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The COCO-256 M0 baseline behaves as expected: PSNR/SSIM/MS-SSIM improve with SNR.",
            "- Fixed-strength blind SD img2img at `strength=0.25` is not a useful refinement setting here.",
            "- M1 degrades image fidelity, LPIPS, CLIP image consistency, COCO caption alignment, and pseudo-label consistency.",
            "- Current semantic diagnostics are auxiliary; the project still needs a frozen main semantic model and Final-Failure metric before S5.",
            "",
            "## Representative Assets Copied",
            "",
        ]
    )
    for item in copied_assets:
        lines.append(f"- `{item['copy']}` from `{item['source']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = resolve_project_path(args.output_dir)
    ensure_output_dir(output_dir, args.overwrite)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    payloads = extract_metrics()
    tables = build_tables(payloads)

    write_csv(output_dir / "stage_progress.csv", tables["stage"])
    write_csv(output_dir / "s1_cifar10_m0.csv", tables["s1"])
    write_csv(output_dir / "coco256_m0_snr_sweep.csv", tables["m0"])
    write_csv(output_dir / "m1_blind_diffusion_summary.csv", tables["m1"])
    save_json(
        output_dir / "summary.json",
        {
            "metadata": {
                "note": "Derived project summary only; no new training, diffusion, or model inference.",
                "sources": [
                    "outputs/EXP-S1-001/metrics.json",
                    "outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export/metrics.json",
                    "outputs/EXP-S2-002/metrics.json",
                    "outputs/EXP-S3-001/metrics.json",
                    "outputs/EXP-S3-002/metrics.json",
                    "outputs/EXP-S3-003/metrics.json",
                ],
                "key_sources": ["scripts/s4_make_project_progress_visual_summary.py"],
            },
            "tables": tables,
        },
    )

    plot_stage_progress(tables["stage"], figures_dir / "stage_progress.png")
    plot_snr_curves(tables, figures_dir / "m0_snr_curves.png")
    plot_m1_quality(tables["m1"], figures_dir / "m1_quality_metrics.png")
    plot_semantic_diagnostics(tables["m1"], figures_dir / "m1_semantic_diagnostics.png")
    plot_negative_deltas(tables["m1"], figures_dir / "m1_negative_deltas.png")
    copied_assets = copy_representative_assets(output_dir)
    make_overview_sheet(copied_assets, figures_dir / "representative_visual_outputs.png")
    (output_dir / "REPORT.md").write_text(make_report(tables, copied_assets), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": project_relative(output_dir),
                "report": project_relative(output_dir / "REPORT.md"),
                "figures": [project_relative(path) for path in sorted(figures_dir.glob("*.png"))],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
