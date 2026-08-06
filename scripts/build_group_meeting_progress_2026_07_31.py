#!/usr/bin/env python3
"""Build figures and a compact data sheet for the 2026-07-31 group meeting report.

All numbers are transcribed from frozen result reports/JSON files.  This script does
not run inference, training, or metric evaluation.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "group_meeting_progress_2026-07-31" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
FONT_BOLD_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT = FontProperties(fname=FONT_PATH)
FONT_BOLD = FontProperties(fname=FONT_BOLD_PATH)

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 180,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "axes.axisbelow": True,
    }
)


def zh(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, fontproperties=FONT_BOLD, fontsize=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontproperties=FONT)
    if ylabel:
        ax.set_ylabel(ylabel, fontproperties=FONT)
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_fontproperties(FONT)


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def timeline():
    events = [
        ("07-17", "SGD vs B1", "确认保真/感知 Pareto\n不能无条件全用 diffusion", "#4575b4"),
        ("07-20", "B1 + diffusion", "简单 gate 失败；S23 仅机制正结果\nS26 route 增益约 +0.093 dB", "#74add1"),
        ("07-21", "外部基线与强主干", "DiffJSCC 暴露旧 backbone 瓶颈\nS33 等码率显著超过 author-JSCC", "#1a9850"),
        ("07-22", "Swin equal-budget", "胜 Base、与 CM 形成 Pareto\nSwin 尚未完全收敛", "#66bd63"),
        ("07-23", "公平性 / 代价 / 大图", "生成链最低保感知点仍慢 165×\nKodak/CLIC：S33 未战胜 Swin", "#fdae61"),
        ("07-30", "RDD-P0", "有方法指纹和分布偏移\n但不能归因于生成先验", "#d73027"),
    ]
    fig, ax = plt.subplots(figsize=(14, 4.8))
    x = np.arange(len(events))
    ax.plot(x, np.zeros_like(x), color="#444", lw=2)
    for i, (date, title, desc, color) in enumerate(events):
        ax.scatter(i, 0, s=180, color=color, zorder=3, edgecolor="white", lw=1.5)
        y = 0.62 if i % 2 == 0 else -0.62
        ax.plot([i, i], [0.08 * np.sign(y), y * 0.63], color=color, lw=1.5)
        va = "bottom" if y > 0 else "top"
        ax.text(i, y, date, ha="center", va=va, fontproperties=FONT_BOLD, fontsize=11, color=color)
        offset = 0.12 if y > 0 else -0.12
        ax.text(i, y + offset, title, ha="center", va=va, fontproperties=FONT_BOLD, fontsize=10)
        ax.text(i, y + 2 * offset, desc, ha="center", va=va, fontproperties=FONT, fontsize=8.7, linespacing=1.4)
    ax.set_xlim(-0.45, len(events) - 0.55)
    ax.set_ylim(-1.38, 1.38)
    ax.axis("off")
    ax.set_title("7 月 17 日至 7 月 30 日：项目判断如何收敛", fontproperties=FONT_BOLD, fontsize=15, pad=12)
    save(fig, "01_timeline.png")


def internal_progress():
    labels = ["S23\n轻量注入", "S26\n跨总体复现", "S27\n全新 512 图复现"]
    dpsnr = [0.000568, 0.093267, 0.092662]
    lpips_gain = [0.001731, 0.007661, 0.007922]
    psnr_ci = [(0.000378, 0.000771), (0.087945, 0.098806), (0.089147, 0.096313)]
    lpips_ci = [(0.001622, 0.001849), (0.006915, 0.008438), (0.007398, 0.008465)]
    x = np.arange(3)
    colors = ["#91bfdb", "#2c7fb8", "#1d91c0"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, vals, cis, title, ylabel in [
        (axes[0], dpsnr, psnr_ci, "相对 B1 的 PSNR 增益", "ΔPSNR (dB，越高越好)"),
        (axes[1], lpips_gain, lpips_ci, "相对 B1 的 LPIPS 改善", "−ΔLPIPS（越高越好）"),
    ]:
        low = [v - c[0] for v, c in zip(vals, cis)]
        high = [c[1] - v for v, c in zip(vals, cis)]
        ax.bar(x, vals, color=colors, width=0.64)
        ax.errorbar(x, vals, yerr=[low, high], fmt="none", ecolor="#222", capsize=4, lw=1.2)
        ax.axhline(0, color="#333", lw=0.8)
        ax.set_xticks(x, labels)
        zh(ax, title, ylabel=ylabel)
        for i, value in enumerate(vals):
            ax.text(i, value + max(vals) * 0.045, f"{value:.6f}", ha="center", fontproperties=FONT, fontsize=9)
    fig.suptitle("旧 B1/diffusion 路线：从微小机制结果到可复现但仍有限的增益", fontproperties=FONT_BOLD, fontsize=14)
    fig.text(
        0.5,
        -0.01,
        "S26/S27 在 1/4/7 dB 使用 fusion，13/19 dB 精确回退 B1；这些结果后来被 S33 强主干取代为历史机制证据。",
        ha="center",
        fontproperties=FONT,
        fontsize=9,
    )
    save(fig, "02_internal_diffusion_progress.png")


def rate_ledger():
    methods = ["S33", "DiffJSCC", "SGD paper upper"]
    main = np.array([16384, 16384, 16384])
    edge = np.array([0, 0, 3328])
    caption = np.array([0, 0, 2144])
    fig, ax = plt.subplots(figsize=(9, 3.8))
    y = np.arange(3)
    ax.barh(y, main, color="#4575b4", label="主图符号")
    ax.barh(y, edge, left=main, color="#fdae61", label="edge")
    ax.barh(y, caption, left=main + edge, color="#d73027", label="caption 最低成本")
    ax.axvline(16384, color="#222", ls="--", lw=1, label="S33 预算 16,384")
    ax.set_yticks(y, methods)
    ax.invert_yaxis()
    zh(ax, "严格通信码率账本", "实际发送的 real symbols / 图")
    ax.legend(prop=FONT, loc="lower right", frameon=False)
    ax.text(21856 + 250, 2, "+33.40%", va="center", fontproperties=FONT_BOLD, color="#b2182b")
    save(fig, "03_rate_ledger.png")


def external_tradeoff_and_cost():
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))

    points = [
        ("S33", 30.4661, 0.119985, "#1b9e77", "o"),
        ("author-JSCC", 29.9861, 0.128342, "#7570b3", "s"),
        ("DiffJSCC", 27.5984, 0.100223, "#d95f02", "D"),
        ("SGD upper", 27.7404, 0.072101, "#e7298a", "^"),
    ]
    ax = axes[0]
    for label, psnr, lpips, color, marker in points:
        face = "none" if label == "SGD upper" else color
        ax.scatter(psnr, lpips, s=100, color=face, edgecolor=color, marker=marker, lw=2, zorder=3)
        ax.annotate(label, (psnr, lpips), xytext=(6, 6), textcoords="offset points", fontproperties=FONT, fontsize=9)
    zh(ax, "256² 共同总体：保真—感知 Pareto", "PSNR (dB，越高越好)", "LPIPS（越低越好）")
    ax.invert_yaxis()
    ax.text(
        0.02,
        0.02,
        "S33 / author / Diff：16,384 real\nSGD：≥21,856 real，non-ranking",
        transform=ax.transAxes,
        fontproperties=FONT,
        fontsize=8.5,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb", alpha=0.9),
    )

    steps = np.array([100, 50, 25, 10, 4])
    latency = np.array([5089.7, 2676.2, 1458.5, 726.3, 433.6])
    lpips = np.array([0.099957, 0.097870, 0.101952, 0.117499, 0.138976])
    ax = axes[1]
    ax.plot(latency, lpips, "-o", color="#d95f02", lw=2, ms=7, label="DiffJSCC")
    for s, x, y in zip(steps, latency, lpips):
        ax.annotate(f"{s}步", (x, y), xytext=(5, 6), textcoords="offset points", fontproperties=FONT, fontsize=8.5)
    ax.scatter([8.833], [0.119902], color="#1b9e77", s=90, marker="*", label="S33（共同 runtime）", zorder=4)
    ax.axvline(1458.5, color="#d95f02", ls="--", alpha=0.45)
    ax.set_xscale("log")
    zh(ax, "DiffJSCC 少步延迟—感知曲线", "单图延迟 (ms，对数轴)", "LPIPS（越低越好）")
    ax.legend(prop=FONT, frameon=False, loc="upper left")
    ax.text(1458.5, 0.132, "最低仍显著保持\nLPIPS 优势：25步\n= 165.1× S33", ha="center", fontproperties=FONT_BOLD, fontsize=8.8)
    fig.suptitle("外部生成式基线：不是全面胜负，而是质量、码率与计算代价的联合权衡", fontproperties=FONT_BOLD, fontsize=14)
    save(fig, "04_external_tradeoff_and_cost.png")


def highres_benchmark():
    names = ["S33", "Swin Base-SA", "Swin CM-SA"]
    colors = ["#1b9e77", "#7570b3", "#e6ab02"]
    datasets = ["Kodak", "CLIC2020 test"]
    psnr = np.array([[29.2070, 29.1593, 29.4073], [32.1842, 32.4473, 32.6751]])
    lpips = np.array([[0.206067, 0.197790, 0.186268], [0.215475, 0.163603, 0.161385]])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    x = np.arange(2)
    width = 0.24
    for i, name in enumerate(names):
        axes[0].bar(x + (i - 1) * width, psnr[:, i], width, label=name, color=colors[i])
        axes[1].bar(x + (i - 1) * width, lpips[:, i], width, label=name, color=colors[i])
    axes[0].set_ylim(28.5, 33.1)
    axes[1].set_ylim(0.13, 0.225)
    for ax in axes:
        ax.set_xticks(x, datasets)
    zh(axes[0], "聚合 PSNR", ylabel="PSNR (dB，越高越好)")
    zh(axes[1], "聚合 LPIPS", ylabel="LPIPS（越低越好）")
    axes[1].legend(prop=FONT, frameon=False, loc="upper right")
    fig.suptitle("领域惯例高分辨率 benchmark：S33 未战胜 SwinJSCC", fontproperties=FONT_BOLD, fontsize=14)
    fig.text(
        0.5,
        -0.01,
        "逐图 actual CBR 严格相同。Kodak：S33 仅 PSNR 追平 Base；CLIC：S33 对 Base/CM 均劣于。",
        ha="center",
        fontproperties=FONT,
        fontsize=9,
    )
    save(fig, "05_highres_swin_benchmark.png")


def rdd_fingerprint():
    labels = ["4臂完整图", "3臂完整图", "4臂中心裁剪", "4臂降采样", "仅两判别式"]
    acc = np.array([0.8396, 0.9059, 0.7852, 0.7102, 0.8693])
    low = np.array([0.7984, 0.8715, 0.7438, 0.6542, 0.8214])
    high = np.array([0.8776, 0.9378, 0.8258, 0.7635, 0.9120])
    chance = np.array([0.25, 1 / 3, 0.25, 0.25, 0.5])
    features = ["DCT 高频变异", "径向频谱 b09", "径向频谱 b11", "高通 MAD", "平均梯度"]
    importance = [0.2132, 0.1915, 0.1844, 0.1482, 0.1303]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(labels))
    axes[0].bar(x, acc, color=["#7b3294", "#008837", "#c2a5cf", "#a6dba0", "#d73027"])
    axes[0].errorbar(x, acc, yerr=[acc - low, high - acc], fmt="none", ecolor="#222", capsize=4)
    axes[0].scatter(x, chance, marker="_", s=280, color="black", lw=2, label="随机水平")
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0].set_ylim(0, 1)
    zh(axes[0], "方法指纹分类准确率", ylabel="准确率")
    axes[0].legend(prop=FONT, frameon=False, loc="lower left")

    y = np.arange(len(features))[::-1]
    axes[1].barh(y, importance, color="#5ab4ac")
    axes[1].set_yticks(y, features)
    zh(axes[1], "最有区分力的统计特征", "置换重要性")
    fig.suptitle("RDD-P0：偏移可识别，但“生成先验导致”未被证明", fontproperties=FONT_BOLD, fontsize=14)
    fig.text(
        0.5,
        -0.01,
        "关键否证：仅 S33 与 author-JSCC 两个无生成先验方法，准确率仍为 86.9%（随机 50%）；指纹主要来自实现与高频差异。",
        ha="center",
        fontproperties=FONT,
        fontsize=9,
    )
    save(fig, "06_rdd_fingerprint.png")


def kodak_example():
    base = ROOT / "paper_idea1b"
    rec = base / "outputs" / "ANALYSIS-IDEA1B-A1-DISCRIMINATIVE-001" / "reconstructions" / "kodak"
    paths = [
        base / "data" / "kodak" / "kodim03.png",
        rec / "s33_strong" / "seed_20260748" / "snr_01" / "442fb3fb0422e28f__kodim03.png",
        rec / "swin_official_base_sa" / "seed_20260748" / "snr_01" / "442fb3fb0422e28f__kodim03.png",
        rec / "swin_capacity_matched_sa" / "seed_20260748" / "snr_01" / "442fb3fb0422e28f__kodim03.png",
    ]
    labels = [
        "原图",
        "S33 | PSNR 31.04 | LPIPS 0.166",
        "Swin Base | PSNR 30.94 | LPIPS 0.167",
        "Swin CM | PSNR 31.31 | LPIPS 0.153",
    ]
    images = [Image.open(p).convert("RGB") for p in paths]
    panel_w, panel_h = 576, 384
    header_h, footer_h = 72, 42
    canvas = Image.new("RGB", (panel_w * 4, header_h + panel_h + footer_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT_PATH, 25)
    small = ImageFont.truetype(FONT_PATH, 20)
    for i, (im, label) in enumerate(zip(images, labels)):
        im = im.resize((panel_w, panel_h), Image.Resampling.LANCZOS)
        canvas.paste(im, (i * panel_w, header_h))
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text((i * panel_w + (panel_w - (bbox[2] - bbox[0])) / 2, 20), label, fill="#111", font=font)
        if i:
            draw.line((i * panel_w, 0, i * panel_w, header_h + panel_h), fill="#bbbbbb", width=2)
    footer = "Kodak kodim03，1 dB，seed 20260748；四列逐图 actual CBR 完全相同。单图仅作直观示例，统计结论以完整 Kodak/CLIC 主表为准。"
    bbox = draw.textbbox((0, 0), footer, font=small)
    draw.text(((canvas.width - (bbox[2] - bbox[0])) / 2, header_h + panel_h + 8), footer, fill="#333", font=small)
    canvas.save(OUT / "07_kodak_1db_visual_example.png")


def data_sheet():
    columns = [
        "stage",
        "population_contract",
        "method",
        "psnr_db",
        "ms_ssim",
        "lpips",
        "semantic_failures",
        "observations",
        "latency_ms",
        "real_symbols",
        "status_note",
    ]
    rows = [
        ["S20", "Imagenette policy-dev 64x3x5", "B1", 28.12459, 0.946697, 0.159398, 35, 960, 2.642, 19712, "旧保真锚点"],
        ["S20", "Imagenette policy-dev 64x3x5", "SGD paper upper", 27.74037, 0.952973, 0.072101, 25, 960, 2064.738, 21856, "non-ranking; 完美 caption; symbols为下界"],
        ["S27", "COCO pristine 512x5", "B1", 27.323569, 0.943408, 0.188371, 1561, 2560, "", 19712, "历史锚点"],
        ["S27", "COCO pristine 512x5", "routed fusion", 27.416232, 0.945718, 0.180449, 1517, 2560, "", 19712, "低SNR fusion，高SNR exact B1"],
        ["S33/S34C", "Imagenette policy-dev 64x3x5", "S33", 30.466064, 0.969708, 0.119985, 9, 960, 8.833, 16384, "共同runtime延迟"],
        ["S33", "Imagenette policy-dev 64x3x5", "author-JSCC", 29.986135, 0.963092, 0.128342, 22, 960, "", 16384, "S33显著超过"],
        ["S34C", "Imagenette policy-dev 64x3x5", "DiffJSCC-100", 27.598398, 0.940799, 0.100223, 23, 960, 5089.671, 16384, "exact-rate Pareto"],
        ["A1", "Kodak 24x5x3", "S33", 29.207021, 0.957358, 0.206067, "", 360, "", "actual CBR=1/24", ""],
        ["A1", "Kodak 24x5x3", "Swin Base-SA", 29.159329, 0.960928, 0.197790, "", 360, "", "actual CBR=1/24", ""],
        ["A1", "Kodak 24x5x3", "Swin CM-SA", 29.407306, 0.962510, 0.186268, "", 360, "", "actual CBR=1/24", "S33劣于此臂"],
        ["A1", "CLIC2020 test 428x5x1", "S33", 32.1842, 0.967450, 0.215475, "", 2140, 189.7, "actual CBR 0.041667-0.063210", "最大2048图延迟"],
        ["A1", "CLIC2020 test 428x5x1", "Swin Base-SA", 32.4473, 0.972799, 0.163603, "", 2140, 439.5, "same as S33 per image", "S33劣于"],
        ["A1", "CLIC2020 test 428x5x1", "Swin CM-SA", 32.6751, 0.974193, 0.161385, "", 2140, 464.4, "same as S33 per image", "S33劣于"],
    ]
    with (OUT / "presentation_key_numbers.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)


def main():
    timeline()
    internal_progress()
    rate_ledger()
    external_tradeoff_and_cost()
    highres_benchmark()
    rdd_fingerprint()
    kodak_example()
    data_sheet()
    print(f"wrote report assets to {OUT}")


if __name__ == "__main__":
    main()
