#!/usr/bin/env python3
"""Derived figures for the 2026-07-03 -> 2026-07-10 progress summary.

This script only reads existing CSV artifacts under outputs/analysis and
renders comparison figures. It does not train models, run diffusion, or
download anything.
"""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS = os.path.join(ROOT, "outputs", "analysis")
OUTDIR = os.path.join(ROOT, "reports", "progress_2026-07-10", "figures")
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

SNRS = [1, 4, 7, 13, 19]


def save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------------
# Figure 1: M0 vs M2 vs M3 PSNR / LPIPS / semantic-failure across SNR
# ---------------------------------------------------------------------------
def fig_m0_m2_m3_per_snr():
    df = pd.read_csv(os.path.join(ANALYSIS, "minimal_closure_report",
                                  "residual_per_snr_quality_semantics.csv"))
    m = {
        "M0-DeepJSCC-HR": ("M0 DeepJSCC", "#555555", "o"),
        "M2-SNRConditionedPixelResidualRestoration": ("M2 residual (raw)", "#1f77b4", "s"),
        "M3-ResidualRestorationTop1Fallback": ("M3 top-1 fallback", "#d62728", "^"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    for key, (label, color, marker) in m.items():
        sub = df[df.method == key].sort_values("snr_db")
        axes[0].plot(sub.snr_db, sub.psnr_db, marker=marker, color=color, label=label)
        axes[1].plot(sub.snr_db, sub.lpips, marker=marker, color=color, label=label)
        axes[2].plot(sub.snr_db, sub.semantic_failure, marker=marker, color=color, label=label)

    axes[0].set_title("Reconstruction quality (PSNR, higher better)")
    axes[0].set_ylabel("PSNR (dB)")
    axes[1].set_title("Perceptual quality (LPIPS, lower better)")
    axes[1].set_ylabel("LPIPS")
    axes[2].set_title("Pseudo semantic failure (lower better)")
    axes[2].set_ylabel("failure rate")
    for ax in axes:
        ax.set_xlabel("SNR (dB)")
        ax.set_xticks(SNRS)
        ax.legend(fontsize=8)
    fig.suptitle("EXP-S4-006 COCO-256 AWGN: M0 / M2 / M3 across SNR (64 img/SNR)", y=1.02)
    save(fig, "fig1_m0_m2_m3_per_snr.png")


# ---------------------------------------------------------------------------
# Figure 2: M1 blind diffusion negative result vs M2/M3 (PSNR delta bar)
# ---------------------------------------------------------------------------
def fig_method_overview():
    labels = [
        "M1 blind\ndiffusion",
        "M3 top-1\nfallback",
        "M3 fixed\nshrink",
        "M3 two-stage\nalpha",
        "M3 receiver\npredictor",
        "M3 adaptive\nalpha",
        "M2 residual\n(raw, unsafe)",
    ]
    # mean PSNR delta vs M0 (validation split where available)
    psnr_delta = [-14.7485, 0.4011, 0.4584, 0.4831, 0.5584, 0.5584, 0.7235]
    new_error = [None, 0, 0, 0, 0, 0, 28]

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = []
    for lab, ne in zip(labels, new_error):
        if lab.startswith("M1"):
            colors.append("#7f7f7f")
        elif ne == 0:
            colors.append("#2ca02c")
        else:
            colors.append("#ff7f0e")
    bars = ax.bar(range(len(labels)), psnr_delta, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean PSNR delta vs M0 (dB)")
    ax.set_title("Method comparison: PSNR gain vs M0 and semantic safety\n"
                 "(green = 0 accepted new error, orange = has new error, gray = negative reference)")
    for i, (v, ne) in enumerate(zip(psnr_delta, new_error)):
        txt = f"{v:+.3f} dB"
        if ne is not None:
            txt += f"\nnew err={ne}"
        ax.text(i, v + (0.15 if v > 0 else -0.9), txt, ha="center", fontsize=7.5)
    ax.set_ylim(-16.5, 1.6)
    save(fig, "fig2_method_overview.png")


# ---------------------------------------------------------------------------
# Figure 3: alpha policy tradeoff (PSNR delta vs accept rate, per split)
# ---------------------------------------------------------------------------
def fig_alpha_policy_tradeoff():
    df = pd.read_csv(os.path.join(ANALYSIS, "minimal_closure_report",
                                  "adaptive_residual_alpha_policy_tradeoff.csv"))
    policy_order = [
        ("top1_full_strength", "top-1 full", "#1f77b4"),
        ("fixed_validation_top1_shrink_schedule", "fixed shrink", "#9467bd"),
        ("adaptive_max_top1_consistent_alpha", "adaptive alpha", "#2ca02c"),
        ("always_full_strength", "always-accept (unsafe)", "#d62728"),
    ]
    splits = ["validation", "held-out", "test-like"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    for ax, split in zip(axes, splits):
        sub = df[df.split == split]
        for pol, label, color in policy_order:
            row = sub[sub.policy == pol]
            if row.empty:
                continue
            x = row.accepted_new_error_count.values[0]
            y = row.delta_psnr_vs_m0_db.values[0]
            ax.scatter(x, y, color=color, s=90, label=label, zorder=3)
            ax.annotate(f"{y:+.3f}", (x, y), textcoords="offset points",
                        xytext=(6, 4), fontsize=7.5)
        ax.set_title(split)
        ax.set_xlabel("accepted new error count (lower = safer)")
        ax.axvline(0, color="green", ls="--", lw=0.8, alpha=0.6)
    axes[0].set_ylabel("PSNR delta vs M0 (dB)")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Residual alpha policies: quality gain vs semantic risk (AlexNet pseudo-label)", y=1.02)
    save(fig, "fig3_alpha_policy_tradeoff.png")


# ---------------------------------------------------------------------------
# Figure 4: edge 2x2 controlled ablation
# ---------------------------------------------------------------------------
def fig_edge_ablation():
    df = pd.read_csv(os.path.join(ANALYSIS, "exp_s4_006_008_009_010_edge_capacity_ablation",
                                  "arm_summary.csv"))
    allrows = df[df.level == "all"].copy()
    order = ["small_no_edge", "small_edge", "large_no_edge", "large_edge"]
    labels = {
        "small_no_edge": "small\nno-edge\n(EXP-006)",
        "small_edge": "small\nedge\n(EXP-010)",
        "large_no_edge": "large\nno-edge\n(EXP-009)",
        "large_edge": "large\nedge\n(EXP-008)",
    }
    allrows = allrows.set_index("arm").loc[order].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(len(order))
    w = 0.35
    axes[0].bar(x - w/2, allrows.raw_refined_delta_vs_m0_db, w, label="raw refined", color="#1f77b4")
    axes[0].bar(x + w/2, allrows.m3_delta_vs_m0_db, w, label="M3 top-1", color="#2ca02c")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([labels[a] for a in order], fontsize=8)
    axes[0].set_ylabel("PSNR delta vs M0 (dB)")
    axes[0].set_title("Edge x capacity 2x2: PSNR gain")
    axes[0].legend(fontsize=8)
    for i, (r, m3) in enumerate(zip(allrows.raw_refined_delta_vs_m0_db, allrows.m3_delta_vs_m0_db)):
        axes[0].text(i - w/2, r + 0.01, f"{r:.3f}", ha="center", fontsize=7)
        axes[0].text(i + w/2, m3 + 0.01, f"{m3:.3f}", ha="center", fontsize=7)

    # raw new error / repair (semantic)
    axes[1].bar(x - w/2, allrows.raw_new_error_count, w, label="raw new error", color="#d62728")
    axes[1].bar(x + w/2, allrows.raw_repair_count, w, label="raw repair", color="#8c564b")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([labels[a] for a in order], fontsize=8)
    axes[1].set_ylabel("count (raw refined, AlexNet pseudo-label)")
    axes[1].set_title("Edge x capacity 2x2: semantic new error vs repair")
    axes[1].legend(fontsize=8)
    fig.suptitle("Capacity- and budget-matched edge-conditioning ablation (64 img/SNR x 5 SNR)", y=1.02)
    save(fig, "fig4_edge_ablation.png")


# ---------------------------------------------------------------------------
# Figure 5: continuous-alpha vs full-strength (quality + ensemble risk)
# ---------------------------------------------------------------------------
def fig_continuous_alpha():
    qs = pd.read_csv(os.path.join(ANALYSIS, "exp_s4_006_continuous_alpha_tail_refiner_audit",
                                  "quality_summary.csv"))
    vs = pd.read_csv(os.path.join(ANALYSIS, "exp_s4_006_continuous_alpha_tail_refiner_audit",
                                  "vote_summary.csv"))
    splits = ["validation", "held-out", "test-like"]
    pol = {
        "continuous_alpha_top1_fallback": ("continuous alpha", "#2ca02c"),
        "full_strength_top1_fallback": ("full-strength", "#1f77b4"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    x = np.arange(len(splits))
    w = 0.35
    qs = qs[qs.level == "policy_split"]
    vs = vs[vs.level == "policy_split"]
    for i, (p, (label, color)) in enumerate(pol.items()):
        vals = []
        for s in splits:
            r = qs[(qs.policy == p) & (qs.split == s)]
            vals.append(r.delta_final_psnr_vs_m0_db.values[0] if not r.empty else np.nan)
        axes[0].bar(x + (i - 0.5) * w, vals, w, label=label, color=color)
        for xi, v in zip(x, vals):
            axes[0].text(xi + (i - 0.5) * w, v + 0.005, f"{v:.3f}", ha="center", fontsize=7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(splits)
    axes[0].set_ylabel("PSNR delta vs M0 (dB)")
    axes[0].set_title("Continuous-alpha tail refiner: PSNR gain")
    axes[0].legend(fontsize=8)

    # ensemble any/majority new error
    for i, (p, (label, color)) in enumerate(pol.items()):
        any_err = []
        maj_err = []
        for s in splits:
            r = vs[(vs.policy == p) & (vs.split == s)]
            any_err.append(r.any_classifier_new_error_count.values[0] if not r.empty else np.nan)
            maj_err.append(r.majority_classifier_new_error_count.values[0] if not r.empty else np.nan)
        axes[1].bar(x + (i - 0.5) * w, any_err, w, label=f"{label}: any-model", color=color, alpha=0.6)
        axes[1].bar(x + (i - 0.5) * w, maj_err, w, label=f"{label}: majority", color=color)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(splits)
    axes[1].set_ylabel("new error count (3-classifier ensemble)")
    axes[1].set_title("Cross-classifier semantic risk")
    axes[1].legend(fontsize=7)
    fig.suptitle("Continuous-alpha learned refiner: strongest training-side candidate, residual ensemble risk", y=1.02)
    save(fig, "fig5_continuous_alpha.png")


# ---------------------------------------------------------------------------
# Figure 6: M3 candidate evolution timeline (PSNR delta progression)
# ---------------------------------------------------------------------------
def fig_m3_evolution():
    stages = [
        ("top-1 fallback", 0.4011, 0.4454, 0.4113),
        ("fixed shrink", 0.4584, 0.4689, 0.4552),
        ("tail-only\nclassif.", 0.4749, 0.4552, 0.4061),
        ("two-stage\nalpha", 0.4831, 0.5009, 0.4875),
        ("receiver\npredictor", 0.5584, 0.5099, 0.4871),
        ("continuous\nalpha", 0.5010, 0.5049, 0.5012),
        ("adaptive alpha\n(post-hoc)", 0.5584, 0.5664, 0.5691),
    ]
    names = [s[0] for s in stages]
    val = [s[1] for s in stages]
    hel = [s[2] for s in stages]
    tst = [s[3] for s in stages]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, val, marker="o", label="validation")
    ax.plot(x, hel, marker="s", label="held-out")
    ax.plot(x, tst, marker="^", label="test-like")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("PSNR delta vs M0 (dB)")
    ax.set_title("M3 candidate evolution (all keep accepted new error = 0 under source AlexNet)\n"
                 "post-hoc adaptive alpha is the empirical ceiling; learned refiners approach it")
    ax.legend()
    for xi, v in zip(x, val):
        ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=7)
    save(fig, "fig6_m3_evolution.png")


if __name__ == "__main__":
    fig_m0_m2_m3_per_snr()
    fig_method_overview()
    fig_alpha_policy_tradeoff()
    fig_edge_ablation()
    fig_continuous_alpha()
    fig_m3_evolution()
    print("all figures written to", OUTDIR)
