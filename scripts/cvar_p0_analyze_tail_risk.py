#!/usr/bin/env python3
"""P2/P3 conditional tail-risk statistics, plots and GO/NO-GO verdict.

Consumes ``diagnostic_samples.csv`` produced by
``scripts/cvar_p0_diagnose_tail_risk.py`` and emits:

*   ``diagnostic_summary.csv``      one row per (arm, snr)
*   ``per_image_tail_stats.csv``    one row per (arm, snr, image)
*   ``variance_decomposition.csv``  within-image (channel) vs between-image (content)
*   ``plots/*.png``                 the five diagnostic figures
*   ``verdict.json``                machine-readable GO / NO-GO

All tail statistics are computed *conditionally*: per image over that image's M
channel realizations, then aggregated over images.  A global pool over
``images x realizations`` is never used for a tail, because it would confuse
image-content difficulty with channel randomness.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

from cadsd_jscc.tail_risk import (  # noqa: E402
    empirical_lower_tail_mean,
    empirical_upper_cvar,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cvar_p0_tail_risk_diagnostic.yaml")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def percentile(values: list[float], fraction: float) -> float:
    """Linear-interpolation percentile, matching numpy's default."""

    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    dx = [value - mean_x for value in xs]
    dy = [value - mean_y for value in ys]
    denominator = math.sqrt(sum(v * v for v in dx)) * math.sqrt(sum(v * v for v in dy))
    if denominator <= 0.0:
        return float("nan")
    return sum(a * b for a, b in zip(dx, dy)) / denominator


def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda index: values[index])
        result = [0.0] * len(values)
        position = 0
        while position < len(order):
            end = position
            while (
                end + 1 < len(order)
                and values[order[end + 1]] == values[order[position]]
            ):
                end += 1
            average = (position + end) / 2.0 + 1.0
            for index in range(position, end + 1):
                result[order[index]] = average
            position = end + 1
        return result

    return pearson(ranks(xs), ranks(ys))


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    directory = resolve(
        config["outputs"]["dry_run_directory"]
        if args.dry_run
        else config["outputs"]["directory"]
    )
    rows = load_rows(directory / "diagnostic_samples.csv")
    metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
    (directory / "plots").mkdir(parents=True, exist_ok=True)

    metrics_cfg = config["metrics"]
    fractions = [float(value) for value in metrics_cfg["tail_fractions"]]
    thresholds = [float(value) for value in metrics_cfg["outage_psnr_thresholds_db"]]
    relative_margin = float(metrics_cfg["relative_outage_margin_db"])
    rule = config["decision_rule"]
    reference_arm = str(rule["reference_arm"])
    candidates = [str(value) for value in rule["candidate_arms"]]

    # (arm, snr, image) -> list of per-realization records
    grouped: dict[tuple[str, float, str], list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        key = (row["arm"], float(row["snr_db"]), row["image_id"])
        grouped[key].append(
            {
                "psnr": float(row["psnr"]),
                "mse": float(row["mse"]),
                "ms_ssim": float(row["ms_ssim"]),
                "lpips": float(row["lpips"]),
                "h_power": float(row["h_power"]),
            }
        )

    arms = sorted({row["arm"] for row in rows})
    snrs = sorted({float(row["snr_db"]) for row in rows})
    images = sorted({row["image_id"] for row in rows})

    # ---------------------------------------------------------------- per image
    per_image: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    per_image_fields = [
        "arm",
        "snr_db",
        "image_id",
        "realizations",
        "mean_psnr",
        "median_psnr",
        "p05_psnr",
        "p10_psnr",
        "std_psnr",
        "min_psnr",
        "mean_mse",
        "mean_lpips",
        "spearman_psnr_vs_h_power",
    ]
    for fraction in fractions:
        tag = f"{int(round(fraction * 100)):02d}"
        per_image_fields += [f"worst{tag}_mean_psnr", f"cvar{tag}_mse", f"cvar{tag}_lpips"]

    with (directory / "per_image_tail_stats.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=per_image_fields)
        writer.writeheader()
        for arm in arms:
            for snr in snrs:
                for image in images:
                    records = grouped.get((arm, snr, image))
                    if not records:
                        continue
                    psnr = [item["psnr"] for item in records]
                    mse = [item["mse"] for item in records]
                    lpips = [item["lpips"] for item in records]
                    hp = [item["h_power"] for item in records]
                    entry: dict[str, Any] = {
                        "arm": arm,
                        "snr_db": f"{snr:.6f}",
                        "image_id": image,
                        "realizations": len(records),
                        "mean_psnr": statistics.mean(psnr),
                        "median_psnr": statistics.median(psnr),
                        "p05_psnr": percentile(psnr, 0.05),
                        "p10_psnr": percentile(psnr, 0.10),
                        "std_psnr": statistics.pstdev(psnr) if len(psnr) > 1 else 0.0,
                        "min_psnr": min(psnr),
                        "mean_mse": statistics.mean(mse),
                        "mean_lpips": statistics.mean(lpips),
                        "spearman_psnr_vs_h_power": spearman(psnr, hp),
                    }
                    psnr_tensor = torch.tensor(psnr)
                    mse_tensor = torch.tensor(mse)
                    lpips_tensor = torch.tensor(lpips)
                    for fraction in fractions:
                        tag = f"{int(round(fraction * 100)):02d}"
                        entry[f"worst{tag}_mean_psnr"] = float(
                            empirical_lower_tail_mean(psnr_tensor, fraction)
                        )
                        entry[f"cvar{tag}_mse"] = float(
                            empirical_upper_cvar(mse_tensor, fraction)
                        )
                        entry[f"cvar{tag}_lpips"] = float(
                            empirical_upper_cvar(lpips_tensor, fraction)
                        )
                    per_image[(arm, snr)].append(entry)
                    writer.writerow(entry)

    # ----------------------------------------------------------------- summary
    awgn_median: dict[float, float] = {}
    for snr in snrs:
        entries = per_image.get((reference_arm, snr), [])
        if entries:
            awgn_median[snr] = statistics.median(
                [entry["median_psnr"] for entry in entries]
            )

    summary_fields = [
        "arm",
        "snr_db",
        "images",
        "realizations_per_image",
        "mean_psnr",
        "median_psnr",
        "p05_psnr",
        "p10_psnr",
        "std_psnr_within_image",
        "mean_mse",
        "mean_lpips",
        "delta_tail_median_minus_p10",
        "delta_worst10_mean_minus_worst10",
        "spearman_psnr_vs_h_power_pooled",
    ]
    for fraction in fractions:
        tag = f"{int(round(fraction * 100)):02d}"
        summary_fields += [f"worst{tag}_mean_psnr", f"cvar{tag}_mse", f"cvar{tag}_lpips"]
    for threshold in thresholds:
        summary_fields.append(f"outage_psnr{int(threshold)}")
    summary_fields.append("outage_relative_awgn_median_minus_3db")

    summary: dict[tuple[str, float], dict[str, Any]] = {}
    with (directory / "diagnostic_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for arm in arms:
            for snr in snrs:
                entries = per_image.get((arm, snr), [])
                if not entries:
                    continue
                pooled = [
                    item
                    for image in images
                    for item in grouped.get((arm, snr, image), [])
                ]
                pooled_psnr = [item["psnr"] for item in pooled]
                relative_threshold = awgn_median.get(snr, float("nan")) - relative_margin
                row: dict[str, Any] = {
                    "arm": arm,
                    "snr_db": f"{snr:.6f}",
                    "images": len(entries),
                    "realizations_per_image": entries[0]["realizations"],
                    # Aggregated as the mean over images of each image's own
                    # conditional statistic, so image count cannot reweight it.
                    "mean_psnr": statistics.mean([e["mean_psnr"] for e in entries]),
                    "median_psnr": statistics.mean([e["median_psnr"] for e in entries]),
                    "p05_psnr": statistics.mean([e["p05_psnr"] for e in entries]),
                    "p10_psnr": statistics.mean([e["p10_psnr"] for e in entries]),
                    "std_psnr_within_image": statistics.mean(
                        [e["std_psnr"] for e in entries]
                    ),
                    "mean_mse": statistics.mean([e["mean_mse"] for e in entries]),
                    "mean_lpips": statistics.mean([e["mean_lpips"] for e in entries]),
                    "spearman_psnr_vs_h_power_pooled": spearman(
                        pooled_psnr, [item["h_power"] for item in pooled]
                    ),
                }
                for fraction in fractions:
                    tag = f"{int(round(fraction * 100)):02d}"
                    for field in (
                        f"worst{tag}_mean_psnr",
                        f"cvar{tag}_mse",
                        f"cvar{tag}_lpips",
                    ):
                        row[field] = statistics.mean([e[field] for e in entries])
                row["delta_tail_median_minus_p10"] = row["median_psnr"] - row["p10_psnr"]
                row["delta_worst10_mean_minus_worst10"] = (
                    row["mean_psnr"] - row["worst10_mean_psnr"]
                )
                # Outage is a probability over all (image, realization) pairs.
                for threshold in thresholds:
                    hits = sum(1 for value in pooled_psnr if value < threshold)
                    row[f"outage_psnr{int(threshold)}"] = hits / len(pooled_psnr)
                if math.isnan(relative_threshold):
                    row["outage_relative_awgn_median_minus_3db"] = float("nan")
                else:
                    hits = sum(1 for value in pooled_psnr if value < relative_threshold)
                    row["outage_relative_awgn_median_minus_3db"] = hits / len(pooled_psnr)
                summary[(arm, snr)] = row
                writer.writerow(row)

    # ------------------------------------------------ variance decomposition
    with (directory / "variance_decomposition.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "arm",
                "snr_db",
                "within_image_channel_variance",
                "between_image_content_variance",
                "channel_share_of_total_variance",
            ],
        )
        writer.writeheader()
        decomposition: dict[tuple[str, float], dict[str, float]] = {}
        for arm in arms:
            for snr in snrs:
                entries = per_image.get((arm, snr), [])
                if len(entries) < 2:
                    continue
                within = statistics.mean([e["std_psnr"] ** 2 for e in entries])
                between = statistics.pvariance([e["mean_psnr"] for e in entries])
                total = within + between
                item = {
                    "arm": arm,
                    "snr_db": f"{snr:.6f}",
                    "within_image_channel_variance": within,
                    "between_image_content_variance": between,
                    "channel_share_of_total_variance": (
                        within / total if total > 0 else float("nan")
                    ),
                }
                decomposition[(arm, snr)] = item
                writer.writerow(item)

    # ------------------------------------------------------------ primary arm
    selection = str(rule.get("primary_arm_selection", "best_mean_psnr_among_rayleigh_arms_per_snr"))
    primary: dict[float, str] = {}
    if selection == "fixed":
        # Used when training matches one deployment arm a priori, so the arm can
        # be fixed before results instead of chosen from the data.
        fixed_arm = str(rule["primary_arm"])
        for snr in snrs:
            if (fixed_arm, snr) not in summary:
                raise RuntimeError(f"fixed primary arm {fixed_arm} missing at {snr} dB")
            primary[snr] = fixed_arm
    elif selection == "best_mean_psnr_among_rayleigh_arms_per_snr":
        for snr in snrs:
            available = [arm for arm in candidates if (arm, snr) in summary]
            if available:
                primary[snr] = max(
                    available, key=lambda arm: summary[(arm, snr)]["mean_psnr"]
                )
    else:
        raise ValueError(f"unknown primary_arm_selection: {selection}")

    # ----------------------------------------------------------------- verdict
    checks: dict[str, Any] = {}
    snr_points_with_2db = [
        snr
        for snr in snrs
        if summary[(primary[snr], snr)]["delta_tail_median_minus_p10"] >= 2.0
    ]
    checks["at_least_two_snr_points_with_median_minus_p10_psnr_ge_2_db"] = {
        "passed": len(snr_points_with_2db) >= 2,
        "snr_points": snr_points_with_2db,
    }
    worst10_gaps = {
        snr: summary[(primary[snr], snr)]["delta_worst10_mean_minus_worst10"]
        for snr in snrs
    }
    checks["worst10_mean_psnr_gap_vs_mean_ge_1_db"] = {
        "passed": any(value >= 1.0 for value in worst10_gaps.values()),
        "gaps_db": worst10_gaps,
    }
    outage_any = {}
    for snr in snrs:
        row = summary[(primary[snr], snr)]
        outage_any[snr] = max(
            [row[f"outage_psnr{int(threshold)}"] for threshold in thresholds]
        )
    checks["nonneglible_outage_at_some_threshold"] = {
        "passed": any(value >= 0.01 for value in outage_any.values()),
        "max_outage_per_snr": outage_any,
    }
    channel_share = {
        snr: decomposition[(primary[snr], snr)]["channel_share_of_total_variance"]
        for snr in snrs
        if (primary[snr], snr) in decomposition
    }
    rank_corr = {
        snr: summary[(primary[snr], snr)]["spearman_psnr_vs_h_power_pooled"]
        for snr in snrs
    }
    checks["tail_attributable_to_h_power_not_image_difficulty"] = {
        "passed": (
            any(value >= 0.5 for value in channel_share.values())
            and all(value >= 0.5 for value in rank_corr.values())
        ),
        "channel_share_of_variance": channel_share,
        "spearman_psnr_vs_h_power": rank_corr,
    }

    all_snr_below_1db = all(
        summary[(primary[snr], snr)]["delta_tail_median_minus_p10"] < 1.0 for snr in snrs
    )
    best_alternative_gap = {}
    for snr in snrs:
        gaps = {
            arm: summary[(arm, snr)]["delta_tail_median_minus_p10"]
            for arm in candidates
            if (arm, snr) in summary
        }
        best_alternative_gap[snr] = min(gaps.values()) if gaps else float("nan")
    tail_removed_by_conditioning = all(
        value < 1.0 for value in best_alternative_gap.values()
    )
    blockers = {
        "median_minus_p10_psnr_lt_1_db_at_all_snr": all_snr_below_1db,
        "tail_removed_by_switching_to_a_better_snr_conditioning_arm": (
            tail_removed_by_conditioning
        ),
    }

    go = all(item["passed"] for item in checks.values()) and not any(blockers.values())
    verdict = {
        "analysis_id": config["analysis_id"],
        "dry_run": args.dry_run,
        "git_commit": metadata["git_commit"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "source_count": metadata["source_count"],
        "realizations_per_image": metadata["num_channel_realizations"],
        "snrs_db": snrs,
        "primary_arm_per_snr": {f"{snr:g}": primary[snr] for snr in snrs},
        "checks": checks,
        "blockers": blockers,
        "decision": "GO" if go else "NO-GO",
    }
    (directory / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------- plots
    plots = directory / "plots"
    styles = {
        "awgn_control": ("#4C6EF5", "o"),
        "rayleigh_nominal_csi": ("#F59F00", "s"),
        "rayleigh_effective_csi": ("#E03131", "^"),
        "rayleigh_effective_csi_clamped": ("#2F9E44", "D"),
    }

    # Figure 1: mean vs tail PSNR for the primary arm.
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for field, label, style in (
        ("mean_psnr", "mean", "-o"),
        ("median_psnr", "median", "-s"),
        ("p10_psnr", "p10", "--^"),
        ("worst10_mean_psnr", "worst-10% mean", ":D"),
    ):
        axis.plot(
            snrs,
            [summary[(primary[snr], snr)][field] for snr in snrs],
            style,
            label=label,
        )
    axis.set_xlabel("nominal SNR (dB)")
    axis.set_ylabel("PSNR (dB)")
    axis.set_title("Conditional mean vs tail PSNR (primary Rayleigh arm)")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots / "psnr_mean_vs_tail.png", dpi=160)
    plt.close(figure)

    # Figure 1b: all arms, mean and p10 — the confound check.
    figure, axis = plt.subplots(figsize=(7.6, 4.8))
    for arm in arms:
        colour, marker = styles.get(arm, ("#868E96", "x"))
        axis.plot(
            snrs,
            [summary[(arm, snr)]["mean_psnr"] for snr in snrs],
            marker=marker,
            color=colour,
            linestyle="-",
            label=f"{arm} mean",
        )
        axis.plot(
            snrs,
            [summary[(arm, snr)]["p10_psnr"] for snr in snrs],
            marker=marker,
            color=colour,
            linestyle="--",
            alpha=0.65,
            label=f"{arm} p10",
        )
    axis.set_xlabel("nominal SNR (dB)")
    axis.set_ylabel("PSNR (dB)")
    axis.set_title("Mean (solid) and p10 (dashed) PSNR by arm")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(plots / "psnr_mean_and_tail_by_arm.png", dpi=160)
    plt.close(figure)

    # Figure 2: outage probability.
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for threshold in thresholds:
        axis.plot(
            snrs,
            [
                summary[(primary[snr], snr)][f"outage_psnr{int(threshold)}"]
                for snr in snrs
            ],
            marker="o",
            label=f"PSNR < {threshold:g} dB",
        )
    axis.plot(
        snrs,
        [
            summary[(primary[snr], snr)]["outage_relative_awgn_median_minus_3db"]
            for snr in snrs
        ],
        marker="x",
        linestyle=":",
        color="#212529",
        label="PSNR < AWGN median - 3 dB",
    )
    axis.set_xlabel("nominal SNR (dB)")
    axis.set_ylabel("outage probability")
    axis.set_title("Outage probability (primary Rayleigh arm)")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(plots / "outage_probability.png", dpi=160)
    plt.close(figure)

    # Figure 3: fading power vs distortion.
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for snr in snrs:
        arm = primary[snr]
        pooled = [item for image in images for item in grouped.get((arm, snr, image), [])]
        axis.scatter(
            [item["h_power"] for item in pooled],
            [item["psnr"] for item in pooled],
            s=3,
            alpha=0.25,
            label=f"{snr:g} dB",
        )
    axis.set_xscale("log")
    axis.set_xlabel(r"$|h|^2$")
    axis.set_ylabel("PSNR (dB)")
    axis.set_title("Fading power vs reconstruction quality (primary arm)")
    axis.grid(alpha=0.3)
    axis.legend(markerscale=4, fontsize=8)
    figure.tight_layout()
    figure.savefig(plots / "fading_power_vs_distortion.png", dpi=160)
    plt.close(figure)

    # Figure 4: PSNR distribution per SNR.
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    data = []
    for snr in snrs:
        arm = primary[snr]
        pooled = [item for image in images for item in grouped.get((arm, snr, image), [])]
        data.append([item["psnr"] for item in pooled])
    axis.boxplot(
        data,
        tick_labels=[f"{snr:g}" for snr in snrs],
        whis=(5, 95),
        showfliers=False,
    )
    axis.set_xlabel("nominal SNR (dB)")
    axis.set_ylabel("PSNR (dB)")
    axis.set_title("PSNR distribution per SNR (primary arm, 5-95% whiskers)")
    axis.grid(alpha=0.3, axis="y")
    figure.tight_layout()
    figure.savefig(plots / "psnr_distribution_by_snr.png", dpi=160)
    plt.close(figure)

    # Figure 4b: empirical CDF, all arms at the lowest SNR.
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    lowest = snrs[0]
    for arm in arms:
        pooled = sorted(
            item["psnr"]
            for image in images
            for item in grouped.get((arm, lowest, image), [])
        )
        colour, _ = styles.get(arm, ("#868E96", "x"))
        axis.plot(
            pooled,
            [(index + 1) / len(pooled) for index in range(len(pooled))],
            color=colour,
            label=arm,
        )
    axis.set_xlabel("PSNR (dB)")
    axis.set_ylabel("empirical CDF")
    axis.set_title(f"PSNR CDF at {lowest:g} dB nominal SNR, all arms")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(plots / "psnr_cdf_all_arms_lowest_snr.png", dpi=160)
    plt.close(figure)

    # Figure 5 support: variance decomposition.
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for arm in arms:
        colour, marker = styles.get(arm, ("#868E96", "x"))
        available = [snr for snr in snrs if (arm, snr) in decomposition]
        axis.plot(
            available,
            [
                decomposition[(arm, snr)]["channel_share_of_total_variance"]
                for snr in available
            ],
            marker=marker,
            color=colour,
            label=arm,
        )
    axis.axhline(0.5, color="#212529", linestyle=":", linewidth=1)
    axis.set_xlabel("nominal SNR (dB)")
    axis.set_ylabel("channel share of PSNR variance")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Within-image (channel) share of total PSNR variance")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(plots / "channel_variance_share.png", dpi=160)
    plt.close(figure)

    print(f"decision: {verdict['decision']}")
    print(f"primary arm per snr: {verdict['primary_arm_per_snr']}")
    for name, item in checks.items():
        print(f"  check {name}: {'PASS' if item['passed'] else 'FAIL'}")
    for name, value in blockers.items():
        print(f"  blocker {name}: {value}")
    print(f"wrote summary and plots under {directory}")


if __name__ == "__main__":
    main()
