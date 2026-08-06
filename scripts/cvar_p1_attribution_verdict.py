#!/usr/bin/env python3
"""P1 attribution verdict: INCONCLUSIVE / END-CVAR / ENTER-CVAR.

Implements the decision table frozen in
``reports/cvar_p1_rayleigh_matched_preregistration_2026-07-31.md`` section 4.
Every threshold below was fixed before the matched model produced any output.

Competence gate (guards against a degenerate model whose tail is small only
because it is uniformly bad):
    aggregate mean PSNR >= P0 best Rayleigh arm aggregate, AND
    no per-SNR regression worse than 0.5 dB against that arm.

Residual tail gate:
    magnitude    median - p10 >= 2.0 dB at >= 2 of the 5 SNR points, AND
    attribution  at those triggering points only, channel variance share >= 0.5
                 and Spearman(PSNR, |h|^2) >= 0.5.

The attribution clause is evaluated only at the triggering SNR points, which
fixes the P0 specification defect where an all-SNR conjunction was applied to a
gate whose magnitude clause required only two points.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

# --- frozen thresholds, see preregistration section 4 -----------------------
COMPETENCE_AGGREGATE_MIN_DB = 27.958479
COMPETENCE_PER_SNR_REFERENCE_DB = {
    1.0: 24.5588,
    4.0: 26.4680,
    7.0: 27.9057,
    13.0: 29.9419,
    19.0: 30.9180,
}
COMPETENCE_MAX_PER_SNR_REGRESSION_DB = 0.5
TAIL_MAGNITUDE_MIN_DB = 2.0
TAIL_MAGNITUDE_MIN_POINTS = 2
ATTRIBUTION_MIN_CHANNEL_SHARE = 0.5
ATTRIBUTION_MIN_SPEARMAN = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matched-directory",
        default="outputs/analysis/ANALYSIS-CVAR-P1-MATCHED-TAIL-RISK-001",
    )
    parser.add_argument(
        "--baseline-directory",
        default="outputs/analysis/ANALYSIS-CVAR-P0-TAIL-RISK-001",
    )
    parser.add_argument("--primary-arm", default="rayleigh_effective_csi")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_summary(directory: Path) -> dict[tuple[str, float], dict[str, Any]]:
    path = directory / "diagnostic_summary.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (row["arm"], float(row["snr_db"])): row for row in csv.DictReader(handle)
        }


def read_decomposition(directory: Path) -> dict[tuple[str, float], dict[str, Any]]:
    path = directory / "variance_decomposition.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (row["arm"], float(row["snr_db"])): row for row in csv.DictReader(handle)
        }


def evaluate_arm(
    arm: str,
    summary: dict[tuple[str, float], dict[str, Any]],
    decomposition: dict[tuple[str, float], dict[str, Any]],
    snrs: list[float],
) -> dict[str, Any]:
    means = {snr: float(summary[(arm, snr)]["mean_psnr"]) for snr in snrs}
    aggregate = statistics.mean(means.values())
    regressions = {
        snr: COMPETENCE_PER_SNR_REFERENCE_DB[snr] - means[snr]
        for snr in snrs
        if snr in COMPETENCE_PER_SNR_REFERENCE_DB
    }
    worst_regression = max(regressions.values()) if regressions else float("nan")
    competence = (
        aggregate >= COMPETENCE_AGGREGATE_MIN_DB
        and worst_regression <= COMPETENCE_MAX_PER_SNR_REGRESSION_DB
    )

    gaps = {
        snr: float(summary[(arm, snr)]["delta_tail_median_minus_p10"]) for snr in snrs
    }
    triggering = [snr for snr in snrs if gaps[snr] >= TAIL_MAGNITUDE_MIN_DB]
    magnitude = len(triggering) >= TAIL_MAGNITUDE_MIN_POINTS

    shares = {
        snr: float(decomposition[(arm, snr)]["channel_share_of_total_variance"])
        for snr in triggering
        if (arm, snr) in decomposition
    }
    spearman = {
        snr: float(summary[(arm, snr)]["spearman_psnr_vs_h_power_pooled"])
        for snr in triggering
    }
    attribution = bool(triggering) and all(
        shares.get(snr, 0.0) >= ATTRIBUTION_MIN_CHANNEL_SHARE
        and spearman.get(snr, 0.0) >= ATTRIBUTION_MIN_SPEARMAN
        for snr in triggering
    )
    residual_tail = magnitude and attribution

    if not competence:
        decision = "INCONCLUSIVE"
    elif residual_tail:
        decision = "ENTER-CVAR"
    else:
        decision = "END-CVAR"

    return {
        "arm": arm,
        "mean_psnr_per_snr": means,
        "aggregate_mean_psnr": aggregate,
        "competence": {
            "passed": competence,
            "aggregate_required_db": COMPETENCE_AGGREGATE_MIN_DB,
            "aggregate_actual_db": aggregate,
            "per_snr_regression_vs_p0_best_arm_db": regressions,
            "worst_regression_db": worst_regression,
            "max_allowed_regression_db": COMPETENCE_MAX_PER_SNR_REGRESSION_DB,
        },
        "residual_tail": {
            "passed": residual_tail,
            "median_minus_p10_db": gaps,
            "triggering_snr_points": triggering,
            "magnitude_passed": magnitude,
            "attribution_passed": attribution,
            "channel_share_at_triggering": shares,
            "spearman_at_triggering": spearman,
        },
        "decision": decision,
    }


def main() -> None:
    args = parse_args()
    matched = resolve(args.matched_directory)
    baseline = resolve(args.baseline_directory)
    summary = read_summary(matched)
    decomposition = read_decomposition(matched)
    baseline_summary = read_summary(baseline)
    snrs = sorted({key[1] for key in summary})

    primary = evaluate_arm(args.primary_arm, summary, decomposition, snrs)

    # Anomaly check: if another Rayleigh arm has a higher aggregate mean PSNR
    # than the a-priori primary arm, the preregistration requires evaluating the
    # tail gate on that arm too.
    others = []
    for arm in ("rayleigh_nominal_csi", "rayleigh_effective_csi_clamped"):
        if (arm, snrs[0]) not in summary or arm == args.primary_arm:
            continue
        aggregate = statistics.mean(
            float(summary[(arm, snr)]["mean_psnr"]) for snr in snrs
        )
        if aggregate > primary["aggregate_mean_psnr"]:
            others.append(evaluate_arm(arm, summary, decomposition, snrs))

    decisions = {primary["arm"]: primary["decision"]}
    for item in others:
        decisions[item["arm"]] = item["decision"]
    unanimous = len(set(decisions.values())) == 1
    final = primary["decision"] if unanimous else "SPLIT_SEE_REPORT"

    metadata = json.loads((matched / "run_metadata.json").read_text(encoding="utf-8"))
    verdict = {
        "analysis_id": "ANALYSIS-CVAR-P1-MATCHED-TAIL-RISK-001",
        "preregistration": (
            "reports/cvar_p1_rayleigh_matched_preregistration_2026-07-31.md"
        ),
        "matched_checkpoint_sha256": metadata["checkpoint_sha256"],
        "git_commit": metadata["git_commit"],
        "source_count": metadata["source_count"],
        "realizations_per_image": metadata["num_channel_realizations"],
        "frozen_thresholds": {
            "competence_aggregate_min_db": COMPETENCE_AGGREGATE_MIN_DB,
            "competence_per_snr_reference_db": COMPETENCE_PER_SNR_REFERENCE_DB,
            "competence_max_per_snr_regression_db": COMPETENCE_MAX_PER_SNR_REGRESSION_DB,
            "tail_magnitude_min_db": TAIL_MAGNITUDE_MIN_DB,
            "tail_magnitude_min_points": TAIL_MAGNITUDE_MIN_POINTS,
            "attribution_min_channel_share": ATTRIBUTION_MIN_CHANNEL_SHARE,
            "attribution_min_spearman": ATTRIBUTION_MIN_SPEARMAN,
        },
        "primary_arm_evaluation": primary,
        "higher_mean_arms_also_evaluated": others,
        "p0_baseline_best_arm_aggregate_db": statistics.mean(
            float(baseline_summary[("rayleigh_nominal_csi", snr)]["mean_psnr"])
            for snr in snrs
        ),
        "decision": final,
    }
    (matched / "attribution_verdict.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8"
    )

    print(f"primary arm: {primary['arm']}")
    print(
        f"  competence: {'PASS' if primary['competence']['passed'] else 'FAIL'} "
        f"(aggregate {primary['aggregate_mean_psnr']:.4f} dB, "
        f"required >= {COMPETENCE_AGGREGATE_MIN_DB:.4f} dB, "
        f"worst per-SNR regression {primary['competence']['worst_regression_db']:+.4f} dB)"
    )
    print(
        f"  residual tail: {'PASS' if primary['residual_tail']['passed'] else 'FAIL'} "
        f"(magnitude {primary['residual_tail']['magnitude_passed']}, "
        f"attribution {primary['residual_tail']['attribution_passed']}, "
        f"triggering {primary['residual_tail']['triggering_snr_points']})"
    )
    for snr in snrs:
        print(
            f"    snr={snr:5.1f}  mean={primary['mean_psnr_per_snr'][snr]:7.3f}  "
            f"median-p10={primary['residual_tail']['median_minus_p10_db'][snr]:6.3f} dB"
        )
    for item in others:
        print(f"  anomaly arm {item['arm']} -> {item['decision']}")
    print(f"\nDecision: {final}")


if __name__ == "__main__":
    main()
