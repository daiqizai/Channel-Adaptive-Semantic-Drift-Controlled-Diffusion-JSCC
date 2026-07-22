#!/usr/bin/env python3
"""Audit and derive the frozen S33 equal-rate strong-vs-author comparison."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cadsd_jscc.external_common import (  # noqa: E402
    canonical_noise_sha256,
    canonical_standard_normal,
)


OUTPUT = ROOT / "outputs/external_baselines/ANALYSIS-S33-STRONG-JSCC-16384-COMPARISON-001"
CONFIG = ROOT / "configs/s33_strong_jscc_16384_external_comparison.yaml"
SCRIPT = Path(__file__).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def cluster_ci(
    rows: list[dict[str, str]], field: str, replicates: int, seed: int
) -> list[float]:
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["sample_id"]].append(float(row[field]))
    keys = sorted(grouped)
    values = np.asarray([np.mean(grouped[key]) for key in keys], dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(replicates, len(values)))
    bootstrap = values[indices].mean(axis=1)
    return [
        float(np.quantile(bootstrap, 0.025)),
        float(np.quantile(bootstrap, 0.975)),
    ]


def mean(rows: list[dict[str, str]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def main() -> None:
    post_path = OUTPUT / "post_analysis.json"
    table_path = OUTPUT / "per_snr_comparison.csv"
    if post_path.exists() or table_path.exists():
        raise FileExistsError("S33 post-analysis output already exists")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    snapshot = OUTPUT / "config_snapshot.yaml"
    summary_path = OUTPUT / "summary.json"
    rows_path = OUTPUT / "per_sample.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = read_csv(rows_path)
    expected = int(config["population"]["expected_rows"])
    keys = {
        (row["sample_id"], int(row["base_seed"]), float(row["snr_db"]))
        for row in rows
    }
    if len(rows) != expected or len(keys) != expected:
        raise RuntimeError("S33 row count or key uniqueness failed")
    if sha256_file(CONFIG) != sha256_file(snapshot):
        raise RuntimeError("S33 config snapshot differs from frozen config")
    if sha256_file(CONFIG) != summary["audit"]["config_sha256"]:
        raise RuntimeError("S33 summary config SHA differs")
    checkpoint = ROOT / config["inputs"]["strong_checkpoint"]
    if sha256_file(checkpoint) != config["inputs"]["strong_checkpoint_sha256"]:
        raise RuntimeError("S33 checkpoint SHA differs")
    s30_path = ROOT / config["inputs"]["s30_per_sample"]
    if sha256_file(s30_path) != config["inputs"]["s30_per_sample_sha256"]:
        raise RuntimeError("S30 source CSV SHA differs")
    s30 = {
        (row["sample_id"], int(row["base_seed"]), float(row["snr_db"])): row
        for row in read_csv(s30_path)
    }
    reference_symbols = int(config["rate"]["canonical_noise_reference_real_symbols"])
    strong_symbols = int(config["rate"]["strong_real_symbols"])
    for row in rows:
        key = (row["sample_id"], int(row["base_seed"]), float(row["snr_db"]))
        old = s30[key]
        for suffix in ("prediction", "failure", "psnr", "ms_ssim", "lpips"):
            if row[f"author_jscc_{suffix}"] != old[f"author_jscc_{suffix}"]:
                raise RuntimeError(f"author row changed for {key}/{suffix}")
        full = canonical_standard_normal(key[1], key[0], key[2], reference_symbols)
        if canonical_noise_sha256(full) != row["canonical_noise_sha256"]:
            raise RuntimeError(f"full canonical noise mismatch for {key}")
        prefix = full[:strong_symbols].contiguous()
        if canonical_noise_sha256(prefix) != row["strong_noise_prefix_sha256"]:
            raise RuntimeError(f"strong noise-prefix mismatch for {key}")
        for field, value in row.items():
            if field.endswith(("_psnr", "_ms_ssim", "_lpips")) and not math.isfinite(
                float(value)
            ):
                raise RuntimeError(f"non-finite {field} for {key}")

    replicates = int(config["metrics"]["bootstrap_replicates"])
    seed = int(config["metrics"]["bootstrap_seed"])
    fields = {
        "psnr": "strong_minus_author_jscc_psnr",
        "ms_ssim": "strong_minus_author_jscc_ms_ssim",
        "lpips": "strong_minus_author_jscc_lpips",
        "failure": "strong_minus_author_jscc_failure",
    }
    aggregate = {
        metric: {
            "delta": mean(rows, field),
            "source_cluster_95ci": cluster_ci(rows, field, replicates, seed),
        }
        for metric, field in fields.items()
    }
    margin = float(config["claim_rule"]["noninferiority_margin_db"])
    lower = aggregate["psnr"]["source_cluster_95ci"][0]
    if lower > 0:
        verdict = "SIGNIFICANTLY_SUPERIOR"
    elif lower > -margin:
        verdict = "NONINFERIOR_WITHIN_0P10_DB"
    elif lower < -margin:
        verdict = "INFERIOR_UNDER_0P10_DB_MARGIN"
    else:
        verdict = "MARGIN_BOUNDARY_UNCERTAIN"

    per_snr: list[dict[str, Any]] = []
    for snr in map(float, config["population"]["snrs_db"]):
        selected = [row for row in rows if float(row["snr_db"]) == snr]
        record: dict[str, Any] = {"snr_db": snr, "rows": len(selected)}
        for method in ("strong", "author_jscc"):
            for metric in ("psnr", "ms_ssim", "lpips"):
                record[f"{method}_{metric}"] = mean(selected, f"{method}_{metric}")
            record[f"{method}_failures"] = sum(
                as_bool(row[f"{method}_failure"]) for row in selected
            )
        for metric, field in fields.items():
            record[f"delta_{metric}"] = mean(selected, field)
            record[f"delta_{metric}_ci_low"], record[f"delta_{metric}_ci_high"] = (
                cluster_ci(selected, field, replicates, seed + int(snr))
            )
        per_snr.append(record)

    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_snr[0]))
        writer.writeheader()
        writer.writerows(per_snr)
    payload = {
        "analysis_id": config["analysis_id"],
        "status": "PASS_INDEPENDENT_ARTIFACT_AUDIT",
        "claim_scope": config["protocol"]["claim_scope"],
        "psnr_margin_verdict": verdict,
        "noninferiority_margin_db": margin,
        "aggregate": aggregate,
        "aggregate_all_quality_metrics_significantly_favor_strong": bool(
            aggregate["psnr"]["source_cluster_95ci"][0] > 0
            and aggregate["ms_ssim"]["source_cluster_95ci"][0] > 0
            and aggregate["lpips"]["source_cluster_95ci"][1] < 0
        ),
        "aggregate_failure_significantly_favors_strong": bool(
            aggregate["failure"]["source_cluster_95ci"][1] < 0
        ),
        "semantic_transitions": summary["strong_vs_author_semantic_transitions"],
        "per_snr": per_snr,
        "audit": {
            "rows": len(rows),
            "unique_keys": len(keys),
            "config_sha256": sha256_file(CONFIG),
            "checkpoint_sha256": sha256_file(checkpoint),
            "per_sample_sha256": sha256_file(rows_path),
            "runner_summary_sha256": sha256_file(summary_path),
            "per_snr_table_sha256": sha256_file(table_path),
            "post_script": str(SCRIPT.relative_to(ROOT)),
            "post_script_sha256": sha256_file(SCRIPT),
            "canonical_full_and_prefix_noise_rows_verified": len(rows),
            "author_rows_exactly_match_s30": len(rows),
            "official_imagenette_validation_accessed": False,
        },
        "runner_label_correction": (
            "The immutable runner summary retained an S31 wording in claim_boundary; "
            "the frozen config and this derived audit correctly identify S33. Metrics are unchanged."
        ),
    }
    post_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
