#!/usr/bin/env python3
"""Add the conservative same-PyTorch-runtime S34D latency comparison."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = (
    ROOT
    / "outputs/analysis/ANALYSIS-S34D-GENERATIVE-INFERENCE-COST-001"
)


def load(name: str) -> dict[str, Any]:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def main() -> None:
    json_path = OUT / "common_runtime_post_analysis.json"
    csv_path = OUT / "latency_comparison_common_torch21.csv"
    if json_path.exists() or csv_path.exists():
        raise FileExistsError("common-runtime post analysis already exists")
    native = load("s33/summary.json")
    common = load("s33_torch21_sensitivity/summary.json")
    diff = load("diffjscc/summary.json")
    sgd = load("sgd/summary.json")
    if common["torch_version"] != diff["torch_version"] or common["torch_version"] != sgd["torch_version"]:
        raise RuntimeError("common-runtime versions are not identical")
    s33_ms = float(common["latency"]["receiver_wall_ms"]["mean"])
    rows: list[dict[str, Any]] = [
        {
            "method": "S33 strong",
            "steps": 0,
            "mean_ms": s33_ms,
            "median_ms": common["latency"]["receiver_wall_ms"]["median"],
            "slowdown_vs_S33_common_runtime": 1.0,
        }
    ]
    for point in diff["quality_curve"]:
        rows.append(
            {
                "method": "DiffJSCC",
                "steps": int(point["steps"]),
                "mean_ms": float(point["receiver_wall_ms"]["mean"]),
                "median_ms": float(point["receiver_wall_ms"]["median"]),
                "slowdown_vs_S33_common_runtime": float(
                    point["receiver_wall_ms"]["mean"]
                )
                / s33_ms,
            }
        )
    sgd_ms = float(sgd["latency"]["receiver_wall_ms"]["mean"])
    rows.append(
        {
            "method": "SGD paper upper",
            "steps": 50,
            "mean_ms": sgd_ms,
            "median_ms": sgd["latency"]["receiver_wall_ms"]["median"],
            "slowdown_vs_S33_common_runtime": sgd_ms / s33_ms,
        }
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    point25 = next(row for row in rows if row["method"] == "DiffJSCC" and row["steps"] == 25)
    point100 = next(row for row in rows if row["method"] == "DiffJSCC" and row["steps"] == 100)
    payload = {
        "status": "PASS",
        "reason": "S33 was remeasured under the exact PyTorch 2.1.0+cu121 runtime used by DiffJSCC and SGD, on the same GPU and entry contract.",
        "same_runtime": common["torch_version"],
        "s33_common_runtime_mean_ms": s33_ms,
        "s33_native_runtime_mean_ms": float(
            native["latency"]["receiver_wall_ms"]["mean"]
        ),
        "diffjscc_25step_mean_ms": point25["mean_ms"],
        "diffjscc_25step_slowdown_common_runtime": point25[
            "slowdown_vs_S33_common_runtime"
        ],
        "diffjscc_25step_slowdown_native_runtime": point25["mean_ms"]
        / float(native["latency"]["receiver_wall_ms"]["mean"]),
        "diffjscc_100step_slowdown_common_runtime": point100[
            "slowdown_vs_S33_common_runtime"
        ],
        "sgd_slowdown_common_runtime": sgd_ms / s33_ms,
        "interpretation": "Use the common-runtime slowdown as the conservative primary ms claim; retain native-runtime and FLOPs ratios as sensitivity evidence.",
        "quality_and_flops_unchanged": True,
        "new_training": False,
        "network_download": False,
        "official_imagenette_validation_accessed": False,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
