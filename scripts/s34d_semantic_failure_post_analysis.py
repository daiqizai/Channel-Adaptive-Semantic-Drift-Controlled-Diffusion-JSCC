#!/usr/bin/env python3
"""Post-analyze S34D semantic-failure deltas on the frozen quality curve."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = (
    ROOT
    / "outputs/analysis/ANALYSIS-S34D-GENERATIVE-INFERENCE-COST-001"
    / "diffjscc/quality_rows.csv"
)
OUTPUT = INPUT.parents[1] / "semantic_failure_post_analysis.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    with INPUT.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for steps in (100, 50, 25, 10, 4):
        subset = [row for row in rows if int(row["steps"]) == steps]
        grouped: defaultdict[str, list[float]] = defaultdict(list)
        for row in subset:
            grouped[row["sample_id"]].append(
                float(row["failure"].lower() == "true")
                - float(row["s33_failure"].lower() == "true")
            )
        values = np.asarray(
            [np.mean(grouped[key]) for key in sorted(grouped)], dtype=np.float64
        )
        rng = np.random.default_rng(20260788 + steps + 1000)
        indices = rng.integers(0, len(values), size=(10000, len(values)))
        bootstrap = values[indices].mean(axis=1)
        result.append(
            {
                "steps": steps,
                "diffjscc_failures": sum(
                    row["failure"].lower() == "true" for row in subset
                ),
                "s33_failures": sum(
                    row["s33_failure"].lower() == "true" for row in subset
                ),
                "new_errors_vs_s33": sum(
                    row["failure"].lower() == "true"
                    and row["s33_failure"].lower() == "false"
                    for row in subset
                ),
                "repairs_vs_s33": sum(
                    row["failure"].lower() == "false"
                    and row["s33_failure"].lower() == "true"
                    for row in subset
                ),
                "failure_rate_delta_vs_s33": float(values.mean()),
                "failure_rate_delta_ci95": [
                    float(np.quantile(bootstrap, 0.025)),
                    float(np.quantile(bootstrap, 0.975)),
                ],
            }
        )
    OUTPUT.write_text(
        json.dumps(
            {
                "status": "PASS",
                "bootstrap_unit": "source_image",
                "bootstrap_replicates": 10000,
                "posthoc_role": "semantic safety interpretation; LPIPS step gate remains preregistered and unchanged",
                "rows": result,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
