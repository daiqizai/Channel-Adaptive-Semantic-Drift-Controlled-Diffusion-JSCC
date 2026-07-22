#!/usr/bin/env python3
"""Verify that the paired-reference kernel exactly replays the frozen sender M3 rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_REFERENCE_REDEFINITION = {
    "reference_final_correct",
    "reference_final_psnr",
    "reference_final_lpips",
}


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key(row: dict[str, str]) -> tuple[int, float, str]:
    return int(row["channel_seed"]), float(row["snr_db"]), row["sample_id"]


def load_rows(path: Path) -> tuple[dict[tuple[int, float, str], dict[str, str]], list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows: dict[tuple[int, float, str], dict[str, str]] = {}
        for row in reader:
            row_key = key(row)
            if row_key in rows:
                raise RuntimeError(f"duplicate row key in {path}: {row_key}")
            rows[row_key] = row
    return rows, fields


def verify(old_path: Path, paired_path: Path) -> dict[str, Any]:
    old_rows, old_fields = load_rows(old_path)
    paired_rows, paired_fields = load_rows(paired_path)
    if old_fields != paired_fields:
        raise RuntimeError("old and paired replay CSV schemas differ")
    if set(old_rows) != set(paired_rows):
        raise RuntimeError("old and paired replay row grids differ")
    mismatches: dict[str, int] = {}
    examples: dict[str, Any] = {}
    for row_key in sorted(old_rows):
        old = old_rows[row_key]
        paired = paired_rows[row_key]
        for field in old_fields:
            if old[field] == paired[field]:
                continue
            mismatches[field] = mismatches.get(field, 0) + 1
            examples.setdefault(field, {"key": row_key, "old": old[field], "paired": paired[field]})
    forbidden = sorted(set(mismatches) - ALLOWED_REFERENCE_REDEFINITION)
    if forbidden:
        raise RuntimeError(f"paired kernel changed frozen row fields: {forbidden}")
    return {
        "status": "PASS",
        "rows": len(old_rows),
        "schema_columns": len(old_fields),
        "row_grid_identical": True,
        "all_m3_and_reference_anchor_raw_fields_exact": True,
        "allowed_reference_redefinition": sorted(ALLOWED_REFERENCE_REDEFINITION),
        "observed_difference_counts": mismatches,
        "difference_examples": examples,
        "old_per_sample_csv": str(old_path),
        "old_per_sample_csv_sha256": sha256_file(old_path),
        "paired_per_sample_csv": str(paired_path),
        "paired_per_sample_csv_sha256": sha256_file(paired_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old",
        default=(
            "outputs/analysis/pc_imagenette_sender_crossmodel_triplet_seed20260727_audit/"
            "per_sample.csv"
        ),
    )
    parser.add_argument(
        "--paired",
        default=(
            "outputs/analysis/pc_imagenette_sender_crossmodel_triplet_seed20260727_"
            "paired_replay_v2/per_sample.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/analysis/pc_imagenette_sender_crossmodel_triplet_seed20260727_"
            "paired_replay_verification"
        ),
    )
    args = parser.parse_args()
    output = resolve(args.output_dir)
    if output.exists():
        raise FileExistsError(output)
    payload = verify(resolve(args.old), resolve(args.paired))
    output.mkdir(parents=True, exist_ok=False)
    (output / "verification.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
