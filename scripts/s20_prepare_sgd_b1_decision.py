#!/usr/bin/env python3
"""Freeze the stratified S20 SGD-JSCC versus B1 decision population."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(value: str | Path, expected: str) -> Path:
    path = resolve(value)
    if not path.is_file() or sha256_file(path) != str(expected):
        raise RuntimeError(f"missing or hash-mismatched frozen input: {path}")
    return path


def bool_value(value: str) -> bool:
    return str(value).strip().lower() == "true"


def main() -> None:
    config_path = resolve("configs/s20_sgd_b1_decision.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "preregistered_before_any_s20_outcome":
        raise RuntimeError("S20 decision config is not preregistered")
    population = config["population"]
    output = resolve(config["outputs"]["population"])
    if output.exists():
        raise FileExistsError(output)

    split_path = require_sha(
        population["split_manifest"], population["split_manifest_sha256"]
    )
    membership_path = require_sha(
        population["frozen_clean_membership_source"],
        population["frozen_clean_membership_source_sha256"],
    )
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("official_val_accessed") is not False:
        raise RuntimeError("split manifest does not keep official validation sealed")
    classes = [str(value) for value in split["classes"]]
    eligible_split = {
        str(item["sample_id"]): item
        for item in split["samples"]
        if str(item["split"]) == str(population["required_split"])
    }

    with membership_path.open(encoding="utf-8", newline="") as handle:
        membership_rows = list(csv.DictReader(handle))
    clean_by_sample: dict[str, bool] = {}
    row_count_by_sample: defaultdict[str, int] = defaultdict(int)
    for row in membership_rows:
        sample_id = str(row["sample_id"])
        row_count_by_sample[sample_id] += 1
        clean_by_sample[sample_id] = clean_by_sample.get(sample_id, True) and bool_value(
            row["clean_correct"]
        )

    prior = yaml.safe_load(resolve(population["prior_pilot_config"]).read_text(encoding="utf-8"))
    excluded = {str(item["sample_id"]) for item in prior["population"]["samples"]}
    by_class: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    source_root = resolve(split["source_train_root"])
    for sample_id, item in eligible_split.items():
        if sample_id in excluded or not clean_by_sample.get(sample_id, False):
            continue
        if row_count_by_sample[sample_id] != len(config["channel"]["snrs_db"]):
            continue
        path = source_root / str(item["relative_path"])
        if not path.is_file() or sha256_file(path) != str(item["content_sha256"]):
            raise RuntimeError(f"source content mismatch: {sample_id}")
        by_class[int(item["class_idx"])].append(dict(item))

    selected: list[dict[str, Any]] = []
    salt = str(population["rank_salt"])
    for class_idx, count in sorted(
        (int(key), int(value)) for key, value in population["per_class_counts"].items()
    ):
        ranked = sorted(
            by_class[class_idx],
            key=lambda item: hashlib.sha256(
                f"{salt}:{item['sample_id']}".encode("utf-8")
            ).hexdigest(),
        )
        if len(ranked) < count:
            raise RuntimeError(f"class {class_idx} has only {len(ranked)} eligible samples")
        selected.extend(ranked[:count])

    expected = int(population["expected_sample_count"])
    if len(selected) != expected or len({item["sample_id"] for item in selected}) != expected:
        raise RuntimeError("S20 selected population has the wrong size or duplicate IDs")
    if {item["sample_id"] for item in selected} & excluded:
        raise RuntimeError("S20 population overlaps the exposed eight-image pilot")

    reference = {
        "phase": config["phase"],
        "study": "sgd_b1_decision_population",
        "analysis_id": config["analysis_id"],
        "status": "preregistered_before_any_pilot_method_output",
        "created_at": config["created_at"],
        "official_val_accessed": False,
        "outcome_claims_allowed": False,
        "population": {
            "dataset": population["dataset"],
            "split_manifest": population["split_manifest"],
            "split_manifest_sha256": population["split_manifest_sha256"],
            "required_split": population["required_split"],
            "frozen_clean_membership_source": population["frozen_clean_membership_source"],
            "frozen_clean_membership_source_sha256": population[
                "frozen_clean_membership_source_sha256"
            ],
            "expected_sample_count": expected,
            "selection_rule": population["selection_rule"],
            "rank_salt": salt,
            "samples": [
                {
                    "sample_id": item["sample_id"],
                    "class_idx": int(item["class_idx"]),
                    "content_sha256": item["content_sha256"],
                }
                for item in selected
            ],
        },
        "channel": config["channel"],
        "evaluator": config["evaluator"],
    }

    output.mkdir(parents=True, exist_ok=False)
    reference_path = output / "population_reference.yaml"
    reference_path.write_text(
        yaml.safe_dump(reference, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    manifest = {
        "analysis_id": config["analysis_id"],
        "official_val_accessed": False,
        "expected_sample_count": expected,
        "excluded_prior_pilot_count": len(excluded),
        "per_class_counts": {
            str(index): sum(int(item["class_idx"]) == index for item in selected)
            for index in range(len(classes))
        },
        "sample_ids": [str(item["sample_id"]) for item in selected],
        "population_reference_sha256": sha256_file(reference_path),
    }
    (output / "population_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy2(config_path, output / "master_config_before_population_freeze.yaml")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
