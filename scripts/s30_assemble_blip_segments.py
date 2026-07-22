#!/usr/bin/env python3
"""Fail-closed assembly of explicitly ordered S30 BLIP2 transport segments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_yaml(value: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(resolve(value).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("config root must be a mapping")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/s30_diffjscc_external_comparison.yaml")
    parser.add_argument("--weight-index", type=int, required=True)
    parser.add_argument("--segments", nargs="+", required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    weights = config["assets"]["blip_weight_files"]
    if args.weight_index < 0 or args.weight_index >= len(weights):
        raise IndexError(args.weight_index)
    spec = weights[args.weight_index]
    destination = resolve(config["assets"]["blip_weights_directory"]) / str(spec["name"])
    assembling = destination.with_name(destination.name + ".assembling")
    if destination.exists() or assembling.exists():
        raise FileExistsError(f"refusing to overwrite: {destination} / {assembling}")

    segments = [resolve(value) for value in args.segments]
    if len(set(segments)) != len(segments):
        raise RuntimeError("duplicate segment path")
    missing = [str(path) for path in segments if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing segments: {missing}")
    observed_segment_bytes = [path.stat().st_size for path in segments]
    expected_bytes = int(spec["bytes"])
    if sum(observed_segment_bytes) != expected_bytes:
        raise RuntimeError(
            f"segment byte sum mismatch: {sum(observed_segment_bytes)} != {expected_bytes}"
        )

    digest = hashlib.sha256()
    written = 0
    with assembling.open("xb") as output:
        for path in segments:
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    output.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
        output.flush()
        os.fsync(output.fileno())
    observed_sha = digest.hexdigest()
    if written != expected_bytes or observed_sha != str(spec["sha256"]):
        raise RuntimeError(
            "assembled BLIP2 weight failed exact validation: "
            f"bytes={written}/{expected_bytes}, sha256={observed_sha}/{spec['sha256']}; "
            f"preserved for diagnosis at {assembling}"
        )
    os.replace(assembling, destination)
    print(
        json.dumps(
            {
                "status": "PASS",
                "weight_index": args.weight_index,
                "destination": str(destination.relative_to(ROOT)),
                "segments": [str(path.relative_to(ROOT)) for path in segments],
                "segment_bytes": observed_segment_bytes,
                "bytes": written,
                "sha256": observed_sha,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
