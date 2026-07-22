#!/usr/bin/env python3
"""Compare the frozen c=6+c=2 guided system against the equal-rate c=8 reference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.utils import save_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/s7_matched_rate_system_cross_split_comparison.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def snr_name(snr: float) -> str:
    return f"snr_{int(snr):02d}db" if float(snr).is_integer() else f"snr_{str(snr).replace('.', 'p')}db"


def image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        return TF.to_tensor(image.convert("RGB"))


def psnr(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    mse = float(torch.mean((reference - candidate).square()).item())
    return -10.0 * math.log10(max(mse, 1e-12))


def read_semantic(path: Path) -> dict[tuple[str, float], dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output: dict[tuple[str, float], dict[str, str]] = {}
    for row in rows:
        key = (str(row["sample"]), float(row["snr_db"]))
        if key in output:
            raise RuntimeError(f"Duplicate semantic row in {path}: {key}")
        output[key] = row
    return output


def validate_rate(config: dict[str, Any]) -> dict[str, Any]:
    rate = config["rate"]
    denominator = int(rate["denominator"])
    reference = int(rate["reference_inner_channel"])
    main = int(rate["matched_main_inner_channel"])
    structure = int(rate["matched_structure_inner_channel"])
    total = int(rate["matched_total_inner_channel"])
    if main + structure != total or total != reference:
        raise RuntimeError(f"Rate mismatch: {main}+{structure}!={total}!={reference}")
    for key, numerator in (
        ("reference_cbr", reference),
        ("matched_main_cbr", main),
        ("matched_structure_cbr", structure),
        ("matched_total_cbr", total),
    ):
        if not math.isclose(float(rate[key]), numerator / denominator, abs_tol=1e-12):
            raise RuntimeError(f"CBR mismatch for {key}")
    return {"reference": reference, "main": main, "structure": structure, "total": total, "denominator": denominator}


def bootstrap(values: np.ndarray, replicates: int, seed: int) -> dict[str, Any]:
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Bootstrap requires a non-empty vector")
    rng = np.random.default_rng(seed)
    sampled = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 256):
        count = min(256, replicates - start)
        indices = rng.integers(0, len(values), size=(count, len(values)), endpoint=False)
        sampled[start : start + count] = values[indices].mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return {
        "estimate": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "replicates": replicates,
        "num_image_clusters": int(len(values)),
    }


def matched_paths(output_dir: Path, snr: float, name: str, validation: bool) -> dict[str, Path]:
    folder = snr_name(snr)
    return {
        "raw": output_dir / "exports" / folder / "refined" / name,
        "top1": output_dir / "exports" / folder / ("final" if validation else "top1_equal_final") / name,
        "confidence": output_dir / "exports" / folder / ("final" if validation else "candidate_final") / name,
    }


def make_visual(
    output_dir: Path,
    rows: list[dict[str, Any]],
    snr: float,
    count: int,
) -> Path:
    selected_ids = sorted({str(row["image_id"]) for row in rows if float(row["snr_db"]) == snr})[:count]
    tensors: list[torch.Tensor] = []
    for role in ("original_path", "reference_path", "main_c6_path", "matched_raw_path"):
        for image_id in selected_ids:
            row = next(
                item
                for item in rows
                if item["image_id"] == image_id and float(item["snr_db"]) == snr
            )
            tensors.append(image_tensor(resolve(row[role])))
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    path = sample_dir / f"{snr_name(snr)}_original_referencec8_mainc6_matchedraw.png"
    save_image(torch.stack(tensors), path, nrow=len(selected_ids), padding=2)
    return path


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    rate_contract = validate_rate(config)
    snrs = [float(value) for value in config["evaluation"]["snrs"]]
    primary_splits = set(str(value) for value in config["evaluation"]["primary_frozen_splits"])
    split_configs = config["inputs"]["splits"]
    plan = {
        "analysis_id": config["analysis_id"],
        "analysis_mode": config["analysis_mode"],
        "rate_contract": rate_contract,
        "snrs": snrs,
        "splits": [item["name"] for item in split_configs],
        "primary_frozen_splits": sorted(primary_splits),
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, "plan": plan}, indent=2, ensure_ascii=False))
        return
    output_dir = resolve(args.output_dir or config["outputs"]["output_dir"])
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output_dir}")
        analysis_root = (PROJECT_ROOT / "outputs" / "analysis").resolve()
        if analysis_root not in output_dir.resolve().parents:
            raise RuntimeError(f"Unsafe overwrite target: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    shutil.copy2(config_path, output_dir / "config.yaml")
    shutil.copy2(SCRIPT_PATH, output_dir / SCRIPT_PATH.name)
    save_json(output_dir / "run_plan.json", plan)

    original_dir = resolve(config["inputs"]["original_dir"])
    reference_export = resolve(config["inputs"]["reference_export_dir"])
    matched_export = resolve(config["inputs"]["matched_export_dir"])
    all_rows: list[dict[str, Any]] = []
    input_hash_rows: list[dict[str, str]] = []
    for split_config in split_configs:
        split = str(split_config["name"])
        start = int(split_config["sample_start"])
        count = int(split_config["sample_count"])
        names = [f"sample_{index:06d}.png" for index in range(start, start + count)]
        reference_semantic = read_semantic(resolve(split_config["reference_semantic_csv"]))
        matched_semantic = read_semantic(resolve(split_config["matched_semantic_csv"]))
        expected = {(name, snr) for name in names for snr in snrs}
        if set(reference_semantic) != expected or set(matched_semantic) != expected:
            raise RuntimeError(f"Semantic grid mismatch for split {split}")
        matched_output = resolve(split_config["matched_output_dir"])
        validation = split == "validation"
        for snr in snrs:
            folder = snr_name(snr)
            for name in names:
                paths = {
                    "original": original_dir / name,
                    "reference": reference_export / "exports" / folder / "reconstruction" / name,
                    "main_c6": matched_export / "exports" / folder / "main_reconstruction" / name,
                    **matched_paths(matched_output, snr, name, validation),
                }
                for role, path in paths.items():
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    input_hash_rows.append(
                        {"split": split, "snr_db": str(snr), "sample": name, "role": role, "path": relative(path), "sha256": sha256_file(path)}
                    )
                tensors = {role: image_tensor(path) for role, path in paths.items()}
                reference_psnr = psnr(tensors["original"], tensors["reference"])
                main_psnr = psnr(tensors["original"], tensors["main_c6"])
                raw_psnr = psnr(tensors["original"], tensors["raw"])
                top1_psnr = psnr(tensors["original"], tensors["top1"])
                confidence_psnr = psnr(tensors["original"], tensors["confidence"])
                ref_sem = reference_semantic[(name, snr)]
                mat_sem = matched_semantic[(name, snr)]
                reference_failure = not bool_value(ref_sem["m0_matches_original_top1"])
                main_failure = not bool_value(mat_sem["m0_matches_original_top1"])
                raw_failure = not bool_value(mat_sem["refined_matches_original_top1"])
                all_rows.append(
                    {
                        "split": split,
                        "primary_frozen_split": split in primary_splits,
                        "image_id": f"{split}:{name}",
                        "sample": name,
                        "snr_db": snr,
                        "original_path": relative(paths["original"]),
                        "reference_path": relative(paths["reference"]),
                        "main_c6_path": relative(paths["main_c6"]),
                        "matched_raw_path": relative(paths["raw"]),
                        "reference_c8_psnr_db": reference_psnr,
                        "main_c6_psnr_db": main_psnr,
                        "matched_raw_psnr_db": raw_psnr,
                        "matched_top1_psnr_db": top1_psnr,
                        "matched_confidence_psnr_db": confidence_psnr,
                        "main_c6_minus_reference_psnr_db": main_psnr - reference_psnr,
                        "matched_raw_minus_reference_psnr_db": raw_psnr - reference_psnr,
                        "matched_raw_minus_main_c6_psnr_db": raw_psnr - main_psnr,
                        "matched_top1_minus_reference_psnr_db": top1_psnr - reference_psnr,
                        "matched_confidence_minus_reference_psnr_db": confidence_psnr - reference_psnr,
                        "reference_c8_failure": reference_failure,
                        "main_c6_failure": main_failure,
                        "matched_raw_failure": raw_failure,
                        "matched_raw_minus_reference_failure": int(raw_failure) - int(reference_failure),
                        "matched_raw_new_error_vs_main_c6": (not main_failure) and raw_failure,
                        "matched_raw_repair_vs_main_c6": main_failure and (not raw_failure),
                    }
                )
    write_csv(output_dir / "per_sample.csv", all_rows)
    write_csv(output_dir / "input_hashes.csv", input_hash_rows)

    summary_rows: list[dict[str, Any]] = []
    for split_config in split_configs:
        split = str(split_config["name"])
        for snr in snrs:
            selected = [row for row in all_rows if row["split"] == split and float(row["snr_db"]) == snr]
            summary_rows.append(
                {
                    "scope": "split_snr",
                    "split": split,
                    "snr_db": snr,
                    "num_rows": len(selected),
                    "main_c6_minus_reference_psnr_db": float(np.mean([row["main_c6_minus_reference_psnr_db"] for row in selected])),
                    "matched_raw_minus_reference_psnr_db": float(np.mean([row["matched_raw_minus_reference_psnr_db"] for row in selected])),
                    "matched_raw_minus_main_c6_psnr_db": float(np.mean([row["matched_raw_minus_main_c6_psnr_db"] for row in selected])),
                    "matched_top1_minus_reference_psnr_db": float(np.mean([row["matched_top1_minus_reference_psnr_db"] for row in selected])),
                    "matched_confidence_minus_reference_psnr_db": float(np.mean([row["matched_confidence_minus_reference_psnr_db"] for row in selected])),
                    "reference_c8_failure_rate": float(np.mean([row["reference_c8_failure"] for row in selected])),
                    "matched_raw_failure_rate": float(np.mean([row["matched_raw_failure"] for row in selected])),
                    "matched_raw_minus_reference_failure": float(np.mean([row["matched_raw_minus_reference_failure"] for row in selected])),
                    "matched_raw_new_error_vs_main_c6": int(sum(row["matched_raw_new_error_vs_main_c6"] for row in selected)),
                    "matched_raw_repair_vs_main_c6": int(sum(row["matched_raw_repair_vs_main_c6"] for row in selected)),
                }
            )
    write_csv(output_dir / "summary.csv", summary_rows)

    grouped_psnr: dict[str, list[float]] = defaultdict(list)
    grouped_failure: dict[str, list[float]] = defaultdict(list)
    frozen_rows = [row for row in all_rows if row["split"] in primary_splits]
    for row in frozen_rows:
        grouped_psnr[str(row["image_id"])].append(float(row["matched_raw_minus_reference_psnr_db"]))
        grouped_failure[str(row["image_id"])].append(float(row["matched_raw_minus_reference_failure"]))
    psnr_clusters = np.asarray([np.mean(grouped_psnr[key]) for key in sorted(grouped_psnr)], dtype=np.float64)
    failure_clusters = np.asarray([np.mean(grouped_failure[key]) for key in sorted(grouped_failure)], dtype=np.float64)
    replicates = int(config["evaluation"]["bootstrap_replicates"])
    seed = int(config["evaluation"]["bootstrap_seed"])
    bootstrap_payload = {
        "primary_scope": sorted(primary_splits),
        "matched_raw_minus_reference_psnr_db": bootstrap(psnr_clusters, replicates, seed),
        "matched_raw_minus_reference_failure": bootstrap(failure_clusters, replicates, seed + 1),
        "psnr_by_split": {},
    }
    for index, split in enumerate(str(item["name"]) for item in split_configs):
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in all_rows:
            if row["split"] == split:
                grouped[str(row["image_id"])].append(float(row["matched_raw_minus_reference_psnr_db"]))
        values = np.asarray([np.mean(grouped[key]) for key in sorted(grouped)], dtype=np.float64)
        bootstrap_payload["psnr_by_split"][split] = bootstrap(values, replicates, seed + 10 + index)
    visual = make_visual(
        output_dir,
        frozen_rows,
        float(config["evaluation"]["save_visual_snr"]),
        int(config["evaluation"]["save_visual_count"]),
    )
    psnr_ci = bootstrap_payload["matched_raw_minus_reference_psnr_db"]
    failure_ci = bootstrap_payload["matched_raw_minus_reference_failure"]
    all_split_snr_positive = all(float(row["matched_raw_minus_reference_psnr_db"]) > 0.0 for row in summary_rows)
    payload = {
        "analysis_id": config["analysis_id"],
        "analysis_mode": config["analysis_mode"],
        "rate_contract": rate_contract,
        "bootstrap": bootstrap_payload,
        "all_split_snr_raw_psnr_point_estimates_positive": all_split_snr_positive,
        "visual": relative(visual),
        "guardrail": "COCO AlexNet pseudo semantics are auxiliary; supervised Imagenette policy-dev remains required.",
    }
    save_json(output_dir / "summary.json", payload)
    lines = [
        "# Matched-Total-Rate Main + Structure System",
        "",
        "Both systems use total CBR `8/48=1/6`: reference is c=8 RGB, matched system is c=6 RGB + c=2 structure.",
        "This is a post-hoc descriptive analysis of frozen runs, not a preregistered final test.",
        "",
        "| Split | Raw ΔPSNR vs c=8 | 95% CI | c=8 / matched raw pseudo failure |",
        "|---|---:|---|---:|",
    ]
    for split in [str(item["name"]) for item in split_configs]:
        interval = bootstrap_payload["psnr_by_split"][split]
        selected = [row for row in all_rows if row["split"] == split]
        reference_failure = float(np.mean([row["reference_c8_failure"] for row in selected]))
        matched_failure = float(np.mean([row["matched_raw_failure"] for row in selected]))
        lines.append(
            f"| {split} | {interval['estimate']:+.4f} dB | "
            f"[{interval['ci_low']:+.4f}, {interval['ci_high']:+.4f}] | "
            f"{reference_failure:.4f} / {matched_failure:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Frozen heldout+testlike+fresh raw PSNR delta: `{psnr_ci['estimate']:+.4f}` dB, "
            f"95% CI `[{psnr_ci['ci_low']:+.4f}, {psnr_ci['ci_high']:+.4f}]` over "
            f"{psnr_ci['num_image_clusters']} image clusters.",
            f"Pseudo failure delta: `{failure_ci['estimate']:+.4f}`, 95% CI "
            f"`[{failure_ci['ci_low']:+.4f}, {failure_ci['ci_high']:+.4f}]`.",
            f"All 4 splits × 5 SNR raw PSNR point estimates are positive: `{all_split_snr_positive}`.",
            "",
            "## Guardrail",
            "",
            "The quality result is genuinely matched-total-rate. The semantic result still uses ImageNet-pretrained "
            "AlexNet pseudo labels and is not sufficient to call the method supervised-safe. Official Imagenette "
            "validation remains sealed; the next decisive step is policy-dev evaluation with independent scratch T_cls.",
            f"Visual: `{relative(visual)}` (rows: original, c=8 reference, c=6 main, matched raw).",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": relative(config_path),
        "config_sha256": sha256_file(config_path),
        "script": relative(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "refiner_checkpoint": relative(resolve(config["inputs"]["matched_refiner_checkpoint"])),
        "refiner_checkpoint_sha256": sha256_file(resolve(config["inputs"]["matched_refiner_checkpoint"])),
        "input_hashes_sha256": sha256_file(output_dir / "input_hashes.csv"),
        "num_rows": len(all_rows),
        "num_input_hash_rows": len(input_hash_rows),
        "official_val_accessed": False,
    }
    save_json(output_dir / "metadata.json", metadata)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
