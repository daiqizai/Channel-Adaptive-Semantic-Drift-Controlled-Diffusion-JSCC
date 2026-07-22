from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cadsd_jscc.metrics import psnr_per_sample
from s5_residual_refiner_pilot import (  # noqa: E402
    build_model,
    gate_tensor,
    load_rgb_tensor,
    snr_name,
    try_load_lpips,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired old-vs-reservation-aware B1 comparison on the exact same reserved "
            "COCO validation reconstructions."
        )
    )
    parser.add_argument(
        "--old-config", default="configs/s13_scaleup_b1_anchor_train.yaml"
    )
    parser.add_argument(
        "--old-checkpoint", default="outputs/EXP-S13-001/checkpoints/best.pt"
    )
    parser.add_argument(
        "--new-config",
        default="configs/s15_uint2_reservation_aware_b1_finetune_pilot.yaml",
    )
    parser.add_argument(
        "--new-checkpoint", default="outputs/EXP-S15-001/checkpoints/best.pt"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/analysis/s15_reservation_aware_b1_paired_comparison",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260728)
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {path}")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def quantize_png(tensor: torch.Tensor) -> torch.Tensor:
    """Match torchvision.save_image's 8-bit PNG conversion before metric evaluation."""
    return torch.floor(tensor.clamp(0.0, 1.0) * 255.0 + 0.5) / 255.0


def load_checkpoint_model(
    config: dict[str, Any], checkpoint_path: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise RuntimeError(f"Invalid refiner checkpoint: {checkpoint_path}")
    model = build_model(config).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


def paired_bootstrap_ci(
    values_by_name: dict[str, list[float]], resamples: int, seed: int
) -> dict[str, float]:
    names = sorted(values_by_name)
    matrix = np.asarray([values_by_name[name] for name in names], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("Paired cluster bootstrap requires at least two image clusters")
    cluster_means = matrix.mean(axis=1)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(names), size=(resamples, len(names)))
    estimates = cluster_means[sampled].mean(axis=1)
    return {
        "estimate": float(cluster_means.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "num_image_clusters": len(names),
        "bootstrap_resamples": int(resamples),
        "bootstrap_seed": int(seed),
    }


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.bootstrap_resamples <= 0:
        raise ValueError("batch-size and bootstrap-resamples must be positive")

    old_config_path = resolve(args.old_config)
    new_config_path = resolve(args.new_config)
    old_checkpoint_path = resolve(args.old_checkpoint)
    new_checkpoint_path = resolve(args.new_checkpoint)
    output_dir = resolve(args.output_dir)
    old_config = load_yaml(old_config_path)
    new_config = load_yaml(new_config_path)

    snrs = [float(value) for value in new_config["snrs"]]
    if snrs != [float(value) for value in old_config["snrs"]]:
        raise RuntimeError("Old and new refiner configs use different SNR grids")
    for key in ["base_channels", "num_blocks", "snr_norm_max", "residual_gates"]:
        if old_config["model"][key] != new_config["model"][key]:
            raise RuntimeError(f"Old and new refiner model.{key} differ")
    if old_config["model"].get("condition_features", []) != new_config["model"].get(
        "condition_features", []
    ):
        raise RuntimeError("Old and new condition features differ")

    start = int(new_config["split"]["eval_sample_start"])
    count = int(new_config["split"]["eval_sample_count"])
    names = [f"sample_{index:06d}.png" for index in range(start, start + count)]
    original_dir = resolve(new_config["inputs"]["original_dir"])
    reserved_root = resolve(new_config["inputs"]["m0_export_dir"])
    reserved_subdir = str(new_config["inputs"].get("m0_reconstruction_subdir", "reconstruction"))

    required = [old_config_path, new_config_path, old_checkpoint_path, new_checkpoint_path]
    required.extend(original_dir / name for name in names)
    for snr in snrs:
        required.extend(
            reserved_root / "exports" / snr_name(snr) / reserved_subdir / name
            for name in names
        )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} required files; first={missing[0]}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")

    manifest = {
        "protocol": "S15 paired old-vs-reservation-aware B1 on identical reserved validation inputs",
        "official_imagenette_accessed": False,
        "old_config": str(old_config_path.relative_to(PROJECT_ROOT)),
        "old_config_sha256": sha256(old_config_path),
        "old_checkpoint": str(old_checkpoint_path.relative_to(PROJECT_ROOT)),
        "old_checkpoint_sha256": sha256(old_checkpoint_path),
        "new_config": str(new_config_path.relative_to(PROJECT_ROOT)),
        "new_config_sha256": sha256(new_config_path),
        "new_checkpoint": str(new_checkpoint_path.relative_to(PROJECT_ROOT)),
        "new_checkpoint_sha256": sha256(new_checkpoint_path),
        "reserved_input_root": str(reserved_root.relative_to(PROJECT_ROOT)),
        "sample_start": start,
        "sample_count": count,
        "snrs_db": snrs,
        "png_quantized_before_metrics": True,
        "batch_size": args.batch_size,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    device = choose_device(args.device)
    old_model, old_checkpoint = load_checkpoint_model(old_config, old_checkpoint_path, device)
    new_model, new_checkpoint = load_checkpoint_model(new_config, new_checkpoint_path, device)
    manifest.update(
        {
            "device": str(device),
            "old_checkpoint_epoch": int(old_checkpoint.get("epoch", -1)),
            "new_checkpoint_epoch": int(new_checkpoint.get("epoch", -1)),
        }
    )
    lpips_model = None
    lpips_error = None
    if not args.skip_lpips:
        lpips_model, lpips_error = try_load_lpips(device, resolve("outputs/cache"))
    manifest["lpips_available"] = lpips_model is not None
    manifest["lpips_error"] = lpips_error

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for snr in snrs:
            reconstruction_dir = reserved_root / "exports" / snr_name(snr) / reserved_subdir
            for offset in range(0, len(names), args.batch_size):
                batch_names = names[offset : offset + args.batch_size]
                reserved = torch.stack(
                    [load_rgb_tensor(reconstruction_dir / name) for name in batch_names]
                ).to(device)
                target = torch.stack(
                    [load_rgb_tensor(original_dir / name) for name in batch_names]
                ).to(device)
                snr_db = torch.full(
                    (len(batch_names),), snr, dtype=torch.float32, device=device
                )
                snr_norm = snr_db / float(new_config["model"]["snr_norm_max"])
                old_prediction = quantize_png(
                    old_model(reserved, snr_norm, gate_tensor(old_config, snr_db, device))
                )
                new_prediction = quantize_png(
                    new_model(reserved, snr_norm, gate_tensor(new_config, snr_db, device))
                )
                old_psnr = psnr_per_sample(old_prediction, target).cpu().tolist()
                new_psnr = psnr_per_sample(new_prediction, target).cpu().tolist()
                old_lpips: list[float | None] = [None] * len(batch_names)
                new_lpips: list[float | None] = [None] * len(batch_names)
                if lpips_model is not None:
                    old_lpips = (
                        lpips_model(old_prediction * 2.0 - 1.0, target * 2.0 - 1.0)
                        .flatten()
                        .cpu()
                        .tolist()
                    )
                    new_lpips = (
                        lpips_model(new_prediction * 2.0 - 1.0, target * 2.0 - 1.0)
                        .flatten()
                        .cpu()
                        .tolist()
                    )
                for name, old_p, new_p, old_l, new_l in zip(
                    batch_names, old_psnr, new_psnr, old_lpips, new_lpips
                ):
                    rows.append(
                        {
                            "sample_name": name,
                            "snr_db": snr,
                            "old_b1_psnr_db": float(old_p),
                            "new_b1_psnr_db": float(new_p),
                            "new_minus_old_psnr_db": float(new_p - old_p),
                            "old_b1_lpips": old_l,
                            "new_b1_lpips": new_l,
                            "new_minus_old_lpips": (
                                float(new_l - old_l)
                                if old_l is not None and new_l is not None
                                else None
                            ),
                        }
                    )

    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["snr_db"])].append(row)
    summary: list[dict[str, Any]] = []
    psnr_by_name: dict[str, list[float]] = defaultdict(list)
    lpips_by_name: dict[str, list[float]] = defaultdict(list)
    for snr_index, snr in enumerate(snrs):
        group = grouped[snr]
        psnr_clusters = {
            str(row["sample_name"]): [float(row["new_minus_old_psnr_db"])]
            for row in group
        }
        psnr_ci = paired_bootstrap_ci(
            psnr_clusters, args.bootstrap_resamples, args.bootstrap_seed + snr_index
        )
        for row in group:
            psnr_by_name[str(row["sample_name"])].append(
                float(row["new_minus_old_psnr_db"])
            )
            if row["new_minus_old_lpips"] is not None:
                lpips_by_name[str(row["sample_name"])].append(
                    float(row["new_minus_old_lpips"])
                )
        summary.append(
            {
                "scope": snr_name(snr),
                "snr_db": snr,
                "num_images": len(group),
                "old_b1_psnr_db": mean([float(row["old_b1_psnr_db"]) for row in group]),
                "new_b1_psnr_db": mean([float(row["new_b1_psnr_db"]) for row in group]),
                "new_minus_old_psnr_db": psnr_ci["estimate"],
                "new_minus_old_psnr_ci95_low": psnr_ci["ci95_low"],
                "new_minus_old_psnr_ci95_high": psnr_ci["ci95_high"],
                "old_b1_lpips": mean(
                    [float(row["old_b1_lpips"]) for row in group if row["old_b1_lpips"] is not None]
                ),
                "new_b1_lpips": mean(
                    [float(row["new_b1_lpips"]) for row in group if row["new_b1_lpips"] is not None]
                ),
                "new_minus_old_lpips": mean(
                    [
                        float(row["new_minus_old_lpips"])
                        for row in group
                        if row["new_minus_old_lpips"] is not None
                    ]
                ),
            }
        )

    aggregate_psnr_ci = paired_bootstrap_ci(
        psnr_by_name, args.bootstrap_resamples, args.bootstrap_seed + len(snrs)
    )
    aggregate_lpips_ci = (
        paired_bootstrap_ci(
            lpips_by_name, args.bootstrap_resamples, args.bootstrap_seed + len(snrs) + 1
        )
        if lpips_by_name
        else None
    )
    all_snr_point_positive = all(
        float(row["new_minus_old_psnr_db"]) > 0.0 for row in summary
    )
    aggregate_ci_positive = aggregate_psnr_ci["ci95_low"] > 0.0
    verdict = "POSITIVE" if all_snr_point_positive and aggregate_ci_positive else "NEGATIVE"
    aggregate_summary = {
        "scope": "all_snrs_image_clustered",
        "snr_db": None,
        "num_images": len(names),
        "old_b1_psnr_db": mean([float(row["old_b1_psnr_db"]) for row in rows]),
        "new_b1_psnr_db": mean([float(row["new_b1_psnr_db"]) for row in rows]),
        "new_minus_old_psnr_db": aggregate_psnr_ci["estimate"],
        "new_minus_old_psnr_ci95_low": aggregate_psnr_ci["ci95_low"],
        "new_minus_old_psnr_ci95_high": aggregate_psnr_ci["ci95_high"],
        "old_b1_lpips": mean(
            [float(row["old_b1_lpips"]) for row in rows if row["old_b1_lpips"] is not None]
        ),
        "new_b1_lpips": mean(
            [float(row["new_b1_lpips"]) for row in rows if row["new_b1_lpips"] is not None]
        ),
        "new_minus_old_lpips": (
            aggregate_lpips_ci["estimate"] if aggregate_lpips_ci is not None else None
        ),
    }
    summary.append(aggregate_summary)

    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "per_sample.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    metrics = {
        "manifest": manifest,
        "decision_rule": {
            "all_snr_mean_psnr_positive": True,
            "aggregate_image_cluster_bootstrap_ci95_low_above_zero": True,
        },
        "decision": {
            "all_snr_mean_psnr_positive": all_snr_point_positive,
            "aggregate_image_cluster_bootstrap_ci95_low_above_zero": aggregate_ci_positive,
            "verdict": verdict,
        },
        "aggregate_psnr_bootstrap": aggregate_psnr_ci,
        "aggregate_lpips_bootstrap": aggregate_lpips_ci,
        "summary": summary,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics["decision"], indent=2))
    print(json.dumps(aggregate_summary, indent=2))


if __name__ == "__main__":
    main()
