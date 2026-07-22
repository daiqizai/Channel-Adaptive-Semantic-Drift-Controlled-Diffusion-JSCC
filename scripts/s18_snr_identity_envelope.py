#!/usr/bin/env python3
"""Select and audit the preregistered S18 SNR-conditioned identity envelope."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import save_image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from cadsd_jscc.channel_matched_latent_diffusion import (  # noqa: E402
    channel_alpha,
    deterministic_ddim,
    normalize_channel_observation,
)
from cadsd_jscc.metrics import ms_ssim_per_sample, psnr_per_sample  # noqa: E402
from cadsd_jscc.snr_identity_envelope import (  # noqa: E402
    apply_correction_envelope,
    envelope_strength,
    select_envelope_policy,
)
from s17_channel_matched_latent_diffusion import (  # noqa: E402
    CachedOriginalDataset,
    active_to_dense,
    build_b1,
    build_denoiser,
    build_jscc,
    canonical_batch_noise,
    classifier_model,
    classify,
    clean_transmitted_active,
    coordinate_contract,
    dense_to_active,
    load_denoiser_checkpoint,
    seed_everything,
)
from s5_residual_refiner_pilot import gate_tensor, try_load_lpips  # noqa: E402


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("config root must be a mapping")
    return config


def validate_contract(config: dict[str, Any], mode: str) -> None:
    allowed = {
        "population_frozen_before_selection_output",
        "policy_frozen_before_holdout_output",
    }
    if config["protocol"]["status"] not in allowed:
        raise RuntimeError("S18 protocol status is not executable")
    if config["protocol"].get("official_imagenette_accessed") is not False:
        raise RuntimeError("official Imagenette validation must remain sealed")
    for key, hash_key in (
        ("prior_source_manifest", "prior_source_manifest_sha256"),
        ("source_manifest", "source_manifest_sha256"),
        ("deepjscc_checkpoint", "deepjscc_checkpoint_sha256"),
        ("latent_diffusion_checkpoint", "latent_diffusion_checkpoint_sha256"),
        ("b1_checkpoint", "b1_checkpoint_sha256"),
    ):
        path = resolve(config["inputs"][key])
        if not path.exists() or sha256_file(path) != str(config["inputs"][hash_key]):
            raise RuntimeError(f"input hash mismatch: {key}")
    if resolve(config["inputs"]["deepjscc_checkpoint"]).resolve() == resolve(
        config["inputs"]["forbidden_deepjscc_checkpoint"]
    ).resolve():
        raise RuntimeError("forbidden DeepJSCC latest checkpoint selected")
    if int(config["rate"]["active_real_symbols"]) != 19712:
        raise RuntimeError("active-symbol contract changed")
    if int(config["rate"]["image_active_real_symbols"]) != 19632:
        raise RuntimeError("image-coordinate contract changed")
    if int(config["rate"]["payload_real_symbols"]) != 80:
        raise RuntimeError("payload reservation changed")
    if mode == "holdout":
        policy = resolve(config["inputs"]["selected_policy"])
        expected = str(config["inputs"]["selected_policy_sha256"])
        if expected.startswith("PENDING_") or not policy.exists() or sha256_file(policy) != expected:
            raise RuntimeError("selected policy is not hash-frozen for holdout")


def stage_names(config: dict[str, Any]) -> tuple[str, ...]:
    return (
        "b0",
        "full",
        *[str(item["name"]) for item in config["envelope"]["candidates"]],
        "b1",
    )


def make_loader(config: dict[str, Any], role: str, device: torch.device) -> DataLoader:
    count = int(
        config["population"][
            "selection_count" if role == config["population"]["selection_role"] else "holdout_count"
        ]
    )
    dataset = CachedOriginalDataset(
        resolve(config["inputs"]["cache_root"]),
        resolve(config["inputs"]["source_manifest"]),
        role,
        start=0,
        count=count,
    )
    return DataLoader(
        dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["evaluation"]["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(config["evaluation"]["num_workers"]) > 0,
    )


@torch.no_grad()
def evaluate(
    config: dict[str, Any],
    mode: str,
    device: torch.device,
    output: Path,
) -> list[dict[str, Any]]:
    role = str(
        config["population"][
            "selection_role" if mode == "selection" else "holdout_role"
        ]
    )
    loader = make_loader(config, role, device)
    jscc = build_jscc(config, device)
    denoiser = build_denoiser(config, device)
    load_denoiser_checkpoint(
        denoiser, resolve(config["inputs"]["latent_diffusion_checkpoint"]), device
    )
    denoiser.eval().requires_grad_(False)
    b1, b1_config = build_b1(config, device)
    reserved, _valid_active, valid_dense = coordinate_contract(jscc, config, device)
    lpips_model, lpips_error = try_load_lpips(device, resolve("outputs/cache"))
    if lpips_model is None:
        raise RuntimeError(lpips_error)
    classifiers: dict[str, torch.nn.Module] = {}
    mean_tensor: torch.Tensor | None = None
    std_tensor: torch.Tensor | None = None
    if mode == "holdout":
        classifiers = {
            name: classifier_model(name, resolve(path), device)
            for name, path in config["classifiers"].items()
            if name in {"alexnet", "resnet18", "mobilenet_v3_small"}
        }
        mean_tensor = torch.tensor(
            config["classifiers"]["imagenet_mean"], device=device
        ).reshape(1, 3, 1, 1)
        std_tensor = torch.tensor(
            config["classifiers"]["imagenet_std"], device=device
        ).reshape(1, 3, 1, 1)
    factor = float(config["channel"]["noise_variance_factor_per_real"])
    reference_snr = float(config["envelope"]["alpha_reference_snr_db"])
    base_seed = int(
        config["channel"][
            "selection_base_seed" if mode == "selection" else "holdout_base_seed"
        ]
    )
    stages = stage_names(config)
    rows: list[dict[str, Any]] = []
    saved_snrs: set[float] = set()
    for snr in [float(value) for value in config["channel"]["snrs_db"]]:
        alpha = float(channel_alpha(snr, factor))
        strengths = {
            str(item["name"]): envelope_strength(
                snr,
                item,
                noise_variance_factor_per_real=factor,
                reference_snr_db=reference_snr,
            )
            for item in config["envelope"]["candidates"]
        }
        for images_cpu, sample_ids in loader:
            images = images_cpu.to(device, non_blocking=True)
            transmitted, clean_active, dense_shape = clean_transmitted_active(
                jscc, images, reserved
            )
            noise_cpu, noise_hashes = canonical_batch_noise(
                list(sample_ids), snr, base_seed, jscc.active_symbols
            )
            jscc.snr_db = snr
            received = jscc.transmit_active(transmitted, noise_cpu.to(device))
            received[:, reserved] = 0.0
            b0 = jscc.decode_active(received, dense_shape).clamp(0.0, 1.0)
            matched_state = active_to_dense(
                jscc, normalize_channel_observation(received, alpha), dense_shape
            )
            full_dense = deterministic_ddim(
                denoiser,
                matched_state,
                valid_dense,
                alpha_start=alpha,
                sampling_steps=int(config["diffusion"]["sampling_steps"]),
                alpha_max=float(config["diffusion"]["train_alpha_max"]),
            )
            full_active = dense_to_active(jscc, full_dense)
            full_active[:, reserved] = 0.0
            candidates: dict[str, torch.Tensor] = {
                "b0": b0,
                "full": jscc.decode_active(full_active, dense_shape).clamp(0.0, 1.0),
            }
            latent_candidates: dict[str, torch.Tensor] = {
                "b0": received,
                "full": full_active,
            }
            for name, strength in strengths.items():
                active = apply_correction_envelope(received, full_active, strength)
                active[:, reserved] = 0.0
                latent_candidates[name] = active
                candidates[name] = jscc.decode_active(active, dense_shape).clamp(0.0, 1.0)
            snr_tensor = torch.full((images.shape[0],), snr, device=device)
            snr_norm = snr_tensor / float(b1_config["model"]["snr_norm_max"])
            candidates["b1"] = b1(
                b0, snr_norm, gate_tensor(b1_config, snr_tensor, device)
            ).clamp(0.0, 1.0)
            quality: dict[str, dict[str, torch.Tensor]] = {}
            for stage, candidate in candidates.items():
                quality[stage] = {
                    "psnr": psnr_per_sample(candidate, images),
                    "ms_ssim": ms_ssim_per_sample(candidate, images),
                    "lpips": lpips_model(
                        candidate * 2.0 - 1.0, images * 2.0 - 1.0
                    ).flatten(),
                }
            latent_mse = {
                stage: (candidate - clean_active).square().sum(dim=1)
                / int(config["rate"]["image_active_real_symbols"])
                for stage, candidate in latent_candidates.items()
            }
            original_classification: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            predictions: dict[str, dict[str, torch.Tensor]] = {}
            if classifiers:
                assert mean_tensor is not None and std_tensor is not None
                for name, classifier in classifiers.items():
                    original_classification[name] = classify(
                        classifier, images, mean_tensor, std_tensor
                    )
                    predictions[name] = {
                        stage: classify(classifier, candidates[stage], mean_tensor, std_tensor)[0]
                        for stage in stages
                    }
            if mode == "holdout" and snr not in saved_snrs:
                count = min(int(config["evaluation"]["sample_grid_count"]), images.shape[0])
                save_image(
                    torch.cat([images[:count], *[candidates[stage][:count] for stage in stages]]).cpu(),
                    output / f"snr_{int(snr):02d}_identity_envelope_grid.png",
                    nrow=count,
                )
                saved_snrs.add(snr)
            for index, sample_id in enumerate(sample_ids):
                row: dict[str, Any] = {
                    "analysis_id": config[
                        "selection_analysis_id" if mode == "selection" else "holdout_analysis_id"
                    ],
                    "sample_id": sample_id,
                    "role": role,
                    "snr_db": snr,
                    "alpha_channel": alpha,
                    "canonical_noise_sha256": noise_hashes[index],
                    "total_real_symbols": int(config["rate"]["active_real_symbols"]),
                    "image_active_real_symbols": int(config["rate"]["image_active_real_symbols"]),
                    "full_strength": 1.0,
                    **{f"{name}_strength": value for name, value in strengths.items()},
                }
                for stage in stages:
                    for metric in ("psnr", "ms_ssim", "lpips"):
                        row[f"{stage}_{metric}"] = float(quality[stage][metric][index])
                for stage, values in latent_mse.items():
                    row[f"{stage}_latent_mse"] = float(values[index])
                for name in classifiers:
                    original_prediction, original_confidence = original_classification[name]
                    row[f"{name}_original_prediction"] = int(original_prediction[index])
                    row[f"{name}_original_confidence"] = float(original_confidence[index])
                    for stage in stages:
                        row[f"{name}_{stage}_prediction"] = int(predictions[name][stage][index])
                if classifiers:
                    for stage in stages:
                        new_votes = 0
                        repair_votes = 0
                        for name in classifiers:
                            original = int(row[f"{name}_original_prediction"])
                            baseline = int(row[f"{name}_b0_prediction"])
                            candidate = int(row[f"{name}_{stage}_prediction"])
                            new_votes += int(baseline == original and candidate != original)
                            repair_votes += int(baseline != original and candidate == original)
                        row[f"majority_{stage}_new"] = new_votes >= 2
                        row[f"majority_{stage}_repair"] = repair_votes >= 2
                rows.append(row)
        print(json.dumps({"mode": mode, "snr_complete": snr, "rows": len(rows)}))
    return rows


def quality_summary(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    stages = stage_names(config)
    summary: dict[str, Any] = {
        "rows": len(rows),
        "images": len({row["sample_id"] for row in rows}),
        "snrs_db": sorted({float(row["snr_db"]) for row in rows}),
        "stages": {},
        "per_snr": [],
    }
    for stage in stages:
        summary["stages"][stage] = {
            metric: mean(rows, f"{stage}_{metric}")
            for metric in ("psnr", "ms_ssim", "lpips")
        }
        if stage != "b0":
            summary["stages"][stage]["psnr_delta_vs_b0"] = (
                summary["stages"][stage]["psnr"] - summary["stages"]["b0"]["psnr"]
            )
            summary["stages"][stage]["lpips_delta_vs_b0"] = (
                summary["stages"][stage]["lpips"] - summary["stages"]["b0"]["lpips"]
            )
    for snr in summary["snrs_db"]:
        subset = [row for row in rows if float(row["snr_db"]) == snr]
        item: dict[str, Any] = {"snr_db": snr, "rows": len(subset), "stages": {}}
        for stage in stages:
            item["stages"][stage] = {
                metric: mean(subset, f"{stage}_{metric}")
                for metric in ("psnr", "ms_ssim", "lpips")
            }
            if stage != "b0":
                item["stages"][stage]["psnr_delta_vs_b0"] = (
                    item["stages"][stage]["psnr"] - item["stages"]["b0"]["psnr"]
                )
                item["stages"][stage]["lpips_delta_vs_b0"] = (
                    item["stages"][stage]["lpips"] - item["stages"]["b0"]["lpips"]
                )
        summary["per_snr"].append(item)
    return summary


def selection_summary(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    summary = quality_summary(rows, config)
    adaptive: list[dict[str, Any]] = []
    for specification in config["envelope"]["candidates"]:
        name = str(specification["name"])
        adaptive.append(
            {
                "name": name,
                "specification": specification,
                "mean_psnr_delta_vs_b0": summary["stages"][name]["psnr_delta_vs_b0"],
                "mean_lpips_delta_vs_b0": summary["stages"][name]["lpips_delta_vs_b0"],
                "per_snr": [
                    {
                        "snr_db": item["snr_db"],
                        "strength": mean(
                            [row for row in rows if float(row["snr_db"]) == item["snr_db"]],
                            f"{name}_strength",
                        ),
                        "psnr_delta_vs_b0": item["stages"][name]["psnr_delta_vs_b0"],
                        "lpips_delta_vs_b0": item["stages"][name]["lpips_delta_vs_b0"],
                    }
                    for item in summary["per_snr"]
                ],
            }
        )
    selected = select_envelope_policy(
        adaptive,
        nonnegative_tolerance_db=float(config["envelope"]["nonnegative_tolerance_db"]),
    )
    return {**summary, "adaptive_candidates": adaptive, "selected": selected}


def holdout_summary(
    rows: list[dict[str, Any]], config: dict[str, Any], selected_name: str
) -> dict[str, Any]:
    summary = quality_summary(rows, config)
    stages = stage_names(config)
    threshold = float(config["evaluation"]["pseudo_original_confidence_min"])
    eligible = [row for row in rows if float(row["alexnet_original_confidence"]) >= threshold]
    summary["alexnet_pseudo_eligible_rows"] = len(eligible)
    summary["pseudo_semantic"] = {}
    for stage in stages:
        summary["pseudo_semantic"][stage] = {
            "alexnet_failure": sum(
                int(row[f"alexnet_{stage}_prediction"])
                != int(row["alexnet_original_prediction"])
                for row in eligible
            ),
            "alexnet_new": sum(
                int(row["alexnet_b0_prediction"]) == int(row["alexnet_original_prediction"])
                and int(row[f"alexnet_{stage}_prediction"])
                != int(row["alexnet_original_prediction"])
                for row in eligible
            ),
            "alexnet_repair": sum(
                int(row["alexnet_b0_prediction"]) != int(row["alexnet_original_prediction"])
                and int(row[f"alexnet_{stage}_prediction"])
                == int(row["alexnet_original_prediction"])
                for row in eligible
            ),
            "majority_new": sum(bool(row[f"majority_{stage}_new"]) for row in rows),
            "majority_repair": sum(bool(row[f"majority_{stage}_repair"]) for row in rows),
        }
    per_snr_selected = [item["stages"][selected_name] for item in summary["per_snr"]]
    low_mid = [row for row in rows if float(row["snr_db"]) <= 7.0]
    selected_low_mid_gain = mean(low_mid, f"{selected_name}_psnr") - mean(low_mid, "b0_psnr")
    full_low_mid_gain = mean(low_mid, "full_psnr") - mean(low_mid, "b0_psnr")
    criteria = config["success_criteria"]
    checks = {
        "selected_mean_psnr_positive_vs_b0": summary["stages"][selected_name][
            "psnr_delta_vs_b0"
        ]
        > float(criteria["selected_minus_b0_mean_psnr_min_db"]),
        "selected_positive_at_least_three_snrs": sum(
            item["psnr_delta_vs_b0"] > 1e-9 for item in per_snr_selected
        )
        >= int(criteria["selected_minus_b0_positive_snr_count_min"]),
        "selected_nonnegative_all_five_snrs": sum(
            item["psnr_delta_vs_b0"] >= -1e-9 for item in per_snr_selected
        )
        >= int(criteria["selected_minus_b0_nonnegative_snr_count_min"]),
        "selected_lpips_nonworse_than_b0": summary["stages"][selected_name][
            "lpips_delta_vs_b0"
        ]
        <= float(criteria["selected_minus_b0_mean_lpips_max"]),
        "selected_retains_low_mid_full_gain": selected_low_mid_gain / full_low_mid_gain
        >= float(criteria["low_mid_snr_full_psnr_gain_retention_min"]),
        "selected_alexnet_new_not_greater_than_repair": summary["pseudo_semantic"][
            selected_name
        ]["alexnet_new"]
        <= summary["pseudo_semantic"][selected_name]["alexnet_repair"],
        "selected_alexnet_new_not_greater_than_full": summary["pseudo_semantic"][
            selected_name
        ]["alexnet_new"]
        <= summary["pseudo_semantic"]["full"]["alexnet_new"],
        "selected_majority_new_not_greater_than_repair": summary["pseudo_semantic"][
            selected_name
        ]["majority_new"]
        <= summary["pseudo_semantic"][selected_name]["majority_repair"],
        "selected_majority_new_not_greater_than_full": summary["pseudo_semantic"][
            selected_name
        ]["majority_new"]
        <= summary["pseudo_semantic"]["full"]["majority_new"],
    }
    summary.update(
        {
            "selected_policy": selected_name,
            "selected_low_mid_psnr_gain": selected_low_mid_gain,
            "full_low_mid_psnr_gain": full_low_mid_gain,
            "low_mid_gain_retention": selected_low_mid_gain / full_low_mid_gain,
            "checks_before_bootstrap": checks,
            "bootstrap_check_pending": "selected_minus_full_psnr_ci_low_gt_0",
            "verdict": "PENDING_BOOTSTRAP" if all(checks.values()) else "NEGATIVE_OR_PARTIAL",
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/s18_snr_identity_envelope.yaml")
    parser.add_argument("--mode", choices=("selection", "holdout"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = load_config(config_path)
    validate_contract(config, args.mode)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed_everything(int(config["seed"]))
    output = resolve(
        args.output_dir
        or config["outputs"]["selection_dir" if args.mode == "selection" else "holdout_dir"]
    )
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    shutil.copy2(config_path, output / "config.yaml")
    shutil.copy2(SCRIPT, output / SCRIPT.name)
    rows = evaluate(config, args.mode, device, output)
    write_csv(output / "per_sample.csv", rows)
    run = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "config": relative(config_path),
        "config_sha256": sha256_file(config_path),
        "source_manifest_sha256": sha256_file(resolve(config["inputs"]["source_manifest"])),
        "deepjscc_checkpoint_sha256": sha256_file(resolve(config["inputs"]["deepjscc_checkpoint"])),
        "latent_diffusion_checkpoint_sha256": sha256_file(
            resolve(config["inputs"]["latent_diffusion_checkpoint"])
        ),
        "rows": len(rows),
        "images": len({row["sample_id"] for row in rows}),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "official_imagenette_accessed": False,
        "download_note": "No download; local COCO/checkpoints only.",
    }
    save_json(output / "run_plan.json", run)
    if args.mode == "selection":
        summary = selection_summary(rows, config)
        selected = summary["selected"]
        policy = {
            "selection_analysis_id": config["selection_analysis_id"],
            "selected_name": selected["name"],
            "selected_specification": selected["specification"],
            "nonnegative_snr_count": selected["nonnegative_snr_count"],
            "mean_psnr_delta_vs_b0": selected["mean_psnr_delta_vs_b0"],
            "mean_lpips_delta_vs_b0": selected["mean_lpips_delta_vs_b0"],
            "per_snr": selected["per_snr"],
            "selection_order": config["envelope"]["selection_order"],
            "source_manifest_sha256": run["source_manifest_sha256"],
            "latent_diffusion_checkpoint_sha256": run[
                "latent_diffusion_checkpoint_sha256"
            ],
            "selection_per_sample_sha256": sha256_file(output / "per_sample.csv"),
            "holdout_accessed": False,
        }
        save_json(output / "summary.json", summary)
        save_json(output / "selected_policy.json", policy)
        run["selected_policy"] = policy
    else:
        policy = json.loads(resolve(config["inputs"]["selected_policy"]).read_text(encoding="utf-8"))
        summary = holdout_summary(rows, config, str(policy["selected_name"]))
        save_json(output / "summary.json", summary)
        run["selected_policy"] = str(policy["selected_name"])
    save_json(output / "STATE.json", {"state": "COMPLETE", **run})
    print(json.dumps({**run, "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
