#!/usr/bin/env python3
"""Evaluate S19 exact fallback once on the pristine S27 population."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from cadsd_jscc.diffusion_fusion import parameter_count  # noqa: E402
from cadsd_jscc.metrics import ms_ssim_per_sample, psnr_per_sample  # noqa: E402
from s17_channel_matched_latent_diffusion import classifier_model, classify  # noqa: E402
from s19_train_and_evaluate_fusion import (  # noqa: E402
    FusionPairDataset,
    build_initial_models,
    gates,
    load_frozen_model,
)
from s21_b1_anchored_gated_fusion import (  # noqa: E402
    load_config,
    resolve,
    save_json,
    seed_everything,
    sha256_file,
    write_csv,
)
from s26_s19_exact_fallback_replication import STAGES, bootstrap, summarize  # noqa: E402
from s5_residual_refiner_pilot import try_load_lpips  # noqa: E402


def validate(config: dict[str, Any]) -> None:
    protocol = config["protocol"]
    if protocol["status"] != "cache_frozen_before_pristine_evaluation_output":
        raise RuntimeError("S27 config is not executable for pristine evaluation")
    for key in (
        "s19_s23_s25_s26_outcomes_known",
        "all_s16_s18_s19_s21_sources_excluded_by_path_and_sha256",
        "frozen_s19_checkpoints_no_retraining",
        "no_selection_or_tuning",
        "one_shot_pristine_holdout_evaluation",
    ):
        if protocol.get(key) is not True:
            raise RuntimeError(f"S27 protocol boundary changed: {key}")
    if protocol.get("official_imagenette_accessed") is not False:
        raise RuntimeError("official Imagenette validation must remain sealed")
    for key, hash_key in (
        ("source_manifest", "source_manifest_sha256"),
        ("cache_manifest", "cache_manifest_sha256"),
        ("b1_config", "b1_config_sha256"),
        ("b1_checkpoint", "b1_checkpoint_sha256"),
        ("control_checkpoint", "control_checkpoint_sha256"),
        ("fusion_checkpoint", "fusion_checkpoint_sha256"),
    ):
        path = resolve(config["inputs"][key])
        expected = str(config["inputs"][hash_key])
        if expected.startswith("PENDING_") or not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"S27 frozen input hash mismatch: {key}")
    metadata = json.loads((resolve(config["outputs"]["population_dir"]) / "population_metadata.json").read_text())
    if int(metadata["excluded_path_overlap"]) != 0 or int(metadata["excluded_sha_overlap"]) != 0:
        raise RuntimeError("S27 population overlaps a prior population")
    if int(config["population"]["roles"]["holdout"]) != 512:
        raise RuntimeError("S27 pristine holdout count changed")
    if [float(value) for value in config["exact_fallback"]["use_frozen_branch_snrs_db"]] != [1.0, 4.0, 7.0]:
        raise RuntimeError("S27 low-SNR route changed")
    if [float(value) for value in config["exact_fallback"]["use_exact_b1_snrs_db"]] != [13.0, 19.0]:
        raise RuntimeError("S27 high-SNR route changed")
    if int(config["rate"]["active_real_symbols"]) != 19712:
        raise RuntimeError("strict rate contract changed")


@torch.no_grad()
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/s27_s19_exact_fallback_fresh_replication.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    os.environ.setdefault("TORCH_HOME", str(resolve("outputs/cache/torch")))
    config_path = resolve(args.config)
    config = load_config(config_path)
    validate(config)
    output = resolve(config["outputs"]["directory"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    shutil.copy2(config_path, output / "config_before_output.yaml")
    shutil.copy2(SCRIPT, output / SCRIPT.name)
    seed_everything(int(config["seed"]))
    device = torch.device(args.device)
    dataset = FusionPairDataset(config, "holdout", train=False)
    loader = DataLoader(
        dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["evaluation"]["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(config["evaluation"]["num_workers"]) > 0,
    )
    control, control_checkpoint = load_frozen_model(config, "control", device)
    fusion, fusion_checkpoint = load_frozen_model(config, "fusion", device)
    _unused_control, _unused_fusion, b1, _b1_config = build_initial_models(config, device)
    b1 = b1.eval().requires_grad_(False)
    lpips_model, lpips_error = try_load_lpips(device, resolve("outputs/cache"))
    if lpips_model is None:
        raise RuntimeError(lpips_error)
    lpips_model.eval().requires_grad_(False)
    classifiers = {
        name: classifier_model(name, resolve(config["classifiers"][name]), device)
        for name in ("alexnet", "resnet18", "mobilenet_v3_small")
    }
    mean = torch.tensor(config["classifiers"]["imagenet_mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(config["classifiers"]["imagenet_std"], device=device).view(1, 3, 1, 1)
    low_snrs = {float(value) for value in config["exact_fallback"]["use_frozen_branch_snrs_db"]}
    rows: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(loader):
        b0 = batch["b0"].to(device, non_blocking=True)
        diffusion = batch["auxiliary"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        snr = batch["snr_db"].to(device, non_blocking=True)
        snr_norm = batch["snr_norm"].to(device, non_blocking=True)
        gate = gates(config, snr, device)
        b1_output = b1(b0, snr_norm, gate)
        control_output = control(b0, b0, snr_norm, gate)
        fusion_output = fusion(b0, diffusion, snr_norm, gate)
        use_branch = torch.tensor(
            [float(value) in low_snrs for value in snr.detach().cpu().tolist()],
            dtype=torch.bool,
            device=device,
        ).view(-1, 1, 1, 1)
        routed_control = torch.where(use_branch, control_output, b1_output)
        routed_fusion = torch.where(use_branch, fusion_output, b1_output)
        candidates = {
            "b0": b0,
            "diffusion": diffusion,
            "b1": b1_output,
            "control": control_output,
            "fusion": fusion_output,
            "routed_control": routed_control,
            "routed_fusion": routed_fusion,
        }
        quality = {
            stage: {
                "psnr": psnr_per_sample(image, target),
                "ms_ssim": ms_ssim_per_sample(image, target),
                "lpips": lpips_model(image * 2.0 - 1.0, target * 2.0 - 1.0).flatten(),
            }
            for stage, image in candidates.items()
        }
        predictions: dict[str, dict[str, torch.Tensor]] = {}
        confidences: dict[str, dict[str, torch.Tensor]] = {}
        for name, classifier in classifiers.items():
            predictions[name] = {}
            confidences[name] = {}
            predictions[name]["original"], confidences[name]["original"] = classify(
                classifier, target, mean, std
            )
            for stage, image in candidates.items():
                predictions[name][stage], confidences[name][stage] = classify(classifier, image, mean, std)
        for index, sample in enumerate(batch["sample"]):
            row: dict[str, Any] = {
                "analysis_id": config["analysis_id"],
                "sample": str(sample),
                "snr_db": float(snr[index].cpu()),
                "routed_fusion_b1_difference": float((routed_fusion[index] - b1_output[index]).abs().max().cpu()),
                "routed_control_b1_difference": float((routed_control[index] - b1_output[index]).abs().max().cpu()),
            }
            for stage in STAGES:
                for metric in ("psnr", "ms_ssim", "lpips"):
                    row[f"{stage}_{metric}"] = float(quality[stage][metric][index].cpu())
            for name in classifiers:
                row[f"{name}_original_prediction"] = int(predictions[name]["original"][index].cpu())
                row[f"{name}_original_confidence"] = float(confidences[name]["original"][index].cpu())
                for stage in STAGES:
                    row[f"{name}_{stage}_prediction"] = int(predictions[name][stage][index].cpu())
                    row[f"{name}_{stage}_confidence"] = float(confidences[name][stage][index].cpu())
            rows.append(row)
        print(json.dumps({"batch": batch_index + 1, "rows": len(rows)}), flush=True)
    expected_rows = int(config["population"]["roles"]["holdout"]) * len(config["channel"]["snrs_db"])
    if len(rows) != expected_rows:
        raise RuntimeError(f"S27 row count changed: {len(rows)} != {expected_rows}")
    per_sample = output / "per_sample.csv"
    write_csv(per_sample, rows)
    summary = summarize(rows)
    replicates = int(config["evaluation"]["bootstrap_replicates"])
    seed = int(config["evaluation"]["bootstrap_seed"])
    comparisons = {
        f"{left}_minus_{right}_{metric}": bootstrap(rows, left, right, metric, seed, replicates)
        for left, right in (("routed_fusion", "routed_control"), ("routed_fusion", "b1"))
        for metric in ("psnr", "ms_ssim", "lpips", "majority_failure")
    }
    high_snr = [item for item in summary["per_snr"] if float(item["snr_db"]) in {13.0, 19.0}]
    criteria = config["success_criteria"]
    checks = {
        "high_snr_exact_b1": max(
            max(float(item["max_routed_fusion_b1_difference"]), float(item["max_routed_control_b1_difference"]))
            for item in high_snr
        ) <= float(criteria["high_snr_exact_b1_max_abs"]),
        "fusion_control_psnr_ci": comparisons["routed_fusion_minus_routed_control_psnr"]["ci95"][0]
        >= float(criteria["routed_fusion_minus_control_psnr_ci_low_min_db"]),
        "fusion_control_lpips_ci": comparisons["routed_fusion_minus_routed_control_lpips"]["ci95"][1]
        <= float(criteria["routed_fusion_minus_control_lpips_ci_high_max"]),
        "fusion_b1_psnr_mean": summary["routed_fusion_minus_b1_psnr"]
        >= float(criteria["routed_fusion_minus_b1_psnr_mean_min_db"]),
        "fusion_b1_psnr_ci": comparisons["routed_fusion_minus_b1_psnr"]["ci95"][0]
        >= float(criteria["routed_fusion_minus_b1_psnr_ci_low_min_db"]),
        "fusion_b1_lpips_ci": comparisons["routed_fusion_minus_b1_lpips"]["ci95"][1]
        <= float(criteria["routed_fusion_minus_b1_lpips_ci_high_max"]),
        "fusion_b1_nonnegative_all_snr": sum(
            float(item["routed_fusion_minus_b1_psnr"]) >= -1e-12 for item in summary["per_snr"]
        ) >= int(criteria["routed_fusion_minus_b1_nonnegative_snr_count_min"]),
        "majority_new_not_greater_than_repair": summary["majority_routed_fusion_new_vs_b1"]
        <= summary["majority_routed_fusion_repair_vs_b1"],
        "majority_failure_not_greater_than_control": summary["majority_routed_fusion_failure"]
        <= summary["majority_routed_control_failure"],
    }
    summary.update(
        {
            "analysis_id": config["analysis_id"],
            "claim_scope": "pristine_image_population_frozen_method_replication",
            "control_epoch": int(control_checkpoint["epoch"]),
            "fusion_epoch": int(fusion_checkpoint["epoch"]),
            "parameter_count_control": parameter_count(control),
            "parameter_count_fusion": parameter_count(fusion),
            "comparisons": comparisons,
            "checks": checks,
            "verdict": "PASS" if all(checks.values()) else "NEGATIVE",
            "target_selection_accessed": False,
            "official_imagenette_accessed": False,
            "downloaded": False,
            "per_sample_sha256": sha256_file(per_sample),
        }
    )
    summary_path = output / "summary.json"
    save_json(summary_path, summary)
    save_json(
        output / "STATE.json",
        {"state": "PRISTINE_REPLICATION_COMPLETE", "summary_sha256": sha256_file(summary_path), **summary},
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
