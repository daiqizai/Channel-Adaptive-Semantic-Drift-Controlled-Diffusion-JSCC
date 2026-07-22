#!/usr/bin/env python3
"""Measure per-sample amplitude headroom along the frozen S23 feature direction."""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from cadsd_jscc.metrics import ms_ssim_per_sample, psnr_per_sample  # noqa: E402
from s17_channel_matched_latent_diffusion import classifier_model, classify  # noqa: E402
from s19_train_and_evaluate_fusion import FusionPairDataset  # noqa: E402
from s21_b1_anchored_gated_fusion import (  # noqa: E402
    anchor_output,
    load_config,
    resolve,
    save_json,
    seed_everything,
    sha256_file,
    write_csv,
)
from s22_b1_feature_injection import (  # noqa: E402
    _b1_gate,
    build_feature_model,
    envelopes,
)
from s5_residual_refiner_pilot import try_load_lpips  # noqa: E402


def validate(config: dict[str, Any]) -> None:
    protocol = config["protocol"]
    if protocol["status"] != "preregistered_before_selection_diagnostic_output":
        raise RuntimeError("S25 config is not executable")
    required_true = (
        "s23_selection_and_holdout_outcomes_known",
        "development_selection_only",
        "target_dependent_oracles_are_upper_bounds_only",
        "no_new_model_training",
        "no_holdout_access",
    )
    if not all(protocol.get(key) is True for key in required_true):
        raise RuntimeError("S25 diagnostic boundary changed")
    if protocol.get("official_imagenette_accessed") is not False:
        raise RuntimeError("official Imagenette validation must remain sealed")
    for key, hash_key in (
        ("source_manifest", "source_manifest_sha256"),
        ("cache_manifest", "cache_manifest_sha256"),
        ("b1_config", "b1_config_sha256"),
        ("b1_checkpoint", "b1_checkpoint_sha256"),
        ("endpoint_checkpoint", "endpoint_checkpoint_sha256"),
    ):
        path = resolve(config["inputs"][key])
        if not path.is_file() or sha256_file(path) != str(config["inputs"][hash_key]):
            raise RuntimeError(f"S25 frozen input hash mismatch: {key}")
    expected_alphas = [0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.35, 0.5, 0.75, 1.0]
    if [float(value) for value in config["amplitude_headroom"]["alphas"]] != expected_alphas:
        raise RuntimeError("S25 amplitude grid changed")
    if int(config["rate"]["active_real_symbols"]) != 19712:
        raise RuntimeError("strict rate contract changed")
    if int(config["rate"]["fusion_side_information_real_symbols"]) != 0:
        raise RuntimeError("unmetered side information introduced")
    for snr in (13, 19):
        if float(config["feature_injection"]["envelope"][str(snr)]) != 0.0:
            raise RuntimeError("high-SNR exact fallback changed")


def load_endpoint(config: dict[str, Any], device: torch.device):
    model, b1_config = build_feature_model(config, device)
    checkpoint = torch.load(resolve(config["inputs"]["endpoint_checkpoint"]), map_location=device)
    if int(checkpoint["epoch"]) != 1:
        raise RuntimeError("S25 requires the frozen one-epoch endpoint")
    if checkpoint["b1_checkpoint_sha256"] != config["inputs"]["b1_checkpoint_sha256"]:
        raise RuntimeError("endpoint was trained against another B1")
    model.aux_projection.load_state_dict(checkpoint["projection_state_dict"], strict=True)
    return model.eval().requires_grad_(False), b1_config


def majority_correct(predictions: dict[str, int], originals: dict[str, int]) -> bool:
    return sum(predictions[name] == originals[name] for name in predictions) >= 2


def summarize_policy(
    name: str, selected: list[dict[str, Any]], grouped: dict[tuple[str, float], list[dict[str, Any]]]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "policy": name,
        "rows": len(selected),
        "mean_psnr": float(np.mean([float(row["psnr"]) for row in selected])),
        "mean_ms_ssim": float(np.mean([float(row["ms_ssim"]) for row in selected])),
        "mean_lpips": float(np.mean([float(row["lpips"]) for row in selected])),
        "majority_failure": sum(not bool(row["candidate_majority_correct"]) for row in selected),
        "majority_new_vs_b1": sum(
            bool(row["b1_majority_correct"]) and not bool(row["candidate_majority_correct"])
            for row in selected
        ),
        "majority_repair_vs_b1": sum(
            not bool(row["b1_majority_correct"]) and bool(row["candidate_majority_correct"])
            for row in selected
        ),
        "alpha_counts": dict(sorted(Counter(float(row["alpha"]) for row in selected).items())),
    }
    result["per_snr"] = {}
    for snr in sorted({float(row["snr_db"]) for row in selected}):
        subset = [row for row in selected if float(row["snr_db"]) == snr]
        result["per_snr"][str(int(snr))] = {
            "rows": len(subset),
            "mean_psnr": float(np.mean([float(row["psnr"]) for row in subset])),
            "mean_ms_ssim": float(np.mean([float(row["ms_ssim"]) for row in subset])),
            "mean_lpips": float(np.mean([float(row["lpips"]) for row in subset])),
            "majority_failure": sum(not bool(row["candidate_majority_correct"]) for row in subset),
            "alpha_counts": dict(sorted(Counter(float(row["alpha"]) for row in subset).items())),
        }
    return result


def bootstrap_difference(
    selected: list[dict[str, Any]], fixed: list[dict[str, Any]], seed: int, replicates: int
) -> dict[str, Any]:
    fixed_map = {(str(row["sample"]), float(row["snr_db"])): row for row in fixed}
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for row in selected:
        key = (str(row["sample"]), float(row["snr_db"]))
        grouped[key[0]].append(float(row["psnr"]) - float(fixed_map[key]["psnr"]))
    names = sorted(grouped)
    matrix = np.asarray([grouped[name] for name in names], dtype=np.float64)
    if matrix.shape[1] != 5:
        raise RuntimeError("bootstrap clusters do not contain five SNR rows")
    cluster_values = matrix.mean(axis=1)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = rng.integers(0, len(names), size=len(names))
        draws[index] = float(cluster_values[sample].mean())
    return {
        "unit": "source_image_cluster_across_five_snrs",
        "clusters": len(names),
        "replicates": replicates,
        "seed": seed,
        "mean": float(cluster_values.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


@torch.no_grad()
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/s25_b1_feature_amplitude_headroom.yaml")
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
    model, b1_config = load_endpoint(config, device)
    dataset = FusionPairDataset(config, "selection", train=False)
    loader = DataLoader(
        dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["evaluation"]["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(config["evaluation"]["num_workers"]) > 0,
    )
    lpips_model, lpips_error = try_load_lpips(device, resolve("outputs/cache"))
    if lpips_model is None:
        raise RuntimeError(lpips_error)
    lpips_model.eval().requires_grad_(False)
    classifier_names = tuple(config["amplitude_headroom"]["semantic_models"])
    classifiers = {
        name: classifier_model(name, resolve(config["classifiers"][name]), device)
        for name in classifier_names
    }
    mean = torch.tensor(config["classifiers"]["imagenet_mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(config["classifiers"]["imagenet_std"], device=device).view(1, 3, 1, 1)
    alphas = [float(value) for value in config["amplitude_headroom"]["alphas"]]
    rows: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(loader):
        b0 = batch["b0"].to(device, non_blocking=True)
        diffusion = batch["auxiliary"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        snr = batch["snr_db"].to(device, non_blocking=True)
        snr_norm = batch["snr_norm"].to(device, non_blocking=True)
        gate = _b1_gate(b1_config, snr, device)
        base_envelope = envelopes(config, snr, device)
        b1 = anchor_output(model.b1, b1_config, b0, snr, snr_norm, device)
        originals: dict[str, torch.Tensor] = {}
        anchors: dict[str, torch.Tensor] = {}
        original_confidences: dict[str, torch.Tensor] = {}
        anchor_confidences: dict[str, torch.Tensor] = {}
        for name, classifier in classifiers.items():
            originals[name], original_confidences[name] = classify(classifier, target, mean, std)
            anchors[name], anchor_confidences[name] = classify(classifier, b1, mean, std)
        candidates = [
            model(b0, diffusion, snr_norm, gate, base_envelope * alpha) for alpha in alphas
        ]
        quality = []
        candidate_predictions: list[dict[str, torch.Tensor]] = []
        candidate_confidences: list[dict[str, torch.Tensor]] = []
        for candidate in candidates:
            quality.append(
                {
                    "psnr": psnr_per_sample(candidate, target),
                    "ms_ssim": ms_ssim_per_sample(candidate, target),
                    "lpips": lpips_model(candidate * 2.0 - 1.0, target * 2.0 - 1.0).flatten(),
                }
            )
            predictions: dict[str, torch.Tensor] = {}
            confidences: dict[str, torch.Tensor] = {}
            for name, classifier in classifiers.items():
                predictions[name], confidences[name] = classify(classifier, candidate, mean, std)
            candidate_predictions.append(predictions)
            candidate_confidences.append(confidences)
        base_projection = model.aux_projection(diffusion - b0).abs().flatten(1).mean(1)
        b0_diffusion_l1 = (diffusion - b0).abs().flatten(1).mean(1)
        b1_diffusion_l1 = (diffusion - b1).abs().flatten(1).mean(1)
        for item_index, sample in enumerate(batch["sample"]):
            original_map = {name: int(originals[name][item_index].cpu()) for name in classifier_names}
            b1_map = {name: int(anchors[name][item_index].cpu()) for name in classifier_names}
            b1_correct = majority_correct(b1_map, original_map)
            for alpha_index, alpha in enumerate(alphas):
                candidate_map = {
                    name: int(candidate_predictions[alpha_index][name][item_index].cpu())
                    for name in classifier_names
                }
                row: dict[str, Any] = {
                    "analysis_id": config["analysis_id"],
                    "sample": str(sample),
                    "snr_db": float(snr[item_index].cpu()),
                    "alpha": alpha,
                    "psnr": float(quality[alpha_index]["psnr"][item_index].cpu()),
                    "ms_ssim": float(quality[alpha_index]["ms_ssim"][item_index].cpu()),
                    "lpips": float(quality[alpha_index]["lpips"][item_index].cpu()),
                    "b1_majority_correct": b1_correct,
                    "candidate_majority_correct": majority_correct(candidate_map, original_map),
                    "base_projection_abs_mean": float(base_projection[item_index].cpu()),
                    "b0_diffusion_l1": float(b0_diffusion_l1[item_index].cpu()),
                    "b1_diffusion_l1": float(b1_diffusion_l1[item_index].cpu()),
                }
                for name in classifier_names:
                    row[f"{name}_original_prediction"] = original_map[name]
                    row[f"{name}_b1_prediction"] = b1_map[name]
                    row[f"{name}_candidate_prediction"] = candidate_map[name]
                    row[f"{name}_original_confidence"] = float(original_confidences[name][item_index].cpu())
                    row[f"{name}_b1_confidence"] = float(anchor_confidences[name][item_index].cpu())
                    row[f"{name}_candidate_confidence"] = float(
                        candidate_confidences[alpha_index][name][item_index].cpu()
                    )
                rows.append(row)
        print(json.dumps({"batch": batch_index + 1, "rows": len(rows)}), flush=True)
    expected = len(dataset) * len(alphas)
    if len(rows) != expected:
        raise RuntimeError(f"S25 row count changed: {len(rows)} != {expected}")
    long_csv = output / "alpha_candidates_long.csv"
    write_csv(long_csv, rows)
    grouped: defaultdict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["sample"]), float(row["snr_db"]))].append(row)
    fixed_alpha = float(config["amplitude_headroom"]["fixed_reference_alpha"])
    fixed: list[dict[str, Any]] = []
    psnr_oracle: list[dict[str, Any]] = []
    lpips_oracle: list[dict[str, Any]] = []
    semantic_safe: list[dict[str, Any]] = []
    choice_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        candidates_for_key = grouped[key]
        fixed_item = next(row for row in candidates_for_key if abs(float(row["alpha"]) - fixed_alpha) < 1e-12)
        psnr_item = max(candidates_for_key, key=lambda row: (float(row["psnr"]), -float(row["alpha"])))
        lpips_item = min(candidates_for_key, key=lambda row: (float(row["lpips"]), float(row["alpha"])))
        safe_candidates = [
            row
            for row in candidates_for_key
            if not (bool(row["b1_majority_correct"]) and not bool(row["candidate_majority_correct"]))
        ]
        safe_item = max(safe_candidates, key=lambda row: (float(row["psnr"]), -float(row["alpha"])))
        fixed.append(fixed_item)
        psnr_oracle.append(psnr_item)
        lpips_oracle.append(lpips_item)
        semantic_safe.append(safe_item)
        choice_rows.append(
            {
                "sample": key[0],
                "snr_db": key[1],
                "fixed_alpha": fixed_alpha,
                "psnr_oracle_alpha": float(psnr_item["alpha"]),
                "lpips_oracle_alpha": float(lpips_item["alpha"]),
                "semantic_safe_psnr_oracle_alpha": float(safe_item["alpha"]),
                "psnr_oracle_minus_fixed_psnr": float(psnr_item["psnr"]) - float(fixed_item["psnr"]),
                "semantic_safe_oracle_minus_fixed_psnr": float(safe_item["psnr"]) - float(fixed_item["psnr"]),
                "semantic_safe_oracle_minus_fixed_lpips": float(safe_item["lpips"]) - float(fixed_item["lpips"]),
            }
        )
    choices_csv = output / "oracle_choices.csv"
    write_csv(choices_csv, choice_rows)
    policy_rows = {
        "fixed_0p15": fixed,
        "psnr_oracle": psnr_oracle,
        "lpips_oracle": lpips_oracle,
        "semantic_safe_psnr_oracle": semantic_safe,
    }
    policies = {
        name: summarize_policy(name, selected, grouped) for name, selected in policy_rows.items()
    }
    reference = policies["fixed_0p15"]
    for name, item in policies.items():
        item["minus_fixed_psnr"] = item["mean_psnr"] - reference["mean_psnr"]
        item["minus_fixed_ms_ssim"] = item["mean_ms_ssim"] - reference["mean_ms_ssim"]
        item["minus_fixed_lpips"] = item["mean_lpips"] - reference["mean_lpips"]
    bootstrap = bootstrap_difference(
        semantic_safe,
        fixed,
        int(config["evaluation"]["bootstrap_seed"]),
        int(config["evaluation"]["bootstrap_replicates"]),
    )
    low_rows = [row for row in semantic_safe if float(row["snr_db"]) in {1.0, 4.0, 7.0}]
    nonreference_fraction = float(
        np.mean([abs(float(row["alpha"]) - fixed_alpha) > 1e-12 for row in low_rows])
    )
    gates = config["decision"]["continue_to_receiver_visible_controller_only_if"]
    safe_summary = policies["semantic_safe_psnr_oracle"]
    checks = {
        "psnr_headroom": safe_summary["minus_fixed_psnr"]
        >= float(gates["semantic_safe_oracle_minus_fixed_psnr_min_db"]),
        "psnr_ci_low": bootstrap["ci95"][0]
        >= float(gates["semantic_safe_oracle_minus_fixed_psnr_ci_low_min_db"]),
        "lpips_not_worse": safe_summary["minus_fixed_lpips"]
        <= float(gates["semantic_safe_oracle_minus_fixed_lpips_max"]),
        "nonreference_fraction": nonreference_fraction
        >= float(gates["low_snr_nonreference_alpha_fraction_min"]),
    }
    summary = {
        "analysis_id": config["analysis_id"],
        "claim_scope": "selection_only_target_dependent_oracle_headroom",
        "selection_rows": len(dataset),
        "candidate_rows": len(rows),
        "unique_images": len({str(row["sample"]) for row in fixed}),
        "alphas": alphas,
        "policies": policies,
        "semantic_safe_oracle_minus_fixed_psnr_bootstrap": bootstrap,
        "low_snr_nonreference_alpha_fraction": nonreference_fraction,
        "checks": checks,
        "continue_to_receiver_visible_controller": all(checks.values()),
        "holdout_accessed": False,
        "official_imagenette_accessed": False,
        "downloaded": False,
        "artifacts": {
            "alpha_candidates_long_sha256": sha256_file(long_csv),
            "oracle_choices_sha256": sha256_file(choices_csv),
        },
    }
    summary_path = output / "summary.json"
    save_json(summary_path, summary)
    save_json(
        output / "STATE.json",
        {"state": "SELECTION_DIAGNOSTIC_COMPLETE", "summary_sha256": sha256_file(summary_path), **summary},
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
