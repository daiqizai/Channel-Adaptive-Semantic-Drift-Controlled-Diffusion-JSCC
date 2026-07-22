#!/usr/bin/env python3
"""Select and audit a monotone convex envelope between frozen B1 and diffusion."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import save_image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from cadsd_jscc.metrics import ms_ssim_per_sample, psnr_per_sample  # noqa: E402
from s17_channel_matched_latent_diffusion import classifier_model, classify  # noqa: E402
from s19_train_and_evaluate_fusion import FusionPairDataset  # noqa: E402
from s21_b1_anchored_gated_fusion import (  # noqa: E402
    anchor_output,
    build_b1,
    load_config,
    resolve,
    save_json,
    seed_everything,
    sha256_file,
    write_csv,
)
from s5_residual_refiner_pilot import try_load_lpips  # noqa: E402


STAGES = ("b0", "diffusion", "b1", "blend")


def validate(config: dict[str, Any], mode: str) -> None:
    expected = {
        "selection": "cache_frozen_before_blend_selection",
        "holdout": "blend_policy_frozen_before_holdout",
        "bootstrap": "blend_policy_frozen_before_holdout",
    }[mode]
    if config["protocol"]["status"] != expected:
        raise RuntimeError(f"S21 convex config is not executable for {mode}")
    if config["protocol"].get("official_imagenette_accessed") is not False:
        raise RuntimeError("official Imagenette validation must remain sealed")
    for key, hash_key in (
        ("source_manifest", "source_manifest_sha256"),
        ("cache_manifest", "cache_manifest_sha256"),
        ("b1_config", "b1_config_sha256"),
        ("b1_checkpoint", "b1_checkpoint_sha256"),
    ):
        path = resolve(config["inputs"][key])
        if not path.is_file() or sha256_file(path) != str(config["inputs"][hash_key]):
            raise RuntimeError(f"input hash mismatch: {key}")
    if mode != "selection":
        policy = resolve(config["inputs"]["selected_blend_policy"])
        if not policy.is_file() or sha256_file(policy) != str(
            config["inputs"]["selected_blend_policy_sha256"]
        ):
            raise RuntimeError("blend policy hash mismatch")


@torch.no_grad()
def selection_metrics(
    config: dict[str, Any], device: torch.device
) -> tuple[dict[float, dict[float, dict[str, float]]], dict[float, dict[str, float]]]:
    dataset = FusionPairDataset(config, "selection", train=False)
    loader = DataLoader(
        dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["evaluation"]["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(config["evaluation"]["num_workers"]) > 0,
    )
    b1, b1_config = build_b1(config, device)
    lpips_model, error = try_load_lpips(device, resolve("outputs/cache"))
    if lpips_model is None:
        raise RuntimeError(error)
    alphas = [float(value) for value in config["blend"]["candidate_alphas"]]
    accum: defaultdict[float, defaultdict[float, defaultdict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    b1_accum: defaultdict[float, defaultdict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for batch in loader:
        b0 = batch["b0"].to(device, non_blocking=True)
        diffusion = batch["auxiliary"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        snr = batch["snr_db"].to(device, non_blocking=True)
        snr_norm = batch["snr_norm"].to(device, non_blocking=True)
        anchor = anchor_output(b1, b1_config, b0, snr, snr_norm, device)
        for snr_value in snr.unique().detach().cpu().tolist():
            value = float(snr_value)
            mask = snr == value
            anchor_subset = anchor[mask]
            diffusion_subset = diffusion[mask]
            target_subset = target[mask]
            count = int(mask.sum().cpu())
            b1_accum[value]["count"] += count
            b1_accum[value]["psnr"] += float(
                psnr_per_sample(anchor_subset, target_subset).sum().cpu()
            )
            b1_accum[value]["lpips"] += float(
                lpips_model(
                    anchor_subset * 2.0 - 1.0,
                    target_subset * 2.0 - 1.0,
                ).sum().cpu()
            )
            for alpha in alphas:
                blend = anchor_subset.lerp(diffusion_subset, alpha)
                accum[value][alpha]["count"] += count
                accum[value][alpha]["psnr"] += float(
                    psnr_per_sample(blend, target_subset).sum().cpu()
                )
                accum[value][alpha]["lpips"] += float(
                    lpips_model(blend * 2.0 - 1.0, target_subset * 2.0 - 1.0)
                    .sum()
                    .cpu()
                )
    metrics = {
        snr: {
            alpha: {
                "mean_psnr": values["psnr"] / values["count"],
                "mean_lpips": values["lpips"] / values["count"],
            }
            for alpha, values in by_alpha.items()
        }
        for snr, by_alpha in accum.items()
    }
    baseline = {
        snr: {
            "mean_psnr": values["psnr"] / values["count"],
            "mean_lpips": values["lpips"] / values["count"],
        }
        for snr, values in b1_accum.items()
    }
    return metrics, baseline


def run_selection(config: dict[str, Any], config_path: Path, device: torch.device) -> None:
    validate(config, "selection")
    output = resolve(config["outputs"]["selection_dir"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    shutil.copy2(config_path, output / "config_before_selection.yaml")
    shutil.copy2(SCRIPT, output / SCRIPT.name)
    seed_everything(int(config["seed"]))
    metrics, baseline = selection_metrics(config, device)
    selectable = [float(value) for value in config["blend"]["selectable_snrs_db"]]
    alphas = [float(value) for value in config["blend"]["candidate_alphas"]]
    candidates: list[dict[str, Any]] = []
    for values in itertools.product(alphas, repeat=len(selectable)):
        if any(values[index] < values[index + 1] for index in range(len(values) - 1)):
            continue
        policy = {snr: alpha for snr, alpha in zip(selectable, values)}
        policy.update({float(snr): 0.0 for snr in config["blend"]["exact_b1_snrs_db"]})
        per_snr = {
            snr: {
                "alpha": policy[snr],
                "mean_psnr": metrics[snr][policy[snr]]["mean_psnr"],
                "mean_lpips": metrics[snr][policy[snr]]["mean_lpips"],
                "psnr_delta_vs_b1": metrics[snr][policy[snr]]["mean_psnr"]
                - baseline[snr]["mean_psnr"],
                "lpips_delta_vs_b1": metrics[snr][policy[snr]]["mean_lpips"]
                - baseline[snr]["mean_lpips"],
            }
            for snr in sorted(policy)
        }
        aggregate_psnr = float(np.mean([item["mean_psnr"] for item in per_snr.values()]))
        aggregate_lpips_delta = float(
            np.mean([item["lpips_delta_vs_b1"] for item in per_snr.values()])
        )
        feasible = all(
            per_snr[snr]["psnr_delta_vs_b1"]
            >= float(
                config["blend"]["selection_constraints"][
                    "each_low_snr_psnr_delta_vs_b1_min_db"
                ]
            )
            - 1e-12
            for snr in selectable
        ) and aggregate_lpips_delta <= float(
            config["blend"]["selection_constraints"]["aggregate_lpips_delta_vs_b1_max"]
        ) + 1e-12
        candidates.append(
            {
                "alphas": {str(int(snr)): policy[snr] for snr in sorted(policy)},
                "per_snr": {str(int(snr)): item for snr, item in per_snr.items()},
                "aggregate_mean_psnr": aggregate_psnr,
                "aggregate_mean_lpips_delta_vs_b1": aggregate_lpips_delta,
                "sum_low_snr_alpha": sum(policy[snr] for snr in selectable),
                "feasible": feasible,
            }
        )
    feasible = [item for item in candidates if item["feasible"]]
    if not feasible:
        raise RuntimeError("zero-alpha B1 policy unexpectedly infeasible")
    selected = max(
        feasible,
        key=lambda item: (
            item["aggregate_mean_psnr"],
            -item["sum_low_snr_alpha"],
            tuple(-item["alphas"][str(int(snr))] for snr in selectable),
        ),
    )
    policy = {
        "analysis_id": config["selection_analysis_id"],
        "selected_alphas": selected["alphas"],
        "selected_metrics": selected,
        "candidate_alphas": alphas,
        "monotonic_rule": config["blend"]["monotonic_rule"],
        "has_nonzero_alpha": any(float(value) > 0 for value in selected["alphas"].values()),
        "official_imagenette_accessed": False,
        "holdout_accessed": False,
    }
    save_json(output / "selected_policy.json", policy)
    write_csv(
        output / "candidate_summary.csv",
        [
            {
                "alpha_1": item["alphas"]["1"],
                "alpha_4": item["alphas"]["4"],
                "alpha_7": item["alphas"]["7"],
                "aggregate_mean_psnr": item["aggregate_mean_psnr"],
                "aggregate_mean_lpips_delta_vs_b1": item[
                    "aggregate_mean_lpips_delta_vs_b1"
                ],
                "feasible": item["feasible"],
            }
            for item in candidates
        ],
    )
    result = {
        **policy,
        "selected_policy_sha256": sha256_file(output / "selected_policy.json"),
        "candidate_count": len(candidates),
        "feasible_count": len(feasible),
    }
    save_json(output / "selection_summary.json", result)
    save_json(output / "STATE.json", {"state": "SELECTION_COMPLETE", **result})
    print(json.dumps(result, ensure_ascii=False, indent=2))


def policy_alphas(config: dict[str, Any]) -> dict[float, float]:
    payload = json.loads(
        resolve(config["inputs"]["selected_blend_policy"]).read_text(encoding="utf-8")
    )
    return {float(key): float(value) for key, value in payload["selected_alphas"].items()}


def semantic_summary(rows: list[dict[str, Any]], summary: dict[str, Any], config: dict[str, Any]) -> None:
    threshold = float(config["evaluation"]["pseudo_original_confidence_min"])
    eligible = [row for row in rows if float(row["alexnet_original_confidence"]) >= threshold]
    summary["alexnet_eligible_rows"] = len(eligible)
    for stage in STAGES:
        summary[f"alexnet_{stage}_failure"] = sum(
            int(row[f"alexnet_{stage}_prediction"]) != int(row["alexnet_original_prediction"])
            for row in eligible
        )
        summary[f"alexnet_{stage}_new_vs_b1"] = sum(
            int(row["alexnet_b1_prediction"]) == int(row["alexnet_original_prediction"])
            and int(row[f"alexnet_{stage}_prediction"]) != int(row["alexnet_original_prediction"])
            for row in eligible
        )
        summary[f"alexnet_{stage}_repair_vs_b1"] = sum(
            int(row["alexnet_b1_prediction"]) != int(row["alexnet_original_prediction"])
            and int(row[f"alexnet_{stage}_prediction"]) == int(row["alexnet_original_prediction"])
            for row in eligible
        )
        stage_correct = [
            sum(
                int(row[f"{name}_{stage}_prediction"])
                == int(row[f"{name}_original_prediction"])
                for name in ("alexnet", "resnet18", "mobilenet_v3_small")
            )
            >= 2
            for row in rows
        ]
        b1_correct = [
            sum(
                int(row[f"{name}_b1_prediction"])
                == int(row[f"{name}_original_prediction"])
                for name in ("alexnet", "resnet18", "mobilenet_v3_small")
            )
            >= 2
            for row in rows
        ]
        summary[f"majority_{stage}_failure"] = sum(not value for value in stage_correct)
        summary[f"majority_{stage}_new_vs_b1"] = sum(
            base and not candidate for base, candidate in zip(b1_correct, stage_correct)
        )
        summary[f"majority_{stage}_repair_vs_b1"] = sum(
            not base and candidate for base, candidate in zip(b1_correct, stage_correct)
        )


@torch.no_grad()
def run_holdout(config: dict[str, Any], config_path: Path, device: torch.device) -> None:
    validate(config, "holdout")
    output = resolve(config["outputs"]["holdout_dir"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    shutil.copy2(config_path, output / "config_before_holdout_access.yaml")
    shutil.copy2(SCRIPT, output / SCRIPT.name)
    dataset = FusionPairDataset(config, "holdout", train=False)
    loader = DataLoader(
        dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["evaluation"]["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(config["evaluation"]["num_workers"]) > 0,
    )
    b1, b1_config = build_b1(config, device)
    lpips_model, error = try_load_lpips(device, resolve("outputs/cache"))
    if lpips_model is None:
        raise RuntimeError(error)
    classifiers = {
        name: classifier_model(name, resolve(config["classifiers"][name]), device)
        for name in ("alexnet", "resnet18", "mobilenet_v3_small")
    }
    mean = torch.tensor(config["classifiers"]["imagenet_mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(config["classifiers"]["imagenet_std"], device=device).view(1, 3, 1, 1)
    alphas = policy_alphas(config)
    rows: list[dict[str, Any]] = []
    saved: set[float] = set()
    for batch in loader:
        b0 = batch["b0"].to(device, non_blocking=True)
        diffusion = batch["auxiliary"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        snr = batch["snr_db"].to(device, non_blocking=True)
        snr_norm = batch["snr_norm"].to(device, non_blocking=True)
        anchor = anchor_output(b1, b1_config, b0, snr, snr_norm, device)
        alpha = torch.tensor([alphas[float(value)] for value in snr.cpu().tolist()], device=device)
        blend = anchor.lerp(diffusion, alpha.view(-1, 1, 1, 1))
        candidates = {"b0": b0, "diffusion": diffusion, "b1": anchor, "blend": blend}
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
            predictions[name], confidences[name] = {}, {}
            prediction, confidence = classify(classifier, target, mean, std)
            predictions[name]["original"], confidences[name]["original"] = prediction, confidence
            for stage, image in candidates.items():
                prediction, confidence = classify(classifier, image, mean, std)
                predictions[name][stage], confidences[name][stage] = prediction, confidence
        batch_snr = float(snr[0].cpu())
        if batch_snr not in saved:
            count = min(int(config["evaluation"]["sample_grid_count"]), b0.shape[0])
            save_image(
                torch.cat([target[:count], *[candidates[stage][:count] for stage in STAGES]]).cpu(),
                output / f"snr_{int(batch_snr):02d}_convex_grid.png",
                nrow=count,
            )
            saved.add(batch_snr)
        for index, sample in enumerate(batch["sample"]):
            row: dict[str, Any] = {
                "analysis_id": config["holdout_analysis_id"],
                "sample": sample,
                "snr_db": float(snr[index].cpu()),
                "alpha": float(alpha[index].cpu()),
                "blend_anchor_max_abs": float((blend[index] - anchor[index]).abs().max().cpu()),
            }
            for stage in STAGES:
                for metric in ("psnr", "ms_ssim", "lpips"):
                    row[f"{stage}_{metric}"] = float(quality[stage][metric][index].cpu())
            for name in classifiers:
                for stage in ("original", *STAGES):
                    row[f"{name}_{stage}_prediction"] = int(predictions[name][stage][index].cpu())
                    row[f"{name}_{stage}_confidence"] = float(confidences[name][stage][index].cpu())
            rows.append(row)
    write_csv(output / "per_sample.csv", rows)
    summary: dict[str, Any] = {
        "analysis_id": config["holdout_analysis_id"],
        "rows": len(rows),
        "images": len({row["sample"] for row in rows}),
        "selected_alphas": {str(int(key)): value for key, value in sorted(alphas.items())},
        "per_snr": [],
    }
    for snr_value in sorted(alphas):
        subset = [row for row in rows if float(row["snr_db"]) == snr_value]
        item: dict[str, Any] = {"snr_db": snr_value, "alpha": alphas[snr_value]}
        for stage in STAGES:
            for metric in ("psnr", "ms_ssim", "lpips"):
                item[f"mean_{stage}_{metric}"] = float(
                    np.mean([float(row[f"{stage}_{metric}"]) for row in subset])
                )
        item["blend_minus_b1_psnr"] = item["mean_blend_psnr"] - item["mean_b1_psnr"]
        item["blend_minus_b1_lpips"] = item["mean_blend_lpips"] - item["mean_b1_lpips"]
        item["max_anchor_difference"] = max(float(row["blend_anchor_max_abs"]) for row in subset)
        summary["per_snr"].append(item)
    for stage in STAGES:
        for metric in ("psnr", "ms_ssim", "lpips"):
            summary[f"mean_{stage}_{metric}"] = float(
                np.mean([float(row[f"{stage}_{metric}"]) for row in rows])
            )
    for metric in ("psnr", "lpips"):
        summary[f"blend_minus_b1_{metric}"] = (
            summary[f"mean_blend_{metric}"] - summary[f"mean_b1_{metric}"]
        )
    semantic_summary(rows, summary, config)
    summary["official_imagenette_accessed"] = False
    save_json(output / "summary.json", summary)
    save_json(output / "STATE.json", {"state": "HOLDOUT_COMPLETE", **summary})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_bootstrap(config: dict[str, Any], config_path: Path) -> None:
    validate(config, "bootstrap")
    holdout = resolve(config["outputs"]["holdout_dir"])
    with (holdout / "per_sample.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["sample"]].append(row)
    names = sorted(grouped)
    if len(names) != int(config["population"]["roles"]["holdout"]):
        raise RuntimeError("holdout image count changed")
    matrix = np.asarray(
        [
            [
                np.mean([float(row["blend_psnr"]) - float(row["b1_psnr"]) for row in grouped[name]]),
                np.mean([float(row["blend_lpips"]) - float(row["b1_lpips"]) for row in grouped[name]]),
            ]
            for name in names
        ]
    )
    rng = np.random.default_rng(int(config["evaluation"]["bootstrap_seed"]))
    replicates = int(config["evaluation"]["bootstrap_replicates"])
    sampled = rng.integers(0, len(names), size=(replicates, len(names)))
    distribution = matrix[sampled].mean(axis=1)
    intervals = {
        key: {
            "mean": float(matrix[:, index].mean()),
            "ci_low": float(np.quantile(distribution[:, index], 0.025)),
            "ci_high": float(np.quantile(distribution[:, index], 0.975)),
        }
        for index, key in enumerate(("blend_minus_b1_psnr", "blend_minus_b1_lpips"))
    }
    summary = json.loads((holdout / "summary.json").read_text(encoding="utf-8"))
    criteria = config["success_criteria"]
    checks = {
        "blend_minus_b1_psnr_ci_low": intervals["blend_minus_b1_psnr"]["ci_low"]
        > float(criteria["blend_minus_b1_psnr_ci_low_min_db"]),
        "blend_minus_b1_lpips_ci_high": intervals["blend_minus_b1_lpips"]["ci_high"]
        < float(criteria["blend_minus_b1_lpips_ci_high_max"]),
        "blend_minus_b1_nonnegative_all_snr_count": sum(
            float(item["blend_minus_b1_psnr"]) >= -1e-12 for item in summary["per_snr"]
        )
        >= int(criteria["blend_minus_b1_nonnegative_all_snr_count_min"]),
        "high_snr_exact_b1": max(
            float(item["max_anchor_difference"])
            for item in summary["per_snr"]
            if float(item["snr_db"]) in {13.0, 19.0}
        )
        <= float(criteria["high_snr_exact_b1_max_abs"]),
        "blend_majority_new_not_greater_than_repair": int(
            summary["majority_blend_new_vs_b1"]
        )
        <= int(summary["majority_blend_repair_vs_b1"]),
    }
    result = {
        "analysis_id": config["bootstrap_analysis_id"],
        "holdout_per_sample_sha256": sha256_file(holdout / "per_sample.csv"),
        "clusters": len(names),
        "replicates": replicates,
        "seed": int(config["evaluation"]["bootstrap_seed"]),
        "intervals": intervals,
        "checks": checks,
        "pass_count": sum(checks.values()),
        "check_count": len(checks),
        "primary_pareto_merge_demonstrated": checks["blend_minus_b1_psnr_ci_low"]
        and checks["blend_minus_b1_lpips_ci_high"],
        "official_imagenette_accessed": False,
    }
    output = resolve(config["outputs"]["bootstrap_dir"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    shutil.copy2(config_path, output / "config.yaml")
    shutil.copy2(SCRIPT, output / SCRIPT.name)
    save_json(output / "bootstrap_summary.json", result)
    save_json(output / "STATE.json", {"state": "COMPLETE", **result})
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/s21_b1_diffusion_convex_envelope.yaml")
    parser.add_argument("--mode", choices=("selection", "holdout", "bootstrap"), required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if args.mode == "selection":
        run_selection(config, config_path, torch.device(args.device))
    elif args.mode == "holdout":
        run_holdout(config, config_path, torch.device(args.device))
    else:
        run_bootstrap(config, config_path)


if __name__ == "__main__":
    main()
