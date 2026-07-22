#!/usr/bin/env python3
"""Train the frozen S23 endpoint and select a preregistered global shrink."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from s21_b1_anchored_gated_fusion import (  # noqa: E402
    load_config,
    resolve,
    save_json,
    seed_everything,
    sha256_file,
)
from s22_b1_feature_injection import (  # noqa: E402
    build_feature_model,
    evaluate_selection,
    make_loaders,
    save_checkpoint,
    train_epoch,
)
from s5_residual_refiner_pilot import try_load_lpips  # noqa: E402


def validate(config: dict[str, Any]) -> None:
    if config["protocol"]["status"] != "preregistered_before_endpoint_training_output":
        raise RuntimeError("S23 config is not executable for selection")
    if config["protocol"].get("official_imagenette_accessed") is not False:
        raise RuntimeError("official Imagenette validation must remain sealed")
    if int(config["training"]["epochs"]) != 1:
        raise RuntimeError("S23 endpoint epoch changed")
    if config["feature_shrink"]["global_alphas"] != [
        0.0,
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.15,
        0.2,
        0.35,
        0.5,
        0.75,
        1.0,
    ]:
        raise RuntimeError("S23 global shrink grid changed")
    for key, hash_key in (
        ("source_manifest", "source_manifest_sha256"),
        ("cache_manifest", "cache_manifest_sha256"),
        ("b1_config", "b1_config_sha256"),
        ("b1_checkpoint", "b1_checkpoint_sha256"),
    ):
        path = resolve(config["inputs"][key])
        if not path.is_file() or sha256_file(path) != config["inputs"][hash_key]:
            raise RuntimeError(f"S23 frozen input hash mismatch: {key}")
    if int(config["rate"]["active_real_symbols"]) != 19712:
        raise RuntimeError("strict rate contract changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/s23_b1_feature_shrink.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    os.environ.setdefault("TORCH_HOME", str(resolve("outputs/cache/torch")))
    config_path = resolve(args.config)
    config = load_config(config_path)
    validate(config)
    device = torch.device(args.device)
    training_output = resolve(config["outputs"]["training_dir"])
    selection_output = resolve(config["outputs"]["selection_dir"])
    if training_output.exists() or selection_output.exists():
        raise FileExistsError("S23 output already exists")
    checkpoint_dir = training_output / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    selection_output.mkdir(parents=True)
    shutil.copy2(config_path, training_output / "config_before_training.yaml")
    shutil.copy2(SCRIPT, training_output / SCRIPT.name)
    seed_everything(int(config["training"]["seed"]))
    train_dataset, selection_dataset, train_loader, selection_loader = make_loaders(
        config, device
    )
    model, b1_config = build_feature_model(config, device)
    optimizer = torch.optim.Adam(
        [model.aux_projection.weight],
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    lpips_model, lpips_error = try_load_lpips(device, resolve("outputs/cache"))
    if lpips_model is None:
        raise RuntimeError(lpips_error)
    lpips_model.eval().requires_grad_(False)
    baseline = evaluate_selection(
        model, b1_config, selection_loader, lpips_model, config, device
    )
    training = train_epoch(
        model, b1_config, train_loader, optimizer, lpips_model, config, device
    )
    endpoint_weight = model.aux_projection.weight.detach().clone()
    endpoint_path = checkpoint_dir / "endpoint_epoch1.pt"
    save_checkpoint(endpoint_path, model, optimizer, 1, {}, config)
    candidates = []
    for alpha in config["feature_shrink"]["global_alphas"]:
        with torch.no_grad():
            model.aux_projection.weight.copy_(endpoint_weight * float(alpha))
        stats = evaluate_selection(
            model, b1_config, selection_loader, lpips_model, config, device
        )
        low_nonnegative = sum(
            stats["per_snr"][str(snr)]["fusion_minus_b1_psnr"] >= -1e-12
            for snr in (1, 4, 7)
        )
        feasible = (
            stats["fusion_minus_b1_lpips"] <= 0.0
            and low_nonnegative >= 3
            and stats["fusion_minus_b1_psnr"] >= -1e-12
        )
        candidates.append(
            {
                "alpha": float(alpha),
                "low_snr_nonnegative_count": low_nonnegative,
                "feasible": feasible,
                **stats,
            }
        )
        print(
            json.dumps(
                {
                    "alpha": alpha,
                    "psnr_delta": stats["fusion_minus_b1_psnr"],
                    "lpips_delta": stats["fusion_minus_b1_lpips"],
                    "low_nonnegative": low_nonnegative,
                    "feasible": feasible,
                }
            ),
            flush=True,
        )
    feasible_candidates = [item for item in candidates if item["feasible"]]
    if not feasible_candidates:
        raise RuntimeError("alpha=0 baseline unexpectedly infeasible")
    selected = sorted(
        feasible_candidates,
        key=lambda item: (-item["mean_fusion_psnr"], item["alpha"]),
    )[0]
    with torch.no_grad():
        model.aux_projection.weight.copy_(endpoint_weight * float(selected["alpha"]))
    selected_epoch = 1 if float(selected["alpha"]) > 0 else 0
    selected_path = checkpoint_dir / "selected.pt"
    save_checkpoint(selected_path, model, optimizer, selected_epoch, selected, config)
    policy = {
        "analysis_id": config["selection_analysis_id"],
        "selected_alpha": selected["alpha"],
        "selected_nonzero": selected_epoch > 0,
        "selected_metrics": selected,
        "candidate_count": len(candidates),
        "feasible_count": len(feasible_candidates),
        "holdout_accessed": False,
        "official_imagenette_accessed": False,
    }
    save_json(selection_output / "selected_policy.json", policy)
    summary = {
        "experiment_id": config["experiment_id"],
        "train_rows": len(train_dataset),
        "selection_rows": len(selection_dataset),
        "baseline": baseline,
        "training": training,
        "endpoint_checkpoint_sha256": sha256_file(endpoint_path),
        "selected_checkpoint_sha256": sha256_file(selected_path),
        "selected_policy_sha256": sha256_file(selection_output / "selected_policy.json"),
        "selected": selected,
        "selected_nonzero": selected_epoch > 0,
        "candidates": candidates,
        "holdout_accessed": False,
        "official_imagenette_accessed": False,
    }
    save_json(training_output / "training_and_selection_summary.json", summary)
    save_json(selection_output / "selection_summary.json", summary)
    shutil.copy2(config_path, selection_output / "config_before_checkpoint_freeze.yaml")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
