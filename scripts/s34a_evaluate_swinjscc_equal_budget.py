#!/usr/bin/env python3
"""Evaluate frozen S34A equal-budget arms against frozen S33 on policy-dev."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from cadsd_jscc.external_common import (  # noqa: E402
    canonical_noise_sha256,
    canonical_standard_normal,
)
from cadsd_jscc.metrics import ms_ssim_per_sample, psnr_per_sample  # noqa: E402
from cadsd_jscc.swinjscc_adapter import (  # noqa: E402
    OfficialSwinJSCCSA,
    trainable_parameter_count,
)
from pc_imagenette_supervised_audit import (  # noqa: E402
    evaluate_probabilities,
    load_scratch_classifier,
)
from s5_residual_refiner_pilot import try_load_lpips  # noqa: E402


SCRIPT = Path(__file__).resolve()
ARMS = ("official_base_sa", "capacity_matched_sa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/s34a_swinjscc_equal_budget_evaluation.yaml"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


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


def require_sha(value: str | Path, expected: str) -> Path:
    path = resolve(value)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != str(expected):
        raise RuntimeError(f"SHA mismatch for {path}: {actual} != {expected}")
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "1.0"}


def load_population(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    population_path = require_sha(
        config["inputs"]["population_reference"],
        config["inputs"]["population_reference_sha256"],
    )
    population_reference = yaml.safe_load(population_path.read_text(encoding="utf-8"))
    population = population_reference["population"]
    manifest_path = require_sha(
        config["inputs"]["split_manifest"], config["inputs"]["split_manifest_sha256"]
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    classes = [str(value) for value in manifest["classes"]]
    by_id = {
        str(item["sample_id"]): item
        for item in manifest["samples"]
        if str(item["split"]) == str(population["required_split"])
    }
    source_root = resolve(manifest["source_train_root"])
    result: list[dict[str, Any]] = []
    for frozen in population["samples"]:
        item = dict(by_id[str(frozen["sample_id"])])
        path = source_root / str(item["relative_path"])
        require_sha(path, str(frozen["content_sha256"]))
        if int(item["class_idx"]) != int(frozen["class_idx"]):
            raise RuntimeError("population class changed")
        item["path"] = path
        result.append(item)
    if len(result) != int(config["population"]["expected_sample_count"]):
        raise RuntimeError("population size changed")
    return result, classes


def build_model(checkpoint: dict[str, Any], expected_arm: str) -> OfficialSwinJSCCSA:
    if checkpoint["arm"] != expected_arm:
        raise RuntimeError(f"checkpoint arm mismatch for {expected_arm}")
    if int(checkpoint["epoch_number_1based"]) > 12:
        raise RuntimeError("extension checkpoint is forbidden in equal-budget evaluation")
    train_config = checkpoint["config"]
    arm = train_config["arms_confirmed"][expected_arm]
    model = OfficialSwinJSCCSA(
        image_size=256,
        latent_channels=64,
        encoder_depths=tuple(int(value) for value in arm["encoder_depths"]),
        decoder_depths=tuple(int(value) for value in arm["decoder_depths"]),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def cluster_ci(
    rows: list[dict[str, Any]], field: str, replicates: int, seed: int
) -> list[float]:
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sample_id"])].append(float(row[field]))
    values = np.asarray(
        [np.mean(grouped[key]) for key in sorted(grouped)], dtype=np.float64
    )
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(0, len(values), size=(replicates, len(values)))
    bootstrap = values[indices].mean(axis=1)
    return [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))]


def metric_summary(
    rows: list[dict[str, Any]], config: dict[str, Any], seed_offset: int
) -> dict[str, Any]:
    replicates = int(config["metrics"]["bootstrap_replicates"])
    base_seed = int(config["metrics"]["bootstrap_seed"]) + seed_offset
    metrics = ("psnr", "ms_ssim", "lpips", "failure")
    means: dict[str, Any] = {}
    intervals: dict[str, Any] = {}
    for method in ("s33", "swin"):
        for metric in metrics:
            field = f"{method}_{metric}"
            if metric == "failure":
                for row in rows:
                    row[field] = float(as_bool(row[field]))
            means[field] = float(np.mean([float(row[field]) for row in rows]))
            intervals[field] = cluster_ci(rows, field, replicates, base_seed)
    for metric in metrics:
        field = f"s33_minus_swin_{metric}"
        means[field] = float(np.mean([float(row[field]) for row in rows]))
        intervals[field] = cluster_ci(rows, field, replicates, base_seed)
    new_error_rows = sum(
        as_bool(row["s33_failure"]) and not as_bool(row["swin_failure"]) for row in rows
    )
    repair_rows = sum(
        not as_bool(row["s33_failure"]) and as_bool(row["swin_failure"]) for row in rows
    )
    return {
        "rows": len(rows),
        "source_clusters": len({str(row["sample_id"]) for row in rows}),
        "means": means,
        "source_image_cluster_95ci": intervals,
        "s33_vs_swin_semantic_transitions": {
            "s33_new_error_rows": int(new_error_rows),
            "s33_repair_rows": int(repair_rows),
        },
    }


def verdict(summary: dict[str, Any], margin: float) -> dict[str, Any]:
    intervals = summary["source_image_cluster_95ci"]
    psnr_lower = float(intervals["s33_minus_swin_psnr"][0])
    if psnr_lower > 0.0:
        psnr_verdict = "S33_SIGNIFICANTLY_EXCEEDS_SWIN"
    elif psnr_lower > -margin:
        psnr_verdict = "S33_NONINFERIOR_TIED_WITH_SWIN"
    elif psnr_lower < -margin:
        psnr_verdict = "S33_INFERIOR_TO_SWIN"
    else:
        psnr_verdict = "EXACT_MARGIN_BOUNDARY_INCONCLUSIVE"
    secondary_conflicts: list[str] = []
    if intervals["s33_minus_swin_ms_ssim"][1] < 0:
        secondary_conflicts.append("MS_SSIM_SIGNIFICANTLY_FAVORS_SWIN")
    if intervals["s33_minus_swin_lpips"][0] > 0:
        secondary_conflicts.append("LPIPS_SIGNIFICANTLY_FAVORS_SWIN")
    if intervals["s33_minus_swin_failure"][0] > 0:
        secondary_conflicts.append("SEMANTIC_FAILURE_SIGNIFICANTLY_FAVORS_SWIN")
    return {
        "psnr_margin_verdict": psnr_verdict,
        "noninferiority_margin_db": margin,
        "secondary_conflicts": secondary_conflicts,
        "overall_wording": "PARETO" if secondary_conflicts else psnr_verdict,
    }


def summarize(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    margin = float(config["claim_rule"]["noninferiority_margin_db"])
    by_arm: dict[str, Any] = {}
    ranking = {
        "S33_SIGNIFICANTLY_EXCEEDS_SWIN": 0,
        "S33_NONINFERIOR_TIED_WITH_SWIN": 1,
        "EXACT_MARGIN_BOUNDARY_INCONCLUSIVE": 2,
        "S33_INFERIOR_TO_SWIN": 3,
    }
    for arm_index, arm in enumerate(ARMS):
        selected = [row for row in rows if row["arm"] == arm]
        aggregate = metric_summary(selected, config, 1000 * arm_index)
        aggregate["verdict"] = verdict(aggregate, margin)
        per_snr: dict[str, Any] = {}
        for snr_index, snr in enumerate(config["population"]["snrs_db"]):
            snr_rows = [row for row in selected if float(row["snr_db"]) == float(snr)]
            value = metric_summary(snr_rows, config, 1000 * arm_index + snr_index + 1)
            value["verdict"] = verdict(value, margin)
            per_snr[str(int(snr))] = value
        by_arm[arm] = {"aggregate": aggregate, "per_snr": per_snr}
    arm_verdicts = {
        arm: by_arm[arm]["aggregate"]["verdict"]["psnr_margin_verdict"] for arm in ARMS
    }
    conservative_arm = max(ARMS, key=lambda arm: ranking[arm_verdicts[arm]])
    any_secondary_conflict = any(
        by_arm[arm]["aggregate"]["verdict"]["secondary_conflicts"] for arm in ARMS
    )
    return {
        "status": "PASS",
        "analysis_id": config["analysis_id"],
        "rows": len(rows),
        "by_arm": by_arm,
        "dual_arm_conservative_verdict": {
            "decisive_arm": conservative_arm,
            "psnr_margin_verdict": arm_verdicts[conservative_arm],
            "overall_wording": "PARETO" if any_secondary_conflict else arm_verdicts[conservative_arm],
            "rule": "take_the_arm_more_adverse_to_S33",
        },
        "claim_scope": "known_policy_dev_not_independent_final_test",
        "official_imagenette_validation_accessed": False,
    }


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["protocol"]["status"] != "freeze_from_coco_val_training_summaries_before_policy_dev_access":
        raise RuntimeError("checkpoint-freeze protocol changed")
    if config["protocol"]["extension_checkpoint_used"] is not False:
        raise RuntimeError("extension checkpoints are forbidden")
    if config["protocol"]["official_imagenette_validation_accessed"] is not False:
        raise RuntimeError("official Imagenette validation must remain sealed")
    if config["metrics"]["primary_quantization"] != "floor_uint8":
        raise RuntimeError("primary quantization contract changed")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("S34A evaluation requires CUDA")

    checkpoint_paths: dict[str, Path] = {}
    checkpoints: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        entry = config["inputs"]["swin_checkpoints"][arm]
        summary_path = resolve(entry["training_summary"])
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        training_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if training_summary["status"] != "complete_equal_budget_only":
            raise RuntimeError(f"equal-budget training is incomplete for {arm}")
        if training_summary["arm"] != arm or int(training_summary["epochs_completed"]) != 12:
            raise RuntimeError(f"training-summary arm/epoch ledger mismatch for {arm}")
        if training_summary["extension_executed"] is not False:
            raise RuntimeError(f"extension checkpoint detected for {arm}")
        path = resolve(entry["path"])
        if relative(path) != str(training_summary["best_checkpoint"]):
            raise RuntimeError(f"training-selected checkpoint path mismatch for {arm}")
        path = require_sha(path, training_summary["best_checkpoint_sha256"])
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if int(checkpoint["epoch_number_1based"]) != int(training_summary["best_epoch"]):
            raise RuntimeError(f"best epoch mismatch for {arm}")
        if checkpoint["phase"] != "continuation":
            raise RuntimeError(f"{arm} final checkpoint is not from the 4+8 continuation")
        checkpoint_paths[arm] = path
        checkpoints[arm] = checkpoint

    s33_path = require_sha(
        config["inputs"]["s33_per_sample"], config["inputs"]["s33_per_sample_sha256"]
    )
    s33_rows = read_csv(s33_path)
    s33_by_key = {
        (str(row["sample_id"]), int(row["base_seed"]), float(row["snr_db"])): row
        for row in s33_rows
    }
    expected_per_arm = (
        int(config["population"]["expected_sample_count"])
        * len(config["population"]["channel_seeds"])
        * len(config["population"]["snrs_db"])
    )
    if len(s33_by_key) != expected_per_arm:
        raise RuntimeError("frozen S33 key count changed")

    samples, classes = load_population(config)
    evaluator, temperature = load_scratch_classifier(
        str(require_sha(config["inputs"]["t_cls_checkpoint"], config["inputs"]["t_cls_checkpoint_sha256"])),
        classes,
        device,
        str(config["evaluator"]["expected_role"]),
    )
    lpips_model, lpips_error = try_load_lpips(device, resolve("outputs/cache"))
    if lpips_model is None:
        raise RuntimeError(lpips_error)
    lpips_model.eval().requires_grad_(False)
    transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()]
    )
    targets = torch.stack(
        [transform(Image.open(item["path"]).convert("RGB")) for item in samples]
    ).to(device)
    eval_config = {
        "imagenette": {
            "normalization_mean": config["evaluator"]["normalization_mean"],
            "normalization_std": config["evaluator"]["normalization_std"],
        }
    }
    with torch.inference_mode():
        source_probabilities = evaluate_probabilities(
            evaluator, temperature, targets, eval_config
        )
    source_confidence, source_prediction = source_probabilities.max(dim=1)
    for index, item in enumerate(samples):
        if int(source_prediction[index]) != int(item["class_idx"]) or float(source_confidence[index]) < float(
            config["evaluator"]["clean_confidence_threshold"]
        ):
            raise RuntimeError(f"source is no longer clean-correct: {item['sample_id']}")

    preflight = {
        "status": "PASS",
        "checkpoints": {
            arm: {
                "path": relative(checkpoint_paths[arm]),
                "sha256": sha256_file(checkpoint_paths[arm]),
                "best_epoch": int(checkpoints[arm]["epoch_number_1based"]),
                "parameters": int(config["models"][arm]["trainable_parameters"]),
            }
            for arm in ARMS
        },
        "s33_rows": len(s33_rows),
        "population_samples": len(samples),
        "expected_rows_total": 2 * expected_per_arm,
        "extension_checkpoint_used": False,
        "official_imagenette_validation_accessed": False,
    }
    if args.preflight:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return

    output = resolve(config["outputs"]["directory"])
    if output.exists() and not args.resume:
        raise FileExistsError(output)
    if not output.exists() and args.resume:
        raise FileNotFoundError(output)
    if not output.exists():
        output.mkdir(parents=True)
        (output / "images").mkdir()
        shutil.copy2(config_path, output / "config_snapshot.yaml")
        shutil.copy2(SCRIPT, output / SCRIPT.name)
    elif sha256_file(output / "config_snapshot.yaml") != sha256_file(config_path):
        raise RuntimeError("resume config differs from output snapshot")

    all_rows: list[dict[str, Any]] = []
    for arm_index, arm in enumerate(ARMS):
        arm_csv = output / f"per_sample_{arm}.csv"
        if args.resume and arm_csv.is_file():
            completed = read_csv(arm_csv)
            if len(completed) != expected_per_arm:
                raise RuntimeError(f"incomplete saved arm CSV: {arm}")
            all_rows.extend(completed)
            continue
        model = build_model(checkpoints[arm], arm).to(device).eval().requires_grad_(False)
        expected_parameters = int(config["models"][arm]["trainable_parameters"])
        if trainable_parameter_count(model) != expected_parameters or model.real_symbols != 16384:
            raise RuntimeError(f"model ledger mismatch for {arm}")
        arm_rows: list[dict[str, Any]] = []
        batch_size = int(config["runtime"]["batch_size"])
        for base_seed in map(int, config["population"]["channel_seeds"]):
            for snr in map(float, config["population"]["snrs_db"]):
                for start in range(0, len(samples), batch_size):
                    end = min(start + batch_size, len(samples))
                    target = targets[start:end]
                    batch_items = samples[start:end]
                    old_rows = [
                        s33_by_key[(str(item["sample_id"]), base_seed, snr)] for item in batch_items
                    ]
                    noises = []
                    for item, old in zip(batch_items, old_rows):
                        reference = canonical_standard_normal(
                            base_seed,
                            str(item["sample_id"]),
                            snr,
                            int(config["rate"]["canonical_noise_reference_real_symbols"]),
                        )
                        if canonical_noise_sha256(reference) != old["canonical_noise_sha256"]:
                            raise RuntimeError("canonical full-noise SHA mismatch")
                        prefix = reference[:16384].contiguous()
                        if canonical_noise_sha256(prefix) != old["strong_noise_prefix_sha256"]:
                            raise RuntimeError("canonical 16,384-D prefix SHA mismatch")
                        noises.append(prefix.reshape(256, 64))
                    noise_batch = torch.stack(noises).to(device)
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    with torch.inference_mode():
                        swin_float, observation = model.forward_with_observation(target, snr, noise_batch)
                        swin_float = swin_float.clamp(0.0, 1.0)
                    torch.cuda.synchronize(device)
                    runtime_ms = (time.perf_counter() - started) * 1000.0 / len(target)
                    swin = torch.floor(swin_float * 255.0).clamp(0.0, 255.0) / 255.0
                    with torch.inference_mode():
                        psnr = psnr_per_sample(swin, target)
                        ms_ssim = ms_ssim_per_sample(swin, target)
                        lpips = lpips_model(swin * 2.0 - 1.0, target * 2.0 - 1.0).flatten()
                        prediction = evaluate_probabilities(
                            evaluator, temperature, swin, eval_config
                        ).argmax(dim=1)
                    for offset, (item, old) in enumerate(zip(batch_items, old_rows)):
                        label = int(item["class_idx"])
                        swin_failure = int(prediction[offset]) != label
                        s33_failure = as_bool(old["strong_failure"])
                        row = {
                            "arm": arm,
                            "sample_id": str(item["sample_id"]),
                            "wnid": item["wnid"],
                            "class_idx": label,
                            "base_seed": base_seed,
                            "snr_db": snr,
                            "canonical_noise_sha256": old["canonical_noise_sha256"],
                            "noise_prefix_sha256": old["strong_noise_prefix_sha256"],
                            "s33_prediction": int(old["strong_prediction"]),
                            "s33_failure": s33_failure,
                            "s33_psnr": float(old["strong_psnr"]),
                            "s33_ms_ssim": float(old["strong_ms_ssim"]),
                            "s33_lpips": float(old["strong_lpips"]),
                            "swin_prediction": int(prediction[offset]),
                            "swin_failure": swin_failure,
                            "swin_psnr": float(psnr[offset]),
                            "swin_ms_ssim": float(ms_ssim[offset]),
                            "swin_lpips": float(lpips[offset]),
                            "s33_minus_swin_failure": float(s33_failure) - float(swin_failure),
                            "s33_minus_swin_psnr": float(old["strong_psnr"]) - float(psnr[offset]),
                            "s33_minus_swin_ms_ssim": float(old["strong_ms_ssim"]) - float(ms_ssim[offset]),
                            "s33_minus_swin_lpips": float(old["strong_lpips"]) - float(lpips[offset]),
                            "swin_normalized_power": float(observation.normalized_power[offset]),
                            "swin_runtime_ms": runtime_ms,
                        }
                        arm_rows.append(row)
                    if start == 0:
                        save_image(
                            torch.cat((target[:4].cpu(), swin[:4].cpu())),
                            output / "images" / f"{arm}_seed_{base_seed}_snr_{int(snr):02d}.png",
                            nrow=min(4, len(target)),
                        )
                write_json(
                    output / "STATE.json",
                    {"status": "running", "arm": arm, "completed_seed": base_seed, "completed_snr": snr},
                )
        if len(arm_rows) != expected_per_arm:
            raise RuntimeError(f"row count mismatch for {arm}")
        power_error = max(abs(float(row["swin_normalized_power"]) - 1.0) for row in arm_rows)
        if power_error > float(config["rate"]["normalized_power_abs_error_max"]):
            raise RuntimeError(f"normalized-power audit failed for {arm}")
        write_csv(arm_csv, arm_rows)
        all_rows.extend(arm_rows)
        del model
        torch.cuda.empty_cache()

    if len(all_rows) != 2 * expected_per_arm:
        raise RuntimeError("dual-arm total row count mismatch")
    for row in all_rows:
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise RuntimeError(f"non-finite {key}")
    write_csv(output / "per_sample.csv", all_rows)
    summary = summarize(config, all_rows)
    summary["audit"] = {
        "config": relative(config_path),
        "config_sha256": sha256_file(config_path),
        "script": relative(SCRIPT),
        "script_sha256": sha256_file(SCRIPT),
        "s33_per_sample": relative(s33_path),
        "s33_per_sample_sha256": sha256_file(s33_path),
        "checkpoint_sha256": {arm: sha256_file(checkpoint_paths[arm]) for arm in ARMS},
        "rows_per_arm": expected_per_arm,
        "official_imagenette_validation_accessed": False,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "STATE.json", {"status": "complete", "extension_executed": False})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
