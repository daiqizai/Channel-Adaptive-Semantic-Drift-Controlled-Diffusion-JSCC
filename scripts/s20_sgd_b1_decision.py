#!/usr/bin/env python3
"""Run and aggregate the preregistered S20 B0/B1 versus SGD-JSCC decision."""

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

from cadsd_jscc.deepjscc_adapter import build_deepjscc_model  # noqa: E402
from cadsd_jscc.external_common import (  # noqa: E402
    canonical_noise_sha256,
    canonical_standard_normal,
)
from cadsd_jscc.external_rate_alignment import ExactRateMaskedDeepJSCC  # noqa: E402
from cadsd_jscc.metrics import ms_ssim_per_sample, psnr_per_sample  # noqa: E402
from cadsd_jscc.semantic_sketch import (  # noqa: E402
    integer_codes_to_bits,
    quantize_probabilities_uniform,
    reserved_symbol_indices,
)
from pc_imagenette_supervised_audit import (  # noqa: E402
    evaluate_probabilities,
    load_scratch_classifier,
)
from s5_residual_refiner_pilot import (  # noqa: E402
    build_model,
    gate_tensor,
    try_load_lpips,
)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_yaml(value: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(resolve(value).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"YAML root must be a mapping: {value}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(value: str | Path, expected: str) -> Path:
    path = resolve(value)
    if not path.is_file() or sha256_file(path) != str(expected):
        raise RuntimeError(f"missing or hash-mismatched frozen input: {path}")
    return path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def evaluator_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "imagenette": {
            "normalization_mean": config["evaluator"]["normalization_mean"],
            "normalization_std": config["evaluator"]["normalization_std"],
        }
    }


def load_population(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    population = config["population"]
    reference_path = require_sha(
        population["generated_reference_config"],
        population["generated_reference_config_sha256"],
    )
    reference = load_yaml(reference_path)
    frozen = reference["population"]
    split_path = require_sha(frozen["split_manifest"], frozen["split_manifest_sha256"])
    split = json.loads(split_path.read_text(encoding="utf-8"))
    classes = [str(value) for value in split["classes"]]
    by_id = {
        str(item["sample_id"]): item
        for item in split["samples"]
        if str(item["split"]) == str(frozen["required_split"])
    }
    source_root = resolve(split["source_train_root"])
    samples: list[dict[str, Any]] = []
    for item in frozen["samples"]:
        sample_id = str(item["sample_id"])
        source = dict(by_id[sample_id])
        path = source_root / str(source["relative_path"])
        require_sha(path, str(item["content_sha256"]))
        if int(source["class_idx"]) != int(item["class_idx"]):
            raise RuntimeError(f"class mismatch for {sample_id}")
        source["path"] = path
        samples.append(source)
    expected = int(frozen["expected_sample_count"])
    if len(samples) != expected or len({item["sample_id"] for item in samples}) != expected:
        raise RuntimeError("population size or uniqueness gate failed")
    return samples, classes


def prepare_sgd_configs(config_path: Path, config: dict[str, Any]) -> None:
    population_path = resolve(config["population"]["generated_reference_config"])
    population_dir = resolve(config["outputs"]["population"])
    for seed in map(int, config["channel"]["base_seeds"]):
        path = population_dir / f"sgd_seed_{seed}_resolved.yaml"
        if path.exists():
            raise FileExistsError(path)
        payload = {
            "phase": config["phase"],
            "study": "author_working_point_rate_alignment",
            "decision_study": config["study"],
            "analysis_id": f"{config['analysis_id']}-SEED-{seed}",
            "status": "preregistered_before_expanded_decision_output",
            "created_at": config["created_at"],
            "official_val_accessed": False,
            "outcome_claims_allowed": False,
            "pilot_claim_scope": config["claim_scope"],
            "population_reference_config": str(population_path.relative_to(ROOT)),
            "channel": {
                "type": config["channel"]["type"],
                "snrs_db": config["channel"]["snrs_db"],
                "base_seed": seed,
                "standard_normal_generator": config["channel"][
                    "standard_normal_generator"
                ],
                "seed_derivation": config["channel"]["seed_derivation"],
                "noise_variance_convention": config["channel"][
                    "noise_variance_convention"
                ],
            },
            "rate": {
                "source_real_dimensions": config["rate"]["source_real_dimensions"],
                "total_real_symbols": config["rate"]["total_real_symbols"],
                "total_complex_channel_uses": config["rate"]["total_complex_channel_uses"],
                "exact_cbr": config["rate"]["exact_cbr"],
                "released_sgd_main_real_symbols": config["rate"]["sgd_main_real_symbols"],
                "released_sgd_active_edge_real_symbols": config["rate"][
                    "sgd_active_edge_real_symbols"
                ],
            },
            "methods": {
                "sgd_jscc_paper_protocol": config["methods"]["sgd_jscc_paper_protocol"]
            },
            "outputs": {
                "root": config["outputs"]["root"],
                "deepjscc": str(
                    resolve(config["outputs"]["baseline"])
                    / "b0_full"
                    / f"seed_{seed}"
                ),
                "sgd_jscc": str(
                    resolve(config["outputs"]["sgd_jscc"]) / f"seed_{seed}"
                ),
                "overwrite_forbidden": True,
            },
        }
        path.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
    shutil.copy2(config_path, population_dir / "master_config_after_population_freeze.yaml")


def metric_tensors(
    target: torch.Tensor, candidate: torch.Tensor, lpips_model: torch.nn.Module
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        return (
            psnr_per_sample(candidate, target),
            ms_ssim_per_sample(candidate, target),
            lpips_model(candidate * 2.0 - 1.0, target * 2.0 - 1.0).flatten(),
        )


def candidate_rows(
    *,
    method: str,
    samples: list[dict[str, Any]],
    labels: torch.Tensor,
    original_prediction: torch.Tensor,
    original_confidence: torch.Tensor,
    prediction: torch.Tensor,
    metrics: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    seed: int,
    snr: float,
    noise_shas: list[str],
    runtime_ms: float,
    peak_memory_mib: float,
    image_real_symbols: int,
    payload_real_symbols: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(samples):
        label = int(labels[index])
        final_correct = int(prediction[index]) == label
        clean = int(original_prediction[index]) == label and float(original_confidence[index]) >= 0.5
        rows.append(
            {
                "method": method,
                "sample_id": item["sample_id"],
                "wnid": item["wnid"],
                "class_idx": label,
                "snr_db": snr,
                "base_seed": seed,
                "canonical_noise_sha256": noise_shas[index],
                "noise_variance_convention": "complex_awgn_per_real_half_variance",
                "total_real_symbols": 19712,
                "total_complex_channel_uses": 9856,
                "cbr": 0.050130208333333336,
                "image_real_symbols": image_real_symbols,
                "payload_real_symbols": payload_real_symbols,
                "original_prediction": int(original_prediction[index]),
                "original_confidence": float(original_confidence[index]),
                "clean_correct": clean,
                "deepjscc_prediction": int(prediction[index]),
                "deepjscc_correct": final_correct,
                "final_prediction": int(prediction[index]),
                "final_correct": final_correct,
                "final_failure": clean and not final_correct,
                "new_error_vs_deepjscc": False,
                "repair_vs_deepjscc": False,
                "deepjscc_psnr": float(metrics[0][index]),
                "deepjscc_ms_ssim": float(metrics[1][index]),
                "deepjscc_lpips": float(metrics[2][index]),
                "final_psnr": float(metrics[0][index]),
                "final_ms_ssim": float(metrics[1][index]),
                "final_lpips": float(metrics[2][index]),
                "runtime_ms_per_image": runtime_ms,
                "peak_gpu_memory_mib": peak_memory_mib,
            }
        )
    return rows


def run_baseline(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    output = resolve(config["outputs"]["baseline"])
    if output.exists():
        raise FileExistsError(output)
    samples, classes = load_population(config)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    exact = config["methods"]["exact_rate_deepjscc"]
    base = build_deepjscc_model(
        resolve(exact["baseline_repo"]),
        int(exact["inner_channel"]),
        str(config["channel"]["type"]),
        float(config["channel"]["snrs_db"][0]),
    )
    jscc = ExactRateMaskedDeepJSCC(
        base,
        dense_symbols=int(config["rate"]["dense_real_symbols"]),
        active_symbols=int(config["rate"]["total_real_symbols"]),
        snr_db=float(config["channel"]["snrs_db"][0]),
    ).to(device)
    exact_checkpoint = torch.load(
        require_sha(exact["checkpoint"], exact["checkpoint_sha256"]), map_location=device
    )
    jscc.load_state_dict(exact_checkpoint["model"], strict=True)
    jscc.eval().requires_grad_(False)

    b1_spec = config["methods"]["b1"]
    b1_config = load_yaml(b1_spec["config"])
    b1 = build_model(b1_config).to(device)
    b1_checkpoint = torch.load(
        require_sha(b1_spec["checkpoint"], b1_spec["checkpoint_sha256"]),
        map_location=device,
    )
    b1.load_state_dict(b1_checkpoint["model_state_dict"], strict=True)
    b1.eval().requires_grad_(False)

    eval_cfg = evaluator_config(config)
    evaluator_spec = config["evaluator"]
    evaluator, evaluator_temperature = load_scratch_classifier(
        str(require_sha(evaluator_spec["checkpoint"], evaluator_spec["checkpoint_sha256"])),
        classes,
        device,
        str(evaluator_spec["expected_role"]),
    )
    sender_spec = config["methods"]["payload_sender"]
    sender, sender_temperature = load_scratch_classifier(
        str(require_sha(sender_spec["checkpoint"], sender_spec["checkpoint_sha256"])),
        classes,
        device,
        "G_aux",
    )
    lpips_model, lpips_error = try_load_lpips(device, resolve("outputs/cache"))
    if lpips_model is None:
        raise RuntimeError(lpips_error)

    transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()]
    )
    target = torch.stack(
        [transform(Image.open(item["path"]).convert("RGB")) for item in samples]
    ).to(device)
    labels = torch.tensor([int(item["class_idx"]) for item in samples], device=device)
    with torch.no_grad():
        original_probability = evaluate_probabilities(
            evaluator, evaluator_temperature, target, eval_cfg
        )
        original_confidence, original_prediction = original_probability.max(dim=1)
        source_probability = evaluate_probabilities(sender, sender_temperature, target, eval_cfg)
        source_codes, _ = quantize_probabilities_uniform(
            source_probability, int(sender_spec["quantization_bits"])
        )
        source_bits = integer_codes_to_bits(
            source_codes, int(sender_spec["quantization_bits"])
        )
        payload = (
            source_bits.to(target.dtype)
            .mul(2.0)
            .sub(1.0)
            .unsqueeze(-1)
            .expand(-1, -1, int(sender_spec["repetitions"]))
            .reshape(len(samples), -1)
        )
    if not bool(
        (
            (original_prediction == labels)
            & (original_confidence >= float(evaluator_spec["clean_confidence_threshold"]))
        ).all()
    ):
        raise RuntimeError("frozen S20 population contains a non-clean T_cls sample")

    reserved = reserved_symbol_indices(
        int(config["rate"]["total_real_symbols"]),
        int(config["rate"]["payload_reserved_real_symbols"]),
        device=device,
    )
    all_rows: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output / "config_snapshot.yaml")
    for seed in map(int, config["channel"]["base_seeds"]):
        per_seed: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for snr in map(float, config["channel"]["snrs_db"]):
            noises_cpu = torch.stack(
                [
                    canonical_standard_normal(
                        seed,
                        str(item["sample_id"]),
                        snr,
                        int(config["rate"]["total_real_symbols"]),
                    )
                    for item in samples
                ]
            )
            noise_shas = [canonical_noise_sha256(row) for row in noises_cpu]
            noises = noises_cpu.to(device)
            jscc.snr_db = snr
            with torch.no_grad():
                active, dense_shape = jscc.encode_active(target)

            torch.cuda.synchronize(device)
            b0_started = time.perf_counter()
            with torch.no_grad():
                full_received = jscc.transmit_active(active, noises)
                b0_full = jscc.decode_active(full_received, dense_shape).clamp(0.0, 1.0)
            torch.cuda.synchronize(device)
            b0_runtime = (time.perf_counter() - b0_started) * 1000.0 / len(samples)

            torch.cuda.synchronize(device)
            b1_started = time.perf_counter()
            with torch.no_grad():
                transmitted = active.clone()
                transmitted[:, reserved] = payload.to(active.dtype)
                norm = transmitted.float().square().sum(dim=1, keepdim=True).clamp_min(1e-12).sqrt()
                transmitted = transmitted * math.sqrt(jscc.active_symbols) / norm.to(
                    transmitted.dtype
                )
                strict_received = jscc.transmit_active(transmitted, noises)
                strict_received[:, reserved] = 0.0
                b0_strict = jscc.decode_active(strict_received, dense_shape).clamp(0.0, 1.0)
                snr_tensor = torch.full((len(samples),), snr, device=device)
                snr_norm = snr_tensor / float(b1_config["model"]["snr_norm_max"])
                b1_output = b1(
                    b0_strict,
                    snr_norm,
                    gate_tensor(b1_config, snr_tensor, device),
                )
            torch.cuda.synchronize(device)
            b1_runtime = (time.perf_counter() - b1_started) * 1000.0 / len(samples)

            peak = torch.cuda.max_memory_allocated(device) / (1024**2)
            for method, candidate, runtime, image_symbols, payload_symbols in (
                ("b0_full", b0_full, b0_runtime, 19712, 0),
                ("b0_strict", b0_strict, b1_runtime, 19632, 80),
                ("b1", b1_output, b1_runtime, 19632, 80),
            ):
                with torch.no_grad():
                    probability = evaluate_probabilities(
                        evaluator, evaluator_temperature, candidate, eval_cfg
                    )
                    prediction = probability.argmax(dim=1)
                    metrics = metric_tensors(target, candidate, lpips_model)
                rows = candidate_rows(
                    method=method,
                    samples=samples,
                    labels=labels,
                    original_prediction=original_prediction,
                    original_confidence=original_confidence,
                    prediction=prediction,
                    metrics=metrics,
                    seed=seed,
                    snr=snr,
                    noise_shas=noise_shas,
                    runtime_ms=runtime,
                    peak_memory_mib=peak,
                    image_real_symbols=image_symbols,
                    payload_real_symbols=payload_symbols,
                )
                per_seed[method].extend(rows)
                all_rows[method].extend(rows)
            grid = torch.cat(
                [target[:8], b0_full[:8], b0_strict[:8], b1_output[:8]], dim=0
            ).cpu()
            save_image(
                grid,
                output / f"seed_{seed}_snr_{int(snr):02d}_source_b0full_b0strict_b1.png",
                nrow=8,
            )
        for method, rows in per_seed.items():
            method_output = output / method / f"seed_{seed}"
            method_output.mkdir(parents=True, exist_ok=False)
            write_csv(method_output / "per_sample.csv", rows)
            (method_output / "summary.json").write_text(
                json.dumps(summarize(rows), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    summary = {method: summarize(rows) for method, rows in all_rows.items()}
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def summarize(rows: list[dict[str, Any] | dict[str, str]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "images": len({str(row["sample_id"]) for row in rows}),
        "channel_seeds": sorted({int(row["base_seed"]) for row in rows}),
        "snrs_db": sorted({float(row["snr_db"]) for row in rows}),
        "mean_psnr": float(np.mean([float(row["final_psnr"]) for row in rows])),
        "mean_ms_ssim": float(np.mean([float(row["final_ms_ssim"]) for row in rows])),
        "mean_lpips": float(np.mean([float(row["final_lpips"]) for row in rows])),
        "final_failures": sum(as_bool(row["final_failure"]) for row in rows),
        "mean_runtime_ms_per_image": float(
            np.mean([float(row["runtime_ms_per_image"]) for row in rows])
        ),
        "peak_gpu_memory_mib": max(float(row["peak_gpu_memory_mib"]) for row in rows),
    }


def keyed(rows: list[dict[str, str]]) -> dict[tuple[int, str, float], dict[str, str]]:
    result = {
        (int(row["base_seed"]), str(row["sample_id"]), float(row["snr_db"])): row
        for row in rows
    }
    if len(result) != len(rows):
        raise RuntimeError("duplicate seed/sample/SNR keys")
    return result


def paired_summary(
    left: dict[tuple[int, str, float], dict[str, str]],
    right: dict[tuple[int, str, float], dict[str, str]],
    config: dict[str, Any],
) -> dict[str, Any]:
    if set(left) != set(right):
        raise RuntimeError("paired method key sets differ")
    for key in left:
        if left[key]["canonical_noise_sha256"] != right[key]["canonical_noise_sha256"]:
            raise RuntimeError(f"canonical noise mismatch: {key}")
    metrics = {
        "psnr": "final_psnr",
        "ms_ssim": "final_ms_ssim",
        "lpips": "final_lpips",
    }
    by_image: defaultdict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    for key in sorted(left):
        sample_id = key[1]
        by_image[sample_id].append(
            (
                float(left[key]["final_psnr"]) - float(right[key]["final_psnr"]),
                float(left[key]["final_ms_ssim"]) - float(right[key]["final_ms_ssim"]),
                float(left[key]["final_lpips"]) - float(right[key]["final_lpips"]),
                float(as_bool(left[key]["final_failure"]))
                - float(as_bool(right[key]["final_failure"])),
            )
        )
    image_ids = sorted(by_image)
    cluster = np.asarray(
        [np.asarray(by_image[sample_id], dtype=np.float64).mean(axis=0) for sample_id in image_ids]
    )
    observed = cluster.mean(axis=0)
    rng = np.random.default_rng(int(config["metrics"]["bootstrap_seed"]))
    replicates = int(config["metrics"]["bootstrap_replicates"])
    sampled = rng.integers(0, len(cluster), size=(replicates, len(cluster)))
    boot = cluster[sampled].mean(axis=1)
    ci = np.quantile(boot, [0.025, 0.975], axis=0)
    all_left = [left[key] for key in sorted(left)]
    all_right = [right[key] for key in sorted(right)]
    result: dict[str, Any] = {
        "left_minus_right": {
            "mean_psnr": float(observed[0]),
            "mean_ms_ssim": float(observed[1]),
            "mean_lpips": float(observed[2]),
            "mean_failure_rate": float(observed[3]),
            "psnr_ci95": [float(ci[0, 0]), float(ci[1, 0])],
            "ms_ssim_ci95": [float(ci[0, 1]), float(ci[1, 1])],
            "lpips_ci95": [float(ci[0, 2]), float(ci[1, 2])],
            "failure_rate_ci95": [float(ci[0, 3]), float(ci[1, 3])],
        },
        "left_failures": sum(as_bool(row["final_failure"]) for row in all_left),
        "right_failures": sum(as_bool(row["final_failure"]) for row in all_right),
        "left_new_errors_vs_right": sum(
            as_bool(right[key]["final_correct"]) and not as_bool(left[key]["final_correct"])
            for key in left
        ),
        "left_repairs_vs_right": sum(
            not as_bool(right[key]["final_correct"]) and as_bool(left[key]["final_correct"])
            for key in left
        ),
        "by_snr": {},
    }
    for snr in map(float, config["channel"]["snrs_db"]):
        keys = [key for key in left if key[2] == snr]
        per_snr = {
            f"mean_{name}_delta": float(
                np.mean([float(left[key][field]) - float(right[key][field]) for key in keys])
            )
            for name, field in metrics.items()
        }
        per_snr.update(
            {
                "mean_failure_rate_delta": float(
                    np.mean(
                        [
                            float(as_bool(left[key]["final_failure"]))
                            - float(as_bool(right[key]["final_failure"]))
                            for key in keys
                        ]
                    )
                ),
                "left_failures": sum(
                    as_bool(left[key]["final_failure"]) for key in keys
                ),
                "right_failures": sum(
                    as_bool(right[key]["final_failure"]) for key in keys
                ),
            }
        )
        result["by_snr"][str(int(snr))] = per_snr
    return result


def run_aggregate(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    output = resolve(config["outputs"]["aggregate"])
    if output.exists():
        raise FileExistsError(output)
    baseline = resolve(config["outputs"]["baseline"])
    sgd_root = resolve(config["outputs"]["sgd_jscc"])
    method_rows: dict[str, list[dict[str, str]]] = {name: [] for name in ("b0_full", "b0_strict", "b1", "sgd")}
    for seed in map(int, config["channel"]["base_seeds"]):
        for method in ("b0_full", "b0_strict", "b1"):
            method_rows[method].extend(
                read_csv(baseline / method / f"seed_{seed}" / "per_sample.csv")
            )
        method_rows["sgd"].extend(
            read_csv(sgd_root / f"seed_{seed}" / "per_sample.csv")
        )
    expected = (
        int(config["population"]["expected_sample_count"])
        * len(config["channel"]["base_seeds"])
        * len(config["channel"]["snrs_db"])
    )
    if any(len(rows) != expected for rows in method_rows.values()):
        raise RuntimeError({name: len(rows) for name, rows in method_rows.items()})
    maps = {name: keyed(rows) for name, rows in method_rows.items()}
    sgd_vs_b1 = paired_summary(maps["sgd"], maps["b1"], config)
    sgd_vs_b0 = paired_summary(maps["sgd"], maps["b0_full"], config)
    b1_vs_b0_strict = paired_summary(maps["b1"], maps["b0_strict"], config)

    delta = sgd_vs_b1["left_minus_right"]
    paper_upper_quality_dominates = (
        float(delta["psnr_ci95"][0]) > 0.0
        and float(delta["lpips_ci95"][1]) < 0.0
        and int(sgd_vs_b1["left_failures"]) <= int(sgd_vs_b1["right_failures"])
    )
    packet_bits = int(config["rate"]["sgd_caption_packet_bits_per_patch"])
    patches = int(config["rate"]["sgd_caption_patches_per_image"])
    minimum_text_symbols = packet_bits * patches
    image_symbols = int(config["rate"]["sgd_main_real_symbols"]) + int(
        config["rate"]["sgd_active_edge_real_symbols"]
    )
    strict_executable = image_symbols + minimum_text_symbols <= int(
        config["rate"]["total_real_symbols"]
    )
    result = {
        "analysis_id": config["analysis_id"],
        "status": "PASS",
        "claim_scope": config["claim_scope"],
        "population": {
            "images": config["population"]["expected_sample_count"],
            "channel_seeds": config["channel"]["base_seeds"],
            "snrs_db": config["channel"]["snrs_db"],
            "rows_per_method": expected,
            "official_val_accessed": False,
        },
        "methods": {name: summarize(rows) for name, rows in method_rows.items()},
        "paired": {
            "sgd_paper_upper_minus_b1": sgd_vs_b1,
            "sgd_paper_upper_minus_b0_full": sgd_vs_b0,
            "b1_minus_b0_strict": b1_vs_b0_strict,
        },
        "strict_rate_audit": {
            "budget_real_symbols": int(config["rate"]["total_real_symbols"]),
            "released_main_plus_edge_real_symbols": image_symbols,
            "minimum_unprotected_caption_real_symbols": minimum_text_symbols,
            "minimum_total_with_caption": image_symbols + minimum_text_symbols,
            "minimum_overrun_real_symbols": image_symbols
            + minimum_text_symbols
            - int(config["rate"]["total_real_symbols"]),
            "released_weight_strict_caption_protocol_executable": strict_executable,
            "interpretation": "requires_reallocating_or_retraining_main_or_edge_to_meter_text",
        },
        "decision": {
            "paper_free_text_upper_bound_quality_and_semantic_dominance": paper_upper_quality_dominates,
            "strict_full_sgd_route_supported": paper_upper_quality_dominates
            and strict_executable,
            "fusion_or_anchor_necessity": (
                "not_established_if_paper_upper_bound_dominates_but_strict_rate_is_unresolved"
                if paper_upper_quality_dominates
                else "retained_because_sgd_does_not_dominate_b1_under_the_favorable_upper_bound"
            ),
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output / "config_snapshot.yaml")
    (output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/s20_sgd_b1_decision.yaml"
    )
    parser.add_argument(
        "--mode", choices=("prepare-sgd-configs", "baseline", "aggregate"), required=True
    )
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = load_yaml(config_path)
    if config.get("status") != "preregistered_before_any_s20_outcome":
        raise RuntimeError("S20 config status changed")
    load_population(config)
    if args.mode == "prepare-sgd-configs":
        prepare_sgd_configs(config_path, config)
    elif args.mode == "baseline":
        summary = run_baseline(config_path, config)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        run_aggregate(config_path, config)


if __name__ == "__main__":
    main()
