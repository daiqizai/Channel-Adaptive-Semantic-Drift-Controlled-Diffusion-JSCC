#!/usr/bin/env python3
"""Reproduce frozen S20 B1 with the original full-population batch schedule."""

from __future__ import annotations

import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from cadsd_jscc.deepjscc_adapter import build_deepjscc_model  # noqa: E402
from cadsd_jscc.external_common import (  # noqa: E402
    canonical_noise_sha256,
    canonical_standard_normal,
)
from cadsd_jscc.external_rate_alignment import ExactRateMaskedDeepJSCC  # noqa: E402
from cadsd_jscc.semantic_sketch import (  # noqa: E402
    integer_codes_to_bits,
    quantize_probabilities_uniform,
    reserved_symbol_indices,
)
from pc_imagenette_supervised_audit import (  # noqa: E402
    evaluate_probabilities,
    load_scratch_classifier,
)
from s20_sgd_b1_decision import (  # noqa: E402
    as_bool,
    candidate_rows,
    evaluator_config,
    keyed,
    load_population,
    load_yaml,
    metric_tensors,
    read_csv,
)
from s21_b1_anchored_gated_fusion import (  # noqa: E402
    load_config,
    resolve,
    save_json,
    sha256_file,
    write_csv,
)
from s5_residual_refiner_pilot import build_model, gate_tensor, try_load_lpips  # noqa: E402


def require_sha(value: str | Path, expected: str) -> Path:
    path = resolve(value)
    if not path.is_file() or sha256_file(path) != str(expected):
        raise RuntimeError(f"missing or hash-mismatched input: {path}")
    return path


def validate(config: dict[str, Any]) -> dict[str, Any]:
    protocol = config["protocol"]
    if protocol["status"] != "registered_after_s28_numeric_reproduction_failure_before_s29_output":
        raise RuntimeError("S29 is not in its executable registered state")
    for key in (
        "s20_and_s28_outcomes_known",
        "diagnostic_only_not_independent_replication",
        "reproduce_original_s20_batch_schedule",
        "no_model_or_method_change",
        "no_threshold_change",
    ):
        if protocol.get(key) is not True:
            raise RuntimeError(f"S29 protocol boundary changed: {key}")
    if protocol.get("official_imagenette_validation_accessed") is not False:
        raise RuntimeError("official validation access is forbidden")
    for key, hash_key in (
        ("s20_config", "s20_config_sha256"),
        ("s28_config", "s28_config_sha256"),
        ("s28_summary", "s28_summary_sha256"),
    ):
        require_sha(config["inputs"][key], config["inputs"][hash_key])
    for item in config["inputs"]["frozen_b1_csvs"]:
        require_sha(item["path"], item["sha256"])
    if int(config["audit"]["batch_size"]) != int(config["audit"]["expected_images"]):
        raise RuntimeError("S29 no longer reproduces the original S20 batch schedule")
    return load_yaml(config["inputs"]["s20_config"])


@torch.no_grad()
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/s29_s28_b1_exact_batch_audit.yaml")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = load_config(config_path)
    s20 = validate(config)
    output = resolve(config["outputs"]["directory"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    shutil.copy2(config_path, output / "config_before_output.yaml")
    shutil.copy2(SCRIPT, output / SCRIPT.name)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    samples, classes = load_population(s20)
    if len(samples) != int(config["audit"]["expected_images"]):
        raise RuntimeError("S20 population size changed")

    exact = s20["methods"]["exact_rate_deepjscc"]
    base = build_deepjscc_model(
        resolve(exact["baseline_repo"]),
        int(exact["inner_channel"]),
        str(s20["channel"]["type"]),
        float(s20["channel"]["snrs_db"][0]),
    )
    jscc = ExactRateMaskedDeepJSCC(
        base,
        dense_symbols=int(s20["rate"]["dense_real_symbols"]),
        active_symbols=int(s20["rate"]["total_real_symbols"]),
        snr_db=float(s20["channel"]["snrs_db"][0]),
    ).to(device)
    checkpoint = torch.load(
        require_sha(exact["checkpoint"], exact["checkpoint_sha256"]), map_location=device
    )
    jscc.load_state_dict(checkpoint["model"], strict=True)
    jscc.eval().requires_grad_(False)

    b1_spec = s20["methods"]["b1"]
    b1_config = load_yaml(b1_spec["config"])
    b1 = build_model(b1_config).to(device)
    b1_checkpoint = torch.load(
        require_sha(b1_spec["checkpoint"], b1_spec["checkpoint_sha256"]), map_location=device
    )
    b1.load_state_dict(b1_checkpoint["model_state_dict"], strict=True)
    b1.eval().requires_grad_(False)

    evaluator_spec = s20["evaluator"]
    evaluator, evaluator_temperature = load_scratch_classifier(
        str(require_sha(evaluator_spec["checkpoint"], evaluator_spec["checkpoint_sha256"])),
        classes,
        device,
        "T_cls",
    )
    sender_spec = s20["methods"]["payload_sender"]
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
    eval_cfg = evaluator_config(s20)
    original_probability = evaluate_probabilities(
        evaluator, evaluator_temperature, target, eval_cfg
    )
    original_confidence, original_prediction = original_probability.max(dim=1)
    source_probability = evaluate_probabilities(sender, sender_temperature, target, eval_cfg)
    source_codes, _ = quantize_probabilities_uniform(
        source_probability, int(sender_spec["quantization_bits"])
    )
    source_bits = integer_codes_to_bits(source_codes, int(sender_spec["quantization_bits"]))
    payload = (
        source_bits.to(target.dtype)
        .mul(2.0)
        .sub(1.0)
        .unsqueeze(-1)
        .expand(-1, -1, int(sender_spec["repetitions"]))
        .reshape(len(samples), -1)
    )
    reserved = reserved_symbol_indices(
        int(s20["rate"]["total_real_symbols"]),
        int(s20["rate"]["payload_reserved_real_symbols"]),
        device=device,
    )

    generated_rows: list[dict[str, Any]] = []
    for seed in map(int, s20["channel"]["base_seeds"]):
        for snr in map(float, s20["channel"]["snrs_db"]):
            noises_cpu = torch.stack(
                [
                    canonical_standard_normal(
                        seed,
                        str(item["sample_id"]),
                        snr,
                        int(s20["rate"]["total_real_symbols"]),
                    )
                    for item in samples
                ]
            )
            noise_shas = [canonical_noise_sha256(row) for row in noises_cpu]
            active, dense_shape = jscc.encode_active(target)
            transmitted = active.clone()
            transmitted[:, reserved] = payload.to(active.dtype)
            norm = transmitted.float().square().sum(dim=1, keepdim=True).clamp_min(1e-12).sqrt()
            transmitted = transmitted * math.sqrt(jscc.active_symbols) / norm.to(transmitted.dtype)
            jscc.snr_db = snr
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            received = jscc.transmit_active(transmitted, noises_cpu.to(device))
            received[:, reserved] = 0.0
            b0 = jscc.decode_active(received, dense_shape).clamp(0.0, 1.0)
            snr_tensor = torch.full((len(samples),), snr, device=device)
            b1_output = b1(
                b0,
                snr_tensor / float(b1_config["model"]["snr_norm_max"]),
                gate_tensor(b1_config, snr_tensor, device),
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            runtime = (time.perf_counter() - started) * 1000.0 / len(samples)
            probability = evaluate_probabilities(evaluator, evaluator_temperature, b1_output, eval_cfg)
            prediction = probability.argmax(dim=1)
            metrics = metric_tensors(target, b1_output, lpips_model)
            generated_rows.extend(
                candidate_rows(
                    method="b1_exact_batch64_reproduction",
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
                    peak_memory_mib=torch.cuda.max_memory_allocated(device) / (1024**2)
                    if device.type == "cuda"
                    else 0.0,
                    image_real_symbols=19632,
                    payload_real_symbols=80,
                )
            )
            print(json.dumps({"seed": seed, "snr_db": snr, "rows": len(generated_rows)}), flush=True)

    expected = int(config["audit"]["expected_rows"])
    if len(generated_rows) != expected:
        raise RuntimeError(f"S29 row count changed: {len(generated_rows)} != {expected}")
    generated_path = output / "per_sample.csv"
    write_csv(generated_path, generated_rows)
    frozen_rows: list[dict[str, str]] = []
    for item in config["inputs"]["frozen_b1_csvs"]:
        frozen_rows.extend(read_csv(require_sha(item["path"], item["sha256"])))
    generated_map = keyed(generated_rows)
    frozen_map = keyed(frozen_rows)
    if set(generated_map) != set(frozen_map):
        raise RuntimeError("S29/frozen B1 key sets differ")

    audit = {
        "rows": expected,
        "batch_size": int(config["audit"]["batch_size"]),
        "noise_sha_mismatches": 0,
        "prediction_mismatches": 0,
        "failure_mismatches": 0,
        "max_abs_psnr_difference": 0.0,
        "max_abs_ms_ssim_difference": 0.0,
        "max_abs_lpips_difference": 0.0,
    }
    for key, row in generated_map.items():
        frozen = frozen_map[key]
        audit["noise_sha_mismatches"] += row["canonical_noise_sha256"] != frozen["canonical_noise_sha256"]
        audit["prediction_mismatches"] += int(row["final_prediction"]) != int(frozen["final_prediction"])
        audit["failure_mismatches"] += as_bool(row["final_failure"]) != as_bool(frozen["final_failure"])
        for name, field in (
            ("psnr", "final_psnr"),
            ("ms_ssim", "final_ms_ssim"),
            ("lpips", "final_lpips"),
        ):
            audit[f"max_abs_{name}_difference"] = max(
                float(audit[f"max_abs_{name}_difference"]),
                abs(float(row[field]) - float(frozen[field])),
            )
    tolerance = float(config["audit"]["metric_abs_tolerance"])
    checks = {
        "noise_sha_exact": audit["noise_sha_mismatches"] <= int(config["audit"]["noise_sha_mismatches_max"]),
        "prediction_exact": audit["prediction_mismatches"] <= int(config["audit"]["prediction_mismatches_max"]),
        "failure_exact": audit["failure_mismatches"] <= int(config["audit"]["failure_mismatches_max"]),
        "psnr_within_tolerance": audit["max_abs_psnr_difference"] <= tolerance,
        "ms_ssim_within_tolerance": audit["max_abs_ms_ssim_difference"] <= tolerance,
        "lpips_within_tolerance": audit["max_abs_lpips_difference"] <= tolerance,
    }
    result = {
        "analysis_id": config["analysis_id"],
        "status": "COMPLETE",
        "scope": "diagnostic_only_not_independent_replication",
        "hypothesis": config["audit"]["hypothesis"],
        "audit": audit,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "NEGATIVE",
        "interpretation": (
            "s28_contract_difference_explained_by_batch_dependent_floating_point_arithmetic"
            if all(checks.values())
            else "s28_contract_difference_not_fully_explained_by_batch_shape"
        ),
        "official_imagenette_validation_accessed": False,
        "downloaded": False,
        "per_sample_sha256": sha256_file(generated_path),
    }
    summary_path = output / "summary.json"
    save_json(summary_path, result)
    save_json(
        output / "STATE.json",
        {"state": "S28_CONTRACT_DIAGNOSTIC_COMPLETE", "summary_sha256": sha256_file(summary_path), **result},
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
