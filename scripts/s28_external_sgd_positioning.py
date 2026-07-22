#!/usr/bin/env python3
"""Run the frozen current method on the frozen S20 population and position it against SGD-JSCC."""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from torchvision import transforms
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
from cadsd_jscc.semantic_sketch import (  # noqa: E402
    integer_codes_to_bits,
    quantize_probabilities_uniform,
)
from cadsd_jscc.snr_identity_envelope import (  # noqa: E402
    apply_correction_envelope,
    envelope_strength,
)
from pc_imagenette_supervised_audit import (  # noqa: E402
    evaluate_probabilities,
    load_scratch_classifier,
)
from s17_channel_matched_latent_diffusion import (  # noqa: E402
    active_to_dense,
    build_denoiser,
    build_jscc,
    canonical_batch_noise,
    coordinate_contract,
    dense_to_active,
    load_denoiser_checkpoint,
    seed_everything,
)
from s19_train_and_evaluate_fusion import gates, load_frozen_model  # noqa: E402
from s20_sgd_b1_decision import (  # noqa: E402
    as_bool,
    evaluator_config,
    keyed,
    load_population,
    paired_summary,
    read_csv,
    summarize,
)
from s21_b1_anchored_gated_fusion import (  # noqa: E402
    load_config,
    resolve,
    save_json,
    sha256_file,
    write_csv,
)
from s5_residual_refiner_pilot import build_model, gate_tensor, try_load_lpips  # noqa: E402


def require_sha(path_value: str | Path, expected: str) -> Path:
    path = resolve(path_value)
    if not path.is_file() or sha256_file(path) != str(expected):
        raise RuntimeError(f"missing or hash-mismatched frozen input: {path}")
    return path


def validate(config: dict[str, Any]) -> dict[str, Any]:
    protocol = config["protocol"]
    if protocol["status"] != "preregistered_before_current_method_external_output":
        raise RuntimeError("S28 config is not in its executable preregistered state")
    for key in (
        "s20_b1_and_sgd_outcomes_known",
        "s27_pristine_replication_outcome_known",
        "frozen_current_method_no_retraining",
        "frozen_s20_population_and_channel_noise",
        "no_selection_or_tuning",
        "one_shot_external_positioning",
    ):
        if protocol.get(key) is not True:
            raise RuntimeError(f"S28 protocol boundary changed: {key}")
    if protocol.get("official_imagenette_validation_accessed") is not False:
        raise RuntimeError("official Imagenette validation must remain sealed")
    scalar_inputs = (
        ("s20_config", "s20_config_sha256"),
        ("s20_aggregate_summary", "s20_aggregate_summary_sha256"),
        ("population_reference", "population_reference_sha256"),
        ("split_manifest", "split_manifest_sha256"),
        ("deepjscc_checkpoint", "deepjscc_checkpoint_sha256"),
        ("latent_diffusion_checkpoint", "latent_diffusion_checkpoint_sha256"),
        ("identity_policy", "identity_policy_sha256"),
        ("b1_config", "b1_config_sha256"),
        ("b1_checkpoint", "b1_checkpoint_sha256"),
        ("control_checkpoint", "control_checkpoint_sha256"),
        ("fusion_checkpoint", "fusion_checkpoint_sha256"),
        ("t_cls_checkpoint", "t_cls_checkpoint_sha256"),
        ("payload_sender_checkpoint", "payload_sender_checkpoint_sha256"),
    )
    for key, hash_key in scalar_inputs:
        require_sha(config["inputs"][key], config["inputs"][hash_key])
    for group in ("frozen_b1_csvs", "frozen_sgd_csvs"):
        for item in config["inputs"][group]:
            require_sha(item["path"], item["sha256"])
    if int(config["rate"]["active_real_symbols"]) != 19712:
        raise RuntimeError("S28 exact-rate contract changed")
    if int(config["rate"]["auxiliary_real_symbols"]) != 0:
        raise RuntimeError("S28 diffusion branch gained side-channel symbols")
    if list(map(float, config["diffusion"]["use_diffusion_snrs_db"])) != [1.0, 4.0, 7.0]:
        raise RuntimeError("S28 low-SNR route changed")
    if list(map(float, config["diffusion"]["exact_b1_fallback_snrs_db"])) != [13.0, 19.0]:
        raise RuntimeError("S28 high-SNR route changed")
    s20 = load_config(require_sha(config["inputs"]["s20_config"], config["inputs"]["s20_config_sha256"]))
    if list(map(float, s20["channel"]["snrs_db"])) != list(map(float, config["channel"]["snrs_db"])):
        raise RuntimeError("S20/S28 SNR contract mismatch")
    if list(map(int, s20["channel"]["base_seeds"])) != list(map(int, config["population"]["channel_seeds"])):
        raise RuntimeError("S20/S28 channel seed contract mismatch")
    return s20


def metric_tensors(
    target: torch.Tensor, candidate: torch.Tensor, lpips_model: torch.nn.Module
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        psnr_per_sample(candidate, target),
        ms_ssim_per_sample(candidate, target),
        lpips_model(candidate * 2.0 - 1.0, target * 2.0 - 1.0).flatten(),
    )


def rows_from_candidate(
    *,
    method: str,
    samples: list[dict[str, Any]],
    labels: torch.Tensor,
    original_prediction: torch.Tensor,
    original_confidence: torch.Tensor,
    prediction: torch.Tensor,
    metrics: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    b1_prediction: torch.Tensor,
    seed: int,
    snr: float,
    noise_shas: list[str],
    runtime_ms: float,
    peak_memory_mib: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(samples):
        label = int(labels[index])
        original_ok = int(original_prediction[index]) == label and float(original_confidence[index]) >= 0.5
        final_ok = int(prediction[index]) == label
        b1_ok = int(b1_prediction[index]) == label
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
                "image_real_symbols": 19632,
                "payload_real_symbols": 80,
                "diffusion_side_information_real_symbols": 0,
                "original_prediction": int(original_prediction[index]),
                "original_confidence": float(original_confidence[index]),
                "clean_correct": original_ok,
                "deepjscc_prediction": int(b1_prediction[index]),
                "deepjscc_correct": b1_ok,
                "final_prediction": int(prediction[index]),
                "final_correct": final_ok,
                "final_failure": original_ok and not final_ok,
                "new_error_vs_deepjscc": b1_ok and not final_ok,
                "repair_vs_deepjscc": (not b1_ok) and final_ok,
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


def load_frozen_rows(config: dict[str, Any], key: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in config["inputs"][key]:
        rows.extend(read_csv(require_sha(item["path"], item["sha256"])))
    expected = int(config["population"]["expected_rows_per_method"])
    if len(rows) != expected:
        raise RuntimeError(f"{key} row count changed: {len(rows)} != {expected}")
    return rows


@torch.no_grad()
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/s28_external_sgd_positioning.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    os.environ.setdefault("TORCH_HOME", str(resolve("outputs/cache/torch")))
    config_path = resolve(args.config)
    config = load_config(config_path)
    s20 = validate(config)
    output = resolve(config["outputs"]["directory"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    shutil.copy2(config_path, output / "config_before_output.yaml")
    shutil.copy2(SCRIPT, output / SCRIPT.name)

    seed_everything(int(config["seed"]))
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    samples, classes = load_population(s20)
    if len(samples) != int(config["population"]["expected_sample_count"]):
        raise RuntimeError("frozen S20 population size changed")
    transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()]
    )
    target_cpu = torch.stack(
        [transform(Image.open(item["path"]).convert("RGB")) for item in samples]
    )
    labels_cpu = torch.tensor([int(item["class_idx"]) for item in samples])

    jscc = build_jscc(config, device)
    denoiser = build_denoiser(config, device)
    load_denoiser_checkpoint(denoiser, resolve(config["inputs"]["latent_diffusion_checkpoint"]), device)
    denoiser.eval().requires_grad_(False)
    reserved, _valid_active, valid_dense = coordinate_contract(jscc, config, device)
    control, _control_checkpoint = load_frozen_model(config, "control", device)
    fusion, _fusion_checkpoint = load_frozen_model(config, "fusion", device)
    b1_config = load_config(resolve(config["inputs"]["b1_config"]))
    b1 = build_model(b1_config).to(device)
    b1_checkpoint = torch.load(resolve(config["inputs"]["b1_checkpoint"]), map_location=device)
    b1.load_state_dict(b1_checkpoint["model_state_dict"], strict=True)
    b1.eval().requires_grad_(False)

    eval_cfg = evaluator_config(config)
    evaluator, evaluator_temperature = load_scratch_classifier(
        str(resolve(config["inputs"]["t_cls_checkpoint"])), classes, device, "T_cls"
    )
    sender, sender_temperature = load_scratch_classifier(
        str(resolve(config["inputs"]["payload_sender_checkpoint"])), classes, device, "G_aux"
    )
    lpips_model, lpips_error = try_load_lpips(device, resolve("outputs/cache"))
    if lpips_model is None:
        raise RuntimeError(lpips_error)
    lpips_model.eval().requires_grad_(False)

    target_all = target_cpu.to(device)
    labels_all = labels_cpu.to(device)
    original_probability = evaluate_probabilities(
        evaluator, evaluator_temperature, target_all, eval_cfg
    )
    original_confidence_all, original_prediction_all = original_probability.max(dim=1)
    if not bool(
        (
            (original_prediction_all == labels_all)
            & (original_confidence_all >= float(config["evaluator"]["clean_confidence_threshold"]))
        ).all()
    ):
        raise RuntimeError("frozen S20 population is no longer clean under T_cls")
    source_probability = evaluate_probabilities(sender, sender_temperature, target_all, eval_cfg)
    source_codes, _ = quantize_probabilities_uniform(
        source_probability, int(config["payload_sender"]["quantization_bits"])
    )
    payload_all = (
        integer_codes_to_bits(source_codes, int(config["payload_sender"]["quantization_bits"]))
        .to(target_all.dtype)
        .mul(2.0)
        .sub(1.0)
        .unsqueeze(-1)
        .expand(-1, -1, int(config["payload_sender"]["repetitions"]))
        .reshape(len(samples), -1)
    )
    if payload_all.shape[1] != int(config["rate"]["payload_real_symbols"]):
        raise RuntimeError("payload width no longer matches the 80-symbol reservation")

    policy = json.loads(resolve(config["inputs"]["identity_policy"]).read_text(encoding="utf-8"))
    if str(policy["selected_name"]) != str(config["diffusion"]["selected_policy_name"]):
        raise RuntimeError("identity diffusion policy changed")
    specification = policy["selected_specification"]
    low_snrs = set(map(float, config["diffusion"]["use_diffusion_snrs_db"]))
    batch_size = int(config["evaluation"]["batch_size"])
    current_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    regenerated_b1_rows: list[dict[str, Any]] = []
    high_snr_exact_pixel_max = 0.0

    for base_seed in map(int, config["population"]["channel_seeds"]):
        for snr in map(float, config["channel"]["snrs_db"]):
            stage_images: dict[str, list[torch.Tensor]] = {
                "b0": [], "diffusion": [], "current": [], "control": [], "b1": []
            }
            runtime_values: list[float] = []
            noise_shas_all: list[str] = []
            for start in range(0, len(samples), batch_size):
                stop = min(start + batch_size, len(samples))
                batch_samples = samples[start:stop]
                target = target_all[start:stop]
                payload = payload_all[start:stop]
                noises_cpu, noise_shas = canonical_batch_noise(
                    [str(item["sample_id"]) for item in batch_samples],
                    snr,
                    base_seed,
                    jscc.active_symbols,
                )
                noise_shas_all.extend(noise_shas)
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
                b0_output = jscc.decode_active(received, dense_shape).clamp(0.0, 1.0)
                snr_tensor = torch.full((stop - start,), snr, device=device)
                snr_norm = snr_tensor / float(b1_config["model"]["snr_norm_max"])
                gate = gate_tensor(b1_config, snr_tensor, device)
                if snr in low_snrs:
                    alpha = float(
                        channel_alpha(snr, float(config["channel"]["noise_variance_factor_per_real"]))
                    )
                    strength = float(
                        envelope_strength(
                            snr,
                            specification,
                            noise_variance_factor_per_real=float(
                                config["channel"]["noise_variance_factor_per_real"]
                            ),
                            reference_snr_db=1.0,
                        )
                    )
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
                    controlled_active = apply_correction_envelope(received, full_active, strength)
                    controlled_active[:, reserved] = 0.0
                    diffusion_output = jscc.decode_active(controlled_active, dense_shape).clamp(0.0, 1.0)
                    current_output = fusion(
                        b0_output, diffusion_output, snr_norm, gates(config, snr_tensor, device)
                    )
                else:
                    diffusion_output = b0_output
                    current_output = b1(b0_output, snr_norm, gate)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                runtime_values.extend(
                    [(time.perf_counter() - started) * 1000.0 / (stop - start)] * (stop - start)
                )
                if snr in low_snrs:
                    b1_output = b1(b0_output, snr_norm, gate)
                    control_output = control(
                        b0_output, b0_output, snr_norm, gates(config, snr_tensor, device)
                    )
                else:
                    b1_output = current_output
                    control_output = b1_output
                    high_snr_exact_pixel_max = max(
                        high_snr_exact_pixel_max,
                        float((current_output - b1_output).abs().max().cpu()),
                    )
                for name, tensor in (
                    ("b0", b0_output),
                    ("diffusion", diffusion_output),
                    ("current", current_output),
                    ("control", control_output),
                    ("b1", b1_output),
                ):
                    stage_images[name].append(tensor.detach())

            candidates = {name: torch.cat(parts, dim=0) for name, parts in stage_images.items()}
            probabilities = {
                name: evaluate_probabilities(evaluator, evaluator_temperature, tensor, eval_cfg)
                for name, tensor in candidates.items()
            }
            predictions = {name: value.argmax(dim=1) for name, value in probabilities.items()}
            qualities = {
                name: metric_tensors(target_all, tensor, lpips_model)
                for name, tensor in candidates.items()
            }
            runtime_ms = float(sum(runtime_values) / len(runtime_values))
            peak = (
                torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda"
                else 0.0
            )
            common = {
                "samples": samples,
                "labels": labels_all,
                "original_prediction": original_prediction_all,
                "original_confidence": original_confidence_all,
                "b1_prediction": predictions["b1"],
                "seed": base_seed,
                "snr": snr,
                "noise_shas": noise_shas_all,
                "runtime_ms": runtime_ms,
                "peak_memory_mib": peak,
            }
            current_rows.extend(
                rows_from_candidate(
                    method="s19_low_snr_fusion_exact_b1_fallback",
                    prediction=predictions["current"],
                    metrics=qualities["current"],
                    **common,
                )
            )
            control_rows.extend(
                rows_from_candidate(
                    method="s19_matched_control_exact_b1_fallback",
                    prediction=predictions["control"],
                    metrics=qualities["control"],
                    **common,
                )
            )
            regenerated_b1_rows.extend(
                rows_from_candidate(
                    method="regenerated_b1_contract_audit",
                    prediction=predictions["b1"],
                    metrics=qualities["b1"],
                    **common,
                )
            )
            if base_seed == int(config["population"]["channel_seeds"][0]):
                grid = torch.cat(
                    [
                        target_all[:8],
                        candidates["b0"][:8],
                        candidates["diffusion"][:8],
                        candidates["current"][:8],
                        candidates["b1"][:8],
                    ],
                    dim=0,
                ).cpu()
                save_image(
                    grid,
                    output / f"seed_{base_seed}_snr_{int(snr):02d}_target_b0_diffusion_current_b1.png",
                    nrow=8,
                )
            print(
                json.dumps(
                    {"seed": base_seed, "snr_db": snr, "rows": len(current_rows)},
                    ensure_ascii=False,
                ),
                flush=True,
            )

    expected = int(config["population"]["expected_rows_per_method"])
    if any(len(rows) != expected for rows in (current_rows, control_rows, regenerated_b1_rows)):
        raise RuntimeError("S28 generated row count changed")
    current_path = output / "current_per_sample.csv"
    control_path = output / "matched_control_per_sample.csv"
    regenerated_path = output / "regenerated_b1_per_sample.csv"
    write_csv(current_path, current_rows)
    write_csv(control_path, control_rows)
    write_csv(regenerated_path, regenerated_b1_rows)

    frozen_b1_rows = load_frozen_rows(config, "frozen_b1_csvs")
    frozen_sgd_rows = load_frozen_rows(config, "frozen_sgd_csvs")
    maps = {
        "current": keyed(current_rows),
        "control": keyed(control_rows),
        "regenerated_b1": keyed(regenerated_b1_rows),
        "frozen_b1": keyed(frozen_b1_rows),
        "sgd": keyed(frozen_sgd_rows),
    }
    if len({frozenset(value) for value in maps.values()}) != 1:
        raise RuntimeError("S28 method key sets differ")

    audit = {
        "noise_sha_mismatches_vs_frozen_b1": 0,
        "prediction_mismatches_vs_frozen_b1": 0,
        "failure_mismatches_vs_frozen_b1": 0,
        "max_abs_psnr_difference_vs_frozen_b1": 0.0,
        "max_abs_ms_ssim_difference_vs_frozen_b1": 0.0,
        "max_abs_lpips_difference_vs_frozen_b1": 0.0,
        "high_snr_current_b1_pixel_max_abs": high_snr_exact_pixel_max,
    }
    for key, row in maps["regenerated_b1"].items():
        frozen = maps["frozen_b1"][key]
        audit["noise_sha_mismatches_vs_frozen_b1"] += (
            row["canonical_noise_sha256"] != frozen["canonical_noise_sha256"]
        )
        audit["prediction_mismatches_vs_frozen_b1"] += int(row["final_prediction"]) != int(
            frozen["final_prediction"]
        )
        audit["failure_mismatches_vs_frozen_b1"] += as_bool(row["final_failure"]) != as_bool(
            frozen["final_failure"]
        )
        for name, field in (
            ("psnr", "final_psnr"),
            ("ms_ssim", "final_ms_ssim"),
            ("lpips", "final_lpips"),
        ):
            audit[f"max_abs_{name}_difference_vs_frozen_b1"] = max(
                float(audit[f"max_abs_{name}_difference_vs_frozen_b1"]),
                abs(float(row[field]) - float(frozen[field])),
            )

    comparisons = {
        "current_minus_b1": paired_summary(maps["current"], maps["frozen_b1"], config),
        "current_minus_sgd_paper_upper": paired_summary(maps["current"], maps["sgd"], config),
        "current_minus_matched_control": paired_summary(maps["current"], maps["control"], config),
    }
    tolerance = float(config["evaluation"]["frozen_b1_metric_abs_tolerance"])
    contract_checks = {
        "noise_sha_exact": audit["noise_sha_mismatches_vs_frozen_b1"] == 0,
        "b1_prediction_exact": audit["prediction_mismatches_vs_frozen_b1"] == 0,
        "b1_failure_exact": audit["failure_mismatches_vs_frozen_b1"] == 0,
        "b1_psnr_reproduced": audit["max_abs_psnr_difference_vs_frozen_b1"] <= tolerance,
        "b1_ms_ssim_reproduced": audit["max_abs_ms_ssim_difference_vs_frozen_b1"] <= tolerance,
        "b1_lpips_reproduced": audit["max_abs_lpips_difference_vs_frozen_b1"]
        <= float(config["evaluation"]["frozen_b1_lpips_abs_tolerance"]),
        "high_snr_exact_b1": high_snr_exact_pixel_max
        <= float(config["evaluation"]["high_snr_exact_b1_pixel_tolerance"]),
    }
    b1_delta = comparisons["current_minus_b1"]["left_minus_right"]
    control_delta = comparisons["current_minus_matched_control"]["left_minus_right"]
    b1_checks = {
        "current_b1_psnr_ci": float(b1_delta["psnr_ci95"][0])
        >= float(config["success_criteria"]["current_minus_b1_psnr_ci_low_min_db"]),
        "current_b1_lpips_ci": float(b1_delta["lpips_ci95"][1])
        <= float(config["success_criteria"]["current_minus_b1_lpips_ci_high_max"]),
        "current_failure_not_greater_than_b1": int(
            comparisons["current_minus_b1"]["left_failures"]
        )
        <= int(comparisons["current_minus_b1"]["right_failures"]),
        "current_control_psnr_ci": float(control_delta["psnr_ci95"][0])
        >= float(config["success_criteria"]["current_minus_control_psnr_ci_low_min_db"]),
        "current_control_lpips_ci": float(control_delta["lpips_ci95"][1])
        <= float(config["success_criteria"]["current_minus_control_lpips_ci_high_max"]),
    }

    sgd_delta = comparisons["current_minus_sgd_paper_upper"]["left_minus_right"]
    if float(sgd_delta["psnr_ci95"][0]) > 0.0 and float(sgd_delta["lpips_ci95"][0]) > 0.0:
        external_relation = "current_higher_psnr_sgd_better_lpips_pareto_tradeoff"
    elif (
        float(sgd_delta["psnr_ci95"][0]) > 0.0
        and float(sgd_delta["lpips_ci95"][1]) < 0.0
        and comparisons["current_minus_sgd_paper_upper"]["left_failures"]
        <= comparisons["current_minus_sgd_paper_upper"]["right_failures"]
    ):
        external_relation = "current_dominates_sgd_paper_upper_on_preregistered_axes"
    elif float(sgd_delta["psnr_ci95"][1]) < 0.0 and float(sgd_delta["lpips_ci95"][1]) < 0.0:
        external_relation = "sgd_paper_upper_higher_psnr_and_better_lpips"
    else:
        external_relation = "mixed_or_statistically_unresolved"

    caption_symbols = int(config["rate"]["sgd_caption_packet_bits_per_patch"]) * int(
        config["rate"]["sgd_caption_patches_per_image"]
    )
    sgd_image_symbols = int(config["rate"]["sgd_main_real_symbols"]) + int(
        config["rate"]["sgd_active_edge_real_symbols"]
    )
    strict_minimum = sgd_image_symbols + caption_symbols
    result = {
        "analysis_id": config["analysis_id"],
        "status": "COMPLETE",
        "claim_scope": "frozen_cross_dataset_external_positioning_not_final_paper_ranking",
        "population": {
            "images": len(samples),
            "channel_seeds": config["population"]["channel_seeds"],
            "snrs_db": config["channel"]["snrs_db"],
            "rows_per_method": expected,
            "official_imagenette_validation_accessed": False,
        },
        "methods": {
            "current": summarize(current_rows),
            "matched_control": summarize(control_rows),
            "b1_frozen": summarize(frozen_b1_rows),
            "sgd_paper_upper": summarize(frozen_sgd_rows),
        },
        "contract_reproduction_audit": audit,
        "contract_checks": contract_checks,
        "b1_improvement_checks": b1_checks,
        "comparisons": comparisons,
        "rate_audit": {
            "current_total_real_symbols": int(config["rate"]["active_real_symbols"]),
            "current_auxiliary_real_symbols": int(config["rate"]["auxiliary_real_symbols"]),
            "sgd_released_image_plus_edge_real_symbols": sgd_image_symbols,
            "sgd_minimum_caption_real_symbols": caption_symbols,
            "sgd_minimum_total_real_symbols_if_text_metered": strict_minimum,
            "sgd_minimum_overrun_real_symbols": strict_minimum
            - int(config["rate"]["active_real_symbols"]),
            "sgd_minimum_overrun_fraction_of_budget": (
                strict_minimum - int(config["rate"]["active_real_symbols"])
            )
            / int(config["rate"]["active_real_symbols"]),
            "strict_equal_total_rate_sgd_released_weights_executable": strict_minimum
            <= int(config["rate"]["active_real_symbols"]),
        },
        "external_relation": external_relation,
        "verdict": "PASS" if all(contract_checks.values()) and all(b1_checks.values()) else "NEGATIVE",
        "downloaded": False,
        "current_per_sample_sha256": sha256_file(current_path),
        "matched_control_per_sample_sha256": sha256_file(control_path),
        "regenerated_b1_per_sample_sha256": sha256_file(regenerated_path),
    }
    summary_path = output / "summary.json"
    save_json(summary_path, result)
    save_json(
        output / "STATE.json",
        {"state": "EXTERNAL_POSITIONING_COMPLETE", "summary_sha256": sha256_file(summary_path), **result},
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
