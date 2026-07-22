#!/usr/bin/env python3
"""Run project M3 or a SING-Zero-style baseline under the frozen common pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from cadsd_jscc.deepjscc_adapter import (  # noqa: E402
    build_deepjscc_model,
    deepjscc_decode,
    deepjscc_encode,
    load_deepjscc_model,
)
from cadsd_jscc.external_rate_alignment import ExactRateMaskedDeepJSCC  # noqa: E402
from cadsd_jscc.external_common import (  # noqa: E402
    canonical_noise_sha256,
    canonical_standard_normal,
    complex_awgn_from_standard_normal,
)
from cadsd_jscc.metrics import ms_ssim_per_sample, psnr_per_sample  # noqa: E402
from cadsd_jscc.semantic_sketch import (  # noqa: E402
    bits_to_integer_codes,
    embed_repeated_sketch,
    integer_codes_to_bits,
    quantize_probabilities_uniform,
    recover_repeated_sketch_and_erase,
)
from pc_imagenette_sender_inbudget_awgn_audit import (  # noqa: E402
    cross_model_triplet_acceptance,
    route_final_candidate,
)
from pc_imagenette_supervised_audit import (  # noqa: E402
    evaluate_probabilities,
    load_scratch_classifier,
    source_semantic_score_tensors,
)
from pc_posterior_consistency_replication import posterior_correct  # noqa: E402
from s10_short_chain_residual_shift_diffusion import (  # noqa: E402
    ShortChainResidualShiftDiffusion,
)
from s5_residual_refiner_pilot import build_model, gate_tensor, try_load_lpips  # noqa: E402


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(resolve(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"YAML root must be a mapping: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(path: str | Path, expected: str) -> Path:
    resolved = resolve(path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    observed = sha256_file(resolved)
    if observed != expected:
        raise RuntimeError(f"SHA-256 mismatch for {resolved}: {observed} != {expected}")
    return resolved


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_and_load_samples(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    if config.get("status") != "preregistered_before_any_pilot_method_output":
        raise RuntimeError("common pilot must remain preregistered before first output")
    if config.get("official_val_accessed") is not False:
        raise RuntimeError("official validation must remain sealed")
    if config.get("outcome_claims_allowed") is not False:
        raise RuntimeError("pilot cannot authorize outcome claims")
    population = config["population"]
    manifest_path = require_sha(
        population["split_manifest"], population["split_manifest_sha256"]
    )
    require_sha(
        population["frozen_clean_membership_source"],
        population["frozen_clean_membership_source_sha256"],
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bool(manifest.get("official_val_accessed")):
        raise RuntimeError("training split manifest records official validation access")
    classes = [str(item) for item in manifest["classes"]]
    by_id = {
        str(item["sample_id"]): item
        for item in manifest["samples"]
        if str(item["split"]) == str(population["required_split"])
    }
    root = resolve(manifest["source_train_root"])
    selected: list[dict[str, Any]] = []
    for frozen in population["samples"]:
        sample_id = str(frozen["sample_id"])
        item = dict(by_id.get(sample_id, {}))
        if not item:
            raise RuntimeError(f"sample is absent from frozen policy-dev split: {sample_id}")
        if int(item["class_idx"]) != int(frozen["class_idx"]):
            raise RuntimeError(f"class mismatch for {sample_id}")
        path = root / str(item["relative_path"])
        require_sha(path, str(frozen["content_sha256"]))
        item["path"] = path
        selected.append(item)
    if len(selected) != 8 or len({item["sample_id"] for item in selected}) != 8:
        raise RuntimeError("frozen pilot must contain exactly eight unique images")
    return selected, classes


def evaluation_config(config: dict[str, Any]) -> dict[str, Any]:
    evaluator = config["evaluator"]
    return {
        "imagenette": {
            "normalization_mean": evaluator["normalization_mean"],
            "normalization_std": evaluator["normalization_std"],
        }
    }


def make_common_reference(
    jscc: torch.nn.Module,
    target: torch.Tensor,
    noises: torch.Tensor,
    snr: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    transmitted = deepjscc_encode(jscc, target)
    if int(transmitted[0].numel()) != 65536:
        raise RuntimeError(f"DeepJSCC latent has {transmitted[0].numel()} real symbols")
    received = complex_awgn_from_standard_normal(transmitted, noises, snr)
    return deepjscc_decode(jscc, received), transmitted, received


def metric_tensors(
    target: torch.Tensor, candidate: torch.Tensor, lpips_model: torch.nn.Module
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        psnr = psnr_per_sample(candidate, target)
        ms_ssim = ms_ssim_per_sample(candidate, target)
        lpips = lpips_model(candidate * 2.0 - 1.0, target * 2.0 - 1.0).flatten()
    return psnr, ms_ssim, lpips


def base_row(
    *,
    method: str,
    item: dict[str, Any],
    snr: float,
    noise_sha: str,
    label: int,
    original_prediction: int,
    original_confidence: float,
    reference_prediction: int,
    final_prediction: int,
    reference_psnr: float,
    reference_ms_ssim: float,
    reference_lpips: float,
    final_psnr: float,
    final_ms_ssim: float,
    final_lpips: float,
    runtime_ms: float,
    peak_memory_mib: float,
) -> dict[str, Any]:
    clean = original_prediction == label and original_confidence >= 0.5
    reference_correct = reference_prediction == label
    final_correct = final_prediction == label
    return {
        "method": method,
        "sample_id": item["sample_id"],
        "wnid": item["wnid"],
        "class_idx": label,
        "snr_db": snr,
        "base_seed": 20260729,
        "canonical_noise_sha256": noise_sha,
        "noise_variance_convention": "complex_awgn_per_real_half_variance",
        "total_real_symbols": 65536,
        "total_complex_channel_uses": 32768,
        "cbr": 1.0 / 6.0,
        "original_prediction": original_prediction,
        "original_confidence": original_confidence,
        "clean_correct": clean,
        "deepjscc_prediction": reference_prediction,
        "deepjscc_correct": reference_correct,
        "final_prediction": final_prediction,
        "final_correct": final_correct,
        "final_failure": clean and not final_correct,
        "new_error_vs_deepjscc": clean and reference_correct and not final_correct,
        "repair_vs_deepjscc": clean and not reference_correct and final_correct,
        "deepjscc_psnr": reference_psnr,
        "deepjscc_ms_ssim": reference_ms_ssim,
        "deepjscc_lpips": reference_lpips,
        "final_psnr": final_psnr,
        "final_ms_ssim": final_ms_ssim,
        "final_lpips": final_lpips,
        "runtime_ms_per_image": runtime_ms,
        "peak_gpu_memory_mib": peak_memory_mib,
    }


def load_shared_models(config: dict[str, Any], classes: list[str], device: torch.device):
    method = config["methods"]["ours_m3"]
    source_config = load_yaml("configs/s13_coco_train2017_c8_scaleup_export.yaml")
    require_sha(method["deepjscc_checkpoint"], method["deepjscc_checkpoint_sha256"])
    jscc = load_deepjscc_model(
        resolve(source_config["baseline"]["repo"]),
        resolve(method["deepjscc_checkpoint"]),
        int(source_config["rate"]["inner_channel"]),
        "AWGN",
        float(config["channel"]["snrs_db"][0]),
        device,
    ).requires_grad_(False)
    eval_cfg = evaluation_config(config)
    evaluator_path = require_sha(
        config["evaluator"]["checkpoint"], config["evaluator"]["checkpoint_sha256"]
    )
    evaluator, evaluator_temperature = load_scratch_classifier(
        str(evaluator_path), classes, device, str(config["evaluator"]["expected_role"])
    )
    lpips_model, lpips_error = try_load_lpips(device, resolve("outputs/cache"))
    if lpips_model is None:
        raise RuntimeError(lpips_error)
    return jscc, evaluator, evaluator_temperature, lpips_model, eval_cfg


def run_ours(
    config: dict[str, Any], samples: list[dict[str, Any]], classes: list[str], output: Path
) -> dict[str, Any]:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    jscc, evaluator, evaluator_temperature, lpips_model, eval_cfg = load_shared_models(
        config, classes, device
    )
    method = config["methods"]["ours_m3"]
    b1_config = load_yaml(method["b1_config"])
    b1 = build_model(b1_config).to(device)
    b1_path = require_sha(method["b1_checkpoint"], method["b1_checkpoint_sha256"])
    b1.load_state_dict(torch.load(b1_path, map_location=device)["model_state_dict"])
    b1.eval().requires_grad_(False)
    diffusion_config = load_yaml(method["diffusion_config"])
    diffusion = ShortChainResidualShiftDiffusion(diffusion_config).to(device)
    diffusion_path = require_sha(
        method["diffusion_checkpoint"], method["diffusion_checkpoint_sha256"]
    )
    diffusion.load_state_dict(
        torch.load(diffusion_path, map_location=device)["model_state_dict"]
    )
    diffusion.eval().requires_grad_(False)
    sender_path = require_sha(method["sender_checkpoint"], method["sender_checkpoint_sha256"])
    sender, sender_temperature = load_scratch_classifier(
        str(sender_path), classes, device, "G_aux"
    )
    gate_path = require_sha(method["gate_checkpoint"], method["gate_checkpoint_sha256"])
    receiver_guard, receiver_guard_temperature = load_scratch_classifier(
        str(gate_path), classes, device, "G_gate"
    )

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
    if not bool(((original_prediction == labels) & (original_confidence >= 0.5)).all()):
        raise RuntimeError("frozen pilot includes a non-clean T_cls image")

    rows: list[dict[str, Any]] = []
    for snr in map(float, config["channel"]["snrs_db"]):
        noises_cpu = torch.stack(
            [
                canonical_standard_normal(
                    int(config["channel"]["base_seed"]), str(item["sample_id"]), snr, 65536
                )
                for item in samples
            ]
        )
        noise_shas = [canonical_noise_sha256(row) for row in noises_cpu]
        noises = noises_cpu.to(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.no_grad():
            reference_b0, latent, _ = make_common_reference(jscc, target, noises, snr)
            source_probability = evaluate_probabilities(sender, sender_temperature, target, eval_cfg)
            source_codes, _ = quantize_probabilities_uniform(source_probability, 2)
            source_bits = integer_codes_to_bits(source_codes, 2)
            source_sketch = source_bits.to(source_probability.dtype).mul(2.0).sub(1.0)
            transmitted, reserved = embed_repeated_sketch(latent, source_sketch, 4)
            received = complex_awgn_from_standard_normal(transmitted, noises, snr)
            recovered_sketch, erased_received = recover_repeated_sketch_and_erase(
                received, 20, 4, reserved
            )
            recovered_bits = (recovered_sketch > 0).to(torch.int64)
            recovered_codes = bits_to_integer_codes(recovered_bits, 10, 2)
            decoded = recovered_codes.to(source_probability.dtype)
            decoded_total = decoded.sum(dim=1, keepdim=True)
            recovered_probability = torch.where(
                decoded_total > 0,
                decoded / decoded_total.clamp_min(1.0),
                torch.full_like(decoded, 0.1),
            )
            b0 = deepjscc_decode(jscc, erased_received)
            snr_tensor = torch.full((len(samples),), snr, device=device)
            snr_norm = snr_tensor / 20.0
            anchor = b1(b0, snr_norm, gate_tensor(b1_config, snr_tensor, device))
            raw = diffusion(
                anchor, snr_norm, gate_tensor(diffusion_config, snr_tensor, device)
            )
        valid_mask = torch.ones(received[0].numel(), dtype=torch.bool, device=device)
        valid_mask[reserved] = False
        posterior = posterior_correct(
            jscc,
            raw,
            erased_received,
            int(method["posterior_steps"]),
            float(method["posterior_normalized_step_size"]),
            valid_mask=valid_mask,
        )
        with torch.no_grad():
            sender_anchor_probability = evaluate_probabilities(
                sender, sender_temperature, anchor, eval_cfg
            )
            sender_posterior_probability = evaluate_probabilities(
                sender, sender_temperature, posterior, eval_cfg
            )
            risk = source_semantic_score_tensors(
                recovered_probability,
                sender_anchor_probability,
                sender_posterior_probability,
            )["fullprob_js_risk"]
            gate_anchor_probability = evaluate_probabilities(
                receiver_guard, receiver_guard_temperature, anchor, eval_cfg
            )
            gate_posterior_probability = evaluate_probabilities(
                receiver_guard, receiver_guard_temperature, posterior, eval_cfg
            )
            (
                sender_accepted,
                receiver_accepted,
                source_anchor_accepted,
                accepted,
            ) = cross_model_triplet_acceptance(
                risk,
                0.0,
                recovered_probability,
                gate_anchor_probability,
                gate_posterior_probability,
            )
            final = route_final_candidate(
                accepted,
                source_anchor_accepted,
                posterior,
                anchor,
                raw,
                str(method["final_routing"]),
            )
        torch.cuda.synchronize(device)
        runtime_ms = (time.perf_counter() - started) * 1000.0 / len(samples)
        with torch.no_grad():
            reference_probability = evaluate_probabilities(
                evaluator, evaluator_temperature, reference_b0, eval_cfg
            )
            final_probability = evaluate_probabilities(
                evaluator, evaluator_temperature, final, eval_cfg
            )
            reference_prediction = reference_probability.argmax(dim=1)
            final_prediction = final_probability.argmax(dim=1)
            reference_metrics = metric_tensors(target, reference_b0, lpips_model)
            final_metrics = metric_tensors(target, final, lpips_model)
        save_image(
            torch.cat([target, reference_b0, b0, anchor, raw, posterior, final]).cpu(),
            output / f"snr_{int(snr):02d}_source_refb0_payloadb0_anchor_raw_post_final.png",
            nrow=len(samples),
        )
        peak = torch.cuda.max_memory_allocated(device) / (1024**2)
        for index, item in enumerate(samples):
            row = base_row(
                method="ours_m3",
                item=item,
                snr=snr,
                noise_sha=noise_shas[index],
                label=int(labels[index]),
                original_prediction=int(original_prediction[index]),
                original_confidence=float(original_confidence[index]),
                reference_prediction=int(reference_prediction[index]),
                final_prediction=int(final_prediction[index]),
                reference_psnr=float(reference_metrics[0][index]),
                reference_ms_ssim=float(reference_metrics[1][index]),
                reference_lpips=float(reference_metrics[2][index]),
                final_psnr=float(final_metrics[0][index]),
                final_ms_ssim=float(final_metrics[1][index]),
                final_lpips=float(final_metrics[2][index]),
                runtime_ms=runtime_ms,
                peak_memory_mib=peak,
            )
            row.update(
                {
                    "accepted": bool(accepted[index]),
                    "sender_accepted": bool(sender_accepted[index]),
                    "receiver_guard_accepted": bool(receiver_accepted[index]),
                    "source_anchor_accepted": bool(source_anchor_accepted[index]),
                    "payload_bit_errors": int((source_bits[index] != recovered_bits[index]).sum()),
                    "payload_exact": bool(torch.equal(source_bits[index], recovered_bits[index])),
                }
            )
            rows.append(row)
    write_csv(output / "per_sample.csv", rows)
    return summarize_rows("ours_m3", rows)


def run_sing_style(
    config: dict[str, Any], samples: list[dict[str, Any]], classes: list[str], output: Path
) -> dict[str, Any]:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    jscc, evaluator, evaluator_temperature, lpips_model, eval_cfg = load_shared_models(
        config, classes, device
    )
    method = config["methods"]["sing_zero_style"]
    os.environ["HF_HOME"] = str(resolve(method["local_cache_dir"]))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"
    from diffusers import StableDiffusionImg2ImgPipeline

    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        method["diffusion_prior"],
        cache_dir=resolve(method["local_cache_dir"]),
        torch_dtype=torch.float16,
        local_files_only=True,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.enable_attention_slicing()

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
    if not bool(((original_prediction == labels) & (original_confidence >= 0.5)).all()):
        raise RuntimeError("frozen pilot includes a non-clean T_cls image")

    rows: list[dict[str, Any]] = []
    for snr in map(float, config["channel"]["snrs_db"]):
        noises_cpu = torch.stack(
            [
                canonical_standard_normal(
                    int(config["channel"]["base_seed"]), str(item["sample_id"]), snr, 65536
                )
                for item in samples
            ]
        )
        noise_shas = [canonical_noise_sha256(row) for row in noises_cpu]
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.no_grad():
            reference_b0, _, _ = make_common_reference(
                jscc, target, noises_cpu.to(device), snr
            )
        generators = []
        for item in samples:
            seed_material = f"sing-style-v1|20260729|{item['sample_id']}|{snr:.6f}"
            seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "big")
            generators.append(torch.Generator(device=device).manual_seed(seed))
        with torch.inference_mode():
            diffusion_output = pipeline(
                prompt=[str(method["prompt"])] * len(samples),
                negative_prompt=[str(method["negative_prompt"])] * len(samples),
                image=reference_b0,
                strength=float(method["strength"]),
                num_inference_steps=int(method["num_inference_steps"]),
                guidance_scale=float(method["guidance_scale"]),
                generator=generators,
                output_type="pt",
            ).images.to(device)
            measured = F.avg_pool2d(reference_b0, kernel_size=2, stride=2)
            diffusion_measured = F.avg_pool2d(diffusion_output, kernel_size=2, stride=2)
            projected_unclamped = (
                F.interpolate(measured, scale_factor=2, mode="nearest")
                + diffusion_output
                - F.interpolate(diffusion_measured, scale_factor=2, mode="nearest")
            )
            final = projected_unclamped.clamp(0.0, 1.0)
            consistency_before = (diffusion_measured - measured).square().flatten(1).mean(1)
            consistency_after = (
                F.avg_pool2d(final, kernel_size=2, stride=2) - measured
            ).square().flatten(1).mean(1)
        torch.cuda.synchronize(device)
        runtime_ms = (time.perf_counter() - started) * 1000.0 / len(samples)
        with torch.no_grad():
            reference_probability = evaluate_probabilities(
                evaluator, evaluator_temperature, reference_b0, eval_cfg
            )
            final_probability = evaluate_probabilities(
                evaluator, evaluator_temperature, final, eval_cfg
            )
            reference_prediction = reference_probability.argmax(dim=1)
            final_prediction = final_probability.argmax(dim=1)
            reference_metrics = metric_tensors(target, reference_b0, lpips_model)
            final_metrics = metric_tensors(target, final, lpips_model)
        save_image(
            torch.cat([target, reference_b0, diffusion_output, final]).cpu(),
            output / f"snr_{int(snr):02d}_source_deepjscc_diffusion_projected.png",
            nrow=len(samples),
        )
        peak = torch.cuda.max_memory_allocated(device) / (1024**2)
        for index, item in enumerate(samples):
            row = base_row(
                method="sing_zero_style",
                item=item,
                snr=snr,
                noise_sha=noise_shas[index],
                label=int(labels[index]),
                original_prediction=int(original_prediction[index]),
                original_confidence=float(original_confidence[index]),
                reference_prediction=int(reference_prediction[index]),
                final_prediction=int(final_prediction[index]),
                reference_psnr=float(reference_metrics[0][index]),
                reference_ms_ssim=float(reference_metrics[1][index]),
                reference_lpips=float(reference_metrics[2][index]),
                final_psnr=float(final_metrics[0][index]),
                final_ms_ssim=float(final_metrics[1][index]),
                final_lpips=float(final_metrics[2][index]),
                runtime_ms=runtime_ms,
                peak_memory_mib=peak,
            )
            row.update(
                {
                    "measurement_mse_before_projection": float(consistency_before[index]),
                    "measurement_mse_after_projection": float(consistency_after[index]),
                    "projection_timing": method["projection_timing"],
                }
            )
            rows.append(row)
    write_csv(output / "per_sample.csv", rows)
    return summarize_rows("sing_zero_style", rows)


def run_exact_rate_deepjscc(
    alignment: dict[str, Any],
    pilot: dict[str, Any],
    samples: list[dict[str, Any]],
    classes: list[str],
    output: Path,
) -> dict[str, Any]:
    """Evaluate the trained exact-author-rate DeepJSCC as its own reference."""

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    method = alignment["methods"]["exact_rate_deepjscc"]
    checkpoint_path = require_sha(method["checkpoint"], method["checkpoint_sha256"])
    train_config = load_yaml("configs/external_author_rate_deepjscc_train_stable.yaml")
    base = build_deepjscc_model(
        resolve(train_config["baseline"]["repo"]),
        int(method["inner_channel"]),
        "AWGN",
        float(alignment["channel"]["snrs_db"][0]),
    )
    model = ExactRateMaskedDeepJSCC(
        base,
        dense_symbols=int(method["dense_real_symbols"]),
        active_symbols=int(method["active_real_symbols"]),
        snr_db=float(alignment["channel"]["snrs_db"][0]),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().requires_grad_(False)

    eval_cfg = evaluation_config(pilot)
    evaluator_path = require_sha(
        pilot["evaluator"]["checkpoint"], pilot["evaluator"]["checkpoint_sha256"]
    )
    evaluator, evaluator_temperature = load_scratch_classifier(
        str(evaluator_path), classes, device, str(pilot["evaluator"]["expected_role"])
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
    if not bool(((original_prediction == labels) & (original_confidence >= 0.5)).all()):
        raise RuntimeError("frozen author-rate pilot includes a non-clean T_cls image")

    active_symbols = int(alignment["rate"]["total_real_symbols"])
    rows: list[dict[str, Any]] = []
    for snr in map(float, alignment["channel"]["snrs_db"]):
        noises_cpu = torch.stack(
            [
                canonical_standard_normal(
                    int(alignment["channel"]["base_seed"]),
                    str(item["sample_id"]),
                    snr,
                    active_symbols,
                )
                for item in samples
            ]
        )
        noise_shas = [canonical_noise_sha256(row) for row in noises_cpu]
        model.snr_db = snr
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            reconstruction = model.forward_with_standard_normal(
                target, noises_cpu.to(device)
            ).clamp(0.0, 1.0)
        torch.cuda.synchronize(device)
        runtime_ms = (time.perf_counter() - started) * 1000.0 / len(samples)
        with torch.no_grad():
            probability = evaluate_probabilities(
                evaluator, evaluator_temperature, reconstruction, eval_cfg
            )
            prediction = probability.argmax(dim=1)
            metrics = metric_tensors(target, reconstruction, lpips_model)
        save_image(
            torch.cat([target, reconstruction]).cpu(),
            output / f"snr_{int(snr):02d}_source_exact_rate_deepjscc.png",
            nrow=len(samples),
        )
        peak = torch.cuda.max_memory_allocated(device) / (1024**2)
        for index, item in enumerate(samples):
            label = int(labels[index])
            final_correct = int(prediction[index]) == label
            rows.append(
                {
                    "method": "deepjscc_exact_author_rate",
                    "sample_id": item["sample_id"],
                    "wnid": item["wnid"],
                    "class_idx": label,
                    "snr_db": snr,
                    "base_seed": int(alignment["channel"]["base_seed"]),
                    "canonical_noise_sha256": noise_shas[index],
                    "noise_variance_convention": "complex_awgn_per_real_half_variance",
                    "total_real_symbols": active_symbols,
                    "total_complex_channel_uses": active_symbols // 2,
                    "cbr": float(alignment["rate"]["exact_cbr"]),
                    "original_prediction": int(original_prediction[index]),
                    "original_confidence": float(original_confidence[index]),
                    "clean_correct": True,
                    "deepjscc_prediction": int(prediction[index]),
                    "deepjscc_correct": final_correct,
                    "final_prediction": int(prediction[index]),
                    "final_correct": final_correct,
                    "final_failure": not final_correct,
                    "new_error_vs_deepjscc": False,
                    "repair_vs_deepjscc": False,
                    "deepjscc_psnr": float(metrics[0][index]),
                    "deepjscc_ms_ssim": float(metrics[1][index]),
                    "deepjscc_lpips": float(metrics[2][index]),
                    "final_psnr": float(metrics[0][index]),
                    "final_ms_ssim": float(metrics[1][index]),
                    "final_lpips": float(metrics[2][index]),
                    "runtime_ms_per_image": runtime_ms,
                    "peak_gpu_memory_mib": peak,
                }
            )
    write_csv(output / "per_sample.csv", rows)
    summary = {
        "method": "deepjscc_exact_author_rate",
        "status": "PASS",
        "rows": len(rows),
        "images": len(samples),
        "snrs_db": alignment["channel"]["snrs_db"],
        "pilot_claim_scope": "integration_and_direction_only",
        "outcome_claims_allowed": False,
        "exact_total_real_symbols": active_symbols,
        "exact_cbr": float(alignment["rate"]["exact_cbr"]),
        "mean_final_psnr": sum(float(row["final_psnr"]) for row in rows) / len(rows),
        "mean_final_ms_ssim": sum(float(row["final_ms_ssim"]) for row in rows) / len(rows),
        "mean_final_lpips": sum(float(row["final_lpips"]) for row in rows) / len(rows),
        "final_failures": sum(bool(row["final_failure"]) for row in rows),
        "mean_runtime_ms_per_image": sum(float(row["runtime_ms_per_image"]) for row in rows)
        / len(rows),
        "peak_gpu_memory_mib": max(float(row["peak_gpu_memory_mib"]) for row in rows),
    }
    summary["by_snr"] = {
        str(int(snr)): {
            "mean_final_psnr": sum(
                float(row["final_psnr"]) for row in rows if float(row["snr_db"]) == snr
            )
            / len(samples),
            "mean_final_ms_ssim": sum(
                float(row["final_ms_ssim"])
                for row in rows
                if float(row["snr_db"]) == snr
            )
            / len(samples),
            "mean_final_lpips": sum(
                float(row["final_lpips"]) for row in rows if float(row["snr_db"]) == snr
            )
            / len(samples),
            "final_failures": sum(
                bool(row["final_failure"]) for row in rows if float(row["snr_db"]) == snr
            ),
        }
        for snr in map(float, alignment["channel"]["snrs_db"])
    }
    return summary


def summarize_rows(method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 40:
        raise RuntimeError(f"{method} produced {len(rows)} rows, expected 40")
    if any(int(row["total_real_symbols"]) != 65536 for row in rows):
        raise RuntimeError("rate gate failed")
    summary: dict[str, Any] = {
        "method": method,
        "status": "PASS",
        "rows": len(rows),
        "images": len({row["sample_id"] for row in rows}),
        "snrs_db": sorted({float(row["snr_db"]) for row in rows}),
        "pilot_claim_scope": "integration_and_direction_only",
        "outcome_claims_allowed": False,
        "mean_final_psnr": sum(float(row["final_psnr"]) for row in rows) / len(rows),
        "mean_final_ms_ssim": sum(float(row["final_ms_ssim"]) for row in rows) / len(rows),
        "mean_final_lpips": sum(float(row["final_lpips"]) for row in rows) / len(rows),
        "final_failures": sum(str(row["final_failure"]).lower() == "true" for row in rows),
        "new_errors_vs_deepjscc": sum(
            str(row["new_error_vs_deepjscc"]).lower() == "true" for row in rows
        ),
        "repairs_vs_deepjscc": sum(
            str(row["repair_vs_deepjscc"]).lower() == "true" for row in rows
        ),
        "mean_runtime_ms_per_image": sum(
            float(row["runtime_ms_per_image"]) for row in rows
        )
        / len(rows),
        "peak_gpu_memory_mib": max(float(row["peak_gpu_memory_mib"]) for row in rows),
    }
    by_snr: dict[str, Any] = {}
    for snr in summary["snrs_db"]:
        subset = [row for row in rows if float(row["snr_db"]) == snr]
        by_snr[str(int(snr))] = {
            "mean_final_psnr": sum(float(row["final_psnr"]) for row in subset) / len(subset),
            "mean_final_ms_ssim": sum(float(row["final_ms_ssim"]) for row in subset)
            / len(subset),
            "mean_final_lpips": sum(float(row["final_lpips"]) for row in subset) / len(subset),
            "final_failures": sum(bool(row["final_failure"]) for row in subset),
            "new_errors_vs_deepjscc": sum(bool(row["new_error_vs_deepjscc"]) for row in subset),
            "repairs_vs_deepjscc": sum(bool(row["repair_vs_deepjscc"]) for row in subset),
        }
    summary["by_snr"] = by_snr
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/external_common_comparison_pilot.yaml")
    parser.add_argument(
        "--method",
        choices=("ours", "sing-zero-style", "exact-rate-deepjscc"),
        required=True,
    )
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = load_yaml(config_path)
    author_rate = args.method == "exact-rate-deepjscc"
    pilot_config = load_yaml(config["population_reference_config"]) if author_rate else config
    samples, classes = validate_and_load_samples(pilot_config)
    if author_rate:
        output = resolve(config["outputs"]["deepjscc"])
    else:
        method_key = "ours" if args.method == "ours" else "sing_zero_style"
        output = resolve(config["outputs"][method_key])
    dry = {
        "analysis_id": config["analysis_id"],
        "method": args.method,
        "output": str(output),
        "sample_ids": [item["sample_id"] for item in samples],
        "snrs_db": config["channel"]["snrs_db"],
        "rows_expected": len(samples) * len(config["channel"]["snrs_db"]),
        "official_val_accessed": False,
        "outcome_claims_allowed": False,
    }
    if not args.run:
        print(json.dumps(dry, ensure_ascii=False, indent=2))
        return
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output / "config_snapshot.yaml")
    if args.method == "ours":
        summary = run_ours(config, samples, classes, output)
    elif author_rate:
        summary = run_exact_rate_deepjscc(config, pilot_config, samples, classes, output)
    else:
        summary = run_sing_style(config, samples, classes, output)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
