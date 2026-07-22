#!/usr/bin/env python3
"""Run the exact-rate low-rate M3 closure on the frozen 8x5 pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from cadsd_jscc.deepjscc_adapter import build_deepjscc_model  # noqa: E402
from cadsd_jscc.external_common import (  # noqa: E402
    canonical_noise_sha256,
    canonical_standard_normal,
)
from cadsd_jscc.external_rate_alignment import ExactRateMaskedDeepJSCC  # noqa: E402
from cadsd_jscc.metrics import ms_ssim_per_sample, psnr_per_sample  # noqa: E402
from cadsd_jscc.semantic_sketch import (  # noqa: E402
    bits_to_integer_codes,
    integer_codes_to_bits,
    quantize_probabilities_uniform,
    reserved_symbol_indices,
)
from external_common_project_pilot import (  # noqa: E402
    evaluation_config,
    require_sha,
    validate_and_load_samples,
)
from pc_imagenette_sender_inbudget_awgn_audit import (  # noqa: E402
    cross_model_triplet_acceptance,
)
from pc_imagenette_supervised_audit import (  # noqa: E402
    evaluate_probabilities,
    load_scratch_classifier,
    source_semantic_score_tensors,
)
from s10_short_chain_residual_shift_diffusion import (  # noqa: E402
    ShortChainResidualShiftDiffusion,
)
from s5_residual_refiner_pilot import (  # noqa: E402
    build_model,
    gate_tensor,
    try_load_lpips,
)


STAGES = ("reference", "b0", "anchor", "raw", "posterior", "final")


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_yaml(value: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(resolve(value).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"YAML root must be a mapping: {value}")
    return payload


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty rows")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalized_payload_active(
    model: ExactRateMaskedDeepJSCC,
    candidate: torch.Tensor,
    reserved: torch.Tensor,
    repeated_payload: torch.Tensor,
) -> torch.Tensor:
    active, _ = model.encode_active(candidate)
    active = active.clone()
    active[:, reserved] = repeated_payload.to(active.dtype)
    norm = active.float().square().sum(dim=1, keepdim=True).clamp_min(1e-12).sqrt()
    return active * (model.active_symbols**0.5 / norm).to(active.dtype)


def active_consistency_per_sample(
    model: ExactRateMaskedDeepJSCC,
    candidate: torch.Tensor,
    received_erased: torch.Tensor,
    valid_mask: torch.Tensor,
    reserved: torch.Tensor,
    repeated_payload: torch.Tensor,
) -> torch.Tensor:
    encoded = normalized_payload_active(model, candidate, reserved, repeated_payload)
    mask = valid_mask.reshape(1, -1).expand(candidate.shape[0], -1)
    count = mask.sum(dim=1).to(encoded.dtype)
    received = received_erased.detach()
    scale = ((received.square() * mask).sum(dim=1) / count).clamp_min(1e-8)
    error = ((encoded - received).square() * mask).sum(dim=1) / count
    return error / scale


def posterior_correct(
    model: ExactRateMaskedDeepJSCC,
    raw: torch.Tensor,
    received_erased: torch.Tensor,
    valid_mask: torch.Tensor,
    reserved: torch.Tensor,
    repeated_payload: torch.Tensor,
    steps: int,
    step_size: float,
) -> torch.Tensor:
    current = raw.detach()
    for _ in range(int(steps)):
        current.requires_grad_(True)
        loss = active_consistency_per_sample(
            model,
            current,
            received_erased,
            valid_mask,
            reserved,
            repeated_payload,
        ).mean()
        gradient = torch.autograd.grad(loss, current)[0]
        rms = gradient.square().flatten(1).mean(dim=1).sqrt().clamp_min(1e-12)
        current = (
            current - float(step_size) * gradient / rms[:, None, None, None]
        ).clamp(0.0, 1.0).detach()
    return current


@torch.no_grad()
def quality_metrics(
    target: torch.Tensor,
    candidate: torch.Tensor,
    lpips_model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        "psnr": psnr_per_sample(candidate, target),
        "ms_ssim": ms_ssim_per_sample(candidate, target),
        "lpips": lpips_model(candidate * 2.0 - 1.0, target * 2.0 - 1.0).flatten(),
    }


def mean(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)


def summarize(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    if len(rows) != 40:
        raise RuntimeError(f"expected 40 rows, got {len(rows)}")
    total = int(config["rate"]["total_real_symbols"])
    if any(int(row["total_real_symbols"]) != total for row in rows):
        raise RuntimeError("exact-rate row gate failed")
    summary: dict[str, Any] = {
        "analysis_id": config["analysis_id"],
        "rows": len(rows),
        "images": len({row["sample_id"] for row in rows}),
        "snrs_db": sorted({float(row["snr_db"]) for row in rows}),
        "total_real_symbols": total,
        "payload_real_symbols": int(config["method"]["payload_real_symbols"]),
        "payload_bit_error_rate": sum(int(row["payload_bit_errors"]) for row in rows)
        / (len(rows) * int(config["method"]["payload_bits"])),
        "payload_exact_rows": sum(bool(row["payload_exact"]) for row in rows),
        "accept_rows": sum(bool(row["accepted"]) for row in rows),
        "mean_consistency_before": mean(rows, "consistency_before"),
        "mean_consistency_after": mean(rows, "consistency_after"),
        "posterior_consistency_increased_rows": sum(
            float(row["consistency_after"]) > float(row["consistency_before"]) + 1e-9
            for row in rows
        ),
    }
    for stage in STAGES:
        summary[f"mean_{stage}_psnr"] = mean(rows, f"{stage}_psnr")
        summary[f"mean_{stage}_ms_ssim"] = mean(rows, f"{stage}_ms_ssim")
        summary[f"mean_{stage}_lpips"] = mean(rows, f"{stage}_lpips")
        summary[f"{stage}_failures"] = sum(not bool(row[f"{stage}_correct"]) for row in rows)
    for stage in ("raw", "posterior", "final"):
        summary[f"{stage}_new_errors_vs_anchor"] = sum(
            bool(row["anchor_correct"]) and not bool(row[f"{stage}_correct"])
            for row in rows
        )
        summary[f"{stage}_repairs_vs_anchor"] = sum(
            not bool(row["anchor_correct"]) and bool(row[f"{stage}_correct"])
            for row in rows
        )
    criteria = config["success_criteria"]
    checks = {
        "rows_complete": len(rows) == 40,
        "exact_rate": all(int(row["total_real_symbols"]) == total for row in rows),
        "posterior_consistency_nonincreasing": summary[
            "posterior_consistency_increased_rows"
        ]
        == 0,
        "final_psnr_within_budget": summary["mean_final_psnr"]
        - summary["mean_anchor_psnr"]
        >= float(criteria["final_mean_psnr_minus_anchor_min_db"]),
        "final_lpips_not_worse": summary["mean_final_lpips"]
        - summary["mean_anchor_lpips"]
        <= float(criteria["final_mean_lpips_minus_anchor_max"]),
        "final_new_not_greater_than_repair": summary["final_new_errors_vs_anchor"]
        <= summary["final_repairs_vs_anchor"],
        "final_failure_not_greater_than_anchor": summary["final_failures"]
        <= summary["anchor_failures"],
    }
    summary["checks"] = checks
    summary["all_pass"] = all(checks.values())
    by_snr: dict[str, Any] = {}
    for snr in summary["snrs_db"]:
        subset = [row for row in rows if float(row["snr_db"]) == snr]
        values: dict[str, Any] = {"rows": len(subset), "accepted": sum(bool(r["accepted"]) for r in subset)}
        for stage in STAGES:
            values[f"mean_{stage}_psnr"] = mean(subset, f"{stage}_psnr")
            values[f"mean_{stage}_lpips"] = mean(subset, f"{stage}_lpips")
            values[f"{stage}_failures"] = sum(
                not bool(row[f"{stage}_correct"]) for row in subset
            )
        by_snr[str(int(snr))] = values
    summary["by_snr"] = by_snr
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lowrate_m3_stage_pilot.yaml")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = load_yaml(config_path)
    if config.get("status") != "preregistered_before_stage_output":
        raise RuntimeError("stage pilot must be preregistered before output")
    if config.get("official_val_accessed") is not False:
        raise RuntimeError("official validation must remain sealed")
    population_config = load_yaml(config["population_reference_config"])
    samples, classes = validate_and_load_samples(population_config)
    output = resolve(config["outputs"]["output_dir"])
    dry = {
        "analysis_id": config["analysis_id"],
        "samples": [item["sample_id"] for item in samples],
        "snrs_db": config["channel"]["snrs_db"],
        "rows_expected": len(samples) * len(config["channel"]["snrs_db"]),
        "rate": config["rate"],
        "output": str(output),
    }
    if not args.run:
        print(json.dumps(dry, ensure_ascii=False, indent=2))
        return
    if output.exists():
        raise FileExistsError(f"output exists, refusing to overwrite: {output}")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output / "config_snapshot.yaml")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    method = config["method"]
    checkpoint_path = require_sha(
        method["exact_checkpoint"], method["exact_checkpoint_sha256"]
    )
    base = build_deepjscc_model(
        resolve(method["baseline_repo"]),
        int(method["inner_channel"]),
        "AWGN",
        float(config["channel"]["snrs_db"][0]),
    )
    jscc = ExactRateMaskedDeepJSCC(
        base,
        dense_symbols=int(config["rate"]["dense_real_symbols"]),
        active_symbols=int(config["rate"]["active_real_symbols"]),
        snr_db=float(config["channel"]["snrs_db"][0]),
    ).to(device)
    jscc.load_state_dict(torch.load(checkpoint_path, map_location=device)["model"], strict=True)
    jscc.eval().requires_grad_(False)

    b1_config = load_yaml(method["b1_config"])
    b1 = build_model(b1_config).to(device)
    b1_path = require_sha(method["b1_checkpoint"], method["b1_checkpoint_sha256"])
    b1.load_state_dict(torch.load(b1_path, map_location=device)["model_state_dict"], strict=True)
    b1.eval().requires_grad_(False)
    diffusion_config = load_yaml(method["diffusion_config"])
    diffusion = ShortChainResidualShiftDiffusion(diffusion_config).to(device)
    diffusion_path = require_sha(
        method["diffusion_checkpoint"], method["diffusion_checkpoint_sha256"]
    )
    diffusion.load_state_dict(
        torch.load(diffusion_path, map_location=device)["model_state_dict"], strict=True
    )
    diffusion.eval().requires_grad_(False)

    eval_cfg = evaluation_config(population_config)
    evaluator_cfg = population_config["evaluator"]
    evaluator_path = require_sha(
        config["evaluator"]["checkpoint"], config["evaluator"]["checkpoint_sha256"]
    )
    evaluator, evaluator_temperature = load_scratch_classifier(
        str(evaluator_path), classes, device, str(config["evaluator"]["expected_role"])
    )
    sender_path = require_sha(method["sender_checkpoint"], method["sender_checkpoint_sha256"])
    sender, sender_temperature = load_scratch_classifier(
        str(sender_path), classes, device, "G_aux"
    )
    gate_path = require_sha(method["gate_checkpoint"], method["gate_checkpoint_sha256"])
    receiver_guard, receiver_guard_temperature = load_scratch_classifier(
        str(gate_path), classes, device, "G_gate"
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
    if not bool(((original_prediction == labels) & (original_confidence >= float(evaluator_cfg["clean_confidence_threshold"]))).all()):
        raise RuntimeError("frozen pilot contains a non-clean evaluator sample")

    active_symbols = int(config["rate"]["active_real_symbols"])
    payload_bits = int(method["payload_bits"])
    repetitions = int(method["payload_repetitions"])
    reserved = reserved_symbol_indices(
        active_symbols, payload_bits * repetitions, device=device
    )
    valid_mask = torch.ones(active_symbols, dtype=torch.bool, device=device)
    valid_mask[reserved] = False
    rows: list[dict[str, Any]] = []
    for snr in map(float, config["channel"]["snrs_db"]):
        noises_cpu = torch.stack(
            [
                canonical_standard_normal(
                    int(config["channel"]["base_seed"]),
                    str(item["sample_id"]),
                    snr,
                    active_symbols,
                )
                for item in samples
            ]
        )
        noise_shas = [canonical_noise_sha256(row) for row in noises_cpu]
        noises = noises_cpu.to(device)
        jscc.snr_db = snr
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.no_grad():
            active, dense_shape = jscc.encode_active(target)
            reference_received = jscc.transmit_active(active, noises)
            reference = jscc.decode_active(reference_received, dense_shape).clamp(0.0, 1.0)
            source_probability = evaluate_probabilities(
                sender, sender_temperature, target, eval_cfg
            )
            source_codes, _ = quantize_probabilities_uniform(source_probability, 2)
            source_bits = integer_codes_to_bits(source_codes, 2)
            payload = (
                source_bits.to(active.dtype)
                .mul(2.0)
                .sub(1.0)
                .unsqueeze(-1)
                .expand(-1, -1, repetitions)
                .reshape(len(samples), -1)
            )
            transmitted = active.clone()
            transmitted[:, reserved] = payload
            norm = transmitted.float().square().sum(dim=1, keepdim=True).clamp_min(1e-12).sqrt()
            transmitted = transmitted * math.sqrt(active_symbols) / norm.to(transmitted.dtype)
            received = jscc.transmit_active(transmitted, noises)
            recovered = received[:, reserved].reshape(len(samples), payload_bits, repetitions).mean(2)
            recovered_bits = (recovered > 0).to(torch.int64)
            recovered_codes = bits_to_integer_codes(recovered_bits, 10, 2)
            decoded = recovered_codes.to(source_probability.dtype)
            decoded_total = decoded.sum(dim=1, keepdim=True)
            recovered_probability = torch.where(
                decoded_total > 0,
                decoded / decoded_total.clamp_min(1.0),
                torch.full_like(decoded, 0.1),
            )
            recovered_payload = (
                recovered_bits.to(active.dtype)
                .mul(2.0)
                .sub(1.0)
                .unsqueeze(-1)
                .expand(-1, -1, repetitions)
                .reshape(len(samples), -1)
            )
            erased_received = received.clone()
            erased_received[:, reserved] = 0.0
            b0 = jscc.decode_active(erased_received, dense_shape).clamp(0.0, 1.0)
            snr_tensor = torch.full((len(samples),), snr, device=device)
            snr_norm = snr_tensor / float(b1_config["model"]["snr_norm_max"])
            anchor = b1(b0, snr_norm, gate_tensor(b1_config, snr_tensor, device))
            raw = diffusion(
                anchor,
                snr_norm,
                gate_tensor(diffusion_config, snr_tensor, device),
            )
            consistency_before = active_consistency_per_sample(
                jscc, raw, erased_received, valid_mask, reserved, recovered_payload
            )
        posterior = posterior_correct(
            jscc,
            raw,
            erased_received,
            valid_mask,
            reserved,
            recovered_payload,
            int(method["posterior_steps"]),
            float(method["posterior_normalized_step_size"]),
        )
        with torch.no_grad():
            consistency_after = active_consistency_per_sample(
                jscc, posterior, erased_received, valid_mask, reserved, recovered_payload
            )
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
            sender_accepted, receiver_accepted, source_anchor_accepted, accepted = (
                cross_model_triplet_acceptance(
                    risk,
                    0.0,
                    recovered_probability,
                    gate_anchor_probability,
                    gate_posterior_probability,
                )
            )
            semantic_accepted = accepted
            activation_min_snr = float(method.get("activation_min_snr_db", -math.inf))
            channel_enabled = snr >= activation_min_snr
            accepted = semantic_accepted & channel_enabled
            final = torch.where(accepted[:, None, None, None], posterior, anchor)
        torch.cuda.synchronize(device)
        runtime_ms = (time.perf_counter() - started) * 1000.0 / len(samples)
        candidates = {
            "reference": reference,
            "b0": b0,
            "anchor": anchor,
            "raw": raw,
            "posterior": posterior,
            "final": final,
        }
        probabilities: dict[str, torch.Tensor] = {}
        metrics: dict[str, dict[str, torch.Tensor]] = {}
        with torch.no_grad():
            for stage, candidate in candidates.items():
                probabilities[stage] = evaluate_probabilities(
                    evaluator, evaluator_temperature, candidate, eval_cfg
                )
                metrics[stage] = quality_metrics(target, candidate, lpips_model)
        save_image(
            torch.cat([target, *[candidates[stage] for stage in STAGES]]).cpu(),
            output / f"snr_{int(snr):02d}_source_ref_b0_anchor_raw_post_final.png",
            nrow=len(samples),
        )
        peak_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
        for index, item in enumerate(samples):
            row: dict[str, Any] = {
                "analysis_id": config["analysis_id"],
                "sample_id": item["sample_id"],
                "wnid": item["wnid"],
                "class_idx": int(labels[index]),
                "snr_db": snr,
                "base_seed": int(config["channel"]["base_seed"]),
                "canonical_noise_sha256": noise_shas[index],
                "total_real_symbols": active_symbols,
                "total_complex_channel_uses": active_symbols // 2,
                "cbr": float(config["rate"]["exact_cbr"]),
                "payload_real_symbols": int(method["payload_real_symbols"]),
                "image_active_real_symbols": int(config["rate"]["image_active_real_symbols"]),
                "payload_bit_errors": int((source_bits[index] != recovered_bits[index]).sum()),
                "payload_exact": bool(torch.equal(source_bits[index], recovered_bits[index])),
                "accepted": bool(accepted[index]),
                "semantic_accepted": bool(semantic_accepted[index]),
                "channel_enabled": channel_enabled,
                "sender_accepted": bool(sender_accepted[index]),
                "receiver_guard_accepted": bool(receiver_accepted[index]),
                "source_anchor_accepted": bool(source_anchor_accepted[index]),
                "consistency_before": float(consistency_before[index]),
                "consistency_after": float(consistency_after[index]),
                "runtime_ms_per_image": runtime_ms,
                "peak_gpu_memory_mib": peak_mib,
            }
            for stage in STAGES:
                prediction = int(probabilities[stage][index].argmax())
                row[f"{stage}_prediction"] = prediction
                row[f"{stage}_correct"] = prediction == int(labels[index])
                row[f"{stage}_psnr"] = float(metrics[stage]["psnr"][index])
                row[f"{stage}_ms_ssim"] = float(metrics[stage]["ms_ssim"][index])
                row[f"{stage}_lpips"] = float(metrics[stage]["lpips"][index])
            rows.append(row)
        print(
            json.dumps(
                {
                    "snr_db": snr,
                    "anchor_psnr": float(metrics["anchor"]["psnr"].mean()),
                    "raw_psnr": float(metrics["raw"]["psnr"].mean()),
                    "posterior_psnr": float(metrics["posterior"]["psnr"].mean()),
                    "final_psnr": float(metrics["final"]["psnr"].mean()),
                    "accepted": int(accepted.sum()),
                },
                indent=2,
            )
        )
    write_csv(output / "per_sample.csv", rows)
    summary = summarize(rows, config)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
