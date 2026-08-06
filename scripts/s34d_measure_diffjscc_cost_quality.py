#!/usr/bin/env python3
"""Measure official DiffJSCC latency components and its step/quality curve."""

from __future__ import annotations

import contextlib
import argparse
import csv
import io
import json
import shutil
import subprocess
import sys
import time
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import einops
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from cadsd_jscc.external_common import (  # noqa: E402
    canonical_standard_normal,
)
from pc_imagenette_supervised_audit import (  # noqa: E402
    evaluate_probabilities,
    load_scratch_classifier,
)
from s30_diffjscc_external_comparison import (  # noqa: E402
    author_process,
    checkpoint_state,
    evaluator_config,
    instantiate_official_model,
    load_population,
    metric_triplet,
    sampler_seed,
)
from model.spaced_sampler import SpacedSampler  # type: ignore  # noqa: E402
from s30_diffjscc_preflight import sha256_file  # noqa: E402
from s5_residual_refiner_pilot import try_load_lpips  # noqa: E402
from utils.image import auto_resize, pad, wavelet_reconstruction  # type: ignore # noqa: E402


SCRIPT = Path(__file__).resolve()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_yaml(value: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(resolve(value).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(value)
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def gpu_inventory() -> dict[str, str]:
    line = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip().splitlines()[0]
    uuid, name, driver = (part.strip() for part in line.split(",", 2))
    return {"uuid": uuid, "name": name, "driver_version": driver}


def sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def timed(device: torch.device, fn: Callable[[], Any]) -> tuple[Any, float]:
    sync(device)
    started = time.perf_counter()
    value = fn()
    sync(device)
    return value, (time.perf_counter() - started) * 1000.0


def profile_flops(fn: Callable[[], Any]) -> int:
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_flops=True,
        record_shapes=True,
    ) as prof:
        fn()
        torch.cuda.synchronize()
    return int(sum(int(event.flops or 0) for event in prof.key_averages()))


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "std": float(array.std(ddof=1)),
    }


def cluster_ci(
    rows: list[dict[str, Any]], field: str, replicates: int, seed: int
) -> list[float]:
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sample_id"])].append(float(row[field]))
    values = np.asarray(
        [np.mean(grouped[key]) for key in sorted(grouped)], dtype=np.float64
    )
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(replicates, len(values)))
    boot = values[indices].mean(axis=1)
    return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def unique_parameter_ledger(model: torch.nn.Module) -> dict[str, Any]:
    components = [
        ("preprocess_jscc", getattr(model, "preprocess_model", None)),
        ("receiver_blip2", getattr(model, "blip_model", None)),
        ("text_conditioner", getattr(model, "cond_stage_model", None)),
        ("diffusion_unet", getattr(getattr(model, "model", None), "diffusion_model", None)),
        ("controlnet", getattr(model, "control_model", None)),
        ("vae", getattr(model, "first_stage_model", None)),
        ("condition_encoder", getattr(model, "cond_encoder", None)),
    ]
    all_unique = {id(p): p.numel() for p in model.parameters()}
    raw: dict[str, int] = {}
    incremental: dict[str, int] = {}
    seen: set[int] = set()
    for name, module in components:
        if module is None:
            raw[name] = 0
            incremental[name] = 0
            continue
        params = list(module.parameters())
        raw[name] = int(sum(p.numel() for p in params))
        new = [p for p in params if id(p) not in seen]
        incremental[name] = int(sum(p.numel() for p in new))
        seen.update(id(p) for p in params)
    incremental["other_registered_parameters"] = int(
        sum(count for pid, count in all_unique.items() if pid not in seen)
    )
    return {
        "unique_live_parameters": int(sum(all_unique.values())),
        "raw_component_counts_may_overlap": raw,
        "deduplicated_incremental_counts_in_listed_order": incremental,
    }


@torch.inference_mode()
def prepare_frontend(
    model: torch.nn.Module,
    source_pil: Image.Image,
    standard_normal: torch.Tensor,
    snr: float,
    device: torch.device,
    measure: bool,
) -> tuple[dict[str, Any], dict[str, float]]:
    timing: dict[str, float] = {}

    def preprocess() -> tuple[np.ndarray, Image.Image, torch.Tensor]:
        resized = auto_resize(source_pil, 512)
        control = pad(np.array(resized), scale=64)
        img_t = torch.tensor(
            np.stack([control]) / 255.0, dtype=torch.float32, device=device
        ).clamp_(0, 1)
        img_t = einops.rearrange(img_t, "n h w c -> n c h w").contiguous()
        return control, resized, img_t

    if measure:
        (control, resized, img_t), timing["preprocess_resize_h2d_ms"] = timed(
            device, preprocess
        )
    else:
        control, resized, img_t = preprocess()
    model.preprocess_model._s30_snr_db = float(snr)
    model.preprocess_model._s30_standard_normal = standard_normal.unsqueeze(0)
    if measure:
        (img_init, cond_snr, _), timing["jscc_frontend_ms"] = timed(
            device, lambda: model.preprocess_model(img_t)
        )
    else:
        img_init, cond_snr, _ = model.preprocess_model(img_t)
    input_img = [model.transform_to_pil(img_init[0])]

    def caption_generate() -> str:
        inputs = model.processor(
            images=input_img, return_tensors="pt", max_length=32
        ).to(device, torch.float16)
        generated_ids = model.blip_model.generate(**inputs)
        generated = model.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return str(generated[0])

    if measure:
        caption, timing["blip2_caption_ms"] = timed(device, caption_generate)
        cond_text, timing["text_conditioning_ms"] = timed(
            device, lambda: model.get_learned_conditioning([caption])
        )
    else:
        caption = caption_generate()
        cond_text = model.get_learned_conditioning([caption])
    return {
        "control": control,
        "resized": resized,
        "img_t": img_t,
        "img_init": img_init,
        "cond_snr": cond_snr,
        "caption": caption,
        "cond_text": cond_text,
    }, timing


@torch.inference_mode()
def sample_timed(
    model: torch.nn.Module,
    prepared: dict[str, Any],
    steps: int,
    rng_seed: int,
    device: torch.device,
    measure: bool,
) -> tuple[torch.Tensor, dict[str, float], dict[str, Any]]:
    timing: dict[str, float] = {}
    sampler = SpacedSampler(model, var_type="fixed_small")

    def make_schedule() -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            sampler.make_schedule(num_steps=int(steps))

    if measure:
        _, timing["sampler_schedule_ms"] = timed(device, make_schedule)
    else:
        make_schedule()
    torch.manual_seed(int(rng_seed))
    torch.cuda.manual_seed_all(int(rng_seed))
    shape = (
        1,
        4,
        prepared["img_t"].shape[-2] // 8,
        prepared["img_t"].shape[-1] // 8,
    )

    def condition_and_init() -> tuple[dict[str, Any], torch.Tensor]:
        condition = {
            "c_spatial": [model.apply_condition_encoder(prepared["img_init"])],
            "c_textual": [prepared["cond_text"]],
            "c_snr": [prepared["cond_snr"]],
        }
        latent = torch.randn(shape, device=device)
        return condition, latent

    if measure:
        (condition, latent), timing["condition_encoder_and_init_ms"] = timed(
            device, condition_and_init
        )
    else:
        condition, latent = condition_and_init()
    time_range = np.flip(sampler.timesteps)
    total_steps = len(sampler.timesteps)

    def denoise_loop() -> torch.Tensor:
        current = latent
        for index_in_loop, step in enumerate(time_range):
            ts = torch.full((1,), step, device=device, dtype=torch.long)
            index = torch.full_like(ts, fill_value=total_steps - index_in_loop - 1)
            current = sampler.p_sample(
                current,
                condition,
                ts,
                index=index,
                cfg_scale=1.0,
                uncond=None,
                cond_fn=None,
            )
        return current

    if measure:
        latent_final, timing["denoiser_loop_ms"] = timed(device, denoise_loop)
    else:
        latent_final = denoise_loop()
    if measure:
        decoded, timing["vae_decode_ms"] = timed(
            device, lambda: (model.decode_first_stage(latent_final) + 1) / 2
        )
        corrected, timing["wavelet_color_fix_ms"] = timed(
            device, lambda: wavelet_reconstruction(decoded, prepared["img_init"])
        )
    else:
        decoded = (model.decode_first_stage(latent_final) + 1) / 2
        corrected = wavelet_reconstruction(decoded, prepared["img_init"])
    return corrected.clamp(0, 1), timing, {
        "sampler": sampler,
        "condition": condition,
        "latent_before_loop": latent,
        "latent_final": latent_final,
    }


def output_to_256(
    output: torch.Tensor, resized: Image.Image, source_pil: Image.Image
) -> np.ndarray:
    pred_np = (
        einops.rearrange(output, "b c h w -> b h w c")[0]
        .mul(255)
        .cpu()
        .numpy()
        .clip(0, 255)
        .astype(np.uint8)
    )
    pred_np = pred_np[: resized.height, : resized.width]
    return np.array(
        Image.fromarray(pred_np).resize(
            source_pil.size, Image.Resampling.LANCZOS
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config_path = resolve("configs/s34d_generative_inference_cost.yaml")
    config = load_yaml(config_path)
    if config["status"] != "preregistered_and_authorized_before_measurement":
        raise RuntimeError("S34D is not authorized")
    if any(
        (
            config["protocol"]["new_training"],
            config["protocol"]["network_download"],
            config["protocol"]["official_imagenette_validation_accessed"],
        )
    ):
        raise RuntimeError("measurement-only contract changed")
    inventory = gpu_inventory()
    if inventory["uuid"] != config["hardware"]["required_gpu_uuid"]:
        raise RuntimeError(f"wrong GPU: {inventory}")
    device = torch.device(config["hardware"]["device"])
    torch.backends.cudnn.benchmark = bool(config["hardware"]["cudnn_benchmark"])
    s30_config_path = resolve(config["inputs"]["s30_config"])
    if sha256_file(s30_config_path) != config["inputs"]["s30_config_sha256"]:
        raise RuntimeError("S30 config changed")
    s30 = load_yaml(s30_config_path)
    samples, classes = load_population(s30)
    state = checkpoint_state(s30)
    model, load_audit = instantiate_official_model(s30, state)
    del state
    model = model.to(device).eval().requires_grad_(False)
    model.preprocess_model._s30_expected_real_symbols = int(
        config["fairness_boundaries"]["diffjscc_real_symbols"]
    )
    from s30_diffjscc_external_comparison import fixed_deepjscc_forward

    # Retain the exact S30 channel implementation already validated against
    # historical output.
    model.preprocess_model.forward = types.MethodType(
        fixed_deepjscc_forward, model.preprocess_model
    )

    source_transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()]
    )
    to_tensor = transforms.ToTensor()
    host_pils = [
        transforms.ToPILImage()(
            source_transform(Image.open(item["path"]).convert("RGB"))
        )
        for item in samples
    ]
    steps_list = list(map(int, config["quality_curve"]["sampling_steps"]))
    snrs = list(map(float, config["quality_curve"]["snrs_db"]))
    base_seed = int(config["quality_curve"]["channel_seeds"][0])
    latency_count = int(config["latency_contract"]["timed_source_images"])

    def standard_noise(index: int, snr: float) -> torch.Tensor:
        full = canonical_standard_normal(
            base_seed,
            str(samples[index]["sample_id"]),
            snr,
            19712,
        )
        return full[: int(config["fairness_boundaries"]["diffjscc_real_symbols"])]

    if args.preflight:
        prepared, _ = prepare_frontend(
            model, host_pils[0], standard_noise(0, 7.0), 7.0, device, False
        )
        restored, _, _ = sample_timed(
            model,
            prepared,
            4,
            sampler_seed(base_seed, str(samples[0]["sample_id"]), 7.0),
            device,
            False,
        )
        pred = output_to_256(restored, prepared["resized"], host_pils[0])
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "output_shape": list(pred.shape),
                    "caption": prepared["caption"],
                    "parameters": unique_parameter_ledger(model),
                    "gpu": inventory,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    output = resolve(config["outputs"]["diffjscc"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    shutil.copy2(config_path, output / "config_snapshot.yaml")
    shutil.copy2(SCRIPT, output / SCRIPT.name)

    @torch.inference_mode()
    def latency_once(index: int, snr: float, steps: int, collect: bool) -> dict[str, Any]:
        source_pil = host_pils[index]
        noise = standard_noise(index, snr)
        sync(device)
        wall_started = time.perf_counter()
        prepared, front_timing = prepare_frontend(
            model, source_pil, noise, snr, device, True
        )
        restored, sample_timing, _ = sample_timed(
            model,
            prepared,
            steps,
            sampler_seed(base_seed, str(samples[index]["sample_id"]), snr),
            device,
            True,
        )
        host_output, output_ms = timed(
            device, lambda: output_to_256(restored, prepared["resized"], source_pil)
        )
        wall_ms = (time.perf_counter() - wall_started) * 1000.0
        if host_output.shape != (256, 256, 3):
            raise RuntimeError("DiffJSCC output shape changed")
        if not collect:
            return {}
        components = {**front_timing, **sample_timing, "postprocess_d2h_resize_ms": output_ms}
        return {
            "sample_id": samples[index]["sample_id"],
            "snr_db": snr,
            "steps": steps,
            "batch_size_source_images": 1,
            "receiver_wall_ms": wall_ms,
            **components,
            "component_sum_ms": sum(components.values()),
            "wall_minus_component_sum_ms": wall_ms - sum(components.values()),
        }

    warmup_count = int(config["latency_contract"]["warmup_source_keys"])
    for index in range(warmup_count):
        latency_once(index, snrs[index % len(snrs)], 4, False)
    latency_rows: list[dict[str, Any]] = []
    for steps in steps_list:
        for snr in snrs:
            for index in range(latency_count):
                row = latency_once(index, snr, steps, True)
                latency_rows.append(row)
                if len(latency_rows) % 20 == 0:
                    print(
                    json.dumps(
                        {
                            "stage": "latency",
                            "steps": steps,
                            "rows": len(latency_rows),
                            "wall_ms": row["receiver_wall_ms"],
                        }
                    ),
                        flush=True,
                    )
    write_csv(output / "latency_rows.csv", latency_rows)

    # Metric models are loaded only after latency measurement so they cannot perturb
    # the primary memory/latency path.
    evaluator, evaluator_temperature = load_scratch_classifier(
        str(resolve(s30["inputs"]["t_cls_checkpoint"])),
        classes,
        device,
        str(s30["evaluator"]["expected_role"]),
    )
    lpips_model, lpips_error = try_load_lpips(device, resolve("outputs/cache"))
    if lpips_model is None:
        raise RuntimeError(lpips_error)
    lpips_model.eval().requires_grad_(False)
    eval_cfg = evaluator_config(s30)
    s33_path = resolve(
        "outputs/external_baselines/ANALYSIS-S33-STRONG-JSCC-16384-COMPARISON-001/per_sample.csv"
    )
    s33_rows = read_csv(s33_path)
    s33_by_key = {
        (row["sample_id"], int(row["base_seed"]), float(row["snr_db"])): row
        for row in s33_rows
    }
    historical = read_csv(resolve(config["inputs"]["s30_per_sample"]))
    historical_by_key = {
        (row["sample_id"], int(row["base_seed"]), float(row["snr_db"])): row
        for row in historical
    }

    quality_rows: list[dict[str, Any]] = []
    for snr in snrs:
        for index, item in enumerate(samples):
            source_pil = host_pils[index]
            target = source_transform(source_pil).unsqueeze(0).to(device)
            noise = standard_noise(index, snr)
            prepared, _ = prepare_frontend(
                model, source_pil, noise, snr, device, False
            )
            for steps in steps_list:
                restored, _timing, _internals = sample_timed(
                    model,
                    prepared,
                    steps,
                    sampler_seed(base_seed, str(item["sample_id"]), snr),
                    device,
                    False,
                )
                pred_np = output_to_256(restored, prepared["resized"], source_pil)
                pred = to_tensor(pred_np).unsqueeze(0).to(device)
                psnr, ms_ssim, lpips_value = metric_triplet(
                    target, pred, lpips_model
                )
                with torch.no_grad():
                    probability = evaluate_probabilities(
                        evaluator, evaluator_temperature, pred, eval_cfg
                    )
                prediction = int(probability.argmax(dim=1)[0])
                key = (str(item["sample_id"]), base_seed, snr)
                s33 = s33_by_key[key]
                historical_row = historical_by_key[key]
                row = {
                    "sample_id": item["sample_id"],
                    "wnid": item["wnid"],
                    "class_idx": int(item["class_idx"]),
                    "base_seed": base_seed,
                    "snr_db": snr,
                    "steps": steps,
                    "sampler_seed": sampler_seed(
                        base_seed, str(item["sample_id"]), snr
                    ),
                    "caption": prepared["caption"],
                    "psnr": psnr,
                    "ms_ssim": ms_ssim,
                    "lpips": lpips_value,
                    "prediction": prediction,
                    "failure": prediction != int(item["class_idx"]),
                    "s33_psnr": float(s33["strong_psnr"]),
                    "s33_ms_ssim": float(s33["strong_ms_ssim"]),
                    "s33_lpips": float(s33["strong_lpips"]),
                    "s33_failure": str(s33["strong_failure"]).lower() == "true",
                    "lpips_minus_s33": lpips_value - float(s33["strong_lpips"]),
                    "psnr_minus_s33": psnr - float(s33["strong_psnr"]),
                    "historical_100step_psnr": float(
                        historical_row["diffjscc_psnr"]
                    ),
                    "historical_100step_lpips": float(
                        historical_row["diffjscc_lpips"]
                    ),
                }
                quality_rows.append(row)
                if len(quality_rows) % 50 == 0:
                    print(
                    json.dumps(
                        {
                            "stage": "quality",
                            "rows": len(quality_rows),
                            "steps": steps,
                            "snr": snr,
                            "lpips": lpips_value,
                        }
                    ),
                        flush=True,
                    )
    write_csv(output / "quality_rows.csv", quality_rows)

    # Historical replay audit protects against accidentally benchmarking a changed
    # 100-step algorithm.
    replay_100 = [row for row in quality_rows if int(row["steps"]) == 100]
    max_psnr_replay_error = max(
        abs(float(row["psnr"]) - float(row["historical_100step_psnr"]))
        for row in replay_100
    )
    max_lpips_replay_error = max(
        abs(float(row["lpips"]) - float(row["historical_100step_lpips"]))
        for row in replay_100
    )
    if max_psnr_replay_error > 1e-6 or max_lpips_replay_error > 1e-6:
        raise RuntimeError(
            "100-step replay changed: "
            f"psnr={max_psnr_replay_error}, lpips={max_lpips_replay_error}"
        )

    curve: list[dict[str, Any]] = []
    replicates = int(config["quality_curve"]["bootstrap_replicates"])
    bootstrap_seed = int(config["quality_curve"]["bootstrap_seed"])
    for steps in steps_list:
        subset = [row for row in quality_rows if int(row["steps"]) == steps]
        delta_ci = cluster_ci(
            subset, "lpips_minus_s33", replicates, bootstrap_seed + steps
        )
        lpips_delta = float(np.mean([float(row["lpips_minus_s33"]) for row in subset]))
        latency_subset = [
            row for row in latency_rows if int(row["steps"]) == steps
        ]
        curve.append(
            {
                "steps": steps,
                "rows": len(subset),
                "mean_psnr": float(np.mean([float(row["psnr"]) for row in subset])),
                "mean_ms_ssim": float(
                    np.mean([float(row["ms_ssim"]) for row in subset])
                ),
                "mean_lpips": float(np.mean([float(row["lpips"]) for row in subset])),
                "failures": int(sum(bool(row["failure"]) for row in subset)),
                "failure_rate": float(
                    np.mean([float(bool(row["failure"])) for row in subset])
                ),
                "lpips_minus_s33": lpips_delta,
                "lpips_minus_s33_ci95_low": delta_ci[0],
                "lpips_minus_s33_ci95_high": delta_ci[1],
                "retains_significant_lpips_advantage_vs_s33": bool(
                    lpips_delta < 0 and delta_ci[1] < 0
                ),
                "receiver_wall_ms": summarize(
                    [float(row["receiver_wall_ms"]) for row in latency_subset]
                ),
                "denoiser_loop_ms": summarize(
                    [float(row["denoiser_loop_ms"]) for row in latency_subset]
                ),
            }
        )
    write_csv(output / "latency_quality_curve.csv", curve)

    # FLOPs lower-bound profiling on one representative source/SNR. We profile
    # fixed components and one p_sample, then expose the auditable linear formula.
    rep_index = 0
    rep_snr = float(config["flops_contract"]["representative_snr_db"])
    rep_noise = standard_noise(rep_index, rep_snr)
    rep_prepared, _ = prepare_frontend(
        model, host_pils[rep_index], rep_noise, rep_snr, device, False
    )
    jscc_flops = profile_flops(
        lambda: model.preprocess_model(rep_prepared["img_t"])
    )

    def representative_caption() -> None:
        input_img = [model.transform_to_pil(rep_prepared["img_init"][0])]
        inputs = model.processor(
            images=input_img, return_tensors="pt", max_length=32
        ).to(device, torch.float16)
        model.blip_model.generate(**inputs)

    caption_flops = profile_flops(representative_caption)
    text_flops = profile_flops(
        lambda: model.get_learned_conditioning([rep_prepared["caption"]])
    )
    sampler = SpacedSampler(model, var_type="fixed_small")
    with contextlib.redirect_stdout(io.StringIO()):
        sampler.make_schedule(4)
    condition_encoder_flops = profile_flops(
        lambda: model.apply_condition_encoder(rep_prepared["img_init"])
    )
    condition = {
        "c_spatial": [model.apply_condition_encoder(rep_prepared["img_init"])],
        "c_textual": [rep_prepared["cond_text"]],
        "c_snr": [rep_prepared["cond_snr"]],
    }
    latent = torch.randn(
        (1, 4, rep_prepared["img_t"].shape[-2] // 8, rep_prepared["img_t"].shape[-1] // 8),
        device=device,
    )
    ts = torch.full((1,), int(np.flip(sampler.timesteps)[0]), device=device, dtype=torch.long)
    index_tensor = torch.full_like(ts, fill_value=3)
    per_step_flops = profile_flops(
        lambda: sampler.p_sample(
            latent,
            condition,
            ts,
            index=index_tensor,
            cfg_scale=1.0,
            uncond=None,
            cond_fn=None,
        )
    )
    vae_decode_flops = profile_flops(
        lambda: (model.decode_first_stage(latent) + 1) / 2
    )
    fixed_flops = (
        jscc_flops
        + caption_flops
        + text_flops
        + condition_encoder_flops
        + vae_decode_flops
    )

    component_fields = [
        "preprocess_resize_h2d_ms",
        "jscc_frontend_ms",
        "blip2_caption_ms",
        "text_conditioning_ms",
        "sampler_schedule_ms",
        "condition_encoder_and_init_ms",
        "denoiser_loop_ms",
        "vae_decode_ms",
        "wavelet_color_fix_ms",
        "postprocess_d2h_resize_ms",
        "receiver_wall_ms",
    ]
    latency_summary = {}
    for steps in steps_list:
        subset = [row for row in latency_rows if int(row["steps"]) == steps]
        latency_summary[str(steps)] = {
            field: summarize([float(row[field]) for row in subset])
            for field in component_fields
        }
    summary = {
        "status": "PASS",
        "method": "Official DiffJSCC OpenImage C16",
        "hardware": inventory,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "batch_size_source_images": 1,
        "internal_resolution": [3, 512, 512],
        "latency_rows": len(latency_rows),
        "quality_rows": len(quality_rows),
        "latency_by_steps": latency_summary,
        "quality_curve": curve,
        "minimum_preregistered_steps_retaining_lpips_advantage_vs_s33": min(
            (
                int(row["steps"])
                for row in curve
                if row["retains_significant_lpips_advantage_vs_s33"]
            ),
            default=None,
        ),
        "parameters": unique_parameter_ledger(model),
        "profiled_flops_lower_bound": {
            "representative_caption_text": rep_prepared["caption"],
            "jscc_frontend": jscc_flops,
            "receiver_blip2_caption": caption_flops,
            "text_conditioning": text_flops,
            "condition_encoder": condition_encoder_flops,
            "one_denoiser_evaluation": per_step_flops,
            "one_vae_decode": vae_decode_flops,
            "fixed_supported_ops_total": fixed_flops,
            "formula": "fixed_supported_ops_total + steps * one_denoiser_evaluation",
            "totals_by_steps": {
                str(steps): int(fixed_flops + steps * per_step_flops)
                for steps in steps_list
            },
            "coverage": config["flops_contract"]["definition"],
            "limitation": config["flops_contract"]["limitation"],
        },
        "replay_audit": {
            "historical_rows": len(replay_100),
            "max_abs_psnr_error": max_psnr_replay_error,
            "max_abs_lpips_error": max_lpips_replay_error,
        },
        "load_audit": load_audit,
        "inputs": {
            "config_sha256": sha256_file(config_path),
            "script_sha256": sha256_file(SCRIPT),
            "s30_config_sha256": sha256_file(s30_config_path),
        },
        "new_training": False,
        "network_download": False,
        "official_imagenette_validation_accessed": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "STATE.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "latency_rows": len(latency_rows),
                "quality_rows": len(quality_rows),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
