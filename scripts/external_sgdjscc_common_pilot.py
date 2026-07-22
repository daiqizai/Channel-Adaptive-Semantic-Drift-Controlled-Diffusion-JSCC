#!/usr/bin/env python3
"""Run the SGD-JSCC common-contract adapter on the frozen 8x5 pilot."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

# pc_imagenette_supervised_audit has transitive optional imports that initialize
# huggingface_hub constants, so the pinned author cache must be selected before
# importing any project evaluation helper below.
os.environ["HF_HOME"] = str(ROOT / "third_party/SGDJSCC/runtime_assets/hf_home")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DIFFUSERS_OFFLINE"] = "1"

from cadsd_jscc.external_common import (  # noqa: E402
    canonical_noise_sha256,
    canonical_standard_normal,
)
from cadsd_jscc.external_rate_alignment import RepeatedSparseChannelAdapter  # noqa: E402
from pc_imagenette_supervised_audit import (  # noqa: E402
    evaluate_probabilities,
    load_scratch_classifier,
)


COMMON_SCRIPT = ROOT / "scripts" / "external_sgdjscc_common_smoke.py"
SPEC = importlib.util.spec_from_file_location("external_sgdjscc_common_helpers", COMMON_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
common = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(common)
native = common.native


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
    if not resolved.is_file() or sha256_file(resolved) != expected:
        raise RuntimeError(f"missing or hash-mismatched frozen input: {resolved}")
    return resolved


def validate_samples(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    if config.get("status") != "preregistered_before_any_pilot_method_output":
        raise RuntimeError("pilot preregistration status changed")
    if config.get("official_val_accessed") is not False:
        raise RuntimeError("official validation must remain sealed")
    population = config["population"]
    manifest_path = require_sha(
        population["split_manifest"], population["split_manifest_sha256"]
    )
    require_sha(
        population["frozen_clean_membership_source"],
        population["frozen_clean_membership_source_sha256"],
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    classes = [str(value) for value in manifest["classes"]]
    by_id = {
        str(item["sample_id"]): item
        for item in manifest["samples"]
        if str(item["split"]) == str(population["required_split"])
    }
    root = resolve(manifest["source_train_root"])
    samples = []
    for frozen in population["samples"]:
        sample_id = str(frozen["sample_id"])
        item = dict(by_id[sample_id])
        path = root / str(item["relative_path"])
        require_sha(path, str(frozen["content_sha256"]))
        if int(item["class_idx"]) != int(frozen["class_idx"]):
            raise RuntimeError(f"class mismatch: {sample_id}")
        item["path"] = path
        samples.append(item)
    expected_count = int(population.get("expected_sample_count", 8))
    if len(samples) != expected_count or len({item["sample_id"] for item in samples}) != expected_count:
        raise RuntimeError(
            f"frozen comparison requires {expected_count} unique images, got {len(samples)}"
        )
    return samples, classes


def load_reference_rows(config: dict[str, Any]) -> dict[tuple[str, float], dict[str, str]]:
    path = resolve(config["outputs"]["ours"]) / "per_sample.csv"
    if not path.is_file():
        raise RuntimeError("ours common-pilot rows must be completed before SGD-JSCC")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keyed = {(row["sample_id"], float(row["snr_db"])): row for row in rows}
    expected = int(config["population"].get("expected_sample_count", 8)) * len(
        config["channel"]["snrs_db"]
    )
    if len(keyed) != expected:
        raise RuntimeError(
            f"ours common reference has {len(keyed)} unique keys, expected {expected}"
        )
    return keyed


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    images = {str(row["sample_id"]) for row in rows}
    snrs = {float(row["snr_db"]) for row in rows}
    expected = len(images) * len(snrs)
    keys = {(str(row["sample_id"]), float(row["snr_db"])) for row in rows}
    if len(rows) != expected or len(keys) != expected:
        raise RuntimeError(
            f"SGD-JSCC produced {len(rows)} rows/{len(keys)} unique keys, expected {expected}"
        )
    summary: dict[str, Any] = {
        "method": "sgd_jscc_common",
        "status": "PASS",
        "rows": len(rows),
        "images": len(images),
        "snrs_db": sorted({float(row["snr_db"]) for row in rows}),
        "pilot_claim_scope": "integration_and_direction_only",
        "outcome_claims_allowed": False,
        "mean_final_psnr": sum(float(row["final_psnr"]) for row in rows) / len(rows),
        "mean_final_ms_ssim": sum(float(row["final_ms_ssim"]) for row in rows) / len(rows),
        "mean_final_lpips": sum(float(row["final_lpips"]) for row in rows) / len(rows),
        "final_failures": sum(bool(row["final_failure"]) for row in rows),
        "new_errors_vs_deepjscc": sum(bool(row["new_error_vs_deepjscc"]) for row in rows),
        "repairs_vs_deepjscc": sum(bool(row["repair_vs_deepjscc"]) for row in rows),
        "mean_runtime_ms_per_image": sum(float(row["runtime_ms_per_image"]) for row in rows)
        / len(rows),
        "peak_gpu_memory_mib": max(float(row["peak_gpu_memory_mib"]) for row in rows),
        "caption_packets_ok": sum(int(row["caption_packets_ok"]) for row in rows),
        "caption_packet_count": sum(int(row["caption_packet_count"]) for row in rows),
    }
    by_snr = {}
    for snr in summary["snrs_db"]:
        subset = [row for row in rows if float(row["snr_db"]) == snr]
        by_snr[str(int(snr))] = {
            "mean_final_psnr": sum(float(row["final_psnr"]) for row in subset) / len(subset),
            "mean_final_ms_ssim": sum(float(row["final_ms_ssim"]) for row in subset)
            / len(subset),
            "mean_final_lpips": sum(float(row["final_lpips"]) for row in subset)
            / len(subset),
            "final_failures": sum(bool(row["final_failure"]) for row in subset),
            "new_errors_vs_deepjscc": sum(bool(row["new_error_vs_deepjscc"]) for row in subset),
            "repairs_vs_deepjscc": sum(bool(row["repair_vs_deepjscc"]) for row in subset),
        }
    summary["by_snr"] = by_snr
    return summary


def run(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    # huggingface_hub/diffusers cache constants are initialized at import time in
    # the pinned SGD-JSCC environment, so point them at the frozen local cache
    # before importing transformers (or any transitive hub consumer).
    reallocation = config.get("study") == "project_working_point_sgd_rate_reallocation"
    author_rate = config.get("study") == "author_working_point_rate_alignment"
    external_mode = reallocation or author_rate
    pilot_config = (
        load_yaml(config["population_reference_config"]) if external_mode else config
    )
    if reallocation:
        method = config["method"]
    elif author_rate:
        method = config["methods"]["sgd_jscc_paper_protocol"]
    else:
        method = config["methods"]["sgd_jscc_common"]
    base_preview = load_yaml(method["base_config"])
    native_preview = load_yaml(base_preview["source"]["native_asset_and_model_config"])
    os.environ["HF_HOME"] = str(resolve(native_preview["assets"]["hf_home"]))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"

    import numpy as np
    import torch
    import torch.nn.functional as F
    from omegaconf import OmegaConf
    from PIL import Image
    from pytorch_msssim import ms_ssim
    from torchvision import transforms
    from torchvision.utils import save_image
    from transformers import AutoProcessor, Blip2ForConditionalGeneration

    samples, classes = validate_samples(pilot_config)
    if author_rate:
        reference_path = resolve(config["outputs"]["deepjscc"]) / "per_sample.csv"
        with reference_path.open(encoding="utf-8", newline="") as handle:
            author_reference = list(csv.DictReader(handle))
        reference_rows = {
            (row["sample_id"], float(row["snr_db"])): row for row in author_reference
        }
        expected_reference_rows = len(samples) * len(config["channel"]["snrs_db"])
        if len(reference_rows) != expected_reference_rows:
            raise RuntimeError(
                "exact-rate DeepJSCC reference must contain "
                f"{expected_reference_rows} rows, got {len(reference_rows)}"
            )
    else:
        reference_rows = load_reference_rows(pilot_config)
    base_path = resolve(method["base_config"])
    base = common.load_config(base_path)
    rate_plan = common.validate_config(base)
    native_payload, asset_validation = common.validate_paths_and_assets(
        base, verify_hashes=True
    )
    runtime = asset_validation["native_asset_validation"]["runtime_status"]
    if not all(
        runtime[name]
        for name in ("blip2_ready", "clip_ready", "scheduler_ready")
    ):
        raise RuntimeError(f"SGD-JSCC runtime assets are incomplete: {runtime}")
    if int(rate_plan["total_real_symbols"]) != 65536:
        raise RuntimeError("SGD-JSCC common rate is not 65,536 real symbols")

    runtime_base = copy.deepcopy(base)
    if reallocation:
        allocation = config["rate"]["allocation"]
        if sum(
            int(allocation[key])
            for key in (
                "main_transmitted_real_symbols",
                "active_edge_transmitted_real_symbols",
                "text_real_symbols",
                "no_information_padding_real_symbols",
            )
        ) != int(config["rate"]["total_real_symbols"]):
            raise RuntimeError("reallocation does not exactly consume the project budget")
        runtime_base["text_transport"]["repetition"] = int(
            allocation["caption_repetition"]
        )
        runtime_base["text_transport"]["real_symbols_per_patch"] = int(
            allocation["caption_packet_bits_per_patch"]
        ) * int(allocation["caption_repetition"])
        runtime_base["channel"]["canonical_noise_layout"] = [
            {
                "name": "main_latent",
                "real_symbols": int(allocation["main_transmitted_real_symbols"]),
            },
            {
                "name": "active_edge",
                "real_symbols": int(allocation["active_edge_transmitted_real_symbols"]),
            },
            {"name": "text_caption", "real_symbols": int(allocation["text_real_symbols"])},
            {
                "name": "no_information_padding",
                "real_symbols": int(allocation["no_information_padding_real_symbols"]),
            },
        ]
        runtime_base["rate_contract"]["total_real_symbols"] = int(
            config["rate"]["total_real_symbols"]
        )
    elif author_rate:
        runtime_base["channel"]["canonical_noise_layout"] = [
            {"name": "main_latent", "real_symbols": 16384},
            {"name": "active_edge", "real_symbols": 3328},
        ]
        runtime_base["rate_contract"]["total_real_symbols"] = int(
            config["rate"]["total_real_symbols"]
        )

    if reallocation:
        output = resolve(config["output_dir"])
    elif author_rate:
        output = resolve(config["outputs"]["sgd_jscc"])
    else:
        output = resolve(config["outputs"]["sgd_jscc"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output / "config_snapshot.yaml")
    shutil.copy2(base_path, output / "adapter_config_snapshot.yaml")
    shutil.copy2(
        resolve(base["source"]["native_asset_and_model_config"]),
        output / "native_config_snapshot.yaml",
    )

    source_path = resolve(base["source"]["path"])
    sys.path.insert(0, str(source_path))
    import inference_one as author
    import clip
    from models.test_advanced_network.autoencoderkl import AutoencoderKL
    from models.test_advanced_network.diffusion_element_wise import DiffusionGenerator
    from models.test_advanced_network.mask_diffusion import MDTv2
    from models.test_advanced_network.mask_diffusion_controlnet import MDTv2_ControlNet
    from models.test_advanced_network.muge_model import Mymodel as MuGEModel

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    native.set_seed(int(config["channel"]["base_seed"]), torch, np)
    model_cfg = native_payload["model"]
    author.config = OmegaConf.create(
        {
            "model": {
                "condition_setting": {
                    "use_semantic": model_cfg["use_semantic"],
                    "use_text": model_cfg["use_text"],
                    "use_controlnet": model_cfg["use_controlnet"],
                    "use_jscc_feature": model_cfg["use_jscc_feature"],
                    "use_gt_text": model_cfg["use_gt_text"],
                },
                "diffusion": {
                    "step_style": model_cfg["step_style"],
                    "diffusion_step": model_cfg["diffusion_step"],
                    "guidance_scale": model_cfg["guidance_scale"],
                    "controlnet_scale": model_cfg["controlnet_scale"],
                    "cfg_method": model_cfg["cfg_method"],
                },
            },
            "transmission": {
                "canny_cr": model_cfg["canny_cr"],
                "channel": "AWGN",
                "mask_method": model_cfg["mask_method"],
                "use_gt_csi": base["channel"]["use_gt_csi"],
            },
            "th": model_cfg["edge_threshold"],
        }
    )
    author.device = device

    transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()]
    )
    source_tensors = [
        transform(Image.open(item["path"]).convert("RGB")).unsqueeze(0).to(device)
        for item in samples
    ]
    patches_and_meta = [author.split_image_v2(tensor) for tensor in source_tensors]
    if any(list(patches.shape) != [4, 3, 128, 128] for patches, _ in patches_and_meta):
        raise RuntimeError("author patch split differs from the frozen 4x128x128 contract")

    processor = AutoProcessor.from_pretrained(
        resolve(native_payload["assets"]["blip2_dir"]), local_files_only=True
    )
    caption_model = Blip2ForConditionalGeneration.from_pretrained(
        resolve(native_payload["assets"]["blip2_dir"]),
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    caption_model.processor = processor
    caption_model.to(device).eval()
    captions: list[list[str]] = []
    caption_runtime_ms: list[float] = []
    for patches, _ in patches_and_meta:
        torch.cuda.synchronize(device)
        caption_started = time.perf_counter()
        with torch.inference_mode():
            nested = author.image_caption(caption_model, patches, device)
        torch.cuda.synchronize(device)
        captions.append(list(nested[0]))
        caption_runtime_ms.append((time.perf_counter() - caption_started) * 1000.0)
    (output / "sender_captions.json").write_text(
        json.dumps(
            {samples[i]["sample_id"]: captions[i] for i in range(len(samples))},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    del caption_model, processor
    torch.cuda.empty_cache()

    checkpoint_dir = resolve(native_payload["assets"]["checkpoint_dir"])
    model = author.JSCC_model(snr=float(config["channel"]["snrs_db"][0]))
    model.load_state_dict(torch.load(checkpoint_dir / "JSCC_model.pth", map_location="cpu"))
    model.to(device).eval()
    canny_net = MuGEModel(encoder_weights=None)
    canny_payload = torch.load(
        checkpoint_dir / "muge-epoch-19-checkpoint.pth", map_location="cpu"
    )
    canny_net.load_state_dict(canny_payload["state_dict"])
    canny_net.to(device).eval()
    denoiser = MDTv2(depth=12, hidden_size=512, patch_size=1, num_heads=8)
    denoiser.load_state_dict(
        torch.load(checkpoint_dir / "diffusion_backbone.pth", map_location="cpu")["model_ema"]
    )
    denoiser = MDTv2_ControlNet(base_model=denoiser, copy_blocks_num=6, hidden_size=512)
    denoiser.load_state_dict(
        torch.load(checkpoint_dir / "diffusion_controlnet.pth", map_location="cpu")["model_ema"]
    )
    denoiser.to(device).eval()
    clip_model, _ = clip.load(
        str(resolve(native_payload["assets"]["clip_model"])), device=device, jit=False
    )
    clip_model.eval()
    ddconfig = {
        "double_z": True,
        "z_channels": 16,
        "resolution": 128,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [],
        "dropout": 0.0,
    }
    vae = AutoencoderKL(ddconfig, 16)
    vae.load_state_dict(model.vae.state_dict(), strict=False)
    vae.to(device).eval()
    pipeline = DiffusionGenerator(denoiser, vae, clip_model, device, torch.float32)

    evaluator_cfg = {
        "imagenette": {
            "normalization_mean": pilot_config["evaluator"]["normalization_mean"],
            "normalization_std": pilot_config["evaluator"]["normalization_std"],
        }
    }
    evaluator_path = require_sha(
        pilot_config["evaluator"]["checkpoint"],
        pilot_config["evaluator"]["checkpoint_sha256"],
    )
    evaluator, evaluator_temperature = load_scratch_classifier(
        str(evaluator_path),
        classes,
        device,
        str(pilot_config["evaluator"]["expected_role"]),
    )
    import lpips

    os.environ.setdefault("TORCH_HOME", str(resolve("outputs/cache/torch")))
    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    all_targets = torch.cat(source_tensors, dim=0)
    labels = torch.tensor([int(item["class_idx"]) for item in samples], device=device)
    with torch.no_grad():
        original_probability = evaluate_probabilities(
            evaluator, evaluator_temperature, all_targets, evaluator_cfg
        )
        original_confidence, original_prediction = original_probability.max(dim=1)
    if not bool(((original_prediction == labels) & (original_confidence >= 0.5)).all()):
        raise RuntimeError("SGD-JSCC sees a non-clean T_cls pilot input")

    rows: list[dict[str, Any]] = []
    for snr in map(float, config["channel"]["snrs_db"]):
        reconstructed_images = []
        runtimes = []
        records_by_image = []
        for index, item in enumerate(samples):
            noise = canonical_standard_normal(
                int(config["channel"]["base_seed"]),
                str(item["sample_id"]),
                snr,
                int(config["rate"]["total_real_symbols"]),
            )
            segments = common.slice_canonical_noise(noise, runtime_base)
            if author_rate:
                decoded_captions = list(captions[index])
                text_records = [
                    {
                        "packet_ok": True,
                        "decoded_packet_bit_errors": 0,
                        "transport": "perfect_unmetered_paper_assumption",
                    }
                    for _ in decoded_captions
                ]
            else:
                decoded_captions, text_records = common.transmit_captions(
                    captions[index], segments["text_caption"], snr, runtime_base, torch
                )
            patches, patch_meta = patches_and_meta[index]
            model.snr = snr
            if reallocation:
                adapter = RepeatedSparseChannelAdapter(
                    main_noise=segments["main_latent"],
                    edge_noise=segments["active_edge"],
                    expected_patches=4,
                    main_per_patch=4096,
                    edge_active_per_patch=832,
                    main_repetition=int(config["rate"]["allocation"]["main_repetition"]),
                    edge_repetition=int(config["rate"]["allocation"]["active_edge_repetition"]),
                    noise_variance_factor=0.5,
                )
            else:
                adapter = common.SparseCommonChannelAdapter(
                    main_noise=segments["main_latent"],
                    edge_noise=segments["active_edge"],
                    expected_patches=4,
                    main_per_patch=4096,
                    edge_active_per_patch=832,
                    noise_variance_factor=0.5,
                )
            adapter.attach(model, torch)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.inference_mode():
                canny_data, canny_uncertainty = author.generate_canny(
                    patches, canny_net, device
                )
                reconstructed_patches, used_caption_text = model(
                    patches,
                    pipe=pipeline,
                    gt_text=[decoded_captions],
                    canny_data=canny_data.to(device),
                    canny_uncertainty=canny_uncertainty.to(device),
                    use_semantic=model_cfg["use_semantic"],
                    use_controlnet=model_cfg["use_controlnet"],
                    use_text=model_cfg["use_text"],
                    use_gt_text=model_cfg["use_gt_text"],
                    canny_cr=model_cfg["canny_cr"],
                    use_jscc_feature=model_cfg["use_jscc_feature"],
                    use_gt_csi=base["channel"]["use_gt_csi"],
                    controlnet_scale=model_cfg["controlnet_scale"],
                    mask_method=model_cfg["mask_method"],
                    diffusion_step=model_cfg["diffusion_step"],
                    step_style=model_cfg["step_style"],
                    cfg_method=model_cfg["cfg_method"],
                    guidance_scale=model_cfg["guidance_scale"],
                    scaling_factor=model_cfg["scaling_factor"],
                )
                reconstructed_patches = reconstructed_patches.clamp(0.0, 1.0)
                reconstructed = author.merge_image_v2(
                    reconstructed_patches, patch_meta
                ).clamp(0.0, 1.0)
            torch.cuda.synchronize(device)
            adapter.require_complete()
            runtimes.append(
                (time.perf_counter() - started) * 1000.0 + caption_runtime_ms[index]
            )
            reconstructed_images.append(reconstructed)
            records_by_image.append(
                {
                    "noise_sha": canonical_noise_sha256(noise),
                    "text_records": text_records,
                    "decoded_captions": decoded_captions,
                    "used_captions": list(used_caption_text),
                    "main_real_symbols": (
                        adapter.main_transmitted_real_symbols
                        if reallocation
                        else adapter.main_real_symbols
                    ),
                    "edge_active_real_symbols": (
                        adapter.edge_transmitted_real_symbols
                        if reallocation
                        else adapter.edge_active_real_symbols
                    ),
                }
            )
        reconstructed_batch = torch.cat(reconstructed_images, dim=0)
        with torch.no_grad():
            final_probability = evaluate_probabilities(
                evaluator, evaluator_temperature, reconstructed_batch, evaluator_cfg
            )
            final_prediction = final_probability.argmax(dim=1)
            mse_values = F.mse_loss(
                reconstructed_batch, all_targets, reduction="none"
            ).flatten(1).mean(1)
            psnr_values = -10.0 * torch.log10(mse_values.clamp_min(1e-12))
            ms_ssim_values = ms_ssim(
                reconstructed_batch, all_targets, data_range=1.0, size_average=False
            )
            lpips_values = lpips_model(
                reconstructed_batch * 2.0 - 1.0, all_targets * 2.0 - 1.0
            ).flatten()
        save_image(
            torch.cat([all_targets, reconstructed_batch]).cpu(),
            output / f"snr_{int(snr):02d}_source_sgdjscc.png",
            nrow=min(8, len(samples)),
        )
        peak = torch.cuda.max_memory_allocated(device) / (1024**2)
        for index, item in enumerate(samples):
            reference = reference_rows[(str(item["sample_id"]), snr)]
            if records_by_image[index]["noise_sha"] != reference["canonical_noise_sha256"]:
                raise RuntimeError("canonical noise differs across environments/methods")
            expected_main = (
                int(config["rate"]["allocation"]["main_transmitted_real_symbols"])
                if reallocation
                else 16384
            )
            expected_edge = (
                int(config["rate"]["allocation"]["active_edge_transmitted_real_symbols"])
                if reallocation
                else 3328
            )
            if int(records_by_image[index]["main_real_symbols"]) != expected_main:
                raise RuntimeError("SGD-JSCC main runtime rate changed")
            if int(records_by_image[index]["edge_active_real_symbols"]) != expected_edge:
                raise RuntimeError("SGD-JSCC active edge runtime rate changed")
            text_records = records_by_image[index]["text_records"]
            final_correct = int(final_prediction[index]) == int(labels[index])
            deepjscc_correct = reference["deepjscc_correct"].lower() == "true"
            row = {
                "method": (
                    "sgd_jscc_main_r2_text_r13"
                    if reallocation
                    else (
                        "sgd_jscc_paper_free_text_author_rate"
                        if author_rate
                        else "sgd_jscc_common"
                    )
                ),
                "sample_id": item["sample_id"],
                "wnid": item["wnid"],
                "class_idx": int(labels[index]),
                "snr_db": snr,
                "base_seed": int(config["channel"]["base_seed"]),
                "canonical_noise_sha256": records_by_image[index]["noise_sha"],
                "noise_variance_convention": "complex_awgn_per_real_half_variance",
                "total_real_symbols": int(config["rate"]["total_real_symbols"]),
                "total_complex_channel_uses": int(config["rate"]["total_complex_channel_uses"]),
                "cbr": float(config["rate"].get("cbr", config["rate"].get("exact_cbr"))),
                "original_prediction": int(original_prediction[index]),
                "original_confidence": float(original_confidence[index]),
                "clean_correct": True,
                "deepjscc_prediction": int(reference["deepjscc_prediction"]),
                "deepjscc_correct": deepjscc_correct,
                "final_prediction": int(final_prediction[index]),
                "final_correct": final_correct,
                "final_failure": not final_correct,
                "new_error_vs_deepjscc": deepjscc_correct and not final_correct,
                "repair_vs_deepjscc": (not deepjscc_correct) and final_correct,
                "deepjscc_psnr": float(reference["deepjscc_psnr"]),
                "deepjscc_ms_ssim": float(reference["deepjscc_ms_ssim"]),
                "deepjscc_lpips": float(reference["deepjscc_lpips"]),
                "final_psnr": float(psnr_values[index]),
                "final_ms_ssim": float(ms_ssim_values[index]),
                "final_lpips": float(lpips_values[index]),
                "runtime_ms_per_image": runtimes[index],
                "peak_gpu_memory_mib": peak,
                "caption_packets_ok": sum(bool(record["packet_ok"]) for record in text_records),
                "caption_packet_count": len(text_records),
                "caption_decoded_bit_errors": sum(
                    int(record["decoded_packet_bit_errors"]) for record in text_records
                ),
                "sender_captions": json.dumps(captions[index], ensure_ascii=False),
                "decoded_captions": json.dumps(
                    records_by_image[index]["decoded_captions"], ensure_ascii=False
                ),
                "main_real_symbols": expected_main,
                "active_edge_real_symbols": expected_edge,
                "text_real_symbols": (
                    int(config["rate"]["allocation"]["text_real_symbols"])
                    if reallocation
                    else (0 if author_rate else int(rate_plan["text_real_symbols"]))
                ),
                "no_information_padding_real_symbols": (
                    int(config["rate"]["allocation"]["no_information_padding_real_symbols"])
                    if reallocation
                    else (
                        0
                        if author_rate
                        else int(rate_plan["no_information_padding_real_symbols"])
                    )
                ),
            }
            rows.append(row)
    write_csv(output / "per_sample.csv", rows)
    summary = summarize(rows)
    if reallocation:
        summary["method"] = "sgd_jscc_main_r2_text_r13"
        summary["interpretation"] = (
            "released_weight_rate_allocation_sensitivity_not_increased_model_capacity"
        )
        summary["allocation"] = config["rate"]["allocation"]
    elif author_rate:
        summary["method"] = "sgd_jscc_paper_free_text_author_rate"
        summary["interpretation"] = (
            "paper_protocol_upper_bound_with_perfect_unmetered_text_not_strict_physical_rate_match"
        )
        summary["image_branch_real_symbols"] = int(config["rate"]["total_real_symbols"])
    summary["environment"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/external_common_comparison_pilot.yaml")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = load_yaml(config_path)
    reallocation = config.get("study") == "project_working_point_sgd_rate_reallocation"
    author_rate = config.get("study") == "author_working_point_rate_alignment"
    external_mode = reallocation or author_rate
    pilot_config = (
        load_yaml(config["population_reference_config"]) if external_mode else config
    )
    samples, _ = validate_samples(pilot_config)
    output = resolve(
        config["output_dir"] if reallocation else config["outputs"]["sgd_jscc"]
    )
    dry = {
        "analysis_id": config["analysis_id"],
        "method": (
            "sgd_jscc_main_r2_text_r13"
            if reallocation
            else (
                "sgd_jscc_paper_free_text_author_rate"
                if author_rate
                else "sgd_jscc_common"
            )
        ),
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
    summary = run(config_path, config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
