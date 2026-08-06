#!/usr/bin/env python3
"""Measure frozen SGD-JSCC paper-protocol inference cost on the S34D GPU."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
os.environ["HF_HOME"] = str(ROOT / "third_party/SGDJSCC/runtime_assets/hf_home")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DIFFUSERS_OFFLINE"] = "1"

from cadsd_jscc.external_common import canonical_standard_normal  # noqa: E402
from external_sgdjscc_common_pilot import (  # noqa: E402
    load_yaml as load_external_yaml,
    require_sha,
    validate_samples,
)


COMMON_SCRIPT = ROOT / "scripts" / "external_sgdjscc_common_smoke.py"
SPEC = importlib.util.spec_from_file_location("s34d_sgd_common_helpers", COMMON_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
common = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(common)
native = common.native
SCRIPT = Path(__file__).resolve()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_yaml(value: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(resolve(value).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(value)
    return payload


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "std": float(array.std(ddof=1)),
    }


def unique_parameter_ledger(components: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: set[int] = set()
    incremental: dict[str, int] = {}
    raw: dict[str, int] = {}
    for name, module in components:
        params = list(module.parameters())
        raw[name] = int(sum(p.numel() for p in params))
        new = [p for p in params if id(p) not in seen]
        incremental[name] = int(sum(p.numel() for p in new))
        seen.update(id(p) for p in params)
    return {
        "unique_live_parameters": int(sum(incremental.values())),
        "raw_component_counts_may_overlap": raw,
        "deduplicated_incremental_counts_in_listed_order": incremental,
    }


class NestedRecorder:
    def __init__(self, torch_module: Any, device: Any) -> None:
        self.torch = torch_module
        self.device = device
        self.values: dict[str, list[float]] = {}

    def reset(self) -> None:
        self.values = {}

    def wrap(self, name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            self.torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            value = fn(*args, **kwargs)
            self.torch.cuda.synchronize(self.device)
            self.values.setdefault(name, []).append(
                (time.perf_counter() - started) * 1000.0
            )
            return value

        return wrapped

    def total(self, name: str) -> float:
        return float(sum(self.values.get(name, [])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config_path = resolve("configs/s34d_generative_inference_cost.yaml")
    config = load_yaml(config_path)
    if config["status"] != "preregistered_and_authorized_before_measurement":
        raise RuntimeError("S34D is not authorized")
    inventory = gpu_inventory()
    if inventory["uuid"] != config["hardware"]["required_gpu_uuid"]:
        raise RuntimeError(f"wrong GPU: {inventory}")

    resolved_path = resolve(config["inputs"]["sgd_seed_config"])
    if common.sha256_file(resolved_path) != config["inputs"]["sgd_seed_config_sha256"]:
        raise RuntimeError("frozen SGD resolved config changed")
    resolved_config = load_external_yaml(resolved_path)
    pilot_config = load_external_yaml(resolved_config["population_reference_config"])
    samples, _classes = validate_samples(pilot_config)
    method = resolved_config["methods"]["sgd_jscc_paper_protocol"]
    base_path = resolve(method["base_config"])
    base = common.load_config(base_path)
    native_payload, asset_validation = common.validate_paths_and_assets(
        base, verify_hashes=True
    )
    runtime_status = asset_validation["native_asset_validation"]["runtime_status"]
    if not all(runtime_status[k] for k in ("blip2_ready", "clip_ready", "scheduler_ready")):
        raise RuntimeError("SGD runtime assets are incomplete")
    os.environ["HF_HOME"] = str(resolve(native_payload["assets"]["hf_home"]))

    import clip
    import torch
    from omegaconf import OmegaConf
    from PIL import Image
    from torchvision import transforms
    from transformers import AutoProcessor, Blip2ForConditionalGeneration

    source_path = resolve(base["source"]["path"])
    sys.path.insert(0, str(source_path))
    import inference_one as author
    from models.test_advanced_network.autoencoderkl import AutoencoderKL
    from models.test_advanced_network.diffusion_element_wise import DiffusionGenerator
    from models.test_advanced_network.mask_diffusion import MDTv2
    from models.test_advanced_network.mask_diffusion_controlnet import MDTv2_ControlNet
    from models.test_advanced_network.muge_model import Mymodel as MuGEModel

    device = torch.device(config["hardware"]["device"])
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = bool(config["hardware"]["cudnn_benchmark"])
    native.set_seed(int(resolved_config["channel"]["base_seed"]), torch, np)
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
    checkpoint_dir = resolve(native_payload["assets"]["checkpoint_dir"])

    processor = AutoProcessor.from_pretrained(
        resolve(native_payload["assets"]["blip2_dir"]), local_files_only=True
    )
    caption_model = Blip2ForConditionalGeneration.from_pretrained(
        resolve(native_payload["assets"]["blip2_dir"]),
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    caption_model.processor = processor
    caption_model.to(device).eval().requires_grad_(False)
    model = author.JSCC_model(
        snr=float(config["latency_contract"]["timed_snrs_db"][0])
    )
    model.load_state_dict(
        torch.load(checkpoint_dir / "JSCC_model.pth", map_location="cpu")
    )
    model.to(device).eval().requires_grad_(False)
    canny_net = MuGEModel(encoder_weights=None)
    canny_payload = torch.load(
        checkpoint_dir / "muge-epoch-19-checkpoint.pth", map_location="cpu"
    )
    canny_net.load_state_dict(canny_payload["state_dict"])
    canny_net.to(device).eval().requires_grad_(False)
    denoiser = MDTv2(depth=12, hidden_size=512, patch_size=1, num_heads=8)
    denoiser.load_state_dict(
        torch.load(checkpoint_dir / "diffusion_backbone.pth", map_location="cpu")[
            "model_ema"
        ]
    )
    denoiser = MDTv2_ControlNet(
        base_model=denoiser, copy_blocks_num=6, hidden_size=512
    )
    denoiser.load_state_dict(
        torch.load(checkpoint_dir / "diffusion_controlnet.pth", map_location="cpu")[
            "model_ema"
        ]
    )
    denoiser.to(device).eval().requires_grad_(False)
    clip_model, _ = clip.load(
        str(resolve(native_payload["assets"]["clip_model"])),
        device=device,
        jit=False,
    )
    clip_model.eval().requires_grad_(False)
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
    pipeline_vae = AutoencoderKL(ddconfig, 16)
    pipeline_vae.load_state_dict(model.vae.state_dict(), strict=False)
    pipeline_vae.to(device).eval().requires_grad_(False)
    pipeline = DiffusionGenerator(
        denoiser, pipeline_vae, clip_model, device, torch.float32
    )
    host_pils = [
        Image.open(item["path"]).convert("RGB") for item in samples
    ]
    transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()]
    )

    recorder = NestedRecorder(torch, device)
    model.vae.encode = recorder.wrap("model_vae_encode", model.vae.encode)
    model.vae.decode = recorder.wrap("final_model_vae_decode", model.vae.decode)
    model.canny_transmission_net.forward = recorder.wrap(
        "edge_jscc", model.canny_transmission_net.forward
    )
    pipeline.encode_text = recorder.wrap("clip_text_encode", pipeline.encode_text)
    pipeline.vae.decode = recorder.wrap(
        "pipeline_internal_vae_decode", pipeline.vae.decode
    )
    pipeline.generate = recorder.wrap("pipeline_generate", pipeline.generate)

    runtime_base = copy.deepcopy(base)
    runtime_base["channel"]["canonical_noise_layout"] = [
        {"name": "main_latent", "real_symbols": 16384},
        {"name": "active_edge", "real_symbols": 3328},
    ]
    runtime_base["rate_contract"]["total_real_symbols"] = 19712

    def sync() -> None:
        torch.cuda.synchronize(device)

    def timed(fn: Callable[[], Any]) -> tuple[Any, float]:
        sync()
        started = time.perf_counter()
        value = fn()
        sync()
        return value, (time.perf_counter() - started) * 1000.0

    @torch.inference_mode()
    def run_once(index: int, snr: float, collect: bool) -> dict[str, Any]:
        item = samples[index]
        noise = canonical_standard_normal(
            int(resolved_config["channel"]["base_seed"]),
            str(item["sample_id"]),
            snr,
            19712,
        )
        segments = common.slice_canonical_noise(noise, runtime_base)
        model.snr = float(snr)
        adapter = common.SparseCommonChannelAdapter(
            main_noise=segments["main_latent"],
            edge_noise=segments["active_edge"],
            expected_patches=4,
            main_per_patch=4096,
            edge_active_per_patch=832,
            noise_variance_factor=0.5,
        )
        adapter.attach(model, torch)
        sync()
        wall_started = time.perf_counter()

        def preprocess() -> tuple[torch.Tensor, Any]:
            source = transform(host_pils[index]).unsqueeze(0).to(device)
            return author.split_image_v2(source)

        (patches, patch_meta), preprocess_ms = timed(preprocess)
        captions_nested, caption_ms = timed(
            lambda: author.image_caption(caption_model, patches, device)
        )
        captions = list(captions_nested[0])
        (canny_data, canny_uncertainty), canny_ms = timed(
            lambda: author.generate_canny(patches, canny_net, device)
        )
        recorder.reset()

        def core() -> tuple[torch.Tensor, list[str]]:
            return model(
                patches,
                pipe=pipeline,
                gt_text=[captions],
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

        (reconstructed_patches, _used), core_ms = timed(core)

        def postprocess() -> torch.Tensor:
            merged = author.merge_image_v2(
                reconstructed_patches.clamp(0, 1), patch_meta
            ).clamp(0, 1)
            return merged.mul(255).to(torch.uint8).cpu()

        host_output, postprocess_ms = timed(postprocess)
        wall_ms = (time.perf_counter() - wall_started) * 1000.0
        adapter.require_complete()
        if tuple(host_output.shape) != (1, 3, 256, 256):
            raise RuntimeError("SGD output shape changed")
        if not collect:
            return {}
        vae_encodes = recorder.values.get("model_vae_encode", [])
        if len(vae_encodes) != 2:
            raise RuntimeError(f"expected two SGD VAE encodes, got {vae_encodes}")
        pipeline_generate = recorder.total("pipeline_generate")
        clip_text = recorder.total("clip_text_encode")
        internal_decode = recorder.total("pipeline_internal_vae_decode")
        solver = pipeline_generate - clip_text - internal_decode
        final_decode = recorder.total("final_model_vae_decode")
        edge_jscc = recorder.total("edge_jscc")
        unattributed = (
            core_ms
            - sum(vae_encodes)
            - edge_jscc
            - pipeline_generate
            - final_decode
        )
        return {
            "sample_id": item["sample_id"],
            "snr_db": snr,
            "batch_size_source_images": 1,
            "executed_denoiser_evaluations": int(model_cfg["diffusion_step"]),
            "receiver_wall_ms": wall_ms,
            "preprocess_h2d_patch_split_ms": preprocess_ms,
            "blip2_caption_ms": caption_ms,
            "edge_extractor_ms": canny_ms,
            "core_model_ms": core_ms,
            "main_vae_encode_ms": vae_encodes[0],
            "edge_jscc_ms": edge_jscc,
            "edge_vae_encode_ms": vae_encodes[1],
            "clip_text_conditioning_ms": clip_text,
            "diffusion_solver_ms": solver,
            "redundant_pipeline_vae_decode_ms": internal_decode,
            "final_vae_decode_ms": final_decode,
            "core_unattributed_channel_mask_stepmatch_ms": unattributed,
            "postprocess_merge_d2h_ms": postprocess_ms,
            "component_top_level_sum_ms": (
                preprocess_ms + caption_ms + canny_ms + core_ms + postprocess_ms
            ),
            "wall_minus_top_level_sum_ms": wall_ms
            - (preprocess_ms + caption_ms + canny_ms + core_ms + postprocess_ms),
        }

    # Preflight tests coexistence of BLIP2 and the complete diffusion receiver.
    for index in range(int(config["latency_contract"]["warmup_source_keys"])):
        run_once(index, float(config["latency_contract"]["timed_snrs_db"][index]), False)
    if args.preflight:
        row = run_once(0, 7.0, True)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "row": row,
                    "parameters": unique_parameter_ledger(
                        [
                            ("receiver_blip2", caption_model),
                            ("edge_extractor_muge", canny_net),
                            ("sgd_jscc_model_including_main_vae_and_edge_jscc", model),
                            ("diffusion_controlnet_denoiser", denoiser),
                            ("clip_text_encoder", clip_model),
                            ("duplicate_pipeline_vae", pipeline_vae),
                        ]
                    ),
                    "gpu": inventory,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    output = resolve(config["outputs"]["sgd"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    shutil.copy2(config_path, output / "config_snapshot.yaml")
    shutil.copy2(SCRIPT, output / SCRIPT.name)
    rows: list[dict[str, Any]] = []
    for snr in map(float, config["latency_contract"]["timed_snrs_db"]):
        for index in range(int(config["latency_contract"]["timed_source_images"])):
            rows.append(run_once(index, snr, True))
            if len(rows) % 20 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "latency",
                            "rows": len(rows),
                            "wall_ms": rows[-1]["receiver_wall_ms"],
                        }
                    ),
                    flush=True,
                )
    write_csv(output / "latency_rows.csv", rows)

    # FLOPs lower bound: profile the three top-level compute sections on one
    # representative input. This avoids falsely treating parameter count as compute.
    rep_index = 0
    rep_snr = float(config["flops_contract"]["representative_snr_db"])
    source = transform(host_pils[rep_index]).unsqueeze(0).to(device)
    patches, patch_meta = author.split_image_v2(source)
    captions = list(author.image_caption(caption_model, patches, device)[0])
    canny_data, canny_uncertainty = author.generate_canny(patches, canny_net, device)
    noise = canonical_standard_normal(
        int(resolved_config["channel"]["base_seed"]),
        str(samples[rep_index]["sample_id"]),
        rep_snr,
        19712,
    )
    segments = common.slice_canonical_noise(noise, runtime_base)
    model.snr = rep_snr
    adapter = common.SparseCommonChannelAdapter(
        main_noise=segments["main_latent"],
        edge_noise=segments["active_edge"],
        expected_patches=4,
        main_per_patch=4096,
        edge_active_per_patch=832,
        noise_variance_factor=0.5,
    )
    adapter.attach(model, torch)

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
            sync()
        return int(sum(int(event.flops or 0) for event in prof.key_averages()))

    caption_flops = profile_flops(
        lambda: author.image_caption(caption_model, patches, device)
    )
    edge_extractor_flops = profile_flops(
        lambda: author.generate_canny(patches, canny_net, device)
    )

    def core_profile() -> Any:
        return model(
            patches,
            pipe=pipeline,
            gt_text=[captions],
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

    core_flops = profile_flops(core_profile)
    fields = [
        "receiver_wall_ms",
        "preprocess_h2d_patch_split_ms",
        "blip2_caption_ms",
        "edge_extractor_ms",
        "core_model_ms",
        "main_vae_encode_ms",
        "edge_jscc_ms",
        "edge_vae_encode_ms",
        "clip_text_conditioning_ms",
        "diffusion_solver_ms",
        "redundant_pipeline_vae_decode_ms",
        "final_vae_decode_ms",
        "core_unattributed_channel_mask_stepmatch_ms",
        "postprocess_merge_d2h_ms",
    ]
    summary = {
        "status": "PASS",
        "method": "SGD-JSCC released paper-protocol upper",
        "hardware": inventory,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "batch_size_source_images": 1,
        "internal_layout": "four_patches_of_3x128x128",
        "latency_rows": len(rows),
        "latency": {
            field: summarize([float(row[field]) for row in rows])
            for field in fields
        },
        "parameters": unique_parameter_ledger(
            [
                ("receiver_blip2", caption_model),
                ("edge_extractor_muge", canny_net),
                ("sgd_jscc_model_including_main_vae_and_edge_jscc", model),
                ("diffusion_controlnet_denoiser", denoiser),
                ("clip_text_encoder", clip_model),
                ("duplicate_pipeline_vae", pipeline_vae),
            ]
        ),
        "profiled_flops_lower_bound": {
            "receiver_blip2_caption": caption_flops,
            "edge_extractor": edge_extractor_flops,
            "core_model_including_50_denoiser_evaluations_and_two_VAE_decodes": core_flops,
            "total": caption_flops + edge_extractor_flops + core_flops,
            "coverage": config["flops_contract"]["definition"],
            "limitation": config["flops_contract"]["limitation"],
        },
        "implementation_observation": {
            "pipeline_internal_VAE_decode_output_is_discarded_by_author_JSCC_model": True,
            "final_VAE_decode_is_executed_after_the_discarded_decode": True,
            "caption_and_diffusion_models_co_resident_during_measurement": True,
            "model_loading_excluded": True,
        },
        "rate_boundary": {
            "executed_main_plus_edge_real_symbols": 19712,
            "minimum_with_unprotected_perfect_caption_accounting": 21856,
            "direct_quality_ranking_allowed": False,
        },
        "inputs": {
            "config_sha256": common.sha256_file(config_path),
            "script_sha256": common.sha256_file(SCRIPT),
            "sgd_resolved_config_sha256": common.sha256_file(resolved_path),
        },
        "new_training": False,
        "network_download": False,
        "official_imagenette_validation_accessed": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "STATE.json").write_text(
        json.dumps({"status": "complete", "rows": len(rows)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
