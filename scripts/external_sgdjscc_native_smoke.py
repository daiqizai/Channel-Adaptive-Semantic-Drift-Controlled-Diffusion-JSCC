#!/usr/bin/env python3
"""Project-side, read-only adapter for a one-image SGD-JSCC author-code smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE_COMMIT = "2188acc0dd2805355d3d0d2e478cbc27b46b4da5"


class SmokeContractError(RuntimeError):
    """Raised when author assets or the frozen smoke contract do not match."""


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_tracked_status(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SmokeContractError("config root must be a mapping")
    return payload


def native_main_real_symbols(shape: list[int]) -> int:
    if len(shape) != 3 or any(int(value) <= 0 for value in shape):
        raise SmokeContractError(f"invalid native latent shape: {shape}")
    result = 1
    for value in shape:
        result *= int(value)
    return result


def make_rate_summary(
    *,
    main_real_symbols: int,
    edge_dense_real_symbols: int,
    edge_active_real_symbols: int,
    text_utf8_bits: int,
    height: int = 128,
    width: int = 128,
) -> dict[str, Any]:
    if not (0 <= edge_active_real_symbols <= edge_dense_real_symbols):
        raise SmokeContractError("edge active symbols must lie within dense symbols")
    denominator = 3 * height * width
    real_dimensions_per_complex_channel_use = 2
    active_total = main_real_symbols + edge_active_real_symbols
    dense_total = main_real_symbols + edge_dense_real_symbols
    return {
        "native_real_source_dimensions": denominator,
        "real_dimensions_per_complex_channel_use": (
            real_dimensions_per_complex_channel_use
        ),
        "main_real_symbols": main_real_symbols,
        "main_real_dimension_ratio": main_real_symbols / denominator,
        "main_complex_channel_uses": main_real_symbols
        / real_dimensions_per_complex_channel_use,
        "main_complex_cbr": main_real_symbols
        / real_dimensions_per_complex_channel_use
        / denominator,
        # Historical field retained for compatibility.  It is a raw real-
        # coordinate ratio, not the project's complex-channel-use CBR.
        "main_real_cbr": main_real_symbols / denominator,
        "edge_dense_real_symbols": edge_dense_real_symbols,
        "edge_active_real_symbols": edge_active_real_symbols,
        "edge_active_fraction": (
            edge_active_real_symbols / edge_dense_real_symbols
            if edge_dense_real_symbols
            else 0.0
        ),
        "main_plus_edge_active_real_symbols": active_total,
        "main_plus_edge_active_real_dimension_ratio": active_total / denominator,
        "main_plus_edge_active_complex_channel_uses": active_total
        / real_dimensions_per_complex_channel_use,
        "main_plus_edge_active_complex_cbr": active_total
        / real_dimensions_per_complex_channel_use
        / denominator,
        "main_plus_edge_active_real_cbr": active_total / denominator,
        "main_plus_edge_dense_real_symbols": dense_total,
        "main_plus_edge_dense_real_dimension_ratio": dense_total / denominator,
        "main_plus_edge_dense_complex_channel_uses": dense_total
        / real_dimensions_per_complex_channel_use,
        "main_plus_edge_dense_complex_cbr": dense_total
        / real_dimensions_per_complex_channel_use
        / denominator,
        "main_plus_edge_dense_real_cbr": dense_total / denominator,
        "text_utf8_bits": text_utf8_bits,
        "text_channel_symbols": None,
        "common_contract_direct_ranking_allowed": False,
        "direct_ranking_blocker": (
            "text transport is perfect/free in the author protocol; no common-contract "
            "channel-symbol mapping is defined"
        ),
    }


def validate_config(payload: dict[str, Any]) -> None:
    errors: list[str] = []
    if payload.get("status") != "preregistered_before_smoke_outcome":
        errors.append("status must be preregistered_before_smoke_outcome")
    if payload.get("official_val_accessed") is not False:
        errors.append("official_val_accessed must remain false")
    if payload.get("outcome_claims_allowed") is not False:
        errors.append("one-image smoke cannot authorize outcome claims")
    if payload.get("comparison_track") != "author_native_smoke_not_direct_ranking":
        errors.append("comparison track must forbid direct ranking")

    source = payload.get("source", {})
    if source.get("commit") != EXPECTED_SOURCE_COMMIT:
        errors.append("unexpected SGD-JSCC source commit")
    if source.get("source_read_only") is not True:
        errors.append("third-party source must remain read-only")

    channel = payload.get("channel", {})
    if channel.get("type") != "AWGN":
        errors.append("smoke channel must be AWGN")
    if not isinstance(channel.get("seed"), int):
        errors.append("channel seed must be explicit")

    rate = payload.get("rate_instrumentation", {})
    for field in (
        "count_main_channel_tensor",
        "count_edge_dense_tensor",
        "count_edge_nonzero_active_tensor",
        "record_text_utf8_bits_without_claiming_channel_cost",
        "unknown_text_channel_symbols_blocks_common_ranking",
    ):
        if rate.get(field) is not True:
            errors.append(f"rate_instrumentation.{field} must be true")
    if payload.get("overwrite_forbidden") is not True:
        errors.append("smoke output must be non-overwriting")
    if errors:
        raise SmokeContractError("\n".join(errors))


def validate_paths_and_assets(
    payload: dict[str, Any], *, verify_hashes: bool
) -> dict[str, Any]:
    source_path = resolve(payload["source"]["path"])
    observed_commit = git_head(source_path)
    if observed_commit != payload["source"]["commit"]:
        raise SmokeContractError(
            f"source commit mismatch: {observed_commit} != {payload['source']['commit']}"
        )
    tracked_status = git_tracked_status(source_path)
    if tracked_status:
        raise SmokeContractError(
            f"third-party tracked source is not read-only/clean: {tracked_status}"
        )

    image_path = resolve(payload["input"]["image"])
    if not image_path.is_file():
        raise SmokeContractError(f"input image missing: {image_path}")

    checkpoint_dir = resolve(payload["assets"]["checkpoint_dir"])
    checkpoint_status: dict[str, Any] = {}
    for name, expected in payload["assets"]["checkpoints"].items():
        path = checkpoint_dir / name
        exists = path.is_file()
        size = path.stat().st_size if exists else None
        observed_hash = sha256_file(path) if exists and verify_hashes else None
        checkpoint_status[name] = {
            "path": str(path),
            "exists": exists,
            "expected_bytes": int(expected["bytes"]),
            "observed_bytes": size,
            "expected_sha256": expected["sha256"],
            "observed_sha256": observed_hash,
        }
        if not exists or size != int(expected["bytes"]):
            raise SmokeContractError(f"checkpoint size mismatch or missing: {name}")
        if verify_hashes and observed_hash != expected["sha256"]:
            raise SmokeContractError(f"checkpoint SHA-256 mismatch: {name}")

    blip_dir = resolve(payload["assets"]["blip2_dir"])
    blip_status: dict[str, Any] = {}
    for name, expected in payload["assets"]["blip2_files"].items():
        path = blip_dir / name
        exists = path.is_file()
        size = path.stat().st_size if exists else None
        observed_hash = sha256_file(path) if exists and verify_hashes else None
        blip_status[name] = {
            "exists": exists,
            "expected_bytes": int(expected["bytes"]),
            "observed_bytes": size,
            "expected_sha256": expected["sha256"],
            "observed_sha256": observed_hash,
        }
        if exists and size != int(expected["bytes"]):
            raise SmokeContractError(f"BLIP2 shard size mismatch: {name}")
        if verify_hashes and observed_hash != expected["sha256"]:
            raise SmokeContractError(f"BLIP2 shard SHA-256 mismatch: {name}")
    clip_path = resolve(payload["assets"]["clip_model"])
    clip_exists = clip_path.is_file() and clip_path.stat().st_size > 0
    clip_hash = sha256_file(clip_path) if clip_exists and verify_hashes else None
    if verify_hashes and clip_hash != payload["assets"]["clip_sha256"]:
        raise SmokeContractError("OpenAI CLIP ViT-L/14 SHA-256 mismatch")
    hf_home = resolve(payload["assets"]["hf_home"])
    runtime_status = {
        "blip2_dir": str(blip_dir),
        "blip2_ready": (
            all(item["exists"] for item in blip_status.values())
            and (blip_dir / "model.safetensors.index.json").is_file()
        ),
        "blip2_shards": blip_status,
        "clip_model": str(clip_path),
        "clip_ready": clip_exists,
        "clip_expected_sha256": payload["assets"]["clip_sha256"],
        "clip_observed_sha256": clip_hash,
        "hf_home": str(hf_home),
        "scheduler_ready": any(hf_home.rglob("scheduler_config.json"))
        if hf_home.exists()
        else False,
    }
    return {
        "source_path": str(source_path),
        "source_commit": observed_commit,
        "source_tracked_clean": True,
        "input_image": str(image_path),
        "checkpoint_status": checkpoint_status,
        "runtime_status": runtime_status,
    }


class ChannelUseRecorder:
    """Record actual tensors passed to author main and edge AWGN channels."""

    def __init__(self) -> None:
        self.main_real_symbols: int | None = None
        self.edge_dense_real_symbols: int | None = None
        self.edge_active_real_symbols: int | None = None
        self._edge_handle: Any = None

    def attach(self, model: Any) -> None:
        original_main_channel = model.channel

        def counted_main(features: Any) -> Any:
            self.main_real_symbols = int(features.numel() // features.shape[0])
            return original_main_channel(features)

        model.channel = counted_main

        def edge_pre_hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            tensor = inputs[0]
            per_batch = tensor[0]
            self.edge_dense_real_symbols = int(per_batch.numel())
            self.edge_active_real_symbols = int((per_batch != 0).sum().item())

        self._edge_handle = model.canny_transmission_net.channel.register_forward_pre_hook(
            edge_pre_hook
        )

    def close(self) -> None:
        if self._edge_handle is not None:
            self._edge_handle.remove()
            self._edge_handle = None

    def require_complete(self) -> None:
        if self.main_real_symbols is None:
            raise SmokeContractError("main channel-use hook was not observed")
        if self.edge_dense_real_symbols is None or self.edge_active_real_symbols is None:
            raise SmokeContractError("edge channel-use hook was not observed")


def set_seed(seed: int, torch: Any, np: Any) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_smoke(
    config_path: Path,
    payload: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    runtime = validation["runtime_status"]
    missing = [
        name
        for name, ready in (
            ("BLIP2", runtime["blip2_ready"]),
            ("OpenAI CLIP ViT-L/14", runtime["clip_ready"]),
            ("Stable Diffusion scheduler", runtime["scheduler_ready"]),
        )
        if not ready
    ]
    if missing:
        raise SmokeContractError(f"runtime assets missing: {missing}")

    output_dir = resolve(payload["output_dir"])
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output_dir / "config_snapshot.yaml")

    started = time.time()
    try:
        # Set cache/offline controls before importing Transformers, Diffusers,
        # or huggingface_hub because those packages cache environment-derived
        # paths at import time.
        os.environ["HF_HOME"] = str(resolve(payload["assets"]["hf_home"]))
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["DIFFUSERS_OFFLINE"] = "1"

        import numpy as np
        import torch
        import torchvision
        import diffusers
        import huggingface_hub
        import transformers
        import xformers
        from omegaconf import OmegaConf
        from PIL import Image
        from torchvision import transforms
        from torchvision.utils import save_image
        from transformers import AutoProcessor, Blip2ForConditionalGeneration

        expected_environment = payload["environment"]
        observed_versions = {
            "python": platform.python_version(),
            "torch": torch.__version__.split("+")[0],
            "torchvision": torchvision.__version__.split("+")[0],
            "transformers": transformers.__version__,
            "diffusers": diffusers.__version__,
            "huggingface_hub": huggingface_hub.__version__,
            "xformers": xformers.__version__,
        }
        for name in (
            "torch",
            "torchvision",
            "transformers",
            "diffusers",
            "huggingface_hub",
            "xformers",
        ):
            expected = str(expected_environment[f"expected_{name}"])
            if observed_versions[name] != expected:
                raise SmokeContractError(
                    f"{name} version mismatch: {observed_versions[name]} != {expected}"
                )
        if not observed_versions["python"].startswith(
            str(expected_environment["expected_python"]) + "."
        ):
            raise SmokeContractError(
                "python version mismatch: "
                f"{observed_versions['python']} != {expected_environment['expected_python']}.x"
            )

        source_path = resolve(payload["source"]["path"])
        sys.path.insert(0, str(source_path))

        import inference_one as author
        import clip
        from models.test_advanced_network.autoencoderkl import AutoencoderKL
        from models.test_advanced_network.diffusion_element_wise import DiffusionGenerator
        from models.test_advanced_network.mask_diffusion import MDTv2
        from models.test_advanced_network.mask_diffusion_controlnet import MDTv2_ControlNet
        from models.test_advanced_network.muge_model import Mymodel as MuGEModel

        device = torch.device(payload["environment"]["cuda_device"])
        if not torch.cuda.is_available():
            raise SmokeContractError("CUDA is required for the author-native smoke")
        torch.cuda.set_device(device)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
        set_seed(int(payload["channel"]["seed"]), torch, np)

        model_cfg = payload["model"]
        author_config = OmegaConf.create(
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
                    "use_gt_csi": payload["channel"]["use_gt_csi"],
                },
                "th": model_cfg["edge_threshold"],
            }
        )
        author.config = author_config
        author.device = device

        image = Image.open(resolve(payload["input"]["image"])).convert("RGB")
        side = min(image.size)
        transform = transforms.Compose(
            [
                transforms.CenterCrop((side, side)),
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ]
        )
        source_tensor = transform(image).unsqueeze(0).to(device)
        save_image(source_tensor.cpu(), output_dir / "source_preprocessed.png")

        # BLIP2 is needed only for the caption. Release it before loading the
        # JSCC, ControlNet, diffusion, and CLIP models so the author-native
        # computation fits on a single 24 GiB GPU without changing its output.
        processor = AutoProcessor.from_pretrained(
            resolve(payload["assets"]["blip2_dir"]), local_files_only=True
        )
        caption_model = Blip2ForConditionalGeneration.from_pretrained(
            resolve(payload["assets"]["blip2_dir"]),
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        caption_model.processor = processor
        caption_model.to(device).eval()
        with torch.inference_mode():
            caption = author.image_caption(caption_model, source_tensor, device)
        del caption_model, processor
        torch.cuda.empty_cache()

        checkpoint_dir = resolve(payload["assets"]["checkpoint_dir"])
        model = author.JSCC_model(snr=float(payload["channel"]["snr_db"]))
        model.load_state_dict(
            torch.load(checkpoint_dir / "JSCC_model.pth", map_location="cpu")
        )
        model.to(device).eval()
        recorder = ChannelUseRecorder()
        recorder.attach(model)

        # The released MuGE checkpoint contains every encoder parameter. Avoid
        # MuGEModel's redundant EfficientNet-B7 ImageNet bootstrap download;
        # strict loading below restores the exact released runtime state.
        canny_net = MuGEModel(encoder_weights=None)
        muge_payload = torch.load(
            checkpoint_dir / "muge-epoch-19-checkpoint.pth", map_location="cpu"
        )
        canny_net.load_state_dict(muge_payload["state_dict"])
        canny_net.to(device).eval()

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
        denoiser.to(device).eval()

        clip_model, _ = clip.load(
            str(resolve(payload["assets"]["clip_model"])), device=device, jit=False
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

        with torch.inference_mode():
            canny_data, canny_uncertainty = author.generate_canny(
                source_tensor, canny_net, device
            )
            reconstructed, caption_text = model(
                source_tensor,
                pipe=pipeline,
                gt_text=caption,
                canny_data=canny_data.to(device),
                canny_uncertainty=canny_uncertainty.to(device),
                use_semantic=model_cfg["use_semantic"],
                use_controlnet=model_cfg["use_controlnet"],
                use_text=model_cfg["use_text"],
                use_gt_text=model_cfg["use_gt_text"],
                canny_cr=model_cfg["canny_cr"],
                use_jscc_feature=model_cfg["use_jscc_feature"],
                use_gt_csi=payload["channel"]["use_gt_csi"],
                controlnet_scale=model_cfg["controlnet_scale"],
                mask_method=model_cfg["mask_method"],
                diffusion_step=model_cfg["diffusion_step"],
                step_style=model_cfg["step_style"],
                cfg_method=model_cfg["cfg_method"],
                guidance_scale=model_cfg["guidance_scale"],
                scaling_factor=model_cfg["scaling_factor"],
            )
        recorder.require_complete()
        recorder.close()

        reconstructed = reconstructed.clamp(0, 1)
        if list(reconstructed.shape) != payload["success_criteria"]["require_output_shape"]:
            raise SmokeContractError(f"unexpected output shape: {list(reconstructed.shape)}")
        if not torch.isfinite(reconstructed).all():
            raise SmokeContractError("non-finite reconstructed output")
        save_image(reconstructed.cpu(), output_dir / "reconstruction.png")

        mse = torch.mean((source_tensor - reconstructed.to(device)) ** 2).item()
        psnr = float(-10.0 * np.log10(max(mse, 1e-12)))
        caption_strings = list(caption_text)
        text_utf8_bits = sum(len(item.encode("utf-8")) * 8 for item in caption_strings)
        rate_summary = make_rate_summary(
            main_real_symbols=int(recorder.main_real_symbols),
            edge_dense_real_symbols=int(recorder.edge_dense_real_symbols),
            edge_active_real_symbols=int(recorder.edge_active_real_symbols),
            text_utf8_bits=text_utf8_bits,
        )
        (output_dir / "rate_accounting.json").write_text(
            json.dumps(rate_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        result = {
            "status": "PASS",
            "analysis_id": payload["analysis_id"],
            "comparison_track": payload["comparison_track"],
            "official_val_accessed": False,
            "outcome_claims_allowed": False,
            "source_image": payload["input"]["image"],
            "snr_db": payload["channel"]["snr_db"],
            "seed": payload["channel"]["seed"],
            "caption": caption_strings,
            "output_shape": list(reconstructed.shape),
            "psnr_db_smoke_only": psnr,
            "elapsed_seconds": time.time() - started,
            "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
            "rate_accounting": rate_summary,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "transformers": transformers.__version__,
                "diffusers": diffusers.__version__,
                "huggingface_hub": huggingface_hub.__version__,
                "xformers": xformers.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device),
            },
        }
        (output_dir / "summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return result
    except Exception as exc:
        failure = {
            "status": "FAIL",
            "analysis_id": payload.get("analysis_id"),
            "official_val_accessed": False,
            "outcome_claims_allowed": False,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
            "python": platform.python_version(),
        }
        (output_dir / "failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/external_sgdjscc_native_smoke.yaml"
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--verify-checkpoint-hashes", action="store_true")
    args = parser.parse_args()

    config_path = resolve(args.config)
    payload = load_config(config_path)
    validate_config(payload)
    validation = validate_paths_and_assets(
        payload,
        verify_hashes=args.verify_checkpoint_hashes or args.run,
    )
    dry_result = {
        "status": "PASS" if not args.run else "READY",
        "mode": "dry_run" if not args.run else "run",
        "analysis_id": payload["analysis_id"],
        "comparison_track": payload["comparison_track"],
        "official_val_accessed": False,
        "outcome_claims_allowed": False,
        "validation": validation,
        "output_dir": str(resolve(payload["output_dir"])),
        "output_exists": resolve(payload["output_dir"]).exists(),
    }
    if not args.run:
        print(json.dumps(dry_result, ensure_ascii=False, indent=2))
        return
    result = run_smoke(config_path, payload, validation)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
