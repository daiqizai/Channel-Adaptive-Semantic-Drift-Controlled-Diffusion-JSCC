#!/usr/bin/env python3
"""Run one 256x256 SGD-JSCC common-contract integration smoke.

This is deliberately labelled a project-side common-contract adapter rather
than an exact author-native reproduction.  It keeps the released networks and
four-patch inference path, but sends captions through an explicit noisy codec,
transmits only the deterministic active edge coordinates, and accounts for one
frozen 65,536-real-coordinate channel budget.
"""

from __future__ import annotations

import argparse
import binascii
import hashlib
import importlib.util
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
NATIVE_SCRIPT = ROOT / "scripts" / "external_sgdjscc_native_smoke.py"
EXPECTED_SOURCE_COMMIT = "2188acc0dd2805355d3d0d2e478cbc27b46b4da5"

_NATIVE_SPEC = importlib.util.spec_from_file_location(
    "external_sgdjscc_native_smoke_for_common", NATIVE_SCRIPT
)
assert _NATIVE_SPEC is not None and _NATIVE_SPEC.loader is not None
native = importlib.util.module_from_spec(_NATIVE_SPEC)
_NATIVE_SPEC.loader.exec_module(native)


class CommonContractError(RuntimeError):
    """Raised when the frozen common-contract adapter is violated."""


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CommonContractError("config root must be a mapping")
    return payload


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


def bytes_to_bits(value: bytes) -> list[int]:
    return [((byte >> shift) & 1) for byte in value for shift in range(7, -1, -1)]


def bits_to_bytes(bits: list[int]) -> bytes:
    if len(bits) % 8:
        raise CommonContractError("bit length must be divisible by eight")
    output = bytearray()
    for start in range(0, len(bits), 8):
        byte = 0
        for bit in bits[start : start + 8]:
            if int(bit) not in (0, 1):
                raise CommonContractError("packet bits must be binary")
            byte = (byte << 1) | int(bit)
        output.append(byte)
    return bytes(output)


def truncate_utf8(text: str, max_bytes: int) -> tuple[str, bool]:
    if max_bytes <= 0:
        raise CommonContractError("max UTF-8 payload bytes must be positive")
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text, False
    chars: list[str] = []
    used = 0
    for char in text:
        raw = char.encode("utf-8")
        if used + len(raw) > max_bytes:
            break
        chars.append(char)
        used += len(raw)
    return "".join(chars), True


def encode_caption_packet(
    text: str, *, max_payload_bytes: int = 64, crc_initial_value: int = 0xFFFF
) -> tuple[list[int], dict[str, Any]]:
    if max_payload_bytes > 255:
        raise CommonContractError("u8 length cannot represent the payload limit")
    transmitted_text, truncated = truncate_utf8(text, max_payload_bytes)
    payload = transmitted_text.encode("utf-8")
    body = bytes([len(payload)]) + payload.ljust(max_payload_bytes, b"\x00")
    crc = binascii.crc_hqx(body, crc_initial_value)
    packet = body + crc.to_bytes(2, "big")
    return bytes_to_bits(packet), {
        "sender_text": text,
        "transmitted_text": transmitted_text,
        "sender_utf8_bytes": len(text.encode("utf-8")),
        "transmitted_utf8_bytes": len(payload),
        "truncated": truncated,
        "packet_bytes": len(packet),
        "packet_bits": len(packet) * 8,
        "crc16": crc,
    }


def decode_caption_packet(
    bits: list[int], *, max_payload_bytes: int = 64, crc_initial_value: int = 0xFFFF
) -> dict[str, Any]:
    expected_bytes = 1 + max_payload_bytes + 2
    if len(bits) != expected_bytes * 8:
        raise CommonContractError(
            f"caption packet has {len(bits)} bits, expected {expected_bytes * 8}"
        )
    packet = bits_to_bytes(bits)
    body, observed_crc_bytes = packet[:-2], packet[-2:]
    observed_crc = int.from_bytes(observed_crc_bytes, "big")
    expected_crc = binascii.crc_hqx(body, crc_initial_value)
    length = int(body[0])
    if length > max_payload_bytes:
        return {
            "decoded_text": "",
            "packet_ok": False,
            "failure": "length_out_of_range",
            "observed_crc16": observed_crc,
            "expected_crc16": expected_crc,
        }
    if observed_crc != expected_crc:
        return {
            "decoded_text": "",
            "packet_ok": False,
            "failure": "crc_mismatch",
            "observed_crc16": observed_crc,
            "expected_crc16": expected_crc,
        }
    try:
        decoded = body[1 : 1 + length].decode("utf-8")
    except UnicodeDecodeError:
        return {
            "decoded_text": "",
            "packet_ok": False,
            "failure": "invalid_utf8",
            "observed_crc16": observed_crc,
            "expected_crc16": expected_crc,
        }
    return {
        "decoded_text": decoded,
        "packet_ok": True,
        "failure": None,
        "observed_crc16": observed_crc,
        "expected_crc16": expected_crc,
    }


def repetition_encode(bits: list[int], repetitions: int) -> list[int]:
    if repetitions <= 0 or repetitions % 2 != 1:
        raise CommonContractError("repetition count must be a positive odd integer")
    return [bit for bit in bits for _ in range(repetitions)]


def repetition_majority_decode(hard_bits: list[int], repetitions: int) -> list[int]:
    if repetitions <= 0 or repetitions % 2 != 1:
        raise CommonContractError("repetition count must be a positive odd integer")
    if len(hard_bits) % repetitions:
        raise CommonContractError("coded bit length is not divisible by repetitions")
    threshold = repetitions // 2
    return [
        int(sum(hard_bits[start : start + repetitions]) > threshold)
        for start in range(0, len(hard_bits), repetitions)
    ]


def make_rate_plan(payload: dict[str, Any]) -> dict[str, Any]:
    rate = payload["rate_contract"]
    main = int(rate["main_real_symbols"])
    edge = int(rate["active_edge_real_symbols"])
    text = int(rate["text_real_symbols"])
    padding = int(rate["no_information_padding_real_symbols"])
    total = main + edge + text + padding
    source_dimensions = int(rate["source_real_dimensions"])
    reals_per_complex = int(rate["real_dimensions_per_complex_channel_use"])
    if total % reals_per_complex:
        raise CommonContractError("total real-symbol count must map to whole complex uses")
    complex_uses = total // reals_per_complex
    return {
        "source_real_dimensions": source_dimensions,
        "real_dimensions_per_complex_channel_use": reals_per_complex,
        "main_real_symbols": main,
        "active_edge_real_symbols": edge,
        "text_real_symbols": text,
        "no_information_padding_real_symbols": padding,
        "information_bearing_real_symbols": main + edge + text,
        "total_real_symbols": total,
        "total_complex_channel_uses": complex_uses,
        "cbr_complex_channel_uses_per_source_real_dimension": (
            complex_uses / source_dimensions
        ),
    }


def validate_config(payload: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if payload.get("status") != "preregistered_before_common_smoke_outcome":
        errors.append("status must be preregistered_before_common_smoke_outcome")
    if payload.get("official_val_accessed") is not False:
        errors.append("official validation must remain sealed")
    if payload.get("outcome_claims_allowed") is not False:
        errors.append("one-image common smoke cannot authorize outcome claims")
    if payload.get("comparison_track") != "common_contract_adapter_smoke_not_stage_ranking":
        errors.append("comparison track must remain a non-ranking common smoke")
    if payload.get("overwrite_forbidden") is not True:
        errors.append("output overwriting must be forbidden")

    source = payload.get("source", {})
    if source.get("commit") != EXPECTED_SOURCE_COMMIT:
        errors.append("unexpected SGD-JSCC source commit")
    if source.get("source_read_only") is not True:
        errors.append("third-party source must remain read-only")

    input_config = payload.get("input", {})
    if input_config.get("source_image_size") != 256:
        errors.append("common source image size must be 256")
    if input_config.get("patch_size") != 128:
        errors.append("author patch size must remain 128")
    if input_config.get("expected_patch_count") != 4:
        errors.append("256x256 common input must produce four author patches")

    channel = payload.get("channel", {})
    if channel.get("type") != "AWGN":
        errors.append("common channel must be AWGN")
    if not isinstance(channel.get("channel_seed"), int):
        errors.append("channel seed must be explicit")
    if channel.get("noise_variance_convention") not in {
        "author_real_awgn_one_over_linear_snr",
        "complex_awgn_per_real_half_variance",
    }:
        errors.append("AWGN real-coordinate variance convention must be explicit")
    layout = channel.get("canonical_noise_layout", [])
    if [item.get("name") for item in layout] != [
        "main_latent",
        "active_edge",
        "text_caption",
        "no_information_padding",
    ]:
        errors.append("canonical noise layout/order differs from the frozen contract")

    text = payload.get("text_transport", {})
    if text.get("max_utf8_payload_bytes_per_patch") != 64:
        errors.append("text payload limit must remain 64 bytes per patch")
    if text.get("packet_bits_per_patch") != 536:
        errors.append("caption packet must remain 536 bits per patch")
    if text.get("repetition") != 21:
        errors.append("caption repetition must remain 21")
    if text.get("crc_failure_action") != "erase_caption_to_empty_string":
        errors.append("CRC failure must fail closed to an empty caption")

    try:
        plan = make_rate_plan(payload)
    except (KeyError, TypeError, ValueError, CommonContractError) as exc:
        errors.append(f"invalid rate plan: {exc}")
        plan = {}
    if plan:
        rate = payload["rate_contract"]
        if plan["total_real_symbols"] != 65536:
            errors.append("common total must be exactly 65,536 real coordinates")
        if plan["total_complex_channel_uses"] != 32768:
            errors.append("common total must be 32,768 complex channel uses")
        if abs(
            plan["cbr_complex_channel_uses_per_source_real_dimension"] - 1.0 / 6.0
        ) > 1e-15:
            errors.append("common CBR must equal 1/6")
        if abs(float(rate.get("target_cbr", -1.0)) - 1.0 / 6.0) > 1e-15:
            errors.append("declared target CBR must equal 1/6")
        layout_total = sum(int(item.get("real_symbols", -1)) for item in layout)
        if layout_total != plan["total_real_symbols"]:
            errors.append("canonical noise layout does not sum to the total budget")

    if errors:
        raise CommonContractError("\n".join(errors))
    return plan


def validate_paths_and_assets(
    payload: dict[str, Any], *, verify_hashes: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_path = resolve(payload["source"]["path"])
    if git_head(source_path) != payload["source"]["commit"]:
        raise CommonContractError("third-party source commit mismatch")
    tracked_status = git_tracked_status(source_path)
    if tracked_status:
        raise CommonContractError(f"third-party tracked files are dirty: {tracked_status}")

    native_config_path = resolve(payload["source"]["native_asset_and_model_config"])
    native_payload = native.load_config(native_config_path)
    native.validate_config(native_payload)
    native_validation = native.validate_paths_and_assets(
        native_payload, verify_hashes=verify_hashes
    )

    image_path = resolve(payload["input"]["image"])
    manifest_path = resolve(payload["input"]["source_manifest"])
    if not image_path.is_file():
        raise CommonContractError(f"common input image missing: {image_path}")
    if not manifest_path.is_file():
        raise CommonContractError(f"source manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if image_path.name not in manifest.get("eval_names", []):
        raise CommonContractError("common smoke input is not in the frozen eval manifest")

    return native_payload, {
        "source_path": str(source_path),
        "source_commit": payload["source"]["commit"],
        "source_tracked_clean": True,
        "input_image": str(image_path),
        "input_sha256": sha256_file(image_path),
        "source_manifest": str(manifest_path),
        "native_asset_config": str(native_config_path),
        "native_asset_validation": native_validation,
    }


def slice_canonical_noise(vector: Any, payload: dict[str, Any]) -> dict[str, Any]:
    segments: dict[str, Any] = {}
    offset = 0
    for item in payload["channel"]["canonical_noise_layout"]:
        count = int(item["real_symbols"])
        segments[str(item["name"])] = vector[offset : offset + count]
        offset += count
    if offset != int(payload["rate_contract"]["total_real_symbols"]):
        raise CommonContractError("canonical noise vector was not consumed exactly")
    return segments


def transmit_captions(
    captions: list[str], text_noise: Any, snr_db: float, payload: dict[str, Any], torch: Any
) -> tuple[list[str], list[dict[str, Any]]]:
    text = payload["text_transport"]
    repetitions = int(text["repetition"])
    max_bytes = int(text["max_utf8_payload_bytes_per_patch"])
    crc_initial = int(text["crc_initial_value"])
    symbols_per_patch = int(text["real_symbols_per_patch"])
    if len(captions) != int(text["captions_per_source_image"]):
        raise CommonContractError("unexpected caption count")
    if int(text_noise.numel()) != symbols_per_patch * len(captions):
        raise CommonContractError("text noise segment has the wrong size")

    noise_rows = text_noise.reshape(len(captions), symbols_per_patch).to(torch.float32)
    variance_factor = (
        0.5
        if payload["channel"]["noise_variance_convention"]
        == "complex_awgn_per_real_half_variance"
        else 1.0
    )
    sigma = (variance_factor / (10.0 ** (float(snr_db) / 10.0))) ** 0.5
    decoded_texts: list[str] = []
    records: list[dict[str, Any]] = []
    for index, caption in enumerate(captions):
        packet_bits, sender = encode_caption_packet(
            caption,
            max_payload_bytes=max_bytes,
            crc_initial_value=crc_initial,
        )
        coded_bits = repetition_encode(packet_bits, repetitions)
        if len(coded_bits) != symbols_per_patch:
            raise CommonContractError("runtime caption code length differs from preregistration")
        tx = torch.tensor(coded_bits, dtype=torch.float32).mul(2.0).sub(1.0)
        rx = tx + noise_rows[index] * sigma
        hard_bits = (rx > 0).to(torch.int64).tolist()
        decoded_bits = repetition_majority_decode(hard_bits, repetitions)
        receiver = decode_caption_packet(
            decoded_bits,
            max_payload_bytes=max_bytes,
            crc_initial_value=crc_initial,
        )
        decoded_texts.append(str(receiver["decoded_text"]))
        records.append(
            {
                "patch_index": index,
                **sender,
                **receiver,
                "coded_real_symbols": len(coded_bits),
                "hard_symbol_errors": sum(
                    int(left != right) for left, right in zip(coded_bits, hard_bits)
                ),
                "decoded_packet_bit_errors": sum(
                    int(left != right) for left, right in zip(packet_bits, decoded_bits)
                ),
            }
        )
    return decoded_texts, records


class SparseCommonChannelAdapter:
    """Inject frozen noise into author main and deterministic active edge coordinates."""

    def __init__(
        self,
        *,
        main_noise: Any,
        edge_noise: Any,
        expected_patches: int,
        main_per_patch: int,
        edge_active_per_patch: int,
        noise_variance_factor: float,
    ) -> None:
        self.main_noise = main_noise
        self.edge_noise = edge_noise
        self.expected_patches = expected_patches
        self.main_per_patch = main_per_patch
        self.edge_active_per_patch = edge_active_per_patch
        self.noise_variance_factor = float(noise_variance_factor)
        if self.noise_variance_factor not in {0.5, 1.0}:
            raise CommonContractError("unsupported real-coordinate AWGN variance factor")
        self.main_real_symbols: int | None = None
        self.edge_dense_real_symbols: int | None = None
        self.edge_active_real_symbols: int | None = None

    def attach(self, model: Any, torch: Any) -> None:
        def common_main(features: Any) -> Any:
            if int(features.shape[0]) != self.expected_patches:
                raise CommonContractError("main channel batch does not equal four patches")
            per_patch = int(features[0].numel())
            if per_patch != self.main_per_patch:
                raise CommonContractError("unexpected author main latent size")
            self.main_real_symbols = int(features.numel())
            z = self.main_noise.to(device=features.device, dtype=features.dtype).reshape_as(
                features
            )
            norm_2 = torch.linalg.norm(features.flatten(start_dim=1), ord=2, dim=1)
            noise_scale = torch.sqrt(
                self.noise_variance_factor
                * (norm_2.square() / per_patch)
                / (10.0 ** (float(model.snr) / 10.0))
            ).reshape([-1, 1, 1, 1])
            return features + z * noise_scale

        model.channel = common_main

        def common_edge(tx: Any, snr: Any) -> Any:
            if int(tx.shape[0]) != self.expected_patches:
                raise CommonContractError("edge channel batch does not equal four patches")
            active_masks = tx != 0
            counts = [int(row.sum().item()) for row in active_masks]
            if any(count != self.edge_active_per_patch for count in counts):
                raise CommonContractError(
                    f"runtime active edge counts differ from frozen mask: {counts}"
                )
            self.edge_dense_real_symbols = int(tx.numel())
            self.edge_active_real_symbols = sum(counts)
            z_rows = self.edge_noise.to(device=tx.device, dtype=tx.dtype).reshape(
                self.expected_patches, self.edge_active_per_patch
            )
            output = torch.zeros_like(tx)
            noise_var = self.noise_variance_factor / (10.0 ** (snr / 10.0))
            for index in range(self.expected_patches):
                mask = active_masks[index]
                output[index, mask] = (
                    tx[index, mask]
                    + z_rows[index] * torch.sqrt(noise_var[index]).reshape(())
                )
            return output

        model.canny_transmission_net.channel.forward = common_edge

    def require_complete(self) -> None:
        if self.main_real_symbols is None:
            raise CommonContractError("main common channel was not observed")
        if self.edge_dense_real_symbols is None or self.edge_active_real_symbols is None:
            raise CommonContractError("edge common channel was not observed")


def run_smoke(
    config_path: Path,
    payload: dict[str, Any],
    native_payload: dict[str, Any],
    validation: dict[str, Any],
    rate_plan: dict[str, Any],
) -> dict[str, Any]:
    runtime = validation["native_asset_validation"]["runtime_status"]
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
        raise CommonContractError(f"runtime assets missing: {missing}")

    output_dir = resolve(payload["output_dir"])
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output_dir / "config_snapshot.yaml")
    shutil.copy2(
        resolve(payload["source"]["native_asset_and_model_config"]),
        output_dir / "native_config_snapshot.yaml",
    )

    started = time.time()
    try:
        os.environ["HF_HOME"] = str(resolve(native_payload["assets"]["hf_home"]))
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

        observed_versions = {
            "python": platform.python_version(),
            "torch": torch.__version__.split("+")[0],
            "torchvision": torchvision.__version__.split("+")[0],
            "transformers": transformers.__version__,
            "diffusers": diffusers.__version__,
            "huggingface_hub": huggingface_hub.__version__,
            "xformers": xformers.__version__,
        }
        expected_environment = native_payload["environment"]
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
                raise CommonContractError(
                    f"{name} version mismatch: {observed_versions[name]} != {expected}"
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
            raise CommonContractError("CUDA is required for the SGD-JSCC common smoke")
        torch.cuda.set_device(device)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
        channel_seed = int(payload["channel"]["channel_seed"])
        native.set_seed(channel_seed, torch, np)

        noise_generator = torch.Generator(device="cpu")
        noise_generator.manual_seed(channel_seed)
        canonical_noise = torch.randn(
            int(rate_plan["total_real_symbols"]),
            generator=noise_generator,
            dtype=torch.float32,
        )
        noise_segments = slice_canonical_noise(canonical_noise, payload)
        canonical_noise_sha256 = hashlib.sha256(
            canonical_noise.numpy().tobytes()
        ).hexdigest()

        model_cfg = native_payload["model"]
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
        if image.size != (256, 256):
            raise CommonContractError(f"common input must be 256x256, got {image.size}")
        source_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
        patches, patch_meta = author.split_image_v2(source_tensor)
        if int(patches.shape[0]) != int(payload["input"]["expected_patch_count"]):
            raise CommonContractError(f"unexpected author patch count: {patches.shape[0]}")
        if list(patches.shape[1:]) != [3, 128, 128]:
            raise CommonContractError(f"unexpected patch shape: {list(patches.shape)}")
        save_image(source_tensor.cpu(), output_dir / "source.png")
        save_image(patches.cpu(), output_dir / "source_patches.png", nrow=2)

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
        with torch.inference_mode():
            sender_caption_nested = author.image_caption(caption_model, patches, device)
        sender_captions = list(sender_caption_nested[0])
        del caption_model, processor
        torch.cuda.empty_cache()

        decoded_captions, text_records = transmit_captions(
            sender_captions,
            noise_segments["text_caption"],
            float(payload["channel"]["snr_db"]),
            payload,
            torch,
        )
        (output_dir / "text_transport.json").write_text(
            json.dumps(text_records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        checkpoint_dir = resolve(native_payload["assets"]["checkpoint_dir"])
        model = author.JSCC_model(snr=float(payload["channel"]["snr_db"]))
        model.load_state_dict(
            torch.load(checkpoint_dir / "JSCC_model.pth", map_location="cpu")
        )
        model.to(device).eval()
        adapter = SparseCommonChannelAdapter(
            main_noise=noise_segments["main_latent"],
            edge_noise=noise_segments["active_edge"],
            expected_patches=4,
            main_per_patch=4096,
            edge_active_per_patch=832,
            noise_variance_factor=(
                0.5
                if payload["channel"]["noise_variance_convention"]
                == "complex_awgn_per_real_half_variance"
                else 1.0
            ),
        )
        adapter.attach(model, torch)

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
            str(resolve(native_payload["assets"]["clip_model"])),
            device=device,
            jit=False,
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
                use_gt_csi=payload["channel"]["use_gt_csi"],
                controlnet_scale=model_cfg["controlnet_scale"],
                mask_method=model_cfg["mask_method"],
                diffusion_step=model_cfg["diffusion_step"],
                step_style=model_cfg["step_style"],
                cfg_method=model_cfg["cfg_method"],
                guidance_scale=model_cfg["guidance_scale"],
                scaling_factor=model_cfg["scaling_factor"],
            )
        adapter.require_complete()
        reconstructed_patches = reconstructed_patches.clamp(0, 1)
        reconstructed = author.merge_image_v2(reconstructed_patches, patch_meta).clamp(0, 1)
        if list(reconstructed.shape) != payload["success_criteria"]["require_output_shape"]:
            raise CommonContractError(
                f"unexpected common output shape: {list(reconstructed.shape)}"
            )
        if not torch.isfinite(reconstructed).all():
            raise CommonContractError("common reconstruction contains non-finite values")
        save_image(reconstructed_patches.cpu(), output_dir / "reconstruction_patches.png", nrow=2)
        save_image(reconstructed.cpu(), output_dir / "reconstruction.png")

        observed_rate = dict(rate_plan)
        observed_rate.update(
            {
                "observed_main_real_symbols": int(adapter.main_real_symbols),
                "observed_edge_dense_tensor_elements": int(
                    adapter.edge_dense_real_symbols
                ),
                "observed_active_edge_real_symbols": int(
                    adapter.edge_active_real_symbols
                ),
                "observed_text_real_symbols": sum(
                    int(record["coded_real_symbols"]) for record in text_records
                ),
                "canonical_noise_sha256": canonical_noise_sha256,
                "common_contract_rate_gate_passed": True,
                "common_contract_direct_ranking_allowed_after_stage_metrics": True,
                "one_image_smoke_direct_ranking_allowed": False,
            }
        )
        expected_observed = {
            "observed_main_real_symbols": rate_plan["main_real_symbols"],
            "observed_active_edge_real_symbols": rate_plan["active_edge_real_symbols"],
            "observed_text_real_symbols": rate_plan["text_real_symbols"],
        }
        for field, expected in expected_observed.items():
            if int(observed_rate[field]) != int(expected):
                raise CommonContractError(
                    f"runtime rate mismatch for {field}: {observed_rate[field]} != {expected}"
                )
        (output_dir / "rate_accounting.json").write_text(
            json.dumps(observed_rate, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        mse = torch.mean((source_tensor - reconstructed) ** 2).item()
        psnr = float(-10.0 * np.log10(max(mse, 1e-12)))
        patch_psnrs = []
        for index in range(4):
            patch_mse = torch.mean(
                (patches[index] - reconstructed_patches[index]) ** 2
            ).item()
            patch_psnrs.append(float(-10.0 * np.log10(max(patch_mse, 1e-12))))

        result = {
            "status": "PASS",
            "analysis_id": payload["analysis_id"],
            "comparison_track": payload["comparison_track"],
            "method_label": "SGD-JSCC common-contract adapter (not author-native)",
            "official_val_accessed": False,
            "outcome_claims_allowed": False,
            "source_image": payload["input"]["image"],
            "snr_db": payload["channel"]["snr_db"],
            "channel_seed": channel_seed,
            "noise_variance_convention": payload["channel"][
                "noise_variance_convention"
            ],
            "patch_count": int(patches.shape[0]),
            "sender_captions": sender_captions,
            "decoded_captions": decoded_captions,
            "used_captions": list(used_caption_text),
            "caption_packets_ok": sum(bool(item["packet_ok"]) for item in text_records),
            "caption_packet_count": len(text_records),
            "caption_decoded_bit_errors": sum(
                int(item["decoded_packet_bit_errors"]) for item in text_records
            ),
            "output_shape": list(reconstructed.shape),
            "psnr_db_smoke_only": psnr,
            "patch_psnr_db_smoke_only": patch_psnrs,
            "elapsed_seconds": time.time() - started,
            "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
            "rate_accounting": observed_rate,
            "semantic_claim": "forbidden_for_one_image_integration_smoke",
            "environment": {
                **observed_versions,
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
        "--config", default="configs/external_sgdjscc_common_smoke.yaml"
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--verify-checkpoint-hashes", action="store_true")
    args = parser.parse_args()

    config_path = resolve(args.config)
    payload = load_config(config_path)
    rate_plan = validate_config(payload)
    native_payload, validation = validate_paths_and_assets(
        payload, verify_hashes=args.verify_checkpoint_hashes or args.run
    )
    dry_result = {
        "status": "PASS" if not args.run else "READY",
        "mode": "dry_run" if not args.run else "run",
        "analysis_id": payload["analysis_id"],
        "comparison_track": payload["comparison_track"],
        "official_val_accessed": False,
        "outcome_claims_allowed": False,
        "rate_plan": rate_plan,
        "validation": validation,
        "output_dir": str(resolve(payload["output_dir"])),
        "output_exists": resolve(payload["output_dir"]).exists(),
    }
    if not args.run:
        print(json.dumps(dry_result, ensure_ascii=False, indent=2))
        return
    result = run_smoke(
        config_path, payload, native_payload, validation, rate_plan
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
