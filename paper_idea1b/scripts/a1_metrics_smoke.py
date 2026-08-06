#!/usr/bin/env python3
"""Smoke-test A1 full-frame LPIPS/DISTS/CLIP metrics on the largest CLIC image."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchmetrics.image.dists import DeepImageStructureAndTextureSimilarity
from torchvision.transforms.functional import pil_to_tensor


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
ARMS = (
    "s33_strong",
    "swin_official_base_sa",
    "swin_capacity_matched_sa",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="paper_idea1b/configs/a1_discriminative_benchmark.yaml"
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_rgb(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        tensor = pil_to_tensor(image.convert("RGB")).float().div_(255.0)
    return tensor.unsqueeze(0).to(device)


def measure(
    operation: Callable[[], float], device: torch.device
) -> dict[str, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    value = float(operation())
    torch.cuda.synchronize(device)
    return {
        "value": value,
        "wall_ms": (time.perf_counter() - started) * 1000.0,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("metric smoke requires CUDA")
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    formal_output = resolve(config["analysis"]["output"])
    reconstruction_summary = (
        formal_output / "summary_clic2020_test_reconstruction.json"
    )
    if not reconstruction_summary.is_file():
        raise FileNotFoundError("CLIC reconstruction must complete before metric smoke")
    reconstruction_rows = read_csv(formal_output / "per_sample_clic2020_test.csv")
    source_rows = [
        row
        for row in read_csv(resolve(config["inputs"]["source_manifest"]["path"]))
        if row["dataset"] == "clic2020_test"
    ]
    largest = max(source_rows, key=lambda row: int(row["width"]) * int(row["height"]))
    sample_id = largest["sample_id"]
    source_path = resolve(largest["path"])
    selected = {
        row["arm"]: row
        for row in reconstruction_rows
        if row["sample_id"] == sample_id
        and int(row["seed"]) == int(config["smoke"]["seed"])
        and float(row["snr_db"]) == float(config["smoke"]["snr_db"])
    }
    if set(selected) != set(ARMS):
        raise RuntimeError("largest-image reconstruction rows are incomplete")

    output = ROOT / "paper_idea1b/outputs/SMOKE-IDEA1B-A1-METRICS-002"
    output.mkdir(parents=True, exist_ok=False)
    results: dict[str, Any] = {
        "status": "running",
        "sample_id": sample_id,
        "source_path": source_path.relative_to(ROOT).as_posix(),
        "width": int(largest["width"]),
        "height": int(largest["height"]),
        "seed": int(config["smoke"]["seed"]),
        "snr_db": float(config["smoke"]["snr_db"]),
        "device": torch.cuda.get_device_name(device),
        "metrics": {},
        "official_imagenette_validation_accessed": False,
    }

    source = load_rgb(source_path, device)
    reconstructions = {
        arm: load_rgb(resolve(selected[arm]["reconstruction_path"]), device)
        for arm in ARMS
    }
    crop_source = source[:, :, :256, :256]

    import lpips

    loaded = time.perf_counter()
    lpips_model = lpips.LPIPS(net="alex").to(device).eval().requires_grad_(False)
    results["metrics"]["lpips"] = {
        "model_load_ms": (time.perf_counter() - loaded) * 1000.0,
        "arms": {},
    }
    with torch.inference_mode():
        _ = lpips_model(crop_source * 2 - 1, crop_source * 2 - 1).item()
        for arm in ARMS:
            reconstruction = reconstructions[arm]
            results["metrics"]["lpips"]["arms"][arm] = measure(
                lambda reconstruction=reconstruction: lpips_model(
                    reconstruction * 2 - 1, source * 2 - 1
                ).item(),
                device,
            )
    del lpips_model
    torch.cuda.empty_cache()

    loaded = time.perf_counter()
    dists_model = DeepImageStructureAndTextureSimilarity(reduction="mean").to(device)
    results["metrics"]["dists"] = {
        "model_load_ms": (time.perf_counter() - loaded) * 1000.0,
        "arms": {},
    }
    with torch.inference_mode():
        _ = dists_model(crop_source, crop_source).item()
        dists_model.reset()
        for arm in ARMS:
            reconstruction = reconstructions[arm]

            def dists_operation(reconstruction: torch.Tensor = reconstruction) -> float:
                value = dists_model(reconstruction, source).item()
                dists_model.reset()
                return value

            results["metrics"]["dists"]["arms"][arm] = measure(
                dists_operation, device
            )
    del dists_model
    torch.cuda.empty_cache()

    import open_clip

    clip_config = config["metrics"]["clip"]
    clip_checkpoint = resolve(clip_config["checkpoint"])
    if sha256_file(clip_checkpoint) != clip_config["sha256"]:
        raise RuntimeError("CLIP checkpoint SHA mismatch")
    loaded = time.perf_counter()
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        model_name=clip_config["model_name"],
        pretrained=str(clip_checkpoint),
        precision="fp32",
        device=device,
        # The SHA-frozen local OpenAI CLIP weight is a TorchScript archive.
        # PyTorch 2.6 rejects TorchScript archives under weights_only=True.
        weights_only=False,
    )
    clip_model = clip_model.eval().requires_grad_(False)
    results["metrics"]["clip_image_cosine"] = {
        "model_load_ms": (time.perf_counter() - loaded) * 1000.0,
        "arms": {},
    }
    with Image.open(source_path) as image:
        clip_source = clip_preprocess(image.convert("RGB")).to(device)
    with torch.inference_mode():
        _ = clip_model.encode_image(torch.stack((clip_source, clip_source))).float()
        for arm in ARMS:
            reconstruction_path = resolve(selected[arm]["reconstruction_path"])
            with Image.open(reconstruction_path) as image:
                clip_reconstruction = clip_preprocess(image.convert("RGB")).to(device)

            def clip_operation(
                clip_reconstruction: torch.Tensor = clip_reconstruction,
            ) -> float:
                encoded = F.normalize(
                    clip_model.encode_image(
                        torch.stack((clip_source, clip_reconstruction))
                    ).float(),
                    dim=-1,
                )
                return (encoded[0] * encoded[1]).sum().item()

            results["metrics"]["clip_image_cosine"]["arms"][arm] = measure(
                clip_operation, device
            )

    results["status"] = "pass"
    results["config_sha256"] = sha256_file(config_path)
    results["script_sha256"] = sha256_file(SCRIPT)
    (output / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
