from __future__ import annotations

import argparse
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
from torchvision.transforms import functional as TF
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cadsd_jscc.metrics import ms_ssim_per_sample, psnr_per_sample, ssim_per_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M1 blind diffusion refinement on exported DeepJSCC images.")
    parser.add_argument("--config", default="configs/s3_m1_blind_diffusion_coco256_awgn.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--snrs", default=None, help="Comma-separated SNR list, overriding config.")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--allow-download", action="store_true", help="Allow diffusers to download model weights.")
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and write no experiment output.")
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def parse_snrs(value: str | None, config: dict[str, Any]) -> list[float]:
    if value is None:
        return [float(item) for item in config["snrs"]]
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def snr_name(snr: float) -> str:
    if float(snr).is_integer():
        return f"snr_{int(snr):02d}db"
    return f"snr_{str(snr).replace('.', 'p')}db"


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def load_rgb_tensor(path: Path) -> torch.Tensor:
    return TF.to_tensor(Image.open(path).convert("RGB"))


def load_rgb_pil(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    return TF.to_pil_image(tensor.detach().cpu().clamp(0, 1))


def list_sample_names(original_dir: Path, reconstruction_dir: Path, num_samples: int) -> list[str]:
    original_names = {path.name for path in original_dir.glob("*.png")}
    reconstruction_names = {path.name for path in reconstruction_dir.glob("*.png")}
    names = sorted(original_names & reconstruction_names)
    if len(names) < num_samples:
        raise RuntimeError(
            f"Need {num_samples} matched samples, found {len(names)} in {original_dir} and {reconstruction_dir}"
        )
    return names[:num_samples]


def load_pipeline(config: dict[str, Any], device: torch.device, allow_download: bool):
    from diffusers import StableDiffusionImg2ImgPipeline

    diffusion = config["diffusion"]
    cache_dir = resolve_project_path(diffusion["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))

    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "cache_dir": str(cache_dir),
        "local_files_only": bool(diffusion.get("local_files_only", True)) and not allow_download,
    }
    if bool(diffusion.get("disable_safety_checker", False)):
        kwargs["safety_checker"] = None
        kwargs["requires_safety_checker"] = False

    pipeline_name = str(diffusion.get("pipeline", "StableDiffusionImg2ImgPipeline"))
    if pipeline_name != "StableDiffusionImg2ImgPipeline":
        raise ValueError(f"Unsupported diffusion pipeline: {pipeline_name}")

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(str(diffusion["model_id"]), **kwargs)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    return pipe


def try_load_lpips(device: torch.device, cache_root: Path):
    try:
        os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))
        import lpips

        model = lpips.LPIPS(net="alex", verbose=False).to(device)
        model.eval()
        return model, None
    except Exception as exc:  # noqa: BLE001 - optional metric should not abort the experiment.
        return None, f"{type(exc).__name__}: {exc}"


def compute_pair_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    lpips_model,
    device: torch.device,
) -> dict[str, float | None]:
    reference = reference.to(device)
    candidate = candidate.to(device)
    metrics: dict[str, float | None] = {}
    with torch.no_grad():
        mse_values = F.mse_loss(candidate, reference, reduction="none").flatten(start_dim=1).mean(dim=1)
        metrics["mse"] = mean(mse_values.detach().cpu().tolist())
        metrics["psnr_db"] = mean(psnr_per_sample(candidate, reference).detach().cpu().tolist())
        metrics["ssim"] = mean(ssim_per_sample(candidate, reference).detach().cpu().tolist())
        metrics["ms_ssim"] = mean(ms_ssim_per_sample(candidate, reference).detach().cpu().tolist())
        if lpips_model is not None:
            lpips_values = lpips_model(candidate * 2.0 - 1.0, reference * 2.0 - 1.0)
            metrics["lpips"] = mean(lpips_values.flatten().detach().cpu().tolist())
        else:
            metrics["lpips"] = None
    return metrics


def refine_batch(
    pipe,
    images: list[Image.Image],
    config: dict[str, Any],
    generator: torch.Generator,
) -> list[Image.Image]:
    diffusion = config["diffusion"]
    result = pipe(
        prompt=[str(diffusion.get("prompt", ""))] * len(images),
        negative_prompt=[str(diffusion.get("negative_prompt", ""))] * len(images),
        image=images,
        strength=float(diffusion["strength"]),
        num_inference_steps=int(diffusion["num_inference_steps"]),
        guidance_scale=float(diffusion["guidance_scale"]),
        generator=generator,
    )
    return list(result.images)


def evaluate_snr(
    snr: float,
    names: list[str],
    config: dict[str, Any],
    pipe,
    lpips_model,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    diffusion = config["diffusion"]
    original_dir = resolve_project_path(config["inputs"]["original_dir"])
    reconstruction_dir = resolve_project_path(config["inputs"]["m0_export_dir"]) / "exports" / snr_name(snr) / "reconstruction"
    refined_dir = output_dir / "exports" / snr_name(snr) / "refined"
    refined_dir.mkdir(parents=True, exist_ok=False)

    batch_size = int(diffusion["batch_size"])
    references: list[torch.Tensor] = []
    reconstructions: list[torch.Tensor] = []
    refined_tensors: list[torch.Tensor] = []
    inference_times: list[float] = []

    for start in range(0, len(names), batch_size):
        batch_names = names[start : start + batch_size]
        input_images = [load_rgb_pil(reconstruction_dir / name) for name in batch_names]
        generator = torch.Generator(device=device).manual_seed(int(config["seed"]) + int(round(snr * 1000)) + start)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        begin = time.perf_counter()
        refined_images = refine_batch(pipe, input_images, config, generator)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_times.append(time.perf_counter() - begin)

        for name, refined_image in zip(batch_names, refined_images):
            refined_image.save(refined_dir / name)

    for name in names:
        references.append(load_rgb_tensor(original_dir / name))
        reconstructions.append(load_rgb_tensor(reconstruction_dir / name))
        refined_tensors.append(load_rgb_tensor(refined_dir / name))

    reference = torch.stack(references)
    reconstruction = torch.stack(reconstructions)
    refined = torch.stack(refined_tensors)

    sample_count = min(int(config["evaluation"]["sample_grid_count"]), len(names))
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    save_image(
        torch.cat([reference[:sample_count], reconstruction[:sample_count], refined[:sample_count]], dim=0),
        sample_dir / f"{snr_name(snr)}_original_reconstruction_refined.png",
        nrow=sample_count,
    )

    return {
        "snr_db": float(snr),
        "num_images": len(names),
        "sample_names": names,
        "input_reconstruction_dir": project_relative(reconstruction_dir),
        "refined_dir": project_relative(refined_dir),
        "m0_reconstruction_vs_original": compute_pair_metrics(reference, reconstruction, lpips_model, device),
        "m1_refined_vs_original": compute_pair_metrics(reference, refined, lpips_model, device),
        "refined_vs_reconstruction": compute_pair_metrics(reconstruction, refined, None, device),
        "diffusion_time_ms_per_image": 1000.0 * sum(inference_times) / max(1, len(names)),
        "sample_grid": project_relative(sample_dir / f"{snr_name(snr)}_original_reconstruction_refined.png"),
    }


def validate_inputs(config: dict[str, Any], snrs: list[float], num_samples: int) -> dict[str, list[str]]:
    original_dir = resolve_project_path(config["inputs"]["original_dir"])
    checkpoint = resolve_project_path(config["inputs"]["checkpoint"])
    forbidden_checkpoint = resolve_project_path(config["inputs"]["forbidden_checkpoint"])
    if not original_dir.exists():
        raise FileNotFoundError(f"Original export directory not found: {original_dir}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Required best checkpoint not found: {checkpoint}")
    if checkpoint == forbidden_checkpoint:
        raise RuntimeError("Config points to forbidden latest.pt checkpoint.")

    sample_names_by_snr: dict[str, list[str]] = {}
    for snr in snrs:
        reconstruction_dir = resolve_project_path(config["inputs"]["m0_export_dir"]) / "exports" / snr_name(snr) / "reconstruction"
        if not reconstruction_dir.exists():
            raise FileNotFoundError(f"Reconstruction export directory not found: {reconstruction_dir}")
        sample_names_by_snr[snr_name(snr)] = list_sample_names(original_dir, reconstruction_dir, num_samples)
    return sample_names_by_snr


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    snrs = parse_snrs(args.snrs, config)
    num_samples = int(args.num_samples or config["evaluation"]["num_samples"])
    config["evaluation"]["num_samples"] = num_samples
    if args.batch_size is not None:
        config["diffusion"]["batch_size"] = int(args.batch_size)

    sample_names_by_snr = validate_inputs(config, snrs, num_samples)
    if args.dry_run:
        print(json.dumps({"status": "ok", "snrs": snrs, "sample_names_by_snr": sample_names_by_snr}, indent=2))
        return

    output_dir = resolve_project_path(args.output_dir or config["outputs"]["output_dir"])
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists, refusing to overwrite: {output_dir}")

    cache_root = resolve_project_path("outputs/cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))

    torch.manual_seed(int(config["seed"]))
    device = resolve_device(args.device)
    pipe = load_pipeline(config, device, allow_download=args.allow_download)
    lpips_model, lpips_error = (None, "Skipped by --skip-lpips") if args.skip_lpips else try_load_lpips(device, cache_root)

    output_dir.mkdir(parents=True)
    shutil.copy2(config_path, output_dir / "config.yaml")

    metadata = {
        "project_version": "N/A (not a project git repo)",
        "config": project_relative(config_path),
        "run_command": " ".join(sys.argv),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "dataset": config["dataset"],
        "image_size": int(config["image_size"]),
        "channel": str(config["channel"]),
        "snrs": snrs,
        "cbr": float(config["cbr"]),
        "seed": int(config["seed"]),
        "num_samples": num_samples,
        "m0_export_dir": config["inputs"]["m0_export_dir"],
        "checkpoint": config["inputs"]["checkpoint"],
        "diffusion": config["diffusion"],
        "lpips_error": lpips_error,
        "key_sources": [
            "scripts/s3_blind_diffusion_refine.py",
            "src/cadsd_jscc/metrics.py",
        ],
    }

    results = []
    for snr in snrs:
        result = evaluate_snr(
            snr=snr,
            names=sample_names_by_snr[snr_name(snr)],
            config=config,
            pipe=pipe,
            lpips_model=lpips_model,
            device=device,
            output_dir=output_dir,
        )
        results.append(result)
        print(json.dumps(result, indent=2))

    payload = {"metadata": metadata, "results": results}
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with (output_dir / "source_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(sample_names_by_snr, handle, indent=2)
    print(json.dumps({"output_dir": project_relative(output_dir), "num_results": len(results)}, indent=2))


if __name__ == "__main__":
    main()
