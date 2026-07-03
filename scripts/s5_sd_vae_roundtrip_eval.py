from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
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
    parser = argparse.ArgumentParser(
        description="Evaluate Stable Diffusion VAE encode/decode roundtrip damage on DeepJSCC M0 exports."
    )
    parser.add_argument("--config", default="configs/s5_sd_vae_roundtrip_coco256_awgn.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--snrs", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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


def load_rgb_tensor(path: Path) -> torch.Tensor:
    return TF.to_tensor(Image.open(path).convert("RGB"))


def load_rgb_pil(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def rate(flags: list[bool]) -> float:
    if not flags:
        return 0.0
    return float(sum(flags) / len(flags))


def list_sample_names(original_dir: Path, reconstruction_dir: Path, num_samples: int) -> list[str]:
    original_names = {path.name for path in original_dir.glob("*.png")}
    reconstruction_names = {path.name for path in reconstruction_dir.glob("*.png")}
    names = sorted(original_names & reconstruction_names)
    if len(names) < num_samples:
        raise RuntimeError(
            f"Need {num_samples} matched samples, found {len(names)} in {original_dir} and {reconstruction_dir}"
        )
    return names[:num_samples]


def validate_inputs(config: dict[str, Any], snrs: list[float], num_samples: int) -> dict[str, list[str]]:
    original_dir = resolve_project_path(config["inputs"]["original_dir"])
    m0_export_dir = resolve_project_path(config["inputs"]["m0_export_dir"])
    checkpoint = resolve_project_path(config["inputs"]["checkpoint"])
    forbidden_checkpoint = resolve_project_path(config["inputs"]["forbidden_checkpoint"])
    classifier_weights = resolve_project_path(config["classifier"]["weights_file"])
    required = [original_dir, m0_export_dir, checkpoint, classifier_weights]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")
    if checkpoint == forbidden_checkpoint:
        raise RuntimeError("Config points to forbidden latest.pt checkpoint.")

    names_by_snr: dict[str, list[str]] = {}
    for snr in snrs:
        reconstruction_dir = m0_export_dir / "exports" / snr_name(snr) / "reconstruction"
        if not reconstruction_dir.exists():
            raise FileNotFoundError(f"Reconstruction export directory not found: {reconstruction_dir}")
        names_by_snr[snr_name(snr)] = list_sample_names(original_dir, reconstruction_dir, num_samples)
    return names_by_snr


def get_project_version() -> str:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:  # noqa: BLE001 - absence of a project git repo is recorded as metadata.
        return "N/A (local directory is not yet a git repo)"
    return value


def check_classifier_cache(config: dict[str, Any], allow_download: bool) -> None:
    weights_file = resolve_project_path(config["classifier"]["weights_file"])
    if weights_file.is_file() and weights_file.stat().st_size > 10 * 1024 * 1024:
        return
    if allow_download:
        resolve_project_path(config["classifier"]["cache_dir"]).mkdir(parents=True, exist_ok=True)
        return
    raise RuntimeError(
        f"Classifier weights not found under project cache: {weights_file}. "
        "Download weights with cleared proxy variables first, or rerun with --allow-download."
    )


def load_classifier(config: dict[str, Any], device: torch.device, allow_download: bool):
    check_classifier_cache(config, allow_download=allow_download)
    cls_cfg = config["classifier"]
    cache_dir = resolve_project_path(cls_cfg["cache_dir"])
    os.environ.setdefault("TORCH_HOME", str(cache_dir))

    import torchvision.models as models

    model_name = str(cls_cfg["model_name"]).lower()
    weights_name = str(cls_cfg["weights"])
    if model_name != "alexnet":
        raise ValueError(f"Unsupported classifier model: {model_name}")
    weights = getattr(models.AlexNet_Weights, weights_name)
    model = models.alexnet(weights=weights).to(device)
    model.eval()
    return model, weights.transforms(), list(weights.meta["categories"])


def load_vae(config: dict[str, Any], device: torch.device, allow_download: bool):
    from diffusers import AutoencoderKL

    vae_cfg = config["vae"]
    cache_dir = resolve_project_path(vae_cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))

    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    vae = AutoencoderKL.from_pretrained(
        str(vae_cfg["model_id"]),
        subfolder=str(vae_cfg.get("subfolder", "vae")),
        cache_dir=str(cache_dir),
        torch_dtype=torch_dtype,
        local_files_only=bool(vae_cfg.get("local_files_only", True)) and not allow_download,
    )
    vae = vae.to(device)
    vae.eval()
    if hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    return vae


def try_load_lpips(device: torch.device, cache_root: Path):
    try:
        os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))
        import lpips

        model = lpips.LPIPS(net="alex", verbose=False).to(device)
        model.eval()
        return model, None
    except Exception as exc:  # noqa: BLE001 - optional metric should not abort the experiment.
        return None, f"{type(exc).__name__}: {exc}"


def vae_scaling_factor(vae, config: dict[str, Any]) -> float:
    value = getattr(vae.config, "scaling_factor", None)
    if value is None:
        value = config["vae"].get("scaling_factor_fallback", 0.18215)
    return float(value)


@torch.no_grad()
def vae_roundtrip_batch(
    vae,
    images: torch.Tensor,
    scaling_factor: float,
    deterministic: bool,
    generator: torch.Generator | None,
    device: torch.device,
) -> torch.Tensor:
    dtype = next(vae.parameters()).dtype
    model_input = (images.to(device=device, dtype=dtype) * 2.0) - 1.0
    posterior = vae.encode(model_input).latent_dist
    if deterministic:
        latents = posterior.mode()
    else:
        latents = posterior.sample(generator=generator)
    latents = latents * scaling_factor
    decoded = vae.decode(latents / scaling_factor).sample
    return ((decoded.float() + 1.0) / 2.0).clamp(0.0, 1.0)


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


def classify_tensors(
    model: torch.nn.Module,
    preprocess,
    paths: list[Path],
    batch_size: int,
    topk: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], float]:
    outputs: list[dict[str, Any]] = []
    elapsed = 0.0
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = torch.stack([preprocess(load_rgb_pil(path)) for path in batch_paths]).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            begin = time.perf_counter()
            logits = model(images)
            probabilities = torch.softmax(logits.float(), dim=-1)
            values, indices = torch.topk(probabilities, k=topk, dim=-1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - begin
            for row_values, row_indices in zip(values.cpu(), indices.cpu()):
                outputs.append(
                    {
                        "top_indices": [int(item) for item in row_indices.tolist()],
                        "top_probs": [float(item) for item in row_values.tolist()],
                    }
                )
    return outputs, elapsed


def label_for(categories: list[str], index: int) -> str:
    if 0 <= index < len(categories):
        return categories[index]
    return f"class_{index}"


def semantic_summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        return {
            "num_images": 0,
            "m0_final_failure": 0.0,
            "m0_vae_final_failure": 0.0,
            "m0_vae_prediction_consistency": 0.0,
            "m0_vae_refinement_drift": 0.0,
            "original_vae_failure": 0.0,
            "original_vae_prediction_consistency": 0.0,
        }
    m0_match_origin = [bool(row["m0_matches_original_top1"]) for row in rows]
    m0_vae_match_origin = [bool(row["m0_vae_matches_original_top1"]) for row in rows]
    m0_vae_match_m0 = [bool(row["m0_vae_matches_m0_top1"]) for row in rows]
    original_vae_match_origin = [bool(row["original_vae_matches_original_top1"]) for row in rows]
    m0_failure = 1.0 - rate(m0_match_origin)
    m0_vae_failure = 1.0 - rate(m0_vae_match_origin)
    original_vae_failure = 1.0 - rate(original_vae_match_origin)
    return {
        "num_images": len(rows),
        "m0_final_failure": m0_failure,
        "m0_vae_final_failure": m0_vae_failure,
        "m0_vae_prediction_consistency": 1.0 - m0_vae_failure,
        "m0_vae_minus_m0_final_failure": m0_vae_failure - m0_failure,
        "m0_vae_drift_origin": m0_vae_failure,
        "m0_vae_refinement_drift": 1.0 - rate(m0_vae_match_m0),
        "original_vae_failure": original_vae_failure,
        "original_vae_prediction_consistency": 1.0 - original_vae_failure,
    }


def subset_summaries(rows: list[dict[str, Any]], thresholds: list[float]) -> dict[str, dict[str, float | int]]:
    summaries = {"all": semantic_summary(rows)}
    for threshold in thresholds:
        key = f"original_conf_ge_{str(threshold).replace('.', 'p')}"
        subset = [row for row in rows if float(row["original_top1_prob"]) >= threshold]
        summaries[key] = semantic_summary(subset)
    return summaries


def roundtrip_and_save(
    vae,
    input_paths: list[Path],
    output_dir: Path,
    batch_size: int,
    scaling_factor: float,
    deterministic: bool,
    seed: int,
    device: torch.device,
) -> float:
    output_dir.mkdir(parents=True, exist_ok=False)
    elapsed = 0.0
    for start in range(0, len(input_paths), batch_size):
        batch_paths = input_paths[start : start + batch_size]
        batch = torch.stack([load_rgb_tensor(path) for path in batch_paths])
        generator = None
        if not deterministic:
            generator = torch.Generator(device=device).manual_seed(seed + start)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        begin = time.perf_counter()
        output = vae_roundtrip_batch(
            vae=vae,
            images=batch,
            scaling_factor=scaling_factor,
            deterministic=deterministic,
            generator=generator,
            device=device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed += time.perf_counter() - begin
        for path, image in zip(batch_paths, output.detach().cpu()):
            save_image(image, output_dir / path.name)
    return elapsed


def evaluate_snr(
    snr: float,
    names: list[str],
    config: dict[str, Any],
    vae,
    classifier_model: torch.nn.Module,
    classifier_preprocess,
    categories: list[str],
    lpips_model,
    scaling_factor: float,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    original_dir = resolve_project_path(config["inputs"]["original_dir"])
    m0_dir = resolve_project_path(config["inputs"]["m0_export_dir"]) / "exports" / snr_name(snr) / "reconstruction"
    snr_output_dir = output_dir / "exports" / snr_name(snr)
    m0_vae_dir = snr_output_dir / "m0_vae_roundtrip"
    original_vae_dir = snr_output_dir / "original_vae_roundtrip"

    original_paths = [original_dir / name for name in names]
    m0_paths = [m0_dir / name for name in names]
    batch_size = int(config["vae"]["batch_size"])
    deterministic = bool(config["vae"].get("deterministic", True))
    seed = int(config["seed"]) + int(round(snr * 1000))

    m0_vae_elapsed = roundtrip_and_save(
        vae=vae,
        input_paths=m0_paths,
        output_dir=m0_vae_dir,
        batch_size=batch_size,
        scaling_factor=scaling_factor,
        deterministic=deterministic,
        seed=seed,
        device=device,
    )
    original_vae_elapsed = 0.0
    if bool(config["vae"].get("compute_original_roundtrip", True)):
        original_vae_elapsed = roundtrip_and_save(
            vae=vae,
            input_paths=original_paths,
            output_dir=original_vae_dir,
            batch_size=batch_size,
            scaling_factor=scaling_factor,
            deterministic=deterministic,
            seed=seed + 100000,
            device=device,
        )

    m0_vae_paths = [m0_vae_dir / name for name in names]
    original_vae_paths = [original_vae_dir / name for name in names]
    cls_batch = int(config["classifier"]["batch_size"])
    topk = int(config["classifier"]["topk"])
    original_preds, t_original = classify_tensors(
        classifier_model, classifier_preprocess, original_paths, cls_batch, topk, device
    )
    m0_preds, t_m0 = classify_tensors(classifier_model, classifier_preprocess, m0_paths, cls_batch, topk, device)
    m0_vae_preds, t_m0_vae = classify_tensors(
        classifier_model, classifier_preprocess, m0_vae_paths, cls_batch, topk, device
    )
    original_vae_preds, t_original_vae = classify_tensors(
        classifier_model, classifier_preprocess, original_vae_paths, cls_batch, topk, device
    )

    references: list[torch.Tensor] = []
    m0_tensors: list[torch.Tensor] = []
    m0_vae_tensors: list[torch.Tensor] = []
    original_vae_tensors: list[torch.Tensor] = []
    per_sample: list[dict[str, Any]] = []
    for idx, name in enumerate(names):
        original_top1 = original_preds[idx]["top_indices"][0]
        m0_top1 = m0_preds[idx]["top_indices"][0]
        m0_vae_top1 = m0_vae_preds[idx]["top_indices"][0]
        original_vae_top1 = original_vae_preds[idx]["top_indices"][0]
        row = {
            "snr_db": float(snr),
            "sample": name,
            "original": project_relative(original_paths[idx]),
            "m0_reconstruction": project_relative(m0_paths[idx]),
            "m0_vae_roundtrip": project_relative(m0_vae_paths[idx]),
            "original_vae_roundtrip": project_relative(original_vae_paths[idx]),
            "original_top1_index": original_top1,
            "original_top1_label": label_for(categories, original_top1),
            "original_top1_prob": original_preds[idx]["top_probs"][0],
            "m0_top1_index": m0_top1,
            "m0_top1_label": label_for(categories, m0_top1),
            "m0_top1_prob": m0_preds[idx]["top_probs"][0],
            "m0_vae_top1_index": m0_vae_top1,
            "m0_vae_top1_label": label_for(categories, m0_vae_top1),
            "m0_vae_top1_prob": m0_vae_preds[idx]["top_probs"][0],
            "original_vae_top1_index": original_vae_top1,
            "original_vae_top1_label": label_for(categories, original_vae_top1),
            "original_vae_top1_prob": original_vae_preds[idx]["top_probs"][0],
            "m0_matches_original_top1": m0_top1 == original_top1,
            "m0_vae_matches_original_top1": m0_vae_top1 == original_top1,
            "m0_vae_matches_m0_top1": m0_vae_top1 == m0_top1,
            "original_vae_matches_original_top1": original_vae_top1 == original_top1,
            "original_vae_matches_m0_top1": original_vae_top1 == m0_top1,
        }
        per_sample.append(row)
        references.append(load_rgb_tensor(original_paths[idx]))
        m0_tensors.append(load_rgb_tensor(m0_paths[idx]))
        m0_vae_tensors.append(load_rgb_tensor(m0_vae_paths[idx]))
        original_vae_tensors.append(load_rgb_tensor(original_vae_paths[idx]))

    reference = torch.stack(references)
    m0 = torch.stack(m0_tensors)
    m0_vae = torch.stack(m0_vae_tensors)
    original_vae = torch.stack(original_vae_tensors)

    sample_count = min(int(config["evaluation"]["sample_grid_count"]), len(names))
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_grid = sample_dir / f"{snr_name(snr)}_original_m0_m0vae_originalvae.png"
    save_image(
        torch.cat(
            [
                reference[:sample_count],
                m0[:sample_count],
                m0_vae[:sample_count],
                original_vae[:sample_count],
            ],
            dim=0,
        ),
        sample_grid,
        nrow=sample_count,
    )

    thresholds = [float(item) for item in config["evaluation"].get("pseudo_clean_conf_thresholds", [])]
    return {
        "snr_db": float(snr),
        "num_images": len(names),
        "sample_names": names,
        "m0_vae_roundtrip_dir": project_relative(m0_vae_dir),
        "original_vae_roundtrip_dir": project_relative(original_vae_dir),
        "sample_grid": project_relative(sample_grid),
        "image_quality": {
            "m0_reconstruction_vs_original": compute_pair_metrics(reference, m0, lpips_model, device),
            "m0_vae_roundtrip_vs_original": compute_pair_metrics(reference, m0_vae, lpips_model, device),
            "m0_vae_roundtrip_vs_m0_reconstruction": compute_pair_metrics(m0, m0_vae, lpips_model, device),
            "original_vae_roundtrip_vs_original": compute_pair_metrics(reference, original_vae, lpips_model, device),
        },
        "semantic_reliability": subset_summaries(per_sample, thresholds),
        "m0_vae_roundtrip_time_ms_per_image": 1000.0 * m0_vae_elapsed / max(1, len(names)),
        "original_vae_roundtrip_time_ms_per_image": 1000.0 * original_vae_elapsed / max(1, len(names)),
        "classification_time_ms_per_image": 1000.0
        * (t_original + t_m0 + t_m0_vae + t_original_vae)
        / max(1, 4 * len(names)),
        "per_sample": per_sample,
    }


def serialize_csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_csv_value(value) for key, value in row.items()})


def fmt(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return float(value - baseline)


def aggregate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        quality = result["image_quality"]
        semantic = result["semantic_reliability"]["all"]
        m0_quality = quality["m0_reconstruction_vs_original"]
        m0_vae_quality = quality["m0_vae_roundtrip_vs_original"]
        m0_vae_vs_m0_quality = quality["m0_vae_roundtrip_vs_m0_reconstruction"]
        original_vae_quality = quality["original_vae_roundtrip_vs_original"]
        m0_psnr = safe_float(m0_quality["psnr_db"])
        m0_vae_psnr = safe_float(m0_vae_quality["psnr_db"])
        m0_lpips = safe_float(m0_quality["lpips"])
        m0_vae_lpips = safe_float(m0_vae_quality["lpips"])
        rows.append(
            {
                "snr_db": float(result["snr_db"]),
                "num_images": int(result["num_images"]),
                "m0_psnr_db": m0_psnr,
                "m0_vae_psnr_db": m0_vae_psnr,
                "m0_vae_delta_psnr_vs_m0_db": delta(m0_vae_psnr, m0_psnr),
                "m0_lpips": m0_lpips,
                "m0_vae_lpips": m0_vae_lpips,
                "m0_vae_delta_lpips_vs_m0": delta(m0_vae_lpips, m0_lpips),
                "m0_vae_vs_m0_psnr_db": safe_float(m0_vae_vs_m0_quality["psnr_db"]),
                "m0_vae_vs_m0_lpips": safe_float(m0_vae_vs_m0_quality["lpips"]),
                "original_vae_psnr_db": safe_float(original_vae_quality["psnr_db"]),
                "original_vae_lpips": safe_float(original_vae_quality["lpips"]),
                "m0_final_failure": float(semantic["m0_final_failure"]),
                "m0_vae_final_failure": float(semantic["m0_vae_final_failure"]),
                "m0_vae_refinement_drift": float(semantic["m0_vae_refinement_drift"]),
                "original_vae_failure": float(semantic["original_vae_failure"]),
                "m0_vae_roundtrip_time_ms_per_image": float(result["m0_vae_roundtrip_time_ms_per_image"]),
            }
        )
    return rows


def make_report(results: list[dict[str, Any]], config: dict[str, Any], scaling_factor: float) -> str:
    rows = aggregate_results(results)
    lines = [
        "# SD VAE Roundtrip Diagnostic",
        "",
        "This S5 diagnostic isolates Stable Diffusion VAE encode/decode damage. It does not run UNet denoising and does not use text prompts.",
        "",
        "## Sources",
        "",
        f"- M0 export: `{config['inputs']['m0_export_dir']}`",
        f"- DeepJSCC checkpoint: `{config['inputs']['checkpoint']}`",
        f"- SD VAE: `{config['vae']['model_id']}` subfolder `{config['vae'].get('subfolder', 'vae')}`",
        f"- VAE scaling factor: `{scaling_factor}`",
        f"- Repository URL provided by user: `{config.get('repository_url', 'N/A')}`",
        "",
        "## Main Table",
        "",
        "| SNR(dB) | M0 PSNR | M0-VAE PSNR | Delta | M0 LPIPS | M0-VAE LPIPS | Delta | M0-VAE vs M0 PSNR | M0 failure | M0-VAE failure | Refinement drift |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {snr:g} | {m0_psnr} | {vae_psnr} | {psnr_delta} | {m0_lpips} | {vae_lpips} | {lpips_delta} | {vae_m0_psnr} | {m0_fail} | {vae_fail} | {ref_drift} |".format(
                snr=float(row["snr_db"]),
                m0_psnr=fmt(row["m0_psnr_db"]),
                vae_psnr=fmt(row["m0_vae_psnr_db"]),
                psnr_delta=fmt(row["m0_vae_delta_psnr_vs_m0_db"]),
                m0_lpips=fmt(row["m0_lpips"]),
                vae_lpips=fmt(row["m0_vae_lpips"]),
                lpips_delta=fmt(row["m0_vae_delta_lpips_vs_m0"]),
                vae_m0_psnr=fmt(row["m0_vae_vs_m0_psnr_db"]),
                m0_fail=fmt(row["m0_final_failure"]),
                vae_fail=fmt(row["m0_vae_final_failure"]),
                ref_drift=fmt(row["m0_vae_refinement_drift"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "- This is a diagnostic, not an M2 or M3 result.",
            "- A large M0-VAE quality drop indicates that generic SD img2img can be bottlenecked before denoising starts.",
            "- COCO ImageNet pseudo-label consistency remains an auxiliary semantic diagnostic, not clean-correct GT classification.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    snrs = parse_snrs(args.snrs, config)
    num_samples = int(args.num_samples or config["evaluation"]["num_samples"])
    config["evaluation"]["num_samples"] = num_samples
    if args.batch_size is not None:
        config["vae"]["batch_size"] = int(args.batch_size)
    names_by_snr = validate_inputs(config, snrs, num_samples)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "snrs": snrs,
                    "sample_names_by_snr": names_by_snr,
                    "vae": config["vae"],
                },
                indent=2,
            )
        )
        return

    output_dir = resolve_project_path(args.output_dir or config["outputs"]["output_dir"])
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists, refusing to overwrite: {output_dir}")

    torch.manual_seed(int(config["seed"]))
    cache_root = resolve_project_path("outputs/cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))
    device = resolve_device(args.device)

    vae = load_vae(config, device, allow_download=args.allow_download)
    scaling_factor = vae_scaling_factor(vae, config)
    classifier_model, classifier_preprocess, categories = load_classifier(
        config, device, allow_download=args.allow_download
    )
    lpips_model, lpips_error = (None, "Skipped by --skip-lpips") if args.skip_lpips else try_load_lpips(device, cache_root)

    output_dir.mkdir(parents=True)
    save_json(output_dir / "source_manifest.json", names_by_snr)
    with config_path.open("r", encoding="utf-8") as handle:
        (output_dir / "config.yaml").write_text(handle.read(), encoding="utf-8")

    results = []
    csv_rows: list[dict[str, Any]] = []
    for snr in snrs:
        result = evaluate_snr(
            snr=snr,
            names=names_by_snr[snr_name(snr)],
            config=config,
            vae=vae,
            classifier_model=classifier_model,
            classifier_preprocess=classifier_preprocess,
            categories=categories,
            lpips_model=lpips_model,
            scaling_factor=scaling_factor,
            device=device,
            output_dir=output_dir,
        )
        results.append(result)
        csv_rows.extend(result["per_sample"])
        printable = {key: value for key, value in result.items() if key != "per_sample"}
        print(json.dumps(printable, indent=2))

    import importlib.metadata as md
    import torchvision

    metadata = {
        "project_version": get_project_version(),
        "repository_url": config.get("repository_url"),
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
        "inputs": config["inputs"],
        "vae": config["vae"],
        "vae_scaling_factor": scaling_factor,
        "classifier": config["classifier"],
        "lpips_error": lpips_error,
        "python_version": platform.python_version(),
        "package_versions": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "diffusers": md.version("diffusers"),
            "transformers": md.version("transformers"),
            "pillow": md.version("pillow"),
            "pytorch-msssim": md.version("pytorch-msssim"),
        },
        "proxy_environment_present": sorted(key for key in os.environ if "proxy" in key.lower()),
        "download_note": "No model download is required when local_files_only is true and caches are present.",
        "key_sources": [
            "scripts/s5_sd_vae_roundtrip_eval.py",
            "src/cadsd_jscc/metrics.py",
        ],
    }
    payload = {
        "metadata": metadata,
        "results": results,
        "summary_rows": aggregate_results(results),
    }
    save_json(output_dir / "metrics.json", payload)
    write_csv(output_dir / "per_sample.csv", csv_rows)
    write_csv(output_dir / "summary.csv", aggregate_results(results))
    (output_dir / "REPORT.md").write_text(make_report(results, config, scaling_factor), encoding="utf-8")
    print(json.dumps({"output_dir": project_relative(output_dir), "num_results": len(results)}, indent=2))


if __name__ == "__main__":
    main()
