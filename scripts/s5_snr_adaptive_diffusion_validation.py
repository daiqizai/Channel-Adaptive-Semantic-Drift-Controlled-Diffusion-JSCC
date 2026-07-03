from __future__ import annotations

import argparse
import csv
import json
import os
import platform
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
    parser = argparse.ArgumentParser(
        description="Run conservative fixed/SNR-adaptive diffusion-strength validation with pseudo semantic metrics."
    )
    parser.add_argument("--config", default="configs/s5_snr_adaptive_diffusion_strength_validation.yaml")
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


def strength_key(snr: float) -> str:
    if float(snr).is_integer():
        return str(int(snr))
    return str(snr)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def load_rgb_tensor(path: Path) -> torch.Tensor:
    return TF.to_tensor(Image.open(path).convert("RGB"))


def load_rgb_pil(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def list_sample_names(original_dir: Path, reconstruction_dir: Path, num_samples: int) -> list[str]:
    original_names = {path.name for path in original_dir.glob("*.png")}
    reconstruction_names = {path.name for path in reconstruction_dir.glob("*.png")}
    names = sorted(original_names & reconstruction_names)
    if len(names) < num_samples:
        raise RuntimeError(
            f"Need {num_samples} matched samples, found {len(names)} in {original_dir} and {reconstruction_dir}"
        )
    return names[:num_samples]


def check_strength_schedule(candidate: dict[str, Any], snrs: list[float]) -> dict[float, float]:
    raw = candidate.get("strengths", {})
    strengths: dict[float, float] = {}
    for snr in snrs:
        key = strength_key(snr)
        if key not in raw:
            raise KeyError(f"Candidate {candidate['name']} is missing strength for SNR {key}")
        value = float(raw[key])
        if not (0.0 < value <= 1.0):
            raise ValueError(f"Invalid strength {value} for candidate {candidate['name']} at SNR {key}")
        strengths[float(snr)] = value
    ordered = sorted(strengths)
    for left, right in zip(ordered, ordered[1:]):
        if strengths[left] + 1e-12 < strengths[right]:
            raise ValueError(
                f"Candidate {candidate['name']} violates non-increasing strength: "
                f"{left} dB has {strengths[left]}, {right} dB has {strengths[right]}"
            )
    return strengths


def validate_inputs(
    config: dict[str, Any],
    snrs: list[float],
    num_samples: int,
) -> tuple[dict[str, list[str]], dict[str, dict[float, float]]]:
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

    strengths_by_candidate = {
        str(candidate["name"]): check_strength_schedule(candidate, snrs) for candidate in config["candidates"]
    }
    return names_by_snr, strengths_by_candidate


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
    strength: float,
    generator: torch.Generator,
) -> list[Image.Image]:
    diffusion = config["diffusion"]
    result = pipe(
        prompt=[str(diffusion.get("prompt", ""))] * len(images),
        negative_prompt=[str(diffusion.get("negative_prompt", ""))] * len(images),
        image=images,
        strength=float(strength),
        num_inference_steps=int(diffusion["num_inference_steps"]),
        guidance_scale=float(diffusion["guidance_scale"]),
        generator=generator,
    )
    return list(result.images)


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
            "accept_rate": 0.0,
            "reject_rate": 0.0,
            "m0_final_failure": 0.0,
            "refined_drift_origin": 0.0,
            "refined_refinement_drift": 0.0,
            "m3_final_failure": 0.0,
            "m3_prediction_consistency": 0.0,
            "m3_refinement_drift": 0.0,
            "false_accept_rate": 0.0,
            "false_reject_rate": 0.0,
        }
    accepted = [bool(row["detector_accept_refined"]) for row in rows]
    m0_match_origin = [bool(row["m0_matches_original_top1"]) for row in rows]
    refined_match_origin = [bool(row["refined_matches_original_top1"]) for row in rows]
    refined_match_m0 = [bool(row["refined_matches_m0_top1"]) for row in rows]
    m3_match_origin = [bool(row["m3_matches_original_top1"]) for row in rows]
    m3_match_m0 = [bool(row["m3_matches_m0_top1"]) for row in rows]
    false_accept = [
        bool(row["detector_accept_refined"]) and not bool(row["refined_matches_original_top1"]) for row in rows
    ]
    false_reject = [
        not bool(row["detector_accept_refined"]) and bool(row["refined_matches_original_top1"]) for row in rows
    ]
    m0_failure = 1.0 - rate(m0_match_origin)
    refined_failure = 1.0 - rate(refined_match_origin)
    m3_failure = 1.0 - rate(m3_match_origin)
    return {
        "num_images": len(rows),
        "accept_rate": rate(accepted),
        "reject_rate": 1.0 - rate(accepted),
        "m0_final_failure": m0_failure,
        "refined_drift_origin": refined_failure,
        "refined_refinement_drift": 1.0 - rate(refined_match_m0),
        "m3_final_failure": m3_failure,
        "m3_prediction_consistency": 1.0 - m3_failure,
        "m3_refinement_drift": 1.0 - rate(m3_match_m0),
        "m3_minus_refined_final_failure": m3_failure - refined_failure,
        "m3_minus_m0_final_failure": m3_failure - m0_failure,
        "false_accept_rate": rate(false_accept),
        "false_reject_rate": rate(false_reject),
        "false_accept_count": int(sum(false_accept)),
        "false_reject_count": int(sum(false_reject)),
    }


def subset_summaries(rows: list[dict[str, Any]], thresholds: list[float]) -> dict[str, dict[str, float | int]]:
    summaries = {"all": semantic_summary(rows)}
    for threshold in thresholds:
        key = f"original_conf_ge_{str(threshold).replace('.', 'p')}"
        subset = [row for row in rows if float(row["original_top1_prob"]) >= threshold]
        summaries[key] = semantic_summary(subset)
    return summaries


def evaluate_candidate_snr(
    candidate_index: int,
    candidate: dict[str, Any],
    snr: float,
    names: list[str],
    strength: float,
    config: dict[str, Any],
    pipe,
    classifier_model: torch.nn.Module,
    classifier_preprocess,
    categories: list[str],
    lpips_model,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    diffusion = config["diffusion"]
    original_dir = resolve_project_path(config["inputs"]["original_dir"])
    m0_dir = resolve_project_path(config["inputs"]["m0_export_dir"]) / "exports" / snr_name(snr) / "reconstruction"
    candidate_dir = output_dir / "candidates" / str(candidate["name"])
    refined_dir = candidate_dir / "exports" / snr_name(snr) / "refined"
    final_dir = candidate_dir / "exports" / snr_name(snr) / "final"
    refined_dir.mkdir(parents=True, exist_ok=False)
    final_dir.mkdir(parents=True, exist_ok=False)

    batch_size = int(diffusion["batch_size"])
    inference_times: list[float] = []
    for start in range(0, len(names), batch_size):
        batch_names = names[start : start + batch_size]
        input_images = [load_rgb_pil(m0_dir / name) for name in batch_names]
        generator = torch.Generator(device=device).manual_seed(
            int(config["seed"]) + candidate_index * 100000 + int(round(snr * 1000)) + start
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        begin = time.perf_counter()
        refined_images = refine_batch(pipe, input_images, config, strength=strength, generator=generator)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_times.append(time.perf_counter() - begin)
        for name, refined_image in zip(batch_names, refined_images):
            refined_image.save(refined_dir / name)

    original_paths = [original_dir / name for name in names]
    m0_paths = [m0_dir / name for name in names]
    refined_paths = [refined_dir / name for name in names]
    cls_batch = int(config["classifier"]["batch_size"])
    topk = int(config["classifier"]["topk"])
    original_preds, t_original = classify_tensors(
        classifier_model, classifier_preprocess, original_paths, cls_batch, topk, device
    )
    m0_preds, t_m0 = classify_tensors(classifier_model, classifier_preprocess, m0_paths, cls_batch, topk, device)
    refined_preds, t_refined = classify_tensors(
        classifier_model, classifier_preprocess, refined_paths, cls_batch, topk, device
    )

    references: list[torch.Tensor] = []
    m0_tensors: list[torch.Tensor] = []
    refined_tensors: list[torch.Tensor] = []
    final_tensors: list[torch.Tensor] = []
    per_sample: list[dict[str, Any]] = []
    for idx, name in enumerate(names):
        original_top1 = original_preds[idx]["top_indices"][0]
        m0_top1 = m0_preds[idx]["top_indices"][0]
        refined_top1 = refined_preds[idx]["top_indices"][0]
        accept = refined_top1 == m0_top1
        final_source = refined_dir / name if accept else m0_dir / name
        final_path = final_dir / name
        shutil.copy2(final_source, final_path)
        m3_prefix = "refined" if accept else "m0"
        m0_matches_origin = m0_top1 == original_top1
        refined_matches_origin = refined_top1 == original_top1
        m3_matches_origin = refined_matches_origin if accept else m0_matches_origin
        m3_matches_m0 = refined_top1 == m0_top1 if accept else True
        row = {
            "candidate": str(candidate["name"]),
            "candidate_method": str(candidate.get("method", "")),
            "snr_db": float(snr),
            "strength": float(strength),
            "sample": name,
            "original": project_relative(original_paths[idx]),
            "m0_reconstruction": project_relative(m0_paths[idx]),
            "refined": project_relative(refined_paths[idx]),
            "m3_final": project_relative(final_path),
            "detector": str(config["failure_handling"]["detector"]),
            "detector_accept_refined": accept,
            "m3_output_kind": "accepted_refined" if accept else "fallback_m0",
            "original_top1_index": original_top1,
            "original_top1_label": label_for(categories, original_top1),
            "original_top1_prob": original_preds[idx]["top_probs"][0],
            "m0_top1_index": m0_top1,
            "m0_top1_label": label_for(categories, m0_top1),
            "m0_top1_prob": m0_preds[idx]["top_probs"][0],
            "refined_top1_index": refined_top1,
            "refined_top1_label": label_for(categories, refined_top1),
            "refined_top1_prob": refined_preds[idx]["top_probs"][0],
            "m3_top1_index": refined_top1 if accept else m0_top1,
            "m3_top1_label": label_for(categories, refined_top1 if accept else m0_top1),
            "m3_top1_prob": refined_preds[idx]["top_probs"][0] if m3_prefix == "refined" else m0_preds[idx]["top_probs"][0],
            "m0_matches_original_top1": m0_matches_origin,
            "refined_matches_original_top1": refined_matches_origin,
            "refined_matches_m0_top1": refined_top1 == m0_top1,
            "m3_matches_original_top1": m3_matches_origin,
            "m3_matches_m0_top1": m3_matches_m0,
            "false_accept": accept and not refined_matches_origin,
            "false_reject": (not accept) and refined_matches_origin,
        }
        per_sample.append(row)
        references.append(load_rgb_tensor(original_paths[idx]))
        m0_tensors.append(load_rgb_tensor(m0_paths[idx]))
        refined_tensors.append(load_rgb_tensor(refined_paths[idx]))
        final_tensors.append(load_rgb_tensor(final_path))

    reference = torch.stack(references)
    m0 = torch.stack(m0_tensors)
    refined = torch.stack(refined_tensors)
    final = torch.stack(final_tensors)
    sample_count = min(int(config["evaluation"]["sample_grid_count"]), len(names))
    sample_dir = candidate_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_grid = sample_dir / f"{snr_name(snr)}_original_m0_refined_m3final.png"
    save_image(
        torch.cat([reference[:sample_count], m0[:sample_count], refined[:sample_count], final[:sample_count]], dim=0),
        sample_grid,
        nrow=sample_count,
    )

    thresholds = [float(item) for item in config["evaluation"].get("pseudo_clean_conf_thresholds", [])]
    return {
        "candidate": str(candidate["name"]),
        "candidate_method": str(candidate.get("method", "")),
        "snr_db": float(snr),
        "strength": float(strength),
        "num_images": len(names),
        "sample_names": names,
        "refined_dir": project_relative(refined_dir),
        "final_dir": project_relative(final_dir),
        "sample_grid": project_relative(sample_grid),
        "image_quality": {
            "m0_reconstruction_vs_original": compute_pair_metrics(reference, m0, lpips_model, device),
            "refined_vs_original": compute_pair_metrics(reference, refined, lpips_model, device),
            "m3_final_vs_original": compute_pair_metrics(reference, final, lpips_model, device),
            "refined_vs_m0_reconstruction": compute_pair_metrics(m0, refined, None, device),
        },
        "semantic_reliability": subset_summaries(per_sample, thresholds),
        "diffusion_time_ms_per_image": 1000.0 * sum(inference_times) / max(1, len(names)),
        "classification_time_ms_per_image": 1000.0 * (t_original + t_m0 + t_refined) / max(1, 3 * len(names)),
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


def aggregate_candidate(results: list[dict[str, Any]]) -> list[dict[str, float | str]]:
    rows = []
    for result in results:
        quality = result["image_quality"]
        semantic = result["semantic_reliability"]["all"]
        rows.append(
            {
                "candidate": result["candidate"],
                "snr_db": float(result["snr_db"]),
                "strength": float(result["strength"]),
                "m0_psnr_db": float(quality["m0_reconstruction_vs_original"]["psnr_db"]),
                "refined_psnr_db": float(quality["refined_vs_original"]["psnr_db"]),
                "m3_psnr_db": float(quality["m3_final_vs_original"]["psnr_db"]),
                "m0_lpips": float(quality["m0_reconstruction_vs_original"]["lpips"])
                if quality["m0_reconstruction_vs_original"]["lpips"] is not None
                else None,
                "refined_lpips": float(quality["refined_vs_original"]["lpips"])
                if quality["refined_vs_original"]["lpips"] is not None
                else None,
                "m3_lpips": float(quality["m3_final_vs_original"]["lpips"])
                if quality["m3_final_vs_original"]["lpips"] is not None
                else None,
                "accept_rate": float(semantic["accept_rate"]),
                "refined_failure": float(semantic["refined_drift_origin"]),
                "m3_final_failure": float(semantic["m3_final_failure"]),
                "m0_final_failure": float(semantic["m0_final_failure"]),
                "false_accept_rate": float(semantic["false_accept_rate"]),
                "false_reject_rate": float(semantic["false_reject_rate"]),
            }
        )
    return rows


def make_report(results: list[dict[str, Any]], config: dict[str, Any], sources: dict[str, str]) -> str:
    rows = aggregate_candidate(results)
    lines = [
        "# SNR-Adaptive Diffusion Strength Validation",
        "",
        "This is a small S5 validation run over existing COCO-256 M0 exports. It uses local Stable Diffusion cache and does not download models.",
        "",
        "## Sources",
        "",
        f"- M0 export: `{sources['m0_export_dir']}`",
        f"- DeepJSCC checkpoint: `{sources['checkpoint']}`",
        f"- Repository URL provided by user: `{config.get('repository_url', 'N/A')}`",
        "",
        "## Candidates",
        "",
    ]
    for candidate in config["candidates"]:
        strengths = ", ".join(f"{key}dB:{value}" for key, value in candidate["strengths"].items())
        lines.append(f"- `{candidate['name']}`: {candidate.get('note', '')} Strengths: {strengths}")
    lines.extend(
        [
            "",
            "## Main Table",
            "",
            "| Candidate | SNR(dB) | Strength | M0 PSNR | Refined PSNR | M3 PSNR | Refined failure | M3 failure | Accept | False accept |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {candidate} | {snr:g} | {strength} | {m0_psnr} | {ref_psnr} | {m3_psnr} | {ref_fail} | {m3_fail} | {accept} | {fa} |".format(
                candidate=row["candidate"],
                snr=float(row["snr_db"]),
                strength=fmt(float(row["strength"])),
                m0_psnr=fmt(float(row["m0_psnr_db"])),
                ref_psnr=fmt(float(row["refined_psnr_db"])),
                m3_psnr=fmt(float(row["m3_psnr_db"])),
                ref_fail=fmt(float(row["refined_failure"])),
                m3_fail=fmt(float(row["m3_final_failure"])),
                accept=fmt(float(row["accept_rate"])),
                fa=fmt(float(row["false_accept_rate"])),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "- This is validation, not a final M3/Ours result.",
            "- COCO pseudo-label consistency is still an auxiliary classifier diagnostic, not clean-correct GT classification.",
            "- A candidate is useful only if it improves the visual/perceptual tradeoff without increasing semantic failure.",
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
        config["diffusion"]["batch_size"] = int(args.batch_size)
    names_by_snr, strengths_by_candidate = validate_inputs(config, snrs, num_samples)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "snrs": snrs,
                    "sample_names_by_snr": names_by_snr,
                    "strengths_by_candidate": {
                        key: {str(snr): value for snr, value in schedule.items()}
                        for key, schedule in strengths_by_candidate.items()
                    },
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

    pipe = load_pipeline(config, device, allow_download=args.allow_download)
    classifier_model, classifier_preprocess, categories = load_classifier(config, device, allow_download=args.allow_download)
    lpips_model, lpips_error = (None, "Skipped by --skip-lpips") if args.skip_lpips else try_load_lpips(device, cache_root)

    output_dir.mkdir(parents=True)
    shutil.copy2(config_path, output_dir / "config.yaml")
    save_json(output_dir / "source_manifest.json", names_by_snr)

    results = []
    csv_rows: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(config["candidates"]):
        schedule = strengths_by_candidate[str(candidate["name"])]
        for snr in snrs:
            result = evaluate_candidate_snr(
                candidate_index=candidate_index,
                candidate=candidate,
                snr=snr,
                names=names_by_snr[snr_name(snr)],
                strength=schedule[float(snr)],
                config=config,
                pipe=pipe,
                classifier_model=classifier_model,
                classifier_preprocess=classifier_preprocess,
                categories=categories,
                lpips_model=lpips_model,
                device=device,
                output_dir=output_dir,
            )
            results.append(result)
            csv_rows.extend(result["per_sample"])
            printable = {key: value for key, value in result.items() if key != "per_sample"}
            print(json.dumps(printable, indent=2))

    import importlib.metadata as md
    import torchvision

    sources = {
        "m0_export_dir": config["inputs"]["m0_export_dir"],
        "checkpoint": config["inputs"]["checkpoint"],
    }
    metadata = {
        "project_version": "N/A (local directory is not yet a git repo)",
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
        "diffusion": config["diffusion"],
        "candidates": config["candidates"],
        "classifier": config["classifier"],
        "failure_handling": config["failure_handling"],
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
            "scripts/s5_snr_adaptive_diffusion_validation.py",
            "src/cadsd_jscc/metrics.py",
        ],
    }
    payload = {
        "metadata": metadata,
        "results": results,
        "summary_rows": aggregate_candidate(results),
    }
    save_json(output_dir / "metrics.json", payload)
    write_csv(output_dir / "per_sample.csv", csv_rows)
    write_csv(output_dir / "summary.csv", aggregate_candidate(results))
    (output_dir / "REPORT.md").write_text(make_report(results, config, sources), encoding="utf-8")
    print(json.dumps({"output_dir": project_relative(output_dir), "num_results": len(results)}, indent=2))


if __name__ == "__main__":
    main()
