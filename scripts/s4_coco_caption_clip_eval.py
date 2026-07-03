from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RE = re.compile(r"^sample_(\d{6})\.png$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate COCO caption CLIP text consistency for M0/M1 outputs.")
    parser.add_argument("--config", default="configs/s4_coco_caption_clip_m1_exp_s2_002.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--snrs", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--allow-download", action="store_true")
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


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def sample_index(name: str) -> int:
    match = SAMPLE_RE.match(name)
    if not match:
        raise ValueError(f"Unexpected sample file name: {name}")
    return int(match.group(1))


def coco_image_id_from_path(path: str | Path) -> int:
    stem = Path(path).stem
    return int(stem)


def list_sample_names(config: dict[str, Any], snr: float, num_samples: int) -> list[str]:
    manifest_path = resolve_project_path(config["inputs"]["source_manifest"])
    manifest = load_json(manifest_path)
    names = manifest.get(snr_name(snr))
    if names is None:
        raise KeyError(f"{snr_name(snr)} not found in {manifest_path}")
    if len(names) < num_samples:
        raise RuntimeError(f"Need {num_samples} samples for {snr_name(snr)}, found {len(names)}")
    return list(names[:num_samples])


def load_export_source_paths(config: dict[str, Any]) -> list[str]:
    manifest_path = resolve_project_path(config["inputs"]["m0_source_manifest"])
    manifest = load_json(manifest_path)
    paths = manifest.get("paths")
    if not isinstance(paths, list):
        raise RuntimeError(f"Expected paths list in {manifest_path}")
    return [str(item) for item in paths]


def load_coco_captions(path: Path) -> dict[int, list[str]]:
    payload = load_json(path)
    annotations = payload.get("annotations")
    if not isinstance(annotations, list):
        raise RuntimeError(f"Expected COCO annotations list in {path}")
    captions_by_id: dict[int, list[str]] = {}
    for item in annotations:
        image_id = int(item["image_id"])
        caption = str(item["caption"]).strip()
        if caption:
            captions_by_id.setdefault(image_id, []).append(caption)
    return captions_by_id


def validate_inputs(config: dict[str, Any], snrs: list[float], num_samples: int) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    original_dir = resolve_project_path(config["inputs"]["original_dir"])
    m0_export_dir = resolve_project_path(config["inputs"]["m0_export_dir"])
    m1_output_dir = resolve_project_path(config["inputs"]["m1_output_dir"])
    checkpoint = resolve_project_path(config["inputs"]["checkpoint"])
    forbidden_checkpoint = resolve_project_path(config["inputs"]["forbidden_checkpoint"])
    captions_path = resolve_project_path(config["inputs"]["coco_captions"])
    required = [
        original_dir,
        m0_export_dir,
        m1_output_dir,
        resolve_project_path(config["inputs"]["m1_metrics"]),
        resolve_project_path(config["inputs"]["source_manifest"]),
        resolve_project_path(config["inputs"]["m0_source_manifest"]),
        captions_path,
        checkpoint,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")
    if checkpoint == forbidden_checkpoint:
        raise RuntimeError("Config points to forbidden latest.pt checkpoint.")

    export_paths = load_export_source_paths(config)
    captions_by_id = load_coco_captions(captions_path)
    names_by_snr: dict[str, list[str]] = {}
    sample_metadata: dict[str, dict[str, Any]] = {}
    for snr in snrs:
        names = list_sample_names(config, snr, num_samples)
        m0_dir = m0_export_dir / "exports" / snr_name(snr) / "reconstruction"
        m1_dir = m1_output_dir / "exports" / snr_name(snr) / "refined"
        for name in names:
            idx = sample_index(name)
            if idx >= len(export_paths):
                raise RuntimeError(f"{name} maps to source index {idx}, but only {len(export_paths)} source paths exist")
            source_path = export_paths[idx]
            image_id = coco_image_id_from_path(source_path)
            captions = captions_by_id.get(image_id, [])
            if not captions:
                raise RuntimeError(f"No COCO captions found for image_id={image_id} from {source_path}")
            for path in [original_dir / name, m0_dir / name, m1_dir / name]:
                if not path.exists():
                    raise FileNotFoundError(f"Matched sample not found: {path}")
            sample_metadata[name] = {
                "sample_index": idx,
                "coco_image_id": image_id,
                "coco_source_path": source_path,
                "num_captions": len(captions),
                "captions": captions,
            }
        names_by_snr[snr_name(snr)] = names
    return names_by_snr, sample_metadata


def check_clip_cache(config: dict[str, Any], allow_download: bool) -> None:
    pretrained_path = config["clip"].get("pretrained_path")
    if pretrained_path:
        local_checkpoint = resolve_project_path(pretrained_path)
        if local_checkpoint.is_file() and local_checkpoint.stat().st_size > 100 * 1024 * 1024:
            return
        if not allow_download:
            raise RuntimeError(
                f"Configured CLIP checkpoint is missing or too small: {local_checkpoint}. "
                "Download it first, or rerun with --allow-download using cleared proxy variables."
            )

    cache_dir = resolve_project_path(config["clip"]["cache_dir"])
    if allow_download:
        cache_dir.mkdir(parents=True, exist_ok=True)
        return
    cached = list(cache_dir.glob("**/*.pt")) + list(cache_dir.glob("**/*.pth")) + list(cache_dir.glob("**/*.bin"))
    if not any(path.stat().st_size > 100 * 1024 * 1024 for path in cached if path.is_file()):
        raise RuntimeError(
            f"No large open_clip checkpoint found under {cache_dir}. "
            "Run with --allow-download using cleared proxy variables for the first run."
        )


def load_clip_model(config: dict[str, Any], device: torch.device, allow_download: bool):
    import open_clip

    check_clip_cache(config, allow_download=allow_download)
    clip_cfg = config["clip"]
    cache_dir = resolve_project_path(clip_cfg["cache_dir"])
    pretrained_path = clip_cfg.get("pretrained_path")
    pretrained = resolve_project_path(pretrained_path) if pretrained_path and resolve_project_path(pretrained_path).is_file() else clip_cfg["pretrained"]
    model, _train_preprocess, eval_preprocess = open_clip.create_model_and_transforms(
        model_name=str(clip_cfg["model_name"]),
        pretrained=str(pretrained),
        precision=str(clip_cfg.get("precision", "fp32")),
        device=device,
        cache_dir=str(cache_dir),
        force_quick_gelu=bool(clip_cfg.get("force_quick_gelu", False)),
        image_mean=tuple(float(item) for item in clip_cfg["image_mean"]) if "image_mean" in clip_cfg else None,
        image_std=tuple(float(item) for item in clip_cfg["image_std"]) if "image_std" in clip_cfg else None,
        image_interpolation=clip_cfg.get("image_interpolation"),
        image_resize_mode=clip_cfg.get("image_resize_mode"),
        weights_only=bool(clip_cfg.get("weights_only", True)),
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(str(clip_cfg["model_name"]))
    return model, eval_preprocess, tokenizer


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def encode_paths(
    model: torch.nn.Module,
    preprocess,
    paths: list[Path],
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    features: list[torch.Tensor] = []
    elapsed = 0.0
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = torch.stack([preprocess(load_image(path)) for path in batch_paths]).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            begin = time.perf_counter()
            encoded = model.encode_image(images)
            encoded = F.normalize(encoded.float(), dim=-1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - begin
            features.append(encoded.detach().cpu())
    return torch.cat(features, dim=0), elapsed


def encode_texts(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    features: list[torch.Tensor] = []
    elapsed = 0.0
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            tokens = tokenizer(batch_texts).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            begin = time.perf_counter()
            encoded = model.encode_text(tokens)
            encoded = F.normalize(encoded.float(), dim=-1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - begin
            features.append(encoded.detach().cpu())
    return torch.cat(features, dim=0), elapsed


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0}
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "min": float(tensor.min().item()),
        "p25": float(torch.quantile(tensor, 0.25).item()),
        "median": float(torch.quantile(tensor, 0.50).item()),
        "p75": float(torch.quantile(tensor, 0.75).item()),
        "max": float(tensor.max().item()),
    }


def caption_scores(image_features: torch.Tensor, text_features: torch.Tensor) -> tuple[list[float], list[float]]:
    sims = image_features @ text_features.T
    max_scores = sims.max(dim=1).values.tolist()
    mean_scores = sims.mean(dim=1).tolist()
    return [float(item) for item in max_scores], [float(item) for item in mean_scores]


def evaluate_snr(
    snr: float,
    names: list[str],
    sample_metadata: dict[str, dict[str, Any]],
    config: dict[str, Any],
    model: torch.nn.Module,
    preprocess,
    tokenizer,
    device: torch.device,
) -> dict[str, Any]:
    original_dir = resolve_project_path(config["inputs"]["original_dir"])
    m0_dir = resolve_project_path(config["inputs"]["m0_export_dir"]) / "exports" / snr_name(snr) / "reconstruction"
    m1_dir = resolve_project_path(config["inputs"]["m1_output_dir"]) / "exports" / snr_name(snr) / "refined"
    batch_size = int(config["clip"]["batch_size"])
    text_batch_size = int(config["clip"].get("text_batch_size", batch_size))

    original_paths = [original_dir / name for name in names]
    m0_paths = [m0_dir / name for name in names]
    m1_paths = [m1_dir / name for name in names]

    original_features, t_original = encode_paths(model, preprocess, original_paths, batch_size, device)
    m0_features, t_m0 = encode_paths(model, preprocess, m0_paths, batch_size, device)
    m1_features, t_m1 = encode_paths(model, preprocess, m1_paths, batch_size, device)

    per_sample = []
    original_caption_max: list[float] = []
    m0_caption_max: list[float] = []
    m1_caption_max: list[float] = []
    original_caption_mean: list[float] = []
    m0_caption_mean: list[float] = []
    m1_caption_mean: list[float] = []
    text_elapsed = 0.0

    for idx, name in enumerate(names):
        captions = list(sample_metadata[name]["captions"])
        text_features, t_text = encode_texts(model, tokenizer, captions, text_batch_size, device)
        text_elapsed += t_text
        o_max, o_mean = caption_scores(original_features[idx : idx + 1], text_features)
        m0_max, m0_mean = caption_scores(m0_features[idx : idx + 1], text_features)
        m1_max, m1_mean = caption_scores(m1_features[idx : idx + 1], text_features)
        original_caption_max.append(o_max[0])
        m0_caption_max.append(m0_max[0])
        m1_caption_max.append(m1_max[0])
        original_caption_mean.append(o_mean[0])
        m0_caption_mean.append(m0_mean[0])
        m1_caption_mean.append(m1_mean[0])
        per_sample.append(
            {
                "sample": name,
                "original": project_relative(original_paths[idx]),
                "m0_reconstruction": project_relative(m0_paths[idx]),
                "m1_refined": project_relative(m1_paths[idx]),
                "coco_image_id": int(sample_metadata[name]["coco_image_id"]),
                "coco_source_path": str(sample_metadata[name]["coco_source_path"]),
                "num_captions": int(sample_metadata[name]["num_captions"]),
                "captions": captions,
                "clip_text_sim_caption_max_original": float(o_max[0]),
                "clip_text_sim_caption_max_m0": float(m0_max[0]),
                "clip_text_sim_caption_max_m1": float(m1_max[0]),
                "clip_text_sim_caption_mean_original": float(o_mean[0]),
                "clip_text_sim_caption_mean_m0": float(m0_mean[0]),
                "clip_text_sim_caption_mean_m1": float(m1_mean[0]),
                "clip_text_delta_max_m1_minus_m0": float(m1_max[0] - m0_max[0]),
                "clip_text_drop_max_m0_minus_m1": float(m0_max[0] - m1_max[0]),
                "clip_text_delta_mean_m1_minus_m0": float(m1_mean[0] - m0_mean[0]),
                "clip_text_drop_mean_m0_minus_m1": float(m0_mean[0] - m1_mean[0]),
            }
        )

    delta_max = [m1 - m0 for m0, m1 in zip(m0_caption_max, m1_caption_max)]
    drop_max = [-item for item in delta_max]
    delta_mean = [m1 - m0 for m0, m1 in zip(m0_caption_mean, m1_caption_mean)]
    drop_mean = [-item for item in delta_mean]

    thresholds = [float(item) for item in config["evaluation"].get("relative_drop_thresholds", [])]
    diagnostic_rates = {
        "m1_caption_max_less_than_m0": float(sum(delta < 0 for delta in delta_max) / len(delta_max)),
        "m1_caption_mean_less_than_m0": float(sum(delta < 0 for delta in delta_mean) / len(delta_mean)),
    }
    for threshold in thresholds:
        key = f"caption_max_drop_ge_{str(threshold).replace('.', 'p')}"
        diagnostic_rates[key] = float(sum(drop >= threshold for drop in drop_max) / len(drop_max))
        key = f"caption_mean_drop_ge_{str(threshold).replace('.', 'p')}"
        diagnostic_rates[key] = float(sum(drop >= threshold for drop in drop_mean) / len(drop_mean))

    top_count = int(config["evaluation"]["top_failure_count"])
    failure_cases = sorted(per_sample, key=lambda item: item["clip_text_drop_max_m0_minus_m1"], reverse=True)[:top_count]

    return {
        "snr_db": float(snr),
        "num_images": len(names),
        "summary": {
            "clip_text_sim_caption_max_original": summarize(original_caption_max),
            "clip_text_sim_caption_max_m0": summarize(m0_caption_max),
            "clip_text_sim_caption_max_m1": summarize(m1_caption_max),
            "clip_text_delta_max_m1_minus_m0": summarize(delta_max),
            "clip_text_drop_max_m0_minus_m1": summarize(drop_max),
            "clip_text_sim_caption_mean_original": summarize(original_caption_mean),
            "clip_text_sim_caption_mean_m0": summarize(m0_caption_mean),
            "clip_text_sim_caption_mean_m1": summarize(m1_caption_mean),
            "clip_text_delta_mean_m1_minus_m0": summarize(delta_mean),
            "clip_text_drop_mean_m0_minus_m1": summarize(drop_mean),
        },
        "diagnostic_rates": diagnostic_rates,
        "encode_time_ms_per_image": 1000.0 * (t_original + t_m0 + t_m1 + text_elapsed) / max(1, 3 * len(names) + len(names)),
        "per_sample": per_sample,
        "failure_cases": failure_cases,
    }


def serialize_csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_csv_value(value) for key, value in row.items()})


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    snrs = parse_snrs(args.snrs, config)
    num_samples = int(args.num_samples or config["evaluation"]["num_samples"])
    config["evaluation"]["num_samples"] = num_samples
    names_by_snr, sample_metadata = validate_inputs(config, snrs, num_samples)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "snrs": snrs,
                    "sample_names_by_snr": names_by_snr,
                    "sample_metadata_preview": {key: sample_metadata[key] for key in sorted(sample_metadata)[:3]},
                },
                indent=2,
            )
        )
        return

    output_dir = resolve_project_path(args.output_dir or config["outputs"]["output_dir"])
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists, refusing to overwrite: {output_dir}")

    torch.manual_seed(int(config["seed"]))
    device = resolve_device(args.device)
    model, preprocess, tokenizer = load_clip_model(config, device, allow_download=args.allow_download)

    output_dir.mkdir(parents=True)
    shutil.copy2(config_path, output_dir / "config.yaml")
    save_json(output_dir / "source_manifest.json", names_by_snr)
    save_json(output_dir / "sample_metadata.json", sample_metadata)

    results = []
    csv_rows: list[dict[str, Any]] = []
    for snr in snrs:
        result = evaluate_snr(
            snr=snr,
            names=names_by_snr[snr_name(snr)],
            sample_metadata=sample_metadata,
            config=config,
            model=model,
            preprocess=preprocess,
            tokenizer=tokenizer,
            device=device,
        )
        results.append(result)
        for row in result["per_sample"]:
            csv_rows.append({"snr_db": float(snr), **row})
        print(json.dumps({k: v for k, v in result.items() if k not in {"per_sample"}}, indent=2))

    import importlib.metadata as md
    import open_clip
    import platform

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
        "inputs": config["inputs"],
        "clip": config["clip"],
        "python_version": platform.python_version(),
        "package_versions": {
            "torch": torch.__version__,
            "open_clip": getattr(open_clip, "__version__", md.version("open_clip_torch")),
            "pillow": md.version("pillow"),
        },
        "proxy_environment_present": sorted(key for key in os.environ if "proxy" in key.lower()),
        "note": (
            "COCO caption CLIP image-text similarity is an auxiliary semantic diagnostic. "
            "It uses COCO captions but does not replace the final clean-correct frozen-classifier metric."
        ),
        "key_sources": ["scripts/s4_coco_caption_clip_eval.py"],
    }
    payload = {"metadata": metadata, "results": results}
    save_json(output_dir / "metrics.json", payload)
    write_csv(output_dir / "per_sample.csv", csv_rows)
    print(json.dumps({"output_dir": project_relative(output_dir), "num_results": len(results)}, indent=2))


if __name__ == "__main__":
    main()
