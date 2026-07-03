from __future__ import annotations

import argparse
import csv
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CLIP image-image consistency for M0/M1 outputs.")
    parser.add_argument("--config", default="configs/s4_clip_consistency_m1_exp_s2_002.yaml")
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


def list_sample_names(config: dict[str, Any], snr: float, num_samples: int) -> list[str]:
    manifest_path = resolve_project_path(config["inputs"]["source_manifest"])
    manifest = load_json(manifest_path)
    names = manifest.get(snr_name(snr))
    if names is None:
        raise KeyError(f"{snr_name(snr)} not found in {manifest_path}")
    if len(names) < num_samples:
        raise RuntimeError(f"Need {num_samples} samples for {snr_name(snr)}, found {len(names)}")
    return list(names[:num_samples])


def validate_inputs(config: dict[str, Any], snrs: list[float], num_samples: int) -> dict[str, list[str]]:
    original_dir = resolve_project_path(config["inputs"]["original_dir"])
    m0_export_dir = resolve_project_path(config["inputs"]["m0_export_dir"])
    m1_output_dir = resolve_project_path(config["inputs"]["m1_output_dir"])
    checkpoint = resolve_project_path(config["inputs"]["checkpoint"])
    forbidden_checkpoint = resolve_project_path(config["inputs"]["forbidden_checkpoint"])
    required = [
        original_dir,
        m0_export_dir,
        m1_output_dir,
        resolve_project_path(config["inputs"]["m1_metrics"]),
        resolve_project_path(config["inputs"]["source_manifest"]),
        checkpoint,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")
    if checkpoint == forbidden_checkpoint:
        raise RuntimeError("Config points to forbidden latest.pt checkpoint.")

    names_by_snr: dict[str, list[str]] = {}
    for snr in snrs:
        names = list_sample_names(config, snr, num_samples)
        m0_dir = m0_export_dir / "exports" / snr_name(snr) / "reconstruction"
        m1_dir = m1_output_dir / "exports" / snr_name(snr) / "refined"
        for name in names:
            for path in [original_dir / name, m0_dir / name, m1_dir / name]:
                if not path.exists():
                    raise FileNotFoundError(f"Matched sample not found: {path}")
        names_by_snr[snr_name(snr)] = names
    return names_by_snr


def check_clip_cache(config: dict[str, Any], allow_download: bool) -> None:
    pretrained_path = config["clip"].get("pretrained_path")
    if pretrained_path:
        local_checkpoint = resolve_project_path(pretrained_path)
        if local_checkpoint.is_file() and local_checkpoint.stat().st_size > 100 * 1024 * 1024:
            return
        if not allow_download:
            raise RuntimeError(
                f"Configured CLIP checkpoint is missing or too small: {local_checkpoint}. "
                "Download it first, or rerun with --allow-download."
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
    return model, eval_preprocess


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


def cosine_diag(a: torch.Tensor, b: torch.Tensor) -> list[float]:
    return (a * b).sum(dim=-1).tolist()


def evaluate_snr(
    snr: float,
    names: list[str],
    config: dict[str, Any],
    model: torch.nn.Module,
    preprocess,
    device: torch.device,
) -> dict[str, Any]:
    original_dir = resolve_project_path(config["inputs"]["original_dir"])
    m0_dir = resolve_project_path(config["inputs"]["m0_export_dir"]) / "exports" / snr_name(snr) / "reconstruction"
    m1_dir = resolve_project_path(config["inputs"]["m1_output_dir"]) / "exports" / snr_name(snr) / "refined"
    batch_size = int(config["clip"]["batch_size"])

    original_paths = [original_dir / name for name in names]
    m0_paths = [m0_dir / name for name in names]
    m1_paths = [m1_dir / name for name in names]

    original_features, t_original = encode_paths(model, preprocess, original_paths, batch_size, device)
    m0_features, t_m0 = encode_paths(model, preprocess, m0_paths, batch_size, device)
    m1_features, t_m1 = encode_paths(model, preprocess, m1_paths, batch_size, device)

    sim_original_m0 = cosine_diag(original_features, m0_features)
    sim_original_m1 = cosine_diag(original_features, m1_features)
    sim_m0_m1 = cosine_diag(m0_features, m1_features)
    delta_m1_minus_m0 = [m1 - m0 for m0, m1 in zip(sim_original_m0, sim_original_m1)]
    drop_m0_minus_m1 = [-delta for delta in delta_m1_minus_m0]

    thresholds = [float(item) for item in config["evaluation"].get("relative_drop_thresholds", [])]
    diagnostic_rates = {
        "m1_less_similar_than_m0": float(sum(delta < 0 for delta in delta_m1_minus_m0) / len(delta_m1_minus_m0)),
    }
    for threshold in thresholds:
        key = f"drop_ge_{str(threshold).replace('.', 'p')}"
        diagnostic_rates[key] = float(sum(drop >= threshold for drop in drop_m0_minus_m1) / len(drop_m0_minus_m1))

    per_sample = []
    for idx, name in enumerate(names):
        per_sample.append(
            {
                "sample": name,
                "original": project_relative(original_paths[idx]),
                "m0_reconstruction": project_relative(m0_paths[idx]),
                "m1_refined": project_relative(m1_paths[idx]),
                "clip_sim_original_m0": float(sim_original_m0[idx]),
                "clip_sim_original_m1": float(sim_original_m1[idx]),
                "clip_sim_m0_m1": float(sim_m0_m1[idx]),
                "clip_delta_m1_minus_m0": float(delta_m1_minus_m0[idx]),
                "clip_drop_m0_minus_m1": float(drop_m0_minus_m1[idx]),
            }
        )

    top_count = int(config["evaluation"]["top_failure_count"])
    failure_cases = sorted(per_sample, key=lambda item: item["clip_drop_m0_minus_m1"], reverse=True)[:top_count]

    return {
        "snr_db": float(snr),
        "num_images": len(names),
        "summary": {
            "clip_sim_original_m0": summarize(sim_original_m0),
            "clip_sim_original_m1": summarize(sim_original_m1),
            "clip_sim_m0_m1": summarize(sim_m0_m1),
            "clip_delta_m1_minus_m0": summarize(delta_m1_minus_m0),
            "clip_drop_m0_minus_m1": summarize(drop_m0_minus_m1),
        },
        "diagnostic_rates": diagnostic_rates,
        "encode_time_ms_per_image": 1000.0 * (t_original + t_m0 + t_m1) / max(1, 3 * len(names)),
        "per_sample": per_sample,
        "failure_cases": failure_cases,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    snrs = parse_snrs(args.snrs, config)
    num_samples = int(args.num_samples or config["evaluation"]["num_samples"])
    config["evaluation"]["num_samples"] = num_samples
    names_by_snr = validate_inputs(config, snrs, num_samples)

    if args.dry_run:
        print(json.dumps({"status": "ok", "snrs": snrs, "sample_names_by_snr": names_by_snr}, indent=2))
        return

    output_dir = resolve_project_path(args.output_dir or config["outputs"]["output_dir"])
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists, refusing to overwrite: {output_dir}")

    torch.manual_seed(int(config["seed"]))
    device = resolve_device(args.device)
    model, preprocess = load_clip_model(config, device, allow_download=args.allow_download)

    output_dir.mkdir(parents=True)
    shutil.copy2(config_path, output_dir / "config.yaml")
    save_json(output_dir / "source_manifest.json", names_by_snr)

    results = []
    csv_rows: list[dict[str, Any]] = []
    for snr in snrs:
        result = evaluate_snr(
            snr=snr,
            names=names_by_snr[snr_name(snr)],
            config=config,
            model=model,
            preprocess=preprocess,
            device=device,
        )
        results.append(result)
        for row in result["per_sample"]:
            csv_rows.append({"snr_db": float(snr), **row})
        print(json.dumps({k: v for k, v in result.items() if k not in {"per_sample"}}, indent=2))

    import open_clip
    import importlib.metadata as md
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
        "note": "CLIP image-image similarity is an auxiliary semantic diagnostic, not the final classification-consistency metric.",
        "key_sources": ["scripts/s4_clip_consistency_eval.py"],
    }
    payload = {"metadata": metadata, "results": results}
    save_json(output_dir / "metrics.json", payload)
    write_csv(output_dir / "per_sample.csv", csv_rows)
    print(json.dumps({"output_dir": project_relative(output_dir), "num_results": len(results)}, indent=2))


if __name__ == "__main__":
    main()
