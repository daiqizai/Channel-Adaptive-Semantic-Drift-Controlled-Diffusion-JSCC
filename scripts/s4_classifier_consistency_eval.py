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
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frozen-classifier pseudo-label consistency for M0/M1 outputs.")
    parser.add_argument("--config", default="configs/s4_classifier_consistency_m1_exp_s2_002.yaml")
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


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def classify_paths(
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
            images = torch.stack([preprocess(load_image(path)) for path in batch_paths]).to(device)
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


def rate(flags: list[bool]) -> float:
    if not flags:
        return 0.0
    return float(sum(flags) / len(flags))


def metrics_for_subset(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        return {
            "num_images": 0,
            "m0_matches_original_top1": 0.0,
            "m1_matches_original_top1": 0.0,
            "m1_matches_m0_top1": 0.0,
            "m0_top5_contains_original_top1": 0.0,
            "m1_top5_contains_original_top1": 0.0,
            "m0_pseudo_drift_origin": 0.0,
            "m1_pseudo_drift_origin": 0.0,
            "m1_refinement_drift": 0.0,
            "m0_pseudo_prediction_consistency": 0.0,
            "m1_pseudo_prediction_consistency": 0.0,
            "m1_minus_m0_pseudo_failure": 0.0,
        }
    m0_match_origin = [bool(row["m0_matches_original_top1"]) for row in rows]
    m1_match_origin = [bool(row["m1_matches_original_top1"]) for row in rows]
    m1_match_m0 = [bool(row["m1_matches_m0_top1"]) for row in rows]
    m0_top5_origin = [bool(row["m0_top5_contains_original_top1"]) for row in rows]
    m1_top5_origin = [bool(row["m1_top5_contains_original_top1"]) for row in rows]
    m0_failure = 1.0 - rate(m0_match_origin)
    m1_failure = 1.0 - rate(m1_match_origin)
    return {
        "num_images": len(rows),
        "m0_matches_original_top1": rate(m0_match_origin),
        "m1_matches_original_top1": rate(m1_match_origin),
        "m1_matches_m0_top1": rate(m1_match_m0),
        "m0_top5_contains_original_top1": rate(m0_top5_origin),
        "m1_top5_contains_original_top1": rate(m1_top5_origin),
        "m0_pseudo_drift_origin": m0_failure,
        "m1_pseudo_drift_origin": m1_failure,
        "m1_refinement_drift": 1.0 - rate(m1_match_m0),
        "m0_pseudo_prediction_consistency": 1.0 - m0_failure,
        "m1_pseudo_prediction_consistency": 1.0 - m1_failure,
        "m1_minus_m0_pseudo_failure": m1_failure - m0_failure,
    }


def label_for(categories: list[str], index: int) -> str:
    if 0 <= index < len(categories):
        return categories[index]
    return f"class_{index}"


def evaluate_snr(
    snr: float,
    names: list[str],
    config: dict[str, Any],
    model: torch.nn.Module,
    preprocess,
    categories: list[str],
    device: torch.device,
) -> dict[str, Any]:
    original_dir = resolve_project_path(config["inputs"]["original_dir"])
    m0_dir = resolve_project_path(config["inputs"]["m0_export_dir"]) / "exports" / snr_name(snr) / "reconstruction"
    m1_dir = resolve_project_path(config["inputs"]["m1_output_dir"]) / "exports" / snr_name(snr) / "refined"
    batch_size = int(config["classifier"]["batch_size"])
    topk = int(config["classifier"]["topk"])

    original_paths = [original_dir / name for name in names]
    m0_paths = [m0_dir / name for name in names]
    m1_paths = [m1_dir / name for name in names]

    original_preds, t_original = classify_paths(model, preprocess, original_paths, batch_size, topk, device)
    m0_preds, t_m0 = classify_paths(model, preprocess, m0_paths, batch_size, topk, device)
    m1_preds, t_m1 = classify_paths(model, preprocess, m1_paths, batch_size, topk, device)

    per_sample = []
    for idx, name in enumerate(names):
        original_top1 = original_preds[idx]["top_indices"][0]
        m0_top1 = m0_preds[idx]["top_indices"][0]
        m1_top1 = m1_preds[idx]["top_indices"][0]
        row = {
            "sample": name,
            "original": project_relative(original_paths[idx]),
            "m0_reconstruction": project_relative(m0_paths[idx]),
            "m1_refined": project_relative(m1_paths[idx]),
            "original_top1_index": original_top1,
            "original_top1_label": label_for(categories, original_top1),
            "original_top1_prob": original_preds[idx]["top_probs"][0],
            "m0_top1_index": m0_top1,
            "m0_top1_label": label_for(categories, m0_top1),
            "m0_top1_prob": m0_preds[idx]["top_probs"][0],
            "m1_top1_index": m1_top1,
            "m1_top1_label": label_for(categories, m1_top1),
            "m1_top1_prob": m1_preds[idx]["top_probs"][0],
            "m0_matches_original_top1": m0_top1 == original_top1,
            "m1_matches_original_top1": m1_top1 == original_top1,
            "m1_matches_m0_top1": m1_top1 == m0_top1,
            "m0_top5_contains_original_top1": original_top1 in m0_preds[idx]["top_indices"],
            "m1_top5_contains_original_top1": original_top1 in m1_preds[idx]["top_indices"],
            "original_top5_indices": original_preds[idx]["top_indices"],
            "original_top5_labels": [label_for(categories, item) for item in original_preds[idx]["top_indices"]],
            "original_top5_probs": original_preds[idx]["top_probs"],
            "m0_top5_indices": m0_preds[idx]["top_indices"],
            "m0_top5_labels": [label_for(categories, item) for item in m0_preds[idx]["top_indices"]],
            "m0_top5_probs": m0_preds[idx]["top_probs"],
            "m1_top5_indices": m1_preds[idx]["top_indices"],
            "m1_top5_labels": [label_for(categories, item) for item in m1_preds[idx]["top_indices"]],
            "m1_top5_probs": m1_preds[idx]["top_probs"],
        }
        per_sample.append(row)

    thresholds = [float(item) for item in config["evaluation"].get("pseudo_clean_conf_thresholds", [])]
    subsets = {"all": metrics_for_subset(per_sample)}
    for threshold in thresholds:
        key = f"original_conf_ge_{str(threshold).replace('.', 'p')}"
        subsets[key] = metrics_for_subset([row for row in per_sample if float(row["original_top1_prob"]) >= threshold])

    failure_cases = sorted(
        [
            row
            for row in per_sample
            if bool(row["m0_matches_original_top1"]) and not bool(row["m1_matches_original_top1"])
        ],
        key=lambda row: (float(row["original_top1_prob"]), float(row["m1_top1_prob"])),
        reverse=True,
    )[: int(config["evaluation"]["top_failure_count"])]

    return {
        "snr_db": float(snr),
        "num_images": len(names),
        "summary": {
            "original_top1_prob": summarize([float(row["original_top1_prob"]) for row in per_sample]),
            "m0_top1_prob": summarize([float(row["m0_top1_prob"]) for row in per_sample]),
            "m1_top1_prob": summarize([float(row["m1_top1_prob"]) for row in per_sample]),
        },
        "pseudo_label_consistency": subsets,
        "classification_time_ms_per_image": 1000.0 * (t_original + t_m0 + t_m1) / max(1, 3 * len(names)),
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
    names_by_snr = validate_inputs(config, snrs, num_samples)

    if args.dry_run:
        print(json.dumps({"status": "ok", "snrs": snrs, "sample_names_by_snr": names_by_snr}, indent=2))
        return

    output_dir = resolve_project_path(args.output_dir or config["outputs"]["output_dir"])
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists, refusing to overwrite: {output_dir}")

    torch.manual_seed(int(config["seed"]))
    device = resolve_device(args.device)
    model, preprocess, categories = load_classifier(config, device, allow_download=args.allow_download)

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
            categories=categories,
            device=device,
        )
        results.append(result)
        for row in result["per_sample"]:
            csv_rows.append({"snr_db": float(snr), **row})
        print(json.dumps({k: v for k, v in result.items() if k not in {"per_sample"}}, indent=2))

    import importlib.metadata as md
    import platform
    import torchvision

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
        "classifier": config["classifier"],
        "python_version": platform.python_version(),
        "package_versions": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "pillow": md.version("pillow"),
        },
        "proxy_environment_present": sorted(key for key in os.environ if "proxy" in key.lower()),
        "note": (
            "Frozen ImageNet classifier pseudo-label consistency is a diagnostic because COCO GT labels are not used. "
            "It does not replace the final clean-correct classification metric required by MILESTONES.md."
        ),
        "key_sources": ["scripts/s4_classifier_consistency_eval.py"],
    }
    payload = {"metadata": metadata, "results": results}
    save_json(output_dir / "metrics.json", payload)
    write_csv(output_dir / "per_sample.csv", csv_rows)
    print(json.dumps({"output_dir": project_relative(output_dir), "num_results": len(results)}, indent=2))


if __name__ == "__main__":
    main()
