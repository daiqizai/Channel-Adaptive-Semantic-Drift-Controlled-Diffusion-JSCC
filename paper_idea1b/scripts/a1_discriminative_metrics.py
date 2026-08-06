#!/usr/bin/env python3
"""Complete A1 perceptual/semantic/distribution metrics and paired verdicts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from cleanfid import features as clean_features
from cleanfid import fid as clean_fid
from PIL import Image
from torchmetrics.image.dists import DeepImageStructureAndTextureSimilarity
from torchvision.transforms.functional import pil_to_tensor


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


SCRIPT = Path(__file__).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="paper_idea1b/configs/a1_discriminative_benchmark.yaml"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--stage", choices=("full_reference", "distribution", "summarize"), required=True
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("refuse to write empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def row_key(row: dict[str, Any]) -> tuple[str, str, int, float]:
    return (
        str(row["arm"]),
        str(row["sample_id"]),
        int(row["seed"]),
        float(row["snr_db"]),
    )


def load_all_reconstruction_rows(
    output: Path, config: dict[str, Any]
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dataset in ("kodak", "clic2020_test"):
        summary_path = output / f"summary_{dataset}_reconstruction.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"{dataset} reconstruction is incomplete")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary["status"] != "reconstruction_complete"
            or summary["paired_actual_rate_and_noise_equal_across_arms"] is not True
            or summary["official_imagenette_validation_accessed"] is not False
        ):
            raise RuntimeError(f"invalid reconstruction summary: {dataset}")
        csv_path = resolve(summary["per_sample_csv"])
        if sha256_file(csv_path) != summary["per_sample_csv_sha256"]:
            raise RuntimeError(f"reconstruction CSV SHA mismatch: {dataset}")
        dataset_rows = read_csv(csv_path)
        expected = (
            int(config["formal"][dataset]["sample_count"])
            * len(config["formal"][dataset]["seeds"])
            * len(config["formal"][dataset]["snrs_db"])
            * len(config["models"]["order"])
        )
        if len(dataset_rows) != expected:
            raise RuntimeError(f"{dataset} row count mismatch")
        rows.extend(dataset_rows)
    if len({row_key(row) for row in rows}) != len(rows):
        raise RuntimeError("duplicate reconstruction row keys")
    return rows


def load_rgb(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        value = pil_to_tensor(image.convert("RGB")).float().div_(255.0)
    return value.unsqueeze(0).to(device)


def build_clip(config: dict[str, Any], device: torch.device):
    import open_clip

    entry = config["metrics"]["clip"]
    checkpoint = resolve(entry["checkpoint"])
    if sha256_file(checkpoint) != str(entry["sha256"]):
        raise RuntimeError("CLIP checkpoint SHA mismatch")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=str(entry["model_name"]),
        pretrained=str(checkpoint),
        precision="fp32",
        device=device,
        # The SHA-frozen local OpenAI CLIP weight is a TorchScript archive.
        # PyTorch 2.6 rejects TorchScript archives under weights_only=True.
        weights_only=False,
    )
    return model.eval().requires_grad_(False), preprocess


@torch.inference_mode()
def clip_embedding(
    model: torch.nn.Module,
    preprocess,
    image_path: Path,
    device: torch.device,
) -> torch.Tensor:
    with Image.open(image_path) as image:
        tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    return F.normalize(model.encode_image(tensor).float(), dim=-1)[0]


def run_full_reference(
    output: Path,
    config: dict[str, Any],
    device: torch.device,
    resume: bool,
) -> None:
    rows = load_all_reconstruction_rows(output, config)
    progress_path = output / "per_sample_perceptual_semantic.jsonl"
    existing = read_jsonl(progress_path)
    completed = {row_key(row) for row in existing}
    if existing and not resume:
        raise RuntimeError("existing full-reference metrics require --resume")
    if len(completed) != len(existing):
        raise RuntimeError("duplicate metric progress keys")

    import lpips

    lpips_model = lpips.LPIPS(net="alex").to(device).eval().requires_grad_(False)
    dists_model = DeepImageStructureAndTextureSimilarity(reduction="mean").to(device)
    clip_model, clip_preprocess = build_clip(config, device)
    clip_source_cache: dict[str, torch.Tensor] = {}
    started = time.perf_counter()
    try:
        for index, row in enumerate(rows, 1):
            key = row_key(row)
            if key in completed:
                continue
            source_path = resolve(row["source_path"])
            reconstruction_path = resolve(row["reconstruction_path"])
            source = load_rgb(source_path, device)
            reconstruction = load_rgb(reconstruction_path, device)
            with torch.inference_mode():
                lpips_value = float(
                    lpips_model(
                        reconstruction * 2.0 - 1.0, source * 2.0 - 1.0
                    ).item()
                )
                dists_value = float(dists_model(reconstruction, source).item())
                dists_model.reset()
            if str(row["sample_id"]) not in clip_source_cache:
                clip_source_cache[str(row["sample_id"])] = clip_embedding(
                    clip_model,
                    clip_preprocess,
                    source_path,
                    device,
                )
            reconstruction_embedding = clip_embedding(
                clip_model,
                clip_preprocess,
                reconstruction_path,
                device,
            )
            clip_value = float(
                (
                    clip_source_cache[str(row["sample_id"])]
                    * reconstruction_embedding
                )
                .sum()
                .item()
            )
            metric_row = {
                "arm": row["arm"],
                "dataset": row["dataset"],
                "sample_id": row["sample_id"],
                "seed": int(row["seed"]),
                "snr_db": float(row["snr_db"]),
                "lpips": lpips_value,
                "dists": dists_value,
                "clip_image_cosine": clip_value,
            }
            if not all(
                math.isfinite(float(metric_row[name]))
                for name in ("lpips", "dists", "clip_image_cosine")
            ):
                raise RuntimeError(f"non-finite perceptual metric: {key}")
            append_jsonl(progress_path, metric_row)
            existing.append(metric_row)
            completed.add(key)
            if len(existing) % 25 == 0:
                write_json(
                    output / "STATE.json",
                    {
                        "status": "full_reference_metrics_running",
                        "completed_rows": len(existing),
                        "expected_rows": len(rows),
                        "elapsed_seconds_this_invocation": time.perf_counter()
                        - started,
                        "official_imagenette_validation_accessed": False,
                    },
                )
            del source, reconstruction
        if len(existing) != len(rows):
            raise RuntimeError("full-reference metric row count mismatch")
        csv_path = output / "per_sample_perceptual_semantic.csv"
        write_csv(csv_path, existing)
        write_json(
            output / "summary_full_reference_metrics.json",
            {
                "status": "complete",
                "rows": len(existing),
                "elapsed_seconds_this_invocation": time.perf_counter() - started,
                "csv": relative(csv_path),
                "csv_sha256": sha256_file(csv_path),
                "lpips": "lpips_0.1.4_alex",
                "dists": "torchmetrics_1.9.0",
                "clip": "open_clip_ViT-B-32_local_frozen_weight",
                "dreamsim": "not_run_not_installed_and_not_selected_after_results",
                "official_imagenette_validation_accessed": False,
                "script_sha256": sha256_file(SCRIPT),
            },
        )
    except Exception:
        write_json(
            output / "STATE.json",
            {
                "status": "full_reference_metrics_failed",
                "completed_rows": len(existing),
                "expected_rows": len(rows),
                "official_imagenette_validation_accessed": False,
                "traceback": traceback.format_exc(),
            },
        )
        raise


def run_distribution(
    output: Path,
    config: dict[str, Any],
    device: torch.device,
    resume: bool,
) -> None:
    load_all_reconstruction_rows(output, config)
    source_root = resolve(config["datasets"]["clic2020_test"]["root"]) if "datasets" in config else resolve("paper_idea1b/data/clic2020_test")
    progress_path = output / "distribution_metrics.jsonl"
    existing = read_jsonl(progress_path)
    completed = {
        (str(row["arm"]), int(row["seed"]), float(row["snr_db"])) for row in existing
    }
    if existing and not resume:
        raise RuntimeError("existing distribution metrics require --resume")
    model = clean_features.build_feature_extractor(
        "clean", device=device, use_dataparallel=False
    )
    source_features = clean_fid.get_folder_features(
        str(source_root),
        model=model,
        num_workers=4,
        batch_size=16,
        device=device,
        mode="clean",
        description="CLIC source",
        verbose=True,
    )
    group_index = 0
    for arm in map(str, config["models"]["order"]):
        for seed in map(int, config["formal"]["clic2020_test"]["seeds"]):
            for snr in map(float, config["formal"]["clic2020_test"]["snrs_db"]):
                key = (arm, seed, snr)
                kid_seed = (
                    int(config["metrics"]["bootstrap_seed"]) + 50000 + group_index
                )
                group_index += 1
                if key in completed:
                    continue
                folder = (
                    output
                    / "reconstructions"
                    / "clic2020_test"
                    / arm
                    / f"seed_{seed}"
                    / f"snr_{int(snr):02d}"
                )
                features = clean_fid.get_folder_features(
                    str(folder),
                    model=model,
                    num_workers=4,
                    batch_size=16,
                    device=device,
                    mode="clean",
                    description=f"{arm} {snr:g}dB",
                    verbose=True,
                )
                if len(features) != 428:
                    raise RuntimeError(f"FID feature count mismatch: {key}")
                np.random.seed(kid_seed)
                row = {
                    "arm": arm,
                    "seed": seed,
                    "snr_db": snr,
                    "source_count": len(source_features),
                    "reconstruction_count": len(features),
                    "fid": float(clean_fid.fid_from_feats(source_features, features)),
                    "kid": float(
                        clean_fid.kernel_distance(
                            source_features,
                            features,
                            num_subsets=100,
                            max_subset_size=428,
                        )
                    ),
                    "kid_rng_seed": kid_seed,
                }
                append_jsonl(progress_path, row)
                existing.append(row)
                completed.add(key)
                write_json(
                    output / "STATE.json",
                    {
                        "status": "distribution_metrics_running",
                        "completed_groups": len(existing),
                        "expected_groups": 15,
                        "official_imagenette_validation_accessed": False,
                    },
                )
    if len(existing) != 15:
        raise RuntimeError("distribution group count mismatch")
    csv_path = output / "distribution_metrics.csv"
    write_csv(csv_path, existing)
    write_json(
        output / "summary_distribution_metrics.json",
        {
            "status": "complete",
            "groups": len(existing),
            "source_features_reused_across_all_groups": True,
            "csv": relative(csv_path),
            "csv_sha256": sha256_file(csv_path),
            "official_imagenette_validation_accessed": False,
            "script_sha256": sha256_file(SCRIPT),
        },
    )


def cluster_ci(
    rows: list[dict[str, Any]], field: str, replicates: int, seed: int
) -> list[float]:
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sample_id"])].append(float(row[field]))
    values = np.asarray(
        [np.mean(grouped[key]) for key in sorted(grouped)], dtype=np.float64
    )
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(replicates, len(values)))
    boot = values[indices].mean(axis=1)
    return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def metric_block(
    rows: list[dict[str, Any]], config: dict[str, Any], seed_offset: int
) -> dict[str, Any]:
    metrics = ("psnr", "ms_ssim", "lpips", "dists", "clip_image_cosine")
    replicates = int(config["metrics"]["bootstrap_replicates"])
    seed = int(config["metrics"]["bootstrap_seed"]) + seed_offset
    result: dict[str, Any] = {
        "rows": len(rows),
        "source_clusters": len({row["sample_id"] for row in rows}),
        "delta_definition": "S33_minus_named_Swin_arm",
        "higher_is_better": ["psnr", "ms_ssim", "clip_image_cosine"],
        "lower_is_better": ["lpips", "dists"],
        "arms": {},
        "comparisons": {},
    }
    for arm in config["models"]["order"]:
        arm_rows = [row for row in rows if row["arm"] == arm]
        result["arms"][arm] = {
            metric: float(np.mean([float(row[metric]) for row in arm_rows]))
            for metric in metrics
        }
    for index, arm in enumerate(config["models"]["order"][1:]):
        pairs = []
        by_key = {
            (
                row["sample_id"],
                int(row["seed"]),
                float(row["snr_db"]),
            ): row
            for row in rows
            if row["arm"] == arm
        }
        for s33 in (row for row in rows if row["arm"] == "s33_strong"):
            key = (s33["sample_id"], int(s33["seed"]), float(s33["snr_db"]))
            other = by_key[key]
            pair = {"sample_id": s33["sample_id"]}
            for metric in metrics:
                pair[f"delta_{metric}"] = float(s33[metric]) - float(other[metric])
            pairs.append(pair)
        comparison: dict[str, Any] = {}
        for metric in metrics:
            field = f"delta_{metric}"
            comparison[metric] = {
                "s33_minus_swin_mean": float(
                    np.mean([float(row[field]) for row in pairs])
                ),
                "source_cluster_95ci": cluster_ci(
                    pairs, field, replicates, seed + index * 100
                ),
            }
        margin = float(config["metrics"]["psnr_noninferiority_margin_db"])
        lower = comparison["psnr"]["source_cluster_95ci"][0]
        if lower > 0:
            verdict = "S33_SIGNIFICANTLY_EXCEEDS_SWIN"
        elif lower >= -margin:
            verdict = "S33_NONINFERIOR_TIED_WITH_SWIN"
        else:
            verdict = "S33_INFERIOR_TO_SWIN"
        comparison["psnr_margin_verdict"] = verdict
        result["comparisons"][arm] = comparison
    arm_verdicts = [
        comparison["psnr_margin_verdict"]
        for comparison in result["comparisons"].values()
    ]
    if "S33_INFERIOR_TO_SWIN" in arm_verdicts:
        result["conservative_two_arm_psnr_verdict"] = "S33_INFERIOR_TO_SWIN"
    elif all(value == "S33_SIGNIFICANTLY_EXCEEDS_SWIN" for value in arm_verdicts):
        result["conservative_two_arm_psnr_verdict"] = (
            "S33_SIGNIFICANTLY_EXCEEDS_BOTH_SWIN_ARMS"
        )
    else:
        result["conservative_two_arm_psnr_verdict"] = (
            "S33_NONINFERIOR_UNDER_WORST_SWIN_ARM"
        )
    return result


def summarize(
    output: Path, config: dict[str, Any], config_path: Path
) -> None:
    reconstruction_rows = load_all_reconstruction_rows(output, config)
    metric_summary = json.loads(
        (output / "summary_full_reference_metrics.json").read_text(encoding="utf-8")
    )
    if metric_summary["status"] != "complete":
        raise RuntimeError("full-reference metrics are incomplete")
    metric_path = resolve(metric_summary["csv"])
    if sha256_file(metric_path) != metric_summary["csv_sha256"]:
        raise RuntimeError("perceptual metric CSV SHA mismatch")
    metrics_by_key = {row_key(row): row for row in read_csv(metric_path)}
    combined: list[dict[str, Any]] = []
    for row in reconstruction_rows:
        metric = metrics_by_key[row_key(row)]
        combined.append(
            {
                **row,
                "lpips": float(metric["lpips"]),
                "dists": float(metric["dists"]),
                "clip_image_cosine": float(metric["clip_image_cosine"]),
            }
        )
    combined_path = output / "per_sample_all_metrics.csv"
    write_csv(combined_path, combined)
    by_dataset: dict[str, Any] = {}
    for dataset_index, dataset in enumerate(("kodak", "clic2020_test")):
        dataset_rows = [row for row in combined if row["dataset"] == dataset]
        aggregate = metric_block(dataset_rows, config, dataset_index * 10000)
        per_snr = {}
        for snr_index, snr in enumerate(config["formal"][dataset]["snrs_db"]):
            per_snr[str(int(snr))] = metric_block(
                [
                    row
                    for row in dataset_rows
                    if float(row["snr_db"]) == float(snr)
                ],
                config,
                dataset_index * 10000 + snr_index + 1,
            )
        by_dataset[dataset] = {"aggregate": aggregate, "per_snr": per_snr}
    distribution_summary_path = output / "summary_distribution_metrics.json"
    if not distribution_summary_path.is_file():
        raise FileNotFoundError("distribution metrics are incomplete")
    distribution_summary = json.loads(
        distribution_summary_path.read_text(encoding="utf-8")
    )
    if distribution_summary["status"] != "complete":
        raise RuntimeError("distribution metrics are incomplete")
    distribution_path = resolve(distribution_summary["csv"])
    if sha256_file(distribution_path) != distribution_summary["csv_sha256"]:
        raise RuntimeError("distribution metric CSV SHA mismatch")
    summary = {
        "status": "complete",
        "analysis_id": config["analysis"]["id"],
        "by_dataset": by_dataset,
        "distribution_metrics": read_csv(distribution_path),
        "claim_scope": {
            "strict_ranking": "S33_vs_Swin_only_same_actual_CBR",
            "semantic_metric": (
                "unlabeled_source_reconstruction_CLIP_similarity;"
                "_no_supervised_failure_rate_on_Kodak_or_CLIC"
            ),
            "diffjscc": "not_run_A2_not_authorized",
            "sgd": "not_run_permanent_nonranking",
            "official_imagenette_validation": "sealed",
        },
        "per_sample_csv": relative(combined_path),
        "per_sample_csv_sha256": sha256_file(combined_path),
        "script_sha256": sha256_file(SCRIPT),
        "config_sha256": sha256_file(config_path),
        "official_imagenette_validation_accessed": False,
    }
    summary_path = output / "summary.json"
    write_json(summary_path, summary)
    write_json(
        output / "STATE.json",
        {
            "status": "complete",
            "official_imagenette_validation_accessed": False,
            "summary_sha256": sha256_file(summary_path),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = resolve(config["analysis"]["output"])
    if not output.is_dir():
        raise FileNotFoundError(output)
    frozen_config = output / "config_snapshot.yaml"
    if sha256_file(frozen_config) != sha256_file(config_path):
        raise RuntimeError("A1 config differs from formal snapshot")
    if not (output / SCRIPT.name).exists():
        shutil.copy2(SCRIPT, output / SCRIPT.name)
    elif sha256_file(output / SCRIPT.name) != sha256_file(SCRIPT):
        raise RuntimeError("metric script differs from formal snapshot")
    device = torch.device(args.device)
    if args.stage != "summarize" and (
        device.type != "cuda" or not torch.cuda.is_available()
    ):
        raise RuntimeError("A1 metrics require CUDA")
    if args.stage == "full_reference":
        run_full_reference(output, config, device, args.resume)
    elif args.stage == "distribution":
        run_distribution(output, config, device, args.resume)
    else:
        summarize(output, config, config_path)


if __name__ == "__main__":
    main()
