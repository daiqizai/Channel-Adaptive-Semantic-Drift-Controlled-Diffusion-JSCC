#!/usr/bin/env python3
"""Find and visualize low-SNR perceptually plausible semantic-risk cases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from cadsd_jscc.external_common import (  # noqa: E402
    canonical_noise_sha256,
    canonical_standard_normal,
)
from cadsd_jscc.metrics import psnr_per_sample  # noqa: E402
from s32_strong_jscc_external_comparison import (  # noqa: E402
    build_model,
    load_population,
    load_yaml,
    require_sha,
    sha256_file,
)


FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
CLASS_NAMES = {
    0: "tench",
    1: "English springer",
    2: "cassette player",
    3: "chain saw",
    4: "church",
    5: "French horn",
    6: "garbage truck",
    7: "gas pump",
    8: "golf ball",
    9: "parachute",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/low_snr_semantic_drift_visual_audit.yaml"
    )
    parser.add_argument("--stage", choices=("prepare", "finalize"), required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT), size)


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def crop_sgd_tile(grid: Image.Image, index: int, reconstruction: bool) -> Image.Image:
    column = index % 8
    row = index // 8 + (8 if reconstruction else 0)
    left = 2 + column * 258
    top = 2 + row * 258
    return grid.crop((left, top, left + 256, top + 256)).convert("RGB")


def tensor_to_pil(value: torch.Tensor) -> Image.Image:
    array = (
        value.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def percentile_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    for rank, index in enumerate(order, 1):
        result[index] = rank / len(values)
    return result


@torch.no_grad()
def classify_paths(
    model: torch.nn.Module,
    preprocess,
    paths: list[Path],
    batch_size: int,
    device: torch.device,
) -> dict[str, int]:
    output: dict[str, int] = {}
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        images = torch.stack(
            [preprocess(Image.open(path).convert("RGB")) for path in batch_paths]
        ).to(device)
        prediction = model(images).argmax(dim=1).cpu().tolist()
        for path, value in zip(batch_paths, prediction):
            output[str(path)] = int(value)
    return output


@torch.no_grad()
def clip_features(
    model: torch.nn.Module,
    preprocess,
    paths: list[Path],
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        images = torch.stack(
            [preprocess(Image.open(path).convert("RGB")) for path in batch_paths]
        ).to(device)
        features = F.normalize(model.encode_image(images).float(), dim=-1).cpu()
        for path, value in zip(batch_paths, features):
            output[str(path)] = value
    return output


def category_color(category: str) -> tuple[int, int, int]:
    return {
        "pending": (105, 105, 105),
        "faithful": (38, 150, 72),
        "reconstruction_failure_blur_noise": (45, 105, 200),
        "semantic_drift_clear_wrong": (215, 35, 35),
        "uncertain": (170, 80, 185),
    }[category]


def triad_image(
    source: Image.Image,
    strong: Image.Image,
    sgd: Image.Image,
    row: dict[str, Any],
    review: dict[str, str] | None = None,
) -> Image.Image:
    review = review or {
        "s33_category": "pending",
        "sgd_category": "pending",
        "s33_note": "",
        "sgd_note": "",
    }
    canvas = Image.new("RGB", (782, 396), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 781, 395), outline=(85, 85, 85), width=3)
    draw.text(
        (10, 7),
        f"#{int(row['rank']):02d} {row['short_id']} | {row['class_name']} | seed {row['base_seed']} | 1 dB",
        fill="black",
        font=font(16, True),
    )
    draw.text(
        (10, 31),
        f"S33 LPIPS {float(row['s33_lpips']):.3f} pctl {float(row['s33_lpips_percentile']):.2f}; "
        f"SGD {float(row['sgd_lpips']):.3f} pctl {float(row['sgd_lpips_percentile']):.2f}",
        fill="black",
        font=font(12),
    )
    draw.text(
        (10, 50),
        f"anomaly score S33/SGD={row['s33_anomaly_score']}/{row['sgd_anomaly_score']} | "
        f"T_cls fail={row['s33_tcls_failure']}/{row['sgd_tcls_failure']} | "
        f"cross-model votes={row['s33_cross_mismatch_votes']}/{row['sgd_cross_mismatch_votes']}",
        fill="black",
        font=font(11),
    )
    labels = ("Original", "S33 pure JSCC", "SGD diffusion")
    for index, label in enumerate(labels):
        draw.text((index * 258 + 82, 72), label, fill="black", font=font(14, True))
    canvas.paste(source, (4, 94))
    canvas.paste(strong, (262, 94))
    canvas.paste(sgd, (520, 94))
    s33_category = review["s33_category"]
    sgd_category = review["sgd_category"]
    draw.rectangle((260, 92, 519, 353), outline=category_color(s33_category), width=5)
    draw.rectangle((518, 92, 777, 353), outline=category_color(sgd_category), width=5)
    draw.text((264, 358), s33_category, fill=category_color(s33_category), font=font(10, True))
    draw.text((522, 358), sgd_category, fill=category_color(sgd_category), font=font(10, True))
    s33_note = str(review.get("s33_note", ""))[:54]
    sgd_note = str(review.get("sgd_note", ""))[:54]
    if s33_note:
        draw.text((264, 376), s33_note, fill=category_color(s33_category), font=font(9))
    if sgd_note:
        draw.text((522, 376), sgd_note, fill=category_color(sgd_category), font=font(9))
    return canvas


def make_sheet(paths: list[Path], output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    canvas = Image.new("RGB", (2364, 2004), (232, 232, 232))
    for index, image in enumerate(images):
        x = 4 + (index % 3) * 786
        y = 4 + (index // 3) * 400
        canvas.paste(image, (x, y))
    canvas.save(output)


def prepare(config: dict[str, Any], config_path: Path, device: torch.device) -> None:
    if config["protocol"]["official_imagenette_validation_accessed"] is not False:
        raise RuntimeError("official validation must remain sealed")
    output = resolve(config["outputs"]["directory"])
    if output.exists():
        raise FileExistsError(output)
    for directory in (
        output / "all_1db" / "source",
        output / "all_1db" / "s33_strong",
        output / "all_1db" / "sgd_jscc",
        output / "triads",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    snr = float(config["selection"]["snr_db"])
    population_manifest = json.loads(
        resolve(config["inputs"]["population_manifest"]).read_text(encoding="utf-8")
    )
    sample_ids = [str(value) for value in population_manifest["sample_ids"]]
    sample_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    if len(sample_ids) != 64 or len(sample_index) != 64:
        raise RuntimeError("population contract changed")

    sgd_rows: list[dict[str, str]] = []
    sgd_root = resolve(config["inputs"]["sgd_root"])
    for path in sorted(sgd_root.glob("seed_*/per_sample.csv")):
        sgd_rows.extend(
            row for row in read_csv(path) if float(row["snr_db"]) == snr
        )
    strong_rows = [
        row
        for row in read_csv(resolve(config["inputs"]["s33_per_sample"]))
        if float(row["snr_db"]) == snr
    ]
    sgd_by_key = {
        (row["sample_id"], int(row["base_seed"])): row for row in sgd_rows
    }
    strong_by_key = {
        (row["sample_id"], int(row["base_seed"])): row for row in strong_rows
    }
    if set(sgd_by_key) != set(strong_by_key) or len(sgd_by_key) != 192:
        raise RuntimeError("1 dB SGD/S33 key sets do not match")

    # Crop every existing formal SGD result and one source tile per image.
    for base_seed in sorted({key[1] for key in sgd_by_key}):
        grid_path = (
            sgd_root
            / f"seed_{base_seed}"
            / f"snr_{int(snr):02d}_source_sgdjscc.png"
        )
        grid = Image.open(grid_path).convert("RGB")
        if grid.size != (2066, 4130):
            raise RuntimeError(f"unexpected SGD montage dimensions: {grid_path}")
        for sample_id in sample_ids:
            index = sample_index[sample_id]
            name = f"sample_{index:02d}_seed_{base_seed}.png"
            crop_sgd_tile(grid, index, True).save(
                output / "all_1db" / "sgd_jscc" / name
            )
            source_path = output / "all_1db" / "source" / f"sample_{index:02d}.png"
            if not source_path.exists():
                crop_sgd_tile(grid, index, False).save(source_path)

    # Replay all 192 S33 1-dB keys under the exact historical environment/batches.
    s33_config = load_yaml(resolve(config["inputs"]["s33_config"]))
    checkpoint_path = require_sha(
        s33_config["inputs"]["strong_checkpoint"],
        s33_config["inputs"]["strong_checkpoint_sha256"],
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint).to(device).eval().requires_grad_(False)
    samples, _ = load_population(s33_config)
    transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()]
    )
    targets_all = torch.stack(
        [transform(Image.open(item["path"]).convert("RGB")) for item in samples]
    ).to(device)
    batch_size = int(s33_config["runtime"]["batch_size"])
    reference_symbols = int(
        s33_config["rate"]["canonical_noise_reference_real_symbols"]
    )
    latent_shape = (
        model.latent_channels,
        model.image_size // 16,
        model.image_size // 16,
    )
    max_psnr_error = 0.0
    for base_seed in sorted({key[1] for key in strong_by_key}):
        for start in range(0, len(samples), batch_size):
            end = min(start + batch_size, len(samples))
            noises: list[torch.Tensor] = []
            for item in samples[start:end]:
                sample_id = str(item["sample_id"])
                historical = strong_by_key[(sample_id, base_seed)]
                reference_noise = canonical_standard_normal(
                    base_seed, sample_id, snr, reference_symbols
                )
                if canonical_noise_sha256(reference_noise) != historical["canonical_noise_sha256"]:
                    raise RuntimeError(f"canonical noise mismatch: {sample_id}/{base_seed}")
                noises.append(
                    reference_noise[: model.real_symbols].reshape(latent_shape)
                )
            with torch.inference_mode():
                reconstructed, _ = model.forward_with_observation(
                    targets_all[start:end], snr, torch.stack(noises).to(device)
                )
                reconstructed = (
                    torch.floor(reconstructed.clamp(0.0, 1.0) * 255.0) / 255.0
                )
                psnr = psnr_per_sample(reconstructed, targets_all[start:end])
            for offset, item in enumerate(samples[start:end]):
                index = start + offset
                historical = strong_by_key[(str(item["sample_id"]), base_seed)]
                error = abs(float(psnr[offset]) - float(historical["strong_psnr"]))
                max_psnr_error = max(max_psnr_error, error)
                if error > 1e-5:
                    raise RuntimeError(f"S33 replay PSNR mismatch: {error}")
                tensor_to_pil(reconstructed[offset]).save(
                    output
                    / "all_1db"
                    / "s33_strong"
                    / f"sample_{index:02d}_seed_{base_seed}.png"
                )
    del model, checkpoint, targets_all
    torch.cuda.empty_cache()

    source_paths = sorted((output / "all_1db" / "source").glob("*.png"))
    strong_paths = sorted((output / "all_1db" / "s33_strong").glob("*.png"))
    sgd_paths = sorted((output / "all_1db" / "sgd_jscc").glob("*.png"))
    all_paths = source_paths + strong_paths + sgd_paths
    if (len(source_paths), len(strong_paths), len(sgd_paths)) != (64, 192, 192):
        raise RuntimeError("materialized image count mismatch")

    # Offline cross-model top-1 consistency with the corresponding source.
    os.environ["TORCH_HOME"] = str(resolve(config["cross_model_classifiers"]["cache_dir"]))
    import torchvision.models as tv_models

    classifier_predictions: dict[str, dict[str, int]] = {}
    classifier_categories: dict[str, list[str]] = {}
    for model_cfg in config["cross_model_classifiers"]["models"]:
        weights_file = resolve(model_cfg["weights_file"])
        if not weights_file.is_file() or weights_file.stat().st_size < 1024 * 1024:
            raise FileNotFoundError(weights_file)
        weights_enum = getattr(tv_models, str(model_cfg["weights_enum"]))
        weights = getattr(weights_enum, str(model_cfg["weights"]))
        classifier = getattr(tv_models, str(model_cfg["model_name"]))(
            weights=weights
        ).to(device).eval().requires_grad_(False)
        key = str(model_cfg["key"])
        classifier_predictions[key] = classify_paths(
            classifier,
            weights.transforms(),
            all_paths,
            int(config["cross_model_classifiers"]["batch_size"]),
            device,
        )
        classifier_categories[key] = list(weights.meta["categories"])
        del classifier
        torch.cuda.empty_cache()

    # Offline CLIP image-image consistency.
    import open_clip

    clip_checkpoint = resolve(config["clip"]["pretrained_path"])
    if not clip_checkpoint.is_file():
        raise FileNotFoundError(clip_checkpoint)
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        model_name=str(config["clip"]["model_name"]),
        pretrained=str(clip_checkpoint),
        precision=str(config["clip"]["precision"]),
        device=device,
        cache_dir=str(resolve(config["clip"]["cache_dir"])),
        weights_only=bool(config["clip"]["weights_only"]),
    )
    clip_model.eval().requires_grad_(False)
    features = clip_features(
        clip_model,
        clip_preprocess,
        all_paths,
        int(config["clip"]["batch_size"]),
        device,
    )
    del clip_model
    torch.cuda.empty_cache()

    classifier_keys = [
        str(row["key"]) for row in config["cross_model_classifiers"]["models"]
    ]
    rows: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        index = sample_index[sample_id]
        source_path = output / "all_1db" / "source" / f"sample_{index:02d}.png"
        for base_seed in sorted({key[1] for key in sgd_by_key}):
            key = (sample_id, base_seed)
            sgd_row = sgd_by_key[key]
            strong_row = strong_by_key[key]
            strong_path = (
                output / "all_1db" / "s33_strong" / f"sample_{index:02d}_seed_{base_seed}.png"
            )
            sgd_path = (
                output / "all_1db" / "sgd_jscc" / f"sample_{index:02d}_seed_{base_seed}.png"
            )
            record: dict[str, Any] = {
                "sample_id": sample_id,
                "short_id": Path(sample_id).name,
                "population_index": index,
                "class_idx": int(strong_row["class_idx"]),
                "class_name": CLASS_NAMES[int(strong_row["class_idx"])],
                "base_seed": base_seed,
                "snr_db": snr,
                "source_path": str(source_path.relative_to(ROOT)),
                "s33_path": str(strong_path.relative_to(ROOT)),
                "sgd_path": str(sgd_path.relative_to(ROOT)),
                "s33_lpips": float(strong_row["strong_lpips"]),
                "sgd_lpips": float(sgd_row["final_lpips"]),
                "s33_psnr": float(strong_row["strong_psnr"]),
                "sgd_psnr": float(sgd_row["final_psnr"]),
                "s33_tcls_prediction": int(strong_row["strong_prediction"]),
                "sgd_tcls_prediction": int(sgd_row["final_prediction"]),
                "s33_tcls_failure": as_bool(strong_row["strong_failure"]),
                "sgd_tcls_failure": as_bool(sgd_row["final_failure"]),
                "s33_clip_similarity": float(
                    torch.dot(features[str(source_path)], features[str(strong_path)])
                ),
                "sgd_clip_similarity": float(
                    torch.dot(features[str(source_path)], features[str(sgd_path)])
                ),
            }
            s33_votes = 0
            sgd_votes = 0
            for classifier_key in classifier_keys:
                predictions = classifier_predictions[classifier_key]
                categories = classifier_categories[classifier_key]
                source_pred = predictions[str(source_path)]
                strong_pred = predictions[str(strong_path)]
                sgd_pred = predictions[str(sgd_path)]
                s33_mismatch = strong_pred != source_pred
                sgd_mismatch = sgd_pred != source_pred
                s33_votes += int(s33_mismatch)
                sgd_votes += int(sgd_mismatch)
                record[f"{classifier_key}_source_prediction"] = source_pred
                record[f"{classifier_key}_source_label"] = categories[source_pred]
                record[f"{classifier_key}_s33_prediction"] = strong_pred
                record[f"{classifier_key}_s33_label"] = categories[strong_pred]
                record[f"{classifier_key}_s33_mismatch"] = s33_mismatch
                record[f"{classifier_key}_sgd_prediction"] = sgd_pred
                record[f"{classifier_key}_sgd_label"] = categories[sgd_pred]
                record[f"{classifier_key}_sgd_mismatch"] = sgd_mismatch
            record["s33_cross_mismatch_votes"] = s33_votes
            record["sgd_cross_mismatch_votes"] = sgd_votes
            rows.append(record)

    s33_percentiles = percentile_ranks([float(row["s33_lpips"]) for row in rows])
    sgd_percentiles = percentile_ranks([float(row["sgd_lpips"]) for row in rows])
    clip_quantile = float(config["selection"]["clip_low_consistency_quantile"])
    s33_clip_threshold = float(
        np.quantile([float(row["s33_clip_similarity"]) for row in rows], clip_quantile)
    )
    sgd_clip_threshold = float(
        np.quantile([float(row["sgd_clip_similarity"]) for row in rows], clip_quantile)
    )
    for index, row in enumerate(rows):
        row["s33_lpips_percentile"] = s33_percentiles[index]
        row["sgd_lpips_percentile"] = sgd_percentiles[index]
        row["s33_clip_low_consistency"] = (
            float(row["s33_clip_similarity"]) <= s33_clip_threshold
        )
        row["sgd_clip_low_consistency"] = (
            float(row["sgd_clip_similarity"]) <= sgd_clip_threshold
        )
        for method in ("s33", "sgd"):
            row[f"{method}_anomaly_score"] = (
                int(row[f"{method}_tcls_failure"])
                * int(config["selection"]["t_cls_failure_weight"])
                + int(row[f"{method}_cross_mismatch_votes"])
                * int(config["selection"]["cross_model_mismatch_weight"])
                + int(row[f"{method}_clip_low_consistency"])
                * int(config["selection"]["clip_low_consistency_weight"])
            )
        row["max_anomaly_score"] = max(
            int(row["s33_anomaly_score"]), int(row["sgd_anomaly_score"])
        )
        row["method_anomaly_contrast"] = abs(
            int(row["s33_anomaly_score"]) - int(row["sgd_anomaly_score"])
        )
        row["joint_anomaly_score"] = int(row["s33_anomaly_score"]) + int(
            row["sgd_anomaly_score"]
        )
        row["mean_lpips_percentile"] = (
            float(row["s33_lpips_percentile"])
            + float(row["sgd_lpips_percentile"])
        ) / 2.0

    write_csv(output / "all_1db_semantic_signals.csv", rows)
    best_max = float(config["selection"]["best_method_lpips_percentile_max"])
    worst_max = float(config["selection"]["worst_method_lpips_percentile_max"])
    candidates = [
        row
        for row in rows
        if int(row["max_anomaly_score"]) > 0
        and min(
            float(row["s33_lpips_percentile"]),
            float(row["sgd_lpips_percentile"]),
        )
        <= best_max
        and max(
            float(row["s33_lpips_percentile"]),
            float(row["sgd_lpips_percentile"]),
        )
        <= worst_max
    ]
    candidates.sort(
        key=lambda row: (
            -int(row["max_anomaly_score"]),
            -int(row["method_anomaly_contrast"]),
            -int(row["joint_anomaly_score"]),
            float(row["mean_lpips_percentile"]),
            str(row["sample_id"]),
            int(row["base_seed"]),
        )
    )
    # First reserve distinct sources where the supervised T_cls itself fails,
    # provided neither method is in the worst LPIPS decile.  This prevents a
    # cross-model contrast score from choosing a different, T_cls-correct seed
    # of the same source and hiding the actual supervised anomaly.
    mandatory_max = float(
        config["selection"]["mandatory_t_cls_failure_lpips_percentile_max"]
    )
    mandatory = [
        row
        for row in rows
        if (bool(row["s33_tcls_failure"]) or bool(row["sgd_tcls_failure"]))
        and max(
            float(row["s33_lpips_percentile"]),
            float(row["sgd_lpips_percentile"]),
        )
        <= mandatory_max
    ]
    mandatory.sort(
        key=lambda row: (
            -(
                int(bool(row["s33_tcls_failure"]))
                + int(bool(row["sgd_tcls_failure"]))
            ),
            -int(row["max_anomaly_score"]),
            -int(row["joint_anomaly_score"]),
            float(row["mean_lpips_percentile"]),
            str(row["sample_id"]),
            int(row["base_seed"]),
        )
    )
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in mandatory:
        if str(row["sample_id"]) in seen:
            continue
        seen.add(str(row["sample_id"]))
        chosen = dict(row)
        chosen["selection_stratum"] = "mandatory_t_cls_failure"
        selected.append(chosen)
    for row in candidates:
        if str(row["sample_id"]) in seen:
            continue
        seen.add(str(row["sample_id"]))
        chosen = dict(row)
        chosen["selection_stratum"] = "cross_model_clip_risk_fill"
        selected.append(chosen)
        if len(selected) == int(config["selection"]["count"]):
            break
    if len(selected) != int(config["selection"]["count"]):
        raise RuntimeError(
            f"semantic/perceptual filter found only {len(selected)} unique sources"
        )
    for rank, row in enumerate(selected, 1):
        row["rank"] = rank
    write_csv(output / "selection.csv", selected)
    for row in selected:
        source = Image.open(resolve(row["source_path"])).convert("RGB")
        strong = Image.open(resolve(row["s33_path"])).convert("RGB")
        sgd = Image.open(resolve(row["sgd_path"])).convert("RGB")
        triad_image(source, strong, sgd, row).save(
            output / "triads" / f"rank_{int(row['rank']):02d}.png"
        )
    make_sheet(
        sorted((output / "triads").glob("rank_*.png")),
        output / "low_snr_semantic_risk_top15_draft.png",
    )
    write_json(
        output / "manual_review.json",
        {
            "instructions": (
                "For each reconstruction choose faithful, "
                "reconstruction_failure_blur_noise, semantic_drift_clear_wrong, or uncertain."
            ),
            "items": [
                {
                    "rank": int(row["rank"]),
                    "sample_id": row["sample_id"],
                    "base_seed": int(row["base_seed"]),
                    "s33_category": "pending",
                    "s33_note": "",
                    "sgd_category": "pending",
                    "sgd_note": "",
                }
                for row in selected
            ],
        },
    )
    write_json(
        output / "audit.json",
        {
            "analysis_id": config["analysis_id"],
            "config": str(config_path.relative_to(ROOT)),
            "config_sha256": file_sha(config_path),
            "script": str(Path(__file__).resolve().relative_to(ROOT)),
            "script_sha256": file_sha(Path(__file__)),
            "s33_checkpoint_sha256": sha256_file(checkpoint_path),
            "s33_replay_max_abs_psnr_error_db": max_psnr_error,
            "one_db_keys": len(rows),
            "candidate_keys_after_filter": len(candidates),
            "mandatory_t_cls_keys": len(mandatory),
            "mandatory_t_cls_unique_sources_selected": sum(
                row["selection_stratum"] == "mandatory_t_cls_failure"
                for row in selected
            ),
            "selected_unique_sources": len(selected),
            "clip_low_thresholds": {
                "s33": s33_clip_threshold,
                "sgd": sgd_clip_threshold,
            },
            "new_training": False,
            "official_imagenette_validation_accessed": False,
            "fair_ranking_forbidden": True,
        },
    )
    print(
        json.dumps(
            {
                "status": "PREPARED",
                "output": str(output.relative_to(ROOT)),
                "candidates": len(candidates),
                "selected": len(selected),
                "s33_replay_max_abs_psnr_error_db": max_psnr_error,
            },
            indent=2,
        )
    )


def finalize(config: dict[str, Any]) -> None:
    output = resolve(config["outputs"]["directory"])
    selection = read_csv(output / "selection.csv")
    review_data = json.loads(
        (output / "manual_review.json").read_text(encoding="utf-8")
    )
    reviews = {int(row["rank"]): row for row in review_data["items"]}
    allowed = set(config["manual_review"]["categories"])
    for row in reviews.values():
        if row["s33_category"] not in allowed or row["sgd_category"] not in allowed:
            raise RuntimeError("manual review contains pending/invalid category")
    for row in selection:
        rank = int(row["rank"])
        source = Image.open(resolve(row["source_path"])).convert("RGB")
        strong = Image.open(resolve(row["s33_path"])).convert("RGB")
        sgd = Image.open(resolve(row["sgd_path"])).convert("RGB")
        triad_image(source, strong, sgd, row, reviews[rank]).save(
            output / "triads" / f"rank_{rank:02d}_reviewed.png"
        )
    final_path = output / "low_snr_semantic_risk_top15_reviewed.png"
    if final_path.exists():
        raise FileExistsError(final_path)
    make_sheet(sorted((output / "triads").glob("rank_*_reviewed.png")), final_path)
    counts: dict[str, dict[str, int]] = {}
    for method in ("s33", "sgd"):
        counts[method] = {
            category: sum(
                row[f"{method}_category"] == category for row in reviews.values()
            )
            for category in sorted(allowed)
        }
    write_json(output / "manual_review_summary.json", {"counts": counts})
    print(json.dumps({"status": "FINALIZED", "counts": counts}, indent=2))


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.stage == "prepare":
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        prepare(config, config_path, device)
    else:
        finalize(config)


if __name__ == "__main__":
    main()
