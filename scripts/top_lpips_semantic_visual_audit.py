#!/usr/bin/env python3
"""Prepare and finalize a frozen top-LPIPS semantic visual audit.

This is a post-hoc visualization utility.  It never trains a model.  SGD-JSCC
tiles are cropped from the already saved formal montages.  Missing S33 tiles
are deterministically replayed from the frozen checkpoint and canonical noise,
then checked against the historical per-sample PSNR.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
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
FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/top_lpips_semantic_visual_audit.yaml"
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(
        str(FONT_BOLD_PATH if bold else FONT_PATH), size=size
    )


def top_unique(
    rows: list[dict[str, str]], metric: str, count: int
) -> list[dict[str, str]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row[metric]),
            str(row["sample_id"]),
            int(row["base_seed"]),
            float(row["snr_db"]),
        ),
    )
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in ordered:
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            continue
        seen.add(sample_id)
        selected.append(row)
        if len(selected) == count:
            return selected
    raise RuntimeError(f"only found {len(selected)} distinct samples")


def crop_grid_tile(grid: Image.Image, index: int, *, reconstruction: bool) -> Image.Image:
    # Formal SGD montage is torchvision.save_image(..., nrow=8): 8 source rows
    # followed by 8 reconstruction rows, tile=256 and padding=2.
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


def pair_image(
    source: Image.Image,
    reconstruction: Image.Image,
    row: dict[str, Any],
    *,
    review: dict[str, str] | None = None,
) -> Image.Image:
    review = review or {}
    category = review.get("category", "pending")
    colors = {
        "pending": (110, 110, 110),
        "faithful": (37, 150, 72),
        "minor_structure_change": (224, 139, 27),
        "semantic_mismatch": (210, 35, 35),
        "uncertain": (175, 90, 190),
    }
    color = colors[category]
    canvas = Image.new("RGB", (526, 350), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 525, 349), outline=color, width=5)
    draw.text((10, 8), f"#{int(row['rank']):02d}  {row['method']}", fill="black", font=font(17, bold=True))
    draw.text(
        (10, 31),
        f"{row['short_id']} | {row['class_name']} | SNR {float(row['snr_db']):g} dB",
        fill="black",
        font=font(13),
    )
    draw.text(
        (10, 50),
        f"LPIPS {float(row['lpips']):.5f} | PSNR {float(row['psnr']):.2f} dB | T_cls failure: {row['auto_failure']}",
        fill="black",
        font=font(12),
    )
    if category != "pending":
        draw.text((385, 9), category, fill=color, font=font(12, bold=True))
    draw.text((105, 76), "Original", fill="black", font=font(14, bold=True))
    draw.text((368, 76), "Reconstruction", fill="black", font=font(14, bold=True))
    canvas.paste(source, (6, 94))
    canvas.paste(reconstruction, (264, 94))
    note = review.get("note", "").strip()
    if note:
        draw.rectangle((6, 320, 520, 344), fill=(255, 245, 230))
        draw.text((10, 325), note[:76], fill=color, font=font(11, bold=True))
    return canvas


def contact_sheet(pair_paths: list[Path], output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in pair_paths]
    canvas = Image.new("RGB", (1608, 1770), (235, 235, 235))
    for index, image in enumerate(images):
        x = 6 + (index % 3) * 534
        y = 6 + (index // 3) * 352
        canvas.paste(image, (x, y))
    canvas.save(output)


def load_selection(output: Path) -> list[dict[str, str]]:
    rows = read_csv(output / "selection.csv")
    if len(rows) != 30:
        raise RuntimeError("selection.csv must contain 30 rows")
    return rows


def prepare(config: dict[str, Any], config_path: Path, device: torch.device) -> None:
    if config["protocol"]["official_imagenette_validation_accessed"] is not False:
        raise RuntimeError("official validation must remain sealed")
    output = resolve(config["outputs"]["directory"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / "pairs" / "sgd_jscc").mkdir(parents=True)
    (output / "pairs" / "s33_strong").mkdir(parents=True)

    population_manifest_path = resolve(config["inputs"]["population_manifest"])
    population_manifest = json.loads(
        population_manifest_path.read_text(encoding="utf-8")
    )
    sample_ids = [str(value) for value in population_manifest["sample_ids"]]
    sample_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    if len(sample_ids) != 64 or len(sample_index) != 64:
        raise RuntimeError("frozen population is not 64 unique samples")

    sgd_root = resolve(config["inputs"]["sgd_root"])
    sgd_rows: list[dict[str, str]] = []
    for path in sorted(sgd_root.glob("seed_*/per_sample.csv")):
        sgd_rows.extend(read_csv(path))
    strong_rows = read_csv(resolve(config["inputs"]["s33_per_sample"]))
    count = int(config["selection"]["count_per_method"])
    selected_sgd = top_unique(sgd_rows, "final_lpips", count)
    selected_strong = top_unique(strong_rows, "strong_lpips", count)

    selection: list[dict[str, Any]] = []
    for method, rows, keys in (
        (
            "sgd_jscc",
            selected_sgd,
            ("final_lpips", "final_psnr", "final_prediction", "final_failure"),
        ),
        (
            "s33_strong",
            selected_strong,
            ("strong_lpips", "strong_psnr", "strong_prediction", "strong_failure"),
        ),
    ):
        for rank, row in enumerate(rows, 1):
            lpips_key, psnr_key, prediction_key, failure_key = keys
            selection.append(
                {
                    "method": method,
                    "rank": rank,
                    "sample_id": row["sample_id"],
                    "short_id": Path(row["sample_id"]).name,
                    "population_index": sample_index[row["sample_id"]],
                    "class_idx": int(row["class_idx"]),
                    "class_name": CLASS_NAMES[int(row["class_idx"])],
                    "base_seed": int(row["base_seed"]),
                    "snr_db": float(row["snr_db"]),
                    "lpips": float(row[lpips_key]),
                    "psnr": float(row[psnr_key]),
                    "prediction": int(row[prediction_key]),
                    "auto_failure": str(row[failure_key]).lower() == "true",
                    "canonical_noise_sha256": row["canonical_noise_sha256"],
                    "reconstruction_source": (
                        "existing_formal_montage_crop"
                        if method == "sgd_jscc"
                        else "frozen_checkpoint_canonical_noise_replay"
                    ),
                }
            )
    write_csv(output / "selection.csv", selection)

    # SGD-JSCC: exact crops from the formal 64-source + 64-reconstruction grids.
    for row in [value for value in selection if value["method"] == "sgd_jscc"]:
        montage = (
            sgd_root
            / f"seed_{int(row['base_seed'])}"
            / f"snr_{int(float(row['snr_db'])):02d}_source_sgdjscc.png"
        )
        grid = Image.open(montage).convert("RGB")
        if grid.size != (2066, 4130):
            raise RuntimeError(f"unexpected formal SGD montage dimensions: {montage}")
        source = crop_grid_tile(grid, int(row["population_index"]), reconstruction=False)
        reconstruction = crop_grid_tile(
            grid, int(row["population_index"]), reconstruction=True
        )
        pair = pair_image(source, reconstruction, row)
        pair.save(output / "pairs" / "sgd_jscc" / f"rank_{int(row['rank']):02d}.png")

    # S33: replay the exact formal batch chunk for every selected key.
    s33_config_path = resolve(config["inputs"]["s33_config"])
    s33_config = load_yaml(s33_config_path)
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
    targets_cpu = [
        transform(Image.open(item["path"]).convert("RGB")) for item in samples
    ]
    targets_all = torch.stack(targets_cpu).to(device)
    formal_rows = {
        (row["sample_id"], int(row["base_seed"]), float(row["snr_db"])): row
        for row in strong_rows
    }
    selected_strong_rows = [
        value for value in selection if value["method"] == "s33_strong"
    ]
    grouped: defaultdict[tuple[int, float, int], list[dict[str, Any]]] = defaultdict(list)
    formal_batch_size = int(s33_config["runtime"]["batch_size"])
    for row in selected_strong_rows:
        batch_start = int(row["population_index"]) // formal_batch_size * formal_batch_size
        grouped[(int(row["base_seed"]), float(row["snr_db"]), batch_start)].append(row)
    latent_shape = (
        model.latent_channels,
        model.image_size // 16,
        model.image_size // 16,
    )
    reference_symbols = int(
        s33_config["rate"].get(
            "canonical_noise_reference_real_symbols", model.real_symbols
        )
    )
    max_psnr_error = 0.0
    replay_tolerance = float(
        config["reconstruction_source"]["s33_replay_psnr_abs_tolerance_db"]
    )
    replay_rows = 0
    for (base_seed, snr, start), wanted in sorted(grouped.items()):
        end = min(start + formal_batch_size, len(samples))
        noises: list[torch.Tensor] = []
        for item in samples[start:end]:
            sample_id = str(item["sample_id"])
            historical = formal_rows[(sample_id, base_seed, snr)]
            reference_noise = canonical_standard_normal(
                base_seed, sample_id, snr, reference_symbols
            )
            if canonical_noise_sha256(reference_noise) != historical["canonical_noise_sha256"]:
                raise RuntimeError(f"canonical noise mismatch: {sample_id}")
            noises.append(reference_noise[: model.real_symbols].reshape(latent_shape))
        noise_batch = torch.stack(noises).to(device)
        with torch.inference_mode():
            value, _ = model.forward_with_observation(
                targets_all[start:end], snr, noise_batch
            )
            value = torch.floor(value.clamp(0.0, 1.0) * 255.0) / 255.0
            psnr = psnr_per_sample(value, targets_all[start:end])
        for row in wanted:
            offset = int(row["population_index"]) - start
            error = abs(float(psnr[offset]) - float(row["psnr"]))
            max_psnr_error = max(max_psnr_error, error)
            if error > replay_tolerance:
                raise RuntimeError(
                    f"S33 replay PSNR mismatch {row['sample_id']}: {error:.9g} dB"
                )
            source = tensor_to_pil(targets_all[int(row["population_index"])])
            reconstruction = tensor_to_pil(value[offset])
            pair = pair_image(source, reconstruction, row)
            pair.save(
                output / "pairs" / "s33_strong" / f"rank_{int(row['rank']):02d}.png"
            )
            replay_rows += 1
    if replay_rows != count:
        raise RuntimeError(f"expected {count} S33 replay rows, got {replay_rows}")

    for method in ("sgd_jscc", "s33_strong"):
        paths = sorted((output / "pairs" / method).glob("rank_*.png"))
        contact_sheet(paths, output / f"{method}_top15_draft.png")

    manual_review = {
        "instructions": (
            "Fill category with faithful, minor_structure_change, "
            "semantic_mismatch, or uncertain; add a concise note."
        ),
        "items": [
            {
                "method": row["method"],
                "rank": int(row["rank"]),
                "category": "pending",
                "note": "",
            }
            for row in selection
        ],
    }
    write_json(output / "manual_review.json", manual_review)
    write_json(
        output / "audit.json",
        {
            "analysis_id": config["analysis_id"],
            "config": str(config_path.relative_to(ROOT)),
            "config_sha256": sha256_file(config_path),
            "script": str(Path(__file__).resolve().relative_to(ROOT)),
            "script_sha256": sha256_file(Path(__file__)),
            "selection_rule": config["selection"]["rule"],
            "count_per_method": count,
            "sgd_reconstruction_source": "existing formal montage crops",
            "s33_reconstruction_source": "frozen checkpoint canonical-noise replay",
            "s33_checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "s33_checkpoint_sha256": sha256_file(checkpoint_path),
            "s33_replay_max_abs_psnr_error_db": max_psnr_error,
            "s33_replay_psnr_abs_tolerance_db": replay_tolerance,
            "new_training": False,
            "official_imagenette_validation_accessed": False,
        },
    )
    print(
        json.dumps(
            {
                "status": "PREPARED",
                "output": str(output.relative_to(ROOT)),
                "selected_per_method": count,
                "s33_replay_max_abs_psnr_error_db": max_psnr_error,
            },
            indent=2,
        )
    )


def finalize(config: dict[str, Any]) -> None:
    output = resolve(config["outputs"]["directory"])
    if not output.is_dir():
        raise FileNotFoundError(output)
    review_path = output / "manual_review.json"
    review_data = json.loads(review_path.read_text(encoding="utf-8"))
    reviews = {
        (str(row["method"]), int(row["rank"])): row
        for row in review_data["items"]
    }
    allowed = set(config["manual_review"]["categories"])
    if any(row["category"] not in allowed for row in reviews.values()):
        raise RuntimeError("manual review still contains pending or invalid categories")
    selection = load_selection(output)
    for row in selection:
        method = row["method"]
        rank = int(row["rank"])
        current = Image.open(output / "pairs" / method / f"rank_{rank:02d}.png")
        source = current.crop((6, 94, 262, 350))
        reconstruction = current.crop((264, 94, 520, 350))
        reviewed = pair_image(source, reconstruction, row, review=reviews[(method, rank)])
        reviewed.save(output / "pairs" / method / f"rank_{rank:02d}_reviewed.png")
    for method in ("sgd_jscc", "s33_strong"):
        paths = sorted((output / "pairs" / method).glob("rank_*_reviewed.png"))
        final_path = output / f"{method}_top15_reviewed.png"
        if final_path.exists():
            raise FileExistsError(final_path)
        contact_sheet(paths, final_path)

    counts: dict[str, dict[str, int]] = {}
    flagged: list[dict[str, Any]] = []
    for method in ("sgd_jscc", "s33_strong"):
        method_reviews = [
            row for (row_method, _), row in reviews.items() if row_method == method
        ]
        counts[method] = {
            category: sum(row["category"] == category for row in method_reviews)
            for category in sorted(allowed)
        }
        for row in method_reviews:
            if row["category"] != "faithful":
                flagged.append(dict(row))
    result_path = output / "manual_review_summary.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    write_json(result_path, {"counts": counts, "flagged": flagged})
    print(json.dumps({"status": "FINALIZED", "counts": counts}, indent=2))


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.stage == "prepare":
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        prepare(config, config_path, device)
    else:
        finalize(config)


if __name__ == "__main__":
    main()
