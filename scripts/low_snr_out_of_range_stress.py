#!/usr/bin/env python3
"""Replay frozen S33/SGD-JSCC on selected sources at -3/-5 dB."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
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
from cadsd_jscc.metrics import ms_ssim_per_sample, psnr_per_sample  # noqa: E402
from pc_imagenette_supervised_audit import (  # noqa: E402
    evaluate_probabilities,
    load_scratch_classifier,
)
from s32_strong_jscc_external_comparison import (  # noqa: E402
    build_model,
    evaluator_config,
    load_population,
    load_yaml,
    require_sha,
    sha256_file,
)
from s5_residual_refiner_pilot import try_load_lpips  # noqa: E402


FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/low_snr_out_of_range_stress.yaml"
    )
    parser.add_argument(
        "--stage", choices=("prepare-s33", "assemble", "finalize"), required=True
    )
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


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT), size)


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


def crop_grid_tile(grid: Image.Image, global_index: int, nrow: int = 8) -> Image.Image:
    column = global_index % nrow
    row = global_index // nrow
    left = 2 + column * 258
    top = 2 + row * 258
    return grid.crop((left, top, left + 256, top + 256)).convert("RGB")


def category_color(category: str) -> tuple[int, int, int]:
    return {
        "pending": (105, 105, 105),
        "faithful": (38, 150, 72),
        "reconstruction_failure_blur_noise": (45, 105, 200),
        "semantic_drift_clear_wrong": (215, 35, 35),
        "uncertain": (170, 80, 185),
    }[category]


def stress_triad(
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
        f"#{int(row['rank']):02d} {row['short_id']} | stress {float(row['snr_db']):g} dB | seed {row['base_seed']}",
        fill="black",
        font=font(16, True),
    )
    draw.text(
        (10, 32),
        f"S33 LPIPS {float(row['s33_lpips']):.3f}, PSNR {float(row['s33_psnr']):.1f}, fail {row['s33_failure']} | "
        f"SGD LPIPS {float(row['sgd_lpips']):.3f}, PSNR {float(row['sgd_psnr']):.1f}, fail {row['sgd_failure']}",
        fill="black",
        font=font(11),
    )
    for index, label in enumerate(("Original", "S33 pure JSCC", "SGD diffusion")):
        draw.text((index * 258 + 82, 70), label, fill="black", font=font(14, True))
    canvas.paste(source, (4, 92))
    canvas.paste(strong, (262, 92))
    canvas.paste(sgd, (520, 92))
    for method, x in (("s33", 260), ("sgd", 518)):
        category = review[f"{method}_category"]
        color = category_color(category)
        draw.rectangle((x, 90, x + 259, 351), outline=color, width=5)
        draw.text((x + 4, 356), category, fill=color, font=font(10, True))
        note = str(review.get(f"{method}_note", ""))[:54]
        if note:
            draw.text((x + 4, 374), note, fill=color, font=font(9))
    return canvas


def make_sheet(paths: list[Path], output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    canvas = Image.new("RGB", (2364, 2004), (232, 232, 232))
    for index, image in enumerate(images):
        canvas.paste(image, (4 + (index % 3) * 786, 4 + (index // 3) * 400))
    canvas.save(output)


def prepare_s33(config: dict[str, Any], config_path: Path, device: torch.device) -> None:
    if config["protocol"]["official_imagenette_validation_accessed"] is not False:
        raise RuntimeError("official validation must remain sealed")
    output = resolve(config["outputs"]["directory"])
    if output.exists():
        raise FileExistsError(output)
    (output / "s33_reference").mkdir(parents=True)
    (output / "s33_images").mkdir()
    (output / "sgd_configs").mkdir()
    (output / "sgd_images").mkdir()
    (output / "triads").mkdir()

    selection_path = resolve(config["inputs"]["selection"])
    selection = read_csv(selection_path)
    if len(selection) != int(config["stress"]["source_count"]):
        raise RuntimeError("frozen selection count changed")
    if len({row["sample_id"] for row in selection}) != len(selection):
        raise RuntimeError("stress selection must have unique source images")

    s33_config = load_yaml(resolve(config["inputs"]["s33_config"]))
    checkpoint_path = require_sha(
        s33_config["inputs"]["strong_checkpoint"],
        s33_config["inputs"]["strong_checkpoint_sha256"],
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint).to(device).eval().requires_grad_(False)
    population, classes = load_population(s33_config)
    by_id = {str(item["sample_id"]): item for item in population}
    chosen = [by_id[row["sample_id"]] for row in selection]
    transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()]
    )
    targets = torch.stack(
        [transform(Image.open(item["path"]).convert("RGB")) for item in chosen]
    ).to(device)
    evaluator, temperature = load_scratch_classifier(
        str(
            require_sha(
                s33_config["inputs"]["t_cls_checkpoint"],
                s33_config["inputs"]["t_cls_checkpoint_sha256"],
            )
        ),
        classes,
        device,
        str(s33_config["evaluator"]["expected_role"]),
    )
    lpips_model, lpips_error = try_load_lpips(device, resolve("outputs/cache"))
    if lpips_model is None:
        raise RuntimeError(lpips_error)
    lpips_model.eval().requires_grad_(False)
    reference_symbols = int(config["stress"]["canonical_noise_reference_real_symbols"])
    base_seed = int(config["stress"]["base_seed"])
    latent_shape = (
        model.latent_channels,
        model.image_size // 16,
        model.image_size // 16,
    )
    rows: list[dict[str, Any]] = []
    eval_cfg = evaluator_config(s33_config)
    for snr in map(float, config["stress"]["snrs_db"]):
        noises = []
        noise_shas = []
        for item in chosen:
            noise = canonical_standard_normal(
                base_seed, str(item["sample_id"]), snr, reference_symbols
            )
            noise_shas.append(canonical_noise_sha256(noise))
            noises.append(noise[: model.real_symbols].reshape(latent_shape))
        with torch.inference_mode():
            reconstructed, _ = model.forward_with_observation(
                targets, snr, torch.stack(noises).to(device)
            )
            reconstructed = torch.floor(reconstructed.clamp(0.0, 1.0) * 255.0) / 255.0
            psnr = psnr_per_sample(reconstructed, targets)
            ms_ssim = ms_ssim_per_sample(reconstructed, targets)
            lpips = lpips_model(
                reconstructed * 2.0 - 1.0, targets * 2.0 - 1.0
            ).flatten()
            probabilities = evaluate_probabilities(
                evaluator, temperature, reconstructed, eval_cfg
            )
            predictions = probabilities.argmax(dim=1)
        for index, (selection_row, item) in enumerate(zip(selection, chosen)):
            label = int(item["class_idx"])
            tensor_to_pil(reconstructed[index]).save(
                output
                / "s33_images"
                / f"rank_{int(selection_row['rank']):02d}_snr_{int(snr)}.png"
            )
            rows.append(
                {
                    "sample_id": item["sample_id"],
                    "snr_db": snr,
                    "base_seed": base_seed,
                    "canonical_noise_sha256": noise_shas[index],
                    "deepjscc_prediction": int(predictions[index]),
                    "deepjscc_correct": int(predictions[index]) == label,
                    "deepjscc_psnr": float(psnr[index]),
                    "deepjscc_ms_ssim": float(ms_ssim[index]),
                    "deepjscc_lpips": float(lpips[index]),
                }
            )
    write_csv(output / "s33_reference" / "per_sample.csv", rows)

    # Materialize a 15-source population config for the unchanged official SGD adapter.
    population_reference = load_yaml(resolve(config["inputs"]["population_reference"]))
    frozen_by_id = {
        str(row["sample_id"]): row
        for row in population_reference["population"]["samples"]
    }
    population_reference["analysis_id"] = config["analysis_id"] + "-SGD-POPULATION"
    population_reference["status"] = "preregistered_before_any_pilot_method_output"
    population_reference["created_at"] = config["created_at"]
    population_reference["population"]["expected_sample_count"] = len(selection)
    population_reference["population"]["selection_rule"] = (
        "frozen_from_ANALYSIS_LOW_SNR_SEMANTIC_DRIFT_AUDIT_003_before_stress"
    )
    population_reference["population"]["samples"] = [
        frozen_by_id[row["sample_id"]] for row in selection
    ]
    population_path = output / "sgd_configs" / "population_reference.yaml"
    population_path.write_text(
        yaml.safe_dump(population_reference, sort_keys=False), encoding="utf-8"
    )

    sgd_config = load_yaml(resolve(config["inputs"]["sgd_resolved_template"]))
    sgd_config["analysis_id"] = config["analysis_id"] + "-SGD"
    sgd_config["created_at"] = config["created_at"]
    sgd_config["population_reference_config"] = str(population_path.relative_to(ROOT))
    sgd_config["channel"]["snrs_db"] = [
        float(value) for value in config["stress"]["snrs_db"]
    ]
    sgd_config["channel"]["base_seed"] = base_seed
    sgd_config["outputs"]["root"] = str(output.relative_to(ROOT))
    sgd_config["outputs"]["deepjscc"] = str(
        (output / "s33_reference").relative_to(ROOT)
    )
    sgd_config["outputs"]["sgd_jscc"] = str((output / "sgd_run").relative_to(ROOT))
    sgd_path = output / "sgd_configs" / "sgd_stress_resolved.yaml"
    sgd_path.write_text(yaml.safe_dump(sgd_config, sort_keys=False), encoding="utf-8")
    shutil.copy2(config_path, output / "config_snapshot.yaml")
    write_json(
        output / "prepare_audit.json",
        {
            "selection": str(selection_path.relative_to(ROOT)),
            "selection_sha256": file_sha(selection_path),
            "source_count": len(selection),
            "stress_snrs_db": config["stress"]["snrs_db"],
            "base_seed": base_seed,
            "s33_checkpoint_sha256": sha256_file(checkpoint_path),
            "new_training": False,
            "official_imagenette_validation_accessed": False,
        },
    )
    print(
        json.dumps(
            {
                "status": "S33_PREPARED",
                "sgd_config": str(sgd_path.relative_to(ROOT)),
                "next_command": (
                    ".venv-sgdjscc/bin/python scripts/external_sgdjscc_common_pilot.py "
                    f"--config {sgd_path.relative_to(ROOT)} --run"
                ),
            },
            indent=2,
        )
    )


def assemble(config: dict[str, Any]) -> None:
    output = resolve(config["outputs"]["directory"])
    selection = read_csv(resolve(config["inputs"]["selection"]))
    s33_rows = read_csv(output / "s33_reference" / "per_sample.csv")
    sgd_rows = read_csv(output / "sgd_run" / "per_sample.csv")
    s33_by_key = {(row["sample_id"], float(row["snr_db"])): row for row in s33_rows}
    sgd_by_key = {(row["sample_id"], float(row["snr_db"])): row for row in sgd_rows}
    expected = len(selection) * len(config["stress"]["snrs_db"])
    if set(s33_by_key) != set(sgd_by_key) or len(sgd_by_key) != expected:
        raise RuntimeError("stress SGD/S33 key mismatch")
    rows: list[dict[str, Any]] = []
    for snr in map(float, config["stress"]["snrs_db"]):
        grid_path = output / "sgd_run" / f"snr_{int(snr):02d}_source_sgdjscc.png"
        grid = Image.open(grid_path).convert("RGB")
        for index, selection_row in enumerate(selection):
            key = (selection_row["sample_id"], snr)
            s33 = s33_by_key[key]
            sgd = sgd_by_key[key]
            rank = int(selection_row["rank"])
            sgd_image_path = output / "sgd_images" / f"rank_{rank:02d}_snr_{int(snr)}.png"
            crop_grid_tile(grid, len(selection) + index).save(sgd_image_path)
            row = {
                "rank": rank,
                "sample_id": selection_row["sample_id"],
                "short_id": selection_row["short_id"],
                "source_path": selection_row["source_path"],
                "snr_db": snr,
                "base_seed": int(config["stress"]["base_seed"]),
                "s33_path": str(
                    (
                        output
                        / "s33_images"
                        / f"rank_{rank:02d}_snr_{int(snr)}.png"
                    ).relative_to(ROOT)
                ),
                "sgd_path": str(sgd_image_path.relative_to(ROOT)),
                "s33_psnr": float(s33["deepjscc_psnr"]),
                "s33_lpips": float(s33["deepjscc_lpips"]),
                "s33_failure": not (str(s33["deepjscc_correct"]).lower() == "true"),
                "sgd_psnr": float(sgd["final_psnr"]),
                "sgd_lpips": float(sgd["final_lpips"]),
                "sgd_failure": str(sgd["final_failure"]).lower() == "true",
            }
            rows.append(row)
            stress_triad(
                Image.open(resolve(row["source_path"])).convert("RGB"),
                Image.open(resolve(row["s33_path"])).convert("RGB"),
                Image.open(resolve(row["sgd_path"])).convert("RGB"),
                row,
            ).save(output / "triads" / f"snr_{int(snr)}_rank_{rank:02d}.png")
        make_sheet(
            sorted((output / "triads").glob(f"snr_{int(snr)}_rank_??.png")),
            output / f"stress_snr_{int(snr)}_draft.png",
        )
    write_csv(output / "stress_rows.csv", rows)
    write_json(
        output / "manual_review.json",
        {
            "items": [
                {
                    "rank": int(row["rank"]),
                    "snr_db": float(row["snr_db"]),
                    "sample_id": row["sample_id"],
                    "s33_category": "pending",
                    "s33_note": "",
                    "sgd_category": "pending",
                    "sgd_note": "",
                }
                for row in rows
            ]
        },
    )
    print(json.dumps({"status": "ASSEMBLED", "rows": len(rows)}, indent=2))


def finalize(config: dict[str, Any]) -> None:
    output = resolve(config["outputs"]["directory"])
    rows = read_csv(output / "stress_rows.csv")
    review_data = json.loads((output / "manual_review.json").read_text(encoding="utf-8"))
    reviews = {
        (int(row["rank"]), float(row["snr_db"])): row
        for row in review_data["items"]
    }
    allowed = set(config["manual_review"]["categories"])
    for review in reviews.values():
        if review["s33_category"] not in allowed or review["sgd_category"] not in allowed:
            raise RuntimeError("manual review contains pending/invalid category")
    for row in rows:
        key = (int(row["rank"]), float(row["snr_db"]))
        stress_triad(
            Image.open(resolve(row["source_path"])).convert("RGB"),
            Image.open(resolve(row["s33_path"])).convert("RGB"),
            Image.open(resolve(row["sgd_path"])).convert("RGB"),
            row,
            reviews[key],
        ).save(
            output
            / "triads"
            / f"snr_{int(float(row['snr_db']))}_rank_{int(row['rank']):02d}_reviewed.png"
        )
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for snr in map(float, config["stress"]["snrs_db"]):
        snr_reviews = [row for row in reviews.values() if float(row["snr_db"]) == snr]
        counts[str(int(snr))] = {}
        for method in ("s33", "sgd"):
            counts[str(int(snr))][method] = {
                category: sum(row[f"{method}_category"] == category for row in snr_reviews)
                for category in sorted(allowed)
            }
        final_path = output / f"stress_snr_{int(snr)}_reviewed.png"
        make_sheet(
            sorted((output / "triads").glob(f"snr_{int(snr)}_rank_??_reviewed.png")),
            final_path,
        )
    write_json(output / "manual_review_summary.json", {"counts": counts})
    print(json.dumps({"status": "FINALIZED", "counts": counts}, indent=2))


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.stage == "prepare-s33":
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        prepare_s33(config, config_path, device)
    elif args.stage == "assemble":
        assemble(config)
    else:
        finalize(config)


if __name__ == "__main__":
    main()
