from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create frozen-classifier pseudo-label failure galleries.")
    parser.add_argument("--input-csv", default="outputs/EXP-S3-002/per_sample.csv")
    parser.add_argument("--output-dir", default="outputs/EXP-S3-002/failure_cases")
    parser.add_argument("--top-k-per-snr", type=int, default=6)
    parser.add_argument("--top-k-global", type=int, default=12)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
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


def snr_name(snr: float) -> str:
    if float(snr).is_integer():
        return f"snr_{int(snr):02d}db"
    return f"snr_{str(snr).replace('.', 'p')}db"


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["snr_db"] = float(str(row["snr_db"]).strip())
        for key in ["original_top1_prob", "m0_top1_prob", "m1_top1_prob"]:
            row[key] = float(str(row[key]).strip())
        for key in [
            "m0_matches_original_top1",
            "m1_matches_original_top1",
            "m1_matches_m0_top1",
            "m0_top5_contains_original_top1",
            "m1_top5_contains_original_top1",
        ]:
            row[key] = parse_bool(row[key])
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def fit_image(path: Path, tile_size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (tile_size, tile_size), "white")
    canvas.paste(image, ((tile_size - image.width) // 2, (tile_size - image.height) // 2))
    return canvas


def short_label(label: str, max_chars: int = 30) -> str:
    if len(label) <= max_chars:
        return label
    return label[: max_chars - 1] + "."


def make_triptych(row: dict[str, Any], tile_size: int) -> Image.Image:
    margin = 18
    gap = 10
    title_h = 88
    label_h = 48
    footer_h = 42
    width = margin * 2 + tile_size * 3 + gap * 2
    height = title_h + tile_size + label_h + footer_h
    canvas = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(17)
    label_font = load_font(14)
    metric_font = load_font(13)

    snr = float(row["snr_db"])
    title = f"{snr:g} dB | {row['sample']} | pseudo-label drift"
    subtitle = (
        f"orig={short_label(str(row['original_top1_label']))} ({row['original_top1_prob']:.3f})  "
        f"M0={short_label(str(row['m0_top1_label']))} ({row['m0_top1_prob']:.3f})  "
        f"M1={short_label(str(row['m1_top1_label']))} ({row['m1_top1_prob']:.3f})"
    )
    draw.text((margin, 14), title, fill=(20, 20, 20), font=title_font)
    draw.text((margin, 44), subtitle, fill=(70, 70, 70), font=metric_font)

    columns = [
        ("Original", str(row["original_top1_label"]), float(row["original_top1_prob"]), resolve_project_path(str(row["original"]))),
        ("M0 Reconstruction", str(row["m0_top1_label"]), float(row["m0_top1_prob"]), resolve_project_path(str(row["m0_reconstruction"]))),
        ("M1 Refined", str(row["m1_top1_label"]), float(row["m1_top1_prob"]), resolve_project_path(str(row["m1_refined"]))),
    ]
    for index, (name, label, prob, path) in enumerate(columns):
        x = margin + index * (tile_size + gap)
        image = fit_image(path, tile_size)
        canvas.paste(image, (x, title_h))
        draw.rectangle((x, title_h, x + tile_size - 1, title_h + tile_size - 1), outline=(210, 210, 210))
        draw.text((x + 6, title_h + tile_size + 6), name, fill=(30, 30, 30), font=label_font)
        draw.text((x + 6, title_h + tile_size + 25), f"{short_label(label, 24)} | p={prob:.3f}", fill=(70, 70, 70), font=metric_font)

    footer = "Auxiliary diagnostic only: ImageNet pseudo-label consistency, not COCO GT clean-correct metric."
    draw.text((margin, title_h + tile_size + label_h + 8), footer, fill=(90, 90, 90), font=metric_font)
    return canvas


def stacked_sheet(images: list[Image.Image], title: str) -> Image.Image:
    if not images:
        raise ValueError("Cannot create an empty sheet.")
    margin = 18
    title_h = 48
    gap = 12
    title_font = load_font(18)
    width = max(image.width for image in images) + margin * 2
    height = title_h + sum(image.height for image in images) + gap * (len(images) - 1) + margin
    canvas = Image.new("RGB", (width, height), (238, 238, 238))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 14), title, fill=(20, 20, 20), font=title_font)
    y = title_h
    for image in images:
        canvas.paste(image, (margin, y))
        y += image.height + gap
    return canvas


def filename_for_row(row: dict[str, Any]) -> str:
    confidence = f"{row['original_top1_prob']:.3f}".replace(".", "p")
    return f"{snr_name(float(row['snr_db']))}_{Path(str(row['sample'])).stem}_origconf_{confidence}.png"


def select_failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = [
        row
        for row in rows
        if bool(row["m0_matches_original_top1"]) and not bool(row["m1_matches_original_top1"])
    ]
    return sorted(
        failures,
        key=lambda row: (float(row["original_top1_prob"]), float(row["m0_top1_prob"]), -float(row["m1_top1_prob"])),
        reverse=True,
    )


def make_markdown(index: dict[str, Any]) -> str:
    lines = [
        "# Frozen Classifier Failure Case Gallery",
        "",
        "Derived from `outputs/EXP-S3-002/per_sample.csv`.",
        "",
        "This is an ImageNet pseudo-label diagnostic. It does not replace a COCO GT clean-correct semantic drift metric.",
        "",
        "## Global Top Cases",
        "",
        "| Rank | SNR(dB) | Sample | Original | M0 | M1 | Image |",
        "|---:|---:|---|---|---|---|---|",
    ]
    for item in index["global_top"]:
        lines.append(
            "| {rank} | {snr_db:g} | `{sample}` | {orig} ({orig_p:.3f}) | {m0} ({m0_p:.3f}) | {m1} ({m1_p:.3f}) | `{image}` |".format(
                rank=item["rank"],
                snr_db=float(item["snr_db"]),
                sample=item["sample"],
                orig=item["original_top1_label"],
                orig_p=float(item["original_top1_prob"]),
                m0=item["m0_top1_label"],
                m0_p=float(item["m0_top1_prob"]),
                m1=item["m1_top1_label"],
                m1_p=float(item["m1_top1_prob"]),
                image=item["triptych"],
            )
        )
    lines.extend(["", "## Per-SNR Sheets", ""])
    for snr_key, item in index["per_snr"].items():
        lines.append(f"- `{snr_key}`: `{item['sheet']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_csv = resolve_project_path(args.input_csv)
    output_dir = resolve_project_path(args.output_dir)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if output_dir.exists():
        if not args.overwrite and any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory already exists and is non-empty: {output_dir}")
        if args.overwrite:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(input_csv)
    failures = select_failures(rows)
    global_rows = failures[: int(args.top_k_global)]

    per_snr_rows: dict[str, list[dict[str, Any]]] = {}
    for row in failures:
        per_snr_rows.setdefault(snr_name(float(row["snr_db"])), []).append(row)
    for key in per_snr_rows:
        per_snr_rows[key] = per_snr_rows[key][: int(args.top_k_per_snr)]

    triptych_dir = output_dir / "triptychs"
    sheets_dir = output_dir / "sheets"
    triptych_dir.mkdir()
    sheets_dir.mkdir()
    triptych_cache: dict[tuple[float, str], str] = {}

    def render_row(row: dict[str, Any]) -> Image.Image:
        key = (float(row["snr_db"]), str(row["sample"]))
        if key not in triptych_cache:
            image = make_triptych(row, int(args.tile_size))
            path = triptych_dir / filename_for_row(row)
            image.save(path)
            triptych_cache[key] = project_relative(path)
            return image
        return Image.open(resolve_project_path(triptych_cache[key])).convert("RGB")

    global_images = [render_row(row) for row in global_rows]
    global_sheet_path = sheets_dir / "global_top_classifier_drift.png"
    stacked_sheet(global_images, f"Global top {len(global_images)} classifier pseudo-label drifts").save(global_sheet_path)

    index: dict[str, Any] = {
        "source_csv": project_relative(input_csv),
        "output_dir": project_relative(output_dir),
        "selection": "m0_matches_original_top1 && !m1_matches_original_top1, sorted by original_top1_prob",
        "top_k_global": int(args.top_k_global),
        "top_k_per_snr": int(args.top_k_per_snr),
        "global_sheet": project_relative(global_sheet_path),
        "global_top": [],
        "per_snr": {},
    }
    for rank, row in enumerate(global_rows, start=1):
        item = dict(row)
        item["rank"] = rank
        item["triptych"] = triptych_cache[(float(row["snr_db"]), str(row["sample"]))]
        index["global_top"].append(item)

    for snr_key, snr_rows in sorted(per_snr_rows.items()):
        images = [render_row(row) for row in snr_rows]
        sheet_path = sheets_dir / f"{snr_key}_top_classifier_drift.png"
        stacked_sheet(images, f"{snr_key}: top {len(images)} classifier pseudo-label drifts").save(sheet_path)
        index["per_snr"][snr_key] = {
            "sheet": project_relative(sheet_path),
            "cases": [
                {
                    **row,
                    "rank": rank,
                    "triptych": triptych_cache[(float(row["snr_db"]), str(row["sample"]))],
                }
                for rank, row in enumerate(snr_rows, start=1)
            ],
        }

    write_csv(output_dir / "global_top_classifier_drift.csv", global_rows)
    with (output_dir / "index.json").open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2)
    (output_dir / "README.md").write_text(make_markdown(index), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": project_relative(output_dir),
                "global_sheet": project_relative(global_sheet_path),
                "num_triptychs": len(triptych_cache),
                "top_k_global": int(args.top_k_global),
                "top_k_per_snr": int(args.top_k_per_snr),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
