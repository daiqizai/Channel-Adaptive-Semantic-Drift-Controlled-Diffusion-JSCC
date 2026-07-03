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
    parser = argparse.ArgumentParser(description="Analyze semantic gate errors for EXP-S4 residual refiner outputs.")
    parser.add_argument("--input-csv", default="outputs/EXP-S4-006/per_sample.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/exp_s4_006_gate_error_analysis")
    parser.add_argument("--top-k-global", type=int, default=12)
    parser.add_argument("--top-k-per-snr", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=224)
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
        row["residual_gate"] = float(str(row["residual_gate"]).strip())
        for key in ["original_top1_prob", "m0_top1_prob", "refined_top1_prob", "m3_top1_prob"]:
            row[key] = float(str(row[key]).strip())
        for key in [
            "detector_accept_refined",
            "m0_matches_original_top1",
            "refined_matches_original_top1",
            "refined_matches_m0_top1",
            "m3_matches_original_top1",
            "m3_matches_m0_top1",
            "false_accept",
            "false_reject",
        ]:
            row[key] = parse_bool(row[key])
        row["case_type"] = classify_case(row)
    return rows


def classify_case(row: dict[str, Any]) -> str:
    accept = parse_bool(row["detector_accept_refined"])
    m0_correct = parse_bool(row["m0_matches_original_top1"])
    refined_correct = parse_bool(row["refined_matches_original_top1"])
    if accept and refined_correct:
        return "accepted_correct"
    if accept and not refined_correct:
        return "accepted_wrong_same_as_m0"
    if (not accept) and refined_correct:
        return "missed_semantic_repair"
    if (not accept) and m0_correct and not refined_correct:
        return "protective_reject"
    return "rejected_both_wrong"


def rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    snrs = sorted({float(row["snr_db"]) for row in rows})
    summary_rows: list[dict[str, Any]] = []
    for snr in snrs:
        subset = [row for row in rows if float(row["snr_db"]) == snr]
        total = len(subset)
        counts = {
            "accepted_correct": sum(row["case_type"] == "accepted_correct" for row in subset),
            "accepted_wrong_same_as_m0": sum(row["case_type"] == "accepted_wrong_same_as_m0" for row in subset),
            "missed_semantic_repair": sum(row["case_type"] == "missed_semantic_repair" for row in subset),
            "protective_reject": sum(row["case_type"] == "protective_reject" for row in subset),
            "rejected_both_wrong": sum(row["case_type"] == "rejected_both_wrong" for row in subset),
        }
        accept_count = sum(bool(row["detector_accept_refined"]) for row in subset)
        false_accept_count = sum(bool(row["false_accept"]) for row in subset)
        false_reject_count = sum(bool(row["false_reject"]) for row in subset)
        m0_correct_count = sum(bool(row["m0_matches_original_top1"]) for row in subset)
        refined_correct_count = sum(bool(row["refined_matches_original_top1"]) for row in subset)
        m3_correct_count = sum(bool(row["m3_matches_original_top1"]) for row in subset)
        summary_rows.append(
            {
                "snr_db": snr,
                "num_images": total,
                "accept_count": accept_count,
                "accept_rate": rate(accept_count, total),
                "reject_count": total - accept_count,
                "reject_rate": rate(total - accept_count, total),
                "m0_correct_count": m0_correct_count,
                "m0_correct_rate": rate(m0_correct_count, total),
                "refined_correct_count": refined_correct_count,
                "refined_correct_rate": rate(refined_correct_count, total),
                "m3_correct_count": m3_correct_count,
                "m3_correct_rate": rate(m3_correct_count, total),
                "false_accept_count": false_accept_count,
                "false_accept_rate": rate(false_accept_count, total),
                "false_reject_count": false_reject_count,
                "false_reject_rate": rate(false_reject_count, total),
                **{f"{key}_count": value for key, value in counts.items()},
                **{f"{key}_rate": rate(value, total) for key, value in counts.items()},
            }
        )
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def short_label(value: str, max_chars: int = 24) -> str:
    value = str(value)
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1] + "."


def case_score(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(row["original_top1_prob"]),
        float(row["refined_top1_prob"]),
        float(row["m0_top1_prob"]),
    )


def make_quad(row: dict[str, Any], tile_size: int) -> Image.Image:
    margin = 18
    gap = 10
    title_h = 96
    label_h = 54
    footer_h = 44
    columns = [
        ("Original", "original", "original_top1_label", "original_top1_prob"),
        ("M0", "m0_reconstruction", "m0_top1_label", "m0_top1_prob"),
        ("Refined", "refined", "refined_top1_label", "refined_top1_prob"),
        ("M3 final", "m3_final", "m3_top1_label", "m3_top1_prob"),
    ]
    width = margin * 2 + tile_size * len(columns) + gap * (len(columns) - 1)
    height = title_h + tile_size + label_h + footer_h
    canvas = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(17)
    label_font = load_font(14)
    metric_font = load_font(13)

    snr = float(row["snr_db"])
    accept = "accept" if bool(row["detector_accept_refined"]) else "reject"
    title = f"{row['case_type']} | {snr:g} dB | {row['sample']} | gate={accept}"
    subtitle = (
        f"orig={short_label(row['original_top1_label'])}  "
        f"M0={short_label(row['m0_top1_label'])}  "
        f"refined={short_label(row['refined_top1_label'])}  "
        f"M3={short_label(row['m3_top1_label'])}"
    )
    draw.text((margin, 14), title, fill=(20, 20, 20), font=title_font)
    draw.text((margin, 43), subtitle, fill=(65, 65, 65), font=metric_font)
    draw.text(
        (margin, 68),
        "Top-1 pseudo-label diagnostic; original label is classifier pseudo-label, not COCO GT.",
        fill=(90, 90, 90),
        font=metric_font,
    )

    for index, (name, path_key, label_key, prob_key) in enumerate(columns):
        x = margin + index * (tile_size + gap)
        path = resolve_project_path(str(row[path_key]))
        image = fit_image(path, tile_size)
        canvas.paste(image, (x, title_h))
        draw.rectangle((x, title_h, x + tile_size - 1, title_h + tile_size - 1), outline=(210, 210, 210))
        draw.text((x + 6, title_h + tile_size + 6), name, fill=(30, 30, 30), font=label_font)
        draw.text(
            (x + 6, title_h + tile_size + 26),
            f"{short_label(row[label_key], 20)} | p={float(row[prob_key]):.3f}",
            fill=(70, 70, 70),
            font=metric_font,
        )

    footer = "Gate uses only refined-vs-M0 top-1 agreement; it cannot repair M0 mistakes under the same classifier."
    draw.text((margin, title_h + tile_size + label_h + 8), footer, fill=(80, 80, 80), font=metric_font)
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
    confidence = f"{float(row['original_top1_prob']):.3f}".replace(".", "p")
    return f"{row['case_type']}_{snr_name(float(row['snr_db']))}_{Path(str(row['sample'])).stem}_origconf_{confidence}.png"


def render_group(
    rows: list[dict[str, Any]],
    output_dir: Path,
    case_type: str,
    top_k_global: int,
    top_k_per_snr: int,
    tile_size: int,
) -> dict[str, Any]:
    group_rows = sorted([row for row in rows if row["case_type"] == case_type], key=case_score, reverse=True)
    group_dir = output_dir / case_type
    triptych_dir = group_dir / "quads"
    sheets_dir = group_dir / "sheets"
    triptych_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[tuple[float, str], str] = {}

    def render_row(row: dict[str, Any]) -> Image.Image:
        key = (float(row["snr_db"]), str(row["sample"]))
        if key not in cache:
            image = make_quad(row, tile_size)
            path = triptych_dir / filename_for_row(row)
            image.save(path)
            cache[key] = project_relative(path)
            return image
        return Image.open(resolve_project_path(cache[key])).convert("RGB")

    selected_global = group_rows[:top_k_global]
    result: dict[str, Any] = {
        "case_type": case_type,
        "num_cases": len(group_rows),
        "global_sheet": None,
        "global_top": [],
        "per_snr": {},
    }
    if selected_global:
        global_images = [render_row(row) for row in selected_global]
        global_sheet = stacked_sheet(global_images, f"{case_type}: global top {len(selected_global)}")
        global_sheet_path = sheets_dir / "global_top.png"
        global_sheet.save(global_sheet_path)
        result["global_sheet"] = project_relative(global_sheet_path)
        for rank, row in enumerate(selected_global, start=1):
            item = index_item(row)
            item["rank"] = rank
            item["quad"] = cache[(float(row["snr_db"]), str(row["sample"]))]
            result["global_top"].append(item)

    for snr in sorted({float(row["snr_db"]) for row in group_rows}):
        snr_rows = [row for row in group_rows if float(row["snr_db"]) == snr][:top_k_per_snr]
        if not snr_rows:
            continue
        images = [render_row(row) for row in snr_rows]
        sheet = stacked_sheet(images, f"{case_type}: {snr:g} dB top {len(snr_rows)}")
        sheet_path = sheets_dir / f"{snr_name(snr)}_top.png"
        sheet.save(sheet_path)
        result["per_snr"][snr_name(snr)] = {
            "snr_db": snr,
            "num_selected": len(snr_rows),
            "sheet": project_relative(sheet_path),
            "cases": [index_item(row) | {"quad": cache[(float(row["snr_db"]), str(row["sample"]))]} for row in snr_rows],
        }
    return result


def index_item(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "snr_db": float(row["snr_db"]),
        "sample": str(row["sample"]),
        "case_type": str(row["case_type"]),
        "detector_accept_refined": bool(row["detector_accept_refined"]),
        "original_top1_label": str(row["original_top1_label"]),
        "original_top1_prob": float(row["original_top1_prob"]),
        "m0_top1_label": str(row["m0_top1_label"]),
        "m0_top1_prob": float(row["m0_top1_prob"]),
        "refined_top1_label": str(row["refined_top1_label"]),
        "refined_top1_prob": float(row["refined_top1_prob"]),
        "m3_top1_label": str(row["m3_top1_label"]),
        "m3_top1_prob": float(row["m3_top1_prob"]),
        "m0_matches_original_top1": bool(row["m0_matches_original_top1"]),
        "refined_matches_original_top1": bool(row["refined_matches_original_top1"]),
        "m3_matches_original_top1": bool(row["m3_matches_original_top1"]),
    }


def make_markdown(index: dict[str, Any], summary_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# EXP-S4-006 Gate Error Analysis",
        "",
        "Derived from `outputs/EXP-S4-006/per_sample.csv`. This analysis does not run a model or download data.",
        "",
        "## Key Guardrail",
        "",
        "The current gate accepts refined output only when `c(refined) == c(M0)`. Under the same classifier, this means M3 top-1 final failure cannot exceed M0 top-1 failure by construction. This is useful as a conservative first gate, but it does not prove semantic reliability under an independent semantic metric.",
        "",
        "## Per-SNR Summary",
        "",
        "| SNR(dB) | N | Accept | Protective Reject | Missed Repair | Accepted Wrong Same As M0 | Rejected Both Wrong |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {snr:g} | {n} | {acc} ({acc_r:.4f}) | {prot} ({prot_r:.4f}) | {miss} ({miss_r:.4f}) | {aw} ({aw_r:.4f}) | {rbw} ({rbw_r:.4f}) |".format(
                snr=float(row["snr_db"]),
                n=int(row["num_images"]),
                acc=int(row["accept_count"]),
                acc_r=float(row["accept_rate"]),
                prot=int(row["protective_reject_count"]),
                prot_r=float(row["protective_reject_rate"]),
                miss=int(row["missed_semantic_repair_count"]),
                miss_r=float(row["missed_semantic_repair_rate"]),
                aw=int(row["accepted_wrong_same_as_m0_count"]),
                aw_r=float(row["accepted_wrong_same_as_m0_rate"]),
                rbw=int(row["rejected_both_wrong_count"]),
                rbw_r=float(row["rejected_both_wrong_rate"]),
            )
        )
    lines.extend(["", "## Case Type Meaning", ""])
    lines.extend(
        [
            "- `protective_reject`: M0 matches the original pseudo-label, refined changes it, and the gate rejects refined. This is the detector doing useful semantic protection.",
            "- `missed_semantic_repair`: M0 is wrong, refined matches the original pseudo-label, but the gate rejects refined because it differs from M0. This is the main cost of top-1 agreement gating.",
            "- `accepted_wrong_same_as_m0`: refined and M0 agree with each other but both differ from the original pseudo-label. This is not additional top-1 damage from refined, but it shows the gate cannot repair M0 semantic failures.",
            "- `rejected_both_wrong`: refined and M0 both differ from the original pseudo-label and refined differs from M0. The fallback avoids one wrong refined label, but still returns wrong M0 under this pseudo-label metric.",
            "",
            "## Gallery Sheets",
            "",
        ]
    )
    for case_type, payload in index["case_types"].items():
        lines.append(f"### {case_type}")
        lines.append("")
        lines.append(f"- Total cases: `{payload['num_cases']}`")
        if payload.get("global_sheet"):
            lines.append(f"- Global sheet: `{payload['global_sheet']}`")
        for snr_key, item in payload.get("per_snr", {}).items():
            lines.append(f"- `{snr_key}` sheet: `{item['sheet']}`")
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
    summary_rows = summarize(rows)
    rows_for_csv = [dict(row) for row in rows]
    write_csv(output_dir / "per_sample_with_case_type.csv", rows_for_csv)
    write_csv(output_dir / "summary.csv", summary_rows)

    case_types = [
        "protective_reject",
        "missed_semantic_repair",
        "accepted_wrong_same_as_m0",
        "rejected_both_wrong",
    ]
    index = {
        "source_csv": project_relative(input_csv),
        "output_dir": project_relative(output_dir),
        "top_k_global": int(args.top_k_global),
        "top_k_per_snr": int(args.top_k_per_snr),
        "case_types": {},
    }
    for case_type in case_types:
        index["case_types"][case_type] = render_group(
            rows=rows,
            output_dir=output_dir,
            case_type=case_type,
            top_k_global=int(args.top_k_global),
            top_k_per_snr=int(args.top_k_per_snr),
            tile_size=int(args.tile_size),
        )

    with (output_dir / "index.json").open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2)
    (output_dir / "REPORT.md").write_text(make_markdown(index, summary_rows), encoding="utf-8")
    print(json.dumps({"output_dir": project_relative(output_dir), "summary_rows": len(summary_rows)}, indent=2))


if __name__ == "__main__":
    main()
