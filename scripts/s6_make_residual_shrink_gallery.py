from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build residual-shrink M3 artifact galleries from existing validation/held-out/test-like CSVs."
    )
    parser.add_argument("--config", default="configs/s6_residual_shrink_artifact_gallery_exp_s4_006.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
            writer.writerow({key: serialize_value(row.get(key, "")) for key in fieldnames})


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def serialize_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def parse_float(value: Any, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    return float(value)


def optional_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def alpha_matches(row: dict[str, str], alpha: Any | None) -> bool:
    if alpha is None:
        return True
    row_alpha = optional_float(row.get("alpha"))
    return row_alpha is not None and abs(row_alpha - float(alpha)) < 1e-9


def row_matches_policy(row: dict[str, str], policy: str, alpha: Any | None = None) -> bool:
    return row.get("policy") == policy and alpha_matches(row, alpha)


def normalize_row(row: dict[str, str], split: str, role: str, label: str) -> dict[str, Any]:
    out: dict[str, Any] = dict(row)
    out["split"] = split
    out["role"] = role
    out["policy_label"] = label
    for key in [
        "accept_candidate",
        "m0_matches_original_top1",
        "candidate_matches_original_top1",
        "candidate_matches_m0_top1",
        "final_matches_original_top1",
        "final_matches_m0_top1",
        "accepted_repair",
        "accepted_new_error",
        "rejected_good",
    ]:
        out[key] = parse_bool(out.get(key))
    for key in [
        "snr_db",
        "alpha",
        "original_top1_prob",
        "m0_top1_prob",
        "candidate_top1_prob",
        "final_top1_prob",
    ]:
        if key in out:
            out[key] = optional_float(out.get(key))
    return out


def selected_categories(row: dict[str, Any]) -> list[str]:
    categories: list[str] = []
    if (
        row["accept_candidate"]
        and row["m0_matches_original_top1"]
        and row["candidate_matches_original_top1"]
        and row["final_matches_original_top1"]
    ):
        categories.append("m3_safe_accept")
    if (
        not row["accept_candidate"]
        and row["m0_matches_original_top1"]
        and not row["candidate_matches_original_top1"]
        and row["final_matches_original_top1"]
    ):
        categories.append("m3_protective_reject")
    if (not row["accept_candidate"]) and row["candidate_matches_original_top1"]:
        categories.append("m3_rejected_good_candidate")
    if row["accepted_repair"]:
        categories.append("m3_accepted_repair")
    if row["accepted_new_error"]:
        categories.append("m3_accepted_new_error")
    return categories


def unsafe_categories(row: dict[str, Any]) -> list[str]:
    categories: list[str] = []
    if row["accepted_new_error"]:
        categories.append("unsafe_accepted_new_error")
    if row["accepted_repair"]:
        categories.append("unsafe_accepted_repair")
    return categories


def sort_case_key(row: dict[str, Any]) -> tuple[float, str, str]:
    return (float(row.get("snr_db") or 0.0), str(row.get("sample", "")), str(row.get("policy_label", "")))


def validate_inputs(config: dict[str, Any]) -> dict[str, str]:
    paths: dict[str, Path] = {
        "jscc_checkpoint": resolve_project_path(config["inputs"]["jscc_checkpoint"]),
        "forbidden_checkpoint": resolve_project_path(config["inputs"]["forbidden_checkpoint"]),
    }
    for split in config["splits"]:
        split_name = str(split["name"])
        paths[f"{split_name}_per_sample_csv"] = resolve_project_path(split["per_sample_csv"])
        paths[f"{split_name}_summary_csv"] = resolve_project_path(split["summary_csv"])
    missing = [f"{key}: {path}" for key, path in paths.items() if key != "forbidden_checkpoint" and not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))
    if paths["jscc_checkpoint"] == paths["forbidden_checkpoint"]:
        raise RuntimeError("Config points to forbidden latest.pt checkpoint.")
    return {key: project_relative(path) for key, path in paths.items()}


def collect_split_rows(split: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    split_name = str(split["name"])
    rows = read_csv(resolve_project_path(split["per_sample_csv"]))
    summary_rows = read_csv(resolve_project_path(split["summary_csv"]))
    selected = split["selected_policy"]
    selected_rows = [
        normalize_row(row, split_name, "selected_m3", str(selected["label"]))
        for row in rows
        if row_matches_policy(row, str(selected["policy"]), selected.get("alpha"))
    ]
    unsafe_rows: list[dict[str, Any]] = []
    for unsafe in split.get("unsafe_policies", []):
        label = str(unsafe["label"])
        policy = str(unsafe["policy"])
        alpha = unsafe.get("alpha")
        unsafe_rows.extend(
            normalize_row(row, split_name, "unsafe", label)
            for row in rows
            if row_matches_policy(row, policy, alpha)
        )
    return selected_rows, unsafe_rows, summary_rows


def summary_row_matches(row: dict[str, str], policy: str, alpha: Any | None) -> bool:
    if row.get("policy") != policy:
        return False
    if str(row.get("snr_db")) != "all":
        return False
    return alpha_matches(row, alpha)


def build_policy_summary(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in config["splits"]:
        split_name = str(split["name"])
        summary_rows = read_csv(resolve_project_path(split["summary_csv"]))
        specs = [("selected_m3", split["selected_policy"])]
        specs.extend(("unsafe", item) for item in split.get("unsafe_policies", []))
        for role, spec in specs:
            policy = str(spec["policy"])
            alpha = spec.get("alpha")
            matches = [row for row in summary_rows if summary_row_matches(row, policy, alpha)]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one summary row for {split_name}:{policy}:{alpha}, got {len(matches)}")
            match = matches[0]
            rows.append(
                {
                    "split": split_name,
                    "role": role,
                    "label": str(spec["label"]),
                    "source_policy": policy,
                    "alpha": "" if alpha is None else float(alpha),
                    "num_images": int(float(match["num_images"])),
                    "accept_rate": parse_float(match["accept_rate"]),
                    "m0_failure_rate": parse_float(match["m0_failure_rate"]),
                    "final_failure_rate": parse_float(match["final_failure_rate"]),
                    "delta_final_failure_vs_m0": parse_float(match["delta_final_failure_vs_m0"]),
                    "delta_psnr_vs_m0_db": parse_float(match["delta_psnr_vs_m0_db"]),
                    "delta_lpips_vs_m0": parse_float(match.get("delta_lpips_vs_m0")),
                    "repair_count": int(float(match["repair_count"])),
                    "accepted_new_error_count": int(float(match["accepted_new_error_count"])),
                    "rejected_good_count": int(float(match["rejected_good_count"])),
                }
            )
    return rows


def case_count_rows(
    selected_rows_by_split: dict[str, list[dict[str, Any]]],
    unsafe_rows_by_split: dict[str, list[dict[str, Any]]],
    categories: list[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split, rows in selected_rows_by_split.items():
        counts = {category: 0 for category in categories}
        for row in rows:
            for category in selected_categories(row):
                counts[category] += 1
        for category in categories:
            if category.startswith("unsafe_"):
                continue
            out.append({"split": split, "role": "selected_m3", "category": category, "count": counts[category]})
    for split, rows in unsafe_rows_by_split.items():
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row["policy_label"]), []).append(row)
        for label, label_rows in grouped.items():
            counts = {category: 0 for category in categories}
            for row in label_rows:
                for category in unsafe_categories(row):
                    counts[category] += 1
            for category in categories:
                if not category.startswith("unsafe_"):
                    continue
                out.append({"split": split, "role": "unsafe", "label": label, "category": category, "count": counts[category]})
    return out


def text_for_column(row: dict[str, Any], column: str) -> str:
    if column == "original":
        label = row.get("original_top1_label", "")
        prob = row.get("original_top1_prob")
    elif column == "m0_reconstruction":
        label = row.get("m0_top1_label", "")
        prob = row.get("m0_top1_prob")
    elif column == "candidate":
        label = row.get("candidate_top1_label", "")
        prob = row.get("candidate_top1_prob")
    else:
        label = row.get("final_top1_label", "")
        prob = row.get("final_top1_prob")
    prob_text = "" if prob in ("", None) else f" {float(prob):.2f}"
    return f"{column}: {label}{prob_text}"


def shorten(text: str, max_len: int = 38) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def make_case_grid(row: dict[str, Any], columns: list[str], image_size: int, output_path: Path) -> None:
    font = ImageFont.load_default()
    header_h = 54
    tile_w = image_size
    tile_h = image_size + header_h
    canvas = Image.new("RGB", (tile_w * len(columns), tile_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, column in enumerate(columns):
        path = resolve_project_path(str(row[column]))
        image = Image.open(path).convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
        x = idx * tile_w
        draw.rectangle([x, 0, x + tile_w - 1, tile_h - 1], outline=(210, 210, 210), width=1)
        draw.text((x + 6, 6), shorten(text_for_column(row, column)), fill=(0, 0, 0), font=font)
        draw.text((x + 6, 24), f"split={row['split']} snr={float(row['snr_db']):g}", fill=(70, 70, 70), font=font)
        canvas.paste(image, (x, header_h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def make_contact_sheet(paths: list[Path], output_path: Path, thumb_width: int = 384, columns: int = 2) -> None:
    if not paths:
        return
    thumbs: list[Image.Image] = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        ratio = thumb_width / img.width
        thumbs.append(img.resize((thumb_width, int(img.height * ratio)), Image.Resampling.BICUBIC))
    rows = (len(thumbs) + columns - 1) // columns
    thumb_h = max(img.height for img in thumbs)
    sheet = Image.new("RGB", (thumb_width * columns, thumb_h * rows), "white")
    for idx, img in enumerate(thumbs):
        x = (idx % columns) * thumb_width
        y = (idx // columns) * thumb_h
        sheet.paste(img, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_").replace(".", "p")


def choose_cases(rows: list[dict[str, Any]], category: str, source: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        categories = selected_categories(row) if source == "selected" else unsafe_categories(row)
        if category in categories:
            selected.append(row)
    return sorted(selected, key=sort_case_key)


def build_case_galleries(
    config: dict[str, Any],
    selected_rows_by_split: dict[str, list[dict[str, Any]]],
    unsafe_rows_by_split: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    max_cases = int(config["gallery"]["max_cases_per_category_per_split"])
    image_size = int(config["gallery"]["image_size"])
    columns = [str(item) for item in config["gallery"]["columns"]]
    cases: list[dict[str, Any]] = []
    gallery_paths_by_category: dict[str, list[Path]] = {}
    for category in config["case_categories"]:
        source = "unsafe" if str(category).startswith("unsafe_") else "selected"
        rows_by_split = unsafe_rows_by_split if source == "unsafe" else selected_rows_by_split
        for split, rows in rows_by_split.items():
            chosen = choose_cases(rows, str(category), source)[:max_cases]
            for idx, row in enumerate(chosen, start=1):
                policy_name = safe_name(str(row["policy_label"]))
                sample_stem = Path(str(row["sample"])).stem
                snr_tag = safe_name(f"snr_{float(row['snr_db']):g}db")
                out_path = (
                    output_dir
                    / "samples"
                    / str(category)
                    / f"{safe_name(split)}_{idx:02d}_{snr_tag}_{safe_name(sample_stem)}_{policy_name}.png"
                )
                make_case_grid(row, columns, image_size, out_path)
                gallery_paths_by_category.setdefault(str(category), []).append(out_path)
                cases.append(
                    {
                        "split": split,
                        "category": str(category),
                        "role": row["role"],
                        "policy": row["policy"],
                        "policy_label": row["policy_label"],
                        "alpha": "" if row.get("alpha") is None else row.get("alpha"),
                        "snr_db": row["snr_db"],
                        "sample": row["sample"],
                        "accept_candidate": row["accept_candidate"],
                        "m0_matches_original_top1": row["m0_matches_original_top1"],
                        "candidate_matches_original_top1": row["candidate_matches_original_top1"],
                        "final_matches_original_top1": row["final_matches_original_top1"],
                        "accepted_repair": row["accepted_repair"],
                        "accepted_new_error": row["accepted_new_error"],
                        "grid": project_relative(out_path),
                    }
                )
    for category, paths in gallery_paths_by_category.items():
        make_contact_sheet(paths, output_dir / "samples" / f"{category}_sheet.png")
    return cases


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "N/A"


def git_dirty_state() -> str:
    try:
        output = subprocess.check_output(["git", "status", "--short"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"
    return "dirty" if output else "clean"


def proxy_environment_present() -> list[str]:
    keys = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "NO_PROXY", "no_proxy"]
    return [key for key in keys if os.environ.get(key)]


def signed(value: Any, digits: int = 4) -> str:
    return f"{float(value):+.{digits}f}"


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    lines = []
    lines.append("| " + " | ".join(label for _key, label in columns) + " |")
    lines.append("|" + "|".join(["---" for _ in columns]) + "|")
    for row in rows:
        values = [str(row.get(key, "")) for key, _label in columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def make_report(
    config: dict[str, Any],
    policy_summary: list[dict[str, Any]],
    case_counts: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> str:
    selected = [row for row in policy_summary if row["role"] == "selected_m3"]
    unsafe_full = [row for row in policy_summary if row["label"] == "AlwaysAcceptFullStrength"]
    unsafe_constrained = [row for row in policy_summary if row["label"] == "AlwaysAcceptValidationConstrained"]
    selected_new_errors = "/".join(str(row["accepted_new_error_count"]) for row in selected)
    selected_psnr = "/".join(signed(row["delta_psnr_vs_m0_db"]) for row in selected)
    full_new_errors = "/".join(str(row["accepted_new_error_count"]) for row in unsafe_full)
    constrained_new_errors = "/".join(str(row["accepted_new_error_count"]) for row in unsafe_constrained)
    count_lookup = {
        (row["split"], row["role"], row.get("label", ""), row["category"]): row["count"] for row in case_counts
    }
    lines = [
        "# Residual Shrink M3 Artifact Gallery",
        "",
        "This derived report collects validation, held-out, and test-like residual-shrink outputs into one audit artifact.",
        "It does not tune alpha, rerun classifiers, train models, run diffusion, or download data.",
        "",
        "## Bottom Line",
        "",
        f"- Selected shrink M3 PSNR deltas on validation/held-out/test-like are `{selected_psnr}` dB vs M0.",
        f"- Selected shrink M3 accepted new errors on validation/held-out/test-like are `{selected_new_errors}`.",
        f"- Always-accept full strength accepted new errors are `{full_new_errors}`.",
        f"- Validation-constrained always-accept accepted new errors are `{constrained_new_errors}`.",
        "- The gallery therefore supports shrink-M3 as a conservative quality-improvement artifact, while keeping always-accept as a negative/unsafe contrast.",
        "",
        "## Policy Summary",
        "",
    ]
    display_summary = []
    for row in policy_summary:
        display_summary.append(
            {
                "split": row["split"],
                "label": row["label"],
                "delta_psnr": signed(row["delta_psnr_vs_m0_db"]),
                "delta_lpips": signed(row["delta_lpips_vs_m0"]),
                "failure_delta": signed(row["delta_final_failure_vs_m0"]),
                "accept": f"{float(row['accept_rate']):.4f}",
                "repair": row["repair_count"],
                "new_error": row["accepted_new_error_count"],
                "rejected_good": row["rejected_good_count"],
            }
        )
    lines += markdown_table(
        display_summary,
        [
            ("split", "Split"),
            ("label", "Policy"),
            ("delta_psnr", "Delta PSNR"),
            ("delta_lpips", "Delta LPIPS"),
            ("failure_delta", "Failure Delta"),
            ("accept", "Accept"),
            ("repair", "Repair"),
            ("new_error", "New Error"),
            ("rejected_good", "Rejected Good"),
        ],
    )
    lines.extend(["", "## Case Counts", ""])
    interesting_counts = []
    for split in [str(item["name"]) for item in config["splits"]]:
        interesting_counts.append(
            {
                "split": split,
                "m3_safe_accept": count_lookup.get((split, "selected_m3", "", "m3_safe_accept"), 0),
                "m3_protective_reject": count_lookup.get((split, "selected_m3", "", "m3_protective_reject"), 0),
                "m3_rejected_good": count_lookup.get((split, "selected_m3", "", "m3_rejected_good_candidate"), 0),
                "m3_new_error": count_lookup.get((split, "selected_m3", "", "m3_accepted_new_error"), 0),
            }
        )
    lines += markdown_table(
        interesting_counts,
        [
            ("split", "Split"),
            ("m3_safe_accept", "Safe Accept"),
            ("m3_protective_reject", "Protective Reject"),
            ("m3_rejected_good", "Rejected Good"),
            ("m3_new_error", "M3 New Error"),
        ],
    )
    lines.extend(["", "## Files", ""])
    lines.extend(
        [
            f"- Policy summary: `{metadata['policy_summary_csv']}`",
            f"- Case counts: `{metadata['case_counts_csv']}`",
            f"- Case index: `{metadata['case_index_csv']}`",
            f"- Sample directory: `{metadata['sample_dir']}`",
            f"- Metadata: `{metadata['metadata_json']}`",
        ]
    )
    if case_rows:
        lines.extend(["", "## Example Sheets", ""])
        sheet_dir = resolve_project_path(metadata["sample_dir"])
        for path in sorted(sheet_dir.glob("*_sheet.png")):
            lines.append(f"- `{project_relative(path)}`")
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- The semantic signal is still frozen AlexNet pseudo-label agreement on COCO crops.",
            "- This is an artifact consolidation pass over existing outputs, not a new independent experiment.",
            "- Always-accept examples are included as negative controls and must not be presented as final M3.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    input_manifest = validate_inputs(config)
    output_dir = resolve_project_path(args.output_dir or config["outputs"]["output_dir"])

    selected_rows_by_split: dict[str, list[dict[str, Any]]] = {}
    unsafe_rows_by_split: dict[str, list[dict[str, Any]]] = {}
    split_counts: dict[str, dict[str, int]] = {}
    for split in config["splits"]:
        selected_rows, unsafe_rows, _summary_rows = collect_split_rows(split)
        split_name = str(split["name"])
        selected_rows_by_split[split_name] = selected_rows
        unsafe_rows_by_split[split_name] = unsafe_rows
        split_counts[split_name] = {
            "selected_rows": len(selected_rows),
            "unsafe_rows": len(unsafe_rows),
        }

    dry_payload = {
        "status": "ok",
        "config": project_relative(config_path),
        "inputs": input_manifest,
        "output_dir": project_relative(output_dir),
        "split_counts": split_counts,
        "proxy_environment_present": proxy_environment_present(),
        "notes": config.get("notes", []),
    }
    if args.dry_run:
        print(json.dumps(dry_payload, indent=2, ensure_ascii=False))
        return

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output_dir / "config.yaml")

    policy_summary = build_policy_summary(config)
    case_counts = case_count_rows(selected_rows_by_split, unsafe_rows_by_split, list(config["case_categories"]))
    case_rows = build_case_galleries(config, selected_rows_by_split, unsafe_rows_by_split, output_dir)

    policy_summary_csv = output_dir / "policy_summary.csv"
    case_counts_csv = output_dir / "case_counts.csv"
    case_index_csv = output_dir / "case_index.csv"
    metadata_json = output_dir / "metadata.json"
    report_path = output_dir / "REPORT.md"
    write_csv(policy_summary_csv, policy_summary)
    write_csv(case_counts_csv, case_counts)
    write_csv(case_index_csv, case_rows)

    metadata = {
        "analysis_id": config["analysis_id"],
        "method": config["method"],
        "project_commit": git_commit(),
        "git_dirty_state": git_dirty_state(),
        "python": sys.version,
        "platform": platform.platform(),
        "config": project_relative(config_path),
        "input_manifest": input_manifest,
        "policy_summary_csv": project_relative(policy_summary_csv),
        "case_counts_csv": project_relative(case_counts_csv),
        "case_index_csv": project_relative(case_index_csv),
        "sample_dir": project_relative(output_dir / "samples"),
        "metadata_json": project_relative(metadata_json),
        "proxy_environment_present": proxy_environment_present(),
        "notes": config.get("notes", []),
    }
    save_json(metadata_json, metadata)
    report_path.write_text(make_report(config, policy_summary, case_counts, case_rows, metadata), encoding="utf-8")
    print(json.dumps({"output_dir": project_relative(output_dir), "report": project_relative(report_path)}, indent=2))


if __name__ == "__main__":
    main()
