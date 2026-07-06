from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RE = re.compile(r"^sample_(\d{6})\.png$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate test-like policies on COCO object-label clean-correct subset with CLIP zero-shot classifier."
    )
    parser.add_argument("--config", default="configs/s5_testlike_coco_object_clip_clean_eval_exp_s4_006.yaml")
    parser.add_argument("--device", default="auto")
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


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def snr_name(snr: float) -> str:
    if float(snr).is_integer():
        return f"snr_{int(snr):02d}db"
    return f"snr_{str(snr).replace('.', 'p')}db"


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def rate(flags: list[bool]) -> float:
    return float(sum(flags) / len(flags)) if flags else 0.0


def sample_index(name: str) -> int:
    match = SAMPLE_RE.match(name)
    if not match:
        raise ValueError(f"Unexpected sample file name: {name}")
    return int(match.group(1))


def coco_image_id_from_path(path: str | Path) -> int:
    return int(Path(path).stem)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def serialize_value(value: Any) -> Any:
    if isinstance(value, bool):
        return bool_text(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_value(row.get(key, "")) for key in fieldnames})


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


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


def load_policy_rows(path: Path, policies: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(path):
        if row["policy"] not in policies:
            continue
        converted: dict[str, Any] = dict(row)
        converted["snr_db"] = float(row["snr_db"])
        for key in [
            "accept_refined",
            "baseline_accept_refined",
            "candidate_accept_refined",
            "new_accept_vs_top1",
            "final_matches_original_top1",
            "m0_matches_original_top1",
            "refined_matches_original_top1",
            "accepted_repair",
            "missed_repair",
            "accepted_new_error",
            "protective_reject",
        ]:
            if key in converted:
                converted[key] = parse_bool(converted[key])
        for key in ["final_psnr_db", "m0_psnr_db", "refined_psnr_db", "clip_sim_m0_refined"]:
            if key in converted and converted[key] != "":
                converted[key] = float(converted[key])
        rows.append(converted)
    if not rows:
        raise RuntimeError(f"No policy rows found in {path}")
    return rows


def load_source_paths(path: Path) -> list[str]:
    manifest = load_json(path)
    paths = manifest.get("paths")
    if not isinstance(paths, list):
        raise RuntimeError(f"Expected paths list in {path}")
    return [str(item) for item in paths]


def dominant_coco_labels(config: dict[str, Any], source_paths: list[str]) -> dict[str, dict[str, Any]]:
    payload = load_json(resolve_project_path(config["inputs"]["coco_instances"]))
    categories = {int(item["id"]): str(item["name"]) for item in payload["categories"]}
    images = {int(item["id"]): item for item in payload["images"]}
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    ignore_crowd = bool(config["labeling"].get("ignore_crowd", True))
    for ann in payload["annotations"]:
        if ignore_crowd and int(ann.get("iscrowd", 0)):
            continue
        anns_by_image[int(ann["image_id"])].append(ann)

    min_area_ratio = float(config["labeling"]["dominant_category_min_area_ratio"])
    min_share = float(config["labeling"]["dominant_category_min_share"])
    labels: dict[str, dict[str, Any]] = {}
    for index, source_path in enumerate(source_paths):
        image_id = coco_image_id_from_path(source_path)
        image = images.get(image_id)
        if image is None:
            continue
        area_by_category: dict[int, float] = defaultdict(float)
        for ann in anns_by_image.get(image_id, []):
            area_by_category[int(ann["category_id"])] += float(ann.get("area", 0.0))
        total_area = sum(area_by_category.values())
        image_area = float(image["width"]) * float(image["height"])
        if total_area <= 0.0 or image_area <= 0.0:
            continue
        category_id, dominant_area = max(area_by_category.items(), key=lambda item: item[1])
        dominant_share = dominant_area / total_area
        dominant_area_ratio = dominant_area / image_area
        usable = dominant_share >= min_share and dominant_area_ratio >= min_area_ratio
        labels[f"sample_{index:06d}.png"] = {
            "sample_index": index,
            "source_path": source_path,
            "coco_image_id": image_id,
            "dominant_category_id": category_id,
            "dominant_label": categories[category_id],
            "dominant_area": dominant_area,
            "total_annotated_area": total_area,
            "image_area": image_area,
            "dominant_share": dominant_share,
            "dominant_area_ratio": dominant_area_ratio,
            "num_categories": len(area_by_category),
            "num_annotations": len(anns_by_image.get(image_id, [])),
            "dominant_label_usable": usable,
        }
    return labels


def validate_inputs(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    paths = {
        "policy_decisions_csv": resolve_project_path(config["inputs"]["policy_decisions_csv"]),
        "m0_source_manifest": resolve_project_path(config["inputs"]["m0_source_manifest"]),
        "coco_instances": resolve_project_path(config["inputs"]["coco_instances"]),
        "source_risk_rule_config": resolve_project_path(config["inputs"]["source_risk_rule_config"]),
        "checkpoint": resolve_project_path(config["inputs"]["checkpoint"]),
        "forbidden_checkpoint": resolve_project_path(config["inputs"]["forbidden_checkpoint"]),
        "clip_checkpoint": resolve_project_path(config["clip"]["pretrained_path"]),
    }
    for key, path in paths.items():
        if key == "forbidden_checkpoint":
            continue
        if not path.exists():
            raise FileNotFoundError(f"Required input missing: {key}: {path}")
    if paths["checkpoint"] == paths["forbidden_checkpoint"]:
        raise RuntimeError("Config points to forbidden latest.pt checkpoint.")
    if paths["clip_checkpoint"].stat().st_size < 100 * 1024 * 1024:
        raise RuntimeError(f"CLIP checkpoint is missing or too small: {paths['clip_checkpoint']}")

    missing_images: list[str] = []
    for row in rows:
        for key in ["original", "m0_reconstruction", "refined", "final_source"]:
            if row.get(key):
                path = resolve_project_path(row[key])
                if not path.exists():
                    missing_images.append(project_relative(path))
    if missing_images:
        raise FileNotFoundError("Missing image paths:\n" + "\n".join(sorted(set(missing_images))[:20]))
    return {key: project_relative(path) for key, path in paths.items()}


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_clip_model(config: dict[str, Any], device: torch.device):
    import open_clip

    clip_cfg = config["clip"]
    cache_dir = resolve_project_path(clip_cfg["cache_dir"])
    pretrained_path = resolve_project_path(clip_cfg["pretrained_path"])
    pretrained = str(pretrained_path) if pretrained_path.is_file() else str(clip_cfg["pretrained"])
    model, _train_preprocess, eval_preprocess = open_clip.create_model_and_transforms(
        model_name=str(clip_cfg["model_name"]),
        pretrained=pretrained,
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


@torch.no_grad()
def encode_text_features(
    model: torch.nn.Module,
    tokenizer,
    labels: list[str],
    templates: list[str],
    device: torch.device,
) -> torch.Tensor:
    per_label_features: list[torch.Tensor] = []
    for label in labels:
        texts = [template.format(label=label) for template in templates]
        tokens = tokenizer(texts).to(device)
        features = model.encode_text(tokens).float()
        features = F.normalize(features, dim=-1)
        feature = F.normalize(features.mean(dim=0, keepdim=True), dim=-1)
        per_label_features.append(feature)
    return torch.cat(per_label_features, dim=0)


@torch.no_grad()
def classify_paths(
    model: torch.nn.Module,
    preprocess,
    text_features: torch.Tensor,
    labels: list[str],
    paths: list[Path],
    batch_size: int,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    logit_scale = model.logit_scale.exp().float().clamp(max=100.0)
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        images = torch.stack([preprocess(load_rgb(path)) for path in batch_paths]).to(device)
        image_features = model.encode_image(images).float()
        image_features = F.normalize(image_features, dim=-1)
        logits = logit_scale * image_features @ text_features.T
        probs = torch.softmax(logits, dim=-1)
        values, indices = torch.topk(probs, k=min(5, len(labels)), dim=-1)
        for path, row_values, row_indices in zip(batch_paths, values.cpu(), indices.cpu()):
            top_indices = [int(item) for item in row_indices.tolist()]
            top_probs = [float(item) for item in row_values.tolist()]
            predictions[project_relative(path)] = {
                "top_indices": top_indices,
                "top_labels": [labels[index] for index in top_indices],
                "top_probs": top_probs,
                "top1_label": labels[top_indices[0]],
                "top1_prob": top_probs[0],
                "top1_margin": top_probs[0] - top_probs[1] if len(top_probs) > 1 else top_probs[0],
            }
    return predictions


def unique_image_paths(rows: list[dict[str, Any]]) -> list[Path]:
    seen: set[str] = set()
    paths: list[Path] = []
    for row in rows:
        for key in ["original", "m0_reconstruction", "refined", "final_source"]:
            value = str(row.get(key, ""))
            if value and value not in seen:
                seen.add(value)
                paths.append(resolve_project_path(value))
    return paths


def enrich_rows(
    rows: list[dict[str, Any]],
    labels_by_sample: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    min_prob = float(config["labeling"]["clean_correct_min_prob"])
    min_margin = float(config["labeling"]["clean_correct_min_margin"])
    output: list[dict[str, Any]] = []
    for row in rows:
        label_meta = labels_by_sample.get(str(row["sample"]), {})
        dominant_label = str(label_meta.get("dominant_label", ""))
        label_usable = bool(label_meta.get("dominant_label_usable", False))
        pred_original = predictions[project_relative(resolve_project_path(row["original"]))]
        pred_m0 = predictions[project_relative(resolve_project_path(row["m0_reconstruction"]))]
        pred_refined = predictions[project_relative(resolve_project_path(row["refined"]))]
        pred_final = predictions[project_relative(resolve_project_path(row["final_source"]))]
        original_clean_correct = (
            label_usable
            and pred_original["top1_label"] == dominant_label
            and float(pred_original["top1_prob"]) >= min_prob
            and float(pred_original["top1_margin"]) >= min_margin
        )
        m0_correct = pred_m0["top1_label"] == dominant_label
        refined_correct = pred_refined["top1_label"] == dominant_label
        final_correct = pred_final["top1_label"] == dominant_label
        enriched = dict(row)
        enriched.update(
            {
                "coco_image_id": label_meta.get("coco_image_id", ""),
                "coco_source_path": label_meta.get("source_path", ""),
                "dominant_label": dominant_label,
                "dominant_category_id": label_meta.get("dominant_category_id", ""),
                "dominant_share": label_meta.get("dominant_share", ""),
                "dominant_area_ratio": label_meta.get("dominant_area_ratio", ""),
                "dominant_label_usable": label_usable,
                "clip_original_top1_label": pred_original["top1_label"],
                "clip_original_top1_prob": pred_original["top1_prob"],
                "clip_original_top1_margin": pred_original["top1_margin"],
                "clip_m0_top1_label": pred_m0["top1_label"],
                "clip_m0_top1_prob": pred_m0["top1_prob"],
                "clip_refined_top1_label": pred_refined["top1_label"],
                "clip_refined_top1_prob": pred_refined["top1_prob"],
                "clip_final_top1_label": pred_final["top1_label"],
                "clip_final_top1_prob": pred_final["top1_prob"],
                "original_clean_correct": original_clean_correct,
                "m0_matches_gt": m0_correct,
                "refined_matches_gt": refined_correct,
                "final_matches_gt": final_correct,
                "final_failure_gt": not final_correct,
                "accepted_repair_gt": bool(row["accept_refined"]) and (not m0_correct) and refined_correct,
                "accepted_new_error_gt": bool(row["accept_refined"]) and m0_correct and (not refined_correct),
                "missed_repair_gt": (not bool(row["accept_refined"])) and (not m0_correct) and refined_correct,
                "protective_reject_gt": (not bool(row["accept_refined"])) and m0_correct and (not refined_correct),
            }
        )
        output.append(enriched)
    return output


def summarize(rows: list[dict[str, Any]], policy: str, snr: float | None = None, clean_only: bool = True) -> dict[str, Any]:
    subset = [row for row in rows if row["policy"] == policy and (snr is None or float(row["snr_db"]) == float(snr))]
    if clean_only:
        subset = [row for row in subset if bool(row["original_clean_correct"])]
    return {
        "subset": "clean_correct" if clean_only else "all_labeled",
        "policy": policy,
        "snr_db": "all" if snr is None else float(snr),
        "num_rows": len(subset),
        "accept_count": sum(bool(row["accept_refined"]) for row in subset),
        "accept_rate": rate([bool(row["accept_refined"]) for row in subset]),
        "m0_failure_gt": 1.0 - rate([bool(row["m0_matches_gt"]) for row in subset]),
        "refined_failure_gt": 1.0 - rate([bool(row["refined_matches_gt"]) for row in subset]),
        "final_failure_gt": 1.0 - rate([bool(row["final_matches_gt"]) for row in subset]),
        "accepted_repair_gt_count": sum(bool(row["accepted_repair_gt"]) for row in subset),
        "accepted_new_error_gt_count": sum(bool(row["accepted_new_error_gt"]) for row in subset),
        "missed_repair_gt_count": sum(bool(row["missed_repair_gt"]) for row in subset),
        "protective_reject_gt_count": sum(bool(row["protective_reject_gt"]) for row in subset),
        "final_psnr_db": mean([float(row["final_psnr_db"]) for row in subset]),
        "m0_psnr_db": mean([float(row["m0_psnr_db"]) for row in subset]),
        "refined_psnr_db": mean([float(row["refined_psnr_db"]) for row in subset]),
        "final_delta_psnr_vs_m0_db": mean([float(row["final_psnr_db"]) for row in subset])
        - mean([float(row["m0_psnr_db"]) for row in subset]),
    }


def add_summary_deltas(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed = {(row["subset"], row["policy"], str(row["snr_db"])): row for row in summary_rows}
    output: list[dict[str, Any]] = []
    for row in summary_rows:
        enriched = dict(row)
        for baseline in ["top1_equal", "raw_conf_gain", "selected_risk_rule"]:
            base = keyed.get((row["subset"], baseline, str(row["snr_db"])))
            if not base:
                continue
            enriched[f"delta_final_failure_gt_vs_{baseline}"] = float(row["final_failure_gt"]) - float(
                base["final_failure_gt"]
            )
            enriched[f"delta_final_psnr_vs_{baseline}_db"] = float(row["final_psnr_db"]) - float(
                base["final_psnr_db"]
            )
            enriched[f"delta_accepted_repair_gt_vs_{baseline}"] = int(row["accepted_repair_gt_count"]) - int(
                base["accepted_repair_gt_count"]
            )
            enriched[f"delta_accepted_new_error_gt_vs_{baseline}"] = int(row["accepted_new_error_gt_count"]) - int(
                base["accepted_new_error_gt_count"]
            )
        output.append(enriched)
    return output


def make_grid(rows: list[dict[str, Any]], output_path: Path, count: int) -> None:
    if not rows:
        return
    rows = rows[:count]
    tile = 160
    label_height = 58
    cols = 4
    canvas = Image.new("RGB", (tile * cols, (tile + label_height) * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for row_index, row in enumerate(rows):
        y = row_index * (tile + label_height)
        image_items = [
            ("original", row["original"]),
            ("m0", row["m0_reconstruction"]),
            ("refined", row["refined"]),
            ("final", row["final_source"]),
        ]
        for col, (label, rel_path) in enumerate(image_items):
            x = col * tile
            image = load_rgb(resolve_project_path(rel_path)).resize((tile, tile), Image.Resampling.BICUBIC)
            canvas.paste(image, (x, y + label_height))
            draw.text((x + 4, y + 4), label, fill=(0, 0, 0), font=font)
        line1 = f"{row['policy']} {row['sample']} {snr_name(float(row['snr_db']))} gt={row['dominant_label']}"
        line2 = (
            f"orig={row['clip_original_top1_label']} m0={row['clip_m0_top1_label']} "
            f"ref={row['clip_refined_top1_label']} final={row['clip_final_top1_label']}"
        )
        draw.text((4, y + 18), line1[:122], fill=(0, 0, 0), font=font)
        draw.text((4, y + 34), line2[:122], fill=(0, 0, 0), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def write_galleries(rows: list[dict[str, Any]], config: dict[str, Any], output_dir: Path) -> dict[str, str]:
    count = int(config["evaluation"]["gallery_rows"])
    clean_rows = [row for row in rows if bool(row["original_clean_correct"])]
    groups = {
        "selected_accepted_new_errors_gt": [
            row for row in clean_rows if row["policy"] == "selected_risk_rule" and bool(row["accepted_new_error_gt"])
        ],
        "selected_accepted_repairs_gt": [
            row for row in clean_rows if row["policy"] == "selected_risk_rule" and bool(row["accepted_repair_gt"])
        ],
        "raw_accepted_new_errors_gt": [
            row for row in clean_rows if row["policy"] == "raw_conf_gain" and bool(row["accepted_new_error_gt"])
        ],
        "raw_accepted_repairs_gt": [
            row for row in clean_rows if row["policy"] == "raw_conf_gain" and bool(row["accepted_repair_gt"])
        ],
    }
    manifest: dict[str, str] = {}
    for name, group_rows in groups.items():
        group_rows = sorted(group_rows, key=lambda row: (float(row["snr_db"]), str(row["sample"])))
        path = output_dir / "galleries" / f"{name}.png"
        make_grid(group_rows, path, count)
        if path.exists():
            manifest[name] = project_relative(path)
    return manifest


def make_report(summary_rows: list[dict[str, Any]], metadata: dict[str, Any]) -> str:
    clean_global = [
        row for row in summary_rows if row["subset"] == "clean_correct" and str(row["snr_db"]) == "all"
    ]
    lines = [
        "# EXP-S4-006 Test-Like COCO Object CLIP Clean-Correct Eval",
        "",
        "This auxiliary diagnostic uses COCO instance labels plus CLIP zero-shot classification over the 80 COCO object categories.",
        "The clean-correct subset keeps rows whose original image is classified as the dominant COCO object label.",
        "",
        "## Clean-Correct Overall",
        "",
        "| Policy | Rows | Final Failure GT | Delta vs Top1 | Final PSNR | Delta PSNR vs Top1 | Repair GT | New Error GT |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in clean_global:
        lines.append(
            "| {policy} | {rows} | {fail:.4f} | {dfail:+.4f} | {psnr:.4f} | {dpsnr:+.4f} | {repair} | {newerr} |".format(
                policy=row["policy"],
                rows=int(row["num_rows"]),
                fail=float(row["final_failure_gt"]),
                dfail=float(row.get("delta_final_failure_gt_vs_top1_equal", 0.0)),
                psnr=float(row["final_psnr_db"]),
                dpsnr=float(row.get("delta_final_psnr_vs_top1_equal_db", 0.0)),
                repair=int(row["accepted_repair_gt_count"]),
                newerr=int(row["accepted_new_error_gt_count"]),
            )
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Per-sample CSV: `{metadata['per_sample_csv']}`",
            f"- Summary CSV: `{metadata['summary_csv']}`",
            f"- By-SNR CSV: `{metadata['by_snr_csv']}`",
            f"- Label audit CSV: `{metadata['label_audit_csv']}`",
            f"- Metadata: `{metadata['metadata_json']}`",
            "",
            "## Caveat",
            "",
            "This is not a fully supervised ImageNet/Imagenette metric. It is a COCO-label clean-correct diagnostic with CLIP as the frozen classifier, intended to reduce reliance on ImageNet pseudo-labels.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    policies = [str(item) for item in config["evaluation"]["policies"]]
    policy_rows = load_policy_rows(resolve_project_path(config["inputs"]["policy_decisions_csv"]), policies)
    input_manifest = validate_inputs(config, policy_rows)
    source_paths = load_source_paths(resolve_project_path(config["inputs"]["m0_source_manifest"]))
    labels_by_sample = dominant_coco_labels(config, source_paths)
    sample_names = sorted({str(row["sample"]) for row in policy_rows})
    usable_samples = [name for name in sample_names if labels_by_sample.get(name, {}).get("dominant_label_usable")]

    dry_run_payload = {
        "status": "ok",
        "rows": len(policy_rows),
        "policies": policies,
        "unique_samples": len(sample_names),
        "dominant_label_usable_unique_samples": len(usable_samples),
        "unique_images_to_classify": len(unique_image_paths(policy_rows)),
        "input_manifest": input_manifest,
        "proxy_environment_present": proxy_environment_present(),
        "output_dir": project_relative(resolve_project_path(args.output_dir or config["outputs"]["output_dir"])),
    }
    if args.dry_run:
        print(json.dumps(dry_run_payload, indent=2, ensure_ascii=False))
        return

    output_dir = resolve_project_path(args.output_dir or config["outputs"]["output_dir"])
    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Use --overwrite to replace it.")
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output_dir / "config.yaml")

    device = resolve_device(args.device)
    model, preprocess, tokenizer = load_clip_model(config, device)
    labels = [item["name"] for item in load_json(resolve_project_path(config["inputs"]["coco_instances"]))["categories"]]
    text_features = encode_text_features(
        model=model,
        tokenizer=tokenizer,
        labels=labels,
        templates=[str(item) for item in config["labeling"]["prompt_templates"]],
        device=device,
    )
    predictions = classify_paths(
        model=model,
        preprocess=preprocess,
        text_features=text_features,
        labels=labels,
        paths=unique_image_paths(policy_rows),
        batch_size=int(config["clip"]["batch_size"]),
        device=device,
    )
    enriched_rows = enrich_rows(policy_rows, labels_by_sample, predictions, config)
    snrs = sorted({float(row["snr_db"]) for row in enriched_rows})
    summary_rows = []
    for clean_only in [True, False]:
        summary_rows += [summarize(enriched_rows, policy, None, clean_only=clean_only) for policy in policies]
        summary_rows += [
            summarize(enriched_rows, policy, snr, clean_only=clean_only)
            for policy in policies
            for snr in snrs
        ]
    summary_rows = add_summary_deltas(summary_rows)
    galleries = write_galleries(enriched_rows, config, output_dir)

    per_sample_csv = output_dir / "per_sample.csv"
    summary_csv = output_dir / "summary.csv"
    by_snr_csv = output_dir / "by_snr.csv"
    label_audit_csv = output_dir / "label_audit.csv"
    metadata_json = output_dir / "metadata.json"
    report_md = output_dir / "REPORT.md"
    write_csv(per_sample_csv, enriched_rows)
    write_csv(summary_csv, [row for row in summary_rows if str(row["snr_db"]) == "all"])
    write_csv(by_snr_csv, [row for row in summary_rows if str(row["snr_db"]) != "all"])
    write_csv(label_audit_csv, [labels_by_sample[name] for name in sorted(labels_by_sample) if name in sample_names])
    metadata = {
        "analysis_id": config["analysis_id"],
        "source_experiment": config["source_experiment"],
        "config": project_relative(config_path),
        "copied_config": project_relative(output_dir / "config.yaml"),
        "output_dir": project_relative(output_dir),
        "per_sample_csv": project_relative(per_sample_csv),
        "summary_csv": project_relative(summary_csv),
        "by_snr_csv": project_relative(by_snr_csv),
        "label_audit_csv": project_relative(label_audit_csv),
        "metadata_json": project_relative(metadata_json),
        "report_md": project_relative(report_md),
        "galleries": galleries,
        "inputs": input_manifest,
        "dry_run_payload": dry_run_payload,
        "clean_correct_min_prob": float(config["labeling"]["clean_correct_min_prob"]),
        "clean_correct_min_margin": float(config["labeling"]["clean_correct_min_margin"]),
        "dominant_category_min_area_ratio": float(config["labeling"]["dominant_category_min_area_ratio"]),
        "dominant_category_min_share": float(config["labeling"]["dominant_category_min_share"]),
        "git_commit": git_commit(),
        "git_dirty_state": git_dirty_state(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "proxy_environment_present": proxy_environment_present(),
        "download_note": "No download is required; COCO annotations and CLIP checkpoint are loaded from local files.",
    }
    save_json(metadata_json, metadata)
    report_md.write_text(make_report(summary_rows, metadata), encoding="utf-8")
    print(json.dumps({"output_dir": project_relative(output_dir), "report": project_relative(report_md)}, indent=2))


if __name__ == "__main__":
    main()
