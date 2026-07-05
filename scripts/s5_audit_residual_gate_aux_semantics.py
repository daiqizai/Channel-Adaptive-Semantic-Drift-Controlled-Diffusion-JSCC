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
import time
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
        description="Audit EXP-S4-006 confidence-gain gate candidate with CLIP image and COCO caption signals."
    )
    parser.add_argument("--config", default="configs/s5_residual_gate_aux_audit_exp_s4_006.yaml")
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


def snr_name(snr: float) -> str:
    if float(snr).is_integer():
        return f"snr_{int(snr):02d}db"
    return f"snr_{str(snr).replace('.', 'p')}db"


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_list(value: Any, cast) -> list[Any]:
    if isinstance(value, list):
        return [cast(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    return [cast(item) for item in text.split("|")]


def read_gate_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["snr_db"] = float(row["snr_db"])
        row["m0_psnr_db"] = float(row["m0_psnr_db"])
        row["refined_psnr_db"] = float(row["refined_psnr_db"])
        for key in [
            "original_top1_index",
            "m0_top1_index",
            "refined_top1_index",
        ]:
            row[key] = int(row[key])
        for key in [
            "original_top1_prob",
            "m0_top1_prob",
            "refined_top1_prob",
        ]:
            row[key] = float(row[key])
        for key in [
            "m0_matches_original_top1",
            "refined_matches_original_top1",
            "refined_matches_m0_top1",
            "detector_accept_refined",
        ]:
            row[key] = parse_bool(row[key])
        for prefix in ["original", "m0", "refined"]:
            row[f"{prefix}_top_indices"] = parse_list(row.get(f"{prefix}_top_indices", ""), int)
            row[f"{prefix}_top_probs"] = parse_list(row.get(f"{prefix}_top_probs", ""), float)
            row[f"{prefix}_top_labels"] = parse_list(row.get(f"{prefix}_top_labels", ""), str)
    return rows


def load_coco_captions(path: Path) -> dict[int, list[str]]:
    payload = load_json(path)
    captions_by_id: dict[int, list[str]] = {}
    for item in payload.get("annotations", []):
        caption = str(item.get("caption", "")).strip()
        if caption:
            captions_by_id.setdefault(int(item["image_id"]), []).append(caption)
    return captions_by_id


def validate_inputs(config: dict[str, Any]) -> dict[str, Any]:
    inputs = config["inputs"]
    paths = {
        "gate_predictions_csv": resolve_project_path(inputs["gate_predictions_csv"]),
        "gate_policy_summary_csv": resolve_project_path(inputs["gate_policy_summary_csv"]),
        "m0_source_manifest": resolve_project_path(inputs["m0_source_manifest"]),
        "coco_captions": resolve_project_path(inputs["coco_captions"]),
        "checkpoint": resolve_project_path(inputs["checkpoint"]),
        "forbidden_checkpoint": resolve_project_path(inputs["forbidden_checkpoint"]),
        "clip_checkpoint": resolve_project_path(config["clip"]["pretrained_path"]),
    }
    for key, path in paths.items():
        if key == "forbidden_checkpoint":
            continue
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {key}: {path}")
    if paths["checkpoint"] == paths["forbidden_checkpoint"]:
        raise RuntimeError("Config points to forbidden latest.pt checkpoint.")
    if paths["clip_checkpoint"].stat().st_size < 100 * 1024 * 1024:
        raise RuntimeError(f"CLIP checkpoint is missing or too small: {paths['clip_checkpoint']}")
    return {key: project_relative(path) for key, path in paths.items()}


def check_rows_have_files(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        for key in ["original", "m0_reconstruction", "refined"]:
            path = resolve_project_path(row[key])
            if not path.exists():
                raise FileNotFoundError(f"Image path from CSV not found: {path}")


def load_clip_model(config: dict[str, Any], device: torch.device):
    import open_clip

    clip_cfg = config["clip"]
    cache_dir = resolve_project_path(clip_cfg["cache_dir"])
    pretrained_path = resolve_project_path(clip_cfg["pretrained_path"])
    model, _train_preprocess, eval_preprocess = open_clip.create_model_and_transforms(
        model_name=str(clip_cfg["model_name"]),
        pretrained=str(pretrained_path),
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


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


@torch.no_grad()
def encode_paths(
    model: torch.nn.Module,
    preprocess,
    paths: list[Path],
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], float]:
    features: dict[str, torch.Tensor] = {}
    elapsed = 0.0
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
        for path, feature in zip(batch_paths, encoded.detach().cpu()):
            features[project_relative(path)] = feature
    return features, elapsed


@torch.no_grad()
def encode_texts(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    outputs: list[torch.Tensor] = []
    elapsed = 0.0
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        tokens = tokenizer(batch_texts).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        begin = time.perf_counter()
        encoded = model.encode_text(tokens)
        encoded = F.normalize(encoded.float(), dim=-1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed += time.perf_counter() - begin
        outputs.append(encoded.detach().cpu())
    return torch.cat(outputs, dim=0), elapsed


def build_caption_features(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, Any]], float]:
    source_paths = [str(item) for item in load_json(resolve_project_path(config["inputs"]["m0_source_manifest"]))["paths"]]
    captions_by_id = load_coco_captions(resolve_project_path(config["inputs"]["coco_captions"]))
    unique_samples = sorted({str(row["sample"]) for row in rows})
    texts: list[str] = []
    sample_slices: dict[str, tuple[int, int]] = {}
    sample_metadata: dict[str, dict[str, Any]] = {}
    for sample in unique_samples:
        idx = sample_index(sample)
        source_path = source_paths[idx]
        image_id = coco_image_id_from_path(source_path)
        captions = captions_by_id.get(image_id, [])
        if not captions:
            raise RuntimeError(f"No COCO captions found for sample={sample}, image_id={image_id}")
        start = len(texts)
        texts.extend(captions)
        sample_slices[sample] = (start, len(texts))
        sample_metadata[sample] = {
            "sample_index": idx,
            "coco_image_id": image_id,
            "coco_source_path": source_path,
            "num_captions": len(captions),
            "captions": captions,
        }
    features, elapsed = encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        batch_size=int(config["clip"]["text_batch_size"]),
        device=device,
    )
    caption_features = {
        sample: features[start:end]
        for sample, (start, end) in sample_slices.items()
    }
    return caption_features, sample_metadata, elapsed


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a * b).sum().item())


def caption_max(image_feature: torch.Tensor, caption_features: torch.Tensor) -> float:
    return float((caption_features @ image_feature).max().item())


def baseline_accept(row: dict[str, Any]) -> bool:
    return int(row["refined_top1_index"]) == int(row["m0_top1_index"])


def candidate_accept(row: dict[str, Any], margin: float) -> bool:
    return baseline_accept(row) or float(row["refined_top1_prob"]) >= float(row["m0_top1_prob"]) + margin


def final_value(row: dict[str, Any], accept: bool, key_prefix: str) -> Any:
    return row[f"refined_{key_prefix}"] if accept else row[f"m0_{key_prefix}"]


def audit_rows(
    rows: list[dict[str, Any]],
    image_features: dict[str, torch.Tensor],
    caption_features: dict[str, torch.Tensor],
    margin: float,
    epsilon: float,
) -> list[dict[str, Any]]:
    audited: list[dict[str, Any]] = []
    for row in rows:
        original_path = str(row["original"])
        m0_path = str(row["m0_reconstruction"])
        refined_path = str(row["refined"])
        original_feature = image_features[original_path]
        m0_feature = image_features[m0_path]
        refined_feature = image_features[refined_path]
        captions = caption_features[str(row["sample"])]

        base_accept = baseline_accept(row)
        cand_accept = candidate_accept(row, margin)
        m0_ok = bool(row["m0_matches_original_top1"])
        refined_ok = bool(row["refined_matches_original_top1"])
        base_final_ok = refined_ok if base_accept else m0_ok
        cand_final_ok = refined_ok if cand_accept else m0_ok
        base_feature = refined_feature if base_accept else m0_feature
        cand_feature = refined_feature if cand_accept else m0_feature
        base_psnr = float(row["refined_psnr_db"] if base_accept else row["m0_psnr_db"])
        cand_psnr = float(row["refined_psnr_db"] if cand_accept else row["m0_psnr_db"])

        clip_m0 = cosine(original_feature, m0_feature)
        clip_refined = cosine(original_feature, refined_feature)
        clip_base = cosine(original_feature, base_feature)
        clip_cand = cosine(original_feature, cand_feature)
        caption_m0 = caption_max(m0_feature, captions)
        caption_refined = caption_max(refined_feature, captions)
        caption_base = caption_max(base_feature, captions)
        caption_cand = caption_max(cand_feature, captions)

        new_accept = cand_accept and not base_accept
        if new_accept and (not m0_ok) and refined_ok:
            case_type = "new_accept_repair"
        elif new_accept and m0_ok and (not refined_ok):
            case_type = "new_accept_new_error"
        elif new_accept and m0_ok and refined_ok:
            case_type = "new_accept_both_correct"
        elif new_accept:
            case_type = "new_accept_both_wrong"
        elif base_accept:
            case_type = "baseline_accept"
        else:
            case_type = "still_reject"

        clip_delta = clip_cand - clip_base
        caption_delta = caption_cand - caption_base
        audited.append(
            {
                "snr_db": float(row["snr_db"]),
                "sample": row["sample"],
                "case_type": case_type,
                "baseline_accept_refined": base_accept,
                "candidate_accept_refined": cand_accept,
                "newly_accepted_by_candidate": new_accept,
                "m0_matches_original_top1": m0_ok,
                "refined_matches_original_top1": refined_ok,
                "baseline_final_matches_original_top1": base_final_ok,
                "candidate_final_matches_original_top1": cand_final_ok,
                "candidate_accepted_repair": cand_accept and (not m0_ok) and refined_ok,
                "candidate_accepted_new_error": cand_accept and m0_ok and (not refined_ok),
                "m0_top1_label": row["m0_top1_label"],
                "m0_top1_prob": row["m0_top1_prob"],
                "refined_top1_label": row["refined_top1_label"],
                "refined_top1_prob": row["refined_top1_prob"],
                "refined_conf_gain_vs_m0": float(row["refined_top1_prob"]) - float(row["m0_top1_prob"]),
                "original": row["original"],
                "m0_reconstruction": row["m0_reconstruction"],
                "refined": row["refined"],
                "m0_psnr_db": row["m0_psnr_db"],
                "refined_psnr_db": row["refined_psnr_db"],
                "baseline_final_psnr_db": base_psnr,
                "candidate_final_psnr_db": cand_psnr,
                "candidate_delta_psnr_vs_baseline_db": cand_psnr - base_psnr,
                "clip_sim_original_m0": clip_m0,
                "clip_sim_original_refined": clip_refined,
                "clip_sim_original_baseline_final": clip_base,
                "clip_sim_original_candidate_final": clip_cand,
                "candidate_delta_clip_vs_baseline": clip_delta,
                "caption_max_m0": caption_m0,
                "caption_max_refined": caption_refined,
                "caption_max_baseline_final": caption_base,
                "caption_max_candidate_final": caption_cand,
                "candidate_delta_caption_vs_baseline": caption_delta,
                "aux_clip_nonworse": clip_delta >= -epsilon,
                "aux_caption_nonworse": caption_delta >= -epsilon,
                "aux_both_nonworse": clip_delta >= -epsilon and caption_delta >= -epsilon,
            }
        )
    return audited


def serialize_value(value: Any) -> Any:
    if isinstance(value, bool):
        return bool_text(value)
    if isinstance(value, (list, dict)):
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_value(row.get(key, "")) for key in fieldnames})


def summarize_subset(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return {
        "subset": name,
        "num_images": len(rows),
        "candidate_accept_rate": rate([bool(row["candidate_accept_refined"]) for row in rows]),
        "new_accept_count": sum(bool(row["newly_accepted_by_candidate"]) for row in rows),
        "candidate_final_failure_rate": 1.0
        - rate([bool(row["candidate_final_matches_original_top1"]) for row in rows]),
        "baseline_final_failure_rate": 1.0
        - rate([bool(row["baseline_final_matches_original_top1"]) for row in rows]),
        "candidate_minus_baseline_failure": (
            1.0 - rate([bool(row["candidate_final_matches_original_top1"]) for row in rows])
        )
        - (
            1.0 - rate([bool(row["baseline_final_matches_original_top1"]) for row in rows])
        ),
        "candidate_final_psnr_db": mean([float(row["candidate_final_psnr_db"]) for row in rows]),
        "baseline_final_psnr_db": mean([float(row["baseline_final_psnr_db"]) for row in rows]),
        "candidate_delta_psnr_vs_baseline_db": mean(
            [float(row["candidate_delta_psnr_vs_baseline_db"]) for row in rows]
        ),
        "candidate_delta_clip_vs_baseline": mean(
            [float(row["candidate_delta_clip_vs_baseline"]) for row in rows]
        ),
        "candidate_delta_caption_vs_baseline": mean(
            [float(row["candidate_delta_caption_vs_baseline"]) for row in rows]
        ),
        "new_accept_clip_nonworse_rate": rate(
            [
                bool(row["aux_clip_nonworse"])
                for row in rows
                if bool(row["newly_accepted_by_candidate"])
            ]
        ),
        "new_accept_caption_nonworse_rate": rate(
            [
                bool(row["aux_caption_nonworse"])
                for row in rows
                if bool(row["newly_accepted_by_candidate"])
            ]
        ),
        "new_accept_both_aux_nonworse_rate": rate(
            [
                bool(row["aux_both_nonworse"])
                for row in rows
                if bool(row["newly_accepted_by_candidate"])
            ]
        ),
        "accepted_repair_count": sum(bool(row["candidate_accepted_repair"]) for row in rows),
        "accepted_new_error_count": sum(bool(row["candidate_accepted_new_error"]) for row in rows),
    }


def make_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = [summarize_subset(rows, "all")]
    for snr in sorted({float(row["snr_db"]) for row in rows}):
        summaries.append(summarize_subset([row for row in rows if float(row["snr_db"]) == snr], snr_name(snr)))
    for case_type in [
        "new_accept_repair",
        "new_accept_new_error",
        "new_accept_both_correct",
        "new_accept_both_wrong",
    ]:
        subset = [row for row in rows if row["case_type"] == case_type]
        summaries.append(summarize_subset(subset, case_type))
    return summaries


def gallery_sort_key(row: dict[str, Any]) -> tuple[float, float]:
    return (
        float(row["candidate_delta_clip_vs_baseline"]) + float(row["candidate_delta_caption_vs_baseline"]),
        float(row["candidate_delta_psnr_vs_baseline_db"]),
    )


def make_quad(row: dict[str, Any], output_path: Path) -> None:
    image_paths = [
        ("original", resolve_project_path(row["original"])),
        ("m0", resolve_project_path(row["m0_reconstruction"])),
        ("refined", resolve_project_path(row["refined"])),
    ]
    final_label = "candidate_final"
    final_path = image_paths[2][1] if bool(row["candidate_accept_refined"]) else image_paths[1][1]
    image_paths.append((final_label, final_path))
    images = [load_image(path).resize((192, 192), Image.Resampling.BICUBIC) for _, path in image_paths]
    label_height = 42
    canvas = Image.new("RGB", (192 * len(images), 192 + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for idx, ((label, _path), image) in enumerate(zip(image_paths, images)):
        x = idx * 192
        canvas.paste(image, (x, label_height))
        draw.text((x + 4, 4), label, fill=(0, 0, 0), font=font)
    detail = (
        f"{row['case_type']} snr={float(row['snr_db']):g} "
        f"dClip={float(row['candidate_delta_clip_vs_baseline']):+.3f} "
        f"dCap={float(row['candidate_delta_caption_vs_baseline']):+.3f}"
    )
    draw.text((4, 22), detail[:120], fill=(0, 0, 0), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def write_galleries(rows: list[dict[str, Any]], output_dir: Path, count: int) -> dict[str, list[str]]:
    gallery_root = output_dir / "galleries"
    manifest: dict[str, list[str]] = {}
    for case_type in [
        "new_accept_repair",
        "new_accept_new_error",
        "new_accept_both_correct",
        "new_accept_both_wrong",
    ]:
        subset = sorted(
            [row for row in rows if row["case_type"] == case_type],
            key=gallery_sort_key,
        )
        paths: list[str] = []
        for idx, row in enumerate(subset[:count]):
            output_path = gallery_root / case_type / f"{idx:02d}_{snr_name(float(row['snr_db']))}_{row['sample']}"
            make_quad(row, output_path)
            paths.append(project_relative(output_path))
        manifest[case_type] = paths
    return manifest


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def make_report(
    summaries: list[dict[str, Any]],
    gallery_manifest: dict[str, list[str]],
    metadata: dict[str, Any],
) -> str:
    all_row = next(row for row in summaries if row["subset"] == "all")
    case_rows = [
        row
        for row in summaries
        if str(row["subset"]).startswith("new_accept_")
    ]
    lines = [
        "# EXP-S4-006 Confidence-Gain Gate Auxiliary Audit",
        "",
        "This derived analysis audits `top1_equal_or_refined_conf_gain_ge_0p05` against the original `top1_equal` gate using CLIP image-image similarity and COCO caption CLIP image-text scores.",
        "",
        "Decision-time inputs remain receiver-side only: M0 and refined classifier predictions. Original images and captions are used only for offline audit.",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Candidate accept rate | {fmt(all_row['candidate_accept_rate'])} |",
        f"| Newly accepted by candidate | {all_row['new_accept_count']} |",
        f"| Candidate final failure | {fmt(all_row['candidate_final_failure_rate'])} |",
        f"| Baseline final failure | {fmt(all_row['baseline_final_failure_rate'])} |",
        f"| Candidate minus baseline failure | {fmt(all_row['candidate_minus_baseline_failure'])} |",
        f"| Candidate delta PSNR vs baseline | {fmt(all_row['candidate_delta_psnr_vs_baseline_db'])} dB |",
        f"| Candidate delta CLIP vs baseline | {fmt(all_row['candidate_delta_clip_vs_baseline'])} |",
        f"| Candidate delta caption vs baseline | {fmt(all_row['candidate_delta_caption_vs_baseline'])} |",
        "",
        "## Newly Accepted Breakdown",
        "",
        "| Subset | N | Accepted repair | Accepted new error | Delta PSNR | Delta CLIP | Delta caption | Aux both nonworse |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in case_rows:
        lines.append(
            "| {subset} | {n} | {repair} | {new_error} | {dpsnr} | {dclip} | {dcap} | {both} |".format(
                subset=row["subset"],
                n=row["num_images"],
                repair=row["accepted_repair_count"],
                new_error=row["accepted_new_error_count"],
                dpsnr=fmt(row["candidate_delta_psnr_vs_baseline_db"]),
                dclip=fmt(row["candidate_delta_clip_vs_baseline"]),
                dcap=fmt(row["candidate_delta_caption_vs_baseline"]),
                both=fmt(row["new_accept_both_aux_nonworse_rate"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Positive candidate evidence: the confidence-gain gate accepts many samples that baseline top-1 agreement rejects, including pseudo-label repairs.",
            "- Risk evidence: accepted new errors are explicitly retained as a separate subset and must be visually reviewed before using the policy as final M3.",
            "- CLIP/caption deltas are auxiliary diagnostics; they do not replace the frozen classifier final-failure metric.",
            "",
            "## Galleries",
            "",
        ]
    )
    for case_type, paths in gallery_manifest.items():
        lines.append(f"- `{case_type}`: {len(paths)} quads")
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Per-sample audit: `{metadata['per_sample_audit_csv']}`",
            f"- Summary: `{metadata['summary_csv']}`",
            f"- Newly accepted: `{metadata['new_accepts_csv']}`",
            f"- Accepted new errors: `{metadata['accepted_new_errors_csv']}`",
            f"- Metadata: `{metadata['metadata_json']}`",
            "",
        ]
    )
    return "\n".join(lines)


def get_project_version() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return "N/A"


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    manifest = validate_inputs(config)
    rows = read_gate_rows(resolve_project_path(config["inputs"]["gate_predictions_csv"]))
    check_rows_have_files(rows)
    if args.dry_run:
        print(json.dumps({"status": "ok", "num_rows": len(rows), "manifest": manifest}, indent=2))
        return

    output_dir = resolve_project_path(args.output_dir or config["outputs"]["output_dir"])
    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory already exists and is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")

    device = resolve_device(args.device)
    model, preprocess, tokenizer = load_clip_model(config, device)
    unique_image_paths = sorted(
        {
            resolve_project_path(row[key])
            for row in rows
            for key in ["original", "m0_reconstruction", "refined"]
        }
    )
    image_features, image_seconds = encode_paths(
        model=model,
        preprocess=preprocess,
        paths=unique_image_paths,
        batch_size=int(config["clip"]["batch_size"]),
        device=device,
    )
    caption_features, sample_metadata, text_seconds = build_caption_features(rows, config, model, tokenizer, device)
    audited = audit_rows(
        rows=rows,
        image_features=image_features,
        caption_features=caption_features,
        margin=float(config["policy"]["refined_conf_gain_margin"]),
        epsilon=float(config["evaluation"].get("auxiliary_nonworse_epsilon", 0.0)),
    )
    summaries = make_summaries(audited)
    new_accepts = [row for row in audited if bool(row["newly_accepted_by_candidate"])]
    accepted_new_errors = [row for row in audited if bool(row["candidate_accepted_new_error"])]
    gallery_manifest = write_galleries(
        new_accepts,
        output_dir,
        count=int(config["evaluation"]["gallery_cases_per_type"]),
    )

    per_sample_csv = output_dir / "per_sample_audit.csv"
    summary_csv = output_dir / "summary.csv"
    new_accepts_csv = output_dir / "new_accepts.csv"
    accepted_new_errors_csv = output_dir / "accepted_new_errors.csv"
    sample_metadata_json = output_dir / "sample_metadata.json"
    metadata_json = output_dir / "metadata.json"
    report_md = output_dir / "REPORT.md"
    write_csv(per_sample_csv, audited)
    write_csv(summary_csv, summaries)
    write_csv(new_accepts_csv, new_accepts)
    write_csv(accepted_new_errors_csv, accepted_new_errors)
    save_json(sample_metadata_json, sample_metadata)

    metadata = {
        "project_version": get_project_version(),
        "config": project_relative(config_path),
        "output_dir": project_relative(output_dir),
        "per_sample_audit_csv": project_relative(per_sample_csv),
        "summary_csv": project_relative(summary_csv),
        "new_accepts_csv": project_relative(new_accepts_csv),
        "accepted_new_errors_csv": project_relative(accepted_new_errors_csv),
        "sample_metadata_json": project_relative(sample_metadata_json),
        "metadata_json": project_relative(metadata_json),
        "report_md": project_relative(report_md),
        "run_command": " ".join(sys.argv),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "num_rows": len(rows),
        "num_unique_images_encoded": len(unique_image_paths),
        "num_unique_samples": len(sample_metadata),
        "clip_image_seconds": image_seconds,
        "clip_text_seconds": text_seconds,
        "policy": config["policy"],
        "clip": config["clip"],
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "proxy_environment_present": sorted(key for key in os.environ if "proxy" in key.lower()),
        "download_note": "No model or data download is required; CLIP weights and COCO captions are loaded from local project files.",
    }
    save_json(metadata_json, metadata)
    report_md.write_text(make_report(summaries, gallery_manifest, metadata), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": project_relative(output_dir),
                "num_rows": len(rows),
                "new_accept_count": len(new_accepts),
                "accepted_new_error_count": len(accepted_new_errors),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
