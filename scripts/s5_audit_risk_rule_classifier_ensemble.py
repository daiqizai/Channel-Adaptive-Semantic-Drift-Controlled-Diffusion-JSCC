from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit selected_risk_rule with multiple frozen torchvision classifiers."
    )
    parser.add_argument("--config", default="configs/s5_risk_rule_classifier_ensemble_audit_exp_s4_006.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
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


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def snr_name(snr: float) -> str:
    if float(snr).is_integer():
        return f"snr_{int(snr):02d}db"
    return f"snr_{str(snr).replace('.', 'p')}db"


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def rate(flags: list[bool]) -> float:
    return float(sum(flags) / len(flags)) if flags else 0.0


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        out["snr_db"] = float(out["snr_db"])
        for key in [
            "accept_refined",
            "baseline_accept_refined",
            "candidate_accept_refined",
            "new_accept_vs_top1",
            "vetoed_candidate_accept",
            "shadow_veto",
            "final_matches_original_top1",
            "accepted_repair",
            "missed_repair",
            "accepted_new_error",
            "protective_reject",
            "vetoed_candidate_repair",
            "vetoed_candidate_new_error",
            "top1_equal_accept_refined",
            "top1_equal_final_matches_original_top1",
        ]:
            if key in out:
                out[key] = parse_bool(out[key])
        for key in [
            "final_psnr_db",
            "top1_equal_final_psnr_db",
            "delta_final_psnr_vs_top1_equal_db",
            "clip_sim_m0_refined",
            "m0_top1_margin",
            "refined_top1_margin",
            "m0_top1_prob",
            "refined_top1_prob",
        ]:
            if key in out and out[key] != "":
                out[key] = float(out[key])
        output.append(out)
    return output


def validate_inputs(config: dict[str, Any]) -> dict[str, str]:
    paths = {
        "selected_policy_per_sample_csv": resolve_project_path(
            config["inputs"]["selected_policy_per_sample_csv"]
        ),
        "selected_policy_summary_csv": resolve_project_path(
            config["inputs"]["selected_policy_summary_csv"]
        ),
        "source_risk_rule_decisions_csv": resolve_project_path(
            config["inputs"]["source_risk_rule_decisions_csv"]
        ),
        "source_risk_rule_config": resolve_project_path(config["inputs"]["source_risk_rule_config"]),
        "checkpoint": resolve_project_path(config["inputs"]["checkpoint"]),
        "forbidden_checkpoint": resolve_project_path(config["inputs"]["forbidden_checkpoint"]),
    }
    for key, path in paths.items():
        if key == "forbidden_checkpoint":
            continue
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {key}: {path}")
    if paths["checkpoint"] == paths["forbidden_checkpoint"]:
        raise RuntimeError("Config points to forbidden latest.pt checkpoint.")
    return {key: project_relative(path) for key, path in paths.items()}


def classifier_status(config: dict[str, Any]) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for model_cfg in config["classifiers"]["models"]:
        weights_file = resolve_project_path(model_cfg["weights_file"])
        statuses.append(
            {
                "key": model_cfg["key"],
                "model_name": model_cfg["model_name"],
                "weights": model_cfg["weights"],
                "weights_file": project_relative(weights_file),
                "available": weights_file.is_file() and weights_file.stat().st_size > 1024 * 1024,
                "bytes": weights_file.stat().st_size if weights_file.exists() else 0,
            }
        )
    return statuses


def validate_source_images(rows: list[dict[str, Any]]) -> list[str]:
    missing: list[str] = []
    for row in rows:
        for key in ["original", "m0_reconstruction", "refined"]:
            path = resolve_project_path(row[key])
            if not path.exists():
                missing.append(project_relative(path))
    return sorted(set(missing))


def load_classifier(model_cfg: dict[str, Any], config: dict[str, Any], device: torch.device, allow_download: bool):
    weights_file = resolve_project_path(model_cfg["weights_file"])
    if (not weights_file.exists() or weights_file.stat().st_size < 1024 * 1024) and not allow_download:
        raise RuntimeError(f"Classifier weights missing from local cache: {weights_file}")
    cache_dir = resolve_project_path(config["classifiers"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(cache_dir))

    import torchvision.models as models

    weights_enum = getattr(models, str(model_cfg["weights_enum"]))
    weights = getattr(weights_enum, str(model_cfg["weights"]))
    builder = getattr(models, str(model_cfg["model_name"]))
    model = builder(weights=weights).to(device)
    model.eval()
    return model, weights.transforms(), list(weights.meta["categories"])


@torch.no_grad()
def classify_paths(
    model: torch.nn.Module,
    preprocess,
    paths: list[Path],
    batch_size: int,
    topk: int,
    device: torch.device,
) -> tuple[dict[str, dict[str, Any]], float]:
    predictions: dict[str, dict[str, Any]] = {}
    elapsed = 0.0
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        images = torch.stack([preprocess(load_rgb(path)) for path in batch_paths]).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        begin = time.perf_counter()
        logits = model(images)
        probs = torch.softmax(logits.float(), dim=-1)
        values, indices = torch.topk(probs, k=topk, dim=-1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed += time.perf_counter() - begin
        for path, row_values, row_indices in zip(batch_paths, values.cpu(), indices.cpu()):
            predictions[project_relative(path)] = {
                "top_indices": [int(item) for item in row_indices.tolist()],
                "top_probs": [float(item) for item in row_values.tolist()],
            }
    return predictions, elapsed


def unique_image_paths(rows: list[dict[str, Any]]) -> list[Path]:
    seen: set[str] = set()
    output: list[Path] = []
    for row in rows:
        for key in ["original", "m0_reconstruction", "refined"]:
            rel = str(row[key])
            if rel not in seen:
                seen.add(rel)
                output.append(resolve_project_path(rel))
    return output


def label_for(categories: list[str], index: int) -> str:
    if 0 <= index < len(categories):
        return categories[index]
    return f"class_{index}"


def eval_model_rows(
    rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    categories: list[str],
    model_key: str,
    margin: float,
) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        pred_by_key = {
            "original": predictions[str(row["original"])],
            "m0": predictions[str(row["m0_reconstruction"])],
            "refined": predictions[str(row["refined"])],
        }
        indices = {key: list(value["top_indices"]) for key, value in pred_by_key.items()}
        probs = {key: list(value["top_probs"]) for key, value in pred_by_key.items()}
        original_top1 = int(indices["original"][0])
        m0_top1 = int(indices["m0"][0])
        refined_top1 = int(indices["refined"][0])
        m0_ok = m0_top1 == original_top1
        refined_ok = refined_top1 == original_top1
        selected_accept = bool(row["accept_refined"])
        selected_ok = refined_ok if selected_accept else m0_ok
        model_top1_accept = refined_top1 == m0_top1
        model_top1_ok = refined_ok if model_top1_accept else m0_ok
        raw_conf_gain_accept = model_top1_accept or float(probs["refined"][0]) >= float(probs["m0"][0]) + margin
        raw_conf_gain_ok = refined_ok if raw_conf_gain_accept else m0_ok
        out = {
            "classifier": model_key,
            "split": row["split"],
            "snr_db": float(row["snr_db"]),
            "sample": row["sample"],
            "selected_accept_refined": selected_accept,
            "selected_new_accept_vs_alexnet_top1": bool(row["new_accept_vs_top1"]),
            "selected_shadow_veto": bool(row["shadow_veto"]),
            "original": row["original"],
            "m0_reconstruction": row["m0_reconstruction"],
            "refined": row["refined"],
            "materialized_final": row["materialized_final"],
            "original_top1_index": original_top1,
            "m0_top1_index": m0_top1,
            "refined_top1_index": refined_top1,
            "original_top1_label": label_for(categories, original_top1),
            "m0_top1_label": label_for(categories, m0_top1),
            "refined_top1_label": label_for(categories, refined_top1),
            "original_top1_prob": float(probs["original"][0]),
            "m0_top1_prob": float(probs["m0"][0]),
            "refined_top1_prob": float(probs["refined"][0]),
            "m0_matches_original_top1": m0_ok,
            "refined_matches_original_top1": refined_ok,
            "refined_matches_m0_top1": refined_top1 == m0_top1,
            "selected_final_matches_original_top1": selected_ok,
            "selected_accepted_repair": selected_accept and (not m0_ok) and refined_ok,
            "selected_accepted_new_error": selected_accept and m0_ok and (not refined_ok),
            "selected_accepted_both_wrong": selected_accept and (not m0_ok) and (not refined_ok),
            "selected_missed_repair": (not selected_accept) and (not m0_ok) and refined_ok,
            "selected_protective_reject": (not selected_accept) and m0_ok and (not refined_ok),
            "model_top1_accept_refined": model_top1_accept,
            "model_top1_final_matches_original_top1": model_top1_ok,
            "raw_conf_gain_accept_refined": raw_conf_gain_accept,
            "raw_conf_gain_final_matches_original_top1": raw_conf_gain_ok,
            "raw_conf_gain_accepted_repair": raw_conf_gain_accept and (not m0_ok) and refined_ok,
            "raw_conf_gain_accepted_new_error": raw_conf_gain_accept and m0_ok and (not refined_ok),
        }
        out_rows.append(out)
    return out_rows


def summarize(rows: list[dict[str, Any]], level: str, classifier: str, split: str, snr: str) -> dict[str, Any]:
    selected_matches = [bool(row["selected_final_matches_original_top1"]) for row in rows]
    m0_matches = [bool(row["m0_matches_original_top1"]) for row in rows]
    refined_matches = [bool(row["refined_matches_original_top1"]) for row in rows]
    top1_matches = [bool(row["model_top1_final_matches_original_top1"]) for row in rows]
    raw_matches = [bool(row["raw_conf_gain_final_matches_original_top1"]) for row in rows]
    return {
        "level": level,
        "classifier": classifier,
        "split": split,
        "snr_db": snr,
        "num_images": len(rows),
        "m0_failure_rate": 1.0 - rate(m0_matches),
        "refined_failure_rate": 1.0 - rate(refined_matches),
        "selected_final_failure_rate": 1.0 - rate(selected_matches),
        "model_top1_final_failure_rate": 1.0 - rate(top1_matches),
        "raw_conf_gain_failure_rate": 1.0 - rate(raw_matches),
        "delta_selected_failure_vs_m0": (1.0 - rate(selected_matches)) - (1.0 - rate(m0_matches)),
        "delta_selected_failure_vs_model_top1": (1.0 - rate(selected_matches)) - (1.0 - rate(top1_matches)),
        "delta_selected_failure_vs_raw_conf_gain": (1.0 - rate(selected_matches)) - (1.0 - rate(raw_matches)),
        "selected_accept_count": sum(bool(row["selected_accept_refined"]) for row in rows),
        "model_top1_accept_count": sum(bool(row["model_top1_accept_refined"]) for row in rows),
        "raw_conf_gain_accept_count": sum(bool(row["raw_conf_gain_accept_refined"]) for row in rows),
        "selected_accepted_repair_count": sum(bool(row["selected_accepted_repair"]) for row in rows),
        "selected_accepted_new_error_count": sum(bool(row["selected_accepted_new_error"]) for row in rows),
        "selected_missed_repair_count": sum(bool(row["selected_missed_repair"]) for row in rows),
        "selected_protective_reject_count": sum(bool(row["selected_protective_reject"]) for row in rows),
        "raw_conf_gain_accepted_repair_count": sum(bool(row["raw_conf_gain_accepted_repair"]) for row in rows),
        "raw_conf_gain_accepted_new_error_count": sum(bool(row["raw_conf_gain_accepted_new_error"]) for row in rows),
    }


def make_summary(per_model_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for classifier in sorted({str(row["classifier"]) for row in per_model_rows}):
        cls_rows = [row for row in per_model_rows if str(row["classifier"]) == classifier]
        rows.append(summarize(cls_rows, "classifier", classifier, "all", "all"))
        for split in sorted({str(row["split"]) for row in cls_rows}):
            split_rows = [row for row in cls_rows if str(row["split"]) == split]
            rows.append(summarize(split_rows, "classifier_split", classifier, split, "all"))
            for snr in sorted({float(row["snr_db"]) for row in split_rows}):
                snr_rows = [row for row in split_rows if float(row["snr_db"]) == snr]
                rows.append(summarize(snr_rows, "classifier_split_snr", classifier, split, snr_name(snr)))
    return rows


def make_vote_rows(source_rows: list[dict[str, Any]], per_model_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_model_rows:
        by_key[(str(row["split"]), float(row["snr_db"]), str(row["sample"]))].append(row)
    output: list[dict[str, Any]] = []
    for row in source_rows:
        key = (str(row["split"]), float(row["snr_db"]), str(row["sample"]))
        model_rows = sorted(by_key[key], key=lambda item: str(item["classifier"]))
        classifiers = [str(item["classifier"]) for item in model_rows]
        new_error_models = [str(item["classifier"]) for item in model_rows if bool(item["selected_accepted_new_error"])]
        repair_models = [str(item["classifier"]) for item in model_rows if bool(item["selected_accepted_repair"])]
        missed_repair_models = [str(item["classifier"]) for item in model_rows if bool(item["selected_missed_repair"])]
        protective_models = [str(item["classifier"]) for item in model_rows if bool(item["selected_protective_reject"])]
        output.append(
            {
                "split": row["split"],
                "snr_db": float(row["snr_db"]),
                "sample": row["sample"],
                "selected_accept_refined": bool(row["accept_refined"]),
                "selected_new_accept_vs_alexnet_top1": bool(row["new_accept_vs_top1"]),
                "selected_shadow_veto": bool(row["shadow_veto"]),
                "original": row["original"],
                "m0_reconstruction": row["m0_reconstruction"],
                "refined": row["refined"],
                "materialized_final": row["materialized_final"],
                "classifier_count": len(model_rows),
                "classifiers": "|".join(classifiers),
                "selected_accepted_new_error_vote_count": len(new_error_models),
                "selected_accepted_new_error_models": "|".join(new_error_models),
                "selected_accepted_repair_vote_count": len(repair_models),
                "selected_accepted_repair_models": "|".join(repair_models),
                "selected_missed_repair_vote_count": len(missed_repair_models),
                "selected_missed_repair_models": "|".join(missed_repair_models),
                "selected_protective_reject_vote_count": len(protective_models),
                "selected_protective_reject_models": "|".join(protective_models),
            }
        )
    return output


def make_vote_summary(vote_rows: list[dict[str, Any]], level: str, split: str, snr: str) -> dict[str, Any]:
    return {
        "level": level,
        "split": split,
        "snr_db": snr,
        "num_images": len(vote_rows),
        "classifier_count": int(vote_rows[0]["classifier_count"]) if vote_rows else 0,
        "selected_accept_count": sum(bool(row["selected_accept_refined"]) for row in vote_rows),
        "any_classifier_new_error_count": sum(int(row["selected_accepted_new_error_vote_count"]) >= 1 for row in vote_rows),
        "majority_classifier_new_error_count": sum(
            int(row["selected_accepted_new_error_vote_count"]) > int(row["classifier_count"]) / 2
            for row in vote_rows
        ),
        "all_classifier_new_error_count": sum(
            int(row["selected_accepted_new_error_vote_count"]) == int(row["classifier_count"]) for row in vote_rows
        ),
        "any_classifier_repair_count": sum(int(row["selected_accepted_repair_vote_count"]) >= 1 for row in vote_rows),
        "majority_classifier_repair_count": sum(
            int(row["selected_accepted_repair_vote_count"]) > int(row["classifier_count"]) / 2
            for row in vote_rows
        ),
        "all_classifier_repair_count": sum(
            int(row["selected_accepted_repair_vote_count"]) == int(row["classifier_count"]) for row in vote_rows
        ),
        "any_classifier_missed_repair_count": sum(int(row["selected_missed_repair_vote_count"]) >= 1 for row in vote_rows),
        "any_classifier_protective_reject_count": sum(
            int(row["selected_protective_reject_vote_count"]) >= 1 for row in vote_rows
        ),
    }


def make_vote_summaries(vote_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [make_vote_summary(vote_rows, "all", "all", "all")]
    for split in sorted({str(row["split"]) for row in vote_rows}):
        split_rows = [row for row in vote_rows if str(row["split"]) == split]
        rows.append(make_vote_summary(split_rows, "split", split, "all"))
        for snr in sorted({float(row["snr_db"]) for row in split_rows}):
            subset = [row for row in split_rows if float(row["snr_db"]) == snr]
            rows.append(make_vote_summary(subset, "split_snr", split, snr_name(snr)))
    return rows


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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_value(row.get(key, "")) for key in fieldnames})


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def make_grid(rows: list[dict[str, Any]], output_path: Path, count: int) -> None:
    if not rows:
        return
    rows = rows[:count]
    tile = 160
    label_height = 48
    cols = 4
    canvas = Image.new("RGB", (tile * cols, (tile + label_height) * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for row_index, row in enumerate(rows):
        y = row_index * (tile + label_height)
        images = [
            ("original", resolve_project_path(row["original"])),
            ("m0", resolve_project_path(row["m0_reconstruction"])),
            ("refined", resolve_project_path(row["refined"])),
            ("final", resolve_project_path(row["materialized_final"])),
        ]
        for col, (label, path) in enumerate(images):
            x = col * tile
            canvas.paste(load_rgb(path).resize((tile, tile), Image.Resampling.BICUBIC), (x, y + label_height))
            draw.text((x + 4, y + 4), label, fill=(0, 0, 0), font=font)
        detail = (
            f"{row['split']} {row['sample']} {snr_name(float(row['snr_db']))} "
            f"accept={bool_text(bool(row['selected_accept_refined']))} "
            f"newerr_votes={row['selected_accepted_new_error_vote_count']} "
            f"repair_votes={row['selected_accepted_repair_vote_count']}"
        )
        draw.text((4, y + 18), detail[:118], fill=(0, 0, 0), font=font)
        models = str(row.get("selected_accepted_new_error_models") or row.get("selected_accepted_repair_models") or "")
        draw.text((4, y + 32), models[:118], fill=(0, 0, 0), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def write_galleries(vote_rows: list[dict[str, Any]], config: dict[str, Any], output_dir: Path) -> dict[str, str]:
    count = int(config["evaluation"]["gallery_rows"])
    gallery_dir = output_dir / "galleries"
    manifest: dict[str, str] = {}
    groups = {
        "any_classifier_selected_new_errors": [
            row for row in vote_rows if int(row["selected_accepted_new_error_vote_count"]) >= 1
        ],
        "majority_classifier_selected_new_errors": [
            row
            for row in vote_rows
            if int(row["selected_accepted_new_error_vote_count"]) > int(row["classifier_count"]) / 2
        ],
        "any_classifier_selected_repairs": [
            row for row in vote_rows if int(row["selected_accepted_repair_vote_count"]) >= 1
        ],
        "all_classifier_selected_repairs": [
            row
            for row in vote_rows
            if int(row["selected_accepted_repair_vote_count"]) == int(row["classifier_count"])
        ],
    }
    for name, rows in groups.items():
        rows = sorted(
            rows,
            key=lambda row: (
                -int(row["selected_accepted_new_error_vote_count"]),
                -int(row["selected_accepted_repair_vote_count"]),
                str(row["split"]),
                float(row["snr_db"]),
                str(row["sample"]),
            ),
        )
        path = gallery_dir / f"{name}.png"
        make_grid(rows, path, count)
        if rows:
            manifest[name] = project_relative(path)
    return manifest


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


def get_git_dirty_state() -> str:
    try:
        output = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return "dirty" if output else "clean"
    except Exception:  # noqa: BLE001
        return "unknown"


def make_report(
    model_summary_rows: list[dict[str, Any]],
    vote_summary_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> str:
    split_rows = [
        row for row in model_summary_rows if row["level"] == "classifier_split" and row["split"] in {"validation", "heldout"}
    ]
    vote_split_rows = [row for row in vote_summary_rows if row["level"] == "split"]
    lines = [
        "# EXP-S4-006 Selected Risk-Rule Classifier Ensemble Audit",
        "",
        "This offline audit re-evaluates fixed `selected_risk_rule` decisions with multiple frozen ImageNet classifiers.",
        "",
        "The ensemble is not used by the receiver-side gate. It only checks whether the AlexNet-tuned rule remains semantically plausible under other classifiers.",
        "",
        "## Per-Classifier Summary",
        "",
        "| Classifier | Split | Selected Failure | Delta vs M0 | Delta vs model top-1 | Selected Repair | Selected New Error |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in split_rows:
        lines.append(
            "| "
            f"{row['classifier']} | {row['split']} | "
            f"{float(row['selected_final_failure_rate']):.4f} | "
            f"{float(row['delta_selected_failure_vs_m0']):+.4f} | "
            f"{float(row['delta_selected_failure_vs_model_top1']):+.4f} | "
            f"{int(row['selected_accepted_repair_count'])} | "
            f"{int(row['selected_accepted_new_error_count'])} |"
        )
    lines.extend(
        [
            "",
            "## Vote Summary",
            "",
            "| Split | Images | Any New Error Vote | Majority New Error Vote | Any Repair Vote | Majority Repair Vote |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in vote_split_rows:
        lines.append(
            "| "
            f"{row['split']} | {int(row['num_images'])} | "
            f"{int(row['any_classifier_new_error_count'])} | "
            f"{int(row['majority_classifier_new_error_count'])} | "
            f"{int(row['any_classifier_repair_count'])} | "
            f"{int(row['majority_classifier_repair_count'])} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Per-model CSV: `{metadata['per_model_csv']}`",
            f"- Per-sample vote CSV: `{metadata['per_sample_votes_csv']}`",
            f"- Model summary CSV: `{metadata['model_summary_csv']}`",
            f"- Vote summary CSV: `{metadata['vote_summary_csv']}`",
            f"- Metadata: `{metadata['metadata_json']}`",
            f"- Galleries: `{metadata['gallery_dir']}`",
            "",
            "## Caveat",
            "",
            "COCO has no ImageNet ground-truth labels here. Each classifier uses its own original-image top-1 as pseudo reference, so this is a robustness audit rather than a final clean-correct metric.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    manifest = validate_inputs(config)
    rows = normalize_rows(read_csv(resolve_project_path(config["inputs"]["selected_policy_per_sample_csv"])))
    missing = validate_source_images(rows)
    if missing:
        raise FileNotFoundError("Missing source images:\n" + "\n".join(missing[:20]))
    statuses = classifier_status(config)
    available = [row for row in statuses if row["available"]]
    dry_run_payload = {
        "status": "ok",
        "num_rows": len(rows),
        "splits": {
            split: sum(1 for row in rows if str(row["split"]) == split)
            for split in sorted({str(row["split"]) for row in rows})
        },
        "unique_images_to_classify": len(unique_image_paths(rows)),
        "classifier_status": statuses,
        "allow_download": bool(args.allow_download),
        "device": str(resolve_device(args.device)),
        "manifest": manifest,
    }
    if args.dry_run:
        print(json.dumps(dry_run_payload, indent=2, ensure_ascii=False))
        return
    if not available and not args.allow_download:
        raise RuntimeError("No classifier weights are available locally. Use --allow-download with cleared proxy env.")

    output_dir = resolve_project_path(args.output_dir or config["outputs"]["output_dir"])
    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory already exists and is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")

    device = resolve_device(args.device)
    all_paths = unique_image_paths(rows)
    batch_size = int(config["classifiers"]["batch_size"])
    topk = int(config["classifiers"]["topk"])
    margin = float(config["evaluation"]["refined_conf_gain_margin"])
    per_model_rows: list[dict[str, Any]] = []
    model_runtime: dict[str, Any] = {}
    loaded_models: list[str] = []
    for model_cfg in config["classifiers"]["models"]:
        key = str(model_cfg["key"])
        model, preprocess, categories = load_classifier(model_cfg, config, device, allow_download=args.allow_download)
        predictions, elapsed = classify_paths(model, preprocess, all_paths, batch_size, topk, device)
        model_rows = eval_model_rows(rows, predictions, categories, key, margin)
        per_model_rows.extend(model_rows)
        model_runtime[key] = {
            "num_images": len(all_paths),
            "inference_seconds": elapsed,
            "device": str(device),
            "weights_file": project_relative(resolve_project_path(model_cfg["weights_file"])),
        }
        loaded_models.append(key)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    model_summary_rows = make_summary(per_model_rows)
    vote_rows = make_vote_rows(rows, per_model_rows)
    vote_summary_rows = make_vote_summaries(vote_rows)
    galleries = write_galleries(vote_rows, config, output_dir)

    per_model_csv = output_dir / "per_model_per_sample.csv"
    per_sample_votes_csv = output_dir / "per_sample_votes.csv"
    model_summary_csv = output_dir / "model_summary.csv"
    vote_summary_csv = output_dir / "vote_summary.csv"
    metadata_json = output_dir / "metadata.json"
    report_md = output_dir / "REPORT.md"
    write_csv(per_model_csv, per_model_rows)
    write_csv(per_sample_votes_csv, vote_rows)
    write_csv(model_summary_csv, model_summary_rows)
    write_csv(vote_summary_csv, vote_summary_rows)
    metadata = {
        "project_version": get_project_version(),
        "git_dirty_state": get_git_dirty_state(),
        "config": project_relative(config_path),
        "output_dir": project_relative(output_dir),
        "per_model_csv": project_relative(per_model_csv),
        "per_sample_votes_csv": project_relative(per_sample_votes_csv),
        "model_summary_csv": project_relative(model_summary_csv),
        "vote_summary_csv": project_relative(vote_summary_csv),
        "metadata_json": project_relative(metadata_json),
        "report_md": project_relative(report_md),
        "gallery_dir": project_relative(output_dir / "galleries"),
        "galleries": galleries,
        "run_command": " ".join(sys.argv),
        "source_inputs": manifest,
        "dry_run_payload": dry_run_payload,
        "classifiers_loaded": loaded_models,
        "classifier_runtime": model_runtime,
        "policy": config["policy"],
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "proxy_environment_present": sorted(key for key in os.environ if "proxy" in key.lower()),
        "download_note": (
            "Classifier weights may be downloaded only when --allow-download is used. "
            "No dataset or diffusion model download is required."
        ),
    }
    save_json(metadata_json, metadata)
    report_md.write_text(make_report(model_summary_rows, vote_summary_rows, metadata), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": project_relative(output_dir),
                "classifiers_loaded": loaded_models,
                "per_model_rows": len(per_model_rows),
                "vote_rows": len(vote_rows),
                "report_md": project_relative(report_md),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
