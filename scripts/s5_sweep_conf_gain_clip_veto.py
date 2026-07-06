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
        description="Sweep receiver-side CLIP M0/refined consistency vetoes for EXP-S4-006 confidence-gain gate."
    )
    parser.add_argument("--config", default="configs/s5_conf_gain_clip_veto_sweep_exp_s4_006.yaml")
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


def rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def safe_threshold(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def snr_name(snr: float) -> str:
    if float(snr).is_integer():
        return f"snr_{int(snr):02d}db"
    return f"snr_{str(snr).replace('.', 'p')}db"


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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


def validate_inputs(config: dict[str, Any]) -> dict[str, str]:
    inputs = config["inputs"]
    paths = {
        "validation_csv": resolve_project_path(inputs["validation_csv"]),
        "heldout_csv": resolve_project_path(inputs["heldout_csv"]),
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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_rows(rows: list[dict[str, str]], split: str, margin: float) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        m0_prob = float(row["m0_top1_prob"])
        refined_prob = float(row["refined_top1_prob"])
        if "baseline_accept_refined" in row and row["baseline_accept_refined"] != "":
            baseline_accept = parse_bool(row["baseline_accept_refined"])
        elif "refined_matches_m0_top1" in row:
            baseline_accept = parse_bool(row["refined_matches_m0_top1"])
        else:
            baseline_accept = str(row.get("m0_top1_index", row.get("m0_top1_label"))) == str(
                row.get("refined_top1_index", row.get("refined_top1_label"))
            )
        candidate_accept = baseline_accept or refined_prob >= m0_prob + margin
        if "candidate_accept_refined" in row and row["candidate_accept_refined"] != "":
            recorded_candidate = parse_bool(row["candidate_accept_refined"])
            if recorded_candidate != candidate_accept:
                raise RuntimeError(
                    f"Candidate policy mismatch in {split} {row['sample']} {row['snr_db']}: "
                    f"recorded={recorded_candidate}, recomputed={candidate_accept}"
                )
        m0_ok = parse_bool(row["m0_matches_original_top1"])
        refined_ok = parse_bool(row["refined_matches_original_top1"])
        normalized.append(
            {
                "split": split,
                "snr_db": float(row["snr_db"]),
                "sample": row["sample"],
                "original": row["original"],
                "m0_reconstruction": row["m0_reconstruction"],
                "refined": row["refined"],
                "m0_top1_label": row.get("m0_top1_label", ""),
                "refined_top1_label": row.get("refined_top1_label", ""),
                "m0_top1_prob": m0_prob,
                "refined_top1_prob": refined_prob,
                "refined_conf_gain_vs_m0": refined_prob - m0_prob,
                "m0_matches_original_top1": m0_ok,
                "refined_matches_original_top1": refined_ok,
                "baseline_accept_refined": baseline_accept,
                "candidate_accept_refined": candidate_accept,
                "newly_accepted_by_candidate": candidate_accept and not baseline_accept,
                "candidate_accepted_repair": candidate_accept and (not m0_ok) and refined_ok,
                "candidate_accepted_new_error": candidate_accept and m0_ok and (not refined_ok),
                "m0_psnr_db": float(row["m0_psnr_db"]),
                "refined_psnr_db": float(row["refined_psnr_db"]),
            }
        )
    return normalized


def check_rows_have_files(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        for key in ["m0_reconstruction", "refined"]:
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
    return model, eval_preprocess


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


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a * b).sum().item())


def add_clip_similarity(rows: list[dict[str, Any]], image_features: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        m0_key = project_relative(resolve_project_path(row["m0_reconstruction"]))
        refined_key = project_relative(resolve_project_path(row["refined"]))
        sim = cosine(image_features[m0_key], image_features[refined_key])
        enriched["clip_sim_m0_refined"] = sim
        enriched["clip_distance_m0_refined"] = 1.0 - sim
        output.append(enriched)
    return output


def build_policy_names(thresholds: list[float]) -> list[str]:
    names = ["top1_equal", "top1_equal_or_refined_conf_gain_ge_0p05"]
    names.extend(f"top1_equal_or_conf_gain_0p05_clip_m0_refined_ge_{safe_threshold(threshold)}" for threshold in thresholds)
    return names


def policy_accept(row: dict[str, Any], policy_name: str) -> bool:
    baseline = bool(row["baseline_accept_refined"])
    candidate = bool(row["candidate_accept_refined"])
    if policy_name == "top1_equal":
        return baseline
    if policy_name == "top1_equal_or_refined_conf_gain_ge_0p05":
        return candidate
    marker = "top1_equal_or_conf_gain_0p05_clip_m0_refined_ge_"
    if policy_name.startswith(marker):
        threshold_text = policy_name.removeprefix(marker).replace("p", ".")
        threshold = float(threshold_text)
        return baseline or (candidate and float(row["clip_sim_m0_refined"]) >= threshold)
    raise ValueError(f"Unknown policy: {policy_name}")


def evaluate_policy(rows: list[dict[str, Any]], split: str, policy_name: str, snr: float | None) -> dict[str, Any]:
    subset = [row for row in rows if row["split"] == split and (snr is None or float(row["snr_db"]) == snr)]
    total = len(subset)
    accepted = 0
    final_correct = 0
    m0_correct = 0
    refined_correct = 0
    accepted_repair = 0
    missed_repair = 0
    protective_reject = 0
    accepted_new_error = 0
    false_accept = 0
    false_reject = 0
    candidate_accept_count = 0
    candidate_new_accept_count = 0
    vetoed_candidate_accept = 0
    vetoed_candidate_repair = 0
    vetoed_candidate_new_error = 0
    final_psnrs: list[float] = []
    m0_psnrs: list[float] = []
    refined_psnrs: list[float] = []
    accepted_clip_sims: list[float] = []
    new_accept_clip_sims: list[float] = []

    for row in subset:
        accept = policy_accept(row, policy_name)
        baseline_accept = bool(row["baseline_accept_refined"])
        candidate_accept = bool(row["candidate_accept_refined"])
        m0_ok = bool(row["m0_matches_original_top1"])
        refined_ok = bool(row["refined_matches_original_top1"])
        final_ok = refined_ok if accept else m0_ok
        new_accept = accept and not baseline_accept
        candidate_new_accept = candidate_accept and not baseline_accept
        final_psnr = float(row["refined_psnr_db"] if accept else row["m0_psnr_db"])
        m0_psnr = float(row["m0_psnr_db"])
        refined_psnr = float(row["refined_psnr_db"])

        accepted += int(accept)
        final_correct += int(final_ok)
        m0_correct += int(m0_ok)
        refined_correct += int(refined_ok)
        accepted_repair += int(accept and (not m0_ok) and refined_ok)
        missed_repair += int((not accept) and (not m0_ok) and refined_ok)
        protective_reject += int((not accept) and m0_ok and (not refined_ok))
        accepted_new_error += int(accept and m0_ok and (not refined_ok))
        false_accept += int(accept and (not refined_ok))
        false_reject += int((not accept) and refined_ok)
        candidate_accept_count += int(candidate_accept)
        candidate_new_accept_count += int(candidate_new_accept)
        vetoed_candidate_accept += int(candidate_accept and not accept)
        vetoed_candidate_repair += int((candidate_accept and not accept) and (not m0_ok) and refined_ok)
        vetoed_candidate_new_error += int((candidate_accept and not accept) and m0_ok and (not refined_ok))
        final_psnrs.append(final_psnr)
        m0_psnrs.append(m0_psnr)
        refined_psnrs.append(refined_psnr)
        if accept:
            accepted_clip_sims.append(float(row["clip_sim_m0_refined"]))
        if new_accept:
            new_accept_clip_sims.append(float(row["clip_sim_m0_refined"]))

    return {
        "split": split,
        "policy": policy_name,
        "snr_db": "all" if snr is None else float(snr),
        "num_images": total,
        "accept_count": accepted,
        "accept_rate": rate(accepted, total),
        "reject_count": total - accepted,
        "reject_rate": rate(total - accepted, total),
        "new_accept_vs_top1_count": sum(
            int(policy_accept(row, policy_name) and not bool(row["baseline_accept_refined"])) for row in subset
        ),
        "candidate_accept_count": candidate_accept_count,
        "candidate_new_accept_count": candidate_new_accept_count,
        "vetoed_candidate_accept_count": vetoed_candidate_accept,
        "vetoed_candidate_repair_count": vetoed_candidate_repair,
        "vetoed_candidate_new_error_count": vetoed_candidate_new_error,
        "m0_failure_rate": 1.0 - rate(m0_correct, total),
        "refined_failure_rate": 1.0 - rate(refined_correct, total),
        "final_failure_rate": 1.0 - rate(final_correct, total),
        "final_correct_rate": rate(final_correct, total),
        "false_accept_count": false_accept,
        "false_accept_rate": rate(false_accept, total),
        "false_reject_count": false_reject,
        "false_reject_rate": rate(false_reject, total),
        "accepted_repair_count": accepted_repair,
        "accepted_repair_rate": rate(accepted_repair, total),
        "missed_repair_count": missed_repair,
        "missed_repair_rate": rate(missed_repair, total),
        "protective_reject_count": protective_reject,
        "protective_reject_rate": rate(protective_reject, total),
        "accepted_new_error_count": accepted_new_error,
        "accepted_new_error_rate": rate(accepted_new_error, total),
        "m0_psnr_db": mean(m0_psnrs),
        "refined_psnr_db": mean(refined_psnrs),
        "final_psnr_db": mean(final_psnrs),
        "final_delta_psnr_vs_m0_db": mean(final_psnrs) - mean(m0_psnrs),
        "final_delta_psnr_vs_refined_db": mean(final_psnrs) - mean(refined_psnrs),
        "accepted_clip_sim_m0_refined_mean": mean(accepted_clip_sims),
        "new_accept_clip_sim_m0_refined_mean": mean(new_accept_clip_sims),
        "new_accept_clip_sim_m0_refined_min": min(new_accept_clip_sims) if new_accept_clip_sims else "",
    }


def add_policy_deltas(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed: dict[tuple[str, str, str], dict[str, Any]] = {
        (str(row["split"]), str(row["snr_db"]), str(row["policy"])): row for row in summary_rows
    }
    output: list[dict[str, Any]] = []
    for row in summary_rows:
        top1 = keyed[(str(row["split"]), str(row["snr_db"]), "top1_equal")]
        conf = keyed[(str(row["split"]), str(row["snr_db"]), "top1_equal_or_refined_conf_gain_ge_0p05")]
        enriched = dict(row)
        for baseline_name, baseline_row in [("top1_equal", top1), ("conf_gain", conf)]:
            suffix = "top1_equal" if baseline_name == "top1_equal" else "conf_gain"
            enriched[f"delta_final_failure_vs_{suffix}"] = (
                float(row["final_failure_rate"]) - float(baseline_row["final_failure_rate"])
            )
            enriched[f"delta_final_psnr_vs_{suffix}_db"] = (
                float(row["final_psnr_db"]) - float(baseline_row["final_psnr_db"])
            )
            enriched[f"delta_accepted_repair_vs_{suffix}"] = (
                int(row["accepted_repair_count"]) - int(baseline_row["accepted_repair_count"])
            )
            enriched[f"delta_accepted_new_error_vs_{suffix}"] = (
                int(row["accepted_new_error_count"]) - int(baseline_row["accepted_new_error_count"])
            )
        output.append(enriched)
    return output


def make_policy_decisions(rows: list[dict[str, Any]], policy_names: list[str]) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for row in rows:
        for policy_name in policy_names:
            accept = policy_accept(row, policy_name)
            baseline_accept = bool(row["baseline_accept_refined"])
            candidate_accept = bool(row["candidate_accept_refined"])
            m0_ok = bool(row["m0_matches_original_top1"])
            refined_ok = bool(row["refined_matches_original_top1"])
            decisions.append(
                {
                    "split": row["split"],
                    "policy": policy_name,
                    "snr_db": row["snr_db"],
                    "sample": row["sample"],
                    "accept_refined": accept,
                    "baseline_accept_refined": baseline_accept,
                    "candidate_accept_refined": candidate_accept,
                    "new_accept_vs_top1": accept and not baseline_accept,
                    "vetoed_candidate_accept": candidate_accept and not accept,
                    "final_matches_original_top1": refined_ok if accept else m0_ok,
                    "accepted_repair": accept and (not m0_ok) and refined_ok,
                    "missed_repair": (not accept) and (not m0_ok) and refined_ok,
                    "accepted_new_error": accept and m0_ok and (not refined_ok),
                    "protective_reject": (not accept) and m0_ok and (not refined_ok),
                    "final_psnr_db": row["refined_psnr_db"] if accept else row["m0_psnr_db"],
                    "clip_sim_m0_refined": row["clip_sim_m0_refined"],
                    "m0_top1_label": row["m0_top1_label"],
                    "refined_top1_label": row["refined_top1_label"],
                    "m0_top1_prob": row["m0_top1_prob"],
                    "refined_top1_prob": row["refined_top1_prob"],
                    "original": row["original"],
                    "m0_reconstruction": row["m0_reconstruction"],
                    "refined": row["refined"],
                }
            )
    return decisions


def make_joint_summary(global_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    policies = sorted({str(row["policy"]) for row in global_rows})
    by_key = {(str(row["split"]), str(row["policy"])): row for row in global_rows}
    output: list[dict[str, Any]] = []
    for policy in policies:
        val = by_key[("validation", policy)]
        held = by_key[("heldout", policy)]
        output.append(
            {
                "policy": policy,
                "validation_final_failure_rate": val["final_failure_rate"],
                "heldout_final_failure_rate": held["final_failure_rate"],
                "validation_final_psnr_db": val["final_psnr_db"],
                "heldout_final_psnr_db": held["final_psnr_db"],
                "validation_delta_psnr_vs_top1_equal_db": val["delta_final_psnr_vs_top1_equal_db"],
                "heldout_delta_psnr_vs_top1_equal_db": held["delta_final_psnr_vs_top1_equal_db"],
                "validation_accepted_repair_count": val["accepted_repair_count"],
                "heldout_accepted_repair_count": held["accepted_repair_count"],
                "validation_accepted_new_error_count": val["accepted_new_error_count"],
                "heldout_accepted_new_error_count": held["accepted_new_error_count"],
                "total_accepted_repair_count": int(val["accepted_repair_count"]) + int(held["accepted_repair_count"]),
                "total_accepted_new_error_count": int(val["accepted_new_error_count"])
                + int(held["accepted_new_error_count"]),
                "total_vetoed_candidate_repair_count": int(val["vetoed_candidate_repair_count"])
                + int(held["vetoed_candidate_repair_count"]),
                "total_vetoed_candidate_new_error_count": int(val["vetoed_candidate_new_error_count"])
                + int(held["vetoed_candidate_new_error_count"]),
                "sum_delta_psnr_vs_top1_equal_db": float(val["delta_final_psnr_vs_top1_equal_db"])
                + float(held["delta_final_psnr_vs_top1_equal_db"]),
            }
        )
    return output


def choose_selected_policy(joint_rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in joint_rows if str(row["policy"]) != "top1_equal"]
    zero_error = [
        row
        for row in candidates
        if int(row["validation_accepted_new_error_count"]) == 0
        and int(row["heldout_accepted_new_error_count"]) == 0
    ]
    if zero_error:
        return sorted(
            zero_error,
            key=lambda row: (
                int(row["total_accepted_repair_count"]),
                float(row["sum_delta_psnr_vs_top1_equal_db"]),
                -int(row["total_vetoed_candidate_repair_count"]),
            ),
            reverse=True,
        )[0]
    return sorted(
        candidates,
        key=lambda row: (
            -int(row["total_accepted_new_error_count"]),
            int(row["total_accepted_repair_count"]),
            float(row["sum_delta_psnr_vs_top1_equal_db"]),
        ),
        reverse=True,
    )[0]


def load_gallery_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB").resize((192, 192), Image.Resampling.BICUBIC)


def make_quad(row: dict[str, Any], output_path: Path) -> None:
    final_path = row["refined"] if parse_bool(row["accept_refined"]) else row["m0_reconstruction"]
    image_specs = [
        ("original", resolve_project_path(row["original"])),
        ("m0", resolve_project_path(row["m0_reconstruction"])),
        ("refined", resolve_project_path(row["refined"])),
        ("final", resolve_project_path(final_path)),
    ]
    images = [load_gallery_image(path) for _label, path in image_specs]
    label_height = 48
    width = 192 * len(images)
    canvas = Image.new("RGB", (width, 192 + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for idx, ((label, _path), image) in enumerate(zip(image_specs, images)):
        x = idx * 192
        canvas.paste(image, (x, label_height))
        draw.text((x + 4, 4), label, fill=(0, 0, 0), font=font)
    detail = (
        f"{row['split']} {float(row['snr_db']):g}dB {row['sample']} "
        f"clip={float(row['clip_sim_m0_refined']):.4f} "
        f"m0={row['m0_top1_label']} ref={row['refined_top1_label']}"
    )
    draw.text((4, 24), detail[:150], fill=(0, 0, 0), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def make_sheet(rows: list[dict[str, Any]], output_path: Path, title: str) -> None:
    if not rows:
        return
    quads: list[Image.Image] = []
    tmp_dir = output_path.parent / "_tmp_quads"
    for idx, row in enumerate(rows):
        tmp_path = tmp_dir / f"{idx:03d}.png"
        make_quad(row, tmp_path)
        quads.append(Image.open(tmp_path).convert("RGB"))
    header_height = 28
    width = max(image.width for image in quads)
    height = header_height + sum(image.height for image in quads)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((6, 8), title[:160], fill=(0, 0, 0), font=font)
    y = header_height
    for image in quads:
        sheet.paste(image, (0, y))
        y += image.height
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def write_galleries(decisions: list[dict[str, Any]], selected_policy: str, output_dir: Path, count: int) -> dict[str, str]:
    gallery_dir = output_dir / "galleries"
    outputs: dict[str, str] = {}
    for policy in ["top1_equal_or_refined_conf_gain_ge_0p05", selected_policy]:
        safe_policy = policy.replace(".", "p")
        subset = [row for row in decisions if row["policy"] == policy]
        for split in ["validation", "heldout"]:
            split_rows = [row for row in subset if row["split"] == split]
            new_errors = [row for row in split_rows if parse_bool(row["accepted_new_error"])]
            new_accepts = [row for row in split_rows if parse_bool(row["new_accept_vs_top1"])]
            new_errors = sorted(new_errors, key=lambda row: float(row["clip_sim_m0_refined"]))[:count]
            new_accepts = sorted(new_accepts, key=lambda row: float(row["clip_sim_m0_refined"]))[:count]
            error_path = gallery_dir / safe_policy / f"{split}_accepted_new_errors.png"
            accept_path = gallery_dir / safe_policy / f"{split}_new_accepts_lowest_clip.png"
            make_sheet(new_errors, error_path, f"{policy} {split} accepted new errors")
            make_sheet(new_accepts, accept_path, f"{policy} {split} new accepts sorted by low CLIP(M0, refined)")
            if error_path.exists():
                outputs[f"{safe_policy}_{split}_accepted_new_errors"] = project_relative(error_path)
            if accept_path.exists():
                outputs[f"{safe_policy}_{split}_new_accepts_lowest_clip"] = project_relative(accept_path)
    return outputs


def make_report(
    global_rows: list[dict[str, Any]],
    joint_rows: list[dict[str, Any]],
    selected: dict[str, Any],
    gallery_manifest: dict[str, str],
    metadata: dict[str, Any],
) -> str:
    by_split_policy = {(str(row["split"]), str(row["policy"])): row for row in global_rows}
    top1_val = by_split_policy[("validation", "top1_equal")]
    conf_val = by_split_policy[("validation", "top1_equal_or_refined_conf_gain_ge_0p05")]
    top1_held = by_split_policy[("heldout", "top1_equal")]
    conf_held = by_split_policy[("heldout", "top1_equal_or_refined_conf_gain_ge_0p05")]
    selected_val = by_split_policy[("validation", str(selected["policy"]))]
    selected_held = by_split_policy[("heldout", str(selected["policy"]))]
    sorted_joint = sorted(
        joint_rows,
        key=lambda row: (
            int(row["total_accepted_new_error_count"]) == 0,
            int(row["total_accepted_repair_count"]),
            float(row["sum_delta_psnr_vs_top1_equal_db"]),
        ),
        reverse=True,
    )

    lines = [
        "# EXP-S4-006 Confidence-Gain CLIP Veto Sweep",
        "",
        "This derived analysis tests a receiver-side second-stage veto for the confidence-gain gate.",
        "The veto only sees M0, refined, and CLIP image-image similarity between M0 and refined; original images and captions are not decision inputs.",
        "",
        "## Anchor Policies",
        "",
        "| Split | Policy | Final Failure | Final PSNR | Accepted Repair | Accepted New Error | Accept |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in [top1_val, conf_val, top1_held, conf_held]:
        lines.append(
            "| {split} | {policy} | {fail:.4f} | {psnr:.4f} | {repair} | {new_error} | {accept:.4f} |".format(
                split=row["split"],
                policy=row["policy"],
                fail=float(row["final_failure_rate"]),
                psnr=float(row["final_psnr_db"]),
                repair=int(row["accepted_repair_count"]),
                new_error=int(row["accepted_new_error_count"]),
                accept=float(row["accept_rate"]),
            )
        )
    lines.extend(
        [
            "",
            "## Selected Conservative Veto",
            "",
            f"- Selected policy: `{selected['policy']}`",
            "- Selection rule: prefer policies with zero accepted new error on both validation and held-out; then maximize accepted repairs and PSNR gain.",
            "",
            "| Split | Final Failure | Delta Failure vs Top1 | Final PSNR | Delta PSNR vs Top1 | Accepted Repair | Accepted New Error | Vetoed Candidate Repair | Vetoed Candidate New Error |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in [selected_val, selected_held]:
        lines.append(
            "| {split} | {fail:.4f} | {dfail:+.4f} | {psnr:.4f} | {dpsnr:+.4f} | {repair} | {new_error} | {vrepair} | {vnew} |".format(
                split=row["split"],
                fail=float(row["final_failure_rate"]),
                dfail=float(row["delta_final_failure_vs_top1_equal"]),
                psnr=float(row["final_psnr_db"]),
                dpsnr=float(row["delta_final_psnr_vs_top1_equal_db"]),
                repair=int(row["accepted_repair_count"]),
                new_error=int(row["accepted_new_error_count"]),
                vrepair=int(row["vetoed_candidate_repair_count"]),
                vnew=int(row["vetoed_candidate_new_error_count"]),
            )
        )
    lines.extend(
        [
            "",
            "## Joint Policy Ranking",
            "",
            "| Policy | Val New Error | Held New Error | Total Repair | Total Vetoed New Error | Total Vetoed Repair | Sum Delta PSNR vs Top1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted_joint[:12]:
        lines.append(
            "| {policy} | {vnew} | {hnew} | {repair} | {veto_new} | {veto_repair} | {dpsnr:+.4f} |".format(
                policy=row["policy"],
                vnew=int(row["validation_accepted_new_error_count"]),
                hnew=int(row["heldout_accepted_new_error_count"]),
                repair=int(row["total_accepted_repair_count"]),
                veto_new=int(row["total_vetoed_candidate_new_error_count"]),
                veto_repair=int(row["total_vetoed_candidate_repair_count"]),
                dpsnr=float(row["sum_delta_psnr_vs_top1_equal_db"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A CLIP M0/refined veto is receiver-side and easy to reproduce, but it is still an auxiliary consistency check.",
            "- If the selected veto removes accepted new errors while preserving some repairs on both splits, it is a better M3 candidate than raw confidence-gain.",
            "- If the selected veto collapses to top-1 behavior or sacrifices most repairs, the next detector should use a stronger receiver-side semantic model or an ensemble.",
            "",
            "## Output Files",
            "",
            f"- `per_sample_with_clip.csv`: `{metadata['per_sample_with_clip_csv']}`",
            f"- `policy_decisions.csv`: `{metadata['policy_decisions_csv']}`",
            f"- `policy_summary.csv`: `{metadata['policy_summary_csv']}`",
            f"- `policy_by_snr.csv`: `{metadata['policy_by_snr_csv']}`",
            f"- `joint_policy_summary.csv`: `{metadata['joint_policy_summary_csv']}`",
            f"- `metadata.json`: `{metadata['metadata_json']}`",
        ]
    )
    if gallery_manifest:
        lines.extend(["", "## Galleries", ""])
        for key, path in sorted(gallery_manifest.items()):
            lines.append(f"- `{key}`: `{path}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    config = load_config(config_path)
    paths = validate_inputs(config)
    margin = float(config["policy"]["refined_conf_gain_margin"])
    validation_rows = normalize_rows(read_csv_rows(resolve_project_path(config["inputs"]["validation_csv"])), "validation", margin)
    heldout_rows = normalize_rows(read_csv_rows(resolve_project_path(config["inputs"]["heldout_csv"])), "heldout", margin)
    rows = validation_rows + heldout_rows
    check_rows_have_files(rows)

    output_dir = resolve_project_path(args.output_dir or config["outputs"]["output_dir"])
    thresholds = [float(item) for item in config["policy"]["veto_thresholds"]]
    policy_names = build_policy_names(thresholds)

    if args.dry_run:
        unique_paths = sorted(
            {
                project_relative(resolve_project_path(row[key]))
                for row in rows
                for key in ["m0_reconstruction", "refined"]
            }
        )
        print(f"Config: {project_relative(config_path)}")
        print(f"Validation rows: {len(validation_rows)}")
        print(f"Held-out rows: {len(heldout_rows)}")
        print(f"Unique CLIP image paths: {len(unique_paths)}")
        print(f"Policies: {len(policy_names)}")
        print(f"Output dir: {project_relative(output_dir)}")
        print(f"Proxy env present: {proxy_environment_present()}")
        return

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output_dir / "config.yaml")

    device = resolve_device(args.device)
    unique_paths = sorted(
        {
            resolve_project_path(row[key])
            for row in rows
            for key in ["m0_reconstruction", "refined"]
        }
    )
    model, preprocess = load_clip_model(config, device)
    image_features, clip_elapsed = encode_paths(
        model=model,
        preprocess=preprocess,
        paths=unique_paths,
        batch_size=int(config["clip"]["batch_size"]),
        device=device,
    )
    rows_with_clip = add_clip_similarity(rows, image_features)

    global_summary = [
        evaluate_policy(rows_with_clip, split, policy_name, None)
        for split in ["validation", "heldout"]
        for policy_name in policy_names
    ]
    global_summary = add_policy_deltas(global_summary)
    by_snr_summary = [
        evaluate_policy(rows_with_clip, split, policy_name, snr)
        for split in ["validation", "heldout"]
        for policy_name in policy_names
        for snr in sorted({float(row["snr_db"]) for row in rows_with_clip if row["split"] == split})
    ]
    by_snr_summary = add_policy_deltas(by_snr_summary)
    decisions = make_policy_decisions(rows_with_clip, policy_names)
    joint_summary = make_joint_summary(global_summary)
    selected = choose_selected_policy(joint_summary)
    gallery_manifest = write_galleries(
        decisions=decisions,
        selected_policy=str(selected["policy"]),
        output_dir=output_dir,
        count=int(config["evaluation"]["gallery_rows"]),
    )

    per_sample_path = output_dir / "per_sample_with_clip.csv"
    decisions_path = output_dir / "policy_decisions.csv"
    policy_summary_path = output_dir / "policy_summary.csv"
    policy_by_snr_path = output_dir / "policy_by_snr.csv"
    joint_path = output_dir / "joint_policy_summary.csv"
    metadata_path = output_dir / "metadata.json"
    report_path = output_dir / "REPORT.md"

    write_csv(per_sample_path, rows_with_clip)
    write_csv(decisions_path, decisions)
    write_csv(policy_summary_path, global_summary)
    write_csv(policy_by_snr_path, by_snr_summary)
    write_csv(joint_path, joint_summary)

    metadata = {
        "analysis_id": config["analysis_id"],
        "source_experiment": config["source_experiment"],
        "config": project_relative(config_path),
        "copied_config": project_relative(output_dir / "config.yaml"),
        "inputs": paths,
        "output_dir": project_relative(output_dir),
        "per_sample_with_clip_csv": project_relative(per_sample_path),
        "policy_decisions_csv": project_relative(decisions_path),
        "policy_summary_csv": project_relative(policy_summary_path),
        "policy_by_snr_csv": project_relative(policy_by_snr_path),
        "joint_policy_summary_csv": project_relative(joint_path),
        "metadata_json": project_relative(metadata_path),
        "report": project_relative(report_path),
        "galleries": gallery_manifest,
        "selected_policy": selected,
        "policy_names": policy_names,
        "clip_elapsed_seconds": clip_elapsed,
        "clip_num_unique_images": len(unique_paths),
        "device": str(device),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "git_commit": git_commit(),
        "git_dirty_state": git_dirty_state(),
        "proxy_environment_present": proxy_environment_present(),
        "download_note": "No model or data download is required; CLIP checkpoint is loaded from local cache.",
    }
    try:
        import open_clip

        metadata["open_clip"] = getattr(open_clip, "__version__", "unknown")
    except Exception as exc:
        metadata["open_clip"] = f"unavailable: {exc}"
    save_json(metadata_path, metadata)
    report_path.write_text(make_report(global_summary, joint_summary, selected, gallery_manifest, metadata), encoding="utf-8")

    print(f"Wrote {project_relative(report_path)}")
    print(f"Selected policy: {selected['policy']}")


if __name__ == "__main__":
    main()
