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
from pathlib import Path
from typing import Any

import torch
import yaml
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from s5_residual_refiner_pilot import (  # noqa: E402
    build_model,
    classify_paths,
    compute_pair_metrics,
    condition_source_name,
    gate_tensor,
    label_for,
    load_classifier,
    load_rgb_tensor,
    load_semantic_sketch_store,
    project_relative,
    residual_gate,
    save_json,
    semantic_sketch_batch_for_names,
    snr_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a residual refiner on held-out/test-like samples and check confidence-gain gate risk."
    )
    parser.add_argument("--config", default="configs/s5_residual_refiner_heldout_gate_exp_s4_006.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def parse_snrs(config: dict[str, Any]) -> list[float]:
    return [float(item) for item in config["snrs"]]


def sample_names(start: int, count: int) -> list[str]:
    return [f"sample_{index:06d}.png" for index in range(start, start + count)]


def sample_range(start: int, count: int) -> set[str]:
    return set(sample_names(start, count))


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def rate(flags: list[bool]) -> float:
    return float(sum(flags) / len(flags)) if flags else 0.0


def bool_text(value: bool) -> str:
    return "true" if value else "false"


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


def get_project_version() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return "N/A (not a project git repo)"


def validate_inputs(config: dict[str, Any], snrs: list[float]) -> dict[str, Any]:
    original_dir = resolve_project_path(config["inputs"]["original_dir"])
    m0_export_dir = resolve_project_path(config["inputs"]["m0_export_dir"])
    refiner_checkpoint = resolve_project_path(config["inputs"]["refiner_checkpoint"])
    source_config = resolve_project_path(config["inputs"]["source_config"])
    checkpoint = resolve_project_path(config["inputs"]["checkpoint"])
    forbidden_checkpoint = resolve_project_path(config["inputs"]["forbidden_checkpoint"])
    classifier_weights = resolve_project_path(config["classifier"]["weights_file"])
    for path in [original_dir, m0_export_dir, refiner_checkpoint, source_config, checkpoint, classifier_weights]:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")
    if checkpoint == forbidden_checkpoint:
        raise RuntimeError("Config points to forbidden latest.pt checkpoint.")
    semantic_store = load_semantic_sketch_store(config)

    split = config["split"]
    names = sample_names(int(split["heldout_sample_start"]), int(split["heldout_sample_count"]))
    source_split = config.get("source_refiner_split", config.get("source_exp_s4_006_split"))
    if source_split is None:
        raise KeyError("Config must define source_refiner_split or legacy source_exp_s4_006_split")
    source_train = sample_range(int(source_split["train_sample_start"]), int(source_split["train_sample_count"]))
    source_eval = sample_range(int(source_split["eval_sample_start"]), int(source_split["eval_sample_count"]))
    overlap_train = sorted(set(names) & source_train)
    overlap_eval = sorted(set(names) & source_eval)
    if overlap_train or overlap_eval:
        source_experiment = str(config.get("source_experiment", "source refiner"))
        raise RuntimeError(
            f"Held-out split overlaps {source_experiment} train/eval: train={overlap_train}, eval={overlap_eval}"
        )

    for name in names:
        if not (original_dir / name).exists():
            raise FileNotFoundError(f"Original sample missing: {original_dir / name}")
    for snr in snrs:
        m0_subdir = str(config["inputs"].get("m0_reconstruction_subdir", "reconstruction"))
        m0_dir = m0_export_dir / "exports" / snr_name(snr) / m0_subdir
        if not m0_dir.exists():
            raise FileNotFoundError(f"M0 reconstruction directory missing: {m0_dir}")
        for name in names:
            if not (m0_dir / name).exists():
                raise FileNotFoundError(f"M0 sample missing: {m0_dir / name}")
        if condition_source_name(config) == "decoded_structure_rgb":
            structure_root = resolve_project_path(config["inputs"]["structure_export_dir"])
            structure_subdir = str(
                config["inputs"].get(
                    "structure_reconstruction_subdir", "structure_reconstruction"
                )
            )
            structure_dir = structure_root / "exports" / snr_name(snr) / structure_subdir
            if not structure_dir.is_dir():
                raise FileNotFoundError(f"Decoded structure directory missing: {structure_dir}")
            for name in names:
                if not (structure_dir / name).is_file():
                    raise FileNotFoundError(f"Decoded structure sample missing: {structure_dir / name}")
        residual_gate(config, snr)
    if semantic_store is not None:
        missing = sorted(set(names) - set(semantic_store["names"]))
        if missing:
            raise RuntimeError(f"Semantic sketch coverage missing held-out samples: {missing[:3]}")
    return {
        "heldout_names": names,
        "num_images_per_snr": len(names),
        "input_paths": {
            "original_dir": project_relative(original_dir),
            "m0_export_dir": project_relative(m0_export_dir),
            "refiner_checkpoint": project_relative(refiner_checkpoint),
            "source_config": project_relative(source_config),
            "checkpoint": project_relative(checkpoint),
            "forbidden_checkpoint": project_relative(forbidden_checkpoint),
            "classifier_weights": project_relative(classifier_weights),
            "semantic_sketch_file": (
                project_relative(semantic_store["path"]) if semantic_store is not None else None
            ),
        },
    }


def candidate_accept(row: dict[str, Any], margin: float) -> bool:
    return bool(row["baseline_accept_refined"]) or (
        float(row["refined_top1_prob"]) >= float(row["m0_top1_prob"]) + margin
    )


def psnr(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    mse = torch.mean((candidate - reference) ** 2).item()
    if mse <= 0.0:
        return 99.0
    return float(10.0 * torch.log10(torch.tensor(1.0 / mse)).item())


@torch.no_grad()
def refine_snr(
    model: torch.nn.Module,
    config: dict[str, Any],
    snr: float,
    names: list[str],
    output_dir: Path,
    device: torch.device,
) -> tuple[list[Path], float]:
    model.eval()
    m0_subdir = str(config["inputs"].get("m0_reconstruction_subdir", "reconstruction"))
    m0_dir = (
        resolve_project_path(config["inputs"]["m0_export_dir"])
        / "exports"
        / snr_name(snr)
        / m0_subdir
    )
    refined_dir = output_dir / "exports" / snr_name(snr) / "refined"
    refined_dir.mkdir(parents=True, exist_ok=True)
    refined_paths: list[Path] = []
    elapsed = 0.0
    batch_size = int(config["inference"]["batch_size"])
    semantic_store = load_semantic_sketch_store(config)
    for start in range(0, len(names), batch_size):
        batch_names = names[start : start + batch_size]
        batch = torch.stack([load_rgb_tensor(m0_dir / name) for name in batch_names]).to(device)
        condition_batch = None
        if condition_source_name(config) == "decoded_structure_rgb":
            structure_subdir = str(
                config["inputs"].get(
                    "structure_reconstruction_subdir", "structure_reconstruction"
                )
            )
            structure_dir = (
                resolve_project_path(config["inputs"]["structure_export_dir"])
                / "exports"
                / snr_name(snr)
                / structure_subdir
            )
            condition_batch = torch.stack(
                [load_rgb_tensor(structure_dir / name) for name in batch_names]
            ).to(device)
        snr_db = torch.full((len(batch_names),), float(snr), dtype=torch.float32, device=device)
        snr_norm = snr_db / float(config["model"]["snr_norm_max"])
        gate = gate_tensor(config, snr_db, device)
        semantic_batch = (
            semantic_sketch_batch_for_names(config, semantic_store, batch_names, snr).to(device)
            if semantic_store is not None
            else None
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        begin = time.perf_counter()
        refined = model(
            batch,
            snr_norm,
            gate,
            condition_image=condition_batch,
            semantic_sketch=semantic_batch,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed += time.perf_counter() - begin
        for name, image in zip(batch_names, refined.detach().cpu()):
            path = refined_dir / name
            save_image(image, path)
            refined_paths.append(path)
    return refined_paths, elapsed


def evaluate_snr(
    model: torch.nn.Module,
    config: dict[str, Any],
    snr: float,
    names: list[str],
    output_dir: Path,
    classifier_model: torch.nn.Module,
    classifier_preprocess,
    categories: list[str],
    lpips_model,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    original_dir = resolve_project_path(config["inputs"]["original_dir"])
    m0_subdir = str(config["inputs"].get("m0_reconstruction_subdir", "reconstruction"))
    m0_dir = (
        resolve_project_path(config["inputs"]["m0_export_dir"])
        / "exports"
        / snr_name(snr)
        / m0_subdir
    )
    refined_paths, refine_seconds = refine_snr(model, config, snr, names, output_dir, device)

    baseline_dir = output_dir / "exports" / snr_name(snr) / "top1_equal_final"
    candidate_dir = output_dir / "exports" / snr_name(snr) / "candidate_final"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    original_paths = [original_dir / name for name in names]
    m0_paths = [m0_dir / name for name in names]
    cls_batch = int(config["classifier"]["batch_size"])
    topk = int(config["classifier"]["topk"])
    original_preds, t_original = classify_paths(
        classifier_model, classifier_preprocess, original_paths, cls_batch, topk, device
    )
    m0_preds, t_m0 = classify_paths(classifier_model, classifier_preprocess, m0_paths, cls_batch, topk, device)
    refined_preds, t_refined = classify_paths(
        classifier_model, classifier_preprocess, refined_paths, cls_batch, topk, device
    )

    margin = float(config["policy"]["refined_conf_gain_margin"])
    per_sample: list[dict[str, Any]] = []
    reference_tensors: list[torch.Tensor] = []
    m0_tensors: list[torch.Tensor] = []
    refined_tensors: list[torch.Tensor] = []
    baseline_tensors: list[torch.Tensor] = []
    candidate_tensors: list[torch.Tensor] = []
    for idx, name in enumerate(names):
        original_top1 = int(original_preds[idx]["top_indices"][0])
        m0_top1 = int(m0_preds[idx]["top_indices"][0])
        refined_top1 = int(refined_preds[idx]["top_indices"][0])
        m0_ok = m0_top1 == original_top1
        refined_ok = refined_top1 == original_top1
        baseline_accept = refined_top1 == m0_top1

        row: dict[str, Any] = {
            "snr_db": float(snr),
            "residual_gate": residual_gate(config, snr),
            "sample": name,
            "original": project_relative(original_paths[idx]),
            "m0_reconstruction": project_relative(m0_paths[idx]),
            "refined": project_relative(refined_paths[idx]),
            "original_top_indices": original_preds[idx]["top_indices"],
            "original_top_probs": original_preds[idx]["top_probs"],
            "original_top_labels": [label_for(categories, item) for item in original_preds[idx]["top_indices"]],
            "m0_top_indices": m0_preds[idx]["top_indices"],
            "m0_top_probs": m0_preds[idx]["top_probs"],
            "m0_top_labels": [label_for(categories, item) for item in m0_preds[idx]["top_indices"]],
            "refined_top_indices": refined_preds[idx]["top_indices"],
            "refined_top_probs": refined_preds[idx]["top_probs"],
            "refined_top_labels": [label_for(categories, item) for item in refined_preds[idx]["top_indices"]],
            "original_top1_index": original_top1,
            "original_top1_label": label_for(categories, original_top1),
            "original_top1_prob": float(original_preds[idx]["top_probs"][0]),
            "m0_top1_index": m0_top1,
            "m0_top1_label": label_for(categories, m0_top1),
            "m0_top1_prob": float(m0_preds[idx]["top_probs"][0]),
            "refined_top1_index": refined_top1,
            "refined_top1_label": label_for(categories, refined_top1),
            "refined_top1_prob": float(refined_preds[idx]["top_probs"][0]),
            "m0_matches_original_top1": m0_ok,
            "refined_matches_original_top1": refined_ok,
            "refined_matches_m0_top1": refined_top1 == m0_top1,
            "baseline_accept_refined": baseline_accept,
        }
        accept_candidate = candidate_accept(row, margin)
        baseline_source = refined_paths[idx] if baseline_accept else m0_paths[idx]
        candidate_source = refined_paths[idx] if accept_candidate else m0_paths[idx]
        baseline_final = baseline_dir / name
        candidate_final = candidate_dir / name
        shutil.copy2(baseline_source, baseline_final)
        shutil.copy2(candidate_source, candidate_final)

        reference = load_rgb_tensor(original_paths[idx])
        m0 = load_rgb_tensor(m0_paths[idx])
        refined = load_rgb_tensor(refined_paths[idx])
        baseline = load_rgb_tensor(baseline_final)
        candidate = load_rgb_tensor(candidate_final)

        baseline_ok = refined_ok if baseline_accept else m0_ok
        candidate_ok = refined_ok if accept_candidate else m0_ok
        row.update(
            {
                "top1_equal_final": project_relative(baseline_final),
                "candidate_final": project_relative(candidate_final),
                "candidate_accept_refined": accept_candidate,
                "newly_accepted_by_candidate": accept_candidate and not baseline_accept,
                "baseline_final_matches_original_top1": baseline_ok,
                "candidate_final_matches_original_top1": candidate_ok,
                "candidate_accepted_repair": accept_candidate and (not m0_ok) and refined_ok,
                "candidate_accepted_new_error": accept_candidate and m0_ok and (not refined_ok),
                "baseline_accepted_repair": baseline_accept and (not m0_ok) and refined_ok,
                "baseline_accepted_new_error": baseline_accept and m0_ok and (not refined_ok),
                "m0_psnr_db": psnr(reference, m0),
                "refined_psnr_db": psnr(reference, refined),
                "baseline_final_psnr_db": psnr(reference, baseline),
                "candidate_final_psnr_db": psnr(reference, candidate),
            }
        )
        row["candidate_delta_psnr_vs_baseline_db"] = (
            float(row["candidate_final_psnr_db"]) - float(row["baseline_final_psnr_db"])
        )
        row["candidate_delta_psnr_vs_m0_db"] = float(row["candidate_final_psnr_db"]) - float(row["m0_psnr_db"])
        per_sample.append(row)

        reference_tensors.append(reference)
        m0_tensors.append(m0)
        refined_tensors.append(refined)
        baseline_tensors.append(baseline)
        candidate_tensors.append(candidate)

    reference_batch = torch.stack(reference_tensors)
    m0_batch = torch.stack(m0_tensors)
    refined_batch = torch.stack(refined_tensors)
    baseline_batch = torch.stack(baseline_tensors)
    candidate_batch = torch.stack(candidate_tensors)
    sample_count = min(int(config["evaluation"]["sample_grid_count"]), len(names))
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_grid = sample_dir / f"{snr_name(snr)}_original_m0_refined_top1_candidate.png"
    save_image(
        torch.cat(
            [
                reference_batch[:sample_count],
                m0_batch[:sample_count],
                refined_batch[:sample_count],
                baseline_batch[:sample_count],
                candidate_batch[:sample_count],
            ],
            dim=0,
        ),
        sample_grid,
        nrow=sample_count,
    )
    quality = {
        "m0_reconstruction_vs_original": compute_pair_metrics(reference_batch, m0_batch, lpips_model, device),
        "refined_vs_original": compute_pair_metrics(reference_batch, refined_batch, lpips_model, device),
        "top1_equal_final_vs_original": compute_pair_metrics(reference_batch, baseline_batch, lpips_model, device),
        "candidate_final_vs_original": compute_pair_metrics(reference_batch, candidate_batch, lpips_model, device),
    }
    summary = summarize_rows(per_sample, snr_name(snr))
    summary.update(
        {
            "snr_db": float(snr),
            "sample_grid": project_relative(sample_grid),
            "refiner_time_ms_per_image": 1000.0 * refine_seconds / max(1, len(names)),
            "classification_time_ms_per_image": 1000.0
            * (t_original + t_m0 + t_refined)
            / max(1, 3 * len(names)),
            "m0_lpips": quality["m0_reconstruction_vs_original"]["lpips"],
            "refined_lpips": quality["refined_vs_original"]["lpips"],
            "baseline_final_lpips": quality["top1_equal_final_vs_original"]["lpips"],
            "candidate_final_lpips": quality["candidate_final_vs_original"]["lpips"],
            "quality": quality,
        }
    )
    return per_sample, summary


def summarize_rows(rows: list[dict[str, Any]], subset: str) -> dict[str, Any]:
    return {
        "subset": subset,
        "num_images": len(rows),
        "m0_failure_rate": 1.0 - rate([bool(row["m0_matches_original_top1"]) for row in rows]),
        "refined_failure_rate": 1.0 - rate([bool(row["refined_matches_original_top1"]) for row in rows]),
        "baseline_accept_rate": rate([bool(row["baseline_accept_refined"]) for row in rows]),
        "candidate_accept_rate": rate([bool(row["candidate_accept_refined"]) for row in rows]),
        "new_accept_count": sum(bool(row["newly_accepted_by_candidate"]) for row in rows),
        "baseline_final_failure_rate": 1.0
        - rate([bool(row["baseline_final_matches_original_top1"]) for row in rows]),
        "candidate_final_failure_rate": 1.0
        - rate([bool(row["candidate_final_matches_original_top1"]) for row in rows]),
        "candidate_minus_baseline_failure": (
            1.0 - rate([bool(row["candidate_final_matches_original_top1"]) for row in rows])
        )
        - (1.0 - rate([bool(row["baseline_final_matches_original_top1"]) for row in rows])),
        "candidate_accepted_repair_count": sum(bool(row["candidate_accepted_repair"]) for row in rows),
        "candidate_accepted_new_error_count": sum(bool(row["candidate_accepted_new_error"]) for row in rows),
        "baseline_accepted_new_error_count": sum(bool(row["baseline_accepted_new_error"]) for row in rows),
        "m0_psnr_db": mean([float(row["m0_psnr_db"]) for row in rows]),
        "refined_psnr_db": mean([float(row["refined_psnr_db"]) for row in rows]),
        "baseline_final_psnr_db": mean([float(row["baseline_final_psnr_db"]) for row in rows]),
        "candidate_final_psnr_db": mean([float(row["candidate_final_psnr_db"]) for row in rows]),
        "candidate_delta_psnr_vs_baseline_db": mean(
            [float(row["candidate_delta_psnr_vs_baseline_db"]) for row in rows]
        ),
        "candidate_delta_psnr_vs_m0_db": mean([float(row["candidate_delta_psnr_vs_m0_db"]) for row in rows]),
    }


def clean_summary_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "quality"}


def make_case_grid(rows: list[dict[str, Any]], output_path: Path, max_cases: int = 12) -> str | None:
    if not rows:
        return None
    selected = rows[:max_cases]
    tensors: list[torch.Tensor] = []
    for row in selected:
        for key in ["original", "m0_reconstruction", "refined", "top1_equal_final", "candidate_final"]:
            tensors.append(load_rgb_tensor(resolve_project_path(row[key])))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(torch.stack(tensors), output_path, nrow=5)
    return project_relative(output_path)


def make_report(summary_rows: list[dict[str, Any]], config: dict[str, Any]) -> str:
    all_row = next(row for row in summary_rows if row["subset"] == "all")
    source_experiment = str(config.get("source_experiment", "source refiner"))
    split_name = str(config.get("split_name", "held-out"))
    lines = [
        f"# {source_experiment} {split_name} Confidence Gate Check",
        "",
        f"This derived check loads the trained {source_experiment} residual refiner and evaluates it on samples outside that refiner's train/eval split.",
        "",
        "Decision-time gate inputs remain receiver-side only: M0 and refined classifier top-1 predictions and confidence.",
        "",
        "## Split",
        "",
        f"- Held-out samples: `sample_{int(config['split']['heldout_sample_start']):06d}.png` to `sample_{int(config['split']['heldout_sample_start']) + int(config['split']['heldout_sample_count']) - 1:06d}.png`",
        f"- Images per SNR: {int(config['split']['heldout_sample_count'])}",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Candidate accept rate | {float(all_row['candidate_accept_rate']):.4f} |",
        f"| Newly accepted by candidate | {int(all_row['new_accept_count'])} |",
        f"| Candidate final failure | {float(all_row['candidate_final_failure_rate']):.4f} |",
        f"| Baseline top-1 final failure | {float(all_row['baseline_final_failure_rate']):.4f} |",
        f"| Candidate minus baseline failure | {float(all_row['candidate_minus_baseline_failure']):+.4f} |",
        f"| Candidate final PSNR | {float(all_row['candidate_final_psnr_db']):.4f} dB |",
        f"| Candidate delta PSNR vs baseline | {float(all_row['candidate_delta_psnr_vs_baseline_db']):+.4f} dB |",
        f"| Candidate delta PSNR vs M0 | {float(all_row['candidate_delta_psnr_vs_m0_db']):+.4f} dB |",
        f"| Accepted repairs | {int(all_row['candidate_accepted_repair_count'])} |",
        f"| Accepted new errors | {int(all_row['candidate_accepted_new_error_count'])} |",
        "",
        "## By SNR",
        "",
        "| SNR | M0 fail | Refined fail | Top-1 fail | Candidate fail | New accept | Repair | New error | Delta PSNR vs top-1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        if row["subset"] == "all":
            continue
        lines.append(
            "| {subset} | {m0:.4f} | {ref:.4f} | {base:.4f} | {cand:.4f} | {new} | {repair} | {err} | {dpsnr:+.4f} |".format(
                subset=row["subset"],
                m0=float(row["m0_failure_rate"]),
                ref=float(row["refined_failure_rate"]),
                base=float(row["baseline_final_failure_rate"]),
                cand=float(row["candidate_final_failure_rate"]),
                new=int(row["new_accept_count"]),
                repair=int(row["candidate_accepted_repair_count"]),
                err=int(row["candidate_accepted_new_error_count"]),
                dpsnr=float(row["candidate_delta_psnr_vs_baseline_db"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- This is a {split_name} pseudo-label check, not a final clean-correct classification result.",
            "- A candidate gate can only be promoted if it improves quality without increasing final failure or accepted new error on held-out data.",
        "- Accepted new errors remain the highest-priority cases for visual and auxiliary semantic review.",
        "",
        "## Review Files",
        "",
        "- `samples/accepted_new_error_review.png`: original / M0 / refined / top-1 final / candidate final for accepted new errors.",
        "- `samples/new_accepts_review.png`: same layout for newly accepted candidate samples.",
        "",
    ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    snrs = parse_snrs(config)
    manifest = validate_inputs(config, snrs)
    if args.dry_run:
        print(json.dumps({"status": "ok", "snrs": snrs, **manifest}, indent=2))
        return

    output_dir = resolve_project_path(args.output_dir or config["outputs"]["output_dir"])
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists, refusing to overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    shutil.copy2(config_path, output_dir / "config.yaml")

    seed = int(config["seed"])
    torch.manual_seed(seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = build_model(config).to(device)
    checkpoint = torch.load(resolve_project_path(config["inputs"]["refiner_checkpoint"]), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    classifier_model, classifier_preprocess, categories = load_classifier(config, device)
    lpips_model = None
    lpips_error = "Skipped by --skip-lpips"
    if not args.skip_lpips:
        try:
            os.environ.setdefault("TORCH_HOME", str(resolve_project_path(config["classifier"]["cache_dir"])))
            import lpips

            lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device)
            lpips_model.eval()
            lpips_error = None
        except Exception as exc:  # noqa: BLE001
            lpips_error = f"{type(exc).__name__}: {exc}"

    all_rows: list[dict[str, Any]] = []
    per_snr_summaries: list[dict[str, Any]] = []
    for snr in snrs:
        rows, summary = evaluate_snr(
            model=model,
            config=config,
            snr=snr,
            names=manifest["heldout_names"],
            output_dir=output_dir,
            classifier_model=classifier_model,
            classifier_preprocess=classifier_preprocess,
            categories=categories,
            lpips_model=lpips_model,
            device=device,
        )
        all_rows.extend(rows)
        per_snr_summaries.append(summary)
        print(json.dumps(clean_summary_row(summary), indent=2))

    summary_rows = [summarize_rows(all_rows, "all")]
    summary_rows[0].update(
        {
            "m0_lpips": mean(
                [float(row["m0_lpips"]) for row in per_snr_summaries if row.get("m0_lpips") is not None]
            )
            if any(row.get("m0_lpips") is not None for row in per_snr_summaries)
            else None,
            "refined_lpips": mean(
                [float(row["refined_lpips"]) for row in per_snr_summaries if row.get("refined_lpips") is not None]
            )
            if any(row.get("refined_lpips") is not None for row in per_snr_summaries)
            else None,
            "baseline_final_lpips": mean(
                [float(row["baseline_final_lpips"]) for row in per_snr_summaries if row.get("baseline_final_lpips") is not None]
            )
            if any(row.get("baseline_final_lpips") is not None for row in per_snr_summaries)
            else None,
            "candidate_final_lpips": mean(
                [float(row["candidate_final_lpips"]) for row in per_snr_summaries if row.get("candidate_final_lpips") is not None]
            )
            if any(row.get("candidate_final_lpips") is not None for row in per_snr_summaries)
            else None,
        }
    )
    summary_rows.extend(clean_summary_row(row) for row in per_snr_summaries)

    accepted_new_errors = [row for row in all_rows if bool(row["candidate_accepted_new_error"])]
    new_accepts = [row for row in all_rows if bool(row["newly_accepted_by_candidate"])]
    review_files = {
        "accepted_new_error_review": make_case_grid(
            accepted_new_errors, output_dir / "samples" / "accepted_new_error_review.png"
        ),
        "new_accepts_review": make_case_grid(new_accepts, output_dir / "samples" / "new_accepts_review.png"),
    }
    write_csv(output_dir / "per_sample.csv", all_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "new_accepts.csv", new_accepts)
    write_csv(output_dir / "accepted_new_errors.csv", accepted_new_errors)
    (output_dir / "REPORT.md").write_text(make_report(summary_rows, config), encoding="utf-8")

    import importlib.metadata as md
    import torchvision

    metadata = {
        "project_version": get_project_version(),
        "repository_url": config.get("repository_url"),
        "config": project_relative(config_path),
        "run_command": " ".join(sys.argv),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "dataset": config["dataset"],
        "image_size": int(config["image_size"]),
        "channel": config["channel"],
        "snrs": snrs,
        "cbr": float(config["cbr"]),
        "seed": seed,
        "inputs": manifest["input_paths"],
        "split": config["split"],
        "source_experiment": config.get("source_experiment", ""),
        "source_refiner_split": config.get("source_refiner_split", config.get("source_exp_s4_006_split")),
        "model": config["model"],
        "policy": config["policy"],
        "classifier": config["classifier"],
        "lpips_error": lpips_error,
        "num_rows": len(all_rows),
        "new_accept_count": len(new_accepts),
        "accepted_new_error_count": len(accepted_new_errors),
        "review_files": review_files,
        "python_version": platform.python_version(),
        "package_versions": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "pillow": md.version("pillow"),
            "pytorch-msssim": md.version("pytorch-msssim"),
        },
        "proxy_environment_present": sorted(key for key in os.environ if "proxy" in key.lower()),
        "download_note": "No model or data download is required; this check loads an existing local refiner checkpoint.",
        "key_sources": [
            "scripts/s5_residual_refiner_heldout_gate_eval.py",
            "scripts/s5_residual_refiner_pilot.py",
            "src/cadsd_jscc/metrics.py",
        ],
    }
    save_json(output_dir / "metadata.json", metadata)
    print(
        json.dumps(
            {
                "output_dir": project_relative(output_dir),
                "num_rows": len(all_rows),
                "new_accept_count": len(new_accepts),
                "accepted_new_error_count": len(accepted_new_errors),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
