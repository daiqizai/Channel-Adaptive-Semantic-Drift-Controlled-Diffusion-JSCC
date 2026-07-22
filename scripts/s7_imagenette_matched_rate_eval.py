#!/usr/bin/env python3
"""Supervised Imagenette policy-dev audit of the matched-total-rate main+structure system."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cadsd_jscc.deepjscc_adapter import build_deepjscc_model, extract_deepjscc_state_dict  # noqa: E402
from cadsd_jscc.semantic_sketch import (  # noqa: E402
    embed_repeated_sketch,
    fixed_rademacher_projection,
    probabilities_to_sketch,
    recover_repeated_sketch_and_erase,
)
from cadsd_jscc.structure import structure_rgb  # noqa: E402
import s6_imagenette_supervised_clean_eval as base  # noqa: E402
from s5_residual_refiner_pilot import (  # noqa: E402
    build_model as build_refiner,
    load_classifier as load_semantic_teacher,
)


SCRIPT_PATH = Path(__file__).resolve()
ARMS = ["reference_c8", "matched_main_c6", "matched_raw", "matched_top1_fallback"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/s7_imagenette_matched_rate_supervised_eval.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return payload


def derived_seed(base_seed: int, offset: int, snr: float, batch_start: int) -> int:
    payload = f"{base_seed}:{offset}:{float(snr):.8f}:{batch_start}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def validate_rate(config: dict[str, Any]) -> dict[str, Any]:
    rate = config["rate"]
    denominator = int(rate["denominator"])
    reference = int(rate["reference_inner_channel"])
    main = int(rate["main_inner_channel"])
    structure = int(rate["structure_inner_channel"])
    total = int(rate["total_inner_channel"])
    if main + structure != total or total != reference:
        raise RuntimeError(f"Rate mismatch: {main}+{structure}!={total}!={reference}")
    for key, numerator in (
        ("reference_cbr", reference),
        ("main_cbr", main),
        ("structure_cbr", structure),
        ("total_cbr", total),
    ):
        if not math.isclose(float(rate[key]), numerator / denominator, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"CBR mismatch for {key}")
    return {"reference": reference, "main": main, "structure": structure, "total": total, "denominator": denominator}


def select_sketch_alpha_indices(
    cosine_matrix: torch.Tensor, alpha_candidates: list[float]
) -> torch.Tensor:
    if cosine_matrix.ndim != 2 or cosine_matrix.shape[1] != len(alpha_candidates):
        raise ValueError("Cosine matrix and alpha candidate count do not match")
    if not alpha_candidates or alpha_candidates != sorted(alpha_candidates):
        raise ValueError("Alpha candidates must be non-empty and sorted")
    if not torch.isfinite(cosine_matrix).all():
        raise ValueError("Non-finite semantic controller score")
    return cosine_matrix.argmax(dim=1)


def load_matched_arm(
    checkpoint_path: Path,
    expected_arm: str,
    inner_channel: int,
    matched_config: dict[str, Any],
    channel_type: str,
    snr: float,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("arm") != expected_arm or int(checkpoint.get("inner_channel", -1)) != inner_channel:
        raise RuntimeError(f"Matched checkpoint contract mismatch: {checkpoint_path}")
    if checkpoint.get("official_val_accessed") is not False:
        raise RuntimeError("Matched checkpoint does not assert official_val_accessed=false")
    model = build_deepjscc_model(
        repo_root=base.resolve_project_path(matched_config["baseline"]["repo"]),
        inner_channel=inner_channel,
        channel=channel_type,
        snr=snr,
    ).to(device)
    model.load_state_dict(extract_deepjscc_state_dict(checkpoint), strict=True)
    model.eval().requires_grad_(False)
    return model, {
        "path": base.project_relative(checkpoint_path),
        "sha256": base.sha256_file(checkpoint_path),
        "arm": expected_arm,
        "inner_channel": inner_channel,
        "actual_cbr": checkpoint.get("actual_cbr"),
        "epoch": checkpoint.get("epoch"),
        "metrics": checkpoint.get("metrics"),
    }


def load_decoded_refiner(
    checkpoint_path: Path, config_path: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    source_config = load_yaml(config_path)
    if source_config["model"].get("condition_source") != "decoded_structure_rgb":
        raise RuntimeError("Matched refiner config is not decoded_structure_rgb")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    embedded = checkpoint.get("config")
    if not isinstance(embedded, dict) or embedded["model"] != source_config["model"]:
        raise RuntimeError("Matched refiner checkpoint/source model config mismatch")
    model = build_refiner(embedded).to(device)
    model.load_state_dict(base.state_dict_from_checkpoint(checkpoint, checkpoint_path), strict=True)
    model.eval().requires_grad_(False)
    return model, embedded, {
        "path": base.project_relative(checkpoint_path),
        "sha256": base.sha256_file(checkpoint_path),
        "source_config": base.project_relative(config_path),
        "source_config_sha256": base.sha256_file(config_path),
        "epoch": checkpoint.get("epoch"),
        "model": embedded["model"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else ("true" if value is True else "false" if value is False else value)
                    for key, value in row.items()
                }
            )


@torch.no_grad()
def evaluate_pipeline(
    records: list[dict[str, Any]],
    loader: torch.utils.data.DataLoader,
    original_predictions: dict[int, dict[str, float | int]],
    config: dict[str, Any],
    base_config: dict[str, Any],
    reference_model: torch.nn.Module,
    main_model: torch.nn.Module,
    structure_model: torch.nn.Module,
    refiner: torch.nn.Module,
    refiner_config: dict[str, Any],
    gate_model: torch.nn.Module,
    gate_temperature: float,
    evaluator: torch.nn.Module,
    evaluator_temperature: float,
    lpips_model: Any,
    device: torch.device,
    semantic_teacher: torch.nn.Module | None = None,
    semantic_preprocess: Any = None,
    semantic_projection: torch.Tensor | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    timings: dict[str, float] = defaultdict(float)
    snrs = [float(value) for value in config["channel"]["snrs"]]
    channel_type = str(config["channel"]["type"])
    channel_seed = int(config["channel"]["channel_seed"])
    quantize = bool(config["evaluation"]["quantize_png"])
    primary_threshold = float(config["evaluation"]["primary_clean_threshold"])
    hybrid_controller = str(config["protocol"].get("pipeline", "matched_structure_raw")) == "hybrid_semantic_controller"
    if hybrid_controller and (
        semantic_teacher is None or semantic_preprocess is None or semantic_projection is None
    ):
        raise RuntimeError("Hybrid semantic controller requires teacher, preprocess and projection")
    alpha_candidates = [float(value) for value in config.get("semantic_controller", {}).get("alpha_candidates", [1.0])]
    if alpha_candidates != sorted(alpha_candidates) or not alpha_candidates:
        raise RuntimeError("semantic_controller.alpha_candidates must be non-empty and sorted")
    for snr in snrs:
        for model in (reference_model, main_model, structure_model):
            model.change_channel(channel_type, snr)
        batch_start = 0
        for images_cpu, indices in loader:
            images = images_cpu.to(device, non_blocking=True)
            batch_size = len(images)
            source_structure = structure_rgb(images, third_channel="maximum")

            base.seed_everything(
                derived_seed(channel_seed, int(config["channel"]["reference_seed_offset"]), snr, batch_start)
            )
            started = time.perf_counter()
            reference = base.quantize_png_tensor(reference_model(images), quantize)
            timings["reference_seconds"] += time.perf_counter() - started

            base.seed_everything(
                derived_seed(channel_seed, int(config["channel"]["main_seed_offset"]), snr, batch_start)
            )
            started = time.perf_counter()
            main = base.quantize_png_tensor(main_model(images), quantize)
            timings["main_seconds"] += time.perf_counter() - started

            base.seed_everything(
                derived_seed(channel_seed, int(config["channel"]["structure_seed_offset"]), snr, batch_start)
            )
            started = time.perf_counter()
            if hybrid_controller:
                source_probabilities = torch.softmax(
                    semantic_teacher(semantic_preprocess(images)).float(), dim=1
                )
                source_sketch = probabilities_to_sketch(source_probabilities, semantic_projection)
                latent = structure_model.encoder(source_structure)
                if latent.ndim != 4:
                    raise RuntimeError(f"Structure encoder lost batch dimension: {tuple(latent.shape)}")
                hybrid_latent, reserved = embed_repeated_sketch(
                    latent, source_sketch, int(config["semantic_payload"]["repetitions"])
                )
                received_latent = structure_model.channel(hybrid_latent)
                recovered_sketch, erased_latent = recover_repeated_sketch_and_erase(
                    received_latent,
                    int(config["semantic_payload"]["sketch_dim"]),
                    int(config["semantic_payload"]["repetitions"]),
                    reserved,
                )
                decoded_structure = base.quantize_png_tensor(
                    structure_model.decoder(erased_latent), quantize
                )
            else:
                source_sketch = recovered_sketch = None
                decoded_structure = base.quantize_png_tensor(structure_model(source_structure), quantize)
            timings["structure_seconds"] += time.perf_counter() - started

            snr_norm = torch.full(
                (batch_size,),
                snr / float(refiner_config["model"]["snr_norm_max"]),
                dtype=torch.float32,
                device=device,
            )
            residual_gate = base.gate_tensor_for_snr(refiner_config, snr, batch_size, device)
            started = time.perf_counter()
            raw_candidate = base.quantize_png_tensor(
                refiner(
                    main,
                    snr_norm,
                    residual_gate,
                    condition_image=decoded_structure,
                    semantic_sketch=recovered_sketch,
                )
                if hybrid_controller
                else refiner(main, snr_norm, residual_gate, condition_image=decoded_structure),
                quantize,
            )
            timings["refiner_seconds"] += time.perf_counter() - started

            if hybrid_controller:
                candidates = [
                    base.quantize_png_tensor(main + alpha * (raw_candidate - main), quantize)
                    for alpha in alpha_candidates
                ]
                candidate_cosines = []
                for candidate in candidates:
                    probabilities = torch.softmax(
                        semantic_teacher(semantic_preprocess(candidate)).float(), dim=1
                    )
                    sketch = probabilities_to_sketch(probabilities, semantic_projection)
                    candidate_cosines.append(F.cosine_similarity(sketch, recovered_sketch, dim=1))
                cosine_matrix = torch.stack(candidate_cosines, dim=1)
                selected_indices = select_sketch_alpha_indices(cosine_matrix, alpha_candidates)
                candidate_stack = torch.stack(candidates, dim=1)
                raw = candidate_stack[
                    torch.arange(batch_size, device=device), selected_indices
                ]
                selected_alpha = raw.new_tensor(alpha_candidates)[selected_indices]
                selected_cosine = cosine_matrix[
                    torch.arange(batch_size, device=device), selected_indices
                ]
                source_recovered_cosine = F.cosine_similarity(
                    source_sketch, recovered_sketch, dim=1
                )
            else:
                raw = raw_candidate
                selected_alpha = torch.ones(batch_size, device=device)
                selected_cosine = torch.full((batch_size,), float("nan"), device=device)
                source_recovered_cosine = torch.full((batch_size,), float("nan"), device=device)

            gate_main_pred, gate_main_conf = base.predict_calibrated(
                gate_model, main, gate_temperature, base_config
            )
            gate_raw_pred, gate_raw_conf = base.predict_calibrated(
                gate_model, raw_candidate, gate_temperature, base_config
            )
            accept = gate_main_pred.eq(gate_raw_pred)
            final = raw_candidate if hybrid_controller else base.quantize_png_tensor(
                torch.where(accept.view(-1, 1, 1, 1), raw, main), quantize
            )

            predictions: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            for arm, tensor in (
                ("reference_c8", reference),
                ("matched_main_c6", main),
                ("matched_raw", raw),
                ("matched_top1_fallback", final),
            ):
                predictions[arm] = base.predict_calibrated(
                    evaluator, tensor, evaluator_temperature, base_config
                )
            qualities = {
                arm: base.quality_per_sample(images, tensor, lpips_model)
                for arm, tensor in (
                    ("reference_c8", reference),
                    ("matched_main_c6", main),
                    ("matched_raw", raw),
                    ("matched_top1_fallback", final),
                )
            }
            structure_mse = F.mse_loss(
                decoded_structure[:, :2], source_structure[:, :2], reduction="none"
            ).flatten(start_dim=1).mean(dim=1)
            for local, dataset_index_raw in enumerate(indices.tolist()):
                dataset_index = int(dataset_index_raw)
                record = records[dataset_index]
                original = original_predictions[dataset_index]
                true_label = int(record["true_label"])
                original_correct = int(original["prediction"]) == true_label
                original_confidence = float(original["confidence"])
                arm_correct: dict[str, bool] = {}
                row: dict[str, Any] = {
                    "image_id": record["image_id"],
                    "wnid": record["wnid"],
                    "true_label": true_label,
                    "snr_db": snr,
                    "channel_seed": channel_seed,
                    "reference_channel_seed": derived_seed(
                        channel_seed, int(config["channel"]["reference_seed_offset"]), snr, batch_start
                    ),
                    "main_channel_seed": derived_seed(
                        channel_seed, int(config["channel"]["main_seed_offset"]), snr, batch_start
                    ),
                    "structure_channel_seed": derived_seed(
                        channel_seed, int(config["channel"]["structure_seed_offset"]), snr, batch_start
                    ),
                    "original_tcls_prediction": int(original["prediction"]),
                    "original_tcls_confidence": original_confidence,
                    "original_correct": original_correct,
                    "clean_primary": original_correct and original_confidence >= primary_threshold,
                    "gate_main_prediction": int(gate_main_pred[local].item()),
                    "gate_main_confidence": float(gate_main_conf[local].item()),
                    "gate_raw_prediction": int(gate_raw_pred[local].item()),
                    "gate_raw_confidence": float(gate_raw_conf[local].item()),
                    "gate_accept": bool(accept[local].item()),
                    "sketch_selected_alpha": float(selected_alpha[local].item()),
                    "sketch_selected_cosine": float(selected_cosine[local].item()),
                    "source_recovered_sketch_cosine": float(source_recovered_cosine[local].item()),
                    "decoded_structure_first2_mse": float(structure_mse[local].item()),
                }
                for arm in ARMS:
                    prediction, confidence = predictions[arm]
                    pred = int(prediction[local].item())
                    correct = pred == true_label
                    arm_correct[arm] = correct
                    row[f"{arm}_tcls_prediction"] = pred
                    row[f"{arm}_tcls_confidence"] = float(confidence[local].item())
                    row[f"{arm}_correct"] = correct
                    row[f"{arm}_failure"] = not correct
                    for metric, values in qualities[arm].items():
                        row[f"{arm}_{metric}"] = (
                            None if values is None else float(values[local].item())
                        )
                row.update(
                    {
                        "raw_new_error_vs_reference": arm_correct["reference_c8"]
                        and not arm_correct["matched_raw"],
                        "raw_repair_vs_reference": (not arm_correct["reference_c8"])
                        and arm_correct["matched_raw"],
                        "raw_new_error_vs_main": arm_correct["matched_main_c6"]
                        and not arm_correct["matched_raw"],
                        "raw_repair_vs_main": (not arm_correct["matched_main_c6"])
                        and arm_correct["matched_raw"],
                        "top1_new_error_vs_reference": arm_correct["reference_c8"]
                        and not arm_correct["matched_top1_fallback"],
                        "top1_repair_vs_reference": (not arm_correct["reference_c8"])
                        and arm_correct["matched_top1_fallback"],
                    }
                )
                rows.append(row)
            batch_start += batch_size
    return rows, dict(timings)


def cluster_values(
    rows: list[dict[str, Any]],
    image_ids: set[str],
    snrs: set[float],
    value: Callable[[dict[str, Any]], float],
) -> np.ndarray:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if str(row["image_id"]) in image_ids and float(row["snr_db"]) in snrs:
            grouped[str(row["image_id"])].append(float(value(row)))
    return np.asarray([np.mean(grouped[key]) for key in sorted(grouped)], dtype=np.float64)


def main() -> None:
    args = parse_args()
    config_path = base.resolve_project_path(args.config)
    config = load_yaml(config_path)
    if config["protocol"].get("split") != "policy_dev" or config["protocol"].get("official_val_is_sealed") is not True:
        raise RuntimeError("This evaluator is policy-dev only and must keep official val sealed")
    base_config_path = base.resolve_project_path(config["protocol"]["base_supervised_config"])
    base_config = load_yaml(base_config_path)
    rate_contract = validate_rate(config)
    base_paths = base.artifact_paths(base_config)
    records, _manifest, manifest_metadata = base.load_manifest_records(
        base_paths["split_manifest"], base_config, "policy_dev", verify_content=True
    )
    device = base.resolve_device(args.device)
    output_dir = base.require_analysis_output_path(
        base.resolve_project_path(args.output_dir or config["outputs"]["output_dir"]),
        "matched-rate Imagenette output",
    )
    plan = {
        "analysis_id": config["analysis_id"],
        "split": "policy_dev",
        "num_images": len(records),
        "snrs": config["channel"]["snrs"],
        "num_rows": len(records) * len(config["channel"]["snrs"]),
        "rate_contract": rate_contract,
        "device": str(device),
        "output_dir": base.project_relative(output_dir),
        "official_val_accessed": False,
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, "plan": plan}, indent=2, ensure_ascii=False))
        return
    base.prepare_output_dir(output_dir, args.overwrite)
    base.save_json(output_dir / "STATE.json", {"state": "IN_PROGRESS", "official_val_accessed": False})
    shutil.copy2(config_path, output_dir / "config.yaml")
    shutil.copy2(SCRIPT_PATH, output_dir / SCRIPT_PATH.name)
    base.save_json(output_dir / "run_plan.json", plan)

    manifest_sha = base.sha256_file(base_paths["split_manifest"])
    protocol_sha = base.protocol_sha256(base_config)
    gate_model, gate_temperature, gate_metadata = base.load_scratch_classifier(
        base_paths["gate_checkpoint"],
        "G_gate",
        base_config["scratch_classifiers"]["G_gate"],
        [str(value) for value in base_config["data"]["classes"]],
        manifest_sha,
        protocol_sha,
        device,
    )
    evaluator, evaluator_temperature, evaluator_metadata = base.load_scratch_classifier(
        base_paths["evaluator_checkpoint"],
        "T_cls",
        base_config["scratch_classifiers"]["T_cls"],
        [str(value) for value in base_config["data"]["classes"]],
        manifest_sha,
        protocol_sha,
        device,
    )
    reference_model, _reference_config, reference_metadata = base.load_deepjscc(
        base.resolve_project_path(config["inputs"]["reference_c8_config"]),
        base.resolve_project_path(config["inputs"]["reference_c8_checkpoint"]),
        str(config["channel"]["type"]),
        float(config["channel"]["snrs"][0]),
        device,
    )
    matched_training_config = load_yaml(
        base.resolve_project_path(config["inputs"]["matched_training_config"])
    )
    main_model, main_metadata = load_matched_arm(
        base.resolve_project_path(config["inputs"]["main_c6_checkpoint"]),
        "main",
        rate_contract["main"],
        matched_training_config,
        str(config["channel"]["type"]),
        float(config["channel"]["snrs"][0]),
        device,
    )
    structure_model, structure_metadata = load_matched_arm(
        base.resolve_project_path(config["inputs"]["structure_c2_checkpoint"]),
        "structure",
        rate_contract["structure"],
        matched_training_config,
        str(config["channel"]["type"]),
        float(config["channel"]["snrs"][0]),
        device,
    )
    refiner, refiner_config, refiner_metadata = load_decoded_refiner(
        base.resolve_project_path(config["inputs"]["refiner_checkpoint"]),
        base.resolve_project_path(config["inputs"]["refiner_config"]),
        device,
    )
    semantic_teacher = semantic_preprocess = semantic_projection = None
    semantic_teacher_metadata = None
    if str(config["protocol"].get("pipeline", "matched_structure_raw")) == "hybrid_semantic_controller":
        semantic_teacher, semantic_preprocess, semantic_categories = load_semantic_teacher(
            config, device
        )
        semantic_teacher.requires_grad_(False).eval()
        semantic_projection = fixed_rademacher_projection(
            len(semantic_categories),
            int(config["semantic_payload"]["sketch_dim"]),
            int(config["semantic_payload"]["projection_seed"]),
            device=device,
        )
        semantic_teacher_metadata = {
            "model": config["classifier"]["model_name"],
            "weights": config["classifier"]["weights"],
            "weights_sha256": base.sha256_file(
                base.resolve_project_path(config["classifier"]["weights_file"])
            ),
            "role": "sender_descriptor_and_receiver_controller_only_not_primary_evaluator",
        }
    lpips_model, lpips_error = base.try_load_lpips(config, device, skip=False)
    if lpips_model is None:
        raise RuntimeError(f"LPIPS is required but unavailable: {lpips_error}")

    dataset = base.ManifestImageDataset(records, int(base_config["data"]["image_size"]))
    loader_config = json.loads(json.dumps(base_config))
    loader_config["evaluation"]["batch_size"] = int(config["evaluation"]["batch_size"])
    loader_config["evaluation"]["num_workers"] = int(config["evaluation"]["num_workers"])
    loader = base.make_loader(dataset, loader_config, device)
    original_predictions = base.classify_originals(
        loader, evaluator, evaluator_temperature, base_config, device
    )
    rows, timings = evaluate_pipeline(
        records,
        loader,
        original_predictions,
        config,
        base_config,
        reference_model,
        main_model,
        structure_model,
        refiner,
        refiner_config,
        gate_model,
        gate_temperature,
        evaluator,
        evaluator_temperature,
        lpips_model,
        device,
        semantic_teacher,
        semantic_preprocess,
        semantic_projection,
    )
    expected_rows = len(records) * len(config["channel"]["snrs"])
    if len(rows) != expected_rows:
        raise RuntimeError(f"Row count mismatch: {len(rows)} vs {expected_rows}")
    write_csv(output_dir / "per_sample.csv", rows)

    clean_ids = {
        str(records[index]["image_id"])
        for index, prediction in original_predictions.items()
        if int(prediction["prediction"]) == int(records[index]["true_label"])
        and float(prediction["confidence"]) >= float(config["evaluation"]["primary_clean_threshold"])
    }
    primary_snrs = {float(value) for value in config["evaluation"]["primary_snrs"]}
    all_snrs = {float(value) for value in config["channel"]["snrs"]}
    primary_rows = [
        row
        for row in rows
        if str(row["image_id"]) in clean_ids and float(row["snr_db"]) in primary_snrs
    ]
    summary: dict[str, Any] = {
        "num_images": len(records),
        "clean_primary_images": len(clean_ids),
        "primary_rows": len(primary_rows),
        "failure_rates": {
            arm: float(np.mean([row[f"{arm}_failure"] for row in primary_rows])) for arm in ARMS
        },
        "events_vs_reference": {
            "raw_new_error": int(sum(row["raw_new_error_vs_reference"] for row in primary_rows)),
            "raw_repair": int(sum(row["raw_repair_vs_reference"] for row in primary_rows)),
            "top1_new_error": int(sum(row["top1_new_error_vs_reference"] for row in primary_rows)),
            "top1_repair": int(sum(row["top1_repair_vs_reference"] for row in primary_rows)),
        },
        "events_vs_main": {
            "raw_new_error": int(sum(row["raw_new_error_vs_main"] for row in primary_rows)),
            "raw_repair": int(sum(row["raw_repair_vs_main"] for row in primary_rows)),
        },
        "quality_deltas": {},
        "sketch_alpha_distribution": {
            str(alpha): int(sum(abs(float(row["sketch_selected_alpha"]) - alpha) < 1e-9 for row in rows))
            for alpha in sorted({float(row["sketch_selected_alpha"]) for row in rows})
        },
    }
    for arm in ("matched_main_c6", "matched_raw", "matched_top1_fallback"):
        summary["quality_deltas"][f"{arm}_minus_reference"] = {
            metric: float(
                np.mean(
                    [
                        float(row[f"{arm}_{metric}"]) - float(row[f"reference_c8_{metric}"])
                        for row in rows
                    ]
                )
            )
            for metric in ("psnr_db", "lpips")
        }

    replicates = int(config["evaluation"]["bootstrap_replicates"])
    seed = int(config["evaluation"]["bootstrap_seed"])
    raw_failure = cluster_values(
        rows,
        clean_ids,
        primary_snrs,
        lambda row: float(row["matched_raw_failure"]) - float(row["reference_c8_failure"]),
    )
    top1_failure = cluster_values(
        rows,
        clean_ids,
        primary_snrs,
        lambda row: float(row["matched_top1_fallback_failure"]) - float(row["reference_c8_failure"]),
    )
    gate_efficacy = cluster_values(
        rows,
        clean_ids,
        primary_snrs,
        lambda row: float(row["matched_top1_fallback_failure"]) - float(row["matched_raw_failure"]),
    )
    raw_psnr = cluster_values(
        rows,
        {str(record["image_id"]) for record in records},
        all_snrs,
        lambda row: float(row["matched_raw_psnr_db"]) - float(row["reference_c8_psnr_db"]),
    )
    raw_lpips = cluster_values(
        rows,
        {str(record["image_id"]) for record in records},
        all_snrs,
        lambda row: float(row["matched_raw_lpips"]) - float(row["reference_c8_lpips"]),
    )
    conditional = base.bootstrap_clustered_conditional_rate(
        [row for row in rows if str(row["image_id"]) in clean_ids],
        clean_ids,
        primary_snrs,
        numerator_function=lambda row: bool(
            row["reference_c8_correct"] and not row["matched_raw_correct"]
        ),
        denominator_function=lambda row: bool(row["reference_c8_correct"]),
        replicates=replicates,
        seed=seed + 4,
        confidence=0.95,
    )
    intervals = {
        "raw_failure_minus_reference": base.bootstrap_mean_ci(raw_failure, replicates, seed),
        "top1_failure_minus_reference": base.bootstrap_mean_ci(top1_failure, replicates, seed + 1),
        "top1_failure_minus_raw": base.bootstrap_mean_ci(gate_efficacy, replicates, seed + 2),
        "raw_psnr_minus_reference_db": base.bootstrap_mean_ci(raw_psnr, replicates, seed + 3),
        "raw_lpips_minus_reference": base.bootstrap_mean_ci(raw_lpips, replicates, seed + 5),
        "raw_new_error_conditional_on_reference_correct": conditional,
    }
    psnr_by_snr = {
        str(snr): float(
            np.mean(
                [
                    float(row["matched_raw_psnr_db"]) - float(row["reference_c8_psnr_db"])
                    for row in rows
                    if float(row["snr_db"]) == snr
                ]
            )
        )
        for snr in sorted(all_snrs)
    }
    criteria = config["success_criteria"]
    success = {
        "raw_failure_noninferior": float(intervals["raw_failure_minus_reference"]["ci_high"])
        <= float(criteria["raw_failure_vs_reference_ci_upper_max"]),
        "raw_new_error_safe": float(conditional["conservative_upper_95"])
        <= float(criteria["raw_new_error_rate_upper_max_conditional_on_reference_correct"]),
        "raw_psnr_positive": float(intervals["raw_psnr_minus_reference_db"]["ci_low"]) > 0.0,
        "raw_psnr_positive_each_snr": all(value > 0.0 for value in psnr_by_snr.values()),
        "raw_lpips_negative": float(intervals["raw_lpips_minus_reference"]["estimate"]) < 0.0,
    }
    if str(config["protocol"].get("pipeline", "matched_structure_raw")) == "hybrid_semantic_controller":
        raw_candidate_gain = float(
            summary["quality_deltas"]["matched_top1_fallback_minus_reference"]["psnr_db"]
        )
        controller_gain = float(
            summary["quality_deltas"]["matched_raw_minus_reference"]["psnr_db"]
        )
        success["controller_improves_raw_failure"] = float(
            intervals["top1_failure_minus_raw"]["ci_low"]
        ) > 0.0
        success["controller_retains_raw_psnr"] = (
            raw_candidate_gain > 0.0
            and controller_gain / raw_candidate_gain
            >= float(criteria["minimum_raw_psnr_gain_retained"])
        )
    success["all_pass"] = all(success.values())
    payload = {
        "analysis_id": config["analysis_id"],
        "rate_contract": rate_contract,
        "summary": summary,
        "bootstrap": intervals,
        "raw_psnr_delta_by_snr": psnr_by_snr,
        "success": success,
        "official_val_accessed": False,
    }
    base.save_json(output_dir / "summary.json", payload)
    failure_ci = intervals["raw_failure_minus_reference"]
    psnr_ci = intervals["raw_psnr_minus_reference_db"]
    lpips_ci = intervals["raw_lpips_minus_reference"]
    primary_label = str(config.get("reporting", {}).get("primary_arm_label", "matched_raw"))
    secondary_label = str(config.get("reporting", {}).get("secondary_arm_label", "matched_top1_fallback"))
    lines = [
        f"# Imagenette {primary_label} Policy-Dev Audit",
        "",
        f"Decision: **{'PASS' if success['all_pass'] else 'FAIL'}** on the preregistered {primary_label} gates.",
        "Official Imagenette validation remained sealed.",
        "",
        "## Equal-rate result",
        "",
        f"- Clean-primary coverage: `{len(clean_ids)}/{len(records)}` images.",
        f"- Raw failure delta vs c=8: `{failure_ci['estimate']:+.6f}`, 95% CI "
        f"`[{failure_ci['ci_low']:+.6f}, {failure_ci['ci_high']:+.6f}]`.",
        f"- Raw new-error conservative upper: `{conditional['conservative_upper_95']:.6f}` from "
        f"`{conditional['event_image_clusters']}/{conditional['eligible_image_clusters']}` event/eligible image clusters.",
        f"- Raw PSNR delta vs c=8: `{psnr_ci['estimate']:+.4f}` dB, 95% CI "
        f"`[{psnr_ci['ci_low']:+.4f}, {psnr_ci['ci_high']:+.4f}]`.",
        f"- Raw LPIPS delta vs c=8: `{lpips_ci['estimate']:+.4f}`, 95% CI "
        f"`[{lpips_ci['ci_low']:+.4f}, {lpips_ci['ci_high']:+.4f}]`.",
        f"- Raw PSNR delta by SNR: `{psnr_by_snr}`.",
        "",
        "## Semantic events in primary SNRs",
        "",
        f"- Matched raw vs c=8: `{summary['events_vs_reference']}`.",
        f"- Matched raw vs c=6 main: `{summary['events_vs_main']}`.",
        f"- Failure rates: `{summary['failure_rates']}`.",
        f"- Arm aliases: `matched_raw={primary_label}`, `matched_top1_fallback={secondary_label}`.",
        f"- Sketch alpha distribution: `{summary['sketch_alpha_distribution']}`.",
        "",
        "## Gates",
        "",
    ]
    for key, passed in success.items():
        if key != "all_pass":
            lines.append(f"- `{key}`: **{'PASS' if passed else 'FAIL'}**")
    lines.extend(
        [
            "",
            "This is policy-development evidence for a frozen 20k warm-start pilot. Passing does not by itself "
            "authorize official-val access or a final-system claim.",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": base.project_relative(config_path),
        "config_sha256": base.sha256_file(config_path),
        "preregistration_sha256": base.sha256_file(
            base.resolve_project_path(config["protocol"]["preregistration"])
        ),
        "script": base.project_relative(SCRIPT_PATH),
        "script_sha256": base.sha256_file(SCRIPT_PATH),
        "manifest": manifest_metadata,
        "classifiers": {"G_gate": gate_metadata, "T_cls": evaluator_metadata},
        "models": {
            "reference": reference_metadata,
            "main": main_metadata,
            "structure": structure_metadata,
            "refiner": refiner_metadata,
            "semantic_teacher": semantic_teacher_metadata,
        },
        "rate_contract": rate_contract,
        "timings": timings,
        "num_rows": len(rows),
        "per_sample_sha256": base.sha256_file(output_dir / "per_sample.csv"),
        "official_val_accessed": False,
    }
    base.save_json(output_dir / "metadata.json", metadata)
    base.save_json(
        output_dir / "STATE.json",
        {
            "state": "COMPLETE",
            "success": success["all_pass"],
            "official_val_accessed": False,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
