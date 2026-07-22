#!/usr/bin/env python3
"""Evaluate sender-derived semantic descriptions on the frozen Imagenette policy-dev run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import s6_imagenette_supervised_clean_eval as base  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
BOOL_COLUMNS = {
    "original_correct",
    "clean_primary",
    "gate_accept",
    "M0_correct",
    "M0_failure",
    "M2_edge_scheduled_correct",
    "M2_edge_scheduled_failure",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SGD-inspired sender semantic descriptions without opening official val."
    )
    parser.add_argument(
        "--config", default="configs/s6_imagenette_source_semantic_description_eval.yaml"
    )
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


def read_base_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    required = {
        "image_id",
        "channel_seed",
        "snr_db",
        "clean_primary",
        "true_label",
        "gate_m0_prediction",
        "gate_m0_confidence",
        "gate_candidate_prediction",
        "gate_candidate_confidence",
        "M0_correct",
        "M2_edge_scheduled_correct",
        "M0_psnr_db",
        "M2_edge_scheduled_psnr_db",
        "M0_lpips",
        "M2_edge_scheduled_lpips",
    }
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise RuntimeError(f"Base per-sample CSV is missing columns: {sorted(missing)}")
    for row in rows:
        for key in BOOL_COLUMNS:
            row[key] = str(row[key]).strip().lower() == "true"
    keys = [(str(row["image_id"]), int(row["channel_seed"]), float(row["snr_db"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Base per-sample CSV has duplicate image/channel-seed/SNR rows")
    return rows


def calibrated_probabilities(
    model: torch.nn.Module,
    images: torch.Tensor,
    temperature: float,
    config: dict[str, Any],
) -> torch.Tensor:
    logits = model(base.normalize_classifier_input(images, config))
    return torch.softmax(logits.float() / float(temperature), dim=1)


def quantize_description(probabilities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    codes = torch.round(probabilities.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    decoded = codes.float()
    sums = decoded.sum(dim=1, keepdim=True)
    if bool(torch.any(sums <= 0.0).item()):
        raise RuntimeError("A quantized source probability vector has zero total mass")
    return codes, decoded / sums


def js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    eps = 1e-12
    left = np.clip(left.astype(np.float64), eps, 1.0)
    right = np.clip(right.astype(np.float64), eps, 1.0)
    left = left / left.sum()
    right = right / right.sum()
    middle = 0.5 * (left + right)
    return float(0.5 * np.sum(left * np.log(left / middle)) + 0.5 * np.sum(right * np.log(right / middle)))


def semantic_scores(source: np.ndarray, m0: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    eps = 1e-12
    source = np.clip(source.astype(np.float64), eps, 1.0)
    source = source / source.sum()
    m0 = np.clip(m0.astype(np.float64), eps, 1.0)
    m0 = m0 / m0.sum()
    candidate = np.clip(candidate.astype(np.float64), eps, 1.0)
    candidate = candidate / candidate.sum()
    source_index = int(np.argmax(source))
    ce_m0 = -float(np.sum(source * np.log(m0)))
    ce_candidate = -float(np.sum(source * np.log(candidate)))
    cos_m0 = 1.0 - float(np.dot(source, m0) / (np.linalg.norm(source) * np.linalg.norm(m0)))
    cos_candidate = 1.0 - float(
        np.dot(source, candidate) / (np.linalg.norm(source) * np.linalg.norm(candidate))
    )
    return {
        "fullprob_cross_entropy_risk": ce_candidate - ce_m0,
        "fullprob_js_risk": js_divergence(source, candidate) - js_divergence(source, m0),
        "fullprob_cosine_risk": cos_candidate - cos_m0,
        "source_top1_logprob_risk": math.log(m0[source_index]) - math.log(candidate[source_index]),
    }


def nested_assignments(
    records: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[dict[str, str], dict[str, Any]]:
    split_config = config["nested_split"]
    seed = int(split_config["seed"])
    fraction = float(split_config["selection_fraction"])
    selection_name = str(split_config["selection_name"])
    audit_name = str(split_config["audit_name"])
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_class[str(record["wnid"])].append(record)
    assignments: dict[str, str] = {}
    counts: dict[str, dict[str, int]] = {}
    for wnid in sorted(by_class):
        ranked = sorted(
            by_class[wnid],
            key=lambda row: hashlib.sha256(
                f"{seed}:{row['image_id']}".encode("utf-8")
            ).hexdigest(),
        )
        selection_count = int(math.floor(len(ranked) * fraction))
        for index, record in enumerate(ranked):
            assignment = selection_name if index < selection_count else audit_name
            image_id = str(record["image_id"])
            if image_id in assignments:
                raise RuntimeError(f"Duplicate nested-split image ID: {image_id}")
            assignments[image_id] = assignment
        counts[wnid] = {
            selection_name: selection_count,
            audit_name: len(ranked) - selection_count,
        }
    return assignments, {
        "method": split_config["method"],
        "seed": seed,
        "selection_fraction": fraction,
        "names": [selection_name, audit_name],
        "per_class_counts": counts,
        "total_counts": {
            selection_name: sum(value[selection_name] for value in counts.values()),
            audit_name: sum(value[audit_name] for value in counts.values()),
        },
    }


@torch.no_grad()
def extract_probability_features(
    records: list[dict[str, Any]],
    loader: torch.utils.data.DataLoader,
    base_rows: list[dict[str, Any]],
    base_config: dict[str, Any],
    description_config: dict[str, Any],
    gate_model: torch.nn.Module,
    gate_temperature: float,
    deepjscc: torch.nn.Module,
    deepjscc_config: dict[str, Any],
    edge_refiner: torch.nn.Module,
    edge_config: dict[str, Any],
    alphas: dict[float, float],
    assignments: dict[str, str],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    existing = {
        (str(row["image_id"]), int(row["channel_seed"]), float(row["snr_db"])): row
        for row in base_rows
    }
    seeds = sorted({int(row["channel_seed"]) for row in base_rows})
    snrs = sorted({float(row["snr_db"]) for row in base_rows})
    source_vectors: dict[int, tuple[list[int], np.ndarray, int, float]] = {}
    for images_cpu, indices in loader:
        images = images_cpu.to(device, non_blocking=True)
        probabilities = calibrated_probabilities(gate_model, images, gate_temperature, base_config)
        codes, decoded = quantize_description(probabilities)
        raw_confidence, raw_prediction = probabilities.max(dim=1)
        for local_index, dataset_index_raw in enumerate(indices.tolist()):
            dataset_index = int(dataset_index_raw)
            source_vectors[dataset_index] = (
                [int(value) for value in codes[local_index].tolist()],
                decoded[local_index].cpu().numpy().astype(np.float64),
                int(raw_prediction[local_index].item()),
                float(raw_confidence[local_index].item()),
            )
    if len(source_vectors) != len(records):
        raise RuntimeError("Source semantic-description extraction is incomplete")

    tolerance = float(description_config["evaluation"]["compare_reconstructed_gate_outputs_abs_tol"])
    feature_rows: list[dict[str, Any]] = []
    max_confidence_error = 0.0
    top1_mismatches = 0
    seen_keys: set[tuple[str, int, float]] = set()
    for channel_seed in seeds:
        for snr in snrs:
            deepjscc.change_channel(str(deepjscc_config["channel"]), float(snr))
            alpha = float(alphas[float(snr)])
            batch_start = 0
            for images_cpu, indices in loader:
                images = images_cpu.to(device, non_blocking=True)
                batch_size = int(images.shape[0])
                call_seed = base.derived_channel_seed(channel_seed, snr, batch_start)
                base.seed_everything(call_seed)
                m0 = base.quantize_png_tensor(deepjscc(images), True)
                snr_norm = torch.full(
                    (batch_size,),
                    float(snr) / float(edge_config["model"]["snr_norm_max"]),
                    dtype=torch.float32,
                    device=device,
                )
                residual_gate = base.gate_tensor_for_snr(edge_config, snr, batch_size, device)
                raw = base.quantize_png_tensor(edge_refiner(m0, snr_norm, residual_gate), True)
                candidate = base.quantize_png_tensor(m0 + alpha * (raw - m0), True)
                m0_probabilities = calibrated_probabilities(
                    gate_model, m0, gate_temperature, base_config
                )
                candidate_probabilities = calibrated_probabilities(
                    gate_model, candidate, gate_temperature, base_config
                )
                m0_confidence, m0_prediction = m0_probabilities.max(dim=1)
                candidate_confidence, candidate_prediction = candidate_probabilities.max(dim=1)
                for local_index, dataset_index_raw in enumerate(indices.tolist()):
                    dataset_index = int(dataset_index_raw)
                    record = records[dataset_index]
                    key = (str(record["image_id"]), channel_seed, float(snr))
                    if key not in existing or key in seen_keys:
                        raise RuntimeError(f"Unexpected or duplicate reconstructed row: {key}")
                    seen_keys.add(key)
                    prior = existing[key]
                    reconstructed_predictions = (
                        int(m0_prediction[local_index].item()),
                        int(candidate_prediction[local_index].item()),
                    )
                    prior_predictions = (
                        int(prior["gate_m0_prediction"]),
                        int(prior["gate_candidate_prediction"]),
                    )
                    if reconstructed_predictions != prior_predictions:
                        top1_mismatches += 1
                    confidence_error = max(
                        abs(float(m0_confidence[local_index].item()) - float(prior["gate_m0_confidence"])),
                        abs(
                            float(candidate_confidence[local_index].item())
                            - float(prior["gate_candidate_confidence"])
                        ),
                    )
                    max_confidence_error = max(max_confidence_error, confidence_error)
                    source_codes, source_probability, source_raw_prediction, source_raw_confidence = (
                        source_vectors[dataset_index]
                    )
                    m0_probability = m0_probabilities[local_index].cpu().numpy().astype(np.float64)
                    candidate_probability = (
                        candidate_probabilities[local_index].cpu().numpy().astype(np.float64)
                    )
                    scores = semantic_scores(source_probability, m0_probability, candidate_probability)
                    enriched = dict(prior)
                    enriched.update(
                        {
                            "nested_split": assignments[str(record["image_id"])],
                            "source_description_codes_uint8": source_codes,
                            "source_description_probability": source_probability.tolist(),
                            "source_raw_prediction": source_raw_prediction,
                            "source_raw_confidence": source_raw_confidence,
                            "source_decoded_prediction": int(np.argmax(source_probability)),
                            "gate_m0_probability": m0_probability.tolist(),
                            "gate_candidate_probability": candidate_probability.tolist(),
                            **scores,
                        }
                    )
                    feature_rows.append(enriched)
                batch_start += batch_size
    if seen_keys != set(existing):
        missing = sorted(set(existing) - seen_keys)[:5]
        raise RuntimeError(f"Probability extraction missed {len(set(existing) - seen_keys)} rows: {missing}")
    if top1_mismatches or max_confidence_error > tolerance:
        raise RuntimeError(
            "Reconstructed gate outputs differ from the frozen base audit: "
            f"top1_mismatches={top1_mismatches}, max_confidence_error={max_confidence_error}, "
            f"tolerance={tolerance}"
        )
    return feature_rows, {
        "num_rows": len(feature_rows),
        "num_images": len(records),
        "channel_seeds": seeds,
        "snrs": snrs,
        "top1_mismatches_vs_base": top1_mismatches,
        "max_confidence_abs_error_vs_base": max_confidence_error,
        "comparison_tolerance": tolerance,
    }


def row_bool(row: dict[str, Any], key: str) -> bool:
    value = row[key]
    return value if isinstance(value, bool) else str(value).strip().lower() == "true"


def final_correct(row: dict[str, Any], accept: bool) -> bool:
    return row_bool(row, "M2_edge_scheduled_correct") if accept else row_bool(row, "M0_correct")


def policy_summary(
    rows: list[dict[str, Any]],
    image_ids: set[str],
    primary_snrs: set[float],
    policy_name: str,
    accept_function: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    subset = [row for row in rows if str(row["image_id"]) in image_ids]
    primary = [
        row
        for row in subset
        if float(row["snr_db"]) in primary_snrs and row_bool(row, "clean_primary")
    ]
    if not subset or not primary:
        raise RuntimeError(f"Empty policy scope for {policy_name}")
    accepted_new_error_rows = 0
    accepted_repair_rows = 0
    protective_reject_rows = 0
    missed_repair_rows = 0
    event_images: set[str] = set()
    eligible_images: set[str] = set()
    failures = 0
    m0_failures = 0
    m2_failures = 0
    primary_accepts = 0
    for row in primary:
        accept = bool(accept_function(row))
        m0_correct = row_bool(row, "M0_correct")
        m2_correct = row_bool(row, "M2_edge_scheduled_correct")
        correct = m2_correct if accept else m0_correct
        failures += int(not correct)
        m0_failures += int(not m0_correct)
        m2_failures += int(not m2_correct)
        primary_accepts += int(accept)
        if m0_correct:
            eligible_images.add(str(row["image_id"]))
        if accept and m0_correct and not m2_correct:
            accepted_new_error_rows += 1
            event_images.add(str(row["image_id"]))
        if accept and not m0_correct and m2_correct:
            accepted_repair_rows += 1
        if not accept and m0_correct and not m2_correct:
            protective_reject_rows += 1
        if not accept and not m0_correct and m2_correct:
            missed_repair_rows += 1
    psnr_gain = 0.0
    m2_psnr_gain = 0.0
    lpips_gain = 0.0
    accepted_all = 0
    for row in subset:
        accept = bool(accept_function(row))
        m0_psnr = float(row["M0_psnr_db"])
        m2_psnr = float(row["M2_edge_scheduled_psnr_db"])
        m0_lpips = float(row["M0_lpips"])
        m2_lpips = float(row["M2_edge_scheduled_lpips"])
        psnr_gain += (m2_psnr if accept else m0_psnr) - m0_psnr
        m2_psnr_gain += m2_psnr - m0_psnr
        lpips_gain += (m2_lpips if accept else m0_lpips) - m0_lpips
        accepted_all += int(accept)
    psnr_gain /= len(subset)
    m2_psnr_gain /= len(subset)
    lpips_gain /= len(subset)
    retained = psnr_gain / m2_psnr_gain if m2_psnr_gain > 0.0 else None
    return {
        "policy": policy_name,
        "num_images": len(image_ids),
        "num_all_snr_rows": len(subset),
        "num_primary_clean_rows": len(primary),
        "primary_accept_rate": primary_accepts / len(primary),
        "all_snr_accept_rate": accepted_all / len(subset),
        "final_failure_rate": failures / len(primary),
        "M0_failure_rate": m0_failures / len(primary),
        "M2_failure_rate": m2_failures / len(primary),
        "final_minus_M0_failure": (failures - m0_failures) / len(primary),
        "final_minus_M2_failure": (failures - m2_failures) / len(primary),
        "accepted_new_error_rows": accepted_new_error_rows,
        "accepted_new_error_image_clusters": len(event_images),
        "eligible_M0_correct_image_clusters": len(eligible_images),
        "accepted_repair_rows": accepted_repair_rows,
        "protective_reject_rows": protective_reject_rows,
        "missed_repair_rows": missed_repair_rows,
        "all_snr_psnr_gain_db": psnr_gain,
        "all_snr_M2_psnr_gain_db": m2_psnr_gain,
        "fraction_M2_psnr_gain_retained": retained,
        "all_snr_lpips_delta": lpips_gain,
    }


def policy_functions(row: dict[str, Any]) -> dict[str, bool]:
    source = int(row["source_decoded_prediction"])
    m0 = int(row["gate_m0_prediction"])
    candidate = int(row["gate_candidate_prediction"])
    label = int(row["true_label"])
    return {
        "M0_reject_all": False,
        "M2_accept_all": True,
        "current_top1_agreement": candidate == m0,
        "learned_source_router_default_accept": not (m0 == source and candidate != source),
        "learned_source_strict_candidate_match": candidate == source,
        "label_oracle_router_default_accept": not (m0 == label and candidate != label),
        "label_oracle_strict_candidate_match": candidate == label,
    }


def select_rule(
    rows: list[dict[str, Any]],
    selection_ids: set[str],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    selection = config["selection"]
    primary_snrs = {float(value) for value in selection["primary_snrs"]}
    grid: list[dict[str, Any]] = []
    for family in selection["eligible_families"]:
        thresholds = [float(value) for value in config["scores"][family]["thresholds"]]
        for threshold in thresholds:
            summary = policy_summary(
                rows,
                selection_ids,
                primary_snrs,
                f"{family}@{threshold:g}",
                lambda row, f=family, t=threshold: float(row[f]) <= t,
            )
            summary.update({"family": family, "threshold": threshold})
            retained = summary["fraction_M2_psnr_gain_retained"]
            summary["selection_feasible"] = bool(
                summary["accepted_new_error_image_clusters"]
                <= int(selection["max_selection_accepted_new_error_image_clusters"])
                and retained is not None
                and retained >= float(selection["min_fraction_m2_psnr_gain_retained"])
            )
            grid.append(summary)
    feasible = [row for row in grid if row["selection_feasible"]]
    if feasible:
        selected = min(
            feasible,
            key=lambda row: (
                float(row["final_failure_rate"]),
                -float(row["all_snr_psnr_gain_db"]),
                str(row["family"]),
                float(row["threshold"]),
            ),
        )
        used_fallback = False
    else:
        selected = min(
            grid,
            key=lambda row: (
                int(row["accepted_new_error_image_clusters"]),
                float(row["final_failure_rate"]),
                -float(row["all_snr_psnr_gain_db"]),
                str(row["family"]),
                float(row["threshold"]),
            ),
        )
        used_fallback = True
    return dict(selected), grid, used_fallback


def clustered_values(
    rows: list[dict[str, Any]],
    image_ids: set[str],
    snrs: set[float] | None,
    clean_only: bool,
    value_function: Callable[[dict[str, Any]], float],
) -> np.ndarray:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        image_id = str(row["image_id"])
        if image_id not in image_ids:
            continue
        if snrs is not None and float(row["snr_db"]) not in snrs:
            continue
        if clean_only and not row_bool(row, "clean_primary"):
            continue
        grouped[image_id].append(float(value_function(row)))
    return np.asarray(
        [np.mean(grouped[key], dtype=np.float64) for key in sorted(grouped)], dtype=np.float64
    )


def audit_intervals(
    rows: list[dict[str, Any]],
    audit_ids: set[str],
    config: dict[str, Any],
    family: str,
    threshold: float,
) -> dict[str, Any]:
    primary_snrs = {float(value) for value in config["selection"]["primary_snrs"]}
    replicates = int(config["evaluation"]["bootstrap_replicates"])
    seed = int(config["evaluation"]["bootstrap_seed"])
    confidence = float(config["evaluation"]["confidence"])

    def accept(row: dict[str, Any]) -> bool:
        return float(row[family]) <= threshold

    efficacy = clustered_values(
        rows,
        audit_ids,
        primary_snrs,
        True,
        lambda row: float(not final_correct(row, accept(row)))
        - float(not row_bool(row, "M2_edge_scheduled_correct")),
    )
    safety = clustered_values(
        rows,
        audit_ids,
        primary_snrs,
        True,
        lambda row: float(not final_correct(row, accept(row)))
        - float(not row_bool(row, "M0_correct")),
    )
    psnr = clustered_values(
        rows,
        audit_ids,
        None,
        False,
        lambda row: (
            float(row["M2_edge_scheduled_psnr_db"])
            if accept(row)
            else float(row["M0_psnr_db"])
        )
        - float(row["M0_psnr_db"]),
    )
    m2_psnr = clustered_values(
        rows,
        audit_ids,
        None,
        False,
        lambda row: float(row["M2_edge_scheduled_psnr_db"]) - float(row["M0_psnr_db"]),
    )
    clean_rows = [row for row in rows if row_bool(row, "clean_primary")]
    conditional = base.bootstrap_clustered_conditional_rate(
        clean_rows,
        audit_ids,
        primary_snrs,
        numerator_function=lambda row: bool(
            accept(row)
            and row_bool(row, "M0_correct")
            and not row_bool(row, "M2_edge_scheduled_correct")
        ),
        denominator_function=lambda row: row_bool(row, "M0_correct"),
        replicates=replicates,
        seed=seed + 3,
        confidence=confidence,
    )
    lpips_values = clustered_values(
        rows,
        audit_ids,
        None,
        False,
        lambda row: (
            float(row["M2_edge_scheduled_lpips"])
            if accept(row)
            else float(row["M0_lpips"])
        )
        - float(row["M0_lpips"]),
    )
    return {
        "gate_efficacy_final_minus_M2_failure": base.bootstrap_mean_ci(
            efficacy, replicates, seed, confidence
        ),
        "safety_final_minus_M0_failure": base.bootstrap_mean_ci(
            safety, replicates, seed + 1, confidence
        ),
        "accepted_new_error_conditional_on_M0_correct": conditional,
        "PSNR_final_minus_M0_db": base.bootstrap_mean_ci(psnr, replicates, seed + 2, confidence),
        "M2_PSNR_minus_M0_db": base.bootstrap_mean_ci(
            m2_psnr, replicates, seed + 4, confidence
        ),
        "LPIPS_final_minus_M0": {
            "estimate": float(lpips_values.mean()),
            "num_clusters": int(len(lpips_values)),
        },
    }


def evaluate_success(
    audit_summary: dict[str, Any], intervals: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    criteria = config["success_criteria"]
    efficacy = intervals["gate_efficacy_final_minus_M2_failure"]
    safety = intervals["safety_final_minus_M0_failure"]
    conditional = intervals["accepted_new_error_conditional_on_M0_correct"]
    psnr = intervals["PSNR_final_minus_M0_db"]
    gates = {
        "gate_efficacy": float(efficacy["ci_high"]) < 0.0,
        "safety_vs_M0": float(safety["ci_high"])
        <= float(criteria["safety_vs_m0_ci_upper_max_absolute"]),
        "accepted_new_error": float(conditional["conservative_upper_95"])
        <= float(criteria["accepted_new_error_rate_ci_upper_max_conditional_on_m0_correct"]),
        "PSNR_positive": float(psnr["ci_low"]) > 0.0,
        "fraction_M2_PSNR_retained": float(audit_summary["fraction_M2_psnr_gain_retained"])
        >= float(criteria["min_fraction_m2_psnr_gain_retained"]),
        "LPIPS_negative": float(audit_summary["all_snr_lpips_delta"]) < 0.0,
    }
    return {"gates": gates, "all_pass": all(gates.values())}


def make_report(
    selected: dict[str, Any],
    used_fallback: bool,
    summaries: list[dict[str, Any]],
    intervals: dict[str, Any],
    success: dict[str, Any],
    split_metadata: dict[str, Any],
) -> str:
    by_key = {(row["scope"], row["policy"]): row for row in summaries}
    selected_name = str(selected["policy"])
    select_row = by_key[("semantic_select", selected_name)]
    audit_row = by_key[("semantic_audit", selected_name)]
    efficacy = intervals["gate_efficacy_final_minus_M2_failure"]
    conditional = intervals["accepted_new_error_conditional_on_M0_correct"]
    psnr = intervals["PSNR_final_minus_M0_db"]
    lines = [
        "# Imagenette Source Semantic Description Result",
        "",
        f"State: **{'PASS' if success['all_pass'] else 'FAIL'}** on the nested semantic-audit gates.",
        "",
        "## Selected learned rule",
        "",
        f"- Rule: `{selected['family']}` with `risk <= {float(selected['threshold']):g}`.",
        f"- Selection fallback objective used: `{used_fallback}`.",
        f"- Nested image counts: `{split_metadata['total_counts']}`.",
        "- Description: scratch G_gate source probability vector, uint8 per class, 80 raw bits, assumed noiseless for this diagnostic.",
        "- Official Imagenette validation remained sealed.",
        "",
        "## Frozen selection versus one-shot nested audit",
        "",
        "| Scope | Failure | Δ vs M2 | New-error image clusters | Repair rows | Accept | ΔPSNR | ΔLPIPS | M2 PSNR retained |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scope, row in (("semantic_select", select_row), ("semantic_audit", audit_row)):
        lines.append(
            f"| {scope} | {row['final_failure_rate']:.6f} | {row['final_minus_M2_failure']:+.6f} | "
            f"{row['accepted_new_error_image_clusters']} | {row['accepted_repair_rows']} | "
            f"{row['all_snr_accept_rate']:.4f} | {row['all_snr_psnr_gain_db']:+.4f} dB | "
            f"{row['all_snr_lpips_delta']:+.4f} | {row['fraction_M2_psnr_gain_retained']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Nested-audit uncertainty",
            "",
            f"- Failure delta vs M2: `{efficacy['estimate']:+.6f}`, 95% CI "
            f"`[{efficacy['ci_low']:+.6f}, {efficacy['ci_high']:+.6f}]`.",
            f"- Accepted-new-error conservative upper: `{conditional['conservative_upper_95']:.6f}` "
            f"from `{conditional['event_image_clusters']}/{conditional['eligible_image_clusters']}` event/eligible image clusters.",
            f"- PSNR delta vs M0: `{psnr['estimate']:+.4f}` dB, 95% CI "
            f"`[{psnr['ci_low']:+.4f}, {psnr['ci_high']:+.4f}]`.",
            "",
            "## Decision",
            "",
        ]
    )
    for gate, passed in success["gates"].items():
        lines.append(f"- `{gate}`: **{'PASS' if passed else 'FAIL'}**")
    lines.extend(
        [
            "",
            "This result is a side-information diagnostic, not a matched-CBR deployment claim. "
            "The labelled oracle comparators are upper-bound diagnostics only and are not the selected learned method.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = base.resolve_project_path(args.config)
    config = load_yaml(config_path)
    base_config_path = base.resolve_project_path(config["protocol"]["base_config"])
    base_config = load_yaml(base_config_path)
    if config["protocol"].get("official_val_is_sealed") is not True:
        raise RuntimeError("This analysis must keep official validation sealed")
    if config["description"].get("channel_model") != "noiseless_diagnostic":
        raise RuntimeError("Only the preregistered noiseless diagnostic is implemented")
    if int(config["description"]["probability_vector_raw_bits"]) != 80:
        raise RuntimeError("The frozen source probability description must record 80 raw bits")
    output_dir = base.require_analysis_output_path(
        base.resolve_project_path(args.output_dir or config["outputs"]["output_dir"]),
        "source-semantic-description output",
    )
    input_dir = base.resolve_project_path(config["protocol"]["input_policy_dev_dir"])
    state_path = input_dir / "STATE.json"
    with state_path.open("r", encoding="utf-8") as handle:
        input_state = json.load(handle)
    if input_state.get("state") != "COMPLETE" or input_state.get("split") != "policy_dev":
        raise RuntimeError("The frozen supervised policy-dev input is not COMPLETE")
    base_csv_path = base.resolve_project_path(config["protocol"]["input_per_sample_csv"])
    base_rows = read_base_rows(base_csv_path)
    paths = base.artifact_paths(base_config)
    records, _manifest, manifest_metadata = base.load_manifest_records(
        paths["split_manifest"], base_config, "policy_dev", verify_content=True
    )
    assignments, split_metadata = nested_assignments(records, config)
    plan = {
        "analysis_id": config["analysis_id"],
        "config": base.project_relative(config_path),
        "base_config": base.project_relative(base_config_path),
        "input_per_sample_csv": base.project_relative(base_csv_path),
        "num_images": len(records),
        "num_base_rows": len(base_rows),
        "nested_split": split_metadata,
        "device": str(base.resolve_device(args.device)),
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

    device = base.resolve_device(args.device)
    manifest_sha = base.sha256_file(paths["split_manifest"])
    protocol_sha = base.protocol_sha256(base_config)
    gate_model, gate_temperature, gate_metadata = base.load_scratch_classifier(
        paths["gate_checkpoint"],
        "G_gate",
        base_config["scratch_classifiers"]["G_gate"],
        [str(item) for item in base_config["data"]["classes"]],
        manifest_sha,
        protocol_sha,
        device,
    )
    snrs = sorted({float(row["snr_db"]) for row in base_rows})
    edge_checkpoint = base.torch_load_checkpoint(paths["edge_checkpoint"])
    no_edge_checkpoint = base.torch_load_checkpoint(paths["no_edge_checkpoint"])
    edge_embedded_config = edge_checkpoint.get("config")
    no_edge_embedded_config = no_edge_checkpoint.get("config")
    if not isinstance(edge_embedded_config, dict) or not isinstance(no_edge_embedded_config, dict):
        raise RuntimeError("Refiner checkpoints must contain embedded configs")
    alphas, _schedule, schedule_metadata = base.load_frozen_schedule(
        paths["schedule"],
        str(base_config["inputs"]["schedule_key"]),
        snrs,
        edge_embedded_config,
        no_edge_embedded_config,
    )
    deepjscc, deepjscc_config, deepjscc_metadata = base.load_deepjscc(
        paths["deepjscc_config"],
        paths["deepjscc_checkpoint"],
        str(base_config["channel"]["type"]),
        snrs[0],
        device,
    )
    edge_refiner, edge_config, edge_metadata = base.load_refiner_model(
        paths["edge_checkpoint"], paths["edge_config"], True, device
    )
    dataset = base.ManifestImageDataset(records, int(base_config["data"]["image_size"]))
    loader_config = json.loads(json.dumps(base_config))
    loader_config["evaluation"]["batch_size"] = int(config["evaluation"]["batch_size"])
    loader_config["evaluation"]["num_workers"] = int(config["evaluation"]["num_workers"])
    loader = base.make_loader(dataset, loader_config, device)
    base.seed_everything(int(config["seed"]))
    feature_rows, extraction_metadata = extract_probability_features(
        records,
        loader,
        base_rows,
        base_config,
        config,
        gate_model,
        gate_temperature,
        deepjscc,
        deepjscc_config,
        edge_refiner,
        edge_config,
        alphas,
        assignments,
        device,
    )
    feature_csv = output_dir / "semantic_features.csv"
    base.write_csv(feature_csv, feature_rows)

    selection_name = str(config["nested_split"]["selection_name"])
    audit_name = str(config["nested_split"]["audit_name"])
    selection_ids = {image_id for image_id, value in assignments.items() if value == selection_name}
    audit_ids = {image_id for image_id, value in assignments.items() if value == audit_name}
    selected, grid, used_fallback = select_rule(feature_rows, selection_ids, config)
    base.write_csv(output_dir / "selection_grid.csv", grid)
    family = str(selected["family"])
    threshold = float(selected["threshold"])
    selected_policy_name = f"{family}@{threshold:g}"
    primary_snrs = {float(value) for value in config["selection"]["primary_snrs"]}

    summary_rows: list[dict[str, Any]] = []
    scopes = ((selection_name, selection_ids), (audit_name, audit_ids), ("full_policy_dev", set(assignments)))
    comparator_names = list(policy_functions(feature_rows[0]))
    for scope_name, image_ids in scopes:
        for policy_name in comparator_names:
            summary = policy_summary(
                feature_rows,
                image_ids,
                primary_snrs,
                policy_name,
                lambda row, name=policy_name: policy_functions(row)[name],
            )
            summary["scope"] = scope_name
            summary_rows.append(summary)
        selected_summary = policy_summary(
            feature_rows,
            image_ids,
            primary_snrs,
            selected_policy_name,
            lambda row: float(row[family]) <= threshold,
        )
        selected_summary["scope"] = scope_name
        summary_rows.append(selected_summary)
    base.write_csv(output_dir / "summary.csv", summary_rows)

    intervals = audit_intervals(feature_rows, audit_ids, config, family, threshold)
    audit_selected = next(
        row
        for row in summary_rows
        if row["scope"] == audit_name and row["policy"] == selected_policy_name
    )
    success = evaluate_success(audit_selected, intervals, config)
    payload = {
        "analysis_id": config["analysis_id"],
        "selected_rule": {
            "family": family,
            "threshold": threshold,
            "policy": selected_policy_name,
            "selection_feasible": bool(selected["selection_feasible"]),
            "used_fallback_objective": used_fallback,
            "selection_summary": selected,
        },
        "nested_split": split_metadata,
        "summaries": summary_rows,
        "nested_audit_intervals": intervals,
        "success": success,
        "description_accounting": config["description"],
        "official_val_accessed": False,
    }
    base.save_json(output_dir / "summary.json", payload)
    report = make_report(selected, used_fallback, summary_rows, intervals, success, split_metadata)
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    metadata = {
        "analysis_id": config["analysis_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_val_accessed": False,
        "input_state": input_state,
        "manifest": manifest_metadata,
        "nested_split": split_metadata,
        "extraction": extraction_metadata,
        "models": {
            "G_gate": gate_metadata,
            "DeepJSCC": deepjscc_metadata,
            "edge_refiner": edge_metadata,
        },
        "schedule": schedule_metadata,
        "hashes": {
            "config_sha256": base.sha256_file(config_path),
            "preregistration_sha256": base.sha256_file(
                base.resolve_project_path(config["protocol"]["preregistration"])
            ),
            "script_sha256": base.sha256_file(SCRIPT_PATH),
            "base_config_sha256": base.sha256_file(base_config_path),
            "base_per_sample_sha256": base.sha256_file(base_csv_path),
            "semantic_features_sha256": base.sha256_file(feature_csv),
        },
        "artifacts": {
            "report": base.project_relative(output_dir / "REPORT.md"),
            "summary_json": base.project_relative(output_dir / "summary.json"),
            "summary_csv": base.project_relative(output_dir / "summary.csv"),
            "selection_grid_csv": base.project_relative(output_dir / "selection_grid.csv"),
            "semantic_features_csv": base.project_relative(feature_csv),
        },
    }
    base.save_json(output_dir / "metadata.json", metadata)
    hashes = {
        name: base.sha256_file(output_dir / filename)
        for name, filename in {
            "report": "REPORT.md",
            "summary_json": "summary.json",
            "summary_csv": "summary.csv",
            "selection_grid_csv": "selection_grid.csv",
            "semantic_features_csv": "semantic_features.csv",
            "metadata_json": "metadata.json",
        }.items()
    }
    base.save_json(
        output_dir / "STATE.json",
        {
            "state": "COMPLETE",
            "official_val_accessed": False,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_hashes": hashes,
        },
    )
    print(report)


if __name__ == "__main__":
    main()
