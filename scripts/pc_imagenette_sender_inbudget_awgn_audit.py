#!/usr/bin/env python3
"""Strict-rate Imagenette audit for an in-budget analog sender probability payload."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from cadsd_jscc.deepjscc_adapter import (  # noqa: E402
    deepjscc_decode,
    deepjscc_encode,
    deepjscc_transmit,
    load_deepjscc_model,
    received_latent_consistency_per_sample,
)
from cadsd_jscc.metrics import psnr_per_sample  # noqa: E402
from cadsd_jscc.semantic_sketch import (  # noqa: E402
    bits_to_integer_codes,
    embed_repeated_sketch,
    integer_codes_to_bits,
    probabilities_to_simplex_sketch,
    quantize_probabilities_uniform,
    recover_repeated_sketch_and_erase,
    semantic_payload_accounting,
    simplex_sketch_to_probabilities,
)
from pc_imagenette_supervised_audit import (  # noqa: E402
    clopper_pearson_upper_95,
    evaluate_probabilities,
    jensen_shannon,
    load_imagenette_samples,
    load_scratch_classifier,
    sha256_file,
    source_semantic_score_tensors,
    summarize_rows,
)
from pc_posterior_consistency_replication import (  # noqa: E402
    load_yaml,
    mean,
    posterior_correct,
    resolve,
    write_csv,
)
from s10_short_chain_residual_shift_diffusion import (  # noqa: E402
    ShortChainResidualShiftDiffusion,
)
from s13_export_coco_train2017_c8_scaleup import derived_seed  # noqa: E402
from s5_residual_refiner_pilot import build_model, gate_tensor, try_load_lpips  # noqa: E402


SENDER_ONLY_CONTROLLER = "imagenette_sender_scratch_fullprob_inbudget_awgn_zero_veto"
DUAL_EVIDENCE_CONTROLLER = (
    "imagenette_sender_scratch_fullprob_and_gate_top1_inbudget_awgn_zero_veto"
)
CROSS_MODEL_TRIPLET_CONTROLLER = (
    "imagenette_sender_scratch_fullprob_crossmodel_triplet_inbudget_awgn_zero_veto"
)
DIGITAL_PAYLOAD_BITS = {2, 3, 4}


def assert_clean_membership_consistency(
    rows: list[dict[str, Any]],
    membership_by_sample: dict[str, dict[str, Any]],
    expected_rows_per_sample: int,
) -> None:
    """Fail closed unless every repeated row reuses one frozen clean decision."""
    if expected_rows_per_sample <= 0:
        raise ValueError("expected_rows_per_sample must be positive")
    counts: dict[str, int] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        expected = membership_by_sample.get(sample_id)
        if expected is None:
            raise RuntimeError(f"row has no frozen clean membership: {sample_id}")
        if (
            bool(row["clean_correct"]) != bool(expected["clean_correct"])
            or int(row["class_idx"]) != int(expected["class_idx"])
            or str(row["wnid"]) != str(expected["wnid"])
            or float(row["original_confidence"]) != float(expected["original_confidence"])
        ):
            raise RuntimeError(f"clean membership changed across rows: {sample_id}")
        counts[sample_id] = counts.get(sample_id, 0) + 1
    if set(counts) != set(membership_by_sample):
        missing = sorted(set(membership_by_sample) - set(counts))[:5]
        extra = sorted(set(counts) - set(membership_by_sample))[:5]
        raise RuntimeError(f"clean membership row coverage mismatch: missing={missing}, extra={extra}")
    wrong_counts = {
        sample_id: count
        for sample_id, count in counts.items()
        if count != expected_rows_per_sample
    }
    if wrong_counts:
        raise RuntimeError(
            "clean membership repetition count mismatch: "
            f"{list(sorted(wrong_counts.items()))[:5]}"
        )


def clustered_mean_values(
    rows: list[dict[str, Any]],
    value_function: Callable[[dict[str, Any]], float],
    *,
    expected_sample_ids: set[str],
    expected_rows_per_sample: int,
) -> tuple[list[str], np.ndarray]:
    """Average repeated conditions within image before image-level inference."""
    if expected_rows_per_sample <= 0:
        raise ValueError("expected_rows_per_sample must be positive")
    grouped: dict[str, list[float]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id not in expected_sample_ids:
            raise RuntimeError(f"unexpected sample in clustered inference: {sample_id}")
        value = float(value_function(row))
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite clustered value for {sample_id}")
        grouped.setdefault(sample_id, []).append(value)
    if set(grouped) != expected_sample_ids:
        missing = sorted(expected_sample_ids - set(grouped))[:5]
        raise RuntimeError(f"clustered inference is missing samples: {missing}")
    wrong_counts = {
        sample_id: len(values)
        for sample_id, values in grouped.items()
        if len(values) != expected_rows_per_sample
    }
    if wrong_counts:
        raise RuntimeError(
            "clustered inference row count mismatch: "
            f"{list(sorted(wrong_counts.items()))[:5]}"
        )
    keys = sorted(grouped)
    values = np.asarray(
        [np.mean(grouped[sample_id], dtype=np.float64) for sample_id in keys],
        dtype=np.float64,
    )
    return keys, values


def paired_cluster_bootstrap_mean_ci(
    values: np.ndarray,
    replicates: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Percentile CI over image-cluster means; repeated seed/SNR rows stay together."""
    materialized = np.asarray(values, dtype=np.float64)
    if materialized.ndim != 1 or materialized.size == 0:
        raise ValueError("cluster bootstrap requires a non-empty 1-D value array")
    if not np.isfinite(materialized).all():
        raise ValueError("cluster bootstrap values must be finite")
    if replicates <= 0:
        raise ValueError("cluster bootstrap replicates must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("cluster bootstrap confidence must lie in (0, 1)")
    rng = np.random.default_rng(int(seed))
    sampled_means = np.empty(int(replicates), dtype=np.float64)
    chunk_size = 256
    for start in range(0, int(replicates), chunk_size):
        count = min(chunk_size, int(replicates) - start)
        indices = rng.integers(
            0,
            materialized.size,
            size=(count, materialized.size),
            endpoint=False,
        )
        sampled_means[start : start + count] = materialized[indices].mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    low, high = np.quantile(sampled_means, [alpha, 1.0 - alpha])
    return {
        "estimate": float(materialized.mean()),
        "ci95_lower": float(low),
        "ci95_upper": float(high),
        "confidence": float(confidence),
        "replicates": int(replicates),
        "seed": int(seed),
        "num_clusters": int(materialized.size),
        "cluster_unit": "sample_id",
    }


def image_cluster_any_event_endpoint(
    rows: list[dict[str, Any]],
    denominator_function: Callable[[dict[str, Any]], bool],
    event_function: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    """Exact image-level any-event endpoint over already selected clean rows."""
    eligible_image_ids: set[str] = set()
    event_image_ids: set[str] = set()
    event_rows = 0
    denominator_rows = 0
    for row in rows:
        sample_id = str(row["sample_id"])
        denominator = bool(denominator_function(row))
        event = bool(event_function(row))
        if event and not denominator:
            raise RuntimeError(
                f"image-cluster event occurred outside its denominator: {sample_id}"
            )
        if denominator:
            denominator_rows += 1
            eligible_image_ids.add(sample_id)
        if event:
            event_rows += 1
            event_image_ids.add(sample_id)
    upper = clopper_pearson_upper_95(
        len(event_image_ids), len(eligible_image_ids)
    )
    return {
        "event_rows": event_rows,
        "denominator_rows": denominator_rows,
        "event_image_ids": event_image_ids,
        "eligible_image_ids": eligible_image_ids,
        "event_image_clusters": len(event_image_ids),
        "eligible_image_clusters": len(eligible_image_ids),
        "image_cluster_any_event_rate": len(event_image_ids) / len(eligible_image_ids),
        "image_cluster_any_event_clopper_pearson_upper_95": upper,
        "cluster_unit": "sample_id",
    }


def dual_evidence_acceptance(
    sender_risk: torch.Tensor,
    threshold: float,
    gate_anchor_probability: torch.Tensor | None = None,
    gate_posterior_probability: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return sender, receiver-guard, and intersected acceptance masks."""
    sender_accepted = sender_risk <= threshold
    if gate_anchor_probability is None and gate_posterior_probability is None:
        guard_accepted = torch.ones_like(sender_accepted)
    elif gate_anchor_probability is None or gate_posterior_probability is None:
        raise ValueError("both receiver-guard probability tensors are required")
    else:
        if gate_anchor_probability.shape != gate_posterior_probability.shape:
            raise ValueError("receiver-guard probability tensor shapes differ")
        if gate_anchor_probability.shape[0] != sender_risk.shape[0]:
            raise ValueError("receiver-guard batch size differs from sender risk")
        guard_accepted = gate_anchor_probability.argmax(dim=1) == (
            gate_posterior_probability.argmax(dim=1)
        )
    return sender_accepted, guard_accepted, sender_accepted & guard_accepted


def cross_model_triplet_acceptance(
    sender_risk: torch.Tensor,
    threshold: float,
    recovered_source_probability: torch.Tensor,
    gate_anchor_probability: torch.Tensor,
    gate_posterior_probability: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Require sender evidence plus recovered-source/anchor/posterior top-1 equality."""
    sender_accepted, receiver_guard_accepted, base_accepted = dual_evidence_acceptance(
        sender_risk,
        threshold,
        gate_anchor_probability,
        gate_posterior_probability,
    )
    if recovered_source_probability.shape != gate_anchor_probability.shape:
        raise ValueError("recovered-source and receiver-guard probability shapes differ")
    cross_model_accepted = recovered_source_probability.argmax(dim=1) == (
        gate_anchor_probability.argmax(dim=1)
    )
    return (
        sender_accepted,
        receiver_guard_accepted,
        cross_model_accepted,
        base_accepted & cross_model_accepted,
    )


def route_final_candidate(
    accepted: torch.Tensor,
    cross_model_source_anchor_accepted: torch.Tensor,
    posterior: torch.Tensor,
    anchor: torch.Tensor,
    raw: torch.Tensor,
    rejected_fallback: str,
) -> torch.Tensor:
    """Route accepted posterior samples and an explicitly configured rejected fallback."""
    if rejected_fallback == "anchor":
        fallback = anchor
    elif rejected_fallback == "raw_on_source_anchor_mismatch_else_anchor":
        fallback = torch.where(
            cross_model_source_anchor_accepted[:, None, None, None], anchor, raw
        )
    else:
        raise ValueError(f"unsupported final_routing.rejected_fallback: {rejected_fallback}")
    return torch.where(accepted[:, None, None, None], posterior, fallback)


def load_reference_rows(path: Path) -> tuple[dict[tuple[int, float, str], dict[str, str]], str]:
    digest = sha256_file(path)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keyed: dict[tuple[int, float, str], dict[str, str]] = {}
    for row in rows:
        key = (int(row["channel_seed"]), float(row["snr_db"]), row["sample_id"])
        if key in keyed:
            raise RuntimeError(f"duplicate reference row key: {key}")
        keyed[key] = row
    return keyed, digest


def reference_bool(row: dict[str, Any], field: str) -> bool:
    raw_value = row[field]
    if isinstance(raw_value, bool):
        return raw_value
    value = str(raw_value).strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(f"invalid boolean {field}={row[field]!r}")
    return value == "true"


def summarize_rate_rows(subset: list[dict[str, Any]]) -> dict[str, Any]:
    result = summarize_rows(subset)
    clean = [row for row in subset if bool(row["clean_correct"])]
    result.update(
        {
            "source_recovered_top1_agreement": sum(
                bool(row["source_recovered_top1_agree"]) for row in subset
            )
            / len(subset),
            "source_recovered_cosine": mean(subset, "source_recovered_cosine"),
            "source_recovered_l1": mean(subset, "source_recovered_l1"),
            "source_recovered_js": mean(subset, "source_recovered_js"),
            "payload_transmitted_power": mean(subset, "payload_transmitted_power"),
            "image_transmitted_power": mean(subset, "image_transmitted_power"),
            "anchor_minus_reference_anchor_psnr": mean(subset, "anchor_psnr")
            - mean(subset, "reference_anchor_psnr"),
            "anchor_minus_reference_anchor_lpips": mean(subset, "anchor_lpips")
            - mean(subset, "reference_anchor_lpips"),
            "raw_minus_reference_raw_psnr": mean(subset, "raw_psnr")
            - mean(subset, "reference_raw_psnr"),
            "raw_minus_reference_raw_lpips": mean(subset, "raw_lpips")
            - mean(subset, "reference_raw_lpips"),
            "final_minus_reference_raw_psnr": mean(subset, "final_psnr")
            - mean(subset, "reference_raw_psnr"),
            "final_minus_reference_raw_lpips": mean(subset, "final_lpips")
            - mean(subset, "reference_raw_lpips"),
            "reference_raw_failure": sum(
                not bool(row["reference_raw_correct"]) for row in clean
            ),
            "reference_raw_new": sum(
                bool(row["reference_anchor_correct"])
                and not bool(row["reference_raw_correct"])
                for row in clean
            ),
            "final_new_vs_reference_raw": sum(
                bool(row["reference_raw_correct"]) and not bool(row["final_correct"])
                for row in clean
            ),
            "final_repair_vs_reference_raw": sum(
                not bool(row["reference_raw_correct"]) and bool(row["final_correct"])
                for row in clean
            ),
            "reference_feasibility_final_failure": sum(
                not bool(row["reference_final_correct"]) for row in clean
            ),
            "perfect_payload_accept_rate": sum(
                bool(row["perfect_payload_accepted"]) for row in clean
            )
            / len(clean),
            "sender_accept_rate": sum(bool(row["sender_accepted"]) for row in clean)
            / len(clean),
            "receiver_guard_accept_rate": sum(
                bool(row["receiver_guard_accepted"]) for row in clean
            )
            / len(clean),
            "receiver_guard_extra_veto_rate": sum(
                bool(row["sender_accepted"])
                and not bool(row["receiver_guard_accepted"])
                for row in clean
            )
            / len(clean),
            "cross_model_source_anchor_accept_rate": sum(
                bool(row["cross_model_source_anchor_accepted"]) for row in clean
            )
            / len(clean),
            "cross_model_extra_veto_rate": sum(
                bool(row["sender_accepted"])
                and bool(row["receiver_guard_accepted"])
                and not bool(row["cross_model_source_anchor_accepted"])
                for row in clean
            )
            / len(clean),
            "payload_noise_decision_change_rate": sum(
                bool(row["payload_noise_changed_decision"]) for row in clean
            )
            / len(clean),
            "perfect_payload_final_failure": sum(
                not bool(row["perfect_payload_final_correct"]) for row in clean
            ),
            "perfect_payload_final_new": sum(
                bool(row["anchor_correct"])
                and not bool(row["perfect_payload_final_correct"])
                for row in clean
            ),
            "perfect_payload_final_repair": sum(
                not bool(row["anchor_correct"])
                and bool(row["perfect_payload_final_correct"])
                for row in clean
            ),
            "perfect_payload_final_minus_reference_raw_psnr": mean(
                subset, "perfect_payload_final_psnr"
            )
            - mean(subset, "reference_raw_psnr"),
            "perfect_payload_final_minus_reference_raw_lpips": mean(
                subset, "perfect_payload_final_lpips"
            )
            - mean(subset, "reference_raw_lpips"),
        }
    )
    digital = [
        row for row in subset if not math.isnan(float(row["payload_bit_error_rate"]))
    ]
    if digital:
        result.update(
            {
                "payload_bit_error_rate": mean(digital, "payload_bit_error_rate"),
                "payload_vector_exact_rate": sum(
                    bool(row["payload_vector_exact"]) for row in digital
                )
                / len(digital),
            }
        )
    return result


def paired_image_cluster_inference(
    rows: list[dict[str, Any]],
    *,
    primary_snrs: set[float],
    replicates: int,
    seed: int,
    all_sample_ids: set[str],
    clean_sample_ids: set[str],
    expected_all_rows_per_sample: int,
    expected_primary_rows_per_sample: int,
) -> dict[str, Any]:
    """Compute paired image-cluster CIs for failure, PSNR, and LPIPS deltas."""
    clean_primary_rows = [
        row
        for row in rows
        if bool(row["clean_correct"]) and float(row["snr_db"]) in primary_snrs
    ]
    failure_keys, failure_values = clustered_mean_values(
        clean_primary_rows,
        lambda row: float(not bool(row["final_correct"]))
        - float(not bool(row["reference_raw_correct"])),
        expected_sample_ids=clean_sample_ids,
        expected_rows_per_sample=expected_primary_rows_per_sample,
    )
    psnr_keys, psnr_values = clustered_mean_values(
        rows,
        lambda row: float(row["final_psnr"]) - float(row["reference_raw_psnr"]),
        expected_sample_ids=all_sample_ids,
        expected_rows_per_sample=expected_all_rows_per_sample,
    )
    lpips_keys, lpips_values = clustered_mean_values(
        rows,
        lambda row: float(row["final_lpips"]) - float(row["reference_raw_lpips"]),
        expected_sample_ids=all_sample_ids,
        expected_rows_per_sample=expected_all_rows_per_sample,
    )
    if psnr_keys != lpips_keys:
        raise RuntimeError("quality inference image clusters differ")
    failure_by_snr: dict[str, Any] = {}
    for snr_index, snr in enumerate(sorted(primary_snrs)):
        snr_rows = [
            row
            for row in clean_primary_rows
            if float(row["snr_db"]) == float(snr)
        ]
        snr_keys, snr_values = clustered_mean_values(
            snr_rows,
            lambda row: float(not bool(row["final_correct"]))
            - float(not bool(row["reference_raw_correct"])),
            expected_sample_ids=clean_sample_ids,
            expected_rows_per_sample=(
                expected_primary_rows_per_sample // len(primary_snrs)
            ),
        )
        if snr_keys != failure_keys:
            raise RuntimeError("per-SNR failure inference image clusters differ")
        failure_by_snr[str(float(snr))] = paired_cluster_bootstrap_mean_ci(
            snr_values, replicates, seed + 100 + snr_index
        )
    return {
        "cluster_unit": "sample_id",
        "resampling_contract": "resample images; retain every seed-by-SNR row per image",
        "primary_failure_rate_delta_final_minus_reference_raw": paired_cluster_bootstrap_mean_ci(
            failure_values, replicates, seed
        ),
        "primary_failure_rate_delta_by_snr": failure_by_snr,
        "all_snr_psnr_delta_final_minus_reference_raw": paired_cluster_bootstrap_mean_ci(
            psnr_values, replicates, seed + 1
        ),
        "all_snr_lpips_delta_final_minus_reference_raw": paired_cluster_bootstrap_mean_ci(
            lpips_values, replicates, seed + 2
        ),
        "primary_failure_cluster_ids_sha256": hashlib.sha256(
            "\n".join(failure_keys).encode("utf-8")
        ).hexdigest(),
        "quality_cluster_ids_sha256": hashlib.sha256(
            "\n".join(psnr_keys).encode("utf-8")
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/pc_imagenette_sender_aux_inbudget_awgn_dev.yaml"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-images", type=int)
    parser.add_argument(
        "--batch-starts",
        help="Comma-separated original policy-dev batch starts for exact replay diagnostics",
    )
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    config = load_yaml(args.config)
    official_val_accessed = config["imagenette"]["required_split"] == "official_val"
    if official_val_accessed:
        if args.dry_run or args.max_images is not None or args.batch_starts or args.output_dir:
            raise RuntimeError("official-val child forbids dry-run/subset/replay/output overrides")
        marker_path = resolve(config["imagenette"]["outcome_consumed_marker"])
        expected_marker_digest = str(config["imagenette"]["outcome_consumed_marker_sha256"])
        if (
            not marker_path.is_file()
            or len(expected_marker_digest) != 64
            or sha256_file(marker_path) != expected_marker_digest
            or os.environ.get("CADSD_OFFICIAL_VAL_AUTHORIZATION") != expected_marker_digest
        ):
            raise RuntimeError("official-val child lacks the one-shot consumed-marker authorization")
        marker_payload = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker_payload.get("state") != "OFFICIAL_VAL_OUTCOME_CONSUMED_BEFORE_MODEL_INFERENCE":
            raise RuntimeError("official-val consumed marker state mismatch")
        staging_root = resolve(config["imagenette"]["official_val_staging_root"]).resolve()
        configured_output = resolve(config["output_dir"]).resolve()
        if staging_root not in configured_output.parents:
            raise RuntimeError("official-val child output escaped the locked staging directory")
    samples, classes = load_imagenette_samples(config)
    if args.max_images is not None:
        if args.max_images <= 0:
            raise ValueError("--max-images must be positive")
        samples = samples[: args.max_images]
    batch_size = int(config["batch_size"])
    if args.batch_starts and args.max_images is not None:
        raise ValueError("--batch-starts and --max-images cannot be combined")
    if args.batch_starts:
        loop_batch_starts = [int(value) for value in args.batch_starts.split(",")]
        if len(loop_batch_starts) != len(set(loop_batch_starts)):
            raise ValueError("--batch-starts must be unique")
        if any(
            start < 0 or start >= len(samples) or start % batch_size != 0
            for start in loop_batch_starts
        ):
            raise ValueError("each diagnostic batch start must be an in-range batch boundary")
    else:
        loop_batch_starts = list(range(0, len(samples), batch_size))
    selected_samples = [
        item
        for start in loop_batch_starts
        for item in samples[start : start + batch_size]
    ]
    channel_seeds = [int(value) for value in config["channel_seeds"]]
    controller = config["controller"]
    controller_type = str(controller["type"])
    if controller_type not in {
        SENDER_ONLY_CONTROLLER,
        DUAL_EVIDENCE_CONTROLLER,
        CROSS_MODEL_TRIPLET_CONTROLLER,
    }:
        raise RuntimeError("strict-rate script requires a supported in-budget controller type")
    rejected_fallback = str(
        config.get("final_routing", {}).get("rejected_fallback", "anchor")
    )
    if rejected_fallback not in {
        "anchor",
        "raw_on_source_anchor_mismatch_else_anchor",
    }:
        raise RuntimeError(
            f"unsupported final_routing.rejected_fallback: {rejected_fallback}"
        )
    if (
        rejected_fallback == "raw_on_source_anchor_mismatch_else_anchor"
        and controller_type != CROSS_MODEL_TRIPLET_CONTROLLER
    ):
        raise RuntimeError(
            "source-anchor-mismatch fallback requires the cross-model triplet controller"
        )
    if str(controller["score"]) != "fullprob_js_risk" or float(controller["threshold"]) != 0.0:
        raise RuntimeError("strict-rate score/threshold must remain fullprob_js_risk/zero")
    if str(config["channel"]["type"]) != "AWGN" or not bool(
        config["channel"]["payload_uses_same_channel_call"]
    ):
        raise RuntimeError("semantic payload must use the same AWGN channel call")
    if bool(config["channel"]["description_is_noiseless"]):
        raise RuntimeError("strict-rate payload cannot use a noiseless description")
    payload_codec = str(controller.get("payload_codec", "analog_simplex_r16"))
    digital_payload_codecs = {
        f"digital_uint{bits}_bpsk_r4" for bits in DIGITAL_PAYLOAD_BITS
    }
    if payload_codec not in {"analog_simplex_r16", *digital_payload_codecs}:
        raise RuntimeError(f"unsupported payload codec: {payload_codec!r}")
    if payload_codec in digital_payload_codecs:
        bits_per_class = int(controller.get("bits_per_class", -1))
        repetitions = int(controller.get("repetitions", -1))
        source_probability_dim = int(controller.get("source_probability_dim", -1))
        expected_codec = f"digital_uint{bits_per_class}_bpsk_r{repetitions}"
        if bits_per_class not in DIGITAL_PAYLOAD_BITS or repetitions != 4:
            raise RuntimeError("digital payload is limited to UInt2/UInt3/UInt4 with BPSK x4")
        if payload_codec != expected_codec:
            raise RuntimeError("digital payload codec name disagrees with its bit/repetition fields")
        if int(controller["payload_vector_dim"]) != source_probability_dim * bits_per_class:
            raise RuntimeError("digital payload dimension must equal classes times bits per class")
    if controller_type == CROSS_MODEL_TRIPLET_CONTROLLER:
        frozen_contract = {
            "source_probability_dim": 10,
            "repetitions": 4,
        }
        for field, expected in frozen_contract.items():
            if int(controller.get(field, -1)) != expected:
                raise RuntimeError(
                    f"cross-model triplet contract changed: {field}={controller.get(field)!r}"
                )
        expected_strings = {
            "reserved_index_rule": "evenly_spread_floor",
            "accept_rule": (
                "source_fullprob_js_risk <= 0 AND recovered_G_aux_source_top1 == "
                "G_gate_anchor_top1 == G_gate_posterior_top1"
            ),
        }
        for field, expected in expected_strings.items():
            if str(controller.get(field)) != expected:
                raise RuntimeError(f"cross-model triplet contract changed: {field}")
        for field in (
            "erase_reserved_symbols_before_decoder",
            "consistency_excludes_reserved_symbols",
        ):
            if controller.get(field) is not True:
                raise RuntimeError(f"cross-model triplet contract requires {field}=true")
        if int(controller.get("reserved_real_symbols", -1)) != (
            int(controller["payload_vector_dim"]) * int(controller["repetitions"])
        ):
            raise RuntimeError("cross-model reserved symbols disagree with the payload contract")

    receiver_guard_config = None
    receiver_guard_checkpoint = None
    receiver_guard_digest = None
    if controller_type in {DUAL_EVIDENCE_CONTROLLER, CROSS_MODEL_TRIPLET_CONTROLLER}:
        receiver_guard_config = controller.get("receiver_guard")
        if not isinstance(receiver_guard_config, dict):
            raise RuntimeError("dual-evidence controller requires receiver_guard config")
        if str(receiver_guard_config.get("type")) != "imagenette_scratch_anchor_top1_guard":
            raise RuntimeError("receiver guard type must remain anchor top-1 agreement")
        if str(receiver_guard_config.get("expected_role")) != "G_gate":
            raise RuntimeError("receiver guard checkpoint role must remain G_gate")
        if str(receiver_guard_config.get("accept_rule")) != (
            "G_gate(posterior).top1 == G_gate(anchor).top1"
        ):
            raise RuntimeError("receiver guard accept rule differs from the frozen natural rule")
        receiver_guard_checkpoint = resolve(str(receiver_guard_config["checkpoint"]))
        receiver_guard_digest = sha256_file(receiver_guard_checkpoint)
        expected_guard_digest = str(receiver_guard_config.get("checkpoint_sha256", ""))
        if not expected_guard_digest or receiver_guard_digest != expected_guard_digest:
            raise RuntimeError("receiver guard checkpoint hash mismatch")
        distinct_checkpoints = {
            resolve(str(controller["checkpoint"])).resolve(),
            receiver_guard_checkpoint.resolve(),
            resolve(str(config["imagenette"]["evaluator_checkpoint"])).resolve(),
        }
        if len(distinct_checkpoints) != 3:
            raise RuntimeError("G_aux, G_gate, and T_cls checkpoints must be distinct")
        cross_model_required = bool(
            receiver_guard_config.get("require_recovered_source_top1_match", False)
        )
        if cross_model_required != (controller_type == CROSS_MODEL_TRIPLET_CONTROLLER):
            raise RuntimeError("cross-model triplet controller contract mismatch")

    accounting = semantic_payload_accounting(
        int(config["rate"]["total_inner_channel"]),
        int(config["rate"]["image_size"]),
        int(controller["payload_vector_dim"]),
        int(controller["repetitions"]),
    )
    expected_accounting = {
        "total_real_symbols": int(config["rate"]["total_real_symbols"]),
        "payload_real_symbols": int(config["rate"]["payload_real_symbols"]),
        "structure_real_symbols_after_reservation": int(
            config["rate"]["image_real_symbols_after_reservation"]
        ),
    }
    for field, expected in expected_accounting.items():
        if int(accounting[field]) != expected:
            raise RuntimeError(
                f"rate accounting mismatch for {field}: {accounting[field]} != {expected}"
            )
    if abs(
        float(accounting["payload_fraction_of_structure"])
        - float(config["rate"]["payload_fraction_of_total"])
    ) > 1e-15:
        raise RuntimeError("payload fraction mismatch")
    if float(config["rate"]["reference_cbr"]) != float(config["rate"]["total_cbr"]):
        raise RuntimeError("total CBR differs from the c=8 reference")
    if controller_type == CROSS_MODEL_TRIPLET_CONTROLLER:
        frozen_rate = config["rate"]
        if (
            int(frozen_rate["image_size"]) != 256
            or int(frozen_rate["reference_inner_channel"]) != 8
            or int(frozen_rate["total_inner_channel"]) != 8
            or int(frozen_rate["total_real_symbols"]) != 65536
            or abs(float(frozen_rate["total_cbr"]) - 1.0 / 6.0) > 1e-15
            or abs(float(frozen_rate["reference_cbr"]) - 1.0 / 6.0) > 1e-15
        ):
            raise RuntimeError("cross-model triplet rate contract differs from frozen c=8/1/6")

    reference_mode = str(config.get("reference_mode", "external_csv"))
    if reference_mode not in {"external_csv", "paired_unpunctured_same_noise"}:
        raise RuntimeError(f"unsupported reference_mode: {reference_mode!r}")
    reference_path: Path | None = None
    reference_rows: dict[tuple[int, float, str], dict[str, Any]] = {}
    reference_digest: str | None = None
    if reference_mode == "external_csv":
        reference_path = resolve(config["reference_noiseless_feasibility_csv"])
        reference_rows, reference_digest = load_reference_rows(reference_path)
        expected_reference_digest = config.get("reference_noiseless_feasibility_csv_sha256")
        if expected_reference_digest and reference_digest != str(expected_reference_digest):
            raise RuntimeError("reference per-sample CSV hash mismatch")
    expected_keys = {
        (seed, float(snr), str(item["sample_id"]))
        for seed in channel_seeds
        for snr in config["snrs"]
        for item in selected_samples
    }
    if reference_mode == "external_csv" and not expected_keys.issubset(reference_rows):
        missing = sorted(expected_keys - set(reference_rows))[:5]
        raise RuntimeError(f"reference CSV is missing required keys: {missing}")

    statistical_inference = config.get("statistical_inference")
    if statistical_inference is not None:
        if not isinstance(statistical_inference, dict):
            raise RuntimeError("statistical_inference must be a mapping")
        if str(statistical_inference.get("cluster_unit")) != "sample_id":
            raise RuntimeError("strict promotion inference must cluster by sample_id")
        if statistical_inference.get("retain_all_seed_snr_rows_per_resampled_image") is not True:
            raise RuntimeError("strict promotion inference must retain all seed/SNR rows")
        if int(statistical_inference.get("bootstrap_replicates", 0)) <= 0:
            raise RuntimeError("strict promotion bootstrap_replicates must be positive")
        if "bootstrap_seed" not in statistical_inference:
            raise RuntimeError("strict promotion bootstrap_seed is required")

    dry_payload = {
        "analysis_id": config["analysis_id"],
        "images": len(selected_samples),
        "original_batch_starts": loop_batch_starts if args.batch_starts else None,
        "classes": classes,
        "channel_seeds": channel_seeds,
        "snrs": config["snrs"],
        "rate_accounting": accounting,
        "payload_codec": payload_codec,
        "controller_type": controller_type,
        "final_routing_rejected_fallback": rejected_fallback,
        "receiver_guard_checkpoint": (
            str(receiver_guard_checkpoint) if receiver_guard_checkpoint is not None else None
        ),
        "receiver_guard_checkpoint_sha256": receiver_guard_digest,
        "reference_mode": reference_mode,
        "reference_csv": str(reference_path) if reference_path is not None else None,
        "reference_csv_sha256": reference_digest,
        "required_reference_keys": len(expected_keys),
        "statistical_inference": statistical_inference,
        "official_val_accessed": official_val_accessed,
    }
    if args.dry_run:
        print(json.dumps(dry_payload, indent=2))
        return

    output = resolve(args.output_dir or config["output_dir"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / "samples").mkdir()
    device = torch.device(args.device)

    source_config = load_yaml(config["source_config"])
    jscc = load_deepjscc_model(
        resolve(source_config["baseline"]["repo"]),
        resolve(config["deepjscc_checkpoint"]),
        int(source_config["rate"]["inner_channel"]),
        "AWGN",
        float(config["snrs"][0]),
        device,
    ).requires_grad_(False)
    b1_config = load_yaml(config["b1_config"])
    b1 = build_model(b1_config).to(device)
    b1.load_state_dict(
        torch.load(resolve(config["b1_checkpoint"]), map_location=device)["model_state_dict"]
    )
    b1.eval().requires_grad_(False)
    diffusion_config = load_yaml(config["diffusion_config"])
    diffusion = ShortChainResidualShiftDiffusion(diffusion_config).to(device)
    diffusion.load_state_dict(
        torch.load(resolve(config["diffusion_checkpoint"]), map_location=device)[
            "model_state_dict"
        ]
    )
    diffusion.eval().requires_grad_(False)
    sender, sender_temperature = load_scratch_classifier(
        str(controller["checkpoint"]), classes, device, str(controller["expected_role"])
    )
    receiver_guard = None
    receiver_guard_temperature = None
    if receiver_guard_checkpoint is not None:
        assert receiver_guard_config is not None
        receiver_guard, receiver_guard_temperature = load_scratch_classifier(
            str(receiver_guard_checkpoint),
            classes,
            device,
            str(receiver_guard_config["expected_role"]),
        )
    evaluator, evaluator_temperature = load_scratch_classifier(
        str(config["imagenette"]["evaluator_checkpoint"]), classes, device, "T_cls"
    )
    lpips_model, lpips_error = try_load_lpips(device, resolve("outputs/cache"))
    if lpips_model is None:
        raise RuntimeError(lpips_error)
    image_transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()]
    )

    rows: list[dict[str, Any]] = []
    clean_membership_by_sample: dict[str, dict[str, Any]] = {}
    repetitions = int(controller["repetitions"])
    probability_dim = int(controller["source_probability_dim"])
    payload_vector_dim = int(controller["payload_vector_dim"])
    for channel_seed in channel_seeds:
        for snr in map(float, config["snrs"]):
            jscc.change_channel("AWGN", snr)
            for start in loop_batch_starts:
                batch = samples[start : start + batch_size]
                target = torch.stack(
                    [image_transform(Image.open(item["path"]).convert("RGB")) for item in batch]
                ).to(device)
                labels = torch.tensor([int(item["class_idx"]) for item in batch], device=device)
                with torch.no_grad():
                    source_probability = evaluate_probabilities(
                        sender, sender_temperature, target, config
                    )
                    source_bits = None
                    if payload_codec == "analog_simplex_r16":
                        channel_free_probability = source_probability
                        source_sketch = probabilities_to_simplex_sketch(source_probability)
                    else:
                        source_codes, channel_free_probability = quantize_probabilities_uniform(
                            source_probability, int(controller["bits_per_class"])
                        )
                        source_bits = integer_codes_to_bits(
                            source_codes, int(controller["bits_per_class"])
                        )
                        source_sketch = source_bits.to(source_probability.dtype).mul(2.0).sub(1.0)
                    latent = deepjscc_encode(jscc, target)
                    reference_received = None
                    if reference_mode == "paired_unpunctured_same_noise":
                        torch.manual_seed(derived_seed(channel_seed, snr, start))
                        reference_received = deepjscc_transmit(jscc, latent)
                    transmitted, reserved = embed_repeated_sketch(
                        latent, source_sketch, repetitions
                    )
                    if transmitted[0].numel() != int(accounting["total_real_symbols"]):
                        raise RuntimeError("runtime c=8 latent size differs from rate contract")
                    torch.manual_seed(derived_seed(channel_seed, snr, start))
                    received = deepjscc_transmit(jscc, transmitted)
                    recovered_sketch, erased_received = recover_repeated_sketch_and_erase(
                        received, payload_vector_dim, repetitions, reserved
                    )
                    if payload_codec == "analog_simplex_r16":
                        recovered_bits = None
                        recovered_probability = simplex_sketch_to_probabilities(
                            recovered_sketch
                        )
                    else:
                        assert source_bits is not None
                        recovered_bits = (recovered_sketch > 0).to(torch.int64)
                        recovered_codes = bits_to_integer_codes(
                            recovered_bits,
                            probability_dim,
                            int(controller["bits_per_class"]),
                        )
                        decoded = recovered_codes.to(source_probability.dtype)
                        decoded_totals = decoded.sum(dim=1, keepdim=True)
                        recovered_probability = torch.where(
                            decoded_totals > 0,
                            decoded / decoded_totals.clamp_min(1.0),
                            torch.full_like(decoded, 1.0 / probability_dim),
                        )
                    b0 = deepjscc_decode(jscc, erased_received)

                valid_mask = torch.ones(
                    received[0].numel(), device=device, dtype=torch.bool
                )
                valid_mask[reserved] = False
                snr_tensor = torch.full((len(batch),), snr, device=device)
                snr_norm = snr_tensor / 20.0
                with torch.no_grad():
                    reference_anchor = None
                    reference_raw = None
                    if reference_received is not None:
                        reference_b0 = deepjscc_decode(jscc, reference_received)
                        reference_anchor = b1(
                            reference_b0,
                            snr_norm,
                            gate_tensor(b1_config, snr_tensor, device),
                        )
                        reference_raw = diffusion(
                            reference_anchor,
                            snr_norm,
                            gate_tensor(diffusion_config, snr_tensor, device),
                        )
                    anchor = b1(b0, snr_norm, gate_tensor(b1_config, snr_tensor, device))
                    raw = diffusion(
                        anchor,
                        snr_norm,
                        gate_tensor(diffusion_config, snr_tensor, device),
                    )
                posterior = posterior_correct(
                    jscc,
                    raw,
                    erased_received,
                    int(config["proximal_steps"]),
                    float(config["normalized_step_size"]),
                    valid_mask=valid_mask,
                )

                with torch.no_grad():
                    sender_anchor_probability = evaluate_probabilities(
                        sender, sender_temperature, anchor, config
                    )
                    sender_posterior_probability = evaluate_probabilities(
                        sender, sender_temperature, posterior, config
                    )
                    sender_scores = source_semantic_score_tensors(
                        recovered_probability,
                        sender_anchor_probability,
                        sender_posterior_probability,
                    )
                    perfect_payload_scores = source_semantic_score_tensors(
                        channel_free_probability,
                        sender_anchor_probability,
                        sender_posterior_probability,
                    )
                    receiver_guard_anchor_probability = None
                    receiver_guard_posterior_probability = None
                    receiver_guard_source_probability = None
                    receiver_guard_oracle_scores = None
                    if receiver_guard is not None:
                        assert receiver_guard_temperature is not None
                        if bool(
                            config.get("diagnostics", {}).get(
                                "receiver_guard_source_oracle", False
                            )
                        ):
                            receiver_guard_source_probability = evaluate_probabilities(
                                receiver_guard,
                                receiver_guard_temperature,
                                target,
                                config,
                            )
                        receiver_guard_anchor_probability = evaluate_probabilities(
                            receiver_guard,
                            receiver_guard_temperature,
                            anchor,
                            config,
                        )
                        receiver_guard_posterior_probability = evaluate_probabilities(
                            receiver_guard,
                            receiver_guard_temperature,
                            posterior,
                            config,
                        )
                        if receiver_guard_source_probability is not None:
                            receiver_guard_oracle_scores = source_semantic_score_tensors(
                                receiver_guard_source_probability,
                                receiver_guard_anchor_probability,
                                receiver_guard_posterior_probability,
                            )
                    if controller_type == CROSS_MODEL_TRIPLET_CONTROLLER:
                        if receiver_guard_anchor_probability is None or (
                            receiver_guard_posterior_probability is None
                        ):
                            raise RuntimeError("cross-model triplet guard probabilities missing")
                        (
                            sender_accepted,
                            receiver_guard_accepted,
                            cross_model_source_anchor_accepted,
                            accepted,
                        ) = cross_model_triplet_acceptance(
                            sender_scores["fullprob_js_risk"],
                            float(controller["threshold"]),
                            recovered_probability,
                            receiver_guard_anchor_probability,
                            receiver_guard_posterior_probability,
                        )
                        (
                            _,
                            _,
                            perfect_cross_model_source_anchor_accepted,
                            perfect_payload_accepted,
                        ) = cross_model_triplet_acceptance(
                            perfect_payload_scores["fullprob_js_risk"],
                            float(controller["threshold"]),
                            channel_free_probability,
                            receiver_guard_anchor_probability,
                            receiver_guard_posterior_probability,
                        )
                    else:
                        sender_accepted, receiver_guard_accepted, accepted = (
                            dual_evidence_acceptance(
                                sender_scores["fullprob_js_risk"],
                                float(controller["threshold"]),
                                receiver_guard_anchor_probability,
                                receiver_guard_posterior_probability,
                            )
                        )
                        cross_model_source_anchor_accepted = torch.ones_like(accepted)
                        _, _, perfect_payload_accepted = dual_evidence_acceptance(
                            perfect_payload_scores["fullprob_js_risk"],
                            float(controller["threshold"]),
                            receiver_guard_anchor_probability,
                            receiver_guard_posterior_probability,
                        )
                        perfect_cross_model_source_anchor_accepted = torch.ones_like(
                            perfect_payload_accepted
                        )
                    final = route_final_candidate(
                        accepted,
                        cross_model_source_anchor_accepted,
                        posterior,
                        anchor,
                        raw,
                        rejected_fallback,
                    )

                    batch_sample_ids = [str(item["sample_id"]) for item in batch]
                    cached_membership = [
                        sample_id in clean_membership_by_sample
                        for sample_id in batch_sample_ids
                    ]
                    if any(cached_membership) and not all(cached_membership):
                        raise RuntimeError("partial clean-membership cache hit within one batch")
                    if not any(cached_membership):
                        original_probability = evaluate_probabilities(
                            evaluator, evaluator_temperature, target, config
                        )
                        computed_confidence, computed_prediction = original_probability.max(dim=1)
                        for membership_index, item in enumerate(batch):
                            sample_id = str(item["sample_id"])
                            if sample_id in clean_membership_by_sample:
                                raise RuntimeError(
                                    f"duplicate clean-membership initialization: {sample_id}"
                                )
                            confidence = float(computed_confidence[membership_index])
                            prediction = int(computed_prediction[membership_index])
                            class_idx = int(labels[membership_index])
                            clean_membership_by_sample[sample_id] = {
                                "sample_id": sample_id,
                                "wnid": str(item["wnid"]),
                                "class_idx": class_idx,
                                "original_prediction": prediction,
                                "original_confidence": confidence,
                                "clean_correct": prediction == class_idx
                                and confidence
                                >= float(config["clean_confidence_threshold"]),
                            }
                    original_confidence = torch.tensor(
                        [
                            float(clean_membership_by_sample[sample_id]["original_confidence"])
                            for sample_id in batch_sample_ids
                        ],
                        device=device,
                        dtype=target.dtype,
                    )
                    original_prediction = torch.tensor(
                        [
                            int(clean_membership_by_sample[sample_id]["original_prediction"])
                            for sample_id in batch_sample_ids
                        ],
                        device=device,
                        dtype=torch.long,
                    )
                    b0_probability = evaluate_probabilities(
                        evaluator, evaluator_temperature, b0, config
                    )
                    anchor_probability = evaluate_probabilities(
                        evaluator, evaluator_temperature, anchor, config
                    )
                    raw_probability = evaluate_probabilities(
                        evaluator, evaluator_temperature, raw, config
                    )
                    posterior_probability = evaluate_probabilities(
                        evaluator, evaluator_temperature, posterior, config
                    )
                    final_probability = evaluate_probabilities(
                        evaluator, evaluator_temperature, final, config
                    )
                    b0_prediction = b0_probability.argmax(dim=1)
                    anchor_prediction = anchor_probability.argmax(dim=1)
                    raw_prediction = raw_probability.argmax(dim=1)
                    posterior_prediction = posterior_probability.argmax(dim=1)
                    final_prediction = final_probability.argmax(dim=1)
                    b0_psnr = psnr_per_sample(b0, target)
                    anchor_psnr = psnr_per_sample(anchor, target)
                    raw_psnr = psnr_per_sample(raw, target)
                    posterior_psnr = psnr_per_sample(posterior, target)
                    final_psnr = psnr_per_sample(final, target)
                    b0_lpips = lpips_model(b0 * 2 - 1, target * 2 - 1).flatten()
                    anchor_lpips = lpips_model(anchor * 2 - 1, target * 2 - 1).flatten()
                    raw_lpips = lpips_model(raw * 2 - 1, target * 2 - 1).flatten()
                    posterior_lpips = lpips_model(
                        posterior * 2 - 1, target * 2 - 1
                    ).flatten()
                    final_lpips = lpips_model(final * 2 - 1, target * 2 - 1).flatten()
                    reference_anchor_prediction = None
                    reference_raw_prediction = None
                    reference_anchor_psnr = None
                    reference_raw_psnr = None
                    reference_anchor_lpips = None
                    reference_raw_lpips = None
                    if reference_anchor is not None and reference_raw is not None:
                        reference_anchor_prediction = evaluate_probabilities(
                            evaluator, evaluator_temperature, reference_anchor, config
                        ).argmax(dim=1)
                        reference_raw_prediction = evaluate_probabilities(
                            evaluator, evaluator_temperature, reference_raw, config
                        ).argmax(dim=1)
                        reference_anchor_psnr = psnr_per_sample(reference_anchor, target)
                        reference_raw_psnr = psnr_per_sample(reference_raw, target)
                        reference_anchor_lpips = lpips_model(
                            reference_anchor * 2 - 1, target * 2 - 1
                        ).flatten()
                        reference_raw_lpips = lpips_model(
                            reference_raw * 2 - 1, target * 2 - 1
                        ).flatten()
                    if rejected_fallback == "anchor":
                        perfect_fallback_correct = anchor_prediction == labels
                        perfect_fallback_psnr = anchor_psnr
                        perfect_fallback_lpips = anchor_lpips
                    else:
                        perfect_fallback_correct = torch.where(
                            perfect_cross_model_source_anchor_accepted,
                            anchor_prediction == labels,
                            raw_prediction == labels,
                        )
                        perfect_fallback_psnr = torch.where(
                            perfect_cross_model_source_anchor_accepted,
                            anchor_psnr,
                            raw_psnr,
                        )
                        perfect_fallback_lpips = torch.where(
                            perfect_cross_model_source_anchor_accepted,
                            anchor_lpips,
                            raw_lpips,
                        )
                    perfect_payload_final_correct = torch.where(
                        perfect_payload_accepted,
                        posterior_prediction == labels,
                        perfect_fallback_correct,
                    )
                    perfect_payload_final_psnr = torch.where(
                        perfect_payload_accepted,
                        posterior_psnr,
                        perfect_fallback_psnr,
                    )
                    perfect_payload_final_lpips = torch.where(
                        perfect_payload_accepted,
                        posterior_lpips,
                        perfect_fallback_lpips,
                    )
                    dc_before = received_latent_consistency_per_sample(
                        jscc, raw, erased_received, valid_mask=valid_mask
                    )
                    dc_after = received_latent_consistency_per_sample(
                        jscc, posterior, erased_received, valid_mask=valid_mask
                    )
                    source_recovered_cosine = F.cosine_similarity(
                        source_probability, recovered_probability, dim=1
                    )
                    source_recovered_l1 = (
                        source_probability - recovered_probability
                    ).abs().mean(dim=1)
                    source_recovered_js = jensen_shannon(
                        source_probability, recovered_probability
                    )
                    source_recovered_top1_agree = (
                        source_probability.argmax(dim=1)
                        == recovered_probability.argmax(dim=1)
                    )
                    transmitted_flat = transmitted.flatten(start_dim=1)
                    payload_transmitted_power = transmitted_flat[:, reserved].square().mean(dim=1)
                    image_transmitted_power = transmitted_flat[:, valid_mask].square().mean(dim=1)
                    if source_bits is None:
                        payload_bit_error_rate = torch.full(
                            (len(batch),), float("nan"), device=device
                        )
                        payload_vector_exact = torch.zeros(
                            len(batch), dtype=torch.bool, device=device
                        )
                    else:
                        assert recovered_bits is not None
                        bit_errors = (source_bits != recovered_bits).sum(dim=1)
                        payload_bit_error_rate = bit_errors.float() / source_bits.shape[1]
                        payload_vector_exact = bit_errors == 0

                if args.batch_starts or start == 0:
                    count = min(8, len(batch))
                    save_image(
                        torch.cat(
                            [
                                target[:count],
                                b0[:count],
                                anchor[:count],
                                raw[:count],
                                posterior[:count],
                                final[:count],
                            ]
                        ),
                        output
                        / "samples"
                        / f"seed_{channel_seed}_snr_{int(snr):02d}_batch_{start:04d}_source_b0_anchor_raw_post_final.png",
                        nrow=count,
                    )

                for index, item in enumerate(batch):
                    key = (channel_seed, snr, str(item["sample_id"]))
                    membership = clean_membership_by_sample[str(item["sample_id"])]
                    if (
                        int(membership["class_idx"]) != int(labels[index])
                        or str(membership["wnid"]) != str(item["wnid"])
                        or int(membership["original_prediction"])
                        != int(original_prediction[index])
                        or float(membership["original_confidence"])
                        != float(original_confidence[index])
                    ):
                        raise RuntimeError(f"frozen clean membership mismatch for {key}")
                    clean = bool(membership["clean_correct"])
                    if reference_mode == "external_csv":
                        reference = reference_rows[key]
                    else:
                        assert reference_anchor_prediction is not None
                        assert reference_raw_prediction is not None
                        assert reference_anchor_psnr is not None
                        assert reference_raw_psnr is not None
                        assert reference_anchor_lpips is not None
                        assert reference_raw_lpips is not None
                        reference = {
                            "class_idx": int(labels[index]),
                            "clean_correct": clean,
                            "original_confidence": float(original_confidence[index]),
                            "anchor_psnr": float(reference_anchor_psnr[index]),
                            "anchor_lpips": float(reference_anchor_lpips[index]),
                            "anchor_correct": bool(
                                reference_anchor_prediction[index] == labels[index]
                            ),
                            "raw_correct": bool(reference_raw_prediction[index] == labels[index]),
                            "final_correct": bool(reference_raw_prediction[index] == labels[index]),
                            "raw_psnr": float(reference_raw_psnr[index]),
                            "raw_lpips": float(reference_raw_lpips[index]),
                            "final_psnr": float(reference_raw_psnr[index]),
                            "final_lpips": float(reference_raw_lpips[index]),
                        }
                    if int(reference["class_idx"]) != int(labels[index]):
                        raise RuntimeError(f"reference class mismatch for {key}")
                    if reference_bool(reference, "clean_correct") != clean:
                        raise RuntimeError(f"reference clean subset mismatch for {key}")
                    if abs(float(reference["original_confidence"]) - float(original_confidence[index])) > 1e-6:
                        raise RuntimeError(f"reference original confidence mismatch for {key}")
                    row = {
                        "channel_seed": channel_seed,
                        "snr_db": snr,
                        "sample_id": item["sample_id"],
                        "wnid": item["wnid"],
                        "class_idx": int(labels[index]),
                        "original_confidence": float(original_confidence[index]),
                        "clean_correct": clean,
                        "controller_type": str(controller["type"]),
                        "accepted": bool(accepted[index]),
                        "sender_accepted": bool(sender_accepted[index]),
                        "receiver_guard_accepted": bool(receiver_guard_accepted[index]),
                        "cross_model_source_anchor_accepted": bool(
                            cross_model_source_anchor_accepted[index]
                        ),
                        "receiver_guard_anchor_prediction": (
                            int(receiver_guard_anchor_probability[index].argmax())
                            if receiver_guard_anchor_probability is not None
                            else -1
                        ),
                        "receiver_guard_source_prediction": (
                            int(receiver_guard_source_probability[index].argmax())
                            if receiver_guard_source_probability is not None
                            else -1
                        ),
                        "receiver_guard_posterior_prediction": (
                            int(receiver_guard_posterior_probability[index].argmax())
                            if receiver_guard_posterior_probability is not None
                            else -1
                        ),
                        "receiver_guard_source_posterior_top1_agree": (
                            bool(
                                receiver_guard_source_probability[index].argmax()
                                == receiver_guard_posterior_probability[index].argmax()
                            )
                            if receiver_guard_source_probability is not None
                            and receiver_guard_posterior_probability is not None
                            else True
                        ),
                        "receiver_guard_oracle_fullprob_js_risk": (
                            float(receiver_guard_oracle_scores["fullprob_js_risk"][index])
                            if receiver_guard_oracle_scores is not None
                            else float("nan")
                        ),
                        "b0_correct": bool(b0_prediction[index] == labels[index]),
                        "anchor_correct": bool(anchor_prediction[index] == labels[index]),
                        "raw_correct": bool(raw_prediction[index] == labels[index]),
                        "posterior_correct": bool(
                            posterior_prediction[index] == labels[index]
                        ),
                        "final_correct": bool(final_prediction[index] == labels[index]),
                        "dc_before": float(dc_before[index]),
                        "dc_after": float(dc_after[index]),
                        "b0_psnr": float(b0_psnr[index]),
                        "anchor_psnr": float(anchor_psnr[index]),
                        "raw_psnr": float(raw_psnr[index]),
                        "posterior_psnr": float(posterior_psnr[index]),
                        "final_psnr": float(final_psnr[index]),
                        "b0_lpips": float(b0_lpips[index]),
                        "anchor_lpips": float(anchor_lpips[index]),
                        "raw_lpips": float(raw_lpips[index]),
                        "posterior_lpips": float(posterior_lpips[index]),
                        "final_lpips": float(final_lpips[index]),
                        "source_prediction": int(source_probability[index].argmax()),
                        "recovered_prediction": int(recovered_probability[index].argmax()),
                        "source_recovered_top1_agree": bool(
                            source_recovered_top1_agree[index]
                        ),
                        "source_recovered_cosine": float(source_recovered_cosine[index]),
                        "source_recovered_l1": float(source_recovered_l1[index]),
                        "source_recovered_js": float(source_recovered_js[index]),
                        "source_probability": json.dumps(
                            source_probability[index].cpu().tolist(), separators=(",", ":")
                        ),
                        "recovered_probability": json.dumps(
                            recovered_probability[index].cpu().tolist(), separators=(",", ":")
                        ),
                        "payload_transmitted_power": float(
                            payload_transmitted_power[index]
                        ),
                        "image_transmitted_power": float(image_transmitted_power[index]),
                        "payload_codec": payload_codec,
                        "payload_bit_error_rate": float(payload_bit_error_rate[index]),
                        "payload_vector_exact": bool(payload_vector_exact[index]),
                        **{
                            f"sender_{name}": float(values[index])
                            for name, values in sender_scores.items()
                        },
                        "perfect_payload_accepted": bool(
                            perfect_payload_accepted[index]
                        ),
                        "payload_noise_changed_decision": bool(
                            perfect_payload_accepted[index] != accepted[index]
                        ),
                        "perfect_payload_fullprob_js_risk": float(
                            perfect_payload_scores["fullprob_js_risk"][index]
                        ),
                        "perfect_payload_final_correct": bool(
                            perfect_payload_final_correct[index]
                        ),
                        "perfect_payload_final_psnr": float(
                            perfect_payload_final_psnr[index]
                        ),
                        "perfect_payload_final_lpips": float(
                            perfect_payload_final_lpips[index]
                        ),
                        "reference_anchor_psnr": float(reference["anchor_psnr"]),
                        "reference_anchor_lpips": float(reference["anchor_lpips"]),
                        "reference_anchor_correct": reference_bool(
                            reference, "anchor_correct"
                        ),
                        "reference_raw_correct": reference_bool(reference, "raw_correct"),
                        "reference_final_correct": reference_bool(
                            reference, "final_correct"
                        ),
                        "reference_raw_psnr": float(reference["raw_psnr"]),
                        "reference_raw_lpips": float(reference["raw_lpips"]),
                        "reference_final_psnr": float(reference["final_psnr"]),
                        "reference_final_lpips": float(reference["final_lpips"]),
                    }
                    rows.append(row)
            print(f"done seed={channel_seed} snr={snr:g}")

    if len(rows) != len(expected_keys):
        raise RuntimeError(f"row count mismatch: {len(rows)} != {len(expected_keys)}")
    actual_keys = {
        (int(row["channel_seed"]), float(row["snr_db"]), str(row["sample_id"]))
        for row in rows
    }
    if len(actual_keys) != len(rows) or actual_keys != expected_keys:
        raise RuntimeError("strict-rate output does not form the complete unique row grid")
    assert_clean_membership_consistency(
        rows,
        clean_membership_by_sample,
        len(channel_seeds) * len(config["snrs"]),
    )
    write_csv(output / "per_sample.csv", rows)
    summary: list[dict[str, Any]] = []
    for snr in map(float, config["snrs"]):
        subset = [row for row in rows if float(row["snr_db"]) == snr]
        summary.append({"snr_db": snr, **summarize_rate_rows(subset)})
    write_csv(output / "summary.csv", summary)
    seed_snr_summary: list[dict[str, Any]] = []
    for channel_seed in channel_seeds:
        for snr in map(float, config["snrs"]):
            subset = [
                row
                for row in rows
                if int(row["channel_seed"]) == channel_seed
                and float(row["snr_db"]) == snr
            ]
            seed_snr_summary.append(
                {
                    "channel_seed": channel_seed,
                    "snr_db": snr,
                    **summarize_rate_rows(subset),
                }
            )
    write_csv(output / "seed_snr_summary.csv", seed_snr_summary)

    clean_sample_ids = {
        sample_id
        for sample_id, membership in clean_membership_by_sample.items()
        if bool(membership["clean_correct"])
    }
    clean_images = len(clean_sample_ids)
    primary = [row for row in summary if float(row["snr_db"]) in config["primary_snrs"]]
    primary_rows = [
        row
        for row in rows
        if bool(row["clean_correct"]) and float(row["snr_db"]) in config["primary_snrs"]
    ]
    anchor_new_endpoint = image_cluster_any_event_endpoint(
        primary_rows,
        lambda row: bool(row["anchor_correct"]),
        lambda row: bool(row["anchor_correct"]) and not bool(row["final_correct"]),
    )
    eligible_image_ids = anchor_new_endpoint["eligible_image_ids"]
    final_new_image_ids = anchor_new_endpoint["event_image_ids"]
    final_new_cluster_upper = float(
        anchor_new_endpoint["image_cluster_any_event_clopper_pearson_upper_95"]
    )
    system_new_endpoint = image_cluster_any_event_endpoint(
        primary_rows,
        lambda row: bool(row["reference_raw_correct"]),
        lambda row: bool(row["reference_raw_correct"])
        and not bool(row["final_correct"]),
    )
    system_eligible_image_ids = system_new_endpoint["eligible_image_ids"]
    system_final_new_image_ids = system_new_endpoint["event_image_ids"]
    system_final_repair_image_ids = {
        str(row["sample_id"])
        for row in primary_rows
        if not bool(row["reference_raw_correct"]) and bool(row["final_correct"])
    }
    system_final_new_cluster_upper = float(
        system_new_endpoint["image_cluster_any_event_clopper_pearson_upper_95"]
    )
    criteria = config["success_criteria"]
    inference_config = config.get("statistical_inference")
    paired_inference = None
    if inference_config is not None:
        paired_inference = paired_image_cluster_inference(
            rows,
            primary_snrs={float(value) for value in config["primary_snrs"]},
            replicates=int(inference_config["bootstrap_replicates"]),
            seed=int(inference_config["bootstrap_seed"]),
            all_sample_ids=set(clean_membership_by_sample),
            clean_sample_ids=clean_sample_ids,
            expected_all_rows_per_sample=len(channel_seeds) * len(config["snrs"]),
            expected_primary_rows_per_sample=len(channel_seeds)
            * len(config["primary_snrs"]),
        )
    clean_ids_by_class = {
        wnid: {
            sample_id
            for sample_id, membership in clean_membership_by_sample.items()
            if bool(membership["clean_correct"])
            and str(membership["wnid"]) == wnid
        }
        for wnid in classes
    }
    primary_seed_rows = [
        row
        for row in seed_snr_summary
        if float(row["snr_db"]) in config["primary_snrs"]
    ]
    gates = {
        "minimum_clean_images": clean_images >= int(config["minimum_clean_images"]),
        "minimum_clean_images_each_class": all(
            len(image_ids) >= int(criteria.get("minimum_clean_images_per_class", 0))
            for image_ids in clean_ids_by_class.values()
        ),
        "exact_total_rate": int(accounting["total_real_symbols"])
        == int(config["rate"]["total_real_symbols"])
        and float(config["rate"]["reference_cbr"])
        == float(config["rate"]["total_cbr"]),
        "payload_uses_shared_awgn": bool(config["channel"]["payload_uses_same_channel_call"])
        and not bool(config["channel"]["description_is_noiseless"]),
        "source_recovered_top1_each_snr": all(
            float(row["source_recovered_top1_agreement"])
            >= float(criteria["minimum_source_recovered_top1_agreement_each_snr"])
            for row in summary
        ),
        "source_recovered_top1_each_seed_snr": all(
            float(row["source_recovered_top1_agreement"])
            >= float(criteria["minimum_source_recovered_top1_agreement_each_snr"])
            for row in seed_snr_summary
        ),
        "source_recovered_cosine_each_snr": all(
            float(row["source_recovered_cosine"])
            >= float(criteria["minimum_source_recovered_cosine_each_snr"])
            for row in summary
        ),
        "source_recovered_cosine_each_seed_snr": all(
            float(row["source_recovered_cosine"])
            >= float(criteria["minimum_source_recovered_cosine_each_snr"])
            for row in seed_snr_summary
        ),
        "masked_dc_decreases_each_snr": all(float(row["dc_delta"]) < 0 for row in summary),
        "masked_dc_decreases_each_seed_snr": all(
            float(row["dc_delta"]) < 0 for row in seed_snr_summary
        ),
        "primary_final_failure_not_above_reference_raw": sum(
            int(row["final_failure"]) for row in primary
        )
        <= sum(int(row["reference_raw_failure"]) for row in primary),
        "primary_final_failure_each_snr_not_above_reference_raw": all(
            int(row["final_failure"]) <= int(row["reference_raw_failure"])
            for row in primary
        ),
        "primary_final_new_not_above_inbudget_raw": sum(
            int(row["final_new"]) for row in primary
        )
        <= sum(int(row["raw_new"]) for row in primary),
        "primary_final_new_each_snr_not_above_inbudget_raw": all(
            int(row["final_new"]) <= int(row["raw_new"]) for row in primary
        ),
        "primary_final_failure_each_seed_not_above_reference_raw": all(
            sum(
                int(row["final_failure"])
                for row in primary_seed_rows
                if int(row["channel_seed"]) == channel_seed
            )
            <= sum(
                int(row["reference_raw_failure"])
                for row in primary_seed_rows
                if int(row["channel_seed"]) == channel_seed
            )
            for channel_seed in channel_seeds
        ),
        "primary_final_failure_each_seed_snr_not_above_reference_raw": all(
            int(row["final_failure"]) <= int(row["reference_raw_failure"])
            for row in primary_seed_rows
        ),
        "primary_final_new_each_seed_not_above_inbudget_raw": all(
            sum(
                int(row["final_new"])
                for row in primary_seed_rows
                if int(row["channel_seed"]) == channel_seed
            )
            <= sum(
                int(row["raw_new"])
                for row in primary_seed_rows
                if int(row["channel_seed"]) == channel_seed
            )
            for channel_seed in channel_seeds
        ),
        "primary_final_new_each_seed_snr_not_above_inbudget_raw": all(
            int(row["final_new"]) <= int(row["raw_new"])
            for row in primary_seed_rows
        ),
        "primary_final_new_cluster_upper_within_limit": final_new_cluster_upper
        <= float(criteria["max_primary_final_new_image_cluster_upper"]),
        "mean_final_minus_reference_raw_psnr_positive": mean(
            summary, "final_minus_reference_raw_psnr"
        )
        > 0,
        "mean_final_minus_reference_raw_lpips_nonpositive": mean(
            summary, "final_minus_reference_raw_lpips"
        )
        <= 0,
    }
    if payload_codec in digital_payload_codecs:
        gates["payload_vector_exact_rate_each_snr"] = all(
            float(row["payload_vector_exact_rate"])
            >= float(criteria["minimum_exact_payload_vector_rate_each_snr"])
            for row in summary
        )
        gates["payload_vector_exact_rate_each_seed_snr"] = all(
            float(row["payload_vector_exact_rate"])
            >= float(criteria["minimum_exact_payload_vector_rate_each_snr"])
            for row in seed_snr_summary
        )
    strict_promotion_gates: dict[str, bool] = {}
    worst_class_failure_delta = None
    if paired_inference is not None:
        failure_inference = paired_inference[
            "primary_failure_rate_delta_final_minus_reference_raw"
        ]
        psnr_inference = paired_inference["all_snr_psnr_delta_final_minus_reference_raw"]
        lpips_inference = paired_inference["all_snr_lpips_delta_final_minus_reference_raw"]
        class_failure_deltas: dict[str, float] = {}
        for wnid in classes:
            class_rows = [
                row
                for row in primary_rows
                if str(row["wnid"]) == wnid
            ]
            if not class_rows:
                raise RuntimeError(f"no clean primary rows for class {wnid}")
            class_failure_deltas[wnid] = float(
                np.mean(
                    [
                        float(not bool(row["final_correct"]))
                        - float(not bool(row["reference_raw_correct"]))
                        for row in class_rows
                    ]
                )
            )
        worst_class_failure_delta = max(class_failure_deltas.values())
        paired_inference["primary_failure_rate_delta_by_class"] = class_failure_deltas
        paired_inference["worst_class_primary_failure_rate_delta"] = (
            worst_class_failure_delta
        )
        strict_promotion_gates = {
            "paired_primary_failure_ci_upper_strictly_below_zero": float(
                failure_inference["ci95_upper"]
            )
            < 0.0,
            "paired_primary_failure_ci_upper_nonpositive_each_snr": all(
                float(endpoint["ci95_upper"]) <= 0.0
                for endpoint in paired_inference[
                    "primary_failure_rate_delta_by_snr"
                ].values()
            ),
            "paired_all_snr_psnr_ci_lower_strictly_above_zero": float(
                psnr_inference["ci95_lower"]
            )
            > 0.0,
            "paired_all_snr_lpips_ci_upper_nonpositive": float(
                lpips_inference["ci95_upper"]
            )
            <= 0.0,
            "final_minus_reference_raw_psnr_positive_each_snr": all(
                float(row["final_minus_reference_raw_psnr"]) > 0.0 for row in summary
            ),
            "worst_class_primary_failure_delta_within_0p02": (
                worst_class_failure_delta <= 0.02
            ),
            "primary_system_new_vs_reference_raw_cluster_upper_within_limit": (
                system_final_new_cluster_upper
                <= float(criteria["max_primary_final_new_image_cluster_upper"])
            ),
        }
    aggregate = {
        "image_population": str(config["imagenette"]["required_split"]),
        "image_population_images": len(selected_samples),
        "policy_dev_images": len(selected_samples),
        "original_batch_starts": loop_batch_starts if args.batch_starts else None,
        "clean_images": clean_images,
        "clean_images_per_class": {
            wnid: len(image_ids) for wnid, image_ids in clean_ids_by_class.items()
        },
        "rows": len(rows),
        "channel_seeds": channel_seeds,
        "rate_accounting": accounting,
        "payload_codec": payload_codec,
        "controller_type": controller_type,
        "final_routing_rejected_fallback": rejected_fallback,
        "receiver_guard_checkpoint": (
            str(receiver_guard_checkpoint) if receiver_guard_checkpoint is not None else None
        ),
        "receiver_guard_checkpoint_sha256": receiver_guard_digest,
        "reference_mode": reference_mode,
        "reference_csv": str(reference_path) if reference_path is not None else None,
        "reference_csv_sha256": reference_digest,
        "source_recovered_top1_agreement": sum(
            bool(row["source_recovered_top1_agree"]) for row in rows
        )
        / len(rows),
        "mean_source_recovered_cosine": mean(rows, "source_recovered_cosine"),
        "mean_source_recovered_l1": mean(rows, "source_recovered_l1"),
        "mean_source_recovered_js": mean(rows, "source_recovered_js"),
        "mean_final_minus_reference_raw_psnr": mean(
            summary, "final_minus_reference_raw_psnr"
        ),
        "mean_final_minus_reference_raw_lpips": mean(
            summary, "final_minus_reference_raw_lpips"
        ),
        "primary_reference_raw_failure": sum(
            int(row["reference_raw_failure"]) for row in primary
        ),
        "primary_inbudget_raw_failure": sum(int(row["raw_failure"]) for row in primary),
        "primary_inbudget_posterior_failure": sum(
            int(row["posterior_failure"]) for row in primary
        ),
        "primary_inbudget_final_failure": sum(
            int(row["final_failure"]) for row in primary
        ),
        "primary_inbudget_raw_new": sum(int(row["raw_new"]) for row in primary),
        "primary_inbudget_posterior_new": sum(
            int(row["posterior_new"]) for row in primary
        ),
        "primary_inbudget_final_new": sum(int(row["final_new"]) for row in primary),
        "primary_final_new_image_clusters": len(final_new_image_ids),
        "primary_final_new_eligible_image_clusters": len(eligible_image_ids),
        "primary_final_new_image_cluster_clopper_pearson_upper_95": final_new_cluster_upper,
        "primary_system_new_vs_reference_raw_rows": int(
            system_new_endpoint["event_rows"]
        ),
        "primary_system_new_vs_reference_raw_denominator_rows": int(
            system_new_endpoint["denominator_rows"]
        ),
        "primary_system_repair_vs_reference_raw_rows": sum(
            int(row["final_repair_vs_reference_raw"]) for row in primary
        ),
        "primary_system_new_vs_reference_raw_image_clusters": len(
            system_final_new_image_ids
        ),
        "primary_system_repair_vs_reference_raw_image_clusters": len(
            system_final_repair_image_ids
        ),
        "primary_system_new_vs_reference_raw_eligible_image_clusters": len(
            system_eligible_image_ids
        ),
        "primary_system_new_vs_reference_raw_clopper_pearson_upper_95": (
            system_final_new_cluster_upper
        ),
        "accept_rate": sum(bool(row["accepted"]) for row in rows) / len(rows),
        "sender_accept_rate": sum(bool(row["sender_accepted"]) for row in rows)
        / len(rows),
        "receiver_guard_accept_rate": sum(
            bool(row["receiver_guard_accepted"]) for row in rows
        )
        / len(rows),
        "receiver_guard_extra_veto_rate": sum(
            bool(row["sender_accepted"])
            and not bool(row["receiver_guard_accepted"])
            for row in rows
        )
        / len(rows),
        "cross_model_source_anchor_accept_rate": sum(
            bool(row["cross_model_source_anchor_accepted"]) for row in rows
        )
        / len(rows),
        "cross_model_extra_veto_rate": sum(
            bool(row["sender_accepted"])
            and bool(row["receiver_guard_accepted"])
            and not bool(row["cross_model_source_anchor_accepted"])
            for row in rows
        )
        / len(rows),
        "perfect_payload_accept_rate": sum(
            bool(row["perfect_payload_accepted"]) for row in rows
        )
        / len(rows),
        "payload_noise_decision_change_rate": sum(
            bool(row["payload_noise_changed_decision"]) for row in rows
        )
        / len(rows),
        "primary_perfect_payload_final_failure": sum(
            int(row["perfect_payload_final_failure"]) for row in primary
        ),
        "primary_perfect_payload_final_new": sum(
            int(row["perfect_payload_final_new"]) for row in primary
        ),
        "primary_perfect_payload_final_repair": sum(
            int(row["perfect_payload_final_repair"]) for row in primary
        ),
        "official_val_accessed": official_val_accessed,
    }
    if payload_codec in digital_payload_codecs:
        aggregate.update(
            {
                "payload_bit_error_rate": mean(rows, "payload_bit_error_rate"),
                "payload_vector_exact_rate": sum(
                    bool(row["payload_vector_exact"]) for row in rows
                )
                / len(rows),
            }
        )
    tradeoff_pass = all(gates.values())
    strict_promotion_pass = tradeoff_pass and all(strict_promotion_gates.values())
    if strict_promotion_gates:
        verdict = (
            "POSITIVE"
            if strict_promotion_pass
            else "PARTIAL_TRADEOFF_POSITIVE"
            if tradeoff_pass
            else "NEGATIVE"
        )
    else:
        verdict = "POSITIVE" if tradeoff_pass else "NEGATIVE"
    payload = {
        "config": config,
        "aggregate": aggregate,
        "summary": summary,
        "seed_snr_summary": seed_snr_summary,
        "gates": gates,
        "strict_promotion_gates": strict_promotion_gates,
        "paired_image_cluster_inference": paired_inference,
        "tradeoff_gates_pass": tradeoff_pass,
        "strict_promotion_gates_pass": strict_promotion_pass,
        "verdict": verdict,
    }
    (output / "metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps({"aggregate": aggregate, "gates": gates, "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
