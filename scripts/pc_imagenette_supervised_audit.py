#!/usr/bin/env python3
"""Supervised Imagenette policy-dev audit for the frozen posterior method."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from cadsd_jscc.deepjscc_adapter import (  # noqa: E402
    deepjscc_forward_with_latents,
    load_deepjscc_model,
    received_latent_consistency_per_sample,
)
from cadsd_jscc.metrics import psnr_per_sample  # noqa: E402
from pc_posterior_consistency_replication import (  # noqa: E402
    classify,
    load_classifier,
    load_yaml,
    mean,
    posterior_correct,
    resolve,
    write_csv,
)
from s10_short_chain_residual_shift_diffusion import ShortChainResidualShiftDiffusion  # noqa: E402
from s13_export_coco_train2017_c8_scaleup import derived_seed  # noqa: E402
from s5_residual_refiner_pilot import build_model, gate_tensor, try_load_lpips  # noqa: E402
from s6_imagenette_source_semantic_description_eval import quantize_description  # noqa: E402


RECEIVER_RISK_FEATURE_VERSION = "receiver_risk_v1"
RECEIVER_RISK_FEATURE_COLUMNS = (
    "snr_db",
    "dc_before",
    "dc_after",
    "dc_delta",
    "anchor_raw_l1",
    "anchor_raw_rmse",
    "anchor_posterior_l1",
    "anchor_posterior_rmse",
    "raw_posterior_l1",
    "raw_posterior_rmse",
    "gate_anchor_confidence",
    "gate_raw_confidence",
    "gate_posterior_confidence",
    "gate_anchor_entropy",
    "gate_raw_entropy",
    "gate_posterior_entropy",
    "gate_anchor_margin",
    "gate_raw_margin",
    "gate_posterior_margin",
    "gate_anchor_posterior_js",
    "gate_raw_posterior_js",
    "gate_anchor_class_retention",
    "gate_raw_class_retention",
    "gate_anchor_posterior_top1_changed",
    "gate_raw_posterior_top1_changed",
    "aux_anchor_confidence",
    "aux_raw_confidence",
    "aux_posterior_confidence",
    "aux_anchor_entropy",
    "aux_raw_entropy",
    "aux_posterior_entropy",
    "aux_anchor_margin",
    "aux_raw_margin",
    "aux_posterior_margin",
    "aux_anchor_posterior_js",
    "aux_raw_posterior_js",
    "aux_anchor_class_retention",
    "aux_raw_class_retention",
    "aux_anchor_posterior_top1_changed",
    "aux_raw_posterior_top1_changed",
    "ensemble_anchor_top1_agree",
    "ensemble_raw_top1_agree",
    "ensemble_posterior_top1_agree",
)

TEACHER_RISK_TARGET_COLUMNS = (
    "teacher_original_confidence",
    "teacher_clean_correct",
    "teacher_anchor_correct",
    "teacher_raw_correct",
    "teacher_posterior_correct",
    "teacher_new_error",
    "teacher_anchor_true_probability",
    "teacher_raw_true_probability",
    "teacher_posterior_true_probability",
    "teacher_posterior_minus_anchor_true_probability",
    "teacher_anchor_true_margin",
    "teacher_raw_true_margin",
    "teacher_posterior_true_margin",
    "teacher_posterior_minus_anchor_true_margin",
)


def load_imagenette_samples(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load the declared Imagenette population under a fail-closed split contract.

    Development runs continue to use the classifier-training manifest's
    ``policy_dev`` records.  A final audit may instead use an independently
    materialized official-validation manifest, but only when the config
    explicitly records that access and pins the manifest SHA-256.
    """
    imagenette = config["imagenette"]
    training_manifest_path = resolve(imagenette["split_manifest"])
    manifest = json.loads(training_manifest_path.read_text())
    if bool(manifest.get("official_val_accessed")):
        raise RuntimeError("split manifest records official validation access")
    required = str(imagenette["required_split"])
    classes = [str(item) for item in manifest["classes"]]
    if required == "policy_dev":
        if imagenette.get("official_val_accessed") is not False:
            raise RuntimeError("policy_dev config must assert official_val_accessed=false")
        samples = [dict(item) for item in manifest["samples"] if item["split"] == required]
        root = resolve(manifest["source_train_root"])
    elif required == "official_val":
        if imagenette.get("official_val_accessed") is not True:
            raise RuntimeError("official_val requires explicit official_val_accessed=true")
        official_manifest_path = resolve(imagenette["official_val_manifest"])
        expected_digest = str(imagenette.get("official_val_manifest_sha256", ""))
        if len(expected_digest) != 64 or sha256_file(official_manifest_path) != expected_digest:
            raise RuntimeError("official validation manifest SHA-256 mismatch")
        official_manifest = json.loads(official_manifest_path.read_text())
        if str(official_manifest.get("role")) != "sealed_official_val_final_test":
            raise RuntimeError("official validation manifest role mismatch")
        if list(official_manifest.get("classes", [])) != classes:
            raise RuntimeError("official validation class order differs from training manifest")
        training_digest = sha256_file(training_manifest_path)
        if str(official_manifest.get("training_split_manifest_sha256")) != training_digest:
            raise RuntimeError("official validation manifest pins a different training manifest")
        samples = [dict(item) for item in official_manifest.get("samples", [])]
        if int(official_manifest.get("sample_count", -1)) != len(samples):
            raise RuntimeError("official validation manifest sample count mismatch")
        expected_count = int(imagenette.get("official_val_expected_count", -1))
        if expected_count <= 0 or len(samples) != expected_count:
            raise RuntimeError("official validation population differs from the frozen count")
        root = resolve(official_manifest["source_val_root"])
        if not root.is_dir():
            raise FileNotFoundError(root)
        if any(str(item.get("split")) != "official_val" for item in samples):
            raise RuntimeError("official validation manifest contains a non-final sample")
    else:
        raise RuntimeError(f"unsupported Imagenette audit split: {required!r}")

    seen: set[str] = set()
    for item in samples:
        sample_id = str(item["sample_id"])
        if sample_id in seen:
            raise RuntimeError(f"duplicate Imagenette sample_id: {sample_id}")
        seen.add(sample_id)
        path = (root / str(item["relative_path"])).resolve()
        if root.resolve() not in path.parents or not path.is_file():
            raise FileNotFoundError(path)
        item["path"] = path
        if required == "official_val":
            if path.stat().st_size != int(item["size_bytes"]):
                raise RuntimeError(f"official validation size mismatch: {sample_id}")
            if bool(imagenette.get("verify_official_val_content_sha256", True)):
                if sha256_file(path) != str(item["content_sha256"]):
                    raise RuntimeError(f"official validation content mismatch: {sample_id}")
    return samples, classes


def load_policy_dev(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Backward-compatible policy-development-only loader."""
    if str(config["imagenette"]["required_split"]) != "policy_dev":
        raise RuntimeError("load_policy_dev cannot open a non-policy_dev population")
    return load_imagenette_samples(config)


def load_scratch_classifier(
    checkpoint_path: str,
    classes: list[str],
    device: torch.device,
    expected_role: str,
):
    path = resolve(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if bool(checkpoint.get("pretrained")) or not bool(checkpoint.get("random_initialization")):
        raise RuntimeError(f"{expected_role} is not the required scratch classifier")
    if checkpoint.get("weights") is not None:
        raise RuntimeError(f"{expected_role} checkpoint does not record weights=None")
    if str(checkpoint.get("role")) != expected_role:
        raise RuntimeError(
            f"scratch classifier role mismatch: checkpoint={checkpoint.get('role')!r}, "
            f"expected={expected_role!r}"
        )
    if checkpoint.get("quality_gate_passed") is not True:
        raise RuntimeError(f"{expected_role} classifier quality gate did not pass")
    if bool(checkpoint.get("official_val_accessed")):
        raise RuntimeError(f"{expected_role} checkpoint records official validation access")
    if checkpoint.get("policy_dev_used_for_training_selection_or_calibration") is not False:
        raise RuntimeError(f"{expected_role} checkpoint does not prove policy-dev separation")
    if str(checkpoint.get("training_split")) != "cls_train":
        raise RuntimeError(f"{expected_role} was not trained on cls_train")
    if str(checkpoint.get("selection_split")) != "cls_cal":
        raise RuntimeError(f"{expected_role} was not selected on cls_cal")
    if str(checkpoint.get("temperature_scaling_split")) != "cls_cal":
        raise RuntimeError(f"{expected_role} was not calibrated on cls_cal")
    if list(checkpoint["idx_to_class"]) != classes:
        raise RuntimeError(f"{expected_role} class order differs from split manifest")
    architecture = str(checkpoint.get("architecture"))
    builders = {
        "mobilenet_v3_small": models.mobilenet_v3_small,
        "resnet18": models.resnet18,
        "efficientnet_b0": models.efficientnet_b0,
    }
    if architecture not in builders:
        raise RuntimeError(f"unsupported scratch classifier architecture: {architecture!r}")
    model = builders[architecture](weights=None, num_classes=len(classes))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    temperature = float(checkpoint["temperature"])
    if not 0 < temperature < float("inf"):
        raise RuntimeError(f"{expected_role} has invalid temperature {temperature}")
    return model.to(device).eval().requires_grad_(False), temperature


@torch.no_grad()
def evaluate_probabilities(
    model,
    temperature: float,
    images: torch.Tensor,
    config: dict[str, Any],
) -> torch.Tensor:
    mean_tensor = torch.tensor(
        config["imagenette"]["normalization_mean"], device=images.device
    )[None, :, None, None]
    std_tensor = torch.tensor(
        config["imagenette"]["normalization_std"], device=images.device
    )[None, :, None, None]
    return torch.softmax(model((images - mean_tensor) / std_tensor) / temperature, dim=1)


@torch.no_grad()
def evaluate(model, temperature: float, images: torch.Tensor, config: dict[str, Any]):
    return evaluate_probabilities(model, temperature, images, config).max(dim=1)


def normalized_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must have shape [batch, classes>=2]")
    safe = probabilities.clamp_min(1e-12)
    return -(safe * safe.log()).sum(dim=1) / math.log(probabilities.shape[1])


def probability_margin(probabilities: torch.Tensor) -> torch.Tensor:
    top2 = probabilities.topk(k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def jensen_shannon(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    midpoint = 0.5 * (first + second)
    first_safe = first.clamp_min(1e-12)
    second_safe = second.clamp_min(1e-12)
    midpoint_safe = midpoint.clamp_min(1e-12)
    return 0.5 * (
        (first_safe * (first_safe.log() - midpoint_safe.log())).sum(dim=1)
        + (second_safe * (second_safe.log() - midpoint_safe.log())).sum(dim=1)
    )


def distribution_pair_features(
    prefix: str,
    anchor_probabilities: torch.Tensor,
    raw_probabilities: torch.Tensor,
    posterior_probabilities: torch.Tensor,
) -> dict[str, torch.Tensor]:
    states = {
        "anchor": anchor_probabilities,
        "raw": raw_probabilities,
        "posterior": posterior_probabilities,
    }
    features: dict[str, torch.Tensor] = {}
    for state, probabilities in states.items():
        features[f"{prefix}_{state}_confidence"] = probabilities.max(dim=1).values
    for state, probabilities in states.items():
        features[f"{prefix}_{state}_entropy"] = normalized_entropy(probabilities)
    for state, probabilities in states.items():
        features[f"{prefix}_{state}_margin"] = probability_margin(probabilities)
    anchor_pred = anchor_probabilities.argmax(dim=1)
    raw_pred = raw_probabilities.argmax(dim=1)
    posterior_pred = posterior_probabilities.argmax(dim=1)
    features[f"{prefix}_anchor_posterior_js"] = jensen_shannon(
        anchor_probabilities, posterior_probabilities
    )
    features[f"{prefix}_raw_posterior_js"] = jensen_shannon(
        raw_probabilities, posterior_probabilities
    )
    features[f"{prefix}_anchor_class_retention"] = (
        posterior_probabilities.gather(1, anchor_pred[:, None]).squeeze(1)
        - anchor_probabilities.gather(1, anchor_pred[:, None]).squeeze(1)
    )
    features[f"{prefix}_raw_class_retention"] = (
        posterior_probabilities.gather(1, raw_pred[:, None]).squeeze(1)
        - raw_probabilities.gather(1, raw_pred[:, None]).squeeze(1)
    )
    features[f"{prefix}_anchor_posterior_top1_changed"] = (
        posterior_pred != anchor_pred
    ).float()
    features[f"{prefix}_raw_posterior_top1_changed"] = (
        posterior_pred != raw_pred
    ).float()
    return features


def receiver_risk_feature_tensors(
    *,
    snr_db: float,
    dc_before: torch.Tensor,
    dc_after: torch.Tensor,
    anchor: torch.Tensor,
    raw: torch.Tensor,
    posterior: torch.Tensor,
    gate_anchor_probabilities: torch.Tensor,
    gate_raw_probabilities: torch.Tensor,
    gate_posterior_probabilities: torch.Tensor,
    aux_anchor_probabilities: torch.Tensor,
    aux_raw_probabilities: torch.Tensor,
    aux_posterior_probabilities: torch.Tensor,
) -> dict[str, torch.Tensor]:
    batch_size = anchor.shape[0]
    features: dict[str, torch.Tensor] = {
        "snr_db": torch.full(
            (batch_size,), float(snr_db), device=anchor.device, dtype=anchor.dtype
        ),
        "dc_before": dc_before,
        "dc_after": dc_after,
        "dc_delta": dc_after - dc_before,
        "anchor_raw_l1": (anchor - raw).abs().flatten(1).mean(dim=1),
        "anchor_raw_rmse": (anchor - raw).square().flatten(1).mean(dim=1).sqrt(),
        "anchor_posterior_l1": (anchor - posterior).abs().flatten(1).mean(dim=1),
        "anchor_posterior_rmse": (
            (anchor - posterior).square().flatten(1).mean(dim=1).sqrt()
        ),
        "raw_posterior_l1": (raw - posterior).abs().flatten(1).mean(dim=1),
        "raw_posterior_rmse": (
            (raw - posterior).square().flatten(1).mean(dim=1).sqrt()
        ),
    }
    features.update(
        distribution_pair_features(
            "gate",
            gate_anchor_probabilities,
            gate_raw_probabilities,
            gate_posterior_probabilities,
        )
    )
    features.update(
        distribution_pair_features(
            "aux",
            aux_anchor_probabilities,
            aux_raw_probabilities,
            aux_posterior_probabilities,
        )
    )
    features.update(
        {
            "ensemble_anchor_top1_agree": (
                gate_anchor_probabilities.argmax(dim=1)
                == aux_anchor_probabilities.argmax(dim=1)
            ).float(),
            "ensemble_raw_top1_agree": (
                gate_raw_probabilities.argmax(dim=1)
                == aux_raw_probabilities.argmax(dim=1)
            ).float(),
            "ensemble_posterior_top1_agree": (
                gate_posterior_probabilities.argmax(dim=1)
                == aux_posterior_probabilities.argmax(dim=1)
            ).float(),
        }
    )
    if tuple(features) != RECEIVER_RISK_FEATURE_COLUMNS:
        missing = sorted(set(RECEIVER_RISK_FEATURE_COLUMNS) - set(features))
        extra = sorted(set(features) - set(RECEIVER_RISK_FEATURE_COLUMNS))
        raise RuntimeError(f"receiver risk feature schema mismatch: missing={missing}, extra={extra}")
    return features


def true_class_margin(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    true_probability = probabilities.gather(1, labels[:, None]).squeeze(1)
    masked = probabilities.clone()
    masked.scatter_(1, labels[:, None], -1.0)
    return true_probability - masked.max(dim=1).values


def source_semantic_score_tensors(
    source_probability: torch.Tensor,
    anchor_probability: torch.Tensor,
    posterior_probability: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if not (
        source_probability.shape == anchor_probability.shape == posterior_probability.shape
    ):
        raise ValueError("source/anchor/posterior probability shapes must match")
    safe_source = source_probability.clamp_min(1e-12)
    safe_anchor = anchor_probability.clamp_min(1e-12)
    safe_posterior = posterior_probability.clamp_min(1e-12)
    source_class = source_probability.argmax(dim=1)
    anchor_cosine = torch.nn.functional.cosine_similarity(
        source_probability, anchor_probability, dim=1
    )
    posterior_cosine = torch.nn.functional.cosine_similarity(
        source_probability, posterior_probability, dim=1
    )
    return {
        "fullprob_cross_entropy_risk": -(
            safe_source * safe_posterior.log()
        ).sum(dim=1)
        + (safe_source * safe_anchor.log()).sum(dim=1),
        "fullprob_js_risk": jensen_shannon(source_probability, posterior_probability)
        - jensen_shannon(source_probability, anchor_probability),
        "fullprob_cosine_risk": anchor_cosine - posterior_cosine,
        "source_top1_logprob_risk": (
            safe_anchor.gather(1, source_class[:, None]).squeeze(1).log()
            - safe_posterior.gather(1, source_class[:, None]).squeeze(1).log()
        ),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_rows(subset: list[dict[str, Any]]) -> dict[str, Any]:
    if not subset:
        raise ValueError("cannot summarize an empty row subset")
    clean = [row for row in subset if bool(row["clean_correct"])]
    if not clean:
        raise ValueError("cannot summarize a subset without clean-correct rows")
    return {
        "clean_rows": len(clean),
        "accept_rate": sum(bool(row["accepted"]) for row in clean) / len(clean),
        "dc_delta": mean(subset, "dc_after") - mean(subset, "dc_before"),
        "posterior_minus_raw_psnr": mean(subset, "posterior_psnr")
        - mean(subset, "raw_psnr"),
        "posterior_minus_raw_lpips": mean(subset, "posterior_lpips")
        - mean(subset, "raw_lpips"),
        "final_minus_raw_psnr": mean(subset, "final_psnr") - mean(subset, "raw_psnr"),
        "final_minus_raw_lpips": mean(subset, "final_lpips")
        - mean(subset, "raw_lpips"),
        "raw_failure": sum(not bool(row["raw_correct"]) for row in clean),
        "posterior_failure": sum(not bool(row["posterior_correct"]) for row in clean),
        "final_failure": sum(not bool(row["final_correct"]) for row in clean),
        "raw_new": sum(
            bool(row["anchor_correct"]) and not bool(row["raw_correct"]) for row in clean
        ),
        "posterior_new": sum(
            bool(row["anchor_correct"]) and not bool(row["posterior_correct"])
            for row in clean
        ),
        "final_new": sum(
            bool(row["anchor_correct"]) and not bool(row["final_correct"])
            for row in clean
        ),
        "raw_repair": sum(
            not bool(row["anchor_correct"]) and bool(row["raw_correct"]) for row in clean
        ),
        "posterior_repair": sum(
            not bool(row["anchor_correct"]) and bool(row["posterior_correct"])
            for row in clean
        ),
        "final_repair": sum(
            not bool(row["anchor_correct"]) and bool(row["final_correct"])
            for row in clean
        ),
    }


def clopper_pearson_upper_95(successes: int, trials: int) -> float:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError(f"invalid binomial counts: successes={successes}, trials={trials}")
    if successes == trials:
        return 1.0
    from scipy.stats import beta

    return float(beta.ppf(0.95, successes + 1, trials - successes))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pc_imagenette_supervised_audit.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    if str(config["imagenette"]["required_split"]) == "official_val":
        raise RuntimeError(
            "standalone posterior audit cannot consume official_val; use the locked paired final runner"
        )
    samples, classes = load_imagenette_samples(config)
    official_val_accessed = config["imagenette"]["required_split"] == "official_val"
    channel_seeds = [int(item) for item in config.get("channel_seeds", [config["seed"]])]
    if not channel_seeds or len(channel_seeds) != len(set(channel_seeds)):
        raise ValueError("channel_seeds must be non-empty and unique")
    risk_config = dict(config.get("risk_features", {}))
    risk_features_enabled = bool(risk_config.get("enabled", False))
    if risk_features_enabled:
        if str(risk_config.get("feature_version")) != RECEIVER_RISK_FEATURE_VERSION:
            raise ValueError(
                "risk_features.feature_version must be "
                f"{RECEIVER_RISK_FEATURE_VERSION!r}"
            )
        forbidden = [
            name
            for name in RECEIVER_RISK_FEATURE_COLUMNS
            if name.startswith("teacher_")
            or "label" in name
            or "class_idx" in name
            or name.startswith("original_")
        ]
        if forbidden:
            raise RuntimeError(f"source-derived receiver features are forbidden: {forbidden}")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "split": config["imagenette"]["required_split"],
                    "images": len(samples),
                    "classes": classes,
                    "snrs": config["snrs"],
                    "channel_seeds": channel_seeds,
                    "risk_features_enabled": risk_features_enabled,
                    "risk_feature_version": (
                        RECEIVER_RISK_FEATURE_VERSION if risk_features_enabled else None
                    ),
                    "receiver_risk_feature_count": (
                        len(RECEIVER_RISK_FEATURE_COLUMNS) if risk_features_enabled else 0
                    ),
                    "receiver_risk_features": (
                        list(RECEIVER_RISK_FEATURE_COLUMNS) if risk_features_enabled else []
                    ),
                    "official_val_accessed": official_val_accessed,
                },
                indent=2,
            )
        )
        return
    output = resolve(config["output_dir"])
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
        torch.load(resolve(config["diffusion_checkpoint"]), map_location=device)["model_state_dict"]
    )
    diffusion.eval().requires_grad_(False)
    controller_config = config.get("controller", {"type": "imagenet_consensus"})
    controller_type = str(controller_config["type"])
    controllers = {}
    scratch_gate = None
    scratch_gate_temperature = None
    sender_controller = None
    sender_controller_temperature = None
    if controller_type == "imagenet_consensus":
        classifier_config = {"classifiers": config["controller_classifiers"]}
        controllers = {
            item["key"]: load_classifier(item, classifier_config, device)
            for item in config["controller_classifiers"]["models"]
        }
    elif controller_type == "imagenette_scratch_top1_fallback":
        scratch_gate, scratch_gate_temperature = load_scratch_classifier(
            str(controller_config["checkpoint"]),
            classes,
            device,
            str(controller_config.get("expected_role", "G_gate")),
        )
    elif controller_type == "imagenette_sender_scratch_fullprob_zero_veto":
        if int(controller_config.get("probability_vector_raw_bits", -1)) != 80:
            raise RuntimeError("sender full-probability diagnostic requires exactly 80 raw bits")
        if str(controller_config.get("probability_quantization")) != (
            "per_class_uint8_then_renormalize"
        ):
            raise RuntimeError("unsupported sender probability quantization")
        if str(controller_config.get("description_channel_model")) != (
            "noiseless_feasibility_only"
        ):
            raise RuntimeError("sender description is only authorized as a noiseless feasibility")
        if str(controller_config.get("score")) != "fullprob_js_risk":
            raise RuntimeError("sender controller score must be frozen to fullprob_js_risk")
        if float(controller_config.get("threshold")) != 0.0:
            raise RuntimeError("sender controller threshold must be the natural zero threshold")
        sender_controller, sender_controller_temperature = load_scratch_classifier(
            str(controller_config["checkpoint"]),
            classes,
            device,
            str(controller_config.get("expected_role", "G_aux")),
        )
    else:
        raise ValueError(f"unsupported controller type: {controller_type!r}")
    risk_auxiliary = None
    risk_auxiliary_temperature = None
    risk_auxiliary_checkpoint = None
    if risk_features_enabled:
        if scratch_gate is None or controller_type != "imagenette_scratch_top1_fallback":
            raise RuntimeError(
                "receiver_risk_v1 requires the frozen scratch G_gate controller model"
            )
        if str(controller_config.get("expected_role", "G_gate")) != "G_gate":
            raise RuntimeError("receiver_risk_v1 requires expected_role=G_gate")
        risk_auxiliary_checkpoint = resolve(str(risk_config["auxiliary_checkpoint"]))
        gate_checkpoint = resolve(str(controller_config["checkpoint"]))
        evaluator_checkpoint = resolve(str(config["imagenette"]["evaluator_checkpoint"]))
        distinct_checkpoints = {
            gate_checkpoint.resolve(),
            risk_auxiliary_checkpoint.resolve(),
            evaluator_checkpoint.resolve(),
        }
        if len(distinct_checkpoints) != 3:
            raise RuntimeError("G_gate, G_aux, and T_cls checkpoints must be distinct")
        risk_auxiliary, risk_auxiliary_temperature = load_scratch_classifier(
            str(risk_auxiliary_checkpoint),
            classes,
            device,
            str(risk_config.get("expected_role", "G_aux")),
        )
    evaluator, temperature = load_scratch_classifier(
        str(config["imagenette"]["evaluator_checkpoint"]), classes, device, "T_cls"
    )
    lpips_model, lpips_error = try_load_lpips(device, resolve("outputs/cache"))
    if lpips_model is None:
        raise RuntimeError(lpips_error)
    image_transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()]
    )
    rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    for channel_seed in channel_seeds:
        for snr in map(float, config["snrs"]):
            jscc.change_channel("AWGN", snr)
            for start in range(0, len(samples), int(config["batch_size"])):
                batch = samples[start : start + int(config["batch_size"])]
                target = torch.stack(
                    [image_transform(Image.open(item["path"]).convert("RGB")) for item in batch]
                ).to(device)
                labels = torch.tensor([int(item["class_idx"]) for item in batch], device=device)
                torch.manual_seed(derived_seed(channel_seed, snr, start))
                b0, _, received = deepjscc_forward_with_latents(jscc, target)
                snr_tensor = torch.full((len(batch),), snr, device=device)
                snr_norm = snr_tensor / 20.0
                with torch.no_grad():
                    anchor = b1(b0, snr_norm, gate_tensor(b1_config, snr_tensor, device))
                    raw = diffusion(
                        anchor,
                        snr_norm,
                        gate_tensor(diffusion_config, snr_tensor, device),
                    )
                posterior = posterior_correct(
                    jscc,
                    raw,
                    received,
                    int(config["proximal_steps"]),
                    float(config["normalized_step_size"]),
                )
                accepted = torch.ones(len(batch), dtype=torch.bool, device=device)
                gate_anchor_probabilities = None
                gate_raw_probabilities = None
                gate_posterior_probabilities = None
                sender_source_probabilities = None
                sender_anchor_probabilities = None
                sender_posterior_probabilities = None
                sender_description_codes = None
                sender_description_probabilities = None
                sender_semantic_scores = None
                if scratch_gate is not None:
                    assert scratch_gate_temperature is not None
                    gate_anchor_probabilities = evaluate_probabilities(
                        scratch_gate, scratch_gate_temperature, anchor, config
                    )
                    gate_posterior_probabilities = evaluate_probabilities(
                        scratch_gate, scratch_gate_temperature, posterior, config
                    )
                    controller_anchor_pred = gate_anchor_probabilities.argmax(dim=1)
                    controller_post_pred = gate_posterior_probabilities.argmax(dim=1)
                    accepted = controller_post_pred == controller_anchor_pred
                    if risk_features_enabled:
                        gate_raw_probabilities = evaluate_probabilities(
                            scratch_gate, scratch_gate_temperature, raw, config
                        )
                elif sender_controller is not None:
                    assert sender_controller_temperature is not None
                    sender_source_probabilities = evaluate_probabilities(
                        sender_controller,
                        sender_controller_temperature,
                        target,
                        config,
                    )
                    sender_anchor_probabilities = evaluate_probabilities(
                        sender_controller,
                        sender_controller_temperature,
                        anchor,
                        config,
                    )
                    sender_posterior_probabilities = evaluate_probabilities(
                        sender_controller,
                        sender_controller_temperature,
                        posterior,
                        config,
                    )
                    sender_description_codes, sender_description_probabilities = (
                        quantize_description(sender_source_probabilities)
                    )
                    sender_semantic_scores = source_semantic_score_tensors(
                        sender_description_probabilities,
                        sender_anchor_probabilities,
                        sender_posterior_probabilities,
                    )
                    controller_anchor_pred = sender_anchor_probabilities.argmax(dim=1)
                    controller_post_pred = sender_posterior_probabilities.argmax(dim=1)
                    accepted = sender_semantic_scores["fullprob_js_risk"] <= float(
                        controller_config["threshold"]
                    )
                else:
                    controller_anchor_pred = torch.full(
                        (len(batch),), -1, dtype=torch.long, device=device
                    )
                    controller_post_pred = controller_anchor_pred.clone()
                    for classifier, preprocess in controllers.values():
                        accepted &= classify(classifier, preprocess, posterior) == classify(
                            classifier, preprocess, anchor
                        )
                final = torch.where(accepted[:, None, None, None], posterior, anchor)
                with torch.no_grad():
                    original_probabilities = evaluate_probabilities(
                        evaluator, temperature, target, config
                    )
                    anchor_probabilities = evaluate_probabilities(
                        evaluator, temperature, anchor, config
                    )
                    raw_probabilities = evaluate_probabilities(
                        evaluator, temperature, raw, config
                    )
                    post_probabilities = evaluate_probabilities(
                        evaluator, temperature, posterior, config
                    )
                    final_probabilities = evaluate_probabilities(
                        evaluator, temperature, final, config
                    )
                    original_conf, original_pred = original_probabilities.max(dim=1)
                    anchor_pred = anchor_probabilities.argmax(dim=1)
                    raw_pred = raw_probabilities.argmax(dim=1)
                    post_pred = post_probabilities.argmax(dim=1)
                    final_pred = final_probabilities.argmax(dim=1)
                    anchor_psnr = psnr_per_sample(anchor, target)
                    raw_psnr = psnr_per_sample(raw, target)
                    post_psnr = psnr_per_sample(posterior, target)
                    final_psnr = psnr_per_sample(final, target)
                    anchor_lpips = lpips_model(anchor * 2 - 1, target * 2 - 1).flatten()
                    raw_lpips = lpips_model(raw * 2 - 1, target * 2 - 1).flatten()
                    post_lpips = lpips_model(posterior * 2 - 1, target * 2 - 1).flatten()
                    final_lpips = lpips_model(final * 2 - 1, target * 2 - 1).flatten()
                    dc_before = received_latent_consistency_per_sample(jscc, raw, received)
                    dc_after = received_latent_consistency_per_sample(jscc, posterior, received)
                    risk_feature_tensors = None
                    if risk_features_enabled:
                        assert gate_anchor_probabilities is not None
                        assert gate_raw_probabilities is not None
                        assert gate_posterior_probabilities is not None
                        assert risk_auxiliary is not None
                        assert risk_auxiliary_temperature is not None
                        aux_anchor_probabilities = evaluate_probabilities(
                            risk_auxiliary,
                            risk_auxiliary_temperature,
                            anchor,
                            config,
                        )
                        aux_raw_probabilities = evaluate_probabilities(
                            risk_auxiliary,
                            risk_auxiliary_temperature,
                            raw,
                            config,
                        )
                        aux_posterior_probabilities = evaluate_probabilities(
                            risk_auxiliary,
                            risk_auxiliary_temperature,
                            posterior,
                            config,
                        )
                        risk_feature_tensors = receiver_risk_feature_tensors(
                            snr_db=snr,
                            dc_before=dc_before,
                            dc_after=dc_after,
                            anchor=anchor,
                            raw=raw,
                            posterior=posterior,
                            gate_anchor_probabilities=gate_anchor_probabilities,
                            gate_raw_probabilities=gate_raw_probabilities,
                            gate_posterior_probabilities=gate_posterior_probabilities,
                            aux_anchor_probabilities=aux_anchor_probabilities,
                            aux_raw_probabilities=aux_raw_probabilities,
                            aux_posterior_probabilities=aux_posterior_probabilities,
                        )
                        teacher_anchor_true_probability = anchor_probabilities.gather(
                            1, labels[:, None]
                        ).squeeze(1)
                        teacher_raw_true_probability = raw_probabilities.gather(
                            1, labels[:, None]
                        ).squeeze(1)
                        teacher_posterior_true_probability = post_probabilities.gather(
                            1, labels[:, None]
                        ).squeeze(1)
                        teacher_anchor_true_margin = true_class_margin(
                            anchor_probabilities, labels
                        )
                        teacher_raw_true_margin = true_class_margin(raw_probabilities, labels)
                        teacher_posterior_true_margin = true_class_margin(
                            post_probabilities, labels
                        )
                if start == 0:
                    count = min(4, len(batch))
                    save_image(
                        torch.cat(
                            [
                                target[:count],
                                anchor[:count],
                                raw[:count],
                                posterior[:count],
                                final[:count],
                            ]
                        ),
                        output
                        / "samples"
                        / f"seed_{channel_seed}_snr_{int(snr):02d}_original_anchor_raw_post_final.png",
                        nrow=count,
                    )
                for index, item in enumerate(batch):
                    clean = bool(original_pred[index] == labels[index]) and float(
                        original_conf[index]
                    ) >= float(config["clean_confidence_threshold"])
                    row = {
                        "channel_seed": channel_seed,
                        "snr_db": snr,
                        "sample_id": item["sample_id"],
                        "wnid": item["wnid"],
                        "class_idx": int(labels[index]),
                        "original_confidence": float(original_conf[index]),
                        "clean_correct": clean,
                        "controller_type": controller_type,
                        "controller_anchor_pred": int(controller_anchor_pred[index]),
                        "controller_posterior_pred": int(controller_post_pred[index]),
                        "accepted": bool(accepted[index]),
                        "anchor_correct": bool(anchor_pred[index] == labels[index]),
                        "raw_correct": bool(raw_pred[index] == labels[index]),
                        "posterior_correct": bool(post_pred[index] == labels[index]),
                        "final_correct": bool(final_pred[index] == labels[index]),
                        "dc_before": float(dc_before[index]),
                        "dc_after": float(dc_after[index]),
                        "anchor_psnr": float(anchor_psnr[index]),
                        "raw_psnr": float(raw_psnr[index]),
                        "posterior_psnr": float(post_psnr[index]),
                        "final_psnr": float(final_psnr[index]),
                        "anchor_lpips": float(anchor_lpips[index]),
                        "raw_lpips": float(raw_lpips[index]),
                        "posterior_lpips": float(post_lpips[index]),
                        "final_lpips": float(final_lpips[index]),
                    }
                    if sender_controller is not None:
                        assert sender_source_probabilities is not None
                        assert sender_description_codes is not None
                        assert sender_description_probabilities is not None
                        assert sender_anchor_probabilities is not None
                        assert sender_posterior_probabilities is not None
                        assert sender_semantic_scores is not None
                        row.update(
                            {
                                "sender_description_role": str(
                                    controller_config.get("expected_role", "G_aux")
                                ),
                                "sender_description_raw_bits": int(
                                    controller_config["probability_vector_raw_bits"]
                                ),
                                "sender_description_channel_model": str(
                                    controller_config["description_channel_model"]
                                ),
                                "sender_source_prediction": int(
                                    sender_source_probabilities[index].argmax()
                                ),
                                "sender_source_confidence": float(
                                    sender_source_probabilities[index].max()
                                ),
                                "sender_description_decoded_prediction": int(
                                    sender_description_probabilities[index].argmax()
                                ),
                                "sender_description_codes_uint8": json.dumps(
                                    sender_description_codes[index].cpu().tolist(),
                                    separators=(",", ":"),
                                ),
                                "sender_description_probability": json.dumps(
                                    sender_description_probabilities[index].cpu().tolist(),
                                    separators=(",", ":"),
                                ),
                                "sender_anchor_prediction": int(
                                    sender_anchor_probabilities[index].argmax()
                                ),
                                "sender_posterior_prediction": int(
                                    sender_posterior_probabilities[index].argmax()
                                ),
                                **{
                                    f"sender_{name}": float(values[index])
                                    for name, values in sender_semantic_scores.items()
                                },
                            }
                        )
                    rows.append(row)
                    if risk_features_enabled:
                        assert risk_feature_tensors is not None
                        risk_row: dict[str, Any] = {
                            "channel_seed": channel_seed,
                            "sample_id": item["sample_id"],
                            "wnid": item["wnid"],
                            "class_idx": int(labels[index]),
                        }
                        risk_row.update(
                            {
                                name: float(risk_feature_tensors[name][index])
                                for name in RECEIVER_RISK_FEATURE_COLUMNS
                            }
                        )
                        risk_row.update(
                            {
                                "teacher_original_confidence": float(original_conf[index]),
                                "teacher_clean_correct": clean,
                                "teacher_anchor_correct": bool(
                                    anchor_pred[index] == labels[index]
                                ),
                                "teacher_raw_correct": bool(raw_pred[index] == labels[index]),
                                "teacher_posterior_correct": bool(
                                    post_pred[index] == labels[index]
                                ),
                                "teacher_new_error": bool(
                                    clean
                                    and anchor_pred[index] == labels[index]
                                    and post_pred[index] != labels[index]
                                ),
                                "teacher_anchor_true_probability": float(
                                    teacher_anchor_true_probability[index]
                                ),
                                "teacher_raw_true_probability": float(
                                    teacher_raw_true_probability[index]
                                ),
                                "teacher_posterior_true_probability": float(
                                    teacher_posterior_true_probability[index]
                                ),
                                "teacher_posterior_minus_anchor_true_probability": float(
                                    teacher_posterior_true_probability[index]
                                    - teacher_anchor_true_probability[index]
                                ),
                                "teacher_anchor_true_margin": float(
                                    teacher_anchor_true_margin[index]
                                ),
                                "teacher_raw_true_margin": float(
                                    teacher_raw_true_margin[index]
                                ),
                                "teacher_posterior_true_margin": float(
                                    teacher_posterior_true_margin[index]
                                ),
                                "teacher_posterior_minus_anchor_true_margin": float(
                                    teacher_posterior_true_margin[index]
                                    - teacher_anchor_true_margin[index]
                                ),
                            }
                        )
                        if tuple(risk_row)[-len(TEACHER_RISK_TARGET_COLUMNS) :] != (
                            TEACHER_RISK_TARGET_COLUMNS
                        ):
                            raise RuntimeError("teacher risk target schema mismatch")
                        risk_rows.append(risk_row)
            print(f"done seed={channel_seed} snr={snr:g}")
    write_csv(output / "per_sample.csv", rows)
    if risk_features_enabled:
        if len(risk_rows) != len(rows):
            raise RuntimeError("risk feature row count differs from audit row count")
        write_csv(output / "risk_features.csv", risk_rows)
    summary: list[dict[str, Any]] = []
    for snr in map(float, config["snrs"]):
        subset = [row for row in rows if float(row["snr_db"]) == snr]
        summary.append({"snr_db": snr, **summarize_rows(subset)})
    write_csv(output / "summary.csv", summary)
    seed_summary: list[dict[str, Any]] = []
    for channel_seed in channel_seeds:
        for snr in map(float, config["snrs"]):
            subset = [
                row
                for row in rows
                if int(row["channel_seed"]) == channel_seed and float(row["snr_db"]) == snr
            ]
            seed_summary.append(
                {"channel_seed": channel_seed, "snr_db": snr, **summarize_rows(subset)}
            )
    write_csv(output / "seed_summary.csv", seed_summary)
    clean_images = len({row["sample_id"] for row in rows if bool(row["clean_correct"])})
    primary = [row for row in summary if float(row["snr_db"]) in config["primary_snrs"]]
    primary_rows = [
        row
        for row in rows
        if bool(row["clean_correct"]) and float(row["snr_db"]) in config["primary_snrs"]
    ]
    eligible_image_ids = {
        str(row["sample_id"]) for row in primary_rows if bool(row["anchor_correct"])
    }
    raw_new_image_ids = {
        str(row["sample_id"])
        for row in primary_rows
        if bool(row["anchor_correct"]) and not bool(row["raw_correct"])
    }
    final_new_image_ids = {
        str(row["sample_id"])
        for row in primary_rows
        if bool(row["anchor_correct"]) and not bool(row["final_correct"])
    }
    final_new_cluster_upper = clopper_pearson_upper_95(
        len(final_new_image_ids), len(eligible_image_ids)
    )
    gates = {
        "minimum_clean_images": clean_images >= int(config["minimum_clean_images"]),
        "dc_all_5_decrease": all(float(row["dc_delta"]) < 0 for row in summary),
        "posterior_psnr_positive_4_of_5": sum(
            float(row["posterior_minus_raw_psnr"]) > 0 for row in summary
        )
        >= 4,
        "posterior_lpips_nonpositive_4_of_5": sum(
            float(row["posterior_minus_raw_lpips"]) <= 0 for row in summary
        )
        >= 4,
        "final_mean_psnr_positive": mean(summary, "final_minus_raw_psnr") > 0,
        "final_mean_lpips_nonpositive": mean(summary, "final_minus_raw_lpips") <= 0,
        "primary_final_new_total_not_increase": sum(int(row["final_new"]) for row in primary)
        <= sum(int(row["raw_new"]) for row in primary),
        "primary_final_new_each_snr_not_increase": all(
            int(row["final_new"]) <= int(row["raw_new"]) for row in primary
        ),
        "primary_final_failure_not_increase": sum(
            int(row["final_failure"]) for row in primary
        )
        <= sum(int(row["raw_failure"]) for row in primary),
    }
    multiseed_gates = config.get("multiseed_gates", {})
    if bool(multiseed_gates.get("enabled", False)):
        primary_by_seed = {
            channel_seed: [
                row
                for row in seed_summary
                if int(row["channel_seed"]) == channel_seed
                and float(row["snr_db"]) in config["primary_snrs"]
            ]
            for channel_seed in channel_seeds
        }
        all_by_seed = {
            channel_seed: [
                row for row in seed_summary if int(row["channel_seed"]) == channel_seed
            ]
            for channel_seed in channel_seeds
        }
        gates.update(
            {
                "dc_all_seed_snr_decrease": all(
                    float(row["dc_delta"]) < 0 for row in seed_summary
                ),
                "final_mean_psnr_positive_each_seed": all(
                    mean(all_by_seed[channel_seed], "final_minus_raw_psnr") > 0
                    for channel_seed in channel_seeds
                ),
                "final_mean_lpips_nonpositive_each_seed": all(
                    mean(all_by_seed[channel_seed], "final_minus_raw_lpips") <= 0
                    for channel_seed in channel_seeds
                ),
                "primary_final_new_each_seed_not_increase": all(
                    sum(int(row["final_new"]) for row in primary_by_seed[channel_seed])
                    <= sum(int(row["raw_new"]) for row in primary_by_seed[channel_seed])
                    for channel_seed in channel_seeds
                ),
                "primary_final_failure_each_seed_not_increase": all(
                    sum(int(row["final_failure"]) for row in primary_by_seed[channel_seed])
                    <= sum(int(row["raw_failure"]) for row in primary_by_seed[channel_seed])
                    for channel_seed in channel_seeds
                ),
                "primary_final_new_image_cluster_upper_within_limit": final_new_cluster_upper
                <= float(multiseed_gates["max_primary_final_new_image_cluster_upper"]),
            }
        )
    aggregate = {
        "policy_dev_images": len(samples),
        "clean_images": clean_images,
        "channel_seeds": channel_seeds,
        "mean_posterior_minus_raw_psnr": mean(summary, "posterior_minus_raw_psnr"),
        "mean_posterior_minus_raw_lpips": mean(summary, "posterior_minus_raw_lpips"),
        "mean_final_minus_raw_psnr": mean(summary, "final_minus_raw_psnr"),
        "mean_final_minus_raw_lpips": mean(summary, "final_minus_raw_lpips"),
        "primary_raw_failure": sum(int(row["raw_failure"]) for row in primary),
        "primary_posterior_failure": sum(int(row["posterior_failure"]) for row in primary),
        "primary_final_failure": sum(int(row["final_failure"]) for row in primary),
        "primary_raw_new": sum(int(row["raw_new"]) for row in primary),
        "primary_posterior_new": sum(int(row["posterior_new"]) for row in primary),
        "primary_final_new": sum(int(row["final_new"]) for row in primary),
        "primary_new_error_eligible_image_clusters": len(eligible_image_ids),
        "primary_raw_new_image_clusters": len(raw_new_image_ids),
        "primary_final_new_image_clusters": len(final_new_image_ids),
        "primary_final_new_image_cluster_clopper_pearson_upper_95": final_new_cluster_upper,
        "official_val_accessed": official_val_accessed,
    }
    if sender_controller is not None:
        aggregate.update(
            {
                "sender_description_role": str(
                    controller_config.get("expected_role", "G_aux")
                ),
                "sender_description_raw_bits": int(
                    controller_config["probability_vector_raw_bits"]
                ),
                "sender_description_channel_model": str(
                    controller_config["description_channel_model"]
                ),
                "sender_fullprob_js_risk_mean": mean(rows, "sender_fullprob_js_risk"),
                "sender_fullprob_js_risk_positive_rate": sum(
                    float(row["sender_fullprob_js_risk"]) > 0 for row in rows
                )
                / len(rows),
                "sender_zero_veto_accept_rate": sum(bool(row["accepted"]) for row in rows)
                / len(rows),
            }
        )
    verdict = "POSITIVE" if all(gates.values()) else "NEGATIVE"
    risk_feature_schema = None
    if risk_features_enabled:
        assert risk_auxiliary_checkpoint is not None
        risk_feature_schema = {
            "feature_version": RECEIVER_RISK_FEATURE_VERSION,
            "row_key_columns": ["channel_seed", "snr_db", "sample_id"],
            "receiver_feature_columns": list(RECEIVER_RISK_FEATURE_COLUMNS),
            "receiver_feature_count": len(RECEIVER_RISK_FEATURE_COLUMNS),
            "teacher_target_columns": list(TEACHER_RISK_TARGET_COLUMNS),
            "teacher_targets_for_development_only": True,
            "teacher_targets_allowed_as_controller_inputs": False,
            "source_image_or_ground_truth_allowed_as_controller_inputs": False,
            "image_cluster_column": "sample_id",
            "row_count": len(risk_rows),
            "G_gate_checkpoint": str(resolve(str(controller_config["checkpoint"]))),
            "G_gate_checkpoint_sha256": sha256_file(
                resolve(str(controller_config["checkpoint"]))
            ),
            "G_aux_checkpoint": str(risk_auxiliary_checkpoint),
            "G_aux_checkpoint_sha256": sha256_file(risk_auxiliary_checkpoint),
            "T_cls_checkpoint": str(resolve(config["imagenette"]["evaluator_checkpoint"])),
            "T_cls_checkpoint_sha256": sha256_file(
                resolve(config["imagenette"]["evaluator_checkpoint"])
            ),
            "official_val_accessed": official_val_accessed,
        }
        (output / "risk_feature_schema.json").write_text(
            json.dumps(risk_feature_schema, indent=2), encoding="utf-8"
        )
    payload = {
        "config": config,
        "aggregate": aggregate,
        "summary": summary,
        "seed_summary": seed_summary,
        "risk_feature_schema": risk_feature_schema,
        "gates": gates,
        "verdict": verdict,
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"aggregate": aggregate, "gates": gates, "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
