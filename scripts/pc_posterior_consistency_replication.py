#!/usr/bin/env python3
"""Frozen independent replication of received-latent posterior correction."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from cadsd_jscc.deepjscc_adapter import (  # noqa: E402
    deepjscc_forward_with_latents,
    load_deepjscc_model,
    received_latent_consistency_loss,
    received_latent_consistency_per_sample,
)
from cadsd_jscc.metrics import psnr_per_sample  # noqa: E402
from s10_short_chain_residual_shift_diffusion import (  # noqa: E402
    ShortChainResidualShiftDiffusion,
)
from s13_export_coco_train2017_c8_scaleup import (  # noqa: E402
    derived_seed,
    discover_images,
    select_paths,
)
from s5_residual_refiner_pilot import (  # noqa: E402
    build_model,
    gate_tensor,
    try_load_lpips,
)
from s5_coco_object_clip_clean_eval import (  # noqa: E402
    encode_text_features,
    load_clip_model,
)


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(resolve(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return payload


def select_rank_block(config: dict[str, Any]) -> list[Path]:
    source = load_yaml(config["source_config"])
    root = resolve(source["inputs"]["source_root"])
    paths = discover_images(root)
    start = int(config["source_rank_start"])
    count = int(config["sample_count"])
    selected = sum(select_paths(paths, root, source["seed"], start, count), [])[start:]
    if len(selected) != count:
        raise RuntimeError(f"rank-block mismatch: expected {count}, got {len(selected)}")
    return selected


def load_classifier(model_config: dict[str, Any], config: dict[str, Any], device: torch.device):
    weights_file = resolve(model_config["weights_file"])
    if not weights_file.is_file() or weights_file.stat().st_size < 1024 * 1024:
        raise RuntimeError(f"local classifier weights missing: {weights_file}")
    os.environ.setdefault("TORCH_HOME", str(resolve(config["classifiers"]["cache_dir"])))
    import torchvision.models as models

    weights_enum = getattr(models, str(model_config["weights_enum"]))
    weights = getattr(weights_enum, str(model_config["weights"]))
    model = getattr(models, str(model_config["model_name"]))(weights=weights)
    return model.to(device).eval().requires_grad_(False), weights.transforms()


@torch.no_grad()
def classify(model: torch.nn.Module, preprocess, images: torch.Tensor) -> torch.Tensor:
    prepared = torch.stack(
        [preprocess(transforms.ToPILImage()(image.cpu())) for image in images]
    ).to(images.device)
    return model(prepared).argmax(dim=1)


def posterior_correct(
    jscc: torch.nn.Module,
    raw: torch.Tensor,
    received: torch.Tensor,
    steps: int,
    step_size: float,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    current = raw.detach()
    for _ in range(int(steps)):
        current.requires_grad_(True)
        loss = received_latent_consistency_loss(
            jscc, current, received, valid_mask=valid_mask
        )
        gradient = torch.autograd.grad(loss, current)[0]
        rms = gradient.square().flatten(start_dim=1).mean(dim=1).sqrt().clamp_min(1e-12)
        current = (
            current - float(step_size) * gradient / rms[:, None, None, None]
        ).clamp(0.0, 1.0).detach()
    return current


def mean(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dominant_coco_labels(
    audit_config: dict[str, Any], paths: list[Path]
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    payload = json.loads(resolve(audit_config["instances_file"]).read_text(encoding="utf-8"))
    categories = {int(item["id"]): str(item["name"]) for item in payload["categories"]}
    category_names = [str(item["name"]) for item in payload["categories"]]
    selected_ids = {int(path.stem) for path in paths}
    images = {
        int(item["id"]): item for item in payload["images"] if int(item["id"]) in selected_ids
    }
    areas: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[int, int] = defaultdict(int)
    for annotation in payload["annotations"]:
        image_id = int(annotation["image_id"])
        if image_id not in selected_ids:
            continue
        if bool(audit_config.get("ignore_crowd", True)) and int(annotation.get("iscrowd", 0)):
            continue
        areas[image_id][int(annotation["category_id"])] += float(annotation.get("area", 0.0))
        counts[image_id] += 1
    output: dict[str, dict[str, Any]] = {}
    for path in paths:
        image_id = int(path.stem)
        image = images.get(image_id)
        category_areas = areas.get(image_id, {})
        total_area = sum(category_areas.values())
        if image is None or not category_areas or total_area <= 0:
            output[str(path)] = {"usable": False}
            continue
        category_id, dominant_area = max(category_areas.items(), key=lambda item: item[1])
        image_area = float(image["width"]) * float(image["height"])
        share = dominant_area / total_area
        area_ratio = dominant_area / image_area
        output[str(path)] = {
            "usable": share >= float(audit_config["dominant_category_min_share"])
            and area_ratio >= float(audit_config["dominant_category_min_area_ratio"]),
            "image_id": image_id,
            "label": categories[category_id],
            "share": share,
            "area_ratio": area_ratio,
            "annotation_count": counts[image_id],
        }
    return output, category_names


@torch.no_grad()
def clip_classify(
    model: torch.nn.Module,
    preprocess,
    text_features: torch.Tensor,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prepared = torch.stack(
        [preprocess(transforms.ToPILImage()(image.cpu())) for image in images]
    ).to(images.device)
    image_features = F.normalize(model.encode_image(prepared).float(), dim=-1)
    logits = model.logit_scale.exp().float().clamp(max=100.0) * image_features @ text_features.T
    probabilities = torch.softmax(logits, dim=-1)
    values, indices = torch.topk(probabilities, k=2, dim=-1)
    return indices[:, 0], values[:, 0], values[:, 0] - values[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/pc002_posterior_consistency_independent_replication.yaml"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    paths = select_rank_block(config)
    if args.dry_run:
        coco_payload: dict[str, Any] = {}
        if isinstance(config.get("coco_object_clip_audit"), dict):
            labels_by_path, _ = dominant_coco_labels(config["coco_object_clip_audit"], paths)
            coco_payload = {
                "dominant_label_usable": sum(
                    bool(item.get("usable")) for item in labels_by_path.values()
                )
            }
        print(
            json.dumps(
                {
                    "count": len(paths),
                    "rank_range": [
                        config["source_rank_start"],
                        config["source_rank_start"] + config["sample_count"] - 1,
                    ],
                    "first": str(paths[0]),
                    "last": str(paths[-1]),
                    "snrs": config["snrs"],
                    "classifiers": [item["key"] for item in config["classifiers"]["models"]],
                    "coco_object_clip_audit": coco_payload,
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
    transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()]
    )
    source_config = load_yaml(config["source_config"])
    jscc = load_deepjscc_model(
        resolve(source_config["baseline"]["repo"]),
        resolve(config["deepjscc_checkpoint"]),
        int(source_config["rate"]["inner_channel"]),
        str(source_config["channel"]["type"]),
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
    classifiers = {
        item["key"]: load_classifier(item, config, device)
        for item in config["classifiers"]["models"]
    }
    lpips_model, lpips_error = try_load_lpips(device, resolve("outputs/cache"))
    if lpips_model is None:
        raise RuntimeError(f"LPIPS is required by the frozen protocol: {lpips_error}")
    coco_audit = config.get("coco_object_clip_audit")
    labels_by_path: dict[str, dict[str, Any]] = {}
    clip_labels: list[str] = []
    clip_model = clip_preprocess = clip_text_features = None
    if isinstance(coco_audit, dict):
        labels_by_path, clip_labels = dominant_coco_labels(coco_audit, paths)
        clip_config = {"clip": coco_audit["clip"]}
        clip_model, clip_preprocess, tokenizer = load_clip_model(clip_config, device)
        clip_text_features = encode_text_features(
            clip_model,
            tokenizer,
            clip_labels,
            [str(template) for template in coco_audit["prompt_templates"]],
            device,
        )

    rows: list[dict[str, Any]] = []
    for snr in map(float, config["snrs"]):
        jscc.change_channel(str(source_config["channel"]["type"]), snr)
        for start in range(0, len(paths), int(config["batch_size"])):
            batch_paths = paths[start : start + int(config["batch_size"])]
            target = torch.stack(
                [transform(Image.open(path).convert("RGB")) for path in batch_paths]
            ).to(device)
            torch.manual_seed(derived_seed(int(config["seed"]), snr, start))
            b0, _, received = deepjscc_forward_with_latents(jscc, target)
            snr_tensor = torch.full((len(target),), snr, device=device)
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
            with torch.no_grad():
                dc_before = received_latent_consistency_per_sample(jscc, raw, received)
                dc_after = received_latent_consistency_per_sample(jscc, posterior, received)
                raw_psnr = psnr_per_sample(raw, target)
                posterior_psnr = psnr_per_sample(posterior, target)
                raw_lpips = lpips_model(raw * 2.0 - 1.0, target * 2.0 - 1.0).flatten()
                posterior_lpips = lpips_model(
                    posterior * 2.0 - 1.0, target * 2.0 - 1.0
                ).flatten()
            predictions: dict[str, dict[str, torch.Tensor]] = {}
            for key, (classifier, preprocess) in classifiers.items():
                predictions[key] = {
                    "original": classify(classifier, preprocess, target),
                    "anchor": classify(classifier, preprocess, anchor),
                    "raw": classify(classifier, preprocess, raw),
                    "posterior": classify(classifier, preprocess, posterior),
                }
            final = None
            accepted = None
            final_psnr = None
            final_lpips = None
            failure_handling = config.get("failure_handling")
            if isinstance(failure_handling, dict):
                controller_keys = failure_handling.get("classifiers")
                if controller_keys is None:
                    controller_keys = [failure_handling["classifier"]]
                controller_keys = [str(key) for key in controller_keys]
                accepted = torch.ones(len(target), dtype=torch.bool, device=device)
                for controller_key in controller_keys:
                    controller = predictions[controller_key]
                    accepted &= controller["posterior"] == controller["anchor"]
                fallback_name = str(failure_handling["fallback"])
                if fallback_name == "anchor":
                    fallback = anchor
                elif fallback_name == "raw":
                    fallback = raw
                else:
                    raise ValueError(f"unsupported fallback: {fallback_name}")
                final = torch.where(accepted[:, None, None, None], posterior, fallback)
                with torch.no_grad():
                    final_psnr = psnr_per_sample(final, target)
                    final_lpips = lpips_model(
                        final * 2.0 - 1.0, target * 2.0 - 1.0
                    ).flatten()
                for key, (classifier, preprocess) in classifiers.items():
                    predictions[key]["final"] = classify(classifier, preprocess, final)
            clip_predictions: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
            if clip_model is not None and clip_preprocess is not None and clip_text_features is not None:
                clip_predictions = {
                    "original": clip_classify(
                        clip_model, clip_preprocess, clip_text_features, target
                    ),
                    "anchor": clip_classify(
                        clip_model, clip_preprocess, clip_text_features, anchor
                    ),
                    "raw": clip_classify(clip_model, clip_preprocess, clip_text_features, raw),
                    "posterior": clip_classify(
                        clip_model, clip_preprocess, clip_text_features, posterior
                    ),
                    "final": clip_classify(
                        clip_model,
                        clip_preprocess,
                        clip_text_features,
                        final if final is not None else posterior,
                    ),
                }
            if start == 0:
                count = min(4, len(target))
                sample_groups = [target[:count], anchor[:count], raw[:count], posterior[:count]]
                if final is not None:
                    sample_groups.append(final[:count])
                save_image(
                    torch.cat(sample_groups, dim=0),
                    output / "samples" / f"snr_{int(snr):02d}_original_anchor_raw_posterior.png",
                    nrow=count,
                )
            for index, path in enumerate(batch_paths):
                row: dict[str, Any] = {
                    "snr_db": snr,
                    "source": str(path.relative_to(ROOT)),
                    "dc_before": float(dc_before[index]),
                    "dc_after": float(dc_after[index]),
                    "raw_psnr": float(raw_psnr[index]),
                    "posterior_psnr": float(posterior_psnr[index]),
                    "raw_lpips": float(raw_lpips[index]),
                    "posterior_lpips": float(posterior_lpips[index]),
                }
                if final is not None and final_psnr is not None and final_lpips is not None:
                    row.update(
                        {
                            "accepted": bool(accepted[index]),
                            "final_psnr": float(final_psnr[index]),
                            "final_lpips": float(final_lpips[index]),
                        }
                    )
                raw_new_votes = post_new_votes = raw_repair_votes = post_repair_votes = 0
                final_new_votes = final_repair_votes = 0
                for key, prediction in predictions.items():
                    original = int(prediction["original"][index])
                    anchor_correct = int(prediction["anchor"][index]) == original
                    raw_correct = int(prediction["raw"][index]) == original
                    post_correct = int(prediction["posterior"][index]) == original
                    row[f"{key}_anchor_correct"] = anchor_correct
                    row[f"{key}_raw_correct"] = raw_correct
                    row[f"{key}_posterior_correct"] = post_correct
                    raw_new_votes += int(anchor_correct and not raw_correct)
                    post_new_votes += int(anchor_correct and not post_correct)
                    raw_repair_votes += int(not anchor_correct and raw_correct)
                    post_repair_votes += int(not anchor_correct and post_correct)
                    if final is not None:
                        final_correct = int(prediction["final"][index]) == original
                        row[f"{key}_final_correct"] = final_correct
                        final_new_votes += int(anchor_correct and not final_correct)
                        final_repair_votes += int(not anchor_correct and final_correct)
                row.update(
                    {
                        "raw_new_votes": raw_new_votes,
                        "posterior_new_votes": post_new_votes,
                        "raw_repair_votes": raw_repair_votes,
                        "posterior_repair_votes": post_repair_votes,
                        "raw_majority_new": raw_new_votes >= 2,
                        "posterior_majority_new": post_new_votes >= 2,
                        "raw_majority_repair": raw_repair_votes >= 2,
                        "posterior_majority_repair": post_repair_votes >= 2,
                    }
                )
                if final is not None:
                    row.update(
                        {
                            "final_new_votes": final_new_votes,
                            "final_repair_votes": final_repair_votes,
                            "final_majority_new": final_new_votes >= 2,
                            "final_majority_repair": final_repair_votes >= 2,
                        }
                    )
                if clip_predictions:
                    label_meta = labels_by_path[str(path)]
                    label_to_index = {label: position for position, label in enumerate(clip_labels)}
                    dominant_index = label_to_index.get(str(label_meta.get("label", "")), -1)
                    original_index = int(clip_predictions["original"][0][index])
                    clean_correct = (
                        bool(label_meta.get("usable"))
                        and original_index == dominant_index
                        and float(clip_predictions["original"][1][index])
                        >= float(coco_audit["clean_correct_min_prob"])
                        and float(clip_predictions["original"][2][index])
                        >= float(coco_audit["clean_correct_min_margin"])
                    )
                    row.update(
                        {
                            "coco_dominant_usable": bool(label_meta.get("usable")),
                            "coco_dominant_label": label_meta.get("label", ""),
                            "coco_dominant_share": label_meta.get("share", ""),
                            "coco_dominant_area_ratio": label_meta.get("area_ratio", ""),
                            "clip_original_clean_correct": clean_correct,
                            "clip_original_top1_prob": float(
                                clip_predictions["original"][1][index]
                            ),
                            "clip_original_top1_margin": float(
                                clip_predictions["original"][2][index]
                            ),
                            "clip_anchor_gt_correct": int(
                                clip_predictions["anchor"][0][index]
                            )
                            == dominant_index,
                            "clip_raw_gt_correct": int(clip_predictions["raw"][0][index])
                            == dominant_index,
                            "clip_posterior_gt_correct": int(
                                clip_predictions["posterior"][0][index]
                            )
                            == dominant_index,
                            "clip_final_gt_correct": int(clip_predictions["final"][0][index])
                            == dominant_index,
                        }
                    )
                rows.append(row)
        print(f"done snr={snr:g}")

    write_csv(output / "per_sample.csv", rows)
    summary: list[dict[str, Any]] = []
    for snr in map(float, config["snrs"]):
        subset = [row for row in rows if float(row["snr_db"]) == snr]
        item: dict[str, Any] = {
            "snr_db": snr,
            "dc_delta": mean(subset, "dc_after") - mean(subset, "dc_before"),
            "posterior_minus_raw_psnr": mean(subset, "posterior_psnr")
            - mean(subset, "raw_psnr"),
            "posterior_minus_raw_lpips": mean(subset, "posterior_lpips")
            - mean(subset, "raw_lpips"),
            "raw_majority_new": sum(bool(row["raw_majority_new"]) for row in subset),
            "posterior_majority_new": sum(
                bool(row["posterior_majority_new"]) for row in subset
            ),
            "raw_majority_repair": sum(bool(row["raw_majority_repair"]) for row in subset),
            "posterior_majority_repair": sum(
                bool(row["posterior_majority_repair"]) for row in subset
            ),
        }
        for key in classifiers:
            item[f"{key}_raw_new"] = sum(
                bool(row[f"{key}_anchor_correct"]) and not bool(row[f"{key}_raw_correct"])
                for row in subset
            )
            item[f"{key}_posterior_new"] = sum(
                bool(row[f"{key}_anchor_correct"])
                and not bool(row[f"{key}_posterior_correct"])
                for row in subset
            )
            item[f"{key}_raw_repair"] = sum(
                not bool(row[f"{key}_anchor_correct"]) and bool(row[f"{key}_raw_correct"])
                for row in subset
            )
            item[f"{key}_posterior_repair"] = sum(
                not bool(row[f"{key}_anchor_correct"])
                and bool(row[f"{key}_posterior_correct"])
                for row in subset
            )
        if isinstance(config.get("failure_handling"), dict):
            item.update(
                {
                    "accept_count": sum(bool(row["accepted"]) for row in subset),
                    "final_minus_raw_psnr": mean(subset, "final_psnr")
                    - mean(subset, "raw_psnr"),
                    "final_minus_raw_lpips": mean(subset, "final_lpips")
                    - mean(subset, "raw_lpips"),
                    "final_majority_new": sum(
                        bool(row["final_majority_new"]) for row in subset
                    ),
                    "final_majority_repair": sum(
                        bool(row["final_majority_repair"]) for row in subset
                    ),
                }
            )
            for key in classifiers:
                item[f"{key}_final_new"] = sum(
                    bool(row[f"{key}_anchor_correct"])
                    and not bool(row[f"{key}_final_correct"])
                    for row in subset
                )
                item[f"{key}_final_repair"] = sum(
                    not bool(row[f"{key}_anchor_correct"])
                    and bool(row[f"{key}_final_correct"])
                    for row in subset
                )
        if isinstance(coco_audit, dict):
            clean = [row for row in subset if bool(row["clip_original_clean_correct"])]
            item.update(
                {
                    "clip_gt_clean_rows": len(clean),
                    "clip_gt_raw_failure": sum(
                        not bool(row["clip_raw_gt_correct"]) for row in clean
                    ),
                    "clip_gt_posterior_failure": sum(
                        not bool(row["clip_posterior_gt_correct"]) for row in clean
                    ),
                    "clip_gt_final_failure": sum(
                        not bool(row["clip_final_gt_correct"]) for row in clean
                    ),
                    "clip_gt_raw_new": sum(
                        bool(row["clip_anchor_gt_correct"])
                        and not bool(row["clip_raw_gt_correct"])
                        for row in clean
                    ),
                    "clip_gt_posterior_new": sum(
                        bool(row["clip_anchor_gt_correct"])
                        and not bool(row["clip_posterior_gt_correct"])
                        for row in clean
                    ),
                    "clip_gt_final_new": sum(
                        bool(row["clip_anchor_gt_correct"])
                        and not bool(row["clip_final_gt_correct"])
                        for row in clean
                    ),
                    "clip_gt_raw_repair": sum(
                        not bool(row["clip_anchor_gt_correct"])
                        and bool(row["clip_raw_gt_correct"])
                        for row in clean
                    ),
                    "clip_gt_posterior_repair": sum(
                        not bool(row["clip_anchor_gt_correct"])
                        and bool(row["clip_posterior_gt_correct"])
                        for row in clean
                    ),
                    "clip_gt_final_repair": sum(
                        not bool(row["clip_anchor_gt_correct"])
                        and bool(row["clip_final_gt_correct"])
                        for row in clean
                    ),
                }
            )
        summary.append(item)
    write_csv(output / "summary.csv", summary)

    psnr_positive = sum(float(item["posterior_minus_raw_psnr"]) > 0 for item in summary)
    lpips_improved = sum(float(item["posterior_minus_raw_lpips"]) <= 0 for item in summary)
    semantic_method = "final" if isinstance(config.get("failure_handling"), dict) else "posterior"
    model_nonincrease = sum(
        sum(int(item[f"{key}_{semantic_method}_new"]) for item in summary)
        <= sum(int(item[f"{key}_raw_new"]) for item in summary)
        for key in classifiers
    )
    gates = {
        "dc_all_5_decrease": all(float(item["dc_delta"]) < 0 for item in summary),
        "mean_psnr_positive_and_4_of_5": mean(summary, "posterior_minus_raw_psnr") > 0
        and psnr_positive >= 4,
        "mean_lpips_nonpositive_and_4_of_5": mean(
            summary, "posterior_minus_raw_lpips"
        )
        <= 0
        and lpips_improved >= 4,
        "majority_new_not_increase": sum(
            int(item[f"{semantic_method}_majority_new"]) for item in summary
        )
        <= sum(int(item["raw_majority_new"]) for item in summary),
        "at_least_2_models_new_not_increase": model_nonincrease >= 2,
    }
    if semantic_method == "final":
        gates["final_mean_psnr_positive"] = mean(summary, "final_minus_raw_psnr") > 0
        gates["final_mean_lpips_nonpositive"] = mean(summary, "final_minus_raw_lpips") <= 0
        semantic_audit = config.get("semantic_audit")
        if isinstance(semantic_audit, dict) and semantic_audit.get("holdout_classifier"):
            holdout = str(semantic_audit["holdout_classifier"])
            gates["holdout_classifier_new_not_increase"] = sum(
                int(item[f"{holdout}_final_new"]) for item in summary
            ) <= sum(int(item[f"{holdout}_raw_new"]) for item in summary)
    if isinstance(coco_audit, dict):
        clean_sources = {
            row["source"] for row in rows if bool(row["clip_original_clean_correct"])
        }
        gates["minimum_coco_clip_clean_images"] = len(clean_sources) >= int(
            coco_audit["minimum_clean_images"]
        )
        gates["clip_gt_final_new_not_increase"] = sum(
            int(item["clip_gt_final_new"]) for item in summary
        ) <= sum(int(item["clip_gt_raw_new"]) for item in summary)
        gates["clip_gt_final_failure_not_increase"] = sum(
            int(item["clip_gt_final_failure"]) for item in summary
        ) <= sum(int(item["clip_gt_raw_failure"]) for item in summary)
    aggregate = {
        "mean_dc_before": mean(rows, "dc_before"),
        "mean_dc_after": mean(rows, "dc_after"),
        "mean_posterior_minus_raw_psnr": mean(summary, "posterior_minus_raw_psnr"),
        "mean_posterior_minus_raw_lpips": mean(summary, "posterior_minus_raw_lpips"),
        "raw_majority_new": sum(int(item["raw_majority_new"]) for item in summary),
        "posterior_majority_new": sum(
            int(item["posterior_majority_new"]) for item in summary
        ),
        "raw_majority_repair": sum(int(item["raw_majority_repair"]) for item in summary),
        "posterior_majority_repair": sum(
            int(item["posterior_majority_repair"]) for item in summary
        ),
    }
    if semantic_method == "final":
        aggregate.update(
            {
                "accept_rate": sum(int(item["accept_count"]) for item in summary) / len(rows),
                "mean_final_minus_raw_psnr": mean(summary, "final_minus_raw_psnr"),
                "mean_final_minus_raw_lpips": mean(summary, "final_minus_raw_lpips"),
                "final_majority_new": sum(
                    int(item["final_majority_new"]) for item in summary
                ),
                "final_majority_repair": sum(
                    int(item["final_majority_repair"]) for item in summary
                ),
            }
        )
    if isinstance(coco_audit, dict):
        aggregate.update(
            {
                "coco_clip_clean_images": len(
                    {row["source"] for row in rows if bool(row["clip_original_clean_correct"])}
                ),
                "clip_gt_raw_failure": sum(
                    int(item["clip_gt_raw_failure"]) for item in summary
                ),
                "clip_gt_posterior_failure": sum(
                    int(item["clip_gt_posterior_failure"]) for item in summary
                ),
                "clip_gt_final_failure": sum(
                    int(item["clip_gt_final_failure"]) for item in summary
                ),
                "clip_gt_raw_new": sum(int(item["clip_gt_raw_new"]) for item in summary),
                "clip_gt_posterior_new": sum(
                    int(item["clip_gt_posterior_new"]) for item in summary
                ),
                "clip_gt_final_new": sum(
                    int(item["clip_gt_final_new"]) for item in summary
                ),
                "clip_gt_raw_repair": sum(
                    int(item["clip_gt_raw_repair"]) for item in summary
                ),
                "clip_gt_posterior_repair": sum(
                    int(item["clip_gt_posterior_repair"]) for item in summary
                ),
                "clip_gt_final_repair": sum(
                    int(item["clip_gt_final_repair"]) for item in summary
                ),
            }
        )
    verdict = "POSITIVE" if all(gates.values()) else "NEGATIVE"
    payload = {
        "config": config,
        "aggregate": aggregate,
        "summary": summary,
        "gates": gates,
        "verdict": verdict,
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"aggregate": aggregate, "gates": gates, "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
