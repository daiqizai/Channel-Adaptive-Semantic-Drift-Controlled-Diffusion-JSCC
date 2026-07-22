#!/usr/bin/env python3
"""精确重放并导出 frozen receiver-risk 审计的尾部失败案例。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from cadsd_jscc.deepjscc_adapter import (  # noqa: E402
    deepjscc_forward_with_latents,
    load_deepjscc_model,
    received_latent_consistency_per_sample,
)
from cadsd_jscc.metrics import psnr_per_sample  # noqa: E402
from pc_fit_receiver_risk_controller import (  # noqa: E402
    parse_bool,
    read_csv,
    row_key,
    verify_hash,
)
from pc_imagenette_supervised_audit import (  # noqa: E402
    evaluate_probabilities,
    jensen_shannon,
    load_policy_dev,
    load_scratch_classifier,
    true_class_margin,
)
from pc_posterior_consistency_replication import (  # noqa: E402
    load_yaml,
    posterior_correct,
    resolve,
    write_csv,
)
from s10_short_chain_residual_shift_diffusion import ShortChainResidualShiftDiffusion  # noqa: E402
from s13_export_coco_train2017_c8_scaleup import derived_seed  # noqa: E402
from s5_residual_refiner_pilot import build_model, gate_tensor, try_load_lpips  # noqa: E402
from s6_imagenette_source_semantic_description_eval import (  # noqa: E402
    quantize_description,
    semantic_scores,
)


def diagnostic_event_type(row: dict[str, Any], primary_snrs: set[float]) -> str | None:
    if not parse_bool(row["clean_correct"]) or float(row["snr_db"]) not in primary_snrs:
        return None
    anchor_correct = parse_bool(row["anchor_correct"])
    posterior_correct = parse_bool(row["posterior_correct"])
    rejected = parse_bool(row["rejected"])
    if anchor_correct and not posterior_correct and not rejected:
        return "missed_new_error"
    if not anchor_correct and posterior_correct and rejected:
        return "rejected_posterior_repair"
    return None


def select_diagnostic_rows(
    rows: list[dict[str, Any]], primary_snrs: set[float]
) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        event_type = diagnostic_event_type(row, primary_snrs)
        if event_type is not None:
            selected.append({**row, "event_type": event_type})
    return selected


def assert_close(name: str, actual: float, expected: float, tolerance: float) -> None:
    if not math.isfinite(actual) or abs(actual - expected) > tolerance:
        raise RuntimeError(
            f"精确重放字段不一致 {name}: actual={actual}, expected={expected}, "
            f"tolerance={tolerance}"
        )


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    return transforms.ToPILImage()(image.detach().cpu().clamp(0, 1))


def save_case_gallery(
    case_dir: Path,
    event_type: str,
    sample_id: str,
    snr_db: float,
    images: dict[str, torch.Tensor],
    predictions: dict[str, dict[str, int]],
    diff_scale: float,
) -> None:
    case_dir.mkdir(parents=True)
    rendered = {
        "source": tensor_to_pil(images["source"]),
        "anchor": tensor_to_pil(images["anchor"]),
        "raw": tensor_to_pil(images["raw"]),
        "posterior": tensor_to_pil(images["posterior"]),
        "diff_x10": tensor_to_pil(
            (images["posterior"] - images["anchor"]).abs() * diff_scale
        ),
    }
    for name, image in rendered.items():
        image.save(case_dir / f"{name}.png")
    labels = ["source", "anchor", "raw", "posterior", f"|post-anchor| x{diff_scale:g}"]
    tile_width, tile_height = rendered["source"].size
    header_height = 72
    canvas = Image.new("RGB", (tile_width * len(labels), tile_height + header_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text(
        (6, 4),
        f"{event_type} | SNR={snr_db:g} dB | {sample_id}",
        fill="black",
        font=font,
    )
    for index, label in enumerate(labels):
        x = index * tile_width
        if label == labels[-1]:
            pred_text = "difference heatmap"
        else:
            pred_text = " ".join(
                f"{model}={predictions[model][label]}" for model in ("T", "G", "A")
            )
        draw.text((x + 6, 28), label, fill="black", font=font)
        draw.text((x + 6, 44), pred_text, fill="black", font=font)
        canvas.paste(rendered["diff_x10" if label == labels[-1] else label], (x, header_height))
    canvas.save(case_dir / "panel.png")


def probability_fields(
    prefix: str,
    probabilities_by_state: dict[str, torch.Tensor],
    labels: torch.Tensor,
    index: int,
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for state, probabilities in probabilities_by_state.items():
        prediction = probabilities.argmax(dim=1)
        confidence = probabilities.max(dim=1).values
        true_probability = probabilities.gather(1, labels[:, None]).squeeze(1)
        margin = true_class_margin(probabilities, labels)
        fields.update(
            {
                f"{prefix}_{state}_prediction": int(prediction[index]),
                f"{prefix}_{state}_confidence": float(confidence[index]),
                f"{prefix}_{state}_true_probability": float(true_probability[index]),
                f"{prefix}_{state}_true_margin": float(margin[index]),
                f"{prefix}_{state}_correct": bool(prediction[index] == labels[index]),
            }
        )
    source_probability = probabilities_by_state["source"][index : index + 1]
    anchor_probability = probabilities_by_state["anchor"][index : index + 1]
    posterior_probability = probabilities_by_state["posterior"][index : index + 1]
    fields[f"{prefix}_source_anchor_js"] = float(
        jensen_shannon(source_probability, anchor_probability)[0]
    )
    fields[f"{prefix}_source_posterior_js"] = float(
        jensen_shannon(source_probability, posterior_probability)[0]
    )
    fields[f"{prefix}_source_posterior_minus_anchor_js"] = (
        fields[f"{prefix}_source_posterior_js"] - fields[f"{prefix}_source_anchor_js"]
    )
    source_pred = int(probabilities_by_state["source"][index].argmax())
    anchor_pred = int(probabilities_by_state["anchor"][index].argmax())
    posterior_pred = int(probabilities_by_state["posterior"][index].argmax())
    fields[f"{prefix}_source_anchor_top1_changed"] = source_pred != anchor_pred
    fields[f"{prefix}_source_posterior_top1_changed"] = source_pred != posterior_pred
    fields[f"{prefix}_oracle_trueclass_posterior_mismatch"] = (
        posterior_pred != int(labels[index])
    )
    codes, decoded = quantize_description(source_probability)
    score_values = semantic_scores(
        decoded[0].cpu().numpy().astype(np.float64),
        anchor_probability[0].cpu().numpy().astype(np.float64),
        posterior_probability[0].cpu().numpy().astype(np.float64),
    )
    fields[f"{prefix}_source_description_codes_uint8"] = json.dumps(
        codes[0].cpu().tolist(), separators=(",", ":")
    )
    fields[f"{prefix}_source_description_probability"] = json.dumps(
        decoded[0].cpu().tolist(), separators=(",", ":")
    )
    for name, value in score_values.items():
        fields[f"{prefix}_{name}"] = float(value)
    return fields


def pairwise_auc(positive: list[float], negative: list[float]) -> float | None:
    if not positive or not negative:
        return None
    comparisons = [
        1.0 if p > n else 0.5 if p == n else 0.0 for p in positive for n in negative
    ]
    return float(np.mean(comparisons))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    source_config_path = resolve(config["source_extraction_config"])
    decisions_path = resolve(config["frozen_decisions_csv"])
    audit_path = resolve(config["frozen_audit_csv"])
    verify_hash(source_config_path, str(config["source_extraction_config_sha256"]))
    verify_hash(decisions_path, str(config["frozen_decisions_sha256"]))
    verify_hash(audit_path, str(config["frozen_audit_sha256"]))
    source_config = load_yaml(source_config_path)
    if bool(source_config["imagenette"].get("official_val_accessed")):
        raise RuntimeError("source config permits official validation access")
    decisions = read_csv(decisions_path)
    audit_rows = read_csv(audit_path)
    audit_by_key = {row_key(row): row for row in audit_rows}
    if len(audit_by_key) != len(audit_rows):
        raise RuntimeError("frozen audit keys are not unique")
    primary_snrs = {float(item) for item in config["selection"]["primary_snrs"]}
    selected = select_diagnostic_rows(decisions, primary_snrs)
    counts = {
        name: sum(row["event_type"] == name for row in selected)
        for name in ("missed_new_error", "rejected_posterior_repair")
    }
    expected = {
        "missed_new_error": int(config["selection"]["expected_missed_new_error_rows"]),
        "rejected_posterior_repair": int(
            config["selection"]["expected_rejected_posterior_repair_rows"]
        ),
    }
    if counts != expected:
        raise RuntimeError(f"diagnostic row counts differ: actual={counts}, expected={expected}")
    samples, classes = load_policy_dev(source_config)
    sample_index = {str(item["sample_id"]): index for index, item in enumerate(samples)}
    if len(sample_index) != len(samples):
        raise RuntimeError("policy-dev sample IDs are not unique")
    batch_size = int(source_config["batch_size"])
    groups: dict[tuple[float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        index = sample_index[str(row["sample_id"])]
        groups[(float(row["snr_db"]), index // batch_size * batch_size)].append(row)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "analysis_id": config["analysis_id"],
                    "selected_rows": len(selected),
                    "event_counts": counts,
                    "replay_batches": len(groups),
                    "channel_seed": source_config["channel_seeds"],
                    "official_val_accessed": False,
                },
                indent=2,
            )
        )
        return

    output = resolve(config["output_dir"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    cases_root = output / "cases"
    cases_root.mkdir()
    device = torch.device(args.device)
    main_source_config = load_yaml(source_config["source_config"])
    jscc = load_deepjscc_model(
        resolve(main_source_config["baseline"]["repo"]),
        resolve(source_config["deepjscc_checkpoint"]),
        int(main_source_config["rate"]["inner_channel"]),
        "AWGN",
        float(source_config["snrs"][0]),
        device,
    ).requires_grad_(False)
    b1_config = load_yaml(source_config["b1_config"])
    b1 = build_model(b1_config).to(device)
    b1.load_state_dict(
        torch.load(resolve(source_config["b1_checkpoint"]), map_location=device)[
            "model_state_dict"
        ]
    )
    b1.eval().requires_grad_(False)
    diffusion_config = load_yaml(source_config["diffusion_config"])
    diffusion = ShortChainResidualShiftDiffusion(diffusion_config).to(device)
    diffusion.load_state_dict(
        torch.load(resolve(source_config["diffusion_checkpoint"]), map_location=device)[
            "model_state_dict"
        ]
    )
    diffusion.eval().requires_grad_(False)
    gate, gate_temperature = load_scratch_classifier(
        source_config["controller"]["checkpoint"], classes, device, "G_gate"
    )
    auxiliary, auxiliary_temperature = load_scratch_classifier(
        source_config["risk_features"]["auxiliary_checkpoint"], classes, device, "G_aux"
    )
    teacher, teacher_temperature = load_scratch_classifier(
        source_config["imagenette"]["evaluator_checkpoint"], classes, device, "T_cls"
    )
    lpips_model, lpips_error = try_load_lpips(device, resolve("outputs/cache"))
    if lpips_model is None:
        raise RuntimeError(lpips_error)
    image_transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()]
    )
    model_specs = {
        "gate": (gate, gate_temperature),
        "aux": (auxiliary, auxiliary_temperature),
        "teacher": (teacher, teacher_temperature),
    }
    output_rows: list[dict[str, Any]] = []
    tolerance = float(config["diagnostics"]["metric_abs_tolerance"])
    channel_seed = int(source_config["channel_seeds"][0])
    for (snr, start), group_rows in sorted(groups.items()):
        jscc.change_channel("AWGN", snr)
        batch = samples[start : start + batch_size]
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
            int(source_config["proximal_steps"]),
            float(source_config["normalized_step_size"]),
        )
        states = {"source": target, "anchor": anchor, "raw": raw, "posterior": posterior}
        with torch.no_grad():
            probabilities = {
                model_name: {
                    state: evaluate_probabilities(model, temperature, images, source_config)
                    for state, images in states.items()
                }
                for model_name, (model, temperature) in model_specs.items()
            }
            dc_before = received_latent_consistency_per_sample(jscc, raw, received)
            dc_after = received_latent_consistency_per_sample(jscc, posterior, received)
            psnr = {
                state: psnr_per_sample(images, target)
                for state, images in states.items()
                if state != "source"
            }
            lpips = {
                state: lpips_model(images * 2 - 1, target * 2 - 1).flatten()
                for state, images in states.items()
                if state != "source"
            }
        batch_index = {str(item["sample_id"]): index for index, item in enumerate(batch)}
        for selected_row in group_rows:
            sample_id = str(selected_row["sample_id"])
            index = batch_index[sample_id]
            existing = audit_by_key[row_key(selected_row)]
            assert_close("dc_before", float(dc_before[index]), float(existing["dc_before"]), tolerance)
            assert_close("dc_after", float(dc_after[index]), float(existing["dc_after"]), tolerance)
            for state, csv_prefix in (
                ("anchor", "anchor"),
                ("raw", "raw"),
                ("posterior", "posterior"),
            ):
                assert_close(
                    f"{state}_psnr",
                    float(psnr[state][index]),
                    float(existing[f"{csv_prefix}_psnr"]),
                    tolerance,
                )
                assert_close(
                    f"{state}_lpips",
                    float(lpips[state][index]),
                    float(existing[f"{csv_prefix}_lpips"]),
                    tolerance,
                )
            teacher_predictions = {
                state: int(probabilities["teacher"][state][index].argmax())
                for state in states
            }
            gate_predictions = {
                state: int(probabilities["gate"][state][index].argmax()) for state in states
            }
            aux_predictions = {
                state: int(probabilities["aux"][state][index].argmax()) for state in states
            }
            if (teacher_predictions["anchor"] == int(labels[index])) != parse_bool(
                existing["anchor_correct"]
            ):
                raise RuntimeError("teacher anchor correctness did not reproduce")
            if (teacher_predictions["posterior"] == int(labels[index])) != parse_bool(
                existing["posterior_correct"]
            ):
                raise RuntimeError("teacher posterior correctness did not reproduce")
            row: dict[str, Any] = {
                "event_type": selected_row["event_type"],
                "channel_seed": channel_seed,
                "snr_db": snr,
                "sample_id": sample_id,
                "wnid": existing["wnid"],
                "class_idx": int(labels[index]),
                "frozen_risk_score": float(selected_row["risk_score"]),
                "frozen_threshold": float(selected_row["threshold"]),
                "frozen_rejected": parse_bool(selected_row["rejected"]),
                "dc_before": float(dc_before[index]),
                "dc_after": float(dc_after[index]),
                "anchor_psnr": float(psnr["anchor"][index]),
                "raw_psnr": float(psnr["raw"][index]),
                "posterior_psnr": float(psnr["posterior"][index]),
                "anchor_lpips": float(lpips["anchor"][index]),
                "raw_lpips": float(lpips["raw"][index]),
                "posterior_lpips": float(lpips["posterior"][index]),
                "anchor_posterior_l1": float(
                    (anchor[index] - posterior[index]).abs().mean()
                ),
                "anchor_posterior_rmse": float(
                    (anchor[index] - posterior[index]).square().mean().sqrt()
                ),
            }
            row.update(probability_fields("gate", probabilities["gate"], labels, index))
            row.update(probability_fields("aux", probabilities["aux"], labels, index))
            row.update(probability_fields("teacher", probabilities["teacher"], labels, index))
            output_rows.append(row)
            safe_name = sample_id.replace("/", "__").replace(".JPEG", "")
            case_dir = cases_root / f"{selected_row['event_type']}__snr{int(snr):02d}__{safe_name}"
            save_case_gallery(
                case_dir,
                str(selected_row["event_type"]),
                sample_id,
                snr,
                {state: images[index] for state, images in states.items()},
                {"T": teacher_predictions, "G": gate_predictions, "A": aux_predictions},
                float(config["diagnostics"]["diff_visualization_scale"]),
            )
        print(f"完成精确重放 snr={snr:g}, batch_start={start}, cases={len(group_rows)}")
    write_csv(output / "per_case.csv", output_rows)
    missed = [row for row in output_rows if row["event_type"] == "missed_new_error"]
    repairs = [
        row for row in output_rows if row["event_type"] == "rejected_posterior_repair"
    ]
    score_names = [
        "fullprob_cross_entropy_risk",
        "fullprob_js_risk",
        "fullprob_cosine_risk",
        "source_top1_logprob_risk",
    ]
    ranking = {
        model: {
            score: pairwise_auc(
                [float(row[f"{model}_{score}"]) for row in missed],
                [float(row[f"{model}_{score}"]) for row in repairs],
            )
            for score in score_names
        }
        for model in ("gate", "aux")
    }
    feasibility = {
        "missed_new_error_rows": len(missed),
        "rejected_posterior_repair_rows": len(repairs),
        "learned_gate_top1_description_detected_missed": sum(
            bool(row["gate_source_posterior_top1_changed"]) for row in missed
        ),
        "learned_aux_top1_description_detected_missed": sum(
            bool(row["aux_source_posterior_top1_changed"]) for row in missed
        ),
        "either_learned_top1_description_detected_missed": sum(
            bool(row["gate_source_posterior_top1_changed"])
            or bool(row["aux_source_posterior_top1_changed"])
            for row in missed
        ),
        "oracle_trueclass_vs_gate_detected_missed": sum(
            bool(row["gate_oracle_trueclass_posterior_mismatch"]) for row in missed
        ),
        "oracle_trueclass_vs_aux_detected_missed": sum(
            bool(row["aux_oracle_trueclass_posterior_mismatch"]) for row in missed
        ),
        "teacher_source_top1_detected_missed": sum(
            bool(row["teacher_source_posterior_top1_changed"]) for row in missed
        ),
        "gate_and_aux_posterior_correct_on_missed": sum(
            bool(row["gate_posterior_correct"]) and bool(row["aux_posterior_correct"])
            for row in missed
        ),
        "fullprob_risk_pairwise_auc_missed_vs_rejected_repairs": ranking,
    }
    payload = {
        "config": config,
        "event_counts": counts,
        "replayed_batches": len(groups),
        "feasibility": feasibility,
        "official_val_accessed": False,
        "status": "POSTHOC_DIAGNOSTIC_COMPLETE",
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
