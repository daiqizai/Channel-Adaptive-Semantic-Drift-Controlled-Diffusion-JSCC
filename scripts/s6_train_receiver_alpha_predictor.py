from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from s6_residual_shrink_selection import (  # noqa: E402
    alpha_name,
    classify_paths,
    compute_pair_metrics,
    load_classifier,
    load_rgb_tensor,
    resolve_device,
    snr_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a tiny receiver-side residual alpha predictor from existing S6 tables."
    )
    parser.add_argument("--config", default="configs/s6_receiver_alpha_predictor_exp_s4_006.yaml")
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def serialize_value(value: Any) -> Any:
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


def bool_from_csv(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def mean(values: list[float | None]) -> float | None:
    clean = [float(item) for item in values if item is not None]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def rate(values: list[bool]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def to_float(value: Any, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    return float(value)


def to_int(value: Any, default: int = 0) -> int:
    if value in ("", None):
        return default
    return int(float(value))


def fmt(value: Any, digits: int = 4) -> str:
    if value in ("", None):
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def signed(value: Any, digits: int = 4) -> str:
    if value in ("", None):
        return ""
    return f"{float(value):+.{digits}f}"


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


def row_key(row: dict[str, Any]) -> tuple[str, float, str]:
    return str(row["split"]), float(row["snr_db"]), str(row["sample"])


def alpha_value(value: Any) -> float:
    if value in ("", None):
        return 0.0
    return float(value)


def alpha_text(alpha: float) -> str:
    if abs(alpha) < 1e-9:
        return ""
    return str(float(alpha))


def split_candidate_roots(config: dict[str, Any]) -> dict[str, Path]:
    return {str(item["name"]): resolve_project_path(item["candidate_root"]) for item in config["splits"]}


def candidate_path(config: dict[str, Any], split: str, alpha: float, snr: float, sample: str) -> Path:
    if abs(alpha) < 1e-9:
        raise ValueError("alpha=0 has no candidate path")
    root = split_candidate_roots(config)[split]
    return root / alpha_name(alpha) / snr_name(snr) / sample


def validate_inputs(config: dict[str, Any]) -> dict[str, Any]:
    paths = {key: resolve_project_path(value) for key, value in config["inputs"].items()}
    paths["classifier_weights"] = resolve_project_path(config["classifier"]["weights_file"])
    for split, root in split_candidate_roots(config).items():
        paths[f"{split}_candidate_root"] = root
    missing = [f"{key}: {path}" for key, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))
    weights = paths["classifier_weights"]
    if not weights.is_file() or weights.stat().st_size < 10 * 1024 * 1024:
        raise RuntimeError(f"Classifier weights missing from local cache: {weights}")

    rows = read_csv(paths["adaptive_per_sample_csv"])
    policies = set(config["source_policies"].values())
    counts: dict[str, int] = {policy: 0 for policy in policies}
    for row in rows:
        if row.get("policy") in counts:
            counts[row["policy"]] += 1
    absent = [policy for policy, count in counts.items() if count == 0]
    if absent:
        raise RuntimeError(f"Missing source policy rows in adaptive per-sample CSV: {absent}")

    candidate_checks = 0
    for row in rows:
        if row.get("policy") != config["source_policies"]["m0"]:
            continue
        split = row["split"]
        snr = float(row["snr_db"])
        sample = row["sample"]
        for alpha in config["alphas"]:
            alpha_float = float(alpha)
            if alpha_float == 0.0:
                continue
            path = candidate_path(config, split, alpha_float, snr, sample)
            if not path.exists():
                raise FileNotFoundError(f"Missing alpha candidate: {path}")
            candidate_checks += 1
    return {
        "paths": {key: project_relative(path) for key, path in paths.items()},
        "policy_counts": counts,
        "candidate_checks": candidate_checks,
    }


def index_rows(rows: list[dict[str, str]]) -> dict[str, dict[tuple[str, float, str], dict[str, str]]]:
    indexed: dict[str, dict[tuple[str, float, str], dict[str, str]]] = {}
    for row in rows:
        indexed.setdefault(row["policy"], {})[row_key(row)] = row
    return indexed


def residual_image_features(m0_path: Path, full_path: Path) -> dict[str, float]:
    m0 = load_rgb_tensor(m0_path)
    full = load_rgb_tensor(full_path)
    diff = full - m0
    abs_diff = diff.abs()
    gray = m0.mean(dim=0)
    gx = (gray[:, 1:] - gray[:, :-1]).abs().mean()
    gy = (gray[1:, :] - gray[:-1, :]).abs().mean()
    return {
        "m0_mean": float(m0.mean().item()),
        "m0_std": float(m0.std(unbiased=False).item()),
        "m0_edge_mean": float(((gx + gy) * 0.5).item()),
        "full_residual_mae": float(abs_diff.mean().item()),
        "full_residual_rmse": float(torch.sqrt((diff**2).mean()).item()),
        "full_residual_p95": float(torch.quantile(abs_diff.flatten(), 0.95).item()),
        "full_residual_max": float(abs_diff.max().item()),
        "full_residual_signed_mean": float(diff.mean().item()),
    }


def build_examples(config: dict[str, Any], source_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[str]]:
    policies = config["source_policies"]
    indexed = index_rows(source_rows)
    split_order = {str(item["name"]): idx for idx, item in enumerate(config["splits"])}
    snrs = [float(item) for item in config["snrs"]]
    feature_names = [
        "snr_scaled",
        *[f"snr_is_{int(snr)}db" for snr in snrs],
        "m0_top1_prob",
        "full_top1_prob",
        "full_conf_delta",
        "full_conf_ratio",
        "full_matches_m0_top1",
        "fixed_schedule_alpha",
        "m0_mean",
        "m0_std",
        "m0_edge_mean",
        "full_residual_mae",
        "full_residual_rmse",
        "full_residual_p95",
        "full_residual_max",
        "full_residual_signed_mean",
    ]
    examples: list[dict[str, Any]] = []
    for key in sorted(indexed[policies["m0"]], key=lambda item: (split_order.get(item[0], 999), item[1], item[2])):
        m0 = indexed[policies["m0"]][key]
        full = indexed[policies["full"]][key]
        fixed = indexed[policies["fixed"]][key]
        oracle = indexed[policies["oracle"]][key]
        split, snr, sample = key
        fixed_alpha = alpha_value(fixed["selected_alpha"])
        m0_prob = float(m0["m0_top1_prob"])
        full_prob = float(full["candidate_top1_prob"])
        full_path = resolve_project_path(full["candidate"])
        image_features = residual_image_features(resolve_project_path(m0["m0_reconstruction"]), full_path)
        feature_map = {
            "snr_scaled": snr / max(snrs),
            **{f"snr_is_{int(item)}db": 1.0 if abs(snr - item) < 1e-9 else 0.0 for item in snrs},
            "m0_top1_prob": m0_prob,
            "full_top1_prob": full_prob,
            "full_conf_delta": full_prob - m0_prob,
            "full_conf_ratio": full_prob / max(m0_prob, 1e-6),
            "full_matches_m0_top1": 1.0 if bool_from_csv(full["candidate_matches_m0_top1"]) else 0.0,
            "fixed_schedule_alpha": fixed_alpha,
            **image_features,
        }
        target_alpha = alpha_value(oracle["selected_alpha"])
        examples.append(
            {
                "split": split,
                "snr_db": snr,
                "sample": sample,
                "features": [float(feature_map[name]) for name in feature_names],
                "target_alpha": target_alpha,
                "m0": m0,
                "full": full,
                "fixed": fixed,
                "oracle": oracle,
            }
        )
    return examples, feature_names


class AlphaPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_units: int, output_dim: int) -> None:
        super().__init__()
        if hidden_units <= 0:
            self.net = nn.Linear(input_dim, output_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_units),
                nn.ReLU(),
                nn.Linear(hidden_units, output_dim),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def tensorize_features(examples: list[dict[str, Any]], mean_values: torch.Tensor, std_values: torch.Tensor) -> torch.Tensor:
    raw = torch.tensor([example["features"] for example in examples], dtype=torch.float32)
    return (raw - mean_values) / std_values.clamp_min(1e-6)


def train_predictor(
    config: dict[str, Any],
    examples: list[dict[str, Any]],
    feature_names: list[str],
    device: torch.device,
) -> tuple[AlphaPredictor, dict[str, Any]]:
    train_split = str(config["predictor"]["train_split"])
    train_examples = [example for example in examples if example["split"] == train_split]
    if not train_examples:
        raise RuntimeError(f"No train examples for split: {train_split}")
    alphas = [float(item) for item in config["alphas"]]
    alpha_to_index = {alpha: idx for idx, alpha in enumerate(alphas)}
    torch.manual_seed(int(config["predictor"]["seed"]))

    raw_train = torch.tensor([example["features"] for example in train_examples], dtype=torch.float32)
    feature_mean = raw_train.mean(dim=0)
    feature_std = raw_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    x_train = tensorize_features(train_examples, feature_mean, feature_std).to(device)
    y_train = torch.tensor([alpha_to_index[float(example["target_alpha"])] for example in train_examples], dtype=torch.long).to(device)

    counts = torch.bincount(y_train.detach().cpu(), minlength=len(alphas)).float()
    if str(config["predictor"].get("class_weighting", "")) == "inverse_frequency":
        weights = counts.sum() / counts.clamp_min(1.0)
        weights = weights / weights.mean().clamp_min(1e-6)
    else:
        weights = torch.ones_like(counts)
    model = AlphaPredictor(
        input_dim=len(feature_names),
        hidden_units=int(config["predictor"]["hidden_units"]),
        output_dim=len(alphas),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["predictor"]["learning_rate"]),
        weight_decay=float(config["predictor"]["weight_decay"]),
    )
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(config["predictor"]["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 25 == 0 or epoch == int(config["predictor"]["epochs"]):
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                acc = float((pred == y_train).float().mean().item())
            history.append({"epoch": epoch, "loss": float(loss.item()), "train_target_accuracy": acc})
    model.eval()
    metadata = {
        "alphas": alphas,
        "alpha_to_index": {str(key): value for key, value in alpha_to_index.items()},
        "feature_names": feature_names,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "class_counts_validation": {str(alphas[idx]): int(counts[idx].item()) for idx in range(len(alphas))},
        "class_weights": {str(alphas[idx]): float(weights[idx].item()) for idx in range(len(alphas))},
        "train_history": history,
    }
    return model, metadata


def predict_examples(
    model: AlphaPredictor,
    examples: list[dict[str, Any]],
    model_metadata: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    alphas = [float(item) for item in model_metadata["alphas"]]
    feature_mean = torch.tensor(model_metadata["feature_mean"], dtype=torch.float32)
    feature_std = torch.tensor(model_metadata["feature_std"], dtype=torch.float32)
    x = tensorize_features(examples, feature_mean, feature_std).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(x), dim=1).detach().cpu()
    predictions: list[dict[str, Any]] = []
    for example, probs in zip(examples, probabilities):
        pred_index = int(torch.argmax(probs).item())
        predictions.append(
            {
                **example,
                "predicted_alpha": alphas[pred_index],
                "predicted_alpha_prob": float(probs[pred_index].item()),
                "class_probabilities": {str(alphas[idx]): float(value) for idx, value in enumerate(probs.tolist())},
            }
        )
    return predictions


def classify_predicted_candidates(
    config: dict[str, Any],
    predictions: list[dict[str, Any]],
    classifier_model,
    classifier_preprocess,
    device: torch.device,
) -> tuple[dict[tuple[str, float, str], dict[str, Any]], dict[str, float]]:
    to_classify: list[dict[str, Any]] = []
    for item in predictions:
        alpha = float(item["predicted_alpha"])
        if alpha == 0.0:
            continue
        path = candidate_path(config, item["split"], alpha, float(item["snr_db"]), item["sample"])
        to_classify.append({**item, "candidate_path": path})
    classified: dict[tuple[str, float, str], dict[str, Any]] = {}
    times: dict[str, float] = {}
    batch_size = int(config["classifier"]["batch_size"])
    topk = int(config["classifier"]["topk"])
    for split in sorted({item["split"] for item in to_classify}):
        split_items = [item for item in to_classify if item["split"] == split]
        if not split_items:
            continue
        preds, elapsed = classify_paths(
            classifier_model,
            classifier_preprocess,
            [item["candidate_path"] for item in split_items],
            batch_size,
            topk,
            device,
        )
        times[f"{split}_predicted_candidates"] = elapsed
        for item, pred in zip(split_items, preds):
            classified[row_key(item)] = {
                "candidate_path": item["candidate_path"],
                "candidate_top1_index": int(pred["top_indices"][0]),
                "candidate_top1_prob": float(pred["top_probs"][0]),
            }
    return classified, times


def make_policy_rows(
    predictions: list[dict[str, Any]],
    classified: dict[tuple[str, float, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in predictions:
        m0 = item["m0"]
        oracle = item["oracle"]
        snr = float(item["snr_db"])
        predicted_alpha = float(item["predicted_alpha"])
        original_top1 = int(m0["original_top1_index"])
        m0_top1 = int(m0["m0_top1_index"])
        m0_prob = float(m0["m0_top1_prob"])
        m0_matches_origin = bool_from_csv(m0["m0_matches_original_top1"])
        if predicted_alpha == 0.0:
            candidate_top1 = m0_top1
            candidate_prob = m0_prob
            candidate_source = resolve_project_path(m0["m0_reconstruction"])
            accept = False
            reason = "predict_fallback_m0"
        else:
            pred = classified[row_key(item)]
            candidate_top1 = int(pred["candidate_top1_index"])
            candidate_prob = float(pred["candidate_top1_prob"])
            candidate_source = pred["candidate_path"]
            accept = candidate_top1 == m0_top1
            reason = "predicted_alpha_top1_consistent" if accept else "predicted_alpha_rejected_by_top1_gate"
        final_source = candidate_source if accept else resolve_project_path(m0["m0_reconstruction"])
        final_top1 = candidate_top1 if accept else m0_top1
        final_prob = candidate_prob if accept else m0_prob
        candidate_matches_origin = candidate_top1 == original_top1
        max_top1_alpha = alpha_value(oracle["max_top1_consistent_alpha"])
        any_candidate_matches_origin = bool_from_csv(oracle["any_candidate_matches_original_top1"])
        rows.append(
            {
                "split": item["split"],
                "policy": "receiver_alpha_predictor_top1_fallback",
                "snr_db": snr,
                "sample": item["sample"],
                "target_alpha": item["target_alpha"],
                "predicted_alpha": predicted_alpha,
                "predicted_alpha_prob": item["predicted_alpha_prob"],
                "prediction_matches_target_alpha": abs(predicted_alpha - float(item["target_alpha"])) < 1e-9,
                "max_top1_consistent_alpha": oracle["max_top1_consistent_alpha"],
                "predicted_alpha_le_max_top1_consistent_alpha": predicted_alpha <= max_top1_alpha + 1e-9,
                "accept_candidate": accept,
                "decision_reason": reason,
                "original": m0["original"],
                "m0_reconstruction": m0["m0_reconstruction"],
                "candidate": project_relative(candidate_source),
                "final_source": project_relative(final_source),
                "original_top1_index": original_top1,
                "original_top1_label": m0["original_top1_label"],
                "original_top1_prob": float(m0["original_top1_prob"]),
                "m0_top1_index": m0_top1,
                "m0_top1_label": m0["m0_top1_label"],
                "m0_top1_prob": m0_prob,
                "candidate_top1_index": candidate_top1,
                "candidate_top1_prob": candidate_prob,
                "final_top1_index": final_top1,
                "final_top1_prob": final_prob,
                "m0_matches_original_top1": m0_matches_origin,
                "candidate_matches_original_top1": candidate_matches_origin,
                "candidate_matches_m0_top1": candidate_top1 == m0_top1,
                "final_matches_original_top1": final_top1 == original_top1,
                "final_matches_m0_top1": final_top1 == m0_top1,
                "accepted_repair": accept and (not m0_matches_origin) and final_top1 == original_top1,
                "accepted_new_error": accept and m0_matches_origin and final_top1 != original_top1,
                "any_candidate_matches_original_top1": any_candidate_matches_origin,
                "missed_repair": (not m0_matches_origin) and final_top1 != original_top1 and any_candidate_matches_origin,
                "class_probabilities": item["class_probabilities"],
            }
        )
    return rows


def stack_paths(paths: list[Path]) -> torch.Tensor:
    return torch.stack([load_rgb_tensor(path) for path in paths])


def summarize_rows(
    split: str,
    policy: str,
    snr: float | str,
    rows: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    references = stack_paths([resolve_project_path(row["original"]) for row in rows])
    m0 = stack_paths([resolve_project_path(row["m0_reconstruction"]) for row in rows])
    final = stack_paths([resolve_project_path(row["final_source"]) for row in rows])
    m0_metrics = compute_pair_metrics(references, m0, None, device, batch_size)
    final_metrics = compute_pair_metrics(references, final, None, device, batch_size)
    m0_failure = 1.0 - rate([bool_from_csv(row["m0_matches_original_top1"]) for row in rows])
    final_failure = 1.0 - rate([bool_from_csv(row["final_matches_original_top1"]) for row in rows])
    return {
        "split": split,
        "policy": policy,
        "snr_db": snr,
        "num_images": len(rows),
        "accept_rate": rate([bool_from_csv(row["accept_candidate"]) for row in rows]),
        "fallback_rate": 1.0 - rate([bool_from_csv(row["accept_candidate"]) for row in rows]),
        "target_alpha_accuracy": rate([bool_from_csv(row["prediction_matches_target_alpha"]) for row in rows]),
        "predicted_alpha_le_oracle_rate": rate([bool_from_csv(row["predicted_alpha_le_max_top1_consistent_alpha"]) for row in rows]),
        "mean_predicted_alpha": mean([float(row["predicted_alpha"]) for row in rows]),
        "mean_accepted_alpha": mean(
            [float(row["predicted_alpha"]) for row in rows if bool_from_csv(row["accept_candidate"]) and float(row["predicted_alpha"]) > 0]
        ),
        "m0_failure_rate": m0_failure,
        "final_failure_rate": final_failure,
        "delta_final_failure_vs_m0": final_failure - m0_failure,
        "repair_count": int(sum(bool_from_csv(row["accepted_repair"]) for row in rows)),
        "accepted_new_error_count": int(sum(bool_from_csv(row["accepted_new_error"]) for row in rows)),
        "missed_repair_count": int(sum(bool_from_csv(row["missed_repair"]) for row in rows)),
        "available_repair_count": int(
            sum(
                (not bool_from_csv(row["m0_matches_original_top1"]))
                and bool_from_csv(row["any_candidate_matches_original_top1"])
                for row in rows
            )
        ),
        "final_mse": final_metrics["mse"],
        "final_psnr_db": final_metrics["psnr_db"],
        "final_ssim": final_metrics["ssim"],
        "final_ms_ssim": final_metrics["ms_ssim"],
        "final_lpips": "",
        "delta_psnr_vs_m0_db": None
        if final_metrics["psnr_db"] is None or m0_metrics["psnr_db"] is None
        else final_metrics["psnr_db"] - m0_metrics["psnr_db"],
        "delta_lpips_vs_m0": "",
        "delta_ms_ssim_vs_m0": None
        if final_metrics["ms_ssim"] is None or m0_metrics["ms_ssim"] is None
        else final_metrics["ms_ssim"] - m0_metrics["ms_ssim"],
    }


def build_summary(config: dict[str, Any], policy_rows: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    batch_size = int(config["evaluation"]["image_batch_size"])
    policy = "receiver_alpha_predictor_top1_fallback"
    for split in [str(item["name"]) for item in config["splits"]]:
        split_rows = [row for row in policy_rows if row["split"] == split]
        for snr in config["snrs"]:
            snr_rows = [row for row in split_rows if abs(float(row["snr_db"]) - float(snr)) < 1e-9]
            summaries.append(summarize_rows(split, policy, float(snr), snr_rows, device, batch_size))
        summaries.append(summarize_rows(split, policy, "all", split_rows, device, batch_size))
    return summaries


def all_row(rows: list[dict[str, Any]], split: str, policy: str) -> dict[str, Any]:
    for row in rows:
        if row.get("split") == split and row.get("policy") == policy and str(row.get("snr_db")) == "all":
            return row
    raise KeyError(f"Missing all row: {split}/{policy}")


def load_comparison_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    adaptive = read_csv(resolve_project_path(config["inputs"]["adaptive_summary_csv"]))
    two_stage = read_csv(resolve_project_path(config["inputs"]["two_stage_summary_csv"]))
    for row in adaptive:
        rows.append({**row, "source": "adaptive_summary"})
    for row in two_stage:
        if row.get("policy") == "full_then_fixed_schedule":
            rows.append({**row, "source": "two_stage_summary"})
    return rows


def plot_tradeoff(config: dict[str, Any], summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    splits = [str(item["name"]) for item in config["splits"]]
    policies = [
        "top1_full_strength",
        "fixed_validation_top1_shrink_schedule",
        "full_then_fixed_schedule",
        "receiver_alpha_predictor_top1_fallback",
        "adaptive_max_top1_consistent_alpha",
        "always_full_strength",
    ]
    labels = {
        "top1_full_strength": "full",
        "fixed_validation_top1_shrink_schedule": "fixed",
        "full_then_fixed_schedule": "2-stage",
        "receiver_alpha_predictor_top1_fallback": "predictor",
        "adaptive_max_top1_consistent_alpha": "adaptive",
        "always_full_strength": "always",
    }
    colors = ["#5875a4", "#cc8963", "#5f9e6e", "#8172b3", "#b55d60", "#8c8c8c"]
    width = 0.13
    x_positions = list(range(len(splits)))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    for idx, policy in enumerate(policies):
        offsets = [x + (idx - (len(policies) - 1) / 2) * width for x in x_positions]
        rows = [all_row(summary_rows, split, policy) for split in splits]
        axes[0].bar(
            offsets,
            [to_float(row["delta_psnr_vs_m0_db"]) for row in rows],
            width=width,
            color=colors[idx],
            label=labels[policy],
        )
        axes[1].bar(
            offsets,
            [to_int(row["accepted_new_error_count"]) for row in rows],
            width=width,
            color=colors[idx],
        )
    for ax, title, ylabel in [
        (axes[0], "Quality Gain", "PSNR delta vs M0 (dB)"),
        (axes[1], "Accepted New Errors", "count"),
    ]:
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(splits)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, ncols=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    lines = ["| " + " | ".join(label for _key, label in columns) + " |"]
    lines.append("|" + "|".join(["---" for _ in columns]) + "|")
    for row in rows:
        values = [fmt(row.get(key, "")) for key, _label in columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def make_report(config: dict[str, Any], summary_rows: list[dict[str, Any]], metadata: dict[str, Any]) -> str:
    splits = [str(item["name"]) for item in config["splits"]]
    predictor = "receiver_alpha_predictor_top1_fallback"
    adaptive = "adaptive_max_top1_consistent_alpha"
    two_stage = "full_then_fixed_schedule"
    predictor_psnr = "/".join(signed(all_row(summary_rows, split, predictor)["delta_psnr_vs_m0_db"]) for split in splits)
    predictor_new = "/".join(str(all_row(summary_rows, split, predictor)["accepted_new_error_count"]) for split in splits)
    two_stage_psnr = "/".join(signed(all_row(summary_rows, split, two_stage)["delta_psnr_vs_m0_db"]) for split in splits)
    adaptive_psnr = "/".join(signed(all_row(summary_rows, split, adaptive)["delta_psnr_vs_m0_db"]) for split in splits)
    display_rows: list[dict[str, Any]] = []
    for split in splits:
        for policy in [
            "top1_full_strength",
            "fixed_validation_top1_shrink_schedule",
            two_stage,
            predictor,
            adaptive,
            "always_full_strength",
        ]:
            row = all_row(summary_rows, split, policy)
            display_rows.append(
                {
                    "split": split,
                    "policy": policy,
                    "delta_psnr": signed(row["delta_psnr_vs_m0_db"]),
                    "failure_delta": signed(row["delta_final_failure_vs_m0"]),
                    "accept": fmt(row["accept_rate"]),
                    "target_acc": fmt(row.get("target_alpha_accuracy", "")),
                    "new_error": row["accepted_new_error_count"],
                    "missed_repair": row["missed_repair_count"],
                }
            )
    lines = [
        "# Receiver Alpha Predictor",
        "",
        "This derived analysis trains a tiny tabular receiver-side alpha predictor on validation only.",
        "It predicts one residual alpha from receiver-visible features, then applies the same top-1 consistency gate before accepting the candidate.",
        "No diffusion, residual regeneration, LPIPS, or external download is used.",
        "",
        "## Bottom Line",
        "",
        f"- Predictor PSNR deltas on validation/held-out/test-like: `{predictor_psnr}` dB.",
        f"- Predictor accepted new errors on validation/held-out/test-like: `{predictor_new}`.",
        f"- Two-stage policy PSNR deltas: `{two_stage_psnr}` dB.",
        f"- Exhaustive adaptive alpha PSNR deltas: `{adaptive_psnr}` dB.",
        "- Interpretation: this is a learned deployability pilot. If it underperforms two-stage or adaptive alpha, it is still useful evidence that the alpha predictor needs richer features or training-time integration.",
        "",
        "## All-Split Policy Summary",
        "",
    ]
    lines += markdown_table(
        display_rows,
        [
            ("split", "Split"),
            ("policy", "Policy"),
            ("delta_psnr", "Delta PSNR"),
            ("failure_delta", "Failure Delta"),
            ("accept", "Accept"),
            ("target_acc", "Target Acc"),
            ("new_error", "New Error"),
            ("missed_repair", "Missed Repair"),
        ],
    )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Summary CSV: `{metadata['summary_csv']}`",
            f"- Per-sample CSV: `{metadata['per_sample_csv']}`",
            f"- Feature table: `{metadata['features_csv']}`",
            f"- Model metadata: `{metadata['model_metadata_json']}`",
            f"- Training history: `{metadata['training_history_csv']}`",
            f"- Tradeoff figure: `{metadata['tradeoff_figure']}`",
            f"- Metadata: `{metadata['metadata_json']}`",
            "",
            "## Caveats",
            "",
            "- The predictor target is the pseudo-label adaptive alpha, not supervised semantic truth.",
            "- The final acceptance still uses the frozen AlexNet top-1 gate, so this is not semantic repair.",
            "- LPIPS is intentionally omitted to avoid external weight loading; compare PSNR/SSIM/MS-SSIM and semantic counts only.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    input_manifest = validate_inputs(config)
    output_dir = resolve_project_path(args.output_dir or config["outputs"]["output_dir"])
    dry_payload = {
        "status": "ok",
        "config": project_relative(config_path),
        "inputs": input_manifest,
        "output_dir": project_relative(output_dir),
        "proxy_environment_present": proxy_environment_present(),
        "notes": config.get("notes", []),
    }
    if args.dry_run:
        print(json.dumps(dry_payload, indent=2, ensure_ascii=False))
        return
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output_dir / "config.yaml")

    device = resolve_device(args.device)
    source_rows = read_csv(resolve_project_path(config["inputs"]["adaptive_per_sample_csv"]))
    examples, feature_names = build_examples(config, source_rows)
    model, model_metadata = train_predictor(config, examples, feature_names, device)
    predictions = predict_examples(model, examples, model_metadata, device)

    classifier_model, classifier_preprocess, _categories = load_classifier(config, device)
    classified, classification_times = classify_predicted_candidates(
        config,
        predictions,
        classifier_model,
        classifier_preprocess,
        device,
    )
    policy_rows = make_policy_rows(predictions, classified)
    predictor_summary = build_summary(config, policy_rows, device)
    comparison_rows = load_comparison_rows(config)
    summary_rows = [*comparison_rows, *predictor_summary]

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tradeoff_figure = figures_dir / "receiver_alpha_predictor_tradeoff.png"
    plot_tradeoff(config, summary_rows, tradeoff_figure)

    summary_csv = output_dir / "summary.csv"
    per_sample_csv = output_dir / "per_sample.csv"
    features_csv = output_dir / "features.csv"
    training_history_csv = output_dir / "training_history.csv"
    model_metadata_json = output_dir / "model_metadata.json"
    metadata_json = output_dir / "metadata.json"
    report_path = output_dir / "REPORT.md"
    write_csv(summary_csv, summary_rows)
    write_csv(per_sample_csv, policy_rows)
    write_csv(
        features_csv,
        [
            {
                "split": item["split"],
                "snr_db": item["snr_db"],
                "sample": item["sample"],
                "target_alpha": item["target_alpha"],
                **{name: value for name, value in zip(feature_names, item["features"])},
            }
            for item in examples
        ],
    )
    write_csv(training_history_csv, model_metadata["train_history"])
    save_json(model_metadata_json, model_metadata)
    torch.save(model.state_dict(), output_dir / "model_state.pt")
    metadata = {
        "analysis_id": config["analysis_id"],
        "method": config["method"],
        "project_commit": git_commit(),
        "git_dirty_state": git_dirty_state(),
        "python": sys.version,
        "platform": platform.platform(),
        "config": project_relative(config_path),
        "input_manifest": input_manifest,
        "summary_csv": project_relative(summary_csv),
        "per_sample_csv": project_relative(per_sample_csv),
        "features_csv": project_relative(features_csv),
        "training_history_csv": project_relative(training_history_csv),
        "model_metadata_json": project_relative(model_metadata_json),
        "model_state": project_relative(output_dir / "model_state.pt"),
        "tradeoff_figure": project_relative(tradeoff_figure),
        "metadata_json": project_relative(metadata_json),
        "device": str(device),
        "classification_times_sec": classification_times,
        "lpips": "omitted",
        "proxy_environment_present": proxy_environment_present(),
        "notes": config.get("notes", []),
    }
    save_json(metadata_json, metadata)
    report_path.write_text(make_report(config, summary_rows, metadata), encoding="utf-8")
    print(json.dumps({"output_dir": project_relative(output_dir), "report": project_relative(report_path)}, indent=2))


if __name__ == "__main__":
    main()
