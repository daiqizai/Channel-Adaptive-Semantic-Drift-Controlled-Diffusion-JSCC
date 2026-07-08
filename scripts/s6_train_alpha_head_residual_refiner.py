from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cadsd_jscc.metrics import ms_ssim_per_sample, psnr_per_sample, ssim_per_sample  # noqa: E402
from s5_residual_refiner_pilot import (  # noqa: E402
    ResidualBlock,
    classify_paths,
    label_for,
    load_classifier,
    load_rgb_pil,
    load_rgb_tensor,
    project_relative,
    resolve_device,
    resolve_project_path,
    save_json,
    snr_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small alpha head on top of the EXP-S4-006 residual refiner."
    )
    parser.add_argument("--config", default="configs/s6_alpha_head_residual_refiner_pilot_exp_s4_006.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


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


def mean(values: list[float | None]) -> float | None:
    clean = [float(item) for item in values if item is not None]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def rate(flags: list[bool]) -> float:
    if not flags:
        return 0.0
    return float(sum(flags) / len(flags))


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
        return "N/A"
    return f"{float(value):.{digits}f}"


def signed(value: Any, digits: int = 4) -> str:
    if value in ("", None):
        return "N/A"
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


def parse_snrs(config: dict[str, Any]) -> list[float]:
    return [float(item) for item in config["snrs"]]


def snr_key(snr: float) -> str:
    if float(snr).is_integer():
        return str(int(snr))
    return str(snr)


def residual_gate(config: dict[str, Any], snr: float) -> float:
    key = snr_key(snr)
    return float(config["model"]["residual_gates"][key])


def gate_tensor(config: dict[str, Any], snr_db: torch.Tensor, device: torch.device) -> torch.Tensor:
    gates = [residual_gate(config, float(item)) for item in snr_db.detach().cpu().tolist()]
    return torch.tensor(gates, dtype=torch.float32, device=device)


def row_key(row: dict[str, str]) -> tuple[str, float, str]:
    return str(row["split"]), float(row["snr_db"]), str(row["sample"])


def alpha_index(alpha: float, alphas: list[float]) -> int:
    return min(range(len(alphas)), key=lambda idx: abs(float(alphas[idx]) - float(alpha)))


def alpha_class_counts(rows: list[dict[str, Any]], num_classes: int) -> list[int]:
    counts = [0 for _ in range(num_classes)]
    for row in rows:
        counts[int(row["target_alpha_index"])] += 1
    return counts


def class_weight_values(config: dict[str, Any], train_rows: list[dict[str, Any]]) -> list[float] | None:
    strategy = str(config["training"].get("class_weighting", "none"))
    if strategy in {"", "none", "false", "off"}:
        return None
    if strategy != "inverse_frequency":
        raise ValueError(f"Unsupported class_weighting strategy: {strategy}")
    counts = alpha_class_counts(train_rows, len(config["alphas"]))
    if any(count == 0 for count in counts):
        raise RuntimeError(f"Cannot use inverse-frequency class weights with empty alpha class: {counts}")
    total = float(sum(counts))
    power = float(config["training"].get("class_weight_power", 1.0))
    raw = [(total / (len(counts) * float(count))) ** power for count in counts]
    normalize_mean = bool(config["training"].get("class_weight_normalize_mean", True))
    if normalize_mean:
        scale = len(raw) / sum(raw)
        raw = [weight * scale for weight in raw]
    return [float(weight) for weight in raw]


def validate_monotonic_gates(config: dict[str, Any], snrs: list[float]) -> dict[str, float]:
    gates = {snr: residual_gate(config, snr) for snr in snrs}
    ordered = sorted(gates)
    for left, right in zip(ordered, ordered[1:]):
        if gates[left] + 1e-12 < gates[right]:
            raise ValueError(f"Residual gate must not increase with SNR: {left}={gates[left]}, {right}={gates[right]}")
    return {str(snr): gate for snr, gate in gates.items()}


def select_policy_rows(config: dict[str, Any], rows: list[dict[str, str]], policy: str) -> list[dict[str, Any]]:
    alphas = [float(item) for item in config["alphas"]]
    selected: list[dict[str, Any]] = []
    for row in rows:
        if row.get("policy") != policy:
            continue
        selected_alpha = to_float(row.get("selected_alpha"), 0.0)
        idx = alpha_index(selected_alpha, alphas)
        item = dict(row)
        item["target_alpha"] = float(alphas[idx])
        item["target_alpha_index"] = int(idx)
        selected.append(item)
    return selected


def validate_inputs(config: dict[str, Any], snrs: list[float]) -> dict[str, Any]:
    base_checkpoint = resolve_project_path(config["inputs"]["base_refiner_checkpoint"])
    adaptive_csv = resolve_project_path(config["inputs"]["adaptive_per_sample_csv"])
    adaptive_summary = resolve_project_path(config["inputs"]["adaptive_summary_csv"])
    two_stage_summary = resolve_project_path(config["inputs"]["two_stage_summary_csv"])
    classifier_weights = resolve_project_path(config["classifier"]["weights_file"])
    paths = [base_checkpoint, adaptive_csv, adaptive_summary, two_stage_summary, classifier_weights]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))
    if not classifier_weights.is_file() or classifier_weights.stat().st_size < 10 * 1024 * 1024:
        raise RuntimeError(f"Classifier weights missing from local cache: {classifier_weights}")

    validate_monotonic_gates(config, snrs)
    rows = read_csv(adaptive_csv)
    target_policy = str(config["source_policy"]["target"])
    target_rows = select_policy_rows(config, rows, target_policy)
    eval_splits = [str(item) for item in config["splits"]["eval_splits"]]
    train_split = str(config["splits"]["train_split"])
    split_counts = {split: 0 for split in eval_splits}
    for row in target_rows:
        split = str(row["split"])
        if split in split_counts:
            split_counts[split] += 1
        for path_key in ["original", "m0_reconstruction"]:
            path = resolve_project_path(row[path_key])
            if not path.exists():
                raise FileNotFoundError(f"Missing image path from adaptive table: {path}")
    if split_counts.get(train_split, 0) == 0:
        raise RuntimeError(f"No training rows found for split={train_split} policy={target_policy}")
    absent = [split for split, count in split_counts.items() if count == 0]
    if absent:
        raise RuntimeError(f"Missing eval split rows: {absent}")
    return {
        "paths": {
            "base_refiner_checkpoint": project_relative(base_checkpoint),
            "adaptive_per_sample_csv": project_relative(adaptive_csv),
            "adaptive_summary_csv": project_relative(adaptive_summary),
            "two_stage_summary_csv": project_relative(two_stage_summary),
            "classifier_weights": project_relative(classifier_weights),
        },
        "target_policy": target_policy,
        "split_counts": split_counts,
        "train_split": train_split,
        "snrs": snrs,
        "residual_gates": validate_monotonic_gates(config, snrs),
        "proxy_environment_present": proxy_environment_present(),
    }


class AlphaPolicyDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], snr_norm_max: float) -> None:
        self.rows = sorted(rows, key=lambda row: (str(row["split"]), float(row["snr_db"]), str(row["sample"])))
        self.snr_norm_max = float(snr_norm_max)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        m0 = load_rgb_tensor(resolve_project_path(row["m0_reconstruction"]))
        target = load_rgb_tensor(resolve_project_path(row["original"]))
        snr = float(row["snr_db"])
        return {
            "m0": m0,
            "target": target,
            "snr_db": torch.tensor(snr, dtype=torch.float32),
            "snr_norm": torch.tensor(snr / self.snr_norm_max, dtype=torch.float32),
            "target_alpha_index": torch.tensor(int(row["target_alpha_index"]), dtype=torch.long),
            "target_alpha": torch.tensor(float(row["target_alpha"]), dtype=torch.float32),
            "split": str(row["split"]),
            "sample": str(row["sample"]),
            "original": str(row["original"]),
            "m0_reconstruction": str(row["m0_reconstruction"]),
        }


class AlphaHeadResidualRefiner(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config["model"]
        base_channels = int(model_cfg["base_channels"])
        self.head = nn.Sequential(
            nn.Conv2d(4, base_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(int(model_cfg["num_blocks"]))])
        self.tail = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)
        hidden = int(model_cfg.get("alpha_head_hidden", base_channels))
        dropout = float(model_cfg.get("alpha_head_dropout", 0.0))
        self.alpha_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels, hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, len(config["alphas"])),
        )

    def set_refiner_frozen(self, frozen: bool) -> None:
        for module in [self.head, self.body, self.tail]:
            for param in module.parameters():
                param.requires_grad = not frozen

    def forward(
        self,
        m0: torch.Tensor,
        snr_norm: torch.Tensor,
        residual_gate_value: torch.Tensor,
        detach_features: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = m0.shape
        snr_map = snr_norm.view(b, 1, 1, 1).expand(b, 1, h, w)
        features = self.body(self.head(torch.cat([m0, snr_map], dim=1)))
        residual = torch.tanh(self.tail(features))
        full_refined = (m0 + residual_gate_value.view(b, 1, 1, 1) * residual).clamp(0.0, 1.0)
        alpha_features = features.detach() if detach_features else features
        alpha_logits = self.alpha_head(alpha_features)
        return full_refined, alpha_logits


def build_model(config: dict[str, Any]) -> AlphaHeadResidualRefiner:
    return AlphaHeadResidualRefiner(config)


def load_base_refiner(model: AlphaHeadResidualRefiner, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = [key for key in missing if key.startswith("alpha_head.")]
    other_missing = [key for key in missing if not key.startswith("alpha_head.")]
    if other_missing:
        raise RuntimeError(f"Unexpected missing refiner keys: {other_missing}")
    other_unexpected = [key for key in unexpected if not key.startswith("alpha_head.")]
    if other_unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {other_unexpected}")
    return {
        "checkpoint_epoch": checkpoint.get("epoch", ""),
        "allowed_missing_alpha_head_keys": allowed_missing,
        "unexpected_alpha_head_keys": [key for key in unexpected if key.startswith("alpha_head.")],
    }


def train_one_epoch(
    model: AlphaHeadResidualRefiner,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    alphas_tensor: torch.Tensor,
    class_weights: torch.Tensor | None,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    freeze_refiner = bool(config["model"].get("freeze_refiner", True))
    losses: list[float] = []
    ce_losses: list[float] = []
    soft_mse_losses: list[float] = []
    accuracies: list[float] = []
    for batch in loader:
        m0 = batch["m0"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        snr_db = batch["snr_db"].to(device, non_blocking=True)
        snr_norm = batch["snr_norm"].to(device, non_blocking=True)
        target_index = batch["target_alpha_index"].to(device, non_blocking=True)
        gate = gate_tensor(config, snr_db, device)

        optimizer.zero_grad(set_to_none=True)
        full_refined, logits = model(m0, snr_norm, gate, detach_features=freeze_refiner)
        ce_loss = F.cross_entropy(logits, target_index, weight=class_weights)
        probs = torch.softmax(logits, dim=-1)
        soft_alpha = torch.matmul(probs, alphas_tensor).view(-1, 1, 1, 1)
        soft_refined = (m0 + soft_alpha * (full_refined.detach() - m0)).clamp(0.0, 1.0)
        soft_mse = F.mse_loss(soft_refined, target)
        soft_l1 = F.l1_loss(soft_refined, target)
        loss = (
            float(config["training"]["ce_weight"]) * ce_loss
            + float(config["training"].get("soft_mse_weight", 0.0)) * soft_mse
            + float(config["training"].get("soft_l1_weight", 0.0)) * soft_l1
        )
        loss.backward()
        grad_clip = float(config["training"].get("grad_clip_norm", 0.0))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        pred = torch.argmax(logits.detach(), dim=-1)
        losses.append(float(loss.detach().cpu()))
        ce_losses.append(float(ce_loss.detach().cpu()))
        soft_mse_losses.append(float(soft_mse.detach().cpu()))
        accuracies.append(float((pred == target_index).float().mean().detach().cpu()))
    return {
        "loss": float(mean(losses) or 0.0),
        "ce_loss": float(mean(ce_losses) or 0.0),
        "soft_mse": float(mean(soft_mse_losses) or 0.0),
        "alpha_accuracy": float(mean(accuracies) or 0.0),
    }


@torch.no_grad()
def quick_eval(
    model: AlphaHeadResidualRefiner,
    loader: DataLoader,
    config: dict[str, Any],
    class_weights: torch.Tensor | None,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    accuracies: list[float] = []
    for batch in loader:
        m0 = batch["m0"].to(device, non_blocking=True)
        snr_db = batch["snr_db"].to(device, non_blocking=True)
        snr_norm = batch["snr_norm"].to(device, non_blocking=True)
        target_index = batch["target_alpha_index"].to(device, non_blocking=True)
        gate = gate_tensor(config, snr_db, device)
        _, logits = model(m0, snr_norm, gate, detach_features=True)
        ce_loss = F.cross_entropy(logits, target_index, weight=class_weights)
        pred = torch.argmax(logits, dim=-1)
        losses.append(float(ce_loss.detach().cpu()))
        accuracies.append(float((pred == target_index).float().mean().detach().cpu()))
    return {
        "eval_ce_loss": float(mean(losses) or 0.0),
        "eval_alpha_accuracy": float(mean(accuracies) or 0.0),
    }


def image_metrics(reference: torch.Tensor, candidate: torch.Tensor, device: torch.device) -> dict[str, float]:
    reference = reference.to(device)
    candidate = candidate.to(device)
    with torch.no_grad():
        mse_values = F.mse_loss(candidate, reference, reduction="none").flatten(start_dim=1).mean(dim=1)
        return {
            "mse": float(mean(mse_values.detach().cpu().tolist()) or 0.0),
            "psnr_db": float(mean(psnr_per_sample(candidate, reference).detach().cpu().tolist()) or 0.0),
            "ssim": float(mean(ssim_per_sample(candidate, reference).detach().cpu().tolist()) or 0.0),
            "ms_ssim": float(mean(ms_ssim_per_sample(candidate, reference).detach().cpu().tolist()) or 0.0),
        }


@torch.no_grad()
def render_split_snr(
    model: AlphaHeadResidualRefiner,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    split: str,
    snr: float,
    output_dir: Path,
    alphas_tensor: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    split_rows = [row for row in rows if str(row["split"]) == split and abs(float(row["snr_db"]) - snr) < 1e-9]
    split_rows = sorted(split_rows, key=lambda row: str(row["sample"]))
    if not split_rows:
        raise RuntimeError(f"No rows for split={split} snr={snr}")

    full_dir = output_dir / "exports" / split.replace(" ", "_") / snr_name(snr) / "full_refined"
    alpha_dir = output_dir / "exports" / split.replace(" ", "_") / snr_name(snr) / "alpha_predicted"
    full_dir.mkdir(parents=True, exist_ok=True)
    alpha_dir.mkdir(parents=True, exist_ok=True)
    batch_size = int(config["evaluation"].get("image_batch_size", config["training"]["batch_size"]))
    full_paths: list[Path] = []
    alpha_paths: list[Path] = []
    predicted_indices: list[int] = []
    predicted_alphas: list[float] = []
    elapsed = 0.0
    for start in range(0, len(split_rows), batch_size):
        batch_rows = split_rows[start : start + batch_size]
        m0 = torch.stack([load_rgb_tensor(resolve_project_path(row["m0_reconstruction"])) for row in batch_rows]).to(device)
        snr_db = torch.full((len(batch_rows),), float(snr), dtype=torch.float32, device=device)
        snr_norm = snr_db / float(config["model"]["snr_norm_max"])
        gate = gate_tensor(config, snr_db, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        begin = time.perf_counter()
        full_refined, logits = model(m0, snr_norm, gate, detach_features=True)
        probs = torch.softmax(logits, dim=-1)
        pred_index = torch.argmax(probs, dim=-1)
        pred_alpha = alphas_tensor[pred_index].view(-1, 1, 1, 1)
        alpha_refined = (m0 + pred_alpha * (full_refined - m0)).clamp(0.0, 1.0)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed += time.perf_counter() - begin
        for row, full_img, alpha_img, idx in zip(batch_rows, full_refined.cpu(), alpha_refined.cpu(), pred_index.cpu()):
            full_path = full_dir / str(row["sample"])
            alpha_path = alpha_dir / str(row["sample"])
            save_image(full_img, full_path)
            save_image(alpha_img, alpha_path)
            full_paths.append(full_path)
            alpha_paths.append(alpha_path)
            predicted_indices.append(int(idx))
            predicted_alphas.append(float(config["alphas"][int(idx)]))
    return {
        "rows": split_rows,
        "full_paths": full_paths,
        "alpha_paths": alpha_paths,
        "predicted_indices": predicted_indices,
        "predicted_alphas": predicted_alphas,
        "refiner_time_ms_per_image": 1000.0 * elapsed / max(1, len(split_rows)),
    }


def policy_metric_row(
    split: str,
    snr: str | float,
    policy: str,
    rows: list[dict[str, Any]],
    reference: torch.Tensor,
    m0: torch.Tensor,
    candidate: torch.Tensor,
    final: torch.Tensor,
    device: torch.device,
    target_accuracy: float | None = None,
    predicted_alphas: list[float] | None = None,
) -> dict[str, Any]:
    m0_metrics = image_metrics(reference, m0, device)
    final_metrics = image_metrics(reference, final, device)
    candidate_metrics = image_metrics(reference, candidate, device)
    accepted = [bool(row.get("accepted", False)) for row in rows]
    m0_match = [bool(row["m0_matches_original_top1"]) for row in rows]
    cand_match = [bool(row["candidate_matches_original_top1"]) for row in rows]
    final_match = [bool(row["final_matches_original_top1"]) for row in rows]
    accepted_new_error = [bool(row["accepted_new_error"]) for row in rows]
    repair = [bool(row["accepted_repair"]) for row in rows]
    missed_repair = [bool(row["missed_repair"]) for row in rows]
    out = {
        "split": split,
        "snr_db": snr,
        "policy": policy,
        "num_images": len(rows),
        "accept_rate": rate(accepted),
        "fallback_rate": 1.0 - rate(accepted),
        "m0_failure_rate": 1.0 - rate(m0_match),
        "candidate_failure_rate": 1.0 - rate(cand_match),
        "final_failure_rate": 1.0 - rate(final_match),
        "delta_final_failure_vs_m0": (1.0 - rate(final_match)) - (1.0 - rate(m0_match)),
        "repair_count": int(sum(repair)),
        "accepted_new_error_count": int(sum(accepted_new_error)),
        "missed_repair_count": int(sum(missed_repair)),
        "candidate_psnr_db": candidate_metrics["psnr_db"],
        "final_psnr_db": final_metrics["psnr_db"],
        "m0_psnr_db": m0_metrics["psnr_db"],
        "delta_psnr_vs_m0_db": final_metrics["psnr_db"] - m0_metrics["psnr_db"],
        "candidate_ssim": candidate_metrics["ssim"],
        "final_ssim": final_metrics["ssim"],
        "candidate_ms_ssim": candidate_metrics["ms_ssim"],
        "final_ms_ssim": final_metrics["ms_ssim"],
    }
    if target_accuracy is not None:
        out["target_alpha_accuracy"] = target_accuracy
    if predicted_alphas:
        out["mean_predicted_alpha"] = float(mean(predicted_alphas) or 0.0)
    return out


def aggregate_summary(rows: list[dict[str, Any]], split: str, policy: str) -> dict[str, Any]:
    subset = [row for row in rows if row["split"] == split and row["policy"] == policy and row["snr_db"] != "all"]
    total = sum(int(row["num_images"]) for row in subset)
    if total == 0:
        raise RuntimeError(f"No summary rows for split={split} policy={policy}")

    def weighted(key: str) -> float:
        return float(sum(float(row[key]) * int(row["num_images"]) for row in subset) / total)

    return {
        "split": split,
        "snr_db": "all",
        "policy": policy,
        "num_images": total,
        "accept_rate": weighted("accept_rate"),
        "fallback_rate": weighted("fallback_rate"),
        "m0_failure_rate": weighted("m0_failure_rate"),
        "candidate_failure_rate": weighted("candidate_failure_rate"),
        "final_failure_rate": weighted("final_failure_rate"),
        "delta_final_failure_vs_m0": weighted("delta_final_failure_vs_m0"),
        "repair_count": int(sum(int(row["repair_count"]) for row in subset)),
        "accepted_new_error_count": int(sum(int(row["accepted_new_error_count"]) for row in subset)),
        "missed_repair_count": int(sum(int(row["missed_repair_count"]) for row in subset)),
        "candidate_psnr_db": weighted("candidate_psnr_db"),
        "final_psnr_db": weighted("final_psnr_db"),
        "m0_psnr_db": weighted("m0_psnr_db"),
        "delta_psnr_vs_m0_db": weighted("delta_psnr_vs_m0_db"),
        "candidate_ssim": weighted("candidate_ssim"),
        "final_ssim": weighted("final_ssim"),
        "candidate_ms_ssim": weighted("candidate_ms_ssim"),
        "final_ms_ssim": weighted("final_ms_ssim"),
        "target_alpha_accuracy": weighted("target_alpha_accuracy") if "target_alpha_accuracy" in subset[0] else "",
        "mean_predicted_alpha": weighted("mean_predicted_alpha") if "mean_predicted_alpha" in subset[0] else "",
    }


def make_plot(summary_rows: list[dict[str, Any]], path: Path) -> None:
    all_rows = [row for row in summary_rows if row["snr_db"] == "all"]
    splits = ["validation", "held-out", "test-like"]
    policy = "alpha_head_predicted_top1_fallback"
    psnr = []
    new_errors = []
    for split in splits:
        row = next(row for row in all_rows if row["split"] == split and row["policy"] == policy)
        psnr.append(float(row["delta_psnr_vs_m0_db"]))
        new_errors.append(float(row["accepted_new_error_count"]))
    fig, ax1 = plt.subplots(figsize=(7, 4))
    x = list(range(len(splits)))
    ax1.bar(x, psnr, color="#4C78A8", label="Delta PSNR")
    ax1.set_ylabel("Delta PSNR vs M0 (dB)")
    ax1.set_xticks(x, splits)
    ax1.axhline(0, color="#333333", linewidth=0.8)
    ax2 = ax1.twinx()
    ax2.plot(x, new_errors, color="#D54E4E", marker="o", label="New errors")
    ax2.set_ylabel("Accepted new errors")
    ax1.set_title("Alpha-head residual refiner pilot")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_report(summary_rows: list[dict[str, Any]], config: dict[str, Any]) -> str:
    all_rows = [row for row in summary_rows if row["snr_db"] == "all"]
    policies = [
        "full_strength_top1_fallback",
        "alpha_head_predicted_top1_fallback",
    ]
    lines = [
        "# Alpha-Head Residual Refiner Pilot",
        "",
        "This pilot attaches a learned alpha head to the EXP-S4-006 residual refiner.",
        "The residual refiner is frozen by default; only the alpha head is trained on validation pseudo targets from adaptive alpha.",
        "No diffusion, LPIPS weight loading, or external download is used.",
        f"Class weighting: `{config['training'].get('class_weighting', 'none')}`.",
        "",
        "## Bottom Line",
        "",
    ]
    for split in ["validation", "held-out", "test-like"]:
        row = next(row for row in all_rows if row["split"] == split and row["policy"] == "alpha_head_predicted_top1_fallback")
        lines.append(
            f"- {split}: alpha-head PSNR delta `{signed(row['delta_psnr_vs_m0_db'])}` dB, "
            f"target-alpha accuracy `{fmt(row.get('target_alpha_accuracy'))}`, "
            f"accepted new errors `{row['accepted_new_error_count']}`."
        )
    lines.extend(
        [
            "",
            "## All-Split Summary",
            "",
            "| Split | Policy | Delta PSNR | Failure Delta | Accept | Target Acc | New Error | Missed Repair |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split in ["validation", "held-out", "test-like"]:
        for policy in policies:
            row = next(row for row in all_rows if row["split"] == split and row["policy"] == policy)
            lines.append(
                "| {split} | {policy} | {dpsnr} | {dfail} | {accept} | {acc} | {new} | {missed} |".format(
                    split=split,
                    policy=policy,
                    dpsnr=signed(row["delta_psnr_vs_m0_db"]),
                    dfail=signed(row["delta_final_failure_vs_m0"]),
                    accept=fmt(row["accept_rate"]),
                    acc=fmt(row.get("target_alpha_accuracy")) if policy.startswith("alpha_head") else "",
                    new=row["accepted_new_error_count"],
                    missed=row["missed_repair_count"],
                )
            )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- The alpha target is a pseudo target from `adaptive_max_top1_consistent_alpha`, not supervised semantic truth.",
            "- This is a training-side exploration pilot, not a new strongest M3 claim.",
            "- The final decision still uses the frozen AlexNet top-1 consistency gate.",
            "- LPIPS is omitted to avoid external weight loading; compare PSNR/SSIM/MS-SSIM and semantic counts only.",
            "",
            "## Files",
            "",
            f"- Summary CSV: `{config['outputs']['output_dir']}/summary.csv`",
            f"- Per-sample CSV: `{config['outputs']['output_dir']}/per_sample.csv`",
            f"- Training history: `{config['outputs']['output_dir']}/train_history.csv`",
            f"- Metadata: `{config['outputs']['output_dir']}/metadata.json`",
        ]
    )
    return "\n".join(lines)


@torch.no_grad()
def evaluate(
    model: AlphaHeadResidualRefiner,
    target_rows: list[dict[str, Any]],
    config: dict[str, Any],
    output_dir: Path,
    classifier_model: torch.nn.Module,
    classifier_preprocess,
    categories: list[str],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    alphas_tensor = torch.tensor([float(item) for item in config["alphas"]], dtype=torch.float32, device=device)
    cls_batch = int(config["classifier"]["batch_size"])
    topk = int(config["classifier"]["topk"])
    for split in config["splits"]["eval_splits"]:
        for snr in parse_snrs(config):
            render = render_split_snr(model, target_rows, config, str(split), float(snr), output_dir, alphas_tensor, device)
            rows = render["rows"]
            original_paths = [resolve_project_path(row["original"]) for row in rows]
            m0_paths = [resolve_project_path(row["m0_reconstruction"]) for row in rows]
            full_paths = render["full_paths"]
            alpha_paths = render["alpha_paths"]
            original_preds, t_original = classify_paths(
                classifier_model, classifier_preprocess, original_paths, cls_batch, topk, device
            )
            m0_preds, t_m0 = classify_paths(classifier_model, classifier_preprocess, m0_paths, cls_batch, topk, device)
            full_preds, t_full = classify_paths(classifier_model, classifier_preprocess, full_paths, cls_batch, topk, device)
            alpha_preds, t_alpha = classify_paths(
                classifier_model, classifier_preprocess, alpha_paths, cls_batch, topk, device
            )

            final_alpha_dir = output_dir / "exports" / str(split).replace(" ", "_") / snr_name(snr) / "final_alpha_head"
            final_full_dir = output_dir / "exports" / str(split).replace(" ", "_") / snr_name(snr) / "final_full_strength"
            final_alpha_dir.mkdir(parents=True, exist_ok=True)
            final_full_dir.mkdir(parents=True, exist_ok=True)
            references: list[torch.Tensor] = []
            m0_tensors: list[torch.Tensor] = []
            alpha_tensors: list[torch.Tensor] = []
            full_tensors: list[torch.Tensor] = []
            final_alpha_tensors: list[torch.Tensor] = []
            final_full_tensors: list[torch.Tensor] = []
            alpha_eval_rows: list[dict[str, Any]] = []
            full_eval_rows: list[dict[str, Any]] = []

            for idx, row in enumerate(rows):
                original_top1 = original_preds[idx]["top_indices"][0]
                m0_top1 = m0_preds[idx]["top_indices"][0]
                alpha_top1 = alpha_preds[idx]["top_indices"][0]
                full_top1 = full_preds[idx]["top_indices"][0]
                m0_matches = m0_top1 == original_top1
                alpha_matches = alpha_top1 == original_top1
                full_matches = full_top1 == original_top1
                alpha_accept = alpha_top1 == m0_top1
                full_accept = full_top1 == m0_top1
                final_alpha_source = alpha_paths[idx] if alpha_accept else m0_paths[idx]
                final_full_source = full_paths[idx] if full_accept else m0_paths[idx]
                final_alpha_path = final_alpha_dir / str(row["sample"])
                final_full_path = final_full_dir / str(row["sample"])
                shutil.copy2(final_alpha_source, final_alpha_path)
                shutil.copy2(final_full_source, final_full_path)
                predicted_alpha = float(render["predicted_alphas"][idx])
                target_alpha = float(row["target_alpha"])
                alpha_common = {
                    "split": split,
                    "snr_db": float(snr),
                    "sample": row["sample"],
                    "target_alpha": target_alpha,
                    "predicted_alpha": predicted_alpha,
                    "target_alpha_index": int(row["target_alpha_index"]),
                    "predicted_alpha_index": int(render["predicted_indices"][idx]),
                    "target_alpha_correct": int(render["predicted_indices"][idx]) == int(row["target_alpha_index"]),
                    "original": project_relative(original_paths[idx]),
                    "m0_reconstruction": project_relative(m0_paths[idx]),
                    "full_refined": project_relative(full_paths[idx]),
                    "alpha_refined": project_relative(alpha_paths[idx]),
                    "alpha_head_final": project_relative(final_alpha_path),
                    "full_strength_final": project_relative(final_full_path),
                    "original_top1_index": original_top1,
                    "original_top1_label": label_for(categories, original_top1),
                    "original_top1_prob": original_preds[idx]["top_probs"][0],
                    "m0_top1_index": m0_top1,
                    "m0_top1_label": label_for(categories, m0_top1),
                    "m0_top1_prob": m0_preds[idx]["top_probs"][0],
                    "alpha_top1_index": alpha_top1,
                    "alpha_top1_label": label_for(categories, alpha_top1),
                    "alpha_top1_prob": alpha_preds[idx]["top_probs"][0],
                    "full_top1_index": full_top1,
                    "full_top1_label": label_for(categories, full_top1),
                    "full_top1_prob": full_preds[idx]["top_probs"][0],
                    "m0_matches_original_top1": m0_matches,
                }
                alpha_row = {
                    **alpha_common,
                    "policy": "alpha_head_predicted_top1_fallback",
                    "candidate_matches_original_top1": alpha_matches,
                    "candidate_matches_m0_top1": alpha_top1 == m0_top1,
                    "accepted": alpha_accept,
                    "final_matches_original_top1": alpha_matches if alpha_accept else m0_matches,
                    "accepted_repair": alpha_accept and (not m0_matches) and alpha_matches,
                    "accepted_new_error": alpha_accept and m0_matches and (not alpha_matches),
                    "missed_repair": (not alpha_accept) and (not m0_matches) and alpha_matches,
                }
                full_row = {
                    **alpha_common,
                    "policy": "full_strength_top1_fallback",
                    "candidate_matches_original_top1": full_matches,
                    "candidate_matches_m0_top1": full_top1 == m0_top1,
                    "accepted": full_accept,
                    "final_matches_original_top1": full_matches if full_accept else m0_matches,
                    "accepted_repair": full_accept and (not m0_matches) and full_matches,
                    "accepted_new_error": full_accept and m0_matches and (not full_matches),
                    "missed_repair": (not full_accept) and (not m0_matches) and full_matches,
                }
                alpha_eval_rows.append(alpha_row)
                full_eval_rows.append(full_row)
                per_sample_rows.extend([alpha_row, full_row])
                references.append(load_rgb_tensor(original_paths[idx]))
                m0_tensors.append(load_rgb_tensor(m0_paths[idx]))
                alpha_tensors.append(load_rgb_tensor(alpha_paths[idx]))
                full_tensors.append(load_rgb_tensor(full_paths[idx]))
                final_alpha_tensors.append(load_rgb_tensor(final_alpha_path))
                final_full_tensors.append(load_rgb_tensor(final_full_path))

            reference_tensor = torch.stack(references)
            m0_tensor = torch.stack(m0_tensors)
            alpha_tensor = torch.stack(alpha_tensors)
            full_tensor = torch.stack(full_tensors)
            final_alpha_tensor = torch.stack(final_alpha_tensors)
            final_full_tensor = torch.stack(final_full_tensors)
            target_accuracy = rate([bool(row["target_alpha_correct"]) for row in alpha_eval_rows])
            summary_rows.append(
                policy_metric_row(
                    str(split),
                    float(snr),
                    "alpha_head_predicted_top1_fallback",
                    alpha_eval_rows,
                    reference_tensor,
                    m0_tensor,
                    alpha_tensor,
                    final_alpha_tensor,
                    device,
                    target_accuracy=target_accuracy,
                    predicted_alphas=render["predicted_alphas"],
                )
            )
            summary_rows.append(
                policy_metric_row(
                    str(split),
                    float(snr),
                    "full_strength_top1_fallback",
                    full_eval_rows,
                    reference_tensor,
                    m0_tensor,
                    full_tensor,
                    final_full_tensor,
                    device,
                )
            )

            sample_count = min(int(config["evaluation"]["sample_grid_count"]), len(rows))
            sample_dir = output_dir / "samples"
            sample_dir.mkdir(parents=True, exist_ok=True)
            save_image(
                torch.cat(
                    [
                        reference_tensor[:sample_count],
                        m0_tensor[:sample_count],
                        full_tensor[:sample_count],
                        alpha_tensor[:sample_count],
                        final_alpha_tensor[:sample_count],
                    ],
                    dim=0,
                ),
                sample_dir / f"{str(split).replace(' ', '_')}_{snr_name(snr)}_original_m0_full_alpha_final.png",
                nrow=sample_count,
            )

            summary_rows[-2]["refiner_time_ms_per_image"] = render["refiner_time_ms_per_image"]
            summary_rows[-2]["classification_time_ms_per_image"] = (
                1000.0 * (t_original + t_m0 + t_full + t_alpha) / max(1, 4 * len(rows))
            )
            summary_rows[-1]["refiner_time_ms_per_image"] = render["refiner_time_ms_per_image"]
            summary_rows[-1]["classification_time_ms_per_image"] = summary_rows[-2]["classification_time_ms_per_image"]

    for split in config["splits"]["eval_splits"]:
        for policy in ["alpha_head_predicted_top1_fallback", "full_strength_top1_fallback"]:
            summary_rows.append(aggregate_summary(summary_rows, str(split), policy))
    return summary_rows, per_sample_rows


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.epochs is not None:
        config["training"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        config["training"]["batch_size"] = int(args.batch_size)

    snrs = parse_snrs(config)
    manifest = validate_inputs(config, snrs)
    if args.dry_run:
        print(json.dumps({"status": "ok", "config": project_relative(config_path), **manifest}, indent=2))
        return

    output_dir = resolve_project_path(args.output_dir or config["outputs"]["output_dir"])
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists, refusing to overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    shutil.copy2(config_path, output_dir / "config.yaml")
    save_json(output_dir / "source_manifest.json", manifest)

    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    all_rows = select_policy_rows(
        config,
        read_csv(resolve_project_path(config["inputs"]["adaptive_per_sample_csv"])),
        str(config["source_policy"]["target"]),
    )
    train_rows = [row for row in all_rows if str(row["split"]) == str(config["splits"]["train_split"])]
    class_weights_list = class_weight_values(config, train_rows)
    train_dataset = AlphaPolicyDataset(train_rows, float(config["model"]["snr_norm_max"]))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )
    eval_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    model = build_model(config)
    base_info = load_base_refiner(model, resolve_project_path(config["inputs"]["base_refiner_checkpoint"]))
    model.set_refiner_frozen(bool(config["model"].get("freeze_refiner", True)))
    model.to(device)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    alphas_tensor = torch.tensor([float(item) for item in config["alphas"]], dtype=torch.float32, device=device)
    class_weights = (
        torch.tensor(class_weights_list, dtype=torch.float32, device=device) if class_weights_list is not None else None
    )
    history: list[dict[str, Any]] = []
    best_eval = float("inf")
    best_path = output_dir / "checkpoints" / "best.pt"
    latest_path = output_dir / "checkpoints" / "latest.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = int(config["training"]["epochs"])
    validate_every = int(config["training"].get("validation_every_epochs", 20))
    for epoch in range(epochs):
        train_stats = train_one_epoch(model, train_loader, optimizer, config, alphas_tensor, class_weights, device)
        row: dict[str, Any] = {"epoch": epoch, **train_stats}
        if (epoch + 1) % validate_every == 0 or epoch == epochs - 1:
            eval_stats = quick_eval(model, eval_loader, config, class_weights, device)
            row.update(eval_stats)
            if eval_stats["eval_ce_loss"] < best_eval:
                best_eval = eval_stats["eval_ce_loss"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config,
                        "eval_stats": eval_stats,
                        "base_info": base_info,
                    },
                    best_path,
                )
        history.append(row)
        print(json.dumps(row, indent=2))
    torch.save(
        {
            "epoch": epochs - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "base_info": base_info,
        },
        latest_path,
    )
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    write_csv(output_dir / "train_history.csv", history)

    classifier_model, classifier_preprocess, categories = load_classifier(config, device)
    summary_rows, per_sample_rows = evaluate(
        model=model,
        target_rows=all_rows,
        config=config,
        output_dir=output_dir,
        classifier_model=classifier_model,
        classifier_preprocess=classifier_preprocess,
        categories=categories,
        device=device,
    )
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "per_sample.csv", per_sample_rows)
    figure_path = output_dir / "figures" / "alpha_head_tradeoff.png"
    make_plot(summary_rows, figure_path)
    (output_dir / "REPORT.md").write_text(make_report(summary_rows, config), encoding="utf-8")

    import importlib.metadata as md
    import torchvision

    metadata = {
        "project_version": git_commit(),
        "git_dirty_state": git_dirty_state(),
        "repository_url": config.get("repository_url"),
        "config": project_relative(config_path),
        "run_command": " ".join(sys.argv),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "dataset": config["dataset"],
        "image_size": int(config["image_size"]),
        "channel": str(config["channel"]),
        "snrs": snrs,
        "cbr": float(config["cbr"]),
        "seed": seed,
        "inputs": config["inputs"],
        "splits": config["splits"],
        "model": config["model"],
        "training": config["training"],
        "classifier": config["classifier"],
        "base_info": base_info,
        "target_alpha_train_counts": alpha_class_counts(train_rows, len(config["alphas"])),
        "class_weights": class_weights_list or [],
        "proxy_environment_present": proxy_environment_present(),
        "download_note": "No model or data download is required; local AlexNet and EXP-S4-006 weights are used.",
        "package_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "pillow": md.version("pillow"),
            "pytorch-msssim": md.version("pytorch-msssim"),
        },
        "outputs": {
            "summary_csv": project_relative(output_dir / "summary.csv"),
            "per_sample_csv": project_relative(output_dir / "per_sample.csv"),
            "train_history_csv": project_relative(output_dir / "train_history.csv"),
            "report": project_relative(output_dir / "REPORT.md"),
            "figure": project_relative(figure_path),
        },
    }
    save_json(output_dir / "metadata.json", metadata)
    print(json.dumps({"output_dir": project_relative(output_dir), "summary_rows": len(summary_rows)}, indent=2))


if __name__ == "__main__":
    main()
