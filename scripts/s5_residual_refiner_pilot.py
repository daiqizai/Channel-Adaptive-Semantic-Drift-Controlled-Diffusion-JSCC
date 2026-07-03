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

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cadsd_jscc.metrics import ms_ssim_per_sample, psnr_per_sample, ssim_per_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate a small SNR-conditioned pixel residual refiner over existing M0 exports."
    )
    parser.add_argument("--config", default="configs/s5_residual_refiner_pilot_coco256_awgn.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--skip-lpips", action="store_true")
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


def parse_snrs(config: dict[str, Any]) -> list[float]:
    return [float(item) for item in config["snrs"]]


def snr_name(snr: float) -> str:
    if float(snr).is_integer():
        return f"snr_{int(snr):02d}db"
    return f"snr_{str(snr).replace('.', 'p')}db"


def snr_key(snr: float) -> str:
    if float(snr).is_integer():
        return str(int(snr))
    return str(snr)


def load_rgb_tensor(path: Path) -> torch.Tensor:
    return TF.to_tensor(Image.open(path).convert("RGB"))


def load_rgb_pil(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def rate(flags: list[bool]) -> float:
    if not flags:
        return 0.0
    return float(sum(flags) / len(flags))


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


def make_sample_names(start: int, count: int) -> list[str]:
    return [f"sample_{index:06d}.png" for index in range(start, start + count)]


def residual_gate(config: dict[str, Any], snr: float) -> float:
    gates = config["model"]["residual_gates"]
    key = snr_key(snr)
    if key not in gates:
        raise KeyError(f"Missing residual gate for SNR {key}")
    return float(gates[key])


def check_residual_gates(config: dict[str, Any], snrs: list[float]) -> dict[float, float]:
    gates = {snr: residual_gate(config, snr) for snr in snrs}
    ordered = sorted(gates)
    for left, right in zip(ordered, ordered[1:]):
        if gates[left] + 1e-12 < gates[right]:
            raise ValueError(
                f"Residual gate must not increase with SNR: {left} dB has {gates[left]}, "
                f"{right} dB has {gates[right]}"
            )
    return gates


def validate_inputs(config: dict[str, Any], snrs: list[float]) -> dict[str, Any]:
    original_dir = resolve_project_path(config["inputs"]["original_dir"])
    m0_export_dir = resolve_project_path(config["inputs"]["m0_export_dir"])
    checkpoint = resolve_project_path(config["inputs"]["checkpoint"])
    forbidden_checkpoint = resolve_project_path(config["inputs"]["forbidden_checkpoint"])
    classifier_weights = resolve_project_path(config["classifier"]["weights_file"])
    for path in [original_dir, m0_export_dir, checkpoint, classifier_weights]:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")
    if checkpoint == forbidden_checkpoint:
        raise RuntimeError("Config points to forbidden latest.pt checkpoint.")

    split = config["split"]
    train_names = make_sample_names(int(split["train_sample_start"]), int(split["train_sample_count"]))
    eval_names = make_sample_names(int(split["eval_sample_start"]), int(split["eval_sample_count"]))
    overlap = sorted(set(train_names) & set(eval_names))
    if overlap:
        raise RuntimeError(f"Train/eval sample split overlaps: {overlap}")

    for name in train_names + eval_names:
        if not (original_dir / name).exists():
            raise FileNotFoundError(f"Original sample missing: {original_dir / name}")
    for snr in snrs:
        m0_dir = m0_export_dir / "exports" / snr_name(snr) / "reconstruction"
        if not m0_dir.exists():
            raise FileNotFoundError(f"M0 reconstruction directory missing: {m0_dir}")
        for name in train_names + eval_names:
            if not (m0_dir / name).exists():
                raise FileNotFoundError(f"M0 sample missing: {m0_dir / name}")

    gates = check_residual_gates(config, snrs)
    return {
        "train_names": train_names,
        "eval_names": eval_names,
        "residual_gates": {str(snr): gate for snr, gate in gates.items()},
    }


def paired_random_crop(m0: torch.Tensor, target: torch.Tensor, crop_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    _, height, width = m0.shape
    if crop_size <= 0 or crop_size >= min(height, width):
        return m0, target
    top = random.randint(0, height - crop_size)
    left = random.randint(0, width - crop_size)
    return (
        m0[:, top : top + crop_size, left : left + crop_size],
        target[:, top : top + crop_size, left : left + crop_size],
    )


class ResidualPairDataset(Dataset):
    def __init__(
        self,
        config: dict[str, Any],
        names: list[str],
        snrs: list[float],
        train: bool,
    ) -> None:
        self.original_dir = resolve_project_path(config["inputs"]["original_dir"])
        self.m0_export_dir = resolve_project_path(config["inputs"]["m0_export_dir"])
        self.snr_norm_max = float(config["model"]["snr_norm_max"])
        self.crop_size = int(config["training"].get("crop_size", 0)) if train else 0
        self.random_flip = bool(config["training"].get("random_flip", False)) if train else False
        self.items: list[tuple[float, str]] = [(float(snr), name) for snr in snrs for name in names]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        snr, name = self.items[index]
        m0_path = self.m0_export_dir / "exports" / snr_name(snr) / "reconstruction" / name
        original_path = self.original_dir / name
        m0 = load_rgb_tensor(m0_path)
        target = load_rgb_tensor(original_path)
        m0, target = paired_random_crop(m0, target, self.crop_size)
        if self.random_flip and random.random() < 0.5:
            m0 = torch.flip(m0, dims=[2])
            target = torch.flip(target, dims=[2])
        return {
            "m0": m0,
            "target": target,
            "snr_db": torch.tensor(float(snr), dtype=torch.float32),
            "snr_norm": torch.tensor(float(snr) / self.snr_norm_max, dtype=torch.float32),
            "name": name,
        }


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SNRConditionedResidualRefiner(nn.Module):
    def __init__(self, base_channels: int, num_blocks: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(4, base_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, m0: torch.Tensor, snr_norm: torch.Tensor, residual_gate_value: torch.Tensor) -> torch.Tensor:
        b, _, h, w = m0.shape
        snr_map = snr_norm.view(b, 1, 1, 1).expand(b, 1, h, w)
        residual = torch.tanh(self.tail(self.body(self.head(torch.cat([m0, snr_map], dim=1)))))
        gate = residual_gate_value.view(b, 1, 1, 1)
        return (m0 + gate * residual).clamp(0.0, 1.0)


def build_model(config: dict[str, Any]) -> SNRConditionedResidualRefiner:
    model_cfg = config["model"]
    return SNRConditionedResidualRefiner(
        base_channels=int(model_cfg["base_channels"]),
        num_blocks=int(model_cfg["num_blocks"]),
    )


def gate_tensor(config: dict[str, Any], snr_db: torch.Tensor, device: torch.device) -> torch.Tensor:
    gates = [residual_gate(config, float(item)) for item in snr_db.detach().cpu().tolist()]
    return torch.tensor(gates, dtype=torch.float32, device=device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    model.train()
    losses: list[float] = []
    mse_losses: list[float] = []
    l1_losses: list[float] = []
    for batch in loader:
        m0 = batch["m0"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        snr_db = batch["snr_db"].to(device, non_blocking=True)
        snr_norm = batch["snr_norm"].to(device, non_blocking=True)
        gate = gate_tensor(config, snr_db, device)

        optimizer.zero_grad(set_to_none=True)
        refined = model(m0, snr_norm, gate)
        mse_loss = F.mse_loss(refined, target)
        l1_loss = F.l1_loss(refined, target)
        loss = float(config["training"]["mse_weight"]) * mse_loss + float(config["training"]["l1_weight"]) * l1_loss
        loss.backward()
        grad_clip = float(config["training"].get("grad_clip_norm", 0.0))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        mse_losses.append(float(mse_loss.detach().cpu()))
        l1_losses.append(float(l1_loss.detach().cpu()))
    return {
        "loss": float(mean(losses) or 0.0),
        "mse_loss": float(mean(mse_losses) or 0.0),
        "l1_loss": float(mean(l1_losses) or 0.0),
    }


@torch.no_grad()
def quick_eval_loss(
    model: nn.Module,
    loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    mse_losses: list[float] = []
    psnrs: list[float] = []
    for batch in loader:
        m0 = batch["m0"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        snr_db = batch["snr_db"].to(device, non_blocking=True)
        snr_norm = batch["snr_norm"].to(device, non_blocking=True)
        gate = gate_tensor(config, snr_db, device)
        refined = model(m0, snr_norm, gate)
        mse_losses.append(float(F.mse_loss(refined, target).detach().cpu()))
        psnrs.extend(psnr_per_sample(refined, target).detach().cpu().tolist())
    return {
        "eval_mse": float(mean(mse_losses) or 0.0),
        "eval_psnr_db": float(mean(psnrs) or 0.0),
    }


def load_classifier(config: dict[str, Any], device: torch.device):
    weights_file = resolve_project_path(config["classifier"]["weights_file"])
    if not weights_file.is_file() or weights_file.stat().st_size < 10 * 1024 * 1024:
        raise RuntimeError(f"Classifier weights missing from local cache: {weights_file}")
    os.environ.setdefault("TORCH_HOME", str(resolve_project_path(config["classifier"]["cache_dir"])))
    import torchvision.models as models

    weights = getattr(models.AlexNet_Weights, str(config["classifier"]["weights"]))
    model = models.alexnet(weights=weights).to(device)
    model.eval()
    return model, weights.transforms(), list(weights.meta["categories"])


def try_load_lpips(device: torch.device, cache_root: Path):
    try:
        os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))
        import lpips

        model = lpips.LPIPS(net="alex", verbose=False).to(device)
        model.eval()
        return model, None
    except Exception as exc:  # noqa: BLE001 - optional metric should not abort the experiment.
        return None, f"{type(exc).__name__}: {exc}"


def compute_pair_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    lpips_model,
    device: torch.device,
) -> dict[str, float | None]:
    reference = reference.to(device)
    candidate = candidate.to(device)
    metrics: dict[str, float | None] = {}
    with torch.no_grad():
        mse_values = F.mse_loss(candidate, reference, reduction="none").flatten(start_dim=1).mean(dim=1)
        metrics["mse"] = mean(mse_values.detach().cpu().tolist())
        metrics["psnr_db"] = mean(psnr_per_sample(candidate, reference).detach().cpu().tolist())
        metrics["ssim"] = mean(ssim_per_sample(candidate, reference).detach().cpu().tolist())
        metrics["ms_ssim"] = mean(ms_ssim_per_sample(candidate, reference).detach().cpu().tolist())
        if lpips_model is not None:
            lpips_values = lpips_model(candidate * 2.0 - 1.0, reference * 2.0 - 1.0)
            metrics["lpips"] = mean(lpips_values.flatten().detach().cpu().tolist())
        else:
            metrics["lpips"] = None
    return metrics


def classify_paths(
    model: torch.nn.Module,
    preprocess,
    paths: list[Path],
    batch_size: int,
    topk: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], float]:
    outputs: list[dict[str, Any]] = []
    elapsed = 0.0
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = torch.stack([preprocess(load_rgb_pil(path)) for path in batch_paths]).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            begin = time.perf_counter()
            logits = model(images)
            probabilities = torch.softmax(logits.float(), dim=-1)
            values, indices = torch.topk(probabilities, k=topk, dim=-1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - begin
            for row_values, row_indices in zip(values.cpu(), indices.cpu()):
                outputs.append(
                    {
                        "top_indices": [int(item) for item in row_indices.tolist()],
                        "top_probs": [float(item) for item in row_values.tolist()],
                    }
                )
    return outputs, elapsed


def label_for(categories: list[str], index: int) -> str:
    if 0 <= index < len(categories):
        return categories[index]
    return f"class_{index}"


def semantic_summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        return {
            "num_images": 0,
            "accept_rate": 0.0,
            "reject_rate": 0.0,
            "m0_final_failure": 0.0,
            "refined_drift_origin": 0.0,
            "refined_refinement_drift": 0.0,
            "m3_final_failure": 0.0,
            "m3_prediction_consistency": 0.0,
            "m3_refinement_drift": 0.0,
            "false_accept_rate": 0.0,
            "false_reject_rate": 0.0,
        }
    accepted = [bool(row["detector_accept_refined"]) for row in rows]
    m0_match_origin = [bool(row["m0_matches_original_top1"]) for row in rows]
    refined_match_origin = [bool(row["refined_matches_original_top1"]) for row in rows]
    refined_match_m0 = [bool(row["refined_matches_m0_top1"]) for row in rows]
    m3_match_origin = [bool(row["m3_matches_original_top1"]) for row in rows]
    m3_match_m0 = [bool(row["m3_matches_m0_top1"]) for row in rows]
    false_accept = [
        bool(row["detector_accept_refined"]) and not bool(row["refined_matches_original_top1"]) for row in rows
    ]
    false_reject = [
        not bool(row["detector_accept_refined"]) and bool(row["refined_matches_original_top1"]) for row in rows
    ]
    m0_failure = 1.0 - rate(m0_match_origin)
    refined_failure = 1.0 - rate(refined_match_origin)
    m3_failure = 1.0 - rate(m3_match_origin)
    return {
        "num_images": len(rows),
        "accept_rate": rate(accepted),
        "reject_rate": 1.0 - rate(accepted),
        "m0_final_failure": m0_failure,
        "refined_drift_origin": refined_failure,
        "refined_refinement_drift": 1.0 - rate(refined_match_m0),
        "m3_final_failure": m3_failure,
        "m3_prediction_consistency": 1.0 - m3_failure,
        "m3_refinement_drift": 1.0 - rate(m3_match_m0),
        "m3_minus_refined_final_failure": m3_failure - refined_failure,
        "m3_minus_m0_final_failure": m3_failure - m0_failure,
        "false_accept_rate": rate(false_accept),
        "false_reject_rate": rate(false_reject),
        "false_accept_count": int(sum(false_accept)),
        "false_reject_count": int(sum(false_reject)),
    }


def subset_summaries(rows: list[dict[str, Any]], thresholds: list[float]) -> dict[str, dict[str, float | int]]:
    summaries = {"all": semantic_summary(rows)}
    for threshold in thresholds:
        key = f"original_conf_ge_{str(threshold).replace('.', 'p')}"
        subset = [row for row in rows if float(row["original_top1_prob"]) >= threshold]
        summaries[key] = semantic_summary(subset)
    return summaries


@torch.no_grad()
def refine_and_save_snr(
    model: nn.Module,
    config: dict[str, Any],
    snr: float,
    names: list[str],
    output_dir: Path,
    device: torch.device,
) -> tuple[list[Path], float]:
    model.eval()
    m0_dir = resolve_project_path(config["inputs"]["m0_export_dir"]) / "exports" / snr_name(snr) / "reconstruction"
    refined_dir = output_dir / "exports" / snr_name(snr) / "refined"
    refined_dir.mkdir(parents=True, exist_ok=False)
    refined_paths: list[Path] = []
    elapsed = 0.0
    batch_size = int(config["training"]["batch_size"])
    for start in range(0, len(names), batch_size):
        batch_names = names[start : start + batch_size]
        batch = torch.stack([load_rgb_tensor(m0_dir / name) for name in batch_names]).to(device)
        snr_db = torch.full((len(batch_names),), float(snr), dtype=torch.float32, device=device)
        snr_norm = snr_db / float(config["model"]["snr_norm_max"])
        gate = gate_tensor(config, snr_db, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        begin = time.perf_counter()
        refined = model(batch, snr_norm, gate)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed += time.perf_counter() - begin
        for name, image in zip(batch_names, refined.detach().cpu()):
            path = refined_dir / name
            save_image(image, path)
            refined_paths.append(path)
    return refined_paths, elapsed


def evaluate_snr(
    model: nn.Module,
    config: dict[str, Any],
    snr: float,
    names: list[str],
    output_dir: Path,
    classifier_model: torch.nn.Module,
    classifier_preprocess,
    categories: list[str],
    lpips_model,
    device: torch.device,
) -> dict[str, Any]:
    original_dir = resolve_project_path(config["inputs"]["original_dir"])
    m0_dir = resolve_project_path(config["inputs"]["m0_export_dir"]) / "exports" / snr_name(snr) / "reconstruction"
    refined_paths, elapsed = refine_and_save_snr(model, config, snr, names, output_dir, device)
    final_dir = output_dir / "exports" / snr_name(snr) / "final"
    final_dir.mkdir(parents=True, exist_ok=False)

    original_paths = [original_dir / name for name in names]
    m0_paths = [m0_dir / name for name in names]
    cls_batch = int(config["classifier"]["batch_size"])
    topk = int(config["classifier"]["topk"])
    original_preds, t_original = classify_paths(classifier_model, classifier_preprocess, original_paths, cls_batch, topk, device)
    m0_preds, t_m0 = classify_paths(classifier_model, classifier_preprocess, m0_paths, cls_batch, topk, device)
    refined_preds, t_refined = classify_paths(
        classifier_model, classifier_preprocess, refined_paths, cls_batch, topk, device
    )

    references: list[torch.Tensor] = []
    m0_tensors: list[torch.Tensor] = []
    refined_tensors: list[torch.Tensor] = []
    final_tensors: list[torch.Tensor] = []
    per_sample: list[dict[str, Any]] = []
    for idx, name in enumerate(names):
        original_top1 = original_preds[idx]["top_indices"][0]
        m0_top1 = m0_preds[idx]["top_indices"][0]
        refined_top1 = refined_preds[idx]["top_indices"][0]
        accept = refined_top1 == m0_top1
        final_source = refined_paths[idx] if accept else m0_paths[idx]
        final_path = final_dir / name
        shutil.copy2(final_source, final_path)
        m0_matches_origin = m0_top1 == original_top1
        refined_matches_origin = refined_top1 == original_top1
        row = {
            "snr_db": float(snr),
            "residual_gate": residual_gate(config, snr),
            "sample": name,
            "original": project_relative(original_paths[idx]),
            "m0_reconstruction": project_relative(m0_paths[idx]),
            "refined": project_relative(refined_paths[idx]),
            "m3_final": project_relative(final_path),
            "detector": str(config["failure_handling"]["detector"]),
            "detector_accept_refined": accept,
            "m3_output_kind": "accepted_refined" if accept else "fallback_m0",
            "original_top1_index": original_top1,
            "original_top1_label": label_for(categories, original_top1),
            "original_top1_prob": original_preds[idx]["top_probs"][0],
            "m0_top1_index": m0_top1,
            "m0_top1_label": label_for(categories, m0_top1),
            "m0_top1_prob": m0_preds[idx]["top_probs"][0],
            "refined_top1_index": refined_top1,
            "refined_top1_label": label_for(categories, refined_top1),
            "refined_top1_prob": refined_preds[idx]["top_probs"][0],
            "m3_top1_index": refined_top1 if accept else m0_top1,
            "m3_top1_label": label_for(categories, refined_top1 if accept else m0_top1),
            "m3_top1_prob": refined_preds[idx]["top_probs"][0] if accept else m0_preds[idx]["top_probs"][0],
            "m0_matches_original_top1": m0_matches_origin,
            "refined_matches_original_top1": refined_matches_origin,
            "refined_matches_m0_top1": refined_top1 == m0_top1,
            "m3_matches_original_top1": refined_matches_origin if accept else m0_matches_origin,
            "m3_matches_m0_top1": refined_top1 == m0_top1 if accept else True,
            "false_accept": accept and not refined_matches_origin,
            "false_reject": (not accept) and refined_matches_origin,
        }
        per_sample.append(row)
        references.append(load_rgb_tensor(original_paths[idx]))
        m0_tensors.append(load_rgb_tensor(m0_paths[idx]))
        refined_tensors.append(load_rgb_tensor(refined_paths[idx]))
        final_tensors.append(load_rgb_tensor(final_path))

    reference = torch.stack(references)
    m0 = torch.stack(m0_tensors)
    refined = torch.stack(refined_tensors)
    final = torch.stack(final_tensors)
    sample_count = min(int(config["evaluation"]["sample_grid_count"]), len(names))
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_grid = sample_dir / f"{snr_name(snr)}_original_m0_refined_m3final.png"
    save_image(
        torch.cat([reference[:sample_count], m0[:sample_count], refined[:sample_count], final[:sample_count]], dim=0),
        sample_grid,
        nrow=sample_count,
    )
    thresholds = [float(item) for item in config["evaluation"].get("pseudo_clean_conf_thresholds", [])]
    return {
        "snr_db": float(snr),
        "residual_gate": residual_gate(config, snr),
        "num_images": len(names),
        "sample_names": names,
        "refined_dir": project_relative(output_dir / "exports" / snr_name(snr) / "refined"),
        "final_dir": project_relative(final_dir),
        "sample_grid": project_relative(sample_grid),
        "image_quality": {
            "m0_reconstruction_vs_original": compute_pair_metrics(reference, m0, lpips_model, device),
            "refined_vs_original": compute_pair_metrics(reference, refined, lpips_model, device),
            "m3_final_vs_original": compute_pair_metrics(reference, final, lpips_model, device),
            "refined_vs_m0_reconstruction": compute_pair_metrics(m0, refined, None, device),
        },
        "semantic_reliability": subset_summaries(per_sample, thresholds),
        "refiner_time_ms_per_image": 1000.0 * elapsed / max(1, len(names)),
        "classification_time_ms_per_image": 1000.0 * (t_original + t_m0 + t_refined) / max(1, 3 * len(names)),
        "per_sample": per_sample,
    }


def serialize_csv_value(value: Any) -> Any:
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_csv_value(value) for key, value in row.items()})


def fmt(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return float(value - baseline)


def aggregate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        quality = result["image_quality"]
        semantic = result["semantic_reliability"]["all"]
        m0_quality = quality["m0_reconstruction_vs_original"]
        refined_quality = quality["refined_vs_original"]
        m3_quality = quality["m3_final_vs_original"]
        m0_psnr = safe_float(m0_quality["psnr_db"])
        refined_psnr = safe_float(refined_quality["psnr_db"])
        m3_psnr = safe_float(m3_quality["psnr_db"])
        m0_lpips = safe_float(m0_quality["lpips"])
        refined_lpips = safe_float(refined_quality["lpips"])
        m3_lpips = safe_float(m3_quality["lpips"])
        rows.append(
            {
                "snr_db": float(result["snr_db"]),
                "residual_gate": float(result["residual_gate"]),
                "num_images": int(result["num_images"]),
                "m0_psnr_db": m0_psnr,
                "refined_psnr_db": refined_psnr,
                "m3_psnr_db": m3_psnr,
                "refined_delta_psnr_vs_m0_db": delta(refined_psnr, m0_psnr),
                "m3_delta_psnr_vs_m0_db": delta(m3_psnr, m0_psnr),
                "m0_lpips": m0_lpips,
                "refined_lpips": refined_lpips,
                "m3_lpips": m3_lpips,
                "refined_delta_lpips_vs_m0": delta(refined_lpips, m0_lpips),
                "m3_delta_lpips_vs_m0": delta(m3_lpips, m0_lpips),
                "m0_final_failure": float(semantic["m0_final_failure"]),
                "refined_failure": float(semantic["refined_drift_origin"]),
                "m3_final_failure": float(semantic["m3_final_failure"]),
                "refined_refinement_drift": float(semantic["refined_refinement_drift"]),
                "accept_rate": float(semantic["accept_rate"]),
                "false_accept_rate": float(semantic["false_accept_rate"]),
                "false_reject_rate": float(semantic["false_reject_rate"]),
                "refiner_time_ms_per_image": float(result["refiner_time_ms_per_image"]),
            }
        )
    return rows


def make_report(results: list[dict[str, Any]], config: dict[str, Any]) -> str:
    rows = aggregate_results(results)
    lines = [
        "# SNR-Conditioned Pixel Residual Refiner Pilot",
        "",
        "This S5 pilot bypasses Stable Diffusion and its VAE. It trains a small pixel-domain residual refiner on existing M0 exports.",
        "",
        "## Split",
        "",
        f"- Train samples: `sample_{int(config['split']['train_sample_start']):06d}.png` to `sample_{int(config['split']['train_sample_start']) + int(config['split']['train_sample_count']) - 1:06d}.png`",
        f"- Eval samples: `sample_{int(config['split']['eval_sample_start']):06d}.png` to `sample_{int(config['split']['eval_sample_start']) + int(config['split']['eval_sample_count']) - 1:06d}.png`",
        "- This is a small validation pilot, not a final M2/M3 result.",
        "",
        "## Main Table",
        "",
        "| SNR(dB) | Gate | M0 PSNR | Refined PSNR | Delta | M0 LPIPS | Refined LPIPS | Delta | M0 failure | Refined failure | M3 failure | Accept |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {snr:g} | {gate} | {m0_psnr} | {ref_psnr} | {psnr_delta} | {m0_lpips} | {ref_lpips} | {lpips_delta} | {m0_fail} | {ref_fail} | {m3_fail} | {accept} |".format(
                snr=float(row["snr_db"]),
                gate=fmt(float(row["residual_gate"])),
                m0_psnr=fmt(row["m0_psnr_db"]),
                ref_psnr=fmt(row["refined_psnr_db"]),
                psnr_delta=fmt(row["refined_delta_psnr_vs_m0_db"]),
                m0_lpips=fmt(row["m0_lpips"]),
                ref_lpips=fmt(row["refined_lpips"]),
                lpips_delta=fmt(row["refined_delta_lpips_vs_m0"]),
                m0_fail=fmt(float(row["m0_final_failure"])),
                ref_fail=fmt(float(row["refined_failure"])),
                m3_fail=fmt(float(row["m3_final_failure"])),
                accept=fmt(float(row["accept_rate"])),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "- This is a pixel-domain restoration pilot, not a diffusion success claim.",
            "- It is useful only if it improves or preserves quality without increasing pseudo semantic failure.",
            "- COCO ImageNet pseudo-label consistency remains an auxiliary semantic diagnostic, not clean-correct GT classification.",
            "",
        ]
    )
    return "\n".join(lines)


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
        print(json.dumps({"status": "ok", "snrs": snrs, **manifest}, indent=2))
        return

    output_dir = resolve_project_path(args.output_dir or config["outputs"]["output_dir"])
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists, refusing to overwrite: {output_dir}")
    output_dir.mkdir(parents=True)
    shutil.copy2(config_path, output_dir / "config.yaml")
    save_json(output_dir / "source_manifest.json", manifest)

    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    train_dataset = ResidualPairDataset(config, manifest["train_names"], snrs, train=True)
    eval_dataset = ResidualPairDataset(config, manifest["eval_names"], snrs, train=False)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    history: list[dict[str, Any]] = []
    best_eval_mse = float("inf")
    best_state_path = output_dir / "checkpoints" / "best.pt"
    latest_state_path = output_dir / "checkpoints" / "latest.pt"
    best_state_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = int(config["training"]["epochs"])
    validate_every = int(config["training"].get("validation_every_epochs", 10))
    for epoch in range(epochs):
        train_stats = train_one_epoch(model, train_loader, optimizer, config, device)
        row: dict[str, Any] = {"epoch": epoch, **train_stats}
        if (epoch + 1) % validate_every == 0 or epoch == epochs - 1:
            eval_stats = quick_eval_loss(model, eval_loader, config, device)
            row.update(eval_stats)
            if eval_stats["eval_mse"] < best_eval_mse:
                best_eval_mse = eval_stats["eval_mse"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config,
                        "eval_stats": eval_stats,
                    },
                    best_state_path,
                )
        history.append(row)
        print(json.dumps(row, indent=2))
    torch.save(
        {
            "epoch": epochs - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        latest_state_path,
    )
    if best_state_path.exists():
        checkpoint = torch.load(best_state_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    write_csv(output_dir / "train_history.csv", history)

    cache_root = resolve_project_path("outputs/cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    classifier_model, classifier_preprocess, categories = load_classifier(config, device)
    lpips_model, lpips_error = (None, "Skipped by --skip-lpips") if args.skip_lpips else try_load_lpips(device, cache_root)

    results = []
    csv_rows: list[dict[str, Any]] = []
    for snr in snrs:
        result = evaluate_snr(
            model=model,
            config=config,
            snr=snr,
            names=manifest["eval_names"],
            output_dir=output_dir,
            classifier_model=classifier_model,
            classifier_preprocess=classifier_preprocess,
            categories=categories,
            lpips_model=lpips_model,
            device=device,
        )
        results.append(result)
        csv_rows.extend(result["per_sample"])
        printable = {key: value for key, value in result.items() if key != "per_sample"}
        print(json.dumps(printable, indent=2))

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
        "channel": str(config["channel"]),
        "snrs": snrs,
        "cbr": float(config["cbr"]),
        "seed": seed,
        "inputs": config["inputs"],
        "split": config["split"],
        "model": config["model"],
        "training": config["training"],
        "classifier": config["classifier"],
        "failure_handling": config["failure_handling"],
        "lpips_error": lpips_error,
        "python_version": platform.python_version(),
        "package_versions": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "pillow": md.version("pillow"),
            "pytorch-msssim": md.version("pytorch-msssim"),
        },
        "proxy_environment_present": sorted(key for key in os.environ if "proxy" in key.lower()),
        "download_note": "No model or data download is required; this pilot uses existing M0 exports.",
        "key_sources": [
            "scripts/s5_residual_refiner_pilot.py",
            "src/cadsd_jscc/metrics.py",
        ],
    }
    payload = {
        "metadata": metadata,
        "train_history": history,
        "results": results,
        "summary_rows": aggregate_results(results),
    }
    save_json(output_dir / "metrics.json", payload)
    write_csv(output_dir / "per_sample.csv", csv_rows)
    write_csv(output_dir / "summary.csv", aggregate_results(results))
    (output_dir / "REPORT.md").write_text(make_report(results, config), encoding="utf-8")
    print(json.dumps({"output_dir": project_relative(output_dir), "num_results": len(results)}, indent=2))


if __name__ == "__main__":
    main()
