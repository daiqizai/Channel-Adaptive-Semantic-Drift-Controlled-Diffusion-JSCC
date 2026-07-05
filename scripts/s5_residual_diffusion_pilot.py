from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from s5_residual_refiner_pilot import (  # noqa: E402
    ResidualBlock,
    ResidualPairDataset,
    aggregate_results,
    evaluate_snr,
    get_project_version,
    load_classifier,
    mean,
    parse_snrs,
    project_relative,
    resolve_device,
    resolve_project_path,
    save_json,
    try_load_lpips,
    validate_inputs,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate a tiny SNR-conditioned DDPM over pixel residuals from existing M0 exports."
    )
    parser.add_argument("--config", default="configs/s5_residual_diffusion_pilot_coco256_awgn.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


class ResidualDiffusionDenoiser(nn.Module):
    def __init__(self, input_channels: int, base_channels: int, num_blocks: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(
        self,
        noisy_residual: torch.Tensor,
        m0: torch.Tensor,
        snr_norm: torch.Tensor,
        t_norm: torch.Tensor,
    ) -> torch.Tensor:
        b, _, h, w = m0.shape
        snr_map = snr_norm.view(b, 1, 1, 1).expand(b, 1, h, w)
        t_map = t_norm.view(b, 1, 1, 1).expand(b, 1, h, w)
        x = torch.cat([noisy_residual, m0, snr_map, t_map], dim=1)
        return self.tail(self.body(self.head(x)))


class SNRConditionedPixelResidualDDPM(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config["model"]
        diffusion_cfg = config["diffusion"]
        self.timesteps = int(diffusion_cfg["timesteps"])
        self.sample_steps = int(config.get("sampling", {}).get("steps", self.timesteps))
        self.denoiser = ResidualDiffusionDenoiser(
            input_channels=int(model_cfg["input_channels"]),
            base_channels=int(model_cfg["base_channels"]),
            num_blocks=int(model_cfg["num_blocks"]),
        )
        betas = torch.linspace(
            float(diffusion_cfg["beta_start"]),
            float(diffusion_cfg["beta_end"]),
            self.timesteps,
            dtype=torch.float32,
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def _extract(self, values: torch.Tensor, timesteps: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return values[timesteps].view(-1, 1, 1, 1).to(device=target.device, dtype=target.dtype)

    def timestep_norm(self, timesteps: torch.Tensor) -> torch.Tensor:
        if self.timesteps <= 1:
            return torch.zeros_like(timesteps, dtype=torch.float32)
        return timesteps.float() / float(self.timesteps - 1)

    def q_sample(self, clean_residual: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            self._extract(self.sqrt_alphas_cumprod, timesteps, clean_residual) * clean_residual
            + self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, clean_residual) * noise
        )

    def predict_x0(self, noisy_residual: torch.Tensor, timesteps: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        return (
            noisy_residual
            - self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, noisy_residual) * epsilon
        ) / self._extract(self.sqrt_alphas_cumprod, timesteps, noisy_residual).clamp_min(1e-8)

    def forward(self, m0: torch.Tensor, snr_norm: torch.Tensor, residual_gate: torch.Tensor) -> torch.Tensor:
        return self.sample(m0, snr_norm, residual_gate)

    def sampling_schedule(self) -> list[int]:
        if self.sample_steps >= self.timesteps:
            return list(range(self.timesteps - 1, -1, -1))
        raw_steps = torch.linspace(self.timesteps - 1, 0, self.sample_steps).round().long().tolist()
        schedule: list[int] = []
        for step in raw_steps:
            if not schedule or schedule[-1] != int(step):
                schedule.append(int(step))
        if not schedule or schedule[-1] != 0:
            schedule.append(0)
        return schedule

    @torch.no_grad()
    def sample(self, m0: torch.Tensor, snr_norm: torch.Tensor, residual_gate: torch.Tensor) -> torch.Tensor:
        b, _, h, w = m0.shape
        residual = torch.randn((b, 3, h, w), device=m0.device, dtype=m0.dtype)
        schedule = self.sampling_schedule()
        for index, step in enumerate(schedule):
            timesteps = torch.full((b,), step, dtype=torch.long, device=m0.device)
            epsilon = self.denoiser(residual, m0, snr_norm, self.timestep_norm(timesteps))
            pred_x0 = self.predict_x0(residual, timesteps, epsilon).clamp(-1.0, 1.0)
            prev_step = schedule[index + 1] if index + 1 < len(schedule) else -1
            if prev_step < 0:
                residual = pred_x0
            else:
                alpha_prev = self.alphas_cumprod[prev_step].view(1, 1, 1, 1).to(
                    device=m0.device,
                    dtype=m0.dtype,
                )
                residual = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1.0 - alpha_prev) * epsilon
        gate = residual_gate.view(b, 1, 1, 1)
        return (m0 + gate * residual.clamp(-1.0, 1.0)).clamp(0.0, 1.0)


def gate_tensor(config: dict[str, Any], snr_db: torch.Tensor, device: torch.device) -> torch.Tensor:
    gates = config["model"]["residual_gates"]
    values = []
    for item in snr_db.detach().cpu().tolist():
        key = str(int(item)) if float(item).is_integer() else str(float(item))
        values.append(float(gates[key]))
    return torch.tensor(values, dtype=torch.float32, device=device)


def normalized_target_residual(m0: torch.Tensor, target: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return ((target - m0) / gate.view(-1, 1, 1, 1).clamp_min(1e-8)).clamp(-1.0, 1.0)


def train_one_epoch(
    model: SNRConditionedPixelResidualDDPM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    model.train()
    losses: list[float] = []
    eps_losses: list[float] = []
    x0_losses: list[float] = []
    for batch in loader:
        m0 = batch["m0"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        snr_db = batch["snr_db"].to(device, non_blocking=True)
        snr_norm = batch["snr_norm"].to(device, non_blocking=True)
        gate = gate_tensor(config, snr_db, device)
        clean_residual = normalized_target_residual(m0, target, gate)
        timesteps = torch.randint(0, model.timesteps, (m0.shape[0],), device=device)
        noise = torch.randn_like(clean_residual)
        noisy_residual = model.q_sample(clean_residual, timesteps, noise)

        optimizer.zero_grad(set_to_none=True)
        epsilon = model.denoiser(noisy_residual, m0, snr_norm, model.timestep_norm(timesteps))
        eps_loss = F.mse_loss(epsilon, noise)
        pred_x0 = model.predict_x0(noisy_residual, timesteps, epsilon).clamp(-1.0, 1.0)
        x0_loss = F.mse_loss(pred_x0, clean_residual)
        loss = (
            float(config["training"].get("epsilon_loss_weight", 1.0)) * eps_loss
            + float(config["training"].get("x0_loss_weight", 0.0)) * x0_loss
        )
        loss.backward()
        grad_clip = float(config["training"].get("grad_clip_norm", 0.0))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        eps_losses.append(float(eps_loss.detach().cpu()))
        x0_losses.append(float(x0_loss.detach().cpu()))
    return {
        "loss": float(mean(losses) or 0.0),
        "epsilon_loss": float(mean(eps_losses) or 0.0),
        "x0_loss": float(mean(x0_losses) or 0.0),
    }


@torch.no_grad()
def quick_eval_loss(
    model: SNRConditionedPixelResidualDDPM,
    loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    eps_losses: list[float] = []
    x0_losses: list[float] = []
    for batch in loader:
        m0 = batch["m0"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        snr_db = batch["snr_db"].to(device, non_blocking=True)
        snr_norm = batch["snr_norm"].to(device, non_blocking=True)
        gate = gate_tensor(config, snr_db, device)
        clean_residual = normalized_target_residual(m0, target, gate)
        timesteps = torch.randint(0, model.timesteps, (m0.shape[0],), device=device)
        noise = torch.randn_like(clean_residual)
        noisy_residual = model.q_sample(clean_residual, timesteps, noise)
        epsilon = model.denoiser(noisy_residual, m0, snr_norm, model.timestep_norm(timesteps))
        pred_x0 = model.predict_x0(noisy_residual, timesteps, epsilon).clamp(-1.0, 1.0)
        eps_losses.append(float(F.mse_loss(epsilon, noise).detach().cpu()))
        x0_losses.append(float(F.mse_loss(pred_x0, clean_residual).detach().cpu()))
    return {
        "eval_epsilon_loss": float(mean(eps_losses) or 0.0),
        "eval_x0_loss": float(mean(x0_losses) or 0.0),
    }


def write_report(results: list[dict[str, Any]], config: dict[str, Any], path: Path) -> None:
    rows = aggregate_results(results)
    lines = [
        "# SNR-Conditioned Pixel Residual Diffusion Pilot",
        "",
        "This pilot bypasses Stable Diffusion, text prompts, and the SD VAE. It trains a small DDPM directly over gated pixel residuals from M0 to the original image.",
        "",
        "## Split",
        "",
        f"- Train samples: `sample_{int(config['split']['train_sample_start']):06d}.png` to `sample_{int(config['split']['train_sample_start']) + int(config['split']['train_sample_count']) - 1:06d}.png`",
        f"- Eval samples: `sample_{int(config['split']['eval_sample_start']):06d}.png` to `sample_{int(config['split']['eval_sample_start']) + int(config['split']['eval_sample_count']) - 1:06d}.png`",
        "- This is a small residual-diffusion design probe, not a final M2/M3 result.",
        "",
        "## Main Table",
        "",
        "| SNR(dB) | Gate | M0 PSNR | Refined PSNR | Delta | M3 PSNR | M3 Delta | M0 failure | Refined failure | M3 failure | Accept | Time ms/img |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {snr:g} | {gate:.4f} | {m0_psnr:.4f} | {refined_psnr:.4f} | {refined_delta:.4f} | {m3_psnr:.4f} | {m3_delta:.4f} | {m0_fail:.4f} | {refined_fail:.4f} | {m3_fail:.4f} | {accept:.4f} | {time_ms:.2f} |".format(
                snr=float(row["snr_db"]),
                gate=float(row["residual_gate"]),
                m0_psnr=float(row["m0_psnr_db"]),
                refined_psnr=float(row["refined_psnr_db"]),
                refined_delta=float(row["refined_delta_psnr_vs_m0_db"]),
                m3_psnr=float(row["m3_psnr_db"]),
                m3_delta=float(row["m3_delta_psnr_vs_m0_db"]),
                m0_fail=float(row["m0_final_failure"]),
                refined_fail=float(row["refined_failure"]),
                m3_fail=float(row["m3_final_failure"]),
                accept=float(row["accept_rate"]),
                time_ms=float(row["refiner_time_ms_per_image"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "- A positive result requires quality gains without worse pseudo semantic failure.",
            "- A negative result is still useful: it tells us whether diffusion needs stronger conditioning, more data, or a different residual parameterization.",
            "- COCO ImageNet pseudo-label consistency remains an auxiliary semantic diagnostic, not ground-truth classification accuracy.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def git_dirty_state() -> str:
    try:
        status = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return "dirty" if status else "clean"
    except Exception:  # noqa: BLE001
        return "unknown"


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
        torch.backends.cudnn.benchmark = True

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

    model = SNRConditionedPixelResidualDDPM(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    history: list[dict[str, Any]] = []
    best_eval_loss = float("inf")
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_state_path = checkpoint_dir / "best.pt"
    latest_state_path = checkpoint_dir / "latest.pt"

    epochs = int(config["training"]["epochs"])
    validate_every = int(config["training"].get("validation_every_epochs", 10))
    for epoch in range(epochs):
        train_stats = train_one_epoch(model, train_loader, optimizer, config, device)
        row: dict[str, Any] = {"epoch": epoch, **train_stats}
        if (epoch + 1) % validate_every == 0 or epoch == epochs - 1:
            eval_stats = quick_eval_loss(model, eval_loader, config, device)
            row.update(eval_stats)
            if eval_stats["eval_epsilon_loss"] < best_eval_loss:
                best_eval_loss = eval_stats["eval_epsilon_loss"]
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

    sampling_seed = int(config.get("sampling", {}).get("seed", seed))
    torch.manual_seed(sampling_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(sampling_seed)

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
        "sampling_seed": sampling_seed,
        "inputs": config["inputs"],
        "split": config["split"],
        "model": config["model"],
        "diffusion": config["diffusion"],
        "training": config["training"],
        "sampling": config["sampling"],
        "classifier": config["classifier"],
        "failure_handling": config["failure_handling"],
        "lpips_error": lpips_error,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "python_version": platform.python_version(),
        "package_versions": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "pillow": md.version("pillow"),
            "pytorch-msssim": md.version("pytorch-msssim"),
        },
        "proxy_environment_present": sorted(key for key in os.environ if "proxy" in key.lower()),
        "download_note": "No model or data download is required; this pilot uses existing M0 exports and local classifier weights.",
        "key_sources": [
            "scripts/s5_residual_diffusion_pilot.py",
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
    write_report(results, config, output_dir / "REPORT.md")
    print(json.dumps({"output_dir": project_relative(output_dir), "num_results": len(results)}, indent=2))


if __name__ == "__main__":
    main()
