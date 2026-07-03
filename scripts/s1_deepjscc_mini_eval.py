from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cadsd_jscc.deepjscc_adapter import load_deepjscc_model
from cadsd_jscc.metrics import ms_ssim_per_sample, psnr_per_sample, ssim_per_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mini evaluation for the DeepJSCC CIFAR-10 baseline.")
    parser.add_argument("--config", default="configs/s1_deepjscc_cifar10_awgn.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--formal", action="store_true")
    return parser.parse_args()


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def build_subset(config: dict, args: argparse.Namespace) -> tuple[Subset, list[int]]:
    data_config = config["data"]
    mini_config = data_config["mini_eval"]
    root = PROJECT_ROOT / data_config["root"]
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR10(root=root, train=False, download=args.download, transform=transform)

    num_samples = int(args.num_samples or mini_config["num_samples"])
    if num_samples > len(dataset):
        raise ValueError(f"num_samples={num_samples} exceeds CIFAR-10 test size {len(dataset)}")

    generator = torch.Generator().manual_seed(int(mini_config["subset_seed"]))
    indices = torch.randperm(len(dataset), generator=generator)[:num_samples].tolist()
    return Subset(dataset, indices), indices


def evaluate_snr(
    config: dict,
    device: torch.device,
    loader: DataLoader,
    snr: float,
    output_dir: Path,
) -> dict:
    torch.manual_seed(int(config["seed"]) + int(round(float(snr) * 100)))
    model = load_deepjscc_model(
        repo_root=PROJECT_ROOT / config["baseline"]["repo"],
        checkpoint_path=PROJECT_ROOT / config["baseline"]["checkpoint"],
        inner_channel=int(config["inner_channel"]),
        channel=str(config["channel"]),
        snr=float(snr),
        device=device,
    )

    mse_values: list[float] = []
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    ms_ssim_values: list[float] = []
    ms_ssim_error: str | None = None
    sample_saved = False

    with torch.no_grad():
        for images, _labels in loader:
            images = images.to(device)
            reconstructed = model(images).clamp(0, 1)

            batch_mse = torch.mean((reconstructed - images) ** 2, dim=(1, 2, 3))
            mse_values.extend(batch_mse.detach().cpu().tolist())
            psnr_values.extend(psnr_per_sample(reconstructed, images).detach().cpu().tolist())
            ssim_values.extend(ssim_per_sample(reconstructed, images).detach().cpu().tolist())

            if ms_ssim_error is None:
                try:
                    ms_ssim_values.extend(ms_ssim_per_sample(reconstructed, images).detach().cpu().tolist())
                except AssertionError as exc:
                    ms_ssim_error = str(exc)

            if not sample_saved:
                sample_count = min(8, images.size(0))
                comparison = torch.cat([images[:sample_count], reconstructed[:sample_count]], dim=0)
                save_image(
                    comparison,
                    output_dir / f"samples_snr_{float(snr):.1f}db.png",
                    nrow=sample_count,
                )
                sample_saved = True

    row = {
        "snr_db": float(snr),
        "num_images": len(psnr_values),
        "mse": mean_or_none(mse_values),
        "psnr_db": mean_or_none(psnr_values),
        "ssim": mean_or_none(ssim_values),
        "ms_ssim": mean_or_none(ms_ssim_values),
    }
    if ms_ssim_error:
        row["ms_ssim_note"] = ms_ssim_error
    return row


def main() -> None:
    args = parse_args()
    config_path = PROJECT_ROOT / args.config
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    device = torch.device(args.device if args.device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(int(config["seed"]))

    mini_config = config["data"]["mini_eval"]
    batch_size = int(args.batch_size or mini_config["batch_size"])
    subset, indices = build_subset(config, args)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(mini_config["num_workers"]),
    )

    output_dir = PROJECT_ROOT / (args.output_dir or config["outputs"]["mini_eval"])
    if args.formal and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Formal experiment output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")

    rows = [
        evaluate_snr(config=config, device=device, loader=loader, snr=float(snr), output_dir=output_dir)
        for snr in config["snr_sweep_db"]
    ]

    subset_record = {
        "dataset": "CIFAR-10 test",
        "subset_seed": int(mini_config["subset_seed"]),
        "indices": indices,
    }
    with (output_dir / "subset_indices.json").open("w", encoding="utf-8") as handle:
        json.dump(subset_record, handle, indent=2)

    result = {
        "note": "Formal baseline experiment." if args.formal else "Mini-eval only. Use EXP-S1-001 for the formal baseline experiment.",
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "run_command": " ".join(sys.argv),
        "device": str(device),
        "batch_size": batch_size,
        "num_samples": len(indices),
        "channel": str(config["channel"]),
        "cbr": float(config["cbr"]),
        "checkpoint": str(config["baseline"]["checkpoint"]),
        "rows": rows,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
