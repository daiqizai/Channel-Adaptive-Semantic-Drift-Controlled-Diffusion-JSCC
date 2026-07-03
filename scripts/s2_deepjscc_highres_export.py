from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cadsd_jscc.datasets import FlatImageDataset
from cadsd_jscc.deepjscc_adapter import build_deepjscc_model, extract_deepjscc_state_dict
from cadsd_jscc.metrics import ms_ssim_per_sample, psnr_per_sample, ssim_per_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and export high-resolution DeepJSCC reconstructions.")
    parser.add_argument("--config", default="configs/s2_deepjscc_coco_val256_awgn_pilot.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--snrs", default="1,4,7,13,19")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--export-count", type=int, default=16)
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def parse_snrs(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def make_eval_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )


def maybe_subset(dataset, size: int | None, seed: int):
    if size is None:
        return dataset
    size = min(int(size), len(dataset))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
    return Subset(dataset, indices)


def dataset_paths(dataset) -> list[str]:
    if isinstance(dataset, Subset):
        base_paths = dataset_paths(dataset.dataset)
        return [base_paths[index] for index in dataset.indices]
    paths = getattr(dataset, "paths", None)
    if paths is None:
        return []
    return [str(Path(path).resolve().relative_to(PROJECT_ROOT)) for path in paths]


def snr_name(snr: float) -> str:
    if float(snr).is_integer():
        return f"snr_{int(snr):02d}db"
    return f"snr_{str(snr).replace('.', 'p')}db"


def save_export_images(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    sample_offset: int,
    export_count: int,
    original_dir: Path,
    reconstruction_dir: Path,
) -> int:
    saved = 0
    for batch_index in range(originals.size(0)):
        sample_index = sample_offset + batch_index
        if sample_index >= export_count:
            break
        filename = f"sample_{sample_index:06d}.png"
        original_path = original_dir / filename
        if not original_path.exists():
            save_image(originals[batch_index], original_path)
        save_image(reconstructions[batch_index], reconstruction_dir / filename)
        saved += 1
    return saved


def evaluate_snr(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    channel: str,
    snr: float,
    max_batches: int | None,
    export_count: int,
    output_dir: Path,
) -> dict:
    model.change_channel(channel, snr)
    model.eval()

    losses: list[float] = []
    psnrs: list[float] = []
    ssims: list[float] = []
    ms_ssims: list[float] = []
    batch_times: list[float] = []
    total_images = 0

    sample_dir = output_dir / "samples"
    export_root = output_dir / "exports"
    original_dir = export_root / "original"
    reconstruction_dir = export_root / snr_name(snr) / "reconstruction"
    sample_dir.mkdir(parents=True, exist_ok=True)
    original_dir.mkdir(parents=True, exist_ok=True)
    reconstruction_dir.mkdir(parents=True, exist_ok=True)

    first_grid_saved = False
    with torch.no_grad():
        for batch_idx, (images, _labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            outputs = model(images).clamp(0, 1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_times.append(time.perf_counter() - start)

            mse_values = F.mse_loss(outputs, images, reduction="none").flatten(start_dim=1).mean(dim=1)
            losses.extend(mse_values.detach().cpu().tolist())
            psnrs.extend(psnr_per_sample(outputs, images).detach().cpu().tolist())
            ssims.extend(ssim_per_sample(outputs, images).detach().cpu().tolist())
            ms_ssims.extend(ms_ssim_per_sample(outputs, images).detach().cpu().tolist())

            if not first_grid_saved:
                count = min(4, images.size(0))
                save_image(
                    torch.cat([images[:count].detach().cpu(), outputs[:count].detach().cpu()], dim=0),
                    sample_dir / f"{snr_name(snr)}_grid.png",
                    nrow=count,
                )
                first_grid_saved = True

            if total_images < export_count:
                save_export_images(
                    originals=images.detach().cpu(),
                    reconstructions=outputs.detach().cpu(),
                    sample_offset=total_images,
                    export_count=export_count,
                    original_dir=original_dir,
                    reconstruction_dir=reconstruction_dir,
                )
            total_images += int(images.size(0))

    return {
        "snr_db": float(snr),
        "num_images": total_images,
        "mse": mean(losses),
        "psnr_db": mean(psnrs),
        "ssim": mean(ssims),
        "ms_ssim": mean(ms_ssims),
        "inference_time_ms_per_image": 1000.0 * sum(batch_times) / max(1, total_images),
        "export_dir": str((export_root / snr_name(snr)).relative_to(PROJECT_ROOT)),
    }


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    seed = int(config["seed"])
    torch.manual_seed(seed)
    device = resolve_device(args.device)
    snrs = parse_snrs(args.snrs)
    image_size = int(config["image_size"])

    checkpoint_path = resolve_project_path(
        args.checkpoint or Path(config["outputs"]["train_dir"]) / "checkpoints" / "best.pt"
    )
    output_dir = resolve_project_path(
        args.output_dir or Path("outputs/eval") / f"{Path(config['outputs']['train_dir']).name}_m0_export"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")

    val_dataset = FlatImageDataset(resolve_project_path(config["data"]["val_root"]), transform=make_eval_transform(image_size))
    val_dataset = maybe_subset(val_dataset, config["data"].get("val_subset"), seed=seed + 1)
    paths = dataset_paths(val_dataset)
    with (output_dir / "source_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"paths": paths, "num_images": len(val_dataset)}, handle, indent=2)

    training = config["training"]
    batch_size = int(args.batch_size or training["batch_size"])
    num_workers = int(args.num_workers if args.num_workers is not None else training["num_workers"])
    loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_deepjscc_state_dict(checkpoint)
    model = build_deepjscc_model(
        repo_root=resolve_project_path(config["baseline"]["repo"]),
        inner_channel=int(config["inner_channel"]),
        channel=str(config["channel"]),
        snr=snrs[0],
    ).to(device)
    model.load_state_dict(state_dict)

    metadata = {
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "run_command": " ".join(sys.argv),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "dataset": config["dataset"],
        "image_size": image_size,
        "channel": str(config["channel"]),
        "snrs": snrs,
        "cbr": float(config["cbr"]),
        "inner_channel": int(config["inner_channel"]),
        "num_images": len(val_dataset),
        "export_count": int(args.export_count),
        "seed": seed,
    }

    results = []
    for snr in snrs:
        torch.manual_seed(seed + int(round(snr * 100)))
        result = evaluate_snr(
            model=model,
            loader=loader,
            device=device,
            channel=str(config["channel"]),
            snr=snr,
            max_batches=args.max_batches,
            export_count=int(args.export_count),
            output_dir=output_dir,
        )
        results.append(result)
        print(json.dumps(result, indent=2))

    payload = {"metadata": metadata, "results": results}
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps({"output_dir": str(output_dir), "results": results}, indent=2))


if __name__ == "__main__":
    main()
