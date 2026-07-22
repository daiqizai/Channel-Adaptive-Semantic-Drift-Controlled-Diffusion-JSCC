#!/usr/bin/env python3
"""Export matched-rate c=6 RGB reconstructions and c=2 decoded structural maps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cadsd_jscc.deepjscc_adapter import build_deepjscc_model, extract_deepjscc_state_dict  # noqa: E402
from cadsd_jscc.metrics import ms_ssim_per_sample, psnr_per_sample, ssim_per_sample  # noqa: E402
from cadsd_jscc.structure import structural_feature_maps, structure_rgb  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/s7_matched_rate_jscc_export_coco256_awgn.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--export-count", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {value}")
    return device


def snr_name(snr: float) -> str:
    return f"snr_{int(snr):02d}db" if float(snr).is_integer() else f"snr_{str(snr).replace('.', 'p')}db"


class ManifestDataset(Dataset):
    def __init__(self, paths: list[Path], image_size: int) -> None:
        self.paths = paths
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB")), index


def load_manifest(path: Path) -> tuple[list[Path], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    raw_paths = payload.get("paths")
    if not isinstance(raw_paths, list) or not raw_paths:
        raise RuntimeError(f"Reference manifest has no paths: {path}")
    paths = [resolve(str(value)) for value in raw_paths]
    if len(paths) != len(set(paths)):
        raise RuntimeError("Reference manifest paths are not unique")
    missing = [item for item in paths if not item.is_file()]
    if missing:
        raise FileNotFoundError(f"Reference manifest has {len(missing)} missing images; first={missing[0]}")
    if int(payload.get("num_images", -1)) != len(paths):
        raise RuntimeError("Reference manifest count mismatch")
    return paths, payload


def validate_rate(config: dict[str, Any]) -> dict[str, Any]:
    rate = config["rate"]
    denominator = int(rate["denominator"])
    main = int(rate["main_inner_channel"])
    structure = int(rate["structure_inner_channel"])
    total = int(rate["total_inner_channel"])
    reference = int(rate["reference_inner_channel"])
    if main + structure != total or total != reference:
        raise RuntimeError(f"Invalid matched rate: {main}+{structure}!={total}!={reference}")
    for key, numerator in (
        ("main_cbr", main),
        ("structure_cbr", structure),
        ("total_cbr", total),
        ("reference_cbr", reference),
    ):
        if not math.isclose(float(rate[key]), numerator / denominator, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"CBR mismatch for {key}")
    return {"main": main, "structure": structure, "total": total, "reference": reference, "denominator": denominator}


def load_arm(
    checkpoint_path: Path,
    expected_arm: str,
    expected_inner_channel: int,
    config: dict[str, Any],
    snr: float,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("arm") != expected_arm:
        raise RuntimeError(f"Checkpoint arm mismatch: {checkpoint.get('arm')} vs {expected_arm}")
    if int(checkpoint.get("inner_channel", -1)) != expected_inner_channel:
        raise RuntimeError("Checkpoint inner-channel mismatch")
    if checkpoint.get("official_val_accessed") is not False:
        raise RuntimeError("Checkpoint does not assert official_val_accessed=false")
    actual_cbr = float(checkpoint.get("actual_cbr", -1.0))
    if not math.isclose(
        actual_cbr,
        expected_inner_channel / int(config["rate"]["denominator"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Checkpoint actual-CBR mismatch")
    model = build_deepjscc_model(
        repo_root=resolve(config["baseline"]["repo"]),
        inner_channel=expected_inner_channel,
        channel=str(config["channel"]["type"]),
        snr=snr,
    ).to(device)
    model.load_state_dict(extract_deepjscc_state_dict(checkpoint), strict=True)
    model.eval().requires_grad_(False)
    return model, {
        "path": relative(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "arm": expected_arm,
        "inner_channel": expected_inner_channel,
        "actual_cbr": actual_cbr,
        "epoch": checkpoint.get("epoch"),
        "metrics": checkpoint.get("metrics"),
    }


def derived_seed(base_seed: int, arm_offset: int, snr: float, batch_start: int) -> int:
    payload = f"{base_seed}:{arm_offset}:{float(snr):.8f}:{batch_start}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def quantize_png(images: torch.Tensor, enabled: bool) -> torch.Tensor:
    images = images.clamp(0.0, 1.0)
    return torch.round(images * 255.0) / 255.0 if enabled else images


@torch.no_grad()
def run_export(
    main_model: torch.nn.Module,
    structure_model: torch.nn.Module,
    loader: DataLoader,
    config: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    export_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    quantize = bool(config["evaluation"]["quantize_png"])
    base_seed = int(config["seed"])
    main_offset = int(config["channel"]["main_seed_offset"])
    structure_offset = int(config["channel"]["structure_seed_offset"])
    snrs = [float(value) for value in config["channel"]["snrs"]]
    original_dir = output_dir / "exports" / "original"
    structure_source_dir = output_dir / "exports" / "structure_source"
    original_dir.mkdir(parents=True, exist_ok=False)
    structure_source_dir.mkdir(parents=True, exist_ok=False)
    timings = {"main_seconds": 0.0, "structure_seconds": 0.0}
    for snr in snrs:
        main_model.change_channel(str(config["channel"]["type"]), snr)
        structure_model.change_channel(str(config["channel"]["type"]), snr)
        folder = snr_name(snr)
        main_dir = output_dir / "exports" / folder / "main_reconstruction"
        structure_dir = output_dir / "exports" / folder / "structure_reconstruction"
        main_dir.mkdir(parents=True, exist_ok=False)
        structure_dir.mkdir(parents=True, exist_ok=False)
        batch_start = 0
        for images_cpu, indices in loader:
            images = images_cpu.to(device, non_blocking=True)
            source_structure = structure_rgb(images, third_channel="maximum")
            torch.manual_seed(derived_seed(base_seed, main_offset, snr, batch_start))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            main = quantize_png(main_model(images), quantize)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings["main_seconds"] += time.perf_counter() - started
            torch.manual_seed(derived_seed(base_seed, structure_offset, snr, batch_start))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            decoded_structure = quantize_png(structure_model(source_structure), quantize)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings["structure_seconds"] += time.perf_counter() - started
            main_psnr = psnr_per_sample(main, images)
            main_ssim = ssim_per_sample(main, images)
            main_ms_ssim = ms_ssim_per_sample(main, images)
            structure_mse = F.mse_loss(
                decoded_structure, source_structure, reduction="none"
            ).flatten(start_dim=1).mean(dim=1)
            structure_psnr = psnr_per_sample(decoded_structure, source_structure)
            source_features = source_structure[:, :2]
            decoded_features = decoded_structure[:, :2]
            feature_mse = F.mse_loss(
                decoded_features, source_features, reduction="none"
            ).flatten(start_dim=1).mean(dim=1)
            for local, dataset_index_raw in enumerate(indices.tolist()):
                dataset_index = int(dataset_index_raw)
                filename = f"sample_{dataset_index:06d}.png"
                if dataset_index < export_count:
                    if not (original_dir / filename).exists():
                        save_image(images[local].cpu(), original_dir / filename)
                        save_image(source_structure[local].cpu(), structure_source_dir / filename)
                    save_image(main[local].cpu(), main_dir / filename)
                    save_image(decoded_structure[local].cpu(), structure_dir / filename)
                rows.append(
                    {
                        "sample_index": dataset_index,
                        "sample": filename,
                        "snr_db": snr,
                        "main_channel_seed": derived_seed(base_seed, main_offset, snr, batch_start),
                        "structure_channel_seed": derived_seed(base_seed, structure_offset, snr, batch_start),
                        "main_psnr_db": float(main_psnr[local].item()),
                        "main_ssim": float(main_ssim[local].item()),
                        "main_ms_ssim": float(main_ms_ssim[local].item()),
                        "structure_mse": float(structure_mse[local].item()),
                        "structure_psnr_db": float(structure_psnr[local].item()),
                        "structure_first2_mse": float(feature_mse[local].item()),
                    }
                )
            batch_start += len(images)
    return rows, timings


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    training_config_path = resolve(config["inputs"]["training_config"])
    with training_config_path.open("r", encoding="utf-8") as handle:
        training_config = yaml.safe_load(handle)
    config["baseline"] = training_config["baseline"]
    rate_contract = validate_rate(config)
    paths, manifest = load_manifest(resolve(config["inputs"]["reference_manifest"]))
    export_count = int(
        args.export_count if args.export_count is not None else config["evaluation"]["export_count"]
    )
    if export_count <= 0 or export_count > len(paths):
        raise ValueError(f"Invalid export_count={export_count} for {len(paths)} manifest images")
    device = resolve_device(args.device)
    dataset = ManifestDataset(paths, int(config["image_size"]))
    loader = DataLoader(
        dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["evaluation"]["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(config["evaluation"]["num_workers"]) > 0,
    )
    snrs = [float(value) for value in config["channel"]["snrs"]]
    main_model, main_metadata = load_arm(
        resolve(config["inputs"]["main_checkpoint"]),
        "main",
        rate_contract["main"],
        config,
        snrs[0],
        device,
    )
    structure_model, structure_metadata = load_arm(
        resolve(config["inputs"]["structure_checkpoint"]),
        "structure",
        rate_contract["structure"],
        config,
        snrs[0],
        device,
    )
    plan = {
        "analysis_id": config["analysis_id"],
        "num_images": len(paths),
        "export_count": export_count,
        "snrs": snrs,
        "rate_contract": rate_contract,
        "main_checkpoint": main_metadata,
        "structure_checkpoint": structure_metadata,
        "independent_arm_noise": config["channel"]["independent_arm_noise"],
        "official_val_accessed": False,
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, "plan": plan}, indent=2, ensure_ascii=False))
        return
    output_dir = resolve(args.output_dir or config["outputs"]["output_dir"])
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output_dir}")
        analysis_root = (PROJECT_ROOT / "outputs" / "eval").resolve()
        if analysis_root not in output_dir.resolve().parents:
            raise RuntimeError(f"Unsafe overwrite target: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    shutil.copy2(config_path, output_dir / "config.yaml")
    shutil.copy2(SCRIPT_PATH, output_dir / SCRIPT_PATH.name)
    save_json(output_dir / "run_plan.json", plan)
    save_json(
        output_dir / "source_manifest.json",
        {
            **manifest,
            "reference_manifest": relative(resolve(config["inputs"]["reference_manifest"])),
            "reference_manifest_sha256": sha256_file(resolve(config["inputs"]["reference_manifest"])),
        },
    )
    rows, timings = run_export(
        main_model, structure_model, loader, config, output_dir, device, export_count
    )
    expected_rows = len(paths) * len(snrs)
    if len(rows) != expected_rows:
        raise RuntimeError(f"Row count mismatch: {len(rows)} vs {expected_rows}")
    write_csv(output_dir / "per_sample.csv", rows)
    summary = []
    for snr in snrs:
        selected = [row for row in rows if float(row["snr_db"]) == snr]
        summary.append(
            {
                "snr_db": snr,
                "num_images": len(selected),
                **{
                    key: float(sum(float(row[key]) for row in selected) / len(selected))
                    for key in (
                        "main_psnr_db",
                        "main_ssim",
                        "main_ms_ssim",
                        "structure_mse",
                        "structure_psnr_db",
                        "structure_first2_mse",
                    )
                },
            }
        )
    write_csv(output_dir / "summary.csv", summary)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": relative(config_path),
        "config_sha256": sha256_file(config_path),
        "script": relative(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "source_manifest_sha256": sha256_file(resolve(config["inputs"]["reference_manifest"])),
        "main": main_metadata,
        "structure": structure_metadata,
        "rate_contract": rate_contract,
        "channel": config["channel"],
        "timings_seconds": timings,
        "num_images": len(paths),
        "export_count": export_count,
        "summary": summary,
        "official_val_accessed": False,
    }
    save_json(output_dir / "metadata.json", metadata)
    save_json(
        output_dir / "STATE.json",
        {
            "state": "COMPLETE",
            "num_rows": len(rows),
            "per_sample_sha256": sha256_file(output_dir / "per_sample.csv"),
            "official_val_accessed": False,
        },
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
