#!/usr/bin/env python3
"""Export a deterministic COCO train2017 c=8 DeepJSCC scale-up cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(ROOT / "src"))

from cadsd_jscc.deepjscc_adapter import (  # noqa: E402
    build_deepjscc_model,
    extract_deepjscc_state_dict,
)
from cadsd_jscc.external_rate_alignment import ExactRateMaskedDeepJSCC  # noqa: E402
from cadsd_jscc.metrics import psnr_per_sample  # noqa: E402
from cadsd_jscc.semantic_sketch import reserved_symbol_indices  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/s13_coco_train2017_c8_scaleup_export.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--train-count", type=int, default=None)
    parser.add_argument("--validation-count", type=int, default=None)
    parser.add_argument("--snrs", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
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


def discover_images(root: Path) -> list[Path]:
    supported = {".jpg", ".jpeg", ".png", ".webp"}
    paths = sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in supported)
    if not paths:
        raise RuntimeError(f"No images found: {root}")
    return paths


def selection_rank(path: Path, source_root: Path, seed: int) -> bytes:
    payload = f"{int(seed)}:{path.resolve().relative_to(source_root.resolve())}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def select_paths(
    paths: list[Path], source_root: Path, seed: int, train_count: int, validation_count: int
) -> tuple[list[Path], list[Path]]:
    total = int(train_count) + int(validation_count)
    if train_count <= 0 or validation_count <= 0 or total > len(paths):
        raise ValueError(
            f"Invalid split counts train={train_count}, validation={validation_count}, available={len(paths)}"
        )
    ranked = sorted(paths, key=lambda path: (selection_rank(path, source_root, seed), str(path)))
    return ranked[:train_count], ranked[train_count:total]


def derived_seed(base_seed: int, snr: float, batch_start: int) -> int:
    payload = f"{int(base_seed)}:{float(snr):.8f}:{int(batch_start)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def snr_name(snr: float) -> str:
    return f"snr_{int(snr):02d}db" if float(snr).is_integer() else f"snr_{str(snr).replace('.', 'p')}db"


class ManifestDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], image_size: int) -> None:
        self.records = records
        self.transform = transforms.Compose(
            [transforms.Resize(image_size), transforms.CenterCrop(image_size), transforms.ToTensor()]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(resolve(self.records[index]["source_path"])) as image:
            return self.transform(image.convert("RGB")), index


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {value}")
    return device


def quantize(images: torch.Tensor, enabled: bool) -> torch.Tensor:
    images = images.clamp(0.0, 1.0)
    return torch.round(images * 255.0) / 255.0 if enabled else images


def validate_rate(config: dict[str, Any]) -> None:
    rate = config["rate"]
    if bool(rate.get("exact_mask", False)):
        active_symbols = int(rate["active_real_symbols"])
        dense_symbols = int(rate["dense_real_symbols"])
        source_symbols = int(rate["source_real_dimensions"])
        if active_symbols <= 0 or active_symbols > dense_symbols or active_symbols % 2:
            raise RuntimeError("Exact-mask active real-symbol count is invalid")
        expected = (active_symbols / 2) / source_symbols
        if int(rate["total_complex_uses"]) != active_symbols // 2:
            raise RuntimeError("Exact-mask complex-use ledger is inconsistent")
    else:
        expected = int(rate["inner_channel"]) / int(rate["denominator"])
    if not math.isclose(float(rate["cbr"]), expected, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"CBR mismatch: {rate['cbr']} vs {expected}")


def balanced_payload(
    batch_size: int, payload_symbols: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Return a deterministic unit-power surrogate without reading semantic labels."""

    if payload_symbols <= 0:
        raise ValueError("payload_symbols must be positive")
    signs = torch.ones(payload_symbols, device=device, dtype=dtype)
    signs[1::2] = -1
    return signs.unsqueeze(0).expand(batch_size, -1)


def exact_mask_forward(
    model: ExactRateMaskedDeepJSCC,
    images: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, int]:
    """Export the image branch under the frozen low-rate payload reservation."""

    active, dense_shape = model.encode_active(images)
    reservation = config["reservation"]
    payload_symbols = int(reservation["payload_real_symbols"])
    reserved = reserved_symbol_indices(
        model.active_symbols, payload_symbols, device=active.device
    )
    transmitted = active.clone()
    transmitted[:, reserved] = balanced_payload(
        active.shape[0], payload_symbols, device=active.device, dtype=active.dtype
    )
    norm = transmitted.float().square().sum(dim=1, keepdim=True).clamp_min(1e-12).sqrt()
    transmitted = transmitted * (model.active_symbols**0.5 / norm).to(transmitted.dtype)
    received = model.transmit_active(transmitted)
    # The receiver consumes the reserved coordinates as an auxiliary payload;
    # the image decoder must not see them as image latent values.
    received[:, reserved] = 0
    return model.decode_active(received, dense_shape), payload_symbols


def make_records(
    train_paths: list[Path], validation_paths: list[Path], exclusion_root: Path
) -> list[dict[str, Any]]:
    exclusion_paths = discover_images(exclusion_root)
    exclusion_names = {path.name for path in exclusion_paths}
    exclusion_hashes = {sha256_file(path) for path in exclusion_paths}
    records: list[dict[str, Any]] = []
    selected_hashes: set[str] = set()
    for role, role_paths in (("train", train_paths), ("validation", validation_paths)):
        for path in role_paths:
            digest = sha256_file(path)
            if path.name in exclusion_names or digest in exclusion_hashes:
                raise RuntimeError(f"Selected train2017 image overlaps exclusion root: {path}")
            if digest in selected_hashes:
                raise RuntimeError(f"Duplicate selected image hash: {path}")
            selected_hashes.add(digest)
            index = len(records)
            records.append(
                {
                    "sample_index": index,
                    "sample": f"sample_{index:06d}.png",
                    "role": role,
                    "source_path": relative(path),
                    "source_sha256": digest,
                }
            )
    return records


@torch.no_grad()
def export_cache(
    model: torch.nn.Module,
    loader: DataLoader,
    records: list[dict[str, Any]],
    config: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    snrs = [float(value) for value in config["channel"]["snrs"]]
    quantize_png = bool(config["evaluation"]["quantize_png"])
    grid_count = int(config["evaluation"]["sample_grid_count"])
    original_dir = output_dir / "exports" / "original"
    original_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for snr in snrs:
        exact_mask = isinstance(model, ExactRateMaskedDeepJSCC)
        if exact_mask:
            model.snr_db = snr
        else:
            model.change_channel(str(config["channel"]["type"]), snr)
        reconstruction_dir = output_dir / "exports" / snr_name(snr) / "reconstruction"
        reconstruction_dir.mkdir(parents=True, exist_ok=False)
        psnrs: list[float] = []
        elapsed = 0.0
        for images_cpu, indices in loader:
            batch_start = int(indices[0])
            images = images_cpu.to(device, non_blocking=True)
            torch.manual_seed(derived_seed(int(config["seed"]), snr, batch_start))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            if exact_mask:
                reconstructed, payload_symbols = exact_mask_forward(model, images, config)
            else:
                reconstructed = model(images)
                payload_symbols = 0
            reconstructions = quantize(reconstructed, quantize_png)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - started
            batch_psnr = psnr_per_sample(reconstructions, images).cpu()
            psnrs.extend(float(value) for value in batch_psnr.tolist())
            for local, index_raw in enumerate(indices.tolist()):
                index = int(index_raw)
                record = records[index]
                name = str(record["sample"])
                if snr == snrs[0]:
                    save_image(quantize(images[local].cpu(), True), original_dir / name)
                save_image(reconstructions[local].cpu(), reconstruction_dir / name)
                rows.append(
                    {
                        **record,
                        "snr_db": snr,
                        "channel_seed": derived_seed(int(config["seed"]), snr, batch_start),
                        "psnr_db": float(batch_psnr[local]),
                        "total_active_real_symbols": int(
                            config["rate"].get("active_real_symbols", 0)
                        ),
                        "payload_reserved_real_symbols": payload_symbols,
                        "image_active_real_symbols": int(
                            config["rate"].get("active_real_symbols", 0)
                        )
                        - payload_symbols,
                    }
                )
        sample_dir = output_dir / "samples"
        sample_dir.mkdir(exist_ok=True)
        grid_names = [str(record["sample"]) for record in records[:grid_count]]
        grid = torch.stack(
            [
                *[transforms.ToTensor()(Image.open(original_dir / name).convert("RGB")) for name in grid_names],
                *[
                    transforms.ToTensor()(Image.open(reconstruction_dir / name).convert("RGB"))
                    for name in grid_names
                ],
            ]
        )
        save_image(grid, sample_dir / f"{snr_name(snr)}_original_reconstruction.png", nrow=grid_count)
        summaries.append(
            {
                "snr_db": snr,
                "num_images": len(records),
                "mean_psnr_db": float(sum(psnrs) / len(psnrs)),
                "inference_time_ms_per_image": 1000.0 * elapsed / len(records),
            }
        )
        print(json.dumps(summaries[-1], indent=2))
    return rows, summaries


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.train_count is not None:
        config["split"]["train_count"] = int(args.train_count)
    if args.validation_count is not None:
        config["split"]["validation_count"] = int(args.validation_count)
    if args.snrs:
        config["channel"]["snrs"] = [float(value.strip()) for value in args.snrs.split(",") if value.strip()]
    if args.batch_size is not None:
        config["evaluation"]["batch_size"] = int(args.batch_size)
    validate_rate(config)

    source_root = resolve(config["inputs"]["source_root"])
    exclusion_root = resolve(config["inputs"]["exclusion_root"])
    checkpoint_path = resolve(config["inputs"]["checkpoint"])
    forbidden = resolve(config["inputs"]["forbidden_checkpoint"])
    for path in (source_root, exclusion_root, checkpoint_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if checkpoint_path.resolve() == forbidden.resolve():
        raise RuntimeError("Config points to forbidden latest.pt")
    paths = discover_images(source_root)
    train_paths, validation_paths = select_paths(
        paths,
        source_root,
        int(config["seed"]),
        int(config["split"]["train_count"]),
        int(config["split"]["validation_count"]),
    )
    plan = {
        "export_id": config["export_id"],
        "available_images": len(paths),
        "train_count": len(train_paths),
        "validation_count": len(validation_paths),
        "snrs": config["channel"]["snrs"],
        "first_train_source": relative(train_paths[0]),
        "first_validation_source": relative(validation_paths[0]),
        "output_dir": str(args.output_dir or config["outputs"]["output_dir"]),
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, "plan": plan}, indent=2))
        return

    output_dir = resolve(args.output_dir or config["outputs"]["output_dir"])
    if output_dir.exists():
        raise FileExistsError(f"Output exists, refusing to overwrite: {output_dir}")
    output_dir.mkdir(parents=True)
    shutil.copy2(config_path, output_dir / "config.yaml")
    shutil.copy2(SCRIPT, output_dir / SCRIPT.name)
    save_json(output_dir / "run_plan.json", plan)
    records = make_records(train_paths, validation_paths, exclusion_root)
    write_csv(output_dir / "source_manifest.csv", records)
    manifest_hash = sha256_file(output_dir / "source_manifest.csv")

    device = resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    base_model = build_deepjscc_model(
        repo_root=resolve(config["baseline"]["repo"]),
        inner_channel=int(config["rate"]["inner_channel"]),
        channel=str(config["channel"]["type"]),
        snr=float(config["channel"]["snrs"][0]),
    ).to(device)
    if bool(config["rate"].get("exact_mask", False)):
        model = ExactRateMaskedDeepJSCC(
            base_model,
            dense_symbols=int(config["rate"]["dense_real_symbols"]),
            active_symbols=int(config["rate"]["active_real_symbols"]),
            snr_db=float(config["channel"]["snrs"][0]),
        ).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
    else:
        model = base_model
        model.load_state_dict(extract_deepjscc_state_dict(checkpoint), strict=True)
    model.eval().requires_grad_(False)
    dataset = ManifestDataset(records, int(config["image_size"]))
    loader = DataLoader(
        dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["evaluation"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    rows, summaries = export_cache(model, loader, records, config, output_dir, device)
    write_csv(output_dir / "per_sample.csv", rows)
    write_csv(output_dir / "summary.csv", summaries)
    save_json(
        output_dir / "metadata.json",
        {
            "run_command": " ".join(sys.argv),
            "config": relative(config_path),
            "script": relative(SCRIPT),
            "checkpoint": relative(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "source_manifest_sha256": manifest_hash,
            "device": str(device),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "official_imagenette_accessed": False,
            "download_note": "No download; local COCO train2017 and checkpoint only.",
            "summaries": summaries,
        },
    )
    expected_rows = len(records) * len(config["channel"]["snrs"])
    if len(rows) != expected_rows:
        raise RuntimeError(f"Export row count mismatch: {len(rows)} vs {expected_rows}")
    print(json.dumps({"output_dir": relative(output_dir), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
