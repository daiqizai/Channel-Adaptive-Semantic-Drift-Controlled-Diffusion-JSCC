#!/usr/bin/env python3
"""Export a COCO c=8 cache with the frozen UInt2 x BPSK-r4 reservation."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from cadsd_jscc.deepjscc_adapter import (  # noqa: E402
    deepjscc_decode,
    deepjscc_encode,
    deepjscc_transmit,
    extract_deepjscc_state_dict,
    build_deepjscc_model,
)
from cadsd_jscc.metrics import psnr_per_sample  # noqa: E402
from cadsd_jscc.semantic_sketch import (  # noqa: E402
    embed_repeated_sketch,
    recover_repeated_sketch_and_erase,
    semantic_payload_accounting,
)
from s13_export_coco_train2017_c8_scaleup import (  # noqa: E402
    ManifestDataset,
    derived_seed,
    discover_images,
    make_records,
    quantize,
    relative,
    resolve,
    resolve_device,
    save_json,
    select_paths,
    sha256_file,
    snr_name,
    validate_rate,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/s15_coco_uint2_reserved_c8_export_pilot.yaml"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def frozen_payload(batch_size: int, payload_bits: int, device: torch.device) -> torch.Tensor:
    """Return a deterministic balanced BPSK payload with exactly unit symbol power."""
    indices = torch.arange(payload_bits, device=device)
    payload = (indices.remainder(2) * 2 - 1).to(torch.float32)
    return payload.unsqueeze(0).expand(batch_size, -1)


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
    payload_bits = int(config["reservation"]["payload_vector_dim"])
    repetitions = int(config["reservation"]["repetitions"])
    expected_reserved = int(config["reservation"]["reserved_real_symbols"])
    original_dir = output_dir / "exports" / "original"
    original_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for snr in snrs:
        model.change_channel(str(config["channel"]["type"]), snr)
        reconstruction_dir = output_dir / "exports" / snr_name(snr) / "reconstruction"
        reconstruction_dir.mkdir(parents=True, exist_ok=False)
        psnrs: list[float] = []
        elapsed = 0.0
        for images_cpu, indices in loader:
            batch_start = int(indices[0])
            images = images_cpu.to(device, non_blocking=True)
            latent = deepjscc_encode(model, images)
            payload = frozen_payload(images.shape[0], payload_bits, device)
            transmitted, reserved = embed_repeated_sketch(latent, payload, repetitions)
            if reserved.numel() != expected_reserved:
                raise RuntimeError("runtime reservation count differs from the frozen rate ledger")
            torch.manual_seed(derived_seed(int(config["seed"]), snr, batch_start))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            received = deepjscc_transmit(model, transmitted)
            _, erased_received = recover_repeated_sketch_and_erase(
                received, payload_bits, repetitions, reserved
            )
            reconstructions = quantize(deepjscc_decode(model, erased_received), quantize_png)
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
                        "payload_codec": "digital_uint2_bpsk_r4",
                        "payload_real_symbols": expected_reserved,
                        "image_real_symbols_after_reservation": latent[0].numel()
                        - expected_reserved,
                        "decoder_reserved_symbols_erased": True,
                    }
                )
        sample_dir = output_dir / "samples"
        sample_dir.mkdir(exist_ok=True)
        grid_names = [str(record["sample"]) for record in records[:grid_count]]
        grid = torch.stack(
            [
                *[
                    transforms.ToTensor()(Image.open(original_dir / name).convert("RGB"))
                    for name in grid_names
                ],
                *[
                    transforms.ToTensor()(Image.open(reconstruction_dir / name).convert("RGB"))
                    for name in grid_names
                ],
            ]
        )
        save_image(
            grid,
            sample_dir / f"{snr_name(snr)}_original_reserved_reconstruction.png",
            nrow=grid_count,
        )
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
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_rate(config)
    accounting = semantic_payload_accounting(
        int(config["rate"]["inner_channel"]),
        int(config["image_size"]),
        int(config["reservation"]["payload_vector_dim"]),
        int(config["reservation"]["repetitions"]),
    )
    if (
        int(accounting["payload_real_symbols"])
        != int(config["reservation"]["reserved_real_symbols"])
        or int(accounting["structure_real_symbols_after_reservation"])
        != int(config["reservation"]["image_real_symbols_after_reservation"])
    ):
        raise RuntimeError("reservation config disagrees with the exact c=8 rate ledger")
    source_root = resolve(config["inputs"]["source_root"])
    exclusion_root = resolve(config["inputs"]["exclusion_root"])
    checkpoint_path = resolve(config["inputs"]["checkpoint"])
    forbidden = resolve(config["inputs"]["forbidden_checkpoint"])
    for path in (source_root, exclusion_root, checkpoint_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if checkpoint_path.resolve() == forbidden.resolve():
        raise RuntimeError("config points to the forbidden latest.pt")
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
        "train_count": len(train_paths),
        "validation_count": len(validation_paths),
        "snrs": config["channel"]["snrs"],
        "rate_accounting": accounting,
        "output_dir": str(args.output_dir or config["outputs"]["output_dir"]),
        "official_imagenette_accessed": False,
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, "plan": plan}, indent=2))
        return
    output_dir = resolve(args.output_dir or config["outputs"]["output_dir"])
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    shutil.copy2(config_path, output_dir / "config.yaml")
    shutil.copy2(SCRIPT, output_dir / SCRIPT.name)
    save_json(output_dir / "run_plan.json", plan)
    records = make_records(train_paths, validation_paths, exclusion_root)
    write_csv(output_dir / "source_manifest.csv", records)
    manifest_hash = sha256_file(output_dir / "source_manifest.csv")
    device = resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_deepjscc_model(
        repo_root=resolve(config["baseline"]["repo"]),
        inner_channel=int(config["rate"]["inner_channel"]),
        channel=str(config["channel"]["type"]),
        snr=float(config["channel"]["snrs"][0]),
    ).to(device)
    model.load_state_dict(extract_deepjscc_state_dict(checkpoint), strict=True)
    model.eval().requires_grad_(False)
    loader = DataLoader(
        ManifestDataset(records, int(config["image_size"])),
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
            "official_imagenette_accessed": False,
            "reservation_equivalence_note": (
                "The decoder erases all payload positions. Fixed balanced BPSK is equivalent "
                "for restoration-cache generation because payload symbol power and count are fixed."
            ),
            "summaries": summaries,
        },
    )
    expected_rows = len(records) * len(config["channel"]["snrs"])
    if len(rows) != expected_rows:
        raise RuntimeError(f"row count mismatch: {len(rows)} != {expected_rows}")
    print(json.dumps({"output_dir": relative(output_dir), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
