#!/usr/bin/env python3
"""Export a rate-accounted c=2 hybrid structure and continuous semantic-sketch path."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from collections import defaultdict
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
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cadsd_jscc.deepjscc_adapter import build_deepjscc_model, extract_deepjscc_state_dict  # noqa: E402
from cadsd_jscc.semantic_sketch import (  # noqa: E402
    embed_repeated_sketch,
    fixed_rademacher_projection,
    probabilities_to_sketch,
    recover_repeated_sketch_and_erase,
    semantic_payload_accounting,
)
from cadsd_jscc.structure import structure_rgb  # noqa: E402
from s7_export_matched_rate_jscc import derived_seed, quantize_png, snr_name  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/s8_hybrid_structure_semantic_export_coco256_awgn.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {value}")
    return device


class OriginalDataset(Dataset):
    def __init__(self, root: Path, count: int, image_size: int) -> None:
        self.names = [f"sample_{index:06d}.png" for index in range(count)]
        missing = [name for name in self.names if not (root / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} originals; first={missing[0]}")
        self.root = root
        self.transform = transforms.Compose(
            [transforms.Resize(image_size), transforms.CenterCrop(image_size), transforms.ToTensor()]
        )

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.root / self.names[index]) as image:
            return self.transform(image.convert("RGB")), index


def load_teacher(config: dict[str, Any], device: torch.device):
    teacher = config["semantic_teacher"]
    weights_file = resolve(teacher["weights_file"])
    if not weights_file.is_file() or weights_file.stat().st_size < 10 * 1024 * 1024:
        raise RuntimeError(f"AlexNet weights are not available locally: {weights_file}")
    os.environ.setdefault("TORCH_HOME", str(resolve(teacher["cache_dir"])))
    import torchvision.models as models

    weights = getattr(models.AlexNet_Weights, str(teacher["weights"]))
    model = models.alexnet(weights=weights).to(device).eval().requires_grad_(False)
    return model, weights.transforms()


def validate_contract(config: dict[str, Any]) -> dict[str, Any]:
    rate = config["rate"]
    denominator = int(rate["denominator"])
    main = int(rate["main_inner_channel"])
    hybrid = int(rate["hybrid_structure_semantic_inner_channel"])
    total = int(rate["total_inner_channel"])
    reference = int(rate["reference_inner_channel"])
    if main + hybrid != total or total != reference:
        raise RuntimeError(f"Rate mismatch: {main}+{hybrid}!={total}!={reference}")
    for key, numerator in (
        ("main_cbr", main),
        ("hybrid_structure_semantic_cbr", hybrid),
        ("total_cbr", total),
        ("reference_cbr", reference),
    ):
        if not math.isclose(float(rate[key]), numerator / denominator, abs_tol=1e-12, rel_tol=0.0):
            raise RuntimeError(f"CBR mismatch for {key}")
    semantic = config["semantic_teacher"]
    payload = config["payload"]
    accounting = semantic_payload_accounting(
        hybrid,
        int(config["image_size"]),
        int(semantic["sketch_dim"]),
        int(payload["repetitions"]),
    )
    if int(payload["reserved_real_symbols"]) != accounting["payload_real_symbols"]:
        raise RuntimeError("Configured reserved payload does not match sketch_dim*repetitions")
    return {
        "main": main,
        "hybrid": hybrid,
        "total": total,
        "reference": reference,
        "denominator": denominator,
        **accounting,
    }


def load_structure_model(config: dict[str, Any], device: torch.device, snr: float):
    checkpoint_path = resolve(config["inputs"]["structure_checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    expected = int(config["rate"]["hybrid_structure_semantic_inner_channel"])
    if checkpoint.get("arm") != "structure" or int(checkpoint.get("inner_channel", -1)) != expected:
        raise RuntimeError("Structure checkpoint arm/channel mismatch")
    training_config = yaml.safe_load(resolve(config["inputs"]["training_config"]).read_text(encoding="utf-8"))
    model = build_deepjscc_model(
        resolve(training_config["baseline"]["repo"]), expected, str(config["channel"]["type"]), snr
    ).to(device)
    model.load_state_dict(extract_deepjscc_state_dict(checkpoint), strict=True)
    model.eval().requires_grad_(False)
    return model, {"path": relative(checkpoint_path), "sha256": sha256_file(checkpoint_path)}


def baseline_first2_means(config: dict[str, Any], count: int) -> dict[float, float]:
    path = resolve(config["inputs"]["matched_export"]) / "per_sample.csv"
    grouped: dict[float, list[float]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["sample_index"]) < count:
                grouped[float(row["snr_db"])].append(float(row["structure_first2_mse"]))
    return {snr: sum(values) / len(values) for snr, values in grouped.items()}


@torch.no_grad()
def compute_source_sketches(
    loader: DataLoader,
    teacher: torch.nn.Module,
    preprocess,
    projection: torch.Tensor,
    device: torch.device,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sketches = torch.empty(count, projection.shape[0], dtype=torch.float32)
    top1 = torch.empty(count, dtype=torch.int64)
    for images_cpu, indices in loader:
        images = images_cpu.to(device, non_blocking=True)
        probabilities = torch.softmax(teacher(preprocess(images)).float(), dim=1)
        batch_sketches = probabilities_to_sketch(probabilities, projection)
        sketches[indices] = batch_sketches.cpu()
        top1[indices] = probabilities.argmax(dim=1).cpu()
    return sketches, top1


@torch.no_grad()
def export_hybrid(
    config: dict[str, Any],
    model: torch.nn.Module,
    loader: DataLoader,
    source_sketches: torch.Tensor,
    top1: torch.Tensor,
    output_dir: Path,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    semantic = config["semantic_teacher"]
    sketch_dim = int(semantic["sketch_dim"])
    repetitions = int(config["payload"]["repetitions"])
    snrs = [float(value) for value in config["channel"]["snrs"]]
    count = len(loader.dataset)
    received_store = torch.empty(len(snrs), count, sketch_dim, dtype=torch.float32)
    rows: list[dict[str, Any]] = []
    quantize = bool(config["evaluation"]["quantize_png"])
    base_seed = int(config["seed"])
    seed_offset = int(config["channel"]["structure_semantic_seed_offset"])
    for snr_index, snr in enumerate(snrs):
        model.change_channel(str(config["channel"]["type"]), snr)
        decoded_dir = output_dir / "exports" / snr_name(snr) / "hybrid_structure_reconstruction"
        decoded_dir.mkdir(parents=True, exist_ok=False)
        batch_start = 0
        for images_cpu, indices in loader:
            images = images_cpu.to(device, non_blocking=True)
            source_structure = structure_rgb(images, third_channel="maximum")
            latent = model.encoder(source_structure)
            if latent.ndim != 4:
                raise RuntimeError(f"Structure encoder lost batch dimension: {tuple(latent.shape)}")
            source_batch_sketch = source_sketches[indices].to(device)
            hybrid_latent, reserved = embed_repeated_sketch(latent, source_batch_sketch, repetitions)
            channel_seed = derived_seed(base_seed, seed_offset, snr, batch_start)
            torch.manual_seed(channel_seed)
            received_latent = model.channel(hybrid_latent)
            recovered, erased_latent = recover_repeated_sketch_and_erase(
                received_latent, sketch_dim, repetitions, reserved
            )
            decoded = quantize_png(model.decoder(erased_latent), quantize)
            received_store[snr_index, indices] = recovered.cpu()
            cosine = F.cosine_similarity(source_batch_sketch, recovered, dim=1)
            first2_mse = F.mse_loss(
                decoded[:, :2], source_structure[:, :2], reduction="none"
            ).flatten(start_dim=1).mean(dim=1)
            full_mse = F.mse_loss(decoded, source_structure, reduction="none").flatten(start_dim=1).mean(dim=1)
            for local, raw_index in enumerate(indices.tolist()):
                index = int(raw_index)
                name = f"sample_{index:06d}.png"
                save_image(decoded[local].cpu(), decoded_dir / name)
                rows.append(
                    {
                        "sample_index": index,
                        "sample": name,
                        "snr_db": snr,
                        "channel_seed": channel_seed,
                        "teacher_top1": int(top1[index]),
                        "semantic_sketch_cosine": float(cosine[local].cpu()),
                        "hybrid_structure_first2_mse": float(first2_mse[local].cpu()),
                        "hybrid_structure_full_mse": float(full_mse[local].cpu()),
                    }
                )
            batch_start += len(images)
    torch.save(
        {
            "format_version": 1,
            "names": list(loader.dataset.names),
            "snrs": snrs,
            "source_sketches": source_sketches,
            "received_sketches": received_store,
            "teacher_top1": top1,
            "sketch_dim": sketch_dim,
            "repetitions": repetitions,
            "projection_seed": int(semantic["projection_seed"]),
            "official_val_accessed": False,
        },
        output_dir / "semantic_sketches.pt",
    )
    return rows, {"num_images": count, "num_rows": len(rows)}


def summarize(config: dict[str, Any], rows: list[dict[str, Any]], baseline: dict[float, float]) -> dict[str, Any]:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["snr_db"])].append(row)
    threshold_cosine = float(config["evaluation"]["minimum_mean_sketch_cosine_each_snr"])
    threshold_mse = float(config["evaluation"]["maximum_relative_first2_structure_mse_increase"])
    by_snr: dict[str, Any] = {}
    for snr, subset in sorted(grouped.items()):
        cosine = sum(float(row["semantic_sketch_cosine"]) for row in subset) / len(subset)
        mse = sum(float(row["hybrid_structure_first2_mse"]) for row in subset) / len(subset)
        ratio = mse / baseline[snr] - 1.0
        by_snr[str(snr)] = {
            "mean_semantic_sketch_cosine": cosine,
            "mean_hybrid_structure_first2_mse": mse,
            "mean_baseline_structure_first2_mse": baseline[snr],
            "relative_first2_mse_increase": ratio,
            "sketch_gate_pass": cosine >= threshold_cosine,
            "structure_gate_pass": ratio <= threshold_mse,
        }
    all_pass = all(
        item["sketch_gate_pass"] and item["structure_gate_pass"] for item in by_snr.values()
    )
    return {"by_snr": by_snr, "all_stage_gates_pass": all_pass}


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    contract = validate_contract(config)
    count = int(config["evaluation"]["export_count"])
    original_dir = resolve(config["inputs"]["original_dir"])
    dataset = OriginalDataset(original_dir, count, int(config["image_size"]))
    device = resolve_device(args.device)
    plan = {
        "analysis_id": config["analysis_id"],
        "num_images": count,
        "snrs": config["channel"]["snrs"],
        "rate_contract": contract,
        "payload": config["payload"],
        "semantic_teacher": config["semantic_teacher"],
        "official_val_accessed": False,
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, "plan": plan}, indent=2))
        return
    output_dir = resolve(args.output_dir or config["outputs"]["output_dir"])
    if output_dir.exists():
        raise FileExistsError(f"Output exists, refusing overwrite: {output_dir}")
    output_dir.mkdir(parents=True)
    shutil.copy2(config_path, output_dir / "config.yaml")
    shutil.copy2(SCRIPT_PATH, output_dir / SCRIPT_PATH.name)
    save_json(output_dir / "run_plan.json", plan)
    loader = DataLoader(
        dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["evaluation"]["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(config["evaluation"]["num_workers"]) > 0,
    )
    teacher, preprocess = load_teacher(config, device)
    semantic = config["semantic_teacher"]
    projection = fixed_rademacher_projection(
        int(semantic["source_probability_dim"]),
        int(semantic["sketch_dim"]),
        int(semantic["projection_seed"]),
        device=device,
    )
    source_sketches, top1 = compute_source_sketches(
        loader, teacher, preprocess, projection, device, count
    )
    del teacher
    model, checkpoint = load_structure_model(config, device, float(config["channel"]["snrs"][0]))
    rows, export_metadata = export_hybrid(
        config, model, loader, source_sketches, top1, output_dir, device
    )
    baseline = baseline_first2_means(config, count)
    summary = summarize(config, rows, baseline)
    metadata = {
        **plan,
        **export_metadata,
        "structure_checkpoint": checkpoint,
        "config_sha256": sha256_file(config_path),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "proxy_environment_present": sorted(key for key in os.environ if "proxy" in key.lower()),
        "downloaded": False,
    }
    write_csv(output_dir / "per_sample.csv", rows)
    save_json(output_dir / "summary.json", summary)
    save_json(output_dir / "metadata.json", metadata)
    save_json(
        output_dir / "STATE.json",
        {"state": "COMPLETE", "all_stage_gates_pass": summary["all_stage_gates_pass"], "official_val_accessed": False},
    )
    lines = [
        "# Hybrid Structure + Semantic Sketch Export",
        "",
        f"Stage gates: **{'PASS' if summary['all_stage_gates_pass'] else 'FAIL'}**.",
        "",
        "| SNR | Sketch cosine | First-2 MSE increase | Sketch | Structure |",
        "|---:|---:|---:|---|---|",
    ]
    for snr, row in summary["by_snr"].items():
        lines.append(
            f"| {snr} | {row['mean_semantic_sketch_cosine']:.6f} | "
            f"{row['relative_first2_mse_increase']:+.4%} | "
            f"{'PASS' if row['sketch_gate_pass'] else 'FAIL'} | "
            f"{'PASS' if row['structure_gate_pass'] else 'FAIL'} |"
        )
    lines.extend(["", "Total rate remains `c=6+c=2=c=8`; official Imagenette validation was not accessed."])
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((output_dir / "REPORT.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
