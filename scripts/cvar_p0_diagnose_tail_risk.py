#!/usr/bin/env python3
"""P0/P2 conditional channel tail-risk diagnostic for the CVaR candidate direction.

Replays the frozen S33B strong JSCC backbone over many independent channel
realizations of the *same* source image, so that channel-induced tail failure
can be separated from image-content difficulty.

Three exactly-paired arms share one encoder pass and one standard-normal noise
draw per (image, snr, realization):

``awgn_control``
    ``h = 1``.  Any tail here exists without fading at all.
``rayleigh_nominal_csi``
    Block-fading ``h`` with zero-forcing equalization, decoder conditioned on
    the *nominal* SNR.  This is the naive deployment.
``rayleigh_effective_csi``
    Same realization, decoder conditioned on the post-equalization effective
    SNR ``nominal + 10*log10(|h|^2)``, which a receiver with perfect CSI can
    always compute.  This is the fair deployment and the arm the GO/NO-GO
    decision is based on.

The encoder is always conditioned on the nominal SNR: block fading without a
feedback link means the transmitter cannot know ``h``.

No training and no checkpoint selection happen here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from cadsd_jscc.datasets import IMAGE_SUFFIXES  # noqa: E402
from cadsd_jscc.external_common import (  # noqa: E402
    canonical_standard_normal,
)
from cadsd_jscc.metrics import ms_ssim_per_sample, psnr_per_sample  # noqa: E402
from cadsd_jscc.tail_risk import (  # noqa: E402
    apply_block_fading_channel,
    block_fading_coefficient,
    effective_snr_db,
)
from s32_strong_jscc_external_comparison import build_model  # noqa: E402
from s5_residual_refiner_pilot import try_load_lpips  # noqa: E402


SAMPLE_FIELDS = [
    "arm",
    "image_id",
    "image_relative_path",
    "snr_db",
    "realization_id",
    "fading_seed_material",
    "h_real",
    "h_imag",
    "h_power",
    "effective_snr_db",
    "decoder_snr_db",
    "mse",
    "psnr",
    "ms_ssim",
    "lpips",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cvar_p0_tail_risk_diagnostic.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--realization-chunk",
        type=int,
        default=None,
        help="override runtime.realization_chunk; results are batch-invariant",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="append to the output directory name, for batch-invariance checks",
    )
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - commit hash is metadata, not a gate.
        return "unknown"


def select_sources(config: dict[str, Any], count: int) -> list[dict[str, Any]]:
    """Fresh COCO val2017 images, disjoint from the S33 checkpoint-selection subset."""

    root = resolve(config["inputs"]["coco_val_root"])
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"no images under {root}")

    selection = config["selection"]
    excluded: set[int] = set()
    if bool(selection["exclude_s33_val_subset"]):
        subset_size = min(int(selection["s33_val_subset_size"]), len(paths))
        generator = torch.Generator().manual_seed(int(selection["s33_val_subset_seed"]))
        excluded = set(torch.randperm(len(paths), generator=generator)[:subset_size].tolist())

    candidates = [
        {"index": index, "path": path}
        for index, path in enumerate(paths)
        if index not in excluded
    ]
    for item in candidates:
        item["content_sha256"] = sha256_file(item["path"])
    candidates.sort(key=lambda item: item["content_sha256"])
    chosen = candidates[:count]
    if len(chosen) != count:
        raise RuntimeError(f"needed {count} sources, found {len(chosen)}")
    for item in chosen:
        item["image_id"] = item["path"].stem
        item["relative_path"] = str(item["path"].relative_to(resolve(".")))
    return chosen


def quantize(images: torch.Tensor) -> torch.Tensor:
    """floor-uint8 output convention used by every other stage in this repo."""

    return torch.floor(images.clamp(0.0, 1.0) * 255.0) / 255.0


def run(
    config: dict[str, Any],
    config_path: Path,
    dry_run: bool,
    device_arg: str | None,
    chunk_override: int | None = None,
    output_suffix: str = "",
) -> Path:
    protocol = config["protocol"]
    if protocol["official_imagenette_validation_accessed"] is not False:
        raise RuntimeError("official validation must remain sealed")
    if protocol["no_training"] is not True:
        raise RuntimeError("this entry point must never train")

    diagnostic = dict(config["diagnostic"])
    runtime = config["runtime"]
    if dry_run:
        overrides = runtime["dry_run"]
        source_count = int(overrides["source_count"])
        diagnostic["num_channel_realizations"] = int(overrides["num_channel_realizations"])
        diagnostic["snrs_db"] = [float(value) for value in overrides["snrs_db"]]
        output = resolve(str(config["outputs"]["dry_run_directory"]) + output_suffix)
    else:
        source_count = int(config["selection"]["source_count"])
        output = resolve(str(config["outputs"]["directory"]) + output_suffix)

    if output.exists() and bool(config["outputs"]["overwrite_forbidden"]):
        raise FileExistsError(f"refusing to overwrite {output}")
    (output / "plots").mkdir(parents=True, exist_ok=True)
    (output / "worst_examples").mkdir(parents=True, exist_ok=True)

    device = torch.device(device_arg or runtime["device"])
    checkpoint_path = resolve(config["inputs"]["strong_checkpoint"])
    actual_sha = sha256_file(checkpoint_path)
    expected_sha = str(config["inputs"]["strong_checkpoint_sha256"])
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"checkpoint SHA mismatch: expected {expected_sha}, got {actual_sha}"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint).to(device).eval().requires_grad_(False)
    real_symbols = int(diagnostic["real_symbols"])
    if model.real_symbols != real_symbols:
        raise RuntimeError(
            f"model has {model.real_symbols} real symbols, config says {real_symbols}"
        )
    latent_shape = (model.latent_channels, model.image_size // 16, model.image_size // 16)

    lpips_model, lpips_error = try_load_lpips(device, resolve("outputs/cache"))
    if lpips_model is None:
        if bool(config["metrics"]["lpips_required"]):
            raise RuntimeError(f"LPIPS is required but unavailable: {lpips_error}")
    else:
        lpips_model.eval().requires_grad_(False)

    sources = select_sources(config, source_count)
    transform = transforms.Compose(
        [
            transforms.Resize(model.image_size),
            transforms.CenterCrop(model.image_size),
            transforms.ToTensor(),
        ]
    )

    arms = config["arms"]
    base_seed = int(diagnostic["base_seed"])
    realizations = int(diagnostic["num_channel_realizations"])
    chunk = int(chunk_override or runtime["realization_chunk"])
    epsilon = float(diagnostic["equalization_epsilon"])
    snrs = [float(value) for value in diagnostic["snrs_db"]]
    clamp_low, clamp_high = (float(value) for value in config["decoder_snr_clamp_db"])

    samples_path = output / "diagnostic_samples.csv"
    handle = samples_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS)
    writer.writeheader()

    started = time.time()
    total_rows = 0
    for source_position, source in enumerate(sources):
        target = transform(Image.open(source["path"]).convert("RGB")).unsqueeze(0).to(device)
        for snr in snrs:
            with torch.inference_mode():
                latent = model.encode(target, snr)
                transmitted, _ = model.normalize_channel_input(latent)

            for start in range(0, realizations, chunk):
                stop = min(start + chunk, realizations)
                span = stop - start
                noises = []
                coefficients = []
                for realization in range(start, stop):
                    noise = canonical_standard_normal(
                        base_seed,
                        f"{source['image_id']}|r{realization}",
                        snr,
                        real_symbols,
                    )
                    noises.append(noise.reshape(latent_shape))
                    coefficients.append(
                        block_fading_coefficient(
                            base_seed, source["image_id"], snr, realization
                        )
                    )
                noise_batch = torch.stack(noises).to(device)
                batch_transmitted = transmitted.expand(span, *transmitted.shape[1:])
                batch_target = target.expand(span, *target.shape[1:])
                h_real = torch.tensor([item[0] for item in coefficients], dtype=torch.float32)
                h_imag = torch.tensor([item[1] for item in coefficients], dtype=torch.float32)
                h_power = (h_real.square() + h_imag.square()).tolist()
                effective = [effective_snr_db(snr, value) for value in h_power]

                for arm in arms:
                    fading = bool(arm["fading"])
                    if fading:
                        arm_h_real, arm_h_imag = h_real, h_imag
                    else:
                        arm_h_real = torch.ones(span, dtype=torch.float32)
                        arm_h_imag = torch.zeros(span, dtype=torch.float32)
                    decoder_snr_mode = str(arm["decoder_snr"])
                    if decoder_snr_mode == "effective":
                        decoder_snr = torch.tensor(effective, dtype=torch.float32, device=device)
                    elif decoder_snr_mode == "effective_clamped":
                        decoder_snr = torch.tensor(
                            effective, dtype=torch.float32, device=device
                        ).clamp(clamp_low, clamp_high)
                    elif decoder_snr_mode == "nominal":
                        decoder_snr = torch.full(
                            (span,), float(snr), dtype=torch.float32, device=device
                        )
                    else:
                        raise ValueError(f"unknown decoder_snr mode: {decoder_snr_mode}")

                    with torch.inference_mode():
                        received = apply_block_fading_channel(
                            batch_transmitted,
                            noise_batch,
                            snr,
                            arm_h_real,
                            arm_h_imag,
                            epsilon=epsilon,
                        )
                        reconstruction = quantize(model.decode(received, decoder_snr))
                        mse = (
                            (reconstruction - batch_target)
                            .square()
                            .flatten(start_dim=1)
                            .mean(dim=1)
                        )
                        psnr = psnr_per_sample(reconstruction, batch_target)
                        ms_ssim = ms_ssim_per_sample(reconstruction, batch_target)
                        if lpips_model is not None:
                            lpips = lpips_model(
                                reconstruction * 2.0 - 1.0, batch_target * 2.0 - 1.0
                            ).flatten()
                        else:
                            lpips = torch.full((span,), float("nan"))

                    for offset in range(span):
                        realization = start + offset
                        writer.writerow(
                            {
                                "arm": str(arm["name"]),
                                "image_id": source["image_id"],
                                "image_relative_path": source["relative_path"],
                                "snr_db": f"{snr:.6f}",
                                "realization_id": realization,
                                "fading_seed_material": (
                                    f"{base_seed}|{source['image_id']}|{snr:.6f}|{realization}"
                                ),
                                "h_real": f"{float(arm_h_real[offset]):.9f}",
                                "h_imag": f"{float(arm_h_imag[offset]):.9f}",
                                "h_power": (
                                    f"{h_power[offset]:.9f}" if fading else "1.000000000"
                                ),
                                "effective_snr_db": (
                                    f"{effective[offset]:.6f}" if fading else f"{snr:.6f}"
                                ),
                                "decoder_snr_db": f"{float(decoder_snr[offset]):.6f}",
                                "mse": f"{float(mse[offset]):.10f}",
                                "psnr": f"{float(psnr[offset]):.6f}",
                                "ms_ssim": f"{float(ms_ssim[offset]):.6f}",
                                "lpips": f"{float(lpips[offset]):.6f}",
                            }
                        )
                        total_rows += 1
        if (source_position + 1) % 10 == 0 or source_position + 1 == len(sources):
            elapsed = time.time() - started
            print(
                f"[{source_position + 1}/{len(sources)}] rows={total_rows} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
    handle.close()

    metadata = {
        "analysis_id": config["analysis_id"],
        "dry_run": dry_run,
        "git_commit": git_commit(),
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256_file(config_path),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": actual_sha,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "source_count": len(sources),
        "num_channel_realizations": realizations,
        "realization_chunk": chunk,
        "snrs_db": snrs,
        "base_seed": base_seed,
        "arms": [str(arm["name"]) for arm in arms],
        "rows": total_rows,
        "lpips_available": lpips_model is not None,
        "lpips_error": None if lpips_model is not None else lpips_error,
        "elapsed_seconds": time.time() - started,
        "sources": [
            {
                "image_id": item["image_id"],
                "relative_path": item["relative_path"],
                "content_sha256": item["content_sha256"],
            }
            for item in sources
        ],
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (output / "config_copy.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    print(f"wrote {total_rows} rows to {samples_path}")
    return output


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config = load_yaml(config_path)
    run(
        config,
        config_path,
        args.dry_run,
        args.device,
        args.realization_chunk,
        args.output_suffix,
    )


if __name__ == "__main__":
    main()
