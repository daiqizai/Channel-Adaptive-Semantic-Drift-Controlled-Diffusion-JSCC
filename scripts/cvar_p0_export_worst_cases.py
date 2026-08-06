#!/usr/bin/env python3
"""P3 worst-case reconstruction export (task book figure 5).

For each SNR and a handful of source images, replays the frozen backbone on the
median, worst-10% and worst channel realization of that image and writes one
labelled comparison strip per case.  Realizations are replayed from
``diagnostic_samples.csv``, so what is rendered is exactly what was measured.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from cadsd_jscc.external_common import canonical_standard_normal  # noqa: E402
from cadsd_jscc.tail_risk import (  # noqa: E402
    apply_block_fading_channel,
    block_fading_coefficient,
    effective_snr_db,
)
from s32_strong_jscc_external_comparison import build_model  # noqa: E402

FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cvar_p0_tail_risk_diagnostic.yaml")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def to_pil(tensor: torch.Tensor) -> Image.Image:
    array = (tensor.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu()
    return Image.fromarray(array.permute(1, 2, 0).numpy())


def label_strip(
    panels: list[tuple[Image.Image, list[str]]], caption: str
) -> Image.Image:
    width = sum(panel.width for panel, _ in panels)
    height = panels[0][0].height
    header, footer = 22, 46
    canvas = Image.new("RGB", (width, height + header + footer), "white")
    try:
        font = ImageFont.truetype(str(FONT_PATH), 13)
        small = ImageFont.truetype(str(FONT_PATH), 11)
    except OSError:
        font = small = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), caption, fill="black", font=font)
    offset = 0
    for panel, lines in panels:
        canvas.paste(panel, (offset, header))
        for index, line in enumerate(lines):
            draw.text(
                (offset + 6, height + header + 2 + index * 13),
                line,
                fill="black",
                font=small,
            )
        offset += panel.width
    return canvas


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    directory = resolve(config["outputs"]["directory"])
    rows = list(csv.DictReader((directory / "diagnostic_samples.csv").open(encoding="utf-8")))
    metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
    verdict = json.loads((directory / "verdict.json").read_text(encoding="utf-8"))
    out = directory / "worst_examples"
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or config["runtime"]["device"])
    checkpoint = torch.load(
        resolve(config["inputs"]["strong_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    model = build_model(checkpoint).to(device).eval().requires_grad_(False)
    transform = transforms.Compose(
        [
            transforms.Resize(model.image_size),
            transforms.CenterCrop(model.image_size),
            transforms.ToTensor(),
        ]
    )
    latent_shape = (model.latent_channels, model.image_size // 16, model.image_size // 16)
    clamp_low, clamp_high = (float(value) for value in config["decoder_snr_clamp_db"])
    base_seed = int(config["diagnostic"]["base_seed"])
    real_symbols = int(config["diagnostic"]["real_symbols"])
    per_snr = int(config["outputs"]["save_worst_examples_per_snr"])
    paths = {item["image_id"]: item["relative_path"] for item in metadata["sources"]}

    grouped: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["arm"], float(row["snr_db"]), row["image_id"])].append(row)

    manifest: list[dict[str, Any]] = []
    for snr_key, arm in verdict["primary_arm_per_snr"].items():
        snr = float(snr_key)
        images = sorted({key[2] for key in grouped if key[0] == arm and key[1] == snr})
        # Rank images by how deep their own conditional tail is, then export the
        # deepest few.  Selection is descriptive only; the verdict is already fixed.
        ranked = sorted(
            images,
            key=lambda image: (
                statistics.median([float(r["psnr"]) for r in grouped[(arm, snr, image)]])
                - min(float(r["psnr"]) for r in grouped[(arm, snr, image)])
            ),
            reverse=True,
        )
        for image in ranked[:per_snr]:
            records = sorted(grouped[(arm, snr, image)], key=lambda r: float(r["psnr"]))
            count = len(records)
            picks = {
                "worst": records[0],
                "worst10pct": records[max(0, round(0.10 * (count - 1)))],
                "median": records[count // 2],
            }
            target = transform(
                Image.open(resolve(paths[image])).convert("RGB")
            ).unsqueeze(0).to(device)
            panels: list[tuple[Image.Image, list[str]]] = [
                (to_pil(target[0]), ["original", f"{image}"])
            ]
            for name in ("median", "worst10pct", "worst"):
                record = picks[name]
                realization = int(record["realization_id"])
                noise = canonical_standard_normal(
                    base_seed, f"{image}|r{realization}", snr, real_symbols
                ).reshape(latent_shape).unsqueeze(0).to(device)
                h_real_value, h_imag_value = block_fading_coefficient(
                    base_seed, image, snr, realization
                )
                h_power = h_real_value**2 + h_imag_value**2
                mode = next(
                    str(item["decoder_snr"])
                    for item in config["arms"]
                    if str(item["name"]) == arm
                )
                if mode == "effective":
                    decoder_snr = effective_snr_db(snr, h_power)
                elif mode == "effective_clamped":
                    decoder_snr = min(
                        max(effective_snr_db(snr, h_power), clamp_low), clamp_high
                    )
                else:
                    decoder_snr = snr
                with torch.inference_mode():
                    latent = model.encode(target, snr)
                    transmitted, _ = model.normalize_channel_input(latent)
                    received = apply_block_fading_channel(
                        transmitted,
                        noise,
                        snr,
                        torch.tensor([h_real_value], dtype=torch.float32),
                        torch.tensor([h_imag_value], dtype=torch.float32),
                        epsilon=float(config["diagnostic"]["equalization_epsilon"]),
                    )
                    reconstruction = model.decode(
                        received,
                        torch.tensor([decoder_snr], dtype=torch.float32, device=device),
                    )
                    reconstruction = (
                        torch.floor(reconstruction.clamp(0.0, 1.0) * 255.0) / 255.0
                    )
                replayed = float(
                    10.0
                    * torch.log10(
                        1.0
                        / (reconstruction - target).square().mean().clamp_min(1e-12)
                    )
                )
                recorded = float(record["psnr"])
                if abs(replayed - recorded) > 0.01:
                    raise RuntimeError(
                        f"replay mismatch for {image}@{snr}: {replayed} vs {recorded}"
                    )
                panels.append(
                    (
                        to_pil(reconstruction[0]),
                        [
                            f"{name} r={realization}",
                            f"|h|^2={h_power:.4f} psnr={recorded:.2f}dB",
                        ],
                    )
                )
            caption = (
                f"{arm}  snr={snr:g}dB  image={image}  "
                f"median-worst spread={float(records[-1]['psnr']) - float(records[0]['psnr']):.2f}dB"
            )
            name = (
                f"snr{snr:g}dB_{image}_worst_r{int(picks['worst']['realization_id'])}"
                f"_h{float(picks['worst']['h_power']):.4f}"
                f"_psnr{float(picks['worst']['psnr']):.2f}.png"
            )
            label_strip(panels, caption).save(out / name)
            manifest.append(
                {
                    "file": name,
                    "arm": arm,
                    "snr_db": snr,
                    "image_id": image,
                    "worst_realization": int(picks["worst"]["realization_id"]),
                    "worst_h_power": float(picks["worst"]["h_power"]),
                    "worst_psnr": float(picks["worst"]["psnr"]),
                    "median_psnr": float(picks["median"]["psnr"]),
                }
            )
            print(f"wrote {name}", flush=True)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"exported {len(manifest)} worst-case strips to {out}")


if __name__ == "__main__":
    main()
