from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cadsd_jscc.deepjscc_adapter import load_deepjscc_model
from cadsd_jscc.metrics import psnr


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().clamp(0, 1)
    array = (tensor.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    return Image.fromarray(array)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for the third-party DeepJSCC baseline.")
    parser.add_argument("--config", default="configs/s1_deepjscc_cifar10_awgn.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = PROJECT_ROOT / args.config
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    device = torch.device(args.device if args.device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(int(config["seed"]))

    output_dir = PROJECT_ROOT / config["outputs"]["smoke_test"]
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = int(config["image_size"])
    images = torch.rand(args.batch_size, 3, image_size, image_size, device=device)

    rows = []
    first_reconstruction_saved = False
    for snr in config["snr_sweep_db"]:
        model = load_deepjscc_model(
            repo_root=PROJECT_ROOT / config["baseline"]["repo"],
            checkpoint_path=PROJECT_ROOT / config["baseline"]["checkpoint"],
            inner_channel=int(config["inner_channel"]),
            channel=str(config["channel"]),
            snr=float(snr),
            device=device,
        )
        with torch.no_grad():
            reconstructed = model(images).clamp(0, 1)
        rows.append(
            {
                "snr_db": float(snr),
                "psnr_db": psnr(reconstructed, images, max_val=1.0),
            }
        )
        if not first_reconstruction_saved:
            tensor_to_image(images[0]).save(output_dir / "synthetic_input.png")
            tensor_to_image(reconstructed[0]).save(output_dir / "synthetic_reconstruction.png")
            first_reconstruction_saved = True

    result = {
        "note": "Smoke test only. These synthetic-image metrics are not experimental results.",
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "device": str(device),
        "rows": rows,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

