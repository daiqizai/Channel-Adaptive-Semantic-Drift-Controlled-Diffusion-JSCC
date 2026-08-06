#!/usr/bin/env python3
"""Run A0 identity sanity for PSNR/MS-SSIM/LPIPS/DISTS/FID/KID."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from cleanfid import fid
from PIL import Image
from torchmetrics.image.dists import DeepImageStructureAndTextureSimilarity
from torchvision.transforms.functional import pil_to_tensor


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cadsd_jscc.metrics import ms_ssim_per_sample, psnr_per_sample  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def load_source_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_rgb_tensor(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        tensor = pil_to_tensor(image.convert("RGB")).float().div_(255.0)
    return tensor.unsqueeze(0).to(device)


def find_metric_weights() -> list[dict[str, Any]]:
    candidates: list[Path] = []
    for root in (
        Path.home() / ".cache" / "torch" / "hub" / "checkpoints",
        Path.home() / ".cache" / "torch" / "hub",
    ):
        if root.exists():
            for path in root.rglob("*"):
                lower = path.name.lower()
                if path.is_file() and any(
                    token in lower for token in ("vgg16", "inception", "pt_inception")
                ):
                    candidates.append(path)
    unique = sorted(set(path.resolve() for path in candidates))
    return [
        {
            "path": relative(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in unique
    ]


def link_metric_weights(config: dict[str, Any]) -> list[dict[str, Any]]:
    download = config["metrics"]["download"]
    root = resolve(download["metric_weights_root"])
    entries = {entry["name"]: entry for entry in download["metric_weights"]}
    vgg = root / entries["torchvision_vgg16"]["filename"]
    inception = root / entries["clean_fid_inception"]["filename"]
    for path, entry in (
        (vgg, entries["torchvision_vgg16"]),
        (inception, entries["clean_fid_inception"]),
    ):
        if not path.is_file() or path.stat().st_size != int(entry["expected_bytes"]):
            raise RuntimeError(f"metric weight is missing or incomplete: {path}")

    links = [
        (Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / vgg.name, vgg),
        (Path("/tmp") / inception.name, inception),
    ]
    rows: list[dict[str, Any]] = []
    for link, target in links:
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.exists() or link.is_symlink():
            if link.resolve() != target.resolve():
                if link.is_file() and sha256_file(link) == sha256_file(target):
                    pass
                else:
                    raise RuntimeError(f"refuse to replace existing metric cache: {link}")
        else:
            link.symlink_to(target.resolve())
        rows.append(
            {
                "workspace_path": relative(target),
                "cache_link": str(link),
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = resolve(config["analysis"]["output_dir"])
    state_path = output / "STATE.json"
    if not state_path.is_file():
        raise FileNotFoundError("prepare_gate_a0 must run first")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") not in {"prepared", "failed"}:
        raise RuntimeError(f"unexpected A0 state: {state}")
    source_rows = load_source_rows(output / "manifests" / "source_images.csv")
    kodak = [row for row in source_rows if row["dataset"] == "kodak"][:4]
    if len(kodak) != 4:
        raise RuntimeError("identity sanity requires four Kodak images")

    metric_weight_links = link_metric_weights(config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dists_metric = DeepImageStructureAndTextureSimilarity(reduction="mean").to(device)
    import lpips

    lpips_metric = lpips.LPIPS(net="alex").to(device).eval()
    per_image: list[dict[str, Any]] = []
    with torch.no_grad():
        for row in kodak:
            tensor = load_rgb_tensor(resolve(row["path"]), device)
            psnr = float(psnr_per_sample(tensor, tensor).item())
            ms_ssim = float(ms_ssim_per_sample(tensor, tensor).item())
            lpips_value = float(lpips_metric(tensor * 2 - 1, tensor * 2 - 1).item())
            dists_value = float(dists_metric(tensor, tensor).item())
            per_image.append(
                {
                    "sample_id": row["sample_id"],
                    "psnr": psnr,
                    "ms_ssim": ms_ssim,
                    "lpips": lpips_value,
                    "dists": dists_value,
                }
            )
    per_image_path = output / "identity_per_image.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)

    clic_root = resolve(config["datasets"]["clic2020_test"]["root"])
    clean_mode = config["metrics"]["distribution"]["clean_fid_mode"]
    fid_value = float(
        fid.compute_fid(
            str(clic_root),
            str(clic_root),
            mode=clean_mode,
            num_workers=4,
            batch_size=16,
            device=device,
            verbose=True,
            use_dataparallel=False,
        )
    )
    kid_value = float(
        fid.compute_kid(
            str(clic_root),
            str(clic_root),
            mode=clean_mode,
            num_workers=4,
            batch_size=16,
            device=device,
            verbose=True,
            use_dataparallel=False,
        )
    )

    thresholds = config["metrics"]["identity_thresholds"]
    checks = {
        # The shared metric intentionally clamps MSE at 1e-12, so exact
        # identity is reported as 120 dB rather than positive infinity.
        "psnr_identity": min(row["psnr"] for row in per_image)
        >= float(thresholds["psnr_identity_min_db"]),
        "ms_ssim_identity": max(
            abs(1.0 - row["ms_ssim"]) for row in per_image
        )
        <= float(thresholds["ms_ssim_abs_from_one_max"]),
        "lpips_identity": max(abs(row["lpips"]) for row in per_image)
        <= float(thresholds["lpips_abs_max"]),
        "dists_identity": max(abs(row["dists"]) for row in per_image)
        <= float(thresholds["dists_abs_max"]),
        "fid_identity": abs(fid_value) <= float(thresholds["fid_abs_max"]),
        "kid_identity": abs(kid_value) <= float(thresholds["kid_abs_max"]),
    }
    summary = {
        "analysis_id": config["analysis"]["id"],
        "status": "complete" if all(checks.values()) else "failed",
        "device": str(device),
        "platform": platform.platform(),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchmetrics": package_version("torchmetrics"),
            "lpips": package_version("lpips"),
            "clean_fid": package_version("clean-fid"),
        },
        "identity_samples": [row["sample_id"] for row in per_image],
        "identity_values": {
            "psnr": [row["psnr"] for row in per_image],
            "ms_ssim": [row["ms_ssim"] for row in per_image],
            "lpips": [row["lpips"] for row in per_image],
            "dists": [row["dists"] for row in per_image],
            "fid_clic_self": fid_value,
            "kid_clic_self": kid_value,
        },
        "checks": checks,
        "metric_weights": metric_weight_links,
        "environment_proxy_keys_present": sorted(
            key for key in os.environ if key.lower() in {"http_proxy", "https_proxy", "all_proxy"}
        ),
        "method_inference_run": False,
        "official_imagenette_validation_accessed": False,
        "per_image_csv": relative(per_image_path),
        "per_image_csv_sha256": sha256_file(per_image_path),
        "script_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(config_path),
    }
    summary_path = output / "identity_sanity_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    state_path.write_text(
        json.dumps(
            {
                "analysis_id": config["analysis"]["id"],
                "status": summary["status"],
                "a1_authorized": False,
                "summary_sha256": sha256_file(summary_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "complete":
        raise RuntimeError(f"identity sanity failed: {checks}")


if __name__ == "__main__":
    main()
