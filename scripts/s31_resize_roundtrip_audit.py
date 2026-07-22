#!/usr/bin/env python3
"""Reproduce the source-only resize round-trip diagnostic used after S30."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml
from PIL import Image
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cadsd_jscc.metrics import ms_ssim_per_sample, psnr_per_sample  # noqa: E402


CONFIG = ROOT / "configs" / "s31_diffjscc_resize_roundtrip_audit.yaml"
SCRIPT = Path(__file__).resolve()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(value: str | Path, expected: str) -> Path:
    path = resolve(value)
    if not path.is_file() or sha256_file(path) != str(expected):
        raise RuntimeError(f"missing or changed frozen input: {path}")
    return path


def main() -> None:
    config: dict[str, Any] = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    output = resolve(config["output_dir"])
    if output.exists():
        raise FileExistsError(output)
    reference = yaml.safe_load(
        require_sha(
            config["inputs"]["population_reference"],
            config["inputs"]["population_reference_sha256"],
        ).read_text(encoding="utf-8")
    )
    manifest = json.loads(
        require_sha(
            config["inputs"]["split_manifest"],
            config["inputs"]["split_manifest_sha256"],
        ).read_text(encoding="utf-8")
    )
    by_id = {str(row["sample_id"]): row for row in manifest["samples"]}
    source_root = resolve(manifest["source_train_root"])
    source_transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()]
    )
    rows: list[dict[str, Any]] = []
    for frozen in reference["population"]["samples"]:
        sample_id = str(frozen["sample_id"])
        source_path = source_root / str(by_id[sample_id]["relative_path"])
        if sha256_file(source_path) != str(frozen["content_sha256"]):
            raise RuntimeError(f"source changed: {sample_id}")
        source = source_transform(Image.open(source_path).convert("RGB"))
        source_pil = transforms.ToPILImage()(source)
        roundtrip_pil = source_pil.resize(
            (512, 512), Image.Resampling.BICUBIC
        ).resize((256, 256), Image.Resampling.LANCZOS)
        roundtrip = transforms.ToTensor()(roundtrip_pil)
        rows.append(
            {
                "sample_id": sample_id,
                "content_sha256": frozen["content_sha256"],
                "psnr": float(psnr_per_sample(roundtrip[None], source[None])[0]),
                "ms_ssim": float(
                    ms_ssim_per_sample(roundtrip[None], source[None])[0]
                ),
            }
        )
    if len(rows) != int(config["expected_samples"]):
        raise RuntimeError("population count changed")
    output.mkdir(parents=True)
    shutil.copy2(CONFIG, output / "config_snapshot.yaml")
    shutil.copy2(SCRIPT, output / SCRIPT.name)
    with (output / "per_sample.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "analysis_id": config["analysis_id"],
        "samples": len(rows),
        "mean_psnr": sum(row["psnr"] for row in rows) / len(rows),
        "minimum_psnr": min(row["psnr"] for row in rows),
        "mean_ms_ssim": sum(row["ms_ssim"] for row in rows) / len(rows),
        "minimum_ms_ssim": min(row["ms_ssim"] for row in rows),
        "claim_boundary": config["claim_boundary"],
        "config_sha256": sha256_file(CONFIG),
        "script_sha256": sha256_file(SCRIPT),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
