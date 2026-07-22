#!/usr/bin/env python3
"""Materialize the preregistered fresh COCO population for S18."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()


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
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty population manifest")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rank(path: Path, source_root: Path, seed: int) -> bytes:
    relative_path = path.resolve().relative_to(source_root.resolve())
    return hashlib.sha256(f"{seed}:{relative_path}".encode("utf-8")).digest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/s18_snr_identity_envelope.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["protocol"]["status"] != "preregistered_before_fresh_population_materialization":
        raise RuntimeError("S18 population contract is not preregistered")
    source_root = resolve(config["inputs"]["source_root"])
    prior_manifest = resolve(config["inputs"]["prior_source_manifest"])
    val_root = resolve(config["inputs"]["coco_val_exclusion_root"])
    if sha256_file(prior_manifest) != str(config["inputs"]["prior_source_manifest_sha256"]):
        raise RuntimeError("prior source manifest hash mismatch")
    with prior_manifest.open("r", encoding="utf-8", newline="") as handle:
        prior_rows = list(csv.DictReader(handle))
    if len(prior_rows) != 11000:
        raise RuntimeError("prior manifest no longer contains the frozen 11,000 sources")
    prior_paths = {str(row["source_path"]) for row in prior_rows}
    prior_hashes = {str(row["source_sha256"]) for row in prior_rows}
    val_names = {path.name for path in val_root.iterdir() if path.is_file()}
    paths = sorted(
        path
        for path in source_root.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        and relative(path) not in prior_paths
        and path.name not in val_names
    )
    total = int(config["population"]["total_count"])
    ranked = sorted(
        paths,
        key=lambda path: (
            rank(path, source_root, int(config["population"]["rank_seed"])),
            str(path),
        ),
    )
    selected = ranked[:total]
    if len(selected) != total:
        raise RuntimeError("fresh population is shorter than its frozen count")
    plan = {
        "available_after_path_and_name_exclusion": len(paths),
        "prior_sources_excluded": len(prior_paths),
        "rank_seed": int(config["population"]["rank_seed"]),
        "selection_count": int(config["population"]["selection_count"]),
        "holdout_count": int(config["population"]["holdout_count"]),
        "first_ranked_source": relative(selected[0]),
        "last_ranked_source": relative(selected[-1]),
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, "plan": plan}, indent=2))
        return
    output = resolve(args.output_dir or config["outputs"]["population_dir"])
    if output.exists():
        raise FileExistsError(output)
    original_dir = output / "exports" / "original"
    original_dir.mkdir(parents=True)
    transform = transforms.Compose(
        [
            transforms.Resize(int(config["image_size"])),
            transforms.CenterCrop(int(config["image_size"])),
            transforms.ToTensor(),
        ]
    )
    records: list[dict[str, Any]] = []
    selection_count = int(config["population"]["selection_count"])
    for index, path in enumerate(selected):
        digest = sha256_file(path)
        if digest in prior_hashes:
            raise RuntimeError(f"fresh source overlaps prior SHA-256: {path}")
        role = (
            str(config["population"]["selection_role"])
            if index < selection_count
            else str(config["population"]["holdout_role"])
        )
        name = f"sample_{index:06d}.png"
        with Image.open(path) as image:
            tensor = transform(image.convert("RGB"))
        save_image(torch.round(tensor * 255.0) / 255.0, original_dir / name)
        records.append(
            {
                "population_rank": index,
                "sample": name,
                "role": role,
                "source_path": relative(path),
                "source_sha256": digest,
            }
        )
    output.mkdir(exist_ok=True)
    shutil.copy2(config_path, output / "config_before_manifest_freeze.yaml")
    shutil.copy2(SCRIPT, output / SCRIPT.name)
    write_csv(output / "source_manifest.csv", records)
    manifest_sha = sha256_file(output / "source_manifest.csv")
    metadata = {
        **plan,
        "source_manifest_sha256": manifest_sha,
        "records": len(records),
        "selection_records": sum(row["role"] == config["population"]["selection_role"] for row in records),
        "holdout_records": sum(row["role"] == config["population"]["holdout_role"] for row in records),
        "prior_path_overlap": sum(row["source_path"] in prior_paths for row in records),
        "prior_sha_overlap": sum(row["source_sha256"] in prior_hashes for row in records),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "official_imagenette_accessed": False,
        "download_note": "No download; local COCO train2017 only.",
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "STATE.json").write_text(
        json.dumps({"state": "COMPLETE", **metadata}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
