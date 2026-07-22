#!/usr/bin/env python3
"""Materialize the preregistered fresh train/selection/holdout population for S19."""

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
    rel = path.resolve().relative_to(source_root.resolve())
    return hashlib.sha256(f"{seed}:{rel}".encode("utf-8")).digest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/s19_diffusion_fusion_ablation.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["protocol"]["status"] != "preregistered_before_population_materialization":
        raise RuntimeError("S19 population contract is not in its preregistered state")
    source_root = resolve(config["inputs"]["source_root"])
    val_root = resolve(config["inputs"]["coco_val_exclusion_root"])
    excluded_paths: set[str] = set()
    excluded_hashes: set[str] = set()
    exclusion_counts: dict[str, int] = {}
    for item in config["inputs"]["exclusion_manifests"]:
        manifest = resolve(item["path"])
        if sha256_file(manifest) != str(item["sha256"]):
            raise RuntimeError(f"exclusion manifest hash mismatch: {manifest}")
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        exclusion_counts[relative(manifest)] = len(rows)
        excluded_paths.update(str(row["source_path"]) for row in rows)
        excluded_hashes.update(str(row["source_sha256"]) for row in rows)
    val_names = {path.name for path in val_root.iterdir() if path.is_file()}
    candidates = sorted(
        path
        for path in source_root.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        and relative(path) not in excluded_paths
        and path.name not in val_names
    )
    total = int(config["population"]["total_count"])
    ranked_candidates = sorted(
        candidates,
        key=lambda path: (rank(path, source_root, int(config["population"]["rank_seed"])), str(path)),
    )
    selected: list[tuple[Path, str]] = []
    selected_hashes: set[str] = set()
    skipped_excluded_sha = 0
    skipped_within_population_sha = 0
    for path in ranked_candidates:
        digest = sha256_file(path)
        if digest in excluded_hashes:
            skipped_excluded_sha += 1
            continue
        if digest in selected_hashes:
            skipped_within_population_sha += 1
            continue
        selected.append((path, digest))
        selected_hashes.add(digest)
        if len(selected) == total:
            break
    if len(selected) != total:
        raise RuntimeError("fresh population is shorter than its frozen count")
    roles = {str(key): int(value) for key, value in config["population"]["roles"].items()}
    if sum(roles.values()) != total:
        raise RuntimeError("population role counts do not sum to total_count")
    boundaries: list[tuple[str, int]] = []
    cumulative = 0
    for role, count in roles.items():
        cumulative += count
        boundaries.append((role, cumulative))
    plan = {
        "available_after_path_and_name_exclusion": len(candidates),
        "exclusion_manifest_counts": exclusion_counts,
        "excluded_unique_paths": len(excluded_paths),
        "excluded_unique_hashes": len(excluded_hashes),
        "rank_seed": int(config["population"]["rank_seed"]),
        "roles": roles,
        "skipped_excluded_sha": skipped_excluded_sha,
        "skipped_within_population_sha": skipped_within_population_sha,
        "first_ranked_source": relative(selected[0][0]),
        "last_ranked_source": relative(selected[-1][0]),
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, "plan": plan}, ensure_ascii=False, indent=2))
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
    for index, (path, digest) in enumerate(selected):
        role = next(role_name for role_name, end in boundaries if index < end)
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
    shutil.copy2(config_path, output / "config_before_population_freeze.yaml")
    shutil.copy2(SCRIPT, output / SCRIPT.name)
    write_csv(output / "source_manifest.csv", records)
    manifest_sha = sha256_file(output / "source_manifest.csv")
    metadata = {
        **plan,
        "source_manifest_sha256": manifest_sha,
        "records": len(records),
        "role_records": {role: sum(row["role"] == role for row in records) for role in roles},
        "excluded_path_overlap": sum(row["source_path"] in excluded_paths for row in records),
        "excluded_sha_overlap": sum(row["source_sha256"] in excluded_hashes for row in records),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "official_imagenette_accessed": False,
        "download_note": "No download; local COCO train2017 only.",
    }
    (output / "population_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "STATE.json").write_text(
        json.dumps({"state": "POPULATION_COMPLETE", **metadata}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
