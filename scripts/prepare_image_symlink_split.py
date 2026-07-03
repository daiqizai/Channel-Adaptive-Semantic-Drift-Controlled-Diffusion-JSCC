from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a reproducible symlink train/val split from an image directory.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--val-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def link_split(paths: list[Path], split_root: Path) -> list[str]:
    split_root.mkdir(parents=True, exist_ok=True)
    relative_targets = []
    for index, source in enumerate(paths):
        link_path = split_root / f"{index:06d}_{source.name}"
        link_path.symlink_to(source.resolve())
        relative_targets.append(str(source.relative_to(PROJECT_ROOT)))
    return relative_targets


def main() -> None:
    args = parse_args()
    source_root = resolve_project_path(args.source_root)
    output_root = resolve_project_path(args.output_root)

    if not source_root.exists():
        raise FileNotFoundError(f"Source directory not found: {source_root}")
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {output_root}")
        shutil.rmtree(output_root)

    paths = sorted(path for path in source_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    required = args.train_size + args.val_size
    if len(paths) < required:
        raise ValueError(f"Need {required} images, found {len(paths)} under {source_root}")

    rng = random.Random(args.seed)
    shuffled = paths[:]
    rng.shuffle(shuffled)
    train_paths = sorted(shuffled[: args.train_size])
    val_paths = sorted(shuffled[args.train_size : required])

    manifest = {
        "source_root": str(source_root.relative_to(PROJECT_ROOT)),
        "output_root": str(output_root.relative_to(PROJECT_ROOT)),
        "seed": args.seed,
        "train_size": len(train_paths),
        "val_size": len(val_paths),
        "train": link_split(train_paths, output_root / "train"),
        "val": link_split(val_paths, output_root / "val"),
    }
    with (output_root / "split_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps({key: manifest[key] for key in ["source_root", "output_root", "seed", "train_size", "val_size"]}, indent=2))


if __name__ == "__main__":
    main()
