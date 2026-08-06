#!/usr/bin/env python3
"""Validate A0 datasets and emit source/native-processing/rate manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import tarfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# Import the existing canonical-noise module to prove that this workspace
# references the shared source tree instead of carrying a copied implementation.
from cadsd_jscc import external_common as shared_external_common  # noqa: E402,F401


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def require_frozen_references(config: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name, entry in config["frozen_external_references"].items():
        path = resolve(entry["path"])
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen reference: {path}")
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            raise RuntimeError(f"SHA mismatch for frozen reference {name}: {actual}")
        rows.append({"name": name, "path": relative(path), "sha256": actual})
    return rows


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ".archive_sha256"
    archive_sha = sha256_file(archive)
    if marker.is_file():
        if marker.read_text(encoding="utf-8").strip() != archive_sha:
            raise RuntimeError(f"existing extraction has another archive SHA: {destination}")
        return
    if any(destination.iterdir()):
        raise RuntimeError(f"refuse to merge into non-empty extraction dir: {destination}")
    with zipfile.ZipFile(archive) as handle:
        root = destination.resolve()
        for info in handle.infolist():
            target = (destination / info.filename).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"unsafe zip member: {info.filename}")
        handle.extractall(destination)
    marker.write_text(archive_sha + "\n", encoding="utf-8")


def image_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def source_rows(dataset: str, root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in image_files(root):
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
            bands = len(image.getbands())
        rows.append(
            {
                "dataset": dataset,
                "sample_id": f"{dataset}/{path.relative_to(root).as_posix()}",
                "path": relative(path),
                "width": width,
                "height": height,
                "stored_mode": mode,
                "stored_bands": bands,
                "evaluation_mode": "RGB",
                "source_real_dimensions": 3 * width * height,
                "file_bytes": path.stat().st_size,
                "content_sha256": sha256_file(path),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"refuse to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def s33_tiles(row: dict[str, Any], tile_size: int) -> Iterable[dict[str, Any]]:
    width, height = int(row["width"]), int(row["height"])
    for top in range(0, height, tile_size):
        for left in range(0, width, tile_size):
            valid_h = min(tile_size, height - top)
            valid_w = min(tile_size, width - left)
            yield {
                "dataset": row["dataset"],
                "sample_id": row["sample_id"],
                "tile_index": (top // tile_size) * math.ceil(width / tile_size)
                + left // tile_size,
                "left": left,
                "top": top,
                "valid_width": valid_w,
                "valid_height": valid_h,
                "transmitted_width": tile_size,
                "transmitted_height": tile_size,
                "pad_right": tile_size - valid_w,
                "pad_bottom": tile_size - valid_h,
            }


def sgd_positions(height: int, width: int, patch_size: int) -> list[tuple[int, int]]:
    stride_h = max(patch_size - (height % patch_size), 1) if height > patch_size else height
    stride_w = max(patch_size - (width % patch_size), 1) if width > patch_size else width
    positions: list[tuple[int, int]] = []
    for i in range((height - 1) // stride_h + 1):
        for j in range((width - 1) // stride_w + 1):
            top = max(0, i * stride_h)
            left = max(0, j * stride_w)
            bottom = min(top + patch_size, height)
            right = min(left + patch_size, width)
            if bottom - top < patch_size:
                top = max(0, bottom - patch_size)
            if right - left < patch_size:
                left = max(0, right - patch_size)
            positions.append((top, left))
    return positions


def diffjscc_internal_size(width: int, height: int, minimum: int, multiple: int) -> tuple[int, int, int, int]:
    if min(width, height) < minimum:
        ratio = minimum / min(width, height)
        resized_w = math.ceil(width * ratio)
        resized_h = math.ceil(height * ratio)
    else:
        resized_w, resized_h = width, height
    padded_w = math.ceil(resized_w / multiple) * multiple
    padded_h = math.ceil(resized_h / multiple) * multiple
    return resized_w, resized_h, padded_w, padded_h


def build_processing_manifests(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    comm = config["communication"]
    tile_size = int(comm["s33_tile_size"])
    real_per_tile = int(comm["s33_real_symbols_per_tile"])
    rate_rows: list[dict[str, Any]] = []
    tile_rows: list[dict[str, Any]] = []
    sgd_rows: list[dict[str, Any]] = []

    for source in rows:
        width, height = int(source["width"]), int(source["height"])
        source_dims = int(source["source_real_dimensions"])
        current_tiles = list(s33_tiles(source, tile_size))
        tile_rows.extend(current_tiles)
        tile_count = len(current_tiles)

        for method, symbols_per_tile in (
            ("s33_strong", int(comm["s33_real_symbols_per_tile"])),
            ("swinjscc_base_or_cm", int(comm["swin_real_symbols_per_tile"])),
        ):
            real_symbols = tile_count * symbols_per_tile
            complex_uses = real_symbols / int(comm["real_coordinates_per_complex_use"])
            rate_rows.append(
                {
                    "dataset": source["dataset"],
                    "sample_id": source["sample_id"],
                    "method": method,
                    "native_processing": f"nonoverlap_{tile_size}px_tiles_edge_pad",
                    "processing_units": tile_count,
                    "internal_width": math.ceil(width / tile_size) * tile_size,
                    "internal_height": math.ceil(height / tile_size) * tile_size,
                    "main_real_symbols": real_symbols,
                    "edge_real_symbols": 0,
                    "caption_real_symbols": 0,
                    "total_real_symbols_lower_bound": real_symbols,
                    "total_real_symbols_exact": real_symbols,
                    "actual_complex_channel_uses": complex_uses,
                    "actual_cbr": complex_uses / source_dims,
                    "rate_status": "exact_accounted",
                    "ranking_eligible": True,
                }
            )

        diff = comm["diffjscc"]
        resized_w, resized_h, padded_w, padded_h = diffjscc_internal_size(
            width,
            height,
            int(diff["minimum_short_edge"]),
            int(diff["pad_multiple"]),
        )
        diff_real = (
            int(diff["main_latent_channels"])
            * (padded_h // int(diff["channel_downsample"]))
            * (padded_w // int(diff["channel_downsample"]))
        )
        diff_complex = diff_real / int(comm["real_coordinates_per_complex_use"])
        rate_rows.append(
            {
                "dataset": source["dataset"],
                "sample_id": source["sample_id"],
                "method": "diffjscc_official_whole_frame",
                "native_processing": "official_auto_resize_min512_pad64_whole_frame",
                "processing_units": 1,
                "internal_width": padded_w,
                "internal_height": padded_h,
                "main_real_symbols": diff_real,
                "edge_real_symbols": 0,
                "caption_real_symbols": 0,
                "total_real_symbols_lower_bound": diff_real,
                "total_real_symbols_exact": diff_real,
                "actual_complex_channel_uses": diff_complex,
                "actual_cbr": diff_complex / source_dims,
                "rate_status": (
                    "preflight_formula_requires_A1_runtime_instrumentation_"
                    f"resize_{resized_w}x{resized_h}_pad_{padded_w}x{padded_h}"
                ),
                "ranking_eligible": False,
            }
        )
        rate_rows.append(
            {
                "dataset": source["dataset"],
                "sample_id": source["sample_id"],
                "method": "diffjscc_author_jscc_whole_frame",
                "native_processing": "official_auto_resize_min512_pad64_whole_frame",
                "processing_units": 1,
                "internal_width": padded_w,
                "internal_height": padded_h,
                "main_real_symbols": diff_real,
                "edge_real_symbols": 0,
                "caption_real_symbols": 0,
                "total_real_symbols_lower_bound": diff_real,
                "total_real_symbols_exact": diff_real,
                "actual_complex_channel_uses": diff_complex,
                "actual_cbr": diff_complex / source_dims,
                "rate_status": (
                    "preflight_formula_requires_A1_runtime_instrumentation_"
                    f"resize_{resized_w}x{resized_h}_pad_{padded_w}x{padded_h}"
                ),
                "ranking_eligible": False,
            }
        )

        sgd = comm["sgd_paper_upper"]
        patch_size = int(sgd["patch_size"])
        positions = sgd_positions(height, width, patch_size)
        for patch_index, (top, left) in enumerate(positions):
            sgd_rows.append(
                {
                    "dataset": source["dataset"],
                    "sample_id": source["sample_id"],
                    "patch_index": patch_index,
                    "left": left,
                    "top": top,
                    "width": patch_size,
                    "height": patch_size,
                    "main_real_symbols": int(sgd["main_real_symbols_per_patch"]),
                    "active_edge_real_symbols": int(
                        sgd["active_edge_real_symbols_per_patch"]
                    ),
                    "caption_accounting": sgd["caption_accounting"],
                }
            )
        sgd_main = len(positions) * int(sgd["main_real_symbols_per_patch"])
        sgd_edge = len(positions) * int(sgd["active_edge_real_symbols_per_patch"])
        lower = sgd_main + sgd_edge
        lower_complex = lower / int(comm["real_coordinates_per_complex_use"])
        rate_rows.append(
            {
                "dataset": source["dataset"],
                "sample_id": source["sample_id"],
                "method": "sgdjscc_released_paper_upper",
                "native_processing": "official_split_image_v2_128px_overlap",
                "processing_units": len(positions),
                "internal_width": "",
                "internal_height": "",
                "main_real_symbols": sgd_main,
                "edge_real_symbols": sgd_edge,
                "caption_real_symbols": "",
                "total_real_symbols_lower_bound": lower,
                "total_real_symbols_exact": "",
                "actual_complex_channel_uses": "",
                "actual_cbr": "",
                "rate_status": "lower_bound_only_perfect_sender_caption_unpriced",
                "ranking_eligible": False,
            }
        )

    return rate_rows, tile_rows, sgd_rows


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not config["analysis"]["forbid_method_inference"]:
        raise RuntimeError("A0 config must forbid method inference")
    output = resolve(config["analysis"]["output_dir"])
    state_path = output / "STATE.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") == "complete":
            raise RuntimeError(f"completed A0 output is immutable: {output}")
    output.mkdir(parents=True, exist_ok=True)

    frozen_rows = require_frozen_references(config)
    datasets = config["datasets"]
    archives = resolve(datasets["archives_dir"])
    archive_rows: list[dict[str, Any]] = []
    kodak_entry = datasets["kodak"]
    kodak_archive = resolve(kodak_entry["mirror_archive_path"])
    if not kodak_archive.is_file():
        raise FileNotFoundError(f"Kodak mirror archive missing: {kodak_archive}")
    expected_kodak_sizes = list(map(int, kodak_entry["expected_file_bytes"]))
    with tarfile.open(kodak_archive, "r:gz") as handle:
        kodak_members = [
            member
            for member in handle.getmembers()
            if "/PhotoCD_PCD0992/" in member.name and member.isfile()
        ]
    if sorted(member.size for member in kodak_members) != sorted(expected_kodak_sizes):
        raise RuntimeError("Kodak mirror members do not match official byte-size table")
    archive_rows.append(
        {
            "name": "kodak_public_mirror",
            "url": kodak_entry["mirror_archive_url"],
            "path": relative(kodak_archive),
            "bytes": kodak_archive.stat().st_size,
            "sha256": sha256_file(kodak_archive),
            "extracted_root": relative(resolve(kodak_entry["root"])),
        }
    )
    for entry in datasets["clic2020_test"]["archives"]:
        archive = archives / entry["filename"]
        if not archive.is_file():
            raise FileNotFoundError(f"download missing: {archive}")
        if archive.stat().st_size != int(entry["expected_bytes"]):
            raise RuntimeError(
                f"archive size mismatch for {archive}: {archive.stat().st_size}"
            )
        destination = resolve(datasets["clic2020_test"]["root"]) / entry["name"]
        safe_extract(archive, destination)
        archive_rows.append(
            {
                "name": entry["name"],
                "url": entry["url"],
                "path": relative(archive),
                "bytes": archive.stat().st_size,
                "sha256": sha256_file(archive),
                "extracted_root": relative(destination),
            }
        )

    all_rows = source_rows("kodak", resolve(datasets["kodak"]["root"]))
    all_rows += source_rows(
        "clic2020_test", resolve(datasets["clic2020_test"]["root"])
    )
    counts = Counter(row["dataset"] for row in all_rows)
    expected = {
        "kodak": int(datasets["kodak"]["count"]),
        "clic2020_test": int(datasets["clic2020_test"]["count"]),
    }
    if dict(counts) != expected:
        raise RuntimeError(f"image-count mismatch: actual={dict(counts)} expected={expected}")
    duplicate_hashes = [
        digest
        for digest, count in Counter(row["content_sha256"] for row in all_rows).items()
        if count > 1
    ]
    if duplicate_hashes:
        raise RuntimeError(f"duplicate source images detected: {len(duplicate_hashes)}")

    rate_rows, tile_rows, sgd_rows = build_processing_manifests(all_rows, config)
    manifests = output / "manifests"
    write_csv(manifests / "source_images.csv", all_rows)
    write_csv(manifests / "method_native_rate_ledger.csv", rate_rows)
    write_csv(manifests / "s33_swin_tile_manifest.csv", tile_rows)
    write_csv(manifests / "sgd_official_patch_manifest.csv", sgd_rows)
    write_csv(manifests / "frozen_external_references.csv", frozen_rows)
    write_csv(manifests / "download_archives.csv", archive_rows)

    summary = {
        "analysis_id": config["analysis"]["id"],
        "status": "prepared",
        "config": relative(config_path),
        "config_sha256": sha256_file(config_path),
        "source_counts": dict(counts),
        "source_modes": dict(Counter(row["stored_mode"] for row in all_rows)),
        "source_total_bytes": sum(int(row["file_bytes"]) for row in all_rows),
        "archive_total_bytes": sum(int(row["bytes"]) for row in archive_rows),
        "rate_rows": len(rate_rows),
        "s33_swin_tile_rows": len(tile_rows),
        "sgd_patch_rows": len(sgd_rows),
        "manifest_hashes": {
            path.name: sha256_file(path)
            for path in sorted(manifests.iterdir())
            if path.is_file()
        },
        "frozen_references": frozen_rows,
        "official_imagenette_validation_accessed": False,
        "method_inference_run": False,
    }
    summary["summary_payload_sha256"] = canonical_json_sha256(summary)
    (output / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    state_path.write_text(
        json.dumps(
            {
                "analysis_id": config["analysis"]["id"],
                "status": "prepared",
                "next_allowed_step": "metric_identity_sanity",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
