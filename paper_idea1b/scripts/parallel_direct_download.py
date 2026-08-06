#!/usr/bin/env python3
"""Direct, resumable, range-parallel downloader for paper idea1b Gate A0."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import math
import os
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
PROXY_KEYS = {
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--dataset",
        choices=(
            "all",
            "kodak",
            "clic_mobile",
            "clic_professional",
            "metric_weights",
        ),
        default="all",
    )
    parser.add_argument("--workers", type=int)
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def direct_environment() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key not in PROXY_KEYS}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def curl_download(url: str, output: Path, expected_bytes: int) -> None:
    if output.is_file() and output.stat().st_size == expected_bytes:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".direct-part")
    command = [
        "curl",
        "--ipv4",
        "--location",
        "--fail",
        "--silent",
        "--show-error",
        "--retry",
        "10",
        "--retry-all-errors",
        "--connect-timeout",
        "30",
        "--output",
        str(temporary),
        url,
    ]
    subprocess.run(command, check=True, env=direct_environment())
    if temporary.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"downloaded byte count mismatch for {url}: "
            f"{temporary.stat().st_size} != {expected_bytes}"
        )
    os.replace(temporary, output)


def download_kodak(config: dict[str, Any], workers: int) -> None:
    entry = config["datasets"]["kodak"]
    root = resolve(entry["root"])
    sizes = list(map(int, entry["expected_file_bytes"]))
    if len(sizes) != int(entry["count"]):
        raise RuntimeError("Kodak expected-size table does not match image count")
    archive = resolve(entry["mirror_archive_path"])
    if not archive.is_file() or archive.stat().st_size == 0:
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix(archive.suffix + ".direct-part")
        subprocess.run(
            [
                "curl",
                "--ipv4",
                "--location",
                "--fail",
                "--retry",
                "10",
                "--retry-all-errors",
                "--connect-timeout",
                "30",
                "--output",
                str(temporary),
                entry["mirror_archive_url"],
            ],
            check=True,
            env=direct_environment(),
        )
        os.replace(temporary, archive)
    with tarfile.open(archive, "r:gz") as handle:
        members = {
            Path(member.name).name: member
            for member in handle.getmembers()
            if "/PhotoCD_PCD0992/" in member.name and member.isfile()
        }
        if len(members) != 24:
            raise RuntimeError(f"Kodak mirror archive contains {len(members)} images")
        root.mkdir(parents=True, exist_ok=True)
        for index, expected in enumerate(sizes, 1):
            member = members[f"{index:02d}.png"]
            if member.size != expected:
                raise RuntimeError(
                    f"Kodak official byte-size mismatch for {index:02d}: "
                    f"{member.size} != {expected}"
                )
            output = root / f"kodim{index:02d}.png"
            if output.is_file() and output.stat().st_size == expected:
                continue
            extracted = handle.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"cannot extract Kodak member {member.name}")
            temporary = output.with_suffix(".png.direct-part")
            with temporary.open("wb") as destination:
                for block in iter(lambda: extracted.read(4 * 1024 * 1024), b""):
                    destination.write(block)
            if temporary.stat().st_size != expected:
                raise RuntimeError("Kodak extracted byte count mismatch")
            os.replace(temporary, output)
    for index, expected in enumerate(sizes, 1):
        output = root / f"kodim{index:02d}.png"
        if output.stat().st_size != expected:
            raise RuntimeError(f"Kodak output invalid: {output}")
        print(f"[Kodak {index:02d}/24] {output.name} sha256={sha256_file(output)}")
    print(f"[Kodak mirror archive] sha256={sha256_file(archive)}")


def ranged_part(
    url: str,
    part: Path,
    start: int,
    end: int,
) -> Path:
    expected = end - start + 1
    if part.is_file() and part.stat().st_size == expected:
        return part
    part.parent.mkdir(parents=True, exist_ok=True)
    temporary = part.with_suffix(".part-tmp")
    command = [
        "curl",
        "--ipv4",
        "--location",
        "--fail",
        "--silent",
        "--show-error",
        "--retry",
        "10",
        "--retry-all-errors",
        "--connect-timeout",
        "30",
        "--range",
        f"{start}-{end}",
        "--output",
        str(temporary),
        url,
    ]
    subprocess.run(command, check=True, env=direct_environment())
    if temporary.stat().st_size != expected:
        raise RuntimeError(
            f"range byte mismatch {start}-{end}: {temporary.stat().st_size} != {expected}"
        )
    os.replace(temporary, part)
    return part


def download_archive(
    entry: dict[str, Any],
    archive_root: Path,
    workers: int,
    chunk_bytes: int,
) -> None:
    output = archive_root / entry["filename"]
    expected_total = int(entry["expected_bytes"])
    if output.is_file() and output.stat().st_size == expected_total:
        print(
            f"[range {entry['name']}] already complete "
            f"sha256={sha256_file(output)}"
        )
        return
    part_root = archive_root / f".{entry['filename']}.range-parts"
    ranges = [
        (index, start, min(start + chunk_bytes - 1, expected_total - 1))
        for index, start in enumerate(range(0, expected_total, chunk_bytes))
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(
                ranged_part,
                entry["url"],
                part_root / f"{index:06d}.part",
                start,
                end,
            ): index
            for index, start, end in ranges
        }
        completed = 0
        report_every = max(1, math.ceil(len(ranges) / 20))
        for future in concurrent.futures.as_completed(future_map):
            future.result()
            completed += 1
            if completed % report_every == 0 or completed == len(ranges):
                print(f"[range {entry['name']}] ranges {completed}/{len(ranges)}")

    assembling = output.with_suffix(output.suffix + ".assembling")
    with assembling.open("wb") as destination:
        for index, _start, _end in ranges:
            part = part_root / f"{index:06d}.part"
            with part.open("rb") as source:
                for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    destination.write(block)
    if assembling.stat().st_size != expected_total:
        raise RuntimeError("assembled archive byte count mismatch")
    os.replace(assembling, output)
    print(
        f"[range {entry['name']}] complete bytes={expected_total} "
        f"sha256={sha256_file(output)}"
    )


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    workers = args.workers or int(config["metrics"]["download"]["parallel_workers"])
    chunk_bytes = int(config["metrics"]["download"]["range_chunk_bytes"])
    if workers < 1 or workers > 64:
        raise ValueError("workers must be in [1,64]")
    print(
        "proxy environment removed:",
        sorted(key for key in os.environ if key in PROXY_KEYS),
    )
    if args.dataset in {"all", "kodak"}:
        download_kodak(config, workers)
    if args.dataset.startswith("clic_"):
        target_name = args.dataset.removeprefix("clic_")
        entries = [
            entry
            for entry in config["datasets"]["clic2020_test"]["archives"]
            if entry["name"] == target_name
        ]
    elif args.dataset == "all":
        entries = config["datasets"]["clic2020_test"]["archives"]
    else:
        entries = []
    archive_root = resolve(config["datasets"]["archives_dir"])
    for entry in entries:
        download_archive(entry, archive_root, workers, chunk_bytes)
    if args.dataset in {"all", "metric_weights"}:
        metric_download = config["metrics"]["download"]
        metric_root = resolve(metric_download["metric_weights_root"])
        for entry in metric_download["metric_weights"]:
            download_archive(entry, metric_root, workers, chunk_bytes)


if __name__ == "__main__":
    main()
