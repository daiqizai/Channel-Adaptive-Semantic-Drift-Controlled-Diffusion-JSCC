#!/usr/bin/env python3
"""Cache paired B0 and S18 identity-controlled diffusion observations for S19."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import save_image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from cadsd_jscc.channel_matched_latent_diffusion import (  # noqa: E402
    channel_alpha,
    deterministic_ddim,
    normalize_channel_observation,
)
from cadsd_jscc.snr_identity_envelope import (  # noqa: E402
    apply_correction_envelope,
    envelope_strength,
)
from s17_channel_matched_latent_diffusion import (  # noqa: E402
    CachedOriginalDataset,
    active_to_dense,
    build_denoiser,
    build_jscc,
    canonical_batch_noise,
    clean_transmitted_active,
    coordinate_contract,
    dense_to_active,
    load_denoiser_checkpoint,
    seed_everything,
)


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


def valid_png(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def atomic_save_image(tensor: torch.Tensor, path: Path) -> None:
    temporary = path.with_name(f"{path.stem}.partial.png")
    save_image(torch.round(tensor.cpu() * 255.0) / 255.0, temporary)
    temporary.replace(path)


def snr_name(snr: float) -> str:
    return f"snr_{int(snr):02d}db" if float(snr).is_integer() else f"snr_{snr:g}db"


def validate(config: dict[str, Any]) -> None:
    if config["protocol"]["status"] != "population_frozen_before_cache_output":
        raise RuntimeError("S19 cache contract is not executable")
    if config["protocol"].get("official_imagenette_accessed") is not False:
        raise RuntimeError("official Imagenette validation must remain sealed")
    for key, hash_key in (
        ("source_manifest", "source_manifest_sha256"),
        ("deepjscc_checkpoint", "deepjscc_checkpoint_sha256"),
        ("latent_diffusion_checkpoint", "latent_diffusion_checkpoint_sha256"),
        ("identity_policy", "identity_policy_sha256"),
        ("b1_config", "b1_config_sha256"),
        ("b1_checkpoint", "b1_checkpoint_sha256"),
    ):
        path = resolve(config["inputs"][key])
        if not path.is_file() or sha256_file(path) != str(config["inputs"][hash_key]):
            raise RuntimeError(f"input hash mismatch: {key}")
    if resolve(config["inputs"]["deepjscc_checkpoint"]).resolve() == resolve(
        config["inputs"]["forbidden_deepjscc_checkpoint"]
    ).resolve():
        raise RuntimeError("forbidden latest DeepJSCC checkpoint selected")
    if int(config["rate"]["active_real_symbols"]) != 19712:
        raise RuntimeError("active-symbol contract changed")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write empty cache manifest")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/s19_diffusion_fusion_ablation.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate(config)
    output = resolve(config["outputs"]["population_dir"])
    cache_manifest = resolve(config["inputs"]["cache_manifest"])
    partial_cache_exists = any((output / "exports").glob("snr_*db"))
    if cache_manifest.exists() or (partial_cache_exists and not args.resume):
        raise FileExistsError("S19 cache outputs already exist")
    device = torch.device(args.device)
    seed_everything(int(config["seed"]))
    jscc = build_jscc(config, device)
    denoiser = build_denoiser(config, device)
    load_denoiser_checkpoint(denoiser, resolve(config["inputs"]["latent_diffusion_checkpoint"]), device)
    denoiser.eval().requires_grad_(False)
    reserved, _valid_active, valid_dense = coordinate_contract(jscc, config, device)
    policy = json.loads(resolve(config["inputs"]["identity_policy"]).read_text(encoding="utf-8"))
    if str(policy["selected_name"]) != str(config["diffusion"]["selected_policy_name"]):
        raise RuntimeError("identity policy name changed")
    specification = policy["selected_specification"]
    cache_diffusion_snrs = {float(value) for value in config["diffusion"]["cache_diffusion_snrs_db"]}
    rows: list[dict[str, Any]] = []
    for role, count in config["population"]["roles"].items():
        dataset = CachedOriginalDataset(
            output, resolve(config["inputs"]["source_manifest"]), str(role), count=int(count)
        )
        loader = DataLoader(
            dataset,
            batch_size=int(config["evaluation"]["batch_size"]),
            shuffle=False,
            num_workers=int(config["evaluation"]["num_workers"]),
            pin_memory=device.type == "cuda",
            persistent_workers=int(config["evaluation"]["num_workers"]) > 0,
        )
        for snr_value in config["channel"]["snrs_db"]:
            snr = float(snr_value)
            stem = snr_name(snr)
            b0_dir = output / "exports" / stem / "reconstruction"
            diffusion_dir = output / "exports" / stem / "identity_diffusion"
            b0_dir.mkdir(parents=True, exist_ok=True)
            if snr in cache_diffusion_snrs:
                diffusion_dir.mkdir(parents=True, exist_ok=True)
            strength = envelope_strength(
                snr,
                specification,
                noise_variance_factor_per_real=float(config["channel"]["noise_variance_factor_per_real"]),
                reference_snr_db=1.0,
            )
            if (snr in cache_diffusion_snrs) != (strength > 0.0):
                raise RuntimeError("cache diffusion SNRs do not match nonzero identity policy")
            alpha = float(channel_alpha(snr, float(config["channel"]["noise_variance_factor_per_real"])))
            for images_cpu, sample_ids in loader:
                noise_cpu, noise_hashes = canonical_batch_noise(
                    list(sample_ids),
                    snr,
                    int(config["channel"]["role_base_seeds"][role]),
                    jscc.active_symbols,
                )
                expected_b0 = [b0_dir / str(sample_id) for sample_id in sample_ids]
                expected_auxiliary = [
                    (diffusion_dir / str(sample_id)) if strength > 0.0 else b0_path
                    for sample_id, b0_path in zip(sample_ids, expected_b0)
                ]
                batch_complete = all(valid_png(path) for path in expected_b0) and all(
                    valid_png(path) for path in expected_auxiliary
                )
                b0 = controlled = None
                if not batch_complete:
                    images = images_cpu.to(device, non_blocking=True)
                    transmitted, _clean_active, dense_shape = clean_transmitted_active(
                        jscc, images, reserved
                    )
                    jscc.snr_db = snr
                    received = jscc.transmit_active(transmitted, noise_cpu.to(device))
                    received[:, reserved] = 0.0
                    b0 = jscc.decode_active(received, dense_shape).clamp(0.0, 1.0)
                    controlled = b0
                    if strength > 0.0:
                        matched_state = active_to_dense(
                            jscc, normalize_channel_observation(received, alpha), dense_shape
                        )
                        full_dense = deterministic_ddim(
                            denoiser,
                            matched_state,
                            valid_dense,
                            alpha_start=alpha,
                            sampling_steps=int(config["diffusion"]["sampling_steps"]),
                            alpha_max=float(config["diffusion"]["train_alpha_max"]),
                        )
                        full_active = dense_to_active(jscc, full_dense)
                        full_active[:, reserved] = 0.0
                        controlled_active = apply_correction_envelope(
                            received, full_active, strength
                        )
                        controlled_active[:, reserved] = 0.0
                        controlled = jscc.decode_active(
                            controlled_active, dense_shape
                        ).clamp(0.0, 1.0)
                for index, sample_id in enumerate(sample_ids):
                    b0_path = b0_dir / str(sample_id)
                    if not valid_png(b0_path):
                        assert b0 is not None
                        atomic_save_image(b0[index], b0_path)
                    b0_sha = sha256_file(b0_path)
                    if strength > 0.0:
                        auxiliary_path = diffusion_dir / str(sample_id)
                        if not valid_png(auxiliary_path):
                            assert controlled is not None
                            atomic_save_image(controlled[index], auxiliary_path)
                        auxiliary_sha = sha256_file(auxiliary_path)
                    else:
                        auxiliary_path = b0_path
                        auxiliary_sha = b0_sha
                    rows.append(
                        {
                            "sample": sample_id,
                            "role": role,
                            "snr_db": snr,
                            "canonical_noise_sha256": noise_hashes[index],
                            "identity_strength": strength,
                            "b0_path": relative(b0_path),
                            "b0_sha256": b0_sha,
                            "fusion_auxiliary_path": relative(auxiliary_path),
                            "fusion_auxiliary_sha256": auxiliary_sha,
                        }
                    )
            print(json.dumps({"role": role, "snr_db": snr, "cached": len(dataset)}), flush=True)
    write_csv(cache_manifest, rows)
    cache_sha = sha256_file(cache_manifest)
    shutil.copy2(config_path, output / "config_before_cache_freeze.yaml")
    shutil.copy2(SCRIPT, output / SCRIPT.name)
    metadata = {
        "cache_manifest_sha256": cache_sha,
        "records": len(rows),
        "unique_samples": len({row["sample"] for row in rows}),
        "role_records": {
            role: sum(row["role"] == role for row in rows)
            for role in config["population"]["roles"]
        },
        "snrs_db": [float(value) for value in config["channel"]["snrs_db"]],
        "b0_files": len(rows),
        "identity_diffusion_files": sum(float(row["identity_strength"]) > 0 for row in rows),
        "resumed_from_partial_cache": bool(args.resume and partial_cache_exists),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "official_imagenette_accessed": False,
        "download_note": "No download; local checkpoints and images only.",
    }
    (output / "cache_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "STATE.json").write_text(
        json.dumps({"state": "CACHE_COMPLETE", **metadata}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
