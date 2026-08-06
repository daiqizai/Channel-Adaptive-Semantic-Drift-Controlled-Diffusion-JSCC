"""Stage 2b of ANALYSIS-RDD-P0-DISTRIBUTION-SHIFT-001: generative-prior proxy references.

Builds reference set (b): source images passed through each method's OWN VAE
(encode -> decode, NO denoising), giving a proxy for that generative prior's footprint.

  vae_sd21  -- DiffJSCC first_stage_model (SD 2.1 AutoencoderKL, z=4)
  vae_sgd   -- SGD AutoencoderKL (embed_dim=16)

Must run under .venv-sgdjscc (has pytorch_lightning / diffusers / einops).
Weights are LOCAL; nothing is downloaded.

Two API divergences are handled explicitly, because getting either wrong silently
corrupts the reference set:
  * LDM  encode -> DiagonalGaussianDistribution ; decode -> Tensor
  * SGD  encode -> AutoencoderKLOutput(latent_dist=...) ; decode -> [Tensor]
  * SGD's forward() injects AWGN via through_channel(); we deliberately call
    encode/decode directly so no channel noise enters the prior proxy.

Latents use the posterior MODE (deterministic), never a sample.

Preregistration: reports/rdd_p0_distribution_shift_preregistration_2026-07-30.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/home/liulu/projects/channel-adaptive-semantic-drift-controlled-diffusion-jscc")
ANALYSIS_ID = "ANALYSIS-RDD-P0-DISTRIBUTION-SHIFT-001"

DIFFJSCC_CKPT = ROOT / "third_party/DiffJSCC/checkpoints/DiffJSCC-OpenImage-CBR-1-96/model.ckpt"
SGD_CKPT = ROOT / "third_party/SGDJSCC/checkpoint/JSCC_model.pth"

# Frozen SHA-256 of the author checkpoints (DiffJSCC value taken from the S30 contract,
# configs/s30_diffjscc_external_comparison.yaml). Both files embed pickled framework
# objects, so weights_only=True cannot load them; we fail closed on the hash instead.
DIFFJSCC_CKPT_SHA256 = "ae1e6df0b706d09857cfa02d399f94cc171d8d0ce44f851d96cb032bd7dec579"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_checkpoint(path: Path, expected_sha: str | None):
    """Load a trusted author checkpoint after verifying its content hash."""
    import torch
    got = sha256_file(path)
    if expected_sha is not None and got != expected_sha:
        raise RuntimeError("checkpoint SHA mismatch for %s: got %s expected %s"
                           % (path, got, expected_sha))
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj, got

DD_BASE = dict(double_z=True, z_channels=4, resolution=256, in_channels=3, out_ch=3,
               ch=128, ch_mult=[1, 2, 4, 4], num_res_blocks=2,
               attn_resolutions=[], dropout=0.0)


def save_u8(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="RGB").save(path, format="PNG", compress_level=1)


def sgd_power_normalize(z):
    """Author's latent power constraint (third_party/SGDJSCC/inference_config.py:41-45).

    SGD's decoder is trained on power-normalized latents: both through_channel() and
    the real inference path at inference_config.py:151 call decode(normalize(z)).
    Decoding a raw posterior mean is out-of-distribution and yields ~12.5 dB instead
    of ~29.5 dB, so this normalization is REQUIRED for a faithful prior proxy.
    """
    import math

    import torch.nn.functional as F

    b, c, h, w = z.shape
    v = z.reshape(b, -1)
    v = F.normalize(v, p=2, dim=1) * math.sqrt(v.shape[1])
    return v.reshape(b, c, h, w)


def roundtrip(model, sources, out_dir: Path, device, kind: str,
              latent_transform=None) -> tuple[int, float]:
    """encode -> posterior.mode() -> [transform] -> decode, in [-1,1] convention."""
    import torch

    n, psnrs = 0, []
    for p in sources:
        arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1)[None].to(device) * 2.0 - 1.0
        with torch.inference_mode():
            enc = model.encode(x)
            posterior = getattr(enc, "latent_dist", enc)   # SGD wraps it, LDM does not
            z = posterior.mode()                           # deterministic, never sample
            if latent_transform is not None:
                z = latent_transform(z)
            dec = model.decode(z)
            if isinstance(dec, (list, tuple)):             # SGD returns [dec]
                dec = dec[0]
        y = ((dec + 1.0) / 2.0).clamp(0.0, 1.0)[0].permute(1, 2, 0).float().cpu().numpy()
        u8 = np.floor(y * 255.0)
        save_u8(u8, out_dir / p.name)
        src = np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)
        mse = float(np.mean((u8 - src) ** 2)) / (255.0 ** 2)
        psnrs.append(-10.0 * np.log10(max(mse, 1e-12)))
        n += 1
    mean_psnr = float(np.mean(psnrs))
    print("  %s: wrote %d images, mean round-trip PSNR = %.3f dB" % (kind, n, mean_psnr))
    return n, mean_psnr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/analysis" / ANALYSIS_ID))
    args = ap.parse_args()

    # Keep third-party hub lookups fully offline.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch

    out = Path(args.out)
    from rdd_p0_common import require_frozen_no_overwrite
    require_frozen_no_overwrite(
        ROOT / "configs/rdd_p0_distribution_shift.yaml",
        [
            out / "reference_sets" / "vae_sd21",
            out / "reference_sets" / "vae_sgd",
            out / "build_vae_references_report.json",
        ],
    )

    real_dir = out / "reference_sets" / "real"
    sources = sorted(real_dir.glob("*.png"))
    if len(sources) != 64:
        print("expected 64 sources in %s, found %d" % (real_dir, len(sources)), file=sys.stderr)
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {"analysis_id": ANALYSIS_ID, "stage": "build_vae_references",
              "downloads": 0, "counts": {},
              "latent": "posterior_mode_deterministic",
              "channel_noise_injected": False,
              "official_validation_accessed": False}

    # ---------------- (b1) DiffJSCC / SD 2.1 first stage ----------------
    sys.path.insert(0, str(ROOT / "third_party/DiffJSCC"))
    from ldm.models.autoencoder import AutoencoderKL as LDMAutoencoderKL  # type: ignore

    sd, sha1 = load_checkpoint(DIFFJSCC_CKPT, DIFFJSCC_CKPT_SHA256)
    report["diffjscc_checkpoint_sha256"] = sha1
    sd = sd.get("state_dict", sd)
    fs = {k[len("first_stage_model."):]: v for k, v in sd.items()
          if k.startswith("first_stage_model.")}
    if not fs:
        raise RuntimeError("no first_stage_model.* weights found")
    print("SD2.1 first_stage tensors:", len(fs))

    vae1 = LDMAutoencoderKL(ddconfig=DD_BASE,
                            lossconfig={"target": "torch.nn.Identity"}, embed_dim=4)
    missing, unexpected = vae1.load_state_dict(fs, strict=False)
    critical = [k for k in missing if not k.startswith("loss.")]
    report["vae_sd21_missing_total"] = len(missing)
    report["vae_sd21_missing_critical"] = critical
    report["vae_sd21_unexpected"] = len(unexpected)
    if critical:
        raise RuntimeError("SD2.1 VAE missing critical keys: %s" % critical[:8])
    vae1 = vae1.to(device).eval().requires_grad_(False)
    report["counts"]["vae_sd21"], report["vae_sd21_roundtrip_psnr_db"] = roundtrip(
        vae1, sources, out / "reference_sets" / "vae_sd21", device, "vae_sd21")
    del vae1
    torch.cuda.empty_cache()

    # ---------------- (b2) SGD 16-channel AutoencoderKL ----------------
    sys.path.insert(0, str(ROOT / "third_party/SGDJSCC"))
    from models.test_advanced_network.autoencoderkl import AutoencoderKL as SGDAutoencoderKL  # type: ignore

    sd2, sha2 = load_checkpoint(SGD_CKPT, None)
    report["sgd_checkpoint_sha256"] = sha2
    sd2 = sd2.get("state_dict", sd2)
    vkeys = {k[len("vae."):]: v for k, v in sd2.items() if k.startswith("vae.")}
    if not vkeys:
        raise RuntimeError("no vae.* weights found in %s" % SGD_CKPT)
    print("SGD vae tensors:", len(vkeys))

    dd = dict(DD_BASE)
    dd["z_channels"] = 16
    vae2 = SGDAutoencoderKL(dd, 16)
    missing2, unexpected2 = vae2.load_state_dict(vkeys, strict=False)
    critical2 = [k for k in missing2 if not k.startswith("loss.")]
    report["vae_sgd_missing_total"] = len(missing2)
    report["vae_sgd_missing_critical"] = critical2
    report["vae_sgd_unexpected"] = len(unexpected2)
    if critical2:
        raise RuntimeError("SGD VAE missing critical keys: %s" % critical2[:8])
    vae2 = vae2.to(device).eval().requires_grad_(False)
    report["sgd_latent_power_normalized"] = True
    report["counts"]["vae_sgd"], report["vae_sgd_roundtrip_psnr_db"] = roundtrip(
        vae2, sources, out / "reference_sets" / "vae_sgd", device, "vae_sgd",
        latent_transform=sgd_power_normalize)

    (out / "build_vae_references_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
