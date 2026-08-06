"""Stage 2 of ANALYSIS-RDD-P0-DISTRIBUTION-SHIFT-001: build candidate target distributions.

Reference sets (all derived from the same 64 source images, so n matches the arms):

  real          (a)  source images                              [written in stage 1]
  vae_sd21      (b1) SD 2.1 AutoencoderKL round-trip, no denoise
  vae_sgd       (b2) SGD 16-ch AutoencoderKL round-trip, no denoise
  blur_s{0.5..2.0} (c) Gaussian low-pass, ALL sigmas reported
  resample_512  (d1) 256->512->Lanczos->256 (DiffJSCC's native grid behaviour)
  jpeg_q{30,70} (d2) natural degradation control

VAE weights are read from LOCAL checkpoints only (no download). Encoding uses the
posterior MEAN (no sampling) so the reference sets are deterministic.

Preregistration: reports/rdd_p0_distribution_shift_preregistration_2026-07-30.md
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path("/home/liulu/projects/channel-adaptive-semantic-drift-controlled-diffusion-jscc")
ANALYSIS_ID = "ANALYSIS-RDD-P0-DISTRIBUTION-SHIFT-001"

BLUR_SIGMAS = [0.5, 1.0, 1.5, 2.0]
JPEG_QUALITIES = [30, 70]

DIFFJSCC_CKPT = ROOT / "third_party/DiffJSCC/checkpoints/DiffJSCC-OpenImage-CBR-1-96/model.ckpt"
SGD_CKPT = ROOT / "third_party/SGDJSCC/checkpoint/JSCC_model.pth"

# SD 2.1 / LDM first-stage ddconfig (matches configs/model/cldm_cnn.yaml).
DD_SD21 = dict(double_z=True, z_channels=4, resolution=256, in_channels=3, out_ch=3,
               ch=128, ch_mult=[1, 2, 4, 4], num_res_blocks=2,
               attn_resolutions=[], dropout=0.0)


def save(arr, path: Path) -> None:
    if isinstance(arr, np.ndarray):
        arr = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.save(path, format="PNG", compress_level=1)


def build_cheap_references(sources: list[Path], out: Path, report: dict) -> None:
    """Blur / resample / JPEG references -- no model weights required."""
    for sigma in BLUR_SIGMAS:
        name = "blur_s%s" % ("%g" % sigma).replace(".", "p")
        for p in sources:
            img = Image.open(p).convert("RGB")
            save(img.filter(ImageFilter.GaussianBlur(radius=sigma)), out / name / p.name)
        report["counts"][name] = len(sources)

    for p in sources:
        img = Image.open(p).convert("RGB")
        up = img.resize((512, 512), Image.BICUBIC)
        save(up.resize((256, 256), Image.LANCZOS), out / "resample_512" / p.name)
    report["counts"]["resample_512"] = len(sources)

    for q in JPEG_QUALITIES:
        name = "jpeg_q%d" % q
        for p in sources:
            img = Image.open(p).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            save(Image.open(buf).convert("RGB"), out / name / p.name)
        report["counts"][name] = len(sources)


def build_vae_references(sources: list[Path], out: Path, report: dict) -> None:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def roundtrip(model, name, scale_in=lambda x: x * 2.0 - 1.0,
                  scale_out=lambda x: (x + 1.0) / 2.0):
        n = 0
        for p in sources:
            arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
            x = torch.from_numpy(arr).permute(2, 0, 1)[None].to(device)
            with torch.inference_mode():
                posterior = model.encode(scale_in(x))
                z = posterior.mean if hasattr(posterior, "mean") else posterior.mode()
                y = model.decode(z)
            y = scale_out(y).clamp(0.0, 1.0)[0].permute(1, 2, 0).cpu().numpy()
            save(np.floor(y * 255.0), out / name / p.name)
            n += 1
        report["counts"][name] = n

    # ---- (b1) DiffJSCC / SD 2.1 first stage ----
    sys.path.insert(0, str(ROOT / "third_party/DiffJSCC"))
    from ldm.models.autoencoder import AutoencoderKL as LDMAutoencoderKL  # type: ignore

    sd = torch.load(DIFFJSCC_CKPT, map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    fs = {k[len("first_stage_model."):]: v for k, v in sd.items()
          if k.startswith("first_stage_model.")}
    if not fs:
        raise RuntimeError("no first_stage_model.* weights in %s" % DIFFJSCC_CKPT)
    vae_sd21 = LDMAutoencoderKL(ddconfig=DD_SD21, lossconfig={"target": "torch.nn.Identity"},
                                embed_dim=4)
    missing, unexpected = vae_sd21.load_state_dict(fs, strict=False)
    # `loss.*` is an Identity stub here and is never used at inference, so only
    # encoder/decoder/quant weights are required to be present.
    critical = [k for k in missing if not k.startswith("loss.")]
    report["vae_sd21_missing_keys"] = len(missing)
    report["vae_sd21_missing_critical"] = critical
    report["vae_sd21_unexpected_keys"] = len(unexpected)
    if critical:
        raise RuntimeError("SD2.1 VAE missing critical keys: %s" % critical[:5])
    vae_sd21 = vae_sd21.to(device).eval().requires_grad_(False)
    roundtrip(vae_sd21, "vae_sd21")
    del vae_sd21
    torch.cuda.empty_cache()

    # ---- (b2) SGD 16-channel AutoencoderKL ----
    sys.path.insert(0, str(ROOT / "third_party/SGDJSCC"))
    from models.test_advanced_network.autoencoderkl import AutoencoderKL as SGDAutoencoderKL  # type: ignore

    sd2 = torch.load(SGD_CKPT, map_location="cpu", weights_only=True)
    sd2 = sd2.get("state_dict", sd2)
    vkeys = {k[len("vae."):]: v for k, v in sd2.items() if k.startswith("vae.")}
    if not vkeys:
        raise RuntimeError("no vae.* weights in %s" % SGD_CKPT)
    dd = dict(DD_SD21)
    dd["z_channels"] = 16
    vae_sgd = SGDAutoencoderKL(dd, 16)
    missing2, unexpected2 = vae_sgd.load_state_dict(vkeys, strict=False)
    critical2 = [k for k in missing2 if not k.startswith("loss.")]
    report["vae_sgd_missing_keys"] = len(missing2)
    report["vae_sgd_missing_critical"] = critical2
    report["vae_sgd_unexpected_keys"] = len(unexpected2)
    if critical2:
        raise RuntimeError("SGD VAE missing critical keys: %s" % critical2[:5])
    vae_sgd = vae_sgd.to(device).eval().requires_grad_(False)
    # SGD divides the latent by 15.45 internally; that is a scalar and cancels in a
    # plain encode->decode round trip, so it is intentionally not applied here.
    roundtrip(vae_sgd, "vae_sgd")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/analysis" / ANALYSIS_ID))
    ap.add_argument("--skip-vae", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    ref_out = out / "reference_sets"
    target_names = [
        *("blur_s%s" % ("%g" % sigma).replace(".", "p") for sigma in BLUR_SIGMAS),
        "resample_512",
        *("jpeg_q%d" % quality for quality in JPEG_QUALITIES),
    ]
    if not args.skip_vae:
        target_names.extend(["vae_sd21", "vae_sgd"])
    from rdd_p0_common import require_frozen_no_overwrite
    require_frozen_no_overwrite(
        ROOT / "configs/rdd_p0_distribution_shift.yaml",
        [out / "build_references_report.json", *(ref_out / name for name in target_names)],
    )

    real_dir = ref_out / "real"
    sources = sorted(real_dir.glob("*.png"))
    if len(sources) != 64:
        print("expected 64 source images in %s, found %d (run stage 1 first)"
              % (real_dir, len(sources)), file=sys.stderr)
        return 2

    ref_out = out / "reference_sets"
    report = {"analysis_id": ANALYSIS_ID, "stage": "build_references",
              "counts": {"real": len(sources)}, "downloads": 0,
              "official_validation_accessed": False}

    build_cheap_references(sources, ref_out, report)
    if not args.skip_vae:
        build_vae_references(sources, ref_out, report)

    (out / "build_references_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
