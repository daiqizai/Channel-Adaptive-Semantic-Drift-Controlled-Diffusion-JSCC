"""Stage 1b of ANALYSIS-RDD-P0-DISTRIBUTION-SHIFT-001: exact S33 replay.

Replays the frozen S33 checkpoint over the shared 64-image policy-dev population
for all 5 SNRs x 3 seeds, reusing the project's proven replay contract
(canonical noise, batching, uint8 floor quantization) verbatim.

This is NOT method development. The checkpoint, noise contract, batch size and
rate are frozen; it only materializes images that were never saved to disk
(historically only 1 dB x 192 were kept). Each reconstruction is verified against
its recorded PSNR and the run aborts on any mismatch.

Preregistration: reports/rdd_p0_distribution_shift_preregistration_2026-07-30.md
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path("/home/liulu/projects/channel-adaptive-semantic-drift-controlled-diffusion-jscc")
ANALYSIS_ID = "ANALYSIS-RDD-P0-DISTRIBUTION-SHIFT-001"

SNRS = [1.0, 4.0, 7.0, 13.0, 19.0]
GATE_S33_PSNR = 1e-5  # same tolerance the project's own replay uses

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def stem(sample_id: str, base_seed: int, snr_db: float) -> str:
    safe = sample_id.replace("/", "__").replace(".JPEG", "")
    return "%s__seed%d__snr%02d" % (safe, base_seed, int(snr_db))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/analysis" / ANALYSIS_ID))
    ap.add_argument("--audit-config",
                    default=str(ROOT / "configs/low_snr_semantic_drift_visual_audit.yaml"))
    args = ap.parse_args()

    import torch
    from PIL import Image
    from torchvision import transforms

    from low_snr_semantic_drift_visual_audit import (  # type: ignore
        read_csv,
        resolve,
        tensor_to_pil,
    )
    from cadsd_jscc.external_common import (  # type: ignore
        canonical_noise_sha256,
        canonical_standard_normal,
    )
    from cadsd_jscc.metrics import psnr_per_sample  # type: ignore
    from s32_strong_jscc_external_comparison import (  # type: ignore
        build_model,
        load_population,
        load_yaml,
        require_sha,
    )

    out = Path(args.out)
    arm_dir = out / "arms" / "s33_strong"
    from rdd_p0_common import require_frozen_no_overwrite
    require_frozen_no_overwrite(
        ROOT / "configs/rdd_p0_distribution_shift.yaml",
        [arm_dir, out / "s33_replay_report.json"],
    )
    arm_dir.mkdir(parents=True, exist_ok=False)

    audit_cfg = load_yaml(Path(args.audit_config))
    s33_cfg = load_yaml(resolve(audit_cfg["inputs"]["s33_config"]))

    # Historical S33 rows keyed by (sample_id, seed, snr) for verification.
    strong_by_key = {}
    for row in read_csv(resolve(audit_cfg["inputs"]["s33_per_sample"])):
        strong_by_key[(str(row["sample_id"]), int(row["base_seed"]),
                       float(row["snr_db"]))] = row
    print("historical S33 rows:", len(strong_by_key))

    ckpt_path = require_sha(s33_cfg["inputs"]["strong_checkpoint"],
                            s33_cfg["inputs"]["strong_checkpoint_sha256"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint).to(device).eval().requires_grad_(False)

    samples, _ = load_population(s33_cfg)
    transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()])
    targets_all = torch.stack(
        [transform(Image.open(item["path"]).convert("RGB")) for item in samples]).to(device)

    batch_size = int(s33_cfg["runtime"]["batch_size"])
    reference_symbols = int(s33_cfg["rate"]["canonical_noise_reference_real_symbols"])
    latent_shape = (model.latent_channels, model.image_size // 16, model.image_size // 16)

    seeds = sorted({k[1] for k in strong_by_key})
    max_err, written = 0.0, 0

    for snr in SNRS:
        for base_seed in seeds:
            for start in range(0, len(samples), batch_size):
                end = min(start + batch_size, len(samples))
                noises = []
                for item in samples[start:end]:
                    sid = str(item["sample_id"])
                    hist = strong_by_key[(sid, base_seed, snr)]
                    ref = canonical_standard_normal(base_seed, sid, snr, reference_symbols)
                    if canonical_noise_sha256(ref) != hist["canonical_noise_sha256"]:
                        raise RuntimeError("canonical noise mismatch: %s/%s/%s"
                                           % (sid, base_seed, snr))
                    noises.append(ref[: model.real_symbols].reshape(latent_shape))
                with torch.inference_mode():
                    recon, _ = model.forward_with_observation(
                        targets_all[start:end], snr, torch.stack(noises).to(device))
                    recon = torch.floor(recon.clamp(0.0, 1.0) * 255.0) / 255.0
                    psnr = psnr_per_sample(recon, targets_all[start:end])
                for offset, item in enumerate(samples[start:end]):
                    sid = str(item["sample_id"])
                    hist = strong_by_key[(sid, base_seed, snr)]
                    err = abs(float(psnr[offset]) - float(hist["strong_psnr"]))
                    max_err = max(max_err, err)
                    if err > GATE_S33_PSNR:
                        raise RuntimeError("S33 replay PSNR mismatch %.8f at %s/%s/%s"
                                           % (err, sid, base_seed, snr))
                    tensor_to_pil(recon[offset]).save(
                        arm_dir / ("%s.png" % stem(sid, base_seed, snr)))
                    written += 1
            print("  snr=%s seed=%s done (written=%d)" % (snr, base_seed, written))

    report = {"analysis_id": ANALYSIS_ID, "stage": "s33_replay",
              "checkpoint_sha_verified": True,
              "images_written": written,
              "max_abs_psnr_error_db": max_err,
              "gate_db": GATE_S33_PSNR,
              "official_validation_accessed": False}
    (out / "s33_replay_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
