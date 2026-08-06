"""Stage 5 of ANALYSIS-RDD-P0-DISTRIBUTION-SHIFT-001: CLIC-428 discriminative complement.

The shared 4-arm population only has n=192 per cell. This stage re-runs the criterion-2
reference-set analysis on the existing CLIC2020 test reconstructions (n=428, full
resolution), for the DISCRIMINATIVE arms that were actually run there
(s33_strong / swin_official_base_sa / swin_capacity_matched_sa).

Purpose: a well-powered check of whether discriminative JSCC output drifts toward a
smoothed (blur) distribution rather than toward the source distribution. This does NOT
touch A1's frozen S33-vs-Swin quality verdict, and no generative arm exists here.

Reference sets are built from the 428 CLIC sources at native resolution.
Preregistration: reports/rdd_p0_distribution_shift_preregistration_2026-07-30.md
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path("/home/liulu/projects/channel-adaptive-semantic-drift-controlled-diffusion-jscc")
ANALYSIS_ID = "ANALYSIS-RDD-P0-DISTRIBUTION-SHIFT-001"

A1 = ROOT / "paper_idea1b/outputs/ANALYSIS-IDEA1B-A1-DISCRIMINATIVE-001"
RECON = A1 / "reconstructions/clic2020_test"
CLIC_SRC = ROOT / "paper_idea1b/data/clic2020_test"

ARMS = ["s33_strong", "swin_official_base_sa", "swin_capacity_matched_sa"]
SNRS = [1.0, 4.0, 7.0, 13.0, 19.0]
SEED = 20260748
BLUR_SIGMAS = [0.5, 1.0, 1.5, 2.0]
JPEG_QUALITIES = [30, 70]
KID_BASE_SEED = 20310800


def kid(f1, f2, seed):
    from cleanfid import fid as clean_fid
    np.random.seed(seed)
    return float(clean_fid.kernel_distance(f1, f2, num_subsets=100,
                                           max_subset_size=min(len(f1), len(f2))))


def feats(path, model, device):
    from cleanfid import fid as clean_fid
    return clean_fid.get_folder_features(str(path), model=model, num_workers=4,
                                         batch_size=8, device=device, mode="clean",
                                         description="", verbose=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/analysis" / ANALYSIS_ID))
    args = ap.parse_args()

    import torch
    from cleanfid import features as clean_features
    from cleanfid import fid as clean_fid

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    from rdd_p0_common import require_frozen_no_overwrite
    require_frozen_no_overwrite(
        ROOT / "configs/rdd_p0_distribution_shift.yaml",
        [
            out / "clic_reference_triangle.json",
            out / "clic_distribution_metrics_matrix.csv",
            out / "clic_criterion2_hits.json",
        ],
    )

    model = clean_features.build_feature_extractor("clean", device=device,
                                                   use_dataparallel=False)

    sources = sorted(CLIC_SRC.rglob("*.png"))
    print("CLIC sources found:", len(sources))
    if len(sources) != 428:
        raise RuntimeError("expected 428 CLIC sources, got %d" % len(sources))

    tmp = Path(tempfile.mkdtemp(prefix="rdd_p0_clic_"))
    try:
        # Build reference sets at native resolution.
        refs = {}
        real_dir = tmp / "real"
        real_dir.mkdir()
        for p in sources:
            (real_dir / (p.stem + ".png")).symlink_to(p.resolve())
        refs["real"] = real_dir

        for sigma in BLUR_SIGMAS:
            name = "blur_s%s" % ("%g" % sigma).replace(".", "p")
            d = tmp / name
            d.mkdir()
            for p in sources:
                Image.open(p).convert("RGB").filter(
                    ImageFilter.GaussianBlur(radius=sigma)).save(
                    d / (p.stem + ".png"), compress_level=1)
            refs[name] = d
            print("  built", name)

        d = tmp / "resample_2x"
        d.mkdir()
        for p in sources:
            im = Image.open(p).convert("RGB")
            w, h = im.size
            im.resize((w * 2, h * 2), Image.BICUBIC).resize((w, h), Image.LANCZOS).save(
                d / (p.stem + ".png"), compress_level=1)
        refs["resample_2x"] = d
        print("  built resample_2x")

        for q in JPEG_QUALITIES:
            name = "jpeg_q%d" % q
            d = tmp / name
            d.mkdir()
            for p in sources:
                im = Image.open(p).convert("RGB")
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=q)
                buf.seek(0)
                Image.open(buf).convert("RGB").save(d / (p.stem + ".png"), compress_level=1)
            refs[name] = d
            print("  built", name)

        ref_feats = {k: feats(v, model, device) for k, v in refs.items()}
        for k, v in ref_feats.items():
            print("  %-14s feats=%s" % (k, v.shape))

        triangle = {}
        for i, (k, f) in enumerate(sorted(ref_feats.items())):
            if k == "real":
                continue
            triangle[k] = {"fid_vs_real": float(clean_fid.fid_from_feats(ref_feats["real"], f)),
                           "kid_vs_real": kid(ref_feats["real"], f, KID_BASE_SEED + i)}
        (out / "clic_reference_triangle.json").write_text(
            json.dumps(triangle, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(triangle, indent=2))

        rows, idx = [], 0
        for arm in ARMS:
            for snr in SNRS:
                folder = RECON / arm / ("seed_%d" % SEED) / ("snr_%02d" % int(snr))
                af = feats(folder, model, device)
                if len(af) != 428:
                    raise RuntimeError("expected 428 recons at %s, got %d" % (folder, len(af)))
                for k, rf in sorted(ref_feats.items()):
                    idx += 1
                    rows.append({"arm": arm, "snr_db": snr, "reference": k,
                                 "n_arm": len(af), "n_reference": len(rf),
                                 "fid": float(clean_fid.fid_from_feats(rf, af)),
                                 "kid": kid(rf, af, KID_BASE_SEED + 2000 + idx),
                                 "kid_rng_seed": KID_BASE_SEED + 2000 + idx})
                print("  %-26s snr=%-5s done" % (arm, snr))

        with open(out / "clic_distribution_metrics_matrix.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        hits = []
        for arm in ARMS:
            for snr in SNRS:
                cell = [r for r in rows if r["arm"] == arm and r["snr_db"] == snr]
                vs_real = next(r for r in cell if r["reference"] == "real")
                for r in cell:
                    if r["reference"] == "real":
                        continue
                    if r["kid"] < vs_real["kid"]:
                        t = triangle.get(r["reference"], {})
                        hits.append({"arm": arm, "snr_db": snr, "reference": r["reference"],
                                     "kid_vs_reference": r["kid"], "kid_vs_real": vs_real["kid"],
                                     "kid_reference_vs_real": t.get("kid_vs_real"),
                                     "criterion2": "strong" if t.get("kid_vs_real", float("inf")) > r["kid"] else "weak"})
        (out / "clic_criterion2_hits.json").write_text(
            json.dumps(hits, indent=2) + "\n", encoding="utf-8")
        print("CLIC criterion2 hits:", len(hits))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
