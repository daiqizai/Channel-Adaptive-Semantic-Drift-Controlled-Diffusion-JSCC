"""Stage 3 of ANALYSIS-RDD-P0-DISTRIBUTION-SHIFT-001: FID/KID against every candidate
target distribution.

For each (arm, SNR) reconstruction set and each reference set R, computes FID and KID.
KID is the PRIMARY metric: with n=192 per cell the 2048-d Inception covariance is
rank-deficient, so FID is upward-biased (reported, but secondary).

Also computes the triangle quantity KID(R, real) so that a "closer to R than to real"
result can be checked for triviality: if R itself sits far from real, criterion 2 may
hold for uninteresting reasons. Preregistered as strong vs weak criterion 2.

Preregistration: reports/rdd_p0_distribution_shift_preregistration_2026-07-30.md
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path("/home/liulu/projects/channel-adaptive-semantic-drift-controlled-diffusion-jscc")
ANALYSIS_ID = "ANALYSIS-RDD-P0-DISTRIBUTION-SHIFT-001"

ARMS = ["s33_strong", "author_jscc", "diffjscc", "sgd_jscc"]
SNRS = [1.0, 4.0, 7.0, 13.0, 19.0]
KID_BASE_SEED = 20310900  # distinct from the A1 range to avoid any confusion


def kid_from_feats(f1, f2, seed, num_subsets=100, max_subset_size=None):
    """KID with max_subset_size pinned to the actual sample size (A1 convention)."""
    from cleanfid import fid as clean_fid
    if max_subset_size is None:
        max_subset_size = min(len(f1), len(f2))
    np.random.seed(seed)
    return float(clean_fid.kernel_distance(f1, f2, num_subsets=num_subsets,
                                           max_subset_size=max_subset_size))


def folder_feats(path: Path, model, device, batch_size=16, num_workers=4):
    from cleanfid import fid as clean_fid
    return clean_fid.get_folder_features(str(path), model=model, num_workers=num_workers,
                                         batch_size=batch_size, device=device,
                                         mode="clean", description="", verbose=False)


def snr_subset(arm_dir: Path, snr: float, tmp: Path) -> Path:
    """Materialize the (arm, SNR) subset as a flat folder cleanfid can read."""
    dst = tmp / ("%s__snr%02d" % (arm_dir.name, int(snr)))
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(arm_dir.glob("*__snr%02d.png" % int(snr))):
        (dst / p.name).symlink_to(p.resolve())
        n += 1
    if n == 0:
        raise RuntimeError("no images for %s at snr %s" % (arm_dir, snr))
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/analysis" / ANALYSIS_ID))
    args = ap.parse_args()

    from cleanfid import features as clean_features
    from cleanfid import fid as clean_fid
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out = Path(args.out)
    from rdd_p0_common import require_frozen_no_overwrite
    require_frozen_no_overwrite(
        ROOT / "configs/rdd_p0_distribution_shift.yaml",
        [
            out / "reference_triangle.json",
            out / "distribution_metrics_matrix.csv",
            out / "distribution_metrics_matrix.jsonl",
            out / "criterion2_hits.json",
        ],
    )

    arms_root = out / "arms"
    refs_root = out / "reference_sets"

    ref_dirs = sorted([d for d in refs_root.iterdir() if d.is_dir()])
    print("reference sets:", [d.name for d in ref_dirs])

    model = clean_features.build_feature_extractor("clean", device=device,
                                                   use_dataparallel=False)

    tmp = Path(tempfile.mkdtemp(prefix="rdd_p0_"))
    try:
        # Reference features (each reference set has n=64 unique source-derived images).
        ref_feats = {d.name: folder_feats(d, model, device) for d in ref_dirs}
        for name, f in ref_feats.items():
            print("  %-16s feats=%s" % (name, f.shape))

        # Triangle: how far is each reference set from real?
        real = ref_feats["real"]
        triangle = {}
        for i, (name, f) in enumerate(sorted(ref_feats.items())):
            if name == "real":
                continue
            triangle[name] = {
                "fid_vs_real": float(clean_fid.fid_from_feats(real, f)),
                "kid_vs_real": kid_from_feats(real, f, KID_BASE_SEED + i),
            }
        (out / "reference_triangle.json").write_text(
            json.dumps(triangle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(triangle, indent=2))

        rows = []
        idx = 0
        for arm in ARMS:
            arm_dir = arms_root / arm
            if not arm_dir.is_dir():
                print("  skip missing arm:", arm)
                continue
            for snr in SNRS:
                sub = snr_subset(arm_dir, snr, tmp)
                af = folder_feats(sub, model, device)
                for name, rf in sorted(ref_feats.items()):
                    idx += 1
                    rows.append({
                        "arm": arm,
                        "snr_db": snr,
                        "reference": name,
                        "n_arm": int(af.shape[0]),
                        "n_reference": int(rf.shape[0]),
                        "fid": float(clean_fid.fid_from_feats(rf, af)),
                        "kid": kid_from_feats(rf, af, KID_BASE_SEED + 1000 + idx),
                        "kid_rng_seed": KID_BASE_SEED + 1000 + idx,
                    })
                print("  %-12s snr=%-4s done (%d refs)" % (arm, snr, len(ref_feats)))

        import csv
        with open(out / "distribution_metrics_matrix.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        with open(out / "distribution_metrics_matrix.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Render criterion 2 per (arm, snr): is any non-real reference closer than real?
        verdicts = []
        for arm in ARMS:
            for snr in SNRS:
                cell = [r for r in rows if r["arm"] == arm and r["snr_db"] == snr]
                if not cell:
                    continue
                vs_real = next(r for r in cell if r["reference"] == "real")
                for r in cell:
                    if r["reference"] == "real":
                        continue
                    if r["kid"] < vs_real["kid"]:
                        t = triangle.get(r["reference"], {})
                        # Weak if the reference is itself further from real than the arm is.
                        strong = t.get("kid_vs_real", float("inf")) > r["kid"]
                        verdicts.append({
                            "arm": arm, "snr_db": snr, "reference": r["reference"],
                            "kid_vs_reference": r["kid"], "kid_vs_real": vs_real["kid"],
                            "kid_reference_vs_real": t.get("kid_vs_real"),
                            "criterion2": "strong" if strong else "weak",
                        })
        (out / "criterion2_hits.json").write_text(
            json.dumps(verdicts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("criterion2 hits:", len(verdicts))
        print(json.dumps(verdicts[:20], indent=2, ensure_ascii=False))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
