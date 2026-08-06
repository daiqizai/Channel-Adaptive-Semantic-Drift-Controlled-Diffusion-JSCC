"""Stage 1 of ANALYSIS-RDD-P0-DISTRIBUTION-SHIFT-001.

Materialize the four arm image sets on the shared 64-image Imagenette policy-dev
population from EXISTING outputs only:

  s33_strong   -- exact deterministic replay of the frozen S33 checkpoint
  author_jscc  -- panel 1 of the S30 DiffJSCC montages
  diffjscc     -- panel 2 of the S30 DiffJSCC montages
  sgd_jscc     -- tile crop of the S20 SGD paper-upper montages

Every arm is verified against its historically recorded PSNR before being accepted.
Also writes the shared source set used to build all reference distributions.

Preregistration: reports/rdd_p0_distribution_shift_preregistration_2026-07-30.md
Config:          configs/rdd_p0_distribution_shift.yaml

No training, no downloads, no official-validation access.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/home/liulu/projects/channel-adaptive-semantic-drift-controlled-diffusion-jscc")
ANALYSIS_ID = "ANALYSIS-RDD-P0-DISTRIBUTION-SHIFT-001"

S30 = ROOT / "outputs/external_baselines/ANALYSIS-S30-DIFFJSCC-COMPARISON-001"
S20 = ROOT / "outputs/external_baselines/ANALYSIS-S20-SGD-B1-DECISION-001"

SNRS = [1.0, 4.0, 7.0, 13.0, 19.0]
SEEDS = [20260748, 20260749, 20260750]

# Preregistered verification gates.
GATE_MONTAGE_PSNR = 0.001
GATE_SGD_PSNR = 0.05
GATE_S33_PSNR = 0.001


def psnr_uint8(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64) / 255.0
    b = np.asarray(b, dtype=np.float64) / 255.0
    mse = float(np.mean((a - b) ** 2))
    return -10.0 * np.log10(max(mse, 1e-12))


def montage_key(sample_id: str, base_seed: int, snr_db: float) -> str:
    material = "%s|%s|%s" % (sample_id, base_seed, float(snr_db))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def crop_sgd_tile(grid: Image.Image, index: int, reconstruction: bool) -> Image.Image:
    """Reuse the already-verified S20 montage geometry (8 cols, stride 258)."""
    column = index % 8
    row = index // 8 + (8 if reconstruction else 0)
    left = 2 + column * 258
    top = 2 + row * 258
    return grid.crop((left, top, left + 256, top + 256)).convert("RGB")


def stem(sample_id: str, base_seed: int, snr_db: float) -> str:
    safe = sample_id.replace("/", "__").replace(".JPEG", "")
    return "%s__seed%d__snr%02d" % (safe, base_seed, int(snr_db))


def save(img: np.ndarray | Image.Image, path: Path) -> None:
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8), mode="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG", optimize=False, compress_level=1)


def build_montage_arms(out: Path, report: dict) -> dict:
    """author_jscc + diffjscc from S30 montages; also emit the shared source set."""
    with open(S30 / "per_sample.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 960, len(rows)

    err = {"author_jscc": [], "diffjscc": []}
    src_written: dict[str, str] = {}
    counts = {"author_jscc": 0, "diffjscc": 0, "source": 0}

    for r in rows:
        sid, seed, snr = r["sample_id"], int(r["base_seed"]), float(r["snr_db"])
        png = S30 / "images" / ("%s.png" % montage_key(sid, seed, snr))
        if not png.exists():
            raise FileNotFoundError(png)
        sheet = np.asarray(Image.open(png).convert("RGB"))
        if sheet.shape != (256, 768, 3):
            raise ValueError("unexpected montage shape %s at %s" % (sheet.shape, png))
        source = sheet[:, 0:256]
        panels = {"author_jscc": sheet[:, 256:512], "diffjscc": sheet[:, 512:768]}

        for arm, recon in panels.items():
            recorded = float(r["%s_psnr" % ("author_jscc" if arm == "author_jscc" else "diffjscc")])
            err[arm].append(abs(psnr_uint8(recon, source) - recorded))
            save(recon, out / "arms" / arm / ("%s.png" % stem(sid, seed, snr)))
            counts[arm] += 1

        # One canonical source copy per unique image (identical across seed/SNR).
        if sid not in src_written:
            key = sid.replace("/", "__").replace(".JPEG", "")
            save(source, out / "reference_sets" / "real" / ("%s.png" % key))
            src_written[sid] = hashlib.sha256(source.tobytes()).hexdigest()
            counts["source"] += 1
        else:
            got = hashlib.sha256(source.tobytes()).hexdigest()
            if got != src_written[sid]:
                raise ValueError("source image not stable across rows for %s" % sid)

    for arm in ("author_jscc", "diffjscc"):
        m = max(err[arm])
        report["gates"]["%s_max_abs_psnr_error_db" % arm] = m
        if m > GATE_MONTAGE_PSNR:
            raise AssertionError("%s PSNR gate failed: %.6f > %.6f" % (arm, m, GATE_MONTAGE_PSNR))

    report["counts"].update(counts)
    report["unique_source_images"] = len(src_written)
    return src_written


def build_sgd_arm(out: Path, src_written: dict, report: dict) -> None:
    with open(S20 / "population/population_manifest.json") as fh:
        manifest = json.load(fh)
    sample_ids = manifest["sample_ids"]
    if len(sample_ids) != 64:
        raise ValueError("expected 64 sample_ids, got %d" % len(sample_ids))

    recorded = {}
    for seed in SEEDS:
        with open(S20 / ("sgd_jscc_paper_protocol/seed_%d/per_sample.csv" % seed)) as fh:
            for r in csv.DictReader(fh):
                recorded[(r["sample_id"], int(r["base_seed"]), float(r["snr_db"]))] = r

    errs, mism, n = [], 0, 0
    for seed in SEEDS:
        for snr in SNRS:
            gp = S20 / ("sgd_jscc_paper_protocol/seed_%d/snr_%02d_source_sgdjscc.png" % (seed, int(snr)))
            grid = Image.open(gp)
            if grid.size != (2066, 4130):
                raise ValueError("unexpected SGD montage size %s at %s" % (grid.size, gp))
            for i, sid in enumerate(sample_ids):
                src = np.asarray(crop_sgd_tile(grid, i, False))
                key = sid.replace("/", "__").replace(".JPEG", "")
                # Cross-check against the DiffJSCC-derived source (byte identity).
                if hashlib.sha256(src.tobytes()).hexdigest() != src_written[sid]:
                    mism += 1
                rec = np.asarray(crop_sgd_tile(grid, i, True))
                row = recorded[(sid, seed, snr)]
                errs.append(abs(psnr_uint8(rec, src) - float(row["final_psnr"])))
                save(rec, out / "arms" / "sgd_jscc" / ("%s.png" % stem(sid, seed, snr)))
                n += 1

    report["gates"]["sgd_source_tile_byte_mismatches"] = mism
    report["gates"]["sgd_max_abs_psnr_error_db"] = max(errs)
    report["gates"]["sgd_median_abs_psnr_error_db"] = float(np.median(errs))
    report["counts"]["sgd_jscc"] = n
    if mism:
        raise AssertionError("SGD source tiles disagree with DiffJSCC panels: %d" % mism)
    if max(errs) > GATE_SGD_PSNR:
        raise AssertionError("SGD PSNR gate failed: %.6f > %.6f" % (max(errs), GATE_SGD_PSNR))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/analysis" / ANALYSIS_ID))
    ap.add_argument("--skip-s33", action="store_true",
                    help="build montage-derived arms only (S33 replay handled separately)")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        print("refusing to write into non-empty %s" % out, file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)

    report: dict = {"analysis_id": ANALYSIS_ID, "stage": "build_arms",
                    "counts": {}, "gates": {},
                    "official_validation_accessed": False,
                    "sgd_non_ranking": True}

    src = build_montage_arms(out, report)
    build_sgd_arm(out, src, report)

    (out / "build_arms_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
