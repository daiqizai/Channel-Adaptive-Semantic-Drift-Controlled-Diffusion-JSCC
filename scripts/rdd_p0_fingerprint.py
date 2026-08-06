"""Stage 4 of ANALYSIS-RDD-P0-DISTRIBUTION-SHIFT-001: method fingerprint classifier.

Decisive test for criterion 1: given a single reconstruction, can a lightweight
classifier identify which method produced it? Accuracy significantly above chance is
direct evidence of identifiable per-method statistical fingerprints.

Anti-leakage: GroupKFold grouped by SOURCE IMAGE. Without this the classifier can win
by recognizing the image content rather than the method.

Preregistered artifact controls (all reported, no post-hoc selection):
  C0 full        -- 256x256, all features
  C1 center 128  -- avoids SGD's documented 4-patch seams at 128 boundaries
  C2 down 128    -- suppresses high-frequency resampling fingerprints
  C3 discriminative-only {s33, author_jscc} -- if high, fingerprints are NOT
                    prior-specific, which weakens the optimistic reading

Preregistration: reports/rdd_p0_distribution_shift_preregistration_2026-07-30.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/home/liulu/projects/channel-adaptive-semantic-drift-controlled-diffusion-jscc")
ANALYSIS_ID = "ANALYSIS-RDD-P0-DISTRIBUTION-SHIFT-001"

ARMS4 = ["s33_strong", "author_jscc", "diffjscc", "sgd_jscc"]
ARMS3 = ["s33_strong", "diffjscc", "sgd_jscc"]
ARMS2 = ["s33_strong", "author_jscc"]

BOOT_REPLICATES = 10000
BOOT_SEED = 20310950


# ----------------------------- features -----------------------------

def dct2(block: np.ndarray) -> np.ndarray:
    from scipy.fftpack import dct
    return dct(dct(block.T, norm="ortho").T, norm="ortho")


def radial_spectrum(gray: np.ndarray, nbins: int = 12):
    f = np.fft.fftshift(np.abs(np.fft.fft2(gray)))
    h, w = f.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    rmax = r.max()
    bins = np.linspace(0, rmax, nbins + 1)
    out = []
    for i in range(nbins):
        m = (r >= bins[i]) & (r < bins[i + 1])
        out.append(float(np.log1p(f[m].mean())) if m.any() else 0.0)
    # log-log slope of the radial profile (spectral falloff)
    centers = 0.5 * (bins[:-1] + bins[1:])
    ok = centers > 0
    slope = float(np.polyfit(np.log(centers[ok] + 1e-8), np.array(out)[ok], 1)[0])
    return out + [slope]


def features_one(img: np.ndarray) -> tuple[list[float], list[str]]:
    """img: HxWx3 uint8."""
    x = img.astype(np.float64) / 255.0
    gray = x.mean(axis=2)
    vals: list[float] = []
    names: list[str] = []

    # 1. Blockwise DCT band energy ratios (8x8 blocks).
    h, w = gray.shape
    bs = 8
    lo, mid, hi = [], [], []
    for i in range(0, h - bs + 1, bs):
        for j in range(0, w - bs + 1, bs):
            c = np.abs(dct2(gray[i:i + bs, j:j + bs]))
            lo.append(c[:3, :3].sum())
            mid.append(c[3:6, 3:6].sum())
            hi.append(c[6:, 6:].sum())
    lo_a, mid_a, hi_a = np.mean(lo), np.mean(mid), np.mean(hi)
    tot = lo_a + mid_a + hi_a + 1e-12
    vals += [lo_a / tot, mid_a / tot, hi_a / tot, float(np.std(hi) / (hi_a + 1e-12))]
    names += ["dct_lo_ratio", "dct_mid_ratio", "dct_hi_ratio", "dct_hi_cv"]

    # 2. Radial power spectrum bands + slope.
    rs = radial_spectrum(gray)
    vals += rs
    names += ["rps_b%02d" % i for i in range(len(rs) - 1)] + ["rps_slope"]

    # 3. High-pass residual statistics (Laplacian).
    k = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)
    from scipy.signal import convolve2d
    res = convolve2d(gray, k, mode="valid")
    vals += [float(res.std()), float(np.mean(np.abs(res))),
             float(((res - res.mean()) ** 3).mean() / (res.std() ** 3 + 1e-12)),
             float(((res - res.mean()) ** 4).mean() / (res.std() ** 4 + 1e-12))]
    names += ["hp_std", "hp_mad", "hp_skew", "hp_kurt"]

    # 4. Local variance / gradient distribution.
    gy, gx = np.gradient(gray)
    gm = np.sqrt(gx ** 2 + gy ** 2)
    vals += [float(gm.mean()), float(gm.std()),
             float(np.percentile(gm, 90)), float(np.percentile(gm, 99))]
    names += ["grad_mean", "grad_std", "grad_p90", "grad_p99"]

    # 5. Inter-channel correlation + chroma energy.
    r, g, b = x[..., 0].ravel(), x[..., 1].ravel(), x[..., 2].ravel()
    vals += [float(np.corrcoef(r, g)[0, 1]), float(np.corrcoef(r, b)[0, 1]),
             float(np.corrcoef(g, b)[0, 1]), float(x.std(axis=2).mean())]
    names += ["corr_rg", "corr_rb", "corr_gb", "chroma_std"]

    # 6. Saturation / clipping fractions.
    vals += [float((img <= 1).mean()), float((img >= 254).mean())]
    names += ["frac_near_black", "frac_near_white"]

    return vals, names


def load_variant(path: Path, variant: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if variant == "center128":
        img = img.crop((64, 64, 192, 192))
    elif variant == "down128":
        img = img.resize((128, 128), Image.BICUBIC)
    return np.asarray(img)


# ----------------------------- evaluation -----------------------------

def parse_stem(name: str):
    """`<wnid>__<file>__seed<seed>__snr<snr>.png` -> (source_key, seed, snr)."""
    base = name[:-4]
    parts = base.split("__")
    seed = next(p for p in parts if p.startswith("seed"))[4:]
    snr = next(p for p in parts if p.startswith("snr"))[3:]
    src = "__".join(p for p in parts if not (p.startswith("seed") or p.startswith("snr")))
    return src, int(seed), int(snr)


def build_matrix(out: Path, arms: list[str], variant: str):
    X, y, groups, meta, names = [], [], [], [], None
    for ai, arm in enumerate(arms):
        for p in sorted((out / "arms" / arm).glob("*.png")):
            src, seed, snr = parse_stem(p.name)
            v, names = features_one(load_variant(p, variant))
            X.append(v)
            y.append(ai)
            groups.append(src)
            meta.append({"arm": arm, "source": src, "seed": seed, "snr": snr})
    return np.asarray(X), np.asarray(y), np.asarray(groups), meta, names


def run_setting(out: Path, arms: list[str], variant: str, label: str, report: dict):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, groups, meta, names = build_matrix(out, arms, variant)
    chance = 1.0 / len(arms)
    print("\n=== %s: X=%s arms=%d chance=%.4f ===" % (label, X.shape, len(arms), chance))

    models = {
        "logreg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=5000, multi_class="auto")),
        "hgb": HistGradientBoostingClassifier(random_state=0),
    }

    res = {}
    for mname, clf in models.items():
        gkf = GroupKFold(n_splits=5)
        pred = np.zeros_like(y)
        for tr, te in gkf.split(X, y, groups):
            clf.fit(X[tr], y[tr])
            pred[te] = clf.predict(X[te])
        correct = (pred == y).astype(float)
        acc = float(correct.mean())

        # Bootstrap CI clustered by source image (matches the project's convention).
        uniq = np.unique(groups)
        rng = np.random.default_rng(BOOT_SEED)
        idx_by_src = {s: np.where(groups == s)[0] for s in uniq}
        boots = np.empty(BOOT_REPLICATES)
        for b in range(BOOT_REPLICATES):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            sel = np.concatenate([idx_by_src[s] for s in pick])
            boots[b] = correct[sel].mean()
        lo, hi = np.percentile(boots, [2.5, 97.5])

        cm = np.zeros((len(arms), len(arms)), dtype=int)
        for t, p in zip(y, pred):
            cm[t, p] += 1
        recall = [float(cm[i, i] / max(cm[i].sum(), 1)) for i in range(len(arms))]

        res[mname] = {
            "accuracy": acc, "ci95": [float(lo), float(hi)], "chance": chance,
            "ci_excludes_chance": bool(lo > chance),
            "confusion_matrix": cm.tolist(), "per_arm_recall": recall,
            "arms": arms, "n_samples": int(len(y)), "n_groups": int(len(uniq)),
        }
        print("  %-7s acc=%.4f CI=[%.4f,%.4f] chance=%.4f excludes=%s"
              % (mname, acc, lo, hi, chance, lo > chance))
        print("     per-arm recall:", ["%.3f" % r for r in recall])

    # Permutation importance on a single stratified-by-group split (logreg pipeline).
    from sklearn.inspection import permutation_importance
    gkf = GroupKFold(n_splits=5)
    tr, te = next(iter(gkf.split(X, y, groups)))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    clf.fit(X[tr], y[tr])
    pi = permutation_importance(clf, X[te], y[te], n_repeats=10, random_state=0)
    order = np.argsort(-pi.importances_mean)
    top = [{"feature": names[i], "importance": float(pi.importances_mean[i]),
            "std": float(pi.importances_std[i])} for i in order[:15]]
    res["top_features"] = top
    print("  top features:", [t["feature"] for t in top[:8]])

    report[label] = res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/analysis" / ANALYSIS_ID))
    args = ap.parse_args()

    out = Path(args.out)
    from rdd_p0_common import require_frozen_no_overwrite
    require_frozen_no_overwrite(
        ROOT / "configs/rdd_p0_distribution_shift.yaml",
        [out / "fingerprint_report.json"],
    )

    report: dict = {"analysis_id": ANALYSIS_ID, "stage": "fingerprint",
                    "bootstrap_replicates": BOOT_REPLICATES,
                    "cv": "GroupKFold(5) grouped by source image",
                    "official_validation_accessed": False}

    run_setting(out, ARMS4, "full", "C0_4arm_full", report)
    run_setting(out, ARMS3, "full", "S_3arm_full", report)
    run_setting(out, ARMS4, "center128", "C1_4arm_center128", report)
    run_setting(out, ARMS4, "down128", "C2_4arm_down128", report)
    run_setting(out, ARMS2, "full", "C3_discriminative_only", report)

    (out / "fingerprint_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\nwrote", out / "fingerprint_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
