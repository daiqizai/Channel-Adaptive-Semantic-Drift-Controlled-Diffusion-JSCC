# Posterior Imagenette Supervised Policy-Dev Audit Result

Date: 2026-07-13. Analysis ID: `ANALYSIS-PC-SUP-001`. Official Imagenette validation was not accessed.

## Verdict

**NEGATIVE under the complete preregistered gate, but the first strong supervised partial-success result for posterior diffusion.** Aggregate supervised reliability and quality improve, while one 7 dB new error violates the frozen per-SNR no-increase gate.

## Results

- Policy-dev images: `1894`; scratch `T_cls` clean-correct at confidence `>=0.50`: `1697`.
- Mean posterior versus S14 raw: PSNR `+0.2654 dB`, LPIPS `-0.00610`.
- Mean controlled final versus raw: PSNR `+0.2543 dB`, LPIPS `-0.00531`.
- Primary SNRs `[1,4,7]`, supervised counts:
  - raw failure/new: `69/4`
  - posterior failure/new: `56/4`
  - controlled final failure/new: `62/4`
- Per-SNR final versus raw new errors:
  - 1 dB: `3 vs 3`
  - 4 dB: `0 vs 1`
  - 7 dB: `1 vs 0` **(gate failure)**
- At 13/19 dB, final failure is `3/2`, with zero new errors.

Every consistency and quality gate passes. Total primary new errors do not increase and primary failure improves `69 → 62`. The only failed gate is the explicitly preregistered requirement that new errors not increase at each primary SNR.

## Interpretation

This resolves the central ambiguity from classifier-ensemble audits: under real Imagenette WNID labels and a scratch independent evaluator, posterior correction has genuine net semantic benefit, not merely pseudo-label churn. It is still not reliable enough to unlock official validation because the 7 dB row contains one new error where raw contains none.

The next method decision is now narrow: retain the frozen posterior restoration component, stop consensus-rule expansion, and train/calibrate a semantic-risk controller on separate development supervision. PC-SUP policy-dev must not be reused as its final audit set. Official validation remains sealed.

Preregistration: `reports/posterior_imagenette_supervised_audit_preregistration_2026-07-13.md`. Outputs: `outputs/analysis/pc_imagenette_supervised_policy_dev/`.
