# Posterior Receiver-Risk Controller Development Preregistration

Date: 2026-07-14. Frozen before the formal fit command. Analysis ID: `ANALYSIS-PC-RISK-CTRL-DEV-001`.

## Status and scope

This is an explicitly non-held-out controller-development package. Its inputs are the already exposed `ANALYSIS-PC-RISK-FEAT-001` policy-dev rows. Passing is necessary to freeze a candidate for a new channel-seed audit, but it is not generalization evidence and cannot unlock official Imagenette validation.

The formal script must verify the frozen SHA-256 values of `risk_features.csv`, `per_sample.csv`, and `risk_feature_schema.json`, require exactly `28,410` aligned unique keys, and reject official-validation access.

## Frozen low-complexity score

The controller is deliberately restricted to six receiver-visible inputs:

- high `G_gate` raw-to-posterior JS divergence;
- high `G_gate` anchor-to-posterior JS divergence;
- high `G_aux` raw-to-posterior JS divergence;
- high `G_aux` anchor-to-posterior JS divergence;
- low `G_gate` posterior confidence;
- low `G_aux` posterior confidence.

For each component, fit an empirical CDF on rows satisfying development-only `teacher_clean_correct AND teacher_anchor_correct`; low-confidence components are sign-reversed before fitting. At application time each raw receiver feature becomes its right-sided empirical percentile, and the risk score is the arithmetic mean of the six percentiles. No classifier, tree, learned coefficient, WNID, class index, sample ID, original-image value, ground-truth value, or `teacher_*` field enters the deployed score.

This transparent score was chosen after exploratory diagnostics on policy-dev. That exploration is part of development and is why the same rows cannot be presented as independent evidence.

## Frozen threshold search

Evaluate target reference rejection rates in exactly this increasing order:

`[0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]`.

For each rate, use NumPy `quantile(..., method="higher")`; reject when `score >= threshold`, otherwise return posterior. A rejected row returns the frozen B1 anchor. Select the first rate passing every gate:

1. primary `[1,4,7]` dB final new-error rows do not exceed S14 raw in total, at any SNR, or for any channel seed;
2. primary final failure rows do not exceed raw in total or for any channel seed;
3. one-sided 95% Clopper-Pearson upper bound for primary final new-error image clusters is at most `0.005`;
4. mean final-minus-raw PSNR is positive and LPIPS is nonpositive overall and for every channel seed;
5. final retains at least `80%` of the posterior mean PSNR gain over raw.

All semantic selection uses frozen `T_cls` only as a development teacher. The controller artifact must serialize only the six receiver feature names, directions, empirical CDFs, arithmetic-mean rule, and selected threshold.

If no candidate passes, the package is negative and no threshold may be invented. If one passes, freeze it before declaring or generating the new audit seed.
