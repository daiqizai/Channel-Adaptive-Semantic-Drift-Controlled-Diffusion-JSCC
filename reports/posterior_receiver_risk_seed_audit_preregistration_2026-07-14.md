# Frozen Receiver-Risk New-Seed Audit Preregistration

Date: 2026-07-14. Frozen before generating any row with channel seed `20260725`. Analysis IDs: feature generation `ANALYSIS-PC-RISK-SEED-AUDIT-FEAT-001`; controller audit `ANALYSIS-PC-RISK-SEED-AUDIT-001`.

## Question and evidence boundary

Does the frozen six-feature percentile controller transfer to a new AWGN realization without threshold or model changes?

Seed `20260725` was absent from the repository's source/config/report ledgers before this preregistration. The audit reuses the already exposed `1894` policy-dev images, so it is a channel-randomness holdout, not an image-population holdout. A pass does not by itself unlock official Imagenette validation.

## Frozen inputs

- Feature extraction config SHA-256: `c1465c585b7e2e12c246668e8b71777c831b4dcf8408dfc6b9ce130ab93d5d33`.
- Controller JSON SHA-256: `3ff792a366074202d1727042c40c0cbc777843a5c65d48276e3dfd9be6199f6f`.
- Empirical-CDF NPZ SHA-256: `2a7f062ab53da9309f6ccad8b9c2a8977b9b79e5d95c760559d674d2503e6956`.
- Controller threshold: `0.8537265316368728`; reject when score is greater than or equal to the threshold.
- Controller inputs: exactly the four frozen `G_gate/G_aux` raw-or-anchor to posterior JS divergences and the two sign-reversed posterior confidences recorded in the controller JSON. `T_cls`, true label, original prediction, WNID, class index, and sample ID are prohibited inputs.
- Candidate/fallback: accept corrected posterior; on rejection return the frozen B1 anchor.
- AWGN seed: exactly `20260725`; SNR `[1,4,7,13,19]` dB; primary `[1,4,7]` dB; all `1894` policy-dev images; exactly `9470` rows.
- S13 B1, S14 diffusion, DeepJSCC, three-step correction, `G_gate`, `G_aux`, and `T_cls` remain byte-frozen from the feature-development package.

No exception list, per-SNR threshold, event-specific rule, checkpoint replacement, or alternate seed is permitted after generation starts.

## Frozen pass gates

All conditions are required:

1. At least `1600` unique clean-correct images; exactly `9470` unique `(sample_id, seed, SNR)` feature/audit rows; all controller scores finite; official validation unaccessed.
2. The clean+anchor-correct reference rejection rate on the new seed is between `5%` and `15%`, guarding against a distribution-collapse pass.
3. Across primary SNRs, controlled final new-error rows do not exceed S14 raw in total or at any SNR.
4. Across primary SNRs, controlled final failure rows do not exceed raw in total or at any SNR.
5. The one-sided 95% Clopper-Pearson upper bound for primary controlled-final new-error image clusters is at most `0.005`.
6. Mean final-minus-raw PSNR is positive and LPIPS nonpositive overall and at every one of the five SNRs.
7. The controlled final retains at least `80%` of the new-seed posterior mean PSNR gain over raw.

The existing scratch top-1 fallback is reported as a frozen reference, not used to tune or override the percentile controller. Failure keeps the controller developmental and the official validation sealed. Passing only establishes channel-seed replication on the same image population; a separate image-population lock remains necessary.
