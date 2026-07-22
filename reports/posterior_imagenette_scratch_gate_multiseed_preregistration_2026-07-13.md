# Posterior Imagenette Scratch-Gate Multi-Seed Preregistration

Date: 2026-07-13. Frozen before non-dry-run. Analysis ID: `ANALYSIS-PC-RISK-REP-001`. Phase remains S5 validation.

## Purpose and evidence boundary

This is a frozen channel-randomness replication of `ANALYSIS-PC-RISK-001`, not a new controller search. It tests whether the remaining supervised tail error is specific to one AWGN realization. It reuses already unsealed Imagenette policy-dev images, so it cannot establish image-population generalization or replace the sealed official validation.

Official Imagenette validation remains inaccessible. The run must reject a manifest or classifier checkpoint that records official-validation access.

## Frozen protocol

- New AWGN seeds: `[20260722, 20260723, 20260724]`; none was used by PC-SUP or PC-RISK-001.
- SNRs: `[1,4,7,13,19]` dB; primary SNRs: `[1,4,7]` dB; `c=8`, CBR `1/6`.
- Images: all `1894` existing policy-dev images for every seed/SNR.
- Clean-correct: scratch calibrated `T_cls(original)=WNID` with confidence `>=0.50`; membership is independent of channel seed.
- Restoration: frozen S13 B1, frozen S14 diffusion, exactly three received-latent correction steps with normalized step size `0.001`.
- Controller: frozen scratch MobileNetV3-Small `G_gate`; accept iff `G_gate(posterior).top1 == G_gate(anchor).top1`, otherwise return anchor.
- Outcome evaluator: frozen scratch ResNet18 `T_cls`, prohibited from entering restoration or controller decisions.
- No threshold, checkpoint, step count, SNR rule, exception list, or event-specific handling may change after the run begins.

Rows must record channel seed. Reports must aggregate both by SNR across seeds and by seed/SNR. Image-cluster tail risk treats an image as an event if the controlled final creates at least one new error at any primary SNR or seed; repeated channel rows do not count as independent images.

## Frozen gates

All conditions are required:

1. At least `1600` unique policy-dev images are clean-correct.
2. Received-latent consistency decreases in every one of the `3×5` seed/SNR cells and in all five seed-aggregated SNR rows.
3. Posterior PSNR is positive at at least four aggregated SNRs and posterior LPIPS is nonpositive at at least four.
4. Controlled-final mean PSNR is positive and LPIPS nonpositive overall and separately for every channel seed.
5. Across primary SNRs, controlled-final new-error rows do not exceed S14 raw in total, at any SNR after aggregating seeds, or for any seed after aggregating primary SNRs.
6. Controlled-final failure rows do not exceed raw across all primary rows or within any channel seed.
7. The one-sided 95% Clopper-Pearson upper bound for primary controlled-final new-error image clusters, conditional on an image having at least one anchor-correct primary row, is at most `0.005`.

A complete pass is necessary before drafting an official-validation final lock, but is not by itself permission to access official validation. A failure keeps official validation sealed and must be reported without seed replacement or controller tuning.
