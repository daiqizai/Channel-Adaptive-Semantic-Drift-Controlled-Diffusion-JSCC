# Received-Latent Posterior Correction Independent Replication Result

Date: 2026-07-13. Analysis ID: `ANALYSIS-PC-002`.

## Verdict

**NEGATIVE under the complete preregistered gate, with a strongly positive quality/data-consistency replication.** The PC-001 posterior correction benefit replicates almost exactly on 256 new images × five SNRs, but semantic new-error risk increases under all three audit classifiers.

## Results

- Mean normalized latent loss: `0.10460 → 0.08328` (`-20.4%` relative); decreases at 5/5 SNRs.
- Posterior minus S14 raw PSNR: `+0.2125 dB`; positive at 5/5 SNRs.
- Posterior minus S14 raw LPIPS: `-0.01078`; improves at 5/5 SNRs.
- Per-SNR PSNR delta: `+0.0326/+0.0657/+0.1158/+0.3230/+0.5254 dB` at `1/4/7/13/19 dB`.
- Ensemble-majority new error: `0 → 2`; repair: `2 → 21`.
- AlexNet new/repair: `15/15 → 20/61`.
- ResNet18 new/repair: `11/9 → 23/66`.
- MobileNetV3-Small new/repair: `15/9 → 53/127`.

The three quality/consistency gates pass. Both semantic robustness gates fail. The correction produces many more repairs, but net or majority improvements cannot hide newly introduced errors.

## Interpretation

The near-identical PC-001/PC-002 PSNR gain (`+0.2124/+0.2125 dB`) and LPIPS gain (`-0.00991/-0.01078`) establish that received-latent correction is not a 64-image accident. However, measurement consistency is not semantic consistency. The next method needs explicit receiver-side semantic failure handling; increasing correction strength is not justified.

Preregistration: `reports/posterior_consistency_independent_replication_preregistration_2026-07-13.md`. Outputs: `outputs/analysis/s16_posterior_consistency_independent_replication/`.
