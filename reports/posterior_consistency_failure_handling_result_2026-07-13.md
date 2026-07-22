# Posterior Correction with Receiver-Side Failure Handling Result

Date: 2026-07-13. Analysis ID: `ANALYSIS-PC-003`.

## Verdict

**NEGATIVE under the complete preregistered cross-model gate; positive quality-preserving risk reduction.** A frozen AlexNet agreement fallback sharply reduces the uncontrolled posterior's ensemble-majority new errors, but one majority new error remains and the two independent audit classifiers still show increased new-error counts.

## Results

- Uncontrolled posterior again replicates: latent loss `0.10508 → 0.08363`, PSNR `+0.2172 dB`, LPIPS `-0.01072`, all quality/consistency gates pass.
- Receiver controller acceptance: `87.66%`.
- Controlled final versus S14 raw: PSNR `+0.2062 dB`, LPIPS `-0.00910`.
- Ensemble-majority new/repair:
  - S14 raw: `0/0`
  - uncontrolled posterior: `4/31`
  - controlled final: `1/11`
- AlexNet final new/repair: `0/0`, by the frozen agreement/fallback design.
- ResNet18 raw versus final new/repair: `11/16 → 19/81`.
- MobileNetV3-Small raw versus final new/repair: `16/13 → 43/87`.

The controller preserves most posterior quality benefit and removes three of four ensemble-majority new errors, but it does not transfer reliably beyond its control classifier. Therefore it is not promoted as a safe M3.

## Stage conclusion

Across PC-001--003, received-latent posterior correction has become a reproducible restoration mechanism, not a final method: two independent 256-image-scale replications retain roughly `+0.21 dB` PSNR and `-0.0107` LPIPS versus S14 raw. The remaining research bottleneck is cross-model semantic control. Next work should train or calibrate a controller on separate development data and leave at least one semantic model or supervised label source untouched for audit; it should not tune correction steps on PC-002/003.

Preregistration: `reports/posterior_consistency_failure_handling_preregistration_2026-07-13.md`. Outputs: `outputs/analysis/s17_posterior_consistency_failure_handling/`.
