# Received-Latent Posterior Correction Pilot Preregistration

Date: 2026-07-13. Frozen before non-dry-run. Analysis ID: `ANALYSIS-PC-001`.

Use 64 previously unused train2017 images ranked immediately after the frozen S13 10k/1k population (rank positions 11000--11063), five SNRs, seed 20260715. No S13/S14 validation image or Imagenette image is accessed.

For each image, reproduce formal c8 transmission and retain the actual received latent `y`; compute frozen S13 B1 anchor and frozen S14 diffusion candidate. Starting from S14 raw, apply exactly three normalized-gradient proximal steps:

```text
Ldc = mean(|E(x)-y|^2) / mean(|y|^2)
g = dLdc/dx
x <- clamp(x - 0.001 * g / RMS(g), 0, 1)
```

No parameter, threshold, or step is selected on this split. Primary feasibility gate: posterior must reduce received-latent loss on all five SNRs without mean PSNR falling more than 0.05 dB below S14 raw. Promotion gate additionally requires mean LPIPS no worse than S14 raw and posterior incremental pseudo new errors relative to B1 not exceed S14 raw. Passing only authorizes a trained posterior sampler; it is not final safety evidence.
