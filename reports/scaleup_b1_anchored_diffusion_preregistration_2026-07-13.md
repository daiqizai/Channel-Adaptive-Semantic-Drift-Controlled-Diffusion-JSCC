# Train2017-Scale B1-Anchored Short-Chain Diffusion Preregistration

Date: 2026-07-13 (Asia/Shanghai)

Status: frozen before the first non-dry-run invocation.

## Question

Does the B1-anchored residual-shift diffusion retain its perceptual gain and eliminate the small-data incremental-risk failure when trained on 10,000 COCO train2017 images with an independent 1,000-image validation split?

## Fixed inputs

- Bare communication cache: `EXPORT-S13-001`, manifest SHA-256 `93ae3f3b47420c6bf84e1b5dce29601db1490f5519bb99fe39dcdad6680e2de9`.
- Frozen anchor: `EXP-S13-001` epoch-9 B1 checkpoint, SHA-256 `80133f9d9649c1a5d9514cf2b4f0d04802b6ebe03cc970bfcec86eddfd165562`.
- Train: 10,000 images × five SNRs = 50,000 pairs/epoch.
- Validation: 1,000 disjoint images × five SNRs.
- Imagenette policy-dev and official validation remain sealed.

## Fixed diffusion

- Pixel-domain residual-shift bridge.
- 20 train timesteps, bridge sigma `0.05`.
- 6 deterministic sampling steps from the frozen B1 anchor.
- Denoiser: 64 base channels, 6 residual blocks.
- Receiver-only Sobel/Laplacian condition computed from the B1 anchor.
- Correction gates: `0.08/0.07/0.06/0.05/0.04` for `1/4/7/13/19` dB.
- Seed: `20260720`, matching S12.

The loss remains exactly the S12 loss; no weight is retuned:

```text
L = MSE + 0.05 pixel-L1 + 0.05 edge-L1 + 0.0001 ResNet18 target KL
```

Training is frozen to 3 epochs. This exposes the model to 150,000 pairs, versus 24,000 total pair exposures in S12. Checkpoint selection uses validation PSNR only after each epoch; LPIPS and semantic events do not select the checkpoint.

## Promotion gate

This exact scale-up diffusion is promising only if all checks pass:

1. mean raw diffusion-minus-B1 LPIPS is negative;
2. mean raw diffusion-minus-B1 PSNR is at least `-0.05 dB`;
3. LPIPS improves at least 4/5 SNR points;
4. raw AlexNet pseudo new-error rows relative to B1 do not exceed repairs;
5. sampling uses no more than 8 steps;
6. all artifacts cover the full 1,000 × 5 validation grid without NaN.

AlexNet is not used in training or checkpoint selection. Its counts are diagnostics, not supervised safety evidence. Passing only authorizes independent risk calibration/audit; failure stops this residual-shift bridge family rather than authorizing validation tuning.
