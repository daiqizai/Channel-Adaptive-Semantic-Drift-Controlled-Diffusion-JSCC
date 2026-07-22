# Scale-Up B1 Anchor Preregistration

Date: 2026-07-12 (Asia/Shanghai)

Status: frozen before the first non-dry-run training invocation.

## Question

Does the receiver-only `c=8 + Sobel/Laplacian residual refiner` remain a strong quality anchor when trained on the independent COCO train2017 scale-up population rather than 160 val2017 development images?

## Fixed data and system

- Cache: `EXPORT-S13-001`, manifest SHA-256 `93ae3f3b47420c6bf84e1b5dce29601db1490f5519bb99fe39dcdad6680e2de9`.
- Train: 10,000 train2017 images × five SNRs.
- Validation: 1,000 disjoint train2017 images × five SNRs.
- Bare communication input: frozen formal `c=8` DeepJSCC cache, AWGN, CBR `1/6`.
- Refiner input: received RGB, received-image Sobel/Laplacian, and SNR only.
- Refiner: 64 base channels, 6 residual blocks, 448,387 parameters.
- Residual gates: `0.12/0.10/0.08/0.05/0.04` at `1/4/7/13/19` dB.
- Training: 10 epochs, batch 16, 128 crops, random flip, `MSE + 0.1 L1`, seed `20260714`.
- Checkpoint selection: mean validation PSNR only, evaluated after every epoch.

No semantic model participates in training or checkpoint selection. Frozen AlexNet and LPIPS are evaluation diagnostics only. No Imagenette split is accessed.

## Promotion rule

The scale-up B1 anchor is usable for the next diffusion stage only if:

1. raw mean PSNR improvement over bare `c=8` is at least `+0.80 dB`;
2. raw PSNR improves at all 5 SNR points;
3. mean LPIPS delta is negative and LPIPS improves at least 4/5 SNR points;
4. raw AlexNet pseudo new-error rows do not exceed repairs;
5. training/evaluation completes without NaN and the saved manifest exactly covers 10k/1k × 5 SNR.

Passing does not establish semantic safety. It only freezes a better-trained deterministic anchor. Failure stops scale-up diffusion until the anchor/data pipeline is corrected; it does not authorize hyperparameter scanning on the validation set.
