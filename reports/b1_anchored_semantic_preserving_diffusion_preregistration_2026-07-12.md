# B1-Anchored Semantic-Preserving Short-Chain Diffusion Preregistration

Date: 2026-07-12 (Asia/Shanghai)

Status: frozen before the first non-dry-run invocation.

## Motivation

`ANALYSIS-S11-001` showed that the fair `c=8 + same-capacity receiver refiner` B1 is stronger than the current `c=6+c=2` decoded-structure system by `0.6217 dB`. Therefore the next diffusion attempt must challenge B1 rather than continue optimizing the weaker structure branch.

`EXP-S10-001` established that an anchor-near residual-shift bridge is stable enough to avoid the multi-dB collapse of naive residual DDPM, but it produced 12 pseudo new errors versus 7 repairs. This follow-up changes only the anchor/condition and preregistered preservation losses; steps, noise schedule, model capacity, seed, data split, and correction gates remain fixed.

## Fixed system

- Communication baseline: formal `c=8` DeepJSCC, CBR `8/48=1/6`, AWGN.
- Deterministic anchor: frozen B1 `EXP-S11-001` receiver-structure residual CNN.
- Anchor inputs: receiver-visible `c=8` reconstruction and SNR only.
- Diffusion structural condition: Sobel magnitude and absolute Laplacian recomputed from the frozen B1 anchor; no transmitted `c=2` side path.
- Pixel-domain residual-shift bridge, 20 train timesteps, 6 deterministic sampling steps, bridge sigma `0.05`.
- Denoiser: 64 base channels, 6 residual blocks, bounded anchor correction.
- Correction gates: `0.08/0.07/0.06/0.05/0.04` for `1/4/7/13/19` dB.
- Split: the same frozen COCO 160 train / 64 eval images.
- Seed: `20260720`, identical to `EXP-S10-001`.
- No Stable Diffusion, VAE, prompt, Imagenette policy-dev, or official validation.

## Training objective

The reconstruction-dominant objective is frozen as:

```text
L = 1.0 * MSE
  + 0.05 * pixel-L1
  + 0.05 * receiver-structure-L1
  + 0.0001 * ResNet18 target-distillation KL
```

The ResNet18 teacher uses an existing local ImageNet checkpoint and is used only during training. Final pseudo new-error diagnostics use frozen AlexNet, avoiding evaluation with the same architecture used in the preservation loss. These remain pseudo-semantic diagnostics, not supervised safety evidence.

The edge and semantic weights are fixed from existing project scales before the run. They may not be adjusted after smoke or formal results. Checkpoint selection remains eval PSNR only, matching S10; semantic diagnostics do not select the checkpoint.

## Decision rule

The exact variant is promising only if all conditions pass:

1. Mean raw diffusion-minus-B1-anchor LPIPS is negative.
2. Mean raw diffusion-minus-B1-anchor PSNR is at least `-0.10 dB`.
3. LPIPS improves at least 3/5 SNR points.
4. Raw AlexNet pseudo new-error rows relative to B1 do not exceed repairs.
5. Sampling uses at most 8 steps.

Secondary reporting must include per-SNR PSNR/LPIPS, top-1 fallback quality, sampling latency, parameter count, raw repair/new-error, and comparison with `EXP-S10-001`. Passing only authorizes a later independent supervised audit; failure is retained and stops further small-data tuning of this bridge family.
