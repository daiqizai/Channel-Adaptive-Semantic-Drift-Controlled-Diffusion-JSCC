# Short-Chain Residual-Shift Diffusion Pilot Preregistration

Date: 2026-07-12 (Asia/Shanghai)

Status: frozen before the first non-dry-run invocation.

## Question

Can a pixel-domain, short-chain conditional diffusion bridge improve the frozen matched-rate decoded-structure residual-CNN anchor without using Stable Diffusion, a VAE, text prompts, or pure-Gaussian full-residual initialization?

## Fixed system

- Channel/rate: AWGN, total CBR `c6+c2=8/48=1/6`.
- Main and structure inputs: frozen S7 matched-rate exports.
- Deterministic anchor: frozen `EXP-S7-002` decoded-structure residual CNN.
- Diffusion input at inference: the deterministic anchor itself.
- Conditioning: decoded structure, normalized SNR, bridge time.
- Reverse process: deterministic residual-shift bridge, at most 6 steps.
- Domain: RGB pixel space; no SD VAE or latent autoencoder.
- Train/eval split: the frozen S7 `160/64` COCO development split.
- This is a design pilot, not a final M2/M3 promotion experiment.

## Forward bridge

For target `x`, deterministic anchor `a`, bridge time `tau in (0,1]`, and Gaussian noise `eps`:

```text
x_tau = (1-tau) * x + tau * a + sigma * sqrt(tau*(1-tau)) * eps
```

The denoiser predicts a bounded correction from `a` to `x`. At inference, sampling starts exactly from `a` at `tau=1` and deterministically follows six predicted bridge updates to `tau=0`. It never starts from an unconstrained Gaussian image/residual.

## Comparisons

- `main_c6`: received main reconstruction.
- `anchor`: frozen S7 residual CNN output.
- `diffusion_raw`: short-chain residual-shift output.
- `diffusion_top1_fallback`: accept raw only when its frozen AlexNet top-1 equals the anchor top-1; otherwise return anchor.

The AlexNet metric is diagnostic only. No supervised-safe claim is permitted from this COCO pilot.

## Pilot decision rule

The pilot is promising only if all of the following hold:

1. Mean `diffusion_raw - anchor` LPIPS is negative.
2. Mean `diffusion_raw - anchor` PSNR is greater than `-0.20 dB`.
3. At least three of five SNR points improve LPIPS.
4. Raw AlexNet pseudo new errors relative to the anchor do not exceed raw repairs.
5. Sampling uses no more than 8 steps and all outputs/metadata are reproducible.

Passing does not promote the method. A later supervised audit must still show controller efficacy and acceptable new-error upper bound. Failure means this residual-shift design is retained as a negative result; it does not authorize further unbounded diffusion tuning.

## Integrity

- No existing output directory may be overwritten.
- Checkpoint selection uses only the frozen COCO eval split PSNR, never Imagenette.
- Existing Imagenette policy-dev and official validation are not accessed.
- No download is required; all checkpoints and classifier weights are local.
