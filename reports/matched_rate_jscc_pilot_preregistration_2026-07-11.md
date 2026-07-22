# Matched-Total-Rate Main + Structure DeepJSCC Pilot Preregistration

Date: 2026-07-11 (Asia/Shanghai)

Status: frozen before c=6/c=2 initialization evaluation or fine-tuning outcomes are inspected.

## Purpose and rate contract

The source-edge oracle established architectural feasibility but did not account for edge transmission. This pilot creates two explicit AWGN DeepJSCC paths using the same backbone convention as the existing formal baseline, whose CBR is `inner_channel / 48`:

```text
main RGB:       c=6, CBR=6/48=1/8
structure RGB:  c=2, CBR=2/48=1/24
total:          c=8, CBR=8/48=1/6
reference:      c=8, CBR=8/48=1/6
```

The structure representation packs sender-derived Sobel magnitude, absolute Laplacian and their channelwise maximum into RGB so the unmodified three-channel DeepJSCC backbone can be used. The receiver will later consume only the first two decoded maps as structural conditions.

## Data and warm start

Both arms use the same deterministic 20,000-image subset of COCO train2017 and the same deterministic 512-image validation subset. This is a convergence/architecture pilot, not the final full-data checkpoint.

The stable c=8 epoch-73 `best.pt` is the sole warm-start source. Latent real channels are ranked before target outcomes using the geometric mean of the encoder final-filter L2 norm and decoder first-filter L2 norm. The top `2c` channels are retained; all shape-compatible layers are copied exactly. Main and structure arms record selected channel indices and source checkpoint SHA-256.

## Training and selection

- Main RGB: 8 epochs, learning rate `5e-5`.
- Structure RGB: 12 epochs, learning rate `1e-4`.
- Both: AWGN 7 dB, batch 32, AdamW, MSE, AMP, gradient clipping 1.0.
- Checkpoint selection: minimum finite validation MSE, including the epoch `-1` warm-start initialization.
- Non-finite loss or metric is fail-closed and cannot overwrite the last finite checkpoint.

## Decision discipline

This pilot succeeds as an implementation milestone if both arms:

1. obey the exact `6+2=8` channel/rate contract;
2. reproduce complete finite validation metrics and checkpoints;
3. do not regress from their own warm-start initialization after the frozen fine-tuning schedule;
4. export receiver-visible RGB/structure reconstructions for the next residual-refiner stage.

The pilot does not yet claim that main+edge beats the c=8 baseline. That claim requires a combined refiner, independent noise realizations, matched total CBR, all five SNRs, LPIPS and semantic-drift evaluation. Official Imagenette validation remains sealed.
