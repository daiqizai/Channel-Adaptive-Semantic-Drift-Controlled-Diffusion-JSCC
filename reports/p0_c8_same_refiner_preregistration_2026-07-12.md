# P0 B1: `c=8` Same-Capacity Refiner Preregistration

Date: 2026-07-12 (Asia/Shanghai)

Status: frozen before the first non-dry-run invocation.

## Question

Does the matched-rate `c=6` main + `c=2` decoded-structure system (`B3`) still outperform a single-path `c=8` reference when the reference receives a receiver-only refiner with the same architecture, training split, training budget, residual gates, and random seed (`B1`)?

This is the first P0 causal/fairness control. It does not by itself complete B0--B4.

## Fixed arms

- `B0`: frozen formal `c=8` DeepJSCC reconstruction, total CBR `8/48=1/6`.
- `B1`: `B0` plus a receiver-only Sobel/Laplacian residual refiner.
- `B3`: frozen S7 `c=6` main + `c=2` decoded structure plus decoded-structure residual refiner (`EXP-S7-002`), total CBR `6/48+2/48=1/6`.

`B1` and `B3` use the same refiner capacity and optimization contract:

- base channels: 64;
- residual blocks: 6;
- epochs: 60;
- train/eval samples: `32:192` / `192:256`;
- SNRs: `[1,4,7,13,19]` dB;
- residual gates: `[0.12,0.10,0.08,0.05,0.04]`;
- crop: 128; batch size: 16;
- loss: `MSE + 0.1 L1`;
- seed: `20260711`.

The only intended refiner-side difference is the structural source:

- `B1`: Sobel/Laplacian recomputed from the receiver-visible `c=8` reconstruction;
- `B3`: the first two channels of the receiver-visible decoded `c=2` structure packet.

The underlying communication systems remain necessarily different (`c=8` versus `c=6+c=2`); this is the intended rate-allocation/representation comparison, not a claim that their encoder training histories are identical.

## Integrity and leakage controls

- Use only existing COCO exports and local checkpoints.
- Do not access Imagenette policy-dev or official validation.
- Select the B1 checkpoint only by the frozen 64-image COCO eval PSNR, matching B3.
- Never overwrite an existing experiment directory.
- Validate that B1 and B3 original PNGs are byte-identical before paired comparison.
- Recompute paired PSNR from PNGs; do not compare rounded summary values.
- Cluster bootstrap by image ID across the five SNRs with 10,000 replicates.
- AlexNet results are pseudo-semantic diagnostics only.

## Primary comparison and decision rule

Primary endpoint:

```text
mean_image,SNR [ PSNR(B3 raw) - PSNR(B1 raw) ]
```

`B3` passes this first structure-increment gate only if:

1. the paired image-cluster bootstrap 95% CI lower bound is greater than 0;
2. at least four of five per-SNR point estimates are positive;
3. `B3` does not have more raw AlexNet pseudo new-error rows than `B1`, where new error means the arm is wrong relative to original top-1 while its own communication input was correct.

Secondary endpoints include B1/B3 raw PSNR relative to B0, M3/top-1 fallback PSNR, repair/new-error counts, parameter count, and measured refiner latency.

Failure does not invalidate the complete matched-rate system; it means the current evidence cannot attribute its gain to decoded structure after giving `c=8` the same-capacity restoration backend. No threshold, gate, epoch, or seed may be retuned after seeing this comparison.
