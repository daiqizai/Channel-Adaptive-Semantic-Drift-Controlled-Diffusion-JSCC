# P0 `c=8 + Same Refiner` Fairness Result

Date: 2026-07-12

## Decision

The current `c=6` main + `c=2` decoded-structure system does **not** pass the first structure-increment fairness gate after the `c=8` reference receives an identically sized and identically trained receiver-only refiner.

This is a negative causal-attribution result, not a failure of the complete system and not a rejection of diffusion.

## Matched contract

`EXP-S11-001` (B1) and `EXP-S7-002` (B3) use:

- total CBR `8/48=1/6`;
- the same 160-image train / 64-image eval split;
- the same seed `20260711`;
- 64 base channels and 6 residual blocks;
- 60 epochs, batch size 16, 128-pixel crops;
- the same `MSE + 0.1 L1` loss;
- the same SNR-dependent residual gates;
- exactly 448,387 refiner parameters;
- approximately 2.5 ms/image refiner latency.

B1 derives Sobel/Laplacian conditions from the received `c=8` reconstruction. B3 receives a separately transmitted and decoded `c=2` structure packet beside the `c=6` main reconstruction.

## Primary result

| Endpoint | Result |
|---|---:|
| B1 raw − bare B0 PSNR | `+1.0192 dB` |
| B3 raw − bare B0 PSNR | `+0.3974 dB` |
| B3 raw − B1 raw PSNR | `-0.6217 dB` |
| 95% image-cluster bootstrap CI | `[-0.6654,-0.5839] dB` |
| Positive SNR points | `0/5` |
| B3 raw − B1 raw LPIPS | `+0.00664` (worse) |
| B1 raw new-error / repair rows | `31 / 45` |
| B3 raw new-error / repair rows | `37 / 57` |

Per-SNR B3-minus-B1 raw PSNR is negative at every point:

| SNR | Delta PSNR | 95% CI |
|---:|---:|---:|
| 1 | `-0.5393` | `[-0.6043,-0.4840]` |
| 4 | `-0.6006` | `[-0.6488,-0.5559]` |
| 7 | `-0.5930` | `[-0.6393,-0.5500]` |
| 13 | `-0.6510` | `[-0.7006,-0.6061]` |
| 19 | `-0.7247` | `[-0.7851,-0.6696]` |

All three preregistered checks fail: the primary CI is not positive, zero of five SNR estimates are positive, and B3 has more pseudo new-error rows than B1.

## What changes

The former comparison against bare `c=8` proved that the complete `c=6+c=2+refiner` system was better than an under-equipped reference. It did not establish that decoded structure caused the improvement. B1 now shows that a single-path `c=8` reconstruction with the same restoration capacity is substantially stronger in both PSNR and LPIPS.

Therefore:

1. Do not describe the current decoded-structure side channel as the main source of gain.
2. Do not spend more experiments tuning the current post-hoc `c=2` structure/sketch allocation to rescue this claim.
3. Use B1 as the new deterministic quality anchor for future diffusion experiments.
4. Keep the project's differentiating question on refinement-induced new error and semantic-risk control.
5. A future semantic side channel is only justified if it supplies sample-specific risk information that B1 cannot infer locally, under an explicit rate/power budget.

## Boundaries

- The `c=8` and `c=6/c=2` encoder training histories are not identical; B3 remains a 20k warm-start pilot.
- AlexNet events are pseudo-semantic diagnostics, not supervised safety evidence.
- This result does not claim that every learned structure representation or semantic token is ineffective.
- Imagenette policy-dev and official validation were not accessed.

Primary artifacts:

- `outputs/EXP-S11-001/`
- `outputs/analysis/s11_p0_b1_b3_paired_comparison/`
- `reports/p0_c8_same_refiner_preregistration_2026-07-12.md`
