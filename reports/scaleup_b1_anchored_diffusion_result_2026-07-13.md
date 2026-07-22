# Train2017-Scale B1-Anchored Diffusion Result

Date: 2026-07-13

## Decision

`EXP-S14-001` is **NEGATIVE** under its preregistered gate. Scale-up fixes the net incremental pseudo-semantic risk (`63 new errors <= 76 repairs`) but removes the LPIPS improvement and exceeds the PSNR budget.

| SNR | Raw ΔPSNR | Raw ΔLPIPS |
|---:|---:|---:|
| 1 | `-0.0747` | `+0.000279` |
| 4 | `-0.0807` | `+0.000287` |
| 7 | `-0.0778` | `+0.000090` |
| 13 | `-0.0863` | `-0.000105` |
| 19 | `-0.0487` | `-0.000145` |
| mean | **`-0.0736 dB`** | **`+0.000081`** |

- Best checkpoint: epoch 2; quick-eval ΔPSNR `-0.0633 dB`.
- Raw incremental new-error/repair: `63/76` (risk gate PASS).
- LPIPS improves only 2/5 SNRs; mean LPIPS worsens (FAIL).
- Mean PSNR is below the preregistered `-0.05 dB` floor (FAIL).
- Six-step latency: `14.97 ms/image`, versus B1 `2.52 ms/image`.
- Top-1 fallback mean ΔPSNR/ΔLPIPS: `-0.0706 dB/+0.000071`.

The 10k/1k scale-up therefore shows that more data can reverse the small-data net semantic harm, but the current reconstruction-dominant residual-shift objective does not add perception beyond the strong B1 anchor. Per preregistration, no learning-rate/loss/step tuning is allowed on this validation split.

This does not prove all diffusion backends are ineffective. It stops this exact bounded residual-shift bridge family. Any later diffusion attempt must be a materially different posterior/data-consistency design with a new development protocol, not another S14 hyperparameter scan.

Artifacts: `outputs/EXP-S14-001/`; anchor cache contains all 55,000 frozen B1 outputs. Imagenette was not accessed and no download occurred.
