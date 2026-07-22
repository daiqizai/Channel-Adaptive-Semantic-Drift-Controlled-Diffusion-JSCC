# Scale-Up B1 Anchor Result

Date: 2026-07-13

## Outcome

`EXPORT-S13-001` and `EXP-S13-001` completed successfully. The 10k/1k COCO train2017 internal split and its receiver-only `c=8` residual restoration anchor pass every preregistered promotion gate.

This checkpoint replaces the 160-image `EXP-S11-001` model as the deterministic anchor for subsequent scale-up diffusion development. It is not itself a diffusion result or supervised semantic-safety claim.

## Data cache integrity

- Source population: local COCO train2017, 118,287 images.
- Deterministic SHA-ranked split: 10,000 train + 1,000 validation.
- Source manifest SHA-256: `93ae3f3b47420c6bf84e1b5dce29601db1490f5519bb99fe39dcdad6680e2de9`.
- Unique selected source paths/hashes: `11,000/11,000`.
- Verified overlap with local val2017 by filename or SHA-256: zero.
- Original PNGs: 11,000.
- Reconstructions: 11,000 at each of five SNRs, 55,000 total.
- Per-sample rows: 55,000 unique `(sample,SNR)` pairs.
- Cache size: approximately 6.9 GB.
- No download or network access.

Bare `c=8` cache quality:

| SNR | Mean PSNR |
|---:|---:|
| 1 | `28.0327 dB` |
| 4 | `30.0414 dB` |
| 7 | `31.5389 dB` |
| 13 | `33.1449 dB` |
| 19 | `33.6661 dB` |

## Scale-up B1 training

- Train grid: 10,000 images × 5 SNR = 50,000 pairs/epoch.
- Validation grid: 1,000 images × 5 SNR = 5,000 pairs.
- Model: receiver RGB + receiver-derived Sobel/Laplacian + SNR, 64 channels × 6 blocks.
- Parameters: 448,387.
- Training: 10 epochs, `MSE + 0.1 L1`, seed `20260714`.
- Best checkpoint: epoch 9, validation PSNR `32.5588 dB`.
- Best checkpoint SHA-256: `80133f9d9649c1a5d9514cf2b4f0d04802b6ebe03cc970bfcec86eddfd165562`.

## Formal validation result

| SNR | Raw ΔPSNR | Raw ΔLPIPS | New error | Repair |
|---:|---:|---:|---:|---:|
| 1 | `+1.7144` | `-0.08469` | 90 | 320 |
| 4 | `+1.3698` | `-0.03980` | 114 | 245 |
| 7 | `+1.2090` | `-0.01849` | 85 | 185 |
| 13 | `+1.2225` | `-0.00904` | 33 | 104 |
| 19 | `+1.3002` | `-0.01157` | 17 | 97 |
| mean / total | **`+1.3632 dB`** | **`-0.03272`** | **339** | **951** |

Top-1 fallback yields mean `ΔPSNR=+0.8384 dB` and `ΔLPIPS=-0.01529`, with final AlexNet failure equal to bare `c=8` by construction. Raw restoration itself reduces pseudo failures by 612 net rows, but the 339 individual new errors remain important and must not be hidden by the net average.

Mean refiner latency is `2.521 ms/image`.

## Preregistered decision

- Mean raw PSNR improvement at least `+0.80 dB`: PASS.
- PSNR improves at 5/5 SNRs: PASS.
- Mean LPIPS improves and at least 4/5 SNRs improve: PASS (`5/5`).
- Raw pseudo new errors do not exceed repairs: PASS (`339 <= 951`).
- Complete finite 10k/1k × 5-SNR artifacts: PASS.

Overall decision: **PASS as the scale-up deterministic anchor**.

## Next use and boundary

The next diffusion experiment must use this frozen checkpoint and the same train/validation population. It must not use the 1,000-image validation split for loss-weight, step, or threshold scanning. A separate calibration split must be introduced before direct semantic-risk controller tuning.

AlexNet results remain pseudo-semantic diagnostics. Imagenette policy-dev and official validation were not accessed.

Artifacts:

- `outputs/eval/s13_coco_train2017_c8_scaleup_10k_1k/`
- `outputs/EXP-S13-001/`
- `reports/coco_train2017_scaleup_protocol_preregistration_2026-07-12.md`
- `reports/scaleup_b1_anchor_preregistration_2026-07-12.md`
