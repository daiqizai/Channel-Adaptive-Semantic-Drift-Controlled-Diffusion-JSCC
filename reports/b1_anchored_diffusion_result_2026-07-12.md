# B1-Anchored Semantic-Preserving Diffusion Result

Date: 2026-07-12

## Decision

`EXP-S12-001` is **negative under the preregistered promotion rule** because its raw diffusion candidate introduces more pseudo new errors than repairs (`8 > 4`).

The quality/perception side is nevertheless materially stronger than `EXP-S10-001`: the B1-anchored bridge improves LPIPS at every SNR while limiting mean PSNR loss to `0.0775 dB`. This establishes a useful architecture boundary but not a promotable semantic-risk-controlled method.

## Fixed design

- Formal `c=8` DeepJSCC communication path, CBR `1/6`.
- Frozen B1 `EXP-S11-001` receiver-structure residual CNN as deterministic anchor.
- Receiver-only Sobel/Laplacian condition recomputed from B1; no `c=2` side path.
- Pixel-domain residual-shift bridge with 20 train timesteps and 6 deterministic sampling steps.
- Same seed, split, denoiser capacity, bridge noise, steps, and correction gates as S10.
- Reconstruction-dominant loss plus receiver-structure L1 and local frozen ResNet18 target-distillation KL.
- Frozen AlexNet used only for final pseudo-semantic diagnostics.

## Formal result

| SNR | Raw ΔPSNR vs B1 | Raw ΔLPIPS vs B1 | New error | Repair |
|---:|---:|---:|---:|---:|
| 1 | `-0.0951` | `-0.000834` | 1 | 0 |
| 4 | `-0.0796` | `-0.000837` | 2 | 1 |
| 7 | `-0.0689` | `-0.000769` | 1 | 2 |
| 13 | `-0.0714` | `-0.000543` | 1 | 0 |
| 19 | `-0.0722` | `-0.000279` | 3 | 1 |
| mean / total | **`-0.0775`** | **`-0.000652`** | **8** | **4** |

Top-1 fallback retains mean `ΔLPIPS=-0.000613` but mean `ΔPSNR=-0.0747 dB`; it returns to B1 whenever AlexNet top-1 changes, so it is a diagnostic guard rather than independent safety evidence.

Mean diffusion sampling latency is `14.97 ms/image`, about 6× the B1 residual anchor's `2.50 ms/image`. The diffusion model has the same 64×6 denoiser scale as S10 and uses 6 steps.

Best checkpoint is epoch 2. Later training loss continues decreasing while eval PSNR degrades by up to more than 1 dB, demonstrating strong small-data overfitting. No early-stop rule or loss weight was changed after observing this.

## Comparison with S10

| Metric | S10 decoded-structure anchor | S12 B1 anchor + preservation | Change |
|---|---:|---:|---:|
| Mean raw ΔPSNR | `-0.1548 dB` | `-0.0775 dB` | `+0.0773 dB` |
| Mean raw ΔLPIPS | `-0.000195` | `-0.000652` | stronger improvement |
| LPIPS-improved SNRs | `5/5` | `5/5` | unchanged |
| New error / repair | `12/7` | `8/4` | both fail risk gate |

The event counts are not a direct matched-arm causal comparison because S10 and S12 use different anchors. They show that neither exact bridge passed its own incremental-risk endpoint.

## Consequence

The experiment supports continuing diffusion only at a different scale/protocol:

1. B1 remains the deterministic baseline and anchor.
2. Stop further 160-image tuning of this residual-shift bridge, as preregistered.
3. If diffusion is continued, build a proper COCO train2017-scale anchor/diffusion training set with independent validation, then freeze once.
4. Semantic-risk control must use independent calibration/evaluation and directly optimize/select new-error risk; a tiny teacher KL is insufficient.
5. Do not unlock Imagenette official validation from this result.

Artifacts:

- `outputs/EXP-S12-001/`
- `reports/b1_anchored_semantic_preserving_diffusion_preregistration_2026-07-12.md`
- `configs/s12_b1_anchored_semantic_preserving_diffusion.yaml`
