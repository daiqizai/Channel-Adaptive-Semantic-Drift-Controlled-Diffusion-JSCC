# Mainline M3 Hybrid Semantic-Controller Result

Date: 2026-07-11 (Asia/Shanghai)

## Integration

The rate-accounted S8 sketch is now integrated into the project mainline rather than treated as a standalone side channel:

```text
M3 = equal-rate DeepJSCC
   + SNR-aware residual restoration
   + transmitted source-semantic checksum
   + receiver-side residual-strength selection
```

Reference `c=8` and proposed `c=6 main + c=2 hybrid structure/semantic` both use total CBR `1/6`. The receiver evaluates PNG-quantized alpha candidates `{0,.25,.5,.75,1}` and selects the one whose fixed AlexNet projection has maximum cosine to the recovered source sketch. No threshold is tuned and the receiver does not see the source image. Scratch ResNet18 `T_cls` remains the independent evaluator.

## Preregistered Imagenette policy-dev audit

The audit covers 1,894 images, five SNRs and 9,470 rows. Primary semantics use 1,697 clean-primary images and SNR `{1,4,7}`. Official Imagenette validation remained sealed.

| Endpoint | M3 result | 95% CI / bound | Gate |
|---|---:|---:|---|
| failure minus reference `c=8` | `-0.023964` | `[-0.030642,-0.017482]` | PASS |
| failure improvement over hybrid raw | `+0.000393` raw-minus-M3 | `[-0.001768,+0.002554]` | **FAIL** |
| new-error conservative upper | `0.015875` | 18/1677 image clusters | **FAIL** (`<=0.005`) |
| PSNR minus reference | `+1.4234 dB` | `[+1.3693,+1.4799]` | PASS |
| LPIPS minus reference | `-0.0265` | `[-0.0276,-0.0255]` | PASS |

Primary-SNR failure rates are reference `3.6142%`, c6 main `4.8321%`, hybrid raw `1.2571%`, and M3 controller `1.2178%`. Relative to hybrid raw, the controller reduces new-error rows/clusters from `41/23` to `29/18`, while repairs fall from `161/109` to `151/105`. It therefore exchanges ten repairs for twelve protected new errors, yielding a small net two-row improvement that is not statistically significant.

M3 retains `1.4234/1.9024=74.8%` of hybrid-raw PSNR gain. Selected alpha counts over all 9,470 rows are `{0:1469, .25:826, .5:1256, .75:1743, 1:4176}`.

## Decision

This is a genuine mainline integration and a meaningful risk-quality tradeoff: source-grounded control removes about 22% of hybrid-raw new-error image clusters while retaining about 75% of its quality gain. However, the preregistered M3 promotion decision is **FAIL** because total failure improvement over raw is inconclusive and the absolute new-error bound remains above 0.5%.

The method should remain the mainline semantic-control ablation/candidate, not the final supervised-safe M3. No more alpha or threshold tuning is allowed on this policy-dev split, and official validation is not unlocked.
