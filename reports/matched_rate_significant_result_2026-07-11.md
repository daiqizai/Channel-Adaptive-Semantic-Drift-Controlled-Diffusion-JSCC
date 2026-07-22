# Matched-Total-Rate Main + Structure Significant Result

Date: 2026-07-11 (Asia/Shanghai)

## Outcome

The project now has a deployable-rate counterpart to the earlier perfect source-edge oracle. The reference and proposed systems use exactly the same DeepJSCC channel-use convention:

```text
reference: c=8 RGB                         total CBR = 8/48 = 1/6
proposed:  c=6 RGB + c=2 structure         total CBR = 6/48 + 2/48 = 1/6
```

The `c=2` arm transmits sender-visible Sobel magnitude and absolute Laplacian through an ordinary independently noisy DeepJSCC model. The receiver refiner consumes only the decoded `c=6` RGB reconstruction, decoded `c=2` structure and SNR. It never receives perfect source edges.

## Frozen pilot components

- Main `c=6`: best epoch 7, validation PSNR `30.7497 dB`, checkpoint SHA-256 `40f36f131b83ec1b3154402525904972d023b4211cb2e53ccb4b8d4e80385b6d`.
- Structure `c=2`: best epoch 11, validation PSNR `30.4991 dB`, checkpoint SHA-256 `4ff825130987d6faa201fb25dcbfc4976fda2aa0e5fba0da56ef3816e3e4734e`.
- Decoded-structure residual refiner: `outputs/EXP-S7-002/checkpoints/best.pt`; trained on the existing COCO validation-derived development split only.
- Reference: stable epoch-73 `c=8` DeepJSCC `best.pt`; the NaN `latest.pt` remains forbidden.

These are 20k-image warm-start pilots, not full-data final checkpoints.

## COCO cross-split equal-rate result

The frozen matched-raw system was compared with reference `c=8` on validation, held-out, test-like and fresh-holdout segments. Every one of the 20 split-by-SNR point estimates was positive.

| Split | Images | Matched raw − reference PSNR | 95% paired image-cluster CI |
|---|---:|---:|---:|
| validation | 64 | `+0.3974 dB` | `[+0.2942,+0.5104]` |
| held-out | 32 | `+0.3261 dB` | `[+0.2207,+0.4190]` |
| test-like | 64 | `+0.4198 dB` | `[+0.3508,+0.4904]` |
| fresh-holdout | 64 | `+0.3600 dB` | `[+0.2686,+0.4461]` |

On the three frozen downstream splits combined, the gain is `+0.3772 dB`, 95% CI `[+0.3274,+0.4253]`, with 160 image clusters. Auxiliary AlexNet pseudo failure changes by `-0.0875`, CI `[-0.1437,-0.0325]`.

## Independent supervised Imagenette policy-dev audit

The audit was preregistered before the new system touched any Imagenette image. It used the independently trained scratch ResNet18 `T_cls`, 1,894 policy-dev images, five SNRs and 10,000 image-cluster bootstrap replicates. Official Imagenette validation remained sealed.

| Endpoint | Result | 95% CI / bound | Preregistered gate |
|---|---:|---:|---|
| raw failure − reference | `-0.021410` | `[-0.028678,-0.013946]` | PASS (`upper <= 0`) |
| raw PSNR − reference | `+1.8341 dB` | `[+1.7742,+1.8949]` | PASS |
| raw LPIPS − reference | `-0.0305` | `[-0.0318,-0.0293]` | PASS |
| new-error conservative upper | `0.024764` | 31/1684 eligible image clusters | **FAIL** (`<=0.005`) |

Across primary SNRs `{1,4,7}`, supervised failure falls from `3.3785%` for reference `c=8` to `1.2375%` for matched raw. There are 159 repair rows and 50 new-error rows relative to reference. Therefore the method is a statistically supported net semantic and quality improvement, but it is not a semantics-preserving drop-in replacement under the deliberately strict per-image new-error criterion.

## Diagnostic and next design decision

A receiver-only fallback between the `c=6` main reconstruction and matched raw cannot solve the remaining safety endpoint. Among reference-correct rows, 9 rows (7 image clusters) are wrong in both main and raw; even an outcome-aware oracle choosing between those two images cannot reach the `<=0.5%` image-cluster upper bound. Simple `G_gate` agreement/confidence thresholds also cannot isolate the risk without discarding the restoration gains.

The next method should therefore stop treating the `c=2` packet as only low-level edge data. It should become a rate-accounted semantic descriptor/checksum channel that conditions reconstruction internally, while retaining some fine-structure capacity. Candidate representations must be trained without using `T_cls` and must be selected on COCO or a newly isolated development split before another preregistered Imagenette audit.

Allowed claim: exact-total-rate decoded structure yields robust reconstruction gains and a significant net supervised failure reduction on policy-dev.

Forbidden claim: the current pilot is semantically lossless, passes the preregistered safety gate, or is authorized for official Imagenette validation.
