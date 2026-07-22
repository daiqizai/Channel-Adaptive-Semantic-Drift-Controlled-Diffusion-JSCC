# Imagenette Hybrid Semantic-Controller Preregistration

Date: 2026-07-11 (Asia/Shanghai)

Status: frozen after S7/S8 results and causal ablations were inspected, but before the hybrid semantic controller was run on any Imagenette image. Official Imagenette validation remains sealed.

## Mainline role

This is not a new side task. It instantiates the project mainline `M3-Ours` as:

```text
equal-rate DeepJSCC + SNR-aware residual restoration
+ rate-accounted source semantic checksum
+ receiver-side semantic consistency control
```

Reference `c=8` and proposed `c=6 main + c=2 hybrid structure/semantic` both use total CBR `1/6`. The 32-D semantic sketch occupies 128 existing real positions inside `c=2`; it adds no rate and traverses the same AWGN path.

## Frozen controller

The frozen `EXP-S8-003` refiner produces hybrid raw. The receiver constructs alpha candidates `{0,0.25,0.5,0.75,1}` between main and raw. Each candidate is PNG-quantized, passed through frozen AlexNet, projected by the same fixed Rademacher matrix, and scored against the recovered source sketch. Maximum cosine wins; exact ties select smaller alpha. There is no learned or selected threshold and the receiver never sees the original image.

AlexNet is part of the method and cannot evaluate it. Primary semantics use the independent scratch ResNet18 `T_cls`; clean-primary membership and primary SNR `{1,4,7}` remain unchanged.

## Primary gates

The internal `matched_raw` field denotes `M3_hybrid_sketch_alpha_controller`; the internal secondary field denotes unattenuated `M2_hybrid_raw`.

1. Controller failure minus reference `c=8`: paired image-cluster CI upper `<=0`.
2. Controller failure minus hybrid raw: paired image-cluster CI upper `<0` (implemented equivalently as hybrid-raw minus controller CI lower `>0`).
3. Among reference-correct rows, controller new-error conservative upper `<=0.005`.
4. Controller PSNR minus reference CI lower `>0`, with positive point estimate at every SNR.
5. Controller retains at least 50% of hybrid-raw mean PSNR gain over reference.
6. Controller LPIPS minus reference is negative.

All gates must pass. Failure is retained and does not authorize another threshold search on policy-dev. Passing is development evidence only and does not automatically unlock official val.
