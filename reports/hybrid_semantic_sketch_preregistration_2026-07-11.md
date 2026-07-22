# Hybrid Structure + Semantic Sketch Preregistration

Date: 2026-07-11 (Asia/Shanghai)

Status: frozen before exporting or evaluating the hybrid payload. The existing S7 matched-rate and Imagenette results were inspected before this protocol. Official Imagenette validation remains sealed.

## Motivation

S7 proved that `c=6` main plus `c=2` decoded structure improves equal-rate quality and net supervised failure, but missed the strict new-error endpoint. Receiver-only main/raw selection cannot repair all events. This follow-up moves coarse semantic information inside the rate-accounted `c=2` path instead of adding another post-hoc threshold.

## Frozen rate and payload

The total-rate contract remains:

```text
reference: c=8 RGB                                  CBR 8/48 = 1/6
hybrid:    c=6 RGB + c=2 structure/semantic         CBR 6/48 + 2/48 = 1/6
```

At 256x256, the `c=2` encoder produces 16,384 real latent symbols. A 32-dimensional semantic sketch is repeated 16 times and overwrites 512 deterministic evenly spread real positions, exactly 3.125% of the existing `c=2` latent. No channel use is added. After the shared AWGN channel, the receiver averages repetitions, normalizes the sketch, erases the 512 reserved positions, and sends the remaining noisy latent to the frozen structure decoder.

## Semantic source and independence

The sender description is a fixed Rademacher projection of frozen torchvision AlexNet ImageNet-1K probabilities. AlexNet is not the final semantic evaluator. It may be used as a training teacher, but AlexNet pseudo-label metrics are circular and cannot establish safety. The primary supervised endpoint remains the separately trained scratch ResNet18 `T_cls`; it never enters encoding, refiner training, model selection or receiver inference.

## Stage gates

Before refiner training:

1. Mean source/recovered sketch cosine must be at least 0.95 at every SNR in `{1,4,7,13,19}`.
2. The decoded structure first-two-channel MSE may increase by at most 10% relative to the frozen S7 `c=2` export at every SNR.
3. Exact payload accounting and `c=6+c=2=c=8` must be machine-validated.

The refiner will warm-start from frozen `EXP-S7-002`, inject the received 32-D sketch through zero-initialized feature modulation, and use a small frozen-AlexNet distillation loss in addition to the existing reconstruction loss. Hyperparameters must be written before its formal run. COCO selection may use reconstruction quality and sketch recovery, not AlexNet semantic improvement.

Before any new Imagenette access, a separate supervised audit addendum must freeze the exact checkpoint and success gates. The previous policy-dev rows may not be used to tune semantic thresholds because the proposed method has no such receiver threshold.

Passing COCO gates is development evidence only. Passing a later policy-dev audit would still not automatically authorize official-val access.
