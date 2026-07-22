# Hybrid Semantic-Sketch Refiner Preregistration

Date: 2026-07-11 (Asia/Shanghai)

Status: frozen after `EXPORT-S8-002` passed its payload gates and before training `EXP-S8-001`.

## Frozen method

The refiner warm-starts from `EXP-S7-002`. Its RGB/structure head, six residual blocks, tail, SNR gates and `c=6` main input are unchanged. It receives the R4 decoded structure and 32-D recovered semantic sketch. A two-layer FiLM branch modulates head features; its last layer is zero initialized, making the initial output exactly equal to the loaded S7 refiner for any sketch.

Training uses the same 160 COCO development images and 64-image validation split. It runs 30 epochs, batch size 8, 192 crop, AdamW learning rate `1e-4`, MSE weight 1.0 and L1 weight 0.1. A frozen AlexNet teacher adds temperature-2 KL with weight `2e-4`. Checkpoint selection is validation MSE only; no classifier outcome enters selection.

Because AlexNet enters training, all AlexNet COCO failure/repair values from this experiment are circular diagnostics and may not be used as a semantic success claim.

## Development gates

1. Training must remain finite and preserve the exact total CBR `c=6+c=2=c=8`.
2. Matched raw must retain positive PSNR versus reference `c=8` at every validation SNR.
3. Mean validation PSNR may not fall more than `0.10 dB` below frozen S7 matched raw.
4. Any semantic promotion requires a new frozen Imagenette policy-dev audit with scratch `T_cls`; AlexNet cannot satisfy it.

No official Imagenette validation access is authorized by this run.
