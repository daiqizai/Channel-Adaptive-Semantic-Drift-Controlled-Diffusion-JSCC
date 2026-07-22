# Auxiliary Scratch Semantic-Risk Feature Model Preregistration

Date: 2026-07-14. Frozen before non-dry-run. Analysis ID: `TRAIN-PC-AUX-001`. Phase remains S5 validation.

## Purpose

The multi-channel-seed audit showed that received-latent posterior correction is stable, while a single scratch top-1 agreement gate does not control tail risk. This work package trains one additional task-matched semantic feature model for a later receiver-visible continuous risk controller. It does not change the diffusion, posterior correction, current controller, or semantic outcome evaluator.

## Frozen separation contract

- Role: `G_aux`, auxiliary receiver-side feature model only.
- Architecture: torchvision EfficientNet-B0, `weights=None`, random initialization.
- Seed: `1618033`.
- Classes: the same ten Imagenette WNIDs and exact class order in the frozen split manifest.
- Parameter fitting: existing `cls_train` only.
- Checkpoint selection and scalar temperature calibration: existing `cls_cal` only.
- Policy-dev is not loaded or evaluated during training; only its manifest hash may be present.
- Official Imagenette validation is not scanned, loaded, evaluated, or used for any decision.
- `T_cls` remains the independent ResNet18 outcome evaluator and is prohibited from entering `G_aux` training or future inference features.

The existing `G_gate` and `T_cls` checkpoints and directories are immutable. `G_aux` uses a new output directory and cannot overwrite them.

## Frozen training recipe

Reuse the already established scratch-classifier recipe without tuning: 80 epochs, SGD, learning rate `0.10`, momentum `0.90`, weight decay `1e-4`, cosine schedule with 5 warmup epochs, label smoothing `0.10`, batch size `128`, the same random crop/RandAugment/erasing pipeline, and no test-time augmentation. Best checkpoint selection uses only `cls_cal` macro top-1; temperature scaling uses the same calibration split.

## Promotion gate

`G_aux` must reach `>=0.85` macro top-1 on `cls_cal`, record `weights=None`, `pretrained=false`, `random_initialization=true`, `policy_dev_used_for_training_selection_or_calibration=false`, and `official_val_accessed=false`. All are required.

A pass only freezes `G_aux` as an eligible receiver-visible feature extractor. It does not promote a controller, change any semantic result, or unlock official validation. A failure is recorded without changing the architecture, seed, epoch budget, or quality threshold in this work package.
