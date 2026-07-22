# Imagenette Supervised Clean-Correct Preregistration

Date: 2026-07-10 (Asia/Shanghai)

Status: preregistered before classifier training, policy-development evaluation, or access to downstream official-validation results.

## Purpose

This experiment closes the main semantic-evaluation gap left by the COCO pseudo-label studies. The COCO-trained DeepJSCC and matched residual refiners remain frozen. Imagenette contributes only an external, single-label semantic audit with ground-truth WNIDs.

Standard ImageNet-pretrained classifiers are not valid primary evidence here: Imagenette was sampled from the ImageNet image pool, so an `IMAGENET1K_V1` model may have trained on exact Imagenette images. The primary gate and evaluator will therefore start from random initialization and use only the controlled Imagenette training split. Any cached ImageNet model result must be labelled as a potentially contaminated sensitivity analysis and must use its full 1000-way output.

## Sealed data protocol

The official `imagenette2-320/train` directory is stratified by WNID and assigned by a deterministic SHA-256 score over seed plus relative path:

- `cls_train`: 70%, classifier parameter fitting only.
- `cls_cal`: 10%, clean macro-top-1 checkpoint selection and scalar temperature calibration only.
- `policy_dev`: 20%, one-seed receiver-policy development; never used for classifier fitting or checkpoint selection.

The official `imagenette2-320/val` directory is a sealed, one-shot final test. It may not be used for early stopping, temperature fitting, threshold selection, gate selection, schedule selection, debugging based on outcomes, or evaluator selection.

The manifest must record relative path, WNID, 10-way label and file SHA-256. Exact content overlap across partitions is forbidden. The official archive must have size `341663724` bytes and MD5 `3df6f0d01a2c9592104656642f5e78a3` before extraction is accepted.

## Frozen semantic models

- `G_gate`: MobileNetV3-Small, `weights=None`, seed `271828`, 10-way output.
- `T_cls`: ResNet18, `weights=None`, seed `314159`, 10-way output; this is the independent primary evaluator.

Both models see only clean `cls_train` images during optimization. Best checkpoints are selected only by `cls_cal` macro top-1, followed by scalar temperature calibration on the same calibration partition. `T_cls` is prohibited from entering the receiver gate, restoration loss, alpha schedule, or any policy selection. No test-time augmentation is used.

Before downstream evaluation, `T_cls` must reach at least `0.85` macro top-1 on `cls_cal`; `G_gate` must reach at least `0.80`. A failure may be addressed only by changing the training recipe while official val remains sealed. Such a change must be recorded before policy-dev is inspected.

## Frozen communication and restoration arms

All communication/restoration weights were trained on COCO and are frozen:

- `M0`: formal DeepJSCC `best.pt` under AWGN.
- `no_edge_scheduled`: EXP-S4-009 capacity-matched no-edge residual refiner under the edge schedule, a structural-conditioning control.
- `M2_edge_scheduled`: EXP-S4-008 edge-conditioned refiner with the already selected monotonic residual-shrink schedule, always accepted.
- `M3_scratch_gate_fallback`: exactly the M2 candidate, accepted only when `G_gate(candidate).top1 == G_gate(M0).top1`; otherwise output M0.

The fixed edge alpha schedule at SNR `[1, 4, 7, 13, 19]` is `[0.75, 0.75, 0.75, 1.00, 0.75]`, giving effective strengths `[0.09, 0.075, 0.06, 0.05, 0.03]`. M0, raw refiner output, shrunken candidate and materialized final are rounded to the 8-bit PNG grid so the streaming experiment matches the distributions used to train and select the existing refiner/schedule.

Policy-dev uses AWGN seed `20260710`. If the preregistered criteria are met, all method and classifier hashes are written into `final_lock`; the official val then runs once with seeds `20260711`, `20260712`, and `20260713`.

## Primary population and endpoints

The primary clean-correct set is fixed before downstream outcomes:

```text
A = {i: T_cls(x_i) = y_i and calibrated_probability(y_i | x_i) >= 0.50}
```

Thresholds `0.0` and `0.7` are sensitivity analyses only. Clean membership is computed once from the same transmitted 256×256 center crop. Final coverage must contain at least 2500 images and at least 150 per WNID; otherwise the evaluator has insufficient coverage and the threshold is not relaxed after the fact.

For every clean-correct image:

```text
M0-Failure        = [T_cls(M0) != y]
Candidate-Failure = [T_cls(candidate) != y]
Final-Failure     = [T_cls(final) != y]

New-Error         = accept and M0 correct and candidate wrong
Repair            = accept and M0 wrong and candidate correct
Protective-Reject = reject and M0 correct and candidate wrong
Missed-Repair     = reject and M0 wrong and candidate correct
```

The jointly primary semantic endpoints use SNR `[1, 4, 7]`:

1. Gate efficacy: `Final-Failure(M3) - Candidate-Failure(M2)`, with the image-cluster bootstrap 95% CI upper bound strictly below zero.
2. Safety versus baseline: `Final-Failure(M3) - M0-Failure`, with the 95% CI upper bound no greater than `+0.005` absolute.
3. Accepted-new-error safety: among clean-correct rows where M0 remains correct, the one-sided 95% upper confidence bound for `accept and candidate wrong` must be no greater than `0.005`. Repairs may not cancel this constraint.

The safety-versus-baseline upper bound also applies separately at each primary SNR, so improvement at one channel condition cannot hide degradation at another. As a classwise guardrail, no WNID may have an M3-minus-M0 failure point estimate above `+0.02`; classwise intervals remain secondary because of their smaller sample sizes.

The jointly required quality conditions are:

- `PSNR(M3)-PSNR(M0)` has a 95% CI lower bound above zero.
- The point estimate is positive at every one of the five SNRs.
- M3 retains at least 50% of the matched M2 PSNR gain.
- LPIPS delta is negative.

Quality is reported over the complete evaluated split, not only the evaluator-selected clean-correct subset. Secondary reporting includes every SNR, classwise macro/worst-class failure, accept rate, all four semantic event counts, PSNR, MS-SSIM, LPIPS and latency.

## Statistics and decision discipline

All intervals use 10,000 paired bootstrap replicates clustered by original image ID. A resampled cluster keeps all its SNR and channel-seed rows. Repeated channel realizations are not treated as independent images. Zero or near-zero event counts receive an exact binomial upper bound in addition to the bootstrap result.

Failure of any primary semantic or quality condition is reported as a negative result; the method is not promoted to supervised-safe M3. If policy-dev fails, policy changes may use only policy-dev and must be frozen before official val. If official val fails, it cannot be renamed validation or reused for tuning.

After official val is unlocked, changing alpha, gate rule, clean threshold, evaluator, checkpoint, channel seeds, or endpoint is forbidden. Only immutable input hashes and the unlock timestamp may be added to the pre-existing lock block.
