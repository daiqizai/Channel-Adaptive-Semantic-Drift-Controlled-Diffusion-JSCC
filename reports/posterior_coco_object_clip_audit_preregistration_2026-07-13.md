# Posterior COCO-Object CLIP Clean-Correct Audit Preregistration

Date: 2026-07-13. Frozen before non-dry-run. Analysis ID: `ANALYSIS-PC-GT-001`. Phase remains S5 validation.

## Question

Determine whether the ImageNet-classifier holdout failures in PC-CTRL correspond to object-level semantic errors under an independent COCO-label-based diagnostic.

## Frozen protocol

- New train2017 SHA-rank positions `11832--12343`, 512 images × five SNRs; disjoint from S13/S14 and every previous PC block.
- AWGN, `c=8`, CBR `1/6`, seed `20260719`.
- Frozen S13 B1, S14 diffusion, PC-001 three-step posterior correction, and PC-CTRL AlexNet+ResNet18 consensus fallback.
- COCO `instances_train2017.json` supplies the dominant object category. A label is usable only when it is at least 50% of annotated object area and 3% of image area.
- Frozen local OpenCLIP ViT-B/32 classifies among the 80 COCO categories. Clean-correct requires original top-1 equal to the dominant label, probability at least `0.20`, and margin at least `0.02`.
- OpenCLIP and COCO labels are not available to the controller.

## Gates

At least 128 unique images must enter the clean-correct subset. Existing posterior consistency/quality and controlled-final quality gates remain active. On clean-correct rows, controlled-final object-level new errors and final failures must each be no greater than S14 raw.

This is a GT-like auxiliary audit because COCO object labels are real annotations but CLIP is still the frozen evaluator. Passing does not replace a final supervised classifier test; failing establishes that the semantic problem is not merely disagreement among ImageNet classifiers.
