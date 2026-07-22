# Received-Latent Posterior Correction Independent Replication Preregistration

Date: 2026-07-13. Frozen before non-dry-run. Analysis ID: `ANALYSIS-PC-002`.

## Purpose

Test whether the positive PC-001 endpoint posterior correction replicates on a larger, completely unused development block and remains plausible under three independent frozen ImageNet classifiers. This is a frozen replication, not a parameter search and not final COCO semantic supervision.

## Frozen data and method

- COCO train2017 images at deterministic SHA-rank positions `11064--11319`, immediately after but disjoint from S13/S14 (`0--10999`) and PC-001 (`11000--11063`).
- 256 images at each of `[1,4,7,13,19] dB`, AWGN, `c=8`, CBR `1/6`; seed `20260716`.
- Frozen S13 B1 anchor and frozen S14 six-step diffusion candidate.
- Starting at S14 raw, exactly 3 normalized received-latent gradient steps with step size `0.001`, identical to PC-001.
- No step, threshold, SNR schedule, model, or checkpoint may be selected on this block.

## Evaluation

- Distortion/perception: per-image PSNR and LPIPS.
- Data consistency: per-image normalized `||E(x)-y||^2 / ||y||^2`.
- Offline semantic robustness probes: AlexNet, ResNet18, MobileNetV3-Small, all using local frozen ImageNet weights.
- For every classifier, a new error means B1 agrees with that classifier's original-image pseudo label and the candidate does not. Repair is the reverse transition.
- Ensemble-majority new error/repair means at least two of three classifiers mark the corresponding event for the same image/SNR row.

## Frozen gates

Replication gate:

1. Mean received-latent loss decreases at all 5 SNRs.
2. Mean posterior-minus-raw PSNR is positive, and PSNR improves at least 4/5 SNRs.
3. Mean posterior-minus-raw LPIPS is non-positive, and LPIPS improves at least 4/5 SNRs.

Semantic robustness gate:

4. Posterior ensemble-majority new-error count does not exceed raw.
5. At least 2/3 individual classifiers have posterior new-error count no greater than raw.

All five gates are required for a positive independent replication. Passing authorizes implementation/training of an interleaved posterior-consistent diffusion sampler on a separate training split. It does not establish final semantic safety.
