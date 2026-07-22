# Posterior Correction with Receiver-Side Failure Handling Preregistration

Date: 2026-07-13. Frozen before non-dry-run. Analysis ID: `ANALYSIS-PC-003`.

PC-002 independently replicated the distortion/perception/data-consistency benefit of PC-001, but failed the preregistered ensemble semantic gate. PC-003 therefore combines the unchanged PC-001 posterior correction with the project's already-established receiver-side top-1 agreement fallback. It does not tune correction strength or a new threshold on PC-002.

## Frozen protocol

- New COCO train2017 SHA-rank positions `11320--11575`, disjoint from S13/S14 and PC-001/002; 256 images × five SNRs.
- AWGN, `[1,4,7,13,19] dB`, `c=8`, CBR `1/6`, seed `20260717`.
- Frozen S13 B1, S14 diffusion, and PC-001 correction: 3 normalized steps of size `0.001`.
- Receiver-side controller: accept posterior iff its frozen AlexNet top-1 equals the B1 anchor top-1; otherwise output B1 anchor. The original image and the two audit classifiers are not available to the controller.
- Offline audit classifiers: frozen AlexNet, ResNet18, MobileNetV3-Small with local weights.

## Gates

The unchanged posterior candidate must again decrease latent loss at all SNRs, improve mean PSNR with at least 4/5 positive SNRs, and improve mean LPIPS with at least 4/5 non-positive SNRs.

The controlled final output must additionally:

1. have positive mean PSNR delta and non-positive mean LPIPS delta versus uncontrolled S14 raw;
2. have ensemble-majority new-error count no greater than S14 raw;
3. have new-error count no greater than raw for at least 2/3 audit classifiers.

All gates are required. Passing is a stage result for a posterior-consistent, receiver-controlled diffusion refinement; it remains pseudo-label evidence rather than final supervised semantic safety.
