# Posterior Consensus Controller Holdout Audit Preregistration

Date: 2026-07-13. Frozen before non-dry-run. Analysis ID: `ANALYSIS-PC-CTRL-001`.

This remains within phase 5 validation and the existing posterior-consistency study. It is not a new project stage.

## Frozen method and split

- New COCO train2017 SHA-rank positions `11576--11831`, disjoint from S13/S14 and PC-001--003; 256 images × `[1,4,7,13,19] dB`.
- AWGN, `c=8`, CBR `1/6`, seed `20260718`.
- Frozen S13 B1, S14 six-step diffusion, and PC-001 correction: exactly 3 normalized received-latent steps of size `0.001`.
- Receiver controller accepts the posterior only when both AlexNet and ResNet18 posterior top-1 equal their corresponding B1-anchor top-1; otherwise it returns B1.
- MobileNetV3-Small is not used by the controller or for parameter selection. It is the frozen holdout semantic audit.

## Gates

The posterior candidate must retain the existing data-consistency/PSNR/LPIPS replication gates. The controlled final output must have positive mean PSNR delta, non-positive mean LPIPS delta, ensemble-majority new errors no greater than S14 raw, and MobileNetV3-Small holdout new errors no greater than raw.

All gates are required. Passing would establish a receiver-visible consensus-control stage result, not final supervised semantic safety. Failure means classifier-consensus routing is not a sufficient solution and should not be expanded by adding more controller classifiers on the same family of splits.
