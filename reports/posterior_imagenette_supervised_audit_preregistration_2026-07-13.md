# Posterior Imagenette Supervised Policy-Dev Audit Preregistration

Date: 2026-07-13. Frozen before non-dry-run. Analysis ID: `ANALYSIS-PC-SUP-001`. Phase remains S5 validation.

## Frozen protocol

- Use only the existing Imagenette `policy_dev` split from `outputs/analysis/imagenette_scratch_classifiers/split_manifest.json` (`1894` images). Official Imagenette validation remains sealed and must not be accessed.
- Ground truth is the Imagenette WNID. Primary evaluator is the existing scratch-trained, calibrated ResNet18 `T_cls`, which used only `cls_train` for training and `cls_cal` for selection/calibration.
- Clean-correct requires `T_cls(original)=WNID` and calibrated confidence at least `0.50`.
- Evaluate all `[1,4,7,13,19] dB`, AWGN, `c=8`, CBR `1/6`, seed `20260721`.
- Freeze S13 B1, S14 diffusion, PC-001 correction (3 normalized steps, size `0.001`), and PC-CTRL AlexNet+ResNet18 consensus fallback. The controller does not use WNID or scratch `T_cls`.

## Gates

1. At least 1600 unique policy-dev images enter the clean-correct subset.
2. Existing received-latent, posterior PSNR/LPIPS, and controlled-final PSNR/LPIPS gates pass.
3. Across primary SNRs `[1,4,7]`, controlled-final supervised new errors do not exceed S14 raw in total or at any individual SNR.
4. Across primary SNRs, controlled-final supervised failures do not exceed S14 raw in total.

All gates are required. Passing is sufficient to retain the controlled posterior method as a supervised policy-dev candidate, but not to unlock official validation automatically. Failing keeps official validation sealed and ends classifier-consensus controller development.
