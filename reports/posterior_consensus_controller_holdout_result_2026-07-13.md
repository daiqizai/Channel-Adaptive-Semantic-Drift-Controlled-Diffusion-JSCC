# Posterior Consensus Controller Holdout Audit Result

Date: 2026-07-13. Analysis ID: `ANALYSIS-PC-CTRL-001`. Phase remains S5 validation.

## Verdict

**NEGATIVE under the preregistered holdout gate.** The two-model receiver consensus controller produces an apparently safe controller-ensemble result while failing on the classifier that was kept out of control. This is a decisive no-go for continuing to stack top-1 agreement rules.

## Results

- Frozen posterior candidate again replicates: latent loss `0.10458 → 0.08347`, PSNR `+0.2119 dB`, LPIPS `-0.01061`; all five SNRs improve.
- AlexNet+ResNet18 consensus-controller coverage: `78.05%`.
- Controlled final versus S14 raw: PSNR `+0.1927 dB`, LPIPS `-0.00791`.
- Three-model majority new error: raw `1`, posterior `1`, controlled final `0`.
- Controller-model new errors are zero by construction:
  - AlexNet raw/final: `17 → 0`
  - ResNet18 raw/final: `22 → 0`
- Unused MobileNetV3-Small holdout new errors increase `12 → 34`; repairs increase `12 → 105`.

Per-SNR final PSNR remains positive (`+0.0385/+0.0641/+0.1062/+0.2739/+0.4805 dB`) and LPIPS remains improved. MobileNet new error is not worse only at 19 dB and increases at the other four SNRs.

## Interpretation

Controller-ensemble majority safety is circular when the same models define acceptance and evaluation. The holdout failure shows that agreement routing is model-specific, even though it preserves quality and creates many repairs. No additional classifier-consensus rule should be added on these splits.

The posterior restoration component remains reproducible. The semantic-control branch should move to a separately trained/calibrated risk model with untouched audit labels, or to COCO-object/Imagenette supervised evaluation. Correction steps and size remain frozen.

Preregistration: `reports/posterior_consensus_controller_holdout_preregistration_2026-07-13.md`. Outputs: `outputs/analysis/pc_controller_holdout_audit/`.
