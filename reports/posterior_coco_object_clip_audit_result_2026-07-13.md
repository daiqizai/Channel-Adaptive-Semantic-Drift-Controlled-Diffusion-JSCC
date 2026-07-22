# Posterior COCO-Object CLIP Clean-Correct Audit Result

Date: 2026-07-13. Analysis ID: `ANALYSIS-PC-GT-001`. Phase remains S5 validation.

## Verdict

**NEGATIVE under the complete preregistered gate; partial semantic improvement.** Real COCO object annotations plus an independent local OpenCLIP evaluator confirm that posterior correction produces net repairs but still introduces object-level new errors.

## Results

- 512 source images; 405 have a usable dominant object label and 195 enter the frozen CLIP clean-correct subset.
- Posterior versus S14 raw: PSNR `+0.2128 dB`, LPIPS `-0.00982`, latent loss `0.10465 → 0.08343`.
- Controlled final versus raw: PSNR `+0.1871 dB`, LPIPS `-0.00740`, coverage `78.71%`.
- Across 195 clean images × five SNRs:
  - raw failure/new/repair: `36/2/1`
  - posterior failure/new/repair: `32/5/8`
  - controlled final failure/new/repair: `32/4/7`

Final failure improves, but final new errors increase `2 → 4`; the new-error gate fails. At 1/4/7/13/19 dB, final new errors are `1/1/1/0/1` versus raw `1/1/0/0/0`.

## Interpretation

The cross-model warnings are not only ImageNet pseudo-label disagreement: a COCO-object GT-like diagnostic observes the same pattern. Posterior correction has positive net semantic utility but lacks a per-sample no-new-error guarantee. Do not tune PC correction or consensus thresholds on this block.

Preregistration: `reports/posterior_coco_object_clip_audit_preregistration_2026-07-13.md`. Outputs: `outputs/analysis/pc_coco_object_clip_audit/`.
