# Posterior Imagenette Scratch-Gate Follow-Up Result

Date: 2026-07-13. Analysis ID: `ANALYSIS-PC-RISK-001`. Official Imagenette validation was not accessed.

## Verdict

**NEGATIVE under the complete preregistered gate, with a meaningful controller improvement.** The task-matched scratch gate improves aggregate supervised reliability, rejects one 1 dB new error, and retains nearly all posterior quality. It still accepts the single 7 dB new error, so the frozen per-SNR no-increase requirement fails.

## Frozen comparison

The only change from `ANALYSIS-PC-SUP-001` is the controller: the ImageNet AlexNet+ResNet18 consensus is replaced by the existing scratch MobileNetV3-Small `G_gate`. The independent scratch ResNet18 `T_cls` remains outcome-only. WNID, original-image predictions, and `T_cls` predictions do not enter the controller.

All `9470` new rows were checked against PC-SUP on sample/SNR identity, clean membership, anchor/raw/posterior correctness, received-latent losses, PSNR, and LPIPS. There were zero mismatches. Thus the raw and posterior candidates are exactly paired and only the final accept/fallback choice changed.

## Results

- Policy-dev images: `1894`; clean-correct images: `1697`; official validation accessed: `false`.
- Posterior versus S14 raw, unchanged from PC-SUP: mean PSNR `+0.26544 dB`, LPIPS `-0.006102`.
- Scratch-gated final versus raw: mean PSNR `+0.26394 dB`, LPIPS `-0.005966`.
- Across all five SNRs, `G_gate` accepted `8428/8485` clean rows (`99.33%`).
- Primary `[1,4,7] dB` failure counts:
  - raw: `69`
  - posterior: `56`
  - scratch-gated final: `57`
  - previous ImageNet-consensus final: `62`
- Primary new-error counts:
  - raw: `4`
  - posterior: `4`
  - scratch-gated final: `3`
  - previous ImageNet-consensus final: `4`

Per primary SNR:

| SNR | accept | raw/post/final failure | raw/post/final new | final PSNR vs raw | final LPIPS vs raw |
|---:|---:|---:|---:|---:|---:|
| 1 dB | 98.82% | 34 / 31 / 31 | 3 / 3 / 2 | +0.02782 dB | -0.006973 |
| 4 dB | 99.29% | 23 / 15 / 16 | 1 / 0 / 0 | +0.06479 dB | -0.007164 |
| 7 dB | 99.23% | 12 / 10 / 10 | 0 / 1 / 1 | +0.12610 dB | -0.005730 |

The failing 7 dB row is `n03425413/n03425413_3069.JPEG`. `G_gate` assigns both anchor and posterior to its class index `2`, so the exact frozen agreement rule accepts the posterior even though independent `T_cls` changes from correct to incorrect. No threshold or exception was added after observing this event.

## Gate audit

Eight of nine implemented checks pass: clean coverage, five-SNR latent decrease, posterior quality/perception, final quality/perception, total primary new-error nonincrease, and primary failure nonincrease. The only failed check is `primary_final_new_each_snr_not_increase`, because 7 dB is `1 > 0`.

## Interpretation and next decision

This is stronger evidence for keeping posterior-consistent diffusion: compared with S14 raw, the scratch-gated method reduces primary supervised failures `69→57` and new errors `4→3` while retaining nearly the full posterior quality gain. It also dominates the prior ImageNet-consensus controller on this development population. It is not a safety result: policy-dev was already unsealed, one strict tail event remains, and `G_gate`/`T_cls` agreement is imperfect.

No more top-1 or scalar-threshold scanning is authorized on this policy-dev set. The frozen restoration candidate is now sufficiently stable for a genuinely independent semantic-risk protocol, where controller development/calibration and final audit populations are separated before outcomes are viewed. Official validation remains sealed until that protocol is explicitly frozen; this follow-up does not unlock it automatically.

Preregistration: `reports/posterior_imagenette_scratch_gate_preregistration_2026-07-13.md`.

Artifacts:

- `outputs/analysis/pc_imagenette_scratch_gate_policy_dev/metrics.json` — SHA-256 `d134d5133a48e44836794831d86762c92a49f2b8caa708e01d918f9fac724dd6`
- `outputs/analysis/pc_imagenette_scratch_gate_policy_dev/per_sample.csv` — SHA-256 `8abf4ed4fb8ccc4622aa171ec8f23a974d41fca7731863bb2236d4eeb93fc181`
