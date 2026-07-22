# Received-Latent Posterior Correction Pilot Result

Date: 2026-07-13. Analysis ID: `ANALYSIS-PC-001`.

## Verdict

**POSITIVE pilot; all preregistered feasibility and promotion gates pass.** Three frozen, normalized received-latent proximal steps turn the frozen S14 diffusion output into a better candidate on all five SNRs. This authorizes development of a trained posterior/data-consistency diffusion sampler. It is not final semantic-safety evidence and does not promote S14 itself.

## Frozen protocol

- Data: 64 previously unused COCO train2017 images at deterministic SHA-rank positions 11000--11063, evaluated at 5 SNRs (`320` rows).
- Seed: `20260715`; AWGN; `c=8`, CBR `1/6`.
- Frozen components: S13 B1 checkpoint, S14 six-step diffusion checkpoint, formal DeepJSCC checkpoint, local AlexNet and LPIPS weights.
- Correction: start from S14 raw and apply exactly three steps of normalized gradient descent on
  `mean(|E(x)-y|^2) / mean(|y|^2)`, step size `0.001`.
- No parameter, threshold, or checkpoint was selected on this split.

## Results

| SNR (dB) | latent loss delta | posterior - raw PSNR (dB) | posterior - raw LPIPS | raw/post new error |
|---:|---:|---:|---:|---:|
| 1 | -0.034064 | +0.0288 | -0.01229 | 0 / 1 |
| 4 | -0.027089 | +0.0594 | -0.01208 | 2 / 1 |
| 7 | -0.020353 | +0.1107 | -0.01060 | 1 / 2 |
| 13 | -0.012543 | +0.3239 | -0.00772 | 0 / 0 |
| 19 | -0.010329 | +0.5394 | -0.00688 | 2 / 1 |
| Mean / total | **-0.020876** | **+0.2124** | **-0.00991** | **5 / 5** |

Mean normalized latent loss falls from `0.10363` to `0.08275`, a relative reduction of about `20.1%`. Mean PSNR rises from `32.4327` to `32.6451 dB`; mean LPIPS falls from `0.04973` to `0.03982`.

Relative to the B1 anchor, raw/posterior repair counts are `2/17`, while new-error counts are `5/5`. Directly comparing posterior to raw yields 19 incorrect-to-correct and 4 correct-to-incorrect AlexNet flips. These are pseudo-label diagnostics, not supervised semantic truth.

## Gate audit

- Primary consistency gate: **PASS**, latent loss decreases at 5/5 SNRs.
- Primary quality floor: **PASS**, mean PSNR delta is `+0.2124 dB` versus the allowed `-0.05 dB` floor.
- Promotion perception gate: **PASS**, mean LPIPS delta is `-0.00991` and improves at 5/5 SNRs.
- Promotion pseudo-risk gate: **PASS**, posterior new errors do not exceed raw (`5 <= 5`).

## Interpretation and boundary

The S14 failure was not evidence that diffusion has no useful role. Its unconstrained residual-shift endpoint was off the received-signal posterior. A very small, receiver-observable data-consistency correction recovers both distortion and perception, with larger PSNR gains at high SNR. That SNR trend is consistent with the received latent becoming a more informative constraint, but it remains an inference from this pilot.

The result is still a 64-image development pilot using one pseudo-label classifier. It does not establish generalization, calibrated semantic reliability, or an end-to-end trained posterior sampler. The next method should incorporate the received-latent consistency term inside diffusion sampling/training, then be evaluated on a new frozen split with classifier-ensemble and supervised semantic audits. Do not tune the three-step pilot on this split.

## Reproduction

```bash
python3 scripts/pc_posterior_consistency_pilot.py --config configs/pc001_posterior_consistency_pilot.yaml --device cuda:0 --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy python3 scripts/pc_posterior_consistency_pilot.py --config configs/pc001_posterior_consistency_pilot.yaml --device cuda:0
```

Outputs: `outputs/analysis/s15_received_latent_posterior_pilot/{per_sample.csv,summary.csv,metrics.json}`.
