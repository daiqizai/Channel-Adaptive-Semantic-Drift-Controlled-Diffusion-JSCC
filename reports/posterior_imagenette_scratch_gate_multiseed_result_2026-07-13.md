# Posterior Imagenette Scratch-Gate Multi-Seed Result

Date: 2026-07-13. Analysis ID: `ANALYSIS-PC-RISK-REP-001`. Official Imagenette validation was not accessed.

## Verdict

**NEGATIVE for semantic tail control; strongly positive and channel-seed-stable for posterior restoration.** Eleven of fifteen frozen gates pass. Every consistency and quality gate passes, and supervised failure improves under every new channel seed. Four new-error/tail gates fail, so the method cannot enter an official-validation final lock.

## Protocol integrity

- Frozen policy-dev population: `1894` images; clean-correct: `1697`.
- New AWGN seeds: `20260722`, `20260723`, `20260724`.
- Five SNRs per seed: `28,410` unique seed/SNR/image rows; no missing or duplicate keys.
- Clean membership is identical across seeds.
- S13 B1, S14 diffusion, three posterior steps, `0.001` normalized step size, scratch `G_gate`, scratch `T_cls`, and all thresholds were unchanged.
- Official validation access remained `false`.

## Main results

Across primary `[1,4,7] dB` rows:

| Arm | Failure rows | New-error rows | New-error image clusters |
|---|---:|---:|---:|
| S14 raw | 196 | 13 | 10 |
| posterior | 164 | 15 | 11 |
| scratch-gated final | **163** | **14** | **11** |

Thus final failure improves `196→163`, but new-error rows increase `13→14` and affected images increase `10→11`. Among `1691` eligible image clusters, final new-error incidence is `11/1691 = 0.6505%`; its one-sided 95% Clopper-Pearson upper bound is `1.0744%`, above the frozen `0.5%` limit. For context, raw is `10/1691 = 0.5914%`, upper bound `1.0010%`.

Quality and measurement consistency remain highly stable:

- mean posterior versus raw: PSNR `+0.26534 dB`, LPIPS `-0.006071`;
- mean scratch-gated final versus raw: PSNR `+0.26334 dB`, LPIPS `-0.005937`;
- received-latent consistency decreases in all `15/15` seed/SNR cells;
- final PSNR is positive and LPIPS nonpositive for every seed.

## Per-SNR aggregate across seeds

| SNR | raw/post/final failure | raw/post/final new | final PSNR vs raw | final LPIPS vs raw |
|---:|---:|---:|---:|---:|
| 1 dB | 103 / 91 / 91 | 8 / 10 / 10 | +0.02748 dB | -0.007007 |
| 4 dB | 59 / 49 / 48 | 4 / 4 / 3 | +0.06448 dB | -0.007045 |
| 7 dB | 34 / 24 / 24 | 1 / 1 / 1 | +0.12656 dB | -0.005702 |
| 13 dB | 12 / 7 / 7 | 0 / 1 / 1 | +0.39975 dB | -0.004757 |
| 19 dB | 6 / 5 / 5 | 0 / 0 / 0 | +0.69844 dB | -0.005174 |

## Per-seed primary result

| Channel seed | raw/post/final failure | raw/post/final new | Gate status versus raw new |
|---:|---:|---:|---|
| 20260722 | 63 / 57 / 56 | 5 / 8 / 7 | fail |
| 20260723 | 67 / 53 / 53 | 6 / 5 / 5 | pass |
| 20260724 | 66 / 54 / 54 | 2 / 2 / 2 | pass |

The scratch gate rejected only `172/25,455` clean rows across all five SNRs (`99.32%` acceptance). It prevented one posterior new-error row and missed no posterior repair, but accepted the other tail events because its anchor and posterior top-1 predictions agreed.

Three final-risk image clusters were absent from raw risk, while two raw-risk clusters disappeared in final, producing the net `+1` cluster. `n03425413/n03425413_3069.JPEG`, the 7 dB event from PC-RISK-001, appears again under seed `20260722` at both 1 and 7 dB. This is evidence of reproducible image susceptibility interacting with channel noise, not a single replaceable bad seed.

## Gate audit

Passed:

- clean coverage;
- all aggregated and all seed/SNR consistency gates;
- posterior PSNR/LPIPS gates;
- final PSNR/LPIPS overall and per seed;
- primary final failure nonincrease overall and per seed.

Failed:

- primary final new-error total nonincrease (`14 > 13`);
- per-SNR nonincrease (1 dB `10 > 8`);
- per-seed nonincrease (seed `20260722`, `7 > 5`);
- image-cluster upper bound (`1.0744% > 0.5%`).

## Research decision

The diffusion-side conclusion is now robust: received-latent posterior correction repeatedly improves distortion, perception, and aggregate supervised failure across channel realizations. The frozen scratch top-1 agreement gate is not a sufficient semantic-risk controller. Replacing the seed, averaging channel rows, or citing net repairs cannot erase the image-cluster tail failure.

Official validation remains sealed. No additional threshold or exception may be selected from these exposed policy-dev outcomes. A next controller must be trained/calibrated on explicitly designated development supervision using receiver-visible continuous risk features, then frozen before evaluation on an unused image population. These multi-seed outputs may be used only as declared controller-development data; they cannot serve as that controller's final audit.

Preregistration: `reports/posterior_imagenette_scratch_gate_multiseed_preregistration_2026-07-13.md`.

Artifacts:

- `outputs/analysis/pc_imagenette_scratch_gate_policy_dev_multiseed/metrics.json` — SHA-256 `933ba33af073b178b888b8244004d4c8d9b9d35bf0d695cd9108b1c6621d44d4`
- `outputs/analysis/pc_imagenette_scratch_gate_policy_dev_multiseed/per_sample.csv` — SHA-256 `45c5167e3426d4efe13b4f443edf42429e75d4d0ef7b28390dba81c2abb07da9`
- `outputs/analysis/pc_imagenette_scratch_gate_policy_dev_multiseed/seed_summary.csv` — SHA-256 `17a36fde894e6fb0f635a904f6094429d1faf0e9d3afc6a644970e339f098aa0`
