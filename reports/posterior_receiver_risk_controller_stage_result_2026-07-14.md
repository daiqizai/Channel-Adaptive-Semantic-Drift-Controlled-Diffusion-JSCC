# Posterior Receiver-Risk Controller Stage Result

Date: 2026-07-14. Work packages: `TRAIN-PC-AUX-001`, `ANALYSIS-PC-RISK-FEAT-001`, `ANALYSIS-PC-RISK-CTRL-DEV-001`, and `ANALYSIS-PC-RISK-SEED-AUDIT-001`. Official Imagenette validation was not accessed.

## Bottom line

The posterior-consistent diffusion restoration component remains worth keeping, but this receiver-only uncertainty controller is rejected.

Across the previous three development seeds and the new audit seed, corrected posterior consistently improves S14 raw by about `+0.265 dB` PSNR and `-0.0061` LPIPS. On the new seed it also reduces primary supervised failure from raw `50` to `45`. Diffusion therefore has reproducible restoration and net-semantic value.

However, the frozen six-feature controller did not catch either new-seed posterior new-error and rejected 11 posterior repairs. Its final primary failure was `56`, worse than raw `50`, and final new-error remained `2` versus raw `0`. The preregistered new-seed verdict is `NEGATIVE`.

## 1. Independent auxiliary semantic model

`G_aux` is a scratch EfficientNet-B0 distinct from scratch MobileNetV3-Small `G_gate` and scratch ResNet18 `T_cls`.

- training/selection/calibration: `cls_train` / `cls_cal` / `cls_cal` only;
- policy-dev used for training, checkpoint selection, or calibration: false;
- official validation accessed: false;
- best epoch: `64/80`;
- `cls_cal` macro top-1: `0.90269984`, above the frozen `0.85` gate;
- temperature: `0.80484366`;
- ECE15: `0.0739071 → 0.0427925` after temperature scaling;
- checkpoint SHA-256: `8e074be6ec854edbc144d95d9fe5cd7d098c61bca853915108952acfa094b455`.

This pass only qualifies `G_aux` as a receiver feature extractor. It is not a semantic-safety result.

## 2. Receiver feature table

`receiver_risk_v1` contains 43 receiver-visible fields: SNR, received-latent consistency, anchor/raw/posterior pixel distances, calibrated confidence/entropy/margin/JS/retention from `G_gate` and `G_aux`, plus cross-model agreement. `T_cls` outcomes are stored separately under `teacher_*` names and are prohibited controller inputs.

Integrity checks passed:

- `28,410 = 1894 × 3 × 5` unique `(sample_id, channel_seed, snr)` rows;
- all 43 receiver features finite;
- feature and audit key sets exactly equal;
- all ordinary audit fields reproduce `ANALYSIS-PC-RISK-REP-001` within `1e-9`;
- schema records three distinct checkpoint hashes and `official_val_accessed=false`.

The repeated baseline result is intentionally unchanged: raw/posterior/scratch-final primary failure `196/164/163`, new-error rows `13/15/14`, and scratch-final new-error cluster upper bound `1.0744%`. The feature package does not reinterpret that negative baseline.

## 3. Transparent development controller

Exploratory image-grouped models did not justify a high-capacity risk learner: five-fold group-hash OOF ROC-AUC was about `0.65` for balanced logistic regression and `0.71` for a small histogram classifier over only 16 all-SNR new-error rows. The frozen candidate therefore uses a low-free-parameter score:

1. four high-risk empirical percentiles: `G_gate/G_aux` raw-to-posterior and anchor-to-posterior JS divergence;
2. two high-risk empirical percentiles: sign-reversed `G_gate/G_aux` posterior confidence;
3. arithmetic mean of the six percentiles;
4. threshold `0.8537265316368728`, the first frozen rejection-rate candidate satisfying every development gate.

No fitted coefficient, WNID, class index, sample ID, source image, ground truth, or teacher field enters the controller.

### Development-only result

| Metric | Raw | Posterior | Frozen risk final |
|---|---:|---:|---:|
| Primary failure rows | 196 | 164 | 180 |
| Primary new-error rows | 13 | 15 | 3 |
| Primary new-error image clusters | 10 | 11 | 3 |
| Cluster upper 95% | — | — | `0.4579%` |
| Mean PSNR gain vs raw | — | `+0.26534 dB` | `+0.23834 dB` |
| Mean LPIPS change vs raw | — | `-0.006071` | `-0.004799` |

The controller retained `89.83%` of posterior PSNR gain. All frozen development gates passed at a `10.0004%` clean+anchor-correct reference rejection rate. These numbers were used to select the threshold and are not holdout evidence.

## 4. New channel-seed audit

Seed `20260725` was absent from source/config/report ledgers before its config, controller hash, threshold, and pass gates were frozen. It uses the same 1894 policy-dev images, so this is only a channel-randomness audit.

### Restoration and controller result

| Metric | Raw | Posterior | Scratch top-1 final | Frozen risk final |
|---|---:|---:|---:|---:|
| Primary failure rows | 50 | 45 | 46 | 56 |
| Primary new-error rows | 0 | 2 | 2 | 2 |
| Primary new-error clusters | 0 | 2 | 2 | 2 |
| Mean PSNR gain vs raw | — | `+0.26535 dB` | `+0.26369 dB` | `+0.23827 dB` |
| Mean LPIPS change vs raw | — | `-0.006064` | `-0.005941` | `-0.004800` |

The risk controller's clean+anchor-correct rejection rate was `9.8610%`, within the preregistered `[5%,15%]` range. It retained `89.80%` of posterior PSNR gain, and quality was positive at all five SNRs. It nevertheless failed the total/per-SNR new-error and failure gates:

- both posterior new-errors occurred at 1 dB and both were accepted;
- 11 anchor-wrong/posterior-correct repairs were rejected;
- final failure rose `45→56`, exceeding raw `50`;
- the cluster upper bound alone passed (`2/1692`, upper `0.3716%`), but cannot override the raw-relative failures.

Verdict: `NEGATIVE`. No seed replacement, threshold change, or event exception is allowed.

## 5. Failure mechanism

The two accepted new-errors were:

| Sample | Risk score | Gate posterior conf. | Aux posterior conf. | Interpretation |
|---|---:|---:|---:|---|
| `n02979186/n02979186_8089.JPEG` | `0.83996` | `0.8798` | `0.9576` | just below threshold, no prior development event |
| `n03425413/n03425413_24914.JPEG` | `0.54799` | `0.9731` | `0.9627` | high-confidence shared blind spot; same image had a development event |

For the second row, all four JS features are between roughly `3e-7` and `6e-6`. Both receiver models are highly confident and see almost no semantic distribution movement, yet independent `T_cls` crosses the true-class boundary. This is not a small threshold miss. A post-hoc threshold low enough to reject both events would reject about `38.3%` of new-seed reference rows.

## 6. Research decision

1. Keep frozen S13 B1 + S14 diffusion + three-step received-latent posterior correction as the current restoration candidate. The new audit adds a fourth seed with essentially identical quality gains and lower posterior failure than raw.
2. Retire `js_uncertainty_mean_percentile_v1` as a promotion candidate. Preserve its negative audit as evidence that receiver-only ensemble uncertainty cannot certify semantic tail safety.
3. Do not continue threshold sweeps or add more receiver classifiers on exposed policy-dev.
4. The next semantic-control method must bring independent information, preferably a task-related, error-protected, strictly rate-accounted sender checksum/token. It should reuse the existing exact-rate payload machinery but must improve on the current 32-D random sketch, whose received-vs-shuffled causal result was already insufficient.
5. Train/select that representation on a new labeled development/calibration population, with received/zero/shuffled semantic and hard new-error gates. Freeze method and protocol before any image-population final audit. Official Imagenette validation remains sealed.

## Artifacts

- `G_aux` checkpoint: `outputs/analysis/imagenette_scratch_risk_classifier/G_aux/checkpoints/best.pt`, SHA-256 `8e074be6ec854edbc144d95d9fe5cd7d098c61bca853915108952acfa094b455`.
- Multiseed risk table: `outputs/analysis/pc_imagenette_receiver_risk_features_multiseed/risk_features.csv`, SHA-256 `7b81120f8a2a23140800845257d19126800fae1e5f2cb78a4c6398266917233d`.
- Frozen controller JSON: `outputs/analysis/pc_imagenette_receiver_risk_controller_dev/controller.json`, SHA-256 `3ff792a366074202d1727042c40c0cbc777843a5c65d48276e3dfd9be6199f6f`.
- Frozen empirical CDFs: `outputs/analysis/pc_imagenette_receiver_risk_controller_dev/empirical_cdfs.npz`, SHA-256 `2a7f062ab53da9309f6ccad8b9c2a8977b9b79e5d95c760559d674d2503e6956`.
- New-seed feature metrics: `outputs/analysis/pc_imagenette_receiver_risk_seed_20260725_features/metrics.json`, SHA-256 `a46d1d2d94748222238ef0c9a800c53ddcfa5365caf076f0a699e6e46f328593`.
- New-seed audit metrics: `outputs/analysis/pc_imagenette_receiver_risk_seed_20260725_audit/metrics.json`, SHA-256 `cb76e6a38cd5b2d25a3b7d15678b60ca8f89598ea49c227d42470e1ac3bf1f93`.

Verification: 51 unit tests, `py_compile`, all dry-runs, exact row/key/finite checks, input/checkpoint/config SHA validation, and `git diff --check` passed. No network access or download occurred.
