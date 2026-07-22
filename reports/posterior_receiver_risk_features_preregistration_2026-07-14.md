# Posterior Receiver-Risk Feature Extraction Preregistration

Date: 2026-07-14. Frozen before non-dry-run. Analysis ID: `ANALYSIS-PC-RISK-FEAT-001`. Phase remains S5 validation.

## Purpose and evidence boundary

This work package converts the already exposed `ANALYSIS-PC-RISK-REP-001` policy-dev rows into a task-matched, receiver-visible continuous feature table for later controller development. It does not evaluate a new controller and is not a held-out result. The three channel seeds `[20260722,20260723,20260724]` and all policy-dev outcomes have already been inspected, so no result from this package may be described as image-population or channel-seed generalization.

Official Imagenette validation remains sealed. The loader must reject any split manifest or classifier checkpoint recording official-validation access.

## Frozen generation path

- Images: all `1894` existing `policy_dev` images; no `cls_train`, `cls_cal`, or official-val image is substituted.
- AWGN: seeds `[20260722,20260723,20260724]`, SNR `[1,4,7,13,19]` dB, `c=8`, CBR `1/6`.
- Candidate: frozen S13 B1 anchor, frozen S14 six-step residual-shift diffusion, then exactly three received-latent correction steps at normalized step size `0.001`.
- `G_gate`: frozen scratch MobileNetV3-Small checkpoint selected/calibrated only on `cls_cal`.
- `G_aux`: frozen scratch EfficientNet-B0 checkpoint from `TRAIN-PC-AUX-001`, selected/calibrated only on `cls_cal` and never evaluated on policy-dev before this package.
- `T_cls`: frozen scratch ResNet18 outcome evaluator. It may create development targets and clean-correct labels but must never enter a receiver feature or deployment decision.
- Existing scratch top-1 fallback is reproduced only as a deterministic regression check; its already known negative verdict is not reinterpreted as the result of this package.

## Receiver feature contract

`receiver_risk_v1` contains only quantities available after decoding at the receiver:

1. SNR and received-latent consistency before/after posterior correction;
2. pixel L1/RMSE distances among anchor, raw diffusion, and corrected posterior;
3. for both `G_gate` and `G_aux`, calibrated confidence, normalized entropy, top-1 margin, Jensen-Shannon change, top-class probability retention, and top-1-change indicators across anchor/raw/posterior;
4. cross-model top-1 agreement on anchor/raw/posterior.

The executable schema is the ordered whitelist `RECEIVER_RISK_FEATURE_COLUMNS`. It must contain no `teacher_*`, original-image prediction, ground-truth label, WNID, or class-index field. `sample_id`, WNID, class index, and channel seed are identifiers or audit metadata only, not model inputs. The generated schema JSON must record the SHA-256 of `G_gate`, `G_aux`, and `T_cls`.

Development-only targets are explicitly prefixed `teacher_`: clean correctness, anchor/raw/posterior correctness, posterior new-error, and `T_cls` true-class probability/margin changes. A later controller may be fitted against these targets on policy-dev, but the final fitted estimator must accept only the frozen receiver whitelist.

## Integrity gates for this package

The extraction is valid only if:

- exactly `1894 × 3 × 5 = 28,410` unique `(sample_id, channel_seed, snr_db)` rows are written;
- the receiver feature table and ordinary audit table have identical keys and row counts;
- every receiver feature is finite;
- the scratch-gate audit summary reproduces `ANALYSIS-PC-RISK-REP-001` semantic counts and quality metrics within deterministic numeric tolerance;
- all three scratch checkpoints are distinct, randomly initialized, quality-gate-passed, policy-dev-separated, and official-val-unaccessed;
- the official validation remains unaccessed.

Passing these gates only makes the feature table eligible for controller development. Before any new channel-seed audit, the estimator family, feature subset, fitting rows, threshold rule, audit seed, and success gates must be frozen in a separate preregistration.
