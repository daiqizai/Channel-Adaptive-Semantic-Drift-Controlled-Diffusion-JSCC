# Imagenette Source-Semantic-Description Router Preregistration

Date: 2026-07-10 (Asia/Shanghai)

Status: frozen after inspecting only the aggregate hard-top-1 source-description diagnostic, but before extracting or inspecting any source/M0/candidate probability vectors and before selecting any continuous-score threshold.

## Question

SGD-JSCC conditions diffusion on sender-derived coarse text and fine edge semantics. This diagnostic asks a narrower question inside the current frozen restoration pipeline: can a compact semantic description of the original distinguish an M2 repair from an M2 semantic injury better than receiver-only top-1 agreement?

No communication, refiner, alpha schedule, classifier checkpoint, evaluator, clean threshold, SNR scope or official-validation rule is changed. Official Imagenette validation remains sealed.

## Side information and accounting

The sender applies the frozen scratch-trained `G_gate` to the original image. Two descriptions are recorded:

- learned top-1 class: 4 raw bits for 10 possible classes;
- calibrated 10-way probability vector: each probability is rounded to uint8 and the decoded vector is renormalized, for 80 raw bits.

The diagnostic assumes these bits arrive without error. It does not call the cost negligible, convert bits to analog channel uses, or claim matched CBR. A deployable follow-up requires an explicit modulation/FEC/error model and a matched total channel budget. Dataset ground-truth class descriptions are reported only as a labelled oracle upper bound and cannot be selected as the learned method.

`T_cls` remains the independent evaluator and never enters the transmitted description, receiver score, threshold selection feature, or inference rule. It enters only policy-development outcome labels and final reporting.

## Nested development discipline

The already unsealed `policy_dev` image IDs are split within each WNID by SHA-256 rank using seed `57721`. The first half is `semantic_select`; the remainder is `semantic_audit`. Only `semantic_select` outcomes may select a score family and threshold. The selected rule is then evaluated once on `semantic_audit`. The previous aggregate hard-top-1 diagnostic used all policy-dev images, so it remains exploratory context rather than independent confirmation; continuous probability vectors and their event-level outcomes have not been inspected.

The primary population and SNR scope remain those of the supervised audit:

```text
A = {i: T_cls(original_i) = y_i and calibrated confidence >= 0.50}
primary SNR = {1, 4, 7} dB
```

## Frozen continuous scores

Let `q` be the decoded uint8 source probability vector, and let `p0` and `p2` be receiver-computed calibrated `G_gate` probabilities for M0 and M2. Every score is oriented so that a larger value means the candidate moved farther from the source description:

```text
CE-risk     = CE(q, p2) - CE(q, p0)
JS-risk     = JS(q, p2) - JS(q, p0)
Cos-risk    = cosine_distance(q, p2) - cosine_distance(q, p0)
Top1-risk   = log p0[argmax(q)] - log p2[argmax(q)]
accept      = risk <= frozen_threshold
```

The exact finite threshold grids are in the config. No new family, interaction, learned risk model or threshold may be added after probability outcomes are inspected without being labelled a new exploratory analysis.

## Selection and audit criteria

On `semantic_select`, feasible rules must have zero accepted-new-error image clusters in the primary scope and retain at least 50% of M2's all-SNR PSNR gain. Among feasible rules, selection first minimizes final failure and then maximizes all-SNR PSNR gain. If none is feasible, the frozen fallback order is: minimize accepted-new-error image clusters, minimize final failure, maximize PSNR gain.

The selected rule is successful on `semantic_audit` only if all of the following hold:

- the image-cluster bootstrap 95% upper endpoint for final failure minus M2 failure is below zero;
- the corresponding upper endpoint versus M0 is at most `+0.005`;
- the conservative one-sided accepted-new-error upper bound is at most `0.005`;
- at least 50% of M2 PSNR gain is retained;
- the PSNR gain versus M0 has a positive 95% lower endpoint;
- LPIPS point delta versus M0 is negative.

Intervals use 10,000 paired image-cluster bootstrap replicates. The accepted-new-error endpoint uses the maximum of the clustered one-sided 95th percentile and an image-any-event Clopper-Pearson upper bound, matching the base supervised audit.

Failure is informative: it means source description alone is insufficient with the current semantic grounding model, and the next method must improve description-image grounding or inject source structure into restoration rather than keep tuning receiver-only scalar gates.
