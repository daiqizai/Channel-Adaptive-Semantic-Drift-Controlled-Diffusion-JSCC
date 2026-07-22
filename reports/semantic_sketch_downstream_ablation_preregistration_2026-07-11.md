# Semantic-Sketch Frozen Downstream Ablation Preregistration

Date: 2026-07-11 (Asia/Shanghai)

Status: frozen after the 64-image validation causal ablation and before running `EXP-S8-002` on any downstream split.

The refiner checkpoint is fixed. The downstream population is the union of held-out samples 0-31, test-like samples 256-319 and fresh-holdout samples 320-383: 160 images, all outside refiner train 32-191 and validation 192-255. Received, zero and deterministic next-manifest shuffled sketches are evaluated with identical main RGB, hybrid structure, SNR and residual gates.

Primary paired PSNR effects pool five SNR rows within image and use 10,000 image-cluster bootstrap replicates. Success requires:

1. Received-minus-zero 95% CI lower endpoint `>0`.
2. Received-minus-shuffled 95% CI lower endpoint `>0`.
3. Both point estimates are non-negative in every one of the 3 split x 5 SNR cells.

No classifier metric is used. AlexNet entered training and cannot establish semantic safety. Passing would show that the transmitted sketch is causally useful for reconstruction across frozen splits; it would only authorize freezing a separate scratch-`T_cls` policy-dev audit, not official validation.
