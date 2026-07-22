# Per-Sample Counterfactual Semantic Refiner Preregistration

Date: 2026-07-11 (Asia/Shanghai)

Status: frozen after `ANALYSIS-S8-003` failed and before training `EXP-S8-003`.

`EXP-S8-002` established a significant received-versus-zero effect on validation and frozen downstream data, but not received-versus-shuffled. Inspection of the frozen implementation found that its hinge ranking used one batch-mean MSE scalar. Positive and negative sample-specific effects could therefore cancel before the hinge.

This follow-up restarts from `EXP-S7-002` and changes exactly the ranking scope: compute received, zero and shuffled MSE separately for every sample, apply both margin hinges per sample, then average. Architecture, R4 payload, data, seed, optimizer, 30 epochs, semantic losses, rank weight 10, margin `5e-6`, rate and checkpoint eligibility remain unchanged.

After training, validation and the same frozen 160-image downstream causal ablations are rerun. Promotion requires received-minus-zero and received-minus-shuffled paired bootstrap CI lower endpoints `>0`, with non-negative split-by-SNR point estimates. Until then, Imagenette policy-dev and official validation remain untouched.
