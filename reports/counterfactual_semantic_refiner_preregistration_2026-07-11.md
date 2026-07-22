# Counterfactual Semantic Refiner Follow-up Preregistration

Date: 2026-07-11 (Asia/Shanghai)

Status: frozen after `ANALYSIS-S8-001` failed and before training `EXP-S8-002`.

`EXP-S8-001` improved reconstruction over S7, but frozen-checkpoint ablation found received-minus-zero PSNR `-0.0282 dB` (95% CI `[-0.0460,-0.0131]`) and received-minus-shuffled `+0.0299 dB` (CI `[-0.0006,+0.0658]`). Its gain therefore cannot be attributed to useful received semantics.

This follow-up restarts from `EXP-S7-002`, not from the failed semantic checkpoint. Architecture, data, R4 payload, SNR gates and total rate are unchanged. It adds:

1. Frozen-teacher projected-sketch cosine loss, weight `5e-4`.
2. Source-probability KL, reduced weight `1e-4`.
3. Received-versus-zero and received-versus-shuffled reconstruction ranking, weight `10`, MSE margin `5e-6`.

The formal run is 30 epochs at learning rate `1e-4`. A validation epoch is checkpoint-eligible only if received-sketch MSE is strictly lower than both zero-sketch and shuffled-sketch MSE. Among eligible epochs, lowest received MSE wins. If no epoch is eligible, training fails closed and no checkpoint is promoted.

After training, the same 64-image x 5-SNR paired causal ablation must be rerun. Semantic-sketch use passes only if received-minus-zero and received-minus-shuffled PSNR bootstrap CI lower endpoints are positive and all per-SNR point estimates are non-negative.

AlexNet cannot provide semantic success evidence because it enters training. No Imagenette access occurs until a checkpoint passes this causal-use gate and a separate scratch-`T_cls` audit protocol is frozen.
