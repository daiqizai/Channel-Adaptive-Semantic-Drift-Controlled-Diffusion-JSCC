# Semantic-Sketch Causal Ablation Preregistration

Date: 2026-07-11 (Asia/Shanghai)

Status: frozen after `EXP-S8-001` completed and before running zero/shuffled-sketch inference.

The checkpoint, validation images, five SNRs and received-sketch outputs are fixed. No model is retrained. Two counterfactuals are evaluated:

- `zeros`: replace the recovered 32-D sketch by zeros.
- `shuffled`: replace each sample's sketch by the next manifest sample's same-SNR sketch.

RGB main, decoded hybrid structure, SNR, checkpoint and residual gates remain unchanged. Therefore received-versus-counterfactual differences isolate whether the trained FiLM path uses sample-specific semantic information; they do not establish supervised semantic safety.

Primary evidence is paired PSNR clustered by the 64 image IDs across all five SNRs, with 10,000 bootstrap replicates. A causal-use result requires the 95% CI lower endpoint of received-minus-zero and received-minus-shuffled PSNR to be positive, plus non-negative received-minus-counterfactual point estimates at every SNR.

AlexNet semantics are not evaluated because AlexNet entered training. Official Imagenette validation remains sealed.
