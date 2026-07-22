# Hybrid Semantic Sketch R4 Follow-up Preregistration

Date: 2026-07-11 (Asia/Shanghai)

Status: frozen after `EXPORT-S8-001` failed its structure-distortion gate and before running this follow-up. Official Imagenette validation remains sealed.

`EXPORT-S8-001` used 32 dimensions repeated 16 times. Sketch cosine passed at every SNR (`0.9882` even at 1 dB), but structure first-two-channel MSE increased by `16.41%` to `29.40%`, failing the fixed 10% limit. Its output is retained unchanged.

This follow-up changes exactly one method hyperparameter: repetition count `16 -> 4`. The semantic source, 32-D projection, seed, structure checkpoint, images, SNRs, channel seed and stage gates remain unchanged. Payload use becomes 128/16,384 real symbols (`0.78125%` of the existing c=2 latent), with no extra total rate.

The same gates apply:

1. Mean sketch cosine `>=0.95` at every SNR.
2. Relative first-two-channel structure MSE increase `<=10%` at every SNR.
3. Exact total rate remains `c=6+c=2=c=8`.

Only if all gates pass may R4 supply the semantic-conditioned refiner. This is a protocol-defined efficiency correction after a failed redundancy setting, not a reinterpretation of S8-001.
