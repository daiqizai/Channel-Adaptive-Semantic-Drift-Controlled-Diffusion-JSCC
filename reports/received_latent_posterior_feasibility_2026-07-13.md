# Received-Latent Posterior Consistency Feasibility

Date: 2026-07-13

The formal DeepJSCC implementation exposes a receiver-visible channel latent and supports differentiable image-to-latent consistency without modifying the third-party model.

Actual checkpoint smoke at 7 dB:

- split `encoder → channel → decoder` versus original `model(x)` max error: `1.788e-7`;
- received latent shape for 256×256, `c=8`: `(B,16,64,64)`;
- normalized transmitted latent power: `0.9999999`;
- received latent power: `1.09734`;
- B0 candidate normalized received-latent loss: `0.06218`;
- image-gradient mean absolute value: `9.38e-6`, finite and nonzero.

Implemented interfaces in `src/cadsd_jscc/deepjscc_adapter.py`:

- `deepjscc_encode`;
- `deepjscc_transmit`;
- `deepjscc_decode`;
- `deepjscc_forward_with_latents`;
- `received_latent_consistency_loss`.

This establishes engineering feasibility only. It does not yet show quality or semantic improvement. The next method must export/store the actual received latent associated with each cached reconstruction and apply a preregistered proximal data-consistency update inside a materially new posterior sampler. It must not tune S14 on the existing validation split.
