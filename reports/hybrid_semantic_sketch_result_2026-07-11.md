# Rate-Accounted Hybrid Semantic Sketch Result

Date: 2026-07-11 (Asia/Shanghai)

## Result

This stage implemented a genuine SGD-inspired coarse-semantic side channel without changing total rate:

```text
reference: c=8 RGB                                  total CBR = 1/6
hybrid:    c=6 RGB + c=2 structure/semantic         total CBR = 1/6
```

A frozen AlexNet probability vector is projected to a 32-D continuous sketch. Repetition-4 uses 128 of the existing 16,384 real `c=2` latent symbols (`0.78125%`), leaving 99.21875% for structure. No extra channel use or error-free metadata is assumed.

## Payload validation

The first repetition-16 setting is retained as a failed experiment: sketch cosine passed, but erased payload locations increased structure MSE by 16.41%-29.40%. The preregistered repetition-4 follow-up passed both gates:

| SNR | Recovered sketch cosine | Structure first-2 MSE increase |
|---:|---:|---:|
| 1 | `0.9552` | `+3.24%` |
| 4 | `0.9772` | `+4.31%` |
| 7 | `0.9880` | `+5.04%` |
| 13 | `0.9970` | `+5.61%` |
| 19 | `0.9992` | `+5.84%` |

## Refiner result

Three frozen training attempts were retained. Ordinary semantic conditioning improved quality but failed received-versus-zero ablation. Batch-mean counterfactual ranking established a received-versus-zero effect but failed shuffled-sketch significance. `EXP-S8-003` changed ranking to per-sample hinges and selected epoch 5; checkpoint SHA-256 is `64754d2da87984c07d699b7b961b16e40fe1742436504dbb59a27df7d706f50f`.

On 160 frozen downstream images across held-out, test-like and fresh-holdout:

- hybrid received-sketch raw versus reference `c=8`: `+0.4691 dB`, 95% image-cluster CI `[+0.4231,+0.5159]`;
- hybrid received-sketch raw versus S7 decoded-structure raw: `+0.0919 dB`, CI `[+0.0746,+0.1147]`;
- received versus zero sketch: `+0.0849 dB`, CI `[+0.0728,+0.0982]`, positive in all 15 split-by-SNR cells;
- received versus shuffled sketch: `+0.0072 dB`, CI `[-0.0023,+0.0170]`; one held-out 1 dB cell is `-0.00036 dB`.

Thus the rate-accounted semantic side signal is causally useful compared with removing it, and the total matched-rate quality result improves materially. However, the correct sample-specific sketch is not significantly better than a wrong sketch. Much of the FiLM benefit behaves like a global nonzero conditioning signal rather than strong semantic grounding.

## Decision

The preregistered causal-use gate is **FAIL** because received-minus-shuffled does not exclude zero. The project must not claim that the current 32-D random projection solves semantic grounding, and it must not spend the existing Imagenette policy-dev set on another audit of this checkpoint. Official Imagenette validation remains sealed.

The next semantic representation should preserve explicitly interpretable class/caption or spatial-token identity instead of a dense random projection. It needs a newly isolated development/audit population before supervised promotion.
