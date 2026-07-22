# Posterior Imagenette Scratch-Gate Follow-Up Preregistration

Date: 2026-07-13. Frozen before non-dry-run. Analysis ID: `ANALYSIS-PC-RISK-001`. Phase remains S5 validation; this is a controller follow-up inside the existing posterior-consistency study, not a new project stage.

## Status and limitation

This audit reuses Imagenette `policy_dev`, whose aggregate `T_cls` outcomes were already inspected in `ANALYSIS-PC-SUP-001`. It is therefore a development diagnostic and cannot serve as independent final confirmation. No threshold, model, checkpoint, or rule is fitted from PC-SUP outcomes: the scratch `G_gate` checkpoint and exact top-1 agreement fallback were frozen by the 2026-07-10 supervised protocol before posterior correction existed. Official Imagenette validation remains sealed and must not be accessed.

## Frozen change

Change exactly one component relative to `ANALYSIS-PC-SUP-001`:

- replace the ImageNet AlexNet+ResNet18 consensus controller with the existing scratch MobileNetV3-Small `G_gate`;
- accept posterior iff `G_gate(posterior).top1 == G_gate(anchor).top1`; otherwise return the S13 B1 anchor;
- keep the independent scratch ResNet18 `T_cls` as the outcome evaluator only. WNID, original image, and `T_cls` outputs never enter the controller.

Both scratch classifiers were trained only on `cls_train`, selected and temperature-calibrated only on `cls_cal`, and did not use policy-dev for fitting. Their architectures, seeds, and roles are independent. The script must reject checkpoints that do not prove these contracts.

Everything else is frozen unchanged: the `1894` policy-dev images, AWGN, `c=8`, CBR `1/6`, SNRs `[1,4,7,13,19]`, seed `20260721`, S13 B1 anchor, S14 diffusion output, three received-latent posterior steps, normalized step size `0.001`, LPIPS implementation, and clean-correct rule (`T_cls(original)=WNID`, calibrated confidence `>=0.50`). Reusing the seed makes every raw/posterior row paired exactly with PC-SUP, so only the controller may change the final output.

## Frozen gates

All gates are required:

1. At least `1600` unique images enter the clean-correct subset.
2. Received-latent consistency decreases at all five SNRs; posterior PSNR is positive at at least four; posterior LPIPS is nonpositive at at least four.
3. Controlled-final mean PSNR is positive and mean LPIPS is nonpositive relative to S14 raw.
4. Across primary SNRs `[1,4,7]`, controlled-final new errors do not exceed raw in total.
5. At each primary SNR, controlled-final new errors do not exceed raw.
6. Across primary SNRs, controlled-final supervised failures do not exceed raw.

A pass retains this task-matched scratch controller as the current supervised development candidate. It does not make the method independently confirmed, does not establish semantic safety, and does not automatically unlock official validation. A failure retires this fixed scratch top-1 fallback for posterior correction; no threshold search on policy-dev is permitted.
