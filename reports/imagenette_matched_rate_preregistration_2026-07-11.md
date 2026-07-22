# Imagenette Matched-Total-Rate Policy-Dev Preregistration

Date: 2026-07-11 (Asia/Shanghai)

Status: frozen after COCO validation/heldout/test-like/fresh quality and pseudo-semantic results were inspected, but before running the matched-rate system on any Imagenette image. Official Imagenette validation remains sealed.

## Question

Does the explicit `c=6` RGB main path plus `c=2` decoded-structure path improve equal-total-rate reconstruction without worsening independently evaluated supervised semantics?

Both systems use total CBR `8/48=1/6`:

```text
reference: c=8 RGB
matched:   c=6 RGB + c=2 Sobel/Laplacian structure
```

The matched refiner was trained only on COCO and is frozen. It receives only receiver-visible c=6 reconstruction, decoded c=2 structural RGB and SNR. Neither Imagenette labels nor either scratch classifier entered communication/refiner training.

## Data and semantic independence

This audit reuses the existing sealed Imagenette protocol and only accesses `policy_dev`. `G_gate` is the frozen scratch MobileNetV3-Small used for the optional top-1 fallback. `T_cls` is the frozen scratch ResNet18 evaluator and never enters inference. Clean membership remains:

```text
A = {i: T_cls(original_i)=y_i and calibrated confidence >= 0.50}
```

The primary semantic scope is SNR `{1,4,7}`. Quality uses all policy-dev images and five SNRs. The official Imagenette validation archive/directory may not be accessed.

## Frozen arms and channel randomness

- `reference_c8`: existing stable epoch-73 c=8 DeepJSCC checkpoint.
- `matched_main_c6`: frozen 20k-pilot c=6 checkpoint.
- `matched_raw`: c=6 output refined with the frozen decoded-structure refiner using the independently transmitted c=2 structural representation.
- `matched_top1_fallback`: accept matched raw only when scratch `G_gate` top-1 agrees with c=6 main; otherwise return c=6 main.

Reference, main and structure paths use independent deterministic AWGN streams from seed `20260711` and fixed arm offsets. All materialized stages are rounded to the 8-bit PNG grid before metrics/classification.

## Primary matched-raw endpoints

1. Supervised failure difference `matched_raw - reference_c8`; paired image-cluster bootstrap 95% CI upper endpoint must be `<=0`.
2. Among rows where `reference_c8` is correct, the conservative accepted/new-error upper bound for `matched_raw` being wrong must be `<=0.005`.
3. PSNR difference `matched_raw - reference_c8`; 95% CI lower endpoint must be positive and every SNR point estimate must be positive.
4. LPIPS difference `matched_raw - reference_c8` must be negative.

The conservative new-error bound is the maximum of the clustered one-sided 95th percentile and image-any-event Clopper-Pearson 95% upper bound. All bootstrap intervals use 10,000 replicates clustered by image ID, preserving all selected SNR rows.

The top-1 fallback is reported as a secondary arm; it is not allowed to rescue a failed matched-raw endpoint by changing the primary method after outcomes are known.

## Decision discipline

Passing policy-dev makes the matched-rate system a supervised-positive development result, not a final result. It does not unlock official val automatically; a full-data checkpoint or an explicitly frozen pilot-method decision is still required. Failure is retained and the official val remains sealed.
