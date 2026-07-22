# COCO train2017 Scale-Up Cache Protocol Preregistration

Date: 2026-07-12 (Asia/Shanghai)

Status: frozen before the first non-dry-run export.

## Purpose

Create an independent COCO train2017 development population for scale-up B1 and diffusion training. Existing COCO val2017 exports, Imagenette policy-dev, and Imagenette official validation are excluded from model selection and training.

## Frozen population

- Source: local COCO2017 `train2017`, 118,287 files.
- Scale-up train: 10,000 images.
- Scale-up validation: 1,000 disjoint images.
- Selection seed: `20260713`.
- Selection rule: rank every relative source path by SHA-256 of `seed:path`; take the first 10,000 for train and the next 1,000 for validation.
- Export sample names: train `sample_000000.png`--`sample_009999.png`; validation `sample_010000.png`--`sample_010999.png`.
- Geometry: deterministic resize-short-side to 256 followed by center crop 256.

The exporter must save source path and source-file SHA-256 for every selected image, verify unique paths/hashes within each role, and verify no selected filename or file hash appears in local COCO val2017.

## Communication cache

- Frozen formal DeepJSCC checkpoint: `outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt`.
- Forbidden checkpoint: `latest.pt`.
- Channel: AWGN.
- CBR: `8/48=1/6` (`inner_channel=8`).
- SNRs: `[1,4,7,13,19]` dB.
- Independent deterministic channel seed for every `(SNR,batch_start)` derived from SHA-256 of the base seed, SNR, and batch offset.
- Output quantized to the PNG uint8 grid before metric calculation and saving.

## Integrity

- Never overwrite an output directory.
- No network or download is required.
- Save config, script, checkpoint hash, source manifest hash, run command, environment versions, per-sample PSNR, per-SNR summary, and sample grids.
- A smoke run must use a separate output directory and cannot change the frozen formal config.
- The formal cache is preparation, not a positive experiment result. It authorizes scale-up B1 training only if all 11,000 images and all five SNR directories are complete.
