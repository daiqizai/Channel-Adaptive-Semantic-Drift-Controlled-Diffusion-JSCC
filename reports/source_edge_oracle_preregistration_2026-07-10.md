# Sender Source-Edge Oracle Preregistration

Date: 2026-07-10 (Asia/Shanghai)

Status: frozen before training or inspecting `EXP-S4-011` outcomes.

## Purpose

The coarse sender-description router failed its nested audit. This experiment tests the other SGD-JSCC mechanism: injecting fine sender-side structure into restoration itself. It is deliberately an oracle feasibility test before spending effort on a separately coded edge channel.

## Frozen comparison

`EXP-S4-011` matches `EXP-S4-008` on COCO sample split, seed, five AWGN SNRs, CBR of the existing main image path, model width/depth, input-channel count, residual gates, optimizer, loss, crop augmentation, epochs and checkpoint selection. The sole intended method difference is the image from which the two structural channels are computed:

- `EXP-S4-008`: receiver-visible M0 Sobel magnitude and absolute Laplacian;
- `EXP-S4-011`: sender original Sobel magnitude and absolute Laplacian.

The sender structural maps are perfectly available to the receiver in this oracle. Their rate and channel errors are not accounted, so the result cannot be presented as a communication-system improvement or as the final M2/M3. The existing main path remains CBR 0.17; total CBR is undefined for this oracle.

## Decision

Primary feasibility endpoint: mean raw-refined PSNR of source-edge minus receiver-M0-edge over the 64-image validation split and five SNRs. A paired sample-cluster bootstrap with 10,000 replicates must have a 95% lower endpoint above zero, and the point estimate must be positive at every SNR.

Secondary endpoints are LPIPS, top-1-fallback PSNR and pseudo semantic event counts. Pseudo labels are diagnostic only. A positive oracle authorizes a follow-up with a separately transmitted lossy edge representation and matched total CBR. A negative or inconsistent oracle stops this direction.
