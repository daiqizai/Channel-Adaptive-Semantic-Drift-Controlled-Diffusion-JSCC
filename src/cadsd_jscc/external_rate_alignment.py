from __future__ import annotations

from typing import Any

import torch


def evenly_spaced_active_indices(dense_symbols: int, active_symbols: int) -> torch.Tensor:
    """Return deterministic unique indices spanning a flattened dense latent."""

    dense_symbols = int(dense_symbols)
    active_symbols = int(active_symbols)
    if dense_symbols <= 0 or active_symbols <= 0 or active_symbols > dense_symbols:
        raise ValueError("require 0 < active_symbols <= dense_symbols")
    indices = torch.div(
        torch.arange(active_symbols, dtype=torch.int64) * dense_symbols,
        active_symbols,
        rounding_mode="floor",
    )
    if int(torch.unique(indices).numel()) != active_symbols:
        raise RuntimeError("evenly spaced index construction produced duplicates")
    return indices


class ExactRateMaskedDeepJSCC(torch.nn.Module):
    """Trainable DeepJSCC wrapper with an exact active real-symbol budget.

    The encoder may emit a denser latent than the link can carry.  A frozen
    flattened mask selects the transmitted coordinates, normalizes only those
    coordinates to unit mean power, applies paired-real complex AWGN, and
    scatters the received coordinates back into a zero-filled dense tensor.
    """

    def __init__(
        self,
        base_model: torch.nn.Module,
        *,
        dense_symbols: int,
        active_symbols: int,
        snr_db: float,
    ) -> None:
        super().__init__()
        if active_symbols % 2:
            raise ValueError("active real-symbol count must be even")
        self.base_model = base_model
        self.dense_symbols = int(dense_symbols)
        self.active_symbols = int(active_symbols)
        self.snr_db = float(snr_db)
        self.register_buffer(
            "active_indices",
            evenly_spaced_active_indices(self.dense_symbols, self.active_symbols),
            persistent=True,
        )

    def encode_active(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        dense = self.base_model.encoder(images)
        if int(dense[0].numel()) != self.dense_symbols:
            raise RuntimeError(
                f"encoder emitted {dense[0].numel()} real symbols, expected {self.dense_symbols}"
            )
        flat = dense.flatten(start_dim=1)
        active = flat.index_select(1, self.active_indices)
        # Compute the power normalization in fp32 under AMP, then return the
        # encoder dtype so gradients still flow through the selection.
        norm = active.float().square().sum(dim=1, keepdim=True).clamp_min(1e-12).sqrt()
        active = active * (self.active_symbols**0.5 / norm).to(active.dtype)
        return active, tuple(dense.shape[1:])

    def transmit_active(
        self, active: torch.Tensor, standard_normal: torch.Tensor | None = None
    ) -> torch.Tensor:
        if tuple(active.shape[1:]) != (self.active_symbols,):
            raise ValueError("active tensor has the wrong per-sample symbol count")
        if standard_normal is None:
            noise = torch.randn_like(active)
        else:
            if int(standard_normal.numel()) != int(active.numel()):
                raise ValueError("canonical noise length does not match active symbols")
            noise = standard_normal.to(device=active.device, dtype=active.dtype).reshape_as(active)
        power = active.float().square().mean(dim=1, keepdim=True)
        sigma = torch.sqrt(power / (2.0 * (10.0 ** (self.snr_db / 10.0))))
        return active + noise * sigma.to(active.dtype)

    def decode_active(
        self, received_active: torch.Tensor, dense_shape: tuple[int, ...]
    ) -> torch.Tensor:
        dense = received_active.new_zeros((received_active.shape[0], self.dense_symbols))
        dense.index_copy_(1, self.active_indices, received_active)
        return self.base_model.decoder(dense.reshape(received_active.shape[0], *dense_shape))

    def forward_with_standard_normal(
        self, images: torch.Tensor, standard_normal: torch.Tensor
    ) -> torch.Tensor:
        active, dense_shape = self.encode_active(images)
        received = self.transmit_active(active, standard_normal)
        return self.decode_active(received, dense_shape)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        active, dense_shape = self.encode_active(images)
        received = self.transmit_active(active)
        return self.decode_active(received, dense_shape)


class RepeatedSparseChannelAdapter:
    """Use independent repetitions for released SGD-JSCC analog branches."""

    def __init__(
        self,
        *,
        main_noise: Any,
        edge_noise: Any,
        expected_patches: int,
        main_per_patch: int,
        edge_active_per_patch: int,
        main_repetition: int,
        edge_repetition: int,
        noise_variance_factor: float,
    ) -> None:
        self.main_noise = main_noise
        self.edge_noise = edge_noise
        self.expected_patches = int(expected_patches)
        self.main_per_patch = int(main_per_patch)
        self.edge_active_per_patch = int(edge_active_per_patch)
        self.main_repetition = int(main_repetition)
        self.edge_repetition = int(edge_repetition)
        self.noise_variance_factor = float(noise_variance_factor)
        if self.main_repetition <= 0 or self.edge_repetition <= 0:
            raise ValueError("analog repetitions must be positive")
        if self.noise_variance_factor not in {0.5, 1.0}:
            raise ValueError("unsupported real-coordinate AWGN variance factor")
        self.main_base_real_symbols: int | None = None
        self.main_transmitted_real_symbols: int | None = None
        self.edge_active_base_real_symbols: int | None = None
        self.edge_transmitted_real_symbols: int | None = None

    def attach(self, model: Any, torch_module: Any) -> None:
        torch = torch_module

        def repeated_main(features: Any) -> Any:
            if int(features.shape[0]) != self.expected_patches:
                raise RuntimeError("main channel batch does not match expected patches")
            if int(features[0].numel()) != self.main_per_patch:
                raise RuntimeError("unexpected released SGD-JSCC main latent size")
            base_count = int(features.numel())
            expected = base_count * self.main_repetition
            if int(self.main_noise.numel()) != expected:
                raise RuntimeError("main repetition noise has the wrong length")
            self.main_base_real_symbols = base_count
            self.main_transmitted_real_symbols = expected
            z = self.main_noise.to(device=features.device, dtype=features.dtype).reshape(
                self.main_repetition, *features.shape
            ).mean(dim=0)
            power = features.flatten(start_dim=1).float().square().mean(dim=1)
            sigma = torch.sqrt(
                self.noise_variance_factor
                * power
                / (10.0 ** (float(model.snr) / 10.0))
            ).reshape([-1, 1, 1, 1])
            return features + z * sigma.to(features.dtype)

        model.channel = repeated_main

        def repeated_edge(tx: Any, snr: Any) -> Any:
            if int(tx.shape[0]) != self.expected_patches:
                raise RuntimeError("edge channel batch does not match expected patches")
            masks = tx != 0
            counts = [int(row.sum().item()) for row in masks]
            if any(count != self.edge_active_per_patch for count in counts):
                raise RuntimeError(f"unexpected active edge counts: {counts}")
            base_count = sum(counts)
            expected = base_count * self.edge_repetition
            if int(self.edge_noise.numel()) != expected:
                raise RuntimeError("edge repetition noise has the wrong length")
            self.edge_active_base_real_symbols = base_count
            self.edge_transmitted_real_symbols = expected
            z = self.edge_noise.to(device=tx.device, dtype=tx.dtype).reshape(
                self.edge_repetition, self.expected_patches, self.edge_active_per_patch
            ).mean(dim=0)
            output = torch.zeros_like(tx)
            noise_var = self.noise_variance_factor / (10.0 ** (snr / 10.0))
            for index in range(self.expected_patches):
                mask = masks[index]
                output[index, mask] = tx[index, mask] + z[index] * torch.sqrt(
                    noise_var[index]
                ).reshape(())
            return output

        model.canny_transmission_net.channel.forward = repeated_edge

    def require_complete(self) -> None:
        if self.main_transmitted_real_symbols is None:
            raise RuntimeError("main repeated channel was not observed")
        if self.edge_transmitted_real_symbols is None:
            raise RuntimeError("edge repeated channel was not observed")
