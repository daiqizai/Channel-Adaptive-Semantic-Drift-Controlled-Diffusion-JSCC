from __future__ import annotations

import torch
from types import SimpleNamespace

from cadsd_jscc.external_rate_alignment import (
    ExactRateMaskedDeepJSCC,
    RepeatedSparseChannelAdapter,
    evenly_spaced_active_indices,
)


class _Encoder(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


class _Decoder(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


class _IdentityJSCC(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Encoder()
        self.decoder = _Decoder()


def test_evenly_spaced_indices_are_unique_and_cover_range() -> None:
    indices = evenly_spaced_active_indices(24576, 19712)
    assert indices.shape == (19712,)
    assert int(torch.unique(indices).numel()) == 19712
    assert int(indices.min()) == 0
    assert int(indices.max()) < 24576


def test_exact_rate_wrapper_normalizes_and_uses_only_active_noise() -> None:
    model = ExactRateMaskedDeepJSCC(
        _IdentityJSCC(), dense_symbols=8, active_symbols=4, snr_db=100.0
    )
    source = torch.arange(1, 9, dtype=torch.float32).reshape(1, 1, 2, 4)
    active, dense_shape = model.encode_active(source)
    assert dense_shape == (1, 2, 4)
    assert torch.allclose(active.square().mean(), torch.tensor(1.0), atol=1e-6)
    result = model.forward_with_standard_normal(source, torch.zeros(4))
    flat = result.flatten()
    inactive = torch.ones(8, dtype=torch.bool)
    inactive[model.active_indices.cpu()] = False
    assert torch.equal(flat[inactive], torch.zeros(4))
    assert int((flat != 0).sum()) == 4


def test_repeated_sparse_adapter_averages_independent_main_copies() -> None:
    model = SimpleNamespace(
        snr=0.0,
        canny_transmission_net=SimpleNamespace(
            channel=SimpleNamespace(forward=lambda tx, snr: tx)
        ),
    )
    main_base = 4 * 4
    adapter = RepeatedSparseChannelAdapter(
        main_noise=torch.cat([torch.ones(main_base), -torch.ones(main_base)]),
        edge_noise=torch.zeros(4 * 2),
        expected_patches=4,
        main_per_patch=4,
        edge_active_per_patch=2,
        main_repetition=2,
        edge_repetition=1,
        noise_variance_factor=0.5,
    )
    adapter.attach(model, torch)
    features = torch.ones(4, 1, 2, 2)
    assert torch.equal(model.channel(features), features)
    edge = torch.tensor([[1.0, 0.0, -1.0]] * 4)
    received_edge = model.canny_transmission_net.channel.forward(
        edge, torch.zeros(4)
    )
    assert torch.equal(received_edge, edge)
    adapter.require_complete()
    assert adapter.main_base_real_symbols == 16
    assert adapter.main_transmitted_real_symbols == 32
    assert adapter.edge_active_base_real_symbols == 8
    assert adapter.edge_transmitted_real_symbols == 8
