from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cadsd_jscc.deepjscc_adapter import (  # noqa: E402
    deepjscc_forward_with_latents,
    received_latent_consistency_loss,
    received_latent_consistency_per_sample,
)


class TinyJSCC(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Conv2d(3, 4, 1, bias=False)
        self.channel = nn.Identity()
        self.decoder = nn.Conv2d(4, 3, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.channel(self.encoder(x)))


class LatentConsistencyTests(unittest.TestCase):
    def test_split_forward_matches_model_forward(self) -> None:
        model = TinyJSCC()
        images = torch.rand(2, 3, 8, 8)
        reconstruction, transmitted, received = deepjscc_forward_with_latents(model, images)
        self.assertTrue(torch.allclose(reconstruction, model(images)))
        self.assertTrue(torch.equal(transmitted, received))

    def test_received_candidate_has_zero_consistency_loss(self) -> None:
        model = TinyJSCC()
        images = torch.rand(2, 3, 8, 8)
        _, _, received = deepjscc_forward_with_latents(model, images)
        loss = received_latent_consistency_loss(model, images, received)
        self.assertLess(float(loss.detach()), 1e-10)

    def test_consistency_loss_backpropagates_to_candidate(self) -> None:
        model = TinyJSCC()
        source = torch.rand(2, 3, 8, 8)
        _, _, received = deepjscc_forward_with_latents(model, source)
        candidate = torch.rand_like(source, requires_grad=True)
        loss = received_latent_consistency_loss(model, candidate, received)
        loss.backward()
        self.assertIsNotNone(candidate.grad)
        self.assertTrue(torch.isfinite(candidate.grad).all())
        self.assertGreater(float(candidate.grad.abs().sum().detach()), 0.0)

    def test_per_sample_consistency_reduces_to_scalar_mean(self) -> None:
        model = TinyJSCC()
        source = torch.rand(3, 3, 8, 8)
        _, _, received = deepjscc_forward_with_latents(model, source)
        candidate = torch.rand_like(source)
        values = received_latent_consistency_per_sample(model, candidate, received)
        self.assertEqual(tuple(values.shape), (3,))
        self.assertTrue(
            torch.allclose(values.mean(), received_latent_consistency_loss(model, candidate, received))
        )

    def test_masked_consistency_ignores_reserved_positions(self) -> None:
        model = TinyJSCC()
        source = torch.rand(2, 3, 8, 8)
        _, _, received = deepjscc_forward_with_latents(model, source)
        candidate = torch.rand_like(source)
        mask = torch.ones(received[0].numel(), dtype=torch.bool)
        mask[[0, 7, 19]] = False
        baseline = received_latent_consistency_per_sample(
            model, candidate, received, valid_mask=mask
        )
        changed = received.clone().flatten(start_dim=1)
        changed[:, ~mask] += 1000.0
        changed = changed.view_as(received)
        repeated = received_latent_consistency_per_sample(
            model, candidate, changed, valid_mask=mask
        )
        self.assertTrue(torch.allclose(baseline, repeated))

    def test_masked_consistency_rejects_empty_mask(self) -> None:
        model = TinyJSCC()
        source = torch.rand(1, 3, 8, 8)
        _, _, received = deepjscc_forward_with_latents(model, source)
        with self.assertRaises(ValueError):
            received_latent_consistency_loss(
                model,
                source,
                received,
                valid_mask=torch.zeros(received[0].numel(), dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()
