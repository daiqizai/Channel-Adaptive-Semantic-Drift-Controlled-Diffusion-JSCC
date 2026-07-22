from __future__ import annotations

import math
import unittest

import torch

from cadsd_jscc.channel_matched_latent_diffusion import (
    ChannelMatchedLatentDenoiser,
    channel_alpha,
    deterministic_ddim,
    masked_mse_per_sample,
    normalize_channel_observation,
    reverse_alpha_schedule,
)


class ChannelMatchedLatentDiffusionTest(unittest.TestCase):
    def test_project_half_variance_mapping(self) -> None:
        gamma = 10.0 ** (1.0 / 10.0)
        expected = 2.0 * gamma / (2.0 * gamma + 1.0)
        self.assertAlmostEqual(channel_alpha(1.0, 0.5), expected, places=12)

    def test_channel_observation_has_forward_diffusion_coefficients(self) -> None:
        alpha = channel_alpha(4.0, 0.5)
        gamma = 10.0 ** 0.4
        x0 = torch.tensor([1.5, -0.5])
        epsilon = torch.tensor([0.25, -1.0])
        received = x0 + epsilon / math.sqrt(2.0 * gamma)
        observed = normalize_channel_observation(received, alpha)
        expected = math.sqrt(alpha) * x0 + math.sqrt(1.0 - alpha) * epsilon
        self.assertTrue(torch.allclose(observed, expected, atol=1e-6))

    def test_schedule_is_increasing_and_keeps_start(self) -> None:
        start = channel_alpha(1.0, 0.5)
        schedule = reverse_alpha_schedule(start, 6, 0.999)
        self.assertAlmostEqual(float(schedule[0]), start, places=6)
        self.assertAlmostEqual(float(schedule[-1]), 0.999, places=6)
        self.assertTrue(bool(torch.all(schedule[1:] > schedule[:-1])))

    def test_model_masks_inactive_coordinates_and_backpropagates(self) -> None:
        model = ChannelMatchedLatentDenoiser(
            latent_channels=2,
            base_channels=8,
            num_blocks=2,
            time_embedding_dim=8,
            group_norm_groups=4,
        )
        latent = torch.randn(3, 2, 8, 8, requires_grad=True)
        mask = torch.ones(2, 8, 8, dtype=torch.bool)
        mask[:, 0, 0] = False
        alpha = torch.tensor([0.72, 0.85, 0.95])
        output = model(latent, alpha, mask)
        self.assertEqual(tuple(output.shape), tuple(latent.shape))
        self.assertTrue(torch.equal(output[:, :, 0, 0], torch.zeros(3, 2)))
        output.square().mean().backward()
        self.assertIsNotNone(latent.grad)

    def test_masked_mse_excludes_invalid_coordinates(self) -> None:
        target = torch.zeros(1, 1, 2, 2)
        prediction = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        mask = torch.tensor([[[True, False], [False, True]]])
        self.assertAlmostEqual(float(masked_mse_per_sample(prediction, target, mask)), 8.5)

    def test_perfect_epsilon_ddim_recovers_clean_latent(self) -> None:
        x0 = torch.randn(2, 1, 4, 4)
        epsilon = torch.randn_like(x0)
        alpha = channel_alpha(1.0, 0.5)
        x_t = math.sqrt(alpha) * x0 + math.sqrt(1.0 - alpha) * epsilon
        mask = torch.ones(1, 4, 4, dtype=torch.bool)

        class Perfect(torch.nn.Module):
            def forward(self, state, current_alpha, valid_mask):
                return epsilon

        recovered = deterministic_ddim(
            Perfect(), x_t, mask, alpha_start=alpha, sampling_steps=4, alpha_max=0.999
        )
        self.assertTrue(torch.allclose(recovered, x0, atol=2e-5))


if __name__ == "__main__":
    unittest.main()
