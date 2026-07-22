from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from s10_short_chain_residual_shift_diffusion import (  # noqa: E402
    ResidualShiftDenoiser,
    ShortChainResidualShiftDiffusion,
    batch_condition,
)


def config() -> dict:
    return {
        "model": {"input_channels": 10, "base_channels": 8, "num_blocks": 1},
        "diffusion": {
            "train_timesteps": 10,
            "sampling_steps": 4,
            "bridge_noise_sigma": 0.05,
        },
    }


class ResidualShiftDiffusionTests(unittest.TestCase):
    def test_bridge_endpoints_are_exact_without_endpoint_noise(self) -> None:
        target = torch.rand(2, 3, 8, 8)
        anchor = torch.rand_like(target)
        noise = torch.randn_like(target)
        tau_zero = torch.zeros(2)
        tau_one = torch.ones(2)
        at_target = ShortChainResidualShiftDiffusion.bridge_state(
            target, anchor, tau_zero, noise, 0.05
        )
        at_anchor = ShortChainResidualShiftDiffusion.bridge_state(
            target, anchor, tau_one, noise, 0.05
        )
        self.assertTrue(torch.allclose(at_target, target))
        self.assertTrue(torch.allclose(at_anchor, anchor))

    def test_zero_initialized_tail_preserves_anchor(self) -> None:
        model = ShortChainResidualShiftDiffusion(config())
        anchor = torch.rand(2, 3, 8, 8)
        structure = torch.rand_like(anchor)
        output = model(
            anchor,
            torch.tensor([0.1, 0.5]),
            torch.tensor([0.08, 0.05]),
            condition_image=structure,
        )
        self.assertTrue(torch.allclose(output, anchor, atol=1e-7))

    def test_denoiser_channel_contract(self) -> None:
        denoiser = ResidualShiftDenoiser(10, 8, 1)
        tensor = torch.rand(2, 3, 8, 8)
        output = denoiser(
            tensor,
            tensor,
            tensor,
            torch.tensor([0.1, 0.2]),
            torch.tensor([0.5, 1.0]),
        )
        self.assertEqual(tuple(output.shape), (2, 3, 8, 8))

    def test_receiver_anchor_structure_needs_no_external_condition(self) -> None:
        receiver_config = config()
        receiver_config["model"]["condition_source"] = "receiver_m0"
        receiver_config["model"]["diffusion_condition_source"] = (
            "receiver_anchor_structural_maps"
        )
        model = ShortChainResidualShiftDiffusion(receiver_config)
        anchor = torch.rand(2, 3, 8, 8)
        condition = batch_condition(receiver_config, anchor, {})
        self.assertEqual(tuple(condition.shape), (2, 2, 8, 8))
        output = model(
            anchor,
            torch.tensor([0.1, 0.5]),
            torch.tensor([0.08, 0.05]),
        )
        self.assertTrue(torch.allclose(output, anchor, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
