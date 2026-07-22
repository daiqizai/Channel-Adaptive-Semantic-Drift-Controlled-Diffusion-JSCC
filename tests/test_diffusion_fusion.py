from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from cadsd_jscc.diffusion_fusion import (  # noqa: E402
    DualInputResidualRefiner,
    expand_b1_state_dict,
    parameter_count,
    residual_gate_tensor,
)
from s5_residual_refiner_pilot import SNRConditionedResidualRefiner  # noqa: E402


class DiffusionFusionTest(unittest.TestCase):
    def test_expansion_is_exact_b1_at_initialization(self) -> None:
        torch.manual_seed(7)
        b1 = SNRConditionedResidualRefiner(
            base_channels=8,
            num_blocks=2,
            condition_features=["sobel_magnitude", "laplacian_abs"],
        )
        torch.nn.init.normal_(b1.tail.weight, std=0.01)
        fusion = DualInputResidualRefiner(base_channels=8, num_blocks=2)
        fusion.load_state_dict(expand_b1_state_dict(b1.state_dict(), fusion), strict=True)
        b0 = torch.rand(3, 3, 17, 19)
        auxiliary = torch.rand_like(b0)
        snr = torch.tensor([0.05, 0.2, 0.35])
        gate = torch.tensor([0.12, 0.10, 0.08])
        expected = b1(b0, snr, gate)
        actual = fusion(b0, auxiliary, snr, gate)
        self.assertLess(float((expected - actual).abs().max().detach()), 1e-6)
        other = fusion(b0, torch.rand_like(b0), snr, gate)
        self.assertTrue(torch.equal(actual, other))

    def test_control_and_fusion_have_identical_parameter_count(self) -> None:
        control = DualInputResidualRefiner(64, 6)
        fusion = DualInputResidualRefiner(64, 6)
        self.assertEqual(parameter_count(control), parameter_count(fusion))
        self.assertEqual(parameter_count(control), 450115)

    def test_gate_lookup(self) -> None:
        values = residual_gate_tensor(
            torch.tensor([1.0, 19.0]), {"1": 0.12, "19": 0.04}, torch.device("cpu")
        )
        self.assertTrue(torch.allclose(values, torch.tensor([0.12, 0.04])))


if __name__ == "__main__":
    unittest.main()
