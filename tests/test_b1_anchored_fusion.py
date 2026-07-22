import unittest

import torch

from cadsd_jscc.b1_anchored_fusion import (
    B1AnchoredGatedAdapter,
    injection_gate_tensor,
)


class B1AnchoredFusionTests(unittest.TestCase):
    def test_zero_initialized_adapter_is_exact_anchor(self) -> None:
        model = B1AnchoredGatedAdapter(base_channels=8, num_blocks=1)
        anchor = torch.rand(2, 3, 16, 16)
        auxiliary = torch.rand_like(anchor)
        output, diagnostics = model(
            anchor,
            auxiliary,
            torch.tensor([0.05, 0.35]),
            torch.tensor([0.12, 0.08]),
            return_diagnostics=True,
        )
        self.assertTrue(torch.equal(output, anchor))
        self.assertEqual(tuple(diagnostics["spatial_gate"].shape), (2, 1, 16, 16))
        self.assertTrue(torch.equal(diagnostics["injection"], torch.zeros_like(anchor)))

    def test_zero_max_injection_is_exact_anchor_after_parameter_change(self) -> None:
        model = B1AnchoredGatedAdapter(base_channels=8, num_blocks=1)
        with torch.no_grad():
            model.residual_head.weight.normal_()
            model.residual_head.bias.normal_()
            model.gate_head.weight.normal_()
        anchor = torch.rand(2, 3, 16, 16)
        auxiliary = torch.rand_like(anchor)
        output = model(
            anchor,
            auxiliary,
            torch.tensor([0.65, 0.95]),
            torch.zeros(2),
        )
        self.assertTrue(torch.equal(output, anchor))

    def test_input_contract_and_gate_lookup(self) -> None:
        gates = {"1": 0.12, "13": 0.0}
        values = injection_gate_tensor(torch.tensor([1.0, 13.0]), gates, torch.device("cpu"))
        self.assertTrue(torch.equal(values, torch.tensor([0.12, 0.0])))
        model = B1AnchoredGatedAdapter(base_channels=8, num_blocks=1)
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            model(
                torch.rand(1, 3, 8, 8),
                torch.rand(1, 3, 7, 8),
                torch.tensor([0.1]),
                torch.tensor([0.1]),
            )

    def test_fixed_one_gate_uses_residual_but_high_snr_is_exact(self) -> None:
        model = B1AnchoredGatedAdapter(
            base_channels=8, num_blocks=1, spatial_gate_mode="fixed_one"
        )
        with torch.no_grad():
            model.residual_head.bias.fill_(0.5)
        anchor = torch.full((2, 3, 8, 8), 0.25)
        auxiliary = torch.full_like(anchor, 0.75)
        output, diagnostics = model(
            anchor,
            auxiliary,
            torch.tensor([0.05, 0.95]),
            torch.tensor([0.12, 0.0]),
            return_diagnostics=True,
        )
        self.assertTrue(torch.equal(diagnostics["spatial_gate"], torch.ones(2, 1, 8, 8)))
        self.assertGreater(float((output[0] - anchor[0]).abs().max().detach()), 0.0)
        self.assertTrue(torch.equal(output[1], anchor[1]))


if __name__ == "__main__":
    unittest.main()
