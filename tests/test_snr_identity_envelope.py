from __future__ import annotations

import unittest

import torch

from cadsd_jscc.snr_identity_envelope import (
    apply_correction_envelope,
    envelope_strength,
    select_envelope_policy,
)


class SNRIdentityEnvelopeTest(unittest.TestCase):
    def test_smooth_strength_is_bounded_and_monotonic(self) -> None:
        spec = {"kind": "smooth_power", "exponent": 0.5}
        values = [envelope_strength(snr, spec) for snr in (1, 4, 7, 13, 19)]
        self.assertAlmostEqual(values[0], 1.0)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in values))
        self.assertTrue(all(left >= right for left, right in zip(values, values[1:])))

    def test_hard_cutoff_has_exact_identity_tail(self) -> None:
        spec = {
            "kind": "hard_cutoff",
            "full_through_snr_db": 7,
            "identity_from_snr_db": 13,
        }
        self.assertEqual(envelope_strength(7, spec), 1.0)
        self.assertEqual(envelope_strength(13, spec), 0.0)
        self.assertEqual(envelope_strength(19, spec), 0.0)

    def test_zero_and_full_envelopes_recover_endpoints(self) -> None:
        received = torch.tensor([[1.0, 2.0]])
        diffusion = torch.tensor([[3.0, -2.0]])
        self.assertTrue(torch.equal(apply_correction_envelope(received, diffusion, 0), received))
        self.assertTrue(torch.equal(apply_correction_envelope(received, diffusion, 1), diffusion))

    def test_policy_selector_prioritizes_nonnegative_snr_count(self) -> None:
        candidates = [
            {
                "name": "higher_mean_but_tail_loss",
                "mean_psnr_delta_vs_b0": 0.3,
                "mean_lpips_delta_vs_b0": -0.1,
                "per_snr": [
                    {"psnr_delta_vs_b0": value} for value in (1.0, 0.5, 0.1, -0.01, 0.0)
                ],
            },
            {
                "name": "identity_safe",
                "mean_psnr_delta_vs_b0": 0.2,
                "mean_lpips_delta_vs_b0": -0.05,
                "per_snr": [
                    {"psnr_delta_vs_b0": value} for value in (0.8, 0.2, 0.01, 0.0, 0.0)
                ],
            },
        ]
        selected = select_envelope_policy(candidates, nonnegative_tolerance_db=1e-9)
        self.assertEqual(selected["name"], "identity_safe")
        self.assertEqual(selected["nonnegative_snr_count"], 5)


if __name__ == "__main__":
    unittest.main()
