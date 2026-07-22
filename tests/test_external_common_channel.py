from __future__ import annotations

import unittest

import torch

from cadsd_jscc.external_common import (
    canonical_noise_seed,
    canonical_standard_normal,
    complex_awgn_from_standard_normal,
    complex_cbr,
)


class ExternalCommonChannelTest(unittest.TestCase):
    def test_seed_is_stable_and_condition_specific(self) -> None:
        first = canonical_noise_seed(20260729, "image-a", 1.0)
        self.assertEqual(first, canonical_noise_seed(20260729, "image-a", 1.0))
        self.assertNotEqual(first, canonical_noise_seed(20260729, "image-b", 1.0))
        self.assertNotEqual(first, canonical_noise_seed(20260729, "image-a", 4.0))

    def test_noise_is_cpu_deterministic(self) -> None:
        first = canonical_standard_normal(20260729, "image-a", 1.0, 32)
        second = canonical_standard_normal(20260729, "image-a", 1.0, 32)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.device.type, "cpu")

    def test_complex_awgn_uses_half_variance_per_real_coordinate(self) -> None:
        transmitted = torch.ones(1, 1, 1, 4)
        z = torch.tensor([1.0, -1.0, 1.0, -1.0])
        received = complex_awgn_from_standard_normal(transmitted, z, 0.0)
        expected_sigma = 2.0**-0.5
        self.assertTrue(
            torch.allclose(
                received.flatten(),
                torch.tensor(
                    [
                        1.0 + expected_sigma,
                        1.0 - expected_sigma,
                        1.0 + expected_sigma,
                        1.0 - expected_sigma,
                    ]
                ),
            )
        )

    def test_cbr_converts_real_coordinates_to_complex_uses(self) -> None:
        self.assertAlmostEqual(complex_cbr(65536, 3 * 256 * 256), 1.0 / 6.0)


if __name__ == "__main__":
    unittest.main()
