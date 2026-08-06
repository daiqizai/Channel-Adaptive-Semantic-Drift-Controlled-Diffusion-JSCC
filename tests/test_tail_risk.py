from __future__ import annotations

import math
import unittest

import torch

from cadsd_jscc.external_common import complex_awgn_from_standard_normal
from cadsd_jscc.tail_risk import (
    apply_block_fading_channel,
    block_fading_coefficient,
    conditional_cvar_objective,
    effective_snr_db,
    empirical_lower_tail_mean,
    empirical_upper_cvar,
    fading_seed,
    tail_count,
)


class EmpiricalCvarTest(unittest.TestCase):
    def test_constant_values_give_back_the_constant(self) -> None:
        values = torch.full((7,), 3.25)
        for fraction in (0.1, 0.2, 0.5, 1.0):
            self.assertAlmostEqual(
                float(empirical_upper_cvar(values, fraction)), 3.25, places=6
            )

    def test_documented_small_cases(self) -> None:
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(float(empirical_upper_cvar(values, 0.2)), 5.0, places=6)
        self.assertAlmostEqual(
            float(empirical_upper_cvar(values, 0.4)), (5.0 + 4.0) / 2.0, places=6
        )
        self.assertAlmostEqual(
            float(empirical_upper_cvar(values, 1.0)), 3.0, places=6
        )

    def test_tail_count_rounds_up_and_clamps(self) -> None:
        self.assertEqual(tail_count(5, 0.2), 1)
        self.assertEqual(tail_count(5, 0.4), 2)
        self.assertEqual(tail_count(4, 0.1), 1)
        self.assertEqual(tail_count(64, 0.1), 7)
        self.assertEqual(tail_count(64, 0.2), 13)

    def test_illegal_tail_fraction_raises(self) -> None:
        values = torch.tensor([1.0, 2.0])
        for fraction in (0.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                empirical_upper_cvar(values, fraction)

    def test_empty_input_raises(self) -> None:
        with self.assertRaises(ValueError):
            empirical_upper_cvar(torch.empty(0), 0.1)

    def test_dim_zero_and_one_are_both_supported(self) -> None:
        matrix = torch.tensor([[1.0, 10.0], [2.0, 20.0]])
        rows = empirical_upper_cvar(matrix, 0.5, dim=1)
        self.assertTrue(torch.allclose(rows, torch.tensor([10.0, 20.0])))
        columns = empirical_upper_cvar(matrix, 0.5, dim=0)
        self.assertTrue(torch.allclose(columns, torch.tensor([2.0, 20.0])))

    def test_lower_tail_mean_tracks_worst_quality(self) -> None:
        psnr = torch.tensor([30.0, 29.0, 28.0, 12.0, 31.0])
        self.assertAlmostEqual(
            float(empirical_lower_tail_mean(psnr, 0.2)), 12.0, places=6
        )
        self.assertAlmostEqual(
            float(empirical_lower_tail_mean(psnr, 0.4)), (12.0 + 28.0) / 2.0, places=6
        )


class ConditionalObjectiveTest(unittest.TestCase):
    def test_per_image_cvar_is_not_a_global_pool(self) -> None:
        # Image 0 is uniformly hard, image 1 is easy but has one channel outlier.
        distortion = torch.tensor([[9.0, 9.0, 9.0, 9.0], [1.0, 1.0, 1.0, 8.0]])
        total, stats = conditional_cvar_objective(distortion, 0.25, 1.0)
        # Per-image worst-25% is 9.0 and 8.0; a global worst-25% pool would have
        # returned only image 0's values and hidden image 1's channel outlier.
        self.assertAlmostEqual(float(total), (9.0 + 8.0) / 2.0, places=6)
        self.assertAlmostEqual(float(stats["loss_mean"]), float(distortion.mean()), 6)

    def test_risk_weight_interpolates(self) -> None:
        distortion = torch.tensor([[1.0, 2.0, 3.0, 100.0]])
        mean = float(distortion.mean())
        cvar = float(empirical_upper_cvar(distortion, 0.25, dim=1).mean())
        for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
            total, _ = conditional_cvar_objective(distortion, 0.25, weight)
            self.assertAlmostEqual(
                float(total), (1.0 - weight) * mean + weight * cvar, places=4
            )

    def test_shape_and_weight_validation(self) -> None:
        with self.assertRaises(ValueError):
            conditional_cvar_objective(torch.ones(4), 0.1, 0.5)
        with self.assertRaises(ValueError):
            conditional_cvar_objective(torch.ones(2, 4), 0.1, 1.5)

    def test_gradient_flows_only_through_tail(self) -> None:
        distortion = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
        total, _ = conditional_cvar_objective(distortion, 0.25, 1.0)
        total.backward()
        self.assertIsNotNone(distortion.grad)
        grad = distortion.grad.flatten().tolist()
        self.assertAlmostEqual(grad[3], 1.0, places=6)
        self.assertEqual(grad[:3], [0.0, 0.0, 0.0])


class BlockFadingChannelTest(unittest.TestCase):
    def test_fading_seed_is_stable_and_condition_specific(self) -> None:
        first = fading_seed(20260731, "image-a", 1.0, 0)
        self.assertEqual(first, fading_seed(20260731, "image-a", 1.0, 0))
        self.assertNotEqual(first, fading_seed(20260731, "image-b", 1.0, 0))
        self.assertNotEqual(first, fading_seed(20260731, "image-a", 4.0, 0))
        self.assertNotEqual(first, fading_seed(20260731, "image-a", 1.0, 1))

    def test_coefficient_is_replayable_and_unit_power(self) -> None:
        first = block_fading_coefficient(20260731, "image-a", 1.0, 3)
        self.assertEqual(first, block_fading_coefficient(20260731, "image-a", 1.0, 3))
        powers = [
            sum(value * value for value in block_fading_coefficient(7, "s", 1.0, index))
            for index in range(20000)
        ]
        self.assertAlmostEqual(sum(powers) / len(powers), 1.0, delta=0.05)

    def test_unit_gain_reduces_to_the_existing_awgn_path(self) -> None:
        torch.manual_seed(0)
        transmitted = torch.randn(3, 2, 4, 8)
        noise = torch.randn(transmitted.numel())
        reference = complex_awgn_from_standard_normal(transmitted, noise, 4.0)
        faded = apply_block_fading_channel(
            transmitted,
            noise,
            4.0,
            torch.ones(3),
            torch.zeros(3),
        )
        self.assertTrue(torch.allclose(reference, faded, atol=1e-5))

    def test_zero_noise_equalization_is_exact(self) -> None:
        torch.manual_seed(1)
        transmitted = torch.randn(2, 1, 4, 8)
        h_real = torch.tensor([0.31, -1.4])
        h_imag = torch.tensor([-0.82, 0.27])
        equalized = apply_block_fading_channel(
            transmitted,
            torch.zeros(transmitted.numel()),
            10.0,
            h_real,
            h_imag,
        )
        self.assertTrue(torch.allclose(transmitted, equalized, atol=1e-5))

    def test_equalized_noise_variance_matches_effective_snr(self) -> None:
        torch.manual_seed(2)
        symbols = 1 << 14
        transmitted = torch.randn(1, symbols)
        noise = torch.randn(symbols)
        h_real = torch.tensor([0.4])
        h_imag = torch.tensor([0.3])
        h_power = 0.4**2 + 0.3**2
        snr_db = 7.0
        equalized = apply_block_fading_channel(
            transmitted, noise, snr_db, h_real, h_imag
        )
        residual = (equalized - transmitted).flatten()
        power = float(transmitted.square().mean())
        gamma_effective = 10.0 ** (effective_snr_db(snr_db, h_power) / 10.0)
        expected = power / (2.0 * gamma_effective)
        self.assertAlmostEqual(
            float(residual.square().mean()) / expected, 1.0, delta=0.05
        )

    def test_effective_snr_is_nominal_snr_plus_fading_power_in_db(self) -> None:
        self.assertAlmostEqual(effective_snr_db(7.0, 1.0), 7.0, places=9)
        self.assertAlmostEqual(effective_snr_db(7.0, 0.1), 7.0 - 10.0, places=9)
        self.assertAlmostEqual(
            effective_snr_db(7.0, 0.5), 7.0 - 10.0 * math.log10(2.0), places=9
        )
        with self.assertRaises(ValueError):
            effective_snr_db(7.0, 0.0)

    def test_per_sample_snr_tensor_matches_scalar_calls(self) -> None:
        torch.manual_seed(3)
        transmitted = torch.randn(3, 2, 4, 8)
        noise = torch.randn(transmitted.numel())
        h_real = torch.tensor([0.6, -1.1, 0.2])
        h_imag = torch.tensor([-0.4, 0.5, 1.3])
        snrs = [1.0, 7.0, 19.0]
        batched = apply_block_fading_channel(
            transmitted, noise, torch.tensor(snrs), h_real, h_imag
        )
        per_sample_noise = noise.reshape(3, -1)
        for index, snr in enumerate(snrs):
            single = apply_block_fading_channel(
                transmitted[index : index + 1],
                per_sample_noise[index],
                snr,
                h_real[index : index + 1],
                h_imag[index : index + 1],
            )
            self.assertTrue(torch.allclose(single[0], batched[index], atol=1e-5))

    def test_per_sample_snr_shape_validation(self) -> None:
        transmitted = torch.randn(2, 8)
        with self.assertRaises(ValueError):
            apply_block_fading_channel(
                transmitted,
                torch.randn(16),
                torch.tensor([1.0, 2.0, 3.0]),
                torch.ones(2),
                torch.zeros(2),
            )

    def test_shape_validation(self) -> None:
        transmitted = torch.randn(2, 8)
        with self.assertRaises(ValueError):
            apply_block_fading_channel(
                transmitted, torch.randn(7), 1.0, torch.ones(2), torch.zeros(2)
            )
        with self.assertRaises(ValueError):
            apply_block_fading_channel(
                transmitted, torch.randn(16), 1.0, torch.ones(3), torch.zeros(3)
            )
        with self.assertRaises(ValueError):
            apply_block_fading_channel(
                torch.randn(2, 7), torch.randn(14), 1.0, torch.ones(2), torch.zeros(2)
            )


if __name__ == "__main__":
    unittest.main()
