from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cadsd_jscc.structure import structural_feature_maps, structure_rgb  # noqa: E402
from s5_residual_refiner_pilot import build_model  # noqa: E402
from s7_train_matched_rate_jscc import (  # noqa: E402
    select_latent_channels,
    validate_rate_contract,
)
from s7_imagenette_matched_rate_eval import select_sketch_alpha_indices  # noqa: E402


class StructureRepresentationTests(unittest.TestCase):
    def test_project_structure_matches_refiner_extractor(self) -> None:
        images = torch.rand(2, 3, 16, 16)
        refiner = build_model(
            {
                "model": {
                    "input_channels": 6,
                    "condition_features": ["sobel_magnitude", "laplacian_abs"],
                    "base_channels": 8,
                    "num_blocks": 1,
                }
            }
        )
        expected = torch.cat(refiner.structural_conditions(images), dim=1)
        actual = structural_feature_maps(images)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7, rtol=0.0))

    def test_structure_rgb_is_bounded_and_keeps_first_two_maps(self) -> None:
        images = torch.rand(2, 3, 16, 16)
        features = structural_feature_maps(images)
        packed = structure_rgb(images)
        self.assertEqual(packed.shape, images.shape)
        self.assertTrue(torch.equal(packed[:, :2], features))
        self.assertGreaterEqual(float(packed.min()), 0.0)
        self.assertLessEqual(float(packed.max()), 1.0)

    def test_zero_initialized_semantic_film_preserves_warm_start(self) -> None:
        base_config = {
            "model": {
                "input_channels": 6,
                "condition_features": ["sobel_magnitude", "laplacian_abs"],
                "condition_source": "decoded_structure_rgb",
                "base_channels": 8,
                "num_blocks": 1,
            }
        }
        semantic_config = {
            "model": {**base_config["model"], "semantic_sketch_dim": 32}
        }
        base = build_model(base_config)
        with torch.no_grad():
            base.tail.weight.normal_(0.0, 0.01)
            base.tail.bias.zero_()
        semantic = build_model(semantic_config)
        incompatible = semantic.load_state_dict(base.state_dict(), strict=False)
        self.assertTrue(all(key.startswith("semantic_modulation.") for key in incompatible.missing_keys))
        self.assertEqual(incompatible.unexpected_keys, [])
        m0 = torch.rand(2, 3, 16, 16)
        condition = torch.rand_like(m0)
        snr = torch.tensor([0.1, 0.4])
        gate = torch.tensor([0.12, 0.10])
        expected = base(m0, snr, gate, condition_image=condition)
        actual = semantic(
            m0,
            snr,
            gate,
            condition_image=condition,
            semantic_sketch=torch.randn(2, 32),
        )
        self.assertTrue(torch.equal(expected, actual))

    def test_semantic_refiner_requires_sketch(self) -> None:
        model = build_model(
            {
                "model": {
                    "input_channels": 4,
                    "condition_features": [],
                    "base_channels": 8,
                    "num_blocks": 1,
                    "semantic_sketch_dim": 32,
                }
            }
        )
        with self.assertRaisesRegex(ValueError, "requires semantic_sketch"):
            model(torch.rand(1, 3, 8, 8), torch.tensor([0.5]), torch.tensor([0.1]))


class RateContractTests(unittest.TestCase):
    def test_exact_six_plus_two_contract(self) -> None:
        config = {
            "protocol": {"cbr_denominator": 48},
            "rate": {
                "main_inner_channel": 6,
                "main_cbr": 0.125,
                "structure_inner_channel": 2,
                "structure_cbr": 1 / 24,
                "total_inner_channel": 8,
                "total_cbr": 1 / 6,
                "reference_inner_channel": 8,
                "reference_cbr": 1 / 6,
            },
        }
        result = validate_rate_contract(config)
        self.assertEqual(result["channels"], {"main": 6, "structure": 2, "total": 8})

    def test_budget_mismatch_is_rejected(self) -> None:
        config = {
            "protocol": {"cbr_denominator": 48},
            "rate": {
                "main_inner_channel": 6,
                "main_cbr": 0.125,
                "structure_inner_channel": 3,
                "structure_cbr": 3 / 48,
                "total_inner_channel": 8,
                "total_cbr": 1 / 6,
                "reference_inner_channel": 8,
                "reference_cbr": 1 / 6,
            },
        }
        with self.assertRaisesRegex(RuntimeError, "Invalid channel budget"):
            validate_rate_contract(config)


class SemanticControllerTests(unittest.TestCase):
    def test_selects_maximum_and_smaller_alpha_on_tie(self) -> None:
        scores = torch.tensor([[0.1, 0.8, 0.8], [0.9, 0.2, 0.3]])
        selected = select_sketch_alpha_indices(scores, [0.0, 0.5, 1.0])
        self.assertEqual(selected.tolist(), [1, 0])

    def test_rejects_unsorted_alpha_candidates(self) -> None:
        with self.assertRaisesRegex(ValueError, "sorted"):
            select_sketch_alpha_indices(torch.ones(1, 2), [1.0, 0.0])


class LatentSelectionTests(unittest.TestCase):
    def test_selection_uses_joint_encoder_decoder_importance(self) -> None:
        encoder = torch.zeros(6, 1, 1, 1)
        decoder = torch.zeros(6, 1, 1, 1)
        encoder[:, 0, 0, 0] = torch.tensor([1.0, 9.0, 4.0, 3.0, 2.0, 8.0])
        decoder[:, 0, 0, 0] = torch.tensor([9.0, 1.0, 4.0, 3.0, 2.0, 0.5])
        indices, _ = select_latent_channels(
            {
                "encoder.conv5.conv.weight": encoder,
                "decoder.tconv1.transconv.weight": decoder,
            },
            target_inner_channel=2,
        )
        self.assertEqual(indices, [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
