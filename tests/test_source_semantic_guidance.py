from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from s6_compare_source_edge_oracle import validate_matched_configs  # noqa: E402
from s6_imagenette_source_semantic_description_eval import (  # noqa: E402
    nested_assignments,
    quantize_description,
    semantic_scores,
)


class SourceDescriptionTests(unittest.TestCase):
    def test_uint8_description_is_decodable_probability_vector(self) -> None:
        probabilities = torch.tensor([[0.001, 0.099, 0.9], [0.2, 0.3, 0.5]])
        codes, decoded = quantize_description(probabilities)
        self.assertEqual(codes.dtype, torch.uint8)
        self.assertTrue(torch.allclose(decoded.sum(dim=1), torch.ones(2)))
        self.assertEqual(decoded.argmax(dim=1).tolist(), [2, 2])

    def test_all_continuous_risks_decrease_when_candidate_moves_toward_source(self) -> None:
        source = np.asarray([0.8, 0.2])
        m0 = np.asarray([0.4, 0.6])
        candidate = np.asarray([0.7, 0.3])
        scores = semantic_scores(source, m0, candidate)
        self.assertTrue(all(value < 0.0 for value in scores.values()))

    def test_nested_split_is_deterministic_and_class_balanced(self) -> None:
        records = [
            {"image_id": f"a/{index}", "wnid": "a"} for index in range(6)
        ] + [{"image_id": f"b/{index}", "wnid": "b"} for index in range(4)]
        config = {
            "nested_split": {
                "seed": 7,
                "selection_fraction": 0.5,
                "selection_name": "select",
                "audit_name": "audit",
                "method": "per_class_sha256_rank_half",
            }
        }
        first, metadata = nested_assignments(records, config)
        second, _ = nested_assignments(records, config)
        self.assertEqual(first, second)
        self.assertEqual(metadata["per_class_counts"]["a"], {"select": 3, "audit": 3})
        self.assertEqual(metadata["per_class_counts"]["b"], {"select": 2, "audit": 2})


class SourceEdgeMatchedContractTests(unittest.TestCase):
    @staticmethod
    def configs() -> tuple[dict, dict]:
        common = {
            "dataset": "COCO2017",
            "image_size": 256,
            "channel": "AWGN",
            "snrs": [1, 4],
            "cbr": 0.17,
            "seed": 42,
            "split": {"train": 1, "eval": 2},
            "model": {
                "input_channels": 6,
                "condition_features": ["sobel_magnitude", "laplacian_abs"],
                "base_channels": 8,
                "num_blocks": 1,
                "snr_norm_max": 20.0,
                "residual_gates": {"1": 0.1, "4": 0.08},
            },
            "training": {"epochs": 2},
        }
        receiver = {**common, "model": {**common["model"], "condition_source": "receiver_m0"}}
        source = {
            **common,
            "model": {**common["model"], "condition_source": "sender_original_oracle"},
        }
        return receiver, source

    def test_only_condition_source_may_differ(self) -> None:
        receiver, source = self.configs()
        result = validate_matched_configs(receiver, source)
        self.assertEqual(result["sole_intended_difference"], "model.condition_source")

    def test_capacity_mismatch_is_rejected(self) -> None:
        receiver, source = self.configs()
        source["model"]["base_channels"] = 16
        with self.assertRaisesRegex(RuntimeError, "not matched"):
            validate_matched_configs(receiver, source)


if __name__ == "__main__":
    unittest.main()
