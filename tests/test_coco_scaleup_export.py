from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from s13_export_coco_train2017_c8_scaleup import (  # noqa: E402
    balanced_payload,
    derived_seed,
    select_paths,
    validate_rate,
)


class ScaleupExportTests(unittest.TestCase):
    def test_hash_ranked_split_is_deterministic_and_disjoint(self) -> None:
        root = Path("/tmp/source")
        paths = [root / f"{index:03d}.jpg" for index in range(20)]
        first = select_paths(paths, root, 17, 8, 4)
        second = select_paths(list(reversed(paths)), root, 17, 8, 4)
        self.assertEqual(first, second)
        self.assertEqual(len(first[0]), 8)
        self.assertEqual(len(first[1]), 4)
        self.assertFalse(set(first[0]) & set(first[1]))

    def test_channel_seed_changes_by_snr_and_batch(self) -> None:
        seeds = {
            derived_seed(7, 1.0, 0),
            derived_seed(7, 4.0, 0),
            derived_seed(7, 1.0, 32),
        }
        self.assertEqual(len(seeds), 3)

    def test_exact_mask_rate_uses_active_complex_symbols(self) -> None:
        validate_rate(
            {
                "rate": {
                    "exact_mask": True,
                    "dense_real_symbols": 24576,
                    "active_real_symbols": 19712,
                    "source_real_dimensions": 196608,
                    "total_complex_uses": 9856,
                    "cbr": 0.050130208333333336,
                }
            }
        )

    def test_exact_mask_rate_rejects_inconsistent_complex_ledger(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_rate(
                {
                    "rate": {
                        "exact_mask": True,
                        "dense_real_symbols": 24576,
                        "active_real_symbols": 19712,
                        "source_real_dimensions": 196608,
                        "total_complex_uses": 9855,
                        "cbr": 0.050130208333333336,
                    }
                }
            )

    def test_balanced_payload_is_deterministic_and_unit_power(self) -> None:
        payload = balanced_payload(3, 80, device=torch.device("cpu"), dtype=torch.float32)
        self.assertEqual(tuple(payload.shape), (3, 80))
        self.assertTrue(torch.equal(payload[0], payload[1]))
        self.assertTrue(torch.equal(payload[0, 0::2], torch.ones(40)))
        self.assertTrue(torch.equal(payload[0, 1::2], -torch.ones(40)))
        self.assertAlmostEqual(float(payload.square().mean()), 1.0)


if __name__ == "__main__":
    unittest.main()
