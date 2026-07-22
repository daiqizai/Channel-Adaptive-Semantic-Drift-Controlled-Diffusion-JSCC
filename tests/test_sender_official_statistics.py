from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import pc_imagenette_sender_inbudget_awgn_audit as sender  # noqa: E402


class FrozenCleanMembershipTests(unittest.TestCase):
    def setUp(self) -> None:
        self.membership = {
            "a": {
                "sample_id": "a",
                "wnid": "class_a",
                "class_idx": 0,
                "original_prediction": 0,
                "original_confidence": 0.75,
                "clean_correct": True,
            },
            "b": {
                "sample_id": "b",
                "wnid": "class_b",
                "class_idx": 1,
                "original_prediction": 0,
                "original_confidence": 0.60,
                "clean_correct": False,
            },
        }

    def rows(self) -> list[dict[str, object]]:
        return [
            {
                "sample_id": sample_id,
                "wnid": membership["wnid"],
                "class_idx": membership["class_idx"],
                "original_confidence": membership["original_confidence"],
                "clean_correct": membership["clean_correct"],
            }
            for _repeat in range(2)
            for sample_id, membership in self.membership.items()
        ]

    def test_repeated_rows_must_reuse_one_membership(self) -> None:
        sender.assert_clean_membership_consistency(self.rows(), self.membership, 2)

    def test_changed_clean_membership_fails_closed(self) -> None:
        rows = self.rows()
        rows[-1]["clean_correct"] = True
        with self.assertRaisesRegex(RuntimeError, "clean membership changed"):
            sender.assert_clean_membership_consistency(rows, self.membership, 2)

    def test_missing_repeated_row_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "repetition count mismatch"):
            sender.assert_clean_membership_consistency(self.rows()[:-1], self.membership, 2)


class ImageClusterEndpointTests(unittest.TestCase):
    def test_system_new_error_is_not_hidden_by_anchor_ineligibility(self) -> None:
        rows = [
            {
                "sample_id": "a",
                "anchor_correct": False,
                "reference_raw_correct": True,
                "final_correct": False,
            },
            {
                "sample_id": "b",
                "anchor_correct": True,
                "reference_raw_correct": True,
                "final_correct": True,
            },
        ]
        anchor = sender.image_cluster_any_event_endpoint(
            rows,
            lambda row: bool(row["anchor_correct"]),
            lambda row: bool(row["anchor_correct"]) and not bool(row["final_correct"]),
        )
        system = sender.image_cluster_any_event_endpoint(
            rows,
            lambda row: bool(row["reference_raw_correct"]),
            lambda row: bool(row["reference_raw_correct"])
            and not bool(row["final_correct"]),
        )
        self.assertEqual(anchor["event_image_clusters"], 0)
        self.assertEqual(system["event_rows"], 1)
        self.assertEqual(system["event_image_clusters"], 1)
        self.assertEqual(system["eligible_image_clusters"], 2)

    def test_event_outside_denominator_is_rejected(self) -> None:
        rows = [{"sample_id": "a", "eligible": False, "event": True}]
        with self.assertRaisesRegex(RuntimeError, "outside its denominator"):
            sender.image_cluster_any_event_endpoint(
                rows,
                lambda row: bool(row["eligible"]),
                lambda row: bool(row["event"]),
            )


class PairedClusterInferenceTests(unittest.TestCase):
    @staticmethod
    def rows() -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for sample_id in ("a", "b"):
            for channel_seed in (11, 12):
                for snr in (1.0, 4.0):
                    reference_correct = sample_id == "a"
                    final_correct = not (
                        sample_id == "a" and channel_seed == 11 and snr == 1.0
                    )
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "channel_seed": channel_seed,
                            "snr_db": snr,
                            "clean_correct": True,
                            "reference_raw_correct": reference_correct,
                            "final_correct": final_correct,
                            "reference_raw_psnr": 20.0,
                            "final_psnr": 20.1 if sample_id == "a" else 20.2,
                            "reference_raw_lpips": 0.20,
                            "final_lpips": 0.19,
                        }
                    )
        return rows

    def test_image_cluster_inference_retains_all_seed_snr_rows(self) -> None:
        result = sender.paired_image_cluster_inference(
            self.rows(),
            primary_snrs={1.0, 4.0},
            replicates=200,
            seed=17,
            all_sample_ids={"a", "b"},
            clean_sample_ids={"a", "b"},
            expected_all_rows_per_sample=4,
            expected_primary_rows_per_sample=4,
        )
        failure = result["primary_failure_rate_delta_final_minus_reference_raw"]
        psnr = result["all_snr_psnr_delta_final_minus_reference_raw"]
        lpips = result["all_snr_lpips_delta_final_minus_reference_raw"]
        self.assertAlmostEqual(float(failure["estimate"]), -0.375)
        self.assertAlmostEqual(float(psnr["estimate"]), 0.15)
        self.assertAlmostEqual(float(lpips["estimate"]), -0.01)
        self.assertEqual(int(failure["num_clusters"]), 2)
        self.assertEqual(set(result["primary_failure_rate_delta_by_snr"]), {"1.0", "4.0"})

    def test_missing_condition_row_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "row count mismatch"):
            sender.paired_image_cluster_inference(
                self.rows()[:-1],
                primary_snrs={1.0, 4.0},
                replicates=20,
                seed=17,
                all_sample_ids={"a", "b"},
                clean_sample_ids={"a", "b"},
                expected_all_rows_per_sample=4,
                expected_primary_rows_per_sample=4,
            )

    def test_constant_cluster_bootstrap_is_exact_and_deterministic(self) -> None:
        values = np.ones(5, dtype=np.float64) * 0.25
        first = sender.paired_cluster_bootstrap_mean_ci(values, 50, 3)
        second = sender.paired_cluster_bootstrap_mean_ci(values, 50, 3)
        self.assertEqual(first, second)
        self.assertEqual(first["estimate"], 0.25)
        self.assertEqual(first["ci95_lower"], 0.25)
        self.assertEqual(first["ci95_upper"], 0.25)


class FinalRoutingTests(unittest.TestCase):
    def test_default_anchor_fallback(self) -> None:
        accepted = torch.tensor([True, False])
        source_anchor = torch.tensor([False, False])
        posterior = torch.tensor([[[[10.0]]], [[[11.0]]]])
        anchor = torch.tensor([[[[20.0]]], [[[21.0]]]])
        raw = torch.tensor([[[[30.0]]], [[[31.0]]]])
        result = sender.route_final_candidate(
            accepted, source_anchor, posterior, anchor, raw, "anchor"
        )
        self.assertEqual(result.flatten().tolist(), [10.0, 21.0])

    def test_source_anchor_mismatch_routes_rejected_sample_to_raw(self) -> None:
        accepted = torch.tensor([False, False, True])
        source_anchor = torch.tensor([False, True, False])
        posterior = torch.tensor([[[[10.0]]], [[[11.0]]], [[[12.0]]]])
        anchor = torch.tensor([[[[20.0]]], [[[21.0]]], [[[22.0]]]])
        raw = torch.tensor([[[[30.0]]], [[[31.0]]], [[[32.0]]]])
        result = sender.route_final_candidate(
            accepted,
            source_anchor,
            posterior,
            anchor,
            raw,
            "raw_on_source_anchor_mismatch_else_anchor",
        )
        self.assertEqual(result.flatten().tolist(), [30.0, 21.0, 12.0])

    def test_unknown_fallback_fails_closed(self) -> None:
        value = torch.zeros((1, 1, 1, 1))
        with self.assertRaisesRegex(ValueError, "unsupported"):
            sender.route_final_candidate(
                torch.tensor([False]),
                torch.tensor([False]),
                value,
                value,
                value,
                "unknown",
            )


if __name__ == "__main__":
    unittest.main()
