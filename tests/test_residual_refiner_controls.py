from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from s5_residual_refiner_pilot import (  # noqa: E402
    build_model,
    condition_feature_names,
    condition_source_name,
)
from s6_audit_continuous_alpha_tail_refiner import normalize_policy_rows  # noqa: E402
from s6_residual_shrink_selection import choose_schedule  # noqa: E402


class StructuralConditionTests(unittest.TestCase):
    def test_edge_conditions_preserve_shape_and_range(self) -> None:
        config = {
            "model": {
                "input_channels": 6,
                "condition_features": ["sobel_magnitude", "laplacian_abs"],
                "base_channels": 8,
                "num_blocks": 1,
            }
        }
        model = build_model(config).eval()
        m0 = torch.rand(2, 3, 16, 16)
        with torch.no_grad():
            output = model(m0, torch.tensor([0.05, 0.95]), torch.tensor([0.12, 0.04]))
        self.assertEqual(output.shape, m0.shape)
        self.assertTrue(torch.isfinite(output).all())
        self.assertGreaterEqual(float(output.min()), 0.0)
        self.assertLessEqual(float(output.max()), 1.0)

    def test_edge_channels_are_the_only_parameter_difference(self) -> None:
        base = {
            "model": {
                "input_channels": 4,
                "condition_features": [],
                "base_channels": 8,
                "num_blocks": 1,
            }
        }
        edge = {
            "model": {
                "input_channels": 6,
                "condition_features": ["sobel_magnitude", "laplacian_abs"],
                "base_channels": 8,
                "num_blocks": 1,
            }
        }
        base_params = sum(parameter.numel() for parameter in build_model(base).parameters())
        edge_params = sum(parameter.numel() for parameter in build_model(edge).parameters())
        self.assertEqual(edge_params - base_params, 2 * 8 * 3 * 3)

    def test_unknown_condition_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported model.condition_features"):
            condition_feature_names({"model": {"condition_features": ["original_edge"]}})

    def test_sender_original_condition_requires_explicit_aligned_image(self) -> None:
        config = {
            "model": {
                "input_channels": 6,
                "condition_features": ["sobel_magnitude", "laplacian_abs"],
                "condition_source": "sender_original_oracle",
                "base_channels": 8,
                "num_blocks": 1,
            }
        }
        self.assertEqual(condition_source_name(config), "sender_original_oracle")
        model = build_model(config).eval()
        m0 = torch.rand(2, 3, 16, 16)
        original = torch.rand_like(m0)
        with self.assertRaisesRegex(ValueError, "requires condition_image"):
            model(m0, torch.tensor([0.05, 0.95]), torch.tensor([0.12, 0.04]))
        with torch.no_grad():
            output = model(
                m0,
                torch.tensor([0.05, 0.95]),
                torch.tensor([0.12, 0.04]),
                condition_image=original,
            )
        self.assertEqual(output.shape, m0.shape)

    def test_unknown_condition_source_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported model.condition_source"):
            condition_source_name(
                {
                    "model": {
                        "condition_features": ["sobel_magnitude"],
                        "condition_source": "magic",
                    }
                }
            )

    def test_decoded_structure_channels_are_consumed_directly(self) -> None:
        config = {
            "model": {
                "input_channels": 6,
                "condition_features": ["sobel_magnitude", "laplacian_abs"],
                "condition_source": "decoded_structure_rgb",
                "base_channels": 8,
                "num_blocks": 1,
            }
        }
        model = build_model(config).eval()
        m0 = torch.rand(2, 3, 16, 16)
        decoded = torch.rand_like(m0)
        with self.assertRaisesRegex(ValueError, "requires condition_image"):
            model(m0, torch.tensor([0.05, 0.95]), torch.tensor([0.12, 0.04]))
        captured: dict[str, torch.Tensor] = {}

        def hook(_module, inputs):
            captured["input"] = inputs[0].detach()

        handle = model.head[0].register_forward_pre_hook(hook)
        try:
            with torch.no_grad():
                model(
                    m0,
                    torch.tensor([0.05, 0.95]),
                    torch.tensor([0.12, 0.04]),
                    condition_image=decoded,
                )
        finally:
            handle.remove()
        self.assertTrue(torch.equal(captured["input"][:, 4:6], decoded[:, :2]))


class MonotonicScheduleTests(unittest.TestCase):
    @staticmethod
    def _rows() -> list[dict[str, float | str]]:
        rows: list[dict[str, float | str]] = []
        best_alpha = {1.0: 0.5, 4.0: 0.5, 7.0: 0.75}
        for snr in [1.0, 4.0, 7.0]:
            rows.append(
                {
                    "policy": "m0",
                    "snr_db": snr,
                    "alpha": "",
                    "final_failure_rate": 0.2,
                    "final_psnr_db": 30.0,
                }
            )
            for alpha in [0.5, 0.75, 1.0]:
                rows.append(
                    {
                        "policy": "top1_fallback_alpha",
                        "snr_db": snr,
                        "alpha": alpha,
                        "final_failure_rate": 0.2,
                        "final_psnr_db": 31.0 - abs(alpha - best_alpha[snr]),
                    }
                )
        return rows

    def test_global_selection_enforces_effective_strength_monotonicity(self) -> None:
        snrs = [1.0, 4.0, 7.0]
        gates = {1.0: 0.12, 4.0: 0.10, 7.0: 0.08}
        unconstrained, _ = choose_schedule(self._rows(), "top1_fallback_alpha", snrs)
        unconstrained_strengths = [gates[snr] * float(unconstrained[snr] or 0.0) for snr in snrs]
        self.assertFalse(
            all(left >= right for left, right in zip(unconstrained_strengths, unconstrained_strengths[1:]))
        )

        constrained, selected = choose_schedule(
            self._rows(),
            "top1_fallback_alpha",
            snrs,
            residual_gates=gates,
            enforce_effective_strength_nonincreasing=True,
        )
        constrained_strengths = [gates[snr] * float(constrained[snr] or 0.0) for snr in snrs]
        self.assertTrue(
            all(left >= right for left, right in zip(constrained_strengths, constrained_strengths[1:]))
        )
        self.assertTrue(
            all(
                row["selection_reason"]
                == "max_mean_psnr_under_m0_failure_and_monotonic_effective_strength"
                for row in selected
            )
        )

    def test_m0_fallback_keeps_constrained_search_feasible(self) -> None:
        rows = self._rows()
        rows = [
            row
            for row in rows
            if not (row["policy"] == "top1_fallback_alpha" and float(row["snr_db"]) == 7.0)
        ]
        choices, _ = choose_schedule(
            rows,
            "top1_fallback_alpha",
            [1.0, 4.0, 7.0],
            residual_gates={1.0: 0.12, 4.0: 0.10, 7.0: 0.08},
            enforce_effective_strength_nonincreasing=True,
        )
        self.assertIsNone(choices[7.0])


class GenericPolicyAuditTests(unittest.TestCase):
    def test_shrink_rows_are_adapted_by_configured_fields(self) -> None:
        source = {
            "__source_id": "fresh",
            "__source_split": "fresh-holdout",
            "policy": "validation_top1_shrink_schedule",
            "snr_db": "7",
            "sample": "sample.png",
            "alpha": "0.75",
            "original": "original.png",
            "m0_reconstruction": "m0.png",
            "candidate": "candidate.png",
            "final_source": "final.png",
            "accept_candidate": "true",
            "m0_matches_original_top1": "true",
            "candidate_matches_original_top1": "true",
            "candidate_matches_m0_top1": "true",
            "final_matches_original_top1": "true",
            "accepted_repair": "false",
            "accepted_new_error": "false",
            "rejected_good": "false",
            "original_top1_index": "1",
            "m0_top1_index": "1",
            "candidate_top1_index": "1",
        }
        config = {
            "policies": [
                {
                    "key": "edge_monotonic_top1_fallback",
                    "source_id": "fresh",
                    "source_policy": "validation_top1_shrink_schedule",
                    "candidate_path_field": "candidate",
                    "final_path_field": "final_source",
                    "accepted_field": "accept_candidate",
                    "alpha_field": "alpha",
                    "missed_repair_field": "rejected_good",
                }
            ]
        }
        rows = normalize_policy_rows([source], config)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["split"], "fresh-holdout")
        self.assertEqual(rows[0]["predicted_alpha"], "0.75")
        self.assertTrue(rows[0]["accepted"])
        self.assertEqual(rows[0]["candidate"], "candidate.png")
        self.assertEqual(rows[0]["final"], "final.png")


if __name__ == "__main__":
    unittest.main()
