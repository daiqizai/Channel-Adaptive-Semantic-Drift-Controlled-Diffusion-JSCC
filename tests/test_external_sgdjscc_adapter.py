from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "external_sgdjscc_native_smoke.py"
SPEC = importlib.util.spec_from_file_location("external_sgdjscc", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ExternalSgdJsccAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = yaml.safe_load(
            (ROOT / "configs" / "external_sgdjscc_native_smoke.yaml").read_text(
                encoding="utf-8"
            )
        )

    def test_frozen_smoke_contract_is_valid(self) -> None:
        MODULE.validate_config(self.payload)

    def test_outcome_claims_fail_closed(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["outcome_claims_allowed"] = True
        with self.assertRaises(MODULE.SmokeContractError):
            MODULE.validate_config(payload)

    def test_native_main_shape_is_4096_real_scalars(self) -> None:
        self.assertEqual(MODULE.native_main_real_symbols([16, 16, 16]), 4096)

    def test_unknown_text_transport_blocks_direct_ranking(self) -> None:
        summary = MODULE.make_rate_summary(
            main_real_symbols=4096,
            edge_dense_real_symbols=16384,
            edge_active_real_symbols=832,
            text_utf8_bits=160,
        )
        self.assertFalse(summary["common_contract_direct_ranking_allowed"])
        self.assertIsNone(summary["text_channel_symbols"])

    def test_native_rate_reports_real_ratio_and_complex_cbr_separately(self) -> None:
        summary = MODULE.make_rate_summary(
            main_real_symbols=4096,
            edge_dense_real_symbols=16384,
            edge_active_real_symbols=832,
            text_utf8_bits=488,
        )
        self.assertAlmostEqual(summary["main_real_dimension_ratio"], 1.0 / 12.0)
        self.assertAlmostEqual(summary["main_complex_cbr"], 1.0 / 24.0)
        self.assertAlmostEqual(
            summary["main_plus_edge_active_complex_cbr"],
            4928 / 2 / (3 * 128 * 128),
        )


if __name__ == "__main__":
    unittest.main()
