from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_external_baseline_contract.py"
SPEC = importlib.util.spec_from_file_location("external_contract", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ExternalBaselineContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = yaml.safe_load(
            (ROOT / "configs" / "external_baseline_comparison_contract.yaml").read_text(
                encoding="utf-8"
            )
        )

    def test_frozen_contract_passes_without_path_checks(self) -> None:
        result = MODULE.validate_contract(self.payload, check_paths=False)
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["outcome_claims_allowed"])

    def test_native_track_cannot_directly_rank(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["comparison_tracks"]["author_native"][
            "direct_ranking_against_ours_allowed"
        ] = True
        with self.assertRaises(MODULE.ContractError):
            MODULE.validate_contract(payload, check_paths=False)

    def test_author_native_cannot_be_relabelled_as_directly_allowed(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["methods"][0]["direct_ranking_status"] = "allowed"
        with self.assertRaises(MODULE.ContractError):
            MODULE.validate_contract(payload, check_paths=False)

    def test_common_adapter_requires_exact_real_symbol_budget(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["methods"][0]["common_adapter"]["total_real_symbols"] -= 1
        with self.assertRaises(MODULE.ContractError):
            MODULE.validate_contract(payload, check_paths=False)

    def test_semantic_new_error_metric_is_mandatory(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["metrics"]["semantic_reliability"].remove("T_cls_new_error")
        with self.assertRaises(MODULE.ContractError):
            MODULE.validate_contract(payload, check_paths=False)


if __name__ == "__main__":
    unittest.main()
