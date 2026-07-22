from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "external_sgdjscc_common_smoke.py"
SPEC = importlib.util.spec_from_file_location("external_sgdjscc_common", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ExternalSgdJsccCommonAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = yaml.safe_load(
            (ROOT / "configs" / "external_sgdjscc_common_smoke.yaml").read_text(
                encoding="utf-8"
            )
        )

    def test_frozen_common_smoke_contract_is_valid(self) -> None:
        plan = MODULE.validate_config(self.payload)
        self.assertEqual(plan["total_real_symbols"], 65536)
        self.assertEqual(plan["total_complex_channel_uses"], 32768)
        self.assertAlmostEqual(
            plan["cbr_complex_channel_uses_per_source_real_dimension"], 1.0 / 6.0
        )

    def test_rate_plan_closes_all_four_branches(self) -> None:
        plan = MODULE.make_rate_plan(self.payload)
        self.assertEqual(plan["main_real_symbols"], 4 * 4096)
        self.assertEqual(plan["active_edge_real_symbols"], 4 * 832)
        self.assertEqual(plan["text_real_symbols"], 4 * 536 * 21)
        self.assertEqual(plan["no_information_padding_real_symbols"], 800)

    def test_caption_packet_round_trip_ascii_and_utf8(self) -> None:
        for caption in (
            "a baseball field with players",
            "一张有猫和树的图像",
        ):
            bits, sender = MODULE.encode_caption_packet(caption)
            receiver = MODULE.decode_caption_packet(bits)
            self.assertTrue(receiver["packet_ok"])
            self.assertEqual(receiver["decoded_text"], sender["transmitted_text"])
            self.assertEqual(len(bits), 536)

    def test_caption_truncation_keeps_valid_utf8(self) -> None:
        caption = "猫" * 30
        bits, sender = MODULE.encode_caption_packet(caption)
        receiver = MODULE.decode_caption_packet(bits)
        self.assertTrue(sender["truncated"])
        self.assertLessEqual(sender["transmitted_utf8_bytes"], 64)
        self.assertTrue(receiver["packet_ok"])
        self.assertEqual(receiver["decoded_text"], sender["transmitted_text"])

    def test_repetition_majority_corrects_ten_flips_per_bit(self) -> None:
        bits = [0, 1, 1, 0]
        coded = MODULE.repetition_encode(bits, 21)
        for group in range(len(bits)):
            start = group * 21
            for offset in range(10):
                coded[start + offset] ^= 1
        self.assertEqual(MODULE.repetition_majority_decode(coded, 21), bits)

    def test_crc_error_erases_caption(self) -> None:
        bits, _ = MODULE.encode_caption_packet("semantic side information")
        bits[20] ^= 1
        receiver = MODULE.decode_caption_packet(bits)
        self.assertFalse(receiver["packet_ok"])
        self.assertEqual(receiver["decoded_text"], "")
        self.assertEqual(receiver["failure"], "crc_mismatch")

    def test_rate_change_fails_closed(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["rate_contract"]["no_information_padding_real_symbols"] = 799
        with self.assertRaises(MODULE.CommonContractError):
            MODULE.validate_config(payload)

    def test_outcome_claims_fail_closed(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["outcome_claims_allowed"] = True
        with self.assertRaises(MODULE.CommonContractError):
            MODULE.validate_config(payload)

    def test_awgn_variance_convention_is_mandatory(self) -> None:
        payload = copy.deepcopy(self.payload)
        del payload["channel"]["noise_variance_convention"]
        with self.assertRaises(MODULE.CommonContractError):
            MODULE.validate_config(payload)


if __name__ == "__main__":
    unittest.main()
