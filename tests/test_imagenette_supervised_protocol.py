from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import s6_imagenette_supervised_clean_eval as evaluator  # noqa: E402
import s6_train_imagenette_scratch_classifiers as trainer  # noqa: E402
import pc_imagenette_supervised_audit as posterior_audit  # noqa: E402
import pc_fit_receiver_risk_controller as risk_controller  # noqa: E402
import pc_export_receiver_risk_failure_cases as failure_cases  # noqa: E402
import pc_imagenette_sender_inbudget_awgn_audit as sender_audit  # noqa: E402
import pc_imagenette_sender_official_val_final as official_sender_audit  # noqa: E402


class ProtocolHashTests(unittest.TestCase):
    def test_final_lock_is_excluded_but_method_is_hashed(self) -> None:
        config = {
            "method": {"name": "scratch_supervised_clean", "version": 1},
            "evaluation": {"primary_snrs": [1.0, 4.0, 7.0]},
            "final_lock": {"unlocked": False, "protocol_sha256": None},
        }
        populated_lock = copy.deepcopy(config)
        populated_lock["final_lock"] = {
            "unlocked": True,
            "locked_at_utc": "2026-07-10T00:00:00Z",
            "protocol_sha256": "f" * 64,
        }
        changed_method = copy.deepcopy(config)
        changed_method["method"]["version"] = 2

        for hash_function in (trainer.protocol_sha256, evaluator.protocol_sha256):
            with self.subTest(hash_function=hash_function.__module__):
                self.assertEqual(hash_function(config), hash_function(populated_lock))
                self.assertNotEqual(hash_function(config), hash_function(changed_method))

    def test_official_sender_protocol_hash_excludes_only_final_lock(self) -> None:
        config = {
            "analysis_id": "x",
            "expected": {"rows": 10},
            "final_lock": {"unlocked": False},
        }
        populated = copy.deepcopy(config)
        populated["final_lock"] = {"unlocked": True, "protocol_sha256": "f" * 64}
        changed = copy.deepcopy(config)
        changed["expected"]["rows"] = 11
        self.assertEqual(
            official_sender_audit.protocol_sha256(config),
            official_sender_audit.protocol_sha256(populated),
        )
        self.assertNotEqual(
            official_sender_audit.protocol_sha256(config),
            official_sender_audit.protocol_sha256(changed),
        )

    def test_official_consumed_marker_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            marker = Path(temporary_directory) / "CONSUMED.json"
            config = {
                "analysis_id": "audit",
                "final_lock": {"protocol_sha256": "a" * 64},
            }
            official_sender_audit.create_marker(marker, config, {"script": "b" * 64})
            payload = json.loads(marker.read_text())
            self.assertEqual(
                payload["state"], "OFFICIAL_VAL_OUTCOME_CONSUMED_BEFORE_MODEL_INFERENCE"
            )
            with self.assertRaises(FileExistsError):
                official_sender_audit.create_marker(marker, config, {"script": "b" * 64})


class TrainerProtocolTests(unittest.TestCase):
    def test_largest_remainder_counts_are_exact_and_exhaustive(self) -> None:
        ratios = {"cls_train": 0.70, "cls_cal": 0.10, "policy_dev": 0.20}
        counts = trainer.largest_remainder_counts(11, ratios)
        self.assertEqual(counts, {"cls_train": 8, "cls_cal": 1, "policy_dev": 2})
        self.assertEqual(sum(counts.values()), 11)

    def test_eval_transform_uses_exact_deepjscc_geometry(self) -> None:
        image_size = 96
        _, eval_transform = trainer.build_transforms(
            {"data": {"image_size": image_size}, "training": {}}
        )
        steps = eval_transform.transforms
        self.assertIsInstance(steps[0], transforms.Resize)
        self.assertEqual(steps[0].size, image_size)
        self.assertIsInstance(steps[1], transforms.CenterCrop)
        self.assertEqual(steps[1].size, (image_size, image_size))

    def test_pretrained_weights_and_flag_are_rejected_explicitly(self) -> None:
        base = {
            "architecture": "resnet18",
            "weights": None,
            "pretrained": False,
        }
        with self.assertRaisesRegex(ValueError, "pretrained weights are forbidden"):
            trainer.build_scratch_model("T_cls", {**base, "weights": "DEFAULT"}, 10)
        with self.assertRaisesRegex(ValueError, "pretrained=True is forbidden"):
            trainer.build_scratch_model("T_cls", {**base, "pretrained": True}, 10)

    def test_auxiliary_efficientnet_is_randomly_initialized_ten_way_model(self) -> None:
        model = trainer.build_scratch_model(
            "G_aux",
            {"architecture": "efficientnet_b0", "weights": None, "pretrained": False},
            10,
        )
        self.assertEqual(model.classifier[-1].out_features, 10)

    @staticmethod
    def _history_row(
        epoch: int,
        macro_top1: float,
        loss: float,
        selected_best: bool,
    ) -> dict[str, object]:
        return {
            "role": "T_cls",
            "epoch": epoch,
            "lr": 0.01,
            "train_loss": 1.0,
            "train_top1": 0.5,
            "train_macro_top1": 0.5,
            "cls_cal_loss": loss,
            "cls_cal_top1": macro_top1,
            "cls_cal_macro_top1": macro_top1,
            "selected_best": selected_best,
            "epoch_seconds": 1.0,
        }

    def test_completed_history_reconstructs_best_epoch_with_tie_break(self) -> None:
        history = [
            self._history_row(1, 0.30, 1.0, True),
            self._history_row(2, 0.20, 0.8, False),
            self._history_row(3, 0.30, 0.7, True),
        ]
        best_epoch, best_macro, best_loss = trainer.validate_completed_history(
            "T_cls", history, expected_epochs=3
        )
        self.assertEqual(best_epoch, 3)
        self.assertAlmostEqual(best_macro, 0.30)
        self.assertAlmostEqual(best_loss, 0.7)

        invalid = copy.deepcopy(history)
        invalid[2]["selected_best"] = False
        with self.assertRaisesRegex(RuntimeError, "selected_best"):
            trainer.validate_completed_history("T_cls", invalid, expected_epochs=3)


class EvaluationPrimitiveTests(unittest.TestCase):
    def test_official_val_loader_requires_explicit_pinned_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            train_root = root / "train"
            val_root = root / "val"
            (train_root / "class0").mkdir(parents=True)
            (val_root / "class0").mkdir(parents=True)
            image_path = val_root / "class0" / "sample.JPEG"
            image_path.write_bytes(b"frozen-official-image-bytes")
            training_manifest_path = root / "training_manifest.json"
            training_manifest = {
                "official_val_accessed": False,
                "source_train_root": str(train_root),
                "classes": ["class0"],
                "samples": [],
            }
            training_manifest_path.write_text(json.dumps(training_manifest))
            official_manifest_path = root / "official_manifest.json"
            relative_path = "class0/sample.JPEG"
            official_manifest = {
                "role": "sealed_official_val_final_test",
                "source_val_root": str(val_root),
                "training_split_manifest_sha256": posterior_audit.sha256_file(
                    training_manifest_path
                ),
                "classes": ["class0"],
                "sample_count": 1,
                "samples": [
                    {
                        "sample_id": f"official_val/{relative_path}",
                        "relative_path": relative_path,
                        "wnid": "class0",
                        "class_idx": 0,
                        "split": "official_val",
                        "content_sha256": hashlib.sha256(
                            image_path.read_bytes()
                        ).hexdigest(),
                        "size_bytes": image_path.stat().st_size,
                    }
                ],
            }
            official_manifest_path.write_text(json.dumps(official_manifest))
            config = {
                "imagenette": {
                    "split_manifest": str(training_manifest_path),
                    "required_split": "official_val",
                    "official_val_accessed": True,
                    "official_val_manifest": str(official_manifest_path),
                    "official_val_manifest_sha256": posterior_audit.sha256_file(
                        official_manifest_path
                    ),
                    "official_val_expected_count": 1,
                    "verify_official_val_content_sha256": True,
                }
            }
            samples, classes = posterior_audit.load_imagenette_samples(config)
            self.assertEqual(classes, ["class0"])
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["path"], image_path.resolve())

            unauthorized = copy.deepcopy(config)
            unauthorized["imagenette"]["official_val_accessed"] = False
            with self.assertRaisesRegex(RuntimeError, "explicit"):
                posterior_audit.load_imagenette_samples(unauthorized)
            wrong_hash = copy.deepcopy(config)
            wrong_hash["imagenette"]["official_val_manifest_sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                posterior_audit.load_imagenette_samples(wrong_hash)

    def test_reference_bool_accepts_runtime_and_csv_booleans(self) -> None:
        self.assertTrue(sender_audit.reference_bool({"value": True}, "value"))
        self.assertFalse(sender_audit.reference_bool({"value": "false"}, "value"))

    def test_dual_evidence_acceptance_is_exact_intersection(self) -> None:
        sender_risk = torch.tensor([-0.1, -0.2, 0.3, -0.4])
        gate_anchor = torch.tensor(
            [[0.8, 0.2], [0.8, 0.2], [0.1, 0.9], [0.1, 0.9]]
        )
        gate_posterior = torch.tensor(
            [[0.7, 0.3], [0.2, 0.8], [0.2, 0.8], [0.8, 0.2]]
        )
        sender, guard, accepted = sender_audit.dual_evidence_acceptance(
            sender_risk, 0.0, gate_anchor, gate_posterior
        )
        self.assertTrue(torch.equal(sender, torch.tensor([True, True, False, True])))
        self.assertTrue(torch.equal(guard, torch.tensor([True, False, True, False])))
        self.assertTrue(torch.equal(accepted, torch.tensor([True, False, False, False])))

    def test_sender_only_acceptance_keeps_receiver_guard_open(self) -> None:
        risk = torch.tensor([-0.1, 0.2])
        sender, guard, accepted = sender_audit.dual_evidence_acceptance(risk, 0.0)
        self.assertTrue(torch.equal(sender, torch.tensor([True, False])))
        self.assertTrue(torch.equal(guard, torch.tensor([True, True])))
        self.assertTrue(torch.equal(accepted, sender))

    def test_cross_model_triplet_requires_all_three_top1_predictions(self) -> None:
        risk = torch.tensor([-0.1, -0.1, -0.1, 0.2])
        recovered = torch.tensor(
            [[0.8, 0.2], [0.8, 0.2], [0.8, 0.2], [0.8, 0.2]]
        )
        anchor = torch.tensor(
            [[0.7, 0.3], [0.2, 0.8], [0.7, 0.3], [0.7, 0.3]]
        )
        posterior = torch.tensor(
            [[0.6, 0.4], [0.1, 0.9], [0.2, 0.8], [0.6, 0.4]]
        )
        sender, receiver, cross, accepted = (
            sender_audit.cross_model_triplet_acceptance(
                risk, 0.0, recovered, anchor, posterior
            )
        )
        self.assertTrue(torch.equal(sender, torch.tensor([True, True, True, False])))
        self.assertTrue(torch.equal(receiver, torch.tensor([True, True, False, True])))
        self.assertTrue(torch.equal(cross, torch.tensor([True, False, True, True])))
        self.assertTrue(torch.equal(accepted, torch.tensor([True, False, False, False])))

    def test_failure_case_selection_is_frozen_to_two_event_types(self) -> None:
        base = {
            "clean_correct": True,
            "snr_db": 1.0,
            "anchor_correct": True,
            "posterior_correct": False,
            "rejected": False,
        }
        self.assertEqual(
            failure_cases.diagnostic_event_type(base, {1.0, 4.0, 7.0}),
            "missed_new_error",
        )
        self.assertEqual(
            failure_cases.diagnostic_event_type(
                {
                    **base,
                    "anchor_correct": False,
                    "posterior_correct": True,
                    "rejected": True,
                },
                {1.0, 4.0, 7.0},
            ),
            "rejected_posterior_repair",
        )
        self.assertIsNone(
            failure_cases.diagnostic_event_type(
                {**base, "snr_db": 13.0}, {1.0, 4.0, 7.0}
            )
        )

    def test_empirical_percentile_uses_frozen_right_side_contract(self) -> None:
        import numpy as np

        reference = np.asarray([1.0, 2.0, 2.0, 4.0])
        values = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
        actual = risk_controller.empirical_percentile(reference, values)
        expected = np.asarray([0.0, 0.25, 0.75, 0.75, 1.0])
        self.assertTrue(np.array_equal(actual, expected))

    def test_controller_selects_first_passing_candidate(self) -> None:
        candidates = [
            {"target_reference_reject_rate": 0.01, "all_gates_pass": False},
            {"target_reference_reject_rate": 0.05, "all_gates_pass": True},
            {"target_reference_reject_rate": 0.10, "all_gates_pass": True},
        ]
        self.assertIs(risk_controller.select_first_passing(candidates), candidates[1])

    def test_receiver_risk_schema_excludes_teacher_and_source_fields(self) -> None:
        features = posterior_audit.RECEIVER_RISK_FEATURE_COLUMNS
        self.assertEqual(len(features), len(set(features)))
        self.assertGreaterEqual(len(features), 40)
        for name in features:
            self.assertFalse(name.startswith("teacher_"))
            self.assertFalse(name.startswith("original_"))
            self.assertNotIn("label", name)
            self.assertNotIn("class_idx", name)

    def test_identical_candidate_has_zero_receiver_shift_features(self) -> None:
        images = torch.rand(2, 3, 8, 8)
        probabilities = torch.tensor(
            [[0.70, 0.20, 0.10], [0.10, 0.25, 0.65]], dtype=torch.float32
        )
        dc = torch.tensor([0.2, 0.3], dtype=torch.float32)
        features = posterior_audit.receiver_risk_feature_tensors(
            snr_db=7.0,
            dc_before=dc,
            dc_after=dc,
            anchor=images,
            raw=images,
            posterior=images,
            gate_anchor_probabilities=probabilities,
            gate_raw_probabilities=probabilities,
            gate_posterior_probabilities=probabilities,
            aux_anchor_probabilities=probabilities,
            aux_raw_probabilities=probabilities,
            aux_posterior_probabilities=probabilities,
        )
        self.assertEqual(tuple(features), posterior_audit.RECEIVER_RISK_FEATURE_COLUMNS)
        for values in features.values():
            self.assertTrue(torch.isfinite(values).all())
        for name in (
            "dc_delta",
            "anchor_raw_l1",
            "anchor_posterior_rmse",
            "raw_posterior_l1",
            "gate_anchor_posterior_js",
            "gate_raw_posterior_js",
            "gate_anchor_class_retention",
            "aux_anchor_posterior_js",
            "aux_raw_class_retention",
            "gate_anchor_posterior_top1_changed",
            "aux_raw_posterior_top1_changed",
        ):
            self.assertTrue(torch.equal(features[name], torch.zeros(2)))

    def test_source_semantic_zero_risk_is_exact_for_anchor_candidate(self) -> None:
        source = torch.tensor([[0.7, 0.2, 0.1]], dtype=torch.float64)
        anchor = torch.tensor([[0.6, 0.3, 0.1]], dtype=torch.float64)
        scores = posterior_audit.source_semantic_score_tensors(source, anchor, anchor)
        self.assertEqual(
            set(scores),
            {
                "fullprob_cross_entropy_risk",
                "fullprob_js_risk",
                "fullprob_cosine_risk",
                "source_top1_logprob_risk",
            },
        )
        for value in scores.values():
            self.assertTrue(torch.allclose(value, torch.zeros(1, dtype=value.dtype)))

    def test_png_quantization_is_idempotent_and_on_uint8_grid(self) -> None:
        images = torch.tensor(
            [[[[ -0.1, 0.0, 0.001, 0.5, 0.999, 1.1 ]]]],
            dtype=torch.float32,
        )
        quantized = evaluator.quantize_png_tensor(images, enabled=True)
        requantized = evaluator.quantize_png_tensor(quantized, enabled=True)

        self.assertTrue(torch.equal(quantized, requantized))
        self.assertGreaterEqual(float(quantized.min()), 0.0)
        self.assertLessEqual(float(quantized.max()), 1.0)
        grid_coordinates = quantized.double() * 255.0
        self.assertTrue(
            torch.allclose(grid_coordinates, grid_coordinates.round(), atol=1e-5, rtol=0.0)
        )

    def test_zero_event_exact_binomial_upper_bound_is_sane(self) -> None:
        self.assertIsNone(evaluator.exact_binomial_upper_95(0, 0))
        upper_100 = evaluator.exact_binomial_upper_95(0, 100)
        upper_1000 = evaluator.exact_binomial_upper_95(0, 1000)
        self.assertIsNotNone(upper_100)
        self.assertIsNotNone(upper_1000)
        assert upper_100 is not None and upper_1000 is not None
        self.assertGreater(upper_100, 0.0)
        self.assertLess(upper_100, 0.04)
        self.assertGreater(upper_1000, 0.0)
        self.assertLess(upper_1000, upper_100)
        exact_zero_event_value = 1.0 - math.pow(0.05, 1.0 / 100.0)
        self.assertAlmostEqual(upper_100, exact_zero_event_value, places=10)

    def test_clustered_conditional_bootstrap_retains_images_and_counts_events(self) -> None:
        rows = [
            {"image_id": "a", "snr_db": 1.0, "eligible": True, "event": True},
            {"image_id": "a", "snr_db": 4.0, "eligible": True, "event": False},
            {"image_id": "b", "snr_db": 1.0, "eligible": True, "event": True},
            {"image_id": "b", "snr_db": 4.0, "eligible": False, "event": False},
            {"image_id": "outside", "snr_db": 1.0, "eligible": True, "event": True},
            {"image_id": "a", "snr_db": 99.0, "eligible": True, "event": True},
        ]
        result = evaluator.bootstrap_clustered_conditional_rate(
            rows=rows,
            image_ids={"a", "b", "c"},
            snrs={1.0, 4.0},
            numerator_function=lambda row: bool(row["event"]),
            denominator_function=lambda row: bool(row["eligible"]),
            replicates=200,
            seed=17,
        )

        self.assertEqual(result["num_clusters"], 2)
        self.assertEqual(result["numerator_events"], 2)
        self.assertEqual(result["denominator_rows"], 3)
        self.assertAlmostEqual(result["estimate"], 2.0 / 3.0)
        self.assertGreater(result["valid_replicates"], 0)
        self.assertLessEqual(result["valid_replicates"], result["replicates"])
        self.assertEqual(result["cluster_unit"], "image_id")


if __name__ == "__main__":
    unittest.main()
