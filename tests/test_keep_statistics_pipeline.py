"""Formula (5) pipeline integration using a real tiny CPU network and frozen K."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
    from safetensors.torch import load_file
except ImportError:
    torch = None

from tests import test_reference_keys as reference_tests
from MDLM.contracts import KEEP_INPUTS_VERSION, KeepStatisticsConfig, file_sha256
from model.adapters.llada_8b_instruct import LLaDA8BInstructAdapter
from util import extract_keep_statistics as keep_tools


@unittest.skipIf(torch is None, "PyTorch and safetensors are required for C pipeline tests")
class KeepStatisticsPipelineTests(unittest.TestCase):
    def setUp(self):
        # 组合已有夹具，复用真实两层网络与 P1 产物，不重复继承和发现 K 的测试。
        fixture = reference_tests.ReferenceKeyTests()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        fixture.extract()
        self.model, self.adapter, self.config = fixture.model, fixture.adapter, fixture.config
        self.p1_dir, self.key_dir = fixture.p1_dir, fixture.output_dir
        self.output_dir = self.p1_dir.parent / "matrices/keep_statistics"
        self.keep_file = self.p1_dir.parent / "keep_inputs.json"
        self.settings = KeepStatisticsConfig(recipe_id="tiny_cpu_keep_v1", chunk_size=2)
        names = self.adapter.target_module_names()
        rows = []
        for index, (ids, masks, positions, first_weights, second_weights) in enumerate((
            ((1, 15, 4, 15), (1, 3), (2, 1), [2.0, 0.0], [0.25, 3.0]),
            ((3, 2, 15, 4), (2,), (3, 0), [0.5, 1.0], [1.0, 2.0]),
        )):
            rows.append({
                "input_id": f"constructed_keep_{index}",
                "source": {"source_dataset": "tiny_cpu_example_only", "source_id": str(index)},
                "input": {"input_ids": list(ids), "attention_mask": [1] * len(ids),
                          "mask_positions": list(masks), "implicit_position_ids": list(range(len(ids)))},
                "capture_positions": list(positions),
                "weights_by_layer": {names[0]: first_weights, names[1]: second_weights},
            })
        self.payload = {
            "artifact_version": KEEP_INPUTS_VERSION, "purpose": "keep_statistics", "example_only": True,
            "recipe_id": self.settings.recipe_id, "mask_policy_id": "explicit_cpu_contexts_v1",
            "position_policy_id": "explicit_position_order_v1", "weight_policy_id": "explicit_per_layer_v1",
            "tokenizer_revision": self.config.tokenizer_revision, "mask_token_id": self.adapter.mask_token_id,
            "inputs": rows,
        }
        self.write_payload(self.payload)

    def write_payload(self, payload):
        self.keep_file.write_text(json.dumps(payload), encoding="utf-8")

    def extract(self, *, reuse_across_rounds=False, batch_keep=False):
        with patch.object(LLaDA8BInstructAdapter, "from_local_tokenizer", return_value=self.adapter), \
                patch.object(LLaDA8BInstructAdapter, "load_model", return_value=self.model), \
                patch.object(LLaDA8BInstructAdapter, "prepare_edit_input", side_effect=AssertionError("no retokenizing edits")), \
                patch.object(LLaDA8BInstructAdapter, "sample_denoise", side_effect=AssertionError("no sampling")):
            return keep_tools.extract_run(self.p1_dir, self.keep_file, self.settings,
                                          reuse_across_rounds=reuse_across_rounds, batch_keep=batch_keep)

    def test_equal_length_batch_replays_and_validates_same_formula(self):
        result = self.extract(batch_keep=True)
        self.assertEqual(result["forward_count"], 5)
        for name, layer in self.read_artifact("manifest.json")["layers"].items():
            matrix = load_file(str(self.output_dir / layer["tensor_file"]))[name]
            expected = torch.zeros_like(matrix, dtype=torch.float64)
            layer_index = int(name.split(".")[3])
            for row in self.payload["inputs"]:
                hidden = self.model.embedding.weight[row["input"]["input_ids"]].detach()
                for block in self.model.model.transformer.blocks[:layer_index]:
                    hidden = block.ff_out(hidden).detach()
                for position, weight in zip(row["capture_positions"], row["weights_by_layer"][name]):
                    expected += weight * torch.outer(hidden[position].double(), hidden[position].double())
            torch.testing.assert_close(matrix, expected.float(), atol=0, rtol=0)
        from util.validate_matrix_artifacts import validate_run
        with patch.object(LLaDA8BInstructAdapter, "from_local_tokenizer", return_value=self.adapter), \
             patch.object(LLaDA8BInstructAdapter, "load_model", return_value=self.model):
            self.assertEqual(validate_run(self.p1_dir, batch_keep=True)["status"], "complete")
        report = json.loads((self.p1_dir.parent / "matrices/validation/qa_report.json").read_text())
        self.assertEqual(report["real_model_forward_count"], 4)

    def read_artifact(self, name):
        return json.loads((self.output_dir / name).read_text(encoding="utf-8"))

    def assert_failed(self, phase):
        manifest = self.read_artifact("manifest.json")
        report = self.read_artifact("qa_report.json")
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["phase"], phase)
        self.assertFalse(manifest["p2_stage_complete"])
        self.assertTrue(report["failure"])

    def test_saved_C_matches_independent_weighted_outer_products_for_each_layer(self):
        original = {name: parameter.detach().clone() for name, parameter in self.model.named_parameters()}
        result = self.extract()
        self.assertEqual(result["forward_count"], 9)  # 基准重放一次，每层两输入各采集/复提一次。
        self.assertTrue(result["example_only"])
        self.assertFalse(Path(self.config.case_file).exists())
        manifest, report = self.read_artifact("manifest.json"), self.read_artifact("qa_report.json")
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["normalization"], "none")
        self.assertFalse(manifest["centered"])
        self.assertFalse(manifest["p2_stage_complete"])
        self.assertTrue(report["baseline_initial_logits_match_p1"])
        self.assertEqual(manifest["qa_report_sha256"], file_sha256(self.output_dir / "qa_report.json"))
        expected_weight_sums = (3.5, 6.25)
        for layer, name in enumerate(self.adapter.target_module_names()):
            vectors, weights = [], []
            for row in self.payload["inputs"]:
                hidden = self.model.embedding.weight[row["input"]["input_ids"]].detach()
                for block in self.model.model.transformer.blocks[:layer]:
                    hidden = block.ff_out(hidden).detach()
                vectors.extend(hidden[row["capture_positions"]].double().unbind())
                weights.extend(row["weights_by_layer"][name])
            expected = sum(weight * torch.outer(vector, vector) for vector, weight in zip(vectors, weights))
            layer_record = manifest["layers"][name]
            matrix_path = self.output_dir / layer_record["tensor_file"]
            matrix = load_file(str(matrix_path))[name]
            torch.testing.assert_close(matrix, expected.float(), atol=0, rtol=0)
            self.assertEqual(matrix.shape, (4 - layer, 4 - layer))
            self.assertEqual(layer_record["vector_count"], 4)
            self.assertEqual(layer_record["positive_weight_count"], 3 if layer == 0 else 4)
            self.assertEqual(layer_record["weight_sum"], expected_weight_sums[layer])
            self.assertEqual(layer_record["tensor_file_sha256"], file_sha256(matrix_path))
            diagnostics = report["layers"][name]["diagnostics"]
            self.assertTrue(diagnostics["all_checks_passed"])
            self.assertTrue(report["layers"][name]["disk_round_trip_passed"])
            self.assertTrue(all(row["capture_replay_max_abs"] == 0 for row in report["layers"][name]["inputs"]))
            # 用独立 D 直接核对公式（5）第二行，检出中心化或隐式平均。
            delta = torch.arange(1, matrix.shape[0] + 1, dtype=torch.float64)[None, :] / 3
            cost = torch.trace(delta @ matrix.double() @ delta.T)
            expected_cost = sum(weight * (delta @ vector).square().sum()
                                for vector, weight in zip(vectors, weights))
            torch.testing.assert_close(cost, expected_cost, atol=1e-6, rtol=1e-6)
        for name, parameter in self.model.named_parameters():
            self.assertTrue(torch.equal(parameter, original[name]))
        snapshot = file_sha256(self.output_dir / "manifest.json")
        with self.assertRaises(FileExistsError):
            self.extract()
        self.assertEqual(snapshot, file_sha256(self.output_dir / "manifest.json"))

    def test_independent_round_reuses_only_verified_C(self):
        with patch.object(keep_tools, "_reusable_c", None):
            first = self.extract(reuse_across_rounds=True)
            saved = self.output_dir.with_name("previous_keep_statistics")
            self.output_dir.rename(saved)
            keep_tools._reusable_c["directory"] = saved
            with patch.object(LLaDA8BInstructAdapter, "forward_keep_with_capture",
                              side_effect=AssertionError("C capture should be reused")):
                second = self.extract(reuse_across_rounds=True)
            self.assertEqual(first["forward_count"], 9)
            self.assertEqual(second["forward_count"], 1)
            self.assertEqual(self.read_artifact("manifest.json")["status"], "complete")
            from util.validate_matrix_artifacts import validate_run
            with patch.object(LLaDA8BInstructAdapter, "from_local_tokenizer", return_value=self.adapter), \
                 patch.object(LLaDA8BInstructAdapter, "load_model", return_value=self.model):
                self.assertEqual(validate_run(self.p1_dir)["status"], "complete")
            for layer in self.adapter.target_module_names():
                old = saved / self.read_artifact("manifest.json")["layers"][layer]["tensor_file"]
                new = self.output_dir / old.name
                self.assertEqual(file_sha256(old), file_sha256(new))
                self.assertEqual(self.read_artifact("qa_report.json")["layers"][layer]["reused_from"], str(old))

    def test_corrupted_cached_C_is_recomputed(self):
        with patch.object(keep_tools, "_reusable_c", None):
            self.extract(reuse_across_rounds=True)
            saved = self.output_dir.with_name("previous_keep_statistics")
            self.output_dir.rename(saved)
            keep_tools._reusable_c["directory"] = saved
            damaged = saved / "c_layer_0.safetensors"
            damaged.write_bytes(damaged.read_bytes() + b"changed")
            original = LLaDA8BInstructAdapter.forward_keep_with_capture
            calls = []

            def capture(adapter, model, model_input, positions):
                calls.append(1)
                return original(adapter, model, model_input, positions)

            with patch.object(LLaDA8BInstructAdapter, "forward_keep_with_capture", new=capture):
                result = self.extract(reuse_across_rounds=True)
            self.assertEqual(result["forward_count"], 9)
            self.assertEqual(len(calls), 8)
            self.assertFalse(self.read_artifact("manifest.json")["reused_c"])

    def test_recipe_purpose_tokenizer_and_layers_are_bound_before_model_loading(self):
        variants = (
            ("recipe", lambda payload: payload.update(recipe_id="another_recipe"), "keep recipe differs"),
            ("purpose", lambda payload: payload.update(purpose="independent_test"), "keep_statistics input"),
            ("tokenizer", lambda payload: payload.update(tokenizer_revision="different_tokenizer"), "tokenizer or mask"),
            ("layers", lambda payload: payload["inputs"][0]["weights_by_layer"].pop(
                self.adapter.target_module_names()[0]), "weight layers differ"),
        )
        for label, mutate, error in variants:
            with self.subTest(field=label):
                payload = copy.deepcopy(self.payload)
                mutate(payload)
                self.write_payload(payload)
                with patch.object(LLaDA8BInstructAdapter, "from_local_tokenizer") as load:
                    with self.assertRaisesRegex(ValueError, error):
                        keep_tools.extract_run(self.p1_dir, self.keep_file, self.settings)
                    load.assert_not_called()
                self.assert_failed("source_artifacts")
                self.output_dir.rename(self.output_dir.with_name("failed_" + label))

    def test_negative_keep_weight_cannot_publish_C(self):
        self.payload["inputs"][0]["weights_by_layer"][self.adapter.target_module_names()[0]][0] = -1
        self.write_payload(self.payload)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            self.extract()
        self.assert_failed("accumulation")
        self.assertFalse(list(self.output_dir.glob("c_layer_*.safetensors")))

    def test_changed_K_tensor_is_rejected_before_model_loading(self):
        tensor_path = self.key_dir / "keys.safetensors"
        tensor_path.write_bytes(tensor_path.read_bytes() + b"changed frozen K")
        with patch.object(LLaDA8BInstructAdapter, "from_local_tokenizer") as load:
            with self.assertRaisesRegex(ValueError, "keys.safetensors digest differs"):
                keep_tools.extract_run(self.p1_dir, self.keep_file, self.settings)
            load.assert_not_called()
        self.assert_failed("source_artifacts")

    def test_changed_baseline_is_rejected_before_loading(self):
        (Path(self.config.model_path) / "model.safetensors").write_bytes(b"different local baseline")
        with patch.object(LLaDA8BInstructAdapter, "from_local_tokenizer") as load:
            with self.assertRaisesRegex(ValueError, "baseline differs from P1/K"):
                keep_tools.extract_run(self.p1_dir, self.keep_file, self.settings)
            load.assert_not_called()
        self.assert_failed("baseline_identity")

    def test_baseline_logits_must_match_frozen_p1(self):
        forward = LLaDA8BInstructAdapter.forward_with_capture

        def different_logits(adapter, model, model_input):
            result = forward(adapter, model, model_input)
            result.mask_logits.add_(1)
            return result

        with patch.object(LLaDA8BInstructAdapter, "forward_with_capture", new=different_logits):
            with self.assertRaisesRegex(ValueError, "baseline logits differ"):
                self.extract()
        self.assert_failed("baseline_replay")
        self.assertFalse(list(self.output_dir.glob("c_layer_*.safetensors")))

    def test_storage_overflow_cannot_publish_C(self):
        forward = LLaDA8BInstructAdapter.forward_keep_with_capture

        def large_inputs(adapter, model, model_input, positions):
            result = forward(adapter, model, model_input, positions)
            for name, values in result.module_inputs.items():
                result.module_inputs[name] = torch.full_like(values, 1e30)
            return result

        with patch.object(LLaDA8BInstructAdapter, "forward_keep_with_capture", new=large_inputs):
            with self.assertRaisesRegex(ValueError, "numeric checks failed"):
                self.extract()
        self.assert_failed("layer_numeric_QA")
        self.assertFalse(list(self.output_dir.glob("c_layer_*.safetensors")))

    def test_keep_inputs_changed_during_capture_cannot_publish_complete(self):
        forward = LLaDA8BInstructAdapter.forward_keep_with_capture
        changed = False

        def capture(adapter, model, model_input, positions):
            nonlocal changed
            result = forward(adapter, model, model_input, positions)
            if not changed:
                self.keep_file.write_text(self.keep_file.read_text(encoding="utf-8") + "\n", encoding="utf-8")
                changed = True
            return result

        with patch.object(LLaDA8BInstructAdapter, "forward_keep_with_capture", new=capture):
            with self.assertRaisesRegex(ValueError, "artifacts changed during C extraction"):
                self.extract()
        self.assert_failed("source_recheck")

    def test_failed_module_input_replay_cannot_publish_C(self):
        forward = LLaDA8BInstructAdapter.forward_keep_with_capture
        call_count = 0

        def capture(adapter, model, model_input, positions):
            nonlocal call_count
            result = forward(adapter, model, model_input, positions)
            call_count += 1
            if call_count == 2:
                name = adapter.target_module_names()[0]
                result.module_inputs[name] = result.module_inputs[name] + 1
            return result

        with patch.object(LLaDA8BInstructAdapter, "forward_keep_with_capture", new=capture):
            with self.assertRaisesRegex(ValueError, "keep module inputs did not replay"):
                self.extract()
        self.assert_failed("accumulation")
        self.assertFalse(list(self.output_dir.glob("c_layer_*.safetensors")))

    def test_save_failure_keeps_failed_manifest_and_report(self):
        with patch("safetensors.torch.save_file", side_effect=OSError("artificial C save failure")):
            with self.assertRaisesRegex(OSError, "artificial C save failure"):
                self.extract()
        self.assert_failed("save_and_reload")
        self.assertFalse(list(self.output_dir.glob("c_layer_*.safetensors")))

    def test_saved_dtype_must_match_frozen_storage_precision(self):
        from safetensors.torch import save_file

        def wrong_dtype(tensors, path, **kwargs):
            save_file({name: value.double() for name, value in tensors.items()}, path, **kwargs)

        with patch("safetensors.torch.save_file", side_effect=wrong_dtype):
            with self.assertRaisesRegex(ValueError, "saved C dtype differs"):
                self.extract()
        self.assert_failed("save_and_reload")

    def test_saved_values_must_round_trip_exactly(self):
        from safetensors.torch import save_file

        def changed_values(tensors, path, **kwargs):
            save_file({name: value + 1 for name, value in tensors.items()}, path, **kwargs)

        with patch("safetensors.torch.save_file", side_effect=changed_values):
            with self.assertRaisesRegex(ValueError, "saved C did not round-trip"):
                self.extract()
        self.assert_failed("save_and_reload")


if __name__ == "__main__":
    unittest.main()
