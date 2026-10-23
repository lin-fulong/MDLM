"""Formula (4) tests on a tiny CPU network and artificial P1 artifacts."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
    from safetensors.torch import load_file
except ImportError:
    torch = None

from MDLM.contracts import (
    P1_ARTIFACT_VERSION, FactSelection, P1RunConfig, ReferenceKeyConfig, file_sha256, local_model_identity,
)
from MDLM.states import FactTrajectory, load_trajectories, save_trajectories
from model.adapters.llada_8b_instruct import LLaDA8BInstructAdapter
from model.base import ForwardCapture, MaskedEditInput
from util.extract_reference_keys import extract_run, load_reference_inputs
from util.filter_compatible_states import filter_run


@unittest.skipIf(torch is None, "PyTorch and safetensors are required for reference-key tests")
class ReferenceKeyTests(unittest.TestCase):
    def setUp(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(block_group_size=1, mask_token_id=15, max_sequence_length=16)
                self.embedding = torch.nn.Embedding(16, 4)
                self.model = torch.nn.Module()
                self.model.transformer = torch.nn.Module()
                self.model.transformer.blocks = torch.nn.ModuleList()
                for width_in, width_out in ((4, 3), (3, 2)):
                    block = torch.nn.Module()
                    block.ff_out = torch.nn.Linear(width_in, width_out)
                    self.model.transformer.blocks.append(block)
                self.head = torch.nn.Linear(2, 16)

            def get_input_embeddings(self):
                return self.embedding

            def forward(self, input_ids, attention_mask, use_cache, return_dict):
                hidden = self.embedding(input_ids)
                for block in self.model.transformer.blocks:
                    hidden = block.ff_out(hidden)
                return SimpleNamespace(logits=self.head(hidden))

        torch.manual_seed(42)
        self.model = TinyModel().eval()
        self.adapter = LLaDA8BInstructAdapter(SimpleNamespace(all_special_ids=[15]), mask_token_id=15, target_layers=(0, 1))
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        model_path = root / "model"
        model_path.mkdir()
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            (model_path / name).write_text("{}", encoding="utf-8")
        (model_path / "model.safetensors").write_bytes(b"artificial model file identity, not real weights")
        identity = local_model_identity(model_path)
        self.config = P1RunConfig(
            run_id="example_only_keys", round_id=0, baseline_id=identity["model_revision"],
            model_name="LLaDA-8B-Instruct", model_path=str(model_path),
            model_revision=identity["model_revision"], tokenizer_revision=identity["tokenizer_revision"],
            case_file=str(root / "dataset_is_deliberately_absent.json"),
            facts=(FactSelection("fact_z", "edit_z"), FactSelection("fact_a", "edit_a")),
            target_layers=(0, 1), window_policy_id="short_answer_exact_tokens_v1",
            sampler_policy_id="confidence_no_remask_v1", seed=2026, steps=None,
            temperature=0.0, replay_count=2, output_root=str(root / "runs"),
        )
        self.p1_dir = self.config.p1_dir()
        self.p1_dir.mkdir(parents=True)
        self.output_dir = self.p1_dir.parent / "matrices/reference_keys"
        trajectories, completed = [], []
        for fact, subject in zip(self.config.facts, (3, 1)):
            initial = MaskedEditInput((4, subject, 15, 15), (1, 1, 1, 1), (2, 3), 1)
            seed = self.config.fact_seed(fact)
            trace = self.adapter.sample_denoise(self.model, initial, steps=2, temperature=0.0, seed=seed)
            trajectories.append(FactTrajectory(fact.case_id, fact.probe_id, (2, 3), (11, 12), seed, trace))
            completed.append({"case_id": fact.case_id, "probe_id": fact.probe_id, "purpose": "example_only",
                              "forward_logits_finite": True, "replayed": True, "logits_replayed": True})
        save_trajectories(self.p1_dir / "trajectories.jsonl", trajectories)
        (self.p1_dir / "collection_report.json").write_text(
            json.dumps({"completed": completed, "failure": None}), encoding="utf-8")
        manifest = {
            "artifact_version": P1_ARTIFACT_VERSION, "collection_status": "complete",
            "config": self.config.as_record(), "config_sha256": self.config.digest(),
            "model_config_sha256": file_sha256(model_path / "config.json"),
            "tokenizer_sha256": file_sha256(model_path / "tokenizer.json"),
            "mask_token_id": 15, "fact_order": [fact.case_id for fact in self.config.facts],
            "replay_fact_indices": [0, 1],
            "forward_conditions": {"attention_mask": "all_ones", "position_ids": "implicit_zero_based",
                                   "attention_bias": None, "use_cache": False, "model_mode": "eval"},
            "trajectories_sha256": file_sha256(self.p1_dir / "trajectories.jsonl"),
            "collection_report_sha256": file_sha256(self.p1_dir / "collection_report.json"),
        }
        (self.p1_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        filter_run(self.p1_dir)

    def extract(self, settings=None):
        with patch.object(LLaDA8BInstructAdapter, "from_local_tokenizer", return_value=self.adapter), \
                patch.object(LLaDA8BInstructAdapter, "load_model", return_value=self.model):
            return extract_run(self.p1_dir, settings)

    def test_columns_are_exact_module_inputs_in_frozen_fact_order(self):
        original_weights = {name: value.detach().clone() for name, value in self.model.named_parameters()}
        with patch.object(self.adapter, "prepare_edit_input", side_effect=AssertionError("must use saved input")), \
                patch.object(self.adapter, "sample_denoise", side_effect=AssertionError("must not resample")):
            result = self.extract()
        keys = load_file(str(self.output_dir / "keys.safetensors"))
        names = self.adapter.target_module_names()
        # 独立计算真实线性层的输入，核对捕获点、主体位置与按列方向。
        expected_first = self.model.embedding.weight[[3, 1]].detach()
        expected_second = self.model.model.transformer.blocks[0].ff_out(expected_first).detach()
        torch.testing.assert_close(keys[names[0]], expected_first.T, rtol=0, atol=0)
        torch.testing.assert_close(keys[names[1]], expected_second.T, rtol=0, atol=0)
        self.assertEqual([list(keys[name].shape) for name in names], [[4, 2], [3, 2]])
        self.assertEqual(result["forward_count"], 4)
        manifest = json.loads((self.output_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual([row["case_id"] for row in manifest["columns"]], ["fact_z", "fact_a"])
        self.assertEqual(manifest["tensor_file_sha256"], file_sha256(self.output_dir / "keys.safetensors"))
        self.assertFalse(manifest["p2_stage_complete"])
        self.assertEqual(manifest["example_only_fact_count"], 2)
        for name, value in self.model.named_parameters():
            self.assertTrue(torch.equal(value, original_weights[name]))
        with self.assertRaises(FileExistsError):
            self.extract()

    def test_changed_trajectory_is_rejected_before_loading_model(self):
        with (self.p1_dir / "trajectories.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with patch.object(LLaDA8BInstructAdapter, "from_local_tokenizer") as factory:
            with self.assertRaisesRegex(ValueError, "trajectories digest"):
                extract_run(self.p1_dir)
            factory.assert_not_called()
        self.assertFalse((self.output_dir / "keys.safetensors").exists())
        report = json.loads((self.output_dir / "qa_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")

    def test_mixed_fact_order_is_rejected(self):
        path = self.p1_dir / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["fact_order"].reverse()
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "fact/probe order"):
            load_reference_inputs(self.p1_dir)

    def test_changed_fixed_state_is_rejected(self):
        path = self.p1_dir / "fixed_states.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        records[0]["states"][0]["input"]["subject_last_position"] = 0
        path.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "saved supervision"):
            load_reference_inputs(self.p1_dir)

    def test_changed_local_baseline_is_rejected(self):
        (Path(self.config.model_path) / "model.safetensors").write_bytes(b"different local baseline")
        with self.assertRaisesRegex(ValueError, "local snapshot"):
            self.extract()
        self.assertFalse((self.output_dir / "keys.safetensors").exists())

    def test_coverage_counts_must_match_collection(self):
        path = self.p1_dir / "coverage_report.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        for field in ("replayed_fact_count", "logits_replayed_fact_count", "example_only_fact_count"):
            with self.subTest(field=field):
                coverage = {**original, field: original[field] + 1}
                path.write_text(json.dumps(coverage), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "coverage counts"):
                    load_reference_inputs(self.p1_dir)

    def test_changed_tokenizer_is_rejected_before_model_loading(self):
        path = Path(self.config.model_path) / "tokenizer_config.json"
        path.write_text('{"changed": true}', encoding="utf-8")
        with patch.object(LLaDA8BInstructAdapter, "from_local_tokenizer") as factory:
            with self.assertRaisesRegex(ValueError, "local snapshot"):
                extract_run(self.p1_dir)
            factory.assert_not_called()

    def test_reference_initial_state_must_be_fully_masked(self):
        path = self.p1_dir / "trajectories.jsonl"
        trajectories = load_trajectories(path)
        trajectory = trajectories[0]
        initial, *following = trajectory.trace.pre_forward_inputs
        trajectories[0] = replace(trajectory, trace=replace(
            trajectory.trace, pre_forward_inputs=(replace(initial, mask_positions=(2,)), *following)))
        save_trajectories(path, trajectories)
        # 更新来源摘要后仍必须被初态的算法约束拒绝。
        for filename in ("manifest.json", "coverage_report.json"):
            record_path = self.p1_dir / filename
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record["trajectories_sha256"] = file_sha256(path)
            record_path.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "full-mask initial state"):
            load_reference_inputs(self.p1_dir)

    def test_initial_logits_must_match_p1(self):
        original = self.adapter.forward_with_capture

        def changed(model, model_input):
            capture = original(model, model_input)
            return ForwardCapture(capture.mask_logits + 1, capture.module_inputs)

        with patch.object(self.adapter, "forward_with_capture", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "initial logits differ"):
                self.extract()
        self.assertFalse((self.output_dir / "keys.safetensors").exists())

    def test_reference_key_must_repeat_within_fixed_tolerance(self):
        original = self.adapter.forward_with_capture
        calls = 0

        def changed(model, model_input):
            nonlocal calls
            calls += 1
            capture = original(model, model_input)
            if calls == 2:
                name = self.adapter.target_module_names()[0]
                capture.module_inputs[name] = capture.module_inputs[name] + 1
            return capture

        with patch.object(self.adapter, "forward_with_capture", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "key did not replay"):
                self.extract()

    def test_nonfinite_key_is_rejected(self):
        original = self.adapter.forward_with_capture

        def changed(model, model_input):
            capture = original(model, model_input)
            name = self.adapter.target_module_names()[0]
            capture.module_inputs[name] = capture.module_inputs[name].clone()
            capture.module_inputs[name][0] = float("nan")
            return capture

        with patch.object(self.adapter, "forward_with_capture", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "finite values"):
                self.extract()

    def test_numeric_choices_are_explicit(self):
        settings = ReferenceKeyConfig()
        for bad in (replace(settings, storage_dtype="float16"), replace(settings, replay_atol=-1),
                    replace(settings, replay_rtol=float("nan"))):
            with self.assertRaises(ValueError):
                bad.check()
        self.extract(replace(settings, storage_dtype="float64"))
        keys = load_file(str(self.output_dir / "keys.safetensors"))
        self.assertTrue(all(value.dtype == torch.float64 for value in keys.values()))

    def assert_failed_publication(self, phase):
        manifest = json.loads((self.output_dir / "manifest.json").read_text(encoding="utf-8"))
        report = json.loads((self.output_dir / "qa_report.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["phase"], phase)
        self.assertTrue(report["failure"])

    def test_save_failure_does_not_publish_complete_artifacts(self):
        with patch("safetensors.torch.save_file", side_effect=OSError("artificial save failure")):
            with self.assertRaisesRegex(OSError, "save failure"):
                self.extract()
        self.assert_failed_publication("save_and_reload")

    def test_disk_round_trip_mismatch_does_not_publish_complete_artifacts(self):
        def changed(path, **kwargs):
            saved = load_file(path, **kwargs)
            name = self.adapter.target_module_names()[0]
            saved[name] = saved[name] + 1
            return saved

        with patch("safetensors.torch.load_file", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "round-trip"):
                self.extract()
        self.assert_failed_publication("save_and_reload")

    def test_p1_change_during_capture_prevents_publication(self):
        original = self.adapter.forward_with_capture

        def changed(model, model_input):
            captured = original(model, model_input)
            with (self.p1_dir / "coverage_report.json").open("a", encoding="utf-8") as stream:
                stream.write("\n")
            return captured

        with patch.object(self.adapter, "forward_with_capture", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "P1 artifacts changed"):
                self.extract()
        self.assert_failed_publication("save_and_reload")

    def test_storage_conversion_overflow_prevents_publication(self):
        original = self.adapter.forward_with_capture

        def changed(model, model_input):
            captured = original(model, model_input)
            return ForwardCapture(captured.mask_logits, {
                name: torch.full(value.shape, 1e40, dtype=torch.float64)
                for name, value in captured.module_inputs.items()
            })

        with patch.object(self.adapter, "forward_with_capture", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "invalid K matrix"):
                self.extract()
        self.assert_failed_publication("capture_and_replay")
        self.assertFalse((self.output_dir / "keys.safetensors").exists())


if __name__ == "__main__":
    unittest.main()
