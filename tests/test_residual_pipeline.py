"""P4 artifact/real-small-model integration; constructed fixtures are not experiment evidence."""

from __future__ import annotations

import copy
from dataclasses import replace
import io
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file

from MDLM.contracts import (KEEP_TRAINING_VERSION, RESIDUAL_CANDIDATE_VERSION,
                           ResidualOptimizationConfig, file_sha256)
from tests import test_writeback_pipeline as writeback_tests
from model.adapters.llada_8b_instruct import LLaDA8BInstructAdapter
from util import optimize_residuals as pipeline, smoke_p4_candidate


class ResidualPipelineTests(unittest.TestCase):
    def setUp(self):
        fixture = writeback_tests.WritebackPipelineTests()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        fixture.solve()
        self.fixture = fixture
        self.p1_dir, self.matrices = fixture.p1_dir, fixture.matrices
        self.model, self.adapter = fixture.fixture.fixture.model, fixture.fixture.fixture.adapter
        self.output = self.p1_dir.parent / "residuals"
        self.settings = ResidualOptimizationConfig({name: 0.001 for name in self.adapter.target_module_names()},
                    max_steps=3, checkpoints=(1, 3), learning_rate=0.03, allow_example_only=True)

    def run_p4(self, settings=None):
        with patch.object(LLaDA8BInstructAdapter, "from_local_tokenizer", return_value=self.adapter), \
                patch.object(LLaDA8BInstructAdapter, "load_model", return_value=self.model):
            return pipeline.optimize_run(self.p1_dir, settings or self.settings)

    def test_two_layer_real_small_model_optimizes_saves_and_preserves_baseline(self):
        before = copy.deepcopy(self.model.state_dict())
        result = self.run_p4()
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["example_only"])
        self.assertFalse(result["p4_stage_complete"])
        manifest = pipeline.read_json(self.output / "manifest.json")
        report = pipeline.read_json(self.output / "optimization_report.json")
        self.assertEqual(manifest["artifact_version"], RESIDUAL_CANDIDATE_VERSION)
        self.assertFalse(manifest["weights_merged"])
        self.assertTrue(report["baseline_unchanged"] and report["fixed_q_unchanged"] and report["zero_residual_replay_passed"])
        self.assertLess(report["optimization"]["selected_loss"]["total"], report["optimization"]["baseline"]["total"])
        saved = load_file(str(self.output / manifest["tensor_file"]))
        self.assertEqual(set(saved), set(self.adapter.target_module_names()))
        for name, value in saved.items():
            self.assertEqual(list(value.shape), manifest["layers"][name]["shape"])
        self.assertTrue(all(torch.equal(value, before[name]) for name, value in self.model.state_dict().items()))
        self.assertTrue(all(not self.model.get_submodule(name)._forward_hooks for name in saved))
        optimizer = torch.load(self.output / "optimizer_state.pt", weights_only=True)
        self.assertTrue(optimizer["state"])
        self.assertEqual(manifest["optimization_report_sha256"], file_sha256(self.output / "optimization_report.json"))
        with self.assertRaises(FileExistsError):
            self.run_p4()

    def test_example_only_permission_and_layer_selection_are_required(self):
        with self.assertRaisesRegex(ValueError, "allow_example_only"):
            self.run_p4(replace(self.settings, allow_example_only=False))
        self.assertEqual(pipeline.read_json(self.output / "manifest.json")["status"], "failed")

    def test_incomplete_q_fails_before_model_load(self):
        self.fixture.fixture.change_json(self.matrices / "writeback_factors/manifest.json", lambda row: row.update(status="failed"))
        with patch.object(LLaDA8BInstructAdapter, "load_model") as load:
            with self.assertRaisesRegex(ValueError, "complete supported P3"):
                pipeline.optimize_run(self.p1_dir, self.settings)
            load.assert_not_called()

    def test_q_layer_shape_dtype_and_source_are_bound(self):
        settings = replace(self.settings, rho_by_layer={"wrong": 1.0})
        with self.assertRaisesRegex(ValueError, "rho layer selection"):
            self.run_p4(settings)

    def test_changed_q_bytes_fail_before_model_load(self):
        manifest = pipeline.read_json(self.matrices / "writeback_factors/manifest.json")
        path = self.matrices / "writeback_factors" / next(iter(manifest["layers"].values()))["tensor_file"]
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "artifact digest differs"):
            self.run_p4()

    def test_old_q_is_rechecked_without_modifying_original_artifacts(self):
        path = self.matrices / "writeback_factors/manifest.json"
        self.fixture.fixture.change_json(path, lambda row: row.update(artifact_version="p3.writeback_factor.1"))
        before = file_sha256(path)
        self.run_p4()
        self.assertEqual(file_sha256(path), before)
        manifest = pipeline.read_json(self.output / "manifest.json")
        self.assertEqual(set(manifest["legacy_q_checks"]), set(self.settings.rho_by_layer))

    def test_source_change_during_training_prevents_complete_manifest(self):
        original = pipeline.optimize_residuals

        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            path = self.p1_dir / "coverage_report.json"
            path.write_bytes(path.read_bytes() + b"\n")
            return result

        with patch.object(pipeline, "optimize_residuals", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "artifact digest differs"):
                self.run_p4()
        self.assertFalse(pipeline.read_json(self.output / "manifest.json")["p4_stage_complete"])

    def test_saved_r_wrong_dtype_is_rejected(self):
        original = pipeline.load_file

        def wrong_dtype(path, **kwargs):
            values = original(path, **kwargs)
            return {name: value.double() for name, value in values.items()} if Path(path).name == "residuals.safetensors" else values

        with patch.object(pipeline, "load_file", side_effect=wrong_dtype):
            with self.assertRaisesRegex(ValueError, "saved R did not round-trip"):
                self.run_p4()

    def test_changed_baseline_files_during_training_prevent_publication(self):
        original = pipeline.optimize_residuals
        config = pipeline.P1RunConfig.from_record(pipeline.read_json(self.p1_dir / "manifest.json")["config"])

        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            path = Path(config.model_path) / "config.json"
            path.write_bytes(path.read_bytes() + b"\n")
            return result

        with patch.object(pipeline, "optimize_residuals", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "artifact digest differs"):
                self.run_p4()

    def test_corrupt_saved_optimizer_state_is_rejected(self):
        original = torch.save

        def wrong_state(state, path):
            original({**state, "state": {}}, path)

        with patch.object(pipeline.torch, "save", side_effect=wrong_state):
            with self.assertRaisesRegex(ValueError, "optimizer state did not round-trip"):
                self.run_p4()

    def test_no_candidate_is_rejected_without_publishing_r(self):
        settings = replace(self.settings, max_steps=1, checkpoints=(1,), learning_rate=1e-300)
        result = self.run_p4(settings)
        self.assertEqual(result["status"], "no_candidate")
        self.assertEqual(pipeline.read_json(self.output / "manifest.json")["status"], "rejected")
        self.assertFalse((self.output / "residuals.safetensors").exists())

    def test_beta_zero_never_loads_keep_reference(self):
        with patch.object(pipeline, "read_json", wraps=pipeline.read_json) as read:
            self.run_p4()
        self.assertFalse(any("keep_train" in str(call.args[0]) for call in read.call_args_list))

    def make_keep_settings(self):
        config = pipeline.P1RunConfig.from_record(pipeline.read_json(self.p1_dir / "manifest.json")["config"])
        recipe_path = self.p1_dir.parent / "constructed_keep_training.json"
        pipeline.write_json(recipe_path, {"artifact_version": KEEP_TRAINING_VERSION, "purpose": "keep_train", "example_only": True,
              "inputs": [{"input": {"input_ids": [2, 15, 15], "attention_mask": [1, 1, 1], "mask_positions": [1, 2]},
                          "positions": [1, 2], "weights": [0.3, 0.7]}]})
        reference = copy.deepcopy(self.model)
        with torch.no_grad():
            reference.head.bias.add_(torch.arange(16) * 0.02)
        settings = replace(self.settings, beta=0.2, keep_train_file=str(recipe_path),
                           reference_model_path=config.model_path, reference_model_revision=config.model_revision,
                           reference_tokenizer_revision=config.tokenizer_revision)
        return settings, reference

    def test_optional_keep_uses_separate_original_reference_model_and_saves_cache(self):
        settings, reference = self.make_keep_settings()
        expected = torch.log_softmax(self.adapter.forward_logits(reference, pipeline.KeepModelInput(
                     (2, 15, 15), (1, 1, 1), (1, 2)), (1, 2)), -1).detach()
        with patch.object(LLaDA8BInstructAdapter, "from_local_tokenizer", return_value=self.adapter), \
                patch.object(LLaDA8BInstructAdapter, "load_model", side_effect=[reference, self.model]) as load:
            result = pipeline.optimize_run(self.p1_dir, settings)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(load.call_count, 2)
        cached = load_file(str(self.output / "theta0_keep_log_probs.safetensors"))["input_0"]
        torch.testing.assert_close(cached, expected)
        manifest = pipeline.read_json(self.output / "manifest.json")
        self.assertEqual(manifest["keep_reference"]["kl_direction"], "theta_0 || candidate")
        report = pipeline.read_json(self.output / "optimization_report.json")
        self.assertGreater(report["optimization"]["baseline"]["keep"], 0)

    def test_development_purpose_is_rejected_as_keep_training_contract(self):
        settings, _ = self.make_keep_settings()
        path = Path(settings.keep_train_file)
        self.fixture.fixture.change_json(path, lambda row: row.update(purpose="keep_dev"))
        with self.assertRaisesRegex(ValueError, "keep-training inputs"):
            self.run_p4(settings)

    def test_wrong_original_reference_identity_stops(self):
        settings, _ = self.make_keep_settings()
        with self.assertRaisesRegex(ValueError, "original theta_0"):
            self.run_p4(replace(settings, reference_model_revision="wrong"))

    def test_changed_original_model_files_are_rejected_even_with_unchanged_candidate_files(self):
        settings, reference = self.make_keep_settings()
        reference_path = self.p1_dir.parent / "original_reference_model"
        shutil.copytree(settings.reference_model_path, reference_path)
        settings = replace(settings, reference_model_path=str(reference_path))
        original = pipeline.optimize_residuals

        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            path = reference_path / "model.safetensors"
            path.write_bytes(path.read_bytes() + b"changed original reference")
            return result

        with patch.object(pipeline, "optimize_residuals", side_effect=changed), \
                patch.object(LLaDA8BInstructAdapter, "from_local_tokenizer", return_value=self.adapter), \
                patch.object(LLaDA8BInstructAdapter, "load_model", side_effect=[reference, self.model]):
            with self.assertRaisesRegex(ValueError, "artifact digest differs"):
                pipeline.optimize_run(self.p1_dir, settings)

    def test_reference_cache_change_during_training_prevents_publication(self):
        settings, reference = self.make_keep_settings()
        original = pipeline.optimize_residuals

        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            path = self.output / "theta0_keep_log_probs.safetensors"
            path.write_bytes(path.read_bytes() + b"changed cache")
            return result

        with patch.object(pipeline, "optimize_residuals", side_effect=changed), \
                patch.object(LLaDA8BInstructAdapter, "from_local_tokenizer", return_value=self.adapter), \
                patch.object(LLaDA8BInstructAdapter, "load_model", side_effect=[reference, self.model]):
            with self.assertRaisesRegex(ValueError, "artifact digest differs"):
                pipeline.optimize_run(self.p1_dir, settings)

    def test_changed_settings_during_training_prevent_publication(self):
        original = pipeline.optimize_residuals

        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            self.settings.rho_by_layer[next(iter(self.settings.rho_by_layer))] = 0.2
            return result

        with patch.object(pipeline, "optimize_residuals", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "settings changed during optimization"):
                self.run_p4()

    def test_alternate_output_preserves_previous_candidate(self):
        self.run_p4()
        before = file_sha256(self.output / "manifest.json")
        alternate = self.output.parent / "residuals_review"
        with patch.object(LLaDA8BInstructAdapter, "from_local_tokenizer", return_value=self.adapter), \
                patch.object(LLaDA8BInstructAdapter, "load_model", return_value=self.model):
            result = pipeline.optimize_run(self.p1_dir, self.settings, output_dir=alternate)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(file_sha256(self.output / "manifest.json"), before)
        self.assertTrue((alternate / "manifest.json").exists())

    def test_adapter_logits_preserve_input_and_cross_layer_autograd(self):
        from MDLM.candidate import CandidateModel
        from MDLM.states import edit_training_examples, load_supervision
        examples = edit_training_examples(load_supervision(self.p1_dir / "fixed_states.jsonl"))
        factors = {name: torch.randn(2, self.model.get_submodule(name).in_features)
                   for name in self.adapter.target_module_names()}
        example = examples[0]
        before = example.model_input
        with CandidateModel(self.model, factors) as candidate:
            logits = self.adapter.forward_logits(self.model, before, example.positions)
            self.assertTrue(logits.requires_grad)
            logits.square().sum().backward()
            self.assertTrue(all(branch.r.grad is not None and branch.r.grad.norm() > 0
                                for branch in candidate.branches.values()))
        self.assertEqual(before, example.model_input)

    def test_smoke_runs_and_protects_existing_artifacts(self):
        config = self.p1_dir.parent / "p4_config.json"
        pipeline.write_json(config, self.settings.as_record())
        arguments = ["smoke_p4_candidate", "--p1-dir", str(self.p1_dir), "--config", str(config)]
        with patch.object(sys, "argv", arguments), \
                patch.object(LLaDA8BInstructAdapter, "from_local_tokenizer", return_value=self.adapter), \
                patch.object(LLaDA8BInstructAdapter, "load_model", return_value=self.model), \
                patch("sys.stdout", new_callable=io.StringIO):
            smoke_p4_candidate.main()
        report = pipeline.read_json(self.p1_dir.parent / "p4_candidate_acceptance_report.json")
        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["p4_stage_complete"])
        with patch.object(sys, "argv", arguments), self.assertRaises(FileExistsError):
            smoke_p4_candidate.main()

    def test_smoke_failure_keeps_failed_report(self):
        arguments = ["smoke_p4_candidate", "--p1-dir", str(self.p1_dir)]
        with patch.object(sys, "argv", arguments), \
                patch.object(smoke_p4_candidate, "optimize_run", side_effect=RuntimeError("constructed failure")):
            with self.assertRaisesRegex(RuntimeError, "constructed failure"):
                smoke_p4_candidate.main()
        report = pipeline.read_json(self.p1_dir.parent / "p4_candidate_acceptance_report.json")
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["p4_stage_complete"])


if __name__ == "__main__":
    unittest.main()
