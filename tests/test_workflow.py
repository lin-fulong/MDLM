"""P5 artifact integration on the existing real tiny CPU model; no formal data."""

import copy
from dataclasses import replace
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from MDLM import workflow
from MDLM.contracts import (AcceptanceConfig, DEVELOPMENT_KEEP_VERSION, FactSelection,
                           MERGED_MODEL_VERSION, file_sha256, local_model_identity)
from model.adapters.llada_8b_instruct import LLaDA8BInstructAdapter
from model.base import EditTarget, MaskedEditInput
from tests import test_residual_pipeline as residual_tests
from util import smoke_p5_acceptance


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        fixture = residual_tests.ResidualPipelineTests()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        fixture.run_p4()
        self.fixture, self.p1_dir = fixture, fixture.p1_dir
        self.model, self.adapter = fixture.model, fixture.adapter
        self.original_model = copy.deepcopy(self.model)
        self.residual_dir = self.p1_dir.parent / "residuals"
        self.output = self.p1_dir.parent / "editing"
        config = workflow.P1RunConfig.from_record(workflow.read_json(self.p1_dir / "manifest.json")["config"])
        self.dev = self.p1_dir.parent / "constructed_dev.json"
        workflow.write_json(self.dev, [{"case_id": case, "quality_control": {"status": "example_only"}}
                                       for case in ("fact_a", "fact_z")])
        self.keep = self.p1_dir.parent / "constructed_keep_dev.json"
        workflow.write_json(self.keep, {"artifact_version": DEVELOPMENT_KEEP_VERSION, "purpose": "keep_dev",
              "example_only": True, "tokenizer_revision": config.tokenizer_revision, "mask_token_id": 15,
              "inputs": [{"input": {"input_ids": [5, 15, 15], "attention_mask": [1, 1, 1], "mask_positions": [1, 2]},
                          "positions": [1, 2], "weights": [.5, .5]}]})
        self.settings = AcceptanceConfig(str(self.dev), (FactSelection("fact_a", "held_out"), FactSelection("fact_z", "held_out")),
                    str(self.keep), config.model_path, config.model_revision, config.tokenizer_revision,
                    delta=1e-6, tau_g=100, tau_u=100, equivalence_atol=1e-6, equivalence_rtol=1e-6,
                    chunk_size=2, allow_example_only=True)

    @staticmethod
    def prepare(case, probe_id):
        inp = MaskedEditInput((5, 15, 15), (1, 1, 1), (1, 2), 0)
        return inp, EditTarget(case["case_id"], probe_id, (11, 12))

    def run_p5(self, settings=None):
        with patch.object(LLaDA8BInstructAdapter, "from_local_tokenizer", return_value=self.adapter), \
                patch.object(LLaDA8BInstructAdapter, "load_model", return_value=self.model), \
                patch.object(self.adapter, "prepare_development_input", side_effect=self.prepare):
            return workflow.run_p5(self.p1_dir, settings or self.settings)

    def assert_recovered(self):
        self.assertTrue(all(torch.equal(value, self.original_model.state_dict()[name]) for name, value in self.model.state_dict().items()))
        self.assertTrue(all(not module._forward_hooks for module in self.model.modules()))

    def change_json(self, path, change):
        record = workflow.read_json(path)
        change(record)
        workflow.write_json(path, record)

    def test_actual_accepted_model_disk_roundtrip_and_reload(self):
        source_hash = file_sha256(self.residual_dir / "manifest.json")
        result = self.run_p5()
        self.assertEqual(result["decision"], "accepted")
        self.assertTrue(result["example_only"])
        self.assertFalse(result["p5_stage_complete"])
        manifest = workflow.read_json(self.output / "manifest.json")
        self.assertEqual(manifest["artifact_version"], MERGED_MODEL_VERSION)
        self.assertTrue(manifest["weights_merged"])
        self.assertFalse(manifest["temporary_branches_active"])
        self.assertEqual(file_sha256(self.residual_dir / "manifest.json"), source_hash)
        saved = load_file(str(self.output / manifest["tensor_file"]))
        for name, weight in saved.items():
            self.assertTrue(torch.equal(weight, self.model.get_submodule(name).weight))
        with patch.object(LLaDA8BInstructAdapter, "from_local_tokenizer", return_value=self.adapter), \
                patch.object(LLaDA8BInstructAdapter, "load_model", return_value=copy.deepcopy(self.original_model)):
            adapter, restored = workflow.load_final_model(self.output)
        self.assertTrue(all(torch.equal(value, restored.state_dict()[name]) for name, value in self.model.state_dict().items()))
        self.assertTrue(all(not module._forward_hooks for module in restored.modules()))
        with self.assertRaises(FileExistsError):
            self.run_p5()

    def test_original_case_schema_heldout_probe_aligns_without_case_mutation(self):
        from tests.test_p0_data import CharacterTokenizer, FIXTURE
        case = workflow.read_json(FIXTURE)
        before = copy.deepcopy(case)
        adapter = LLaDA8BInstructAdapter(CharacterTokenizer(), mask_token_id=126336, target_layers=(0,))
        inp, target = adapter.prepare_development_input(case, "birth_test")
        self.assertEqual(target.token_ids, tuple(map(ord, "海岚市")))
        self.assertTrue(all(inp.input_ids[position] == 126336 for position in inp.mask_positions))
        self.assertEqual(inp.input_ids[inp.subject_last_position], ord("舟"))
        self.assertEqual(case, before)

    def test_all_rejected_artifact_keeps_actual_baseline(self):
        result = self.run_p5(replace(self.settings, delta=1e6))
        self.assertEqual(result["decision"], "rejected")
        self.assertIsNone(result["selected_alpha"])
        report = workflow.read_json(self.output / "acceptance_report.json")
        self.assertTrue(report["baseline_restored"])
        self.assertFalse(workflow.read_json(self.output / "manifest.json")["weights_merged"])
        self.assert_recovered()
        _, loaded = self.reload_with(copy.deepcopy(self.original_model))
        self.assertTrue(all(torch.equal(value, self.original_model.state_dict()[name]) for name, value in loaded.state_dict().items()))

    def test_example_permission_required_before_model_loading(self):
        with patch.object(LLaDA8BInstructAdapter, "load_model") as load:
            with self.assertRaisesRegex(ValueError, "explicit permission"):
                workflow.run_p5(self.p1_dir, replace(self.settings, allow_example_only=False))
            load.assert_not_called()

    def test_incomplete_p4_is_rejected_before_loading(self):
        self.change_json(self.residual_dir / "manifest.json", lambda row: row.update(status="rejected"))
        with patch.object(LLaDA8BInstructAdapter, "load_model") as load:
            with self.assertRaisesRegex(ValueError, "complete audited P4"):
                workflow.run_p5(self.p1_dir, self.settings)
            load.assert_not_called()

    def test_mixed_round_identity_is_rejected(self):
        self.change_json(self.residual_dir / "manifest.json", lambda row: row.update(baseline_id="another"))
        with self.assertRaisesRegex(ValueError, "identity/order"):
            self.run_p5()

    def test_failed_p4_qa_cannot_be_rehashed_into_a_usable_candidate(self):
        path = self.residual_dir / "optimization_report.json"
        self.change_json(path, lambda row: row.update(baseline_unchanged=False))
        self.change_json(self.residual_dir / "manifest.json", lambda row: row.update(optimization_report_sha256=file_sha256(path)))
        with self.assertRaisesRegex(ValueError, "QA did not pass"):
            self.run_p5()

    def test_example_q_cannot_be_relabelled_as_formal_p4(self):
        self.change_json(self.residual_dir / "manifest.json", lambda row: row.update(example_only=False, p4_stage_complete=True))
        with self.assertRaisesRegex(ValueError, "relabel"):
            self.run_p5()

    def test_changed_r_digest_and_rehashed_wrong_shape_are_rejected(self):
        path = self.residual_dir / "residuals.safetensors"
        tensors = load_file(str(path))
        tensors[next(iter(tensors))] = torch.zeros(1, 1)
        save_file(tensors, str(path))
        with self.assertRaisesRegex(ValueError, "artifact digest"):
            workflow.load_residual_candidate(self.p1_dir, self.residual_dir, self.settings)
        self.change_json(self.residual_dir / "manifest.json", lambda row: row.update(tensor_file_sha256=file_sha256(path)))
        with self.assertRaisesRegex(ValueError, "R shape/dtype"):
            self.run_p5()

    def test_changed_q_binding_in_r_manifest_is_rejected(self):
        self.change_json(self.residual_dir / "manifest.json", lambda row: next(iter(row["layers"].values())).update(q_sha256="wrong"))
        with self.assertRaisesRegex(ValueError, "Q binding"):
            self.run_p5()

    def test_original_model_identity_is_explicitly_required(self):
        with self.assertRaisesRegex(ValueError, "original theta_0"):
            self.run_p5(replace(self.settings, reference_model_revision="wrong"))

    def test_preservation_readouts_must_be_mask_positions_and_normalized(self):
        self.change_json(self.keep, lambda row: row["inputs"][0].update(positions=[0], weights=[1]))
        with self.assertRaisesRegex(ValueError, "formula .16."):
            self.run_p5()
        self.assert_recovered()

    def test_preservation_nonunit_weight_sum_is_rejected(self):
        self.change_json(self.keep, lambda row: row["inputs"][0].update(weights=[1, 1]))
        with self.assertRaisesRegex(ValueError, "sum to one"):
            self.run_p5()

    def test_training_purpose_cannot_replace_development_contract(self):
        self.change_json(self.keep, lambda row: row.update(purpose="keep_train"))
        with self.assertRaisesRegex(ValueError, "development preservation"):
            self.run_p5()

    def test_distinct_original_reference_is_cached_then_baseline_loaded(self):
        path = self.p1_dir.parent / "theta0"
        shutil.copytree(self.settings.reference_model_path, path)
        (path / "config.json").write_text('{"constructed_original":true}')
        identity = local_model_identity(path)
        reference = copy.deepcopy(self.original_model)
        with torch.no_grad():
            reference.head.bias.add_(torch.arange(16) * .02)
        settings = replace(self.settings, reference_model_path=str(path), reference_model_revision=identity["model_revision"])
        with patch.object(LLaDA8BInstructAdapter, "from_local_tokenizer", return_value=self.adapter), \
                patch.object(LLaDA8BInstructAdapter, "load_model", side_effect=[reference, self.model]) as load, \
                patch.object(self.adapter, "prepare_development_input", side_effect=self.prepare):
            workflow.run_p5(self.p1_dir, settings)
        self.assertEqual(load.call_count, 2)
        cached = load_file(str(self.output / "theta0_keep_log_probs.safetensors"))["input_0"]
        with torch.no_grad():
            expected = workflow.log_probabilities(self.adapter.forward_logits(reference,
                MaskedEditInput((5, 15, 15), (1, 1, 1), (1, 2), 0), (1, 2)))
        torch.testing.assert_close(cached, expected)

    def test_source_change_after_acceptance_recovers_and_blocks_publication(self):
        original = workflow.select_merged_candidate

        def changed(*args):
            report = original(*args)
            self.dev.write_bytes(self.dev.read_bytes() + b"\n")
            return report

        with patch.object(workflow, "select_merged_candidate", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "artifact digest"):
                self.run_p5()
        self.assert_recovered()
        manifest = workflow.read_json(self.output / "manifest.json")
        self.assertEqual(manifest["status"], "failed")
        self.assertFalse(manifest["p5_stage_complete"] or manifest["weights_merged"])

    def test_saved_weights_wrong_dtype_recovers_and_blocks_publication(self):
        original = workflow.load_file

        def corrupt(path, **kwargs):
            tensors = original(path, **kwargs)
            return {name: value.double() for name, value in tensors.items()} if Path(path).name == "final_target_weights.safetensors" else tensors

        with patch.object(workflow, "load_file", side_effect=corrupt):
            with self.assertRaisesRegex(ValueError, "disk round-trip"):
                self.run_p5()
        self.assert_recovered()

    def test_save_failure_after_accepted_trial_recovers_baseline(self):
        original = workflow.save_file

        def fail(tensors, path, **kwargs):
            if Path(path).name == "final_target_weights.safetensors":
                raise OSError("constructed disk full")
            return original(tensors, path, **kwargs)

        with patch.object(workflow, "save_file", side_effect=fail):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.run_p5()
        self.assert_recovered()

    def test_config_change_during_trials_recovers_baseline(self):
        original = workflow.select_merged_candidate

        def changed(*args):
            report = original(*args)
            object.__setattr__(self.settings, "tau_g", 200)
            return report

        with patch.object(workflow, "select_merged_candidate", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "settings changed"):
                self.run_p5()
        self.assert_recovered()

    def test_corrupt_final_weights_prevent_model_reload(self):
        self.run_p5()
        path = self.output / "final_target_weights.safetensors"
        path.write_bytes(path.read_bytes() + b"corruption")
        with patch.object(LLaDA8BInstructAdapter, "load_model") as load:
            with self.assertRaisesRegex(ValueError, "artifact digest"):
                workflow.load_final_model(self.output)
            load.assert_not_called()

    def test_rehashed_bad_final_weights_do_not_modify_loaded_model(self):
        self.run_p5()
        path = self.output / "final_target_weights.safetensors"
        tensors = load_file(str(path))
        tensors[next(iter(tensors))].add_(1)
        save_file(tensors, str(path))
        self.change_json(self.output / "manifest.json", lambda row: row.update(tensor_file_sha256=file_sha256(path)))
        loaded = copy.deepcopy(self.original_model)
        with patch.object(LLaDA8BInstructAdapter, "from_local_tokenizer", return_value=self.adapter), \
                patch.object(LLaDA8BInstructAdapter, "load_model", return_value=loaded):
            with self.assertRaisesRegex(ValueError, "target weight contract"):
                workflow.load_final_model(self.output)
        self.assertTrue(all(torch.equal(value, self.original_model.state_dict()[name]) for name, value in loaded.state_dict().items()))

    def test_reloaded_logits_mismatch_recovers_loaded_baseline(self):
        self.run_p5()
        path = self.output / "final_dev_logits.safetensors"
        values = {name: value + 1 for name, value in load_file(str(path)).items()}
        save_file(values, str(path))
        self.change_json(self.output / "manifest.json", lambda row: row.update(final_logits_sha256=file_sha256(path)))
        loaded = copy.deepcopy(self.original_model)
        with patch.object(LLaDA8BInstructAdapter, "from_local_tokenizer", return_value=self.adapter), \
                patch.object(LLaDA8BInstructAdapter, "load_model", return_value=loaded):
            with self.assertRaisesRegex(ValueError, "logits differ"):
                workflow.load_final_model(self.output)
        self.assertTrue(all(torch.equal(value, self.original_model.state_dict()[name]) for name, value in loaded.state_dict().items()))

    def test_smoke_runs_configured_rejection_and_reload_in_one_entry(self):
        self.adapter.tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: "constructed prompt",
                                                 encode=lambda *args, **kwargs: [5])
        config_file = self.p1_dir.parent / "smoke_config.json"
        workflow.write_json(config_file, self.settings.as_record())
        original = smoke_p5_acceptance.run_p5
        def frozen_before_start(*args, **kwargs):
            root = self.p1_dir.parent / "p5_smoke"
            self.assertTrue((root / "configured_settings.json").is_file())
            self.assertTrue((root / "forced_rejection_settings.json").is_file())
            return original(*args, **kwargs)
        with patch.object(LLaDA8BInstructAdapter, "from_local_tokenizer", return_value=self.adapter), \
                patch.object(LLaDA8BInstructAdapter, "load_model", side_effect=lambda *a, **kw: copy.deepcopy(self.original_model)), \
                patch.object(self.adapter, "prepare_development_input", side_effect=self.prepare), \
                patch.object(smoke_p5_acceptance, "run_p5", side_effect=frozen_before_start):
            report = smoke_p5_acceptance.smoke(self.p1_dir, residual_dir=self.residual_dir, config_file=config_file)
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["actual_model_reload_passed"])
        self.assertEqual(report["forced_rejection"]["decision"], "rejected")
        self.assertFalse(report["p5_stage_complete"])

    def reload_with(self, model):
        with patch.object(LLaDA8BInstructAdapter, "from_local_tokenizer", return_value=self.adapter), \
                patch.object(LLaDA8BInstructAdapter, "load_model", return_value=model):
            return workflow.load_final_model(self.output)

    def test_reload_requires_complete_adapter_target_whitelist(self):
        self.run_p5()
        path = self.output / "final_target_weights.safetensors"
        tensors = load_file(str(path))
        missing = next(iter(tensors))
        del tensors[missing]
        save_file(tensors, str(path))
        def change(row):
            del row["layers"][missing]
            row["tensor_file_sha256"] = file_sha256(path)
        self.change_json(self.output / "manifest.json", change)
        with self.assertRaisesRegex(ValueError, "whitelist"):
            self.reload_with(copy.deepcopy(self.original_model))

    def test_reload_selected_alpha_must_match_acceptance_report(self):
        self.run_p5()
        self.change_json(self.output / "manifest.json", lambda row: row.update(selected_alpha=.123))
        with self.assertRaisesRegex(ValueError, "selection"):
            self.reload_with(copy.deepcopy(self.original_model))

    def test_reload_checks_increment_file_digest(self):
        self.run_p5()
        path = self.output / "candidate_increments.safetensors"
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "artifact digest"):
            self.reload_with(copy.deepcopy(self.original_model))

    def test_reload_rechecks_sources_after_model_loading(self):
        self.run_p5()
        loaded = copy.deepcopy(self.original_model)
        def change(*args, **kwargs):
            self.dev.write_bytes(self.dev.read_bytes() + b"\n")
            return loaded
        with patch.object(LLaDA8BInstructAdapter, "from_local_tokenizer", return_value=self.adapter), \
                patch.object(LLaDA8BInstructAdapter, "load_model", side_effect=change):
            with self.assertRaisesRegex(ValueError, "artifact digest"):
                workflow.load_final_model(self.output)
        self.assertTrue(all(torch.equal(value, self.original_model.state_dict()[name]) for name, value in loaded.state_dict().items()))

    def test_reload_detects_target_mutation_even_within_logits_tolerance(self):
        self.run_p5()
        loaded = copy.deepcopy(self.original_model)
        original = self.adapter.forward_logits
        def change(model, *args):
            result = original(model, *args)
            with torch.no_grad():
                model.model.transformer.blocks[0].ff_out.weight.add_(1e-7)
            return result
        with patch.object(self.adapter, "forward_logits", side_effect=change):
            with self.assertRaisesRegex(RuntimeError, "integrity"):
                self.reload_with(loaded)
        self.assertTrue(all(torch.equal(value, self.original_model.state_dict()[name]) for name, value in loaded.state_dict().items()))

    def test_rehashed_final_weights_still_have_to_satisfy_merge_formula(self):
        self.run_p5()
        path = self.output / "final_target_weights.safetensors"
        tensors = load_file(str(path))
        name = next(iter(tensors))
        tensors[name][0, 0] += 1e-7
        save_file(tensors, str(path))
        def change(row):
            row["tensor_file_sha256"] = file_sha256(path)
            row["layers"][name]["sha256"] = workflow.tensor_sha256(tensors[name])
        self.change_json(self.output / "manifest.json", change)
        with self.assertRaisesRegex(ValueError, "merge formula"):
            self.reload_with(copy.deepcopy(self.original_model))


if __name__ == "__main__":
    unittest.main()
