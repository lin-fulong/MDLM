"""The production entrypoint orders stages and reports the actual P5 decision."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import main as pipeline


class MainPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root / "model"
        self.model.mkdir()
        self.dataset = self.root / "dataset.json"
        self.dataset.write_text(json.dumps({
            "case_id": "fact_1", "quality_control": {"status": "gold_ready"},
            "prompts": {"edit": [{"probe_id": "edit_1"}]},
        }), encoding="utf-8")
        self.keep = self.root / "keep.json"
        self.keep.write_text('{"example_only": false}', encoding="utf-8")
        self.dev = self.root / "dev.json"
        self.dev.write_text('{"case_id": "dev_1"}', encoding="utf-8")
        self.dev_keep = self.root / "dev_keep.json"
        self.dev_keep.write_text('{"example_only": false}', encoding="utf-8")
        self.config = self.root / "config.json"
        self.choices = {
            "dataset": str(self.dataset), "model_path": str(self.model),
            "output_root": str(self.root / "runs"), "target_layers": [0],
            "cache_dir": str(self.root / "cache"),
            "keep_inputs": str(self.keep),
            "p2_keep": {"recipe_id": "frozen_keep"},
            "p3": {"ridge_by_layer": {"model.transformer.blocks.0.ff_out": 0.01}},
            "p4": {"rho_by_layer": {"model.transformer.blocks.0.ff_out": 0.001}},
            "p5": {"dev_edit_file": str(self.dev), "dev_keep_file": str(self.dev_keep),
                   "dev_edit_probes": [{"case_id": "dev_1", "probe_id": "qa_1"}]},
        }
        self.config.write_text(json.dumps(self.choices), encoding="utf-8")
        self.identity = {"model_revision": "model_hash", "tokenizer_revision": "tokenizer_hash"}

    def test_incomplete_config_reports_all_missing_inputs_before_run(self):
        template = {"dataset": str(self.dataset), "model_path": str(self.model),
                    "keep_inputs": "REPLACE_WITH_KEEP", "p2_keep": {}, "p5": {}}
        issues = pipeline.config_setup_issues(template)
        self.assertTrue(any("keep_inputs" in issue for issue in issues))
        self.assertTrue(any("p5.dev_edit_file" in issue for issue in issues))
        self.assertTrue(any("p5.dev_keep_file" in issue for issue in issues))
        self.assertTrue(any("p2_keep.recipe_id" in issue for issue in issues))
        self.assertTrue(any("p5.dev_edit_probes" in issue for issue in issues))

    @patch.object(pipeline, "local_model_identity")
    @patch.object(pipeline.subprocess, "run")
    def test_existing_stage_files_are_reused_and_corruption_recomputes_dependents(self, execute, identity):
        execute.side_effect = self.fake_stage
        identity.return_value = self.identity
        pipeline.one_round(self.config, run_id_override="trial", request_key="same")
        execute.reset_mock()
        workspace = self.root / "runs/trial/round_0"
        result = pipeline.one_round(self.config, workspace=workspace, request_key="same")
        execute.assert_not_called()
        self.assertTrue(result["stages"]["P2_KEYS"]["cache_hit"])
        (workspace / "matrices/reference_keys/manifest.json").write_text("corrupted", encoding="utf-8")
        pipeline.one_round(self.config, workspace=workspace, request_key="same")
        self.assertTrue(any("util.extract_reference_keys" in call.args[0] for call in execute.call_args_list))
        self.assertTrue(list((workspace / "failed_attempts").glob("p2_keys_*")))

    def fake_stage(self, command, **kwargs):
        module = command[command.index("-m") + 1]
        round_dir = self.root / "runs" / "trial" / "round_0"
        outputs = {
            "util.collect_denoise_states": round_dir / "p1/coverage_report.json",
            "util.extract_reference_keys": round_dir / "matrices/reference_keys/manifest.json",
            "util.extract_keep_statistics": round_dir / "matrices/keep_statistics/manifest.json",
            "util.validate_matrix_artifacts": round_dir / "matrices/validation/manifest.json",
            "util.solve_writeback_factor": round_dir / "matrices/writeback_factors/manifest.json",
            "util.optimize_residuals": round_dir / "residuals/manifest.json",
            "util.merge_and_accept": round_dir / "editing/acceptance_report.json",
        }
        if module == "util.smoke_p0_llada":
            path = Path(command[command.index("--output") + 1])
            value = [{"case_id": "fact_1", "probe_id": "edit_1", "real_forward": True,
                      "replay_within_tolerance": True, "full_mask_input": True,
                      "mask_count": 1, "target_token_count": 1,
                      "target_modules": {"model.transformer.blocks.0.ff_out": {}},
                      "ffn_input_shapes": {"model.transformer.blocks.0.ff_out": [1]}}]
        else:
            path = outputs[module]
            value = {"status": "complete"}
            if module == "util.merge_and_accept":
                value.update(decision="rejected", selection={"selected_alpha": None},
                             baseline_restored=True)
                path.parent.mkdir(parents=True, exist_ok=True)
                (round_dir / "editing/manifest.json").write_text(json.dumps({
                    "status": "rejected", "example_only": False, "p5_stage_complete": True,
                }), encoding="utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    @patch.object(pipeline, "local_model_identity")
    @patch.object(pipeline.subprocess, "run")
    def test_runs_every_stage_and_reports_rejection(self, execute, identity):
        execute.side_effect = self.fake_stage
        identity.return_value = self.identity
        result = pipeline.one_round(self.config, run_id_override="trial")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["decision"], "rejected")
        self.assertIsNone(result["final_model_dir"])
        self.assertTrue(result["baseline_restored"])
        self.assertEqual(list(result["stages"]), [
            "P0_LAYER_0", "P1", "P2_KEYS", "P2_KEEP", "P2_QA", "P3", "P4", "P5"])
        self.assertEqual(execute.call_count, 8)
        self.assertTrue((self.root / "runs/trial/round_0/p1_config.json").exists())

    @patch.object(pipeline, "local_model_identity")
    @patch.object(pipeline.subprocess, "run")
    def test_stage_failure_stops_and_preserves_report(self, execute, identity):
        identity.return_value = self.identity

        def fail_after_p0(command, **kwargs):
            if "util.collect_denoise_states" in command:
                raise RuntimeError("P1 failed")
            self.fake_stage(command, **kwargs)

        execute.side_effect = fail_after_p0
        with self.assertRaisesRegex(RuntimeError, "P1 failed"):
            pipeline.one_round(self.config, run_id_override="trial")
        report = json.loads((self.root / "runs/trial/round_0/pipeline_report.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["phase"], "P1")
        self.assertEqual(execute.call_count, 2)

    def test_example_inputs_are_rejected_before_model_loading(self):
        self.keep.write_text('{"example_only": true}', encoding="utf-8")
        with patch.object(pipeline, "local_model_identity") as identity:
            with self.assertRaisesRegex(ValueError, "non-example"):
                pipeline.one_round(self.config, run_id_override="trial")
            identity.assert_not_called()

    @patch.object(pipeline, "local_model_identity")
    @patch.object(pipeline.subprocess, "run")
    def test_no_candidate_is_a_rejected_round_and_does_not_run_p5(self, execute, identity):
        identity.return_value = self.identity

        def reject_at_p4(command, **kwargs):
            if "util.optimize_residuals" in command:
                path = self.root / "runs/trial/round_0/residuals/manifest.json"
                path.parent.mkdir(parents=True)
                path.write_text('{"status": "rejected"}', encoding="utf-8")
                raise subprocess.CalledProcessError(1, command)
            self.fake_stage(command, **kwargs)

        execute.side_effect = reject_at_p4
        result = pipeline.one_round(self.config, run_id_override="trial")
        self.assertEqual((result["status"], result["decision"]), ("complete", "rejected"))
        self.assertFalse(any("util.merge_and_accept" in call.args[0] for call in execute.call_args_list))

    @patch.object(pipeline, "local_model_identity")
    @patch.object(pipeline, "one_round")
    def test_device_selection_reaches_all_stage_configs(self, run_round, identity):
        identity.return_value = self.identity
        run_round.return_value = {"decision": "rejected", "selected_alpha": None,
                                  "final_model_dir": None, "checkpoint_dir": None}
        pipeline.run_dataset(self.config, run_id_override="device_choice", use_cache=False, device_override="cpu")
        effective = json.loads((self.root / "runs/device_choice/effective_config.json").read_text(encoding="utf-8"))
        self.assertEqual(effective["execution"], {"backend": "resident", "device": "cpu"})
        self.assertEqual(effective["p1"]["device_map"], "cpu")
        self.assertEqual(effective["p2_keep"]["device"], "cpu")
        self.assertEqual(effective["p3"]["device"], "cpu")

    @patch.object(pipeline, "local_model_identity")
    @patch.object(pipeline, "one_round")
    def test_dataset_override_and_source_case_grouping(self, run_round, identity):
        selected = self.root / "other_dataset.json"
        cases = [json.loads(self.dataset.read_text(encoding="utf-8")) for _ in range(2)]
        for index, case in enumerate(cases):
            case.update(case_id=f"other_{index}", source_case_id="one_source")
        selected.write_text(json.dumps(cases), encoding="utf-8")
        identity.return_value = self.identity
        run_round.return_value = {"decision": "rejected", "selected_alpha": None,
                                  "final_model_dir": None, "checkpoint_dir": None}
        result = pipeline.run_dataset(self.config, run_id_override="override", dataset_override=selected,
                                      cache_override=self.root / "cache", use_cache=False)
        self.assertEqual((result["fact_count"], result["case_count"]), (2, 1))
        self.assertEqual(run_round.call_args.kwargs["selected_case_ids"], ["other_0", "other_1"])
        effective = json.loads((self.root / "runs/override/effective_config.json").read_text(encoding="utf-8"))
        self.assertEqual(effective["dataset"], str(selected))
        self.assertTrue((self.root / "runs/override/evaluation_plan.json").is_file())

    @patch.object(pipeline, "local_model_identity")
    @patch.object(pipeline.subprocess, "run")
    def test_dataset_mode_prepares_inputs_and_second_run_reuses_entire_pipeline(self, execute, identity):
        from util import prepare_dataset_inputs
        for filename in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            (self.model / filename).write_text("{}", encoding="utf-8")
        case = json.loads(self.dataset.read_text(encoding="utf-8"))
        case["prompts"]["preservation"] = [
            {"probe_id": f"keep_{i}", "prompt": f"keep {i}?", "answer": "old"} for i in range(4)]
        self.dataset.write_text(json.dumps([case]), encoding="utf-8")
        self.choices["input_preparation"] = {"mode": "dataset"}
        self.choices["cache_dir"] = str(self.root / "cache")
        self.config.write_text(json.dumps(self.choices), encoding="utf-8")
        tokenizer = SimpleNamespace(all_special_ids=[], encode=lambda text, **kwargs: [ord(c) for c in text],
                                    apply_chat_template=lambda messages, **kwargs: messages[0]["content"] + " answer:")
        adapter = SimpleNamespace(tokenizer=tokenizer, mask_token_id=99999)
        identity.return_value = self.identity
        execute.side_effect = self.fake_stage
        with patch.object(prepare_dataset_inputs.LLaDA8BInstructAdapter, "from_local_tokenizer", return_value=adapter), \
             patch("util.evaluate_accuracy.evaluate_run", return_value={"status": "unavailable"}):
            first = pipeline.run_dataset(self.config, run_id_override="trial")
        self.assertEqual(first["status"], "complete")
        execute.reset_mock()
        with patch.object(prepare_dataset_inputs.LLaDA8BInstructAdapter, "from_local_tokenizer") as tokenize, \
             patch("util.evaluate_accuracy.evaluate_run", return_value={"status": "unavailable"}):
            second = pipeline.run_dataset(self.config, run_id_override="rerun")
        execute.assert_not_called()
        tokenize.assert_not_called()
        self.assertTrue(second["rounds"][0]["cache_workspace"])
        self.assertEqual(first["rounds"][0]["report"], second["rounds"][0]["report"])

    def test_default_resets_and_sequential_chains_only_accepted_edits(self):
        cases = [json.loads(self.dataset.read_text(encoding="utf-8"))]
        second = json.loads(json.dumps(cases[0]))
        second["case_id"] = "fact_2"
        cases.append(second)
        self.dataset.write_text(json.dumps(cases), encoding="utf-8")
        checkpoint = self.root / "accepted_checkpoint"

        for sequential, expected in ((False, [self.model, self.model]),
                                     (True, [self.model, checkpoint])):
            seen = []

            def fake_round(config, **kwargs):
                seen.append(kwargs["baseline_model"])
                accepted = kwargs["round_index"] == 0
                return {"decision": "accepted" if accepted else "rejected",
                        "selected_alpha": 0.5 if accepted else None,
                        "final_model_dir": str(self.root / "editing") if accepted else None,
                        "checkpoint_dir": str(checkpoint) if sequential and accepted else None}

            with patch.object(pipeline, "local_model_identity", return_value=self.identity), \
                 patch.object(pipeline, "one_round", side_effect=fake_round):
                result = pipeline.run_dataset(self.config, sequential=sequential,
                                              run_id_override="seq" if sequential else "ind")
            self.assertEqual(seen, expected)
            self.assertEqual((result["accepted_count"], result["rejected_count"]), (1, 1))
            self.assertEqual(result["final_checkpoint_dir"], str(checkpoint) if sequential else None)

    @patch.object(pipeline, "local_model_identity")
    @patch.object(pipeline.subprocess, "run", side_effect=RuntimeError("stop before model forward"))
    def test_sequential_round_uses_previous_baseline_and_original_theta0(self, execute, identity):
        checkpoint = self.root / "prior_checkpoint"
        checkpoint.mkdir()
        identity.return_value = {"model_revision": "edited_hash", "tokenizer_revision": "tokenizer_hash"}
        with self.assertRaisesRegex(RuntimeError, "stop before model forward"):
            pipeline.one_round(self.config, run_id_override="trial", round_index=1,
                               selected_case_id="fact_1", baseline_model=checkpoint,
                               sequential=True, original_identity=self.identity)
        round_dir = self.root / "runs/trial/round_1"
        p1 = json.loads((round_dir / "p1_config.json").read_text(encoding="utf-8"))
        p5 = json.loads((round_dir / "p5_config.json").read_text(encoding="utf-8"))
        self.assertEqual(p1["model_path"], str(checkpoint))
        self.assertEqual(p1["baseline_id"], "edited_hash")
        self.assertEqual(p1["round_id"], 1)
        self.assertEqual(p5["reference_model_path"], str(self.model))
        self.assertEqual(p5["reference_model_revision"], "model_hash")


if __name__ == "__main__":
    unittest.main()
