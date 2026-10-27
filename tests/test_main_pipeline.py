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
                   "dev_edit_probes": [{"case_id": "fact_1", "probe_id": "edit_1"}]},
        }
        self.config.write_text(json.dumps(self.choices), encoding="utf-8")
        self.identity = {"model_revision": "model_hash", "tokenizer_revision": "tokenizer_hash"}

    def fake_prepare(self, dataset, model, layers, cache_dir, **kwargs):
        from data.readers.json_cases import load_cases
        prepared = self.root / ("prepared_" + dataset.stem)
        prepared.mkdir(exist_ok=True)
        probes = [{"case_id": case["case_id"], "probe_id": case["prompts"]["edit"][0]["probe_id"]}
                  for case in load_cases(dataset)]
        (prepared / "development_probes.json").write_text(json.dumps(probes), encoding="utf-8")
        (prepared / "evaluation_inputs.json").write_text('{"inputs": []}', encoding="utf-8")
        manifest = prepared / "manifest.json"
        evaluation_file = prepared / "evaluation_inputs.json"
        manifest.write_text(json.dumps({"status": "complete",
                                        "files": {"evaluation_inputs.json": str(evaluation_file)},
                                        "files_sha256": {str(evaluation_file): pipeline.file_sha256(evaluation_file)}}),
                            encoding="utf-8")
        return {"cache_hit": False, "manifest": str(manifest), "cache_key": "test",
                "recipe_id": "frozen_keep", "files": {
                    "editing_cases.json": str(dataset), "keep_statistics.json": str(self.keep),
                    "development_probes.json": str(prepared / "development_probes.json"),
                    "development_keep.json": str(self.dev_keep),
                    "keep_training.json": str(self.keep),
                    "evaluation_inputs.json": str(prepared / "evaluation_inputs.json")}}

    def test_required_paths_fail_before_model_loading(self):
        for key in ("dataset", "model_path"):
            with self.assertRaises(ValueError):
                pipeline.required_path({}, key)
            with self.assertRaises(FileNotFoundError):
                pipeline.required_path({key: str(self.root / "missing")}, key)
        self.assertEqual(pipeline.required_path(self.choices, "dataset"), self.dataset)
        self.choices["input_preparation"] = {"mode": "explicit"}
        with self.assertRaisesRegex(ValueError, "只支持从数据集"):
            pipeline.prepare_dataset_run(self.choices, None, None, None, True)

    @patch.object(pipeline, "local_model_identity")
    @patch.object(pipeline.subprocess, "run")
    def test_existing_experiment_is_not_overwritten(self, execute, identity):
        execute.side_effect = self.fake_stage
        identity.return_value = self.identity
        pipeline.one_round(self.config, run_id_override="trial")
        execute.reset_mock()
        report = self.root / "runs/trial/round_0/pipeline_report.json"
        saved = report.read_bytes()
        with self.assertRaises(FileExistsError):
            pipeline.one_round(self.config, run_id_override="trial")
        execute.assert_not_called()
        self.assertEqual(report.read_bytes(), saved)

    def fake_stage(self, command, **kwargs):
        module = command[command.index("-m") + 1]
        round_dir = Path(command[command.index("-m") + 2]).parent
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
    def test_c_reuse_is_enabled_only_for_independent_resident_rounds(self, identity):
        identity.return_value = self.identity
        self.choices["execution"] = {"backend": "resident", "device": "cpu"}
        self.config.write_text(json.dumps(self.choices), encoding="utf-8")

        def stage(label, arguments, artifact, accepted=("complete",), **kwargs):
            if label.startswith("P0_"):
                return [{"case_id": "fact_1", "probe_id": "edit_1", "real_forward": True,
                         "replay_within_tolerance": True, "full_mask_input": True,
                         "mask_count": 1, "target_token_count": 1,
                         "target_modules": {"model.transformer.blocks.0.ff_out": {}},
                         "ffn_input_shapes": {"model.transformer.blocks.0.ff_out": [1]}}]
            return {"status": "rejected" if label == "P4" else "complete"}

        for sequential in (False, True):
            with patch.object(pipeline, "run_stage", side_effect=stage) as run:
                pipeline.one_round(self.config, run_id_override=f"mode_{sequential}", sequential=sequential)
            command = next(call.args[1] for call in run.call_args_list if call.args[0] == "P2_KEEP")
            self.assertEqual("--reuse-across-rounds" in command, not sequential)
            self.assertNotIn("--batch-keep", command)
        self.choices["p2_keep"]["device"] = "cuda:0"
        self.config.write_text(json.dumps(self.choices), encoding="utf-8")
        with patch.object(pipeline, "run_stage", side_effect=stage) as run:
            pipeline.one_round(self.config, run_id_override="batched")
        for label in ("P2_KEEP", "P2_QA"):
            command = next(call.args[1] for call in run.call_args_list if call.args[0] == label)
            self.assertIn("--batch-keep", command)

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
        with patch.object(pipeline, "prepare_inputs", side_effect=self.fake_prepare):
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
        with patch.object(pipeline, "prepare_inputs", side_effect=self.fake_prepare):
            result = pipeline.run_dataset(self.config, run_id_override="override", dataset_override=selected,
                                          cache_override=self.root / "cache", use_cache=False)
        self.assertEqual((result["fact_count"], result["case_count"]), (2, 1))
        self.assertEqual(run_round.call_args.kwargs["selected_case_ids"], ["other_0", "other_1"])
        effective = json.loads((self.root / "runs/override/effective_config.json").read_text(encoding="utf-8"))
        self.assertEqual(effective["dataset"], str(selected))
        report = json.loads((self.root / "runs/override/run_report.json").read_text())
        self.assertEqual(report["evaluation_status"], "unavailable")

    @patch.object(pipeline, "local_model_identity")
    @patch.object(pipeline.subprocess, "run")
    def test_input_cache_is_reused_but_each_experiment_runs_all_stages(self, execute, identity):
        from util import prepare_dataset_inputs
        for filename in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            (self.model / filename).write_text("{}", encoding="utf-8")
        case = json.loads(self.dataset.read_text(encoding="utf-8"))
        case["prompts"]["preservation"] = [
            {"probe_id": f"keep_{i}", "prompt": f"keep {i}?", "answer": "old"} for i in range(4)]
        self.dataset.write_text(json.dumps([case]), encoding="utf-8")
        self.choices["input_preparation"] = {"settings": {}}
        self.choices["cache_dir"] = str(self.root / "cache")
        del self.choices["keep_inputs"]
        del self.choices["p2_keep"]["recipe_id"]
        del self.choices["p5"]["dev_edit_file"]
        del self.choices["p5"]["dev_keep_file"]
        del self.choices["p5"]["dev_edit_probes"]
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
        self.assertEqual(execute.call_count, 8)
        tokenize.assert_not_called()
        self.assertNotEqual(first["rounds"][0]["report"], second["rounds"][0]["report"])
        self.assertTrue(Path(first["rounds"][0]["report"]).is_file())
        self.assertTrue(Path(second["rounds"][0]["report"]).is_file())

    def test_single_case_selection_keeps_full_dataset_preparation(self):
        cases = [json.loads(self.dataset.read_text()) for _ in range(2)]
        cases[1]["case_id"] = "fact_2"
        self.dataset.write_text(json.dumps(cases), encoding="utf-8")
        with patch.object(pipeline, "local_model_identity", return_value=self.identity), \
             patch.object(pipeline, "one_round", return_value={"decision": "rejected", "selected_alpha": None,
                          "final_model_dir": None, "checkpoint_dir": None}) as run_round, \
             patch.object(pipeline, "prepare_inputs", side_effect=self.fake_prepare) as prepare:
            result = pipeline.run_dataset(self.config, run_id_override="single", case_ids=["fact_2"])
        self.assertEqual(prepare.call_args.args[0], self.dataset)
        self.assertEqual(run_round.call_args.kwargs["selected_case_ids"], ["fact_2"])
        self.assertEqual((result["fact_count"], result["case_count"]), (1, 1))

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
                 patch.object(pipeline, "one_round", side_effect=fake_round), \
                 patch.object(pipeline, "prepare_inputs", side_effect=self.fake_prepare):
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
                               selected_case_ids=["fact_1"], baseline_model=checkpoint,
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
