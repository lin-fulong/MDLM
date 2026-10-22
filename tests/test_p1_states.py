"""Small artificial state traces for formulas (1)–(3); not model evidence."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from MDLM.contracts import P1_ARTIFACT_VERSION, P1RunConfig, config_digest, file_sha256
from MDLM.states import (
    FactTrajectory,
    build_supervision,
    input_from_record,
    input_record,
    load_supervision,
    load_trajectories,
    save_supervision,
    save_trajectories,
)
from model.adapters.llada_8b_instruct import LLaDA8BInstructAdapter
from model.base import DenoiseTrace, EditTarget, MaskedEditInput
from util.collect_denoise_states import collect_run
from util.filter_compatible_states import filter_run


MASK = 99


def state(tokens: tuple[int, ...], masks: tuple[int, ...]) -> MaskedEditInput:
    return MaskedEditInput(tokens, (1,) * len(tokens), masks, 0)


class P1StateTests(unittest.TestCase):
    def setUp(self):
        first = DenoiseTrace(
            (
                state((7, MASK, MASK, MASK), (1, 2, 3)),
                state((7, 11, MASK, MASK), (2, 3)),
                state((7, 11, MASK, MASK), (2, 3)),
                state((7, 11, 88, MASK), (3,)),
            ),
            ("a" * 64, "b" * 64, "c" * 64, "d" * 64),
            (((1, 11),), (), ((2, 88),), ((3, 77),)),
            state((7, 11, 88, 77), ()),
        )
        second = DenoiseTrace(
            (state((7, MASK), (1,)),),
            ("e" * 64,),
            (((1, 22),),),
            state((7, 22), ()),
        )
        self.trajectories = [
            FactTrajectory("fact_1", "edit_1", (1, 2, 3), (11, 12, 13), 1, first),
            FactTrajectory("fact_2", "edit_2", (1,), (22,), 2, second),
        ]

    def write_filter_inputs(self, run_dir, trajectories=None):
        save_trajectories(run_dir / "trajectories.jsonl", self.trajectories if trajectories is None else trajectories)
        config = {
            "run_id": "example_only", "round_id": 0, "baseline_id": "unverified",
            "facts": [{"case_id": item.case_id, "probe_id": item.probe_id} for item in self.trajectories],
        }
        collection = {"failure": None, "completed": [
            {"case_id": item.case_id, "probe_id": item.probe_id, "purpose": "example_only",
             "replayed": False, "logits_replayed": False} for item in self.trajectories
        ]}
        (run_dir / "collection_report.json").write_text(json.dumps(collection), encoding="utf-8")
        manifest = {
            "artifact_version": P1_ARTIFACT_VERSION, "collection_status": "complete",
            "mask_token_id": MASK, "fact_order": [item.case_id for item in self.trajectories],
            "config": config, "config_sha256": config_digest(config),
            "trajectories_sha256": file_sha256(run_dir / "trajectories.jsonl"),
            "collection_report_sha256": file_sha256(run_dir / "collection_report.json"),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    def test_formula_one_and_three_with_two_facts(self):
        facts, report = build_supervision(self.trajectories, MASK)
        self.assertEqual([len(item["states"]) for item in facts], [2, 1])
        self.assertEqual(report["facts"][0]["duplicate_count"], 1)
        self.assertEqual(report["facts"][0]["conflict_count"], 1)
        self.assertEqual(report["facts"][0]["terminal_final_count"], 1)
        self.assertTrue(report["facts"][0]["full_mask_initial_compatible"])
        self.assertEqual(report["supervision_position_count"], 6)
        self.assertAlmostEqual(facts[0]["states"][0]["supervision"][0]["weight"], 1 / 12)
        self.assertAlmostEqual(facts[0]["states"][1]["supervision"][0]["weight"], 1 / 8)
        self.assertEqual([term["position"] for term in facts[0]["states"][1]["supervision"]], [2, 3])
        self.assertEqual([term["target_token_id"] for term in facts[0]["states"][1]["supervision"]], [12, 13])
        self.assertAlmostEqual(report["facts"][0]["weight_sum"], 0.5)
        self.assertAlmostEqual(report["facts"][1]["weight_sum"], 0.5)
        self.assertAlmostEqual(report["global_weight_sum"], 1.0)
        self.assertTrue(report["formula_checks_passed"])

    def test_complete_input_and_artifact_round_trip(self):
        original = self.trajectories[0].trace.pre_forward_inputs[0]
        self.assertEqual(input_from_record(input_record(original)), original)
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "trajectories.jsonl"
            fixed_path = Path(directory) / "fixed_states.jsonl"
            save_trajectories(raw, self.trajectories)
            self.assertEqual(load_trajectories(raw), self.trajectories)
            fixed, _ = build_supervision(load_trajectories(raw), MASK)
            save_supervision(fixed_path, fixed)
            self.assertEqual(load_supervision(fixed_path), fixed)

    def test_remasking_is_rejected(self):
        original = self.trajectories[0]
        remasked = DenoiseTrace(
            (original.trace.pre_forward_inputs[0], original.trace.pre_forward_inputs[1],
             state((7, MASK, MASK, MASK), (1, 2, 3))),
            original.trace.mask_logit_sha256[:3],
            original.trace.revealed_tokens[:3],
            original.trace.final_input,
        )
        bad = FactTrajectory(original.case_id, original.probe_id, original.answer_positions,
                             original.target_token_ids, original.seed, remasked)
        with self.assertRaisesRegex(ValueError, "masked again"):
            build_supervision([bad], MASK)

    def test_invalid_answer_and_mask_positions_are_rejected(self):
        original = self.trajectories[1]
        initial = original.trace.pre_forward_inputs[0]
        variants = [
            (replace(original, answer_positions=(), target_token_ids=()), "nonempty"),
            (replace(original, answer_positions=(1, 1), target_token_ids=(22, 22)), "unique"),
            (replace(original, answer_positions=(-1,)), "out of range"),
            (replace(original, trace=replace(original.trace, pre_forward_inputs=(
                replace(initial, input_ids=(MASK, MASK)),))), "mask positions"),
            (replace(self.trajectories[0], trace=replace(self.trajectories[0].trace,
                pre_forward_inputs=(self.trajectories[0].trace.pre_forward_inputs[0],
                    replace(self.trajectories[0].trace.pre_forward_inputs[1], mask_positions=(2, 2, 3)),
                    *self.trajectories[0].trace.pre_forward_inputs[2:]))), "mask positions"),
        ]
        for trajectory, message in variants:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                build_supervision([trajectory], MASK)

    def test_missing_initial_sample_has_clear_error(self):
        trace = replace(self.trajectories[1].trace, pre_forward_inputs=())
        with self.assertRaisesRegex(ValueError, "no pre-forward input"):
            FactTrajectory.from_sample(EditTarget("fact", "edit", (22,)), 1, trace)

    def test_example_config_is_explicitly_unverified(self):
        path = Path(__file__).parents[1] / "configs" / "p1_example_only.json"
        config = P1RunConfig.from_file(path)
        self.assertIn("example_only", config.run_id)
        self.assertEqual(config.fact_seed(config.facts[0]), config.fact_seed(config.facts[0]))
        self.assertEqual(len(config.digest()), 64)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["steps"], None)

    def test_filter_run_writes_fixed_artifacts_and_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self.write_filter_inputs(run_dir)
            report = filter_run(run_dir)
            self.assertTrue(report["formula_checks_passed"])
            self.assertEqual(report["example_only_fact_count"], 2)
            self.assertEqual(len(load_supervision(run_dir / "fixed_states.jsonl")), 2)
            self.assertTrue((run_dir / "coverage_report.json").exists())

    def test_filter_rejects_changed_or_mixed_artifacts(self):
        for change in ("config", "trajectory", "probe", "collection", "version"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                run_dir = Path(directory)
                manifest = self.write_filter_inputs(run_dir)
                if change == "config":
                    manifest["config"]["round_id"] = 1
                elif change in ("trajectory", "probe"):
                    modified = [replace(self.trajectories[0], probe_id="wrong_probe"), self.trajectories[1]]
                    save_trajectories(run_dir / "trajectories.jsonl", modified)
                    if change == "probe":
                        manifest["trajectories_sha256"] = file_sha256(run_dir / "trajectories.jsonl")
                elif change == "collection":
                    (run_dir / "collection_report.json").write_text(
                        json.dumps({"completed": [], "failure": None}), encoding="utf-8")
                    manifest["collection_report_sha256"] = file_sha256(run_dir / "collection_report.json")
                else:
                    manifest["artifact_version"] = "unsupported"
                (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaises(ValueError):
                    filter_run(run_dir)
                self.assertFalse((run_dir / "fixed_states.jsonl").exists())
                report = json.loads((run_dir / "coverage_report.json").read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "failed")
                self.assertTrue(report["failure"])

    def test_failed_formula_check_does_not_publish_supervision(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self.write_filter_inputs(run_dir)
            with patch("util.filter_compatible_states.build_supervision", return_value=([], {"formula_checks_passed": False})):
                with self.assertRaisesRegex(ValueError, "checks failed"):
                    filter_run(run_dir)
            self.assertFalse((run_dir / "fixed_states.jsonl").exists())
            report = json.loads((run_dir / "coverage_report.json").read_text(encoding="utf-8"))
            self.assertFalse(report["formula_checks_passed"])
            self.assertEqual(report["status"], "failed")

    def test_collection_pipeline_with_artificial_adapter(self):
        outer = self

        class ArtificialAdapter:
            mask_token_id = MASK

            def load_model(self, *_args, **_kwargs):
                return object()

            def prepare_edit_input(self, case, probe_id):
                return state((7, MASK), (1,)), EditTarget(case["case_id"], probe_id, (22,))

            def sample_denoise(self, _model, model_input, *, steps, temperature, seed):
                outer.assertEqual(steps, 1)
                outer.assertEqual(temperature, 0.0)
                return DenoiseTrace((model_input,), ("a" * 64,), (((1, 22),),), state((7, 22), ()))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            case_path = root / "cases.json"
            case_path.write_text(json.dumps({
                "case_id": "fact_1", "quality_control": {"status": "example_only"}
            }), encoding="utf-8")
            template = json.loads((Path(__file__).parents[1] / "configs" / "p1_example_only.json").read_text(encoding="utf-8"))
            template.update({
                "model_path": str(model_dir), "case_file": str(case_path),
                "facts": [{"case_id": "fact_1", "probe_id": "edit_1"}],
                "output_root": str(root / "runs"),
            })
            config_path = root / "p1.json"
            config_path.write_text(json.dumps(template), encoding="utf-8")
            config = P1RunConfig.from_file(config_path)
            with patch.object(LLaDA8BInstructAdapter, "from_local_tokenizer", return_value=ArtificialAdapter()):
                report = collect_run(config)
            self.assertTrue(report["formula_checks_passed"])
            self.assertEqual(report["replayed_fact_count"], 1)
            self.assertEqual(report["logits_replayed_fact_count"], 1)
            self.assertEqual(report["example_only_fact_count"], 1)
            self.assertEqual(len(load_trajectories(config.p1_dir() / "trajectories.jsonl")), 1)
            with patch.object(LLaDA8BInstructAdapter, "from_local_tokenizer") as load_tokenizer:
                with self.assertRaises(FileExistsError):
                    collect_run(config)
                load_tokenizer.assert_not_called()

            class ChangedLogitsAdapter(ArtificialAdapter):
                calls = 0

                def sample_denoise(self, *args, **kwargs):
                    self.calls += 1
                    trace = super().sample_denoise(*args, **kwargs)
                    if self.calls == 2:
                        return DenoiseTrace(trace.pre_forward_inputs, ("b" * 64,),
                                            trace.revealed_tokens, trace.final_input)
                    return trace

            template["run_id"] = "example_only_changed_logits"
            config_path.write_text(json.dumps(template), encoding="utf-8")
            changed_config = P1RunConfig.from_file(config_path)
            with patch.object(LLaDA8BInstructAdapter, "from_local_tokenizer", return_value=ChangedLogitsAdapter()):
                with self.assertRaisesRegex(ValueError, "mask logits did not replay"):
                    collect_run(changed_config)
            failure = json.loads((changed_config.p1_dir() / "collection_report.json").read_text(encoding="utf-8"))
            self.assertIn("mask logits did not replay", failure["failure"])
            self.assertEqual(failure["phase"], "replay")
            self.assertEqual(failure["failed_case_id"], "fact_1")

            class UnavailableModelAdapter(ArtificialAdapter):
                def load_model(self, *_args, **_kwargs):
                    raise RuntimeError("example model loading failure")

            failed_config = replace(config, run_id="example_only_setup_failure")
            with patch.object(LLaDA8BInstructAdapter, "from_local_tokenizer", return_value=UnavailableModelAdapter()):
                with self.assertRaisesRegex(RuntimeError, "model loading failure"):
                    collect_run(failed_config)
            failure = json.loads((failed_config.p1_dir() / "collection_report.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["phase"], "setup")
            manifest = json.loads((failed_config.p1_dir() / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["collection_status"], "failed")


if __name__ == "__main__":
    unittest.main()
