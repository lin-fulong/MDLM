"""Keep recipe preparation and orchestration tests; no 8B weights are loaded."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from MDLM.contracts import KEEP_INPUTS_VERSION, KeepStatisticsConfig, file_sha256
from model.adapters.llada_8b_instruct import LLaDA8BInstructAdapter
from util import smoke_p2_keep


class KeepSmokeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.p1_dir = self.root / "runs/example_only_keep/round_0/p1"
        self.p1_dir.mkdir(parents=True)
        template = json.loads((smoke_p2_keep.PROJECT_ROOT / "configs/p1_example_only.json").read_text(encoding="utf-8"))
        template.update({"case_file": "this_dataset_is_not_read.json", "target_layers": [0, 1]})
        (self.p1_dir / "manifest.json").write_text(json.dumps({"config": template}), encoding="utf-8")
        self.config_path = self.root / "keep_config.json"
        self.config_path.write_text(json.dumps(KeepStatisticsConfig("constructed_keep_two_inputs_v1").as_record()), encoding="utf-8")
        self.arguments = ["smoke_p2_keep", "--p1-dir", str(self.p1_dir), "--config", str(self.config_path)]

    def fake_freeze(self, p1_dir, fixture, output_path):
        self.assertEqual(p1_dir, self.p1_dir)
        output_path.write_text('{"example_only": true}', encoding="utf-8")
        return {"example_only": True}

    def fake_extract(self, p1_dir, keep_file, settings):
        self.assertTrue(keep_file.exists())
        directory = p1_dir.parent / "matrices/keep_statistics"
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text('{"status": "complete"}', encoding="utf-8")
        return {"status": "complete", "example_only": True, "output_dir": str(directory)}

    def test_freezer_records_explicit_masks_positions_and_unscaled_weights(self):
        adapter = SimpleNamespace(
            tokenizer=SimpleNamespace(apply_chat_template=lambda *args, **kwargs: [4, 3, 2]),
            mask_token_id=15,
            target_module_names=lambda: ("model.transformer.blocks.0.ff_out", "model.transformer.blocks.1.ff_out"),
        )
        output = self.p1_dir.parent / "keep_inputs.json"
        fixture = smoke_p2_keep.PROJECT_ROOT / "tests/fixtures/p2_keep_example_only.json"
        with patch.object(LLaDA8BInstructAdapter, "from_local_tokenizer", return_value=adapter):
            payload = smoke_p2_keep.freeze_example_inputs(self.p1_dir, fixture, output)
        self.assertEqual(payload["artifact_version"], KEEP_INPUTS_VERSION)
        self.assertEqual(payload["purpose"], "keep_statistics")
        self.assertTrue(payload["example_only"])
        self.assertEqual(payload["inputs"][0]["input"]["input_ids"], [4, 3, 2, 15, 15])
        self.assertEqual(payload["inputs"][0]["capture_positions"], [2, 3])
        for name in adapter.target_module_names():
            self.assertEqual(payload["inputs"][0]["weights_by_layer"][name], [0.5, 1.5])
            self.assertEqual(payload["inputs"][1]["weights_by_layer"][name], [0.0, 2.0])
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), payload)
        with self.assertRaises(FileExistsError):
            smoke_p2_keep.freeze_example_inputs(self.p1_dir, fixture, output)

    def test_existing_p1_mode_skips_upstream_and_binds_c_report(self):
        with patch.object(sys, "argv", self.arguments), patch("sys.stdout", new_callable=io.StringIO), \
                patch.object(smoke_p2_keep.subprocess, "run") as upstream, \
                patch.object(smoke_p2_keep, "freeze_example_inputs", side_effect=self.fake_freeze), \
                patch.object(smoke_p2_keep, "extract_run", side_effect=self.fake_extract):
            smoke_p2_keep.main()
        upstream.assert_not_called()
        root = self.p1_dir.parent
        report = json.loads((root / "p2_keep_acceptance_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["stage_order"], ["P2_C"])
        self.assertNotIn("upstream_reports", report)
        self.assertFalse(report["p2_stage_complete"])
        self.assertFalse(report["formal_dataset_acceptance"])
        self.assertEqual(report["manifest_sha256"], file_sha256(root / "matrices/keep_statistics/manifest.json"))

    def test_changed_raw_recipe_during_tokenizing_cannot_publish_frozen_inputs(self):
        fixture = self.root / "raw_keep.json"
        fixture.write_bytes((smoke_p2_keep.PROJECT_ROOT / "tests/fixtures/p2_keep_example_only.json").read_bytes())
        original_digest = file_sha256(fixture)

        def tokenize(*args, **kwargs):
            changed = json.loads(fixture.read_text(encoding="utf-8"))
            changed["inputs"][0]["weights"][0] = 9.0
            fixture.write_text(json.dumps(changed), encoding="utf-8")
            return [4, 3, 2]

        adapter = SimpleNamespace(
            tokenizer=SimpleNamespace(apply_chat_template=tokenize), mask_token_id=15,
            target_module_names=lambda: ("model.transformer.blocks.0.ff_out", "model.transformer.blocks.1.ff_out"),
        )
        output = self.p1_dir.parent / "keep_inputs.json"
        with patch.object(LLaDA8BInstructAdapter, "from_local_tokenizer", return_value=adapter):
            with self.assertRaisesRegex(ValueError, "raw keep recipe changed"):
                smoke_p2_keep.freeze_example_inputs(self.p1_dir, fixture, output)
        self.assertNotEqual(file_sha256(fixture), original_digest)
        self.assertFalse(output.exists())

    def test_fresh_mode_runs_upstream_before_keep_freezing(self):
        self.p1_dir = self.root / "runs/fresh_keep/round_0/p1"
        order = []

        def upstream(command, **kwargs):
            self.p1_dir.mkdir(parents=True)
            self.assertEqual(command[:3], [sys.executable, "-m", "util.smoke_p2_keys"])
            for filename in ("acceptance_report.json", "p2_keys_acceptance_report.json"):
                (self.p1_dir.parent / filename).write_text('{"status": "passed"}', encoding="utf-8")
            order.append("P0_P1_K")

        def freeze(*args):
            order.append("freeze_keep")
            return self.fake_freeze(*args)

        def extract(*args):
            order.append("C")
            return self.fake_extract(*args)

        arguments = ["smoke_p2_keep", "--layer", "0", "--run-id", "fresh_keep",
                     "--output-root", str(self.root / "runs"), "--config", str(self.config_path)]
        with patch.object(sys, "argv", arguments), patch("sys.stdout", new_callable=io.StringIO), \
                patch.object(smoke_p2_keep.subprocess, "run", side_effect=upstream), \
                patch.object(smoke_p2_keep, "freeze_example_inputs", side_effect=freeze), \
                patch.object(smoke_p2_keep, "extract_run", side_effect=extract):
            smoke_p2_keep.main()
        self.assertEqual(order, ["P0_P1_K", "freeze_keep", "C"])
        root = self.p1_dir.parent
        report = json.loads((root / "p2_keep_acceptance_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["stage_order"], ["P0", "P1", "P2_K", "P2_C"])
        for stage, filename in (("P0_P1", "acceptance_report.json"), ("P2_K", "p2_keys_acceptance_report.json")):
            self.assertEqual(report["upstream_reports"][stage]["sha256"], file_sha256(root / filename))

    def test_upstream_failure_stops_keep_experiment(self):
        def failed(command, **kwargs):
            (self.root / "runs/failed_keep/round_0").mkdir(parents=True)
            raise subprocess.CalledProcessError(1, command)

        arguments = ["smoke_p2_keep", "--layer", "0", "--run-id", "failed_keep",
                     "--output-root", str(self.root / "runs"), "--config", str(self.config_path)]
        with patch.object(sys, "argv", arguments), patch("sys.stdout", new_callable=io.StringIO), \
                patch.object(smoke_p2_keep.subprocess, "run", side_effect=failed), \
                patch.object(smoke_p2_keep, "freeze_example_inputs") as freeze, \
                patch.object(smoke_p2_keep, "extract_run") as extract:
            with self.assertRaises(subprocess.CalledProcessError):
                smoke_p2_keep.main()
        freeze.assert_not_called()
        extract.assert_not_called()
        report = json.loads((self.root / "runs/failed_keep/round_0/p2_keep_acceptance_report.json").read_text())
        self.assertEqual(report["status"], "failed")

    def test_c_failure_is_recorded_without_claiming_acceptance(self):
        with patch.object(sys, "argv", self.arguments), patch("sys.stdout", new_callable=io.StringIO), \
                patch.object(smoke_p2_keep, "freeze_example_inputs", side_effect=self.fake_freeze), \
                patch.object(smoke_p2_keep, "extract_run", side_effect=ValueError("artificial C failure")):
            with self.assertRaisesRegex(ValueError, "C failure"):
                smoke_p2_keep.main()
        report = json.loads((self.p1_dir.parent / "p2_keep_acceptance_report.json").read_text())
        self.assertEqual((report["status"], report["phase"]), ("failed", "keep_statistics"))
        self.assertNotIn("manifest_sha256", report)

    def test_existing_keep_inputs_are_preserved_without_starting_experiment(self):
        path = self.p1_dir.parent / "keep_inputs.json"
        path.write_bytes(b"existing keep artifact")
        with patch.object(sys, "argv", self.arguments), patch.object(smoke_p2_keep, "extract_run") as extract:
            with self.assertRaises(FileExistsError):
                smoke_p2_keep.main()
        extract.assert_not_called()
        self.assertEqual(path.read_bytes(), b"existing keep artifact")


if __name__ == "__main__":
    unittest.main()
