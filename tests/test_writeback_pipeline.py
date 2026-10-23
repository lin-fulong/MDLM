"""P3 artifact integration with accepted two-layer tiny CPU P2 outputs."""

from __future__ import annotations

import io
import sys
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
from safetensors.torch import load_file

from MDLM.contracts import WRITEBACK_FACTOR_VERSION, WritebackConfig, file_sha256
from model.adapters.llada_8b_instruct import LLaDA8BInstructAdapter
from tests import test_matrix_artifacts as matrix_tests
from util import solve_writeback_factor as pipeline, smoke_p3_writeback


class WritebackPipelineTests(unittest.TestCase):
    def setUp(self):
        fixture = matrix_tests.MatrixArtifactTests()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        fixture.validate()
        self.fixture = fixture
        self.p1_dir, self.matrices = fixture.p1_dir, fixture.root
        self.output = self.matrices / "writeback_factors"
        self.settings = WritebackConfig({name: 0.3 + index * 0.2
                                        for index, name in enumerate(fixture.fixture.adapter.target_module_names())},
                                       allow_example_only=True, chunk_size=2)

    def solve(self, settings=None):
        with patch.object(LLaDA8BInstructAdapter, "load_model", side_effect=AssertionError("P3 must not load a model")):
            return pipeline.solve_run(self.p1_dir, settings or self.settings)

    def test_saved_q_matches_independent_solve_for_both_widths(self):
        result = self.solve()
        self.assertTrue(result["example_only"])
        self.assertFalse(result["p3_stage_complete"])
        manifest = pipeline.read_json(self.output / "manifest.json")
        report = pipeline.read_json(self.output / "qa_report.json")
        self.assertEqual((manifest["status"], report["status"]), ("complete", "complete"))
        self.assertEqual(manifest["artifact_version"], WRITEBACK_FACTOR_VERSION)
        self.assertEqual(report["model_forward_count"], 0)
        self.assertEqual(manifest["qa_report_sha256"], file_sha256(self.output / "qa_report.json"))
        c_manifest = pipeline.read_json(self.matrices / "keep_statistics/manifest.json")
        for name, info in manifest["layers"].items():
            k = load_file(str(self.matrices / "reference_keys/keys.safetensors"))[name].double()
            c = load_file(str(self.matrices / "keep_statistics" / c_manifest["layers"][name]["tensor_file"]))[name].double()
            q = load_file(str(self.output / info["tensor_file"]))[name]
            expected = torch.linalg.solve(c + k @ k.T + info["ridge"] * torch.eye(len(k)), k).T
            torch.testing.assert_close(q, expected.float(), atol=1e-7, rtol=1e-6)
            self.assertEqual(info["tensor_file_sha256"], file_sha256(self.output / info["tensor_file"]))
            self.assertTrue(report["layers"][name]["disk_round_trip_passed"])
            self.assertLess(report["layers"][name]["normal_equation_relative_residual"], 1e-8)
            self.assertLess(report["layers"][name]["stored_normal_equation_relative_residual"], 1e-5)
        with self.assertRaises(FileExistsError):
            self.solve()

    def test_example_requires_explicit_permission_in_frozen_config(self):
        with self.assertRaisesRegex(ValueError, "allow_example_only"):
            self.solve(replace(self.settings, allow_example_only=False))
        self.assertEqual(pipeline.read_json(self.output / "manifest.json")["status"], "failed")

    def test_incomplete_p2_qa_is_rejected(self):
        self.fixture.change_json(self.matrices / "validation/manifest.json", lambda item: item.update(status="failed"))
        with self.assertRaisesRegex(ValueError, "P2 unified acceptance is incomplete"):
            self.solve()

    def test_changed_matrix_is_rejected_before_solving(self):
        path = self.matrices / "reference_keys/keys.safetensors"
        path.write_bytes(path.read_bytes() + b"modified")
        with patch.object(pipeline, "solve_writeback_factor") as solve:
            with self.assertRaisesRegex(ValueError, "artifact digest differs"):
                self.solve()
            solve.assert_not_called()

    def test_mixed_baseline_and_missing_ridge_layer_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "ridge layer selection differs"):
            self.solve(replace(self.settings, ridge_by_layer={"wrong_layer": 0.3}))

    def test_mixed_p2_baseline_is_rejected(self):
        self.fixture.change_json(self.matrices / "validation/manifest.json", lambda item: item.update(baseline_id="another"))
        with self.assertRaisesRegex(ValueError, "cannot mix P2 rounds"):
            self.solve()

    def test_source_change_during_solve_prevents_publication(self):
        original = pipeline.solve_writeback_factor
        source = self.matrices / "validation/qa_report.json"

        def solve(*args):
            result = original(*args)
            source.write_bytes(source.read_bytes() + b"\n")
            return result

        with patch.object(pipeline, "solve_writeback_factor", side_effect=solve):
            with self.assertRaisesRegex(ValueError, "artifact digest differs"):
                self.solve()
        self.assertEqual(pipeline.read_json(self.output / "qa_report.json")["phase"], "source_recheck")
        self.assertFalse(pipeline.read_json(self.output / "manifest.json")["p3_stage_complete"])

    def test_save_failure_retains_diagnostics(self):
        with patch("safetensors.torch.save_file", side_effect=OSError("artificial Q save failure")):
            with self.assertRaisesRegex(OSError, "Q save failure"):
                self.solve()
        report = pipeline.read_json(self.output / "qa_report.json")
        self.assertEqual((report["status"], report["phase"]), ("failed", "save_and_reload"))
        self.assertTrue(report["layers"])

    def test_disk_dtype_mismatch_is_rejected(self):
        def wrong_precision(path, **kwargs):
            return {name: value.double() for name, value in load_file(path, **kwargs).items()}

        with patch("safetensors.torch.load_file", side_effect=wrong_precision):
            with self.assertRaisesRegex(ValueError, "round-trip exactly"):
                self.solve()

    def test_disk_value_mismatch_with_correct_dtype_is_rejected(self):
        def wrong_values(path, **kwargs):
            return {name: value + 1 for name, value in load_file(path, **kwargs).items()}

        with patch("safetensors.torch.load_file", side_effect=wrong_values):
            with self.assertRaisesRegex(ValueError, "round-trip exactly"):
                self.solve()

    def test_normal_equation_failure_retains_layer_diagnostics(self):
        with patch("MDLM.writeback.relative_residual", side_effect=[0.0, 1.0]):
            with self.assertRaisesRegex(ValueError, "normal equation residual"):
                self.solve()
        report = pipeline.read_json(self.output / "qa_report.json")
        self.assertEqual(report["status"], "failed")
        row = next(iter(report["layers"].values()))
        self.assertEqual(row["normal_equation_relative_residual"], 1.0)
        self.assertFalse(list(self.output.glob("q_layer_*.safetensors")))

    def test_settings_mutation_during_solve_prevents_publication(self):
        original = pipeline.solve_writeback_factor
        name = next(iter(self.settings.ridge_by_layer))

        def solve(*args):
            result = original(*args)
            self.settings.ridge_by_layer[name] *= 2
            return result

        with patch.object(pipeline, "solve_writeback_factor", side_effect=solve):
            with self.assertRaisesRegex(ValueError, "settings changed during solve"):
                self.solve()
        self.assertEqual(pipeline.read_json(self.output / "manifest.json")["status"], "failed")

    def test_smoke_uses_completed_p2_and_reports_zero_model_forwards(self):
        config_path = self.p1_dir.parent / "p3_config.json"
        pipeline.write_json(config_path, self.settings.as_record())
        arguments = ["smoke_p3_writeback", "--p1-dir", str(self.p1_dir), "--config", str(config_path)]
        with patch.object(sys, "argv", arguments), patch("sys.stdout", new_callable=io.StringIO):
            smoke_p3_writeback.main()
        report = pipeline.read_json(self.p1_dir.parent / "p3_writeback_acceptance_report.json")
        self.assertEqual((report["status"], report["model_forward_count"]), ("passed", 0))
        self.assertFalse(report["p3_stage_complete"])
        with patch.object(sys, "argv", arguments), self.assertRaises(FileExistsError):
            smoke_p3_writeback.main()

    def test_smoke_retains_failure_report_without_claiming_acceptance(self):
        arguments = ["smoke_p3_writeback", "--p1-dir", str(self.p1_dir)]
        with patch.object(sys, "argv", arguments), \
                patch.object(smoke_p3_writeback, "solve_run", side_effect=ValueError("artificial P3 failure")):
            with self.assertRaisesRegex(ValueError, "artificial P3 failure"):
                smoke_p3_writeback.main()
        report = pipeline.read_json(self.p1_dir.parent / "p3_writeback_acceptance_report.json")
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["p3_stage_complete"])


if __name__ == "__main__":
    unittest.main()
