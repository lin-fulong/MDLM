"""P2 bundle acceptance with a real tiny CPU model; no 8B weights are loaded."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
    from safetensors.torch import load_file, save_file
except ImportError:
    torch = None

from tests import test_keep_statistics_pipeline as keep_tests
from model.adapters.llada_8b_instruct import LLaDA8BInstructAdapter
from MDLM.contracts import KeepStatisticsConfig, config_digest, file_sha256
from model.base import ForwardCapture
from util.validate_matrix_artifacts import check_matrix, validate_run


@unittest.skipIf(torch is None, "PyTorch and safetensors are required for P2 bundle tests")
class MatrixArtifactTests(unittest.TestCase):
    def setUp(self):
        fixture = keep_tests.KeepStatisticsPipelineTests()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        fixture.extract()
        self.fixture = fixture
        self.p1_dir = fixture.p1_dir
        self.root = self.p1_dir.parent / "matrices"
        self.output = self.root / "validation"

    def validate(self):
        with patch.object(LLaDA8BInstructAdapter, "from_local_tokenizer", return_value=self.fixture.adapter), \
                patch.object(LLaDA8BInstructAdapter, "load_model", return_value=self.fixture.model):
            return validate_run(self.p1_dir)

    def change_json(self, path: Path, change):
        record = json.loads(path.read_text(encoding="utf-8"))
        change(record)
        path.write_text(json.dumps(record), encoding="utf-8")

    def test_two_layer_bundle_replays_and_recomputes_c(self):
        before = {name: value.detach().clone() for name, value in self.fixture.model.named_parameters()}
        result = self.validate()
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["example_only"])
        self.assertFalse(result["p2_stage_complete"])
        manifest = json.loads((self.output / "manifest.json").read_text())
        report = json.loads((self.output / "qa_report.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(report["real_model_forward_count"], 6)  # 两条编辑初态，两个层各两条保持输入。
        self.assertEqual(report["layer_count"], 2)
        self.assertFalse(report["full_spectrum_checked"])
        self.assertTrue(all(row["replayed_vector_count"] == 4 for row in report["layers"].values()))
        for name, value in self.fixture.model.named_parameters():
            self.assertTrue(torch.equal(value, before[name]))
        with self.assertRaises(FileExistsError):
            self.validate()

    def test_changed_c_file_is_rejected_before_model_loading(self):
        manifest = json.loads((self.root / "keep_statistics/manifest.json").read_text())
        tensor = self.root / "keep_statistics" / next(iter(manifest["layers"].values()))["tensor_file"]
        tensor.write_bytes(tensor.read_bytes() + b"changed")
        with patch.object(LLaDA8BInstructAdapter, "load_model") as load:
            with self.assertRaisesRegex(ValueError, "artifact digest differs"):
                self.validate()
            load.assert_not_called()
        self.assertEqual(json.loads((self.output / "manifest.json").read_text())["status"], "failed")

    def test_mixed_baseline_is_rejected_before_model_loading(self):
        path = self.root / "keep_statistics/manifest.json"
        self.change_json(path, lambda record: record.update(baseline_id="another_baseline"))
        with patch.object(LLaDA8BInstructAdapter, "load_model") as load:
            with self.assertRaisesRegex(ValueError, "run, baseline"):
                self.validate()
            load.assert_not_called()

    def test_example_purpose_cannot_be_relabelled_as_formal(self):
        path = self.root / "keep_statistics/manifest.json"
        self.change_json(path, lambda record: record.update(example_only=False))
        with self.assertRaisesRegex(ValueError, "example-only purpose differs"):
            self.validate()

    def test_changed_column_order_is_rejected(self):
        path = self.root / "reference_keys/manifest.json"
        self.change_json(path, lambda record: record["columns"].reverse())
        # C's K-manifest binding must be updated too; column validation still rejects the swap.
        keep_path = self.root / "keep_statistics/manifest.json"
        self.change_json(keep_path, lambda record: record["key_files_sha256"].update(
            {str(path): file_sha256(path)}))
        with self.assertRaisesRegex(ValueError, "K column order"):
            self.validate()

    def test_rehashed_wrong_c_is_rejected_by_independent_recomputation(self):
        path = self.root / "keep_statistics/manifest.json"
        manifest = json.loads(path.read_text())
        name = next(iter(manifest["layers"]))
        tensor_path = self.root / "keep_statistics" / manifest["layers"][name]["tensor_file"]
        values = load_file(str(tensor_path))
        values[name] = values[name] + torch.eye(len(values[name]))
        save_file(values, str(tensor_path))
        self.change_json(path, lambda record: record["layers"][name].update(
            tensor_file_sha256=file_sha256(tensor_path)))
        with self.assertRaisesRegex(ValueError, "trace differs"):
            self.validate()

    def test_c_qa_failure_is_rejected_even_with_matching_digest(self):
        report_path = self.root / "keep_statistics/qa_report.json"
        name = next(iter(json.loads(report_path.read_text())["layers"]))
        self.change_json(report_path, lambda record: record["layers"][name]["diagnostics"].update(
            all_checks_passed=False))
        manifest_path = self.root / "keep_statistics/manifest.json"
        self.change_json(manifest_path, lambda record: record.update(
            qa_report_sha256=file_sha256(report_path)))
        with self.assertRaisesRegex(ValueError, "C extraction QA failed"):
            self.validate()

    def test_source_change_during_replay_prevents_publication(self):
        source = self.root / "keep_statistics/qa_report.json"
        original = LLaDA8BInstructAdapter.forward_keep_with_capture
        changed = False

        def capture(adapter, model, model_input, positions):
            nonlocal changed
            result = original(adapter, model, model_input, positions)
            if not changed:
                source.write_bytes(source.read_bytes() + b"\n")
                changed = True
            return result

        with patch.object(LLaDA8BInstructAdapter, "forward_keep_with_capture", new=capture):
            with self.assertRaisesRegex(ValueError, "artifact digest differs"):
                self.validate()
        self.assertEqual(json.loads((self.output / "manifest.json").read_text())["status"], "failed")

    def test_key_replay_does_not_hide_double_precision_differences(self):
        path = self.root / "reference_keys/manifest.json"
        record = json.loads(path.read_text())
        record["settings"].update(replay_atol=0.0, replay_rtol=0.0)
        record["settings_sha256"] = config_digest(record["settings"])
        path.write_text(json.dumps(record), encoding="utf-8")
        self.change_json(self.root / "keep_statistics/manifest.json", lambda row: row["key_files_sha256"].update(
            {str(path): file_sha256(path)}))
        original = LLaDA8BInstructAdapter.forward_with_capture

        def changed(adapter, model, model_input):
            result = original(adapter, model, model_input)
            values = {name: value.double() for name, value in result.module_inputs.items()}
            value = next(iter(values.values()))
            index = value.abs().argmax()
            value[index] += value[index].abs() * 1e-9  # 小于 float32 舍入间隔，超过冻结的零容差。
            return ForwardCapture(result.mask_logits, values)

        with patch.object(LLaDA8BInstructAdapter, "forward_with_capture", new=changed):
            with self.assertRaisesRegex(ValueError, "K replay differs"):
                self.validate()


@unittest.skipIf(torch is None, "PyTorch and safetensors are required for matrix numeric tests")
class MatrixNumericTests(unittest.TestCase):
    def test_finite_large_projected_moment_does_not_overflow_symmetrization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            key_path, keep_path = root / "k.safetensors", root / "c.safetensors"
            save_file({"layer": torch.ones(1, 1, dtype=torch.float64)}, str(key_path))
            save_file({"layer": torch.full((1, 1), 1e308, dtype=torch.float64)}, str(keep_path))
            expected = {"vectors": torch.ones(1, 1, dtype=torch.float64), "indices": torch.tensor([0]),
                        "direct": torch.full((1, 1), 1e308, dtype=torch.float64),
                        "projection": torch.full((1, 1), 1e308, dtype=torch.float64),
                        "trace": 1e308, "vector_count": 1}
            report = check_matrix("layer", keep_path, key_path, 1, 1, "torch.float64", "float64",
                                  KeepStatisticsConfig("constructed_large_moment"), expected)
            self.assertEqual(report["projected_eigenvalue_min"], 1e308)
            json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
