"""Small resident-model and real CUDA checks, without loading the 8B checkpoint."""

import hashlib
import io
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from MDLM.contracts import WritebackConfig, cached_file_hashes, file_sha256
from MDLM.keep_statistics import SecondMomentAccumulator
from MDLM.merge import WeightMerge
from MDLM.acceptance import cumulative_change
from MDLM.writeback import solve_writeback_factor
from util.gpu_runtime import ResidentModel, execute_stage, execution_context


def tiny_model(device="cpu"):
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.transformer = torch.nn.Module()
    block = torch.nn.Module()
    block.ff_out = torch.nn.Linear(3, 2, bias=False)
    model.model.transformer.blocks = torch.nn.ModuleList([block])
    return model.to(device).eval()


class ResidentRuntimeTests(unittest.TestCase):
    def test_accepted_weight_is_reset_before_reuse_without_reloading(self):
        model = tiny_model()
        loader = Mock(return_value=model)
        pool = ResidentModel(loader, [0], "cpu")
        self.addCleanup(pool.close)
        first = pool.load("original", device_map="auto")
        weight = first.model.transformer.blocks[0].ff_out.weight
        baseline = weight.detach().clone()
        with torch.no_grad():
            weight.add_(5)
        second = pool.load("original", device_map="auto")
        self.assertIs(first, second)
        torch.testing.assert_close(weight, baseline, atol=0, rtol=0)
        loader.assert_called_once()
        self.assertEqual(loader.call_args.kwargs["device_map"], "cpu")
        self.assertEqual(pool.reuse_count, 1)

    def test_different_baseline_loads_and_leaked_branch_stops_reuse(self):
        loader = Mock(side_effect=[tiny_model(), tiny_model()])
        pool = ResidentModel(loader, [0], "cpu")
        self.addCleanup(pool.close)
        first, second = pool.load("original"), pool.load("checkpoint")
        self.assertIsNot(first, second)
        self.assertEqual(loader.call_count, 2)
        module = second.model.transformer.blocks[0].ff_out
        module._mdlm_candidate_owner = object()
        try:
            with self.assertRaisesRegex(RuntimeError, "残差分支"):
                pool.load("checkpoint")
        finally:
            del module._mdlm_candidate_owner

    def test_stage_runs_in_same_process_and_restores_cli_on_failure(self):
        previous = sys.argv
        cwd = Path.cwd()
        stream = io.StringIO()

        def stage_main():
            self.assertEqual(sys.argv, ["constructed_stage", "--flag"])
            print("stage output")
            raise SystemExit(1)

        import subprocess
        with patch("util.gpu_runtime.importlib.import_module", return_value=SimpleNamespace(main=stage_main)), \
                patch("util.gpu_runtime.subprocess.run") as spawn:
            with self.assertRaises(subprocess.CalledProcessError):
                execute_stage(["constructed_stage", "--flag"], stream=stream, cwd=cwd, resident=True)
            spawn.assert_not_called()
        self.assertIs(sys.argv, previous)
        self.assertEqual(Path.cwd(), cwd)
        self.assertIn("stage output", stream.getvalue())

    def test_stage_logging_restores_transformers_handler_before_evaluation(self):
        from transformers.utils import logging as transformers_logging

        root = logging.getLogger("transformers")
        previous_handlers, previous_level, previous_propagate = list(root.handlers), root.level, root.propagate
        outside = io.StringIO()
        original_stdout = sys.stdout
        try:
            root.handlers.clear()
            with patch.object(transformers_logging, "_default_handler", None), \
                    patch.object(sys, "stderr", outside), tempfile.TemporaryDirectory() as temporary:
                def stage_main():
                    print("stage stdout")
                    print("stage stderr", file=sys.stderr)
                    transformers_logging.get_logger("transformers.stage_test").warning("stage warning")
                    if sys.argv[1] == "fail":
                        raise SystemExit(1)

                with patch("util.gpu_runtime.importlib.import_module", return_value=SimpleNamespace(main=stage_main)):
                    for index in range(3):
                        with (Path(temporary) / f"stage_{index}.log").open("w") as stream:
                            if index == 2:
                                import subprocess
                                with self.assertRaises(subprocess.CalledProcessError):
                                    execute_stage(["constructed_stage", "fail"], stream=stream,
                                                  cwd=Path.cwd(), resident=True)
                            else:
                                execute_stage(["constructed_stage", str(index)], stream=stream,
                                              cwd=Path.cwd(), resident=True)
                        self.assertIs(sys.stdout, original_stdout)
                        self.assertIs(sys.stderr, outside)
                        self.assertIn("stage stdout", (Path(temporary) / f"stage_{index}.log").read_text())
                        self.assertIn("stage stderr", (Path(temporary) / f"stage_{index}.log").read_text())
                        self.assertIn("stage warning", (Path(temporary) / f"stage_{index}.log").read_text())
                handler = transformers_logging._default_handler
                self.assertIs(handler.stream, outside)
                self.assertIs(getattr(handler.flush, "__self__", None), outside)
                from util import evaluate_accuracy as accuracy
                run_dir = Path(temporary)
                inputs = run_dir / "evaluation_inputs.json"
                inputs.write_text("{}")
                (run_dir / "effective_config.json").write_text('{"target_layers": [0], "execution": {"device": "cpu"}}')
                (run_dir / "run_report.json").write_text(
                    '{"status": "complete", "mode": "independent", "original_model": "unused", '
                    '"rounds": [{"round_id": 0, "case_id": "case", "fact_ids": ["case"], '
                    '"baseline_model": "unused", "decision": "rejected", "final_model_dir": null}]}')
                probe = {"case_id": "case", "probe_id": "probe", "role": "independent_test"}

                def load_tokenizer(**kwargs):
                    transformers_logging.get_logger("transformers.evaluation_test").warning("evaluation warning")
                    return SimpleNamespace(load_model=lambda *args, **kwargs: object())

                with patch.object(accuracy, "evaluation_inputs", return_value=(inputs, [probe])), \
                        patch.object(accuracy.LLaDA8BInstructAdapter, "from_local_tokenizer",
                                     side_effect=load_tokenizer), \
                        patch.object(accuracy, "score_probe", return_value={**probe, "correct": True}):
                    self.assertEqual(accuracy.evaluate_run(run_dir)["status"], "complete")
                self.assertIn("evaluation warning", outside.getvalue())
        finally:
            root.handlers[:] = previous_handlers
            root.setLevel(previous_level)
            root.propagate = previous_propagate

    def test_hash_cache_reuses_unchanged_file_and_detects_rewrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "weights"
            path.write_bytes(b"first")
            with cached_file_hashes(), patch("MDLM.contracts.hashlib.sha256", wraps=hashlib.sha256) as digest:
                first = file_sha256(path)
                self.assertEqual(first, file_sha256(path))
                self.assertEqual(digest.call_count, 1)
                path.write_bytes(b"other")
                self.assertNotEqual(first, file_sha256(path))
                self.assertEqual(digest.call_count, 2)

    def test_hash_cache_does_not_survive_context_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.safetensors"
            path.write_bytes(b"weights")
            with cached_file_hashes():
                expected = file_sha256(path)
            with cached_file_hashes(), patch("MDLM.contracts.hashlib.sha256", wraps=hashlib.sha256) as digest:
                self.assertEqual(file_sha256(path), expected)
                self.assertEqual(file_sha256(path), expected)
                digest.assert_called_once()
            with patch("MDLM.contracts.hashlib.sha256", wraps=hashlib.sha256) as digest:
                self.assertEqual(file_sha256(path), expected)
                digest.assert_called_once()

    def test_cuda_unavailable_fails_before_loading_instead_of_falling_back(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "无法使用 CUDA"):
                with execution_context({"backend": "resident", "device": "cuda:0"}, [0]):
                    self.fail("unreachable")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class CudaNumericsTests(unittest.TestCase):
    def test_resident_cuda_stages_share_one_load_and_reset_edited_weights(self):
        from model.adapters.llada_8b_instruct import LLaDA8BInstructAdapter
        model = tiny_model("cuda:0")
        weight = model.model.transformer.blocks[0].ff_out.weight
        baseline = weight.detach().clone()

        def stage_main():
            loaded = LLaDA8BInstructAdapter.load_model("constructed_gpu_model", torch_dtype="auto")
            self.assertIs(loaded, model)
            self.assertTrue(all(parameter.device.type == "cuda" for parameter in loaded.parameters()))
            torch.testing.assert_close(weight, baseline, atol=0, rtol=0)
            if sys.argv[1] == "mutate":
                with torch.no_grad():
                    weight.add_(1)

        with patch.object(LLaDA8BInstructAdapter, "load_model", return_value=model) as loader:
            with execution_context({"backend": "resident", "device": "cuda:0"}, [0]) as pool:
                with patch("util.gpu_runtime.importlib.import_module", return_value=SimpleNamespace(main=stage_main)):
                    for step in ("mutate", "next_case"):
                        execute_stage(["constructed_gpu_stage", step], stream=io.StringIO(), cwd=Path.cwd(), resident=True)
                self.assertEqual((pool.load_count, pool.reuse_count), (1, 1))
            loader.assert_called_once()

    def test_gpu_c_and_q_match_independent_cpu_equations(self):
        values = torch.tensor([[1., 2., 3.], [2., 0., -1.]], dtype=torch.float64)
        accumulator = SecondMomentAccumulator(3, device="cuda:0")
        accumulator.add(values.cuda(), [0.5, 0.25])
        expected_c = values.T @ torch.diag(torch.tensor([0.5, 0.25], dtype=torch.float64)) @ values
        self.assertEqual(accumulator.matrix.device.type, "cuda")
        torch.testing.assert_close(accumulator.matrix.cpu(), expected_c, atol=1e-12, rtol=1e-12)
        self.assertTrue(accumulator.diagnose(accumulator.matrix.float(), chunk_size=2)["all_checks_passed"])
        keys = torch.tensor([[1.], [0.], [0.5]], dtype=torch.float64)
        settings = WritebackConfig({"layer": .01}, device="cuda:0", storage_dtype="float64", chunk_size=2)
        q, report = solve_writeback_factor(keys, expected_c, .01, settings)
        expected_q = torch.linalg.solve(expected_c + keys @ keys.T + .01 * torch.eye(3, dtype=torch.float64), keys).T
        torch.testing.assert_close(q, expected_q, atol=1e-11, rtol=1e-11)
        self.assertEqual(report["device"], "cuda:0")
        self.assertLess(report["normal_equation_relative_residual"], 1e-10)

    def test_cuda_statistics_artifact_passes_independent_streamed_validation(self):
        from dataclasses import replace
        from tests.test_keep_statistics_pipeline import KeepStatisticsPipelineTests
        from model.adapters.llada_8b_instruct import LLaDA8BInstructAdapter
        from util.validate_matrix_artifacts import validate_run
        fixture = KeepStatisticsPipelineTests()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        fixture.settings = replace(fixture.settings, device="cuda:0")
        fixture.extract()
        with patch.object(LLaDA8BInstructAdapter, "from_local_tokenizer", return_value=fixture.adapter), \
                patch.object(LLaDA8BInstructAdapter, "load_model", return_value=fixture.model):
            result = validate_run(fixture.p1_dir)
        self.assertEqual(result["status"], "complete")

    def test_gpu_merge_and_parameter_gate_match_cpu_and_recover_exactly(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=False)).double().cuda().eval()
        original = model[0].weight.detach().cpu().clone()
        q = {"0": torch.tensor([[1., 2., 3.]], dtype=torch.float64)}
        r = {"0": torch.tensor([[.01], [.02]], dtype=torch.float64)}
        merger = WeightMerge(model, q, r, compute_dtype="float64", chunk_size=2)
        self.assertEqual(merger.increments["0"].device.type, "cuda")
        merger.apply(.5)
        expected = original + .5 * r["0"] @ q["0"]
        torch.testing.assert_close(model[0].weight.cpu(), expected, atol=1e-12, rtol=1e-12)
        change = cumulative_change(merger, {"0": original}, 1e-12)
        self.assertAlmostEqual(change, float((expected - original).norm() / original.norm()), places=11)
        merger.diagnostics(q, r, {"0": torch.ones(3, 1, dtype=torch.float64)})
        merger.restore()
        self.assertTrue(torch.equal(model[0].weight.cpu(), original))


if __name__ == "__main__":
    unittest.main()
