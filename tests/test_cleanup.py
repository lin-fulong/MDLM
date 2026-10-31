"""清理异常语句后的算法回归：微型模型、已提交基线和失败恢复。"""

import ast
import csv
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
import importlib
import io
from itertools import product
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import torch

from test_evaluation import AdapterModel, ByteTokenizer
from model.adapters.llada_8b_instruct import LLaDA8BInstructAdapter
from model.base import KeepModelInput, ModelAdapter
from util.nethook import register_output_hooks
from util.io import read_json, write_json
from util.experiment_log import ExperimentLog
from util.resource_usage import EditResourceUsage


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "c505588"


def committed_module(path):
    """读取清理前已提交源码作数值对照，不写入工作区。"""
    name = "_cleanup_before_" + path.replace("/", "_").removesuffix(".py")
    if name not in sys.modules:
        source = subprocess.check_output(["git", "show", f"{BASELINE}:{path}"], cwd=ROOT).decode("utf-8")
        module = ModuleType(name)
        module.__file__ = str(ROOT / path)
        sys.modules[name] = module
        exec(compile(source, f"{BASELINE}:{path}", "exec"), module.__dict__)
    return sys.modules[name]


class OffsetTokenizer(ByteTokenizer):
    def __call__(self, text, *, return_offsets_mapping=False, **kwargs):
        # ASCII 偏移用于轻量回归，真实 tokenizer 的边界需在实际模型环境核验。
        result = {"input_ids": self.encode(text)}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result


def fixture(adapter_class=LLaDA8BInstructAdapter):
    model = AdapterModel()
    adapter = adapter_class(OffsetTokenizer(), mask_token_id=0, target_layers=(0,))
    cases = [{"case_id": "f0", "fact_core": {"subject": "Alice"}, "prompts": {
        "edit": [{"probe_id": "e", "prompt": "Alice?", "answer": "B"}],
        "preservation": [{"probe_id": str(i), "prompt": f"Bob {i}?", "answer": "A"} for i in range(4)],
    }}]
    return model, adapter, cases


class AlgorithmRegressionTests(unittest.TestCase):
    def setUp(self):
        # 每项测试隔离临时基线和 MDLM 导入，保留独立评价的依赖边界检查。
        self.modules = {name: value for name, value in sys.modules.items()
                        if name == "MDLM" or name.startswith(("MDLM.", "_cleanup_before_"))}

    def tearDown(self):
        # 保留 PyTorch 延迟加载的模块，避免重复注册其底层算子。
        for name in list(sys.modules):
            if name == "MDLM" or name.startswith(("MDLM.", "_cleanup_before_")):
                del sys.modules[name]
        sys.modules.update(self.modules)

    def test_experiment_log_tees_terminal_streams_and_summary(self):
        with tempfile.TemporaryDirectory(prefix="mdlm_log_") as temporary:
            root = Path(temporary)
            log = ExperimentLog(root / "runs/edit", algorithm="mdlm", model="base",
                                dataset="cases.jsonl", kind="edit", parameters={"mode": "independent"})
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr), log.capture():
                print("stdout marker")
                print("stderr marker", file=sys.stderr)
                log.finish({"status": "complete", "case_count": 1})
            content = log.path.read_text(encoding="utf-8")
            self.assertIn("stdout marker", stdout.getvalue())
            self.assertIn("stderr marker", stderr.getvalue())
            self.assertIn("stdout marker", content)
            self.assertIn("stderr marker", content)
            self.assertIn("status: complete", content)

            failed = ExperimentLog(root / "runs/failed", algorithm="mdlm", model="base",
                                   dataset="cases.jsonl", kind="edit", parameters={"mode": "independent"})
            with self.assertRaises(ValueError):
                with failed.capture():
                    int("not an integer")
            failure_content = failed.path.read_text(encoding="utf-8")
            self.assertIn("[uncaught_exception]", failure_content)
            self.assertIn("ValueError", failure_content)
            self.assertIn("status: failed", failure_content)

    def test_edit_resource_csv_cpu_and_failure(self):
        with tempfile.TemporaryDirectory(prefix="mdlm_resource_") as temporary:
            output = Path(temporary) / "runs" / "edit"
            monitor = EditResourceUsage(output, use_cuda=False)
            monitor.start_case("source_1")
            monitor.finish_case()
            self.assertTrue((output / "result/resource_report.partial.csv").is_file())
            monitor.start_case("source_2")
            result = monitor.finish("failed")
            with Path(result["resource_csv_file"]).open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["item"] for row in rows], ["total", "source_1", "source_2"])
            self.assertEqual([row["status"] for row in rows], ["failed", "complete", "failed"])
            self.assertEqual(rows[0]["total_samples"], "2")
            self.assertEqual(rows[0]["peak_vram_gib"], "")
            self.assertGreaterEqual(float(rows[0]["elapsed_seconds"]), 0)
            self.assertFalse((output / "result/resource_report.partial.csv").exists())

    def test_no_explicit_exception_statements(self):
        paths = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.py"],
                                        cwd=ROOT).decode("utf-8").splitlines()
        found = []
        for path in paths:
            file = ROOT / path
            if file.is_file():
                found.extend((path, node.lineno) for node in ast.walk(ast.parse(file.read_text(encoding="utf-8-sig")))
                             if isinstance(node, (ast.Raise, ast.Assert)))
        self.assertEqual(found, [])

    def test_state_sampling_alignment_and_supervision_match_baseline(self):
        current = importlib.import_module("MDLM.states")
        old = committed_module("MDLM/states.py")
        previous_adapter = committed_module("model/adapters/llada_8b_instruct.py").LLaDA8BInstructAdapter
        for temperature in (0, 0.7):
            for steps in (None, 1, 2):
                model, adapter, cases = fixture()
                cases[0]["prompts"]["edit"][0]["answer"] = "BC"
                old_model, old_adapter, old_cases = fixture(previous_adapter)
                old_cases[0]["prompts"]["edit"][0]["answer"] = "BC"
                before = old.collect_trajectories(old_model, old_adapter, old_cases, steps=steps,
                                                  temperature=temperature)
                after = current.collect_trajectories(model, adapter, cases, steps=steps,
                                                     temperature=temperature)
                self.assertEqual([asdict(t) for t in before], [asdict(t) for t in after])
                self.assertEqual(old.build_supervision(before, 0), current.build_supervision(after, 0))
                prepared = [module.prepare_inputs(cases, adapter) for module in (old, current)]
                for values in prepared:
                    values["development_probes"] = [asdict(item) for item in values["development_probes"]]
                self.assertEqual(*prepared)

    def test_both_evaluation_protocols_match_baseline(self):
        from evaluation import inference
        from test_evaluation import Adapter, TinyModel, TARGET

        old = committed_module("evaluation/inference.py")
        for protocol in (inference.PROTOCOL, inference.GOLD_PROTOCOL):
            for steps in (None, 1, 2, 9):
                for temperature in (0, 0.7, 1.5):
                    for edited, dtype in product((False, True), (torch.float32, torch.float16, torch.bfloat16)):
                        results, inputs = [], []
                        for module in (old, inference):
                            model, adapter = TinyModel(), Adapter()
                            model.layer.weight.data.fill_(float(edited))
                            forward = adapter.forward_logits
                            adapter.forward_logits = lambda *args, fn=forward, dt=dtype: fn(*args).to(dt)
                            prediction = module.generate(model, adapter, module.GenerationRequest((81, 124), 3, 2026),
                                steps=steps, temperature=temperature, protocol=protocol,
                                target=TARGET if protocol == inference.GOLD_PROTOCOL else None)
                            results.append(asdict(prediction))
                            inputs.append(adapter.inputs)
                        self.assertEqual(*results)
                        self.assertEqual(*inputs)

    def test_keys_second_moment_factor_optimization_and_merge_match_baseline(self):
        from MDLM import merge, optimize, states, writeback

        old_modules = {name: committed_module(f"MDLM/{name}.py") for name in ("states", "writeback", "optimize", "merge")}
        previous_adapter = committed_module("model/adapters/llada_8b_instruct.py").LLaDA8BInstructAdapter
        outputs = []
        for modules, adapter_class in ((old_modules, previous_adapter),
                                       ({"states": states, "writeback": writeback, "optimize": optimize, "merge": merge},
                                        LLaDA8BInstructAdapter)):
            s, w, o, m = (modules[name] for name in ("states", "writeback", "optimize", "merge"))
            model, adapter, cases = fixture(adapter_class)
            name = adapter.target_module_names()[0]
            prepared = s.prepare_inputs(cases, adapter)
            trajectories = s.collect_trajectories(model, adapter, cases)
            facts, supervision = s.build_supervision(trajectories, 0)
            edits = s.edit_training_examples(facts)
            keys = w.extract_keys(model, adapter, trajectories)
            matrices = dict(w.keep_matrices(model, adapter, prepared["statistics"]))
            factors = {name: w.solve_writeback_factor(keys[name], matrices[name], 0.01,
                                                     w.WritebackConfig({name: 0.01}))}
            originals = {name: model.get_submodule(name).weight.detach().clone()}
            keep_input = KeepModelInput((66, 0), (1, 1), (1,))
            reference = o.log_probabilities(adapter.forward_logits(model, keep_input, (1,))).detach()
            keeps = [o.KeepTrainingExample(keep_input, (1,), (1.,), reference)]
            settings = o.ResidualOptimizationConfig({name: 0.001}, beta=0.3, max_steps=4,
                                                     checkpoints=(1, 2, 4), learning_rate=0.1)
            with o.CandidateModel(model, factors) as candidate:
                result = o.optimize_residuals(candidate, adapter, edits, keeps, settings)
            self.assertIsNotNone(result.residuals)
            merger = m.WeightMerge(model, factors, result.residuals)
            accepted = m.select_merged_candidate(merger, adapter, factors, result.residuals, edits, keeps,
                                                 originals, m.AcceptanceConfig(delta=0, tau_g=10, tau_u=1e12))
            self.assertEqual(accepted["status"], "accepted")
            outputs.append((keys, matrices, factors, result, accepted, model.get_submodule(name).weight.detach().clone()))
            self.assertFalse(model.get_submodule(name)._forward_hooks)
        before, after = outputs
        for i in range(3):
            for name, tensor in before[i].items():
                self.assertTrue(torch.equal(tensor, after[i][name]))
        self.assertEqual(before[3].report, after[3].report)
        for name, tensor in before[3].residuals.items():
            self.assertTrue(torch.equal(tensor, after[3].residuals[name]))
        self.assertEqual(before[4], after[4])
        self.assertTrue(torch.equal(before[5], after[5]))

    def test_hook_and_candidate_installation_failures_restore_state(self):
        from MDLM.optimize import CandidateModel

        model, adapter, _ = fixture()
        name = adapter.target_module_names()[0]
        layer = model.get_submodule(name)
        original = list((p, p.requires_grad) for p in model.parameters())
        model.train()
        with self.assertRaises(AttributeError):
            register_output_hooks(model, {name: lambda *args: None, "missing": lambda *args: None})
        self.assertFalse(layer._forward_hooks)
        with patch("MDLM.optimize.register_output_hooks", Mock(side_effect=RuntimeError("安装失败"))):
            with self.assertRaisesRegex(RuntimeError, "安装失败"):
                CandidateModel(model, {name: torch.eye(2)})
        self.assertTrue(model.training)
        self.assertEqual([p.requires_grad for p, _ in original], [flag for _, flag in original])
        self.assertFalse(hasattr(layer, "_mdlm_candidate_owner"))
        self.assertFalse(layer._forward_hooks)

    def test_edit_runner_and_evaluation_end_to_end(self):
        import main
        from evaluation import inference
        from evaluation.runner import run_evaluation

        for mode in ("independent", "sequential"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(prefix="mdlm_full_") as temporary:
                root = Path(temporary)
                model, adapter, cases = fixture()
                name = adapter.target_module_names()[0]
                cases.append({**cases[0], "case_id": "f1"})
                source = root / "cases.json"
                write_json(source, cases)
                config = root / "config.json"
                write_json(config, {"dataset_config": str(ROOT / "configs/datasets/cases.json"),
                    "editor_config": str(ROOT / "configs/editors/mdlm.json"),
                    "model_config": str(ROOT / "configs/models/llada_8b_instruct.json"),
                    "dataset_path": str(source), "model_path": str(root / "base"),
                    "target_layers": [0], "torch_dtype": "float32", "mode": mode,
                    "execution": {"backend": "resident", "device": "cpu"}, "output_root": str(root / "runs"),
                    "ridge": 0.01, "rho": 0.001, "optimization_steps": 4, "selection_steps": [1, 4],
                    "learning_rate": 0.1, "tau_u": 1e14, "tau_g": 10})
                adapter.load_model = Mock(return_value=model)
                with patch("MDLM.runner.get_adapter", return_value=adapter), redirect_stdout(io.StringIO()):
                    main.main(["--config", str(config), "--run-id", "edit"])
                report = read_json(root / "runs/edit/run_report.json")
                self.assertEqual(report["status"], "complete")
                with Path(report["resource_csv_file"]).open(encoding="utf-8-sig", newline="") as stream:
                    resource_rows = list(csv.DictReader(stream))
                self.assertEqual([row["item"] for row in resource_rows], ["total", "f0", "f1"])
                self.assertEqual(resource_rows[0]["total_samples"], "2")
                self.assertEqual(resource_rows[0]["status"], "complete")
                self.assertEqual(report["case_count"], 2)
                self.assertEqual(report["accepted_count"], 2)
                self.assertEqual(read_json(root / "runs/edit/run_report.json"), report)
                edit_log = Path(report["log_file"])
                self.assertEqual(edit_log.parent, root / "logs/mdlm/base/cases")
                self.assertIn(f"mode-{mode}", edit_log.name)
                self.assertIn("round:", edit_log.read_text(encoding="utf-8"))
                if mode == "independent":
                    self.assertTrue(torch.equal(model.get_submodule(name).weight, torch.zeros(2, 2)))
                else:
                    self.assertTrue(torch.count_nonzero(model.get_submodule(name).weight))
                for protocol in (inference.PROTOCOL, inference.GOLD_PROTOCOL):
                    runtime = AdapterModel()
                    adapter.load_model = Mock(return_value=runtime)
                    evaluation_config = root / "evaluate.json"
                    write_json(evaluation_config, {"protocol": protocol, "groups": ["edit", "preservation"],
                        "device": "cpu", "output_root": str(root / "evaluations")})
                    with patch("evaluation.inference.get_adapter", return_value=adapter), redirect_stdout(io.StringIO()):
                        result = run_evaluation(evaluation_config, run_dir=root / "runs/edit", eval_id=protocol)
                    self.assertEqual(result["status"], "complete")
                    self.assertEqual(result["coverage"], 1)
                    self.assertEqual(result["completed_tasks"], 10)
                    evaluation_log = Path(result["log_file"])
                    self.assertEqual(evaluation_log.parent, root / "logs/mdlm/base/cases")
                    self.assertIn(f"protocol-{protocol}", evaluation_log.name)
                    self.assertIn("completed_tasks: 10", evaluation_log.read_text(encoding="utf-8"))
                    self.assertTrue(torch.equal(runtime.get_submodule(name).weight, torch.zeros(2, 2)))

    def test_edit_runner_failure_report_and_round_rollback(self):
        from MDLM import runner

        for mode in ("independent", "sequential"):
            for phase in ("round", "context"):
                with self.subTest(mode=mode, phase=phase), tempfile.TemporaryDirectory(prefix="mdlm_failure_") as temporary:
                    root = Path(temporary)
                    model, adapter, cases = fixture()
                    name = adapter.target_module_names()[0]
                    source = root / "cases.json"
                    write_json(source, cases)
                    config = root / "config.json"
                    write_json(config, {"dataset_path": str(source), "model_path": str(root / "base"),
                        "target_layers": [0], "mode": mode, "output_root": str(root / "runs"),
                        "execution": {"backend": "resident", "device": "cpu"}, "ridge": 0.01, "rho": 0.001,
                        "optimization_steps": 1, "selection_steps": [1]})
                    adapter.load_model = Mock(return_value=model)

                    def failed_round(*args, **kwargs):
                        with torch.no_grad():
                            model.get_submodule(name).weight.fill_(42)
                        Mock(side_effect=RuntimeError("编辑失败"))()

                    @contextmanager
                    def failed_context(settings):
                        yield
                        Mock(side_effect=RuntimeError("环境恢复失败"))()

                    with patch("MDLM.runner.get_adapter", return_value=adapter), redirect_stdout(io.StringIO()):
                        operation = (patch("MDLM.runner.edit_round", side_effect=failed_round) if phase == "round"
                                     else patch("MDLM.runner.execution_context", side_effect=failed_context))
                        with operation, self.assertRaisesRegex(RuntimeError, "编辑失败|环境恢复失败"):
                            runner.run_dataset(config, run_id_override="failure")
                    report = read_json(root / "runs/failure/run_report.json")
                    self.assertEqual(report["status"], "failed")
                    self.assertIn("失败", report["failure"])
                    self.assertIn("status: failed", Path(report["log_file"]).read_text(encoding="utf-8"))
                    if phase == "round":
                        self.assertEqual(report["current_case_id"], "f0")
                        self.assertTrue(torch.equal(model.get_submodule(name).weight, torch.zeros(2, 2)))
                    self.assertFalse(model.get_submodule(name)._forward_hooks)

    def test_alignment_and_abstract_contracts(self):
        from data.validation.alignment import align_edit_prompt

        tokenizer = OffsetTokenizer()
        alignment = align_edit_prompt(tokenizer, "Alice?", "Alice", "B")
        self.assertEqual(alignment.subject_positions, tuple(range(5)))
        self.assertEqual(alignment.answer_ids, (67,))
        with self.assertRaises(SystemExit):
            align_edit_prompt(tokenizer, "Bob?", "Alice", "B")
        with self.assertRaises(SystemExit):
            align_edit_prompt(tokenizer, "Alice?", None, "B")
        with self.assertRaises(TypeError):
            ModelAdapter()
        self.assertIn("forward_logits", ModelAdapter.__abstractmethods__)

    def test_subject_ambiguity_and_unstable_answer_stop(self):
        from data.validation.alignment import align_edit_prompt
        from evaluation.inference import prepare_request
        from test_evaluation import Adapter, task

        # 对应真实 UltraEditBench 案例 0004695；不能取 What 的首字母作为实体 W。
        text = "What area does W include that is known for its residential neighborhoods and local amenities?"
        with self.assertRaisesRegex(SystemExit, "唯一"):
            align_edit_prompt(OffsetTokenizer(), text, "W", "A")
        adapter = Adapter()
        with patch.object(adapter.tokenizer, "encode", side_effect=[[81], [82, 65]]):
            with self.assertRaisesRegex(SystemExit, "边界"):
                prepare_request(adapter, task(), 2026)
        for answer in ("", "\x00", "\x7f"):
            with self.assertRaises(SystemExit):
                prepare_request(adapter, replace(task(), answer=answer), 2026)

    def test_keep_data_and_statistics_constraints(self):
        from MDLM import states, writeback, merge

        model, adapter, cases = fixture()
        for scale in (-1, 0, float("nan")):
            with self.assertRaisesRegex(SystemExit, "statistics_scale"):
                states.prepare_inputs(cases, adapter, settings={"statistics_scale": scale})
        for count in (0, 1):
            with self.assertRaisesRegex(SystemExit, "至少需要两道"):
                states.prepare_inputs(cases, adapter, keep_pool=cases[0]["prompts"]["preservation"][:count])
        with self.assertRaises(SystemExit):
            writeback.SecondMomentAccumulator(2).add(torch.ones(1, 2), [-1])
        with self.assertRaisesRegex(SystemExit, "非空开发"):
            merge.development_metrics(None, adapter, [], [], {}, 1e-12)

    def test_nan_sampling_and_invalid_trajectory_stop(self):
        from dataclasses import replace
        from MDLM import states, writeback
        from model.base import ForwardCapture

        model, adapter, cases = fixture()
        initial, _ = states.prepare_edit_input(adapter, cases[0], "e")
        with patch("MDLM.states.capture_edit", return_value=ForwardCapture(torch.full((1, 258), float("nan")), {})):
            with self.assertRaisesRegex(SystemExit, "采样概率"):
                states.sample_denoise(model, adapter, initial, steps=1, temperature=0, seed=2026)
        trajectory = states.collect_trajectories(model, adapter, cases)[0]
        bad = replace(trajectory, trace=replace(trajectory.trace, revealed_tokens=(((999, 65),),)))
        with self.assertRaisesRegex(SystemExit, "揭示记录"):
            states.classify_fact(bad, 0)
        name = adapter.target_module_names()[0]
        with torch.no_grad():
            model.get_submodule(name).weight.fill_(2)
        with self.assertRaisesRegex(SystemExit, "K 取键"):
            writeback.extract_keys(model, adapter, [trajectory])

    def test_semantic_stop_saves_report_and_restores_round(self):
        from MDLM import runner

        for mode in ("independent", "sequential", "misspelled"):
            with tempfile.TemporaryDirectory(prefix="mdlm_stop_") as temporary:
                root = Path(temporary)
                model, adapter, cases = fixture()
                name = adapter.target_module_names()[0]
                write_json(root / "cases.json", cases)
                write_json(root / "config.json", {"dataset_path": str(root / "cases.json"),
                    "model_path": str(root / "base"), "target_layers": [0], "mode": mode,
                    "output_root": str(root / "runs"), "ridge": 0.01, "rho": 0.001,
                    "execution": {"backend": "resident", "device": "cpu"}})
                def fail_round(*args, **kwargs):
                    with torch.no_grad():
                        model.get_submodule(name).weight.fill_(42)
                    sys.exit("模拟语义错误")
                adapter.load_model = Mock(return_value=model)
                with patch("MDLM.runner.get_adapter", return_value=adapter), \
                        patch("MDLM.runner.edit_round", side_effect=fail_round), redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        runner.run_dataset(root / "config.json", run_id_override="stop")
                report = read_json(root / "runs/stop/run_report.json")
                self.assertEqual(report["status"], "failed")
                self.assertIn("mode" if mode == "misspelled" else "模拟语义错误", report["failure"])
                self.assertTrue(torch.equal(model.get_submodule(name).weight, torch.zeros(2, 2)))

    def test_p5_numerical_equivalence_and_gate_rejections_match_baseline(self):
        from MDLM import merge, optimize, states

        previous = committed_module("MDLM/merge.py")
        previous_adapter = committed_module("model/adapters/llada_8b_instruct.py").LLaDA8BInstructAdapter

        class TrialModel(AdapterModel):
            def forward(self, *args, **kwargs):
                output = super().forward(*args, **kwargs)
                layer = self.model.transformer.blocks[0].ff_out
                owner = getattr(layer, "_mdlm_candidate_owner", None)
                value = float((owner.branches[self.name].r if owner else layer.weight)[1, 0].detach())
                if self.failure == "numeric" and value > 1.5:
                    output.logits[..., 66] = torch.nan
                elif self.failure == "equivalence" and owner is None and value > 1.5:
                    output.logits[..., 66] += 1
                elif self.failure == "fatal" and owner is None and value > 0:
                    self.fail()
                return output

        for failure in ("numeric", "equivalence", "gate", "fatal"):
            reports, weights = [], []
            for module, adapter_class in ((previous, previous_adapter), (merge, LLaDA8BInstructAdapter)):
                model = TrialModel()
                adapter = adapter_class(OffsetTokenizer(), mask_token_id=0, target_layers=(0,))
                name = adapter.target_module_names()[0]
                model.name, model.failure = name, failure
                model.fail = Mock(side_effect=RuntimeError("结构错误"))
                inputs = KeepModelInput((66, 0), (1, 1), (1,))
                edit = states.EditTrainingExample("f0", inputs, (1,), (67,), (1.,))
                logs = optimize.log_probabilities(adapter.forward_logits(model, inputs, (1,))).detach()
                keep = optimize.KeepTrainingExample(inputs, (1,), (1.,), logs)
                factors = {name: torch.eye(2)}
                residuals = {name: torch.tensor([[0., 0.], [2., 0.]])}
                original = {name: model.get_submodule(name).weight.detach().clone()}
                merger = module.WeightMerge(model, factors, residuals)
                settings = module.AcceptanceConfig(delta=100 if failure == "gate" else 0,
                    tau_g=10, tau_u=1e14, equivalence_atol=0, equivalence_rtol=0)
                if failure == "fatal":
                    with self.assertRaisesRegex(RuntimeError, "结构错误"):
                        module.select_merged_candidate(merger, adapter, factors, residuals, [edit], [keep], original, settings)
                else:
                    report = module.select_merged_candidate(merger, adapter, factors, residuals,
                                                             [edit], [keep], original, settings)
                    reports.append(report)
                    self.assertEqual(report["status"], "rejected" if failure == "gate" else "accepted")
                    if failure in ("numeric", "equivalence"):
                        self.assertEqual(report["selected_alpha"], 0.5)
                        self.assertTrue(report["trials"][0]["baseline_restored"])
                weight = model.get_submodule(name).weight
                weights.append(weight.detach().clone())
                if failure in ("gate", "fatal"):
                    self.assertTrue(torch.equal(weight, original[name]))
                self.assertFalse(model.get_submodule(name)._forward_hooks)
                self.assertFalse(hasattr(model.get_submodule(name), "_mdlm_candidate_owner"))
            if reports:
                self.assertEqual(*reports)
            self.assertTrue(torch.equal(*weights))


if __name__ == "__main__":
    unittest.main()
