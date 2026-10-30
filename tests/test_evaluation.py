"""评价协议的 CPU 轻量验证：模拟 logits、真实临时权重和完整运行器。"""

from contextlib import redirect_stdout
from dataclasses import asdict, replace
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from safetensors.torch import save_file

from evaluation.artifacts import file_digest, read_evaluation_data, read_run
from evaluation.framework import EvalTask, UltraEditFramework, get_framework
from evaluation.inference import (GOLD_PROTOCOL, PROTOCOL, GenerationRequest, LLaDABackend,
                                  Prediction, generate, prepare_request)
from evaluation.reporting import ScoreSummary, paired_scores, token_accuracy, write_record
from evaluation.runner import run_evaluation
from util.io import read_json, write_json
from data.readers.base import iter_records
from data.readers.fever import FEVERReader
from data.readers.ultraeditbench import UltraEditBenchReader
from data.readers.wikibigedit import WikiBigEditReader
from data.readers.zsre import ZSREReader


TARGET = (65, 66, 67)  # A、B、C
ROOT = Path(__file__).resolve().parents[1]


class Tokenizer:
    all_special_ids = [0, 127]

    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1, bias=False)
        self.layer.weight.data.zero_()
        self.config = SimpleNamespace(max_sequence_length=256)
        self.eval()


class Adapter:
    """复现方案示例；每次前向保存输入，用来核查回填时机。"""

    mask_token_id = 0
    tokenizer = Tokenizer()

    def __init__(self):
        self.inputs = []
        self.restores = []

    def render_prompt(self, prompt):
        return prompt + "|"

    def forward_logits(self, model, current, positions):
        self.inputs.append(current)
        begin = len(current.input_ids) - 3
        state = current.input_ids[begin:]
        edited = bool(model.layer.weight.item())
        if edited:
            candidates = {i: (TARGET[i], 0.8) for i in range(3)}
        elif state == (0, 0, 0):
            candidates = {0: (68, 0.4), 1: (88, 0.9), 2: (67, 0.6)}
        elif state == (0, 66, 0):
            candidates = {0: (65, 0.85), 2: (90, 0.7)}
        elif state == (65, 66, 0):
            candidates = {2: (67, 0.95)}
        else:
            candidates = {i: (88, 0.8) for i in range(3)}
        logits = torch.full((len(positions), 128), -torch.inf)
        for row, position in enumerate(positions):
            token, probability = candidates[position - begin]
            logits[row, token] = torch.tensor(probability).log()
            # .4 仍为最大概率；将剩余质量分给两个其他普通 token。
            logits[row, 85] = logits[row, 86] = torch.tensor((1 - probability) / 2).log()
            logits[row, 0] = logits[row, 127] = 100  # 必须屏蔽。
        return logits

    def load_model(self, *args, **kwargs):
        self.model = TinyModel()
        return self.model

    def snapshot_target_weights(self, model):
        return model.layer.weight.detach().clone()

    def restore_target_weights(self, model, snapshot):
        with torch.no_grad():
            model.layer.weight.copy_(snapshot)
        self.restores.append(float(model.layer.weight.detach()))

    def target_module_names(self):
        return ("layer",)


def task(group="edit", source="s0", index=0, decision="accepted"):
    return EvalTask(str(index), source, source, str(index), group, "Q", "ABC", decision, None)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.model, self.adapter = TinyModel(), Adapter()
        self.request = GenerationRequest((81, 124), 3, 2026)

    def gold(self, **kwargs):
        return generate(self.model, self.adapter, self.request, protocol=GOLD_PROTOCOL, target=TARGET, **kwargs)

    def test_document_example_and_replay(self):
        prediction = self.gold()
        self.assertEqual(prediction.token_ids, (65, 88, 67))
        self.assertEqual(prediction.correct, (1, 0, 1))
        self.assertEqual([r["positions"] for r in prediction.trace], [[1], [0], [2]])
        self.assertEqual([x.input_ids[-3:] for x in self.adapter.inputs],
                         [(0, 0, 0), (0, 66, 0), (65, 66, 0)])
        self.assertAlmostEqual(token_accuracy(prediction.token_ids, TARGET), 2 / 3)
        self.assertAlmostEqual(sum(prediction.correct) / 3, 2 / 3)
        # 从记录重建各轮状态，核查每个位置恰好评分一次且评分前保持 mask。
        state, seen = [0, 0, 0], set()
        for entry, model_input in zip(prediction.trace, self.adapter.inputs):
            self.assertEqual(tuple(state), model_input.input_ids[-3:])
            for p, value, gold, correct in zip(entry["positions"], entry["predicted_ids"],
                                              entry["target_ids"], entry["correct"]):
                self.assertNotIn(p, seen)
                self.assertEqual(state[p], 0)
                self.assertEqual(correct, int(value == gold))
                seen.add(p)
            for p, gold in zip(entry["positions"], entry["target_ids"]):
                state[p] = gold
        self.assertEqual(seen, {0, 1, 2})
        self.assertEqual(tuple(state), TARGET)
        self.assertNotEqual(prediction.token_ids, tuple(state))
        self.assertAlmostEqual(prediction.trace[0]["confidence"][0], 0.9, places=6)

    def test_multi_position_round_and_step_limits(self):
        for steps, counts in [(None, [1, 1, 1]), (1, [3]), (2, [2, 1]), (3, [1, 1, 1]), (9, [1, 1, 1])]:
            with self.subTest(steps=steps):
                self.adapter.inputs.clear()
                prediction = self.gold(steps=steps)
                self.assertEqual([len(r["positions"]) for r in prediction.trace], counts)
                self.assertEqual(len(self.adapter.inputs), len(counts))
                self.assertEqual(sorted(p for r in prediction.trace for p in r["positions"]), [0, 1, 2])
                if steps == 2:
                    self.assertEqual(prediction.trace[0]["positions"], [1, 2])
                    self.assertEqual(self.adapter.inputs[1].input_ids[-3:], (0, 66, 67))
        # 两协议单轮时使用同一个请求和随机流，应得到完全相同的评分预测。
        for temperature in (0, 0.6, 1.5):
            old = generate(self.model, Adapter(), self.request, steps=1, temperature=temperature)
            new = self.gold(steps=1, temperature=temperature)
            self.assertEqual(old.token_ids, new.token_ids)

    def test_sampling_ties_and_global_rng(self):
        self.model.layer.weight.data.fill_(1)
        greedy = self.gold()
        self.assertEqual([r["positions"] for r in greedy.trace], [[0], [1], [2]])
        torch.manual_seed(7)
        original_rng = torch.get_rng_state().clone()
        first = self.gold(temperature=1.3)
        second = self.gold(temperature=1.3)
        self.assertEqual(asdict(first), asdict(second))
        self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))
        self.assertFalse(set(first.token_ids) & {0, 127})

    def test_lengths_and_remainder_distribution(self):
        for length in (1, 2, 5, 7):
            for steps in (1, 2, 3, None, 20):
                with self.subTest(length=length, steps=steps):
                    inputs = []

                    def forward(model, current, positions):
                        inputs.append(current)
                        logits = torch.zeros(len(positions), 128)
                        logits[:, 65] = 3
                        return logits

                    request = replace(self.request, answer_length=length)
                    with patch.object(self.adapter, "forward_logits", side_effect=forward):
                        result = generate(self.model, self.adapter, request, steps=steps,
                                          protocol=GOLD_PROTOCOL, target=(65,) * length)
                    rounds = min(steps, length) if steps is not None else length
                    counts = [length // rounds + (t < length % rounds) for t in range(rounds)]
                    self.assertEqual([len(r["positions"]) for r in result.trace], counts)
                    self.assertEqual(result.correct, (1,) * length)
                    self.assertEqual([p for r in result.trace for p in r["positions"]], list(range(length)))
                    for state, row in zip(inputs, result.trace):
                        for position in row["positions"]:
                            self.assertEqual(state.input_ids[len(request.prompt_ids) + position], 0)

    def test_temperature_and_confidence_use_sampled_candidate(self):
        # 指定抽样结果，使低概率候选参与选位，避免误用每位置最大概率。
        logits = torch.full((3, 128), -torch.inf)
        logits[:, 65], logits[:, 66] = torch.tensor([3.0, 2.0, 1.0]), 0
        sampled = torch.tensor([[66], [65], [65]])
        with patch.object(self.adapter, "forward_logits", return_value=logits), \
                patch("evaluation.inference.torch.multinomial", return_value=sampled) as draw:
            result = self.gold(steps=1, temperature=2)
        masked = logits.clone()
        masked[:, [0, 127]] = -torch.inf
        probabilities = torch.softmax(masked / 2, dim=-1)
        self.assertTrue(torch.equal(draw.call_args.args[0], probabilities))
        self.assertEqual(result.trace[0]["positions"], [1, 2, 0])
        self.assertAlmostEqual(result.trace[0]["confidence"][-1], float(probabilities[0, 66]))

    def test_alignment_and_seeds(self):
        request, target = prepare_request(self.adapter, task(), 2026)
        self.assertEqual(request.prompt_ids, (81, 124))
        self.assertEqual(target, TARGET)
        expected = int.from_bytes(hashlib.sha256(b"2026:diffusion_short_v1:0").digest()[:8], "big") % (2**63)
        self.assertEqual(request.seed, expected)
        gold_request, _ = prepare_request(self.adapter, task(), 2026, GOLD_PROTOCOL)
        self.assertNotEqual(request.seed, gold_request.seed)

    def test_native_generation_and_scoring_errors(self):
        for target in (None, (), (65,)):
            with self.subTest(target=target), self.assertRaises(SystemExit):
                generate(self.model, self.adapter, self.request, protocol=GOLD_PROTOCOL, target=target)
        with self.assertRaises(SystemExit):
            self.gold(steps="two")
        with self.assertRaises(SystemExit):
            self.gold(temperature="hot")
        with self.assertRaises(TypeError):
            generate(self.model, self.adapter, self.request, extra=1)
        with patch.object(self.adapter, "forward_logits", return_value=torch.full((3, 128), -torch.inf)):
            with self.assertRaises(SystemExit):
                self.gold(temperature=1)
        with self.assertRaises(ValueError):
            token_accuracy((65,), TARGET)
        with self.assertRaises(ZeroDivisionError):
            token_accuracy((), ())

    def test_invalid_protocol_inputs_stop_before_forward(self):
        for protocol in (PROTOCOL, GOLD_PROTOCOL):
            for steps in (0, -1, True):
                with self.assertRaises(SystemExit):
                    generate(self.model, self.adapter, self.request, protocol=protocol,
                             target=TARGET if protocol == GOLD_PROTOCOL else None, steps=steps)
        self.assertEqual(self.adapter.inputs, [])
        self.model.config.max_sequence_length = 4
        with self.assertRaisesRegex(SystemExit, "上下文"):
            self.gold()

    def test_old_protocol_matches_committed_implementation(self):
        # 使用本次修改前的真实源码对照，不在仓库内复制一套旧生成实现。
        source = subprocess.check_output(["git", "show", "35a906a:evaluation/inference.py"], cwd=ROOT).decode("utf-8")
        legacy_types = ModuleType("evaluation.types")
        legacy_types.GenerationRequest, legacy_types.Prediction = GenerationRequest, Prediction
        namespace = {"__name__": "evaluation.legacy", "__package__": "evaluation"}
        with patch.dict(sys.modules, {"evaluation.types": legacy_types}):
            exec(compile(source, "legacy_inference.py", "exec"), namespace)
        for steps in (None, 1, 2, 9):
            for temperature in (0, 0.7, 1.5):
                with self.subTest(steps=steps, temperature=temperature):
                    old_adapter, new_adapter = Adapter(), Adapter()
                    old = namespace["generate"](self.model, old_adapter, self.request, steps=steps, temperature=temperature)
                    new = generate(self.model, new_adapter, self.request, steps=steps, temperature=temperature)
                    self.assertEqual(old.token_ids, new.token_ids)
                    self.assertEqual(old.text, new.text)
                    self.assertEqual(old_adapter.inputs, new_adapter.inputs)


class ReportingTests(unittest.TestCase):
    def test_five_groups_averages_and_subsets(self):
        framework = get_framework("ultraedit")
        tasks = [task(group, source, i, decision)
                 for group in framework.metrics
                 for i, (source, decision) in enumerate([("a", "accepted"), ("b", "rejected"),
                                                        ("b", "rejected"), ("b", "rejected")])]
        summary = framework.aggregate(tasks, list(framework.metrics))
        for t in tasks:
            after = (65, 66, 67) if t.source_case_id == "a" else (88, 88, 88)
            summary.add(t, paired_scores((88, 88, 88), after, TARGET))
        results = summary.export()
        self.assertEqual(set(results["all"]), {"ES", "GS", "LS", "person_score", "mhop_score"})
        for name in framework.metrics.values():
            row = results["all"][name]
            self.assertEqual(row["probe_macro"], {"before": 0, "after": 0.25, "delta": 0.25})
            self.assertEqual(row["case_macro"], {"before": 0, "after": 0.5, "delta": 0.5})
            self.assertEqual(results["accepted"][name]["case_macro"]["after"], 1)
            self.assertEqual(results["rejected"][name]["case_macro"]["after"], 0)
            self.assertEqual(row["status"], "complete")
        empty = ScoreSummary([task()], {"edit": "ES", "persona": "person_score"}).export()
        self.assertEqual(empty["all"]["ES"]["status"], "incomplete")
        self.assertEqual(empty["all"]["person_score"]["status"], "not_applicable")
        self.assertIsNone(empty["all"]["person_score"]["case_macro"]["after"])


class RunnerTests(unittest.TestCase):
    def run_fixture(self, mode, decisions, protocol, fail=False, write_failure=False,
                    checkpoint_failure=False, close_failure=False, nonfinite=False, source_change=False):
        with tempfile.TemporaryDirectory(prefix="mdlm_eval_") as temporary:
            root = Path(temporary)
            run = root / "run"
            run.mkdir()
            model_path = root / "base_model"
            model_path.mkdir()
            groups = list(UltraEditFramework.metrics)
            cases, rounds = [], []
            metadata = {"editor": "mdlm", "model_path": str(model_path),
                        "target_layers": [0], "torch_dtype": "float32"}

            def checkpoint(name, value):
                path = run / name
                path.mkdir()
                write_json(path / "model.json", metadata)
                save_file({"layer": torch.tensor([[value]], dtype=torch.float32)},
                          str(path / "target_weights.safetensors"))
                return str(path)

            for index, decision in enumerate(decisions):
                cid = f"s{index}"
                cases.append({"case_id": cid, "source_case_id": cid,
                              "prompts": {g: [{"probe_id": g, "prompt": "Q", "answer": "ABC"}] for g in groups}})
                rounds.append({"round_id": index, "case_id": cid, "fact_ids": [cid], "decision": decision,
                               "final_model_dir": checkpoint(f"case_{index}", index + 1.0)
                               if mode == "independent" and decision == "accepted" else None})
            source = root / "source.json"
            write_json(source, cases)
            report = {"editor": "mdlm", "status": "complete", "mode": mode, "seed": 2026,
                      "rounds": rounds, "case_count": len(rounds), "fact_count": len(cases),
                      "source_dataset": str(source), "original_model": str(model_path),
                      "final_checkpoint_dir": checkpoint("final_model", float("accepted" in decisions))
                      if mode == "sequential" else None}
            write_json(run / "run_report.json", report)
            config = root / "eval.json"
            write_json(config, {"protocol": protocol, "device": "cpu", "output_root": str(root / "out"),
                                "target_layers": [0], "torch_dtype": "float32"})
            hashes = {p: file_digest(p) for p in root.rglob("*") if p.is_file()}
            _, parsed_rounds, _ = read_run(run)
            self.assertEqual([r["checkpoint"] is not None for r in parsed_rounds],
                             [mode == "sequential" or d == "accepted" for d in decisions])
            adapter = Adapter()
            if source_change:
                forward = adapter.forward_logits
                def mutate_source(*args):
                    # 只改本测试临时源文件，模拟评价期间被外部进程替换。
                    source.write_text(source.read_text(encoding="utf-8") + " ", encoding="utf-8")
                    return forward(*args)
                adapter.forward_logits = mutate_source
            if fail:
                adapter.forward_logits = Mock(side_effect=RuntimeError("模拟前向失败"))
            if nonfinite:
                adapter.forward_logits = Mock(return_value=torch.full((3, 128), float("nan")))
            if checkpoint_failure:
                adapter.target_module_names = lambda: ("other",)

            failed_write = Mock(side_effect=OSError("模拟写入失败"))
            failed_close = Mock(side_effect=RuntimeError("模拟恢复失败"))
            original_close = LLaDABackend.close

            def persist(stream, record):
                if write_failure and record["phase"] == "paired":
                    failed_write()
                write_record(stream, record)

            def close(backend):
                original_close(backend)
                if close_failure:
                    failed_close()

            with patch("evaluation.inference.get_adapter", return_value=adapter), \
                    patch("evaluation.runner.write_record", side_effect=persist), \
                    patch.object(LLaDABackend, "close", close), redirect_stdout(io.StringIO()):
                if fail or write_failure or checkpoint_failure or close_failure or nonfinite or source_change:
                    with self.assertRaises((RuntimeError, OSError, KeyError, AttributeError, SystemExit)):
                        run_evaluation(config, run_dir=run, eval_id="test")
                    result = read_json(root / "out/test/summary.json")
                    self.assertEqual(result["status"], "failed")
                    self.assertEqual(result["error_count"], 1)
                    self.assertEqual(result["completed_tasks"], 5 * len(decisions) if close_failure or source_change else 0)
                    self.assertEqual(result["coverage"], 1 if close_failure or source_change else 0)
                    if checkpoint_failure:
                        self.assertIsNone(result["current_task_id"])
                else:
                    result = run_evaluation(config, run_dir=run, eval_id="test")
                    self.assertEqual(result["status"], "complete")
                    self.assertEqual(result["coverage"], 1)
                    records = [json.loads(line) for line in (root / "out/test/predictions.jsonl").read_text(encoding="utf-8").splitlines()]
                    paired = [r for r in records if r["phase"] == "paired"]
                    self.assertEqual(len(paired), 5 * len(decisions))
                    manifest = read_json(root / "out/test/manifest.json")
                    self.assertEqual(manifest["protocol"], protocol)
                    self.assertEqual(manifest["generation"], {"steps": None, "temperature": 0.0})
                    for row in paired:
                        edited = ("accepted" in decisions) if mode == "sequential" else row["decision"] == "accepted"
                        self.assertEqual(row["after"]["token_ids"], list(TARGET) if edited else row["before"]["token_ids"])
                        if not edited:
                            self.assertEqual(row["before"], row["after"])
                        if protocol == GOLD_PROTOCOL:
                            self.assertAlmostEqual(row["scores"]["token_accuracy"]["before"], 2 / 3)
                            self.assertEqual(row["before"]["correct"], [1, 0, 1])
                            self.assertEqual(len(row["before"]["trace"]), 3)
                    self.assertFalse(any(name == "MDLM" or name.startswith("MDLM.") for name in sys.modules))
            self.assertEqual(float(adapter.model.layer.weight.detach()), 0)
            if source_change:
                self.assertNotEqual(hashes.pop(source), file_digest(source))
            self.assertEqual(hashes, {p: file_digest(p) for p in hashes})

    def test_native_checkpoint_errors_restore_baseline(self):
        adapter = Adapter()
        with tempfile.TemporaryDirectory(prefix="mdlm_weights_") as temporary, \
                patch("evaluation.inference.get_adapter", return_value=adapter):
            path = Path(temporary)
            backend = LLaDABackend({"model_path": temporary, "target_layers": [0], "torch_dtype": "float32"}, "cpu")
            for weights, error in (({"other": torch.ones(1, 1)}, KeyError),
                                   ({"layer": torch.ones(2, 2)}, SystemExit),
                                   ({"layer": torch.ones(1, 1, dtype=torch.float64)}, SystemExit),
                                   ({"layer": torch.full((1, 1), float("nan"))}, SystemExit)):
                save_file(weights, str(path / "target_weights.safetensors"))
                with self.assertRaises(error):
                    backend.use_checkpoint(path)
                self.assertEqual(float(adapter.model.layer.weight.detach()), 0)
            # 只有符合模型精度的完整权重才可进入评价。
            save_file({"layer": torch.ones(1, 1)}, str(path / "target_weights.safetensors"))
            backend.use_checkpoint(path)
            self.assertEqual(adapter.model.layer.weight.dtype, torch.float32)
            self.assertEqual(float(adapter.model.layer.weight.detach()), 1)
            backend.close()

    def test_broadcast_checkpoint_is_rejected(self):
        adapter, model = Adapter(), AdapterModel()
        name = "model.transformer.blocks.0.ff_out"
        from util.nethook import restore_weights, snapshot_weights
        adapter.load_model = Mock(return_value=model)
        adapter.target_module_names = lambda: (name,)
        adapter.snapshot_target_weights = lambda m: snapshot_weights(m, (name,))
        adapter.restore_target_weights = lambda m, s: restore_weights(m, s, (name,))
        with tempfile.TemporaryDirectory(prefix="mdlm_broadcast_") as temporary, \
                patch("evaluation.inference.get_adapter", return_value=adapter):
            path = Path(temporary)
            backend = LLaDABackend({"model_path": temporary, "target_layers": [0], "torch_dtype": "float32"}, "cpu")
            save_file({name: torch.ones(1, 2)}, str(path / "target_weights.safetensors"))
            with self.assertRaises(SystemExit):
                backend.use_checkpoint(path)
            self.assertTrue(torch.equal(model.get_submodule(name).weight, torch.zeros(2, 2)))
            backend.close()

    def test_all_checkpoint_layers_are_validated_before_copy(self):
        # 第二层不合法时，第一层也不能写入；模型保持原始权重。
        from util.nethook import restore_weights, snapshot_weights

        adapter = Adapter()
        model = TinyModel()
        model.other = torch.nn.Linear(1, 1, bias=False)
        model.other.weight.data.zero_()
        adapter.load_model = Mock(return_value=model)
        adapter.target_module_names = lambda: ("layer", "other")
        adapter.snapshot_target_weights = lambda m: snapshot_weights(m, ("layer", "other"))
        adapter.restore_target_weights = lambda m, s: restore_weights(m, s, ("layer", "other"))
        with tempfile.TemporaryDirectory(prefix="mdlm_partial_") as temporary, \
                patch("evaluation.inference.get_adapter", return_value=adapter):
            path = Path(temporary)
            backend = LLaDABackend({"model_path": temporary, "target_layers": [0], "torch_dtype": "float32"}, "cpu")
            save_file({"layer": torch.ones(1, 1), "other": torch.ones(2, 2)},
                      str(path / "target_weights.safetensors"))
            with self.assertRaises(SystemExit):
                backend.use_checkpoint(path)
            self.assertEqual(float(model.layer.weight.detach()), 0)
            self.assertEqual(float(model.other.weight.detach()), 0)
            backend.close()

    def test_modes_and_rejected_cases_for_both_protocols(self):
        for protocol in (PROTOCOL, GOLD_PROTOCOL):
            for mode, decisions in (("independent", ["accepted", "rejected", "accepted"]),
                                    ("independent", ["rejected"]), ("sequential", ["accepted", "rejected"]),
                                    ("sequential", ["rejected"])):
                with self.subTest(protocol=protocol, mode=mode, decisions=decisions):
                    self.run_fixture(mode, decisions, protocol)

    def test_failure_still_writes_summary_and_restores_weights(self):
        self.run_fixture("independent", ["accepted"], GOLD_PROTOCOL, fail=True)

    def test_failed_record_is_not_counted_as_completed(self):
        self.run_fixture("independent", ["accepted"], GOLD_PROTOCOL, write_failure=True)

    def test_checkpoint_failure_does_not_blame_previous_probe(self):
        self.run_fixture("independent", ["accepted"], GOLD_PROTOCOL, checkpoint_failure=True)

    def test_cleanup_failure_is_not_marked_complete(self):
        self.run_fixture("independent", ["accepted"], GOLD_PROTOCOL, close_failure=True)

    def test_handled_outer_exception_does_not_mark_success_failed(self):
        try:
            int("模拟外层异常")
        except ValueError:
            self.run_fixture("independent", ["accepted"], GOLD_PROTOCOL)

    def test_nonfinite_greedy_is_failed_not_zero_score(self):
        for protocol in (PROTOCOL, GOLD_PROTOCOL):
            self.run_fixture("independent", ["accepted"], protocol, nonfinite=True)

    def test_input_version_change_is_reported_failed(self):
        self.run_fixture("independent", ["accepted"], GOLD_PROTOCOL, source_change=True)

    def test_preflight_failures_are_recorded_without_model_loading(self):
        for failure in ("failed_report", "empty_group", "bad_steps", "duplicate_id"):
            with tempfile.TemporaryDirectory(prefix="mdlm_preflight_") as temporary:
                root = Path(temporary)
                cases = [{"case_id": "s", "prompts": {"edit": [{"probe_id": "e", "prompt": "Q", "answer": "ABC"}]}}]
                if failure == "duplicate_id":
                    cases *= 2
                write_json(root / "cases.json", cases)
                report = {"editor": "mdlm", "status": "failed" if failure == "failed_report" else "complete",
                          "mode": "independent", "case_count": 1, "fact_count": 1,
                          "rounds": [{"round_id": 0, "case_id": "s", "fact_ids": ["s"],
                                      "decision": "rejected", "final_model_dir": None}],
                          "source_dataset": str(root / "cases.json"), "original_model": str(root / "base")}
                write_json(root / "run_report.json", report)
                config = {"device": "cpu", "output_root": str(root / "out"),
                          "target_layers": [0], "torch_dtype": "float32"}
                if failure == "empty_group":
                    config["groups"] = ["persona"]
                if failure == "bad_steps":
                    config["generation"] = {"steps": 0}
                write_json(root / "eval.json", config)
                with patch("evaluation.inference.get_adapter") as loader:
                    with self.assertRaises(SystemExit):
                        run_evaluation(root / "eval.json", run_dir=root, eval_id="failed")
                loader.assert_not_called()
                summary = read_json(root / "out/failed/summary.json")
                self.assertEqual(summary["status"], "failed")
                self.assertTrue(summary["failure"])
                self.assertEqual(summary["completed_tasks"], 0)
                self.assertIsNone(summary["coverage"])
                if failure == "failed_report":
                    process = subprocess.run([sys.executable, "-B", str(ROOT / "evaluate.py"),
                        "--config", str(root / "eval.json"), "--run-dir", str(root), "--eval-id", "cli_failed"],
                        capture_output=True, text=True)
                    self.assertNotEqual(process.returncode, 0)
                    cli_summary = read_json(root / "out/cli_failed/summary.json")
                    self.assertEqual(cli_summary["status"], "failed")


class ReaderTests(unittest.TestCase):
    def test_reader_semantic_errors_stop(self):
        with self.assertRaises(SystemExit):
            FEVERReader(options={"rephrase_policy": "single_fixd"})
        with self.assertRaises(SystemExit):
            ZSREReader(options={"target_filed": "alt"})
        with tempfile.TemporaryDirectory(prefix="mdlm_duplicate_") as temporary:
            path = Path(temporary) / "data.json"
            row = {"case_id": 1, "subject": "Alice", "src": "Alice?", "ans": "A",
                   "rephrase": "Alice again?", "loc": "Bob", "loc_ans": "B"}
            write_json(path, [row, row])
            with self.assertRaisesRegex(SystemExit, "重复案例"):
                list(ZSREReader().iter_cases(path))

    def test_json_jsonl_bom_limits_and_text_preservation(self):
        rows = [{"case_id": 1, "text": " 中文🙂 "}, {"case_id": 2, "text": "é"}]
        with tempfile.TemporaryDirectory(prefix="mdlm_reader_") as temporary:
            root = Path(temporary)
            formats = [
                ("array.json", json.dumps(rows, ensure_ascii=False), rows),
                ("object.json", json.dumps(rows[0], ensure_ascii=False), rows[:1]),
                ("lines.JSONL", "\r\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\r\n\r\n", rows),
            ]
            for name, text, expected in formats:
                for bom in ("", "\ufeff"):
                    with self.subTest(name=name, bom=bool(bom)):
                        path = root / name
                        path.write_text(bom + text, encoding="utf-8")
                        self.assertEqual(list(iter_records(path)), list(enumerate(expected)))
                        self.assertEqual(list(iter_records(path, limit=1)), [(0, expected[0])])

    def test_native_parsing_field_errors_and_generator_cleanup(self):
        import ijson

        with tempfile.TemporaryDirectory(prefix="mdlm_reader_") as temporary:
            path = Path(temporary) / "data.json"
            for text in ("{", ""):
                path.write_text(text, encoding="utf-8")
                with self.subTest(text=text), self.assertRaises(ijson.JSONError):
                    list(iter_records(path))
            path = path.with_suffix(".jsonl")
            path.write_text('{"id": 1}\nnot-json\n', encoding="utf-8")
            self.assertEqual(list(iter_records(path, limit=1)), [(0, {"id": 1})])
            with self.assertRaises(ValueError):
                list(iter_records(path))
            for limit in (-1, "one"):
                with self.assertRaises(SystemExit):
                    list(iter_records(path, limit=limit))
            path.write_text('{"id": 1}\n{"id": 2}\n', encoding="utf-8")
            opened, original_open = [], Path.open

            def tracked_open(*args, **kwargs):
                stream = original_open(*args, **kwargs)
                opened.append(stream)
                return stream

            with patch.object(Path, "open", tracked_open):
                records = iter_records(path)
                next(records)
                records.close()
            self.assertTrue(opened and all(stream.closed for stream in opened))
            row = {"case_id": 1, "subject": "Alice", "src": "Alice?", "ans": " A ",
                   "rephrase": "Where is Alice?", "loc": "Bob", "loc_ans": " B "}
            del row["src"]
            write_json(path.with_suffix(".json"), [row])
            with self.assertRaises(KeyError):
                list(ZSREReader().iter_cases(path.with_suffix(".json")))

    def test_reader_fields_keep_probes_and_zsre_target_choice(self):
        with tempfile.TemporaryDirectory(prefix="mdlm_reader_") as temporary:
            root = Path(temporary)
            row = {"case_id": "one", "subject": "Alice", "ans": " A ", "alt": " C ",
                   "src": "Alice?", "prompt": "Alice?", "update": "Alice?",
                   "rephrase": "Where is Alice?", "rephrase_prompt": "Where is Alice?",
                   "loc": "Bob", "loc_ans": " B ", "personas": "As a historian, Alice?",
                   "mhop": "Next?", "mhop_ans": " D "}
            for reader in (ZSREReader(), UltraEditBenchReader(), WikiBigEditReader()):
                path = root / f"{reader.name}.json"
                write_json(path, [row])
                case = list(reader.iter_cases(path))[0]
                self.assertEqual(case["prompts"]["edit"][0]["answer"], " A ")
                self.assertEqual(case["prompts"]["generalization"][0]["answer"], " A ")
                expected_keep = "Bob?" if reader.name == "zsre" else "Bob"
                keep = list(reader.iter_keep_probes(path))[0]
                self.assertEqual((keep["prompt"], keep["answer"]), (expected_keep, " B "))
                if reader.name == "wikibigedit":
                    self.assertEqual(case["prompts"]["portability"][0]["answer"], " D ")
                    self.assertEqual(case["prompts"]["persona"][0]["answer"], " A ")
            path = root / "zsre.json"
            changed = list(ZSREReader(options={"target_field": "alt"}).iter_cases(path))[0]
            self.assertEqual(changed["prompts"]["edit"][0]["answer"], " C ")
            self.assertEqual(changed["prompts"]["generalization"][0]["answer"], " C ")

    def test_fever_selection_and_optional_mhop(self):
        with tempfile.TemporaryDirectory(prefix="mdlm_reader_") as temporary:
            path = Path(temporary) / "fever.json"
            row = {"case_id": 7, "prompt": "Alice?", "ans": "old", "alt": "new",
                   "equiv_prompt": ["Alice one?", "Alice two?", "Alice three?"],
                   "unrel_prompt": "Bob?", "unrel_ans": "unchanged"}
            write_json(path, [row])
            first = list(FEVERReader(seed=2026).iter_cases(path))[0]
            second = list(FEVERReader(seed=2026).iter_cases(path))[0]
            self.assertEqual(first, second)
            self.assertEqual(len(first["prompts"]["generalization"]), 1)
            all_variants = list(FEVERReader(options={"rephrase_policy": "all"}).iter_cases(path))[0]
            self.assertEqual([p["prompt"] for p in all_variants["prompts"]["generalization"]], row["equiv_prompt"])
            self.assertTrue(all(p["answer"] == "new" for p in all_variants["prompts"]["generalization"]))
            self.assertEqual(first["prompts"]["preservation"][0]["answer"], "unchanged")
            wiki = {"subject": "Alice", "update": "Alice?", "ans": "A", "rephrase": "Again?",
                    "loc": "Bob?", "loc_ans": "B", "personas": "As a historian, Alice?"}
            for missing in (None, "", "null", " null "):
                case = WikiBigEditReader().convert(path, 0, {**wiki, "mhop": missing, "mhop_ans": missing})
                self.assertEqual(case["prompts"]["portability"], [])


class ByteTokenizer:
    """仅模拟分词结果；可无损处理真实样本中的 Unicode，不使用真实模型资源。"""

    all_special_ids = [0, 257]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.chat_call = (messages, tokenize, add_generation_prompt)
        return "user:" + messages[0]["content"] + "|assistant:"

    def encode(self, text, **kwargs):
        return [value + 1 for value in text.encode("utf-8")]

    def decode(self, ids, **kwargs):
        return bytes(value - 1 for value in ids).decode("utf-8", errors="replace")


class AdapterModel(torch.nn.Module):
    """符合实际适配器接口的微型模型，执行真实 Tensor 前向及目标层权重切换。"""

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(258, 2)
        self.model = torch.nn.Module()
        self.model.transformer = torch.nn.Module()
        self.model.transformer.blocks = torch.nn.ModuleList([torch.nn.Module()])
        self.model.transformer.blocks[0].ff_out = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.embedding.weight[:, 0] = 1
            self.embedding.weight[:, 1] = 0
            self.model.transformer.blocks[0].ff_out.weight.zero_()
        self.config = SimpleNamespace(max_sequence_length=512, mask_token_id=0, block_group_size=1)
        self.observed_weights = []
        self.eval()

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask, use_cache, return_dict):
        layer = self.model.transformer.blocks[0].ff_out
        self.observed_weights.append(tuple(layer.weight.detach().flatten().tolist()))
        hidden = layer(self.embedding(input_ids))
        logits = hidden.new_full((*input_ids.shape, 258), -5)
        logits[..., 66] = hidden[..., 0] + 1  # 字节 token A
        logits[..., 67] = hidden[..., 1]      # 字节 token B
        return SimpleNamespace(logits=logits)


class IntegrationTests(unittest.TestCase):
    def test_cli_real_adapter_and_mdlm_save_format(self):
        import evaluate
        from model.adapters.llada_8b_instruct import LLaDA8BInstructAdapter

        # 调用 MDLM 实际保存函数，不执行编辑；退出后恢复模块表，隔离其他测试。
        with patch.dict(sys.modules):
            from MDLM.runner import save_final_model

            for protocol in (PROTOCOL, GOLD_PROTOCOL):
                for mode in ("independent", "sequential"):
                    with self.subTest(protocol=protocol, mode=mode), \
                            tempfile.TemporaryDirectory(prefix="mdlm_cli_") as temporary:
                        root = Path(temporary)
                        run = root / "run"
                        run.mkdir()
                        original_path, relocated_path = root / "base_old", root / "base_relocated"
                        relocated_path.mkdir()
                        tokenizer = ByteTokenizer()
                        adapter = LLaDA8BInstructAdapter(tokenizer, mask_token_id=0, target_layers=(0,))
                        source_model, runtime_model = AdapterModel(), AdapterModel()
                        name = adapter.target_module_names()[0]
                        cases, rounds = [], []
                        # 同一源案例两个事实，另有拒绝案例和第二个接受案例。
                        for index, fact_ids in enumerate((["f0", "f1"], ["f2"], ["f3"])):
                            source_id = f"s{index}"
                            decision = "rejected" if index == 1 else "accepted"
                            for fact_id in fact_ids:
                                cases.append({"case_id": fact_id, "source_case_id": source_id,
                                              "prompts": {g: [{"probe_id": g, "prompt": "问题 Q", "answer": "ABC"}]
                                                          for g in UltraEditFramework.metrics}})
                            checkpoint = None
                            if decision == "accepted":
                                with torch.no_grad():
                                    source_model.get_submodule(name).weight.copy_(
                                        torch.tensor([[0., 0.], [3., 0.]]) if index == 0
                                        else torch.tensor([[4., 0.], [0., 0.]]))
                                if mode == "independent":
                                    checkpoint = save_final_model(source_model, adapter, run / f"case_{index}",
                                                                  model_path=original_path, torch_dtype="float32",
                                                                  device_map="cpu")
                            rounds.append({"round_id": index, "case_id": source_id, "fact_ids": fact_ids,
                                           "decision": decision, "final_model_dir": str(checkpoint) if checkpoint else None})
                        final = save_final_model(source_model, adapter, run / "final_model",
                                                 model_path=original_path, torch_dtype="float32", device_map="cpu") \
                            if mode == "sequential" else None
                        source = root / "source.json"
                        write_json(source, cases)
                        # 旧路径不可用，验证 CLI 路径覆盖和元数据选择确实生效。
                        report = {"editor": "mdlm", "status": "complete", "mode": mode, "rounds": rounds,
                                  "seed": 2026, "case_count": 3, "fact_count": 4,
                                  "source_dataset": str(root / "old/source.json"),
                                  "dataset_reading": {"reader": "native", "case_count": 4},
                                  "original_model": str(original_path.resolve()),
                                  "final_checkpoint_dir": str(final) if final else None}
                        write_json(run / "run_report.json", report)
                        config = root / "eval.json"
                        write_json(config, {"protocol": protocol, "run_dir": str(root / "missing"),
                                            "device": "cuda:999", "output_root": str(root / "out")})
                        protected = {p: file_digest(p) for p in root.rglob("*") if p.is_file()}
                        output = io.StringIO()
                        with patch("evaluation.inference.get_adapter", return_value=adapter), \
                                patch.object(adapter, "load_model", return_value=runtime_model) as loader, \
                                redirect_stdout(output):
                            evaluate.main(["--config", str(config), "--run-dir", str(run), "--device", "cpu",
                                           "--model-path", str(relocated_path), "--dataset", str(source), "--eval-id", "cli"])
                        self.assertIn('"status": "complete"', output.getvalue())
                        self.assertEqual(loader.call_args.args[0], str(relocated_path.resolve()))
                        self.assertEqual(loader.call_args.kwargs, {"torch_dtype": "float32", "device_map": "cpu"})
                        summary = read_json(root / "out/cli/summary.json")
                        self.assertEqual(summary["expected_tasks"], 20)
                        self.assertEqual(summary["completed_tasks"], 20)
                        self.assertEqual(summary["metrics"]["rejected"]["LS"]["status"], "complete")
                        rows = [json.loads(line) for line in
                                (root / "out/cli/predictions.jsonl").read_text(encoding="utf-8").splitlines()]
                        for row in rows:
                            if row["phase"] != "paired":
                                continue
                            self.assertEqual(row["before"]["token_ids"], [66, 66, 66])
                            expected = [67, 67, 67] if mode == "independent" and row["source_case_id"] == "s0" else [66, 66, 66]
                            self.assertEqual(row["after"]["token_ids"], expected)
                            if row["checkpoint"] is None:
                                self.assertEqual(row["before"], row["after"])
                        weights_seen = set(runtime_model.observed_weights)
                        self.assertIn((0., 0., 0., 0.), weights_seen)
                        self.assertIn((4., 0., 0., 0.), weights_seen)
                        if mode == "independent":
                            self.assertIn((0., 0., 3., 0.), weights_seen)
                        self.assertTrue(torch.equal(runtime_model.get_submodule(name).weight, torch.zeros(2, 2)))
                        self.assertEqual(tokenizer.chat_call, ([{"role": "user", "content": "问题 Q"}], False, True))
                        self.assertEqual(protected, {p: file_digest(p) for p in protected})

    def test_existing_dataset_samples_and_task_mapping(self):
        from data.dataset import load_dataset

        checked = 0
        framework = get_framework("ultraedit")
        for reader in ("zsre", "fever", "wikibigedit", "ultraeditbench"):
            config = read_json(ROOT / f"configs/datasets/{reader}.json")
            source = ROOT / config["dataset_path"]
            if not source.is_file():
                continue
            with self.subTest(reader=reader):
                before = (source.stat().st_size, source.stat().st_mtime_ns)
                bundle = load_dataset({**config, "source_limit": 2, "keep_path": None, "keep_limit": None})
                report = {"source_dataset": str(source), "seed": 2026, "dataset_reading": bundle.metadata}
                if reader == "fever":
                    report["dataset_reading"]["reader_options"]["subject_path"] = "不存在的主体标注"
                reread = read_evaluation_data(report)
                self.assertEqual(bundle.cases, reread.cases)
                rounds = [{"case_id": c["source_case_id"], "fact_ids": [c["case_id"]],
                           "decision": "rejected", "checkpoint": None} for c in reread.cases]
                tasks = framework.build_tasks(reread.cases, rounds, list(framework.metrics))
                expected_per_case = 5 if reader == "wikibigedit" else 3
                self.assertEqual(len(tasks), expected_per_case * 2)
                by_id = {c["case_id"]: c for c in reread.cases}
                for t in tasks:
                    probe = by_id[t.case_id]["prompts"][t.group][0]
                    self.assertEqual((t.prompt, t.answer), (probe["prompt"], probe["answer"]))
                # 真实样本 + 临时模拟编辑产物，经实际入口运行；不加载真实大模型。
                import evaluate
                from model.adapters.llada_8b_instruct import LLaDA8BInstructAdapter

                with tempfile.TemporaryDirectory(prefix="mdlm_samples_") as temporary, patch.dict(sys.modules):
                    from MDLM.runner import save_final_model

                    root = Path(temporary)
                    run, base = root / "run", root / "base"
                    run.mkdir()
                    base.mkdir()
                    adapter = LLaDA8BInstructAdapter(ByteTokenizer(), mask_token_id=0, target_layers=(0,))
                    runtime, edited = AdapterModel(), AdapterModel()
                    runtime.config.max_sequence_length = 8192
                    with torch.no_grad():
                        edited.get_submodule(adapter.target_module_names()[0]).weight[1, 0] = 3
                    checkpoint = save_final_model(edited, adapter, run / "case_0", model_path=base,
                                                  torch_dtype="float32", device_map="cpu")
                    selected_rounds = [
                        {**r, "round_id": i, "decision": "accepted" if i == 0 else "rejected",
                         "final_model_dir": str(checkpoint) if i == 0 else None}
                        for i, r in enumerate(rounds)]
                    write_json(run / "run_report.json", {
                        **report, "editor": "mdlm", "status": "complete", "mode": "independent",
                        "case_count": 2, "fact_count": 2, "rounds": selected_rounds, "original_model": str(base)})
                    config_path = root / "eval.json"
                    write_json(config_path, {"protocol": GOLD_PROTOCOL, "device": "cpu",
                                             "generation": {"steps": 2, "temperature": 0},
                                             "output_root": str(root / "out")})
                    with patch("evaluation.inference.get_adapter", return_value=adapter), \
                            patch.object(adapter, "load_model", return_value=runtime), redirect_stdout(io.StringIO()):
                        evaluate.main(["--config", str(config_path), "--run-dir", str(run), "--eval-id", "samples"])
                    result = read_json(root / "out/samples/summary.json")
                    self.assertEqual(result["status"], "complete")
                    self.assertEqual(result["completed_tasks"], len(tasks))
                    self.assertEqual(result["coverage"], 1)
                    if reader != "wikibigedit":
                        self.assertEqual(result["metrics"]["all"]["person_score"]["status"], "not_applicable")
                    predictions = [json.loads(line) for line in
                                   (root / "out/samples/predictions.jsonl").read_text(encoding="utf-8").splitlines()]
                    before_rows = [r for r in predictions if r["phase"] == "before"]
                    self.assertEqual({r["task_id"] for r in before_rows}, {t.task_id for t in tasks})
                    for row in before_rows:
                        saved = row["prediction"]
                        self.assertEqual(saved["correct"], [int(a == b) for a, b in
                                                           zip(saved["token_ids"], row["target_ids"])])
                self.assertEqual(before, (source.stat().st_size, source.stat().st_mtime_ns))
                checked += 1
        if not checked:
            self.skipTest("当前环境没有本地 UltraEdit 数据文件")


if __name__ == "__main__":
    unittest.main()
