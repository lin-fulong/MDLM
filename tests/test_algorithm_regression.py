"""固定小模型的 P0～P5 数值回归；基准来自重构前提交。"""

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace
import unittest

import torch

from MDLM.acceptance import select_merged_candidate
from MDLM.candidate import CandidateModel
from MDLM.contracts import AcceptanceConfig, FactSelection, ResidualOptimizationConfig, WritebackConfig
from MDLM.keep_statistics import SecondMomentAccumulator
from MDLM.merge import WeightMerge
from MDLM.optimize import KeepTrainingExample, log_probabilities, optimize_residuals
from MDLM.states import EditTrainingExample, FactTrajectory, build_supervision, edit_training_examples
from MDLM.writeback import solve_writeback_factor
from model.adapters.llada_8b_instruct import LLaDA8BInstructAdapter
from model.base import DenoiseTrace, KeepModelInput, MaskedEditInput


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(block_group_size=1, mask_token_id=3, max_sequence_length=16)
        self.embedding = torch.nn.Embedding(4, 3)
        self.model = torch.nn.Module()
        self.model.transformer = torch.nn.Module()
        self.model.transformer.blocks = torch.nn.ModuleList()
        for _ in range(2):
            block = torch.nn.Module()
            block.ff_out = torch.nn.Linear(3, 3)
            self.model.transformer.blocks.append(block)
        self.head = torch.nn.Linear(3, 4)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask, use_cache, return_dict):
        hidden = self.embedding(input_ids)
        hidden = hidden + hidden.mean(dim=1, keepdim=True)
        for block in self.model.transformer.blocks:
            hidden = torch.tanh(block.ff_out(hidden))
        return SimpleNamespace(logits=self.head(hidden))


class Tokenizer:
    all_special_ids = [3]

    def apply_chat_template(self, messages, **kwargs):
        return "Q:"

    def encode(self, text, **kwargs):
        return [{"Q": 2, ":": 1, "A": 0, "B": 1, "C": 2}[char] for char in text]

    def decode(self, ids, **kwargs):
        return "".join("ABC"[value] for value in ids if value != 3)


def algorithm_snapshot():
    """只记录数值和决策，不包含耗时、路径或工程元数据。"""
    torch.manual_seed(42)
    model = TinyModel().double().eval()
    adapter = LLaDA8BInstructAdapter(Tokenizer(), mask_token_id=3, target_layers=(0, 1))
    names = adapter.target_module_names()
    trajectories, keys = [], {name: [] for name in names}
    for index in range(2):
        initial = MaskedEditInput((2, index, 3, 3, 3), (1,) * 5, (2, 3, 4), 1)
        capture = adapter.forward_with_capture(model, initial)
        for name in names:
            keys[name].append(capture.module_inputs[name])
        trace = adapter.sample_denoise(model, initial, steps=3, temperature=0.0, seed=2026 + index)
        targets = [trace.final_input.input_ids[position] for position in initial.mask_positions]
        conflict_position = trace.revealed_tokens[1][0][0]
        targets[initial.mask_positions.index(conflict_position)] = (targets[initial.mask_positions.index(conflict_position)] + 1) % 3
        # 重复完整输入，保证状态去重和冲突筛选都参与回归。
        trace = DenoiseTrace((initial,) + trace.pre_forward_inputs,
                            (trace.mask_logit_sha256[0],) + trace.mask_logit_sha256,
                            ((),) + trace.revealed_tokens, trace.final_input)
        trajectories.append(FactTrajectory(str(index), str(index), initial.mask_positions,
                                            tuple(targets), 2026 + index, trace))
    keys = {name: torch.stack(values, dim=1) for name, values in keys.items()}
    facts, supervision = build_supervision(trajectories, 3)
    edits = edit_training_examples(facts)
    keep_input = KeepModelInput((2, 1, 3, 3), (1,) * 4, (2, 3))
    keep_capture = adapter.forward_keep_with_capture(model, keep_input, (2, 3))
    matrices, factors = {}, {}
    for name in names:
        accumulator = SecondMomentAccumulator(3, accumulation_dtype="float64", seed=2026)
        accumulator.add(keep_capture.module_inputs[name], [.3, .7])
        matrices[name] = accumulator.matrix
        factors[name], _ = solve_writeback_factor(keys[name], matrices[name], .3,
            WritebackConfig({name: .3}, compute_dtype="float64", storage_dtype="float64", chunk_size=2))
    keeps = [KeepTrainingExample(keep_input, (2, 3), (.3, .7), log_probabilities(keep_capture.mask_logits))]
    settings = ResidualOptimizationConfig({name: .03 for name in names}, beta=.4,
        keep_train_file="fixture", reference_model_path="fixture", reference_model_revision="fixture",
        reference_tokenizer_revision="fixture", learning_rate=.03, max_steps=6,
        checkpoints=(1, 3, 6), residual_dtype="float64")
    with CandidateModel(model, factors, dtype="float64") as candidate:
        optimized = optimize_residuals(candidate, adapter, edits, keeps, settings)
    residuals = optimized.residuals
    originals = {name: model.get_submodule(name).weight.detach().clone() for name in names}
    development = [EditTrainingExample(item.case_id, item.trace.pre_forward_inputs[0],
        item.answer_positions, item.target_token_ids, (1 / 6,) * 3) for item in trajectories]
    acceptance = AcceptanceConfig("fixture", (FactSelection("0", "0"),), "fixture", "fixture", "fixture", "fixture",
        alphas=(3., 1., .25), delta=1e-6, tau_g=100, tau_u=100, compute_dtype="float64",
        equivalence_atol=1e-12, equivalence_rtol=1e-12, chunk_size=2)
    merger = WeightMerge(model, factors, residuals, compute_dtype="float64", chunk_size=2)
    accepted = select_merged_candidate(merger, adapter, factors, residuals, development, keeps, originals, acceptance)
    weights = {name: model.get_submodule(name).weight.detach().clone() for name in names}
    rejected = select_merged_candidate(merger, adapter, factors, residuals, development, keeps, originals,
                                        replace(acceptance, delta=1e6))
    return {
        "supervision": facts, "supervision_report": supervision,
        "tensors": {label: {name: value.tolist() for name, value in values.items()} for label, values in
                    (("K", keys), ("C", matrices), ("Q", factors), ("R", residuals), ("W", weights))},
        "selected_step": optimized.report["selected_step"],
        "objectives": [row["loss"] for row in optimized.report["history"] if "loss" in row],
        "accepted": accepted, "rejected": rejected,
        "baseline_restored": all(torch.equal(model.get_submodule(name).weight, value) for name, value in originals.items()),
    }


class AlgorithmRegressionTests(unittest.TestCase):
    def test_frozen_small_model_matches_before_refactor(self):
        before_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            snapshot = algorithm_snapshot()
        finally:
            torch.set_num_threads(before_threads)
        self.assertTrue(snapshot["baseline_restored"])
        self.assertEqual(snapshot["accepted"]["status"], "accepted")
        self.assertEqual(snapshot["rejected"]["status"], "rejected")
        # cea7353、CPU float64、单线程：逐阶段摘要保证所有元素逐位一致。
        expected = {
            "K": "9743e11b43669c5bb0c519326c31155ece5bbf82059c85e98017600c4bfe0fc8",
            "C": "fbea27de6c64f8fdcd25856f5eadc70a46d0241bba2a272ea9a1583738ac4e1a",
            "Q": "6c39301387bf7cac11df8d8791f2495d69a8013dfa5f0a0bc12b1e4e87ef447e",
            "R": "2355b59102e13a0893b90ad29f9664a752ea19c5621ff88504c587afdcd25175",
            "W": "108cfe768e9fbeb0febac1e194415e906111cd19c226b71a95fba037a21914dc",
        }
        for name, values in snapshot["tensors"].items():
            digest = hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            self.assertEqual(digest, expected[name], name)


if __name__ == "__main__":
    unittest.main()
