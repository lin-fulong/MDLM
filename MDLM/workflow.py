"""串联公式（1）～（18）：已采集轨迹 → 固定监督 → K/C → Q → R → 合并与接受。"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from MDLM.acceptance import AcceptanceConfig, select_merged_candidate
from MDLM.candidate import CandidateModel
from MDLM.merge import WeightMerge
from MDLM.optimize import KeepTrainingExample, ResidualOptimizationConfig, log_probabilities, optimize_residuals
from MDLM.states import build_supervision, edit_training_examples
from MDLM.writeback import WritebackConfig, solve_writeback_factor
from model.adapters.llada_8b_instruct import LLaDA8BInstructAdapter
from util.extract_keep_statistics import keep_input_from_record, keep_matrices
from util.extract_reference_keys import extract_keys
from util.io import read_json, write_json


@torch.no_grad()
def reference_examples(model, adapter, rows) -> list[KeepTrainingExample]:
    """公式（10）（16）（17）：缓存最初 theta_0 的完整词表 log p，供保持 KL 复用。

    runner 在首次编辑前调用，训练与开发保持输入分别准备，保留给定位置与权重。
    """
    examples = []
    for row in rows:
        model_input = keep_input_from_record(row["input"])
        positions, weights = tuple(row["positions"]), tuple(row["weights"])
        logs = log_probabilities(adapter.forward_logits(model, model_input, positions)).cpu()
        examples.append(KeepTrainingExample(model_input, positions, weights, logs))
    return examples


def edit_round(model, adapter, trajectories, statistics, dev_edits, dev_keeps, originals, *,
               writeback: WritebackConfig, optimization: ResidualOptimizationConfig,
               acceptance: AcceptanceConfig, keep_train=(), p2_keys=None, p2_keep=None) -> dict:
    """执行一轮编辑：theta_b 为传入模型，originals 始终为最初 W_0。"""
    # 公式（1）～（3）：筛选固定兼容状态，沿用事实/状态/位置的全局监督权重。
    facts, supervision = build_supervision(trajectories, adapter.mask_token_id)
    edits = edit_training_examples(facts)

    # 公式（4）～（9）：按事实顺序提取 K，逐层累计 C、解 HX=K，得到固定 Q=Xᵀ。
    keys = extract_keys(model, adapter, trajectories, **(p2_keys or {}))
    factors = {}
    for name, matrix in keep_matrices(model, adapter, statistics, **(p2_keep or {})):
        factors[name] = solve_writeback_factor(
            keys[name], matrix, writeback.ridge_by_layer[name], writeback)
        del matrix

    # 公式（10）～（12）：R 从零开始，Q 和基准权重固定，多层分支联合优化。
    with CandidateModel(model, factors, dtype=optimization.residual_dtype) as candidate:
        optimized = optimize_residuals(candidate, adapter, edits, list(keep_train), optimization)
    result = {"supervision": supervision, "optimization": optimized.report}
    if optimized.residuals is None:
        # 无有限、非零且优于 J(0) 的 R 时结束本轮；退出候选上下文已移除分支。
        return {**result, "decision": "rejected", "selected_alpha": None,
                "baseline_restored": True, "selection": None}

    # 公式（13）（15）～（18）：从 W_b 重建 αRQ 试步，按三个开发条件接受或恢复。
    merger = WeightMerge(model, factors, optimized.residuals,
                         compute_dtype=acceptance.compute_dtype, chunk_size=acceptance.chunk_size)
    selection = select_merged_candidate(merger, adapter, factors, optimized.residuals,
                                        dev_edits, dev_keeps, originals, acceptance)
    return {**result, "decision": selection["status"], "selected_alpha": selection["selected_alpha"],
            "baseline_restored": selection["baseline_restored"], "selection": selection}


def save_final_model(model, adapter, output: Path, *, model_path: Path,
                     torch_dtype="auto", device_map="auto") -> Path:
    """文档 11.2（未编号）：保存 theta_final=theta_acc 的目标权重及重载参数。"""
    output.mkdir(parents=True, exist_ok=False)
    weights = {name: model.get_submodule(name).weight.detach().cpu().contiguous().clone()
               for name in adapter.target_module_names()}
    save_file(weights, str(output / "target_weights.safetensors"))
    write_json(output / "model.json", {"editor": "mdlm", "model_path": str(model_path.resolve()),
               "target_layers": list(adapter.target_layers), "torch_dtype": torch_dtype,
               "device_map": device_map})
    return output


def load_final_model(output_dir: str | Path, *, device_map: str | None = None):
    """加载原始模型，再覆盖已保存的最终目标权重；仍需 model.json 指向的原模型。"""
    output = Path(output_dir)
    metadata = read_json(output / "model.json")
    adapter = LLaDA8BInstructAdapter.from_local_tokenizer(
        model_path=metadata["model_path"], target_layers=tuple(metadata["target_layers"]))
    model = adapter.load_model(metadata["model_path"], torch_dtype=metadata["torch_dtype"],
                               device_map=device_map or metadata["device_map"])
    weights = load_file(str(output / "target_weights.safetensors"))
    with torch.no_grad():
        for name, value in weights.items():
            weight = model.get_submodule(name).weight
            weight.copy_(value.to(weight.device))
    return adapter, model
