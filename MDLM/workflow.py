"""内存中的 P1～P5：状态 → K/C → Q → R → 实际合并与开发验收。"""

from __future__ import annotations

import math
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
from util.io import read_json, require, write_json


@torch.no_grad()
def reference_examples(model, adapter, rows) -> list[KeepTrainingExample]:
    """在首次编辑前计算 theta_0 的完整词表参考，供全部轮次复用。"""
    examples = []
    for row in rows:
        model_input = keep_input_from_record(row["input"])
        positions, weights = tuple(row["positions"]), tuple(row["weights"])
        require(positions and len(positions) == len(weights) and set(positions) <= set(model_input.mask_positions)
                and all(model_input.input_ids[p] == adapter.mask_token_id for p in positions)
                and all(isinstance(w, (int, float)) and not isinstance(w, bool) and math.isfinite(w) and w >= 0
                        for w in weights), "保持 KL 的位置和权重不符合公式（16）")
        logs = log_probabilities(adapter.forward_logits(model, model_input, positions)).cpu()
        examples.append(KeepTrainingExample(model_input, positions, weights, logs))
    require(examples and math.isclose(math.fsum(w for item in examples for w in item.weights),
                                      1.0, abs_tol=1e-12, rel_tol=1e-12),
            "保持参考的全局位置权重之和必须为 1")
    return examples


def edit_round(model, adapter, trajectories, statistics, dev_edits, dev_keeps, originals, *,
               writeback: WritebackConfig, optimization: ResidualOptimizationConfig,
               acceptance: AcceptanceConfig, keep_train=(), p2_keys=None, p2_keep=None) -> dict:
    """直接使用既有数学实现；不创建阶段目录，不序列化中间张量。"""
    names = set(adapter.target_module_names())
    require(names == set(writeback.ridge_by_layer) == set(optimization.rho_by_layer),
            "P3/P4 的逐层参数必须与目标层完全一致")
    facts, supervision = build_supervision(trajectories, adapter.mask_token_id)
    require(supervision["formula_checks_passed"], "P1 的三级监督权重不符合公式（2）（3）")
    edits = edit_training_examples(facts)

    # P2/P3：K 的列顺序保持事实顺序，C 逐层计算并按原存储精度交给 Q。
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
        return {**result, "decision": "rejected", "selected_alpha": None,
                "baseline_restored": True, "selection": None}

    # P5：沿用 W_b + αRQ、因子/合并前向等价检查、三个开发门槛和精确回退。
    merger = WeightMerge(model, factors, optimized.residuals,
                         compute_dtype=acceptance.compute_dtype, chunk_size=acceptance.chunk_size)
    try:
        merger.diagnostics(factors, optimized.residuals, keys)
        selection = select_merged_candidate(merger, adapter, factors, optimized.residuals,
                                            dev_edits, dev_keeps, originals, acceptance)
    except Exception:
        merger.restore()
        raise
    return {**result, "decision": selection["status"], "selected_alpha": selection["selected_alpha"],
            "baseline_restored": selection["baseline_restored"], "selection": selection}


def save_final_model(model, adapter, output: Path, *, model_path: Path,
                     torch_dtype="auto", device_map="auto") -> Path:
    """只保存最终目标层权重，加载时覆盖到原始模型上。"""
    output.mkdir(parents=True, exist_ok=False)
    weights = {name: model.get_submodule(name).weight.detach().cpu().contiguous().clone()
               for name in adapter.target_module_names()}
    save_file(weights, str(output / "target_weights.safetensors"))
    saved = load_file(str(output / "target_weights.safetensors"))
    require(set(saved) == set(weights) and all(
        saved[name].dtype == value.dtype and torch.equal(saved[name], value) for name, value in weights.items()),
        "最终权重读回不一致")
    write_json(output / "model.json", {"editor": "mdlm", "model_path": str(model_path.resolve()),
               "target_layers": list(adapter.target_layers), "torch_dtype": torch_dtype,
               "device_map": device_map})
    return output


def load_final_model(output_dir: str | Path, *, device_map: str | None = None):
    """最终结果自包含于原始模型路径和目标层权重，不依赖 P1～P4 文件。"""
    output = Path(output_dir)
    metadata = read_json(output / "model.json")
    adapter = LLaDA8BInstructAdapter.from_local_tokenizer(
        model_path=metadata["model_path"], target_layers=tuple(metadata["target_layers"]))
    model = adapter.load_model(metadata["model_path"], torch_dtype=metadata["torch_dtype"],
                               device_map=device_map or metadata["device_map"])
    adapter.inspect_target_modules(model)
    weights = load_file(str(output / "target_weights.safetensors"))
    require(set(weights) == set(adapter.target_module_names()), "最终权重的目标层不一致")
    snapshot = adapter.snapshot_target_weights(model)
    try:
        with torch.no_grad():
            for name, value in weights.items():
                weight = model.get_submodule(name).weight
                require(value.shape == weight.shape and value.dtype == weight.dtype
                        and bool(torch.isfinite(value).all()), "最终权重的形状、精度或有限性无效")
            for name, value in weights.items():
                model.get_submodule(name).weight.copy_(value.to(model.get_submodule(name).weight.device))
    except Exception:
        adapter.restore_target_weights(model, snapshot)
        raise
    return adapter, model
