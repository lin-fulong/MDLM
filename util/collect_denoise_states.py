"""P1：从本轮基准模型自主采样固定轨迹，供公式（1）～（3）筛选和监督。"""

from __future__ import annotations

import math
import random

import torch

from MDLM.states import FactTrajectory, fact_seed


def collect_trajectories(model, adapter, cases: list[dict], *, seed=2026, steps=None,
                         temperature=0.0, replay_count=1) -> list[FactTrajectory]:
    """每事实采集一条全 mask 起始的轨迹，并按原种子抽样核对可重放性。"""
    if type(seed) is not int or seed < 0 or (steps is not None and (type(steps) is not int or steps < 1)):
        raise ValueError("seed 必须为非负整数，steps 必须为正整数或 null")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature 必须为有限非负数")
    if type(replay_count) is not int or not 0 <= replay_count <= len(cases):
        raise ValueError("replay_count 必须在 0 与本轮事实数之间")
    target_names = set(adapter.inspect_target_modules(model))
    trajectories = []
    for case in cases:
        probes = case["prompts"]["edit"]
        if len(probes) != 1:
            raise ValueError("每个编辑事实必须且只能选择一个编辑探针")
        probe_id = probes[0]["probe_id"]
        initial, target = adapter.prepare_edit_input(case, probe_id)
        answer_positions = initial.mask_positions
        if len(answer_positions) != len(target.token_ids) or any(
            initial.input_ids[p] != adapter.mask_token_id for p in answer_positions
        ):
            raise ValueError("P0 要求完整短答案 mask 输入")

        # P0：同一全 mask 输入前向两次，核对目标层、logits 和捕获输入。
        captured = adapter.forward_with_capture(model, initial)
        repeated = adapter.forward_with_capture(model, initial)
        if set(captured.module_inputs) != target_names or set(repeated.module_inputs) != target_names:
            raise ValueError("P0 捕获层与目标层不一致")
        if not torch.allclose(captured.mask_logits, repeated.mask_logits, atol=1e-2, rtol=1e-3) or any(
            not torch.allclose(captured.module_inputs[name], repeated.module_inputs[name], atol=1e-2, rtol=1e-3)
            for name in target_names
        ):
            raise ValueError("P0 前向未能在原有容差内重放")

        # 目标答案只附在监督侧；采样器仅接收 initial，不接收 target。
        selected_seed = fact_seed(seed, case["case_id"], probe_id)
        trace = adapter.sample_denoise(model, initial, steps=steps or len(answer_positions),
                                       temperature=temperature, seed=selected_seed)
        trajectories.append(FactTrajectory.from_sample(target, selected_seed, trace))

    # 重放使用实际轨迹步数及逐事实种子，保持原抽样顺序和比较规则。
    for index in sorted(random.Random(seed).sample(range(len(trajectories)), replay_count)):
        saved = trajectories[index]
        replay = adapter.sample_denoise(model, saved.trace.pre_forward_inputs[0],
                                       steps=len(saved.trace.pre_forward_inputs),
                                       temperature=temperature, seed=saved.seed)
        if replay != saved.trace:
            raise ValueError(f"{saved.case_id}: 去噪轨迹或 logits 未能重放")
    return trajectories
