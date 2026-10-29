"""自主采样去噪轨迹，按公式（1）～（3）筛选兼容状态并建立固定监督。"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass

import torch

from model.base import DenoiseTrace, EditTarget, MaskedEditInput


@dataclass(frozen=True)
class FactSelection:
    """事实及其问题的选择标识。"""

    case_id: str
    probe_id: str


def fact_seed(seed: int, case_id: str, probe_id: str) -> int:
    """沿用逐事实随机流：改变处理顺序不改变采样种子。"""
    key = f"{seed}:{case_id}:{probe_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63)


def input_record(model_input: MaskedEditInput) -> dict:
    """保留完整输入字段，供状态去重后的监督结构使用；不写入文件。"""
    return {
        "input_ids": list(model_input.input_ids),
        "attention_mask": list(model_input.attention_mask),
        "mask_positions": list(model_input.mask_positions),
        "subject_last_position": model_input.subject_last_position,
        "implicit_position_ids": list(range(len(model_input.input_ids))),
    }


def input_from_record(record: dict) -> MaskedEditInput:
    """还原模型输入，核对 LLaDA 从零开始的隐式位置编号。"""
    if record["implicit_position_ids"] != list(range(len(record["input_ids"]))):
        raise ValueError("saved implicit position IDs differ from the model input length")
    return MaskedEditInput(
        tuple(record["input_ids"]),
        tuple(record["attention_mask"]),
        tuple(record["mask_positions"]),
        record["subject_last_position"],
    )


@dataclass(frozen=True)
class FactTrajectory:
    """一个事实的自主采样轨迹；目标答案独立存放，只用于筛选和监督。"""

    case_id: str
    probe_id: str
    answer_positions: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    seed: int
    trace: DenoiseTrace

    @classmethod
    def from_sample(cls, target: EditTarget, seed: int, trace: DenoiseTrace) -> "FactTrajectory":
        """从全 mask 初态取得答案窗口，组装事实轨迹。"""
        if not trace.pre_forward_inputs:
            raise ValueError(f"{target.case_id}: sampler returned no pre-forward input")
        return cls(target.case_id, target.probe_id, trace.pre_forward_inputs[0].mask_positions,
                   target.token_ids, seed, trace)


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


def classify_fact(trajectory: FactTrajectory, mask_token_id: int) -> tuple[list[dict], dict]:
    """公式（1）：完整输入去重后，保留尚有 mask 且已揭示答案均正确的状态。"""
    trace = trajectory.trace
    inputs = trace.pre_forward_inputs
    answer_positions = trajectory.answer_positions
    targets = trajectory.target_token_ids

    # 本算法使用非空答案窗口、全 mask 初态和不重新掩码的受控去噪轨迹。
    if not answer_positions or answer_positions != tuple(sorted(set(answer_positions))):
        raise ValueError(f"{trajectory.case_id}: answer positions must be nonempty, unique and ordered")
    if len(answer_positions) != len(targets):
        raise ValueError(f"{trajectory.case_id}: answer positions and target length differ")
    if not inputs:
        raise ValueError(f"{trajectory.case_id}: full-mask initial state is missing")
    initial = inputs[0]
    if any(position < 0 or position >= len(initial.input_ids) for position in answer_positions):
        raise ValueError(f"{trajectory.case_id}: answer position is out of range")
    if not 0 <= initial.subject_last_position < min(answer_positions):
        raise ValueError(f"{trajectory.case_id}: subject must precede the answer window")
    if initial.mask_positions != answer_positions:
        raise ValueError(f"{trajectory.case_id}: first state is not the full-mask initial state")
    if len(trace.revealed_tokens) != len(inputs):
        raise ValueError(f"{trajectory.case_id}: reveal count differs from forward count")
    if len(trace.mask_logit_sha256) != len(inputs):
        raise ValueError(f"{trajectory.case_id}: logit digest count differs from forward count")

    compatible: list[dict] = []
    discarded: list[dict] = []
    seen: set[MaskedEditInput] = set()
    previous_masks = set(answer_positions)
    for step, state in enumerate(inputs):
        if (len(state.input_ids) != len(initial.input_ids)
                or len(state.input_ids) != len(state.attention_mask)
                or any(value != 1 for value in state.attention_mask)):
            raise ValueError(f"{trajectory.case_id}: saved attention input differs from the forward contract")
        current_masks = set(state.mask_positions)
        if not current_masks <= previous_masks:
            raise ValueError(f"{trajectory.case_id}: a revealed token was masked again")
        if state.mask_positions != tuple(
            position for position, token in enumerate(state.input_ids) if token == mask_token_id
        ):
            raise ValueError(f"{trajectory.case_id}: saved mask positions differ from input IDs")
        previous_masks = current_masks
        # 按全部模型输入字段去重，不能只比较剩余 mask 或可见文本。
        if state in seen:
            discarded.append({"step": step, "reason": "duplicate_full_input"})
            continue
        seen.add(state)
        if not current_masks:
            discarded.append({"step": step, "reason": "terminal"})
        elif any(
            state.input_ids[position] != target
            for position, target in zip(answer_positions, targets)
            if position not in current_masks
        ):
            discarded.append({"step": step, "reason": "conflict"})
        else:
            compatible.append({"step": step, "input": input_record(state),
                               "remaining_mask_positions": list(state.mask_positions)})

    # 核对每次揭示确实产生下一输入，保持上下文、已揭示内容和 mask 变化一致。
    if trace.final_input.mask_positions or mask_token_id in trace.final_input.input_ids:
        raise ValueError(f"{trajectory.case_id}: denoising did not reach a terminal state")
    for step, current in enumerate(inputs):
        following = inputs[step + 1] if step + 1 < len(inputs) else trace.final_input
        expected_ids = list(current.input_ids)
        revealed_positions = set()
        for position, token in trace.revealed_tokens[step]:
            if position not in current.mask_positions or position in revealed_positions or token == mask_token_id:
                raise ValueError(f"{trajectory.case_id}: invalid revealed token at step {step}")
            expected_ids[position] = token
            revealed_positions.add(position)
        expected = MaskedEditInput(
            tuple(expected_ids),
            current.attention_mask,
            tuple(position for position in current.mask_positions if position not in revealed_positions),
            current.subject_last_position,
        )
        if following != expected:
            raise ValueError(f"{trajectory.case_id}: saved input transition differs from revealed tokens at step {step}")
    report = {
        "case_id": trajectory.case_id,
        "probe_id": trajectory.probe_id,
        "pre_forward_count": len(inputs),
        "unique_input_count": len(seen),
        "compatible_count": len(compatible),
        "full_mask_initial_compatible": bool(compatible and compatible[0]["step"] == 0),
        "conflict_count": sum(item["reason"] == "conflict" for item in discarded),
        "duplicate_count": sum(item["reason"] == "duplicate_full_input" for item in discarded),
        "terminal_pre_forward_count": sum(item["reason"] == "terminal" for item in discarded),
        "terminal_final_count": 1,
        "discarded": discarded,
    }
    return compatible, report


def build_supervision(trajectories: list[FactTrajectory], mask_token_id: int) -> tuple[list[dict], dict]:
    """公式（2）（3）：为事实、兼容状态、剩余 mask 三级平均准备位置权重。"""
    fact_count = len(trajectories)
    if not fact_count:
        raise ValueError("no selected facts were collected")
    fixed: list[dict] = []
    reports: list[dict] = []
    for trajectory in trajectories:
        states, report = classify_fact(trajectory, mask_token_id)
        answer_by_position = dict(zip(trajectory.answer_positions, trajectory.target_token_ids))
        total_states = len(states)
        for state in states:
            positions = state["remaining_mask_positions"]
            # 公式（3）：w_i,t,j = 1 / (N·|T_i|·|M_i,t|)，不再对样本另取平均。
            weight = 1.0 / (fact_count * total_states * len(positions))
            state["supervision"] = [
                {"position": position, "target_token_id": answer_by_position[position], "weight": weight}
                for position in positions
            ]
        report["weight_sum"] = math.fsum(
            term["weight"] for state in states for term in state["supervision"]
        )
        fixed.append({
            "case_id": trajectory.case_id,
            "probe_id": trajectory.probe_id,
            "answer_positions": list(trajectory.answer_positions),
            "target_token_ids": list(trajectory.target_token_ids),
            "states": states,
        })
        reports.append(report)
    # 每个事实总权重为 1/N，全体为 1；合法的全 mask 初态保证 T_i 非空。
    total_weight = math.fsum(item["weight_sum"] for item in reports)
    formula_checks_passed = all(
        item["full_mask_initial_compatible"]
        and math.isclose(item["weight_sum"], 1 / fact_count, abs_tol=1e-12)
        for item in reports
    ) and math.isclose(total_weight, 1.0, abs_tol=1e-12)
    return fixed, {
        "fact_count": fact_count,
        "compatible_state_count": sum(item["compatible_count"] for item in reports),
        "conflict_state_count": sum(item["conflict_count"] for item in reports),
        "supervision_position_count": sum(
            len(state["supervision"]) for fact in fixed for state in fact["states"]
        ),
        "global_weight_sum": total_weight,
        "formula_checks_passed": formula_checks_passed,
        "facts": reports,
    }


@dataclass(frozen=True)
class EditTrainingExample:
    """一个固定兼容状态及其监督；模型输入与目标 token 分开存放。"""

    case_id: str
    model_input: MaskedEditInput
    positions: tuple[int, ...]
    targets: tuple[int, ...]
    weights: tuple[float, ...]


def edit_training_examples(facts: list[dict]) -> list[EditTrainingExample]:
    """沿用公式（3）的权重，交给 optimize.py 计算公式（2）的损失。"""
    return [EditTrainingExample(
        fact["case_id"], input_from_record(state["input"]),
        tuple(term["position"] for term in state["supervision"]),
        tuple(term["target_token_id"] for term in state["supervision"]),
        tuple(term["weight"] for term in state["supervision"]),
    ) for fact in facts for state in fact["states"]]
