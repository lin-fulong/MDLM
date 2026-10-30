"""MDLM 输入与状态：固定保持划分，自主去噪，筛选兼容状态并建立监督。"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch

from data.validation.alignment import align_edit_prompt
from model.base import ForwardCapture, KeepModelInput, MaskedEditInput
from util.io import config_digest
from util.nethook import tensor_sha256


@dataclass(frozen=True)
class EditTarget:
    """公式（1）～（3）的目标短答案 a_i*；仅用于筛选和监督，不传入自主采样器。"""

    case_id: str
    probe_id: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class DenoiseTrace:
    """公式（1）的采样来源：逐步前向前输入、logits 摘要、揭示记录及终态。

    前三项按采样步一一对应；终态单独保存，不作为非空 mask 监督状态。
    """

    pre_forward_inputs: tuple[MaskedEditInput, ...]
    mask_logit_sha256: tuple[str, ...]
    revealed_tokens: tuple[tuple[tuple[int, int], ...], ...]  # 每步的 (位置, token_id)。
    final_input: MaskedEditInput


@dataclass(frozen=True)
class FactSelection:
    """事实及其问题的选择标识。"""

    case_id: str
    probe_id: str


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
        return cls(target.case_id, target.probe_id, trace.pre_forward_inputs[0].mask_positions,
                   target.token_ids, seed, trace)


@dataclass(frozen=True)
class EditTrainingExample:
    """一个固定兼容状态及其监督；模型输入与目标 token 分开存放。"""

    case_id: str
    model_input: MaskedEditInput
    positions: tuple[int, ...]
    targets: tuple[int, ...]
    weights: tuple[float, ...]


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
    """还原模型输入；LLaDA 继续使用从零开始的隐式位置编号。"""
    return MaskedEditInput(
        tuple(record["input_ids"]),
        tuple(record["attention_mask"]),
        tuple(record["mask_positions"]),
        record["subject_last_position"],
    )


def keep_input_from_record(record):
    """还原已准备的保持输入。"""
    return KeepModelInput(tuple(record["input_ids"]), tuple(record["attention_mask"]), tuple(record["mask_positions"]))


def prepare_edit_input(adapter, case: dict, probe_id: str) -> tuple[MaskedEditInput, EditTarget]:
    """公式（1）～（4）：选择编辑探针，分别构造全 mask 初态与监督答案。"""
    probe = next(p for p in case["prompts"]["edit"] if p["probe_id"] == probe_id)
    return _prepare_short_answer(adapter, case, probe)


def prepare_development_input(adapter, case: dict, probe_id: str) -> tuple[MaskedEditInput, EditTarget]:
    """公式（18）：从各探针组查找固定开发题，沿用短答案对齐。"""
    probe = next(p for group in case["prompts"].values() for p in group if p["probe_id"] == probe_id)
    return _prepare_short_answer(adapter, case, probe)


def _prepare_short_answer(adapter, case: dict, probe: dict) -> tuple[MaskedEditInput, EditTarget]:
    """按真实聊天模板定位主体；答案只决定窗口长度并单独返回为监督。"""
    rendered = adapter.render_prompt(probe["prompt"])
    alignment = align_edit_prompt(
        adapter.tokenizer, rendered, case["fact_core"]["subject"], probe["answer"]
    )
    begin = len(alignment.prompt_ids)
    positions = tuple(range(begin, begin + len(alignment.answer_ids)))
    token_ids = alignment.prompt_ids + (adapter.mask_token_id,) * len(positions)
    return (
        MaskedEditInput(
            input_ids=token_ids,
            attention_mask=(1,) * len(token_ids),
            mask_positions=positions,
            subject_last_position=alignment.subject_last_position,
        ),
        EditTarget(case["case_id"], probe["probe_id"], alignment.answer_ids),
    )


DEFAULT_SETTINGS = {"seed": 2026, "dev_fraction": 0.25, "test_fraction": 0.25,
                    "statistics_scale": 1.0}


def encode_keep(adapter, prompt: str, answer: str) -> tuple[dict, list[int], list[int]]:
    """编码全 mask 短答案上下文，返回输入、KL 读出位置和 C 的激活捕获位置。"""
    rendered = adapter.render_prompt(prompt)
    prompt_ids = list(adapter.tokenizer.encode(rendered, add_special_tokens=False))
    full_ids = list(adapter.tokenizer.encode(rendered + answer, add_special_tokens=False))
    begin = len(prompt_ids)
    answer_ids = full_ids[begin:]
    # 公式（10）（16）（17）：KL 只读答案 mask；答案内容不进入模型输入。
    positions = list(range(begin, begin + len(answer_ids)))
    tokens = prompt_ids + [adapter.mask_token_id] * len(answer_ids)
    return {"input_ids": tokens, "attention_mask": [1] * len(tokens),
            "mask_positions": positions, "implicit_position_ids": list(range(len(tokens)))}, positions, [begin - 1, *positions]


def prepare_inputs(cases: list[dict], adapter, *, settings: dict | None = None,
                   keep_pool: list[dict] | None = None) -> dict:
    """从显式保持池或完整案例集准备输入；沿用原去重、划分和位置权重。"""
    policy = {**DEFAULT_SETTINGS, **(settings or {})}
    scale = policy["statistics_scale"]

    # 按 strip 后的题目摘要去重，保留首次出现的原始 prompt 和 answer。
    pool = {}
    probes = keep_pool if keep_pool is not None else (
        probe for case in cases for probe in case["prompts"].get("preservation", []))
    for probe in probes:
        key = config_digest({"prompt": probe["prompt"].strip()})
        pool.setdefault(key, {"prompt": probe["prompt"], "answer": probe["answer"]})
    ordered = sorted(pool, key=lambda key: config_digest({"seed": policy["seed"], "prompt": key}))
    count = len(ordered)
    # 摘要排序不使用全局随机状态；计数沿用截断、至少一条开发题及一条训练题的规则。
    dev_count = min(count - 1, max(1, int(count * policy["dev_fraction"])))
    test_count = min(count - dev_count - 1, max(1, int(count * policy["test_fraction"]))) \
        if policy["test_fraction"] > 0 and count >= 3 else 0
    train_begin = dev_count + test_count
    partition = {"dev": ordered[:dev_count], "test": ordered[dev_count:train_begin],
                 "statistics_train": ordered[train_begin:]}
    names = adapter.target_module_names()
    statistics, training, development = [], [], []
    # test 仅保留划分编号；先编码训练再编码开发，C 与保持训练共用训练上下文。
    for group, readouts in (("statistics_train", training), ("dev", development)):
        selected = partition[group]
        sample_count = len(selected)
        for key in selected:
            record, positions, captures = encode_keep(adapter, pool[key]["prompt"], pool[key]["answer"])
            # 公式（10）（16）的 b_{h,j}=1/(题数×该题答案 mask 数)，各组总权重为 1。
            readouts.append({"input": record, "positions": positions,
                             "weights": [1 / (sample_count * len(positions))] * len(positions)})
            if group == "statistics_train":
                # 公式（5）的 ω=scale/(题数×捕获位置数)；包含提示末位置，总尺度为 scale。
                statistics.append({"input": record, "capture_positions": captures,
                                   "weights_by_layer": {name: [scale / (sample_count * len(captures))] * len(captures)
                                                        for name in names}})
    # 公式（18）的 F_dev：沿用 edit_probe，按原案例/探针顺序选择，不自动改为留出题。
    return {"statistics": statistics, "training": training, "development": development,
            "development_probes": [FactSelection(case["case_id"], probe["probe_id"])
                                   for case in cases for probe in case["prompts"]["edit"]],
            "partition": partition}


def fact_seed(seed: int, case_id: str, probe_id: str) -> int:
    """沿用逐事实随机流：改变处理顺序不改变采样种子。"""
    key = f"{seed}:{case_id}:{probe_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63)


def capture_edit(model, adapter, model_input: MaskedEditInput) -> ForwardCapture:
    """公式（4）：由算法选择主体末位置，适配器只捕获指定位置的实际输入。"""
    result = adapter.forward_with_capture(
        model, ((model_input, (model_input.subject_last_position,)),),
    )[0]
    return ForwardCapture(result.mask_logits, {name: value[0] for name, value in result.module_inputs.items()})


def sample_denoise(
    model, adapter, model_input: MaskedEditInput, *, steps: int, temperature: float, seed: int
) -> DenoiseTrace:
    """公式（1）的状态来源：自主采样并按置信度揭示现有答案 mask，不接收目标答案。"""
    mask_count = len(model_input.mask_positions)

    generator = None
    current_ids = list(model_input.input_ids)
    remaining = list(model_input.mask_positions)
    states: list[MaskedEditInput] = []
    logit_digests: list[str] = []
    reveals: list[tuple[tuple[int, int], ...]] = []
    blocked_ids = sorted(set(adapter.tokenizer.all_special_ids) | {adapter.mask_token_id})

    for step in range(steps):
        # 公式（1）的候选状态必须是每次前向之前的独立快照。
        current = MaskedEditInput(
            tuple(current_ids), model_input.attention_mask, tuple(remaining), model_input.subject_last_position
        )
        states.append(current)
        raw_logits = capture_edit(model, adapter, current).mask_logits
        # 原精度 logits 摘要保留在轨迹中；采样副本才转 float32。
        logit_digests.append(tensor_sha256(raw_logits))
        logits = raw_logits.float().clone()
        logits[:, [token for token in blocked_ids if 0 <= token < logits.shape[-1]]] = -torch.inf
        # 当前采样约定（无独立公式编号）：屏蔽特殊 token 后，T>0 按 softmax(logits/T) 抽样。
        probabilities = torch.softmax(logits / temperature if temperature else logits, dim=-1)
        if temperature:
            if generator is None:
                generator = torch.Generator(device=logits.device)
                generator.manual_seed(seed)
            predicted = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
        else:
            predicted = logits.argmax(dim=-1)
        confidence = probabilities.gather(1, predicted[:, None]).squeeze(1)

        # 每步揭示 floor(M/S) 个，前 M%S 步多一个；置信度并列时按原位置排序。
        count = mask_count // steps + (step < mask_count % steps)
        ranked = sorted(range(len(remaining)), key=lambda i: (-float(confidence[i]), remaining[i]))
        selected = sorted(ranked[:count])
        revealed = tuple((remaining[i], int(predicted[i])) for i in selected)
        for position, token_id in revealed:
            current_ids[position] = token_id
        reveals.append(revealed)
        selected_set = set(selected)
        remaining = [position for i, position in enumerate(remaining) if i not in selected_set]

    return DenoiseTrace(
        tuple(states),
        tuple(logit_digests),
        tuple(reveals),
        MaskedEditInput(tuple(current_ids), model_input.attention_mask, (), model_input.subject_last_position),
    )


def collect_trajectories(model, adapter, cases: list[dict], *, seed=2026, steps=None,
                         temperature=0.0) -> list[FactTrajectory]:
    """每事实采集一条全 mask 起始的轨迹；保持原逐事实随机流。"""
    trajectories = []
    for case in cases:
        probes = case["prompts"]["edit"]
        probe_id = probes[0]["probe_id"]
        initial, target = prepare_edit_input(adapter, case, probe_id)
        answer_positions = initial.mask_positions

        # 目标答案只附在监督侧；采样器仅接收 initial，不接收 target。
        selected_seed = fact_seed(seed, case["case_id"], probe_id)
        trace = sample_denoise(model, adapter, initial, steps=steps or len(answer_positions),
                                       temperature=temperature, seed=selected_seed)
        trajectories.append(FactTrajectory.from_sample(target, selected_seed, trace))

    return trajectories


def classify_fact(trajectory: FactTrajectory, mask_token_id: int) -> tuple[list[dict], dict]:
    """公式（1）：完整输入去重后，保留尚有 mask 且已揭示答案均正确的状态。"""
    trace = trajectory.trace
    inputs = trace.pre_forward_inputs
    answer_positions = trajectory.answer_positions
    targets = trajectory.target_token_ids

    compatible: list[dict] = []
    discarded: list[dict] = []
    seen: set[MaskedEditInput] = set()
    for step, state in enumerate(inputs):
        current_masks = set(state.mask_positions)
        # 按全部模型输入字段去重，不能只比较剩余 mask 或可见文本。
        if state in seen:
            discarded.append({"step": step, "reason": "duplicate_full_input"})
            continue
        seen.add(state)
        if not current_masks:
            discarded.append({"step": step, "reason": "terminal"})
        elif any(
            state.input_ids[position] != target
            for position, target in zip(answer_positions, targets, strict=True)
            if position not in current_masks
        ):
            discarded.append({"step": step, "reason": "conflict"})
        else:
            compatible.append({"step": step, "input": input_record(state),
                               "remaining_mask_positions": list(state.mask_positions)})

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
    fixed: list[dict] = []
    reports: list[dict] = []
    for trajectory in trajectories:
        states, report = classify_fact(trajectory, mask_token_id)
        answer_by_position = dict(zip(trajectory.answer_positions, trajectory.target_token_ids, strict=True))
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


def edit_training_examples(facts: list[dict]) -> list[EditTrainingExample]:
    """沿用公式（3）的权重，交给 optimize.py 计算公式（2）的损失。"""
    return [EditTrainingExample(
        fact["case_id"], input_from_record(state["input"]),
        tuple(term["position"] for term in state["supervision"]),
        tuple(term["target_token_id"] for term in state["supervision"]),
        tuple(term["weight"] for term in state["supervision"]),
    ) for fact in facts for state in fact["states"]]
