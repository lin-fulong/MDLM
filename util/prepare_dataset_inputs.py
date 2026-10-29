"""为公式（5）（10）（16）准备固定保持输入，为公式（18）选择开发编辑探针。"""

from __future__ import annotations

import math

from MDLM.states import FactSelection
from util.io import config_digest


DEFAULT_SETTINGS = {"seed": 2026, "dev_fraction": 0.25, "test_fraction": 0.25,
                    "statistics_scale": 1.0, "dev_edit_policy": "edit_probe"}


def encode_keep(adapter, prompt: str, answer: str) -> tuple[dict, list[int], list[int]]:
    """编码全 mask 短答案上下文，返回输入、KL 读出位置和 C 的激活捕获位置。"""
    rendered = adapter.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    prompt_ids = list(adapter.tokenizer.encode(rendered, add_special_tokens=False))
    full_ids = list(adapter.tokenizer.encode(rendered + answer, add_special_tokens=False))
    begin = len(prompt_ids)
    answer_ids = full_ids[begin:]
    if not prompt_ids or full_ids[:begin] != prompt_ids or not answer_ids:
        raise ValueError("保持题无法建立稳定的短答案 mask 边界")
    if adapter.mask_token_id in prompt_ids or set(answer_ids) & set(adapter.tokenizer.all_special_ids):
        raise ValueError("保持题含有不允许的预置 mask 或答案特殊 token")
    # 公式（10）（16）（17）：KL 只读答案 mask；答案内容不进入模型输入。
    positions = list(range(begin, begin + len(answer_ids)))
    tokens = prompt_ids + [adapter.mask_token_id] * len(answer_ids)
    return {"input_ids": tokens, "attention_mask": [1] * len(tokens),
            "mask_positions": positions, "implicit_position_ids": list(range(len(tokens)))}, positions, [begin - 1, *positions]


def prepare_inputs(cases: list[dict], adapter, *, settings: dict | None = None) -> dict:
    """从完整案例集去重并固定保持划分，按题/位置均分权重，返回内存记录。"""
    policy = {**DEFAULT_SETTINGS, **(settings or {})}
    if set(policy) != set(DEFAULT_SETTINGS) or policy["dev_edit_policy"] != "edit_probe":
        raise ValueError("不支持的输入生成参数；dev_edit_policy 当前支持 edit_probe")
    if type(policy["seed"]) is not int or policy["seed"] < 0:
        raise ValueError("seed 必须为非负整数")
    fractions = (policy["dev_fraction"], policy["test_fraction"])
    if any(not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) or not 0 <= x < 1
           for x in fractions) or not 0 < fractions[0] or sum(fractions) >= 1:
        raise ValueError("保持题划分比例必须满足 dev>0、test>=0、dev+test<1")
    scale = policy["statistics_scale"]
    if not isinstance(scale, (int, float)) or isinstance(scale, bool) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("statistics_scale 必须为有限正数")
    if not cases:
        raise ValueError("所选数据集没有可执行的编辑案例")

    # 按 strip 后的题目摘要去重，保留首次出现的原始 prompt 和 answer。
    pool = {}
    for case in cases:
        for probe in case["prompts"].get("preservation", []):
            key = config_digest({"prompt": probe["prompt"].strip()})
            pool.setdefault(key, {"prompt": probe["prompt"], "answer": probe["answer"]})
    ordered = sorted(pool, key=lambda key: config_digest({"seed": policy["seed"], "prompt": key}))
    count = len(ordered)
    if count < 2:
        raise ValueError("自动生成保持统计和开发保持输入至少需要两道不同的 preservation 题")
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
