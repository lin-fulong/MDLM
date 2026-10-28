"""从完整数据集在内存中准备保持统计、训练和开发输入。"""

from __future__ import annotations

import math

from MDLM.states import FactSelection
from util.io import config_digest


DEFAULT_SETTINGS = {"seed": 2026, "dev_fraction": 0.25, "test_fraction": 0.25,
                    "statistics_scale": 1.0, "dev_edit_policy": "edit_probe"}


def encode_keep(adapter, prompt: str, answer: str) -> tuple[dict, list[int], list[int]]:
    rendered = adapter.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    ids = list(adapter.tokenizer.encode(rendered, add_special_tokens=False))
    full = list(adapter.tokenizer.encode(rendered + answer, add_special_tokens=False))
    targets = full[len(ids):]
    if not ids or full[:len(ids)] != ids or not targets:
        raise ValueError("保持题无法建立稳定的短答案 mask 边界")
    if adapter.mask_token_id in ids or set(targets) & set(adapter.tokenizer.all_special_ids):
        raise ValueError("保持题含有不允许的预置 mask 或答案特殊 token")
    positions = list(range(len(ids), len(ids) + len(targets)))
    tokens = ids + [adapter.mask_token_id] * len(targets)
    return {"input_ids": tokens, "attention_mask": [1] * len(tokens),
            "mask_positions": positions, "implicit_position_ids": list(range(len(tokens)))}, positions, [len(ids) - 1, *positions]


def prepare_inputs(cases: list[dict], adapter, *, settings: dict | None = None) -> dict:
    """沿用原来的题目去重、确定性划分和逐位置权重，仅取消文件缓存。"""
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

    pool = {}
    for case in cases:
        for probe in case["prompts"].get("preservation", []):
            key = config_digest({"prompt": probe["prompt"].strip()})
            pool.setdefault(key, {"prompt": probe["prompt"], "answer": probe["answer"]})
    ordered = sorted(pool, key=lambda key: config_digest({"seed": policy["seed"], "prompt": key}))
    if len(ordered) < 2:
        raise ValueError("自动生成保持统计和开发保持输入至少需要两道不同的 preservation 题")
    dev_count = min(len(ordered) - 1, max(1, int(len(ordered) * policy["dev_fraction"])))
    test_count = min(len(ordered) - dev_count - 1, max(1, int(len(ordered) * policy["test_fraction"]))) \
        if policy["test_fraction"] > 0 and len(ordered) >= 3 else 0
    partition = {"dev": ordered[:dev_count], "test": ordered[dev_count:dev_count + test_count],
                 "statistics_train": ordered[dev_count + test_count:]}
    names = adapter.target_module_names()
    statistics, training, development = [], [], []
    for group in ("statistics_train", "dev"):
        selected = partition[group]
        for key in selected:
            record, positions, captures = encode_keep(adapter, pool[key]["prompt"], pool[key]["answer"])
            readout = {"input": record, "positions": positions,
                       "weights": [1 / (len(selected) * len(positions))] * len(positions)}
            if group == "dev":
                development.append(readout)
            else:
                training.append(readout)
                statistics.append({"input": record, "capture_positions": captures,
                                   "weights_by_layer": {name: [scale / (len(selected) * len(captures))] * len(captures)
                                                        for name in names}})
    return {"statistics": statistics, "training": training, "development": development,
            "development_probes": [FactSelection(case["case_id"], probe["probe_id"])
                                   for case in cases for probe in case["prompts"]["edit"]],
            "partition": partition}
