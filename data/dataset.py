"""读取和转换编辑案例，准备固定保持划分、上下文及位置权重。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from MDLM.states import FactSelection
from util.io import config_digest


def load_cases(path: str | Path) -> list[dict]:
    """读取 JSON/JSONL，返回原顺序的案例列表；审核源格式调用 adapt_reviewed_cases。"""
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        if path.suffix == ".jsonl":
            payload = [json.loads(line) for line in stream if line.strip()]
        else:
            payload = json.load(stream)
    cases = payload if isinstance(payload, list) else [payload]
    # 首条决定整份数据的格式；保留后续公式（2）（3）的事实及公式（4）的 K 列顺序。
    if cases and "edit_requests" in cases[0] and "edit_questions" in cases[0]:
        return adapt_reviewed_cases(cases)
    return cases


def case_purpose(case: dict) -> str:
    """返回旧用途标签；当前无正式调用方，不参与选例或保持数据划分。"""
    if case["quality_control"]["status"] == "example_only":
        return "example_only"
    if case.get("split") == "test":
        return "independent_test"
    if case.get("split") in ("gold", "pilot", "dev"):
        return "development"
    return "edit"


def _probe(source: dict, *, answer_field: str, used_for_edit: bool, role: str) -> dict:
    """把一道题映射为探针，保留原题、指定答案及用途，不重新分词。"""
    return {"probe_id": source["probe_id"], "prompt": source["question"],
            "answer": source[answer_field], "used_for_edit": used_for_edit, "role": role}


def adapt_reviewed_cases(rows: list[dict]) -> list[dict]:
    """按源案例/请求顺序构造 D_edit；同源事实由 runner 分组联合编辑。"""
    adapted = []
    for row in rows:
        if not row["gold_ready"]:
            continue
        requests, questions = row["edit_requests"], row["edit_questions"]
        count = len(requests)
        if count != len(questions):
            raise ValueError(f"{row['case_id']}: edit_requests and edit_questions cannot be paired by index")
        for index, (request, question) in enumerate(zip(requests, questions)):
            # 编辑题没有 request_id，沿用同下标配对；公式（2）（3）（4）沿用此事实顺序。
            case_id = row["case_id"] if count == 1 else request["request_id"]
            # 保持题只挂在首个请求下，供后续公式（5）（10）（16）的固定输入划分使用。
            prompts = {"edit": [_probe(question, answer_field="expected_new_answer",
                                        used_for_edit=True, role="edit")],
                       "free_form_qa": [], "portability": [],
                       "preservation": [_probe(source, answer_field="expected_unchanged_answer",
                                               used_for_edit=False, role="preservation")
                                        for source in row["preservation_questions"]] if index == 0 else []}
            # fact_core 保存请求事实；真正的逐 token 监督取上面编辑题的 answer。
            adapted.append({
                "case_id": case_id,
                "source_case_id": row["case_id"],
                "source_request_id": request["request_id"],
                "dataset_group": row["dataset_group"],
                "fact_core": {"subject": request["subject"], "relation": request["relation"],
                              "old_object": request["old_answer"],
                              "new_object": request["experimental_new_answer"]},
                "prompts": prompts,
                "quality_control": {"status": row["review_status"]},
            })
    return adapted


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


def main() -> None:
    """可选转换入口：读取 JSONL，按源/事实编号选例，写出 JSON 或 JSONL。"""
    parser = argparse.ArgumentParser(description="将 gold_ready 源案例转换为编辑事实。")
    parser.add_argument("source", type=Path, help="reviewed cases.jsonl")
    parser.add_argument("output", type=Path, help="output .json or .jsonl; source is never modified")
    parser.add_argument("--case-id", help="select one source or expanded runtime case ID")
    args = parser.parse_args()
    with args.source.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    cases = adapt_reviewed_cases(rows)
    if args.case_id is not None:
        cases = [case for case in cases if args.case_id in (case["case_id"], case["source_case_id"])]
        if not cases:
            parser.error(f"no gold_ready case matches {args.case_id!r}")
    if args.output.suffix == ".jsonl":
        contents = "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases)
    elif args.output.suffix == ".json":
        contents = json.dumps(cases[0] if len(cases) == 1 else cases, ensure_ascii=False, indent=2) + "\n"
    else:
        parser.error("output must end in .json or .jsonl")
    args.output.write_text(contents, encoding="utf-8")
    print(f"wrote {len(cases)} editing cases to {args.output}")


if __name__ == "__main__":
    main()
