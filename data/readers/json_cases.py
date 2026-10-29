"""读取可信案例，统一列表形式，并按源格式分派转换。"""

from __future__ import annotations

import json
from pathlib import Path


def load_cases(path: str | Path) -> list[dict]:
    """读取 JSON/JSONL，返回原顺序的案例列表；审核源格式交给 reviewed_cases。"""
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        if path.suffix == ".jsonl":
            payload = [json.loads(line) for line in stream if line.strip()]
        else:
            payload = json.load(stream)
    cases = payload if isinstance(payload, list) else [payload]
    # 首条决定整份数据的格式；保留后续公式（2）（3）的事实及公式（4）的 K 列顺序。
    if cases and "edit_requests" in cases[0] and "edit_questions" in cases[0]:
        from data.readers.reviewed_cases import adapt_reviewed_cases

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
