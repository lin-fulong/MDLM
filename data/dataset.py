"""统一数据集入口；保留原生 JSON/JSONL 读取和审核源格式转换接口。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import islice
import json
from pathlib import Path

from data.readers.base import iter_records, validate_limit
from data.readers.fever import FEVERReader
from data.readers.ultraeditbench import UltraEditBenchReader
from data.readers.wikibigedit import WikiBigEditReader
from data.readers.zsre import ZSREReader
from util.io import project_path


# native 复用下面已有的读取函数，保留其格式识别、编号和展开规则。
READERS = {"native": None, "zsre": ZSREReader, "fever": FEVERReader,
           "wikibigedit": WikiBigEditReader, "ultraeditbench": UltraEditBenchReader}


@dataclass
class DatasetBundle:
    cases: list[dict]
    keep_pool: list[dict] | None
    metadata: dict


def _native_cases(path, limit):
    if limit is None:
        return load_cases(path)
    rows = [row for _, row in iter_records(path, limit=limit)]
    if rows and "edit_requests" in rows[0] and "edit_questions" in rows[0]:
        return adapt_reviewed_cases(rows)
    return rows


def load_dataset(config: dict, *, dataset_path_override=None) -> DatasetBundle:
    """按配置读取文本；资源路径相对项目根目录，命令行覆盖路径相对当前目录。

    source_limit 限定源记录范围，独立于 runner 的 case_ids/max_cases。
    未提供 keep_path 时返回 None，让原算法沿用案例内的保持题来源。
    """
    name = config.get("dataset_reader", "native")
    if not isinstance(name, str) or name not in READERS:
        raise ValueError(f"未知 dataset_reader: {name!r}；可选 {', '.join(READERS)}")
    limit = validate_limit(config.get("source_limit"), "source_limit")
    keep_limit = validate_limit(config.get("keep_limit"), "keep_limit")
    path = (Path(dataset_path_override).resolve() if dataset_path_override is not None
            else project_path(config["dataset_path"]))
    keep_path = project_path(config["keep_path"]) if config.get("keep_path") is not None else None
    if keep_limit is not None and keep_path is None:
        raise ValueError("keep_limit 需要同时指定 keep_path")
    raw_options = config.get("reader_options", {})
    if not isinstance(raw_options, dict):
        raise ValueError("reader_options 必须是对象")
    options = dict(raw_options)
    if options.get("subject_path") is not None:
        options["subject_path"] = str(project_path(options["subject_path"]))
    seed = config.get("seed", 2026)
    keep_pool = None
    if name == "native":
        if options:
            raise ValueError("native 读取器不接受 reader_options")
        cases = _native_cases(path, limit)
        if keep_path is not None:
            keep_pool = list(islice((p for case in load_cases(keep_path)
                                    for p in case["prompts"].get("preservation", [])), keep_limit))
    else:
        reader = READERS[name](options=options, seed=seed)
        cases = list(reader.iter_cases(path, limit=limit))
        if keep_path is not None:
            keep_pool = list(reader.iter_keep_probes(keep_path, limit=keep_limit))
    if keep_pool is not None and not keep_pool:
        raise ValueError(f"保持题来源为空: {keep_path}")
    return DatasetBundle(cases, keep_pool, {
        "reader": name, "source_path": str(path), "source_limit": limit,
        "case_count": len(cases), "reader_options": options, "seed": seed,
        "keep_path": str(keep_path) if keep_path is not None else None,
        "keep_limit": keep_limit,
        "keep_probe_count": len(keep_pool) if keep_pool is not None else None,
    })


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
