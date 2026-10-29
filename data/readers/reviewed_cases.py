"""将 gold_ready 源案例按请求展开为算法事实；沿用源数据的审核结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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
    parser = argparse.ArgumentParser(description="Materialize gold_ready editing cases for P0-P5.")
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
