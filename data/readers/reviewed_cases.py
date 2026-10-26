"""Adapt the reviewed cases.jsonl layout to the existing editing runtime interface.

This is a field adapter, not a data-quality validator. The source's gold_ready
flag selects rows for editing; all other review decisions remain with the source.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _probe(source: dict, *, answer_field: str, used_for_edit: bool, role: str) -> dict:
    return {
        "probe_id": source["probe_id"],
        "prompt": source["question"],
        "answer": source[answer_field],
        "used_for_edit": used_for_edit,
        "role": role,
    }


def adapt_reviewed_cases(rows: list[dict]) -> list[dict]:
    """Return one runtime case per approved edit request, in source order."""
    adapted = []
    for row in rows:
        if not row["gold_ready"]:
            continue
        requests, questions = row["edit_requests"], row["edit_questions"]
        if len(requests) != len(questions):
            raise ValueError(f"{row['case_id']}: edit_requests and edit_questions cannot be paired by index")
        for index, (request, question) in enumerate(zip(requests, questions)):
            # The source has no request_id on edit_questions; their shared index
            # is the only explicit pairing available. Keep the original IDs.
            case_id = row["case_id"] if len(requests) == 1 else request["request_id"]
            prompts = {"edit": [_probe(question, answer_field="expected_new_answer",
                                        used_for_edit=True, role="edit")],
                       "free_form_qa": [], "portability": [], "preservation": []}
            if index == 0:
                # Chain-level probes are attached once, never duplicated across
                # the component facts of a multi-request MQuAKE case.
                for source in row["independent_test_questions"]:
                    group = "portability" if ":portability:" in source["probe_id"] else "free_form_qa"
                    prompts[group].append(_probe(source, answer_field="expected_new_answer",
                                                 used_for_edit=False, role="independent_test"))
                prompts["preservation"] = [
                    _probe(source, answer_field="expected_unchanged_answer",
                           used_for_edit=False, role="preservation")
                    for source in row["preservation_questions"]
                ]
            adapted.append({
                "case_id": case_id,
                "source_case_id": row["case_id"],
                "source_request_id": request["request_id"],
                "dataset_group": row["dataset_group"],
                "fact_core": {
                    "subject": request["subject"],
                    "relation": request["relation"],
                    "old_object": request["old_answer"],
                    "new_object": request["experimental_new_answer"],
                },
                "prompts": prompts,
                "quality_control": {"status": row["review_status"]},
            })
    return adapted


def main() -> None:
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
