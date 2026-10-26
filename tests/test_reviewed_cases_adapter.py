"""Interface regression for the reviewed dataset's editing layout."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from data.readers.json_cases import case_purpose, evaluation_inputs, load_cases
from data.readers.reviewed_cases import adapt_reviewed_cases
from model.adapters.llada_8b_instruct import LLaDA8BInstructAdapter
from tests.test_p0_data import CharacterTokenizer


def row(case_id: str, count: int = 1) -> dict:
    requests = []
    edits = []
    for index in range(count):
        subject = ("Alice", "Bob")[index]
        requests.append({"request_id": f"{case_id}:rewrite:{index}", "subject": subject,
                         "relation": "related_to", "old_answer": "Old",
                         "experimental_new_answer": "New"})
        edits.append({"probe_id": f"{case_id}:edit:{index}", "question": f"{subject} is related to ",
                      "expected_new_answer": "New"})
    return {"case_id": case_id, "dataset_group": "MQuAKE", "gold_ready": True,
            "review_status": "human_review_passed_gold_ready", "edit_requests": requests,
            "edit_questions": edits,
            "independent_test_questions": [{"probe_id": f"{case_id}:portability:0",
                                            "question": "Where does the chain lead?",
                                            "expected_new_answer": "New"}],
            "preservation_questions": [{"probe_id": f"{case_id}:same_subject:0",
                                        "question": "What stays the same?",
                                        "expected_unchanged_answer": "Old"}]}


class ReviewedCasesAdapterTests(unittest.TestCase):
    def test_reader_selects_gold_ready_and_preserves_single_case_id(self):
        approved = row("one")
        pending = row("pending")
        pending["gold_ready"] = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text("\n".join(json.dumps(x) for x in (approved, pending)) + "\n", encoding="utf-8")
            cases = load_cases(path)
        self.assertEqual([case["case_id"] for case in cases], ["one"])
        self.assertEqual(case_purpose(cases[0]), "edit")
        self.assertEqual(cases[0]["prompts"]["edit"][0]["probe_id"], "one:edit:0")
        self.assertEqual([x["role"] for x in evaluation_inputs(cases[0])],
                         ["independent_test", "preservation"])
        self.assertTrue(all("answer" not in x for x in evaluation_inputs(cases[0])))

    def test_multi_request_expands_facts_without_duplicating_chain_probes(self):
        cases = adapt_reviewed_cases([row("chain", 2)])
        self.assertEqual([case["case_id"] for case in cases],
                         ["chain:rewrite:0", "chain:rewrite:1"])
        self.assertEqual([case["fact_core"]["subject"] for case in cases], ["Alice", "Bob"])
        self.assertEqual([len(evaluation_inputs(case)) for case in cases], [2, 0])
        adapter = LLaDA8BInstructAdapter(CharacterTokenizer(), mask_token_id=126336, target_layers=(0,))
        for case in cases:
            masked, target = adapter.prepare_edit_input(case, case["prompts"]["edit"][0]["probe_id"])
            self.assertEqual(len(masked.mask_positions), len(target.token_ids))
            self.assertEqual(target.case_id, case["case_id"])

    def test_unpaired_requests_fail_at_interface_boundary(self):
        source = row("chain", 2)
        source["edit_questions"].pop()
        with self.assertRaisesRegex(ValueError, "cannot be paired"):
            adapt_reviewed_cases([source])


if __name__ == "__main__":
    unittest.main()
