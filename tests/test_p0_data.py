"""P0 contract checks with an artificial example, never an experiment result."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from data.readers.json_cases import case_purpose, evaluation_inputs, load_cases
from data.schema import CaseValidationError, check_dataset, validate_case
from data.validation.alignment import AlignmentError, align_edit_prompt
from model.registry import get_adapter


FIXTURE = Path(__file__).parent / "fixtures" / "p0_case_example.json"


class CharacterTokenizer:
    """Tiny stand-in for contract tests; not evidence about real LLaDA tokenization."""

    all_special_ids: tuple[int, ...] = ()

    def apply_chat_template(self, messages, **_kwargs):
        return "用户：" + messages[0]["content"] + "\n助手："

    def __call__(self, text, **kwargs):
        output = {"input_ids": [ord(char) for char in text]}
        if kwargs.get("return_offsets_mapping"):
            output["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return output

    def decode(self, ids, **_kwargs):
        return "".join(chr(token) for token in ids)


class P0DataTests(unittest.TestCase):
    def setUp(self):
        self.case = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_read_and_case_roles(self):
        self.assertEqual(load_cases(FIXTURE), [self.case])
        self.assertEqual(case_purpose(self.case), "example_only")
        self.assertEqual(evaluation_inputs(self.case)[0]["role"], "development_validation")
        self.assertEqual(check_dataset(FIXTURE), 1)

    def test_reader_skips_quality_checks_and_manual_check_rejects_duplicates(self):
        second = copy.deepcopy(self.case)
        second["case_id"] = "constructed_birth_city_0002"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text(
                "\n".join(json.dumps(case, ensure_ascii=False) for case in (self.case, second)) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(len(load_cases(path)), 2)
            path.write_text(json.dumps(self.case, ensure_ascii=False) + "\n" + json.dumps(self.case, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(len(load_cases(path)), 2)
            with self.assertRaisesRegex(CaseValidationError, "duplicate case_id"):
                check_dataset(path)
            invalid = copy.deepcopy(self.case)
            del invalid["fact_core"]["subject"]
            path.write_text(json.dumps(invalid, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(load_cases(path), [invalid])
            with self.assertRaisesRegex(CaseValidationError, "subject"):
                check_dataset(path)

    def test_required_types_and_usage(self):
        bad = copy.deepcopy(self.case)
        del bad["fact_core"]["subject"]
        with self.assertRaisesRegex(CaseValidationError, "subject"):
            validate_case(bad)
        bad = copy.deepcopy(self.case)
        bad["prompts"]["edit"][0]["used_for_edit"] = "true"
        with self.assertRaisesRegex(CaseValidationError, "used_for_edit"):
            validate_case(bad)
        bad = copy.deepcopy(self.case)
        bad["prompts"]["paraphrase"][0]["used_for_edit"] = True
        validate_case(bad)
        self.assertEqual(evaluation_inputs(bad), [])

    def test_old_answer_alias_and_prompt_target_are_rejected(self):
        bad = copy.deepcopy(self.case)
        bad["fact_core"]["new_object_aliases"].append("青榆市")
        with self.assertRaisesRegex(CaseValidationError, "old_object conflicts"):
            validate_case(bad)
        bad = copy.deepcopy(self.case)
        bad["prompts"]["edit"][0]["prompt"] += "海岚市"
        with self.assertRaisesRegex(CaseValidationError, "prompt contains target"):
            validate_case(bad)
        bad = copy.deepcopy(self.case)
        bad["prompts"]["paraphrase"][0]["prompt"] += "海岚市"
        self.assertEqual(evaluation_inputs(bad)[0]["prompt"], bad["prompts"]["paraphrase"][0]["prompt"])

    def test_edit_target_separate_from_model_input(self):
        adapter = get_adapter("LLaDA-8B-Instruct", tokenizer=CharacterTokenizer(), mask_token_id=126336, target_layers=(0,))
        model_input, target = adapter.prepare_edit_input(self.case, "birth_edit")
        self.assertEqual(len(model_input.mask_positions), len(target.token_ids))
        self.assertTrue(all(model_input.input_ids[i] == 126336 for i in model_input.mask_positions))
        self.assertEqual(target.token_ids, tuple(map(ord, "海岚市")))
        self.assertEqual(model_input.input_ids[model_input.subject_last_position], ord("舟"))
        self.assertEqual(adapter.target_module_names(), ("model.transformer.blocks.0.ff_out",))
        self.assertNotIn("answer", evaluation_inputs(self.case)[0])
        self.assertNotIn("target_proposition", evaluation_inputs(self.case)[0])
        self.assertTrue(evaluation_inputs(self.case)[0]["example_only"])

    def test_alignment_rejects_ambiguous_subject(self):
        for prompt, subject in (("林舟问林舟", "林舟"), ("哈哈哈", "哈哈")):
            with self.subTest(subject=subject), self.assertRaises(AlignmentError):
                align_edit_prompt(CharacterTokenizer(), prompt, subject, "海岚市")

    def test_alignment_accepts_overlapping_byte_token_offsets(self):
        class ByteSplitTokenizer(CharacterTokenizer):
            def __call__(self, text, **kwargs):
                output = super().__call__(text, **kwargs)
                output["input_ids"][1:2] = [9001, 9002]
                if "offset_mapping" in output:
                    output["offset_mapping"][1:2] = [(1, 2), (1, 2)]
                return output

        aligned = align_edit_prompt(ByteSplitTokenizer(), "林舟问：", "林舟", "海岚市")
        self.assertEqual(aligned.subject_positions, (0, 1, 2))
        self.assertEqual(aligned.subject_last_position, 2)

    def test_alignment_accepts_subject_token_with_leading_space(self):
        class SpaceMergedTokenizer(CharacterTokenizer):
            def __call__(self, text, **kwargs):
                output = {"input_ids": [9001, *map(ord, text[6:])]}
                if kwargs.get("return_offsets_mapping"):
                    output["offset_mapping"] = [(0, 6), *[(i, i + 1) for i in range(6, len(text))]]
                return output

        aligned = align_edit_prompt(SpaceMergedTokenizer(), " Alice?", "Alice", "Paris")
        self.assertEqual(aligned.subject_positions, (0,))
        self.assertEqual(aligned.answer_ids, tuple(map(ord, "Paris")))

    def test_alignment_rejects_gap_in_subject_offsets(self):
        class GappedTokenizer(CharacterTokenizer):
            def __call__(self, text, **kwargs):
                output = super().__call__(text, **kwargs)
                if "offset_mapping" in output:
                    output["offset_mapping"][1] = (2, 2)
                return output

        with self.assertRaisesRegex(AlignmentError, "subject"):
            align_edit_prompt(GappedTokenizer(), "林舟问：", "林舟", "海岚市")

    def test_alignment_rejects_special_target_and_prompt_mask(self):
        tokenizer = CharacterTokenizer()
        tokenizer.all_special_ids = (ord("海"),)
        with self.assertRaisesRegex(AlignmentError, "special tokens"):
            align_edit_prompt(tokenizer, "林舟问：", "林舟", "海岚市")

        class PromptMaskTokenizer(CharacterTokenizer):
            def __call__(self, text, **kwargs):
                output = super().__call__(text, **kwargs)
                output["input_ids"] = [126336 if token == ord("？") else token for token in output["input_ids"]]
                return output

        adapter = get_adapter("LLaDA-8B-Instruct", tokenizer=PromptMaskTokenizer(), mask_token_id=126336, target_layers=(0,))
        with self.assertRaisesRegex(ValueError, "already contains"):
            adapter.prepare_edit_input(self.case, "birth_edit")

    def test_independent_test_probe_role(self):
        other = copy.deepcopy(self.case)
        other["split"] = "test"
        self.assertEqual(evaluation_inputs(other)[0]["role"], "independent_test")


if __name__ == "__main__":
    unittest.main()
