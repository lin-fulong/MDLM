"""读取 ZSRE 字段并转换为统一文本案例。"""

from .base import BaseReader


class ZSREReader(BaseReader):
    name = "zsre"
    allowed_options = frozenset({"target_field"})
    question_fields = {"generalization": "rephrase"}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_field = self.options.get("target_field", "ans")

    def keep_pair(self, row):
        question, answer = super().keep_pair(row)
        return question + "?", answer

    def convert(self, path, index, row):
        answer = row[self.target_field]
        case = self.make_case(path, index, row, row["src"], answer, subject=row["subject"])
        case["metadata"].update(target_field=self.target_field, original_ans=row.get("ans"),
                                original_alt=row.get("alt"))
        return case
