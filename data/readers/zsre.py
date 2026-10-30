"""读取 ZSRE 字段并转换为统一文本案例。"""

from .base import BaseReader, required_text


class ZSREReader(BaseReader):
    name = "zsre"
    allowed_options = frozenset({"target_field"})
    question_fields = {"generalization": "rephrase"}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_field = self.options.get("target_field", "ans")
        assert self.target_field in ("ans", "alt"), "zsre.target_field 只支持 ans 或 alt"

    def keep_pair(self, row):
        question, answer = super().keep_pair(row)
        return question + "?", answer

    def convert(self, path, index, row):
        answer = required_text(row, self.target_field)
        case = self.make_case(path, index, row, required_text(row, "src"), answer,
                              subject=required_text(row, "subject"))
        case["metadata"].update(target_field=self.target_field, original_ans=row.get("ans"),
                                original_alt=row.get("alt"))
        return case
