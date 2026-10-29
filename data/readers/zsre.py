"""适配 UltraEdit/data/zsre.py 的字段映射，返回文本案例。"""

from .base import BaseReader, probe, required_text


class ZSREReader(BaseReader):
    name = "zsre"
    allowed_options = frozenset({"target_field"})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_field = self.options.get("target_field", "ans")
        if self.target_field not in ("ans", "alt"):
            raise ValueError("zsre.target_field 只支持 ans 或 alt")

    def keep_pair(self, row):
        question, answer = super().keep_pair(row)
        return question + "?", answer

    def convert(self, path, index, row):
        answer = required_text(row, self.target_field)
        case = self.make_case(path, index, row, required_text(row, "src"), answer,
                              subject=required_text(row, "subject"))
        case["prompts"]["generalization"] = [probe(
            case["case_id"], "generalization", required_text(row, "rephrase"), answer)]
        case["metadata"].update(target_field=self.target_field, original_ans=row.get("ans"),
                                original_alt=row.get("alt"))
        return case
