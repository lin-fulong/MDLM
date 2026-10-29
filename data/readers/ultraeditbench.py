"""读取 UltraEditBench 字段，并保留原始主体和编号。"""

from .base import BaseReader, required_text


class UltraEditBenchReader(BaseReader):
    name = "ultraeditbench"
    question_fields = {"generalization": "rephrase_prompt"}

    def convert(self, path, index, row):
        answer = required_text(row, "ans")
        case = self.make_case(path, index, row, required_text(row, "prompt"), answer,
                              subject=required_text(row, "subject"))
        case["metadata"]["target_field"] = "ans"
        return case
