"""适配 UltraEdit/data/ultraeditbench.py，并保留原始主体和编号。"""

from .base import BaseReader, probe, required_text


class UltraEditBenchReader(BaseReader):
    name = "ultraeditbench"

    def convert(self, path, index, row):
        answer = required_text(row, "ans")
        case = self.make_case(path, index, row, required_text(row, "prompt"), answer,
                              subject=required_text(row, "subject"))
        case["prompts"]["generalization"] = [probe(
            case["case_id"], "generalization", required_text(row, "rephrase_prompt"), answer)]
        case["metadata"]["target_field"] = "ans"
        return case
