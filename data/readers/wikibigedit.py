"""适配 UltraEdit/data/wikibigedit.py，缺失多跳题不生成探针。"""

from .base import BaseReader, probe, required_text


class WikiBigEditReader(BaseReader):
    name = "wikibigedit"

    def convert(self, path, index, row):
        answer = required_text(row, "ans")
        case = self.make_case(path, index, row, required_text(row, "update"), answer,
                              subject=required_text(row, "subject"))
        for role, field in (("generalization", "rephrase"), ("persona", "personas")):
            case["prompts"][role] = [probe(case["case_id"], role, required_text(row, field), answer)]
        question, target = row.get("mhop"), row.get("mhop_ans")
        missing = lambda value: value is None or (isinstance(value, str) and value.strip() in ("", "null"))
        if missing(question) != missing(target):
            raise ValueError("mhop 和 mhop_ans 必须同时存在或同时缺失")
        if not missing(question):
            case["prompts"]["portability"] = [probe(case["case_id"], "portability", question, target)]
        case["metadata"].update(target_field="ans", tag=row.get("tag"))
        return case
