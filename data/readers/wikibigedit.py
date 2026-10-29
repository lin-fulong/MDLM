"""读取 WikiBigEdit 字段；缺失多跳题时不生成探针。"""

from .base import BaseReader, probe, required_text


class WikiBigEditReader(BaseReader):
    name = "wikibigedit"
    question_fields = {"generalization": "rephrase", "persona": "personas"}

    def convert(self, path, index, row):
        answer = required_text(row, "ans")
        case = self.make_case(path, index, row, required_text(row, "update"), answer,
                              subject=required_text(row, "subject"))
        question, target = row.get("mhop"), row.get("mhop_ans")
        missing = lambda value: value is None or (isinstance(value, str) and value.strip() in ("", "null"))
        # 多跳题可以缺失，但问题和答案必须成对存在。
        if missing(question) != missing(target):
            raise ValueError("mhop 和 mhop_ans 必须同时存在或同时缺失")
        if not missing(question):
            case["prompts"]["portability"] = [probe(case["case_id"], "portability", question, target)]
        case["metadata"].update(target_field="ans", tag=row.get("tag"))
        return case
