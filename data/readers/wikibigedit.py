"""读取 WikiBigEdit 字段；缺失多跳题时不生成探针。"""

from .base import BaseReader, probe


class WikiBigEditReader(BaseReader):
    name = "wikibigedit"
    question_fields = {"generalization": "rephrase", "persona": "personas"}

    def convert(self, path, index, row):
        answer = row["ans"]
        case = self.make_case(path, index, row, row["update"], answer, subject=row["subject"])
        question, target = row.get("mhop"), row.get("mhop_ans")
        question_missing = question is None or (isinstance(question, str) and question.strip() in ("", "null"))
        # 源格式用空值或 "null" 表示没有多跳题。
        if not question_missing:
            case["prompts"]["portability"] = [probe(case["case_id"], "portability", question, target)]
        case["metadata"].update(target_field="ans", tag=row.get("tag"))
        return case
