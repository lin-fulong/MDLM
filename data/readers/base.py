"""原始记录迭代、文本探针和外部数据读取器的共用边界。"""

from __future__ import annotations

from contextlib import closing
from itertools import islice
import json
from pathlib import Path


def validate_limit(value, name="limit"):
    assert value is None or (type(value) is int and value >= 1), f"{name} 必须为正整数或 null"
    return value


def iter_records(path, *, limit=None):
    """按原顺序读取对象、数组或 JSONL；限量按源记录计，解析错误直接向上传递。"""
    path = Path(path)
    validate_limit(limit)
    jsonl = path.suffix.lower() == ".jsonl"
    with path.open("r" if jsonl else "rb", encoding="utf-8-sig" if jsonl else None) as stream:
        if jsonl:
            records = (json.loads(line) for line in stream if line.strip())
        else:
            # 仅外部 JSON 数组/对象需要 ijson；不包装依赖库和解析器的原始异常。
            import ijson

            start = 3 if stream.read(3) == b"\xef\xbb\xbf" else 0
            stream.seek(start)
            first = stream.read(1)
            while first and first in b" \r\n\t":
                first = stream.read(1)
            assert first in (b"[", b"{"), f"{path}: 顶层必须是 JSON 数组或对象"
            stream.seek(start)
            records = ijson.items(stream, "item" if first == b"[" else "", use_float=True)
        for index, row in enumerate(islice(records, limit)):
            assert isinstance(row, dict), f"{path}: 源记录 {index} 必须是对象"
            yield index, row


def required_text(row, field):
    value = row.get(field)
    assert isinstance(value, str) and value.strip(), f"{field} 必须为非空字符串"
    return value


def probe(case_id, role, prompt, answer, index=0):
    # 不 strip、不添加空格；分词和模板由下游模型输入构造负责。
    assert all(isinstance(value, str) and value.strip() for value in (prompt, answer)), "prompt 和 answer 必须为非空字符串"
    return {"probe_id": f"{case_id}:{role}:{index}", "prompt": prompt, "answer": answer,
            "role": role, "used_for_edit": role == "edit"}


class BaseReader:
    name = ""
    allowed_options = frozenset()
    keep_fields = ("loc", "loc_ans")
    # 各数据集只声明固定字段映射；这些附加问题不进入 edit 监督。
    question_fields = {}

    def __init__(self, *, options=None, seed=2026):
        assert options is None or isinstance(options, dict), "reader_options 必须是对象"
        self.options = dict(options or {})
        unknown = self.options.keys() - self.allowed_options
        assert not unknown, f"{self.name}: 未知读取参数 {sorted(unknown)}"
        assert type(seed) is int and seed >= 0, "seed 必须为非负整数"
        self.seed = seed

    def identity(self, path, index, row):
        original = row.get("case_id", index)
        assert (
            not isinstance(original, bool) and isinstance(original, (int, str)) and str(original).strip()
        ), "case_id 必须为非空字符串或整数"
        return f"{self.name}:{Path(path).stem}:{original}"

    def keep_pair(self, row):
        return tuple(required_text(row, field) for field in self.keep_fields)

    def make_case(self, path, index, row, prompt, answer, *, subject=None, old_object=None):
        case_id = self.identity(path, index, row)
        prompts = {key: [] for key in (
            "edit", "generalization", "preservation", "persona", "portability", "free_form_qa")}
        prompts["edit"] = [probe(case_id, "edit", prompt, answer)]
        prompts["preservation"] = [probe(case_id, "preservation", *self.keep_pair(row))]
        for role, field in self.question_fields.items():
            prompts[role] = [probe(case_id, role, required_text(row, field), answer)]
        return {"case_id": case_id, "source_case_id": case_id, "dataset_group": self.name,
                "fact_core": {"subject": subject, "relation": row.get("relation"),
                              "old_object": old_object, "new_object": answer},
                "prompts": prompts, "quality_control": {"status": "imported"},
                "metadata": {"source_file": str(Path(path)), "source_index": index,
                             "original_case_id": row.get("case_id"),
                             "subject_missing": subject is None}}

    def iter_cases(self, path, *, limit=None):
        seen = set()
        # 转换错误直接抛出，closing 仍保证源文件及时释放。
        with closing(iter_records(path, limit=limit)) as records:
            for index, row in records:
                case = self.convert(path, index, row)
                case_id = case["case_id"]
                assert case_id not in seen, f"duplicate case_id: {case_id}"
                seen.add(case_id)
                yield case

    def iter_keep_probes(self, path, *, limit=None):
        """只读取无关问题，不要求编辑主体、改写或多跳字段存在。"""
        with closing(iter_records(path, limit=limit)) as records:
            for index, row in records:
                yield probe(self.identity(path, index, row), "preservation", *self.keep_pair(row))


def validate_edit_subjects(cases):
    """对 runner 已选中的外部案例作字符级预检；精确 token 对齐仍在 states 中。"""
    for case in cases:
        subject = case["fact_core"].get("subject")
        assert isinstance(subject, str) and subject.strip(), (
            f"{case['case_id']}: 缺少编辑主体；FEVER 可通过 reader_options.subject_path 补充")
        prompt = case["prompts"]["edit"][0]["prompt"]
        begin = prompt.find(subject)
        assert begin >= 0 and prompt.find(subject, begin + 1) < 0, (
            f"{case['case_id']}: subject 必须在编辑问题中恰好出现一次")
