"""原始记录迭代、文本探针和外部数据读取器的共用边界。"""

from __future__ import annotations

from contextlib import closing
from itertools import islice
import json
import sys
from pathlib import Path


def validate_limit(limit):
    if limit is not None and (type(limit) is not int or limit < 1):
        sys.exit("读取上限必须为正整数或 null")


def iter_records(path, *, limit=None):
    """按原顺序读取对象、数组或 JSONL；限量按源记录计，解析错误直接向上传递。"""
    validate_limit(limit)
    path = Path(path)
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
            stream.seek(start)
            records = ijson.items(stream, "item" if first == b"[" else "", use_float=True)
        for index, row in enumerate(islice(records, limit)):
            yield index, row


def probe(case_id, role, prompt, answer, index=0):
    # 不 strip、不添加空格；分词和模板由下游模型输入构造负责。
    if any(not isinstance(text, str) or not text.strip() for text in (prompt, answer)):
        sys.exit(f"{case_id}: {role} 的问题和答案必须为非空文本")
    return {"probe_id": f"{case_id}:{role}:{index}", "prompt": prompt, "answer": answer,
            "role": role, "used_for_edit": role == "edit"}


class BaseReader:
    name = ""
    allowed_options = frozenset()
    keep_fields = ("loc", "loc_ans")
    # 各数据集只声明固定字段映射；这些附加问题不进入 edit 监督。
    question_fields = {}

    def __init__(self, *, options=None, seed=2026):
        self.options = dict(options or {})
        if self.options.keys() - self.allowed_options:
            sys.exit(f"{self.name}: 未知读取参数 {sorted(self.options.keys() - self.allowed_options)}")
        self.seed = seed

    def identity(self, path, index, row):
        original = row.get("case_id", index)
        if isinstance(original, bool) or not isinstance(original, (str, int)) or not str(original).strip():
            sys.exit("案例 ID 必须为非空字符串或整数")
        return f"{self.name}:{Path(path).stem}:{original}"

    def keep_pair(self, row):
        return tuple(row[field] for field in self.keep_fields)

    def make_case(self, path, index, row, prompt, answer, *, subject=None, old_object=None):
        case_id = self.identity(path, index, row)
        prompts = {key: [] for key in (
            "edit", "generalization", "preservation", "persona", "portability", "free_form_qa")}
        prompts["edit"] = [probe(case_id, "edit", prompt, answer)]
        prompts["preservation"] = [probe(case_id, "preservation", *self.keep_pair(row))]
        for role, field in self.question_fields.items():
            prompts[role] = [probe(case_id, role, row[field], answer)]
        return {"case_id": case_id, "source_case_id": case_id, "dataset_group": self.name,
                "fact_core": {"subject": subject, "relation": row.get("relation"),
                              "old_object": old_object, "new_object": answer},
                "prompts": prompts, "quality_control": {"status": "imported"},
                "metadata": {"source_file": str(Path(path)), "source_index": index,
                             "original_case_id": row.get("case_id"),
                             "subject_missing": subject is None}}

    def iter_cases(self, path, *, limit=None):
        # 转换错误直接抛出，closing 仍保证源文件及时释放。
        seen = set()
        with closing(iter_records(path, limit=limit)) as records:
            for index, row in records:
                case = self.convert(path, index, row)
                if case["case_id"] in seen:
                    sys.exit(f"重复案例 ID: {case['case_id']}")
                seen.add(case["case_id"])
                yield case

    def iter_keep_probes(self, path, *, limit=None):
        """只读取无关问题，不要求编辑主体、改写或多跳字段存在。"""
        with closing(iter_records(path, limit=limit)) as records:
            for index, row in records:
                yield probe(self.identity(path, index, row), "preservation", *self.keep_pair(row))
