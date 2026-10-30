"""读取 FEVER 字段；主体标注和改写选择显式、可复现。"""

import hashlib
import json
from pathlib import Path

from .base import BaseReader, probe, required_text


class FEVERReader(BaseReader):
    name = "fever"
    keep_fields = ("unrel_prompt", "unrel_ans")
    allowed_options = frozenset({"rephrase_policy", "subject_path"})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.policy = self.options.get("rephrase_policy", "single_fixed")
        assert self.policy in ("single_fixed", "all"), "fever.rephrase_policy 只支持 single_fixed 或 all"
        self.subjects = {}
        if self.options.get("subject_path") is not None:
            with Path(self.options["subject_path"]).open(encoding="utf-8-sig") as stream:
                self.subjects = json.load(stream)
            assert isinstance(self.subjects, dict) and all(
                isinstance(value, str) and value.strip() for value in self.subjects.values()
            ), "subject_path 必须指向 {完整案例ID: 主体字符串} 的 JSON 对象"

    def convert(self, path, index, row):
        case_id = self.identity(path, index, row)
        subject = row.get("subject")
        annotated = self.subjects.get(case_id)
        if subject is not None:
            required_text(row, "subject")
            assert annotated is None or annotated == subject, "主体标注与原始 subject 冲突"
        subject = subject if subject is not None else annotated
        answer = required_text(row, "alt")
        case = self.make_case(path, index, row, required_text(row, "prompt"), answer,
                              subject=subject, old_object=required_text(row, "ans"))
        variants = row.get("equiv_prompt")
        assert isinstance(variants, list) and variants and all(
            isinstance(value, str) and value.strip() for value in variants
        ), "equiv_prompt 必须是非空字符串列表"
        indices = range(len(variants))
        if self.policy == "single_fixed":
            # 案例独立选题；读取顺序和全局随机状态不影响选中的改写。
            digest = hashlib.sha256(f"{self.seed}:{case_id}:rephrase".encode("utf-8")).digest()
            indices = [int.from_bytes(digest[:8], "big") % len(variants)]
        case["prompts"]["generalization"] = [
            probe(case_id, "generalization", variants[i], answer, i) for i in indices]
        case["metadata"].update(target_field="alt", rephrase_policy=self.policy,
                                rephrase_indices=list(indices))
        return case
