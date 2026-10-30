"""读取 FEVER 字段；主体标注和改写选择显式、可复现。"""

import hashlib
import json
import sys
from pathlib import Path

from .base import BaseReader, probe


class FEVERReader(BaseReader):
    name = "fever"
    allowed_options = frozenset({"rephrase_policy", "subject_path"})
    keep_fields = ("unrel_prompt", "unrel_ans")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.policy = self.options.get("rephrase_policy", "single_fixed")
        if self.policy not in ("single_fixed", "all"):
            sys.exit("FEVER rephrase_policy 必须为 single_fixed 或 all")
        self.subjects = {}
        if self.options.get("subject_path") is not None:
            with Path(self.options["subject_path"]).open(encoding="utf-8-sig") as stream:
                self.subjects = json.load(stream)

    def convert(self, path, index, row):
        case_id = self.identity(path, index, row)
        subject = row.get("subject")
        annotated = self.subjects.get(case_id)
        if subject is not None and annotated is not None and subject != annotated:
            sys.exit(f"{case_id}: 主体标注与原始 subject 冲突")
        subject = subject if subject is not None else annotated
        answer = row["alt"]
        case = self.make_case(path, index, row, row["prompt"], answer,
                              subject=subject, old_object=row["ans"])
        variants = row["equiv_prompt"]
        if not isinstance(variants, list) or not variants or any(
            not isinstance(text, str) or not text.strip() for text in variants
        ):
            sys.exit(f"{case_id}: equiv_prompt 必须为非空问题列表")
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
