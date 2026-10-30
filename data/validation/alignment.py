"""对齐主体和短答案 token，为公式（1）～（4）及开发编辑提供输入位置。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TokenAlignment:
    """模板化提示、覆盖主体的连续 token 位置，以及单独保存的监督答案。"""

    prompt_ids: tuple[int, ...]
    subject_positions: tuple[int, ...]
    answer_ids: tuple[int, ...]

    @property
    def subject_last_position(self) -> int:
        """公式（4）的取键位置：覆盖主体的最后一个 token，下标从零开始。"""
        return self.subject_positions[-1]


def align_edit_prompt(tokenizer: Any, rendered_prompt: str, subject: str, answer: str) -> TokenAlignment:
    """按源数据给定的主体和稳定答案边界，构造监督 token 的位置。"""
    start = rendered_prompt.index(subject)
    end = start + len(subject)
    encoded = tokenizer(rendered_prompt, add_special_tokens=False, return_offsets_mapping=True)
    prompt_ids = tuple(encoded["input_ids"])
    offsets = encoded["offset_mapping"]
    full_ids = tuple(tokenizer(rendered_prompt + answer, add_special_tokens=False)["input_ids"])
    answer_begin = len(prompt_ids)
    # 公式（4）：按字符区间相交找主体 token，不单独分词主体，以保留上下文边界。
    positions = tuple(i for i, (left, right) in enumerate(offsets) if right > left and right > start and left < end)
    # 公式（1）～（3）：答案独立为监督；适配器随后从 answer_begin 建立等长全 mask 窗口。
    answer_ids = full_ids[answer_begin:]
    return TokenAlignment(prompt_ids, positions, answer_ids)
