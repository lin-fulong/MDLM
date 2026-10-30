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
    """定位唯一主体，保持提示分词前缀不变，并取得可精确解码的短答案 token。"""
    assert all(isinstance(value, str) and value for value in (rendered_prompt, subject, answer)), (
        "rendered prompt, subject and answer must be nonempty strings")
    start = rendered_prompt.find(subject)
    assert start >= 0 and rendered_prompt.find(subject, start + 1) == -1, (
        "subject must occur exactly once in the rendered edit prompt")
    end = start + len(subject)
    encoded = tokenizer(rendered_prompt, add_special_tokens=False, return_offsets_mapping=True)
    prompt_ids = tuple(encoded["input_ids"])
    offsets = encoded["offset_mapping"]
    full_ids = tuple(tokenizer(rendered_prompt + answer, add_special_tokens=False)["input_ids"])
    answer_begin = len(prompt_ids)
    assert len(offsets) == answer_begin, "offset count differs from prompt token count"
    # 公式（4）：按字符区间相交找主体 token，不单独分词主体，以保留上下文边界。
    positions = tuple(i for i, (left, right) in enumerate(offsets) if right > left and right > start and left < end)
    cursor = start
    previous_left = -1
    for position in positions:
        left, right = offsets[position]
        # 字节级 token 可共享字符偏移，也可包含主体边缘的空格；核对覆盖范围即可。
        assert left <= cursor and left >= previous_left and left >= 0 and right <= len(rendered_prompt), (
            "subject token offsets have a gap or invalid order")
        cursor = max(cursor, min(right, end))
        previous_left = left
    # positions 按 token 下标递增；数量等于首尾跨度即连续，无需另建 range 元组。
    assert positions and cursor == end and len(positions) == positions[-1] - positions[0] + 1, (
        "subject is not fully covered by contiguous tokens")
    # 公式（1）～（3）：答案独立为监督；适配器随后从 answer_begin 建立等长全 mask 窗口。
    assert full_ids[:answer_begin] == prompt_ids, "answer changes prompt tokenization; no stable mask boundary"
    answer_ids = full_ids[answer_begin:]
    assert answer_ids, "answer tokenization is empty"
    assert not set(answer_ids) & set(tokenizer.all_special_ids), "answer contains tokenizer special tokens"
    decoded_answer = tokenizer.decode(answer_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    assert decoded_answer == answer, f"answer token IDs decode to {decoded_answer!r}, expected {answer!r}"
    return TokenAlignment(prompt_ids, positions, answer_ids)
