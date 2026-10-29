"""对齐主体和短答案 token，为公式（1）～（4）及开发编辑提供输入位置。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class AlignmentError(ValueError):
    """主体位置或短答案边界无法按当前 tokenizer 确定。"""


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
    if not all(isinstance(value, str) and value for value in (rendered_prompt, subject, answer)):
        raise AlignmentError("rendered prompt, subject and answer must be nonempty strings")
    start = rendered_prompt.find(subject)
    if start < 0 or rendered_prompt.find(subject, start + 1) != -1:
        raise AlignmentError("subject must occur exactly once in the rendered edit prompt")
    end = start + len(subject)
    try:
        encoded = tokenizer(rendered_prompt, add_special_tokens=False, return_offsets_mapping=True)
        prompt_ids = tuple(encoded["input_ids"])
        offsets = encoded["offset_mapping"]
        full_ids = tuple(tokenizer(rendered_prompt + answer, add_special_tokens=False)["input_ids"])
    except (KeyError, TypeError, ValueError, NotImplementedError) as exc:
        raise AlignmentError("tokenizer must provide exact offset_mapping for rendered prompt") from exc
    answer_begin = len(prompt_ids)
    if len(offsets) != answer_begin:
        raise AlignmentError("offset count differs from prompt token count")
    # 公式（4）：按字符区间相交找主体 token，不单独分词主体，以保留上下文边界。
    positions = tuple(i for i, (left, right) in enumerate(offsets) if right > left and right > start and left < end)
    cursor = start
    previous_left = -1
    for position in positions:
        left, right = offsets[position]
        # 字节级 token 可共享字符偏移，也可包含主体边缘的空格；核对覆盖范围即可。
        if left > cursor or left < previous_left or left < 0 or right > len(rendered_prompt):
            raise AlignmentError("subject token offsets have a gap or invalid order")
        cursor = max(cursor, min(right, end))
        previous_left = left
    # positions 按 token 下标递增；数量等于首尾跨度即连续，无需另建 range 元组。
    if not positions or cursor != end or len(positions) != positions[-1] - positions[0] + 1:
        raise AlignmentError("subject is not fully covered by contiguous tokens")
    # 公式（1）～（3）：答案独立为监督；适配器随后从 answer_begin 建立等长全 mask 窗口。
    if full_ids[:answer_begin] != prompt_ids:
        raise AlignmentError("answer changes prompt tokenization; no stable mask boundary")
    answer_ids = full_ids[answer_begin:]
    if not answer_ids:
        raise AlignmentError("answer tokenization is empty")
    if set(answer_ids) & set(tokenizer.all_special_ids):
        raise AlignmentError("answer contains tokenizer special tokens")
    decoded_answer = tokenizer.decode(answer_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if decoded_answer != answer:
        raise AlignmentError(f"answer token IDs decode to {decoded_answer!r}, expected {answer!r}")
    return TokenAlignment(prompt_ids, positions, answer_ids)
