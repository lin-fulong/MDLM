"""对齐主体和短答案 token，为公式（1）～（4）及开发编辑提供输入位置。"""

from __future__ import annotations

from dataclasses import dataclass
import sys
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


def encode_answer(tokenizer, rendered_prompt, answer, mask_token_id=None):
    """编辑与评价共用稳定边界：p=Tok(prompt)，y=Tok(prompt+answer)[|p|:]。"""
    if not isinstance(answer, str) or not answer.strip():
        sys.exit("标准答案必须为非空文本")
    prompt = tuple(tokenizer.encode(rendered_prompt, add_special_tokens=False))
    full = tuple(tokenizer.encode(rendered_prompt + answer, add_special_tokens=False))
    target = full[len(prompt):]
    if not prompt or full[:len(prompt)] != prompt or not target:
        sys.exit("答案分词边界不稳定，无法建立完整的 mask 窗口")
    if mask_token_id in prompt or set(target) & set(tokenizer.all_special_ids):
        sys.exit("提示含预置 mask 或答案含特殊 token")
    if tokenizer.decode(target, skip_special_tokens=False, clean_up_tokenization_spaces=False) != answer:
        sys.exit("答案 token 无法精确解码为标准答案")
    return prompt, target


def align_edit_prompt(tokenizer: Any, rendered_prompt: str, subject: str, answer: str) -> TokenAlignment:
    """按源数据给定的主体和稳定答案边界，构造监督 token 的位置。"""
    if not isinstance(subject, str) or not subject.strip():
        sys.exit("缺少有效编辑主体，请在案例 fact_core.subject 中提供")
    start = rendered_prompt.find(subject)
    if start < 0 or rendered_prompt.find(subject, start + 1) >= 0:
        sys.exit("编辑主体必须在模板化问题中唯一出现，不能猜测主体位置")
    end = start + len(subject)
    encoded = tokenizer(rendered_prompt, add_special_tokens=False, return_offsets_mapping=True)
    prompt_ids, answer_ids = encode_answer(tokenizer, rendered_prompt, answer)
    offsets = encoded["offset_mapping"]
    if tuple(encoded["input_ids"]) != prompt_ids or len(offsets) != len(prompt_ids):
        sys.exit("tokenizer 的 token 与字符偏移不一致")
    # 公式（4）：按字符区间相交找主体 token，不单独分词主体，以保留上下文边界。
    positions = tuple(i for i, (left, right) in enumerate(offsets) if right > left and right > start and left < end)
    cursor = start
    previous_left = -1
    for position in positions:
        left, right = offsets[position]
        if left > cursor or left < previous_left or left < 0 or right > len(rendered_prompt):
            sys.exit("主体 token 偏移不连续或越界")
        cursor, previous_left = max(cursor, min(right, end)), left
    if not positions or cursor != end or len(positions) != positions[-1] - positions[0] + 1:
        sys.exit("主体未被连续 token 完整覆盖")
    # 公式（1）～（3）：答案独立为监督；适配器随后从 answer_begin 建立等长全 mask 窗口。
    return TokenAlignment(prompt_ids, positions, answer_ids)
