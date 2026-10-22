"""Exact tokenizer alignment for P0 edit prompts and short answers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class AlignmentError(ValueError):
    pass


@dataclass(frozen=True)
class TokenAlignment:
    prompt_ids: tuple[int, ...]
    subject_positions: tuple[int, ...]
    answer_ids: tuple[int, ...]

    @property
    def subject_last_position(self) -> int:
        return self.subject_positions[-1]


def align_edit_prompt(tokenizer: Any, rendered_prompt: str, subject: str, answer: str) -> TokenAlignment:
    """Locate tokens covering the unique subject and a stable answer boundary."""
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
    if len(offsets) != len(prompt_ids):
        raise AlignmentError("offset count differs from prompt token count")
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
    if not positions or cursor != end or positions != tuple(range(positions[0], positions[-1] + 1)):
        raise AlignmentError("subject is not fully covered by contiguous tokens")
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise AlignmentError("answer changes prompt tokenization; no stable mask boundary")
    answer_ids = full_ids[len(prompt_ids) :]
    if not answer_ids:
        raise AlignmentError("answer tokenization is empty")
    if set(answer_ids) & set(tokenizer.all_special_ids):
        raise AlignmentError("answer contains tokenizer special tokens")
    decoded_answer = tokenizer.decode(
        answer_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    if decoded_answer != answer:
        raise AlignmentError(
            f"answer token IDs decode to {decoded_answer!r}, expected {answer!r}"
        )
    return TokenAlignment(prompt_ids, positions, answer_ids)
