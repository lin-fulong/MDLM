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
    """Require a unique subject and a stable token boundary before the answer."""
    if not all(isinstance(value, str) and value for value in (rendered_prompt, subject, answer)):
        raise AlignmentError("rendered prompt, subject and answer must be nonempty strings")
    if rendered_prompt.count(subject) != 1:
        raise AlignmentError("subject must occur exactly once in the rendered edit prompt")
    start = rendered_prompt.index(subject)
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
    positions = tuple(i for i, (left, right) in enumerate(offsets) if right > start and left < end)
    cursor = start
    for position in positions:
        left, right = offsets[position]
        if left != cursor or right <= left or right > end:
            raise AlignmentError("subject does not align to contiguous whole tokens")
        cursor = right
    if not positions or cursor != end:
        raise AlignmentError("subject does not align to contiguous whole tokens")
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
