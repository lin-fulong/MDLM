"""Small P0 interface shared by diffusion-model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MaskedEditInput:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    mask_positions: tuple[int, ...]
    subject_last_position: int


@dataclass(frozen=True)
class EditTarget:
    """Supervision kept separate from the model-facing input."""

    case_id: str
    probe_id: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class ForwardCapture:
    mask_logits: Any
    module_inputs: dict[str, Any]


@dataclass(frozen=True)
class TargetWeightSnapshot:
    """In-process baseline; only valid for the model instance that created it."""

    model_id: int
    parameter_ids: dict[str, int]
    weights: dict[str, Any]


class ModelAdapter(ABC):
    """P0 model boundary; P1 may add autonomous denoising behind this API."""

    @abstractmethod
    def prepare_edit_input(self, case: dict, probe_id: str) -> tuple[MaskedEditInput, EditTarget]:
        raise NotImplementedError

    @abstractmethod
    def forward_with_capture(self, model: Any, model_input: MaskedEditInput) -> ForwardCapture:
        raise NotImplementedError

    @abstractmethod
    def snapshot_target_weights(self, model: Any) -> TargetWeightSnapshot:
        raise NotImplementedError

    @abstractmethod
    def restore_target_weights(self, model: Any, snapshot: TargetWeightSnapshot) -> None:
        raise NotImplementedError

    def sample_denoise(self, model: Any, model_input: MaskedEditInput, *, seed: int):
        """P1 hook, deliberately unavailable until a sampler is specified."""
        raise NotImplementedError("autonomous denoising belongs to P1 and is not implemented")
