"""P0/P1 interface shared by diffusion-model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
from typing import Any


def tensor_sha256(tensor: Any) -> str:
    """Digest the original dtype, shape and bytes without reducing precision."""
    import torch

    raw_bytes = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    header = f"{tensor.dtype}:{tuple(tensor.shape)}:".encode("ascii")
    return hashlib.sha256(header + raw_bytes).hexdigest()


@dataclass(frozen=True)
class MaskedEditInput:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    mask_positions: tuple[int, ...]
    subject_last_position: int


@dataclass(frozen=True)
class KeepModelInput:
    """Frozen preservation context, independent of edit subjects and targets."""

    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    mask_positions: tuple[int, ...]


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
class DenoiseTrace:
    """Pre-forward inputs, raw mask-logit digests, reveals, and terminal input."""

    pre_forward_inputs: tuple[MaskedEditInput, ...]
    mask_logit_sha256: tuple[str, ...]
    revealed_tokens: tuple[tuple[tuple[int, int], ...], ...]
    final_input: MaskedEditInput


@dataclass(frozen=True)
class TargetWeightSnapshot:
    """In-process baseline; only valid for the model instance that created it."""

    model_id: int
    parameter_ids: dict[str, int]
    weights: dict[str, Any]


class ModelAdapter(ABC):
    """Shared model boundary for P0 alignment and P1 autonomous denoising."""

    @abstractmethod
    def prepare_edit_input(self, case: dict, probe_id: str) -> tuple[MaskedEditInput, EditTarget]:
        raise NotImplementedError

    @abstractmethod
    def forward_with_capture(self, model: Any, model_input: MaskedEditInput) -> ForwardCapture:
        raise NotImplementedError

    def forward_keep_with_capture(
        self, model: Any, model_input: KeepModelInput, capture_positions: tuple[int, ...]
    ) -> ForwardCapture:
        """Return each selected module's actual inputs as [position, input feature]."""
        raise NotImplementedError("preservation-input capture is unavailable for this model adapter")

    @abstractmethod
    def snapshot_target_weights(self, model: Any) -> TargetWeightSnapshot:
        raise NotImplementedError

    @abstractmethod
    def restore_target_weights(self, model: Any, snapshot: TargetWeightSnapshot) -> None:
        raise NotImplementedError

    def sample_denoise(
        self, model: Any, model_input: MaskedEditInput, *, steps: int, temperature: float, seed: int
    ) -> DenoiseTrace:
        """Sample without receiving the edit target; adapters define their own token policy."""
        raise NotImplementedError("autonomous denoising is unavailable for this model adapter")
