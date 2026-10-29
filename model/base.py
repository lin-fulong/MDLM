"""模型适配器共用的输入、监督、轨迹与权重快照；具体前向和采样由适配器实现。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
from typing import Any


def tensor_sha256(tensor: Any) -> str:
    """按原 dtype、形状和连续原始字节计算摘要，供采样/取键重放核对；无对应公式。"""
    import torch

    header = f"{tensor.dtype}:{tuple(tensor.shape)}:".encode("ascii")
    digest = hashlib.sha256(header)
    digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class MaskedEditInput:
    """公式（1）～（4）：完整编辑状态 x_{i,t}，不含监督答案 a_i*。"""

    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    mask_positions: tuple[int, ...]
    subject_last_position: int  # 公式（4）：主体末 token 的实际 FFN 输入取键位置。


@dataclass(frozen=True)
class KeepModelInput:
    """公式（5）（10）（16）：固定保持上下文，与编辑主体和目标答案分开。"""

    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    mask_positions: tuple[int, ...]


@dataclass(frozen=True)
class EditTarget:
    """公式（1）～（3）的目标短答案 a_i*；仅用于筛选和监督，不传入自主采样器。"""

    case_id: str
    probe_id: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class ForwardCapture:
    """原始完整词表 mask logits，以及公式（4）（5）所需的实际 FFN 输入。"""

    mask_logits: Any
    module_inputs: dict[str, Any]  # 编辑取键为 [d_in]，保持捕获为 [位置数, d_in]。


@dataclass(frozen=True)
class DenoiseTrace:
    """公式（1）的采样来源：逐步前向前输入、logits 摘要、揭示记录及终态。

    前三项按采样步一一对应；终态单独保存，不作为非空 mask 监督状态。
    """

    pre_forward_inputs: tuple[MaskedEditInput, ...]
    mask_logit_sha256: tuple[str, ...]
    revealed_tokens: tuple[tuple[tuple[int, int], ...], ...]  # 每步的 (位置, token_id)。
    final_input: MaskedEditInput


@dataclass(frozen=True)
class TargetWeightSnapshot:
    """文档 10.4 的本轮 W_b 快照，仅供同一模型恢复；固定 W_0 由 runner 另存。"""

    model_id: int
    parameter_ids: dict[str, int]
    weights: dict[str, Any]


class ModelAdapter(ABC):
    """统一输入对齐、激活捕获、可微前向、自主去噪和本轮权重恢复接口。"""

    @abstractmethod
    def prepare_edit_input(self, case: dict, probe_id: str) -> tuple[MaskedEditInput, EditTarget]:
        """公式（1）～（4）：分别返回全 mask 编辑输入和仅供监督使用的目标。"""
        raise NotImplementedError

    def prepare_development_input(self, case: dict, probe_id: str) -> tuple[MaskedEditInput, EditTarget]:
        """公式（18）：构造固定开发短答案输入，参考答案与模型输入分离。"""
        raise NotImplementedError("development probe alignment is unavailable for this adapter")

    @abstractmethod
    def forward_with_capture(self, model: Any, model_input: MaskedEditInput) -> ForwardCapture:
        """公式（4）：返回原始 mask logits 和主体末位置的实际目标模块输入。"""
        raise NotImplementedError

    def forward_keep_with_capture(self, model: Any, model_input: KeepModelInput,
                                  capture_positions: tuple[int, ...]) -> ForwardCapture:
        """公式（5）：按给定位置顺序返回实际激活，形状为 [位置数, d_in]。"""
        raise NotImplementedError("preservation-input capture is unavailable for this model adapter")

    def forward_logits(self, model: Any, model_input: MaskedEditInput | KeepModelInput,
                       positions: tuple[int, ...]) -> Any:
        """公式（2）（10）～（12）（16）（17）：返回指定位置完整词表 logits，保留候选计算图。"""
        raise NotImplementedError("differentiable candidate forward is unavailable for this adapter")

    @abstractmethod
    def snapshot_target_weights(self, model: Any) -> TargetWeightSnapshot:
        """文档 10.4（未编号）：保存本轮基准目标权重 W_b。"""
        raise NotImplementedError

    @abstractmethod
    def restore_target_weights(self, model: Any, snapshot: TargetWeightSnapshot) -> None:
        """文档 10.4（未编号）：精确恢复 W_b，供拒绝、异常或独立案例切换使用。"""
        raise NotImplementedError

    def sample_denoise(self, model: Any, model_input: MaskedEditInput, *, steps: int,
                       temperature: float, seed: int) -> DenoiseTrace:
        """公式（1）的前置采样：只接收输入和采样参数，由具体模型决定揭示策略。"""
        raise NotImplementedError("autonomous denoising is unavailable for this model adapter")
