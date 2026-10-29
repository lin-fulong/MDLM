"""模型共用的输入、前向结果及适配接口；不定义编辑算法的监督或采样策略。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from util.nethook import TargetWeightSnapshot


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
class ForwardCapture:
    """原始完整词表 mask logits，以及调用方指定位置的实际模块输入。"""

    mask_logits: Any
    module_inputs: dict[str, Any]  # 适配器返回 [位置数, d_in]；算法取单个位置后可为 [d_in]。


class ModelAdapter(ABC):
    """模型模板、目标模块、指定位置捕获、可微前向与权重恢复接口。"""

    @abstractmethod
    def render_prompt(self, prompt: str) -> str:
        """按模型的聊天模板渲染提示；答案窗口和监督由算法构造。"""
        raise NotImplementedError

    @abstractmethod
    def target_module_names(self) -> tuple[str, ...]:
        """返回模型实际的目标线性层名称，保留配置层序。"""
        raise NotImplementedError

    @abstractmethod
    def inspect_target_modules(self, model: Any) -> dict:
        """核对具体模型的层结构，返回目标层输入/输出维度。"""
        raise NotImplementedError

    @abstractmethod
    def forward_with_capture(self, model: Any, requests: tuple, *, module_names=None,
                             require_all_logits_finite: bool = True) -> list[ForwardCapture]:
        """对等长的 (输入, 捕获位置) 请求前向，保留请求顺序及原精度。"""
        raise NotImplementedError

    def forward_logits(self, model: Any, model_input: MaskedEditInput | KeepModelInput,
                       positions: tuple[int, ...]) -> Any:
        """读取给定位置的完整词表 logits，保留调用方所需计算图。"""
        raise NotImplementedError("differentiable candidate forward is unavailable for this adapter")

    @abstractmethod
    def snapshot_target_weights(self, model: Any) -> TargetWeightSnapshot:
        """复制目标权重，供调用方恢复本轮基准。"""
        raise NotImplementedError

    @abstractmethod
    def restore_target_weights(self, model: Any, snapshot: TargetWeightSnapshot) -> None:
        """将快照复制回原参数，保留参数身份、精度和设备。"""
        raise NotImplementedError
