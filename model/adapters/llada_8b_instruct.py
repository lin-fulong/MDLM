"""LLaDA 模型加载、聊天模板、指定模块/位置的激活捕获与可微前向。"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from model.base import ForwardCapture, KeepModelInput, MaskedEditInput, ModelAdapter
from util.nethook import TargetWeightSnapshot, capture_inputs, restore_weights, snapshot_weights


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "LLaDA-8B-Instruct"


class LLaDA8BInstructAdapter(ModelAdapter):
    """提供 LLaDA 的模型操作，不决定编辑输入、去噪或知识保持策略。"""

    def __init__(self, tokenizer: Any, *, mask_token_id: int, target_layers: tuple[int, ...]):
        """固定 tokenizer、mask 标记及可编辑层；目标为独立的 ff_out 线性层。"""
        self.tokenizer = tokenizer
        if not target_layers or len(set(target_layers)) != len(target_layers) or any(
            type(layer) is not int or layer < 0 for layer in target_layers
        ):
            sys.exit("目标层必须为非空、不重复的非负层号")
        self.mask_token_id = mask_token_id
        self.target_layers = target_layers

    @classmethod
    def from_local_tokenizer(
        cls, *, model_path: str | Path = DEFAULT_MODEL_PATH, target_layers: tuple[int, ...]
    ) -> "LLaDA8BInstructAdapter":
        """从本地 tokenizer/config 读取模板与 mask ID，不下载模型文件。"""
        from transformers import AutoConfig, AutoTokenizer

        model_path = str(Path(model_path))
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        return cls(tokenizer, mask_token_id=config.mask_token_id, target_layers=target_layers)

    @staticmethod
    def load_model(model_path: str | Path = DEFAULT_MODEL_PATH, **kwargs):
        """按调用方的精度和设备设置加载本地权重，并固定为 eval 模式。"""
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), trust_remote_code=True, local_files_only=True, **kwargs
        )
        model.eval()
        return model

    def render_prompt(self, prompt: str) -> str:
        """沿用模型的 user 模板与 generation prompt，不加入答案。"""
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )

    def target_module_names(self) -> tuple[str, ...]:
        """公式（4）～（15）的目标模块 ℓ，顺序沿用 target_layers。"""
        return tuple(f"model.transformer.blocks.{layer}.ff_out" for layer in self.target_layers)

    def inspect_target_modules(self, model: Any) -> dict[str, dict[str, int | bool]]:
        """读取目标 FFN 输出线性层的实际维度。"""
        result: dict[str, dict[str, int | bool]] = {}
        pointers = set()
        for name in self.target_module_names():
            module = model.get_submodule(name)
            pointer = (str(module.weight.device), module.weight.untyped_storage().data_ptr())
            if pointer in pointers or module.weight.shape != (module.out_features, module.in_features):
                sys.exit("目标层共享权重或权重方向不符合独立线性层约定")
            pointers.add(pointer)
            result[name] = {
                "in_features": module.in_features,
                "out_features": module.out_features,
                "has_bias": module.bias is not None,
            }
        return result

    def snapshot_target_weights(self, model: Any) -> TargetWeightSnapshot:
        """复制目标权重到 CPU。"""
        return snapshot_weights(model, self.target_module_names())

    def restore_target_weights(self, model: Any, snapshot: TargetWeightSnapshot) -> None:
        """将快照精确复制回原参数。"""
        restore_weights(model, snapshot, self.target_module_names())

    def forward_with_capture(
        self, model: Any, requests: tuple, *, module_names=None,
    ) -> list[ForwardCapture]:
        """单条/等长批量共用前向；捕获哪些层和位置均由调用方指定，不做 padding。"""
        descriptions = self.inspect_target_modules(model)
        if module_names is not None:
            descriptions = {name: descriptions[name] for name in module_names}
        return self._forward_capture(model, requests, descriptions)

    def _forward(self, model: Any, model_inputs: tuple[MaskedEditInput | KeepModelInput, ...]):
        """同位置读取完整词表 logits；梯度模式由调用方决定，不进行因果移位。"""
        import torch

        device = model.get_input_embeddings().weight.device
        if model.training:
            sys.exit("MDLM 前向必须使用 eval 模式")
        if any(len(item.attention_mask) != len(item.input_ids) or item.mask_positions != tuple(
            p for p, token in enumerate(item.input_ids) if token == self.mask_token_id
        ) for item in model_inputs):
            sys.exit("模型输入的注意力长度或 mask 位置不一致")
        limit = getattr(model.config, "max_sequence_length", None)
        if limit is not None and any(len(item.input_ids) > limit for item in model_inputs):
            sys.exit("模型输入超过上下文长度，不自动截断")
        return model(
            input_ids=torch.tensor([item.input_ids for item in model_inputs], dtype=torch.long, device=device),
            attention_mask=torch.tensor([item.attention_mask for item in model_inputs], dtype=torch.long, device=device),
            use_cache=False, return_dict=True,
        ).logits

    def _forward_capture(
        self, model: Any,
        requests: tuple[tuple[MaskedEditInput | KeepModelInput, tuple[int, ...]], ...], descriptions: dict,
    ) -> list[ForwardCapture]:
        """调用通用 pre-hook，按请求封装原精度 logits 和实际模块输入。"""
        import torch

        names = tuple(descriptions)
        positions_by_request = tuple(positions for _, positions in requests)
        with capture_inputs(model, names, positions_by_request) as (captured, counts):
            # 同一模型后续要优化 R；no_grad 生成的 RoPE 缓存可安全用于反向传播。
            with torch.no_grad():
                logits = self._forward(model, tuple(item for item, _ in requests))
            if any(counts[name] != 1 for name in names):
                sys.exit("取键和保持统计要求目标模块每次前向恰好调用一次")
            results = []
            for index, (model_input, positions) in enumerate(requests):
                mask_logits = logits[index, list(model_input.mask_positions)].detach().clone()
                # 沿用原返回顺序：单条按实际调用顺序，批量按配置的目标层顺序。
                module_inputs = {name: captured[name][index]
                                 for name in (captured if len(requests) == 1 else names)}
                results.append(ForwardCapture(mask_logits, module_inputs))
            return results

    def forward_logits(
        self, model: Any, model_input: MaskedEditInput | KeepModelInput, positions: tuple[int, ...],
    ):
        """公式（2）（10）～（12）（16）（17）：完整词表读出，保留 R 的跨层计算图。"""
        logits = self._forward(model, (model_input,))
        return logits[0, list(positions)]
