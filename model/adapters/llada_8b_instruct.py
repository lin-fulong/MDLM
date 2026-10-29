"""LLaDA 模型加载、聊天模板、指定模块/位置的激活捕获与可微前向。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model.base import ForwardCapture, KeepModelInput, MaskedEditInput, ModelAdapter
from util.nethook import TargetWeightSnapshot, capture_inputs, restore_weights, snapshot_weights


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "LLaDA-8B-Instruct"


class LLaDA8BInstructAdapter(ModelAdapter):
    """提供 LLaDA 的模型操作，不决定编辑输入、去噪或知识保持策略。"""

    def __init__(self, tokenizer: Any, *, mask_token_id: int, target_layers: tuple[int, ...]):
        """固定 tokenizer、mask 标记及可编辑层；目标为独立的 ff_out 线性层。"""
        if not target_layers or len(set(target_layers)) != len(target_layers) or any(
            type(layer) is not int or layer < 0 for layer in target_layers
        ):
            raise ValueError("target_layers must be unique nonnegative layer indices")
        if type(mask_token_id) is not int or mask_token_id < 0:
            raise ValueError("mask_token_id must be a nonnegative integer")
        self.tokenizer = tokenizer
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
        if any(layer >= config.n_layers for layer in target_layers):
            raise ValueError(f"target layer must be below {config.n_layers}")
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
        """确认独立的 FFN 输出线性层，返回 W 为 [d_out, d_in] 的实际维度。"""
        import torch

        if getattr(model.config, "block_group_size", None) != 1:
            raise ValueError("this LLaDA adapter requires block_group_size=1")
        modules = dict(model.named_modules())
        result: dict[str, dict[str, int | bool]] = {}
        seen_weights: set[tuple[str, int]] = set()
        for name in self.target_module_names():
            module = modules.get(name)
            if module is None or not isinstance(module, torch.nn.Linear):
                raise ValueError(f"target FFN output linear module absent: {name}")
            weight_key = (
                str(module.weight.device),
                module.weight.untyped_storage().data_ptr() if not module.weight.is_meta else id(module.weight),
            )
            if weight_key in seen_weights:
                raise ValueError(f"target FFN weight is shared: {name}")
            seen_weights.add(weight_key)
            if tuple(module.weight.shape) != (module.out_features, module.in_features):
                raise ValueError(f"unexpected weight direction: {name}")
            result[name] = {
                "in_features": module.in_features,
                "out_features": module.out_features,
                "has_bias": module.bias is not None,
            }
        return result

    def snapshot_target_weights(self, model: Any) -> TargetWeightSnapshot:
        """核对 LLaDA 目标层后，复制权重到 CPU 并绑定原参数对象。"""
        self.inspect_target_modules(model)
        return snapshot_weights(model, self.target_module_names())

    def restore_target_weights(self, model: Any, snapshot: TargetWeightSnapshot) -> None:
        """核对 LLaDA 目标层后，将快照精确复制回原参数。"""
        self.inspect_target_modules(model)
        restore_weights(model, snapshot, self.target_module_names())

    def forward_with_capture(
        self, model: Any, requests: tuple, *, module_names=None, require_all_logits_finite=True,
    ) -> list[ForwardCapture]:
        """单条/等长批量共用前向；捕获哪些层和位置均由调用方指定，不做 padding。"""
        if not requests:
            raise ValueError("capture batch must be nonempty")
        descriptions = self.inspect_target_modules(model)
        if module_names is not None:
            descriptions = {name: descriptions[name] for name in module_names}
        if len({len(item.input_ids) for item, _ in requests}) != 1:
            raise ValueError("capture batch requires equal input lengths")
        for model_input, positions in requests:
            self._check_capture_input(model, model_input, positions)
        return self._forward_capture(model, requests, descriptions,
                                     require_all_logits_finite=require_all_logits_finite)

    def _check_capture_input(self, model: Any, model_input: MaskedEditInput | KeepModelInput,
                             capture_positions: tuple[int, ...]) -> None:
        """核对模型输入和显式捕获位置；允许没有 mask 的上下文。"""
        if getattr(model.config, "mask_token_id", None) != self.mask_token_id:
            raise ValueError("model and adapter mask_token_id disagree")
        if model.training:
            raise ValueError("preservation capture requires model.eval() for reproducible outputs")
        length = len(model_input.input_ids)
        if not 0 < length <= model.config.max_sequence_length:
            raise ValueError("preservation input length must be within model max_sequence_length")
        if len(model_input.attention_mask) != length or any(value != 1 for value in model_input.attention_mask):
            raise ValueError("preservation input requires an all-ones attention mask")
        if not capture_positions or len(set(capture_positions)) != len(capture_positions) or any(
            type(position) is not int or not 0 <= position < length for position in capture_positions
        ):
            raise ValueError("capture positions must be unique in-range token positions")
        if len(set(model_input.mask_positions)) != len(model_input.mask_positions) or any(
            type(position) is not int or not 0 <= position < length for position in model_input.mask_positions
        ):
            raise ValueError("preservation mask positions must be unique in-range token positions")
        actual_masks = {i for i, token in enumerate(model_input.input_ids) if token == self.mask_token_id}
        if set(model_input.mask_positions) != actual_masks:
            raise ValueError("preservation mask positions differ from actual mask tokens")

    def _forward(self, model: Any, model_inputs: tuple[MaskedEditInput | KeepModelInput, ...]):
        """同位置读取完整词表 logits；梯度模式由调用方决定，不进行因果移位。"""
        import torch

        device = model.get_input_embeddings().weight.device
        return model(
            input_ids=torch.tensor([item.input_ids for item in model_inputs], dtype=torch.long, device=device),
            attention_mask=torch.tensor([item.attention_mask for item in model_inputs], dtype=torch.long, device=device),
            use_cache=False, return_dict=True,
        ).logits

    def _forward_capture(
        self, model: Any,
        requests: tuple[tuple[MaskedEditInput | KeepModelInput, tuple[int, ...]], ...], descriptions: dict,
        *, require_all_logits_finite: bool,
    ) -> list[ForwardCapture]:
        """调用通用 pre-hook，按请求封装原精度 logits 和实际模块输入。"""
        import torch

        names = tuple(descriptions)
        positions_by_request = tuple(positions for _, positions in requests)
        with capture_inputs(model, names, positions_by_request) as (captured, call_counts):
            # 同一模型后续要优化 R；no_grad 生成的 RoPE 缓存可安全用于反向传播。
            with torch.no_grad():
                logits = self._forward(model, tuple(item for item, _ in requests))
            if logits.ndim != 3 or logits.shape[:2] != (len(requests), len(requests[0][0].input_ids)):
                raise ValueError("unexpected LLaDA logits shape")
            if require_all_logits_finite and not torch.isfinite(logits).all():
                raise ValueError("nonfinite preservation logits")
            if any(call_counts[name] != 1 for name in names):
                raise ValueError(f"target FFN call counts are not one: {call_counts}")
            results = []
            for index, (model_input, positions) in enumerate(requests):
                mask_logits = logits[index, list(model_input.mask_positions)].detach().clone()
                # 沿用原返回顺序：单条按实际调用顺序，批量按配置的目标层顺序。
                module_inputs = {name: captured[name][index]
                                 for name in (captured if len(requests) == 1 else names)}
                if not torch.isfinite(mask_logits).all():
                    raise ValueError("nonfinite mask logits")
                if any(value.shape != (len(positions), descriptions[name]["in_features"])
                       for name, value in module_inputs.items()):
                    raise ValueError("captured FFN input dimension differs from linear in_features")
                if any(not torch.isfinite(value).all() for value in module_inputs.values()):
                    raise ValueError("nonfinite target module input")
                results.append(ForwardCapture(mask_logits, module_inputs))
            return results

    def forward_logits(
        self, model: Any, model_input: MaskedEditInput | KeepModelInput, positions: tuple[int, ...]
    ):
        """公式（2）（10）～（12）（16）（17）：完整词表读出，保留 R 的跨层计算图。"""
        import torch

        length = len(model_input.input_ids)
        if model.training or getattr(model.config, "mask_token_id", None) != self.mask_token_id:
            raise ValueError("P4 requires eval mode and the frozen mask token identity")
        if not 0 < length <= model.config.max_sequence_length or len(model_input.attention_mask) != length:
            raise ValueError("P4 input length differs from the model contract")
        if any(value != 1 for value in model_input.attention_mask) or not positions or any(
            type(position) is not int or not 0 <= position < length for position in positions
        ):
            raise ValueError("P4 requires all-ones attention and in-range readout positions")
        logits = self._forward(model, (model_input,))
        if logits.ndim != 3 or logits.shape[:2] != (1, length):
            raise ValueError("unexpected P4 logits shape")
        selected = logits[0, list(positions)]
        if not torch.isfinite(selected).all():
            raise FloatingPointError("nonfinite P4 readout logits")
        return selected
