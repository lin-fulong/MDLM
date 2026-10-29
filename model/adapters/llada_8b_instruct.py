"""LLaDA 的短答案输入、实际 FFN 输入捕获、可微前向与自主去噪。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from data.validation.alignment import align_edit_prompt
from model.base import DenoiseTrace, EditTarget, ForwardCapture, KeepModelInput, MaskedEditInput, ModelAdapter, TargetWeightSnapshot, tensor_sha256


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "LLaDA-8B-Instruct"


class LLaDA8BInstructAdapter(ModelAdapter):
    """将 LLaDA 的模型调用转换为 MDLM 所需的输入、激活、轨迹与本轮快照。"""

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

    def prepare_edit_input(self, case: dict, probe_id: str) -> tuple[MaskedEditInput, EditTarget]:
        """公式（1）～（4）：选择编辑探针，分别构造全 mask 初态与监督答案。"""
        probe = next(p for p in case["prompts"]["edit"] if p["probe_id"] == probe_id)
        return self._prepare_short_answer(case, probe)

    def prepare_development_input(self, case: dict, probe_id: str) -> tuple[MaskedEditInput, EditTarget]:
        """公式（18）：从各探针组查找固定开发题，沿用短答案对齐。"""
        probe = next(p for group in case["prompts"].values() for p in group if p["probe_id"] == probe_id)
        return self._prepare_short_answer(case, probe)

    def _prepare_short_answer(self, case: dict, probe: dict) -> tuple[MaskedEditInput, EditTarget]:
        """按真实聊天模板定位主体；答案只决定窗口长度并单独返回为监督。"""
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": probe["prompt"]}], tokenize=False, add_generation_prompt=True
        )
        alignment = align_edit_prompt(
            self.tokenizer, rendered, case["fact_core"]["subject"], probe["answer"]
        )
        if self.mask_token_id in alignment.prompt_ids:
            raise ValueError(f"{case['case_id']}: edit prompt already contains the model mask token")
        begin = len(alignment.prompt_ids)
        positions = tuple(range(begin, begin + len(alignment.answer_ids)))
        token_ids = alignment.prompt_ids + (self.mask_token_id,) * len(positions)
        return (
            MaskedEditInput(
                input_ids=token_ids,
                attention_mask=(1,) * len(token_ids),
                mask_positions=positions,
                subject_last_position=alignment.subject_last_position,
            ),
            EditTarget(case["case_id"], probe["probe_id"], alignment.answer_ids),
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
        """文档 10.4：复制本轮 W_b 到 CPU，并绑定原模型及参数对象。"""
        self.inspect_target_modules(model)
        modules = dict(model.named_modules())
        weights = {}
        parameter_ids = {}
        for name in self.target_module_names():
            weight = modules[name].weight
            if weight.is_meta:
                raise ValueError(f"cannot snapshot offloaded/meta weight: {name}")
            weights[name] = weight.detach().to("cpu", copy=True)
            parameter_ids[name] = id(weight)
        return TargetWeightSnapshot(id(model), parameter_ids, weights)

    def restore_target_weights(self, model: Any, snapshot: TargetWeightSnapshot) -> None:
        """文档 10.4：将 W_b 精确复制回原参数对象，保留参数身份和精度。"""
        import torch

        self.inspect_target_modules(model)
        names = self.target_module_names()
        if snapshot.model_id != id(model):
            raise ValueError("snapshot belongs to another model instance")
        if set(snapshot.weights) != set(names) or set(snapshot.parameter_ids) != set(names):
            raise ValueError("snapshot layer names differ from selected target layers")
        modules = dict(model.named_modules())
        for name in names:
            weight, original = modules[name].weight, snapshot.weights[name]
            if id(weight) != snapshot.parameter_ids[name]:
                raise ValueError(f"target parameter was replaced after snapshot: {name}")
            if weight.is_meta or not isinstance(original, torch.Tensor):
                raise ValueError(f"invalid or offloaded snapshot: {name}")
            if weight.shape != original.shape or weight.dtype != original.dtype:
                raise ValueError(f"snapshot shape/dtype differs from model: {name}")
        with torch.no_grad():
            for name in names:
                weight = modules[name].weight
                weight.copy_(snapshot.weights[name].to(weight.device))

    def forward_with_capture(self, model: Any, model_input: MaskedEditInput) -> ForwardCapture:
        """公式（4）：读取原始 mask logits 和主体末 token 的实际 ff_out 输入。"""
        descriptions = self.inspect_target_modules(model)
        if getattr(model.config, "mask_token_id", None) != self.mask_token_id:
            raise ValueError("model and adapter mask_token_id disagree")
        if model.training:
            raise ValueError("P0 capture requires model.eval() for reproducible outputs")
        if len(model_input.input_ids) > model.config.max_sequence_length:
            raise ValueError("masked input exceeds model max_sequence_length")
        if len(model_input.attention_mask) != len(model_input.input_ids) or any(
            value != 1 for value in model_input.attention_mask
        ):
            raise ValueError("P0 full-mask input requires an all-ones attention mask")
        if not 0 <= model_input.subject_last_position < len(model_input.input_ids):
            raise ValueError("subject token position is out of range")
        if not model_input.mask_positions or any(
            pos < 0 or pos >= len(model_input.input_ids) or model_input.input_ids[pos] != self.mask_token_id
            for pos in model_input.mask_positions
        ):
            raise ValueError("mask positions do not contain the model mask token")
        all_mask_positions = {i for i, token in enumerate(model_input.input_ids) if token == self.mask_token_id}
        if len(model_input.mask_positions) != len(all_mask_positions) or set(model_input.mask_positions) != all_mask_positions:
            raise ValueError("model input contains mask tokens outside the answer window")
        if model_input.subject_last_position >= min(model_input.mask_positions):
            raise ValueError("subject token must precede the answer mask window")
        result = self._forward_capture(
            model, ((model_input, (model_input.subject_last_position,)),), descriptions,
            require_all_logits_finite=False,
        )[0]
        # 公式（4）的单事实键为 [d_in]；公式（5）的保持激活保留位置维。
        return ForwardCapture(result.mask_logits, {name: value[0] for name, value in result.module_inputs.items()})

    def forward_keep_with_capture(
        self, model: Any, model_input: KeepModelInput, capture_positions: tuple[int, ...]
    ) -> ForwardCapture:
        """公式（5）：按指定位置顺序捕获保持激活，返回 [位置数, d_in]。"""
        descriptions = self.inspect_target_modules(model)
        self._check_keep_capture_input(model, model_input, capture_positions)
        return self._forward_capture(
            model, ((model_input, capture_positions),), descriptions, require_all_logits_finite=True,
        )[0]

    def _check_keep_capture_input(self, model: Any, model_input: KeepModelInput,
                                  capture_positions: tuple[int, ...]) -> None:
        """核对保持前向的输入条件及捕获位置，允许没有 mask 的保持上下文。"""
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

    def forward_keep_batch_with_capture(
        self, model: Any, requests: tuple[tuple[KeepModelInput, tuple[int, ...]], ...]
    ) -> list[ForwardCapture]:
        """公式（5）：等长保持输入共同前向，不 padding，保持原位置与请求顺序。"""
        if not requests:
            raise ValueError("preservation batch must be nonempty")
        if len(requests) == 1:
            model_input, positions = requests[0]
            return [self.forward_keep_with_capture(model, model_input, positions)]
        descriptions = self.inspect_target_modules(model)
        if len({len(item.input_ids) for item, _ in requests}) != 1:
            raise ValueError("preservation batch requires equal input lengths")
        for model_input, positions in requests:
            self._check_keep_capture_input(model, model_input, positions)
        return self._forward_capture(model, requests, descriptions, require_all_logits_finite=True)

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
        """公式（4）（5）：单条/批量共用 pre-hook，复制实际 FFN 输入并及时移除 hook。"""
        import torch

        names = self.target_module_names()
        captured: dict[str, Any] = {}
        call_counts = {name: 0 for name in names}
        handles = []

        def hook_for(name: str):
            def capture(_module, inputs):
                call_counts[name] += 1
                captured[name] = [inputs[0][index, list(positions)].detach().clone()
                                  for index, (_, positions) in enumerate(requests)]
            return capture

        try:
            for name in names:
                handles.append(model.get_submodule(name).register_forward_pre_hook(hook_for(name)))
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
        finally:
            for handle in handles:
                handle.remove()

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

    def sample_denoise(
        self, model: Any, model_input: MaskedEditInput, *, steps: int, temperature: float, seed: int
    ) -> DenoiseTrace:
        """公式（1）的状态来源：自主采样并按置信度揭示现有答案 mask，不接收目标答案。"""
        import torch

        mask_count = len(model_input.mask_positions)
        if type(steps) is not int or not 1 <= steps <= mask_count:
            raise ValueError("denoising steps must be between 1 and the answer mask count")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")

        generator = None
        current_ids = list(model_input.input_ids)
        remaining = list(model_input.mask_positions)
        states: list[MaskedEditInput] = []
        logit_digests: list[str] = []
        reveals: list[tuple[tuple[int, int], ...]] = []
        blocked_ids = sorted(set(self.tokenizer.all_special_ids) | {self.mask_token_id})

        for step in range(steps):
            # 公式（1）的候选状态必须是每次前向之前的独立快照。
            current = MaskedEditInput(
                tuple(current_ids), model_input.attention_mask, tuple(remaining), model_input.subject_last_position
            )
            states.append(current)
            raw_logits = self.forward_with_capture(model, current).mask_logits
            # 原精度 logits 摘要留在内存，用于采样/取键重放；采样副本才转 float32。
            logit_digests.append(tensor_sha256(raw_logits))
            logits = raw_logits.float().clone()
            logits[:, [token for token in blocked_ids if 0 <= token < logits.shape[-1]]] = -torch.inf
            if not torch.isfinite(logits).any(dim=-1).all():
                raise ValueError("no finite non-special token can be sampled")
            # 当前采样约定（无独立公式编号）：屏蔽特殊 token 后，T>0 按 softmax(logits/T) 抽样。
            probabilities = torch.softmax(logits / temperature if temperature else logits, dim=-1)
            if not torch.isfinite(probabilities).all():
                raise ValueError("nonfinite sampling probabilities")
            if temperature:
                if generator is None:
                    generator = torch.Generator(device=logits.device)
                    generator.manual_seed(seed)
                predicted = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
            else:
                predicted = logits.argmax(dim=-1)
            confidence = probabilities.gather(1, predicted[:, None]).squeeze(1)

            # 每步揭示 floor(M/S) 个，前 M%S 步多一个；置信度并列时按原位置排序。
            count = mask_count // steps + (step < mask_count % steps)
            ranked = sorted(range(len(remaining)), key=lambda i: (-float(confidence[i]), remaining[i]))
            selected = sorted(ranked[:count])
            revealed = tuple((remaining[i], int(predicted[i])) for i in selected)
            for position, token_id in revealed:
                current_ids[position] = token_id
            reveals.append(revealed)
            selected_set = set(selected)
            remaining = [position for i, position in enumerate(remaining) if i not in selected_set]

        return DenoiseTrace(
            tuple(states),
            tuple(logit_digests),
            tuple(reveals),
            MaskedEditInput(tuple(current_ids), model_input.attention_mask, (), model_input.subject_last_position),
        )
