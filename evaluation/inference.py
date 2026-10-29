"""评估专用的定长全 mask 推理；不使用编辑轨迹、取键或优化代码。"""

import hashlib
import math

import torch
from safetensors.torch import load_file

from model.base import KeepModelInput
from model.registry import get_adapter
from .types import GenerationRequest, Prediction


PROTOCOL = "diffusion_short_v1"


def validate_generation(config):
    if set(config) - {"steps", "temperature"}:
        raise ValueError("generation 只支持 steps 和 temperature")
    steps, temperature = config.get("steps"), config.get("temperature", 0.0)
    if steps is not None and (type(steps) is not int or steps < 1):
        raise ValueError("steps 必须为正整数或 null")
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature 必须为有限非负数")
    return {"steps": steps, "temperature": temperature}


def prepare_request(adapter, task, seed):
    # 连同提示分词，再核对前缀，避免答案边界处的 token 合并改变标签。
    rendered = adapter.render_prompt(task.prompt)
    prompt = tuple(adapter.tokenizer.encode(rendered, add_special_tokens=False))
    full = tuple(adapter.tokenizer.encode(rendered + task.answer, add_special_tokens=False))
    target = full[len(prompt):]
    if not prompt or full[:len(prompt)] != prompt or not target:
        raise ValueError(f"{task.case_id}/{task.probe_id}: 无法建立稳定的答案 token 边界")
    if adapter.mask_token_id in prompt or set(target) & set(adapter.tokenizer.all_special_ids):
        raise ValueError("问题含预置 mask 或参考答案含特殊 token")
    if adapter.tokenizer.decode(target, skip_special_tokens=False, clean_up_tokenization_spaces=False) != task.answer:
        raise ValueError("参考答案无法精确分词/解码")
    derived = int.from_bytes(hashlib.sha256(f"{seed}:{PROTOCOL}:{task.task_id}".encode()).digest()[:8], "big") % (2**63)
    # 推理请求只含提示、长度和种子；参考答案留给独立评分。
    return GenerationRequest(prompt, len(target), derived), target


@torch.no_grad()
def generate(model, adapter, request, *, steps=None, temperature=0.0):
    length = request.answer_length
    if type(length) is not int or length < 1:
        raise ValueError("答案窗口必须非空")
    begin = len(request.prompt_ids)
    steps = min(steps, length) if steps is not None else length
    tokens = list(request.prompt_ids) + [adapter.mask_token_id] * length
    if len(tokens) > model.config.max_sequence_length:
        raise ValueError("评估输入超过模型上下文长度；不自动截断")
    remaining = list(range(begin, len(tokens)))
    generator = None
    blocked = set(adapter.tokenizer.all_special_ids) | {adapter.mask_token_id}
    for step in range(steps):
        current = KeepModelInput(tuple(tokens), (1,) * len(tokens), tuple(remaining))
        logits = adapter.forward_logits(model, current, tuple(remaining)).float().clone()
        logits[:, [i for i in blocked if 0 <= i < logits.shape[-1]]] = -torch.inf
        if not torch.isfinite(logits).any(dim=-1).all():
            raise ValueError("没有可生成的普通 token")
        probabilities = torch.softmax(logits / temperature if temperature else logits, dim=-1)
        if not torch.isfinite(probabilities).all():
            raise ValueError("非有限生成概率")
        if temperature:
            if generator is None:
                # 每道题独立随机流，不推进全局 RNG。
                generator = torch.Generator(device=logits.device).manual_seed(request.seed)
            predicted = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
        else:
            predicted = logits.argmax(-1)
        confidence = probabilities.gather(1, predicted[:, None]).squeeze(-1)
        count = length // steps + (step < length % steps)
        # 均分每步揭示量；置信度并列时按位置排序，保持结果可复现。
        selected = sorted(range(len(remaining)), key=lambda i: (-float(confidence[i]), remaining[i]))[:count]
        for i in selected:
            tokens[remaining[i]] = int(predicted[i])
        chosen = set(selected)
        remaining = [position for i, position in enumerate(remaining) if i not in chosen]
    ids = tuple(tokens[begin:])
    return Prediction(ids, adapter.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False))


class LLaDABackend:
    """只操作本评估进程的模型副本，切换独立案例前恢复原始目标权重。"""

    def __init__(self, settings, device):
        self.adapter = get_adapter(model_path=settings["model_path"], target_layers=tuple(settings["target_layers"]))
        self.model = self.adapter.load_model(settings["model_path"], torch_dtype=settings["torch_dtype"], device_map=device)
        self.original = self.adapter.snapshot_target_weights(self.model)

    def use_checkpoint(self, path):
        self.adapter.restore_target_weights(self.model, self.original)
        if path is None:
            return
        weights = load_file(str(path / "target_weights.safetensors"))
        names = self.adapter.target_module_names()
        if set(weights) != set(names):
            raise ValueError("最终权重的目标层集合不匹配")
        parameters = {name: self.model.get_submodule(name).weight for name in names}
        # 先验证全部权重，再统一复制，防止只写入一部分层。
        for name, parameter in parameters.items():
            value = weights[name]
            if value.shape != parameter.shape or value.dtype != parameter.dtype or not torch.isfinite(value).all():
                raise ValueError(f"最终权重形状、精度或数值非法: {name}")
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(weights[name].to(parameter.device))

    def prepare(self, task, seed):
        return prepare_request(self.adapter, task, seed)

    def predict(self, request, generation):
        return generate(self.model, self.adapter, request, **generation)

    def close(self):
        self.adapter.restore_target_weights(self.model, self.original)
        del self.model, self.original
