"""通用模型操作：激活捕获、临时 hook、权重快照及原精度张量摘要。"""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import sys

import torch
from torch import nn


def tensor_sha256(tensor: torch.Tensor) -> str:
    """按 dtype、形状及连续原始字节计算摘要，不转换浮点精度。"""
    header = f"{tensor.dtype}:{tuple(tensor.shape)}:".encode("ascii")
    digest = hashlib.sha256(header)
    digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def remove_hooks(handles) -> None:
    """仅移除调用方持有的 hook，保留模型上原有的其他 hook。"""
    for handle in handles:
        handle.remove()
    handles.clear()


def register_output_hooks(model: nn.Module, callbacks: dict) -> list:
    """按给定顺序安装输出 hook；原样传递张量，由回调决定是否保留梯度。"""
    handles = []
    installed = False
    try:
        for name, callback in callbacks.items():
            handles.append(model.get_submodule(name).register_forward_hook(callback))
        installed = True
        return handles
    finally:
        if not installed and isinstance(sys.exc_info()[1], Exception):
            remove_hooks(handles)


@contextmanager
def capture_inputs(model: nn.Module, names, positions_by_request):
    """捕获 [batch, sequence, hidden] 输入的指定位置副本，并记录模块调用次数。"""
    captured, counts, handles = {}, dict.fromkeys(names, 0), []

    def hook_for(name):
        def capture(_module, inputs):
            counts[name] += 1
            captured[name] = [inputs[0][index, list(positions)].detach().clone()
                              for index, positions in enumerate(positions_by_request)]
        return capture

    try:
        for name in names:
            handles.append(model.get_submodule(name).register_forward_pre_hook(hook_for(name)))
        yield captured, counts
    finally:
        remove_hooks(handles)


@contextmanager
def capture_module_outputs(model: nn.Module, names):
    """独立复制每次调用的全位置 Tensor 输出，保持调用次序和原精度。"""
    outputs = {name: [] for name in names}
    handles = register_output_hooks(model, {
        name: (lambda module, inputs, output, name=name: outputs[name].append(output.detach().clone()))
        for name in names
    })
    try:
        yield outputs
    finally:
        remove_hooks(handles)


def prepare_autograd(model: nn.Module) -> None:
    """把推理预热产生的缓存转为普通张量，供后续可微前向使用。"""
    # 原位替换缓存值，保留共享缓存字典、数值、精度与设备。
    with torch.inference_mode(False), torch.no_grad():
        for module in model.modules():
            for name, cache in vars(module).items():
                if isinstance(cache, dict) and (name == "_buffers" or "cache" in name.lower()):
                    for key, value in cache.items():
                        if isinstance(value, torch.Tensor) and torch.is_inference(value):
                            cache[key] = value.clone()


@dataclass(frozen=True)
class TargetWeightSnapshot:
    """同一模型的指定模块权重副本；参数身份用于精确原位恢复。"""
    model_id: int
    parameter_ids: dict[str, int]
    weights: dict[str, torch.Tensor]


def snapshot_weights(model: nn.Module, names) -> TargetWeightSnapshot:
    """按给定模块名复制 weight 到 CPU，不改变精度或原参数。"""
    modules = dict(model.named_modules())
    weights, parameter_ids = {}, {}
    for name in names:
        weight = modules[name].weight
        assert not weight.is_meta, f"cannot snapshot offloaded/meta weight: {name}"
        weights[name] = weight.detach().to("cpu", copy=True)
        parameter_ids[name] = id(weight)
    return TargetWeightSnapshot(id(model), parameter_ids, weights)


def restore_weights(model: nn.Module, snapshot: TargetWeightSnapshot, names) -> None:
    """先核对全部参数，再原位复制快照，保留参数身份和设备。"""
    assert snapshot.model_id == id(model), "snapshot belongs to another model instance"
    assert (
        set(snapshot.weights) == set(names)
        and set(snapshot.parameter_ids) == set(names)
    ), "snapshot layer names differ from selected target layers"
    modules = dict(model.named_modules())
    for name in names:
        weight, original = modules[name].weight, snapshot.weights[name]
        assert id(weight) == snapshot.parameter_ids[name], f"target parameter was replaced after snapshot: {name}"
        assert not weight.is_meta and isinstance(original, torch.Tensor), f"invalid or offloaded snapshot: {name}"
        assert (
            weight.shape == original.shape
            and weight.dtype == original.dtype
        ), f"snapshot shape/dtype differs from model: {name}"
    with torch.no_grad():
        for name in names:
            weight = modules[name].weight
            weight.copy_(snapshot.weights[name].to(weight.device))
