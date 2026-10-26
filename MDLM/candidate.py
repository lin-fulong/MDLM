"""Formula (11): temporary all-position branches on a frozen real network."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def require_differentiable_model(model: nn.Module) -> None:
    """Reject inference hooks even when hidden inside a SequentialHook on a frozen layer."""
    hooks = [getattr(module, "_hf_hook", None) for module in model.modules()]
    while hooks:
        hook = hooks.pop()
        if getattr(hook, "offload", False) or getattr(hook, "no_grad", False):
            raise ValueError("P4 requires resident layers with continuous gradients; inference offload/no_grad is unsupported")
        hooks.extend(getattr(hook, "hooks", ()))
    # 兼容此前推理预热留下的 inference tensor；LLaDA 多层共享同一 BufferCache。
    # 原位替换缓存条目，保持共享字典、值、精度和设备一致，不改动模型参数。
    seen = set()
    with torch.inference_mode(False), torch.no_grad():
        for module in model.modules():
            for name, cache in vars(module).items():
                if not isinstance(cache, dict) or (name != "_buffers" and "cache" not in name.lower()):
                    continue
                if id(cache) in seen:
                    continue
                seen.add(id(cache))
                for key, value in cache.items():
                    if isinstance(value, torch.Tensor) and torch.is_inference(value):
                        cache[key] = value.clone()


class ResidualBranch(nn.Module):
    def __init__(self, q: torch.Tensor, out_features: int, device: torch.device):
        super().__init__()
        self.register_buffer("q", q.detach().to(device=device, copy=True))
        self.r = nn.Parameter(torch.zeros(out_features, len(q), dtype=q.dtype, device=device))

    def forward(self, inputs: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        # 公式（11）：作用于每个实际位置；输入 k 保留来自前层 R 的梯度。
        update = F.linear(F.linear(inputs.to(self.r.dtype), self.q), self.r)
        return output + update.to(output.dtype)


class CandidateModel:
    """Install/remove factor hooks; baseline tensors are never modified or replaced."""

    def __init__(self, model: nn.Module, factors: dict[str, torch.Tensor], *, dtype: str = "float32"):
        if not factors:
            raise ValueError("candidate requires at least one target layer")
        require_differentiable_model(model)
        self.model, self.branches, self.handles = model, {}, []
        self.closed = False
        self.original_flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
        self.original_modes = [(module, module.training) for module in model.modules()]
        modules = dict(model.named_modules())
        aliases = {}
        for name, parameter in model.named_parameters(remove_duplicate=False):
            if not parameter.is_meta:
                storage = (str(parameter.device), parameter.untyped_storage().data_ptr())
                aliases.setdefault(storage, []).append(name)
        for name, q in factors.items():
            module = modules.get(name)
            if not isinstance(module, nn.Linear) or module.weight.is_meta:
                raise ValueError(f"candidate requires a resident linear target: {name}")
            if getattr(module, "_mdlm_candidate_owner", None) is not None:
                raise ValueError("a candidate branch is already installed on this target")
            storage = (str(module.weight.device), module.weight.untyped_storage().data_ptr())
            # 未选中模块的别名同样会随实际写回改变，不能只检查选中层之间的共享。
            if len(aliases[storage]) != 1:
                raise ValueError("shared target weights require an explicit tied-parameter adapter")
            if q.ndim != 2 or q.shape[1] != module.in_features or q.shape[0] < 1 or not torch.isfinite(q).all():
                raise ValueError(f"candidate Q shape or finiteness differs at {name}")
            if q.dtype != getattr(torch, dtype):
                raise ValueError("P4 residual_dtype must equal saved Q dtype to keep Q unchanged")
            self.branches[name] = ResidualBranch(q, module.out_features, module.weight.device)
        try:
            model.eval()
            for parameter, _ in self.original_flags:
                parameter.requires_grad_(False)
            for name, branch in self.branches.items():
                modules[name]._mdlm_candidate_owner = self
                self.handles.append(modules[name].register_forward_hook(
                    lambda module, inputs, output, branch=branch: branch(inputs[0], output)
                ))
        except Exception:
            self.close()
            raise

    def parameters(self) -> list[nn.Parameter]:
        return [branch.r for branch in self.branches.values()]

    def residuals(self) -> dict[str, torch.Tensor]:
        return {name: branch.r.detach().to("cpu", copy=True) for name, branch in self.branches.items()}

    def set_residuals(self, residuals: dict[str, torch.Tensor]) -> None:
        if set(residuals) != set(self.branches):
            raise ValueError("candidate residual layer names differ")
        with torch.no_grad():
            for name, branch in self.branches.items():
                value = residuals[name]
                if value.shape != branch.r.shape or value.dtype != branch.r.dtype or not torch.isfinite(value).all():
                    raise ValueError(f"candidate residual shape/dtype/values differ at {name}")
                branch.r.copy_(value.to(branch.r.device))

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for name in self.branches:
            module = self.model.get_submodule(name)
            if getattr(module, "_mdlm_candidate_owner", None) is self:
                del module._mdlm_candidate_owner
        for parameter, flag in self.original_flags:
            parameter.requires_grad_(flag)
        for module, training in self.original_modes:
            module.training = training

    def __enter__(self) -> "CandidateModel":
        return self

    def __exit__(self, *args) -> None:
        self.close()
