"""公式（11）：在冻结网络的所有实际位置上加入临时 R(Qk) 分支。"""

import torch
from torch import nn
from torch.nn import functional as F


def prepare_autograd(model: nn.Module) -> None:
    """将推理预热产生的缓存转为普通张量，使 LLaDA 缓存可参与反向传播。"""
    # 原位替换缓存值，保留共享缓存字典、数值、精度与设备。
    with torch.inference_mode(False), torch.no_grad():
        for module in model.modules():
            for name, cache in vars(module).items():
                if isinstance(cache, dict) and (name == "_buffers" or "cache" in name.lower()):
                    for key, value in cache.items():
                        if isinstance(value, torch.Tensor) and torch.is_inference(value):
                            cache[key] = value.clone()


class ResidualBranch(nn.Module):
    """保存固定 Q 和从零开始的可训练 R。"""

    def __init__(self, q: torch.Tensor, out_features: int, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.register_buffer("q", q.detach().to(device=device, copy=True))
        self.r = nn.Parameter(torch.zeros(out_features, len(q), dtype=dtype, device=device))

    def forward(self, inputs: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        # 公式（11）：W_b k + b + R(Qk)；不 detach 输入，保留跨层梯度。
        update = F.linear(F.linear(inputs.to(self.r.dtype), self.q), self.r)
        return output + update.to(output.dtype)


class CandidateModel:
    """冻结基准参数、安装多层分支，退出时移除分支并恢复原有状态。"""

    def __init__(self, model: nn.Module, factors: dict[str, torch.Tensor], *, dtype: str = "float32"):
        prepare_autograd(model)
        self.model, self.handles, self.closed = model, [], False
        self.original_flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
        self.original_modes = [(module, module.training) for module in model.modules()]
        modules = {name: model.get_submodule(name) for name in factors}
        self.branches = {
            name: ResidualBranch(q, modules[name].out_features, modules[name].weight.device, getattr(torch, dtype))
            for name, q in factors.items()
        }
        model.eval()
        for parameter, _ in self.original_flags:
            parameter.requires_grad_(False)
        for name, branch in self.branches.items():
            modules[name]._mdlm_candidate_owner = self
            self.handles.append(modules[name].register_forward_hook(
                lambda module, inputs, output, branch=branch: branch(inputs[0], output)))

    def parameters(self) -> list[nn.Parameter]:
        """公式（10）：优化器只接收 R，Q 和基准参数保持固定。"""
        return [branch.r for branch in self.branches.values()]

    def residuals(self) -> dict[str, torch.Tensor]:
        """复制当前 R 到 CPU，用于保留最佳检查点。"""
        return {name: branch.r.detach().to("cpu", copy=True) for name, branch in self.branches.items()}

    @torch.no_grad()
    def set_residuals(self, residuals: dict[str, torch.Tensor]) -> None:
        """装入指定 R，供恢复最佳候选或合并前向使用。"""
        for name, branch in self.branches.items():
            branch.r.copy_(residuals[name].to(branch.r.device))

    def close(self) -> None:
        """移除临时分支，逐项恢复基准参数的梯度标志和模块模式。"""
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
