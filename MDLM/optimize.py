"""公式（10）～（12）：安装临时 R(Qk) 分支，固定 Q，联合优化并选择多层残差 R。"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from MDLM.states import EditTrainingExample
from model.base import KeepModelInput, ModelAdapter


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


@dataclass(frozen=True)
class ResidualOptimizationConfig:
    """公式（10）：逐层 ρ、保持系数 β、Adam 预算和预定检查点。"""

    rho_by_layer: dict[str, float]
    beta: float = 0.0
    learning_rate: float = 0.01
    max_steps: int = 20
    checkpoints: tuple[int, ...] = (1, 5, 10, 20)
    microbatch_size: int = 1
    residual_dtype: str = "float32"
    seed: int = 2026


@dataclass(frozen=True)
class KeepTrainingExample:
    """保持训练输入、位置权重及最初 theta_0 的完整词表参考。"""

    model_input: KeepModelInput
    positions: tuple[int, ...]
    weights: tuple[float, ...]
    reference_log_probs: torch.Tensor


@dataclass
class OptimizationResult:
    """选中的 R，以及目标值轨迹和最佳步数。"""

    residuals: dict[str, torch.Tensor] | None
    report: dict


def log_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """半精度 logits 转 float32 计算 CE/KL；float64 输入保留原精度。"""
    return F.log_softmax(logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits, dim=-1)


def edit_loss_from_logits(logits: torch.Tensor, example: EditTrainingExample) -> torch.Tensor:
    """公式（3）：按 P1 给定的全局位置权重求交叉熵之和。"""
    log_probs = log_probabilities(logits)
    targets = torch.tensor(example.targets, device=logits.device, dtype=torch.long)
    weights = torch.tensor(example.weights, device=logits.device, dtype=log_probs.dtype)
    return -(log_probs.gather(1, targets[:, None]).squeeze(1) * weights).sum()


def edit_loss(adapter: ModelAdapter, model, example: EditTrainingExample) -> torch.Tensor:
    """重放一个固定编辑状态并计算其加权短答案损失。"""
    return edit_loss_from_logits(adapter.forward_logits(model, example.model_input, example.positions), example)


def keep_loss_from_logits(logits: torch.Tensor, example: KeepTrainingExample) -> torch.Tensor:
    """公式（10）的保持项：完整词表 KL(p_theta0 || p_candidate)，按固定位置权重求和。"""
    log_probs = log_probabilities(logits)
    reference = example.reference_log_probs.to(device=log_probs.device, dtype=log_probs.dtype)
    weights = torch.tensor(example.weights, device=log_probs.device, dtype=log_probs.dtype)
    return (F.kl_div(log_probs, reference, log_target=True, reduction="none").sum(-1) * weights).sum()


def keep_loss(adapter: ModelAdapter, model, example: KeepTrainingExample) -> torch.Tensor:
    """重放一条保持训练输入并计算其加权 KL。"""
    return keep_loss_from_logits(adapter.forward_logits(model, example.model_input, example.positions), example)


def regularization(candidate: CandidateModel, settings: ResidualOptimizationConfig) -> torch.Tensor:
    """公式（10）：Σ_l (ρ_l/2)||R_l||²，先缩放再平方，保留原有精度与计算顺序。"""
    device = candidate.parameters()[0].device
    return sum((branch.r.double() * (math.sqrt(settings.rho_by_layer[name]) / math.sqrt(2))).square().sum().to(device)
               for name, branch in candidate.branches.items())


def accumulate_data_gradients(candidate: CandidateModel, adapter: ModelAdapter,
                              edits: list[EditTrainingExample], keeps: list[KeepTrainingExample],
                              settings: ResidualOptimizationConfig) -> None:
    """公式（12）：逐微批累计数据梯度，每个完整优化步只加入一次 ρR。"""
    device = candidate.parameters()[0].device
    for examples, loss_function, coefficient in ((edits, edit_loss, 1.0), (keeps, keep_loss, settings.beta)):
        if coefficient == 0:
            continue
        for begin in range(0, len(examples), settings.microbatch_size):
            # 公式（3）：沿用全局位置权重，不在微批内重新平均。
            loss = sum(coefficient * loss_function(adapter, candidate.model, item).to(device)
                       for item in examples[begin:begin + settings.microbatch_size])
            loss.backward()
    regularization(candidate, settings).backward()


@torch.no_grad()
def evaluate_objective(candidate: CandidateModel, adapter: ModelAdapter, edits: list[EditTrainingExample],
                       keeps: list[KeepTrainingExample], settings: ResidualOptimizationConfig) -> dict:
    """公式（10）：计算 J = F + βL_keep + Σ_l (ρ_l/2)||R_l||²，用于检查点选择。"""
    edit = math.fsum(float(edit_loss(adapter, candidate.model, item)) for item in edits)
    keep = math.fsum(float(keep_loss(adapter, candidate.model, item)) for item in keeps) if settings.beta else 0.0
    penalty = float(regularization(candidate, settings))
    return {"edit": edit, "keep": keep, "regularization": penalty,
            "total": math.fsum((edit, settings.beta * keep, penalty))}


def optimize_residuals(candidate: CandidateModel, adapter: ModelAdapter, edits: list[EditTrainingExample],
                       keeps: list[KeepTrainingExample], settings: ResidualOptimizationConfig) -> OptimizationResult:
    """从 R=0 开始联合优化，在预定检查点中选择有限、非零且优于 J(0) 的最佳 R。"""
    torch.manual_seed(settings.seed)
    optimizer = torch.optim.Adam(candidate.parameters(), lr=settings.learning_rate,
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    baseline = evaluate_objective(candidate, adapter, edits, keeps, settings)
    report = {"baseline": baseline, "history": [{"step": 0, "loss": baseline}], "selected_step": None}
    best_residuals, best_total = None, math.inf
    for step in range(1, settings.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulate_data_gradients(candidate, adapter, edits, keeps, settings)
        optimizer.step()
        if step in settings.checkpoints:
            loss = evaluate_objective(candidate, adapter, edits, keeps, settings)
            report["history"].append({"step": step, "loss": loss})
            # 算法的候选选择条件：R 有限且非零，J(R) 有限并严格优于基准及已有最佳值。
            parameters = candidate.parameters()
            finite = math.isfinite(loss["total"]) and all(bool(torch.isfinite(p).all()) for p in parameters)
            nonzero = any(bool(torch.count_nonzero(p)) for p in parameters)
            if finite and nonzero and loss["total"] < min(baseline["total"], best_total):
                best_total, best_residuals = loss["total"], candidate.residuals()
                report["selected_step"] = step
    report["status"] = "no_candidate" if best_residuals is None else "candidate_ready"
    if best_residuals is not None:
        candidate.set_residuals(best_residuals)
        report["selected_loss"] = evaluate_objective(candidate, adapter, edits, keeps, settings)
    return OptimizationResult(best_residuals, report)
