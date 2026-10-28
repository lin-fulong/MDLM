"""公式（3）（10）（12）：固定 Q，按全局位置权重联合优化多层残差 R。"""

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from MDLM.candidate import CandidateModel
from MDLM.states import EditTrainingExample
from model.base import KeepModelInput, ModelAdapter


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
