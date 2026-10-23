"""Formulas (3), (10) and (12): finite-budget joint optimization of fixed-Q residuals."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import resource
import time
from typing import Callable

import torch
from torch.nn import functional as F

from MDLM.candidate import CandidateModel
from MDLM.contracts import ResidualOptimizationConfig
from MDLM.states import EditTrainingExample
from MDLM.writeback import stable_frobenius_norm
from model.base import KeepModelInput, ModelAdapter


@dataclass(frozen=True)
class KeepTrainingExample:
    model_input: KeepModelInput
    positions: tuple[int, ...]
    weights: tuple[float, ...]
    reference_log_probs: torch.Tensor  # 完整词表，来自最初 theta_0；不从 theta_b 替代。


@dataclass
class OptimizationResult:
    residuals: dict[str, torch.Tensor] | None
    optimizer_state: dict | None
    report: dict


class OptimizationError(RuntimeError):
    def __init__(self, message: str, report: dict):
        super().__init__(message)
        self.report = report


def log_probabilities(logits: torch.Tensor) -> torch.Tensor:
    # 半精度模型的 CE/KL 在 float32 中计算；float64 小例保留精度以检查梯度。
    return F.log_softmax(logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits, dim=-1)


def edit_loss(adapter: ModelAdapter, model, example: EditTrainingExample) -> torch.Tensor:
    logits = adapter.forward_logits(model, example.model_input, example.positions)
    log_probs = log_probabilities(logits)
    targets = torch.tensor(example.targets, device=logits.device, dtype=torch.long)
    weights = torch.tensor(example.weights, device=logits.device, dtype=log_probs.dtype)
    # 公式（3）：P1 的全局位置权重直接求和，微批内不能重新平均。
    return -(log_probs.gather(1, targets[:, None]).squeeze(1) * weights).sum()


def keep_loss(adapter: ModelAdapter, model, example: KeepTrainingExample) -> torch.Tensor:
    log_probs = log_probabilities(adapter.forward_logits(model, example.model_input, example.positions))
    reference = example.reference_log_probs.to(device=log_probs.device, dtype=log_probs.dtype)
    if reference.shape != log_probs.shape:
        raise ValueError("theta_0 and candidate full-vocabulary logits shapes differ")
    weights = torch.tensor(example.weights, device=log_probs.device, dtype=log_probs.dtype)
    return (F.kl_div(log_probs, reference, log_target=True, reduction="none").sum(-1) * weights).sum()


def regularization(candidate: CandidateModel, settings: ResidualOptimizationConfig) -> torch.Tensor:
    device = candidate.parameters()[0].device
    # 公式（10）的等价加权平方：先缩放再平方，避免 R² 先上溢/下溢。
    return sum((branch.r.double() * (math.sqrt(settings.rho_by_layer[name]) / math.sqrt(2))).square().sum().to(device)
               for name, branch in candidate.branches.items())


def accumulate_data_gradients(candidate: CandidateModel, adapter: ModelAdapter,
                              edits: list[EditTrainingExample], keeps: list[KeepTrainingExample],
                              settings: ResidualOptimizationConfig) -> None:
    """One global step: bounded microbatch graphs, then exactly one residual penalty."""
    device = candidate.parameters()[0].device
    for examples, loss_function, coefficient in ((edits, edit_loss, 1.0), (keeps, keep_loss, settings.beta)):
        if coefficient == 0:
            continue
        for begin in range(0, len(examples), settings.microbatch_size):
            loss = sum(coefficient * loss_function(adapter, candidate.model, item).to(device)
                       for item in examples[begin:begin + settings.microbatch_size])
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite P4 microbatch loss")
            loss.backward()
    for name, branch in candidate.branches.items():
        if branch.r.grad is None:
            raise RuntimeError(f"missing data gradient at {name}; the candidate graph is disconnected")
    # 公式（12）：先确认真实数据图连通，再加 rho R；不能让正则掩盖断链。
    penalty = regularization(candidate, settings)
    if not torch.isfinite(penalty):
        raise FloatingPointError("nonfinite P4 residual regularization")
    penalty.backward()


@torch.no_grad()
def evaluate_objective(candidate: CandidateModel, adapter: ModelAdapter, edits: list[EditTrainingExample],
                       keeps: list[KeepTrainingExample], settings: ResidualOptimizationConfig) -> dict:
    edit = math.fsum(float(edit_loss(adapter, candidate.model, item)) for item in edits)
    keep = math.fsum(float(keep_loss(adapter, candidate.model, item)) for item in keeps) if settings.beta else 0.0
    penalty = float(regularization(candidate, settings))
    total = math.fsum((edit, settings.beta * keep, penalty))
    if not all(math.isfinite(value) for value in (edit, keep, penalty, total)):
        raise FloatingPointError("nonfinite P4 checkpoint objective")
    return {"edit": edit, "keep": keep, "regularization": penalty, "total": total}


@torch.no_grad()
def residual_diagnostics(candidate: CandidateModel) -> dict:
    result = {}
    for name, branch in candidate.branches.items():
        r = branch.r.double()
        r_norm, delta_norm = float(stable_frobenius_norm(r)), 0.0
        # 分块计算 ||RQ||，避免为了记录诊断常驻一个完整增量矩阵。
        for begin in range(0, branch.q.shape[1], 256):
            block_norm = float(stable_frobenius_norm(r @ branch.q[:, begin:begin + 256].double()))
            delta_norm = math.hypot(delta_norm, block_norm)
        if not math.isfinite(r_norm) or not math.isfinite(delta_norm):
            raise FloatingPointError("nonfinite residual or weight-increment norm")
        result[name] = {"residual_norm": r_norm, "increment_norm": delta_norm}
    return result


def cpu_optimizer_state(optimizer: torch.optim.Optimizer) -> dict:
    state = copy.deepcopy(optimizer.state_dict())
    for values in state["state"].values():
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                values[name] = value.cpu()
    return state


def optimizer_states_equal(saved: dict, expected: dict) -> bool:
    """Adam state must round-trip with the same groups, moments, dtype and step."""
    if set(saved) != set(expected) or saved["param_groups"] != expected["param_groups"]:
        return False
    if set(saved["state"]) != set(expected["state"]):
        return False
    for key, values in expected["state"].items():
        actual = saved["state"][key]
        if set(actual) != set(values):
            return False
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                if not isinstance(actual[name], torch.Tensor) or actual[name].dtype != value.dtype or not torch.equal(actual[name], value):
                    return False
            elif actual[name] != value:
                return False
    return True


def optimize_residuals(candidate: CandidateModel, adapter: ModelAdapter, edits: list[EditTrainingExample],
                       keeps: list[KeepTrainingExample], settings: ResidualOptimizationConfig,
                       *, on_progress: Callable[[dict], None] | None = None) -> OptimizationResult:
    settings.check()
    if set(settings.rho_by_layer) != set(candidate.branches) or not edits:
        raise ValueError("P4 requires selected layers and nonempty fixed edit supervision")
    fact_count = len({item.case_id for item in edits})
    if any(branch.r.shape[1] != fact_count for branch in candidate.branches.values()):
        raise ValueError("P4 Q rows and R columns must equal the fixed fact count N")
    if any(branch.r.dtype != getattr(torch, settings.residual_dtype) for branch in candidate.branches.values()):
        raise ValueError("candidate residual dtype differs from the frozen P4 configuration")
    if bool(settings.beta) != bool(keeps):
        raise ValueError("P4 keep-training examples must be present exactly when beta>0")
    if any(torch.count_nonzero(parameter) for parameter in candidate.parameters()):
        raise ValueError("P4 residual optimization must start from R=0")
    torch.manual_seed(settings.seed)
    optimizer = torch.optim.Adam(candidate.parameters(), lr=settings.learning_rate,
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    report = {"status": "running", "phase": "baseline_objective", "history": [], "selected_step": None,
              "fact_count": fact_count, "state_count": len(edits),
              "supervision_position_count": sum(len(item.positions) for item in edits),
              "edit_weight_sum": math.fsum(weight for item in edits for weight in item.weights),
              "keep_weight_sum": math.fsum(weight for item in keeps for weight in item.weights),
              "optimizer": {"name": "adam", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0}}
    started = time.perf_counter()
    best_residuals, best_optimizer, best_total = None, None, math.inf
    try:
        baseline = evaluate_objective(candidate, adapter, edits, keeps, settings)
        report["baseline"] = baseline
        report["history"].append({"step": 0, "loss": baseline, "layers": residual_diagnostics(candidate)})
        for step in range(1, settings.max_steps + 1):
            report.update(phase="backward", current_step=step)
            optimizer.zero_grad(set_to_none=True)
            accumulate_data_gradients(candidate, adapter, edits, keeps, settings)
            gradients = {}
            for name, branch in candidate.branches.items():
                if branch.r.grad is None or not torch.isfinite(branch.r.grad).all():
                    raise FloatingPointError(f"missing/nonfinite residual gradient at {name}")
                gradients[name] = float(stable_frobenius_norm(branch.r.grad.double()))
            if not all(math.isfinite(value) for value in gradients.values()):
                raise FloatingPointError("nonfinite residual gradient norm")
            report["phase"] = "optimizer_step"
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in candidate.parameters()):
                raise FloatingPointError("nonfinite residual after optimizer step")
            if any(not torch.isfinite(value).all() for state in optimizer.state.values()
                   for value in state.values() if isinstance(value, torch.Tensor)):
                raise FloatingPointError("nonfinite Adam state after optimizer step")
            row = {"step": step, "gradient_norms": gradients}
            if step in settings.checkpoints:
                report["phase"] = "checkpoint"
                loss = evaluate_objective(candidate, adapter, edits, keeps, settings)
                layers = residual_diagnostics(candidate)
                row.update(loss=loss, layers=layers)
                if any(info["residual_norm"] > 0 for info in layers.values()) and loss["total"] < min(
                    baseline["total"], best_total
                ):
                    best_total = loss["total"]
                    best_residuals, best_optimizer = candidate.residuals(), cpu_optimizer_state(optimizer)
                    report["selected_step"] = step
            report["history"].append(row)
            if on_progress:
                on_progress(report)
        if best_residuals is None:
            report.update(status="no_candidate", phase="complete", failure="no finite nonzero checkpoint improved J(0)")
        else:
            candidate.set_residuals(best_residuals)
            selected = evaluate_objective(candidate, adapter, edits, keeps, settings)
            if selected["total"] >= baseline["total"]:
                raise RuntimeError("reloaded selected residuals no longer improve J(0)")
            report.update(status="candidate_ready", phase="complete", selected_loss=selected,
                          selected_layers=residual_diagnostics(candidate), failure=None)
    except Exception as error:
        report.update(status="failed", failure=str(error))
        raise OptimizationError(str(error), report) from error
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        report["cuda_peak_allocated_bytes"] = {
            str(device): torch.cuda.max_memory_allocated(device)
            for device in {parameter.device for parameter in candidate.parameters()} if device.type == "cuda"
        }
    return OptimizationResult(best_residuals, best_optimizer, report)
