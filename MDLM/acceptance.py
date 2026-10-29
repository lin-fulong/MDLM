"""公式（16）～（18）：按预定公共 α 试步，以三个独立开发条件决定接受或回退。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

import torch

from MDLM.merge import EquivalenceError, WeightMerge, merge_with_equivalence
from MDLM.optimize import edit_loss, edit_loss_from_logits, keep_loss, keep_loss_from_logits
from MDLM.states import EditTrainingExample
from MDLM.writeback import stable_frobenius_norm


@dataclass(frozen=True)
class AcceptanceConfig:
    """公式（15）～（18）的公共 α 顺序、开发门槛与合并精度。"""

    alphas: tuple[float, ...] = (1.0, 0.5, 0.25)
    delta: float = 0.001
    tau_g: float = 0.05
    tau_u: float = 0.05
    epsilon: float = 1e-12
    compute_dtype: str = "float32"
    equivalence_atol: float = 0.125
    equivalence_rtol: float = 0.02
    chunk_size: int = 256


def development_edits(cases: list[dict], selections: tuple, adapter) -> list[EditTrainingExample]:
    """构造固定开发编辑输入，沿用事实 → 探针 → 答案 token 的均分协议。"""
    by_id = {case["case_id"]: case for case in cases}
    counts = Counter(item.case_id for item in selections)
    examples = []
    for item in selections:
        model_input, target = adapter.prepare_development_input(by_id[item.case_id], item.probe_id)
        positions = model_input.mask_positions
        # F_dev 沿用公式（2）（3）的分级均值形式，此处用开发探针替代训练状态。
        weight = 1 / (len(counts) * counts[item.case_id] * len(positions))
        examples.append(EditTrainingExample(item.case_id, model_input, positions, target.token_ids,
                                            (weight,) * len(positions)))
    return examples


@torch.no_grad()
def cumulative_change(merger: WeightMerge, originals: dict, epsilon: float) -> float:
    """公式（17）：当前实际存储权重相对最初 W_0 的整体净变化比例。"""
    scale = epsilon
    for name, weight in merger.parameters.items():
        original = originals[name]
        scale = max(scale, float(original.abs().max()), float(weight.detach().abs().max()))
    numerator, denominator = 0.0, 0.0
    for name, weight in merger.parameters.items():
        original = originals[name]
        for begin in range(0, weight.shape[1], merger.chunk_size):
            columns = slice(begin, begin + merger.chunk_size)
            # 公式（17）分子分母用同一尺度；先缩放，避免差值/整体范数上溢。
            w0 = original[:, columns].to(device=weight.device, dtype=torch.float64) / scale
            difference = weight[:, columns].detach().double() / scale - w0
            numerator = math.hypot(numerator, float(stable_frobenius_norm(difference)))
            denominator = math.hypot(denominator, float(stable_frobenius_norm(w0)))
    divisor = max(denominator, epsilon / scale)
    value = numerator / divisor if divisor else math.inf
    if not math.isfinite(value):
        raise FloatingPointError("nonfinite cumulative parameter change")
    return value


@torch.no_grad()
def development_metrics(merger, adapter, edits, keeps, originals, epsilon, logits=None) -> dict:
    """计算 F_dev、完整词表 G_dev 和累计 U，可复用合并一致性前向的 logits。"""
    if not edits or not keeps:
        raise ValueError("P5 requires both development edit and preservation readouts even when training beta=0")
    edit = math.fsum(float(edit_loss(adapter, merger.model, item) if logits is None
                          else edit_loss_from_logits(logits[index], item)) for index, item in enumerate(edits))
    # 公式（16）：复用完整词表 KL(theta_0 || theta_alpha)，开发保持始终执行。
    keep = math.fsum(float(keep_loss(adapter, merger.model, item) if logits is None
                          else keep_loss_from_logits(logits[len(edits) + index], item))
                     for index, item in enumerate(keeps))
    change = cumulative_change(merger, originals, epsilon)
    if not all(math.isfinite(value) for value in (edit, keep, change)):
        raise FloatingPointError("nonfinite P5 development metrics")
    return {"f_dev": edit, "g_dev": keep, "u": change}


def gate_decision(metrics: dict, baseline_edit: float, settings: AcceptanceConfig) -> dict:
    """公式（18）：F_dev ≤ F_dev(W_b)−δ、G_dev ≤ τ_G、U ≤ τ_U 同时成立。"""
    limits = {"f_dev": baseline_edit - settings.delta, "g_dev": settings.tau_g, "u": settings.tau_u}
    checks = {name: math.isfinite(metrics[name]) and metrics[name] <= limit for name, limit in limits.items()}
    return {"passed": all(checks.values()), "checks": checks, "limits": limits,
            "reasons": [name + " exceeds its limit" for name, passed in checks.items() if not passed]}


def select_merged_candidate(merger: WeightMerge, adapter, factors: dict, residuals: dict,
                            edits: list, keeps: list, originals: dict, settings: AcceptanceConfig) -> dict:
    """依序接受首个通过等价性与三个门槛的 α；全部拒绝或异常时恢复 W_b。"""
    report = {"status": "running", "selected_alpha": None, "trials": [], "baseline_restored": False}
    try:
        merger.restore()
        report["baseline"] = development_metrics(merger, adapter, edits, keeps, originals, settings.epsilon)
        inputs = [(item.model_input, item.positions) for item in (*edits, *keeps)]
        for alpha in settings.alphas:
            row = {"alpha": alpha, "accepted": False}
            report["trials"].append(row)
            try:
                # 先证明同 alpha 的因子/合并前向一致，再对实际合并模型验收。
                merged_logits = []
                row["equivalence"] = merge_with_equivalence(merger, adapter, factors, residuals, inputs, alpha,
                    atol=settings.equivalence_atol, rtol=settings.equivalence_rtol, merged_logits=merged_logits)
                row["metrics"] = development_metrics(merger, adapter, edits, keeps, originals, settings.epsilon,
                                                     merged_logits)
                row["gate"] = gate_decision(row["metrics"], report["baseline"]["f_dev"], settings)
                row["accepted"] = row["gate"]["passed"]
            except (EquivalenceError, FloatingPointError) as error:
                row["reasons"] = [str(error)]
            if row["accepted"]:
                report.update(status="accepted", selected_alpha=alpha, failure=None)
                return report
            merger.restore()
            row["baseline_restored"] = True
        report.update(status="rejected", baseline_restored=True, failure=None)
        return report
    except Exception:
        merger.restore()
        raise
