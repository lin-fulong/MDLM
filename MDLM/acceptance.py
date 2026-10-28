"""Formulas (16)–(18): independent development gates and ordered common trials."""

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
    edit_protocol: str = "full_mask_mean_fact_probe_token_v1"

    def check(self) -> None:
        if not self.alphas or any(isinstance(a, bool) or not isinstance(a, (int, float))
                                 or not math.isfinite(a) or a <= 0 for a in self.alphas):
            raise ValueError("P5 requires a finite ordered sequence of positive common alphas")
        for name in ("delta", "tau_g", "tau_u", "epsilon", "equivalence_atol", "equivalence_rtol"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"P5 {name} must be finite and nonnegative")
        if self.delta == 0 or self.epsilon == 0:
            raise ValueError("P5 delta and epsilon must be positive")
        if self.compute_dtype not in ("float32", "float64") or type(self.chunk_size) is not int or self.chunk_size < 1:
            raise ValueError("P5 requires float32/float64 merge arithmetic and positive chunk_size")
        if self.edit_protocol != "full_mask_mean_fact_probe_token_v1":
            raise ValueError("unsupported P5 development protocol")


class AcceptanceError(RuntimeError):
    def __init__(self, message: str, report: dict):
        super().__init__(message)
        self.report = report


def development_edits(cases: list[dict], selections: tuple, adapter) -> list[EditTrainingExample]:
    """Trusted original case schema; one full-mask readout per explicitly selected probe."""
    by_id = {case["case_id"]: case for case in cases}
    counts = Counter(item.case_id for item in selections)
    examples = []
    for item in selections:
        model_input, target = adapter.prepare_development_input(by_id[item.case_id], item.probe_id)
        positions = model_input.mask_positions
        # 冻结开发协议：事实均分、同事实探针均分、答案 token 均分。
        weight = 1 / (len(counts) * counts[item.case_id] * len(positions))
        examples.append(EditTrainingExample(item.case_id, model_input, positions, target.token_ids,
                                            (weight,) * len(positions)))
    return examples


@torch.no_grad()
def cumulative_change(merger: WeightMerge, originals: dict, epsilon: float) -> float:
    """Formula (17): actual current minus original weights, never a sum of step norms."""
    if set(originals) != set(merger.parameters):
        raise ValueError("original W_0 layer selection differs from actual merged model")
    scale = epsilon
    for name, weight in merger.parameters.items():
        original = originals[name]
        if original.shape != weight.shape or not torch.isfinite(original).all():
            raise ValueError(f"original W_0 shape/finiteness differs at {name}")
        if not torch.isfinite(weight).all():
            raise FloatingPointError("nonfinite actual weights in cumulative change")
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
    if not edits or not keeps:
        raise ValueError("P5 requires both development edit and preservation readouts even when training beta=0")
    if logits is not None and len(logits) != len(edits) + len(keeps):
        raise ValueError("P5 cached logits do not cover all development readouts")
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
    limits = {"f_dev": baseline_edit - settings.delta, "g_dev": settings.tau_g, "u": settings.tau_u}
    checks = {name: math.isfinite(metrics[name]) and metrics[name] <= limit for name, limit in limits.items()}
    return {"passed": all(checks.values()), "checks": checks, "limits": limits,
            "reasons": [name + " exceeds its limit" for name, passed in checks.items() if not passed]}


def select_merged_candidate(merger: WeightMerge, adapter, factors: dict, residuals: dict,
                            edits: list, keeps: list, originals: dict, settings: AcceptanceConfig) -> dict:
    settings.check()
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
                # check_integrity 已将当前目标层摘要与 apply() 后的冻结摘要逐项比较。
                merger.check_integrity()
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
    except Exception as error:
        report.update(status="failed", baseline_restored=False, failure=str(error))
        try:
            merger.restore()
            report["baseline_restored"] = True
        except Exception as recovery_error:
            report["recovery_failure"] = str(recovery_error)
        raise AcceptanceError(str(error), report) from error
