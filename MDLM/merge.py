"""公式（13）～（18）：合并权重、核对前向一致性，并按开发门槛接受或回退。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import sys

import torch

from MDLM.optimize import CandidateModel, edit_loss, edit_loss_from_logits, keep_loss, keep_loss_from_logits
from MDLM.states import EditTrainingExample, prepare_development_input
from MDLM.writeback import stable_frobenius_norm
from util.nethook import capture_module_outputs


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

    def __post_init__(self):
        if (not self.alphas or any(not math.isfinite(a) or a <= 0 for a in self.alphas)
                or not math.isfinite(self.epsilon) or self.epsilon <= 0
                or any(not math.isfinite(x) or x < 0 for x in (
                    self.delta, self.tau_g, self.tau_u, self.equivalence_atol, self.equivalence_rtol))):
            sys.exit("P5 的公共步长、门槛或等价容差无效")


class WeightMerge:
    """保存本轮基准 W_b 和固定增量 ΔW，每次试步都从 W_b 重建。

    公式（14）的 R*(QK−I) 是参考键上的软拟合误差，不是合并增量或接受门槛。
    """

    @torch.no_grad()
    def __init__(self, model, factors: dict, residuals: dict, *, compute_dtype="float32", chunk_size=256):
        """公式（13）：ΔW = R*Q；沿用分块乘法和原合并精度。"""
        self.model, self.chunk_size = model, chunk_size
        self.dtype = getattr(torch, compute_dtype)
        if type(chunk_size) is not int or chunk_size < 1:
            sys.exit("合并分块大小必须为正整数")
        self.baseline, self.increments, self.parameters = {}, {}, {}
        for name, q in factors.items():
            weight, r = model.get_submodule(name).weight, residuals[name]
            if q.ndim != 2 or q.shape[1] != weight.shape[1] or r.shape != (weight.shape[0], q.shape[0]):
                sys.exit("RQ 的形状必须与目标权重完全一致")
            self.parameters[name] = weight
            self.baseline[name] = weight.detach().clone()
            device = weight.device
            q_device, r_device = q.to(device=device, dtype=self.dtype), r.to(device=device, dtype=self.dtype)
            delta = torch.empty(weight.shape, dtype=self.dtype, device=device)
            for begin in range(0, q.shape[1], chunk_size):
                # 公式（13）：真正写回的是 RQ；按列分块限制中间内存。
                delta[:, begin:begin + chunk_size] = r_device @ q_device[:, begin:begin + chunk_size]
            self.increments[name] = delta

    @torch.no_grad()
    def apply(self, alpha: float) -> str | None:
        """公式（15）：全部编辑层共用 α；数值溢出返回本试步的拒绝原因。"""
        for name, weight in self.parameters.items():
            for begin in range(0, weight.shape[1], self.chunk_size):
                columns = slice(begin, begin + self.chunk_size)
                # 公式（15）：基准加一次公共 alpha 增量，绝不沿用上个试步。
                value = (self.baseline[name][:, columns].to(self.dtype)
                         + alpha * self.increments[name][:, columns]).to(dtype=weight.dtype, device=weight.device)
                if not torch.isfinite(value).all():
                    return f"merged weight overflows its model dtype at {name}"
                weight[:, columns].copy_(value)

    @torch.no_grad()
    def restore(self) -> None:
        """直接复制保存的 W_b，拒绝或异常时精确恢复本轮基准。"""
        for name, weight in self.parameters.items():
            weight.copy_(self.baseline[name].to(weight.device))


@torch.no_grad()
def capture_outputs(adapter, model, model_input, positions, names):
    """记录因子/实际合并模型的读出；通用 hook 只负责逐次复制目标层输出。"""
    with capture_module_outputs(model, names) as outputs:
        logits = adapter.forward_logits(model, model_input, positions).detach().clone()
        if not torch.isfinite(logits).all():
            return None
        if any(not values for values in outputs.values()):
            sys.exit("P5 的目标模块没有实际参与前向")
        return logits, outputs


@torch.no_grad()
def merge_with_equivalence(merger: WeightMerge, adapter, factors: dict, residuals: dict,
                           inputs: list, alpha: float, *, atol: float, rtol: float,
                           merged_logits: list[torch.Tensor] | None = None) -> dict:
    """公式（11）（15）：比较 W_b k+b+αR(Qk) 与 (W_b+αRQ)k+b。"""
    merger.restore()
    expected = []
    scaled = {name: alpha * value for name, value in residuals.items()}
    if any(not torch.isfinite(value).all() for value in scaled.values()):
        return {"passed": False, "reason": "scaled factor candidate is nonfinite"}
    dtype = str(next(iter(factors.values())).dtype).removeprefix("torch.")
    with CandidateModel(merger.model, factors, dtype=dtype) as candidate:
        candidate.set_residuals(scaled)
        for model_input, positions in inputs:
            captured = capture_outputs(adapter, merger.model, model_input, positions, factors)
            if captured is None:
                return {"passed": False, "reason": "nonfinite P4 readout logits"}
            expected.append(captured)
    # 临时分支已关闭，实际权重只加入一次 αRQ；溢出沿用原试步拒绝条件。
    failure = merger.apply(alpha)
    if failure is not None:
        return {"passed": False, "reason": failure}
    maxima = {name: 0.0 for name in factors}
    logits_max = 0.0
    for (model_input, positions), (expected_logits, expected_outputs) in zip(inputs, expected):
        captured = capture_outputs(adapter, merger.model, model_input, positions, factors)
        if captured is None:
            return {"passed": False, "reason": "nonfinite P4 readout logits"}
        logits, outputs = captured
        if logits.shape != expected_logits.shape or not torch.allclose(logits, expected_logits, atol=atol, rtol=rtol):
            return {"passed": False, "reason": "actual merged logits differ from the scaled factor candidate"}
        if merged_logits is not None:
            merged_logits.append(logits)
        logits_max = max(logits_max, float((logits.double() - expected_logits.double()).abs().max()))
        for name in factors:
            if len(outputs[name]) != len(expected_outputs[name]):
                return {"passed": False, "reason": f"target invocation count differs at {name}"}
            for actual, reference in zip(outputs[name], expected_outputs[name]):
                if actual.shape != reference.shape or not torch.allclose(actual, reference, atol=atol, rtol=rtol):
                    return {"passed": False, "reason": f"actual merged target outputs differ at {name}"}
                maxima[name] = max(maxima[name], float((actual.double() - reference.double()).abs().max()))
    return {"passed": True, "input_count": len(inputs), "logits_max_abs": logits_max, "target_max_abs": maxima}


def development_edits(cases: list[dict], selections: tuple, adapter) -> list[EditTrainingExample]:
    """构造固定开发编辑输入，沿用事实 → 探针 → 答案 token 的均分协议。"""
    by_id = {case["case_id"]: case for case in cases}
    counts = Counter(item.case_id for item in selections)
    examples = []
    for item in selections:
        model_input, target = prepare_development_input(adapter, by_id[item.case_id], item.probe_id)
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
    return value


@torch.no_grad()
def development_metrics(merger, adapter, edits, keeps, originals, epsilon, logits=None) -> dict:
    """计算 F_dev、完整词表 G_dev 和累计 U，可复用合并一致性前向的 logits。"""
    if not edits or not keeps:
        sys.exit("P5 必须同时提供非空开发编辑集和开发保持集，beta=0 也不能跳过保持验收")
    edit = math.fsum(float(edit_loss(adapter, merger.model, item) if logits is None
                          else edit_loss_from_logits(logits[index], item)) for index, item in enumerate(edits))
    # 公式（16）：复用完整词表 KL(theta_0 || theta_alpha)，开发保持始终执行。
    keep = math.fsum(float(keep_loss(adapter, merger.model, item) if logits is None
                          else keep_loss_from_logits(logits[len(edits) + index], item))
                     for index, item in enumerate(keeps))
    change = cumulative_change(merger, originals, epsilon)
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
        if not all(math.isfinite(value) for value in report["baseline"].values()):
            sys.exit("P5 基准指标非有限，无法进行有效验收")
        inputs = [(item.model_input, item.positions) for item in (*edits, *keeps)]
        for alpha in settings.alphas:
            row = {"alpha": alpha, "accepted": False}
            report["trials"].append(row)
            try:
                # 先证明同 alpha 的因子/合并前向一致，再对实际合并模型验收。
                merged_logits = []
                equivalence = merge_with_equivalence(merger, adapter, factors, residuals, inputs, alpha,
                    atol=settings.equivalence_atol, rtol=settings.equivalence_rtol, merged_logits=merged_logits)
                if not equivalence["passed"]:
                    row["reasons"] = [equivalence["reason"]]
                else:
                    row["equivalence"] = equivalence
                    metrics = development_metrics(merger, adapter, edits, keeps, originals, settings.epsilon,
                                                  merged_logits)
                    # 仍先检查累计变化，再检查全部开发指标，保持原拒绝原因和报告字段。
                    if not math.isfinite(metrics["u"]):
                        row["reasons"] = ["nonfinite cumulative parameter change"]
                    elif not all(math.isfinite(value) for value in metrics.values()):
                        row["reasons"] = ["nonfinite P5 development metrics"]
                    else:
                        row["metrics"] = metrics
                        row["gate"] = gate_decision(metrics, report["baseline"]["f_dev"], settings)
                        row["accepted"] = row["gate"]["passed"]
            except FloatingPointError as error:
                row["reasons"] = [str(error)]
            if row["accepted"]:
                report.update(status="accepted", selected_alpha=alpha, failure=None)
                return report
            merger.restore()
            row["baseline_restored"] = True
        report.update(status="rejected", baseline_restored=True, failure=None)
        return report
    finally:
        if report["status"] == "running":
            merger.restore()
