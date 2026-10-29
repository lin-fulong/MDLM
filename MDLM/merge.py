"""公式（13）～（15）：形成实际增量 RQ，按公共 α 写回，并核对因子前向与合并前向。"""

from __future__ import annotations

import torch

from MDLM.candidate import CandidateModel


class EquivalenceError(RuntimeError):
    """同一输入、同一 α 下的因子前向与实际合并前向超出预定容差。"""


class WeightMerge:
    """保存本轮基准 W_b 和固定增量 ΔW，每次试步都从 W_b 重建。

    公式（14）的 R*(QK−I) 是参考键上的软拟合误差，不是合并增量或接受门槛。
    """

    @torch.no_grad()
    def __init__(self, model, factors: dict, residuals: dict, *, compute_dtype="float32", chunk_size=256):
        """公式（13）：ΔW = R*Q；沿用分块乘法和原合并精度。"""
        self.model, self.chunk_size = model, chunk_size
        self.dtype = getattr(torch, compute_dtype)
        self.baseline, self.increments, self.parameters = {}, {}, {}
        for name, q in factors.items():
            weight, r = model.get_submodule(name).weight, residuals[name]
            self.parameters[name] = weight
            self.baseline[name] = weight.detach().clone()
            device = weight.device
            q_device, r_device = q.to(device=device, dtype=self.dtype), r.to(device=device, dtype=self.dtype)
            delta = torch.empty(weight.shape, dtype=self.dtype, device=device)
            for begin in range(0, q.shape[1], chunk_size):
                # 公式（13）：真正写回的是 RQ；按列分块限制中间内存。
                delta[:, begin:begin + chunk_size] = r_device @ q_device[:, begin:begin + chunk_size]
            if not torch.isfinite(delta).all():
                raise FloatingPointError(f"nonfinite actual increment at {name}")
            self.increments[name] = delta

    @torch.no_grad()
    def apply(self, alpha: float) -> None:
        """公式（15）：全部编辑层共用 α，仅覆盖目标权重，偏置保持原值。"""
        for name, weight in self.parameters.items():
            for begin in range(0, weight.shape[1], self.chunk_size):
                columns = slice(begin, begin + self.chunk_size)
                # 公式（15）：基准加一次公共 alpha 增量，绝不沿用上个试步。
                value = (self.baseline[name][:, columns].to(self.dtype)
                         + alpha * self.increments[name][:, columns]).to(dtype=weight.dtype, device=weight.device)
                if not torch.isfinite(value).all():
                    raise FloatingPointError(f"merged weight overflows its model dtype at {name}")
                weight[:, columns].copy_(value)

    @torch.no_grad()
    def restore(self) -> None:
        """直接复制保存的 W_b，拒绝或异常时精确恢复本轮基准。"""
        for name, weight in self.parameters.items():
            weight.copy_(self.baseline[name].to(weight.device))


@torch.no_grad()
def capture_outputs(adapter, model, model_input, positions, names):
    """记录目标层每次调用的全位置输出及指定位置 logits，退出时移除捕获 hook。"""
    outputs, handles = {name: [] for name in names}, []
    try:
        for name in names:
            handles.append(model.get_submodule(name).register_forward_hook(
                lambda module, inputs, output, name=name: outputs[name].append(output.detach().clone())))
        logits = adapter.forward_logits(model, model_input, positions).detach().clone()
        if any(not values for values in outputs.values()):
            raise RuntimeError("a P5 target module was not called")
        return logits, outputs
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def merge_with_equivalence(merger: WeightMerge, adapter, factors: dict, residuals: dict,
                           inputs: list, alpha: float, *, atol: float, rtol: float,
                           merged_logits: list[torch.Tensor] | None = None) -> dict:
    """公式（11）（15）：比较 W_b k+b+αR(Qk) 与 (W_b+αRQ)k+b。"""
    merger.restore()
    expected = []
    scaled = {name: alpha * value for name, value in residuals.items()}
    if any(not torch.isfinite(value).all() for value in scaled.values()):
        raise FloatingPointError("scaled factor candidate is nonfinite")
    dtype = str(next(iter(factors.values())).dtype).removeprefix("torch.")
    with CandidateModel(merger.model, factors, dtype=dtype) as candidate:
        candidate.set_residuals(scaled)
        for model_input, positions in inputs:
            expected.append(capture_outputs(adapter, merger.model, model_input, positions, factors))
    merger.apply(alpha)  # 临时分支已关闭，实际权重只加入一次 αRQ。
    maxima = {name: 0.0 for name in factors}
    logits_max = 0.0
    for (model_input, positions), (expected_logits, expected_outputs) in zip(inputs, expected):
        logits, outputs = capture_outputs(adapter, merger.model, model_input, positions, factors)
        if logits.shape != expected_logits.shape or not torch.allclose(logits, expected_logits, atol=atol, rtol=rtol):
            raise EquivalenceError("actual merged logits differ from the scaled factor candidate")
        if merged_logits is not None:
            merged_logits.append(logits)
        logits_max = max(logits_max, float((logits.double() - expected_logits.double()).abs().max()))
        for name in factors:
            if len(outputs[name]) != len(expected_outputs[name]):
                raise EquivalenceError(f"target invocation count differs at {name}")
            for actual, reference in zip(outputs[name], expected_outputs[name]):
                if actual.shape != reference.shape or not torch.allclose(actual, reference, atol=atol, rtol=rtol):
                    raise EquivalenceError(f"actual merged target outputs differ at {name}")
                maxima[name] = max(maxima[name], float((actual.double() - reference.double()).abs().max()))
    return {"passed": True, "input_count": len(inputs), "logits_max_abs": logits_max, "target_max_abs": maxima}
