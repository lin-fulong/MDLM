"""Formulas (13)–(15): actual weight updates, equivalence and exact baseline recovery."""

from __future__ import annotations

import math

import torch
from torch import nn

from MDLM.candidate import CandidateModel
from MDLM.writeback import stable_frobenius_norm
from model.base import tensor_sha256


class EquivalenceError(RuntimeError):
    pass


class WeightMerge:
    """Keep baseline copies beside target weights; rebuild every trial from them."""

    @torch.no_grad()
    def __init__(self, model, factors: dict, residuals: dict, *, compute_dtype="float32", chunk_size=256):
        if not factors or set(factors) != set(residuals):
            raise ValueError("P5 Q/R target layers differ")
        if compute_dtype not in ("float32", "float64") or type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("invalid P5 merge dtype or chunk size")
        self.model, self.chunk_size = model, chunk_size
        self.dtype = getattr(torch, compute_dtype)
        dimensions = {q.shape[0] for q in factors.values() if q.ndim == 2}
        if len(dimensions) != 1 or min(dimensions) < 1:
            raise ValueError("P5 layers must share one nonempty fact dimension N")
        self.baseline, self.increments, self.parameters = {}, {}, {}
        self.versions = {name: (id(p), p._version) for name, p in model.named_parameters()}
        self.buffers = {name: (id(b), b._version) for name, b in model.named_buffers()}
        aliases = {}
        for name, parameter in model.named_parameters(remove_duplicate=False):
            if not parameter.is_meta:
                aliases.setdefault((str(parameter.device), parameter.untyped_storage().data_ptr()), []).append(name)
        for name, q in factors.items():
            module, r = model.get_submodule(name), residuals[name]
            if not isinstance(module, nn.Linear) or module.weight.is_meta:
                raise ValueError(f"P5 requires a resident linear target: {name}")
            if getattr(module, "_mdlm_candidate_owner", None) is not None:
                raise ValueError("close temporary branches before actual weight merging")
            storage = (str(module.weight.device), module.weight.untyped_storage().data_ptr())
            if len(aliases[storage]) != 1:
                raise ValueError("shared target weights require an explicit tied-parameter adapter")
            if q.ndim != 2 or r.shape != (module.out_features, q.shape[0]) or q.shape[1] != module.in_features:
                raise ValueError(f"P5 Q/R dimensions differ at {name}")
            if q.dtype != r.dtype or q.dtype not in (torch.float32, torch.float64) or (
                q.dtype == torch.float64 and self.dtype != torch.float64
            ):
                raise ValueError("P5 merge precision must preserve saved Q/R precision")
            if module.weight.dtype == torch.float64 and self.dtype != torch.float64:
                raise ValueError("P5 merge arithmetic must preserve baseline precision")
            if not torch.isfinite(q).all() or not torch.isfinite(r).all():
                raise FloatingPointError("nonfinite P5 factors")
            self.parameters[name] = module.weight
            self.baseline[name] = module.weight.detach().clone()
            if not torch.isfinite(self.baseline[name]).all():
                raise FloatingPointError(f"nonfinite baseline weight at {name}")
            device = module.weight.device
            q_device, r_device = q.to(device=device, dtype=self.dtype), r.to(device=device, dtype=self.dtype)
            delta = torch.empty(module.weight.shape, dtype=self.dtype, device=device)
            for begin in range(0, q.shape[1], chunk_size):
                # 公式（13）：真正写回的是 RQ；按列分块限制中间内存。
                delta[:, begin:begin + chunk_size] = r_device @ q_device[:, begin:begin + chunk_size]
            if not torch.isfinite(delta).all():
                raise FloatingPointError(f"nonfinite actual increment at {name}")
            self.increments[name] = delta
        self._remember_target_writes()

    def _remember_target_writes(self) -> None:
        # 只在受控写入后更新期望值，前向期间的任何目标修改都必须被发现。
        for name, weight in self.parameters.items():
            self.versions[name + ".weight"] = (id(weight), weight._version)
        self.expected_weight_hashes = self.weight_hashes()

    def check_integrity(self) -> None:
        current = dict(self.model.named_parameters())
        if set(current) != set(self.versions) or any(
            (id(current[name]), current[name]._version) != (identity, version)
            for name, (identity, version) in self.versions.items()
        ) or {name: (id(b), b._version) for name, b in self.model.named_buffers()} != self.buffers or (
            self.weight_hashes() != self.expected_weight_hashes
        ):
            raise RuntimeError("model parameter/buffer integrity changed outside the P5 weight whitelist")

    @torch.no_grad()
    def apply(self, alpha: float) -> None:
        self.check_integrity()
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("P5 common alpha must be finite and positive")
        for name, weight in self.parameters.items():
            for begin in range(0, weight.shape[1], self.chunk_size):
                columns = slice(begin, begin + self.chunk_size)
                # 公式（15）：基准加一次公共 alpha 增量，绝不沿用上个试步。
                value = (self.baseline[name][:, columns].to(self.dtype)
                         + alpha * self.increments[name][:, columns]).to(dtype=weight.dtype, device=weight.device)
                if not torch.isfinite(value).all():
                    raise FloatingPointError(f"merged weight overflows its model dtype at {name}")
                weight[:, columns].copy_(value)
        self._remember_target_writes()

    @torch.no_grad()
    def restore(self) -> None:
        for name, weight in self.parameters.items():
            weight.copy_(self.baseline[name].to(weight.device))
        self._remember_target_writes()
        self.check_integrity()
        if any(not torch.equal(weight.detach(), self.baseline[name]) for name, weight in self.parameters.items()):
            raise RuntimeError("P5 baseline recovery failed")

    def weight_hashes(self) -> dict:
        return {name: tensor_sha256(weight) for name, weight in self.parameters.items()}

    @torch.no_grad()
    def diagnostics(self, factors: dict, residuals: dict, keys: dict) -> dict:
        result = {}
        for name, k in keys.items():
            device = self.parameters[name].device
            q, r, k = (value.to(device=device, dtype=torch.float64) for value in (factors[name], residuals[name], k))
            # 公式（14）：软拟合误差允许非零；V* 仅是参考键上的辅助量。
            error = r @ (q @ k - torch.eye(q.shape[0], dtype=torch.float64, device=device))
            result[name] = {"increment_norm": float(stable_frobenius_norm(self.increments[name].double())),
                            "soft_fit_error_norm": float(stable_frobenius_norm(error)),
                            "auxiliary_v_norm": float(stable_frobenius_norm(self.baseline[name].double() @ k + r))}
            if not all(math.isfinite(value) for value in result[name].values()):
                raise FloatingPointError("nonfinite reference-key diagnostics")
        return result


@torch.no_grad()
def capture_outputs(adapter, model, model_input, positions, names):
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
                           inputs: list, alpha: float, *, atol: float, rtol: float) -> dict:
    """Compare the same inputs/alpha, then leave only actual merged weights active."""
    merger.restore()
    expected = []
    scaled = {name: alpha * value for name, value in residuals.items()}
    if any(not torch.isfinite(value).all() for value in scaled.values()):
        raise FloatingPointError("scaled factor candidate is nonfinite")
    with CandidateModel(merger.model, factors, dtype=str(next(iter(factors.values())).dtype).removeprefix("torch.")) as candidate:
        candidate.set_residuals(scaled)
        for model_input, positions in inputs:
            expected.append(capture_outputs(adapter, merger.model, model_input, positions, factors))
    merger.apply(alpha)
    maxima = {name: 0.0 for name in factors}
    logits_max = 0.0
    for (model_input, positions), (expected_logits, expected_outputs) in zip(inputs, expected):
        logits, outputs = capture_outputs(adapter, merger.model, model_input, positions, factors)
        if logits.shape != expected_logits.shape or not torch.allclose(logits, expected_logits, atol=atol, rtol=rtol):
            raise EquivalenceError("actual merged logits differ from the scaled factor candidate")
        logits_max = max(logits_max, float((logits.double() - expected_logits.double()).abs().max()))
        for name in factors:
            if len(outputs[name]) != len(expected_outputs[name]):
                raise EquivalenceError(f"target invocation count differs at {name}")
            for actual, reference in zip(outputs[name], expected_outputs[name]):
                if actual.shape != reference.shape or not torch.allclose(actual, reference, atol=atol, rtol=rtol):
                    raise EquivalenceError(f"actual merged target outputs differ at {name}")
                maxima[name] = max(maxima[name], float((actual.double() - reference.double()).abs().max()))
    merger.check_integrity()
    return {"passed": True, "input_count": len(inputs), "logits_max_abs": logits_max, "target_max_abs": maxima}
