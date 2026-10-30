"""公式（4）～（9）：提取参考键 K，逐层累计保持矩阵 C，并求固定写回因子 Q。"""

from dataclasses import dataclass
import math
import sys

import torch

from MDLM.states import capture_edit, keep_input_from_record
from util.nethook import tensor_sha256


@dataclass(frozen=True)
class WritebackConfig:
    """逐层 λ > 0、求解精度、Q 存储精度和计算设备。"""

    ridge_by_layer: dict[str, float]
    compute_dtype: str = "float64"
    storage_dtype: str = "float32"
    device: str = "cpu"


def extract_keys(model, adapter, trajectories, *, storage_dtype="float32",
                 replay_atol=1e-2, replay_rtol=1e-3) -> dict:
    """公式（4）：K_ell = [k_1,ell, ..., k_N,ell]，形状为 [输入维度, 事实数]。"""
    descriptions = adapter.inspect_target_modules(model)
    dtype = getattr(torch, storage_dtype)
    columns = {name: [] for name in descriptions}
    for trajectory in trajectories:
        # 每事实只取全 mask 初态；适配器捕获主体末 token 处 FFN 输出线性层的输入。
        initial = trajectory.trace.pre_forward_inputs[0]
        captured = capture_edit(model, adapter, initial)
        repeated = capture_edit(model, adapter, initial)
        if any(tensor_sha256(value.mask_logits) != trajectory.trace.mask_logit_sha256[0]
               for value in (captured, repeated)):
            sys.exit("K 取键前向与采样全 mask 初态不一致")
        for name in descriptions:
            key = captured.module_inputs[name]
            stored = key.detach().to(device="cpu", dtype=dtype)
            if (key.shape != (descriptions[name]["in_features"],) or not torch.isfinite(key).all()
                    or not torch.allclose(key, repeated.module_inputs[name], atol=replay_atol, rtol=replay_rtol)
                    or not torch.allclose(key.double().cpu(), stored.double(), atol=replay_atol, rtol=replay_rtol)):
                sys.exit(f"{name}: K 形状、有限性或重放精度不符合约定")
            columns[name].append(stored)
    # 沿第 1 维堆叠，保持与 trajectories 完全相同的事实列序，不作状态平均或归一化。
    return {name: torch.stack(values, dim=1).contiguous() for name, values in columns.items()}


class SecondMomentAccumulator:
    """逐批累计一层的 C，不保存激活或生成诊断报告。"""

    def __init__(self, width: int, *, accumulation_dtype: str = "float64", device: str = "cpu"):
        self.matrix = torch.zeros((width, width), dtype=getattr(torch, accumulation_dtype), device=device)

    @torch.no_grad()
    def add(self, activations: torch.Tensor, weights: list[float]) -> None:
        """activations 每行是 u_mᵀ；weights 是对应的非负统计权重。"""
        values = activations.detach().to(device=self.matrix.device, dtype=torch.float64)
        weights = torch.as_tensor(weights, dtype=torch.float64, device=self.matrix.device)
        if (weights.shape != (len(values),) or not torch.isfinite(weights).all()
                or (weights < 0).any() or not torch.isfinite(values).all()):
            sys.exit("C 的激活和非负位置权重必须有限且逐项对应")
        positive = weights > 0

        # 公式（5）：v_m = sqrt(ω_m)u_m，C += VᵀV；不去均值、不归一化。
        weighted = (values[positive] * weights[positive].sqrt()[:, None]).to(self.matrix.dtype)
        self.matrix.addmm_(weighted.T, weighted)

        # 沿用原有分块计算：上三角复制到下三角，消除 BLAS 归约产生的非对称舍入。
        if len(weighted):
            width = self.matrix.shape[0]
            for start in range(0, width, 256):
                end = min(width, start + 256)
                if start:
                    self.matrix[start:end, :start].copy_(self.matrix[:start, start:end].T)
                block = self.matrix[start:end, start:end]
                lower = torch.tril_indices(end - start, end - start, offset=-1, device=self.matrix.device)
                block[lower[0], lower[1]] = block[lower[1], lower[0]]


def equal_length_groups(records, *, enabled=False):
    """按原顺序对连续等长输入分批：至多 4 条、256 token 预算，超长单条保留。"""
    start = 0
    while start < len(records):
        length = len(records[start][1].input_ids)
        limit = min(4, max(1, 256 // max(length, 1))) if enabled else 1
        end = start + 1
        while end < len(records) and end - start < limit and len(records[end][1].input_ids) == length:
            end += 1
        yield records[start:end]
        start = end


def keep_matrices(model, adapter, rows, *, accumulation_dtype="float64", storage_dtype="float32",
                  device="cpu", batch_keep=False):
    """公式（5）：每层返回 (模块名, C)，每批保持输入只前向一次。"""
    records = [(row, keep_input_from_record(row["input"]), tuple(row["capture_positions"])) for row in rows]
    for name in adapter.target_module_names():
        accumulator = SecondMomentAccumulator(
            model.get_submodule(name).in_features, accumulation_dtype=accumulation_dtype, device=device)
        for group in equal_length_groups(records, enabled=batch_keep):
            requests = tuple((model_input, positions) for _, model_input, positions in group)
            captures = adapter.forward_with_capture(model, requests, module_names=(name,))
            activations = torch.cat([captured.module_inputs[name] for captured in captures])
            weights = [weight for row, _, _ in group for weight in row["weights_by_layer"][name]]
            accumulator.add(activations, weights)
        matrix = accumulator.matrix.to(dtype=getattr(torch, storage_dtype))
        yield name, matrix
        del accumulator, matrix


def stable_frobenius_norm(value: torch.Tensor) -> torch.Tensor:
    """缩放后计算 Frobenius 范数，供残差优化、合并和接受阶段复用。"""
    scale = value.abs().max()
    if bool(scale == 0):
        return scale
    return scale * (value / scale).norm()


@torch.no_grad()
def solve_writeback_factor(keys: torch.Tensor, keep: torch.Tensor, ridge: float,
                           settings: WritebackConfig) -> torch.Tensor:
    """返回固定的 CPU Q，形状为 [事实数, 输入维度]；不修改 K、C 或模型权重。

    公式（6）的软拟合目标经公式（7）得到 DH = RKᵀ，因此只需预先求 Q。
    """
    values = keys.detach().to(device=settings.device, dtype=getattr(torch, settings.compute_dtype))
    if not math.isfinite(ridge) or ridge <= 0:
        sys.exit("写回 ridge 必须为有限正数")
    if (values.ndim != 2 or keep.shape != (values.shape[0], values.shape[0])
            or not torch.isfinite(values).all() or not torch.isfinite(keep).all()):
        sys.exit("K/C 的形状或有限性不符合写回方程")

    # 公式（7）（8）：H = C + KKᵀ + λI；C 半正定且 λ > 0 时，H 正定。
    system = keep.detach().to(device=settings.device, dtype=values.dtype, copy=True)
    system.addmm_(values, values.T)
    system.diagonal().add_(ridge)

    # 公式（8）（9）：Cholesky 分解 H = LLᵀ，再解 HX = K，不显式求逆。
    factor = torch.linalg.cholesky(system)
    del system
    solution = torch.cholesky_solve(values, factor)
    del factor

    # 公式（9）：Q = Xᵀ，后续优化 R 时 Q 固定，权重增量为 ΔW = RQ。
    return solution.T.to(device="cpu", dtype=getattr(torch, settings.storage_dtype)).contiguous()
