"""公式（6）～（9）：由 K、C 和 λ 预先求固定写回因子 Q，供后续 ΔW = RQ 使用。"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WritebackConfig:
    """逐层 λ > 0、求解精度、Q 存储精度和计算设备。"""

    ridge_by_layer: dict[str, float]
    compute_dtype: str = "float64"
    storage_dtype: str = "float32"
    device: str = "cpu"


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
