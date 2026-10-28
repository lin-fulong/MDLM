"""公式（5）：按显式权重累计未中心化保持矩阵 C = Σ_m ω_m u_m u_mᵀ。"""

import torch


class SecondMomentAccumulator:
    """逐批累计一层的 C，不保存激活或生成诊断报告。"""

    def __init__(self, width: int, *, accumulation_dtype: str = "float64", device: str = "cpu"):
        self.matrix = torch.zeros((width, width), dtype=getattr(torch, accumulation_dtype), device=device)

    @torch.no_grad()
    def add(self, activations: torch.Tensor, weights: list[float]) -> None:
        """activations 每行是 u_mᵀ；weights 是对应的非负统计权重。"""
        values = activations.detach().to(device=self.matrix.device, dtype=torch.float64)
        weights = torch.as_tensor(weights, dtype=torch.float64, device=self.matrix.device)
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
