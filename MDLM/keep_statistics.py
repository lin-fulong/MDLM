"""Streaming uncentered second moments and bounded-memory checks for formula (5)."""

from __future__ import annotations

import math

import torch


class SecondMomentAccumulator:
    """Accumulate C on the configured device without retaining activations."""

    def __init__(
        self, width: int, *, accumulation_dtype: str = "float64", projection_count: int = 8,
        direct_block_size: int = 16, seed: int = 2026, device: str = "cpu",
    ):
        if type(width) is not int or width < 1:
            raise ValueError("width must be a positive integer")
        if accumulation_dtype not in ("float32", "float64"):
            raise ValueError("accumulation_dtype must be float32 or float64")
        if type(projection_count) is not int or projection_count < 1:
            raise ValueError("projection_count must be a positive integer")
        if type(direct_block_size) is not int or direct_block_size < 1:
            raise ValueError("direct_block_size must be a positive integer")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.width = width
        self.device = torch.device(device)
        self.matrix = torch.zeros((width, width), dtype=getattr(torch, accumulation_dtype), device=self.device)
        self.vector_count = 0
        self.positive_weight_count = 0
        self.weight_sum = 0.0
        self.expected_trace = 0.0
        generator = torch.Generator(device="cpu").manual_seed(seed)
        vectors = torch.randn((width, min(width, projection_count)), generator=generator,
                              dtype=torch.float64)
        self.projection_vectors = torch.linalg.qr(vectors.to(self.device), mode="reduced").Q
        self.direct_indices = torch.randperm(width, generator=generator)[:min(width, direct_block_size)].sort().values.to(self.device)
        self._direct_expected = torch.zeros((len(self.direct_indices), len(self.direct_indices)), dtype=torch.float64, device=self.device)
        count = self.projection_vectors.shape[1]
        self._projection_expected = torch.zeros((count, count), dtype=torch.float64, device=self.device)

    def add(self, activations: torch.Tensor, weights: list[float]) -> None:
        """Add explicit nonnegative weights; neither center nor normalize the samples."""
        if not isinstance(activations, torch.Tensor) or activations.ndim != 2 or activations.shape[1] != self.width:
            raise ValueError("activations must have shape [sample_count, width]")
        if activations.is_complex():
            raise ValueError("activations must be real")
        values = activations.detach().to(device=self.device, dtype=torch.float64)
        weight_tensor = torch.as_tensor(weights, dtype=torch.float64, device=self.device)
        if weight_tensor.ndim != 1 or len(weight_tensor) != len(values):
            raise ValueError("one weight is required per activation vector")
        if not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(weight_tensor).all()):
            raise ValueError("activations and weights must be finite")
        if bool((weight_tensor < 0).any()):
            raise ValueError("weights must be nonnegative")

        sample_count = len(values)
        positive = weight_tensor > 0
        positive_count = int(positive.sum())
        total_weight = math.fsum((self.weight_sum, *weight_tensor.tolist()))
        # 零权重项不参与乘法，避免大数在目标累计精度溢出后形成 0×inf。
        values, weight_tensor = values[positive], weight_tensor[positive]

        # 公式（5）：(sqrt(ω)u)(sqrt(ω)u)ᵀ 与 ωuuᵀ 等价，不改变保持尺度。
        # 先吸收权重再平方，避免 u² 或低精度 ω 的中间溢出/下溢。
        weighted_values = values * weight_tensor.sqrt()[:, None]
        stored_values = weighted_values.to(self.matrix.dtype)
        if not bool(torch.isfinite(stored_values).all()):
            raise ValueError("weighted activations exceed the accumulation precision")
        self.matrix.addmm_(stored_values.T, stored_values)
        if positive_count:
            # 公式（5）的外积严格对称；BLAS 两侧可能采用不同归约顺序。
            # 分块复用上三角结果，保持同一外积定义/权重，不平均或重缩放 C。
            for start in range(0, self.width, 256):
                end = min(self.width, start + 256)
                if start:
                    self.matrix[start:end, :start].copy_(self.matrix[:start, start:end].T)
                block = self.matrix[start:end, start:end]
                lower = torch.tril_indices(end - start, end - start, offset=-1, device=self.device)
                block[lower[0], lower[1]] = block[lower[1], lower[0]]
        projected = weighted_values @ self.projection_vectors
        direct = weighted_values[:, self.direct_indices]
        for row, projection in zip(direct, projected):
            # 小坐标块逐项外积与投影矩独立累计，供保存精度与实现核对。
            self._direct_expected.add_(torch.outer(row, row))
            self._projection_expected.add_(torch.outer(projection, projection))
        self.vector_count += sample_count
        self.positive_weight_count += positive_count
        self.weight_sum = total_weight
        self.expected_trace += float(weighted_values.square().sum())

    def diagnose(
        self, matrix: torch.Tensor, *, atol: float = 1e-6, rtol: float = 1e-5,
        chunk_size: int = 512,
    ) -> dict:
        """Check a stored C in row blocks; only the small projection uses eigvalsh."""
        if not isinstance(matrix, torch.Tensor) or matrix.shape != (self.width, self.width) or matrix.is_complex():
            raise ValueError("matrix must be a real [width, width] tensor")
        if not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0:
            raise ValueError("diagnostic tolerances must be finite and nonnegative")
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        vectors = self.projection_vectors
        projection = torch.zeros_like(self._projection_expected)
        finite, symmetric = True, True
        symmetry_max_diff, trace = 0.0, 0.0
        diagonal_min, diagonal_max, diagonal_abs_max = math.inf, -math.inf, 0.0
        for start in range(0, self.width, chunk_size):
            end = min(self.width, start + chunk_size)
            rows = matrix[start:end].detach().to(device=self.device, dtype=torch.float64)
            columns = matrix[:, start:end].T.detach().to(device=self.device, dtype=torch.float64)
            if not bool(torch.isfinite(rows).all()) or not bool(torch.isfinite(columns).all()):
                finite = False
                continue
            difference = (rows - columns).abs()
            symmetry_max_diff = max(symmetry_max_diff, float(difference.max()))
            symmetric = symmetric and bool((difference <= atol + rtol * columns.abs()).all())
            diagonal = rows[torch.arange(end - start, device=self.device), torch.arange(start, end, device=self.device)]
            trace += float(diagonal.sum())
            diagonal_min = min(diagonal_min, float(diagonal.min()))
            diagonal_max = max(diagonal_max, float(diagonal.max()))
            diagonal_abs_max = max(diagonal_abs_max, float(diagonal.abs().max()))
            projection.add_(vectors[start:end].T @ (rows @ vectors))

        indices = self.direct_indices.to(matrix.device)
        direct = matrix[indices[:, None], indices[None, :]].detach().to(device=self.device, dtype=torch.float64)
        direct_finite = bool(torch.isfinite(direct).all())
        expected_finite = math.isfinite(self.expected_trace) and math.isfinite(self.weight_sum)
        expected_finite = expected_finite and bool(torch.isfinite(self._direct_expected).all())
        expected_finite = expected_finite and bool(torch.isfinite(self._projection_expected).all())
        projection_finite = bool(torch.isfinite(projection).all())
        direct_diff = (direct - self._direct_expected).abs()
        direct_max_diff = float(direct_diff.max()) if direct_finite and expected_finite else None
        if direct_max_diff is not None and not math.isfinite(direct_max_diff):
            direct_max_diff = None
        # 抵消后的近零非对角元，以独立对角几何尺度核对，避免逐元素相对误差误拒。
        diagonal_roots = self._direct_expected.diagonal().sqrt()
        direct_scale = diagonal_roots[:, None] * diagonal_roots[None, :]
        direct_bound = atol + rtol * direct_scale
        direct_bound_finite = bool(torch.isfinite(direct_bound).all())
        direct_matches = direct_finite and expected_finite and direct_bound_finite \
            and bool((direct_diff <= direct_bound).all())
        positive_scale = direct_scale > 0
        direct_normalized_error = None
        if direct_finite and expected_finite:
            if bool(positive_scale.any()):
                direct_normalized_error = float((direct_diff[positive_scale] / direct_scale[positive_scale]).max())
            elif direct_max_diff == 0:
                direct_normalized_error = 0.0
        if direct_normalized_error is not None and not math.isfinite(direct_normalized_error):
            direct_normalized_error = None
        projection_max_diff = float((projection - self._projection_expected).abs().max()) \
            if finite and expected_finite and projection_finite else None
        if projection_max_diff is not None and not math.isfinite(projection_max_diff):
            projection_max_diff = None
        # 正交投影的误差使用原二阶矩 trace 尺度，仍严格使用配置中的 atol/rtol。
        # 存储舍入可向理论零投影引入小量；此阈值是数值容差，不是全谱 PSD 证明。
        projection_scale = self.expected_trace if expected_finite else None
        projection_bound = atol + rtol * projection_scale if projection_scale is not None else None
        if projection_bound is not None and not math.isfinite(projection_bound):
            projection_bound = None
        projection_matches = projection_max_diff is not None and projection_bound is not None \
            and projection_max_diff <= projection_bound
        projection_normalized_error = projection_max_diff / projection_scale \
            if projection_max_diff is not None and projection_scale > 0 else None
        if projection_max_diff == 0 and projection_scale == 0:
            projection_normalized_error = 0.0
        if projection_normalized_error is not None and not math.isfinite(projection_normalized_error):
            projection_normalized_error = None
        trace_finite = math.isfinite(trace)
        trace_matches = finite and trace_finite and expected_finite and math.isclose(
            trace, self.expected_trace, abs_tol=atol, rel_tol=rtol)
        nonnegative_diagonal = finite and diagonal_min >= -(atol + rtol * diagonal_abs_max)
        eigen_min = eigen_max = None
        projected_psd = False
        if finite and projection_finite:
            # 只检查低维投影的谱；低秩、零矩阵合法，不要求 C 正定。
            eigenvalues = torch.linalg.eigvalsh(projection * 0.5 + projection.T * 0.5)
            if bool(torch.isfinite(eigenvalues).all()):
                eigen_min, eigen_max = float(eigenvalues.min()), float(eigenvalues.max())
                projected_psd = projection_bound is not None and eigen_min >= -projection_bound
        checks = {
            "finite": finite and trace_finite and expected_finite and projection_finite,
            "symmetric": finite and symmetric,
            "nonnegative_diagonal": nonnegative_diagonal,
            "trace_matches": trace_matches,
            "direct_outer_product_matches": bool(direct_matches),
            "projected_second_moment_matches": bool(projection_matches),
            "projected_psd": projected_psd,
        }
        return {
            **checks,
            "all_checks_passed": all(checks.values()),
            "width": self.width,
            "dtype": str(matrix.dtype).removeprefix("torch."),
            "vector_count": self.vector_count,
            "positive_weight_count": self.positive_weight_count,
            "weight_sum": self.weight_sum if math.isfinite(self.weight_sum) else None,
            "expected_trace": self.expected_trace if math.isfinite(self.expected_trace) else None,
            "trace": trace if finite and trace_finite else None,
            "diagonal": {"min": diagonal_min if finite else None,
                         "max": diagonal_max if finite else None,
                         "mean": trace / self.width if finite and trace_finite else None},
            "symmetry_max_diff": symmetry_max_diff if finite and math.isfinite(symmetry_max_diff) else None,
            "direct_outer_product_max_diff": direct_max_diff,
            "direct_reference_scale": "sqrt(E_ii)*sqrt(E_jj)",
            "direct_reference_scale_max": float(direct_scale.max()) if expected_finite else None,
            "direct_error_bound_max": float(direct_bound.max()) if expected_finite and direct_bound_finite else None,
            "direct_normalized_error": direct_normalized_error,
            "projected_second_moment_max_diff": projection_max_diff,
            "projection_reference_scale": projection_scale,
            "projection_error_bound": projection_bound,
            "projection_normalized_error": projection_normalized_error,
            "projected_psd_negative_tolerance": projection_bound,
            "projection_count": vectors.shape[1],
            "projected_eigenvalue_min": eigen_min,
            "projected_eigenvalue_max": eigen_max,
            "full_spectrum_checked": False,
            "atol": atol,
            "rtol": rtol,
        }
