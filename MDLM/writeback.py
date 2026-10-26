"""Stable fixed-factor solve for formulas (6)--(9); no model weights are changed."""

from __future__ import annotations

import math
import time

import torch

from MDLM.contracts import WritebackConfig


class WritebackSolveError(ValueError):
    """Retain numerical diagnostics even when a layer cannot publish Q."""

    def __init__(self, message: str, diagnostics: dict):
        super().__init__(message)
        self.diagnostics = diagnostics


def stable_frobenius_norm(value: torch.Tensor) -> torch.Tensor:
    """Scale before squaring so finite small/large Frobenius norms stay representable."""
    scale = value.abs().max()
    if bool(scale == 0):
        return scale
    return scale * (value / scale).norm()


def relative_residual(keys: torch.Tensor, keep: torch.Tensor, solution: torch.Tensor,
                      ridge: float, *, epsilon: float, chunk_size: int) -> float:
    """Compute ||(C + KKᵀ + λI)X - K|| / max(||K||, ε) in float64 blocks."""
    # 用原始 C/K 复核求解与保存后的 X，不以分解器内部重构的 H 代替。
    device = solution.device
    keys64, solution64 = keys.detach().to(device=device, dtype=torch.float64), solution.double()
    small = keys64.T @ solution64
    error_norm = torch.zeros((), device=device, dtype=torch.float64)
    for start in range(0, len(keys), chunk_size):
        end = min(len(keys), start + chunk_size)
        rows = keep[start:end].detach().to(device=device, dtype=torch.float64)
        error = rows @ solution64 + keys64[start:end] @ small + ridge * solution64[start:end] - keys64[start:end]
        # 分块范数用 hypot 合并，避免平方和下溢为 0 或溢出为 inf。
        error_norm = torch.hypot(error_norm, stable_frobenius_norm(error))
    return float(error_norm / stable_frobenius_norm(keys64).clamp_min(epsilon))


@torch.no_grad()
def solve_writeback_factor(keys: torch.Tensor, keep: torch.Tensor, ridge: float,
                           settings: WritebackConfig) -> tuple[torch.Tensor, dict]:
    """Return a detached CPU Q [fact, input feature] and a numerical report."""
    settings.check()
    started = time.perf_counter()
    reported_ridge = ridge if isinstance(ridge, (int, float)) and math.isfinite(ridge) else None
    report = {"status": "partial", "phase": "inputs", "ridge": reported_ridge,
              "compute_dtype": settings.compute_dtype, "storage_dtype": settings.storage_dtype,
              "device": settings.device, "solver": settings.solver, "failure_policy": settings.failure_policy,
              "residual_norm_method": "scaled_frobenius_hypot"}

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    try:
        require(isinstance(ridge, (int, float)) and not isinstance(ridge, bool)
                and math.isfinite(ridge) and ridge > 0, "ridge must be finite and positive")
        require(keys.ndim == 2 and min(keys.shape) > 0 and keep.shape == (len(keys), len(keys))
                and keys.is_floating_point() and keep.is_floating_point(), "K/C dimensions or dtypes are invalid")
        width, count = keys.shape
        report.update({"key_shape": list(keys.shape), "keep_shape": list(keep.shape),
                       "q_shape": [count, width]})
        device = torch.device(settings.device)
        if device.type == "cuda":
            require(torch.cuda.is_available(), "the configured CUDA device is unavailable")
        # C 在 GPU 上复用，避免四次方程复核重复搬运同一方阵。
        keep = keep.detach().to(device=device)
        keys = keys.detach().to(device=device)
        for start in range(0, width, settings.chunk_size):
            rows = keep[start:start + settings.chunk_size].detach().double()
            columns = keep[:, start:start + settings.chunk_size].T.detach().double()
            require(bool(torch.isfinite(rows).all()) and bool(torch.isfinite(columns).all()), "C contains nonfinite values")
            require(torch.allclose(rows, columns, atol=settings.symmetry_atol, rtol=settings.symmetry_rtol),
                    "C is not symmetric within the frozen tolerance")
        require(bool(torch.isfinite(keys).all()), "K contains nonfinite values")
        values = keys.detach().to(device=device, dtype=getattr(torch, settings.compute_dtype))
        # 公式（8）（9）：按冻结 λ 构造 H，不缩放 C，不补 jitter，也不改动输入矩阵。
        report["phase"] = "system"
        system = keep.detach().to(device=device, dtype=values.dtype, copy=True)
        system.addmm_(values, values.T)
        diagonal = system.diagonal().clone()
        system.diagonal().add_(ridge)
        require(bool((system.diagonal() > diagonal).all()), "ridge is not representable at the configured precision")
        infinity_norm = 0.0
        for start in range(0, width, settings.chunk_size):
            rows = system[start:start + settings.chunk_size]
            require(bool(torch.isfinite(rows).all()), "H contains nonfinite values")
            infinity_norm = max(infinity_norm, float(rows.double().abs().sum(dim=1).max()))
        key_norm = float(stable_frobenius_norm(values.double()))
        require(math.isfinite(infinity_norm) and math.isfinite(key_norm), "H/K scale exceeds diagnostic precision")
        report.update({"h_infinity_norm": infinity_norm, "ridge_to_h_scale": ridge / max(infinity_norm, ridge),
                       "h_diagonal_min": float(system.diagonal().min()),
                       "h_diagonal_max": float(system.diagonal().max()),
                       "key_norm": key_norm,
                       "estimated_dense_working_bytes": (2 * width * width + 3 * width * count) * values.element_size()})
        report["phase"] = "cholesky"
        factor, info = torch.linalg.cholesky_ex(system, check_errors=False)
        report["cholesky_info"] = int(info)
        require(int(info) == 0, "H Cholesky factorization failed; frozen ridge is unchanged")
        require(bool(torch.isfinite(factor.diagonal()).all()), "Cholesky factor contains nonfinite diagonal values")
        report.update({"cholesky_diagonal_min": float(factor.diagonal().min()),
                       "cholesky_diagonal_max": float(factor.diagonal().max())})
        # 最后一个行块是 H 的视图，必须一起释放，才能释放整个方阵存储。
        del rows, system
        solution = torch.cholesky_solve(values, factor)
        del factor
        require(bool(torch.isfinite(solution).all()), "linear solve produced nonfinite values")
        report["phase"] = "residual"
        error = relative_residual(keys, keep, solution, ridge,
                                  epsilon=settings.residual_epsilon, chunk_size=settings.chunk_size)
        report["relative_residual"] = error if math.isfinite(error) else None
        require(math.isfinite(error) and error <= settings.residual_tolerance, "linear solve residual exceeds the frozen tolerance")
        # 公式（7）（9）还要求 QH=Kᵀ；近似对称的 C 不能只用 HX=K 代替此检查。
        normal_error = relative_residual(keys, keep.T, solution, ridge,
                                         epsilon=settings.residual_epsilon, chunk_size=settings.chunk_size)
        report["normal_equation_relative_residual"] = normal_error if math.isfinite(normal_error) else None
        require(math.isfinite(normal_error) and normal_error <= settings.residual_tolerance,
                "Q normal equation residual exceeds the frozen tolerance")
        # 公式（9）：Q=Xᵀ；保存精度也要通过原方程残差，不靠提高原捕获精度宣称更精确。
        q = solution.T.detach().to(device="cpu", dtype=getattr(torch, settings.storage_dtype)).contiguous()
        require(bool(torch.isfinite(q).all()), "stored Q contains nonfinite values")
        stored_error = relative_residual(keys, keep, q.T.to(device), ridge,
                                         epsilon=settings.residual_epsilon, chunk_size=settings.chunk_size)
        report["stored_relative_residual"] = stored_error if math.isfinite(stored_error) else None
        require(math.isfinite(stored_error) and stored_error <= settings.stored_residual_tolerance,
                "stored Q residual exceeds the frozen tolerance")
        stored_normal_error = relative_residual(keys, keep.T, q.T.to(device), ridge,
                                                epsilon=settings.residual_epsilon, chunk_size=settings.chunk_size)
        report["stored_normal_equation_relative_residual"] = stored_normal_error if math.isfinite(stored_normal_error) else None
        require(math.isfinite(stored_normal_error) and stored_normal_error <= settings.stored_residual_tolerance,
                "stored Q normal equation residual exceeds the frozen tolerance")
        report.update({"status": "complete", "phase": "complete", "failure": None,
                       "elapsed_seconds": time.perf_counter() - started})
        try:
            import resource
            report["process_peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        except ImportError:
            report["process_peak_rss_bytes"] = None
        if device.type == "cuda":
            report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        return q, report
    except Exception as error:
        report.update({"status": "failed", "failure": str(error), "elapsed_seconds": time.perf_counter() - started})
        raise WritebackSolveError(str(error), report) from error
