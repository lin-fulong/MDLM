"""公式（5）的独立重放检查，直接核对内存中的 C。"""

import math
import torch
from model.base import tensor_sha256
from util.io import require


@torch.no_grad()
def validate_matrix(matrix, model, adapter, records, name, captured_digests, *, seed=2026,
                    projection_count=8, direct_block_size=16, chunk_size=512,
                    atol=1e-6, rtol=1e-5, batch_keep=False):
    from util.extract_keep_statistics import equal_length_groups
    width, device = matrix.shape[0], matrix.device
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    vectors = torch.linalg.qr(torch.randn((width, min(width, projection_count)),
                              generator=generator, dtype=torch.float64).to(device), mode="reduced").Q
    indices = torch.randperm(width, generator=generator)[:min(width, direct_block_size)].sort().values.to(device)
    expected_direct = torch.zeros((len(indices), len(indices)), dtype=torch.float64, device=device)
    expected_projection = torch.zeros((vectors.shape[1], vectors.shape[1]), dtype=torch.float64, device=device)
    expected_trace, index = 0.0, 0
    for group in equal_length_groups(records, enabled=batch_keep):
        captures = adapter.forward_keep_batch_with_capture(model, tuple((inp, pos) for _, inp, pos in group))
        for (row, _, positions), captured in zip(group, captures):
            values = captured.module_inputs[name].detach().to(device=device, dtype=torch.float64)
            require((tensor_sha256(captured.mask_logits), tensor_sha256(captured.module_inputs[name]))
                    == captured_digests[index], "C 的独立重放与原始捕获不同")
            index += 1
            weights = torch.as_tensor(row["weights_by_layer"][name], dtype=torch.float64, device=device)
            require(values.shape == (len(positions), width) and len(weights) == len(positions)
                    and bool(torch.isfinite(values).all()) and bool(torch.isfinite(weights).all())
                    and bool((weights >= 0).all()), "C 的独立重放输入无效")
            weighted = values[weights > 0] * weights[weights > 0].sqrt()[:, None]
            direct, projected = weighted[:, indices], weighted @ vectors
            expected_direct += direct.T @ direct
            expected_projection += projected.T @ projected
            expected_trace += float(weighted.square().sum())
    actual_projection = torch.zeros_like(expected_projection)
    actual_direct = torch.empty_like(expected_direct)
    trace, diagonal_min = 0.0, math.inf
    for start in range(0, width, chunk_size):
        end = min(width, start + chunk_size)
        rows, columns = matrix[start:end].double(), matrix[:, start:end].T.double()
        require(bool(torch.isfinite(rows).all()), "C 包含非有限值")
        require(bool(((rows - columns).abs() <= atol + rtol * columns.abs()).all()), "C 不满足原有对称容差")
        diagonal = rows[torch.arange(end - start, device=device), torch.arange(start, end, device=device)]
        trace += float(diagonal.sum())
        diagonal_min = min(diagonal_min, float(diagonal.min()))
        actual_projection += vectors[start:end].T @ (rows @ vectors)
        selected = (indices >= start) & (indices < end)
        if bool(selected.any()):
            actual_direct[selected] = rows[indices[selected] - start][:, indices]
    bound = atol + rtol * expected_trace
    require(math.isfinite(trace) and math.isclose(trace, expected_trace, abs_tol=atol, rel_tol=rtol),
            "C 的 trace 与独立加权输入不一致")
    require(diagonal_min >= -bound, "C 存在超出容差的负对角值")
    direct_scale = expected_direct.diagonal().clamp_min(0).sqrt()
    require(bool(((actual_direct - expected_direct).abs()
                  <= atol + rtol * direct_scale[:, None] * direct_scale[None, :]).all()), "C 的独立外积不一致")
    require(float((actual_projection - expected_projection).abs().max()) <= bound, "C 的独立投影矩不一致")
    eigen_min = float(torch.linalg.eigvalsh(actual_projection * 0.5 + actual_projection.T * 0.5).min())
    require(math.isfinite(eigen_min) and eigen_min >= -bound, "C 的投影半正定检查未通过")
