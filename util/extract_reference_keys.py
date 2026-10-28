"""公式（4）：从本轮全 mask 初态提取 K，不重分词、不回填答案。"""

import torch

from model.base import tensor_sha256


def extract_keys(model, adapter, trajectories, *, storage_dtype="float32",
                 replay_atol=1e-2, replay_rtol=1e-3) -> dict:
    if storage_dtype not in ("float32", "float64"):
        raise ValueError("K 精度必须为 float32 或 float64")
    descriptions = adapter.inspect_target_modules(model)
    columns = {name: [] for name in descriptions}
    for trajectory in trajectories:
        initial = trajectory.trace.pre_forward_inputs[0]
        captured = adapter.forward_with_capture(model, initial)
        repeated = adapter.forward_with_capture(model, initial)
        if any(tensor_sha256(value.mask_logits) != trajectory.trace.mask_logit_sha256[0]
               for value in (captured, repeated)):
            raise ValueError("K 初态 logits 与 P1 基准不一致")
        if set(captured.module_inputs) != set(columns) or set(repeated.module_inputs) != set(columns):
            raise ValueError("K 捕获层与目标层不一致")
        for name, info in descriptions.items():
            key, replay = captured.module_inputs[name], repeated.module_inputs[name]
            if key.shape != (info["in_features"],) or not torch.isfinite(key).all() or not torch.allclose(
                key, replay, atol=replay_atol, rtol=replay_rtol
            ):
                raise ValueError(f"K 的形状、有限性或重放检查失败：{name}")
            stored = key.detach().to(device="cpu", dtype=getattr(torch, storage_dtype))
            if not torch.allclose(key.double().cpu(), stored.double(), atol=replay_atol, rtol=replay_rtol):
                raise ValueError("K 的配置精度超出原有重放容差")
            columns[name].append(stored)
    return {name: torch.stack(values, dim=1).contiguous() for name, values in columns.items()}
