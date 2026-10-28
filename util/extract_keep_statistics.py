"""公式（5）：逐层累计 C；一次只保留一个方阵，交给 Q 求解后释放。"""

import torch

from MDLM.keep_statistics import SecondMomentAccumulator
from model.base import KeepModelInput, tensor_sha256


def equal_length_groups(records, input_index=1, enabled=False):
    start = 0
    while start < len(records):
        length = len(records[start][input_index].input_ids)
        limit = min(4, max(1, 256 // max(length, 1))) if enabled else 1
        end = start + 1
        while end < len(records) and end - start < limit and len(records[end][input_index].input_ids) == length:
            end += 1
        yield records[start:end]
        start = end


def keep_input_from_record(record):
    if record.get("implicit_position_ids", list(range(len(record["input_ids"])))) != list(range(len(record["input_ids"]))):
        raise ValueError("保持输入的位置编号与序列长度不一致")
    return KeepModelInput(tuple(record["input_ids"]), tuple(record["attention_mask"]), tuple(record["mask_positions"]))


def keep_matrices(model, adapter, rows, *, accumulation_dtype="float64", storage_dtype="float32",
                  replay_atol=1e-2, replay_rtol=1e-3, check_atol=1e-6, check_rtol=1e-5,
                  qa_seed=2026, projection_count=8, direct_block_size=16, chunk_size=512,
                  device="cpu", batch_keep=False):
    if not rows or storage_dtype not in ("float32", "float64"):
        raise ValueError("C 需要非空保持输入和 float32/float64 存储精度")
    names = set(adapter.target_module_names())
    if any(set(row["weights_by_layer"]) != names for row in rows):
        raise ValueError("保持权重与目标层不一致")
    records = [(row, keep_input_from_record(row["input"]), tuple(row["capture_positions"])) for row in rows]
    for layer in adapter.target_layers:
        name = f"model.transformer.blocks.{layer}.ff_out"
        layer_adapter = type(adapter)(adapter.tokenizer, mask_token_id=adapter.mask_token_id, target_layers=(layer,))
        accumulator = SecondMomentAccumulator(
            model.get_submodule(name).in_features, accumulation_dtype=accumulation_dtype,
            projection_count=projection_count, direct_block_size=direct_block_size, seed=qa_seed, device=device)
        digests = []
        for group in equal_length_groups(records, enabled=batch_keep):
            requests = tuple((model_input, positions) for _, model_input, positions in group)
            captures = layer_adapter.forward_keep_batch_with_capture(model, requests)
            repeats = layer_adapter.forward_keep_batch_with_capture(model, requests)
            values, weights = [], []
            for (row, _, _), captured, repeated in zip(group, captures, repeats):
                actual, replay = captured.module_inputs[name], repeated.module_inputs[name]
                if tensor_sha256(captured.mask_logits) != tensor_sha256(repeated.mask_logits) or not torch.allclose(
                    actual, replay, atol=replay_atol, rtol=replay_rtol
                ):
                    raise ValueError(f"C 保持输入未能重放：{name}")
                values.append(actual)
                weights.extend(row["weights_by_layer"][name])
                digests.append((tensor_sha256(captured.mask_logits), tensor_sha256(actual)))
            accumulator.add(torch.cat(values), weights)
        matrix = accumulator.matrix.to(dtype=getattr(torch, storage_dtype))
        diagnostics = accumulator.diagnose(matrix, atol=check_atol, rtol=check_rtol, chunk_size=chunk_size)
        if not diagnostics["all_checks_passed"]:
            raise ValueError(f"C 的原有数值检查未通过：{name}")
        from util.validate_matrix_artifacts import validate_matrix
        validate_matrix(matrix, model, layer_adapter, records, name, digests, seed=qa_seed,
                        projection_count=projection_count, direct_block_size=direct_block_size,
                        chunk_size=chunk_size, atol=check_atol, rtol=check_rtol, batch_keep=batch_keep)
        yield name, matrix, diagnostics
        del accumulator, matrix
