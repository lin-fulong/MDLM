"""公式（5）：逐层捕获保持激活并累计 C，交给本层 Q 求解后释放。"""

import torch

from MDLM.keep_statistics import SecondMomentAccumulator
from model.base import KeepModelInput


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


def keep_input_from_record(record):
    """还原已准备的保持输入，核对从零开始的隐式位置编号。"""
    position_ids = list(range(len(record["input_ids"])))
    if record.get("implicit_position_ids", position_ids) != position_ids:
        raise ValueError("保持输入的位置编号与序列长度不一致")
    return KeepModelInput(tuple(record["input_ids"]), tuple(record["attention_mask"]), tuple(record["mask_positions"]))


def keep_matrices(model, adapter, rows, *, accumulation_dtype="float64", storage_dtype="float32",
                  device="cpu", batch_keep=False):
    """公式（5）：每层返回 (模块名, C)，每批保持输入只前向一次。"""
    records = [(row, keep_input_from_record(row["input"]), tuple(row["capture_positions"])) for row in rows]
    for layer in adapter.target_layers:
        name = f"model.transformer.blocks.{layer}.ff_out"
        layer_adapter = type(adapter)(adapter.tokenizer, mask_token_id=adapter.mask_token_id, target_layers=(layer,))
        accumulator = SecondMomentAccumulator(
            model.get_submodule(name).in_features, accumulation_dtype=accumulation_dtype, device=device)
        for group in equal_length_groups(records, enabled=batch_keep):
            requests = tuple((model_input, positions) for _, model_input, positions in group)
            captures = layer_adapter.forward_keep_batch_with_capture(model, requests)
            activations = torch.cat([captured.module_inputs[name] for captured in captures])
            weights = [weight for row, _, _ in group for weight in row["weights_by_layer"][name]]
            accumulator.add(activations, weights)
        matrix = accumulator.matrix.to(dtype=getattr(torch, storage_dtype))
        yield name, matrix
        del accumulator, matrix
