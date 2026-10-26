"""Evaluate held-out short-answer accuracy for a completed editing run."""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import torch
from safetensors.torch import load_file

from MDLM.contracts import file_sha256
from model.adapters.llada_8b_instruct import LLaDA8BInstructAdapter
from model.base import MaskedEditInput, tensor_sha256


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_answer(value: str) -> str:
    """Ignore case, Unicode width and repeated whitespace; keep answer content."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def make_masked_input(adapter: LLaDA8BInstructAdapter, prompt: str, answer: str) -> MaskedEditInput:
    tokenizer = adapter.tokenizer
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    prefix = tuple(tokenizer.encode(rendered, add_special_tokens=False))
    full = tuple(tokenizer.encode(rendered + answer, add_special_tokens=False))
    if not prefix or full[:len(prefix)] != prefix or len(full) == len(prefix):
        raise ValueError("测试题无法建立稳定的短答案 mask 边界")
    if adapter.mask_token_id in prefix:
        raise ValueError("测试题提示词包含 mask token")
    count = len(full) - len(prefix)
    ids = prefix + (adapter.mask_token_id,) * count
    return MaskedEditInput(ids, (1,) * len(ids), tuple(range(len(prefix), len(ids))), len(prefix) - 1)


def score_probe(adapter: LLaDA8BInstructAdapter, model, probe: dict, *, seed: int = 2026) -> dict:
    masked = make_masked_input(adapter, probe["prompt"], probe["answer"])
    with torch.no_grad():
        trace = adapter.sample_denoise(model, masked, steps=len(masked.mask_positions), temperature=0.0, seed=seed)
    predicted_ids = [trace.final_input.input_ids[position] for position in masked.mask_positions]
    prediction = adapter.tokenizer.decode(predicted_ids, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)
    return {"case_id": probe["case_id"], "probe_id": probe["probe_id"],
            "role": probe["role"], "prediction": prediction, "reference": probe["answer"],
            "correct": normalize_answer(prediction) == normalize_answer(probe["answer"]),
            "answer_token_count": len(masked.mask_positions)}


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    correct = sum(item["correct"] for item in rows)
    return {"correct": correct, "total": total, "accuracy": correct / total if total else None}


def evaluation_inputs(run_dir: Path, report: dict, choices: dict) -> tuple[Path | None, list[dict]]:
    configured = choices.get("evaluation", {}).get("inputs_file")
    if configured:
        path = Path(configured).expanduser()
        path = path if path.is_absolute() else Path(__file__).resolve().parents[1] / path
    elif report.get("prepared_inputs"):
        manifest = read_json(Path(report["prepared_inputs"]))
        path = Path(manifest["files"]["evaluation_inputs.json"])
        expected = manifest["files_sha256"][str(path)]
        if file_sha256(path) != expected:
            raise ValueError("测试输入文件与冻结的预处理清单不一致")
    else:
        return None, []
    payload = read_json(path)
    inputs = payload["inputs"]
    if not isinstance(inputs, list) or any(
        not isinstance(item, dict) or any(not isinstance(item.get(key), str) for key in
                                          ("case_id", "probe_id", "role", "prompt", "answer"))
        for item in inputs
    ):
        raise ValueError(f"测试输入格式错误：{path}")
    if any(item["role"] not in ("independent_test", "preservation_test") for item in inputs):
        raise ValueError("准确率只接受独立测试题和保持测试题")
    identities = [(item["role"], item["case_id"], item["probe_id"]) for item in inputs]
    if len(set(identities)) != len(identities):
        raise ValueError("测试输入包含重复探针")
    return path, inputs


def apply_final_weights(model, final_dir: Path, adapter: LLaDA8BInstructAdapter) -> None:
    manifest = read_json(final_dir / "manifest.json")
    if manifest["status"] != "accepted" or not manifest["p5_stage_complete"]:
        raise ValueError(f"P5 产物并非已接受的正式编辑：{final_dir}")
    tensor_path = final_dir / manifest["tensor_file"]
    if file_sha256(tensor_path) != manifest["tensor_file_sha256"]:
        raise ValueError(f"P5 权重文件摘要不一致：{tensor_path}")
    weights = load_file(str(tensor_path))
    if set(weights) != set(adapter.target_module_names()):
        raise ValueError("P5 权重层与实验配置不一致")
    for name, value in weights.items():
        target = model.get_submodule(name).weight
        info = manifest["layers"][name]
        if (target.shape != value.shape or target.dtype != value.dtype
                or tensor_sha256(value) != info["sha256"]
                or tensor_sha256(target) != manifest["baseline_weight_sha256"][name]):
            raise ValueError(f"P5 权重与本轮 baseline 不匹配：{name}")
    with torch.no_grad():
        for name, value in weights.items():
            target = model.get_submodule(name).weight
            target.copy_(value.to(target.device))


def evaluate_run(run_dir: str | Path, *, device: str | None = None) -> dict:
    """Score each round's baseline and accepted result without rerunning P0-P5."""
    run_dir = Path(run_dir).resolve()
    report = read_json(run_dir / "run_report.json")
    if report["status"] != "complete":
        raise ValueError("只有已完成的运行才能计算准确率")
    choices = read_json(run_dir / "effective_config.json")
    input_path, inputs = evaluation_inputs(run_dir, report, choices)
    output = run_dir / "accuracy_report.json"
    if not inputs:
        result = {"status": "unavailable", "reason": "未配置独立测试题或保持测试题", "inputs": str(input_path) if input_path else None}
        write_json(output, result)
        return result

    layers = tuple(choices["target_layers"])
    original_model = Path(report["original_model"])
    adapter = LLaDA8BInstructAdapter.from_local_tokenizer(model_path=original_model, target_layers=layers)
    selected_device = device or choices.get("execution", {}).get("device") or choices.get("p1", {}).get("device_map", "auto")
    dtype = choices.get("p1", {}).get("torch_dtype", "auto")
    seed = choices.get("p1", {}).get("seed", 2026)
    records = []
    for row in report["rounds"]:
        probes = [item for item in inputs if item["role"] == "preservation_test" or
                  (item["role"] == "independent_test" and item["case_id"] in row["fact_ids"])]
        if not probes:
            continue
        model = adapter.load_model(row["baseline_model"], torch_dtype=dtype, device_map=selected_device)
        before = [score_probe(adapter, model, item, seed=seed) for item in probes]
        if row["decision"] == "accepted":
            snapshot = adapter.snapshot_target_weights(model)
            try:
                apply_final_weights(model, Path(row["final_model_dir"]), adapter)
                after = [score_probe(adapter, model, item, seed=seed) for item in probes]
            finally:
                adapter.restore_target_weights(model, snapshot)
        else:
            after = [dict(item) for item in before]
        record = {"round_id": row["round_id"], "case_id": row["case_id"], "decision": row["decision"],
                  "baseline_model": row["baseline_model"], "final_model_dir": row["final_model_dir"],
                  "before": before, "after": after}
        for role in ("independent_test", "preservation_test"):
            label = "independent" if role == "independent_test" else "preservation"
            record[label] = {"before": summarize([item for item in before if item["role"] == role]),
                             "after": summarize([item for item in after if item["role"] == role])}
        records.append(record)
        print(f"[评价 {row['round_id'] + 1}/{len(report['rounds'])}] {row['case_id']} | "
              f"独立题 {record['independent']['after']['correct']}/{record['independent']['after']['total']} | "
              f"保持题 {record['preservation']['after']['correct']}/{record['preservation']['after']['total']}", flush=True)

    expected = {(item["case_id"], item["probe_id"]) for item in inputs if item["role"] == "independent_test"}
    observed = [(item["case_id"], item["probe_id"]) for row in records for item in row["after"]
                if item["role"] == "independent_test"]
    if set(observed) != expected or len(observed) != len(expected):
        raise ValueError("部分独立测试题未被本次编辑轮次唯一覆盖")

    def aggregate(role: str, phase: str) -> dict:
        return summarize([item for row in records for item in row[phase] if item["role"] == role])

    result = {"status": "complete", "metric": "normalized_exact_match",
              "protocol": "确定性全 mask 去噪；答案窗口长度取参考答案 token 数；Unicode NFKC、大小写与空白归一后精确匹配",
              "inputs": str(input_path), "input_sha256": file_sha256(input_path),
              "mode": report["mode"], "rounds": records,
              "independent": {"unique_probe_count": len(expected),
                              **{phase: aggregate("independent_test", phase) for phase in ("before", "after")}},
              "preservation": {"unique_probe_count": len({(item["case_id"], item["probe_id"])
                                                        for item in inputs if item["role"] == "preservation_test"}),
                               **{phase: aggregate("preservation_test", phase) for phase in ("before", "after")}}}
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="已完成实验的输出目录")
    parser.add_argument("--device", help="cuda:0、cpu 等计算设备；默认使用原实验配置")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    choices = read_json(run_dir / "effective_config.json")
    from util.gpu_runtime import execution_context
    execution = dict(choices.get("execution", {}))
    if args.device:
        execution.update(backend="resident", device=args.device)
    with execution_context(execution, choices["target_layers"]):
        result = evaluate_run(run_dir, device=args.device)
    print(json.dumps({key: result.get(key) for key in ("status", "independent", "preservation")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
