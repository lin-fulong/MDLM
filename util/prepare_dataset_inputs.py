"""Freeze dataset-derived P2/P4/P5 inputs and reserve independent evaluation probes."""

from __future__ import annotations

import argparse
import json
import math
import uuid
from pathlib import Path

from MDLM.contracts import (
    DEVELOPMENT_KEEP_VERSION, KEEP_INPUTS_VERSION, KEEP_TRAINING_VERSION,
    config_digest, file_sha256,
)
from data.readers.json_cases import load_cases
from model.adapters.llada_8b_instruct import DEFAULT_MODEL_PATH, LLaDA8BInstructAdapter
from util.pipeline_cache import cache_lock


PREPARATION_VERSION = "mdlm.dataset_inputs.1"
DEFAULT_SETTINGS = {"seed": 2026, "dev_fraction": 0.25, "test_fraction": 0.25,
                    "statistics_scale": 1.0, "dev_edit_policy": "edit_probe"}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def valid_files(files: dict[str, str]) -> bool:
    try:
        return isinstance(files, dict) and bool(files) and all(Path(path).is_file() and file_sha256(path) == digest
                                   for path, digest in files.items())
    except OSError:
        return False


def tokenizer_identity(model: Path) -> dict:
    hashes = {name: file_sha256(model / name) for name in (
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")}
    return {"tokenizer_revision": "local-files-sha256:" + config_digest(hashes),
            "tokenizer_files_sha256": hashes, "model_config_sha256": file_sha256(model / "config.json")}


def encode_keep(adapter, prompt: str, answer: str) -> tuple[dict, list[int], list[int]]:
    rendered = adapter.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    ids = list(adapter.tokenizer.encode(rendered, add_special_tokens=False))
    full = list(adapter.tokenizer.encode(rendered + answer, add_special_tokens=False))
    targets = full[len(ids):]
    if not ids or full[:len(ids)] != ids or not targets:
        raise ValueError("保持题无法建立稳定的短答案 mask 边界")
    if adapter.mask_token_id in ids or set(targets) & set(adapter.tokenizer.all_special_ids):
        raise ValueError("保持题含有不允许的预置 mask 或答案特殊 token")
    positions = list(range(len(ids), len(ids) + len(targets)))
    tokens = ids + [adapter.mask_token_id] * len(targets)
    return {"input_ids": tokens, "attention_mask": [1] * len(tokens),
            "mask_positions": positions, "implicit_position_ids": list(range(len(tokens)))}, positions, [len(ids) - 1, *positions]


def prepare_inputs(dataset: Path, model_path: Path, layers: tuple[int, ...], cache_dir: Path,
                   *, settings: dict | None = None, use_cache: bool = True) -> dict:
    lock_key = config_digest({"dataset": file_sha256(dataset), "model": str(model_path.resolve()),
                              "layers": layers, "settings": settings})
    with cache_lock(cache_dir / "locks" / ("prepare_" + lock_key + ".lock")):
        return _prepare_inputs(dataset, model_path, layers, cache_dir, settings=settings, use_cache=use_cache)


def _prepare_inputs(dataset: Path, model_path: Path, layers: tuple[int, ...], cache_dir: Path,
                    *, settings: dict | None = None, use_cache: bool = True) -> dict:
    """Only tokenize reviewed data; model activations and matrices are computed later."""
    dataset, model_path, cache_dir = dataset.resolve(), model_path.resolve(), cache_dir.resolve()
    policy = {**DEFAULT_SETTINGS, **(settings or {})}
    if set(policy) != set(DEFAULT_SETTINGS) or policy["dev_edit_policy"] != "edit_probe":
        raise ValueError("不支持的输入生成参数；dev_edit_policy 当前支持 edit_probe")
    if type(policy["seed"]) is not int or policy["seed"] < 0 or not layers or len(set(layers)) != len(layers):
        raise ValueError("seed 和目标层配置无效")
    if any(type(layer) is not int or layer < 0 for layer in layers):
        raise ValueError("目标层必须为非负整数")
    fractions = (policy["dev_fraction"], policy["test_fraction"])
    if any(not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) or not 0 <= x < 1
           for x in fractions) or not 0 < fractions[0] or sum(fractions) >= 1:
        raise ValueError("保持题划分比例必须满足 dev>0、test>=0、dev+test<1")
    if not isinstance(policy["statistics_scale"], (int, float)) or isinstance(policy["statistics_scale"], bool) \
            or not math.isfinite(policy["statistics_scale"]) or policy["statistics_scale"] <= 0:
        raise ValueError("statistics_scale 必须为有限正数")
    source_digest = file_sha256(dataset)
    identity = tokenizer_identity(model_path)
    request = {"version": PREPARATION_VERSION, "dataset_sha256": source_digest,
               "tokenizer": identity, "target_layers": list(layers), "settings": policy,
               "implementation": {str(path.name): file_sha256(path) for path in (
                   Path(__file__), Path(__file__).resolve().parents[1] / "data/readers/reviewed_cases.py",
                   Path(__file__).resolve().parents[1] / "data/readers/json_cases.py",
                   Path(__file__).resolve().parents[1] / "model/adapters/llada_8b_instruct.py")}}
    key = config_digest(request)
    output = cache_dir / "datasets" / (key if use_cache else key + "_" + uuid.uuid4().hex[:8])
    manifest_path = output / "manifest.json"
    if use_cache and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = {}
        if isinstance(manifest, dict) and manifest.get("status") == "complete" and manifest.get("request") == request and valid_files(manifest.get("files_sha256", {})):
            return {**manifest, "cache_hit": True, "manifest": str(manifest_path)}
    if output.exists():
        output.rename(output.with_name(output.name + ".invalid_" + uuid.uuid4().hex[:8]))
    output.mkdir(parents=True)
    manifest = {"status": "running", "request": request, "cache_key": key,
                "source_dataset": str(dataset), "cache_dir": str(output)}
    write_json(manifest_path, manifest)
    try:
        cases = load_cases(dataset)
        if not cases:
            raise ValueError("所选数据集没有可执行的编辑案例")
        adapter = LLaDA8BInstructAdapter.from_local_tokenizer(model_path=model_path, target_layers=layers)
        # 按题目内容分组后划分用途，重复题目不会同时进入保持训练和开发保持。
        pool = {}
        evaluation = []
        for case in cases:
            for probe in case["prompts"].get("preservation", []):
                prompt_key = config_digest({"prompt": probe["prompt"].strip()})
                item = pool.setdefault(prompt_key, {"prompt": probe["prompt"], "answer": probe["answer"], "sources": []})
                item["sources"].append({"case_id": case["case_id"], "probe_id": probe["probe_id"]})
            for group, probes in case["prompts"].items():
                if group in ("edit", "preservation"):
                    continue
                evaluation.extend({"case_id": case["case_id"], **probe} for probe in probes)
        ordered = sorted(pool, key=lambda key_: config_digest({"seed": policy["seed"], "prompt": key_}))
        if len(ordered) < 2:
            raise ValueError("自动生成保持统计和开发保持输入至少需要两道不同的 preservation 题")
        dev_count = min(len(ordered) - 1, max(1, int(len(ordered) * policy["dev_fraction"])))
        test_count = min(len(ordered) - dev_count - 1, max(1, int(len(ordered) * policy["test_fraction"]))) \
            if policy["test_fraction"] > 0 and len(ordered) >= 3 else 0
        partition = {"dev": ordered[:dev_count], "test": ordered[dev_count:dev_count + test_count],
                     "statistics_train": ordered[dev_count + test_count:]}
        encoded = {key_: encode_keep(adapter, pool[key_]["prompt"], pool[key_]["answer"])
                   for key_ in partition["dev"] + partition["statistics_train"]}
        example_only = any(case["quality_control"]["status"] == "example_only" for case in cases)
        common = {"example_only": example_only, "tokenizer_revision": identity["tokenizer_revision"],
                  "mask_token_id": adapter.mask_token_id}
        names = [f"model.transformer.blocks.{layer}.ff_out" for layer in layers]
        statistics, training, development = [], [], []
        for group in ("statistics_train", "dev"):
            selected = partition[group]
            for key_ in selected:
                record, positions, captures = encoded[key_]
                source = {"source_dataset": dataset.stem, "probes": pool[key_]["sources"]}
                readout = {"input": record, "positions": positions,
                           "weights": [1 / (len(selected) * len(positions))] * len(positions),
                           "source": source, "input_id": key_}
                if group == "dev":
                    development.append(readout)
                else:
                    training.append(readout)
                    statistics.append({"input_id": key_, "source": source, "input": record,
                                       "capture_positions": captures,
                                       "weights_by_layer": {name: [policy["statistics_scale"] / (len(selected) * len(captures))] * len(captures)
                                                            for name in names}})
        recipe_id = "dataset_keep_" + key[:24]
        payloads = {
            "editing_cases.json": cases,
            "development_probes.json": [{"case_id": case["case_id"], "probe_id": probe["probe_id"]}
                                        for case in cases for probe in case["prompts"]["edit"]],
            "keep_statistics.json": {**common, "artifact_version": KEEP_INPUTS_VERSION,
                                     "recipe_id": recipe_id, "purpose": "keep_statistics",
                                     "mask_policy_id": "answer_length_full_mask_v1",
                                     "position_policy_id": "last_prompt_and_all_answer_masks_v1",
                                     "weight_policy_id": "equal_prompt_equal_position_explicit_scale_v1", "inputs": statistics},
            "keep_training.json": {**common, "artifact_version": KEEP_TRAINING_VERSION,
                                   "purpose": "keep_train", "inputs": training},
            "development_keep.json": {**common, "artifact_version": DEVELOPMENT_KEEP_VERSION,
                                      "purpose": "keep_dev", "inputs": development},
            "partitions.json": {"policy": policy, "preservation": partition,
                                "sources": {key_: pool[key_]["sources"] for key_ in pool},
                                "dev_edit_policy": "复用编辑题计算 P5 短答案损失；独立测试题留给后续评价"},
        }
        for key_ in partition["test"]:
            evaluation.append({"case_id": pool[key_]["sources"][0]["case_id"],
                               "probe_id": pool[key_]["sources"][0]["probe_id"], "role": "preservation_test",
                               "prompt": pool[key_]["prompt"], "answer": pool[key_]["answer"]})
        payloads["evaluation_inputs.json"] = {"status": "pending_evaluation", "inputs": evaluation}
        for filename, payload in payloads.items():
            write_json(output / filename, payload)
        if file_sha256(dataset) != source_digest or tokenizer_identity(model_path) != identity:
            raise ValueError("生成输入期间数据集或 tokenizer 文件发生变化")
        manifest.update(status="complete", recipe_id=recipe_id, fact_count=len(cases),
                        files={name: str(output / name) for name in payloads},
                        files_sha256={str(output / name): file_sha256(output / name) for name in payloads})
        write_json(manifest_path, manifest)
        return {**manifest, "cache_hit": False, "manifest": str(manifest_path)}
    except Exception as error:
        manifest.update(status="failed", failure=str(error))
        write_json(manifest_path, manifest)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--layers", type=int, nargs="+", default=[0])
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/cache"))
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    result = prepare_inputs(args.dataset, args.model_path, tuple(args.layers), args.cache_dir, use_cache=not args.no_cache)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
