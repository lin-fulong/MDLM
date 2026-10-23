"""Extract formula (5) uncentered C from an explicit frozen keep-input recipe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from MDLM.contracts import (
    KEEP_INPUTS_VERSION, KEEP_STATISTICS_VERSION, REFERENCE_KEYS_VERSION,
    KeepStatisticsConfig, config_digest, file_sha256, local_model_identity,
)
from MDLM.keep_statistics import SecondMomentAccumulator
from model.adapters.llada_8b_instruct import LLaDA8BInstructAdapter
from model.base import KeepModelInput, tensor_sha256
from util.extract_reference_keys import load_reference_inputs, write_json


def keep_input_from_record(record: dict) -> KeepModelInput:
    if record["implicit_position_ids"] != list(range(len(record["input_ids"]))):
        raise ValueError("keep implicit position IDs differ from the frozen forward contract")
    return KeepModelInput(tuple(record["input_ids"]), tuple(record["attention_mask"]),
                          tuple(record["mask_positions"]))


def load_key_reference(p1_dir: Path, config, source_hashes: dict) -> tuple[dict, dict]:
    """Bind C to the same completed K, baseline, layers and P1 inputs."""
    key_dir = p1_dir.parent / "matrices/reference_keys"
    with (key_dir / "manifest.json").open(encoding="utf-8") as stream:
        keys = json.load(stream)
    if keys["artifact_version"] != REFERENCE_KEYS_VERSION or keys["status"] != "complete":
        raise ValueError("reference K is incomplete or uses an unsupported artifact version")
    if (keys["p1_config_sha256"] != config.digest() or keys["source_files_sha256"] != source_hashes
            or any(keys[field] != getattr(config, field) for field in ("run_id", "round_id", "baseline_id"))):
        raise ValueError("reference K belongs to a different P1 run or baseline")
    expected_names = {f"model.transformer.blocks.{layer}.ff_out" for layer in config.target_layers}
    if set(keys["layers"]) != expected_names:
        raise ValueError("reference K target layers differ from the frozen selection")
    paths = {key_dir / "manifest.json": file_sha256(key_dir / "manifest.json")}
    for filename, field in ((keys["tensor_file"], "tensor_file_sha256"), ("qa_report.json", "qa_report_sha256")):
        path = key_dir / filename
        paths[path] = file_sha256(path)
        if paths[path] != keys[field]:
            raise ValueError(f"reference K {filename} digest differs")
    with (key_dir / "qa_report.json").open(encoding="utf-8") as stream:
        if json.load(stream)["status"] != "complete":
            raise ValueError("reference K QA did not pass")
    return keys, paths


def extract_run(p1_dir: str | Path, keep_file: str | Path, settings: KeepStatisticsConfig) -> dict:
    """Accumulate one layer at a time on CPU; the model remains read-only."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    settings.check()
    p1_dir, keep_file = Path(p1_dir).resolve(), Path(keep_file).resolve()
    output_dir = p1_dir.parent / "matrices/keep_statistics"
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path, report_path = output_dir / "manifest.json", output_dir / "qa_report.json"
    manifest = {"artifact_version": KEEP_STATISTICS_VERSION, "status": "partial", "formula": 5,
                "settings": settings.as_record(), "settings_sha256": config_digest(settings.as_record()),
                "source_p1_dir": str(p1_dir), "source_keep_file": str(keep_file),
                "centered": False, "normalization": "none", "p2_stage_complete": False,
                "memory_policy": "one_layer_at_a_time_cpu_accumulation_and_chunked_QA"}
    report = {"status": "partial", "phase": "source_artifacts", "forward_count": 0, "layers": {}}
    write_json(manifest_path, manifest)
    try:
        source, config, trajectories, source_hashes, _ = load_reference_inputs(p1_dir)
        if config.model_name != "LLaDA-8B-Instruct":
            raise ValueError("keep statistics currently support only LLaDA-8B-Instruct")
        keys, key_hashes = load_key_reference(p1_dir, config, source_hashes)
        keep_digest = file_sha256(keep_file)
        with keep_file.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        # 核对的是保持统计的冻结配方/用途，不扫描原数据或调用 schema.py。
        if payload["artifact_version"] != KEEP_INPUTS_VERSION or payload["purpose"] != "keep_statistics":
            raise ValueError("expected a frozen keep_statistics input artifact")
        if payload["recipe_id"] != settings.recipe_id:
            raise ValueError("keep recipe differs from the selected statistics configuration")
        if (payload["tokenizer_revision"] != config.tokenizer_revision
                or payload["mask_token_id"] != source["mask_token_id"]):
            raise ValueError("keep tokenizer or mask ID differs from the frozen baseline")
        if not payload["inputs"]:
            raise ValueError("no keep inputs were selected")
        names = set(keys["layers"])
        records = []
        for row in payload["inputs"]:
            if set(row["weights_by_layer"]) != names:
                raise ValueError("explicit keep weight layers differ from K target layers")
            records.append((row, keep_input_from_record(row["input"]), tuple(row["capture_positions"])))
        write_json(output_dir / "keep_inputs.json", payload)
        manifest.update({
            "run_id": config.run_id, "round_id": config.round_id, "baseline_id": config.baseline_id,
            "p1_config_sha256": config.digest(), "source_files_sha256": source_hashes,
            "key_files_sha256": {str(path): value for path, value in key_hashes.items()},
            "keep_inputs_sha256": keep_digest, "source_config": config.as_record(),
            "recipe": {field: payload[field] for field in (
                "recipe_id", "purpose", "example_only", "mask_policy_id", "position_policy_id", "weight_policy_id")},
            "example_only": bool(payload["example_only"] or keys["example_only_fact_count"]),
            "forward_conditions": source["forward_conditions"],
        })
        write_json(manifest_path, manifest)
        report["phase"] = "baseline_identity"
        print("Checking C baseline model and tokenizer file hashes...", flush=True)
        identity = local_model_identity(config.model_path)
        if (identity["model_revision"] != config.model_revision
                or identity["tokenizer_revision"] != config.tokenizer_revision
                or identity != keys["model_identity"]):
            raise ValueError("keep statistics baseline differs from P1/K")
        manifest["model_identity"] = identity
        report["phase"] = "model_loading"
        adapter = LLaDA8BInstructAdapter.from_local_tokenizer(
            model_path=config.model_path, target_layers=config.target_layers)
        if adapter.mask_token_id != source["mask_token_id"]:
            raise ValueError("local model mask ID differs from P1/K")
        model = adapter.load_model(config.model_path, torch_dtype=config.torch_dtype, device_map=config.device_map)
        descriptions = adapter.inspect_target_modules(model)
        for name, info in descriptions.items():
            if any(keys["layers"][name][field] != value for field, value in info.items()):
                raise ValueError(f"K and C module dimensions differ at {name}")
        report["phase"] = "baseline_replay"
        reference = trajectories[0]
        report["forward_count"] += 1
        replay = adapter.forward_with_capture(model, reference.trace.pre_forward_inputs[0])
        if tensor_sha256(replay.mask_logits) != reference.trace.mask_logit_sha256[0]:
            raise ValueError("C baseline logits differ from the P1 initial state")
        report["baseline_initial_logits_match_p1"] = True
        manifest["runtime"] = {"torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
                               "device_map": {name: str(device) for name, device in getattr(model, "hf_device_map", {}).items()}}
        write_json(manifest_path, manifest)
        layers = {}
        for layer in config.target_layers:
            name = f"model.transformer.blocks.{layer}.ff_out"
            report["phase"], report["current_layer"] = "accumulation", name
            print(f"Accumulating C for {name} on CPU...", flush=True)
            # 按层重放保持输入，只驻留这一层的方阵，不保存全量激活。
            layer_adapter = LLaDA8BInstructAdapter(
                adapter.tokenizer, mask_token_id=adapter.mask_token_id, target_layers=(layer,))
            accumulator = SecondMomentAccumulator(
                descriptions[name]["in_features"], accumulation_dtype=settings.accumulation_dtype,
                projection_count=settings.projection_count, direct_block_size=settings.direct_block_size,
                seed=settings.qa_seed)
            completed = []
            groups = {}
            report["layers"][name] = {"status": "partial", "inputs": completed}
            capture_dtype = None
            for row, model_input, positions in records:
                report["current_input_id"] = row["input_id"]
                report["forward_count"] += 1
                captured = layer_adapter.forward_keep_with_capture(model, model_input, positions)
                report["forward_count"] += 1
                repeated = layer_adapter.forward_keep_with_capture(model, model_input, positions)
                if tensor_sha256(captured.mask_logits) != tensor_sha256(repeated.mask_logits):
                    raise ValueError(f"{row['input_id']}: keep mask logits did not replay")
                values, replay_values = captured.module_inputs[name], repeated.module_inputs[name]
                if values.shape != replay_values.shape or not torch.allclose(
                    values, replay_values, atol=settings.replay_atol, rtol=settings.replay_rtol
                ):
                    raise ValueError(f"{row['input_id']}: keep module inputs did not replay at {name}")
                if capture_dtype is not None and capture_dtype != str(values.dtype):
                    raise ValueError(f"keep capture dtype changed between inputs at {name}")
                capture_dtype = str(values.dtype)
                weights = row["weights_by_layer"][name]
                accumulator.add(values, weights)
                group = groups.setdefault(row["source"].get("source_dataset", "unspecified"),
                                          {"input_count": 0, "vector_count": 0, "weight_sum": 0.0})
                group["input_count"] += 1
                group["vector_count"] += len(positions)
                group["weight_sum"] += sum(weights)
                completed.append({
                    "input_id": row["input_id"], "source": row["source"],
                    "input_sha256": config_digest(row["input"]),
                    "capture_positions": list(positions), "weights": weights,
                    "weight_sum": sum(weights), "mask_count": len(model_input.mask_positions),
                    "mask_logits_sha256": tensor_sha256(captured.mask_logits),
                    "capture_sha256": tensor_sha256(values),
                    "capture_replay_max_abs": float((values.double() - replay_values.double()).abs().max()),
                })
            report["phase"] = "layer_numeric_QA"
            stored = accumulator.matrix.to(dtype=getattr(torch, settings.storage_dtype))
            diagnostics = accumulator.diagnose(
                stored, atol=settings.check_atol, rtol=settings.check_rtol, chunk_size=settings.chunk_size)
            report["layers"][name]["diagnostics"] = diagnostics
            if not diagnostics["all_checks_passed"]:
                raise ValueError(f"formula (5) numeric checks failed at {name}")
            report["phase"] = "save_and_reload"
            tensor_path = output_dir / f"c_layer_{layer}.safetensors"
            save_file({name: stored}, str(tensor_path), metadata={
                "artifact_version": KEEP_STATISTICS_VERSION, "recipe_id": settings.recipe_id,
                "p1_config_sha256": config.digest(),
            })
            # Safetensors 按行读取核对，避免为验收再复制完整的输入维方阵。
            with safe_open(str(tensor_path), framework="pt", device="cpu") as stream:
                if set(stream.keys()) != {name}:
                    raise ValueError("saved C module name differs")
                view = stream.get_slice(name)
                if view.get_shape() != list(stored.shape):
                    raise ValueError("saved C shape differs")
                for start in range(0, stored.shape[0], settings.chunk_size):
                    end = min(stored.shape[0], start + settings.chunk_size)
                    saved_rows = view[start:end]
                    # torch.equal 不比较 dtype，必须单独核对冻结的保存精度。
                    if saved_rows.dtype != stored.dtype:
                        raise ValueError("saved C dtype differs from the frozen storage precision")
                    if not torch.equal(saved_rows, stored[start:end]):
                        raise ValueError("saved C did not round-trip exactly")
            layers[name] = {
                **descriptions[name], "shape": list(stored.shape),
                "accumulation_dtype": settings.accumulation_dtype, "storage_dtype": settings.storage_dtype,
                "capture_dtype": capture_dtype, "vector_count": accumulator.vector_count,
                "positive_weight_count": accumulator.positive_weight_count, "weight_sum": accumulator.weight_sum,
                "zero_weight_count": accumulator.vector_count - accumulator.positive_weight_count,
                "source_groups": groups,
                "tensor_file": tensor_path.name, "tensor_file_sha256": file_sha256(tensor_path),
            }
            report["layers"][name].update({"status": "complete", "disk_round_trip_passed": True})
            write_json(report_path, report)
            del accumulator, stored
        report["phase"] = "source_recheck"
        if (file_sha256(keep_file) != keep_digest
                or any(file_sha256(p1_dir / name) != value for name, value in source_hashes.items())
                or any(file_sha256(path) != value for path, value in key_hashes.items())):
            raise ValueError("P1/K/keep input artifacts changed during C extraction")
        if file_sha256(output_dir / "keep_inputs.json") != keep_digest:
            # 输入 JSON 可有不同排版；按完整记录的规范化摘要核对冻结副本。
            with (output_dir / "keep_inputs.json").open(encoding="utf-8") as stream:
                if config_digest(json.load(stream)) != config_digest(payload):
                    raise ValueError("saved keep inputs differ from the frozen recipe")
        report.update({"status": "complete", "phase": "complete", "failure": None})
        write_json(report_path, report)
        manifest.update({"status": "complete", "layers": layers,
                         "saved_keep_inputs_sha256": file_sha256(output_dir / "keep_inputs.json"),
                         "qa_report_sha256": file_sha256(report_path)})
        write_json(manifest_path, manifest)
    except Exception as error:
        manifest["status"] = "failed"
        report.update({"status": "failed", "failure": str(error)})
        write_json(report_path, report)
        write_json(manifest_path, manifest)
        raise
    return {"status": "complete", "output_dir": str(output_dir), "layers": layers,
            "forward_count": report["forward_count"], "example_only": manifest["example_only"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("p1_dir", type=Path, help="completed P1 directory with matching K artifacts")
    parser.add_argument("keep_inputs", type=Path, help="frozen keep_statistics input JSON")
    parser.add_argument("--config", type=Path, required=True, help="explicit formula (5) recipe and numeric choices")
    args = parser.parse_args()
    report = extract_run(args.p1_dir, args.keep_inputs, KeepStatisticsConfig.from_file(args.config))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
