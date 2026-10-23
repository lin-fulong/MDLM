"""Extract formula (4) K from the saved P1 full-mask initial inputs only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from MDLM.contracts import (
    P1_ARTIFACT_VERSION, REFERENCE_KEYS_VERSION, P1RunConfig, ReferenceKeyConfig,
    config_digest, file_sha256, local_model_identity,
)
from MDLM.states import build_supervision, input_record, load_supervision, load_trajectories
from model.adapters.llada_8b_instruct import LLaDA8BInstructAdapter
from model.base import tensor_sha256


def write_json(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_reference_inputs(p1_dir: Path):
    """Check frozen algorithm artifacts; never reopen or validate the dataset."""
    names = ("manifest.json", "trajectories.jsonl", "collection_report.json",
             "fixed_states.jsonl", "coverage_report.json")
    sources = {name: file_sha256(p1_dir / name) for name in names}
    with (p1_dir / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest["artifact_version"] != P1_ARTIFACT_VERSION or manifest["collection_status"] != "complete":
        raise ValueError("P1 collection is incomplete or uses an unsupported artifact version")
    config = P1RunConfig.from_record(manifest["config"])
    if config.digest() != manifest["config_sha256"]:
        raise ValueError("P1 configuration digest differs")
    for name in ("trajectories", "collection_report"):
        suffix = ".jsonl" if name == "trajectories" else ".json"
        if sources[name + suffix] != manifest[name + "_sha256"]:
            raise ValueError(f"P1 {name} digest differs")
    if manifest["forward_conditions"] != {
        "attention_mask": "all_ones", "position_ids": "implicit_zero_based",
        "attention_bias": None, "use_cache": False, "model_mode": "eval",
    }:
        raise ValueError("P1 forward conditions differ from the LLaDA replay contract")
    trajectories = load_trajectories(p1_dir / "trajectories.jsonl")
    with (p1_dir / "collection_report.json").open(encoding="utf-8") as stream:
        collection = json.load(stream)
    with (p1_dir / "coverage_report.json").open(encoding="utf-8") as stream:
        coverage = json.load(stream)
    selected = [(fact.case_id, fact.probe_id) for fact in config.facts]
    if ([(trace.case_id, trace.probe_id) for trace in trajectories] != selected
            or manifest["fact_order"] != [fact.case_id for fact in config.facts]
            or [(row["case_id"], row["probe_id"]) for row in collection["completed"]] != selected):
        raise ValueError("P1 fact/probe order differs from the frozen column selection")
    if collection["failure"] is not None or any(not row["forward_logits_finite"] for row in collection["completed"]):
        raise ValueError("P1 collection did not complete with finite logits")
    if coverage["status"] != "complete" or not coverage["formula_checks_passed"]:
        raise ValueError("P1 fixed supervision did not pass formula checks")
    for field in ("run_id", "round_id", "baseline_id"):
        if coverage[field] != manifest["config"][field]:
            raise ValueError(f"P1 coverage {field} differs")
    if (coverage["config_sha256"] != config.digest()
            or coverage["trajectories_sha256"] != sources["trajectories.jsonl"]):
        raise ValueError("P1 coverage refers to different inputs")
    # 复算只用于核对冻结状态；不发布新监督，也不改变 P1 的集合或权重。
    fixed, expected_coverage = build_supervision(trajectories, manifest["mask_token_id"])
    if load_supervision(p1_dir / "fixed_states.jsonl") != fixed or any(
        coverage[key] != value for key, value in expected_coverage.items()
    ):
        raise ValueError("P1 saved supervision or coverage differs from the frozen trajectories")
    replayed = {index for index, row in enumerate(collection["completed"])
                if row["replayed"] and row["logits_replayed"]}
    if replayed != set(manifest["replay_fact_indices"]) or len(replayed) != config.replay_count:
        raise ValueError("P1 configured trajectory replay did not pass")
    # 覆盖报告的统计也须来自同一份采集报告，不能只核对公式部分。
    collection_counts = {
        "replayed_fact_count": sum(row["replayed"] for row in collection["completed"]),
        "logits_replayed_fact_count": sum(row["logits_replayed"] for row in collection["completed"]),
        "example_only_fact_count": sum(row["purpose"] == "example_only" for row in collection["completed"]),
    }
    if any(coverage.get(field) != count for field, count in collection_counts.items()):
        raise ValueError("P1 coverage counts differ from the collection report")
    return manifest, config, trajectories, sources, collection


def extract_run(p1_dir: str | Path, settings: ReferenceKeyConfig | None = None) -> dict:
    """Read theta_b once and capture/replay one reference column per fact/layer."""
    import torch
    from safetensors.torch import load_file, save_file

    settings = settings or ReferenceKeyConfig()
    settings.check()
    p1_dir = Path(p1_dir).resolve()
    output_dir = p1_dir.parent / "matrices" / "reference_keys"
    # 每轮只发布一次 K；失败目录同样保留，重新实验应使用新的 run_id。
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"artifact_version": REFERENCE_KEYS_VERSION, "status": "partial", "formula": 4,
                "settings": settings.as_record(), "settings_sha256": config_digest(settings.as_record()),
                "source_p1_dir": str(p1_dir), "p2_stage_complete": False,
                "reference_state": "P1 pre_forward_inputs[0]",
                "capture_point": "FFN output linear input at subject_last_position",
                "column_axis": "frozen fact order"}
    report = {"status": "partial", "phase": "p1_artifacts", "completed": []}
    manifest_path, report_path = output_dir / "manifest.json", output_dir / "qa_report.json"
    write_json(manifest_path, manifest)
    try:
        source, config, trajectories, source_hashes, collection = load_reference_inputs(p1_dir)
        if config.model_name != "LLaDA-8B-Instruct":
            raise ValueError("reference key extraction currently supports only LLaDA-8B-Instruct")
        manifest.update({
            "run_id": config.run_id, "round_id": config.round_id, "baseline_id": config.baseline_id,
            "p1_config_sha256": config.digest(), "source_files_sha256": source_hashes,
            "source_config": config.as_record(), "forward_conditions": source["forward_conditions"],
        })
        report["phase"] = "baseline_identity"
        print("Checking local baseline model and tokenizer file hashes...", flush=True)
        identity = local_model_identity(config.model_path)
        if (identity["model_revision"] != config.model_revision
                or identity["tokenizer_revision"] != config.tokenizer_revision):
            raise ValueError("P1 model/tokenizer revision is not this local snapshot; recollect with frozen file hashes")
        model_path = Path(config.model_path)
        if (file_sha256(model_path / "config.json") != source["model_config_sha256"]
                or file_sha256(model_path / "tokenizer.json") != source["tokenizer_sha256"]):
            raise ValueError("local model/tokenizer files differ from P1 collection")
        manifest["model_identity"] = identity
        write_json(manifest_path, manifest)
        report["phase"] = "model_loading"
        adapter = LLaDA8BInstructAdapter.from_local_tokenizer(
            model_path=config.model_path, target_layers=config.target_layers
        )
        if adapter.mask_token_id != source["mask_token_id"]:
            raise ValueError("local mask token ID differs from P1")
        model = adapter.load_model(config.model_path, torch_dtype=config.torch_dtype, device_map=config.device_map)
        descriptions = adapter.inspect_target_modules(model)
        manifest["runtime"] = {"torch": str(torch.__version__),
                               "cuda_runtime": torch.version.cuda,
                               "device_map": {name: str(device) for name, device in getattr(model, "hf_device_map", {}).items()}}
        columns = {name: [] for name in descriptions}
        capture_dtypes = {}
        column_records = []
        report["phase"] = "capture_and_replay"
        for column, trajectory in enumerate(trajectories):
            report["current_case_id"] = trajectory.case_id
            initial = trajectory.trace.pre_forward_inputs[0]
            # 公式（4）：直接重放 P1 全 mask 初态，不重分词、不采样、不回填答案。
            captured = adapter.forward_with_capture(model, initial)
            if tensor_sha256(captured.mask_logits) != trajectory.trace.mask_logit_sha256[0]:
                raise ValueError(f"{trajectory.case_id}: initial logits differ from the P1 baseline")
            replay = adapter.forward_with_capture(model, initial)
            if tensor_sha256(replay.mask_logits) != trajectory.trace.mask_logit_sha256[0]:
                raise ValueError(f"{trajectory.case_id}: replay logits differ from the P1 baseline")
            if set(captured.module_inputs) != set(descriptions) or set(replay.module_inputs) != set(descriptions):
                raise ValueError("captured module names differ from the frozen target layers")
            differences = {}
            for name, info in descriptions.items():
                key, repeated = captured.module_inputs[name], replay.module_inputs[name]
                if (tuple(key.shape) != (info["in_features"],) or repeated.shape != key.shape
                        or not torch.isfinite(key).all() or not torch.isfinite(repeated).all()):
                    raise ValueError(f"{trajectory.case_id}: invalid reference key shape or finite values at {name}")
                if not torch.allclose(key, repeated, atol=settings.replay_atol, rtol=settings.replay_rtol):
                    raise ValueError(f"{trajectory.case_id}: reference key did not replay at {name}")
                dtype = str(key.dtype)
                if name in capture_dtypes and capture_dtypes[name] != dtype:
                    raise ValueError(f"capture dtype changed between facts at {name}")
                capture_dtypes[name] = dtype
                differences[name] = float((key.double() - repeated.double()).abs().max())
                columns[name].append(key.detach().to(device="cpu", dtype=getattr(torch, settings.storage_dtype)))
            column_records.append({
                "column": column, "case_id": trajectory.case_id, "probe_id": trajectory.probe_id,
                "subject_last_position": initial.subject_last_position,
                "mask_positions": list(initial.mask_positions),
                "initial_input_sha256": config_digest(input_record(initial)),
                "initial_mask_logits_sha256": trajectory.trace.mask_logit_sha256[0],
            })
            report["completed"].append({"case_id": trajectory.case_id, "probe_id": trajectory.probe_id,
                                        "initial_logits_match_p1": True, "key_replay_max_abs": differences})
        # 每层按事实列序堆叠为 d_in × N；不对 token、状态或事实求平均。
        keys = {name: torch.stack(values, dim=1).contiguous() for name, values in columns.items()}
        layers = {}
        for name, matrix in keys.items():
            if matrix.shape != (descriptions[name]["in_features"], len(trajectories)) or not torch.isfinite(matrix).all():
                raise ValueError(f"invalid K matrix at {name}")
            layers[name] = {**descriptions[name], "shape": list(matrix.shape),
                            "capture_dtype": capture_dtypes[name], "storage_dtype": str(matrix.dtype),
                            "column_norms": matrix.double().norm(dim=0).tolist()}
        report["phase"] = "save_and_reload"
        tensor_path = output_dir / "keys.safetensors"
        # 独立目录中直接保存；仅 manifest.status=complete 表示可用，避免挂载盘重命名可见性问题。
        save_file(keys, str(tensor_path), metadata={"artifact_version": REFERENCE_KEYS_VERSION,
                                                   "p1_config_sha256": config.digest()})
        saved = load_file(str(tensor_path), device="cpu")
        if set(saved) != set(keys) or any(not torch.equal(saved[name], value) for name, value in keys.items()):
            raise ValueError("saved reference keys did not round-trip exactly")
        if any(file_sha256(p1_dir / name) != digest for name, digest in source_hashes.items()):
            raise ValueError("P1 artifacts changed during key extraction")
        tensor_digest = file_sha256(tensor_path)
        report.update({"status": "complete", "phase": "complete", "fact_count": len(trajectories),
                       "layer_count": len(keys), "forward_count": 2 * len(trajectories),
                       "disk_round_trip_passed": True, "failure": None})
        write_json(report_path, report)
        manifest.update({
            "status": "complete", "columns": column_records, "layers": layers,
            "tensor_file": tensor_path.name, "tensor_file_sha256": tensor_digest,
            "qa_report_sha256": file_sha256(report_path),
            "example_only_fact_count": sum(row["purpose"] == "example_only" for row in collection["completed"]),
        })
        write_json(manifest_path, manifest)
    except Exception as error:
        manifest["status"] = "failed"
        report.update({"status": "failed", "failure": str(error)})
        write_json(report_path, report)
        write_json(manifest_path, manifest)
        raise
    return {"status": "complete", "output_dir": str(output_dir), "fact_count": len(trajectories),
            "layers": layers, "forward_count": report["forward_count"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("p1_dir", type=Path, help="completed runs/<run_id>/round_<n>/p1 directory")
    parser.add_argument("--config", type=Path, help="reference-key numeric choices; defaults match the example config")
    args = parser.parse_args()
    settings = ReferenceKeyConfig.from_file(args.config) if args.config else ReferenceKeyConfig()
    print(json.dumps(extract_run(args.p1_dir, settings), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
