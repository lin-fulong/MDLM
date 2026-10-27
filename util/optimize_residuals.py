"""Run P4 on accepted fixed P1/P2/P3 artifacts; save residuals without merging weights."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from MDLM.candidate import CandidateModel, require_differentiable_model
from MDLM.contracts import (
    KEEP_TRAINING_VERSION, RESIDUAL_CANDIDATE_VERSION, WRITEBACK_FACTOR_VERSION,
    P1RunConfig, ResidualOptimizationConfig, WritebackConfig, config_digest, file_sha256, local_model_identity,
)
from MDLM.optimize import (
    KeepTrainingExample, OptimizationError, log_probabilities, optimize_residuals, optimizer_states_equal,
)
from MDLM.states import edit_training_examples, load_supervision
from MDLM.writeback import relative_residual
from model.adapters.llada_8b_instruct import LLaDA8BInstructAdapter
from model.base import KeepModelInput, tensor_sha256
from util.io import checked_file, read_json, require, write_json
from util.solve_writeback_factor import load_matrix_bundle


def bind_model_files(sources: dict, model_path: str | Path, identity: dict) -> None:
    """Keep the originally read model/tokenizer hashes for the final source recheck."""
    for field in ("model_files_sha256", "tokenizer_files_sha256"):
        for name, digest in identity[field].items():
            path = str(Path(model_path).resolve() / name)
            require(path not in sources or sources[path] == digest, "model reference hashes conflict within this run")
            sources[path] = digest


def load_fixed_factors(p1_dir: Path, settings: ResidualOptimizationConfig):
    """Bind Q to its unchanged P2 inputs; old Q is accepted only after new residual checks."""
    directory = p1_dir.parent / "matrices/writeback_factors"
    sources = {str(directory / name): file_sha256(directory / name) for name in ("manifest.json", "qa_report.json")}
    q_manifest, qa = read_json(directory / "manifest.json"), read_json(directory / "qa_report.json")
    require(q_manifest["status"] == "complete" and q_manifest["artifact_version"] in (
        WRITEBACK_FACTOR_VERSION, "p3.writeback_factor.1"
    ), "P4 requires complete supported P3 factors")
    checked_file(directory / "qa_report.json", q_manifest["qa_report_sha256"])
    require(qa["status"] == "complete" and qa["failure"] is None, "P3 QA did not pass")
    q_settings = WritebackConfig(**q_manifest["settings"])
    q_settings.check()
    require(config_digest(q_settings.as_record()) == q_manifest["settings_sha256"], "P3 settings digest differs")
    bundle, keys, keep, bound_sources = load_matrix_bundle(p1_dir, q_settings)
    require(q_manifest["source_files_sha256"] == bound_sources, "P3 source binding differs from accepted P2")
    for field in ("run_id", "round_id", "baseline_id", "p1_config_sha256", "fact_order",
                  "target_layers", "model_identity", "example_only"):
        require(q_manifest[field] == bundle[field], f"P3 {field} differs from accepted P2")
    require(Path(q_manifest["source_p1_dir"]).resolve() == p1_dir, "P3 directory belongs to another run")
    require(set(q_manifest["layers"]) == set(keys["layers"]) == set(settings.rho_by_layer),
            "P4 rho layer selection differs from accepted Q")
    if bundle["example_only"]:
        require(settings.allow_example_only, "example-only inputs require allow_example_only=true")
    else:
        require(q_manifest["p3_stage_complete"] is True, "formal P4 requires completed P3 stage acceptance")
    sources.update(bound_sources)
    factors, legacy_checks = {}, {}
    for name, info in q_manifest["layers"].items():
        path = directory / info["tensor_file"]
        checked_file(path, info["tensor_file_sha256"])
        sources[str(path)] = info["tensor_file_sha256"]
        saved = load_file(str(path), device="cpu")
        require(set(saved) == {name}, "Q tensor layer names differ")
        q = saved[name]
        shape = [len(bundle["fact_order"]), keys["layers"][name]["in_features"]]
        require(list(q.shape) == info["shape"] == shape and str(q.dtype) == info["storage_dtype"]
                and q.dtype == getattr(torch, settings.residual_dtype) and torch.isfinite(q).all()
                and info["fixed_during_residual_optimization"] is True, "Q shape/dtype/fixed contract differs")
        require(all(info[field] == keys["layers"][name][field] for field in (
            "in_features", "out_features", "has_bias"
        )) and info["ridge"] == q_settings.ridge_by_layer[name], "Q dimensions or ridge differ")
        if q_manifest["artifact_version"] == "p3.writeback_factor.1":
            # 历史 v1 保持原样；用同轮原始 C/K 核对保存 Q 的两侧方程后才允许使用。
            from safetensors import safe_open
            matrices = p1_dir.parent / "matrices"
            with safe_open(str(matrices / "reference_keys" / keys["tensor_file"]), framework="pt") as stream:
                k = stream.get_tensor(name)
            c = load_file(str(matrices / "keep_statistics" / keep["layers"][name]["tensor_file"]))[name]
            residuals = [relative_residual(k, matrix, q.T, info["ridge"], epsilon=q_settings.residual_epsilon,
                                          chunk_size=q_settings.chunk_size) for matrix in (c, c.T)]
            require(all(math.isfinite(value) and value <= q_settings.stored_residual_tolerance for value in residuals),
                    "historical Q failed saved two-sided residual checks")
            legacy_checks[name] = {"stored_relative_residual": residuals[0],
                                   "stored_normal_equation_relative_residual": residuals[1]}
            del k, c
        else:
            row = qa["layers"][name]
            require(row["status"] == "complete" and row["disk_round_trip_passed"] is True and all(
                math.isfinite(row[field]) and row[field] <= tolerance
                for field, tolerance in (("relative_residual", q_settings.residual_tolerance),
                                         ("normal_equation_relative_residual", q_settings.residual_tolerance),
                                         ("stored_relative_residual", q_settings.stored_residual_tolerance),
                                         ("stored_normal_equation_relative_residual", q_settings.stored_residual_tolerance))
            ), "P3 two-sided residual QA did not pass")
        factors[name] = q
    for path, digest in sources.items():
        checked_file(Path(path), digest)
    return q_manifest, factors, sources, legacy_checks


def prepare_keep_training(settings: ResidualOptimizationConfig, config: P1RunConfig,
                          output_dir: Path, sources: dict) -> tuple[list[KeepTrainingExample], dict | None]:
    if settings.beta == 0:
        return [], None
    path = Path(settings.keep_train_file).resolve()
    sources[str(path)] = file_sha256(path)
    recipe = read_json(path)
    # 这是新产生的保持训练计算合同；原始案例结构不变，不做数据集质量/泄漏检查。
    require(recipe["artifact_version"] == KEEP_TRAINING_VERSION and recipe["purpose"] == "keep_train",
            "P4 requires explicitly designated keep-training inputs")
    require(type(recipe["example_only"]) is bool, "keep-training purpose must be explicit")
    require(not recipe["example_only"] or settings.allow_example_only, "example-only keep-training inputs are disabled")
    identity = local_model_identity(settings.reference_model_path)
    require(identity["model_revision"] == settings.reference_model_revision
            and identity["tokenizer_revision"] == settings.reference_tokenizer_revision == config.tokenizer_revision,
            "keep reference must be the explicitly frozen original theta_0 with the same tokenizer")
    bind_model_files(sources, settings.reference_model_path, identity)
    reference_adapter = LLaDA8BInstructAdapter.from_local_tokenizer(
        model_path=settings.reference_model_path, target_layers=config.target_layers)
    reference_model = reference_adapter.load_model(settings.reference_model_path,
                                                  torch_dtype=config.torch_dtype, device_map=config.device_map)
    examples, tensors = [], {}
    for index, row in enumerate(recipe["inputs"]):
        record = row["input"]
        model_input = KeepModelInput(tuple(record["input_ids"]), tuple(record["attention_mask"]),
                                     tuple(record["mask_positions"]))
        positions, weights = tuple(row["positions"]), tuple(row["weights"])
        require(len(weights) == len(positions) and weights and all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
            for value in weights), "keep training readout weights must be finite nonnegative and position-aligned")
        with torch.no_grad():
            log_probs = log_probabilities(reference_adapter.forward_logits(reference_model, model_input, positions)).cpu()
        examples.append(KeepTrainingExample(model_input, positions, weights, log_probs))
        tensors[f"input_{index}"] = log_probs.contiguous()
    require(examples and math.isclose(math.fsum(w for item in examples for w in item.weights),
                                      1.0, rel_tol=1e-12, abs_tol=1e-12), "keep training weights must sum to one")
    del reference_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    checked_file(path, sources[str(path)])
    cache_path = output_dir / "theta0_keep_log_probs.safetensors"
    save_file(tensors, str(cache_path))
    cache = load_file(str(cache_path))
    require(set(cache) == set(tensors) and all(cache[name].dtype == value.dtype and torch.equal(cache[name], value)
                                             for name, value in tensors.items()), "theta_0 cache did not round-trip exactly")
    sources[str(cache_path)] = file_sha256(cache_path)
    return examples, {"model_identity": identity, "source_file": str(path), "example_only": recipe["example_only"],
                      "cache_file": cache_path.name, "cache_sha256": sources[str(cache_path)],
                      "input_count": len(examples), "kl_direction": "theta_0 || candidate", "full_vocabulary": True}


def optimize_run(p1_dir: str | Path, settings: ResidualOptimizationConfig,
                 *, output_dir: str | Path | None = None) -> dict:
    settings.check()
    p1_dir = Path(p1_dir).resolve()
    output = Path(output_dir).resolve() if output_dir is not None else p1_dir.parent / "residuals"
    output.mkdir(parents=True, exist_ok=False)
    manifest_path, report_path = output / "manifest.json", output / "optimization_report.json"
    manifest = {"artifact_version": RESIDUAL_CANDIDATE_VERSION, "status": "partial",
                "source_p1_dir": str(p1_dir), "settings": settings.as_record(),
                "settings_sha256": config_digest(settings.as_record()), "p4_stage_complete": False,
                "weights_merged": False, "formal_dataset_acceptance": False}
    report = {"status": "partial", "phase": "source_artifacts", "zero_residual_replay_forward_count": 0}
    write_json(manifest_path, manifest)
    try:
        q_manifest, factors, sources, legacy_checks = load_fixed_factors(p1_dir, settings)
        for field in ("run_id", "round_id", "baseline_id", "p1_config_sha256", "fact_order",
                      "target_layers", "model_identity", "example_only"):
            manifest[field] = q_manifest[field]
        config = P1RunConfig.from_record(read_json(p1_dir / "manifest.json")["config"])
        require(config.model_name == "LLaDA-8B-Instruct", "P4 currently supports LLaDA-8B-Instruct")
        edits = edit_training_examples(load_supervision(p1_dir / "fixed_states.jsonl"))
        manifest.update(source_files_sha256=sources, legacy_q_checks=legacy_checks)
        write_json(manifest_path, manifest)
        report["phase"] = "original_reference"
        keeps, reference = prepare_keep_training(settings, config, output, sources)
        if reference is not None:
            manifest["example_only"] = manifest["example_only"] or reference["example_only"]
        manifest["keep_reference"] = reference
        report["phase"] = "baseline_identity"
        print("P4: checking frozen baseline, then loading the real candidate model...", flush=True)
        require(local_model_identity(config.model_path) == q_manifest["model_identity"], "P4 local baseline identity differs")
        bind_model_files(sources, config.model_path, q_manifest["model_identity"])
        adapter = LLaDA8BInstructAdapter.from_local_tokenizer(model_path=config.model_path, target_layers=config.target_layers)
        require(adapter.mask_token_id == read_json(p1_dir / "manifest.json")["mask_token_id"], "P4 mask identity differs")
        model = adapter.load_model(config.model_path, torch_dtype=config.torch_dtype, device_map=config.device_map)
        adapter.inspect_target_modules(model)
        # 禁止推理专用 offload：其 no_grad 包装会切断后续冻结层对早期 R 的梯度。
        require_differentiable_model(model)
        versions = {name: (id(parameter), parameter._version) for name, parameter in model.named_parameters()}
        weight_hashes = {name: tensor_sha256(model.get_submodule(name).weight) for name in factors}
        with torch.no_grad():
            original_logits = [adapter.forward_logits(model, item.model_input, item.positions).cpu() for item in edits]
        report["zero_residual_replay_forward_count"] += len(edits)
        report["phase"] = "zero_residual_replay"
        manifest["runtime"] = {"torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
                               "model_dtype": str(next(model.parameters()).dtype),
                               "target_devices": {name: str(model.get_submodule(name).weight.device) for name in factors}}
        with CandidateModel(model, factors, dtype=settings.residual_dtype) as candidate:
            with torch.no_grad():
                for example, expected in zip(edits, original_logits):
                    actual = adapter.forward_logits(model, example.model_input, example.positions).cpu()
                    require(torch.allclose(actual, expected, atol=settings.equivalence_atol, rtol=settings.equivalence_rtol),
                            "R=0 logits differ from baseline")
                    report["zero_residual_replay_forward_count"] += 1
            report["zero_residual_replay_passed"] = True
            q_hashes = {name: tensor_sha256(branch.q) for name, branch in candidate.branches.items()}
            report["phase"] = "joint_optimization"
            write_json(report_path, report)

            def progress(record):
                report["optimization"] = record
                write_json(report_path, report)
                step = record["history"][-1]
                if "loss" in step:
                    print(f"P4 step {step['step']}: J={step['loss']['total']:.8g}", flush=True)

            try:
                result = optimize_residuals(candidate, adapter, edits, keeps, settings, on_progress=progress)
            except OptimizationError as error:
                report["optimization"] = error.report
                raise
            report["optimization"] = result.report
            require(all(tensor_sha256(branch.q) == q_hashes[name] for name, branch in candidate.branches.items()),
                    "fixed Q changed during residual optimization")
            report["fixed_q_unchanged"] = True
            if result.residuals is not None:
                report["phase"] = "save_and_reload"
                path = output / "residuals.safetensors"
                save_file(result.residuals, str(path), metadata={"artifact_version": RESIDUAL_CANDIDATE_VERSION,
                          "p1_config_sha256": manifest["p1_config_sha256"], "settings_sha256": manifest["settings_sha256"]})
                saved = load_file(str(path))
                require(set(saved) == set(result.residuals) and all(saved[name].dtype == value.dtype
                        and torch.equal(saved[name], value) for name, value in result.residuals.items()),
                        "saved R did not round-trip exactly")
                torch.save(result.optimizer_state, output / "optimizer_state.pt")
                saved_optimizer = torch.load(output / "optimizer_state.pt", map_location="cpu", weights_only=True)
                require(optimizer_states_equal(saved_optimizer, result.optimizer_state), "optimizer state did not round-trip exactly")
                report["optimizer_disk_round_trip_passed"] = True
                manifest.update(tensor_file=path.name, tensor_file_sha256=file_sha256(path),
                                optimizer_file="optimizer_state.pt", optimizer_file_sha256=file_sha256(output / "optimizer_state.pt"),
                                selected_step=result.report["selected_step"],
                                layers={name: {"shape": list(value.shape), "dtype": str(value.dtype),
                                              "q_sha256": q_hashes[name]} for name, value in saved.items()})
        require({name: (id(parameter), parameter._version) for name, parameter in model.named_parameters()} == versions
                and all(tensor_sha256(model.get_submodule(name).weight) == digest for name, digest in weight_hashes.items()),
                "baseline parameters changed during P4")
        report["baseline_unchanged"] = True
        report["phase"] = "source_recheck"
        for path, digest in sources.items():
            checked_file(Path(path), digest)
        require(config_digest(settings.as_record()) == manifest["settings_sha256"], "P4 settings changed during optimization")
        ready = result.residuals is not None
        report.update(status="complete" if ready else "no_candidate", phase="complete", failure=result.report["failure"])
        write_json(report_path, report)
        manifest.update(status="complete" if ready else "rejected", optimization_report_sha256=file_sha256(report_path),
                        p4_stage_complete=ready and not manifest["example_only"])
        write_json(manifest_path, manifest)
        return {"status": report["status"], "output_dir": str(output), "report": str(report_path),
                "example_only": manifest["example_only"], "p4_stage_complete": manifest["p4_stage_complete"]}
    except Exception as error:
        report.update(status="failed", failure=str(error))
        manifest["status"] = "failed"
        write_json(report_path, report)
        write_json(manifest_path, manifest)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("p1_dir", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="new directory for a repeated run; existing outputs are never overwritten")
    args = parser.parse_args()
    result = optimize_run(args.p1_dir, ResidualOptimizationConfig.from_file(args.config), output_dir=args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
