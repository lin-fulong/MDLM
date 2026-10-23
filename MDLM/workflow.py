"""Single-round P5 orchestration on immutable, accepted P1–P4 artifacts."""

from __future__ import annotations

from dataclasses import asdict
import gc
import math
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from data.readers.json_cases import load_cases
from MDLM.acceptance import AcceptanceError, development_edits, gate_decision, select_merged_candidate
from MDLM.contracts import (AcceptanceConfig, DEVELOPMENT_KEEP_VERSION, MERGED_MODEL_VERSION,
                           P1RunConfig, RESIDUAL_CANDIDATE_VERSION, ResidualOptimizationConfig,
                           config_digest, file_sha256, local_model_identity)
from MDLM.merge import WeightMerge
from MDLM.optimize import KeepTrainingExample, log_probabilities
from model.adapters.llada_8b_instruct import LLaDA8BInstructAdapter
from model.base import KeepModelInput, tensor_sha256
from util.extract_reference_keys import write_json
from util.optimize_residuals import bind_model_files, load_fixed_factors
from util.validate_matrix_artifacts import checked_file, read_json, require


def load_residual_candidate(p1_dir: Path, residual_dir: Path, settings: AcceptanceConfig):
    """Validate algorithm artifact provenance, without validating raw dataset quality."""
    manifest_path = residual_dir / "manifest.json"
    sources = {str(manifest_path): file_sha256(manifest_path)}
    manifest = read_json(manifest_path)
    require(manifest["status"] == "complete" and manifest["artifact_version"] == RESIDUAL_CANDIDATE_VERSION
            and manifest["weights_merged"] is False, "P5 requires a complete audited P4 residual candidate")
    require(Path(manifest["source_p1_dir"]).resolve() == p1_dir, "P4 candidate belongs to another round")
    p4_settings = ResidualOptimizationConfig(**{**manifest["settings"],
                                               "checkpoints": tuple(manifest["settings"]["checkpoints"])})
    p4_settings.check()
    require(config_digest(p4_settings.as_record()) == manifest["settings_sha256"], "P4 settings digest differs")
    q_manifest, factors, bound, _ = load_fixed_factors(p1_dir, p4_settings)
    require(all(manifest[field] == q_manifest[field] for field in (
        "run_id", "round_id", "baseline_id", "p1_config_sha256", "fact_order", "target_layers", "model_identity"
    )), "P4 identity/order differs from the same-round Q factors")
    require(all(manifest["source_files_sha256"].get(path) == digest for path, digest in bound.items()),
            "P4 sources differ from same-round P1/P2/P3")
    sources.update(manifest["source_files_sha256"])
    for name, digest in (("optimization_report.json", manifest["optimization_report_sha256"]),
                         (manifest["tensor_file"], manifest["tensor_file_sha256"]),
                         (manifest["optimizer_file"], manifest["optimizer_file_sha256"])):
        path = residual_dir / name
        checked_file(path, digest)
        sources[str(path)] = digest
    report = read_json(residual_dir / "optimization_report.json")
    require(report["status"] == "complete" and report["failure"] is None and all(report[field] is True for field in (
        "baseline_unchanged", "fixed_q_unchanged", "zero_residual_replay_passed", "optimizer_disk_round_trip_passed"
    )), "P4 candidate QA did not pass")
    optimization = report["optimization"]
    require(optimization["status"] == "candidate_ready" and optimization["selected_step"] == manifest["selected_step"]
            and math.isfinite(optimization["selected_loss"]["total"])
            and optimization["selected_loss"]["total"] < optimization["baseline"]["total"],
            "P4 selected checkpoint did not improve J(0)")
    residuals = load_file(str(residual_dir / manifest["tensor_file"]))
    require(set(residuals) == set(factors) == set(manifest["layers"]), "P4 R/Q layer selection differs")
    for name, r in residuals.items():
        info = manifest["layers"][name]
        require(list(r.shape) == info["shape"] == [q_manifest["layers"][name]["out_features"], factors[name].shape[0]]
                and str(r.dtype) == info["dtype"] and r.dtype == factors[name].dtype and torch.isfinite(r).all()
                and tensor_sha256(factors[name]) == info["q_sha256"], "saved R shape/dtype or fixed Q binding differs")
    require(any(torch.count_nonzero(r) for r in residuals.values()), "P5 requires a nonzero P4 candidate")
    require(not (manifest["example_only"] or q_manifest["example_only"]) or settings.allow_example_only,
            "example-only P5 requires explicit permission")
    require(not q_manifest["example_only"] or manifest["example_only"], "P4 cannot relabel example factors as formal")
    require(manifest["example_only"] or manifest["p4_stage_complete"], "formal P5 requires P4 stage acceptance")
    for path, digest in sources.items():
        checked_file(Path(path), digest)
    return manifest, factors, residuals, sources


def save_and_reload(path: Path, tensors: dict) -> None:
    save_file({name: value.detach().cpu().contiguous() for name, value in tensors.items()}, str(path))
    saved = load_file(str(path))
    require(set(saved) == set(tensors) and all(saved[name].dtype == value.dtype and torch.equal(saved[name], value.cpu())
                                             for name, value in tensors.items()), "P5 tensor disk round-trip differs")


def prepare_development_reference(settings: AcceptanceConfig, config: P1RunConfig, factors: dict,
                                  output: Path, sources: dict):
    paths = [Path(settings.dev_edit_file).resolve(), Path(settings.dev_keep_file).resolve()]
    for path in paths:
        sources[str(path)] = file_sha256(path)
    cases, recipe = load_cases(paths[0]), read_json(paths[1])
    require(recipe["artifact_version"] == DEVELOPMENT_KEEP_VERSION and recipe["purpose"] == "keep_dev",
            "P5 requires explicitly frozen development preservation inputs")
    example_only = recipe["example_only"] or any(case["quality_control"]["status"] == "example_only" for case in cases)
    require(not example_only or settings.allow_example_only, "example-only development inputs require permission")
    identity = local_model_identity(settings.reference_model_path)
    require(identity["model_revision"] == settings.reference_model_revision
            and identity["tokenizer_revision"] == settings.reference_tokenizer_revision == config.tokenizer_revision
            == recipe["tokenizer_revision"], "P5 requires the frozen original theta_0 and matching tokenizer")
    bind_model_files(sources, settings.reference_model_path, identity)
    adapter = LLaDA8BInstructAdapter.from_local_tokenizer(model_path=settings.reference_model_path,
                                                       target_layers=config.target_layers)
    require(adapter.mask_token_id == recipe["mask_token_id"], "development preservation mask identity differs")
    edits = development_edits(cases, settings.dev_edit_probes, adapter)
    reference = adapter.load_model(settings.reference_model_path, torch_dtype=config.torch_dtype, device_map=config.device_map)
    adapter.inspect_target_modules(reference)
    originals = {name: reference.get_submodule(name).weight.detach().to("cpu", copy=True) for name in factors}
    keeps, cache = [], {}
    for index, row in enumerate(recipe["inputs"]):
        record = row["input"]
        model_input = KeepModelInput(tuple(record["input_ids"]), tuple(record["attention_mask"]), tuple(record["mask_positions"]))
        positions, weights = tuple(row["positions"]), tuple(row["weights"])
        # 公式（16）：同一合法 mask 位置、完整词表、非负归一权重。
        require(positions and set(positions) <= set(model_input.mask_positions) and len(weights) == len(positions)
                and all(model_input.input_ids[p] == adapter.mask_token_id for p in positions)
                and all(isinstance(w, (int, float)) and not isinstance(w, bool) and math.isfinite(w) and w >= 0 for w in weights),
                "development KL positions/weights differ from formula (16)")
        with torch.no_grad():
            logs = log_probabilities(adapter.forward_logits(reference, model_input, positions)).cpu()
        keeps.append(KeepTrainingExample(model_input, positions, weights, logs))
        cache[f"input_{index}"] = logs
    require(keeps and math.isclose(math.fsum(w for item in keeps for w in item.weights), 1.0, abs_tol=1e-12, rel_tol=1e-12),
            "development preservation weights must sum to one")
    for name, tensors in (("theta0_keep_log_probs.safetensors", cache), ("theta0_target_weights.safetensors", originals)):
        path = output / name
        save_and_reload(path, tensors)
        sources[str(path)] = file_sha256(path)
    write_json(output / "development_inputs.json", {"edit_protocol": settings.edit_protocol,
               "edit": [asdict(item) for item in edits], "keep": recipe, "example_only": bool(example_only)})
    sources[str(output / "development_inputs.json")] = file_sha256(output / "development_inputs.json")
    return reference, adapter, edits, keeps, originals, identity, bool(example_only)


def run_p5(p1_dir: str | Path, settings: AcceptanceConfig, *, residual_dir=None, output_dir=None) -> dict:
    settings.check()
    p1_dir = Path(p1_dir).resolve()
    residual_dir = Path(residual_dir).resolve() if residual_dir is not None else p1_dir.parent / "residuals"
    output = Path(output_dir).resolve() if output_dir is not None else p1_dir.parent / "editing"
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"artifact_version": MERGED_MODEL_VERSION, "status": "partial", "p5_stage_complete": False,
                "source_p1_dir": str(p1_dir), "source_residual_dir": str(residual_dir), "settings": settings.as_record(),
                "settings_sha256": config_digest(settings.as_record()), "formal_dataset_acceptance": False}
    report = {"status": "partial", "phase": "sources", "baseline_restored": False}
    merger = None
    write_json(output / "manifest.json", manifest)
    try:
        p4, factors, residuals, sources = load_residual_candidate(p1_dir, residual_dir, settings)
        config = P1RunConfig.from_record(read_json(p1_dir / "manifest.json")["config"])
        for field in ("run_id", "round_id", "baseline_id", "p1_config_sha256", "fact_order", "target_layers", "model_identity"):
            manifest[field] = p4[field]
        require(config.model_name == "LLaDA-8B-Instruct", "P5 currently supports LLaDA-8B-Instruct")
        report["phase"] = "original_reference"
        print("P5: preparing original theta_0 weights and development KL reference...", flush=True)
        model, adapter, edits, keeps, originals, identity, example = prepare_development_reference(
            settings, config, factors, output, sources)
        if p4["keep_reference"] is not None:
            require(p4["keep_reference"]["model_identity"] == identity, "P5 original theta_0 differs from P4 keep reference")
        manifest.update(example_only=bool(p4["example_only"] or example), reference_identity=identity)
        require(local_model_identity(config.model_path) == p4["model_identity"], "P5 baseline identity differs")
        bind_model_files(sources, config.model_path, p4["model_identity"])
        if identity != p4["model_identity"]:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            adapter = LLaDA8BInstructAdapter.from_local_tokenizer(model_path=config.model_path, target_layers=config.target_layers)
            model = adapter.load_model(config.model_path, torch_dtype=config.torch_dtype, device_map=config.device_map)
        adapter.inspect_target_modules(model)
        require(adapter.mask_token_id == read_json(p1_dir / "manifest.json")["mask_token_id"], "P5 baseline mask identity differs")
        merger = WeightMerge(model, factors, residuals, compute_dtype=settings.compute_dtype, chunk_size=settings.chunk_size)
        manifest["baseline_weight_sha256"] = merger.weight_hashes()
        key_path = p1_dir.parent / "matrices/reference_keys"
        keys = load_file(str(key_path / read_json(key_path / "manifest.json")["tensor_file"]))
        report["reference_key_diagnostics"] = merger.diagnostics(factors, residuals, keys)
        del keys
        report["phase"] = "common_alpha_trials"
        print("P5: actual merge equivalence and development gates...", flush=True)
        try:
            report["selection"] = select_merged_candidate(merger, adapter, factors, residuals, edits, keeps, originals, settings)
        except AcceptanceError as error:
            report["selection"] = error.report
            raise
        report["phase"] = "save_and_reload"
        selection = report["selection"]
        weights = {name: weight.detach().cpu().clone() for name, weight in merger.parameters.items()}
        for name, tensors in (("final_target_weights.safetensors", weights), ("candidate_increments.safetensors", merger.increments)):
            save_and_reload(output / name, tensors)
        with torch.no_grad():
            final_logits = {f"input_{index}": adapter.forward_logits(model, item.model_input, item.positions).cpu()
                            for index, item in enumerate((*edits, *keeps))}
        save_and_reload(output / "final_dev_logits.safetensors", final_logits)
        manifest.update(tensor_file="final_target_weights.safetensors",
                        tensor_file_sha256=file_sha256(output / "final_target_weights.safetensors"),
                        increment_file="candidate_increments.safetensors",
                        increment_file_sha256=file_sha256(output / "candidate_increments.safetensors"),
                        final_logits_file="final_dev_logits.safetensors",
                        final_logits_sha256=file_sha256(output / "final_dev_logits.safetensors"),
                        layers={name: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": tensor_sha256(value)}
                                for name, value in weights.items()},
                        selected_alpha=selection["selected_alpha"], temporary_branches_active=False,
                        weights_merged=selection["status"] == "accepted", source_files_sha256=sources)
        report["phase"] = "source_recheck"
        merger.check_integrity()
        for path, digest in sources.items():
            checked_file(Path(path), digest)
        require(config_digest(settings.as_record()) == manifest["settings_sha256"], "P5 settings changed during trials")
        require(merger.weight_hashes() == {name: row["sha256"] for name, row in manifest["layers"].items()},
                "final actual weights changed during saving")
        require(all(getattr(model.get_submodule(name), "_mdlm_candidate_owner", None) is None for name in factors),
                "temporary residual branches remain installed")
        report.update(status="complete", phase="complete", decision=selection["status"],
                      baseline_restored=selection["baseline_restored"], disk_round_trip_passed=True, failure=None)
        write_json(output / "acceptance_report.json", report)
        manifest.update(status=selection["status"], p5_stage_complete=not manifest["example_only"],
                        acceptance_report_sha256=file_sha256(output / "acceptance_report.json"))
        write_json(output / "manifest.json", manifest)
        return {"status": "complete", "decision": selection["status"], "selected_alpha": selection["selected_alpha"],
                "output_dir": str(output), "report": str(output / "acceptance_report.json"),
                "example_only": manifest["example_only"], "p5_stage_complete": manifest["p5_stage_complete"]}
    except Exception as error:
        if merger is not None:
            try:
                merger.restore()
                report["baseline_restored"] = True
            except Exception as recovery_error:
                report.update(baseline_restored=False, recovery_failure=str(recovery_error))
        report.update(status="failed", failure=str(error))
        manifest.update(status="failed", p5_stage_complete=False, weights_merged=False)
        write_json(output / "acceptance_report.json", report)
        write_json(output / "manifest.json", manifest)
        raise


def load_final_model(output_dir: str | Path):
    """Reload the immutable baseline plus saved actual target weights, with no factor hooks."""
    output = Path(output_dir).resolve()
    manifest_digest = file_sha256(output / "manifest.json")
    manifest = read_json(output / "manifest.json")
    require(manifest["artifact_version"] == MERGED_MODEL_VERSION and manifest["status"] in ("accepted", "rejected"),
            "P5 final model artifact did not complete")
    require(config_digest(manifest["settings"]) == manifest["settings_sha256"], "P5 reload settings digest differs")
    settings = AcceptanceConfig.from_record(manifest["settings"])
    checked_file(output / "acceptance_report.json", manifest["acceptance_report_sha256"])
    report = read_json(output / "acceptance_report.json")
    require(report["status"] == "complete" and report["decision"] == manifest["status"], "P5 acceptance report differs")
    selection, accepted = report["selection"], manifest["status"] == "accepted"
    trials = selection["trials"]
    require(selection["status"] == manifest["status"] and selection["selected_alpha"] == manifest["selected_alpha"]
            and manifest["weights_merged"] is accepted and manifest["temporary_branches_active"] is False
            and 0 < len(trials) <= len(settings.alphas)
            and [row["alpha"] for row in trials] == list(settings.alphas[:len(trials)]), "P5 reload selection differs")
    require((accepted and trials[-1]["accepted"] is True and all(row["accepted"] is False for row in trials[:-1])
             and trials[-1]["alpha"] == manifest["selected_alpha"])
            or (not accepted and manifest["selected_alpha"] is None and len(trials) == len(settings.alphas)
                and all(row["accepted"] is False for row in trials) and selection["baseline_restored"] is True),
            "P5 reload selection order/recovery differs")
    for row in trials:
        if "gate" in row:
            decision = gate_decision(row["metrics"], selection["baseline"]["f_dev"], settings)
            require(decision == row["gate"] and row["accepted"] == decision["passed"]
                    and row["equivalence"]["passed"] is True, "P5 reload selection gate differs")
        else:
            require(row["accepted"] is False and row.get("reasons"), "P5 reload selection has no rejection reason")
    reload_sources = {**manifest["source_files_sha256"], str(output / "manifest.json"): manifest_digest}
    for name, digest in (("acceptance_report.json", manifest["acceptance_report_sha256"]),
                         (manifest["tensor_file"], manifest["tensor_file_sha256"]),
                         (manifest["increment_file"], manifest["increment_file_sha256"]),
                         (manifest["final_logits_file"], manifest["final_logits_sha256"])):
        reload_sources[str(output / name)] = digest
    for path, digest in reload_sources.items():
        checked_file(Path(path), digest)
    weights = load_file(str(output / manifest["tensor_file"]))
    increments = load_file(str(output / manifest["increment_file"]))
    require(set(weights) == set(manifest["layers"]), "P5 saved target layers differ")
    config = P1RunConfig.from_record(read_json(Path(manifest["source_p1_dir"]) / "manifest.json")["config"])
    require(local_model_identity(config.model_path) == manifest["model_identity"], "P5 reload baseline identity differs")
    adapter = LLaDA8BInstructAdapter.from_local_tokenizer(model_path=config.model_path, target_layers=config.target_layers)
    model = adapter.load_model(config.model_path, torch_dtype=config.torch_dtype, device_map=config.device_map)
    whitelist = set(adapter.target_module_names())
    require(set(weights) == set(increments) == set(manifest["baseline_weight_sha256"]) == whitelist
            and tuple(manifest["target_layers"]) == config.target_layers, "P5 reload target whitelist differs")
    adapter.inspect_target_modules(model)
    for name, value in weights.items():
        weight, info = model.get_submodule(name).weight, manifest["layers"][name]
        require(list(value.shape) == info["shape"] == list(weight.shape) and str(value.dtype) == info["dtype"]
                and value.dtype == weight.dtype and torch.isfinite(value).all() and tensor_sha256(value) == info["sha256"]
                and tensor_sha256(weight) == manifest["baseline_weight_sha256"][name], "P5 actual target weight contract differs")
        increment = increments[name]
        require(increment.shape == weight.shape and increment.dtype == getattr(torch, settings.compute_dtype)
                and torch.isfinite(increment).all(), "P5 reload increment contract differs")
        for begin in range(0, weight.shape[1], settings.chunk_size):
            columns = slice(begin, begin + settings.chunk_size)
            baseline = weight[:, columns].detach().cpu()
            expected_weight = (baseline.to(increment.dtype) + manifest["selected_alpha"] * increment[:, columns]).to(weight.dtype) if accepted else baseline
            require(torch.equal(value[:, columns], expected_weight), "P5 final weights violate the selected merge formula")
    snapshot = adapter.snapshot_target_weights(model)
    buffer_versions = {name: (id(b), b._version) for name, b in model.named_buffers()}
    try:
        with torch.no_grad():
            for name, value in weights.items():
                model.get_submodule(name).weight.copy_(value.to(model.get_submodule(name).weight.device))
        versions = {name: (id(p), p._version) for name, p in model.named_parameters()}
        checked_file(output / manifest["final_logits_file"], manifest["final_logits_sha256"])
        expected = load_file(str(output / manifest["final_logits_file"]))
        inputs = read_json(output / "development_inputs.json")
        readouts = [(row["model_input"], row["positions"]) for row in inputs["edit"]]
        readouts.extend((row["input"], row["positions"]) for row in inputs["keep"]["inputs"])
        require(set(expected) == {f"input_{i}" for i in range(len(readouts))}, "P5 reload readout coverage differs")
        for index, (record, positions) in enumerate(readouts):
            model_input = KeepModelInput(tuple(record["input_ids"]), tuple(record["attention_mask"]), tuple(record["mask_positions"]))
            with torch.no_grad():
                actual = adapter.forward_logits(model, model_input, tuple(positions)).cpu()
            require(actual.shape == expected[f"input_{index}"].shape and torch.allclose(actual, expected[f"input_{index}"], atol=manifest["settings"]["equivalence_atol"],
                                   rtol=manifest["settings"]["equivalence_rtol"]), "P5 reloaded actual model logits differ")
        if versions != {name: (id(p), p._version) for name, p in model.named_parameters()} or (
            buffer_versions != {name: (id(b), b._version) for name, b in model.named_buffers()}
        ) or any(tensor_sha256(model.get_submodule(name).weight) != manifest["layers"][name]["sha256"] for name in weights):
            raise RuntimeError("P5 reloaded model integrity changed during forward")
        for path, digest in reload_sources.items():
            checked_file(Path(path), digest)
    except Exception:
        adapter.restore_target_weights(model, snapshot)
        raise
    return adapter, model
