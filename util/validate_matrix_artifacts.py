"""Accept one frozen P1/K/C round after real-model replay and independent matrix QA."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from MDLM.contracts import (
    KEEP_INPUTS_VERSION, KEEP_STATISTICS_VERSION, MATRIX_BUNDLE_VERSION,
    REFERENCE_KEYS_VERSION, KeepStatisticsConfig, ReferenceKeyConfig,
    config_digest, file_sha256, local_model_identity,
)
from MDLM.states import input_record
from model.adapters.llada_8b_instruct import LLaDA8BInstructAdapter
from model.base import tensor_sha256
from util.extract_keep_statistics import keep_input_from_record
from util.extract_reference_keys import load_reference_inputs, write_json


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def checked_file(path: Path, digest: str) -> None:
    require(file_sha256(path) == digest, f"artifact digest differs: {path}")


def check_sources(p1_dir: Path, config, p1_manifest: dict, p1_hashes: dict) -> tuple[dict, dict, dict]:
    """Bind all saved artifacts to the same baseline, order, layers and recipe."""
    root = p1_dir.parent / "matrices"
    key_dir, keep_dir = root / "reference_keys", root / "keep_statistics"
    keys, keep = read_json(key_dir / "manifest.json"), read_json(keep_dir / "manifest.json")
    key_report, keep_report = read_json(key_dir / "qa_report.json"), read_json(keep_dir / "qa_report.json")
    require(keys["artifact_version"] == REFERENCE_KEYS_VERSION and keys["status"] == "complete"
            and keys["formula"] == 4, "K artifact is incomplete or has the wrong version/formula")
    require(keep["artifact_version"] == KEEP_STATISTICS_VERSION and keep["status"] == "complete"
            and keep["formula"] == 5, "C artifact is incomplete or has the wrong version/formula")
    require(key_report["status"] == keep_report["status"] == "complete"
            and key_report["disk_round_trip_passed"], "K/C extraction QA is incomplete")
    require(keep_report["baseline_initial_logits_match_p1"], "C baseline replay did not pass")
    names = [f"model.transformer.blocks.{layer}.ff_out" for layer in config.target_layers]
    for artifact in (keys, keep):
        require(all(artifact[field] == getattr(config, field) for field in ("run_id", "round_id", "baseline_id"))
                and artifact["p1_config_sha256"] == config.digest()
                and config_digest(artifact["source_config"]) == config.digest()
                and artifact["source_files_sha256"] == p1_hashes
                and artifact["forward_conditions"] == p1_manifest["forward_conditions"]
                and Path(artifact["source_p1_dir"]).resolve() == p1_dir
                and set(artifact["layers"]) == set(names), "P1/K/C run, baseline, order or layers differ")
    require(keys["settings_sha256"] == config_digest(keys["settings"])
            and keep["settings_sha256"] == config_digest(keep["settings"]), "matrix settings digest differs")
    key_settings = ReferenceKeyConfig(**keys["settings"])
    keep_settings = KeepStatisticsConfig(**keep["settings"])
    key_settings.check()
    keep_settings.check()
    require(keys["reference_state"] == "P1 pre_forward_inputs[0]"
            and keys["column_axis"] == "frozen fact order"
            and keep["centered"] is False and keep["normalization"] == "none",
            "K/C extraction definitions differ from formulas (4) and (5)")
    require(keys["model_identity"] == keep["model_identity"], "K/C baseline identities differ")
    identity = local_model_identity(config.model_path)
    require(identity == keys["model_identity"] and identity["model_revision"] == config.model_revision
            and identity["tokenizer_revision"] == config.tokenizer_revision,
            "local model/tokenizer baseline differs from P1/K/C")
    require(identity["model_files_sha256"]["config.json"] == p1_manifest["model_config_sha256"]
            and identity["tokenizer_files_sha256"]["tokenizer.json"] == p1_manifest["tokenizer_sha256"],
            "P1 model/tokenizer files differ")
    checked_file(key_dir / "qa_report.json", keys["qa_report_sha256"])
    checked_file(key_dir / keys["tensor_file"], keys["tensor_file_sha256"])
    checked_file(keep_dir / "qa_report.json", keep["qa_report_sha256"])
    checked_file(keep_dir / "keep_inputs.json", keep["saved_keep_inputs_sha256"])
    checked_file(Path(keep["source_keep_file"]), keep["keep_inputs_sha256"])
    require(read_json(keep_dir / "keep_inputs.json") == read_json(Path(keep["source_keep_file"])),
            "saved keep inputs differ from their source")
    require(keep["key_files_sha256"] == {
        str(path): file_sha256(path) for path in (
            key_dir / "manifest.json", key_dir / keys["tensor_file"], key_dir / "qa_report.json")
    }, "C refers to different K files")
    for name in names:
        checked_file(keep_dir / keep["layers"][name]["tensor_file"],
                     keep["layers"][name]["tensor_file_sha256"])
        require(all(keys["layers"][name][field] == keep["layers"][name][field]
                    for field in ("in_features", "out_features", "has_bias")),
                f"K/C module dimensions differ at {name}")
        require(keep_report["layers"][name]["status"] == "complete"
                and keep_report["layers"][name]["disk_round_trip_passed"]
                and keep_report["layers"][name]["diagnostics"]["all_checks_passed"],
                f"C extraction QA failed at {name}")
    return keys, keep, {"K": key_report, "C": keep_report}


def check_matrix(name: str, path: Path, key_path: Path, width: int, count: int,
                 key_dtype: str, keep_dtype: str, settings: KeepStatisticsConfig,
                 expected: dict) -> dict:
    """Check saved C in row blocks against replayed formula (5) projections."""
    import torch
    from safetensors import safe_open

    atol, rtol = settings.check_atol, settings.check_rtol
    with safe_open(str(key_path), framework="pt", device="cpu") as stream:
        require(name in stream.keys(), f"missing K tensor at {name}")
        keys = stream.get_tensor(name)
    require(tuple(keys.shape) == (width, count) and str(keys.dtype) == key_dtype
            and bool(torch.isfinite(keys).all()), f"K shape, dtype or finite values differ at {name}")
    with safe_open(str(path), framework="pt", device="cpu") as stream:
        require(set(stream.keys()) == {name}, f"C tensor names differ at {name}")
        view = stream.get_slice(name)
        require(view.get_shape() == [width, width], f"C shape differs at {name}")
        projection = torch.zeros_like(expected["projection"])
        trace, symmetry_max, diagonal_min = 0.0, 0.0, math.inf
        direct = torch.empty_like(expected["direct"])
        indices = expected["indices"]
        for start in range(0, width, settings.chunk_size):
            end = min(width, start + settings.chunk_size)
            rows = view[start:end]
            require(rows.dtype == getattr(torch, keep_dtype) and bool(torch.isfinite(rows).all()),
                    f"C dtype or finite values differ at {name}")
            rows = rows.double()
            columns = view[:, start:end].T.double()
            difference = (rows - columns).abs()
            symmetry_max = max(symmetry_max, float(difference.max()))
            require(bool((difference <= atol + rtol * columns.abs()).all()), f"C is asymmetric at {name}")
            diagonal = rows[torch.arange(end - start), torch.arange(start, end)]
            trace += float(diagonal.sum())
            diagonal_min = min(diagonal_min, float(diagonal.min()))
            projection += expected["vectors"][start:end].T @ (rows @ expected["vectors"])
            chosen = (indices >= start) & (indices < end)
            if bool(chosen.any()):
                direct[chosen] = rows[indices[chosen] - start][:, indices]
        scale = expected["trace"]
        bound = atol + rtol * scale
        require(math.isfinite(trace) and math.isclose(trace, scale, abs_tol=atol, rel_tol=rtol),
                f"C trace differs from replayed weighted inputs at {name}")
        require(diagonal_min >= -bound, f"C has a negative diagonal at {name}")
        direct_scale = expected["direct"].diagonal().clamp_min(0).sqrt()
        direct_bound = atol + rtol * direct_scale[:, None] * direct_scale[None, :]
        require(bool(((direct - expected["direct"]).abs() <= direct_bound).all()),
                f"C direct outer-product block differs at {name}")
        projection_error = float((projection - expected["projection"]).abs().max())
        require(projection_error <= bound, f"C projected second moment differs at {name}")
        # 先乘 1/2 再相加，避免有限二阶矩在对称化中间值上溢。
        eigen_min = float(torch.linalg.eigvalsh(projection * 0.5 + projection.T * 0.5).min())
        require(math.isfinite(eigen_min) and eigen_min >= -bound, f"C projected PSD check failed at {name}")
        return {"shape": [width, width], "storage_dtype": keep_dtype, "finite": True,
                "symmetric": True, "symmetry_max_diff": symmetry_max, "trace": trace,
                "replayed_trace": scale, "direct_outer_product_max_diff": float(
                    (direct - expected["direct"]).abs().max()),
                "projection_max_diff": projection_error, "projected_eigenvalue_min": eigen_min,
                "projected_psd_negative_tolerance": bound, "full_spectrum_checked": False,
                "replayed_vector_count": expected["vector_count"]}


def validate_run(p1_dir: str | Path) -> dict:
    """Publish a complete bundle only after all source, replay and matrix checks pass."""
    import torch
    from safetensors import safe_open

    p1_dir = Path(p1_dir).resolve()
    output_dir = p1_dir.parent / "matrices" / "validation"
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path, report_path = output_dir / "manifest.json", output_dir / "qa_report.json"
    manifest = {"artifact_version": MATRIX_BUNDLE_VERSION, "status": "partial",
                "source_p1_dir": str(p1_dir), "p2_stage_complete": False}
    report = {"status": "partial", "phase": "source_artifacts", "layers": {},
              "real_model_forward_count": 0}
    write_json(manifest_path, manifest)
    try:
        p1, config, trajectories, p1_hashes, collection = load_reference_inputs(p1_dir)
        keys, keep, source_reports = check_sources(p1_dir, config, p1, p1_hashes)
        keep_inputs = read_json(p1_dir.parent / "matrices/keep_statistics/keep_inputs.json")
        require(keep_inputs["artifact_version"] == KEEP_INPUTS_VERSION
                and keep_inputs["purpose"] == "keep_statistics"
                and keep_inputs["recipe_id"] == keep["settings"]["recipe_id"]
                and keep_inputs["tokenizer_revision"] == config.tokenizer_revision
                and keep_inputs["mask_token_id"] == p1["mask_token_id"],
                "frozen keep recipe differs from P1/C")
        if "raw_source_file" in keep_inputs:
            checked_file(Path(keep_inputs["raw_source_file"]), keep_inputs["raw_source_sha256"])
        require(keep["recipe"] == {field: keep_inputs[field] for field in (
            "recipe_id", "purpose", "example_only", "mask_policy_id", "position_policy_id", "weight_policy_id")},
            "C frozen recipe identity differs")
        require(len(keys["columns"]) == len(trajectories)
                and source_reports["K"]["fact_count"] == len(trajectories)
                and source_reports["K"]["layer_count"] == len(config.target_layers)
                and keys["example_only_fact_count"] == sum(
                    row["purpose"] == "example_only" for row in collection["completed"]),
                "K column count or example purpose differs")
        require(keep["example_only"] is bool(
            keep_inputs["example_only"] or keys["example_only_fact_count"]),
            "P2 example-only purpose differs from frozen inputs")
        names = [f"model.transformer.blocks.{layer}.ff_out" for layer in config.target_layers]
        require(all(set(row["weights_by_layer"]) == set(names) for row in keep_inputs["inputs"]),
                "keep weights target different layers")
        manifest.update({"run_id": config.run_id, "round_id": config.round_id,
                         "baseline_id": config.baseline_id, "p1_config_sha256": config.digest(),
                         "model_identity": keys["model_identity"], "fact_order": p1["fact_order"],
                         "target_layers": list(config.target_layers),
                         "recipe_id": keep["settings"]["recipe_id"],
                         "source_manifests_sha256": {
                             "P1": p1_hashes["manifest.json"],
                             "K": file_sha256(p1_dir.parent / "matrices/reference_keys/manifest.json"),
                             "C": file_sha256(p1_dir.parent / "matrices/keep_statistics/manifest.json")},
                         "example_only": bool(keep["example_only"]),
                         "formal_dataset_acceptance": False})
        write_json(manifest_path, manifest)
        report["phase"] = "real_model_replay"
        adapter = LLaDA8BInstructAdapter.from_local_tokenizer(
            model_path=config.model_path, target_layers=config.target_layers)
        require(adapter.mask_token_id == p1["mask_token_id"], "replay mask token differs")
        model = adapter.load_model(config.model_path, torch_dtype=config.torch_dtype, device_map=config.device_map)
        descriptions = adapter.inspect_target_modules(model)
        require(set(descriptions) == set(names), "replay target layers differ")
        with safe_open(str(p1_dir.parent / "matrices/reference_keys" / keys["tensor_file"]),
                       framework="pt", device="cpu") as stream:
            require(set(stream.keys()) == set(names), "K tensor layer names differ")
            saved_keys = {name: stream.get_tensor(name) for name in names}
        for name in names:
            require(saved_keys[name].shape == (descriptions[name]["in_features"], len(trajectories))
                    and str(saved_keys[name].dtype) == keys["layers"][name]["storage_dtype"]
                    and bool(torch.isfinite(saved_keys[name]).all()), f"K tensor shape/dtype differs at {name}")
            norms = saved_keys[name].double().norm(dim=0).tolist()
            require(len(norms) == len(keys["layers"][name]["column_norms"])
                    and all(math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-8)
                            for left, right in zip(norms, keys["layers"][name]["column_norms"])),
                    f"K column norms differ at {name}")
        key_settings = ReferenceKeyConfig(**keys["settings"])
        for index, trajectory in enumerate(trajectories):
            initial = trajectory.trace.pre_forward_inputs[0]
            column = keys["columns"][index]
            require(column == {"column": index, "case_id": trajectory.case_id,
                               "probe_id": trajectory.probe_id,
                               "subject_last_position": initial.subject_last_position,
                               "mask_positions": list(initial.mask_positions),
                               "initial_input_sha256": config_digest(input_record(initial)),
                               "initial_mask_logits_sha256": trajectory.trace.mask_logit_sha256[0]},
                    "K column order or initial input differs")
            captured = adapter.forward_with_capture(model, initial)
            report["real_model_forward_count"] += 1
            require(tensor_sha256(captured.mask_logits) == column["initial_mask_logits_sha256"],
                    f"P1/K initial logits replay differs for {trajectory.case_id}")
            for name in names:
                value = captured.module_inputs[name].detach().cpu()
                stored = saved_keys[name]
                require(tuple(stored.shape) == (descriptions[name]["in_features"], len(trajectories))
                        and str(stored.dtype) == keys["layers"][name]["storage_dtype"]
                        and bool(torch.isfinite(stored).all())
                        # 不降为 float32，否则会掩盖超过冻结容差的 float64 差异。
                        and torch.allclose(value.double(), stored[:, index].double(),
                                           atol=key_settings.replay_atol, rtol=key_settings.replay_rtol),
                        f"K replay differs at {name}, column {index}")
        del saved_keys
        settings = KeepStatisticsConfig(**keep["settings"])
        settings.check()
        for name in names:
            report["phase"] = "matrix_recomputation"
            layer = keep["layers"][name]
            width = descriptions[name]["in_features"]
            require(layer["shape"] == [width, width]
                    and keys["layers"][name]["shape"] == [width, len(trajectories)]
                    and keys["layers"][name]["storage_dtype"] == "torch." + key_settings.storage_dtype
                    and layer["storage_dtype"] == settings.storage_dtype
                    and layer["accumulation_dtype"] == settings.accumulation_dtype
                    and all(layer[field] == descriptions[name][field]
                            for field in ("in_features", "out_features", "has_bias")),
                    f"K/C shapes or module dimensions differ at {name}")
            generator = torch.Generator(device="cpu").manual_seed(settings.qa_seed + 1)
            vectors = torch.linalg.qr(torch.randn((width, min(width, settings.projection_count)),
                                                 generator=generator, dtype=torch.float64), mode="reduced").Q
            indices = torch.randperm(width, generator=generator)[:min(width, settings.direct_block_size)].sort().values
            expected = {"vectors": vectors, "indices": indices,
                        "direct": torch.zeros((len(indices), len(indices)), dtype=torch.float64),
                        "projection": torch.zeros((vectors.shape[1], vectors.shape[1]), dtype=torch.float64),
                        "trace": 0.0, "vector_count": 0, "positive_weight_count": 0,
                        "weights": [], "source_groups": {}}
            layer_adapter = LLaDA8BInstructAdapter(
                adapter.tokenizer, mask_token_id=adapter.mask_token_id,
                target_layers=(config.target_layers[names.index(name)],))
            rows = source_reports["C"]["layers"][name]["inputs"]
            require(len(rows) == len(keep_inputs["inputs"]), f"C input count differs at {name}")
            for input_row, recorded in zip(keep_inputs["inputs"], rows):
                positions = tuple(input_row["capture_positions"])
                model_input = keep_input_from_record(input_row["input"])
                require(recorded["input_id"] == input_row["input_id"]
                        and recorded["source"] == input_row["source"]
                        and recorded["capture_positions"] == list(positions)
                        and recorded["input_sha256"] == config_digest(input_row["input"])
                        and recorded["weights"] == input_row["weights_by_layer"][name]
                        and recorded["mask_count"] == len(model_input.mask_positions)
                        and math.isclose(recorded["weight_sum"], math.fsum(recorded["weights"])),
                        f"C keep input order or weights differ at {name}")
                captured = layer_adapter.forward_keep_with_capture(model, model_input, positions)
                report["real_model_forward_count"] += 1
                require(tensor_sha256(captured.mask_logits) == recorded["mask_logits_sha256"]
                        and tensor_sha256(captured.module_inputs[name]) == recorded["capture_sha256"],
                        f"C keep input replay differs at {name}: {recorded['input_id']}")
                values = captured.module_inputs[name].detach().cpu().double()
                weights = torch.as_tensor(recorded["weights"], dtype=torch.float64)
                require(tuple(values.shape) == (len(positions), width)
                        and len(weights) == len(positions)
                        and bool(torch.isfinite(values).all())
                        and bool(torch.isfinite(weights).all()) and bool((weights >= 0).all()),
                        f"C replayed inputs or weights are invalid at {name}")
                weighted = values[weights > 0] * weights[weights > 0].sqrt()[:, None]
                direct = weighted[:, indices]
                projected = weighted @ vectors
                expected["direct"] += direct.T @ direct
                expected["projection"] += projected.T @ projected
                expected["trace"] += float(weighted.square().sum())
                expected["vector_count"] += len(positions)
                expected["positive_weight_count"] += int((weights > 0).sum())
                expected["weights"].extend(recorded["weights"])
                group = expected["source_groups"].setdefault(
                    recorded["source"].get("source_dataset", "unspecified"),
                    {"input_count": 0, "vector_count": 0, "weight_sum": 0.0})
                group["input_count"] += 1
                group["vector_count"] += len(positions)
                group["weight_sum"] += sum(recorded["weights"])
            require(expected["vector_count"] == layer["vector_count"]
                    and expected["positive_weight_count"] == layer["positive_weight_count"]
                    and layer["zero_weight_count"] == layer["vector_count"] - layer["positive_weight_count"]
                    and math.isclose(math.fsum(expected["weights"]), layer["weight_sum"]),
                    f"C vector counts or weight scale differ at {name}")
            require(expected["source_groups"] == layer["source_groups"],
                    f"C source group counts or weight scale differ at {name}")
            report["layers"][name] = check_matrix(
                name, p1_dir.parent / "matrices/keep_statistics" / layer["tensor_file"],
                p1_dir.parent / "matrices/reference_keys" / keys["tensor_file"],
                width, len(trajectories), keys["layers"][name]["storage_dtype"],
                layer["storage_dtype"], settings, expected)
            write_json(report_path, report)
        report["phase"] = "source_recheck"
        for filename, digest in p1_hashes.items():
            checked_file(p1_dir / filename, digest)
        for label, filename in (("K", "reference_keys"), ("C", "keep_statistics")):
            checked_file(p1_dir.parent / "matrices" / filename / "manifest.json",
                         manifest["source_manifests_sha256"][label])
        for path, digest in (
            (p1_dir.parent / "matrices/reference_keys/qa_report.json", keys["qa_report_sha256"]),
            (p1_dir.parent / "matrices/reference_keys" / keys["tensor_file"], keys["tensor_file_sha256"]),
            (p1_dir.parent / "matrices/keep_statistics/qa_report.json", keep["qa_report_sha256"]),
            (p1_dir.parent / "matrices/keep_statistics/keep_inputs.json", keep["saved_keep_inputs_sha256"]),
            (Path(keep["source_keep_file"]), keep["keep_inputs_sha256"]),
        ):
            checked_file(path, digest)
        if "raw_source_file" in keep_inputs:
            checked_file(Path(keep_inputs["raw_source_file"]), keep_inputs["raw_source_sha256"])
        for layer in keep["layers"].values():
            checked_file(p1_dir.parent / "matrices/keep_statistics" / layer["tensor_file"],
                         layer["tensor_file_sha256"])
        report.update({"status": "complete", "phase": "complete", "failure": None,
                       "fact_count": len(trajectories), "layer_count": len(names),
                       "full_spectrum_checked": False})
        write_json(report_path, report)
        manifest.update({"status": "complete", "p2_stage_complete": not manifest["example_only"],
                         "qa_report_sha256": file_sha256(report_path)})
        write_json(manifest_path, manifest)
    except Exception as error:
        report.update({"status": "failed", "failure": str(error)})
        manifest["status"] = "failed"
        write_json(report_path, report)
        write_json(manifest_path, manifest)
        raise
    return {"status": "complete", "output_dir": str(output_dir),
            "p2_stage_complete": manifest["p2_stage_complete"],
            "example_only": manifest["example_only"], "report": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("p1_dir", type=Path, help="completed runs/<run_id>/round_<n>/p1 directory")
    args = parser.parse_args()
    print(json.dumps(validate_run(args.p1_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
