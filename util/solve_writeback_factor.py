"""Solve fixed P3 Q factors from a complete, unchanged P2 matrix bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from MDLM.contracts import (
    MATRIX_BUNDLE_VERSION, REFERENCE_KEYS_VERSION, KEEP_STATISTICS_VERSION,
    WRITEBACK_FACTOR_VERSION, WritebackConfig, config_digest, file_sha256,
)
from MDLM.writeback import WritebackSolveError, solve_writeback_factor
from util.extract_reference_keys import load_reference_inputs, write_json
from util.validate_matrix_artifacts import checked_file, read_json, require


def load_matrix_bundle(p1_dir: Path, settings: WritebackConfig) -> tuple[dict, dict, dict, dict]:
    """Check immutable algorithm artifacts; no dataset or model weights are read."""
    p1, config, trajectories, p1_hashes, collection = load_reference_inputs(p1_dir)
    matrices = p1_dir.parent / "matrices"
    bundle_digest = file_sha256(matrices / "validation/manifest.json")
    bundle = read_json(matrices / "validation/manifest.json")
    qa = read_json(matrices / "validation/qa_report.json")
    keys = read_json(matrices / "reference_keys/manifest.json")
    keep = read_json(matrices / "keep_statistics/manifest.json")
    require(bundle["artifact_version"] == MATRIX_BUNDLE_VERSION and bundle["status"] == "complete",
            "P2 unified acceptance is incomplete or has an unsupported version")
    checked_file(matrices / "validation/qa_report.json", bundle["qa_report_sha256"])
    require(qa["status"] == "complete" and qa["failure"] is None, "P2 unified QA did not pass")
    for label, path in (("P1", p1_dir / "manifest.json"),
                        ("K", matrices / "reference_keys/manifest.json"),
                        ("C", matrices / "keep_statistics/manifest.json")):
        checked_file(path, bundle["source_manifests_sha256"][label])
    require(keys["artifact_version"] == REFERENCE_KEYS_VERSION and keys["status"] == "complete"
            and keep["artifact_version"] == KEEP_STATISTICS_VERSION and keep["status"] == "complete",
            "K/C artifacts are incomplete or have unsupported versions")
    names = [f"model.transformer.blocks.{layer}.ff_out" for layer in config.target_layers]
    require(set(settings.ridge_by_layer) == set(names), "ridge layer selection differs from the accepted P2 layers")
    require(bundle["target_layers"] == list(config.target_layers) and bundle["fact_order"] == p1["fact_order"]
            and qa["fact_count"] == len(trajectories) and qa["layer_count"] == len(names)
            and set(qa["layers"]) == set(names), "P2 accepted facts or layers differ")
    for record in (bundle, keys, keep):
        require(all(record[field] == getattr(config, field) for field in ("run_id", "round_id", "baseline_id"))
                and record["p1_config_sha256"] == config.digest()
                and Path(record["source_p1_dir"]).resolve() == p1_dir,
                "P3 cannot mix P2 rounds, baselines or configurations")
    for record in (keys, keep):
        require(record["source_files_sha256"] == p1_hashes
                and config_digest(record["source_config"]) == config.digest()
                and set(record["layers"]) == set(names), "K/C sources differ from accepted P1 artifacts")
    require(bundle["model_identity"] == keys["model_identity"] == keep["model_identity"]
            and bundle["model_identity"]["model_revision"] == config.model_revision
            and bundle["model_identity"]["tokenizer_revision"] == config.tokenizer_revision,
            "P2 baseline model/tokenizer identities differ")
    require(keep["centered"] is False and keep["normalization"] == "none"
            and bundle["recipe_id"] == keep["settings"]["recipe_id"], "C definition or recipe differs")
    example_only = bool(keep["example_only"] or any(row["purpose"] == "example_only" for row in collection["completed"]))
    require(type(bundle["example_only"]) is bool and bundle["example_only"] == example_only,
            "P2 example-only label differs from its source artifacts")
    if example_only:
        require(settings.allow_example_only, "example-only matrices require explicit allow_example_only=true")
    else:
        require(bundle["p2_stage_complete"] is True, "formal P3 requires completed P2 stage acceptance")
    sources = {str(p1_dir / name): digest for name, digest in p1_hashes.items()}
    # 使用读入时绑定的摘要，不能把处理中改变的文件重新摘要后视作同一来源。
    sources.update({
        str(matrices / "validation/manifest.json"): bundle_digest,
        str(matrices / "validation/qa_report.json"): bundle["qa_report_sha256"],
        str(matrices / "reference_keys/manifest.json"): bundle["source_manifests_sha256"]["K"],
        str(matrices / "keep_statistics/manifest.json"): bundle["source_manifests_sha256"]["C"],
    })
    for directory, record in ((matrices / "reference_keys", keys), (matrices / "keep_statistics", keep)):
        sources[str(directory / "qa_report.json")] = record["qa_report_sha256"]
        if directory.name == "reference_keys":
            sources[str(directory / record["tensor_file"])] = record["tensor_file_sha256"]
        else:
            sources[str(directory / "keep_inputs.json")] = record["saved_keep_inputs_sha256"]
            for layer in record["layers"].values():
                sources[str(directory / layer["tensor_file"])] = layer["tensor_file_sha256"]
    for path, digest in sources.items():
        checked_file(Path(path), digest)
    return bundle, keys, keep, sources


def solve_run(p1_dir: str | Path, settings: WritebackConfig) -> dict:
    """Solve/save one layer at a time, preserving both successful and failed evidence."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    settings.check()
    p1_dir = Path(p1_dir).resolve()
    output_dir = p1_dir.parent / "matrices/writeback_factors"
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path, report_path = output_dir / "manifest.json", output_dir / "qa_report.json"
    manifest = {"artifact_version": WRITEBACK_FACTOR_VERSION, "status": "partial", "formula": 9,
                "source_p1_dir": str(p1_dir), "settings": settings.as_record(),
                "settings_sha256": config_digest(settings.as_record()), "p3_stage_complete": False}
    report = {"status": "partial", "phase": "source_artifacts", "layers": {}, "model_forward_count": 0}
    report["runtime"] = {"torch": str(torch.__version__), "cpu_threads": torch.get_num_threads()}
    write_json(manifest_path, manifest)
    try:
        bundle, keys, keep, sources = load_matrix_bundle(p1_dir, settings)
        for field in ("run_id", "round_id", "baseline_id", "p1_config_sha256", "model_identity",
                      "fact_order", "target_layers", "example_only"):
            manifest[field] = bundle[field]
        manifest["source_files_sha256"] = sources
        manifest["formal_dataset_acceptance"] = False
        write_json(manifest_path, manifest)
        matrices = p1_dir.parent / "matrices"
        layers = {}
        for layer in bundle["target_layers"]:
            name = f"model.transformer.blocks.{layer}.ff_out"
            report.update({"phase": "solve", "current_layer": name})
            print(f"Solving fixed Q for {name} on {settings.device}...", flush=True)
            with safe_open(str(matrices / "reference_keys" / keys["tensor_file"]), framework="pt", device="cpu") as stream:
                require(set(stream.keys()) == set(keys["layers"]), "K tensor layer names differ")
                k = stream.get_tensor(name)
            with safe_open(str(matrices / "keep_statistics" / keep["layers"][name]["tensor_file"]),
                           framework="pt", device="cpu") as stream:
                require(set(stream.keys()) == {name}, f"C tensor names differ at {name}")
                c = stream.get_tensor(name)
            info = keys["layers"][name]
            require(tuple(k.shape) == (info["in_features"], len(bundle["fact_order"]))
                    and list(k.shape) == info["shape"] and str(k.dtype) == info["storage_dtype"]
                    and tuple(c.shape) == (info["in_features"], info["in_features"])
                    and list(c.shape) == keep["layers"][name]["shape"]
                    and c.dtype == getattr(torch, keep["layers"][name]["storage_dtype"]),
                    f"accepted K/C shape or storage dtype differs at {name}")
            try:
                q, diagnostics = solve_writeback_factor(k, c, settings.ridge_by_layer[name], settings)
            except WritebackSolveError as error:
                report["layers"][name] = error.diagnostics
                raise
            report["layers"][name] = diagnostics
            report["phase"] = "save_and_reload"
            path = output_dir / f"q_layer_{layer}.safetensors"
            save_file({name: q}, str(path), metadata={
                "artifact_version": WRITEBACK_FACTOR_VERSION,
                "p1_config_sha256": bundle["p1_config_sha256"],
                "settings_sha256": manifest["settings_sha256"],
            })
            saved = load_file(str(path), device="cpu")
            require(set(saved) == {name} and saved[name].dtype == q.dtype
                    and torch.equal(saved[name], q), f"saved Q did not round-trip exactly at {name}")
            diagnostics["disk_round_trip_passed"] = True
            layers[name] = {"shape": list(q.shape), "in_features": info["in_features"],
                            "out_features": info["out_features"], "has_bias": info["has_bias"],
                            "ridge": settings.ridge_by_layer[name], "storage_dtype": str(q.dtype),
                            "tensor_file": path.name, "tensor_file_sha256": file_sha256(path),
                            "fixed_during_residual_optimization": True}
            write_json(report_path, report)
            del k, c, q, saved
        report["phase"] = "source_recheck"
        for path, digest in sources.items():
            checked_file(Path(path), digest)
        require(config_digest(settings.as_record()) == manifest["settings_sha256"], "P3 settings changed during solve")
        report.update({"status": "complete", "phase": "complete", "failure": None,
                       "fact_count": len(bundle["fact_order"]), "layer_count": len(layers)})
        write_json(report_path, report)
        manifest.update({"status": "complete", "layers": layers,
                         "qa_report_sha256": file_sha256(report_path),
                         "p3_stage_complete": not manifest["example_only"]})
        write_json(manifest_path, manifest)
    except Exception as error:
        report.update({"status": "failed", "failure": str(error)})
        manifest["status"] = "failed"
        write_json(report_path, report)
        write_json(manifest_path, manifest)
        raise
    return {"status": "complete", "output_dir": str(output_dir), "layers": layers,
            "example_only": manifest["example_only"], "p3_stage_complete": manifest["p3_stage_complete"],
            "report": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("p1_dir", type=Path, help="P1 directory with completed P2 unified acceptance")
    parser.add_argument("--config", type=Path, required=True, help="frozen per-layer ridge and solve choices")
    args = parser.parse_args()
    print(json.dumps(solve_run(args.p1_dir, WritebackConfig.from_file(args.config)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
