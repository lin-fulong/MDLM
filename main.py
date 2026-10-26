"""Run LLaDA knowledge editing for every selected fact (P0 through P5).

The input JSON freezes experiment choices. Existing stage implementations own the
math and their artifact checks; this file only prepares inputs and orders them.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from MDLM.contracts import (
    AcceptanceConfig, FactSelection, KeepStatisticsConfig, P1RunConfig,
    ReferenceKeyConfig, ResidualOptimizationConfig, WritebackConfig,
    config_digest, file_sha256, local_model_identity,
)
from data.readers.json_cases import load_cases
from util.prepare_dataset_inputs import prepare_inputs
from util.pipeline_cache import archive_partial, files_match, round_key, save_index, snapshot


ROOT = Path(__file__).resolve().parent


def read_json(path: Path) -> object:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def project_path(value: str) -> Path:
    """Resolve config paths from the repository root, independent of shell cwd."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def required_path(config: dict, key: str) -> Path:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must name an existing file or directory")
    path = project_path(value)
    if not path.exists():
        raise FileNotFoundError(f"{key}: {path}")
    return path


def config_setup_issues(choices: dict) -> list[str]:
    """Explain absent experiment inputs before allocating a run directory."""
    paths = {
        "dataset": choices.get("dataset"),
        "model_path": choices.get("model_path"),
    }
    mode = choices.get("input_preparation", {}).get("mode", "explicit")
    if mode == "explicit":
        paths.update({"keep_inputs": choices.get("keep_inputs"),
                      "p5.dev_edit_file": choices.get("p5", {}).get("dev_edit_file"),
                      "p5.dev_keep_file": choices.get("p5", {}).get("dev_keep_file")})
    issues = []
    for label, value in paths.items():
        if not isinstance(value, str) or not value or value.startswith("REPLACE_WITH_"):
            issues.append(f"{label}: specify a real path")
        elif not project_path(value).exists():
            issues.append(f"{label}: file or directory does not exist: {project_path(value)}")
    if mode == "dataset":
        return issues
    if mode != "explicit":
        return [*issues, "input_preparation.mode: choose dataset or explicit"]
    recipe = choices.get("p2_keep", {}).get("recipe_id")
    if not isinstance(recipe, str) or not recipe or recipe.startswith("REPLACE_WITH_"):
        issues.append("p2_keep.recipe_id: specify the frozen keep recipe ID")
    probes = choices.get("p5", {}).get("dev_edit_probes")
    if not isinstance(probes, list) or not probes or any(
        not isinstance(item, dict) or any(
            not isinstance(item.get(key), str) or not item[key]
            or item[key].startswith("REPLACE_WITH_") for key in ("case_id", "probe_id")
        ) for item in probes
    ):
        issues.append("p5.dev_edit_probes: select real development probes")
    return issues


def one_round(config_file: Path, *, run_id_override: str | None = None,
              round_index: int = 0, selected_case_id: str | None = None,
              baseline_model: Path | None = None, sequential: bool = False,
              original_identity: dict | None = None, selected_case_ids: list[str] | None = None,
              workspace: Path | None = None, request_key: str | None = None,
              use_cache: bool = True, baseline_identity: dict | None = None) -> dict:
    choices = read_json(config_file)
    dataset = required_path(choices, "dataset")
    original_model = required_path(choices, "model_path")
    model = baseline_model.resolve() if baseline_model is not None else original_model
    keep_inputs = required_path(choices, "keep_inputs")
    p5_values = dict(choices["p5"])
    dev_edit_file = required_path(p5_values, "dev_edit_file")
    dev_keep_file = required_path(p5_values, "dev_keep_file")
    for label, path in (("dataset", dataset), ("keep_inputs", keep_inputs),
                        ("dev_edit_file", dev_edit_file), ("dev_keep_file", dev_keep_file)):
        if not path.is_file():
            raise ValueError(f"{label} must be a file: {path}")
    if not model.is_dir() or not original_model.is_dir():
        raise ValueError("model_path and baseline_model must be model directories")
    layers = choices["target_layers"]
    if not isinstance(layers, list) or not layers or any(type(n) is not int or n < 0 for n in layers):
        raise ValueError("target_layers must be a nonempty list of nonnegative layer indices")
    names = {f"model.transformer.blocks.{layer}.ff_out" for layer in layers}
    if len(names) != len(layers):
        raise ValueError("target_layers must not contain duplicates")

    # 输入已由预处理脚本生成，或由配置显式指定；这里核对运行用途。
    if read_json(keep_inputs)["example_only"] or read_json(dev_keep_file)["example_only"]:
        raise ValueError("main.py requires non-example P2 and P5 preservation inputs")
    cases = load_cases(dataset)
    selections = selected_case_ids or ([selected_case_id] if selected_case_id is not None else None)
    if selections is not None:
        cases = [case for case in cases if case["case_id"] in selections]
    if not cases:
        raise ValueError("the dataset adapter selected no editing facts")
    if any(case["quality_control"]["status"] == "example_only" for case in cases):
        raise ValueError("main.py requires non-example editing facts")
    facts = []
    for case in cases:
        probes = case["prompts"]["edit"]
        if len(probes) != 1:
            raise ValueError(f"{case['case_id']}: exactly one runtime edit probe is required per fact")
        facts.append(FactSelection(case["case_id"], probes[0]["probe_id"]))
    if len({fact.case_id for fact in facts}) != len(facts):
        raise ValueError("runtime fact IDs must be unique")

    keys = ReferenceKeyConfig(**choices.get("p2_keys", {}))
    keys.check()
    keep = KeepStatisticsConfig(**choices["p2_keep"])
    keep.check()
    q = WritebackConfig(**choices["p3"])
    q.check()
    theta0_identity = original_identity or local_model_identity(original_model)
    identity = baseline_identity or (theta0_identity if model == original_model else local_model_identity(model))
    r_values = dict(choices["p4"])
    if "checkpoints" in r_values:
        r_values["checkpoints"] = tuple(r_values["checkpoints"])
    if r_values.get("beta", 0) > 0:
        r_values.setdefault("reference_model_path", str(original_model))
        r_values.setdefault("reference_model_revision", theta0_identity["model_revision"])
        r_values.setdefault("reference_tokenizer_revision", theta0_identity["tokenizer_revision"])
        r_values["keep_train_file"] = str(required_path(r_values, "keep_train_file"))
    r = ResidualOptimizationConfig(**r_values)
    r.check()
    if set(q.ridge_by_layer) != names or set(r.rho_by_layer) != names:
        raise ValueError("P3/P4 layer names must match target_layers exactly")
    if q.allow_example_only or r.allow_example_only or p5_values.get("allow_example_only", False):
        raise ValueError("the main entrypoint does not allow example-only stage inputs")
    if r.beta > 0 and (project_path(r.reference_model_path) != original_model
                       or r.reference_model_revision != theta0_identity["model_revision"]
                       or r.reference_tokenizer_revision != theta0_identity["tokenizer_revision"]):
        raise ValueError("P4 theta_0 reference must match the original model directory")

    run_id = run_id_override or choices.get("run_id") or (
        "edit_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    )
    if not isinstance(run_id, str) or run_id in ("", ".", "..") or "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must be one directory name")
    output_root = project_path(choices.get("output_root", "runs"))
    if workspace is not None:
        output_root, run_id = workspace.parent.parent, workspace.parent.name
        round_index = int(workspace.name.removeprefix("round_"))
    round_dir = output_root / run_id / f"round_{round_index}"
    round_dir.mkdir(parents=True, exist_ok=workspace is not None)
    report_path = round_dir / "pipeline_report.json"
    report = {"status": "running", "phase": "freeze_inputs", "run_id": run_id,
              "round_id": round_index, "mode": "sequential" if sequential else "independent",
              "baseline_model": str(model), "original_model": str(original_model),
              "fact_count": len(facts), "target_layers": layers, "stages": {},
              "source_dataset": str(dataset), "source_dataset_sha256": file_sha256(dataset),
              "source_keep_inputs": str(keep_inputs), "source_dev_edit_file": str(dev_edit_file),
              "source_dev_keep_file": str(dev_keep_file),
              "config_file": str(config_file.resolve()), "config_sha256": file_sha256(config_file)}
    if workspace is not None and report_path.is_file():
        existing = read_json(report_path)
        if existing.get("request_key") != request_key:
            raise ValueError("缓存运行与当前请求不匹配")
        report["stages"] = existing.get("stages", {})
    report["request_key"] = request_key
    write_json(report_path, report)

    upstream = {}

    def stage(label: str, arguments: list[str], artifact: Path, accepted: tuple[str, ...]) -> dict:
        dependency = config_digest({"arguments": arguments, "configs": {
            str(path): file_sha256(path) for path in sorted(round_dir.glob("*_config.json"))},
            "upstream": upstream, "request_key": request_key})
        prior = report["stages"].get(label, {})
        if use_cache and prior.get("status") in accepted and prior.get("dependency") == dependency and files_match(prior.get("files_sha256", {})):
            result = read_json(artifact)
            status = result["status"] if isinstance(result, dict) else "complete"
            if status in accepted:
                prior["cache_hit"] = True
                upstream[label] = prior["files_sha256"]
                print(f"[{label}] 复用本地缓存：{artifact}", flush=True)
                return result
        report["phase"] = label
        write_json(report_path, report)
        log_path = round_dir / f"{label.lower()}.log"
        report["stages"][label] = {"status": "running", "artifact": str(artifact), "log": str(log_path)}
        write_json(report_path, report)
        print(f"[{label}] {log_path}", flush=True)
        target = artifact if artifact.parent == round_dir else artifact.parent
        archive_partial(target, round_dir / "failed_attempts", label)
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as stream:
            try:
                from util.gpu_runtime import execute_stage
                execute_stage(arguments, stream=stream, cwd=ROOT,
                              resident=choices.get("execution", {}).get("backend") == "resident")
            except subprocess.CalledProcessError:
                if label != "P4" or not artifact.is_file() or read_json(artifact).get("status") != "rejected":
                    raise
        result = read_json(artifact)
        status = result["status"] if isinstance(result, dict) else "complete"
        if status not in accepted:
            raise RuntimeError(f"{label} artifact status is {status!r}: {artifact}")
        digests = snapshot(target)
        report["stages"][label] = {"status": status, "artifact": str(artifact), "log": str(log_path),
                                    "dependency": dependency, "files_sha256": digests, "cache_hit": False,
                                    "elapsed_seconds": time.perf_counter() - started}
        upstream[label] = digests
        write_json(report_path, report)
        print(f"[{label}] {status} | {report['stages'][label]['elapsed_seconds']:.1f}s", flush=True)
        return result

    try:
        write_json(round_dir / "model_identity.json", identity)
        # Freeze the adapter output once. The source file is never changed.
        case_file = round_dir / "editing_cases.json"
        write_json(case_file, cases)
        write_json(round_dir / "keep_inputs.json", read_json(keep_inputs))
        keep_file = round_dir / "keep_inputs.json"
        write_json(round_dir / "source_config.json", choices)
        p1_values = dict(choices.get("p1", {}))
        p1 = P1RunConfig(
            run_id=run_id, round_id=round_index, baseline_id=identity["model_revision"],
            model_name="LLaDA-8B-Instruct", model_path=str(model),
            model_revision=identity["model_revision"],
            tokenizer_revision=identity["tokenizer_revision"],
            case_file=str(case_file), facts=tuple(facts), target_layers=tuple(layers),
            window_policy_id="short_answer_exact_tokens_v1",
            sampler_policy_id="confidence_no_remask_v1",
            seed=p1_values.pop("seed", 2026), steps=p1_values.pop("steps", None),
            temperature=p1_values.pop("temperature", 0.0),
            replay_count=p1_values.pop("replay_count", 1), output_root=str(output_root),
            torch_dtype=p1_values.pop("torch_dtype", "auto"),
            device_map=p1_values.pop("device_map", "auto"),
        )
        if p1_values:
            raise ValueError(f"unsupported P1 settings: {sorted(p1_values)}")
        p1.check()
        p5_values.update(dev_edit_file=str(dev_edit_file), dev_keep_file=str(dev_keep_file),
                         reference_model_path=str(original_model),
                         reference_model_revision=theta0_identity["model_revision"],
                         reference_tokenizer_revision=theta0_identity["tokenizer_revision"])
        selected = {fact.case_id for fact in facts}
        if choices.get("input_preparation", {}).get("mode") == "dataset":
            p5_values["dev_edit_probes"] = [item for item in p5_values["dev_edit_probes"] if item["case_id"] in selected]
        if not p5_values["dev_edit_probes"]:
            raise ValueError("当前编辑案例没有配置相应的 P5 开发探针")
        p5 = AcceptanceConfig.from_record(p5_values)
        for name, value in (("p1", p1.as_record()), ("p2_keys", keys.as_record()),
                            ("p2_keep", keep.as_record()), ("p3", q.as_record()),
                            ("p4", r.as_record()), ("p5", p5.as_record())):
            write_json(round_dir / f"{name}_config.json", value)

        for layer in layers:
            p0_report = round_dir / f"p0_layer_{layer}.json"
            label = f"P0_LAYER_{layer}"
            rows = stage(label, ["util.smoke_p0_llada", str(case_file), "--model-path", str(model),
                         "--layer", str(layer), "--forward", "--output", str(p0_report),
                         "--device-map", p1.device_map, "--torch-dtype", p1.torch_dtype], p0_report, ("complete",))
            module = f"model.transformer.blocks.{layer}.ff_out"
            if len(rows) != len(facts) or [
                (row["case_id"], row["probe_id"]) for row in rows
            ] != [(fact.case_id, fact.probe_id) for fact in facts] or any(
                not row["real_forward"] or not row["replay_within_tolerance"]
                or not row["full_mask_input"] or row["mask_count"] != row["target_token_count"]
                or set(row["target_modules"]) != {module}
                or set(row["ffn_input_shapes"]) != {module} for row in rows
            ):
                raise RuntimeError(f"P0 did not cover and replay all {len(facts)} facts at layer {layer}")

        p1_dir = round_dir / "p1"
        stage("P1", ["util.collect_denoise_states", str(round_dir / "p1_config.json")],
              p1_dir / "coverage_report.json", ("complete",))
        stage("P2_KEYS", ["util.extract_reference_keys", str(p1_dir),
              "--config", str(round_dir / "p2_keys_config.json")],
              round_dir / "matrices/reference_keys/manifest.json", ("complete",))
        stage("P2_KEEP", ["util.extract_keep_statistics", str(p1_dir), str(keep_file),
              "--config", str(round_dir / "p2_keep_config.json")],
              round_dir / "matrices/keep_statistics/manifest.json", ("complete",))
        stage("P2_QA", ["util.validate_matrix_artifacts", str(p1_dir)],
              round_dir / "matrices/validation/manifest.json", ("complete",))
        stage("P3", ["util.solve_writeback_factor", str(p1_dir),
              "--config", str(round_dir / "p3_config.json")],
              round_dir / "matrices/writeback_factors/manifest.json", ("complete",))
        p4_result = stage("P4", ["util.optimize_residuals", str(p1_dir),
              "--config", str(round_dir / "p4_config.json")],
              round_dir / "residuals/manifest.json", ("complete", "rejected"))
        if p4_result["status"] == "rejected":
            report.update(status="complete", phase="complete", decision="rejected", selected_alpha=None,
                          decision_reason="P4 未得到合格残差候选", final_model_dir=None,
                          checkpoint_dir=None, baseline_restored=True)
            write_json(report_path, report)
            return report
        p5_dir = round_dir / "editing"
        p5_result = stage("P5", ["util.merge_and_accept", str(p1_dir),
              "--config", str(round_dir / "p5_config.json"),
              "--residual-dir", str(round_dir / "residuals"), "--output-dir", str(p5_dir)],
              p5_dir / "acceptance_report.json", ("complete",))
        decision = p5_result["decision"]
        if decision not in ("accepted", "rejected"):
            raise RuntimeError(f"unexpected P5 decision: {decision!r}")
        final_manifest = read_json(p5_dir / "manifest.json")
        if final_manifest["status"] != decision or final_manifest["example_only"] or not final_manifest["p5_stage_complete"]:
            raise RuntimeError("P5 final manifest does not represent a completed non-example round")
        if decision == "accepted":
            report["phase"] = "P5_RELOAD"
            write_json(report_path, report)
            dependency = config_digest(upstream)
            prior = report["stages"].get("P5_RELOAD", {})
            if not (use_cache and prior.get("dependency") == dependency and files_match(prior.get("files_sha256", {}))):
                from MDLM.workflow import load_final_model
                adapter, edited_model = load_final_model(p5_dir)
                del adapter, edited_model
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                report["stages"]["P5_RELOAD"] = {"status": "complete", "artifact": str(p5_dir / "manifest.json"),
                                                  "dependency": dependency, "files_sha256": snapshot(p5_dir)}
        checkpoint_dir = None
        if sequential and decision == "accepted":
            report["phase"] = "CHECKPOINT_EXPORT"
            write_json(report_path, report)
            from util.export_edited_checkpoint import export_checkpoint
            checkpoint_dir = round_dir / "checkpoint"
            dependency = config_digest(upstream)
            prior = report["stages"].get("CHECKPOINT_EXPORT", {})
            if not (use_cache and prior.get("dependency") == dependency and files_match(prior.get("files_sha256", {}))):
                archive_partial(checkpoint_dir, round_dir / "failed_attempts", "CHECKPOINT_EXPORT")
                export_checkpoint(model, p5_dir, checkpoint_dir)
                report["stages"]["CHECKPOINT_EXPORT"] = {
                    "status": "complete", "artifact": str(checkpoint_dir / "export_report.json"),
                    "dependency": dependency, "files_sha256": snapshot(checkpoint_dir)}
        report.update(status="complete", phase="complete", decision=decision,
                      selected_alpha=p5_result["selection"]["selected_alpha"],
                      final_model_dir=str(p5_dir) if decision == "accepted" else None,
                      checkpoint_dir=str(checkpoint_dir) if checkpoint_dir is not None else None,
                      baseline_restored=p5_result["baseline_restored"])
        write_json(report_path, report)
        return report
    except Exception as error:
        report.update(status="failed", failure=str(error))
        if report["phase"] in report["stages"]:
            report["stages"][report["phase"]]["status"] = "failed"
        write_json(report_path, report)
        raise


def run_dataset(config_file: Path, *, sequential: bool = False,
                run_id_override: str | None = None, dataset_override: Path | None = None,
                model_override: Path | None = None, cache_override: Path | None = None,
                use_cache: bool = True, device_override: str | None = None,
                rehash_model: bool = False) -> dict:
    """Keep one GPU model alive across stages and independent editing cases."""
    print("[启动] 正在初始化 PyTorch 和计算设备...", flush=True)
    from util.gpu_runtime import execution_context
    choices = read_json(config_file)
    execution = dict(choices.get("execution", {}))
    if device_override is not None:
        execution["device"] = device_override
        execution.setdefault("backend", "resident")
    cache_dir = cache_override.resolve() if cache_override is not None else project_path(choices.get("cache_dir", "runs/cache"))
    runtime_settings = {**execution, "hash_cache_path": str(cache_dir / "file_hashes.json"),
                        "rehash_files": rehash_model or not use_cache}
    with execution_context(runtime_settings, choices["target_layers"]):
        return _run_dataset(config_file, sequential=sequential, run_id_override=run_id_override,
                            dataset_override=dataset_override, model_override=model_override,
                            cache_override=cache_override, use_cache=use_cache,
                            execution_override=execution)


def _run_dataset(config_file: Path, *, sequential: bool = False,
                 run_id_override: str | None = None, dataset_override: Path | None = None,
                 model_override: Path | None = None, cache_override: Path | None = None,
                 use_cache: bool = True, execution_override: dict | None = None) -> dict:
    """Prepare the selected dataset, then edit each source case and reuse matching stages."""
    config_file = config_file.resolve()
    choices = read_json(config_file)
    if execution_override:
        choices["execution"] = execution_override
    execution = choices.get("execution", {})
    if "device" in execution or execution.get("backend") == "resident":
        device = execution.setdefault("device", "cuda:0")
        choices.setdefault("p1", {})["device_map"] = device
        choices.setdefault("p2_keep", {})["device"] = device
        choices["p3"]["device"] = device
    if dataset_override is not None:
        choices["dataset"] = str(dataset_override.resolve())
    if model_override is not None:
        choices["model_path"] = str(model_override.resolve())
    issues = config_setup_issues(choices)
    if issues:
        raise ValueError("full-run configuration is incomplete:\n- " + "\n- ".join(issues))
    source_dataset = required_path(choices, "dataset")
    original_model = required_path(choices, "model_path")
    choices["dataset"], choices["model_path"] = str(source_dataset), str(original_model)
    choices["source_dataset"] = str(source_dataset)
    choices["source_dataset_sha256"] = file_sha256(source_dataset)
    cache_dir = cache_override.resolve() if cache_override is not None else project_path(choices.get("cache_dir", "runs/cache"))
    print("[模型身份] 检查模型文件（未变化的权重复用本地摘要）...", flush=True)
    identity_started = time.perf_counter()
    theta0_identity = local_model_identity(original_model)
    print(f"[模型身份] 完成 | {time.perf_counter() - identity_started:.1f}s", flush=True)
    preparation = None
    if choices.get("input_preparation", {}).get("mode", "explicit") == "dataset":
        preparation = prepare_inputs(source_dataset, original_model, tuple(choices["target_layers"]), cache_dir,
                                     settings=choices["input_preparation"].get("settings"), use_cache=use_cache)
        print("[输入准备] " + ("复用缓存：" if preparation["cache_hit"] else "已生成：") + preparation["manifest"], flush=True)
        files = preparation["files"]
        choices["dataset"] = files["editing_cases.json"]
        choices["keep_inputs"] = files["keep_statistics.json"]
        choices.setdefault("p2_keep", {})["recipe_id"] = preparation["recipe_id"]
        choices["p5"].update(dev_edit_file=files["editing_cases.json"],
                              dev_edit_probes=read_json(Path(files["development_probes.json"])),
                              dev_keep_file=files["development_keep.json"])
        if choices["p4"].get("beta", 0) > 0:
            choices["p4"]["keep_train_file"] = files["keep_training.json"]
        choices["prepared_inputs"] = {"manifest": preparation["manifest"], "cache_key": preparation["cache_key"],
                                      "manifest_sha256": file_sha256(preparation["manifest"])}
    for key in ("dataset", "keep_inputs"):
        choices[key] = str(required_path(choices, key))
    for key in ("dev_edit_file", "dev_keep_file"):
        choices["p5"][key] = str(required_path(choices["p5"], key))
    if choices["p4"].get("beta", 0) > 0:
        choices["p4"]["keep_train_file"] = str(required_path(choices["p4"], "keep_train_file"))
    dataset = Path(choices["dataset"])
    cases = load_cases(dataset)
    if not cases:
        raise ValueError("the dataset adapter selected no editing facts")
    case_ids = [case["case_id"] for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("runtime fact IDs must be unique")
    # 同一源案例的多个编辑请求在一轮中联合执行，避免拆散链式事实。
    groups = {}
    for case in cases:
        groups.setdefault(case.get("source_case_id", case["case_id"]), []).append(case["case_id"])
    run_id = run_id_override or choices.get("run_id") or (
        "edit_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    )
    if not isinstance(run_id, str) or run_id in ("", ".", "..") or "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must be one directory name")
    root = project_path(choices.get("output_root", "runs")) / run_id
    root.mkdir(parents=True, exist_ok=False)
    effective_config = root / "effective_config.json"
    write_json(effective_config, choices)
    report_path = root / "run_report.json"
    report = {"status": "running", "mode": "sequential" if sequential else "independent",
              "run_id": run_id, "fact_count": len(case_ids), "case_count": len(groups),
              "source_dataset": str(source_dataset), "original_model": str(original_model),
              "source_dataset_sha256": choices["source_dataset_sha256"],
              "prepared_inputs": preparation["manifest"] if preparation else None,
              "execution": execution,
              "rounds": [], "current_case_id": None, "evaluation_status": "pending"}
    write_json(report_path, report)
    baseline = original_model
    try:
        report["original_model_identity"] = theta0_identity
        write_json(report_path, report)
        for index, (case_id, fact_ids) in enumerate(groups.items()):
            report["current_case_id"] = case_id
            write_json(report_path, report)
            print(f"[round {index + 1}/{len(groups)}] {case_id} | baseline: {baseline}", flush=True)
            baseline_identity = theta0_identity if baseline == original_model else local_model_identity(baseline)
            key = round_key({**choices, "_baseline_path": str(baseline), "_sequential": sequential},
                            fact_ids, baseline_identity, theta0_identity, ROOT)
            index_path = cache_dir / "rounds" / (key + ".json")
            workspace, round_config = None, effective_config
            if use_cache and index_path.is_file():
                entry = read_json(index_path)
                prior_workspace, prior_config = Path(entry["workspace"]), Path(entry["config"])
                if entry.get("key") == key and (prior_workspace / "pipeline_report.json").is_file() \
                        and prior_config.is_file() and file_sha256(prior_config) == entry.get("config_sha256") \
                        and read_json(prior_workspace / "pipeline_report.json").get("request_key") == key:
                    workspace, round_config = prior_workspace, prior_config
            actual_workspace = workspace or root / f"round_{index}"
            if use_cache:
                save_index(index_path, {"key": key, "workspace": str(actual_workspace), "config": str(round_config),
                                        "config_sha256": file_sha256(round_config)})
            result = one_round(round_config, run_id_override=run_id, round_index=index,
                               selected_case_ids=fact_ids, baseline_model=baseline, workspace=workspace,
                               sequential=sequential, original_identity=theta0_identity,
                               request_key=key, use_cache=use_cache, baseline_identity=baseline_identity)
            report["rounds"].append({"round_id": index, "case_id": case_id,
                                     "baseline_model": str(baseline),
                                     "decision": result["decision"],
                                     "selected_alpha": result["selected_alpha"],
                                     "final_model_dir": result["final_model_dir"],
                                     "checkpoint_dir": result["checkpoint_dir"],
                                     "fact_ids": fact_ids, "cache_workspace": workspace is not None,
                                     "report": str(actual_workspace / "pipeline_report.json")})
            write_json(root / f"round_{index}_reference.json", report["rounds"][-1])
            if sequential and result["decision"] == "accepted":
                baseline = Path(result["checkpoint_dir"])
            write_json(report_path, report)
        report.update(status="complete", current_case_id=None,
                      accepted_count=sum(row["decision"] == "accepted" for row in report["rounds"]),
                      rejected_count=sum(row["decision"] == "rejected" for row in report["rounds"]),
                      final_checkpoint_dir=str(baseline) if sequential else None)
        write_json(report_path, report)
        from util.evaluate_accuracy import evaluate_run
        accuracy = evaluate_run(root, device=execution.get("device"))
        report["evaluation_status"] = accuracy["status"]
        report["accuracy_report"] = str(root / "accuracy_report.json")
        report["independent_accuracy"] = accuracy.get("independent")
        report["preservation_accuracy"] = accuracy.get("preservation")
        write_json(root / "evaluation_plan.json", {"status": accuracy["status"],
                   "inputs": accuracy.get("inputs"), "report": report["accuracy_report"],
                   "mode": report["mode"]})
        write_json(report_path, report)
        return report
    except Exception as error:
        report.update(status="failed", failure=str(error))
        write_json(report_path, report)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/full_run.json", help="实验配置 JSON")
    parser.add_argument("--dataset", type=Path, help="本次实验数据集，覆盖配置中的 dataset")
    parser.add_argument("--model-path", type=Path, help="原始模型目录，覆盖配置中的 model_path")
    parser.add_argument("--device", help="计算设备，如 cuda:0（默认配置）或 cpu")
    parser.add_argument("--rehash-model", action="store_true", help="重新读取权重计算摘要，不使用跨运行摘要缓存")
    parser.add_argument("--cache-dir", type=Path, help="预处理和算法产物缓存目录")
    parser.add_argument("--no-cache", action="store_true", help="本次重新计算，不读取已有缓存")
    parser.add_argument("--run-id", help="optional unique output directory name")
    parser.add_argument("--evaluate-run", type=Path,
                        help="只评价已完成实验目录，不重复执行 P0-P5")
    parser.add_argument("mode", nargs="?", choices=("sequential",),
                        help="chain accepted edits; default resets to the original model for each fact")
    parser.add_argument("--sequential", action="store_true", help="same as positional sequential")
    args = parser.parse_args()
    if args.evaluate_run is not None:
        from util.evaluate_accuracy import evaluate_run
        from util.gpu_runtime import execution_context
        run_dir = args.evaluate_run.resolve()
        prior = read_json(run_dir / "run_report.json")
        choices = read_json(run_dir / "effective_config.json")
        execution = dict(choices.get("execution", {}))
        if args.device:
            execution.update(backend="resident", device=args.device)
        with execution_context(execution, choices["target_layers"]):
            accuracy = evaluate_run(run_dir, device=args.device)
        prior["evaluation_status"] = accuracy["status"]
        prior["accuracy_report"] = str(run_dir / "accuracy_report.json")
        prior["independent_accuracy"] = accuracy.get("independent")
        prior["preservation_accuracy"] = accuracy.get("preservation")
        write_json(run_dir / "run_report.json", prior)
        write_json(run_dir / "evaluation_plan.json", {"status": accuracy["status"],
                   "inputs": accuracy.get("inputs"), "report": prior["accuracy_report"],
                   "mode": prior["mode"]})
        print(json.dumps({key: accuracy.get(key) for key in ("status", "independent", "preservation")},
                         ensure_ascii=False, indent=2))
        return
    if not args.config.is_file():
        parser.error(f"configuration file does not exist: {args.config}. Copy "
                     "configs/full_run.template.json to configs/full_run.json and fill in the required inputs")
    selected_choices = read_json(args.config)
    if args.dataset is not None:
        selected_choices["dataset"] = str(args.dataset.resolve())
    if args.model_path is not None:
        selected_choices["model_path"] = str(args.model_path.resolve())
    issues = config_setup_issues(selected_choices)
    if issues:
        parser.error("configuration is incomplete:\n- " + "\n- ".join(issues))
    result = run_dataset(args.config, sequential=bool(args.mode or args.sequential),
                         run_id_override=args.run_id, dataset_override=args.dataset,
                         model_override=args.model_path, cache_override=args.cache_dir, use_cache=not args.no_cache,
                         device_override=args.device, rehash_model=args.rehash_model)
    print(json.dumps({key: result.get(key) for key in (
        "status", "mode", "fact_count", "accepted_count", "rejected_count", "final_checkpoint_dir",
        "evaluation_status", "independent_accuracy", "preservation_accuracy")},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
