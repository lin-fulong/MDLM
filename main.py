"""Run LLaDA knowledge editing for every selected fact (P0 through P5).

The input JSON freezes experiment choices. Existing stage implementations own the
math and their artifact checks; this file only prepares inputs and orders them.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import uuid
from dataclasses import dataclass
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


def choose_run_id(configured: str | None, override: str | None) -> str:
    run_id = override or configured or (
        "edit_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    )
    if not isinstance(run_id, str) or run_id in ("", ".", "..") or "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must be one directory name")
    return run_id


def config_setup_issues(choices: dict) -> list[str]:
    """Check the two user-supplied paths; all stage inputs are derived from the dataset."""
    paths = {"dataset": choices.get("dataset"), "model_path": choices.get("model_path")}
    issues = []
    for label, value in paths.items():
        if not isinstance(value, str) or not value or value.startswith("REPLACE_WITH_"):
            issues.append(f"{label}: specify a real path")
        elif not project_path(value).exists():
            issues.append(f"{label}: file or directory does not exist: {project_path(value)}")
    if choices.get("input_preparation", {}).get("mode", "dataset") != "dataset":
        issues.append("input_preparation.mode: only automatic dataset preparation is supported")
    return issues


@dataclass(frozen=True)
class RoundInputs:
    choices: dict
    dataset: Path
    original_model: Path
    model: Path
    keep_inputs: Path
    dev_edit_file: Path
    dev_keep_file: Path
    layers: list[int]
    cases: list[dict]
    facts: list[FactSelection]
    keys: ReferenceKeyConfig
    keep: KeepStatisticsConfig
    q: WritebackConfig
    residual: ResidualOptimizationConfig
    p5_values: dict
    original_identity: dict
    baseline_identity: dict


def load_round_inputs(config_file: Path, baseline_model: Path | None,
                      selected_case_ids: list[str] | None, original_identity: dict | None,
                      baseline_identity: dict | None) -> RoundInputs:
    """Validate one round's frozen inputs and stage settings."""
    choices = read_json(config_file)
    dataset = required_path(choices, "dataset")
    original_model = required_path(choices, "model_path")
    model = baseline_model.resolve() if baseline_model is not None else original_model
    keep_inputs = required_path(choices, "keep_inputs")
    p5_values = dict(choices["p5"])
    dev_edit_file = required_path(p5_values, "dev_edit_file")
    dev_keep_file = required_path(p5_values, "dev_keep_file")
    for label, file in (("dataset", dataset), ("keep_inputs", keep_inputs),
                        ("dev_edit_file", dev_edit_file), ("dev_keep_file", dev_keep_file)):
        if not file.is_file():
            raise ValueError(f"{label} must be a file: {file}")
    if not model.is_dir() or not original_model.is_dir():
        raise ValueError("model_path and baseline_model must be model directories")
    layers = choices["target_layers"]
    if not isinstance(layers, list) or not layers or any(type(n) is not int or n < 0 for n in layers):
        raise ValueError("target_layers must be a nonempty list of nonnegative layer indices")
    names = {f"model.transformer.blocks.{layer}.ff_out" for layer in layers}
    if len(names) != len(layers):
        raise ValueError("target_layers must not contain duplicates")
    if read_json(keep_inputs)["example_only"] or read_json(dev_keep_file)["example_only"]:
        raise ValueError("main.py requires non-example P2 and P5 preservation inputs")

    cases = load_cases(dataset)
    if selected_case_ids is not None:
        cases = [case for case in cases if case["case_id"] in selected_case_ids]
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
    theta0 = original_identity or local_model_identity(original_model)
    identity = baseline_identity or (theta0 if model == original_model else local_model_identity(model))
    residual_values = dict(choices["p4"])
    if "checkpoints" in residual_values:
        residual_values["checkpoints"] = tuple(residual_values["checkpoints"])
    if residual_values.get("beta", 0) > 0:
        residual_values.setdefault("reference_model_path", str(original_model))
        residual_values.setdefault("reference_model_revision", theta0["model_revision"])
        residual_values.setdefault("reference_tokenizer_revision", theta0["tokenizer_revision"])
        residual_values["keep_train_file"] = str(required_path(residual_values, "keep_train_file"))
    residual = ResidualOptimizationConfig(**residual_values)
    residual.check()
    if set(q.ridge_by_layer) != names or set(residual.rho_by_layer) != names:
        raise ValueError("P3/P4 layer names must match target_layers exactly")
    if q.allow_example_only or residual.allow_example_only or p5_values.get("allow_example_only", False):
        raise ValueError("the main entrypoint does not allow example-only stage inputs")
    if residual.beta > 0 and (project_path(residual.reference_model_path) != original_model
                              or residual.reference_model_revision != theta0["model_revision"]
                              or residual.reference_tokenizer_revision != theta0["tokenizer_revision"]):
        raise ValueError("P4 theta_0 reference must match the original model directory")
    return RoundInputs(choices, dataset, original_model, model, keep_inputs, dev_edit_file,
                       dev_keep_file, layers, cases, facts, keys, keep, q, residual,
                       p5_values, theta0, identity)


def start_round(config_file: Path, inputs: RoundInputs, run_id_override: str | None,
                round_index: int, sequential: bool, workspace: Path | None,
                request_key: str | None) -> tuple[Path, dict]:
    """Create the round directory and its initial report."""
    run_id = choose_run_id(inputs.choices.get("run_id"), run_id_override)
    output_root = project_path(inputs.choices.get("output_root", "runs"))
    if workspace is not None:
        output_root, run_id = workspace.parent.parent, workspace.parent.name
        round_index = int(workspace.name.removeprefix("round_"))
    round_dir = output_root / run_id / f"round_{round_index}"
    round_dir.mkdir(parents=True, exist_ok=workspace is not None)
    report_path = round_dir / "pipeline_report.json"
    report = {"status": "running", "phase": "freeze_inputs", "run_id": run_id,
              "round_id": round_index, "mode": "sequential" if sequential else "independent",
              "baseline_model": str(inputs.model), "original_model": str(inputs.original_model),
              "fact_count": len(inputs.facts), "target_layers": inputs.layers, "stages": {},
              "source_dataset": str(inputs.dataset), "source_dataset_sha256": file_sha256(inputs.dataset),
              "source_keep_inputs": str(inputs.keep_inputs), "source_dev_edit_file": str(inputs.dev_edit_file),
              "source_dev_keep_file": str(inputs.dev_keep_file),
              "config_file": str(config_file.resolve()), "config_sha256": file_sha256(config_file)}
    if workspace is not None and report_path.is_file():
        existing = read_json(report_path)
        if existing.get("request_key") != request_key:
            raise ValueError("缓存运行与当前请求不匹配")
        report["stages"] = existing.get("stages", {})
    report["request_key"] = request_key
    write_json(report_path, report)
    return round_dir, report


class StageRunner:
    """Execute one stage and reuse only matching, intact artifacts."""

    def __init__(self, round_dir: Path, report: dict, request_key: str | None,
                 use_cache: bool, resident: bool):
        self.round_dir = round_dir
        self.report = report
        self.request_key = request_key
        self.use_cache = use_cache
        self.resident = resident
        self.upstream = {}

    def run(self, label: str, arguments: list[str], artifact: Path,
            accepted: tuple[str, ...] = ("complete",)) -> dict:
        report_path = self.round_dir / "pipeline_report.json"
        dependency = config_digest({"arguments": arguments, "configs": {
            str(path): file_sha256(path) for path in sorted(self.round_dir.glob("*_config.json"))},
            "upstream": self.upstream, "request_key": self.request_key})
        prior = self.report["stages"].get(label, {})
        if (self.use_cache and prior.get("status") in accepted and prior.get("dependency") == dependency
                and files_match(prior.get("files_sha256", {}))):
            result = read_json(artifact)
            status = result["status"] if isinstance(result, dict) else "complete"
            if status in accepted:
                prior["cache_hit"] = True
                self.upstream[label] = prior["files_sha256"]
                print(f"[{label}] 复用本地缓存：{artifact}", flush=True)
                return result
        self.report["phase"] = label
        write_json(report_path, self.report)
        log_path = self.round_dir / f"{label.lower()}.log"
        self.report["stages"][label] = {"status": "running", "artifact": str(artifact), "log": str(log_path)}
        write_json(report_path, self.report)
        print(f"[{label}] {log_path}", flush=True)
        target = artifact if artifact.parent == self.round_dir else artifact.parent
        archive_partial(target, self.round_dir / "failed_attempts", label)
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as stream:
            try:
                from util.gpu_runtime import execute_stage
                execute_stage(arguments, stream=stream, cwd=ROOT, resident=self.resident)
            except subprocess.CalledProcessError:
                if label != "P4" or not artifact.is_file() or read_json(artifact).get("status") != "rejected":
                    raise
        result = read_json(artifact)
        status = result["status"] if isinstance(result, dict) else "complete"
        if status not in accepted:
            raise RuntimeError(f"{label} artifact status is {status!r}: {artifact}")
        digests = snapshot(target)
        self.report["stages"][label] = {"status": status, "artifact": str(artifact), "log": str(log_path),
                                        "dependency": dependency, "files_sha256": digests, "cache_hit": False,
                                        "elapsed_seconds": time.perf_counter() - started}
        self.upstream[label] = digests
        write_json(report_path, self.report)
        print(f"[{label}] {status} | {self.report['stages'][label]['elapsed_seconds']:.1f}s", flush=True)
        return result


def freeze_round_configs(inputs: RoundInputs, round_dir: Path, run_id: str,
                         round_index: int) -> P1RunConfig:
    """Write exactly the frozen inputs and settings consumed by P0–P5."""
    write_json(round_dir / "model_identity.json", inputs.baseline_identity)
    case_file = round_dir / "editing_cases.json"
    write_json(case_file, inputs.cases)
    write_json(round_dir / "keep_inputs.json", read_json(inputs.keep_inputs))
    write_json(round_dir / "source_config.json", inputs.choices)
    p1_values = dict(inputs.choices.get("p1", {}))
    p1 = P1RunConfig(
        run_id=run_id, round_id=round_index, baseline_id=inputs.baseline_identity["model_revision"],
        model_name="LLaDA-8B-Instruct", model_path=str(inputs.model),
        model_revision=inputs.baseline_identity["model_revision"],
        tokenizer_revision=inputs.baseline_identity["tokenizer_revision"],
        case_file=str(case_file), facts=tuple(inputs.facts), target_layers=tuple(inputs.layers),
        window_policy_id="short_answer_exact_tokens_v1",
        sampler_policy_id="confidence_no_remask_v1",
        seed=p1_values.pop("seed", 2026), steps=p1_values.pop("steps", None),
        temperature=p1_values.pop("temperature", 0.0),
        replay_count=p1_values.pop("replay_count", 1),
        output_root=str(round_dir.parent.parent),
        torch_dtype=p1_values.pop("torch_dtype", "auto"),
        device_map=p1_values.pop("device_map", "auto"),
    )
    if p1_values:
        raise ValueError(f"unsupported P1 settings: {sorted(p1_values)}")
    p1.check()
    p5_values = dict(inputs.p5_values)
    p5_values.update(dev_edit_file=str(inputs.dev_edit_file), dev_keep_file=str(inputs.dev_keep_file),
                     reference_model_path=str(inputs.original_model),
                     reference_model_revision=inputs.original_identity["model_revision"],
                     reference_tokenizer_revision=inputs.original_identity["tokenizer_revision"])
    selected = {fact.case_id for fact in inputs.facts}
    p5_values["dev_edit_probes"] = [probe for probe in p5_values["dev_edit_probes"]
                                    if probe["case_id"] in selected]
    if not p5_values["dev_edit_probes"]:
        raise ValueError("当前编辑案例没有配置相应的 P5 开发探针")
    p5 = AcceptanceConfig.from_record(p5_values)
    for name, value in (("p1", p1.as_record()), ("p2_keys", inputs.keys.as_record()),
                        ("p2_keep", inputs.keep.as_record()), ("p3", inputs.q.as_record()),
                        ("p4", inputs.residual.as_record()), ("p5", p5.as_record())):
        write_json(round_dir / f"{name}_config.json", value)
    return p1


def check_p0(rows: list[dict], facts: list[FactSelection], module: str) -> None:
    """Require a real full-mask forward and replay for every selected fact."""
    if len(rows) != len(facts) or [(row["case_id"], row["probe_id"]) for row in rows] != [
        (fact.case_id, fact.probe_id) for fact in facts
    ] or any(not row["real_forward"] or not row["replay_within_tolerance"]
             or not row["full_mask_input"] or row["mask_count"] != row["target_token_count"]
             or set(row["target_modules"]) != {module}
             or set(row["ffn_input_shapes"]) != {module} for row in rows):
        raise RuntimeError(f"P0 did not cover and replay all {len(facts)} facts at {module}")


def run_round_stages(inputs: RoundInputs, runner: StageRunner, p1: P1RunConfig) -> dict | None:
    """Run P0 through P5 in their fixed dependency order."""
    round_dir = runner.round_dir
    case_file = round_dir / "editing_cases.json"
    for layer in inputs.layers:
        label = f"P0_LAYER_{layer}"
        module = f"model.transformer.blocks.{layer}.ff_out"
        artifact = round_dir / f"p0_layer_{layer}.json"
        rows = runner.run(label, ["util.smoke_p0_llada", str(case_file), "--model-path", str(inputs.model),
                                  "--layer", str(layer), "--forward", "--output", str(artifact),
                                  "--device-map", p1.device_map, "--torch-dtype", p1.torch_dtype], artifact)
        check_p0(rows, inputs.facts, module)
    p1_dir = round_dir / "p1"
    runner.run("P1", ["util.collect_denoise_states", str(round_dir / "p1_config.json")],
               p1_dir / "coverage_report.json")
    runner.run("P2_KEYS", ["util.extract_reference_keys", str(p1_dir),
                           "--config", str(round_dir / "p2_keys_config.json")],
               round_dir / "matrices/reference_keys/manifest.json")
    keep_command = ["util.extract_keep_statistics", str(p1_dir), str(round_dir / "keep_inputs.json"),
                    "--config", str(round_dir / "p2_keep_config.json")]
    if runner.resident and runner.report["mode"] == "independent":
        keep_command.append("--reuse-across-rounds")
    batch_keep = runner.resident and inputs.keep.device.startswith("cuda")
    if batch_keep:
        keep_command.append("--batch-keep")
    runner.run("P2_KEEP", keep_command,
               round_dir / "matrices/keep_statistics/manifest.json")
    qa_command = ["util.validate_matrix_artifacts", str(p1_dir)]
    if batch_keep:
        qa_command.append("--batch-keep")
    runner.run("P2_QA", qa_command,
               round_dir / "matrices/validation/manifest.json")
    runner.run("P3", ["util.solve_writeback_factor", str(p1_dir),
                      "--config", str(round_dir / "p3_config.json")],
               round_dir / "matrices/writeback_factors/manifest.json")
    p4 = runner.run("P4", ["util.optimize_residuals", str(p1_dir),
                            "--config", str(round_dir / "p4_config.json")],
                    round_dir / "residuals/manifest.json", ("complete", "rejected"))
    if p4["status"] == "rejected":
        return None
    return runner.run("P5", ["util.merge_and_accept", str(p1_dir),
                             "--config", str(round_dir / "p5_config.json"),
                             "--residual-dir", str(round_dir / "residuals"),
                             "--output-dir", str(round_dir / "editing")],
                      round_dir / "editing/acceptance_report.json")


def verify_p5(round_dir: Path, result: dict) -> str:
    """Check that the saved P5 artifact matches its acceptance decision."""
    decision = result["decision"]
    if decision not in ("accepted", "rejected"):
        raise RuntimeError(f"unexpected P5 decision: {decision!r}")
    manifest = read_json(round_dir / "editing/manifest.json")
    if manifest["status"] != decision or manifest["example_only"] or not manifest["p5_stage_complete"]:
        raise RuntimeError("P5 final manifest does not represent a completed non-example round")
    return decision


def reload_accepted_model(runner: StageRunner) -> None:
    """Verify that an accepted P5 model can be reconstructed from disk."""
    round_dir, report = runner.round_dir, runner.report
    p5_dir = round_dir / "editing"
    report["phase"] = "P5_RELOAD"
    write_json(round_dir / "pipeline_report.json", report)
    dependency = config_digest(runner.upstream)
    prior = report["stages"].get("P5_RELOAD", {})
    if runner.use_cache and prior.get("dependency") == dependency and files_match(prior.get("files_sha256", {})):
        return
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


def export_accepted_checkpoint(inputs: RoundInputs, runner: StageRunner) -> Path:
    """Write the accepted weights as the next sequential baseline."""
    round_dir, report = runner.round_dir, runner.report
    report["phase"] = "CHECKPOINT_EXPORT"
    write_json(round_dir / "pipeline_report.json", report)
    from util.export_edited_checkpoint import export_checkpoint
    checkpoint_dir = round_dir / "checkpoint"
    dependency = config_digest(runner.upstream)
    prior = report["stages"].get("CHECKPOINT_EXPORT", {})
    if not (runner.use_cache and prior.get("dependency") == dependency
            and files_match(prior.get("files_sha256", {}))):
        archive_partial(checkpoint_dir, round_dir / "failed_attempts", "CHECKPOINT_EXPORT")
        export_checkpoint(inputs.model, round_dir / "editing", checkpoint_dir)
        report["stages"]["CHECKPOINT_EXPORT"] = {
            "status": "complete", "artifact": str(checkpoint_dir / "export_report.json"),
            "dependency": dependency, "files_sha256": snapshot(checkpoint_dir)}
    return checkpoint_dir


def one_round(config_file: Path, *, run_id_override: str | None = None,
              round_index: int = 0, baseline_model: Path | None = None,
              sequential: bool = False, original_identity: dict | None = None,
              selected_case_ids: list[str] | None = None, workspace: Path | None = None,
              request_key: str | None = None, use_cache: bool = True,
              baseline_identity: dict | None = None) -> dict:
    """Orchestrate one source case without implementing stage mathematics."""
    inputs = load_round_inputs(config_file, baseline_model, selected_case_ids,
                               original_identity, baseline_identity)
    round_dir, report = start_round(config_file, inputs, run_id_override, round_index,
                                    sequential, workspace, request_key)
    report_path = round_dir / "pipeline_report.json"
    runner = StageRunner(round_dir, report, request_key, use_cache,
                         inputs.choices.get("execution", {}).get("backend") == "resident")
    try:
        p1 = freeze_round_configs(inputs, round_dir, report["run_id"], report["round_id"])
        p5 = run_round_stages(inputs, runner, p1)
        if p5 is None:
            report.update(status="complete", phase="complete", decision="rejected", selected_alpha=None,
                          decision_reason="P4 未得到合格残差候选", final_model_dir=None,
                          checkpoint_dir=None, baseline_restored=True)
        else:
            decision = verify_p5(round_dir, p5)
            if decision == "accepted":
                reload_accepted_model(runner)
            checkpoint = export_accepted_checkpoint(inputs, runner) if sequential and decision == "accepted" else None
            report.update(status="complete", phase="complete", decision=p5["decision"],
                          selected_alpha=p5["selection"]["selected_alpha"],
                          final_model_dir=str(round_dir / "editing") if p5["decision"] == "accepted" else None,
                          checkpoint_dir=str(checkpoint) if checkpoint is not None else None,
                          baseline_restored=p5["baseline_restored"])
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


@dataclass(frozen=True)
class DatasetInputs:
    choices: dict
    source_dataset: Path
    original_model: Path
    dataset: Path
    cache_dir: Path
    original_identity: dict
    preparation: dict


def prepare_dataset_run(config_file: Path, dataset_override: Path | None,
                        model_override: Path | None, cache_override: Path | None,
                        use_cache: bool, execution_override: dict | None) -> DatasetInputs:
    """Prepare all stage inputs from the selected dataset before starting rounds."""
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
    choices["model_path"] = str(original_model)
    choices["source_dataset"] = str(source_dataset)
    choices["source_dataset_sha256"] = file_sha256(source_dataset)
    cache_dir = cache_override.resolve() if cache_override is not None else project_path(choices.get("cache_dir", "runs/cache"))
    print("[模型身份] 检查模型文件（未变化的权重复用本地摘要）...", flush=True)
    started = time.perf_counter()
    identity = local_model_identity(original_model)
    print(f"[模型身份] 完成 | {time.perf_counter() - started:.1f}s", flush=True)
    preparation = prepare_inputs(source_dataset, original_model, tuple(choices["target_layers"]), cache_dir,
                                 settings=choices.get("input_preparation", {}).get("settings"), use_cache=use_cache)
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
    return DatasetInputs(choices, source_dataset, original_model, Path(choices["dataset"]),
                         cache_dir, identity, preparation)


def record_evaluation(root: Path, report: dict, device: str | None) -> dict:
    """Evaluate completed rounds and save the two summary reports."""
    from util.evaluate_accuracy import evaluate_run
    accuracy = evaluate_run(root, device=device)
    report["evaluation_status"] = accuracy["status"]
    report["accuracy_report"] = str(root / "accuracy_report.json")
    report["independent_accuracy"] = accuracy.get("independent")
    report["preservation_accuracy"] = accuracy.get("preservation")
    write_json(root / "evaluation_plan.json", {"status": accuracy["status"],
               "inputs": accuracy.get("inputs"), "report": report["accuracy_report"],
               "mode": report["mode"]})
    write_json(root / "run_report.json", report)
    return accuracy


def _run_dataset(config_file: Path, *, sequential: bool = False,
                 run_id_override: str | None = None, dataset_override: Path | None = None,
                 model_override: Path | None = None, cache_override: Path | None = None,
                 use_cache: bool = True, execution_override: dict | None = None) -> dict:
    """Edit each source case and aggregate the completed rounds."""
    config_file = config_file.resolve()
    prepared = prepare_dataset_run(config_file, dataset_override, model_override, cache_override,
                                   use_cache, execution_override)
    choices, source_dataset, original_model = prepared.choices, prepared.source_dataset, prepared.original_model
    dataset, cache_dir = prepared.dataset, prepared.cache_dir
    theta0_identity, preparation = prepared.original_identity, prepared.preparation
    execution = choices.get("execution", {})
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
    run_id = choose_run_id(choices.get("run_id"), run_id_override)
    root = project_path(choices.get("output_root", "runs")) / run_id
    root.mkdir(parents=True, exist_ok=False)
    effective_config = root / "effective_config.json"
    write_json(effective_config, choices)
    report_path = root / "run_report.json"
    report = {"status": "running", "mode": "sequential" if sequential else "independent",
              "run_id": run_id, "fact_count": len(case_ids), "case_count": len(groups),
              "source_dataset": str(source_dataset), "original_model": str(original_model),
              "source_dataset_sha256": choices["source_dataset_sha256"],
              "prepared_inputs": preparation["manifest"],
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
        record_evaluation(root, report, execution.get("device"))
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
        from util.gpu_runtime import execution_context
        run_dir = args.evaluate_run.resolve()
        prior = read_json(run_dir / "run_report.json")
        choices = read_json(run_dir / "effective_config.json")
        execution = dict(choices.get("execution", {}))
        if args.device:
            execution.update(backend="resident", device=args.device)
        with execution_context(execution, choices["target_layers"]):
            accuracy = record_evaluation(run_dir, prior, args.device)
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
