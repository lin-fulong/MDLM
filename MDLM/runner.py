"""MDLM 运行入口：一次加载模型，在内存中逐案例执行算法，只保存最终结果。"""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
import uuid
import torch

from MDLM.acceptance import AcceptanceConfig, development_edits
from MDLM.optimize import ResidualOptimizationConfig
from MDLM.workflow import edit_round, reference_examples, save_final_model
from MDLM.writeback import WritebackConfig
from data.readers.json_cases import load_cases
from model.adapters.llada_8b_instruct import LLaDA8BInstructAdapter
from util.collect_denoise_states import collect_trajectories
from util.gpu_runtime import execution_context
from util.io import read_json, write_json
from util.prepare_dataset_inputs import prepare_inputs


ROOT = Path(__file__).resolve().parents[1]


def project_path(value) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def choose_run_id(configured: str | None, override: str | None) -> str:
    run_id = override or configured or (
        "edit_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    if not isinstance(run_id, str) or run_id in ("", ".", "..") or "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must be one directory name")
    return run_id


def stage_settings(kind, values):
    """读取数学参数；旧磁盘流程的路径和版本绑定字段已不再参与计算。"""
    values = dict(values)
    legacy = {"dev_edit_file", "dev_edit_probes", "dev_keep_file", "keep_train_file",
              "reference_model_path", "reference_model_revision", "reference_tokenizer_revision"}
    if values.pop("allow_example_only", False):
        raise ValueError("正式入口不允许 example-only 输入")
    for name in legacy:
        values.pop(name, None)
    if "checkpoints" in values:
        values["checkpoints"] = tuple(values["checkpoints"])
    if "alphas" in values:
        values["alphas"] = tuple(values["alphas"])
    unknown = set(values) - {item.name for item in fields(kind)}
    if unknown:
        raise ValueError(f"{kind.__name__} 中存在不支持的参数：{sorted(unknown)}")
    settings = kind(**values)
    settings.check()
    return settings


def select_cases(cases, case_ids):
    if not case_ids:
        return cases
    selected = set(case_ids)
    available = {case["case_id"] for case in cases} | {case.get("source_case_id", case["case_id"]) for case in cases}
    if selected - available:
        raise ValueError(f"数据集中没有这些可编辑案例：{sorted(selected - available)}")
    return [case for case in cases if case["case_id"] in selected
            or case.get("source_case_id", case["case_id"]) in selected]


def run_dataset(config_file: Path, *, sequential=False, run_id_override=None, dataset_override=None,
                model_override=None, device_override=None, case_ids=None) -> dict:
    choices = read_json(config_file)
    dataset = dataset_override.resolve() if dataset_override is not None else project_path(choices["dataset"])
    model_path = model_override.resolve() if model_override is not None else project_path(choices["model_path"])
    layers = tuple(choices["target_layers"])
    if not layers or len(set(layers)) != len(layers) or any(type(layer) is not int or layer < 0 for layer in layers):
        raise ValueError("target_layers 必须为非空且不重复的非负层编号")
    if choices.get("input_preparation", {}).get("mode", "dataset") != "dataset":
        raise ValueError("只支持从数据集自动准备实验输入")
    cases = load_cases(dataset)
    if not cases or any(case["quality_control"]["status"] == "example_only" for case in cases):
        raise ValueError("正式入口需要非空且非 example-only 的编辑事实")
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("runtime fact IDs must be unique")
    selected = select_cases(cases, case_ids)
    groups = {}
    for case in selected:
        groups.setdefault(case.get("source_case_id", case["case_id"]), []).append(case)

    execution = dict(choices.get("execution", {}))
    if device_override is not None:
        execution["device"] = device_override
        execution.setdefault("backend", "resident")
    p1 = dict(choices.get("p1", {}))
    device = execution.get("device", "cuda:0") if execution.get("backend") == "resident" else p1.get("device_map", "auto")
    dtype = p1.pop("torch_dtype", "auto")
    p1.pop("device_map", None)
    q_values, keep_values = dict(choices["p3"]), dict(choices.get("p2_keep", {}))
    keep_values.pop("recipe_id", None)
    if "device" in execution or execution.get("backend") == "resident":
        device = execution.get("device", "cuda:0")
        q_values["device"] = keep_values["device"] = device
    keep_values["batch_keep"] = execution.get("backend") == "resident" and keep_values.get("device", "cpu").startswith("cuda")
    q = WritebackConfig(**q_values)
    optimization = stage_settings(ResidualOptimizationConfig, choices["p4"])
    acceptance = stage_settings(AcceptanceConfig, choices["p5"])

    run_id = choose_run_id(choices.get("run_id"), run_id_override)
    output = project_path(choices.get("output_root", "runs")) / run_id
    output.mkdir(parents=True, exist_ok=False)
    report = {"editor": "mdlm", "status": "running", "run_id": run_id,
              "mode": "sequential" if sequential else "independent",
              "fact_count": len(selected), "case_count": len(groups),
              "source_dataset": str(dataset), "original_model": str(model_path),
              "selected_case_ids": case_ids, "rounds": [], "current_case_id": None}
    try:
        with execution_context(execution, layers):
            adapter = LLaDA8BInstructAdapter.from_local_tokenizer(model_path=model_path, target_layers=layers)
            # 完整数据集先划分，再选择编辑案例，保持与原流程相同的训练/开发用途。
            prepared = prepare_inputs(cases, adapter, settings=choices.get("input_preparation", {}).get("settings"))
            model = adapter.load_model(model_path, torch_dtype=dtype, device_map=device)
            if execution.get("backend") == "resident" and str(device).startswith("cuda"):
                requested = torch.device(device)
                if requested.index is None:
                    requested = torch.device("cuda", torch.cuda.current_device())
                if any(parameter.device != requested for parameter in model.parameters()):
                    raise RuntimeError("模型未完整加载到指定 GPU，停止运行以避免 CPU 卸载")
            adapter.inspect_target_modules(model)
            originals = {name: model.get_submodule(name).weight.detach().cpu().clone()
                         for name in adapter.target_module_names()}
            dev_keeps = reference_examples(model, adapter, prepared["development"])
            train_keeps = reference_examples(model, adapter, prepared["training"]) if optimization.beta > 0 else []
            for index, (case_id, group) in enumerate(groups.items()):
                report["current_case_id"] = case_id
                print(f"[round {index + 1}/{len(groups)}] {case_id}", flush=True)
                baseline = adapter.snapshot_target_weights(model)
                try:
                    trajectories = collect_trajectories(model, adapter, group, **p1)
                    fact_ids = {case["case_id"] for case in group}
                    probes = tuple(item for item in prepared["development_probes"] if item.case_id in fact_ids)
                    dev_edits = development_edits(group, probes, adapter)
                    result = edit_round(
                        model, adapter, trajectories, prepared["statistics"], dev_edits, dev_keeps, originals,
                        writeback=q, optimization=optimization, acceptance=acceptance, keep_train=train_keeps,
                        p2_keys=choices.get("p2_keys"), p2_keep=keep_values)
                    final_dir = None
                    if not sequential and result["decision"] == "accepted":
                        final_dir = save_final_model(model, adapter, output / f"case_{index}",
                                                     model_path=model_path, torch_dtype=dtype, device_map=device)
                    report["rounds"].append({"round_id": index, "case_id": case_id,
                        "fact_ids": [case["case_id"] for case in group], "decision": result["decision"],
                        "selected_alpha": result["selected_alpha"],
                        "selected_step": result["optimization"]["selected_step"], "selection": result["selection"],
                        "final_model_dir": str(final_dir) if final_dir is not None else None})
                except Exception:
                    adapter.restore_target_weights(model, baseline)
                    raise
                if not sequential:
                    adapter.restore_target_weights(model, baseline)
            final_dir = save_final_model(model, adapter, output / "final_model", model_path=model_path,
                                         torch_dtype=dtype, device_map=device) if sequential else None
            report.update(status="complete", current_case_id=None,
                          accepted_count=sum(row["decision"] == "accepted" for row in report["rounds"]),
                          rejected_count=sum(row["decision"] == "rejected" for row in report["rounds"]),
                          final_checkpoint_dir=str(final_dir) if final_dir is not None else None)
    except Exception as error:
        report.update(status="failed", failure=str(error))
        raise
    finally:
        write_json(output / "run_report.json", report)
    return report
