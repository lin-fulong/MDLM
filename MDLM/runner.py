"""MDLM 运行流程：固定最初参照，按源案例编辑，保存和重载最终结果。"""

from __future__ import annotations

from pathlib import Path
import math
import sys

import torch
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm

from MDLM.merge import AcceptanceConfig, WeightMerge, development_edits, select_merged_candidate
from MDLM.optimize import (CandidateModel, KeepTrainingExample, ResidualOptimizationConfig,
                           log_probabilities, optimize_residuals)
from MDLM.states import (build_supervision, collect_trajectories, edit_training_examples,
                         keep_input_from_record, prepare_inputs)
from MDLM.writeback import WritebackConfig, extract_keys, keep_matrices, solve_writeback_factor
from data.dataset import load_dataset
from model.registry import get_adapter
from util.gpu_runtime import execution_context
from util.experiment_log import ExperimentLog
from util.io import choose_run_id, project_path, read_config, read_json, write_json
from util.resource_usage import EditResourceUsage


def select_cases(cases, case_ids):
    """按源案例或事实编号选例，保留数据集顺序；源案例会选中其全部事实。"""
    if not case_ids:
        return cases
    selected = set(case_ids)
    available = {case["case_id"] for case in cases} | {case.get("source_case_id", case["case_id"]) for case in cases}
    if selected - available:
        sys.exit(f"没有这些可编辑案例：{sorted(selected - available)}")
    return [case for case in cases if case["case_id"] in selected
            or case.get("source_case_id", case["case_id"]) in selected]


def layer_coefficients(value, names) -> dict:
    """λ/ρ 标量用于全部目标层；列表按模型目标层顺序逐项对应。"""
    if isinstance(value, (int, float)):
        result = dict.fromkeys(names, value)
    else:
        result = dict(zip(names, value, strict=True))
    if any(not math.isfinite(x) or x <= 0 for x in result.values()):
        sys.exit("ridge/rho 必须为有限正数")
    return result


def run_dataset(config_file: Path, **options) -> dict:
    """先建立运行报告；输入错误、原生运行错误和主动终止都保存失败原因。"""
    choices = read_config(config_file)
    run_id = choose_run_id(choices.get("run_id"), options.pop("run_id_override", None))
    output = project_path(choices.get("output_root", "runs")) / run_id
    output.mkdir(parents=True, exist_ok=False)
    execution = choices.get("execution", {})
    device = options.get("device_override") or execution.get("device", choices.get("device_map", "auto"))
    monitor = EditResourceUsage(output, use_cuda=str(device).startswith("cuda") or device == "auto")
    report = {"editor": "mdlm", "status": "running", "run_id": run_id,
              "rounds": [], "current_case_id": None}
    mode = choices.get("mode", "independent") if options.get("sequential") is None else (
        "sequential" if options["sequential"] else "independent")
    dataset = options.get("dataset_override") or choices.get("dataset_path")
    model = options.get("model_override") or choices.get("model_path")
    log = ExperimentLog(
        output, algorithm=choices.get("editor", "mdlm"), model=model, dataset=dataset, kind="edit",
        parameters={"mode": mode, "layers": "-".join(map(str, choices.get("target_layers", []))),
                    "steps": choices.get("optimization_steps", 20),
                    "ridge": choices.get("ridge", 0.01), "rho": choices.get("rho", 0.001),
                    "seed": choices.get("seed", 2026),
                    "cases": options.get("max_cases_override") or choices.get("max_cases") or "all"})
    with log.capture():
        try:
            return _run_dataset(choices, output, report, resource_monitor=monitor, **options)
        finally:
            if report["status"] != "complete":
                report.update(status="failed", failure=str(sys.exc_info()[1]))
            report.update(monitor.finish(report["status"]))
            report["log_file"] = str(log.path)
            log.finish(report)
            write_json(output / "run_report.json", report)


def _run_dataset(choices, output, report, *, sequential=None, dataset_override=None,
                model_override=None, device_override=None, case_ids=None, max_cases_override=None,
                resource_monitor=None) -> dict:
    """独立模式每案例从 theta_0 开始；连续模式接续上一案例的接受结果。

    每案例执行一次固定监督、残差优化与写回验收，不执行同一事实集的可选刷新。
    """
    seed = choices.get("seed", 2026)
    if type(seed) is not int or seed < 0:
        sys.exit("随机种子必须为非负整数")
    # 主配置种子统一控制保持划分、去噪采样和残差优化。
    input_settings = {"seed": seed, **{name: choices[key] for name, key in (
        ("dev_fraction", "keep_dev_fraction"), ("test_fraction", "keep_test_fraction"),
        ("statistics_scale", "statistics_scale")) if key in choices}}
    mode = choices.get("mode", "independent") if sequential is None else (
        "sequential" if sequential else "independent")
    if mode not in ("independent", "sequential"):
        sys.exit("mode 必须为 independent 或 sequential")
    sequential = mode == "sequential"
    case_ids = case_ids if case_ids is not None else choices.get("case_ids")
    max_cases = max_cases_override if max_cases_override is not None else choices.get("max_cases")
    if max_cases is not None and (type(max_cases) is not int or max_cases < 1):
        sys.exit("max_cases 必须为正整数或 null")
    dataset = dataset_override.resolve() if dataset_override is not None else project_path(choices["dataset_path"])
    model_path = model_override.resolve() if model_override is not None else project_path(choices["model_path"])
    layers = tuple(choices["target_layers"])
    dataset_bundle = load_dataset(choices, dataset_path_override=dataset)
    cases = dataset_bundle.cases
    if not cases or len({case["case_id"] for case in cases}) != len(cases):
        sys.exit("编辑案例必须非空且事实 ID 唯一")
    # 同一源案例的事实联合编辑；分组顺序沿用数据集，不按编号排序。
    selected = select_cases(cases, case_ids)
    groups = {}
    for case in selected:
        groups.setdefault(case.get("source_case_id", case["case_id"]), []).append(case)
    # 先选例、再按源案例限量；不拆散同源事实，也不截断保持输入的数据源。
    if max_cases is not None:
        groups = dict(list(groups.items())[:max_cases])
    execution = dict(choices.get("execution", {}))
    if device_override is not None:
        execution["device"] = device_override
        execution.setdefault("backend", "resident")
    key_options = {name: choices["key_" + name] for name in ("replay_atol", "replay_rtol")
                   if "key_" + name in choices}
    sampling = {"seed": seed, "steps": choices.get("denoise_steps"), "temperature": choices.get("temperature", 0.0),
                "replay_count": choices.get("replay_count", 1)}
    dtype, device = choices.get("torch_dtype", "auto"), choices.get("device_map", "auto")
    compute_device = "cpu"
    # 执行设备优先于模型 device_map；C/Q 沿用同一设备与保持批处理规则。
    if "device" in execution or execution.get("backend") == "resident":
        device = execution.get("device", "cuda:0")
        compute_device = device
    keep_options = {"device": compute_device,
                    "batch_keep": execution.get("backend") == "resident" and compute_device.startswith("cuda")}
    optimization_values = {name: choices[key] for name, key in (
        ("beta", "beta"), ("learning_rate", "learning_rate"), ("max_steps", "optimization_steps"),
        ("checkpoints", "selection_steps"), ("microbatch_size", "microbatch_size")) if key in choices}
    acceptance = AcceptanceConfig(**{key: choices[key] for key in (
        "alphas", "delta", "tau_g", "tau_u", "epsilon", "equivalence_atol", "equivalence_rtol", "chunk_size")
        if key in choices})

    report.update(seed=seed, mode=mode, max_cases=max_cases,
                  fact_count=sum(len(group) for group in groups.values()), case_count=len(groups),
                  source_dataset=str(dataset), original_model=str(model_path),
                  dataset_reading=dataset_bundle.metadata, selected_case_ids=case_ids)
    with execution_context(execution):
        resource_monitor.start_gpu()
        adapter = get_adapter(model_path=model_path, target_layers=layers)
        # 公式（6）（10）：算法提供 λ/ρ，具体模块名由模型适配器确定。
        names = adapter.target_module_names()
        q = WritebackConfig(layer_coefficients(choices["ridge"], names), device=compute_device)
        optimization = ResidualOptimizationConfig(
            layer_coefficients(choices["rho"], names), seed=seed, **optimization_values)
        if (not math.isfinite(optimization.beta) or optimization.beta < 0
                or type(optimization.max_steps) is not int or optimization.max_steps < 1
                or type(optimization.microbatch_size) is not int or optimization.microbatch_size < 1
                or not optimization.checkpoints or any(type(n) is not int or not 1 <= n <= optimization.max_steps
                                                      for n in optimization.checkpoints)):
            sys.exit("残差优化预算、检查点、微批大小或 beta 无效")
        # 保持池来自显式 keep_path 或完整 cases，不随 case_ids/max_cases 改变。
        prepared = prepare_inputs(cases, adapter, settings=input_settings,
                                  keep_pool=dataset_bundle.keep_pool)
        model = adapter.load_model(model_path, torch_dtype=dtype, device_map=device)
        # 公式（10）（16）（17）：W_0 与 p_theta_0 在首次编辑前固定，之后不刷新。
        originals = {name: model.get_submodule(name).weight.detach().cpu().clone()
                     for name in adapter.target_module_names()}
        dev_keeps = reference_examples(model, adapter, prepared["development"])
        train_keeps = reference_examples(model, adapter, prepared["training"]) if optimization.beta > 0 else []
        for index, (case_id, group) in enumerate(tqdm(
                groups.items(), total=len(groups), desc="Editing", unit="case",
                disable=not sys.stderr.isatty())):
            report["current_case_id"] = case_id
            resource_monitor.start_case(case_id)
            # 文档 3.1（未编号）：theta_b 为当前模型；快照用于本轮恢复。
            baseline = adapter.snapshot_target_weights(model)
            round_complete = False
            try:
                # 公式（1）～（3）的前置采样：使用本轮 theta_b，自主生成固定轨迹。
                trajectories = collect_trajectories(model, adapter, group, **sampling)
                fact_ids = {case["case_id"] for case in group}
                probes = tuple(item for item in prepared["development_probes"] if item.case_id in fact_ids)
                dev_edits = development_edits(group, probes, adapter)
                # 公式（1）～（13）（15）～（18）：固定监督、重算 K/C/Q，从 R=0 优化并验收。
                result = edit_round(
                    model, adapter, trajectories, prepared["statistics"], dev_edits, dev_keeps, originals,
                    writeback=q, optimization=optimization, acceptance=acceptance, keep_train=train_keeps,
                    keep_options=keep_options, key_options=key_options)
                final_dir = None
                if not sequential and result["decision"] == "accepted":
                    final_dir = save_final_model(model, adapter, output / f"case_{index}",
                                                 model_path=model_path, torch_dtype=dtype, device_map=device)
                report["rounds"].append({"round_id": index, "case_id": case_id,
                    "fact_ids": [case["case_id"] for case in group], "decision": result["decision"],
                    "selected_alpha": result["selected_alpha"],
                    "selected_step": result["optimization"]["selected_step"], "selection": result["selection"],
                    "final_model_dir": str(final_dir) if final_dir is not None else None})
                round_complete = True
            finally:
                if not round_complete:
                    adapter.restore_target_weights(model, baseline)
            # 独立模式保存结果后恢复 theta_0；连续模式保留本轮接受模型，再处理下一案例。
            if not sequential:
                adapter.restore_target_weights(model, baseline)
            resource_monitor.finish_case()
        # 文档 11.2（未编号）：theta_final=theta_acc；只保存最终权重和运行摘要。
        final_dir = save_final_model(model, adapter, output / "final_model", model_path=model_path,
                                     torch_dtype=dtype, device_map=device) if sequential else None
    report.update(status="complete", current_case_id=None,
                  accepted_count=sum(row["decision"] == "accepted" for row in report["rounds"]),
                  rejected_count=sum(row["decision"] == "rejected" for row in report["rounds"]),
                  final_checkpoint_dir=str(final_dir) if final_dir is not None else None)
    return report


@torch.no_grad()
def reference_examples(model, adapter, rows) -> list[KeepTrainingExample]:
    """公式（10）（16）（17）：缓存最初 theta_0 的完整词表 log p，供保持 KL 复用。

    runner 在首次编辑前调用，训练与开发保持输入分别准备，保留给定位置与权重。
    """
    examples = []
    for row in rows:
        model_input = keep_input_from_record(row["input"])
        positions, weights = tuple(row["positions"]), tuple(row["weights"])
        logs = log_probabilities(adapter.forward_logits(model, model_input, positions)).cpu()
        examples.append(KeepTrainingExample(model_input, positions, weights, logs))
    return examples


def edit_round(model, adapter, trajectories, statistics, dev_edits, dev_keeps, originals, *,
               writeback: WritebackConfig, optimization: ResidualOptimizationConfig,
               acceptance: AcceptanceConfig, keep_train=(), keep_options=None, key_options=None) -> dict:
    """执行一轮编辑：theta_b 为传入模型，originals 始终为最初 W_0。"""
    # 公式（1）～（3）：筛选固定兼容状态，沿用事实/状态/位置的全局监督权重。
    facts, supervision = build_supervision(trajectories, adapter.mask_token_id)
    edits = edit_training_examples(facts)

    # 公式（4）～（9）：按事实顺序提取 K，逐层累计 C、解 HX=K，得到固定 Q=Xᵀ。
    keys = extract_keys(model, adapter, trajectories, **(key_options or {}))
    factors = {}
    for name, matrix in keep_matrices(model, adapter, statistics, **(keep_options or {})):
        factors[name] = solve_writeback_factor(
            keys[name], matrix, writeback.ridge_by_layer[name], writeback)
        del matrix

    # 公式（10）～（12）：R 从零开始，Q 和基准权重固定，多层分支联合优化。
    with CandidateModel(model, factors, dtype=optimization.residual_dtype) as candidate:
        optimized = optimize_residuals(candidate, adapter, edits, list(keep_train), optimization)
    result = {"supervision": supervision, "optimization": optimized.report}
    if optimized.residuals is None:
        # 无有限、非零且优于 J(0) 的 R 时结束本轮；退出候选上下文已移除分支。
        return {**result, "decision": "rejected", "selected_alpha": None,
                "baseline_restored": True, "selection": None}

    # 公式（13）（15）～（18）：从 W_b 重建 αRQ 试步，按三个开发条件接受或恢复。
    merger = WeightMerge(model, factors, optimized.residuals,
                         compute_dtype=acceptance.compute_dtype, chunk_size=acceptance.chunk_size)
    selection = select_merged_candidate(merger, adapter, factors, optimized.residuals,
                                        dev_edits, dev_keeps, originals, acceptance)
    return {**result, "decision": selection["status"], "selected_alpha": selection["selected_alpha"],
            "baseline_restored": selection["baseline_restored"], "selection": selection}


def save_final_model(model, adapter, output: Path, *, model_path: Path,
                     torch_dtype="auto", device_map="auto") -> Path:
    """文档 11.2（未编号）：保存 theta_final=theta_acc 的目标权重及重载参数。"""
    output.mkdir(parents=True, exist_ok=False)
    weights = {name: model.get_submodule(name).weight.detach().cpu().contiguous().clone()
               for name in adapter.target_module_names()}
    save_file(weights, str(output / "target_weights.safetensors"))
    write_json(output / "model.json", {"editor": "mdlm", "model_path": str(model_path.resolve()),
               "target_layers": list(adapter.target_layers), "torch_dtype": torch_dtype,
               "device_map": device_map})
    return output


def load_final_model(output_dir: str | Path, *, device_map: str | None = None):
    """加载原始模型，再覆盖已保存的最终目标权重；仍需 model.json 指向的原模型。"""
    output = Path(output_dir)
    metadata = read_json(output / "model.json")
    adapter = get_adapter(
        model_path=metadata["model_path"], target_layers=tuple(metadata["target_layers"]))
    model = adapter.load_model(metadata["model_path"], torch_dtype=metadata["torch_dtype"],
                               device_map=device_map or metadata["device_map"])
    weights = load_file(str(output / "target_weights.safetensors"))
    adapter.restore_target_weights(model, weights)
    return adapter, model
