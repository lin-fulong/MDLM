"""MDLM 实验入口：固定最初参照，按源案例执行单轮编辑，保存最终结果。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import uuid

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
    """配置内的相对路径以项目根目录为起点。"""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def choose_run_id(configured: str | None, override: str | None) -> str:
    """按命令行、配置、自动生成的优先级选取单层输出目录名。"""
    run_id = override or configured or (
        "edit_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    if not isinstance(run_id, str) or run_id in ("", ".", "..") or "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must be one directory name")
    return run_id


def select_cases(cases, case_ids):
    """按源案例或事实编号选例，保留数据集顺序；源案例会选中其全部事实。"""
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
    """独立模式每案例从 theta_0 开始；连续模式接续上一案例的接受结果。

    每案例执行一次固定监督、残差优化与写回验收，不执行同一事实集的可选刷新。
    """
    choices = read_json(config_file)
    dataset = dataset_override.resolve() if dataset_override is not None else project_path(choices["dataset"])
    model_path = model_override.resolve() if model_override is not None else project_path(choices["model_path"])
    layers = tuple(choices["target_layers"])
    cases = load_cases(dataset)
    # 同一源案例的事实联合编辑；分组顺序沿用数据集，不按编号排序。
    selected = select_cases(cases, case_ids)
    groups = {}
    for case in selected:
        groups.setdefault(case.get("source_case_id", case["case_id"]), []).append(case)

    execution = dict(choices.get("execution", {}))
    if device_override is not None:
        execution["device"] = device_override
        execution.setdefault("backend", "resident")
    p1 = dict(choices.get("p1", {}))
    dtype = p1.pop("torch_dtype", "auto")
    device = p1.pop("device_map", "auto")
    q_values, keep_values = dict(choices["p3"]), dict(choices.get("p2_keep", {}))
    keep_values.pop("recipe_id", None)
    # 执行设备优先于 p1.device_map，并沿用 C/Q 的设备与保持批处理设置。
    if "device" in execution or execution.get("backend") == "resident":
        device = execution.get("device", "cuda:0")
        q_values["device"] = keep_values["device"] = device
    keep_values["batch_keep"] = execution.get("backend") == "resident" and keep_values.get("device", "cpu").startswith("cuda")
    q = WritebackConfig(**q_values)
    optimization = ResidualOptimizationConfig(**choices["p4"])
    acceptance = AcceptanceConfig(**choices["p5"])

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
            # 用完整 cases 划分保持输入，不让 --case-id 改变训练/开发集合。
            prepared = prepare_inputs(cases, adapter, settings=choices.get("input_preparation", {}).get("settings"))
            model = adapter.load_model(model_path, torch_dtype=dtype, device_map=device)
            # 公式（10）（16）（17）：W_0 与 p_theta_0 在首次编辑前固定，之后不刷新。
            originals = {name: model.get_submodule(name).weight.detach().cpu().clone()
                         for name in adapter.target_module_names()}
            dev_keeps = reference_examples(model, adapter, prepared["development"])
            train_keeps = reference_examples(model, adapter, prepared["training"]) if optimization.beta > 0 else []
            for index, (case_id, group) in enumerate(groups.items()):
                report["current_case_id"] = case_id
                print(f"[round {index + 1}/{len(groups)}] {case_id}", flush=True)
                # 文档 3.1（未编号）：theta_b 为当前模型；快照用于本轮恢复。
                baseline = adapter.snapshot_target_weights(model)
                try:
                    # 公式（1）～（3）的前置采样：使用本轮 theta_b，自主生成固定轨迹。
                    trajectories = collect_trajectories(model, adapter, group, **p1)
                    fact_ids = {case["case_id"] for case in group}
                    probes = tuple(item for item in prepared["development_probes"] if item.case_id in fact_ids)
                    dev_edits = development_edits(group, probes, adapter)
                    # 公式（1）～（13）（15）～（18）：固定监督、重算 K/C/Q，从 R=0 优化并验收。
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
                # 独立模式保存结果后恢复 theta_0；连续模式保留本轮接受模型，再处理下一案例。
                if not sequential:
                    adapter.restore_target_weights(model, baseline)
            # 文档 11.2（未编号）：theta_final=theta_acc；只保存最终权重和运行摘要。
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
