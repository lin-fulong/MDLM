"""仅对已完成编辑的产物进行评估，完全独立于 MDLM 编辑运行器。"""

from dataclasses import asdict
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import platform
import sys

from util.io import choose_run_id, config_digest, project_path, read_config, write_json
from .artifacts import file_digest, model_settings, read_evaluation_data, read_run
from .framework import get_framework
from .reporting import write_record


def run_evaluation(config_file, *, run_dir=None, device=None, model_path=None, dataset_path=None, eval_id=None):
    """读取产物 → 全部 before → 按权重分组 after → 评分和保存。"""
    # 模型依赖只在执行评估时导入；--help 不加载模型或导入 MDLM。
    import torch
    from util.gpu_runtime import execution_context
    from .inference import (GOLD_PROTOCOL, PROTOCOL, PROTOCOLS, LLaDABackend,
                            generate, prepare_request)

    config = read_config(Path(config_file))
    framework = get_framework(config.get("framework", "ultraedit"))
    protocol = config.get("protocol", PROTOCOL)
    description = PROTOCOLS[protocol]
    generation = {"steps": None, "temperature": 0.0, **config.get("generation", {})}
    run_dir = Path(run_dir).resolve() if run_dir is not None else project_path(config["run_dir"])
    report_hash = file_digest(run_dir / "run_report.json")
    report, rounds, artifacts = read_run(run_dir)
    seed = config.get("seed", report.get("seed", 2026))
    bundle = read_evaluation_data(report, dataset_override=dataset_path)
    source_hash = file_digest(bundle.metadata["source_path"])
    groups = config.get("groups", list(framework.metrics))
    tasks = framework.build_tasks(bundle.cases, rounds, groups)
    settings = model_settings(report, artifacts, config, override=model_path)
    device = device or config.get("device", "cuda:0")
    identifier = choose_run_id(config.get("eval_id"), eval_id)
    output = project_path(config.get("output_root", "evaluations")) / identifier
    output.mkdir(parents=True, exist_ok=False)
    summary = framework.aggregate(tasks, groups)
    accepted = sum(row["decision"] == "accepted" for row in rounds)
    # 运行口径只定义一次，运行清单和最终汇总使用相同统计。
    run_info = {
        "framework": config.get("framework", "ultraedit"), "protocol": protocol,
        "mode": report["mode"], "case_count": len(rounds), "accepted_count": accepted,
        "rejected_count": len(rounds) - accepted, "acceptance_rate": accepted / len(rounds),
    }
    manifest = {
        **run_info,
        "protocol_description": description,
        "prediction_semantics": ("各位置被选中时的评分预测；完全匹配及一致率不代表自由生成"
                                 if protocol == GOLD_PROTOCOL else "自主去噪最终答案"),
        "trace_positions": "zero_based_answer_offsets; step_count=len(trace); count=len(positions)",
        "generation": generation, "seed": seed, "device": device, "model": settings,
        "run_dir": str(run_dir), "run_report_sha256": report_hash, "artifacts": artifacts,
        "dataset": bundle.metadata, "source_sha256_at_evaluation": source_hash,
        "source_identity_at_edit_time": "not_recorded_in_edit_report",
        "base_weights_identity_at_edit_time": "not_recorded_in_edit_report",
        "locality_scope": "benchmark_preservation; independence_from_edit_keep_pool_not_verified",
        "aggregation": "case_macro_primary; probe_macro_matches_UltraEdit_averaging",
        "groups": groups, "expected_tasks": len(tasks),
        "config": config, "python": platform.python_version(), "torch": torch.__version__,
    }
    task_hash = hashlib.sha256()
    for task in tasks:
        task_hash.update(json.dumps(asdict(task), ensure_ascii=False, sort_keys=True, default=str).encode())
        task_hash.update(b"\n")
    manifest["tasks_sha256"] = task_hash.hexdigest()
    write_json(output / "manifest.json", manifest)
    current, status, failure = None, "running", None
    baseline = {}
    try:
        # 驻留精度设置仅作用于独立评估进程，退出时由原公共上下文恢复。
        with execution_context({"backend": "resident", "device": device}), \
                (output / "predictions.jsonl").open("x", encoding="utf-8") as stream, ExitStack() as resources:
            backend = LLaDABackend(settings, device)
            resources.callback(backend.close)
            manifest["tokenizer_class"] = type(backend.adapter.tokenizer).__name__
            manifest["mask_token_id"] = backend.adapter.mask_token_id
            manifest["model_config"] = backend.model.config.to_dict() if hasattr(backend.model.config, "to_dict") else None
            manifest["evaluation_code_sha256"] = {
                str(path.relative_to(Path(__file__).parent)): file_digest(path)
                for path in sorted(Path(__file__).parent.rglob("*.py"))}
            write_json(output / "manifest.json", manifest)
            for index, task in enumerate(tasks):
                current = task.task_id
                request, target = prepare_request(backend.adapter, task, seed, protocol)
                prediction = generate(backend.model, backend.adapter, request, **generation,
                                      protocol=protocol, target=target if protocol == GOLD_PROTOCOL else None)
                request_data = asdict(request)
                baseline[task.task_id] = (prediction, target, request)
                write_record(stream, {"phase": "before", "task_id": task.task_id,
                                      "case_id": task.case_id, "probe_id": task.probe_id,
                                      "prompt": task.prompt, "answer": task.answer,
                                      "prediction": asdict(prediction), "target_ids": target,
                                      "request": request_data, "request_sha256": config_digest(request_data),
                                      "seed": request.seed})
                if index % 100 == 0:
                    print(f"[baseline {index + 1}/{len(tasks)}]", flush=True)
            current = None
            by_checkpoint = {}
            for task in tasks:
                by_checkpoint.setdefault(task.checkpoint, []).append(task)
            for checkpoint, selected in by_checkpoint.items():
                if checkpoint is not None:
                    backend.use_checkpoint(checkpoint)
                print(f"[evaluate] {checkpoint or 'original (rejected)'}: {len(selected)} probes", flush=True)
                for task in selected:
                    current = task.task_id
                    before, target, request = baseline.pop(task.task_id)
                    if checkpoint is None:
                        # 独立拒绝案例的最终模型就是 theta_0，复用完全相同的配对预测。
                        after = before
                    else:
                        # 复用不可变请求与标准标签，两侧各自从全 mask 开始预测。
                        after = generate(backend.model, backend.adapter, request, **generation,
                                         protocol=protocol, target=target if protocol == GOLD_PROTOCOL else None)
                    scores = framework.score(before, after, target)
                    write_record(stream, {"phase": "paired", "task_id": task.task_id,
                        "source_case_id": task.source_case_id, "case_id": task.case_id,
                        "probe_id": task.probe_id, "group": task.group, "decision": task.decision,
                        "checkpoint": str(checkpoint) if checkpoint else None,
                        "before": asdict(before), "after": asdict(after), "scores": scores})
                    # 只有逐题结果成功保存，才计入已完成数量。
                    summary.add(task, scores)
                    current = None
        status = "complete"
    finally:
        # 上下文退出和权重恢复成功后才算完成；异常在记录后自然传播。
        if status != "complete":
            status, failure = "failed", str(sys.exc_info()[1])
        # coverage = 已成功保存的配对题数 / 预期题数；失败状态始终随汇总保存。
        result = {**run_info, "status": status, "failure": failure, "current_task_id": current,
                  "eval_id": identifier, "output_dir": str(output),
                  "expected_tasks": len(tasks), "completed_tasks": summary.completed,
                  "error_count": int(failure is not None),
                  "coverage": summary.completed / len(tasks), "metrics": summary.export()}
        write_json(output / "summary.json", result)
    return result
