"""仅对已完成编辑的产物进行评估，完全独立于 MDLM 编辑运行器。"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

from util.io import choose_run_id, config_digest, project_path, read_config, write_json
from .artifacts import file_digest, model_settings, read_evaluation_data, read_run
from .registry import get_framework
from .reporting import write_record


def run_evaluation(config_file, *, run_dir=None, device=None, model_path=None, dataset_path=None, eval_id=None):
    # 模型依赖只在执行评估时导入；--help 不加载模型或导入 MDLM。
    import torch
    from util.gpu_runtime import execution_context
    from .inference import LLaDABackend, PROTOCOL, validate_generation

    config = read_config(Path(config_file))
    framework = get_framework(config.get("framework", "ultraedit"))
    if config.get("protocol", PROTOCOL) != PROTOCOL:
        raise ValueError(f"当前只实现 {PROTOCOL}，自由生成/长答案协议尚未接入")
    generation = validate_generation(config.get("generation", {}))
    if run_dir is None and not config.get("run_dir"):
        raise ValueError("请通过 --run-dir 或评估配置指定已完成的编辑目录")
    run_dir = Path(run_dir).resolve() if run_dir is not None else project_path(config["run_dir"])
    report_hash = file_digest(run_dir / "run_report.json")
    report, rounds, artifacts = read_run(run_dir)
    seed = config.get("seed", report.get("seed", 2026))
    if type(seed) is not int or seed < 0:
        raise ValueError("评估 seed 必须为非负整数")
    bundle = read_evaluation_data(report, dataset_override=dataset_path)
    source_hash = file_digest(bundle.metadata["source_path"])
    groups = config.get("groups", list(framework.metrics))
    tasks = framework.build_tasks(bundle.cases, rounds, groups)
    settings = model_settings(report, artifacts, config, override=model_path)
    device = device or config.get("device", "cuda:0")
    if not isinstance(device, str) or not (device == "cpu" or device == "cuda" or device.startswith("cuda:")):
        raise ValueError("评估 device 需为 cpu、cuda 或 cuda:N")
    identifier = choose_run_id(config.get("eval_id"), eval_id)
    output = project_path(config.get("output_root", "evaluations")) / identifier
    for protected in (run_dir, Path(settings["model_path"])):
        if output.resolve().is_relative_to(protected.resolve()):
            raise ValueError("评估输出不能位于编辑产物或原始模型目录内")
    output.mkdir(parents=True, exist_ok=False)
    summary = framework.aggregate(tasks, groups)
    accepted = sum(row["decision"] == "accepted" for row in rounds)
    # 运行口径只定义一次，运行清单和最终汇总使用相同统计。
    run_info = {
        "framework": config.get("framework", "ultraedit"), "protocol": PROTOCOL,
        "mode": report["mode"], "case_count": len(rounds), "accepted_count": accepted,
        "rejected_count": len(rounds) - accepted, "acceptance_rate": accepted / len(rounds),
    }
    manifest = {
        **run_info,
        "protocol_description": "参考答案仅决定窗口长度；全 mask 自主去噪；不是原自回归 teacher forcing",
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
    backend, current, status, failure = None, None, "running", None
    baseline = {}
    try:
        # 驻留精度设置仅作用于独立评估进程，退出时由原公共上下文恢复。
        with execution_context({"backend": "resident", "device": device}), \
                (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
            backend = LLaDABackend(settings, device)
            manifest["tokenizer_class"] = type(backend.adapter.tokenizer).__name__
            manifest["mask_token_id"] = backend.adapter.mask_token_id
            manifest["model_config"] = backend.model.config.to_dict() if hasattr(backend.model.config, "to_dict") else None
            manifest["evaluation_code_sha256"] = {
                str(path.relative_to(Path(__file__).parent)): file_digest(path)
                for path in sorted(Path(__file__).parent.rglob("*.py"))}
            write_json(output / "manifest.json", manifest)
            for index, task in enumerate(tasks):
                current = task.task_id
                request, target = backend.prepare(task, seed)
                prediction = backend.predict(request, generation)
                signature = config_digest(asdict(request))
                baseline[task.task_id] = (prediction, target, signature)
                write_record(stream, {"phase": "before", "task_id": task.task_id,
                                      "case_id": task.case_id, "probe_id": task.probe_id,
                                      "prompt": task.prompt, "answer": task.answer,
                                      "prediction": asdict(prediction), "target_ids": target,
                                      "request_sha256": signature, "seed": request.seed})
                if index % 100 == 0:
                    print(f"[baseline {index + 1}/{len(tasks)}]", flush=True)
            by_checkpoint = {}
            for task in tasks:
                by_checkpoint.setdefault(task.checkpoint, []).append(task)
            for checkpoint, selected in by_checkpoint.items():
                if checkpoint is not None:
                    if file_digest(checkpoint / "target_weights.safetensors") != artifacts[str(checkpoint)]["weights_sha256"]:
                        raise ValueError("评估期间模型产物发生变化")
                    backend.use_checkpoint(checkpoint)
                print(f"[evaluate] {checkpoint or 'original (rejected)'}: {len(selected)} probes", flush=True)
                for task in selected:
                    current = task.task_id
                    before, target, signature = baseline.pop(task.task_id)
                    if checkpoint is None:
                        # 独立拒绝案例的最终模型就是 theta_0，复用完全相同的配对预测。
                        after = before
                    else:
                        request, repeated_target = backend.prepare(task, seed)
                        if config_digest(asdict(request)) != signature or repeated_target != target:
                            raise ValueError("基线与编辑后的评估输入不一致")
                        after = backend.predict(request, generation)
                    scores = framework.score(before, after, target)
                    summary.add(task, scores)
                    write_record(stream, {"phase": "paired", "task_id": task.task_id,
                        "source_case_id": task.source_case_id, "case_id": task.case_id,
                        "probe_id": task.probe_id, "group": task.group, "decision": task.decision,
                        "checkpoint": str(checkpoint) if checkpoint else None,
                        "before": asdict(before), "after": asdict(after), "scores": scores})
            current = None
            if file_digest(run_dir / "run_report.json") != report_hash or file_digest(bundle.metadata["source_path"]) != source_hash:
                raise ValueError("评估期间编辑报告或源数据发生变化")
            for path, item in artifacts.items():
                if (file_digest(Path(path) / "model.json") != item["metadata_sha256"]
                        or file_digest(Path(path) / "target_weights.safetensors") != item["weights_sha256"]):
                    raise ValueError("评估期间模型产物发生变化")
            status = "complete"
    except BaseException as exc:
        status, failure = "failed", str(exc)
        raise
    finally:
        try:
            if backend is not None:
                backend.close()
        except BaseException as exc:
            status, failure = "failed", f"恢复评估模型失败: {exc}; 原错误: {failure}"
            raise
        finally:
            result = {**run_info, "status": status, "failure": failure, "current_task_id": current,
                      "eval_id": identifier, "output_dir": str(output),
                      "expected_tasks": len(tasks), "completed_tasks": summary.completed,
                      "error_count": int(failure is not None),
                      "coverage": summary.completed / len(tasks), "metrics": summary.export()}
            write_json(output / "summary.json", result)
    return result
