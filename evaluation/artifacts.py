"""只读解析 MDLM 最终产物；拒绝案例和连续模式由报告明确区分。"""

import hashlib
from pathlib import Path

from data.dataset import load_dataset
from util.io import project_path, read_json


def file_digest(path):
    """分块计算文件 SHA-256，用于核查评价期间输入是否变化。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_run(run_dir):
    """读取完整编辑报告，为每轮确定 after 权重并登记产物摘要。"""
    run_dir = Path(run_dir).resolve()
    report = read_json(run_dir / "run_report.json")
    mode = report.get("mode")
    assert (report.get("editor") == "mdlm" and report.get("status") == "complete"
            and mode in ("independent", "sequential") and report.get("rounds")), \
        "需要已完成的 MDLM 报告、有效编辑模式和非空 rounds"
    rounds, artifacts, sources, facts = [], {}, set(), set()

    def checkpoint(reference, expected):
        """在当前运行目录定位产物，校验引用并读取元数据和摘要。"""
        # 允许整体移动运行目录，但不信任任意目录或缺失权重。
        assert reference and str(reference).replace("\\", "/").rstrip("/").split("/")[-1] == expected, \
            f"编辑报告缺少或错误引用 {expected}"
        path = run_dir / expected
        assert path.resolve().parent == run_dir, "模型产物必须位于当前运行目录内"
        metadata = read_json(path / "model.json")
        weights = path / "target_weights.safetensors"
        artifacts[str(path)] = {"metadata": metadata, "weights_sha256": file_digest(weights),
                                "metadata_sha256": file_digest(path / "model.json")}
        return path

    final = checkpoint(report.get("final_checkpoint_dir"), "final_model") if mode == "sequential" else None
    # after = θ_case（独立接受）、θ_0（独立拒绝）或 θ_final（连续模式）。
    for index, row in enumerate(report["rounds"]):
        source, ids, decision = row.get("case_id"), row.get("fact_ids"), row.get("decision")
        assert (isinstance(source, str) and source and source not in sources and row.get("round_id") == index
                and decision in ("accepted", "rejected") and isinstance(ids, list) and ids
                and all(isinstance(item, str) and item for item in ids)
                and len(set(ids)) == len(ids) and not facts.intersection(ids)), \
            f"编辑报告的第 {index} 轮包含非法或重复的案例/事实"
        sources.add(source)
        facts.update(ids)
        path = final
        if mode == "independent":
            if decision == "accepted":
                path = checkpoint(row.get("final_model_dir"), f"case_{index}")
            else:
                assert row.get("final_model_dir") is None, "独立模式拒绝案例不应带有最终权重"
        rounds.append({**row, "checkpoint": path})
    assert report.get("case_count") == len(rounds) and report.get("fact_count") == len(facts), \
        "编辑报告的案例/事实数量不一致"
    return report, rounds, artifacts


def read_evaluation_data(report, dataset_override=None):
    """恢复编辑时的读取器、范围和选题种子，不重新读取编辑保持池。"""
    saved = report.get("dataset_reading", {})
    options = dict(saved.get("reader_options", {}))
    # 主体标注不决定评估问题和答案，评估不要求重建编辑主体。
    options.pop("subject_path", None)
    config = {"dataset_reader": saved.get("reader", "native"),
              "dataset_path": saved.get("source_path", report["source_dataset"]),
              "source_limit": saved.get("source_limit"), "reader_options": options,
              "seed": saved.get("seed", report.get("seed", 2026))}
    bundle = load_dataset(config, dataset_path_override=dataset_override)
    assert saved.get("case_count") is None or saved["case_count"] == len(bundle.cases), \
        "当前数据读取数量与编辑报告不一致"
    return bundle


def model_settings(report, artifacts, config, override=None):
    """从产物恢复层号和加载精度；全部拒绝时由评价配置补充。"""
    metadata = [item["metadata"] for item in artifacts.values()]
    if metadata:
        layers, dtype = metadata[0]["target_layers"], metadata[0]["torch_dtype"]
        expected = ("mdlm", report["original_model"], layers, dtype)
        assert all((item["editor"], item["model_path"], item["target_layers"], item["torch_dtype"]) == expected
                   for item in metadata), "模型产物的来源、目标层或精度与编辑报告不一致"
    else:
        # 全部拒绝时没有 model.json，直接从评价配置读取必要设置。
        layers, dtype = config["target_layers"], config["torch_dtype"]
    return {"model_path": str(Path(override).resolve() if override else project_path(report["original_model"])),
            "target_layers": layers, "torch_dtype": dtype}
