"""只读解析 MDLM 最终产物；拒绝案例和连续模式由报告明确区分。"""

import hashlib
from pathlib import Path

from data.dataset import load_dataset
from util.io import project_path, read_json


def file_digest(path):
    """分块计算文件 SHA-256，记录评价使用的输入版本。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_run(run_dir):
    """读取完整编辑报告，为每轮确定 after 权重并登记产物摘要。"""
    run_dir = Path(run_dir).resolve()
    report = read_json(run_dir / "run_report.json")
    mode = report["mode"]
    rounds, artifacts = [], {}

    def checkpoint(expected):
        """按 MDLM 保存规则定位产物，允许整体移动运行目录。"""
        path = run_dir / expected
        metadata = read_json(path / "model.json")
        weights = path / "target_weights.safetensors"
        artifacts[str(path)] = {"metadata": metadata, "weights_sha256": file_digest(weights),
                                "metadata_sha256": file_digest(path / "model.json")}
        return path

    final = checkpoint("final_model") if mode == "sequential" else None
    # after = θ_case（独立接受）、θ_0（独立拒绝）或 θ_final（连续模式）。
    for index, row in enumerate(report["rounds"]):
        decision = row["decision"]
        path = final
        if mode == "independent":
            if decision == "accepted":
                path = checkpoint(f"case_{index}")
        rounds.append({**row, "checkpoint": path})
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
    return bundle


def model_settings(report, artifacts, config, override=None):
    """从产物恢复层号和加载精度；全部拒绝时由评价配置补充。"""
    metadata = [item["metadata"] for item in artifacts.values()]
    if metadata:
        layers, dtype = metadata[0]["target_layers"], metadata[0]["torch_dtype"]
    else:
        # 全部拒绝时没有 model.json，直接从评价配置读取必要设置。
        layers, dtype = config["target_layers"], config["torch_dtype"]
    return {"model_path": str(Path(override).resolve() if override else project_path(report["original_model"])),
            "target_layers": layers, "torch_dtype": dtype}
