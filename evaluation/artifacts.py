"""只读解析 MDLM 最终产物；拒绝案例和连续模式由报告明确区分。"""

import hashlib
from pathlib import Path

from data.dataset import load_dataset
from util.io import project_path, read_json


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_run(run_dir):
    run_dir = Path(run_dir).resolve()
    report = read_json(run_dir / "run_report.json")
    if report.get("editor") != "mdlm" or report.get("status") != "complete":
        raise ValueError("只评估已正常完成的 MDLM 编辑运行")
    mode = report.get("mode")
    if mode not in ("independent", "sequential") or not report.get("rounds"):
        raise ValueError("编辑报告必须包含有效模式和非空 rounds")
    rounds, artifacts, sources, facts = [], {}, set(), set()

    def checkpoint(reference, expected):
        # 允许整体移动运行目录，但不信任任意目录或缺失权重。
        if not reference or str(reference).replace("\\", "/").rstrip("/").split("/")[-1] != expected:
            raise ValueError(f"编辑报告缺少或错误引用 {expected}")
        path = run_dir / expected
        if path.resolve().parent != run_dir:
            raise ValueError("模型产物必须位于当前运行目录内")
        metadata = read_json(path / "model.json")
        weights = path / "target_weights.safetensors"
        artifacts[str(path)] = {"metadata": metadata, "weights_sha256": file_digest(weights),
                                "metadata_sha256": file_digest(path / "model.json")}
        return path

    final = checkpoint(report.get("final_checkpoint_dir"), "final_model") if mode == "sequential" else None
    # 连续模式统一回测最终模型；独立模式只为接受案例加载各自权重。
    for index, row in enumerate(report["rounds"]):
        source, ids, decision = row.get("case_id"), row.get("fact_ids"), row.get("decision")
        if (not isinstance(source, str) or not source or source in sources or row.get("round_id") != index
                or decision not in ("accepted", "rejected") or not isinstance(ids, list) or not ids
                or any(not isinstance(item, str) or not item for item in ids)
                or len(set(ids)) != len(ids) or facts.intersection(ids)):
            raise ValueError(f"编辑报告的第 {index} 轮包含非法或重复的案例/事实")
        sources.add(source)
        facts.update(ids)
        path = final
        if mode == "independent":
            if decision == "accepted":
                path = checkpoint(row.get("final_model_dir"), f"case_{index}")
            elif row.get("final_model_dir") is not None:
                raise ValueError("独立模式拒绝案例不应带有最终权重")
        rounds.append({**row, "checkpoint": path})
    if report.get("case_count") != len(rounds) or report.get("fact_count") != len(facts):
        raise ValueError("编辑报告的案例/事实数量不一致")
    return report, rounds, artifacts


def read_evaluation_data(report, dataset_override=None):
    saved = report.get("dataset_reading", {})
    options = dict(saved.get("reader_options", {}))
    # 主体标注不决定评估问题和答案，评估不要求重建编辑主体。
    options.pop("subject_path", None)
    config = {"dataset_reader": saved.get("reader", "native"),
              "dataset_path": saved.get("source_path", report["source_dataset"]),
              "source_limit": saved.get("source_limit"), "reader_options": options,
              "seed": saved.get("seed", report.get("seed", 2026))}
    bundle = load_dataset(config, dataset_path_override=dataset_override)
    if saved.get("case_count") is not None and saved["case_count"] != len(bundle.cases):
        raise ValueError("当前数据读取数量与编辑报告不一致")
    return bundle


def model_settings(report, artifacts, config, override=None):
    metadata = [item["metadata"] for item in artifacts.values()]
    for item in metadata:
        if item.get("editor") != "mdlm" or item.get("model_path") != report["original_model"]:
            raise ValueError("模型产物与编辑报告的原始模型不一致")
    if metadata:
        layers, dtype = metadata[0]["target_layers"], metadata[0]["torch_dtype"]
        if any(item["target_layers"] != layers or item["torch_dtype"] != dtype for item in metadata):
            raise ValueError("同一运行的模型产物具有不同目标层或精度")
    else:
        # 全部拒绝的旧独立运行没有 model.json，必须显式提供原模型设置。
        if "target_layers" not in config or "torch_dtype" not in config:
            raise ValueError("全部拒绝的独立运行需在评估配置提供 target_layers 和 torch_dtype")
        layers, dtype = config["target_layers"], config["torch_dtype"]
    return {"model_path": str(Path(override).resolve() if override else project_path(report["original_model"])),
            "target_layers": layers, "torch_dtype": dtype}
