"""读取配置、保存结果，并生成确定性数据划分所需的摘要。"""

import hashlib
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    """读取 UTF-8 JSON 配置或最终模型元数据。"""
    return json.loads(path.read_text(encoding="utf-8"))


def read_config(path: Path) -> dict:
    """合并主配置引用的三个 JSON；引用相对主配置目录，主配置同名项优先。"""
    main = read_json(path)
    values = {}
    for key in ("model_config", "dataset_config", "editor_config"):
        if key in main:
            values.update(read_json(path.parent / main.pop(key)))
    return {**values, **main}


def write_json(path: Path, value: object) -> None:
    """以固定缩进和 UTF-8 保存运行摘要或最终模型元数据。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def config_digest(values: dict) -> str:
    """生成稳定排序键，固定公式（5）（10）（16）所用保持输入的划分。"""
    payload = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
