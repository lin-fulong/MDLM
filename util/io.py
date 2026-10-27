"""实验记录的 JSON 读写和基本检查。"""

from __future__ import annotations

import json
from pathlib import Path

from MDLM.contracts import file_sha256


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def checked_file(path: Path, digest: str) -> None:
    require(file_sha256(path) == digest, f"artifact digest differs: {path}")
