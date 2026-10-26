"""Content identities for prepared inputs and resumable P0–P5 stage artifacts."""

from __future__ import annotations

import importlib.metadata
import json
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

from MDLM.contracts import config_digest, file_sha256


def files_match(files: dict[str, str]) -> bool:
    try:
        return isinstance(files, dict) and bool(files) and all(Path(path).is_file() and file_sha256(path) == digest
                                   for path, digest in files.items())
    except OSError:
        return False


@contextmanager
def cache_lock(path: Path):
    """Serialize writers on WSL/Linux; kernel releases the lock after a crash."""
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def snapshot(path: Path) -> dict[str, str]:
    paths = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    return {str(p.resolve()): file_sha256(p) for p in paths}


def code_identity(root: Path) -> str:
    paths = {root / "main.py", root / "model/base.py", root / "data/validation/alignment.py"}
    for folder in ("MDLM", "util", "data/readers", "model/adapters"):
        paths.update((root / folder).glob("*.py"))
    return config_digest({str(path.relative_to(root)): file_sha256(path) for path in sorted(paths)})


def runtime_identity() -> dict:
    result = {"python": sys.version}
    for package in ("torch", "transformers", "tokenizers", "safetensors", "accelerate"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "unavailable"
    return result


def round_key(choices: dict, facts: list[str], baseline: dict, original: dict, root: Path) -> str:
    settings = {key: value for key, value in choices.items()
                if key not in ("run_id", "output_root", "cache_dir", "source_dataset")}
    paths = [choices["dataset"], choices["keep_inputs"], choices["p5"]["dev_edit_file"], choices["p5"]["dev_keep_file"]]
    if choices["p4"].get("beta", 0) > 0:
        paths.append(choices["p4"]["keep_train_file"])
    return config_digest({"version": "mdlm.round_cache.1", "settings": settings, "facts": facts,
                          "inputs": {str(path): file_sha256(path) for path in paths},
                          "baseline": baseline, "original": original,
                          "code": code_identity(root), "runtime": runtime_identity()})


def archive_partial(path: Path, archive_root: Path, label: str) -> None:
    if path.exists():
        archive_root.mkdir(parents=True, exist_ok=True)
        path.rename(archive_root / (label.lower() + "_" + uuid.uuid4().hex[:8]))


def save_index(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex[:8] + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
