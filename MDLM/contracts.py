"""Versioned inputs for P1 state collection; dataset quality is checked separately."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


P1_ARTIFACT_VERSION = "p1.1"


def config_digest(values: dict) -> str:
    payload = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FactSelection:
    """One edit probe represents one fact in the formula (2) denominator N."""

    case_id: str
    probe_id: str


@dataclass(frozen=True)
class P1RunConfig:
    run_id: str
    round_id: int
    baseline_id: str
    model_name: str
    model_path: str
    model_revision: str
    tokenizer_revision: str
    case_file: str
    facts: tuple[FactSelection, ...]
    target_layers: tuple[int, ...]
    window_policy_id: str
    sampler_policy_id: str
    seed: int
    steps: int | None
    temperature: float
    replay_count: int
    output_root: str = "runs"
    torch_dtype: str = "auto"
    device_map: str = "auto"

    @classmethod
    def from_file(cls, path: str | Path) -> "P1RunConfig":
        with Path(path).open(encoding="utf-8") as stream:
            values = json.load(stream)
        values["facts"] = tuple(FactSelection(**item) for item in values["facts"])
        values["target_layers"] = tuple(values["target_layers"])
        config = cls(**values)
        config.check()
        return config

    def check(self) -> None:
        """Check run choices, not the trusted dataset's contents."""
        if not self.run_id or self.run_id in (".", "..") or "/" in self.run_id or "\\" in self.run_id:
            raise ValueError("run_id must be one path component")
        if type(self.round_id) is not int or self.round_id < 0:
            raise ValueError("round_id must be a nonnegative integer")
        if not self.baseline_id or not self.model_revision or not self.tokenizer_revision:
            raise ValueError("baseline and model/tokenizer revisions must be identified")
        if not self.facts or len({item.case_id for item in self.facts}) != len(self.facts):
            raise ValueError("select one edit probe for each fact")
        if self.window_policy_id != "short_answer_exact_tokens_v1":
            raise ValueError("unsupported P1 answer window policy")
        if self.sampler_policy_id != "confidence_no_remask_v1":
            raise ValueError("unsupported P1 denoising policy")
        if not self.target_layers or len(set(self.target_layers)) != len(self.target_layers) or any(
            type(layer) is not int or layer < 0 for layer in self.target_layers
        ):
            raise ValueError("target_layers must be unique nonnegative indices")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.steps is not None and (type(self.steps) is not int or self.steps < 1):
            raise ValueError("steps must be positive or null for one step per answer token")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if type(self.replay_count) is not int or not 0 <= self.replay_count <= len(self.facts):
            raise ValueError("replay_count must be between zero and the selected fact count")

    def as_record(self) -> dict:
        return asdict(self)

    def digest(self) -> str:
        return config_digest(self.as_record())

    def p1_dir(self) -> Path:
        return Path(self.output_root) / self.run_id / f"round_{self.round_id}" / "p1"

    def fact_seed(self, fact: FactSelection) -> int:
        # 每个事实使用独立随机流，改变处理顺序不会改变其采样种子。
        key = f"{self.seed}:{fact.case_id}:{fact.probe_id}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63)
