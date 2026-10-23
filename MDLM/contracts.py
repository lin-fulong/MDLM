"""Versioned run choices; dataset quality is checked separately."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


P1_ARTIFACT_VERSION = "p1.1"
REFERENCE_KEYS_VERSION = "p2.keys.1"
KEEP_INPUTS_VERSION = "p2.keep_inputs.1"
KEEP_STATISTICS_VERSION = "p2.keep_statistics.1"
MATRIX_BUNDLE_VERSION = "p2.matrix_bundle.1"
WRITEBACK_FACTOR_VERSION = "p3.writeback_factor.2"
RESIDUAL_CANDIDATE_VERSION = "p4.residual_candidate.2"
KEEP_TRAINING_VERSION = "p4.keep_training.1"


def config_digest(values: dict) -> str:
    payload = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_model_identity(model_path: str | Path) -> dict:
    """Identify a local safetensors model, its Python implementation and tokenizer."""
    model_path = Path(model_path)
    weights = sorted(model_path.glob("*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"no safetensors weights found in {model_path}")
    model_files = [model_path / "config.json", *sorted(model_path.glob("*.py")), *weights]
    index = model_path / "model.safetensors.index.json"
    if index.exists():
        model_files.append(index)
    tokenizer_files = [model_path / "tokenizer.json", model_path / "tokenizer_config.json",
                       model_path / "special_tokens_map.json"]
    model_hashes = {path.name: file_sha256(path) for path in model_files}
    tokenizer_hashes = {path.name: file_sha256(path) for path in tokenizer_files}
    return {
        "model_files_sha256": model_hashes,
        "tokenizer_files_sha256": tokenizer_hashes,
        "model_revision": "local-files-sha256:" + config_digest(model_hashes),
        "tokenizer_revision": "local-files-sha256:" + config_digest(tokenizer_hashes),
    }


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
        return cls.from_record(values)

    @classmethod
    def from_record(cls, values: dict) -> "P1RunConfig":
        values = dict(values)
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


@dataclass(frozen=True)
class ReferenceKeyConfig:
    """Formula (4) storage and repeat-capture tolerances, fixed before extraction."""

    storage_dtype: str = "float32"
    replay_atol: float = 1e-2
    replay_rtol: float = 1e-3

    @classmethod
    def from_file(cls, path: str | Path) -> "ReferenceKeyConfig":
        with Path(path).open(encoding="utf-8") as stream:
            config = cls(**json.load(stream))
        config.check()
        return config

    def check(self) -> None:
        if self.storage_dtype not in ("float32", "float64"):
            raise ValueError("reference key storage_dtype must be float32 or float64")
        if any(not math.isfinite(value) or value < 0 for value in (self.replay_atol, self.replay_rtol)):
            raise ValueError("reference key replay tolerances must be finite and nonnegative")

    def as_record(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class KeepStatisticsConfig:
    """Explicit formula (5) recipe identity and numerical choices; no implicit scale."""

    recipe_id: str
    accumulation_dtype: str = "float64"
    storage_dtype: str = "float32"
    replay_atol: float = 1e-2
    replay_rtol: float = 1e-3
    check_atol: float = 1e-6
    check_rtol: float = 1e-5
    qa_seed: int = 2026
    projection_count: int = 8
    direct_block_size: int = 16
    chunk_size: int = 512

    @classmethod
    def from_file(cls, path: str | Path) -> "KeepStatisticsConfig":
        with Path(path).open(encoding="utf-8") as stream:
            config = cls(**json.load(stream))
        config.check()
        return config

    def check(self) -> None:
        if not self.recipe_id:
            raise ValueError("keep recipe_id must be explicitly identified")
        if any(dtype not in ("float32", "float64") for dtype in (self.accumulation_dtype, self.storage_dtype)):
            raise ValueError("keep accumulation/storage dtype must be float32 or float64")
        if any(not math.isfinite(value) or value < 0 for value in (
            self.replay_atol, self.replay_rtol, self.check_atol, self.check_rtol
        )):
            raise ValueError("keep replay/check tolerances must be finite and nonnegative")
        if type(self.qa_seed) is not int or self.qa_seed < 0:
            raise ValueError("keep qa_seed must be a nonnegative integer")
        if any(type(value) is not int or value < 1 for value in (
            self.projection_count, self.direct_block_size, self.chunk_size
        )):
            raise ValueError("keep projection, direct block and chunk sizes must be positive integers")

    def as_record(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class WritebackConfig:
    """Freeze formula (9) ridge coefficients and numerical choices before solving."""

    ridge_by_layer: dict[str, float]
    compute_dtype: str = "float64"
    storage_dtype: str = "float32"
    device: str = "cpu"
    solver: str = "cholesky"
    failure_policy: str = "stop"
    symmetry_atol: float = 1e-7
    symmetry_rtol: float = 1e-6
    residual_tolerance: float = 1e-8
    stored_residual_tolerance: float = 1e-5
    residual_epsilon: float = 1e-12
    chunk_size: int = 256
    allow_example_only: bool = False

    @classmethod
    def from_file(cls, path: str | Path) -> "WritebackConfig":
        with Path(path).open(encoding="utf-8") as stream:
            config = cls(**json.load(stream))
        config.check()
        return config

    def check(self) -> None:
        if not isinstance(self.ridge_by_layer, dict) or not self.ridge_by_layer or any(
            not isinstance(name, str) or not name or isinstance(value, bool)
            or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
            for name, value in self.ridge_by_layer.items()
        ):
            raise ValueError("one explicit finite positive ridge is required per target layer")
        if any(dtype not in ("float32", "float64") for dtype in (self.compute_dtype, self.storage_dtype)):
            raise ValueError("writeback compute/storage dtype must be float32 or float64")
        if self.solver != "cholesky" or self.failure_policy != "stop":
            raise ValueError("writeback supports only frozen cholesky with stop-on-failure")
        if not isinstance(self.device, str) or (self.device != "cpu" and self.device != "cuda" and not (
            self.device.startswith("cuda:") and self.device[5:].isdigit()
        )):
            raise ValueError("writeback device must be cpu, cuda or cuda:<index>")
        for name in ("symmetry_atol", "symmetry_rtol", "residual_tolerance",
                     "stored_residual_tolerance", "residual_epsilon"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"writeback {name} must be finite and nonnegative")
        if self.residual_epsilon == 0:
            raise ValueError("writeback residual_epsilon must be positive")
        if type(self.chunk_size) is not int or self.chunk_size < 1:
            raise ValueError("writeback chunk_size must be a positive integer")
        if type(self.allow_example_only) is not bool:
            raise ValueError("allow_example_only must be boolean")

    def as_record(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ResidualOptimizationConfig:
    """Formula (10): explicit finite budget; rho penalizes R, independently of lambda."""

    rho_by_layer: dict[str, float]
    beta: float = 0.0
    learning_rate: float = 0.01
    max_steps: int = 20
    checkpoints: tuple[int, ...] = (1, 5, 10, 20)
    microbatch_size: int = 1
    residual_dtype: str = "float32"
    seed: int = 2026
    optimizer: str = "adam"
    equivalence_atol: float = 0.01
    equivalence_rtol: float = 0.001
    keep_train_file: str | None = None
    reference_model_path: str | None = None
    reference_model_revision: str | None = None
    reference_tokenizer_revision: str | None = None
    allow_example_only: bool = False

    @classmethod
    def from_file(cls, path: str | Path) -> "ResidualOptimizationConfig":
        with Path(path).open(encoding="utf-8") as stream:
            values = json.load(stream)
        values["checkpoints"] = tuple(values.get("checkpoints", cls.checkpoints))
        config = cls(**values)
        config.check()
        return config

    def check(self) -> None:
        if not isinstance(self.rho_by_layer, dict) or not self.rho_by_layer or any(
            not isinstance(name, str) or not name or isinstance(value, bool)
            or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
            for name, value in self.rho_by_layer.items()
        ):
            raise ValueError("one finite positive rho is required per target layer")
        for name in ("beta", "learning_rate", "equivalence_atol", "equivalence_rtol"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"P4 {name} must be finite and nonnegative")
        if self.learning_rate == 0 or self.optimizer != "adam":
            raise ValueError("P4 uses explicitly configured Adam with a positive learning rate and no weight decay")
        if any(type(value) is not int or value < 1 for value in (self.max_steps, self.microbatch_size)):
            raise ValueError("P4 step and microbatch sizes must be positive integers")
        if not self.checkpoints or tuple(sorted(set(self.checkpoints))) != self.checkpoints or any(
            type(step) is not int or not 1 <= step <= self.max_steps for step in self.checkpoints
        ) or self.checkpoints[-1] != self.max_steps:
            raise ValueError("P4 checkpoints must be ordered unique steps including max_steps")
        if self.residual_dtype not in ("float32", "float64") or type(self.seed) is not int or self.seed < 0:
            raise ValueError("P4 requires float32/float64 residuals and a nonnegative integer seed")
        reference = (self.keep_train_file, self.reference_model_path,
                     self.reference_model_revision, self.reference_tokenizer_revision)
        if self.beta > 0 and not all(isinstance(value, str) and value for value in reference):
            raise ValueError("beta>0 requires explicit keep-training inputs and original theta_0 identity")
        if self.beta == 0 and any(value is not None for value in reference):
            raise ValueError("beta=0 must leave keep-training and theta_0 fields null")
        if type(self.allow_example_only) is not bool:
            raise ValueError("allow_example_only must be boolean")

    def as_record(self) -> dict:
        return asdict(self)
