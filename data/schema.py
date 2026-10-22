"""Validation for the agreed 0.3-proposal case JSON; no dataset-specific fields."""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "0.3-proposal"
PROMPT_GROUPS = ("edit", "paraphrase", "cloze", "free_form_qa", "portability")
# A split partitions *cases*. used_for_edit partitions probes inside each case.
SPLIT_PURPOSE = {
    "train": "edit",
    "gold": "development",
    "pilot": "development",
    "dev": "development",
    "test": "independent_test",
}


class CaseValidationError(ValueError):
    """A case cannot be used without silently guessing a required field."""


def normalize_text(text: str) -> str:
    """Normalize visible text for conservative exact-match leakage checks."""
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def assert_answer_absent(prompt: str, answers: list[str] | tuple[str, ...], path: str) -> None:
    """Reject prompts that literally contain an answer or one of its aliases."""
    visible = normalize_text(prompt)
    for answer in answers:
        target = normalize_text(answer)
        if target and target in visible:
            raise CaseValidationError(f"{path}: prompt contains target answer/alias {answer!r}")


def _mapping(value: Any, path: str) -> dict:
    if not isinstance(value, dict):
        raise CaseValidationError(f"{path}: expected object")
    return value


def _field(obj: dict, name: str, path: str, kind: type, *, nonempty: bool = False) -> Any:
    if name not in obj:
        raise CaseValidationError(f"{path}.{name}: required field is missing")
    value = obj[name]
    if kind is bool:
        valid = type(value) is bool
    else:
        valid = isinstance(value, kind) and not isinstance(value, bool)
    if not valid or (nonempty and (not value or (isinstance(value, str) and not value.strip()))):
        raise CaseValidationError(f"{path}.{name}: expected {'nonempty ' if nonempty else ''}{kind.__name__}")
    return value


def _strings(obj: dict, name: str, path: str) -> list[str]:
    values = _field(obj, name, path, list)
    if any(not isinstance(item, str) or not item.strip() for item in values):
        raise CaseValidationError(f"{path}.{name}: expected a list of nonempty strings")
    return values


def validate_case(case: Any) -> dict:
    """Validate the supplied structure in place and return it unchanged."""
    case = _mapping(case, "case")
    _field(case, "case_id", "case", str, nonempty=True)
    version = _field(case, "schema_version", "case", str, nonempty=True)
    if version != SCHEMA_VERSION:
        raise CaseValidationError(f"case.schema_version: expected {SCHEMA_VERSION!r}")
    split = _field(case, "split", "case", str, nonempty=True)
    if split not in SPLIT_PURPOSE:
        raise CaseValidationError(f"case.split: unsupported value {split!r}")
    direction = _field(case, "edit_direction", "case", str, nonempty=True)
    _field(case, "language", "case", str, nonempty=True)

    fact = _mapping(_field(case, "fact_core", "case", dict), "case.fact_core")
    for name in ("subject", "relation", "relation_text", "old_object", "new_object", "edit_type", "domain"):
        _field(fact, name, "case.fact_core", str, nonempty=True)
    _strings(fact, "new_object_aliases", "case.fact_core")
    old_answer = normalize_text(fact["old_object"])
    new_answers = [fact["new_object"], *fact["new_object_aliases"]]
    if old_answer in {normalize_text(answer) for answer in new_answers}:
        raise CaseValidationError("case.fact_core: old_object conflicts with new_object/aliases")

    truth = _mapping(_field(case, "truth_scope", "case", dict), "case.truth_scope")
    for name in ("public_fact", "dataset_internal", "constructed_fact", "counterfactual", "privacy_sensitive"):
        _field(truth, name, "case.truth_scope", bool)
    if direction == "counterfactual" and not truth["counterfactual"]:
        raise CaseValidationError("case.truth_scope.counterfactual conflicts with edit_direction")
    source = _mapping(_field(case, "source", "case", dict), "case.source")
    for name in ("source_dataset", "source_id", "source_version", "annotation_method"):
        _field(source, name, "case.source", str, nonempty=True)

    prompts = _mapping(_field(case, "prompts", "case", dict), "case.prompts")
    seen_ids: set[str] = set()
    edit_count = 0
    for group in PROMPT_GROUPS:
        probes = _field(prompts, group, "case.prompts", list)
        for index, raw in enumerate(probes):
            path = f"case.prompts.{group}[{index}]"
            probe = _mapping(raw, path)
            probe_id = _field(probe, "probe_id", path, str, nonempty=True)
            _field(probe, "prompt", path, str, nonempty=True)
            _field(probe, "answer", path, str, nonempty=True)
            used = _field(probe, "used_for_edit", path, bool)
            if probe_id in seen_ids:
                raise CaseValidationError(f"{path}.probe_id: duplicate {probe_id!r}")
            seen_ids.add(probe_id)
            if group == "edit" and not used:
                raise CaseValidationError(f"{path}.used_for_edit: edit probe must be marked true")
            # A non-edit group may contain a probe used during training. It is
            # excluded from held-out evaluation, although P0 only prepares the
            # canonical prompts.edit input for the main algorithm.
            if group == "edit":
                edit_count += 1
                if normalize_text(probe["answer"]) not in {normalize_text(answer) for answer in new_answers}:
                    raise CaseValidationError(f"{path}.answer: edit target differs from new_object/aliases")
                assert_answer_absent(probe["prompt"], new_answers, f"{path}.prompt")
            if "answer_aliases" in probe:
                _strings(probe, "answer_aliases", path)
            for name in ("reference_type", "target_proposition"):
                if name in probe:
                    _field(probe, name, path, str, nonempty=True)
    if not edit_count:
        raise CaseValidationError("case.prompts.edit: at least one edit probe is required")

    profile = _mapping(_field(case, "generation_profile", "case", dict), "case.generation_profile")
    for name in ("mode", "window_policy_id"):
        _field(profile, name, "case.generation_profile", str, nonempty=True)
    _field(case, "trajectory_probes", "case", list)
    kg = _mapping(_field(case, "kg_consequence", "case", dict), "case.kg_consequence")
    for name in ("expected_ripples", "preserve_edges", "forbidden_leakage"):
        _field(kg, name, "case.kg_consequence", list)
    locality = _mapping(_field(case, "locality", "case", dict), "case.locality")
    for name in ("same_subject", "same_relation", "semantic_neighbor", "unrelated"):
        _field(locality, name, "case.locality", list)
    quality = _mapping(_field(case, "quality_control", "case", dict), "case.quality_control")
    _field(quality, "status", "case.quality_control", str, nonempty=True)
    _field(case, "metadata", "case", dict)
    return case


def check_dataset(path: str | Path) -> int:
    """Validate a JSON/JSONL file when explicitly requested; return its case count."""
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            cases = [json.loads(line) for line in stream if line.strip()]
    elif path.suffix == ".json":
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        cases = payload if isinstance(payload, list) else [payload]
    else:
        raise CaseValidationError(f"{path}: expected .json or .jsonl")
    if not cases:
        raise CaseValidationError(f"{path}: no cases")
    seen: set[str] = set()
    for index, case in enumerate(cases):
        try:
            validate_case(case)
        except CaseValidationError as exc:
            raise CaseValidationError(f"{path}[{index}]: {exc}") from exc
        case_id = case["case_id"]
        if case_id in seen:
            raise CaseValidationError(f"{path}[{index}]: duplicate case_id {case_id!r}")
        seen.add(case_id)
    return len(cases)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manually check the P0 case structure in a JSON/JSONL dataset.")
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    try:
        count = check_dataset(args.dataset)
    except (CaseValidationError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        parser.exit(1, f"数据集检查失败：{exc}\n")
    print(f"数据集初步检查通过：{count} 条案例（不代表正式数据质量验收通过）")


if __name__ == "__main__":
    main()
