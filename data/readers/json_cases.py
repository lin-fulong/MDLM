"""Read trusted case data and prepare answer-free evaluation inputs."""

from __future__ import annotations

import json
from pathlib import Path


def load_cases(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            cases = [json.loads(line) for line in stream if line.strip()]
    else:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        cases = payload if isinstance(payload, list) else [payload]
    if cases and "edit_requests" in cases[0] and "edit_questions" in cases[0]:
        from data.readers.reviewed_cases import adapt_reviewed_cases

        return adapt_reviewed_cases(cases)
    return cases


def case_purpose(case: dict) -> str:
    """Describe the case's intended use without validating it."""
    if case["quality_control"]["status"] == "example_only":
        return "example_only"
    if case.get("split") == "test":
        return "independent_test"
    if case.get("split") in ("gold", "pilot", "dev"):
        return "development"
    return "edit"


def evaluation_inputs(case: dict) -> list[dict]:
    """Expose held-out questions without passing answer references to the model."""
    inputs = []
    for probes in case["prompts"].values():
        for probe in probes:
            if probe["used_for_edit"]:
                continue
            if "role" in probe:
                role = probe["role"]
            elif case.get("split") == "test":
                role = "independent_test"
            elif case.get("split") == "train":
                role = "edit_diagnostic"
            else:
                role = "development_validation"
            inputs.append({
                "case_id": case["case_id"],
                "probe_id": probe["probe_id"],
                "prompt": probe["prompt"],
                "role": role,
                "example_only": case["quality_control"]["status"] == "example_only",
            })
    return inputs
