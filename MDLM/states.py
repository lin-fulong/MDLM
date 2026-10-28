"""P1 complete denoising states and supervision from formulas (1)–(3)."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from model.base import DenoiseTrace, EditTarget, MaskedEditInput


@dataclass(frozen=True)
class FactSelection:
    case_id: str
    probe_id: str


def fact_seed(seed: int, case_id: str, probe_id: str) -> int:
    """沿用逐事实随机流：改变处理顺序不改变采样种子。"""
    key = f"{seed}:{case_id}:{probe_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63)


def input_record(model_input: MaskedEditInput) -> dict:
    """Save every model-facing field and LLaDA's implicit zero-based positions."""
    return {
        "input_ids": list(model_input.input_ids),
        "attention_mask": list(model_input.attention_mask),
        "mask_positions": list(model_input.mask_positions),
        "subject_last_position": model_input.subject_last_position,
        "implicit_position_ids": list(range(len(model_input.input_ids))),
    }


def input_from_record(record: dict) -> MaskedEditInput:
    if record["implicit_position_ids"] != list(range(len(record["input_ids"]))):
        raise ValueError("saved implicit position IDs differ from the model input length")
    return MaskedEditInput(
        tuple(record["input_ids"]),
        tuple(record["attention_mask"]),
        tuple(record["mask_positions"]),
        record["subject_last_position"],
    )


@dataclass(frozen=True)
class FactTrajectory:
    case_id: str
    probe_id: str
    answer_positions: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    seed: int
    trace: DenoiseTrace

    @classmethod
    def from_sample(
        cls, target: EditTarget, seed: int, trace: DenoiseTrace
    ) -> "FactTrajectory":
        if not trace.pre_forward_inputs:
            raise ValueError(f"{target.case_id}: sampler returned no pre-forward input")
        return cls(
            target.case_id,
            target.probe_id,
            trace.pre_forward_inputs[0].mask_positions,
            target.token_ids,
            seed,
            trace,
        )

    def as_record(self) -> dict:
        return {
            "case_id": self.case_id,
            "probe_id": self.probe_id,
            "answer_positions": list(self.answer_positions),
            "target_token_ids": list(self.target_token_ids),
            "seed": self.seed,
            "pre_forward_inputs": [input_record(item) for item in self.trace.pre_forward_inputs],
            "mask_logit_sha256": list(self.trace.mask_logit_sha256),
            "revealed_tokens": [
                [[position, token] for position, token in step] for step in self.trace.revealed_tokens
            ],
            "final_input": input_record(self.trace.final_input),
        }

    @classmethod
    def from_record(cls, record: dict) -> "FactTrajectory":
        trace = DenoiseTrace(
            tuple(input_from_record(item) for item in record["pre_forward_inputs"]),
            tuple(record["mask_logit_sha256"]),
            tuple(
                tuple((position, token) for position, token in step)
                for step in record["revealed_tokens"]
            ),
            input_from_record(record["final_input"]),
        )
        return cls(
            record["case_id"],
            record["probe_id"],
            tuple(record["answer_positions"]),
            tuple(record["target_token_ids"]),
            record["seed"],
            trace,
        )


def classify_fact(trajectory: FactTrajectory, mask_token_id: int) -> tuple[list[dict], dict]:
    """Formula (1): retain masked states with no conflicting revealed answer token."""
    if not trajectory.answer_positions or trajectory.answer_positions != tuple(sorted(set(trajectory.answer_positions))):
        raise ValueError(f"{trajectory.case_id}: answer positions must be nonempty, unique and ordered")
    if len(trajectory.answer_positions) != len(trajectory.target_token_ids):
        raise ValueError(f"{trajectory.case_id}: answer positions and target length differ")
    if not trajectory.trace.pre_forward_inputs:
        raise ValueError(f"{trajectory.case_id}: full-mask initial state is missing")
    initial = trajectory.trace.pre_forward_inputs[0]
    if any(position < 0 or position >= len(initial.input_ids) for position in trajectory.answer_positions):
        raise ValueError(f"{trajectory.case_id}: answer position is out of range")
    if not 0 <= initial.subject_last_position < min(trajectory.answer_positions):
        raise ValueError(f"{trajectory.case_id}: subject must precede the answer window")
    if trajectory.trace.pre_forward_inputs[0].mask_positions != trajectory.answer_positions:
        raise ValueError(f"{trajectory.case_id}: first state is not the full-mask initial state")
    if len(trajectory.trace.revealed_tokens) != len(trajectory.trace.pre_forward_inputs):
        raise ValueError(f"{trajectory.case_id}: reveal count differs from forward count")
    if len(trajectory.trace.mask_logit_sha256) != len(trajectory.trace.pre_forward_inputs):
        raise ValueError(f"{trajectory.case_id}: logit digest count differs from forward count")

    compatible: list[dict] = []
    discarded: list[dict] = []
    seen: set[MaskedEditInput] = set()
    previous_masks = set(trajectory.answer_positions)
    for step, state in enumerate(trajectory.trace.pre_forward_inputs):
        if len(state.input_ids) != len(initial.input_ids) or len(state.input_ids) != len(state.attention_mask) or any(value != 1 for value in state.attention_mask):
            raise ValueError(f"{trajectory.case_id}: saved attention input differs from the forward contract")
        if not set(state.mask_positions) <= previous_masks:
            raise ValueError(f"{trajectory.case_id}: a revealed token was masked again")
        current_masks = set(state.mask_positions)
        if state.mask_positions != tuple(
            position for position, token in enumerate(state.input_ids) if token == mask_token_id
        ):
            raise ValueError(f"{trajectory.case_id}: saved mask positions differ from input IDs")
        previous_masks = current_masks
        # frozen 输入已包含全部前向字段，可直接按完整输入去重。
        if state in seen:
            discarded.append({"step": step, "reason": "duplicate_full_input"})
            continue
        seen.add(state)
        if not current_masks:
            discarded.append({"step": step, "reason": "terminal"})
        elif any(
            state.input_ids[position] != target
            for position, target in zip(trajectory.answer_positions, trajectory.target_token_ids)
            if position not in current_masks
        ):
            discarded.append({"step": step, "reason": "conflict"})
        else:
            compatible.append({"step": step, "input": input_record(state), "remaining_mask_positions": list(state.mask_positions)})

    if trajectory.trace.final_input.mask_positions or mask_token_id in trajectory.trace.final_input.input_ids:
        raise ValueError(f"{trajectory.case_id}: denoising did not reach a terminal state")
    for step, current in enumerate(trajectory.trace.pre_forward_inputs):
        following = (
            trajectory.trace.pre_forward_inputs[step + 1]
            if step + 1 < len(trajectory.trace.pre_forward_inputs)
            else trajectory.trace.final_input
        )
        expected_ids = list(current.input_ids)
        revealed_positions = set()
        for position, token in trajectory.trace.revealed_tokens[step]:
            if position not in current.mask_positions or position in revealed_positions or token == mask_token_id:
                raise ValueError(f"{trajectory.case_id}: invalid revealed token at step {step}")
            expected_ids[position] = token
            revealed_positions.add(position)
        expected = MaskedEditInput(
            tuple(expected_ids),
            current.attention_mask,
            tuple(position for position in current.mask_positions if position not in revealed_positions),
            current.subject_last_position,
        )
        if following != expected:
            raise ValueError(f"{trajectory.case_id}: saved input transition differs from revealed tokens at step {step}")
    report = {
        "case_id": trajectory.case_id,
        "probe_id": trajectory.probe_id,
        "pre_forward_count": len(trajectory.trace.pre_forward_inputs),
        "unique_input_count": len(seen),
        "compatible_count": len(compatible),
        "full_mask_initial_compatible": bool(compatible and compatible[0]["step"] == 0),
        "conflict_count": sum(item["reason"] == "conflict" for item in discarded),
        "duplicate_count": sum(item["reason"] == "duplicate_full_input" for item in discarded),
        "terminal_pre_forward_count": sum(item["reason"] == "terminal" for item in discarded),
        "terminal_final_count": 1,
        "discarded": discarded,
    }
    return compatible, report


def build_supervision(trajectories: list[FactTrajectory], mask_token_id: int) -> tuple[list[dict], dict]:
    """Formulas (2)–(3): normalize by fact, compatible state, then remaining mask."""
    fact_count = len(trajectories)
    if not fact_count:
        raise ValueError("no selected facts were collected")
    fixed: list[dict] = []
    reports: list[dict] = []
    for trajectory in trajectories:
        states, report = classify_fact(trajectory, mask_token_id)
        answer_by_position = dict(zip(trajectory.answer_positions, trajectory.target_token_ids))
        total_states = len(states)
        for state in states:
            positions = state["remaining_mask_positions"]
            # 公式（3）：每个监督位置的权重是 1/(N·|T_i|·|M_i,t|)。
            weight = 1.0 / (fact_count * total_states * len(positions))
            state["supervision"] = [
                {"position": position, "target_token_id": answer_by_position[position], "weight": weight}
                for position in positions
            ]
        report["weight_sum"] = math.fsum(
            term["weight"] for state in states for term in state["supervision"]
        )
        fixed.append({
            "case_id": trajectory.case_id,
            "probe_id": trajectory.probe_id,
            "answer_positions": list(trajectory.answer_positions),
            "target_token_ids": list(trajectory.target_token_ids),
            "states": states,
        })
        reports.append(report)
    total_weight = math.fsum(item["weight_sum"] for item in reports)
    formula_checks_passed = all(
        item["full_mask_initial_compatible"]
        and math.isclose(item["weight_sum"], 1 / fact_count, abs_tol=1e-12)
        for item in reports
    ) and math.isclose(total_weight, 1.0, abs_tol=1e-12)
    return fixed, {
        "fact_count": fact_count,
        "compatible_state_count": sum(item["compatible_count"] for item in reports),
        "conflict_state_count": sum(item["conflict_count"] for item in reports),
        "supervision_position_count": sum(
            len(state["supervision"]) for fact in fixed for state in fact["states"]
        ),
        "global_weight_sum": total_weight,
        "formula_checks_passed": formula_checks_passed,
        "facts": reports,
    }


@dataclass(frozen=True)
class EditTrainingExample:
    case_id: str
    model_input: MaskedEditInput
    positions: tuple[int, ...]
    targets: tuple[int, ...]
    weights: tuple[float, ...]


def edit_training_examples(facts: list[dict]) -> list[EditTrainingExample]:
    """Reuse P1 formula (3) weights unchanged; never insert targets into inputs."""
    return [EditTrainingExample(
        fact["case_id"], input_from_record(state["input"]),
        tuple(term["position"] for term in state["supervision"]),
        tuple(term["target_token_id"] for term in state["supervision"]),
        tuple(term["weight"] for term in state["supervision"]),
    ) for fact in facts for state in fact["states"]]
