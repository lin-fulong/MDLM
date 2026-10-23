"""Run one-case P0/P1 real-model checks and save a technical acceptance report.

The bundled case is example_only. A passing report is integration evidence,
not a formal dataset or multi-fact algorithm acceptance result.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from MDLM.contracts import FactSelection, P1RunConfig, local_model_identity
from MDLM.states import load_supervision, load_trajectories
from data.readers.json_cases import case_purpose, load_cases
from model.adapters.llada_8b_instruct import DEFAULT_MODEL_PATH
from util.collect_denoise_states import collect_run


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CASE = PROJECT_ROOT / "tests/fixtures/p0_case_example.json"


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def environment_record() -> dict:
    import torch
    import transformers

    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available()}


def check_p0(rows: list[dict], *, case_id: str, probe_id: str, layer: int) -> dict:
    """Assert the real full-mask forward and ff_out input replay evidence."""
    if len(rows) != 1:
        raise ValueError(f"P0 expected one edit probe, got {len(rows)}")
    row = rows[0]
    name = f"model.transformer.blocks.{layer}.ff_out"
    if (row["case_id"], row["probe_id"]) != (case_id, probe_id):
        raise ValueError("P0 case/probe differs from the frozen selection")
    if not row["real_forward"] or not row["replay_within_tolerance"]:
        raise ValueError("P0 real forward or replay did not pass")
    if not row["full_mask_input"] or row["mask_count"] != row["target_token_count"] or row["mask_count"] < 1:
        raise ValueError("P0 answer window is not fully masked")
    modules = row["target_modules"]
    if set(modules) != {name} or set(row["ffn_input_shapes"]) != {name}:
        raise ValueError("P0 did not capture the selected ff_out input")
    if row["ffn_input_shapes"][name] != [modules[name]["in_features"]]:
        raise ValueError("P0 captured input dimension differs from ff_out weight")
    if len(row["mask_logits_shape"]) != 2 or row["mask_logits_shape"][0] != row["mask_count"]:
        raise ValueError("P0 mask logits do not match the answer window")
    return {"case_id": case_id, "probe_id": probe_id, "purpose": row["purpose"],
            "mask_count": row["mask_count"], "target_module": name,
            "ffn_input_width": modules[name]["in_features"], "real_forward_replayed": True}


def check_p1(run_dir: Path, *, case_id: str, probe_id: str, purpose: str) -> dict:
    """Independently check saved real trajectory and formula (1)-(3) weights."""
    with (run_dir / "coverage_report.json").open(encoding="utf-8") as stream:
        coverage = json.load(stream)
    with (run_dir / "collection_report.json").open(encoding="utf-8") as stream:
        collection = json.load(stream)
    trajectories = load_trajectories(run_dir / "trajectories.jsonl")
    fixed = load_supervision(run_dir / "fixed_states.jsonl")
    if coverage["status"] != "complete" or not coverage["formula_checks_passed"]:
        raise ValueError("P1 coverage or formula checks did not pass")
    if coverage["fact_count"] != 1 or len(trajectories) != 1 or len(fixed) != 1:
        raise ValueError("P1 result is not the selected one-fact run")
    if (coverage["replayed_fact_count"] != 1 or coverage["logits_replayed_fact_count"] != 1
            or len(collection["completed"]) != 1 or collection["failure"] is not None):
        raise ValueError("P1 saved trajectory and logits were not replayed")
    row, trace, fact = coverage["facts"][0], trajectories[0], fixed[0]
    completed = collection["completed"][0]
    if any((item["case_id"], item["probe_id"]) != (case_id, probe_id)
           for item in (row, fact, completed)) or (trace.case_id, trace.probe_id) != (case_id, probe_id):
        raise ValueError("P1 fact/probe differs from the frozen selection")
    if (completed["purpose"] != purpose or not completed["forward_logits_finite"]
            or not completed["replayed"] or not completed["logits_replayed"]):
        raise ValueError("P1 purpose or finite-logits evidence differs")
    if not row["full_mask_initial_compatible"] or row["compatible_count"] < 1:
        raise ValueError("P1 full-mask initial state is not compatible")
    if (not trace.trace.pre_forward_inputs or
            trace.trace.pre_forward_inputs[0].mask_positions != trace.answer_positions or
            len(trace.answer_positions) != len(trace.target_token_ids) or
            len(trace.trace.mask_logit_sha256) != len(trace.trace.pre_forward_inputs) or
            trace.trace.final_input.mask_positions):
        raise ValueError("P1 saved denoising trajectory is incomplete")
    states = fact["states"]
    if len(states) != row["compatible_count"]:
        raise ValueError("P1 compatible state count differs from fixed supervision")
    answer = dict(zip(trace.answer_positions, trace.target_token_ids))
    weights = []
    for state in states:
        positions = state["remaining_mask_positions"]
        if not positions or [term["position"] for term in state["supervision"]] != positions:
            raise ValueError("P1 supervision positions differ from remaining masks")
        expected = 1.0 / (len(states) * len(positions))  # 公式（3），此验收中 N=1。
        for term in state["supervision"]:
            if term["target_token_id"] != answer[term["position"]] or not math.isclose(
                term["weight"], expected, rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError("P1 target token or three-level weight differs")
            weights.append(term["weight"])
    if not math.isclose(math.fsum(weights), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("P1 one-fact and global weights must sum to one")
    if (not math.isclose(row["weight_sum"], 1.0, rel_tol=0, abs_tol=1e-12)
            or not math.isclose(coverage["global_weight_sum"], 1.0, rel_tol=0, abs_tol=1e-12)):
        raise ValueError("P1 coverage weight sum differs")
    return {"case_id": case_id, "probe_id": probe_id, "purpose": purpose,
            "pre_forward_count": len(trace.trace.pre_forward_inputs),
            "compatible_state_count": len(states), "conflict_state_count": row["conflict_count"],
            "weight_sum": math.fsum(weights), "trajectory_and_logits_replayed": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-file", type=Path, default=EXAMPLE_CASE)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--layer", type=int, required=True, help="explicitly selected LLaDA ff_out layer")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--run-id", help="unique run name; defaults to a UTC timestamp and random suffix")
    args = parser.parse_args()
    if args.layer < 0:
        parser.error("--layer must be nonnegative")
    case_file, model_path = args.case_file.resolve(), args.model_path.resolve()
    cases = load_cases(case_file)
    if len(cases) != 1 or len(cases[0]["prompts"]["edit"]) != 1:
        raise ValueError("this acceptance entry point selects exactly one case and one edit probe")
    case = cases[0]
    case_id, probe_id = case["case_id"], case["prompts"]["edit"][0]["probe_id"]
    purpose = case_purpose(case)
    run_id = args.run_id or "single_case_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    if run_id in ("", ".", "..") or "/" in run_id or "\\" in run_id:
        parser.error("--run-id must be one path component")
    output_root = args.output_root.resolve()
    run_root = output_root / run_id / "round_0"
    run_root.mkdir(parents=True, exist_ok=False)
    print(f"Acceptance artifacts: {run_root}", flush=True)
    report_path = run_root / "acceptance_report.json"
    report = {"status": "running", "scope": "single_case_technical_acceptance",
              "formal_dataset_acceptance": False, "case_id": case_id, "probe_id": probe_id,
              "purpose": purpose, "run_id": run_id, "target_layer": args.layer,
              "phase": "model_identity"}
    save_json(report_path, report)
    try:
        print("[1/3] Hashing local model and tokenizer files...", flush=True)
        identity = local_model_identity(model_path)
        save_json(run_root / "model_identity.json", identity)
        save_json(run_root / "environment.json", environment_record())
        config = P1RunConfig(
            run_id=run_id, round_id=0, baseline_id=identity["model_revision"],
            model_name="LLaDA-8B-Instruct", model_path=str(model_path),
            model_revision=identity["model_revision"],
            tokenizer_revision=identity["tokenizer_revision"], case_file=str(case_file),
            facts=(FactSelection(case_id, probe_id),), target_layers=(args.layer,),
            window_policy_id="short_answer_exact_tokens_v1",
            sampler_policy_id="confidence_no_remask_v1", seed=2026, steps=None,
            temperature=0.0, replay_count=1, output_root=str(output_root),
        )
        config.check()
        save_json(run_root / "run_config.json", config.as_record())
        report["phase"] = "p0_real_forward"
        save_json(report_path, report)
        print("[2/3] Running P0 full-mask forward, ff_out capture and replay...", flush=True)
        p0_path = run_root / "p0_report.json"
        subprocess.run(
            [sys.executable, "-m", "util.smoke_p0_llada", str(case_file),
             "--model-path", str(model_path), "--layer", str(args.layer),
             "--forward", "--output", str(p0_path)],
            cwd=PROJECT_ROOT, check=True,
        )
        with p0_path.open(encoding="utf-8") as stream:
            p0_rows = json.load(stream)
        report["p0"] = check_p0(p0_rows, case_id=case_id, probe_id=probe_id, layer=args.layer)
        report["phase"] = "p1_real_trajectory"
        save_json(report_path, report)
        print("[3/3] Running P1 autonomous denoising, replay and supervision checks...", flush=True)
        collect_run(config)
        report["p1"] = check_p1(config.p1_dir(), case_id=case_id, probe_id=probe_id, purpose=purpose)
        report["status"] = "passed"
        report["phase"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        save_json(report_path, report)
        raise
    save_json(report_path, report)
    print(json.dumps({"status": report["status"], "report": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
