"""Run real P0 -> P1 -> K -> C checks, or reuse P1/K and run only C.

Each fresh run uses LLaDA-8B and separate example_only edit/keep inputs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from MDLM.contracts import KEEP_INPUTS_VERSION, KeepStatisticsConfig, P1RunConfig, file_sha256
from model.adapters.llada_8b_instruct import DEFAULT_MODEL_PATH, LLaDA8BInstructAdapter
from util.accept_p0_p1_single_case import PROJECT_ROOT
from util.extract_keep_statistics import extract_run
from util.io import write_json
from util.validate_matrix_artifacts import validate_run


def freeze_example_inputs(p1_dir: Path, fixture_file: Path, output_path: Path) -> dict:
    """Tokenize the demo recipe once; no edit answers or evaluation probes are read."""
    if output_path.exists():
        raise FileExistsError(f"frozen keep inputs already exist: {output_path}")
    with (p1_dir / "manifest.json").open(encoding="utf-8") as stream:
        config = P1RunConfig.from_record(json.load(stream)["config"])
    raw_digest = file_sha256(fixture_file)
    with fixture_file.open(encoding="utf-8") as stream:
        fixture = json.load(stream)
    if (fixture["purpose"] != "keep_statistics" or fixture["example_only"] is not True
            or fixture["mask_policy_id"] != "append_explicit_mask_window_v1"
            or fixture["position_policy_id"] != "last_prompt_token_and_first_mask_v1"):
        raise ValueError("this smoke input builder only supports the explicit example_only keep recipe")
    adapter = LLaDA8BInstructAdapter.from_local_tokenizer(
        model_path=config.model_path, target_layers=config.target_layers)
    rows = []
    for item in fixture["inputs"]:
        prompt = tuple(adapter.tokenizer.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}], tokenize=True, add_generation_prompt=True))
        if not prompt or item["mask_count"] < 1:
            raise ValueError("example recipe requires a prompt and a nonempty explicit mask window")
        ids = prompt + (adapter.mask_token_id,) * item["mask_count"]
        positions = [len(prompt) - 1, len(prompt)]
        rows.append({
            "input_id": item["input_id"], "source": item["source"],
            "input": {"input_ids": list(ids), "attention_mask": [1] * len(ids),
                      "mask_positions": list(range(len(prompt), len(ids))),
                      "implicit_position_ids": list(range(len(ids)))},
            "capture_positions": positions,
            "weights_by_layer": {name: list(item["weights"]) for name in adapter.target_module_names()},
        })
    payload = {
        "artifact_version": KEEP_INPUTS_VERSION, "recipe_id": fixture["recipe_id"],
        "purpose": "keep_statistics", "example_only": True,
        "mask_policy_id": fixture["mask_policy_id"], "position_policy_id": fixture["position_policy_id"],
        "weight_policy_id": fixture["weight_policy_id"], "note": fixture["note"],
        "tokenizer_revision": config.tokenizer_revision, "mask_token_id": adapter.mask_token_id,
        "raw_source_file": str(fixture_file.resolve()), "raw_source_sha256": raw_digest,
        "inputs": rows,
    }
    # 分词只使用这份冻结配方；来源在处理中改变时不能发布错误的摘要。
    if file_sha256(fixture_file) != raw_digest:
        raise ValueError("raw keep recipe changed during tokenizing")
    write_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--p1-dir", type=Path, help="reuse completed P1/K from an existing round")
    source.add_argument("--layer", type=int, help="create a fresh P0/P1/K/C example run for this layer")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--run-id", help="unique name for a fresh run")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/p2_keep_example_only.json")
    args = parser.parse_args()
    settings = KeepStatisticsConfig.from_file(args.config)
    if args.p1_dir is not None:
        if args.run_id is not None:
            parser.error("--run-id is only used with a fresh --layer run")
        p1_dir = args.p1_dir.resolve()
    else:
        if args.layer < 0:
            parser.error("--layer must be nonnegative")
        run_id = args.run_id or "example_only_p2_keep_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        if run_id in ("", ".", "..") or "/" in run_id or "\\" in run_id:
            parser.error("--run-id must be one path component")
        p1_dir = args.output_root.resolve() / run_id / "round_0/p1"
        if p1_dir.parent.exists():
            raise FileExistsError("existing experiment will not be overwritten")
    run_root = p1_dir.parent
    frozen_path, report_path = run_root / "keep_inputs.json", run_root / "p2_keep_acceptance_report.json"
    if any(path.exists() for path in (frozen_path, report_path, run_root / "matrices/keep_statistics",
                                      run_root / "matrices/validation")):
        raise FileExistsError("existing keep statistics experiment will not be overwritten")
    stage_order = ["P0", "P1", "P2_K", "P2_C", "P2_QA"] if args.p1_dir is None else ["P2_C", "P2_QA"]
    report = {"status": "running", "phase": "upstream_P0_P1_K", "example_only": True,
              "scope": "real_model_stage_checks", "stage_order": stage_order,
              "formal_dataset_acceptance": False, "p2_stage_complete": False}
    print("Real model experiment order: " + " -> ".join(stage_order), flush=True)
    print(f"C smoke artifacts: {run_root}", flush=True)
    try:
        if args.p1_dir is None:
            subprocess.run(
                [sys.executable, "-m", "util.smoke_p2_keys", "--layer", str(args.layer),
                 "--model-path", str(args.model_path.resolve()), "--run-id", run_id,
                 "--output-root", str(args.output_root.resolve())], cwd=PROJECT_ROOT, check=True,
            )
            report["phase"] = "upstream_reports"
            report["upstream_reports"] = {}
            # 汇总的是本次真实调用生成的报告，不增加前向或代替完整 P2 产物验收。
            for stage, filename in (("P0_P1", "acceptance_report.json"),
                                    ("P2_K", "p2_keys_acceptance_report.json")):
                path = run_root / filename
                with path.open(encoding="utf-8") as stream:
                    previous = json.load(stream)
                if previous["status"] != "passed":
                    raise ValueError(f"{stage} real-model checks did not pass")
                report["upstream_reports"][stage] = {
                    "status": "passed", "report": str(path), "sha256": file_sha256(path),
                }
        report["phase"] = "freeze_keep_recipe"
        freeze_example_inputs(p1_dir, PROJECT_ROOT / "tests/fixtures/p2_keep_example_only.json", frozen_path)
        report["phase"] = "keep_statistics"
        write_json(report_path, report)
        print("Running P2_C: baseline replay, keep capture, C extraction and disk checks...", flush=True)
        result = extract_run(p1_dir, frozen_path, settings)
        if result["status"] != "complete" or not result["example_only"]:
            raise ValueError("example C extraction did not pass")
        report["phase"] = "matrix_artifact_validation"
        write_json(report_path, report)
        validation = validate_run(p1_dir)
        if validation["status"] != "complete" or not validation["example_only"]:
            raise ValueError("example P2 matrix bundle did not pass")
        report.update({"status": "passed", "phase": "complete", "keep_statistics": result,
                       "manifest_sha256": file_sha256(Path(result["output_dir"]) / "manifest.json"),
                       "matrix_validation": validation,
                       "p2_stage_complete": validation["p2_stage_complete"]})
    except Exception as error:
        report.update({"status": "failed", "failure": str(error)})
        if run_root.exists():
            write_json(report_path, report)
        raise
    write_json(report_path, report)
    print(json.dumps({"status": "passed", "stage_order": stage_order,
                      "report": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
