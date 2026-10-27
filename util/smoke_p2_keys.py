"""One-command real LLaDA P0/P1 + formula (4) check on the example_only case."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from MDLM.contracts import ReferenceKeyConfig, file_sha256
from model.adapters.llada_8b_instruct import DEFAULT_MODEL_PATH
from util.accept_p0_p1_single_case import PROJECT_ROOT
from util.extract_reference_keys import extract_run
from util.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/p2_keys_example_only.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--run-id", help="unique run name; automatically generated if omitted")
    args = parser.parse_args()
    if args.layer < 0:
        parser.error("--layer must be nonnegative")
    run_id = args.run_id or "example_only_p2_keys_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    if run_id in ("", ".", "..") or "/" in run_id or "\\" in run_id:
        parser.error("--run-id must be one path component")
    settings = ReferenceKeyConfig.from_file(args.config)
    output_root = args.output_root.resolve()
    run_root = output_root / run_id / "round_0"
    if run_root.exists():
        raise FileExistsError(f"existing experiment will not be overwritten: {run_root}")
    report_path = run_root / "p2_keys_acceptance_report.json"
    report = {"status": "running", "scope": "single_case_technical_acceptance",
              "example_only": True, "formal_dataset_acceptance": False,
              "p2_stage_complete": False, "phase": "p0_p1"}
    print(f"P2 key smoke artifacts: {run_root}", flush=True)
    try:
        # 独立进程依次加载 P0/P1，进程退出后释放显存，再加载 P2 的同一基准。
        subprocess.run(
            [sys.executable, "-m", "util.accept_p0_p1_single_case", "--layer", str(args.layer),
             "--model-path", str(args.model_path.resolve()), "--run-id", run_id,
             "--output-root", str(output_root)], cwd=PROJECT_ROOT, check=True,
        )
        with (run_root / "acceptance_report.json").open(encoding="utf-8") as stream:
            previous = json.load(stream)
        if previous["status"] != "passed":
            raise ValueError("P0/P1 single-case checks did not pass")
        report["phase"] = "reference_keys"
        write_json(report_path, report)
        result = extract_run(run_root / "p1", settings)
        keys_dir = Path(result["output_dir"])
        with (keys_dir / "manifest.json").open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest["status"] != "complete" or manifest["example_only_fact_count"] != 1:
            raise ValueError("reference key artifacts did not pass the single-case checks")
        report.update({"status": "passed", "phase": "complete", "keys": result,
                       "p0_p1_report_sha256": file_sha256(run_root / "acceptance_report.json"),
                       "key_manifest_sha256": file_sha256(keys_dir / "manifest.json")})
    except Exception as error:
        report.update({"status": "failed", "failure": str(error)})
        if run_root.exists():
            write_json(report_path, report)
        raise
    write_json(report_path, report)
    print(json.dumps({"status": report["status"], "report": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
