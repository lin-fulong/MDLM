"""One command for real P4 candidate forward/backward, optimization and saved-artifact checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from MDLM.contracts import ResidualOptimizationConfig, file_sha256
from util.accept_p0_p1_single_case import PROJECT_ROOT
from util.io import write_json
from util.optimize_residuals import optimize_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1-dir", type=Path, required=True, help="same-round P1 directory with accepted P2 and fixed P3 Q")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/p4_example_only.json")
    args = parser.parse_args()
    p1_dir = args.p1_dir.resolve()
    path = p1_dir.parent / "p4_candidate_acceptance_report.json"
    if path.exists() or (p1_dir.parent / "residuals").exists():
        raise FileExistsError("existing P4 experiment will not be overwritten")
    report = {"status": "running", "p4_stage_complete": False, "weights_merged": False,
              "formal_dataset_acceptance": False}
    try:
        result = optimize_run(p1_dir, ResidualOptimizationConfig.from_file(args.config))
        report.update(status="passed" if result["status"] == "complete" else "failed", candidate=result,
                      example_only=result["example_only"], p4_stage_complete=result["p4_stage_complete"],
                      manifest_sha256=file_sha256(Path(result["output_dir"]) / "manifest.json"))
    except Exception as error:
        report.update(status="failed", failure=str(error))
        if p1_dir.parent.exists():
            write_json(path, report)
        raise
    write_json(path, report)
    print(json.dumps({"status": report["status"], "report": str(path)}, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
