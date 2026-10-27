"""Accept P3 Q factors from existing real P2 artifacts; no model is loaded."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from MDLM.contracts import WritebackConfig, file_sha256
from util.accept_p0_p1_single_case import PROJECT_ROOT
from util.io import write_json
from util.solve_writeback_factor import solve_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/p3_writeback_example_only.json")
    args = parser.parse_args()
    p1_dir = args.p1_dir.resolve()
    path = p1_dir.parent / "p3_writeback_acceptance_report.json"
    if path.exists() or (p1_dir.parent / "matrices/writeback_factors").exists():
        raise FileExistsError("existing P3 experiment will not be overwritten")
    report = {"status": "running", "phase": "solve_Q", "model_forward_count": 0,
              "formal_dataset_acceptance": False, "p3_stage_complete": False}
    try:
        result = solve_run(p1_dir, WritebackConfig.from_file(args.config))
        report.update({"status": "passed", "phase": "complete", "writeback_factors": result,
                       "manifest_sha256": file_sha256(Path(result["output_dir"]) / "manifest.json"),
                       "example_only": result["example_only"], "p3_stage_complete": result["p3_stage_complete"]})
    except Exception as error:
        report.update({"status": "failed", "failure": str(error)})
        if p1_dir.parent.exists():
            write_json(path, report)
        raise
    write_json(path, report)
    print(json.dumps({"status": "passed", "model_forward_count": 0, "report": str(path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
