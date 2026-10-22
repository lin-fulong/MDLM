"""Build fixed P1 supervision and a coverage report from saved trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from MDLM.contracts import P1_ARTIFACT_VERSION, config_digest, file_sha256
from MDLM.states import build_supervision, load_trajectories, save_supervision


def filter_run(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    report_path = run_dir / "coverage_report.json"
    fixed_path = run_dir / "fixed_states.jsonl"
    if report_path.exists() or fixed_path.exists():
        raise FileExistsError("fixed P1 artifacts already exist for this run")
    report: dict = {}
    try:
        with (run_dir / "manifest.json").open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest["artifact_version"] != P1_ARTIFACT_VERSION:
            raise ValueError("unsupported P1 artifact version; recollect with the current code")
        if manifest["collection_status"] != "complete":
            raise ValueError("state collection is incomplete; fixed supervision cannot be published")
        # 这里只核对本轮算法产物的来源，仍不调用输入数据集质量检查。
        if config_digest(manifest["config"]) != manifest["config_sha256"]:
            raise ValueError("saved run configuration digest differs")
        for name in ("trajectories", "collection_report"):
            suffix = ".jsonl" if name == "trajectories" else ".json"
            if file_sha256(run_dir / (name + suffix)) != manifest[name + "_sha256"]:
                raise ValueError(f"{name} digest differs from the collection manifest")
        trajectories = load_trajectories(run_dir / "trajectories.jsonl")
        selected = [(item["case_id"], item["probe_id"]) for item in manifest["config"]["facts"]]
        if ([(item.case_id, item.probe_id) for item in trajectories] != selected
                or [item.case_id for item in trajectories] != manifest["fact_order"]):
            raise ValueError("collected fact/probe order differs from the frozen selection")
        with (run_dir / "collection_report.json").open(encoding="utf-8") as stream:
            collection = json.load(stream)
        if collection["failure"] is not None or [
            (item["case_id"], item["probe_id"]) for item in collection["completed"]
        ] != selected:
            raise ValueError("collection report differs from the completed fact/probe selection")
        facts, report = build_supervision(trajectories, manifest["mask_token_id"])
        report.update({
            "run_id": manifest["config"]["run_id"],
            "round_id": manifest["config"]["round_id"],
            "baseline_id": manifest["config"]["baseline_id"],
            "config_sha256": manifest["config_sha256"],
            "trajectories_sha256": manifest["trajectories_sha256"],
            "replayed_fact_count": sum(item["replayed"] for item in collection["completed"]),
            "logits_replayed_fact_count": sum(item["logits_replayed"] for item in collection["completed"]),
            "example_only_fact_count": sum(item["purpose"] == "example_only" for item in collection["completed"]),
        })
        if not report["formula_checks_passed"]:
            raise ValueError("formula (1)-(3) checks failed; fixed supervision cannot be published")
        save_supervision(fixed_path, facts)
        report["status"] = "complete"
    except Exception as error:
        report.update({"status": "failed", "failure": str(error)})
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="runs/<run_id>/round_<n>/p1")
    args = parser.parse_args()
    report = filter_run(args.run_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
