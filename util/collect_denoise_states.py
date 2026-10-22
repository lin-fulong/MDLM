"""Collect real-model P1 trajectories, then build fixed formula (1)–(3) supervision."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from MDLM.contracts import P1_ARTIFACT_VERSION, P1RunConfig, file_sha256
from MDLM.states import FactTrajectory, load_trajectories
from data.readers.json_cases import case_purpose, load_cases
from model.adapters.llada_8b_instruct import LLaDA8BInstructAdapter
from util.filter_compatible_states import filter_run


def collect_run(config: P1RunConfig) -> dict:
    """Load trusted cases, sample each selected fact, and save every pre-forward input."""
    config.check()
    if config.model_name != "LLaDA-8B-Instruct":
        raise ValueError("P1 currently has a sampler only for LLaDA-8B-Instruct")
    run_dir = config.p1_dir()
    # 加载大模型之前先占用独立运行目录；成功与失败产物均不覆盖。
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "artifact_version": P1_ARTIFACT_VERSION,
        "collection_status": "partial",
        "config": config.as_record(),
        "config_sha256": config.digest(),
        "forward_conditions": {
            "attention_mask": "all_ones",
            "position_ids": "implicit_zero_based",
            "attention_bias": None,
            "use_cache": False,
            "model_mode": "eval",
        },
        "fact_order": [item.case_id for item in config.facts],
        "replay_fact_indices": sorted(random.Random(config.seed).sample(range(len(config.facts)), config.replay_count)),
    }
    manifest_path = run_dir / "manifest.json"
    report_path = run_dir / "collection_report.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    completed: list[dict] = []
    phase = "setup"
    current_case_id = None
    try:
        cases = {case["case_id"]: case for case in load_cases(config.case_file)}
        model_path = Path(config.model_path)
        manifest["case_file_sha256"] = file_sha256(config.case_file)
        manifest["model_config_sha256"] = file_sha256(model_path / "config.json")
        manifest["tokenizer_sha256"] = file_sha256(model_path / "tokenizer.json")
        adapter = LLaDA8BInstructAdapter.from_local_tokenizer(
            model_path=config.model_path, target_layers=config.target_layers
        )
        manifest["mask_token_id"] = adapter.mask_token_id
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        model = adapter.load_model(config.model_path, torch_dtype=config.torch_dtype, device_map=config.device_map)
        phase = "sampling"
        with (run_dir / "trajectories.jsonl").open("w", encoding="utf-8") as stream:
            for fact in config.facts:
                current_case_id = fact.case_id
                case = cases[fact.case_id]
                masked_input, target = adapter.prepare_edit_input(case, fact.probe_id)
                seed = config.fact_seed(fact)
                steps = config.steps or len(masked_input.mask_positions)
                # 采样接口没有 target 参数；目标 token 仅在轨迹完成后写入监督侧。
                trace = adapter.sample_denoise(
                    model, masked_input, steps=steps, temperature=config.temperature, seed=seed
                )
                trajectory = FactTrajectory.from_sample(target, seed, trace)
                stream.write(json.dumps(trajectory.as_record(), ensure_ascii=False) + "\n")
                stream.flush()
                completed.append({
                    "case_id": fact.case_id,
                    "probe_id": fact.probe_id,
                    "purpose": case_purpose(case),
                    "pre_forward_count": len(trace.pre_forward_inputs),
                    "forward_logits_finite": True,
                    "replayed": False,
                    "logits_replayed": False,
                })
        phase = "replay"
        saved_trajectories = load_trajectories(run_dir / "trajectories.jsonl")
        for index in manifest["replay_fact_indices"]:
            saved = saved_trajectories[index]
            current_case_id = saved.case_id
            # 从磁盘读取首态重新采样，核对保存后的完整前向输入轨迹。
            replay = adapter.sample_denoise(
                model,
                saved.trace.pre_forward_inputs[0],
                steps=len(saved.trace.pre_forward_inputs),
                temperature=config.temperature,
                seed=saved.seed,
            )
            if replay != saved.trace:
                raise ValueError(f"{saved.case_id}: saved inputs, reveals, or mask logits did not replay")
            completed[index]["replayed"] = True
            completed[index]["logits_replayed"] = True
    except Exception as error:
        manifest["collection_status"] = "failed"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report_path.write_text(
            json.dumps({"completed": completed, "failure": str(error), "phase": phase,
                        "failed_case_id": current_case_id}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise

    report_path.write_text(
        json.dumps({"completed": completed, "failure": None}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest["collection_status"] = "complete"
    manifest["trajectories_sha256"] = file_sha256(run_dir / "trajectories.jsonl")
    manifest["collection_report_sha256"] = file_sha256(report_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return filter_run(run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="P1 JSON run configuration")
    args = parser.parse_args()
    config = P1RunConfig.from_file(args.config)
    report = collect_run(config)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
