"""Real LLaDA P5 constructed-case checks: configured gates, forced rejection, reload."""

from dataclasses import replace
import argparse
import gc
import json
from pathlib import Path

import torch

from MDLM.contracts import AcceptanceConfig, DEVELOPMENT_KEEP_VERSION, P1RunConfig, file_sha256
from MDLM.workflow import load_final_model, run_p5
from model.adapters.llada_8b_instruct import LLaDA8BInstructAdapter
from util.extract_reference_keys import write_json
from util.validate_matrix_artifacts import read_json, require


def smoke(p1_dir: Path, *, residual_dir: Path | None = None, output_dir: Path | None = None,
          config_file=Path("configs/p5_example_only.json")):
    p1_dir = p1_dir.resolve()
    output = output_dir.resolve() if output_dir is not None else p1_dir.parent / "p5_smoke"
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "example_only": True, "p5_stage_complete": False}
    try:
        config = P1RunConfig.from_record(read_json(p1_dir / "manifest.json")["config"])
        require(config.round_id == 0, "the constructed P5 smoke uses theta_0=theta_b only in round 0")
        q = read_json(p1_dir.parent / "matrices/writeback_factors/manifest.json")
        values = read_json(config_file)
        adapter = LLaDA8BInstructAdapter.from_local_tokenizer(model_path=config.model_path, target_layers=config.target_layers)
        prompt = adapter.tokenizer.apply_chat_template([{"role": "user", "content": "日常常识：清水通常是什么颜色？"}],
                                                       tokenize=False, add_generation_prompt=True)
        ids = tuple(adapter.tokenizer.encode(prompt, add_special_tokens=False))
        positions = (len(ids), len(ids) + 1, len(ids) + 2)
        keep_path = output / "constructed_development_keep.json"
        write_json(keep_path, {"artifact_version": DEVELOPMENT_KEEP_VERSION, "purpose": "keep_dev", "example_only": True,
                   "tokenizer_revision": config.tokenizer_revision, "mask_token_id": adapter.mask_token_id,
                   "metadata": {"note": "独立人工构造开发保持输入，仅作技术联调，不是正式验收数据"},
                   "inputs": [{"input": {"input_ids": list(ids + (adapter.mask_token_id,) * 3),
                                           "attention_mask": [1] * (len(ids) + 3), "mask_positions": list(positions)},
                               "positions": list(positions), "weights": [1 / 3] * 3}]})
        values.update(dev_keep_file=str(keep_path), reference_model_path=config.model_path,
                      reference_model_revision=q["model_identity"]["model_revision"],
                      reference_tokenizer_revision=config.tokenizer_revision, allow_example_only=True)
        settings = AcceptanceConfig.from_record(values)
        write_json(output / "configured_settings.json", settings.as_record())
        # 拒绝路径的固定测试参数，启动前已写入；不为首个实验放宽任何门槛。
        rejection = replace(settings, delta=1e6)
        write_json(output / "forced_rejection_settings.json", rejection.as_record())
        report["configured"] = run_p5(p1_dir, settings, residual_dir=residual_dir, output_dir=output / "configured")
        report["forced_rejection"] = run_p5(p1_dir, rejection, residual_dir=residual_dir, output_dir=output / "forced_rejection")
        rejected = read_json(output / "forced_rejection/acceptance_report.json")
        require(report["forced_rejection"]["decision"] == "rejected" and rejected["baseline_restored"],
                "forced rejection did not recover the baseline")
        adapter, model = load_final_model(output / "configured")
        del adapter, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        report.update(status="passed", actual_model_reload_passed=True,
                      configured_manifest_sha256=file_sha256(output / "configured/manifest.json"))
        return report
    except Exception as error:
        report.update(status="failed", failure=str(error))
        raise
    finally:
        write_json(output / "p5_acceptance_report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1-dir", type=Path, required=True)
    parser.add_argument("--residual-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/p5_example_only.json"))
    args = parser.parse_args()
    result = smoke(args.p1_dir, residual_dir=args.residual_dir, output_dir=args.output_dir, config_file=args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
