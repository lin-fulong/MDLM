"""Real LLaDA P0 smoke check; the bundled case is example_only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.readers.json_cases import case_purpose, load_cases
from model.adapters.llada_8b_instruct import DEFAULT_MODEL_PATH, LLaDA8BInstructAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_file", type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device-map", default="auto", help="模型计算设备，如 cuda:0")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--layer", type=int, required=True, help="one explicitly chosen FFN layer")
    parser.add_argument("--forward", action="store_true", help="load 8B weights and run two replay-checked forwards")
    parser.add_argument("--output", type=Path, help="also save the JSON report to this path")
    args = parser.parse_args()
    cases = load_cases(args.case_file)
    adapter = LLaDA8BInstructAdapter.from_local_tokenizer(model_path=args.model_path, target_layers=(args.layer,))
    results = []
    model = adapter.load_model(args.model_path, torch_dtype=args.torch_dtype, device_map=args.device_map) if args.forward else None
    module_info = adapter.inspect_target_modules(model) if model is not None else None
    for case in cases:
        for probe in case["prompts"]["edit"]:
            model_input, target = adapter.prepare_edit_input(case, probe["probe_id"])
            row = {
                "case_id": case["case_id"],
                "probe_id": probe["probe_id"],
                "purpose": case_purpose(case),
                "mask_count": len(model_input.mask_positions),
                "full_mask_input": len(model_input.mask_positions) == len(target.token_ids)
                and all(model_input.input_ids[pos] == adapter.mask_token_id for pos in model_input.mask_positions),
                "subject_last_position": model_input.subject_last_position,
                "target_token_count": len(target.token_ids),
                "real_forward": False,
            }
            if model is not None:
                result = adapter.forward_with_capture(model, model_input)
                replay = adapter.forward_with_capture(model, model_input)
                import torch

                if not torch.allclose(result.mask_logits, replay.mask_logits, atol=1e-2, rtol=1e-3):
                    raise ValueError(f"{case['case_id']}: mask logits did not replay within tolerance")
                for name in result.module_inputs:
                    if not torch.allclose(result.module_inputs[name], replay.module_inputs[name], atol=1e-2, rtol=1e-3):
                        raise ValueError(f"{case['case_id']}: {name} input did not replay within tolerance")
                row["real_forward"] = True
                row["replay_within_tolerance"] = True
                row["mask_logits_shape"] = list(result.mask_logits.shape)
                row["target_modules"] = module_info
                row["ffn_input_shapes"] = {name: list(value.shape) for name, value in result.module_inputs.items()}
            results.append(row)
    if args.output is not None:
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
