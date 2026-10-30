"""评估已完成的 MDLM 运行；独立入口，不触发模型编辑。"""

import argparse
import json
from pathlib import Path

from util.experiment_log import append_command_output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parent / "configs/evaluation/mdlm_ultraedit.json")
    parser.add_argument("--run-dir", type=Path, help="已完成编辑的 runs/<run_id> 目录")
    parser.add_argument("--device", help="cpu 或 cuda:0")
    parser.add_argument("--model-path", type=Path, help="原始模型迁移后的本地路径")
    parser.add_argument("--dataset", type=Path, help="同一源数据迁移后的文件路径，不改变读取器或案例 ID")
    parser.add_argument("--eval-id", help="尚不存在的评估输出目录名")
    args = parser.parse_args(argv)
    from evaluation.runner import run_evaluation
    result = run_evaluation(args.config, run_dir=args.run_dir, device=args.device,
                            model_path=args.model_path, dataset_path=args.dataset, eval_id=args.eval_id)
    output = json.dumps({key: result[key] for key in (
        "status", "output_dir", "expected_tasks", "completed_tasks", "coverage")},
        ensure_ascii=False, indent=2)
    print(output)
    append_command_output(result["log_file"], output)


if __name__ == "__main__":
    main()
