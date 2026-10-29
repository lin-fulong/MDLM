"""知识编辑算法的统一入口：选择算法，读取启动参数，执行实验。"""

from __future__ import annotations

import argparse
import json
from importlib import import_module
from pathlib import Path

from util.io import read_json


ROOT = Path(__file__).resolve().parent

# 新算法在此登记模块路径，模块须提供 run_dataset(config_file, **options)。
# 按需导入，避免运行其他算法时加载 MDLM 及其模型依赖。
EDITORS = {"mdlm": "MDLM.runner"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--editor", type=str.lower, choices=EDITORS,
                        help="覆盖主配置中的算法名称，缺省 mdlm，不区分大小写")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/main.json", help="主配置 JSON")
    parser.add_argument("--dataset", type=Path, help="覆盖配置中的数据集路径")
    parser.add_argument("--model-path", type=Path, help="覆盖配置中的原始模型目录")
    parser.add_argument("--device", help="计算设备，如 cuda:0 或 cpu")
    parser.add_argument("--case-id", action="append", help="选择源案例或事实，可重复指定")
    parser.add_argument("--max-cases", type=int, help="最多编辑的源案例数，覆盖主配置")
    parser.add_argument("--run-id", help="本次实验的输出目录名")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--mode", dest="edit_mode", choices=("independent", "sequential"), help="覆盖主配置的编辑模式")
    modes.add_argument("--sequential", action="store_true", help="将已接受的编辑作为下一轮基准")
    parser.add_argument("mode", nargs="?", choices=("sequential",),
                        help="兼容原有连续编辑命令；默认每轮从原始模型开始")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.config.is_file():
        parser.error(f"配置文件不存在：{args.config}；请通过 --config 指定主配置")
    if args.mode and args.edit_mode == "independent":
        parser.error("sequential 与 --mode independent 不能同时使用")

    editor = args.editor or read_json(args.config).get("editor", "mdlm").lower()
    if editor not in EDITORS:
        parser.error(f"未注册的编辑算法：{editor}")
    run = import_module(EDITORS[editor]).run_dataset
    result = run(
        args.config,
        sequential=True if args.mode or args.sequential else (
            args.edit_mode == "sequential" if args.edit_mode is not None else None),
        run_id_override=args.run_id,
        dataset_override=args.dataset,
        model_override=args.model_path,
        device_override=args.device,
        case_ids=args.case_id,
        max_cases_override=args.max_cases,
    )
    summary = {key: result.get(key) for key in (
        "status", "mode", "fact_count", "accepted_count", "rejected_count", "final_checkpoint_dir")}
    print(json.dumps({"editor": editor, **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
