"""Run P5 actual weight merging, development gates and baseline recovery."""

import argparse
import json
from pathlib import Path

from MDLM.contracts import AcceptanceConfig
from MDLM.workflow import run_p5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("p1_dir", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--residual-dir", type=Path, help="complete P4 version 2 candidate directory")
    parser.add_argument("--output-dir", type=Path, help="new output directory; existing outputs are preserved")
    args = parser.parse_args()
    result = run_p5(args.p1_dir, AcceptanceConfig.from_file(args.config),
                    residual_dir=args.residual_dir, output_dir=args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # 门槛拒绝是正常算法结果；仅执行错误抛异常，拒绝不能伪装成接受。


if __name__ == "__main__":
    main()
