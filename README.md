# MDLM：掩码扩散语言模型知识编辑

本项目实现面向掩码扩散语言模型（MDLM）的事实知识编辑，目前适配本地 LLaDA-8B-Instruct。编辑流程在单进程中完成状态采样、写回因子求解、残差优化与权重合并验收；评价由独立入口运行。算法推导见[完整算法逻辑](docs/最终逻辑%20V1.0.md)，参数说明见[配置参数说明](docs/配置参数说明.md)。

## 环境准备

推荐 WSL/Linux、Python 3.10 和可运行 LLaDA-8B-Instruct 的 CUDA 设备。先安装与设备匹配的 PyTorch 2.5.1，再安装项目依赖：

```bash
conda create -n mdlm python=3.10
conda activate mdlm
python -m pip install -r requirement.txt
python -m pip check
```

`requirement.txt` 固定了已验证的依赖版本，但不是完整环境锁文件。将模型文件放在 `model/models/LLaDA-8B-Instruct/`，将案例数据放在 `data/datasets/cases.jsonl`；两者不随仓库分发。完整 8B 运行需要足够的显存和内存。

## 配置与编辑

在仓库根目录运行。默认配置由 `configs/main.json` 汇总：

- `configs/models/llada_8b_instruct.json`：模型路径、目标层和设备（默认 `cuda:0`）。
- `configs/datasets/cases.json`：数据路径与保持题划分。
- `configs/editors/mdlm.json`：采样、优化和验收参数。

先用一个案例检查完整编辑流程（已存在的运行目录名不能复用）：

```bash
python -u main.py --config configs/main.json --case-id cf_00001 --run-id "smoke_$(date +%Y%m%d_%H%M%S)"
```

对数据集中所有可编辑案例运行：

```bash
RUN_ID="cases_$(date +%Y%m%d_%H%M%S)"
python -u main.py --config configs/main.json --mode independent --run-id "$RUN_ID"
```

`independent` 是默认模式，每个源案例从原始模型开始；如需将已接受的编辑逐轮累积到同一个模型，使用 `--mode sequential`。可用 `--max-cases N` 限制源案例数，或重复传入 `--case-id ID` 选择案例；`--dataset`、`--model-path`、`--device` 可覆盖对应配置。

原生审核格式的 `cases.jsonl` 会读取完整文件，并只将 `gold_ready: true` 的记录转换为编辑事实。同一源案例的多个事实在一轮中联合编辑。`--case-id` 和 `--max-cases` 只限制编辑范围，不缩小默认保持题来源。

编辑报告保存在 `runs/<run_id>/run_report.json`。独立模式只为通过验收的案例保存 `case_<序号>/model.json` 与 `target_weights.safetensors`；连续模式在结束时保存 `final_model/`。报告中的 `accepted` 表示通过编辑器的内部验收门槛，最终答案表现需另行评价。

## 评价

编辑完成后，指定对应的运行目录启动评价：

```bash
python -u evaluate.py --config configs/evaluation/mdlm_ultraedit.json --run-dir "runs/$RUN_ID" --eval-id "${RUN_ID}_short"
```

默认协议 `diffusion_short_v1` 使用自主去噪。若要评价“先预测并评分、再回填正确答案”的 `diffusion_gold_v1`，将 `configs/evaluation/mdlm_ultraedit.json` 中的 `protocol` 改为 `diffusion_gold_v1`，再使用新的 `--eval-id` 运行同一命令；无需重新编辑模型。两种协议均按答案 token 正确率统计，并额外记录完整答案匹配率；它们是定长短答案诊断，不能直接等同于原版 UltraEdit 的自回归评价。

评价结果保存在 `evaluations/<eval_id>/`，包括 `summary.json`、`predictions.jsonl` 和 `manifest.json`。ES、GS、LS 等指标只对数据中实际存在的题组计算；缺失题组显示为 `not_applicable`。当前审核源格式的转换只提供编辑题和保持题，未将 `independent_test_questions` 映射为 GS 题组，因而不要把缺失的 GS 解读为零分。评价协议和指标细节见 [UltraEdit 评价文档](docs/UltraEdit评价/)。
