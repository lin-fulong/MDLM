# MDLM 扩散知识编辑项目

本项目实现掩码扩散语言模型的事实知识编辑，算法定义见 [完整算法逻辑](docs/最终逻辑%20V1.0.md)。
当前适配 LLaDA-8B-Instruct。算法在同一进程中执行 P0～P5，阶段间直接传递内存对象，只保存最终编辑结果和运行摘要。
当前文件职责和逐文件阅读顺序见[现行代码结构](docs/项目结构设计/00_当前代码结构与阅读顺序.md)。

## 环境

运行基线为 WSL/Linux、Python 3.10 和 `requirement.txt` 中固定的依赖。
先安装与设备匹配的 PyTorch 2.5.1，再安装其余依赖；已配置好的 `mdlm` 环境可以直接使用。

```bash
conda activate mdlm
python -m pip install -r requirement.txt
python -m pip check
```

模型目录默认为 `model/models/LLaDA-8B-Instruct`，数据集默认为 `data/datasets/cases.jsonl`。
模型文件和数据集不上传至 Git。
小模型测试能验证计算和接口；完整 8B 实验仍需要足够的显存与内存。

## 启动

`configs/` 仅保留 `full_run.json`。直接修改其中的数据集、模型、目标层和算法超参数，逐项含义见[配置参数说明](docs/配置参数说明.md)，然后运行：

```bash
python -u main.py --editor mdlm --config configs/full_run.json
```

`--editor` 不区分大小写，省略时默认 MDLM。
命令行路径相对当前目录，JSON 中的相对路径相对项目根目录。

```bash
# 覆盖数据集或模型目录
python -u main.py --editor mdlm --dataset data/datasets/cases.jsonl --model-path model/models/LLaDA-8B-Instruct

# 选择计算设备
python -u main.py --editor mdlm --device cuda:0

# 选择源案例；可重复传入多个 --case-id
python -u main.py --editor mdlm --case-id cf_00001

# 连续编辑：下一轮从上一轮已接受的模型继续
python -u main.py --editor mdlm --sequential
```

默认独立编辑：每个源案例从最初模型开始，同一源案例的多个事实联合编辑。
连续编辑只接续已接受的结果，拒绝时恢复本轮基准。
保持输入始终由完整数据集确定，案例筛选只决定编辑对象，因此单案例与全数据集实验使用相同划分。
`--sequential` 表示跨源案例接续编辑：每个源案例执行一轮，拒绝后继续处理后续案例。
它与算法文档中同一事实集的可选第二轮刷新不同；当前入口未实现该刷新。

准备输入不再写缓存，`--cache-dir` 和 `--no-cache` 已移除。
旧的逐阶段磁盘命令已取消；算法计算均通过统一入口执行。
配置中的 `execution.backend` 继续保留原有的精度设置语义，运行流程统一为单进程。

## 算法代码

- `model/base.py`：模型输入、监督答案、前向捕获、去噪轨迹与本轮权重快照，以及适配器接口；其中的公式编号说明数据和接口用途。
- `model/adapters/llada_8b_instruct.py`：LLaDA 本地加载、模板对齐、单条/批量实际 FFN 输入捕获、可微完整词表读出、自主去噪与本轮权重恢复。
- `data/validation/alignment.py`：按实际 tokenizer 偏移定位主体 token，取得稳定边界的监督答案；为公式（1）～（4）和开发编辑提供位置与目标。
- `data/readers/reviewed_cases.py`：按 gold_ready 筛选源案例、逐请求展开事实，保留联合编辑来源；共享保持题只附到首个事实，监督答案取编辑题字段。
- `data/readers/json_cases.py`：读取 JSON/JSONL 并统一列表形式，按首条记录识别是否调用审核源格式转换；用途标签辅助函数目前没有正式调用方。
- `MDLM/runner.py`：按源案例分组、加载一次模型，固定最初 θ₀/W₀ 参照，控制独立/连续编辑与本轮恢复，保存最终结果。
- `MDLM/states.py`：去噪状态的兼容性筛选与三级监督权重，公式（1）～（3）。
- `util/prepare_dataset_inputs.py`：固定保持题划分、全 mask 上下文及题/位置权重，为公式（5）（10）（16）和开发编辑准备输入；统计与训练共用训练分区，开发保持单独划分。
- `util/extract_reference_keys.py`：从本轮全 mask 初态提取 K，公式（4）。
- `MDLM/keep_statistics.py` 与 `util/extract_keep_statistics.py`：每批保持输入前向一次，按公式（5）逐层累计 C；直接返回矩阵。
- `MDLM/writeback.py`：以原有 λ 和 Cholesky 方法求解并直接返回固定 Q，公式（6）～（9）。
- `MDLM/candidate.py` 与 `MDLM/optimize.py`：安装全位置 R(Qk) 分支，固定 Q、联合优化多层 R，按预定检查点选择优于 J(0) 的候选，公式（10）～（12）。
- `MDLM/merge.py` 与 `MDLM/acceptance.py`：按公共 α 从基准重建实际权重，核对因子/合并前向，按三个独立开发门槛接受或精确回退，公式（13）～（18）。
- `MDLM/workflow.py`：准备最初 θ₀ 的保持参考，串联单轮监督 → K/C → Q → R → 开发接受，并保存、重载最终目标权重。
- `util/gpu_runtime.py`：选择运行设备；resident 模式关闭 TF32，退出时恢复原设置。它不实现编号公式，只固定公式（1）～（18）的共同运行环境。
- `util/io.py`：读取配置、保存运行摘要和最终模型元数据，并为保持输入生成确定性划分摘要；摘要服务公式（5）（10）（16）的输入准备，本身不计算公式。

`contracts.py` 已删除。数学参数放在对应算法模块的参数类中；
不再需要阶段 manifest、文件摘要、版本合同或临时 JSON 配置来衔接计算。

算法定义沿用原实现：K 的事实列序、C 的显式权重与未中心化累计、K/C/Q/R 的配置精度、
Adam 预算与检查点、公共 α 的尝试顺序、KL(theta_0 || candidate) 和开发门槛均保持一致。
C 按层累计并立即交给 Q 求解，避免同时保留所有层的大方阵。
原有 test_fraction 仍保留为不参与保持训练和开发验收的留出划分，避免改变训练输入。

## 最终结果

独立编辑对每个接受的源案例保存一个结果目录：

```text
runs/<run_id>/
  run_report.json
  case_0/
    model.json
    target_weights.safetensors
```

连续编辑只在全部轮次完成后保存 `final_model/`，其中同样只有 `model.json` 和
`target_weights.safetensors`。运行摘要记录各轮接受/拒绝、所选 α 和开发指标。
不会保存 P0～P5 的轨迹、K/C/Q/R、阶段报告、参考 logits 或优化器状态。

保存的是最终目标层权重，加载时仍需原始模型目录：

```python
from MDLM.workflow import load_final_model

adapter, model = load_final_model("runs/<run_id>/final_model", device_map="cuda:0")
```

## 增加其他编辑算法

在新模块中实现 `run_dataset(config_file, **options)`，并在 `main.py` 的
`EDITORS` 中登记 `"算法名": "包名.runner"`。各算法读取自己的配置，入口按需导入对应模块：

```bash
python main.py --editor 算法名 --config configs/对应配置.json
```

## 本地验证

`tests/` 仅在本地保留。新的回归测试用重构前保存的小模型基准，逐元素比较
K、C、Q、R、最终权重、优化损失和验收决策，并检查连续编辑、失败回退、CUDA 与最终结果重载：

```bash
python -m unittest tests.test_memory_pipeline tests.test_memory_runner tests.test_memory_preparation tests.test_memory_gpu -b -q
```

原始算法推导和历史联调记录保留在 `docs/`；其中的阶段文件和旧命令描述对应当时版本。
