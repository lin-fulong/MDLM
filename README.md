# MDLM 扩散知识编辑项目

本项目实现掩码扩散语言模型的事实知识编辑，算法定义见 [完整算法逻辑](docs/最终逻辑%20V1.0.md)。
当前适配 LLaDA-8B-Instruct。算法在同一进程中完成状态采样、固定因子求解、残差优化及合并接受，函数间直接传递内存对象，只保存最终编辑结果和运行摘要。
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

`configs/main.json` 组合模型、数据集和算法配置。分别在 `models/llada_8b_instruct.json` 修改模型与目标层，在 `datasets/cases.json` 修改数据路径与输入划分，在 `editors/mdlm.json` 修改算法超参数；逐项含义见[配置参数说明](docs/配置参数说明.md)。然后运行：

```bash
python -u main.py --config configs/main.json
```

算法默认读取主配置的 `editor`；`--editor` 可覆盖，不区分大小写，均未指定时使用 MDLM。
命令行路径相对当前目录，三个配置引用相对主配置目录，数据、模型和输出路径相对项目根目录。
λ/ρ 为标量时应用到全部目标层，也可按 `target_layers` 顺序填写等长列表；无需在算法配置中重复模型模块名。
`mdlm.json` 使用平铺参数名：`denoise_steps`、`ridge`、`rho`、`optimization_steps`、`selection_steps` 等；不再使用 P1～P5 分组，去噪步数与优化步数分别配置，候选选择步数不表示保存磁盘文件。
随机种子统一在 `configs/main.json` 设置 `seed`，默认为 `2026`，共同控制保持数据划分、轨迹采样与残差优化；子配置不再设置种子。`run_report.json` 记录本次使用的种子。
主配置的 `mode` 选择 independent/sequential，`max_cases` 限制源案例数，`case_ids` 选择源案例/事实。数据配置直接填写 `dataset_path`、`keep_dev_fraction`、`keep_test_fraction` 和 `statistics_scale`。

```bash
# 覆盖数据集或模型目录
python -u main.py --editor mdlm --dataset data/datasets/cases.jsonl --model-path model/models/LLaDA-8B-Instruct

# 选择计算设备
python -u main.py --editor mdlm --device cuda:0

# 选择源案例；可重复传入多个 --case-id
python -u main.py --editor mdlm --case-id cf_00001

# 连续编辑前 10 个源案例；命令行覆盖主配置
python -u main.py --mode sequential --max-cases 10

# 临时改回独立编辑；原 --sequential 及位置参数 sequential 仍兼容
python -u main.py --mode independent
```

默认独立编辑：每个源案例从最初模型开始，同一源案例的多个事实联合编辑。
连续编辑只接续已接受的结果，拒绝时恢复本轮基准。
保持输入始终由完整数据集确定，先按 case_ids 筛选、再按源案例分组和 max_cases 限量；数据集顺序不变，案例数不等于事实数。命令行 --case-id 替换配置列表，--mode/--max-cases 覆盖对应配置。选源案例时不拆散其事实，显式选事实编号时只编辑该事实，因此单案例与全数据集实验使用相同保持划分。
`--sequential` 表示跨源案例接续编辑：每个源案例执行一轮，拒绝后继续处理后续案例。
它与算法文档中同一事实集的可选第二轮刷新不同；当前入口未实现该刷新。

准备输入不再写缓存，`--cache-dir` 和 `--no-cache` 已移除。
旧的逐阶段磁盘命令已取消；算法计算均通过统一入口执行。
配置中的 `execution.backend` 继续保留原有的精度设置语义，运行流程统一为单进程。

## 算法代码

- `model/base.py`：模型输入、监督答案、前向捕获、去噪轨迹与本轮权重快照，以及适配器接口；其中的公式编号说明数据和接口用途。
- `model/adapters/llada_8b_instruct.py`：LLaDA 本地加载、模板对齐、单条/批量实际 FFN 输入捕获、可微完整词表读出、自主去噪与本轮权重恢复。
- `data/validation/alignment.py`：按实际 tokenizer 偏移定位主体 token，取得稳定边界的监督答案；为公式（1）～（4）和开发编辑提供位置与目标。
- `model/registry.py`：模型适配器选择入口，目前保留供后续模型接入，现行 runner 直接使用 LLaDA 适配器。
- `data/dataset.py`：读取 JSON/JSONL、展开 gold_ready 源案例，并准备固定保持划分、全 mask 上下文与题/位置权重；统计与训练共用训练分区，开发保持单独划分。
- `MDLM/runner.py`：组合配置、按源案例运行独立/连续编辑，固定最初 θ₀/W₀；`edit_round` 串联单轮算法，并提供最终权重保存和重载函数。
- `MDLM/states.py`：自主采集去噪轨迹、筛选兼容状态、构造三级监督权重，公式（1）～（3）。
- `MDLM/writeback.py`：提取 K、逐层累计 C，以 Cholesky 求解固定 Q，公式（4）～（9）；每层 C 求解后释放。
- `MDLM/optimize.py`：安装全位置 R(Qk) 分支，固定 Q、联合优化多层 R，按预定检查点选择优于 J(0) 的候选，公式（10）～（12）。
- `MDLM/merge.py`：按公共 α 从基准重建实际权重，核对因子/合并前向，按三个独立开发门槛接受或精确回退，公式（13）～（18）。
- `util/gpu_runtime.py`：选择运行设备；resident 模式关闭 TF32，退出时恢复原设置。它不实现编号公式，只固定公式（1）～（18）的共同运行环境。
- `util/io.py`：组合主配置引用的模型、数据集与算法参数，保存运行摘要和最终模型元数据，并为保持输入生成确定性划分摘要；摘要服务公式（5）（10）（16）的输入准备，本身不计算公式。

`contracts.py` 已删除。数学参数放在对应算法模块的参数类中；
不再需要阶段 manifest、文件摘要、版本合同或临时 JSON 配置来衔接计算。

算法定义沿用原实现：K 的事实列序、C 的显式权重与未中心化累计、K/C/Q/R 的配置精度、
Adam 预算与检查点、公共 α 的尝试顺序、KL(theta_0 || candidate) 和开发门槛均保持一致。
C 按层累计并立即交给 Q 求解，避免同时保留所有层的大方阵。
保持题按 keep_dev_fraction/keep_test_fraction 留出：开发保持题不参与 C 统计/保持训练，用于已有的接受门槛；额外留出题不参与训练或候选选择，预留独立检验，当前不执行评价。其余保持题用于 C 统计和可选保持训练，原比例与权重保持不变。

原有数据读取和准备脚本已合并到 `data/dataset.py`，可选格式转换命令为：

```bash
python -m data.dataset data/datasets/cases.jsonl /tmp/edit_cases.jsonl --case-id cf_00001
```

通常直接启动 `main.py` 即可自动读取和转换，无需先生成转换文件。

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
from MDLM.runner import load_final_model

adapter, model = load_final_model("runs/<run_id>/final_model", device_map="cuda:0")
```

## 增加其他编辑算法

在新模块中实现 `run_dataset(config_file, **options)`，并在 `main.py` 的
`EDITORS` 中登记 `"算法名": "包名.runner"`。主配置中同步选择 `editor` 和 `editor_config`；新算法可用 `util.io.read_config` 读取组合后的配置，入口按需导入对应模块：

```bash
python main.py --editor 算法名 --config configs/对应主配置.json
```

## 本地验证

`tests/` 仅在本地保留。新的回归测试用重构前保存的小模型基准，逐元素比较
K、C、Q、R、最终权重、优化损失和验收决策，并检查连续编辑、失败回退、CUDA 与最终结果重载：

```bash
python -m unittest tests.test_module_merge tests.test_split_config tests.test_main_cli tests.test_memory_pipeline tests.test_memory_runner tests.test_memory_preparation tests.test_memory_gpu -b -q
```

原始算法推导和历史联调记录保留在 `docs/`；其中的阶段文件和旧命令描述对应当时版本。
