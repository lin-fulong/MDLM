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
算法输入、分词对齐和权重约束使用 `assert`，请使用普通 `python` 启动，不使用会移除断言的 `python -O` / `-OO`。

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
保持输入由完整已读取案例集或显式 keep_path 确定，不受 case_ids/max_cases 影响；先按 case_ids 筛选、再按源案例分组和 max_cases 限量。数据集顺序不变，案例数不等于事实数。命令行 --case-id 替换配置列表，--mode/--max-cases 覆盖对应配置。选源案例时不拆散其事实，显式选事实编号时只编辑该事实，因此相同读取配置下的单案例与全数据集实验使用相同保持划分。
`--sequential` 表示跨源案例接续编辑：每个源案例执行一轮，拒绝后继续处理后续案例。
它与算法文档中同一事实集的可选第二轮刷新不同；当前入口未实现该刷新。

准备输入不再写缓存，`--cache-dir` 和 `--no-cache` 已移除。
旧的逐阶段磁盘命令已取消；算法计算均通过统一入口执行。
配置中的 `execution.backend` 继续保留原有的精度设置语义，运行流程统一为单进程。

## 算法代码

- `model/base.py`：共用模型输入、前向捕获结果与适配接口；监督答案和去噪轨迹由 MDLM 管理。
- `model/adapters/llada_8b_instruct.py`：LLaDA 本地加载、聊天模板、目标层名、指定位置的单条/等长批量捕获、可微读出与通用快照工具的调用。
- `data/validation/alignment.py`：按实际 tokenizer 偏移定位主体 token，取得稳定边界的监督答案；为公式（1）～（4）和开发编辑提供位置与目标。
- `model/registry.py`：runner 和最终结果重载共用的适配器工厂，默认加载 LLaDA 的本地 tokenizer。
- `data/dataset.py`：按配置选择读取器；保留原生 JSON/JSONL、gold_ready 源案例展开及可选转换命令。
- `data/readers/`：ZSRE、UltraEditBench、WikiBigEdit、FEVER 的文本映射、流式读取与必要字段检查；不执行模型分词或评估。
- `MDLM/runner.py`：组合配置、按源案例运行独立/连续编辑，固定最初 θ₀/W₀；`edit_round` 串联单轮算法，并提供最终权重保存和重载函数。
- `MDLM/states.py`：编辑/开发/保持输入与固定分区、自主去噪、兼容筛选、三级监督权重；公式（1）～（3）及后续 K/C/KL 的输入配方。
- `MDLM/writeback.py`：提取 K、逐层累计 C，以 Cholesky 求解固定 Q，公式（4）～（9）；每层 C 求解后释放。
- `MDLM/optimize.py`：安装全位置 R(Qk) 分支，固定 Q、联合优化多层 R，按预定检查点选择优于 J(0) 的候选，公式（10）～（12）。
- `MDLM/merge.py`：按公共 α 从基准重建实际权重，核对因子/合并前向，按三个独立开发门槛接受或精确回退，公式（13）～（18）。
- `util/gpu_runtime.py`：选择运行设备；resident 模式关闭 TF32，退出时恢复原设置。它不实现编号公式，只固定公式（1）～（18）的共同运行环境。
- `util/nethook.py`：通用 hook、指定位置输入/逐次输出捕获、权重快照/恢复、推理缓存转换与原精度摘要；不决定算法策略。
- `util/io.py`：配置/JSON、项目相对路径、运行目录名和确定性 JSON 摘要。

`contracts.py` 已删除。数学参数放在对应算法模块的参数类中；
不再需要阶段 manifest、文件摘要、版本合同或临时 JSON 配置来衔接计算。

算法定义沿用原实现：K 的事实列序、C 的显式权重与未中心化累计、K/C/Q/R 的配置精度、
Adam 预算与检查点、公共 α 的尝试顺序、KL(theta_0 || candidate) 和开发门槛均保持一致。
C 按层累计并立即交给 Q 求解，避免同时保留所有层的大方阵。
保持题按 keep_dev_fraction/keep_test_fraction 留出：开发保持题不参与 C 统计/保持训练，用于已有的接受门槛；额外留出题不参与训练或候选选择，预留独立检验，当前不执行评价。其余保持题用于 C 统计和可选保持训练，原比例与权重保持不变。

`data/dataset.py` 负责读取和源格式转换；MDLM 专用的保持划分、输入编码和位置权重在 `MDLM/states.py`。可选格式转换命令为：

```bash
python -m data.dataset data/datasets/cases.jsonl /tmp/edit_cases.jsonl --case-id cf_00001
```

通常直接启动 `main.py` 即可自动读取和转换，无需先生成转换文件。

接入外部数据时，在主配置中选择对应的 `datasets/*.json`；具体字段、限量、独立保持来源和 FEVER 主体标注见[可选数据读取器](docs/配置参数说明.md#可选数据读取器)。默认原生配置保持不变；数据集 JSON/JSONL 原文件只读。外部示例配置有显式读取上限，`max_cases` 只限制编辑数量。改写、persona 和多跳问题由独立评估入口使用。

`MDLM/` 保留五个核心文件。算法决定捕获哪些层和位置，model 提供模型操作，util 只提供通用机制；model/data/util 均不反向依赖 MDLM。

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

## 独立 UltraEdit 评估

UltraEdit 评价资料集中在 [docs/UltraEdit评价](docs/UltraEdit评价/)：[原版指标计算逻辑](docs/UltraEdit评价/原版UltraEdit评估指标计算逻辑.md) 解释正确答案前缀与逐题评分；[当前 MDLM 评价实现](docs/UltraEdit评价/MDLM接入后UltraEdit评估指标计算逻辑.md) 说明自主去噪评分；[标准答案回填评价方案](docs/UltraEdit评价/MDLM置信度顺序标准答案回填评价方案.md) 说明已实现的 diffusion_gold_v1；[evaluation 文件职责](docs/UltraEdit评价/evaluation文件功能说明.md) 说明评价模块各文件的功能边界。

完成编辑后，在 WSL 的 `mdlm` 环境中执行（将目录名替换为实际运行目录）：

```bash
python -u evaluate.py --run-dir "runs/<run_id>" --config configs/evaluation/ultraedit.json
```

这是独立评估入口，读取已经完成的编辑报告和最终目标层权重。`evaluation/` 不调用编辑、取键、残差优化或合并验收；现有 MDLM 算法文件、模型适配器和数据读取器保持原实现。评估使用独立模型实例，串行切换案例时先恢复原始权重。原始 JSON/JSONL 和编辑产物只读，输出放在 `evaluations/<eval_id>/`。

当前实现 `ultraedit` 框架，支持 `diffusion_short_v1`（自主去噪）和 `diffusion_gold_v1`（先预测、先评分、再回填）。旧配置 `configs/evaluation/ultraedit.json` 保持原协议；新协议使用 `configs/evaluation/ultraedit_gold.json`，也可修改配置的 `protocol` 切换。对应指标为 edit→ES、generalization→GS、preservation→LS、persona→person_score、portability→mhop_score。它们均为逐题答案 token 正确比例；LS 相对于无关题标准答案计算。另记录整段 token 完全匹配率和编辑前后输出一致率。

两个协议均采用模型原聊天模板和全 mask 答案窗口。旧协议只用参考答案确定长度并事后评分；新协议在每轮选位后先保存预测并评分，再将本轮所选位置统一回填标准 token，供下一轮使用。新协议返回的 token_ids/text 是评分预测的拼接，correct 保存逐位置对错，trace 保存逐轮记录，不能将其视为自主生成答案。每步预测剩余位置，按预测置信度揭示 token，并列时按位置排序。`generation.steps: null` 表示每步揭示一个 token；指定正整数时实际步数取 min(steps, 答案 token 数)，各步均分揭示数量，余数优先分配给前几步。`temperature: 0.0` 为贪心预测，正数使用任务专属随机流；特殊 token 全部屏蔽。基线与编辑后使用同一任务派生种子。该协议是有目标长度条件的扩散短答案诊断，不能将其数值视为原 UltraEdit 自回归 teacher forcing 的直接复现，也不用于长答案自由生成。

`summary.json` 同时提供 `case_macro`（先源案例内平均，再跨案例平均，主口径）和 `probe_macro`（逐题平均，对应原 UltraEdit 的汇总口径）。分组包括 all、accepted、rejected；缺少 persona/mhop 等题目时标为 not_applicable，分数为 null。独立拒绝案例的最终状态为原始模型，计入全部案例并复用基线预测；连续编辑统一使用 final_model 回测所有已尝试事实，包括当轮拒绝的事实。当前没有逐轮快照，因此不提供连续编辑逐轮曲线。

评估参数与编辑参数完全独立：

| 参数 | 含义 |
| --- | --- |
| `framework` / `protocol` | ultraedit；协议选 diffusion_short_v1 或 diffusion_gold_v1 |
| `run_dir` | 已完成编辑目录，可由 --run-dir 覆盖；命令行路径相对当前目录，配置资源路径相对项目根 |
| `groups` | 选择要评估的问题组，默认全部五组 |
| `device` | cpu、cuda 或 cuda:N，可由 --device 覆盖 |
| `seed`（可选） | 默认继承编辑报告的种子，可显式覆盖评估生成种子；源数据改写选择始终使用原读取种子 |
| `generation.steps` / `generation.temperature` | 评估专用去噪步数和温度 |
| `output_root` / `eval_id` | 独立输出位置；--eval-id 可指定尚未存在的目录名 |

数据读取器、目标答案字段、读取范围和原读取种子从 run_report 恢复；只评估报告 rounds 中实际尝试的事实。源数据主体标注不参与预测，因此评估不要求提供 FEVER 的主体文件。--dataset 仅用于同一文件迁移后的路径覆盖，不改变读取器或文件名构成的案例 ID；--model-path 用于原模型路径迁移。保存目录可以整体移动，评估从当前 run-dir 下的 case_N/final_model 读取产物。

若独立运行全部被拒绝，原流程不保存 model.json；此时需要在评估配置中显式提供本次编辑原有的 target_layers 和 torch_dtype。也可通过 model_config 引用原模型配置（相对评估配置目录），并补充 torch_dtype。例如原编辑确实采用第 0 层和自动加载精度时，填写 `"target_layers": [0], "torch_dtype": "auto"`。存在已保存模型时，以产物 metadata 的层与精度为准。

新协议运行示例：`python -u evaluate.py --run-dir "runs/<run_id>" --config configs/evaluation/ultraedit_gold.json`。两个配置默认 `temperature=0.0`、`steps=null`。新协议两侧各自重新按置信度选位，顺序可能不同；trace.positions 为从 0 开始的答案位置，trace 长度即实际轮数，每轮 positions 长度即选位数量。新协议的完全匹配和前后一致率均比较评分预测，不使用回填后的输入。

每次运行生成 manifest.json、predictions.jsonl、summary.json。manifest 记录源数据/目标权重/评估代码摘要、任务摘要、配置及环境；predictions 保存逐题基线和配对预测、参考标签与分数；summary 保存覆盖率及错误信息。失败即停止并标为 failed，部分指标不可当作完整结果；不静默略过运行错误，不覆盖已有评估目录。无效产物在加载模型前报错。

现有编辑产物没有源数据和原模型权重的编辑时摘要，因此 manifest 会明确注明这两项历史身份未验证，当前摘要只证明评估时版本。当前 LS 属于 benchmark preservation，未验证与编辑保持池独立，不应称为独立保持测试集。原始模型路径迁移必须指向同一份基础权重。

以后可用普通类实现 metrics、build_tasks、score、aggregate，在 `evaluation.framework.FRAMEWORKS` 登记；无需继承基类。当前已实现的推理协议仅服务短答案；长答案任务和新的推理后端需要另行接入，不能通过修改协议名称就启用。

## 增加其他编辑算法

在新模块中实现 `run_dataset(config_file, **options)`，并在 `main.py` 的
`EDITORS` 中登记 `"算法名": "包名.runner"`。主配置中同步选择 `editor` 和 `editor_config`；新算法可用 `util.io.read_config` 读取组合后的配置，入口按需导入对应模块：

```bash
python main.py --editor 算法名 --config configs/对应主配置.json
```

原始算法推导和历史联调记录保留在 `docs/`；其中的阶段文件和旧命令描述对应当时版本。
