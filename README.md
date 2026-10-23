# MDLM 扩散知识编辑项目

本项目按[完整算法逻辑](docs/最终逻辑%20V1.0.md)实现掩码扩散语言模型的事实知识编辑。P0 模型与数据接口、P1 固定状态与监督、P2 的 K/C 提取和统一验收、P3 固定 Q、P4 真实候选前向与 R 联合优化已实现，并取得真实 LLaDA 单案例的技术验收证据。正式数据、保持配方、多事实/多层验收和 P5/P6 编辑闭环仍待完成。开发顺序及进度见[开发进度记录](docs/开发进度/01_开发顺序与进度记录.md)和[待办事项](docs/开发进度/02_开发待办事项.md)，代码审核见[代码审查进度](docs/开发进度/03_代码文件审查进度.md)。

## 基础扩散模型

四个已下载的指令模型位于 `model/models/`；下表链接指向各模型的原始发布页，可查看模型卡与权重文件。首版 P0 适配和接口检查使用 **LLaDA-8B-Instruct**；其余模型尚需单独完成适配与真实前向验收。

| 模型 | 本地目录 | 下载链接 |
| --- | --- | --- |
| Dream-v0-Instruct-7B | `model/models/Dream-v0-Instruct-7B/` | [Dream-org/Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) |
| LLaDA-MoE-7B-A1B-Instruct | `model/models/LLaDA-MoE-7B-A1B-Instruct/` | [inclusionAI/LLaDA-MoE-7B-A1B-Instruct](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct) |
| iLLaDA-8B-Instruct | `model/models/iLLaDA-8B-Instruct/` | [GSAI-ML/iLLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/iLLaDA-8B-Instruct) |
| LLaDA-8B-Instruct | `model/models/LLaDA-8B-Instruct/` | [GSAI-ML/LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) |

## 依赖环境

当前以 **Python 3.10 + LLaDA 8B 的基础适配路径**作为依赖基线。`requirement.txt` 只列现阶段确定的直接依赖：PyTorch 用于模型前向、梯度和矩阵求解；Transformers 用于模型及 Tokenizer 加载；Accelerate 用于大模型加载/设备管理；NumPy、SciPy 用于数据和数值校验；Safetensors 用于模型/矩阵张量读写；tqdm 用于长时间提取任务的进度显示。LLaDA 官方示例指定 `transformers==4.38.2`，所以暂不照搬 EasyEdit 当前的 Transformers 5.x 依赖组合。本机 WSL 环境已通过真实 8B 单案例联调；这份依赖清单尚不代表后续训练和完整实验已验收。

在项目根目录创建 Conda 环境：

```bash
conda create -n mdlm python=3.10 -y
conda activate mdlm
python -m pip install --upgrade pip
```

先按设备安装与 `requirement.txt` **同版**的 PyTorch。以下二选一；CUDA 11.8、12.4 或其他平台请使用 [PyTorch 官方 2.5.1 安装命令](https://docs.pytorch.org/get-started/previous-versions/)选择相应 wheel，不能根据本机 CUDA 版本盲用示例：

```bash
# CPU 环境
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# 或：CUDA 12.1 环境
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

然后安装项目依赖并检查环境：

```bash
python -m pip install -r requirement.txt
python -m pip check
python -c "import torch, transformers, accelerate, numpy, scipy, safetensors, tqdm; print('torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'available:', torch.cuda.is_available()); print('transformers:', transformers.__version__)"
```

`requirement.txt` 中的 `torch==2.5.1` 会沿用先安装的同版 CPU/CUDA wheel；若直接运行 `pip install -r requirement.txt`，pip 按当前包索引选取 wheel，未必符合目标设备。8B 模型的实际可运行性仍取决于显存、模型修订、驱动和后续适配代码。首次 P0 真实前向通过后，应保存 `python -m pip freeze` 输出及模型/Tokenizer 修订，作为实验环境记录。

暂不加入 `datasets`、`peft`、`vllm`、`flash-attn`、外部裁判 SDK、`torchvision` 或 `torchaudio`：当前尚未确定数据读取方式、加速方案和裁判提供方，算法主线也没有图像或音频需求。选定首个数据集和模型后，只添加实际用到的包并核验其版本；更换为其他扩散模型时先验证其 Transformers 版本要求。

## P0 人工案例接口检查

`tests/fixtures/p0_case_example.json` 是 `0.3-proposal` 结构的人工示例，`quality_control.status=example_only`，不能进入正式实验。`split=gold` 表示开发联调；同一案例中 `prompts.edit` 是编辑输入，其他 `used_for_edit=false` 的探针是开发验证题。独立测试案例使用 `split=test`。正式数据的质量和划分应在算法运行前完成审核。

轻量回归检查（不加载下载的 8B 权重）：

```bash
python -m unittest discover -s tests -v
```

2026-10-22 C 审核修复后：使用 WSL `mdlm` 环境（`/home/chi/miniconda3/envs/mdlm/bin/python`），91 项轻量测试全部通过，其中 C 共 44 项（8 项捕获、17 项数学、12 项提取链路、7 项入口）。覆盖公式（5）、极端尺度、近零抵消、两层独立权重、基准绑定、来源改变及保存精度/数值失败。此前本地 LLaDA tokenizer 的人工中文、英文及字节拆分主体对齐通过。轻量检查与真实 8B 技术联调分别记录。

算法运行时直接读取预先审核的数据，不检查文件后缀、空数据集、字段质量、重复 ID、答案字面泄漏或跨划分泄漏。需要检查案例结构、字段用途及重复 `case_id` 时，在项目根目录手动运行独立工具：

```bash
python data/schema.py tests/fixtures/p0_case_example.json
# 正式数据接入后，将路径替换为对应的 .json 或 .jsonl 文件
```

命令通过只表示 P0 基础结构检查通过；正式数据的其他审核在算法运行前独立完成。Tokenizer 对齐仍是构造模型输入的必要计算步骤。

安装并核验真实模型环境后，先运行 tokenizer 对齐，再按显存/内存情况显式选择 `--forward` 执行两次真实前向及目标 FFN 输入重放检查：

```bash
python -m util.smoke_p0_llada tests/fixtures/p0_case_example.json --layer 0
python -m util.smoke_p0_llada tests/fixtures/p0_case_example.json --layer 0 --forward
```

`--layer 0` 仅为 P0 冒烟检查，不代表正式编辑目标层已确定。短答案 mask 数量来自真实 tokenizer 的编辑答案；`generation_profile` 只供后续自由回答使用。模型前向只接收 mask 输入，目标 token 和评价参考不传入模型输入。本地 LLaDA `config.json` 记录 `transformers_version=4.46.3`，而 `requirement.txt` 暂列 `4.38.2`；需以真实加载和前向检查确定可用版本，再冻结环境。

手动运行 `schema.py` 时会拒绝旧答案与新答案别名冲突、编辑问题直接包含目标答案等情况。适配器准备输入时仍会检查答案 token 中的特殊 token 和问题区预置模型 mask，以保证 mask 位置和模型前向计算正确。正式实验不得把 `example_only` 当作正式案例。

## P1 固定去噪状态与监督

`configs/p1_example_only.json` 仅展示配置字段，指向人工构造的 `example_only` 案例和未冻结的模型/层选择，不能作为正式实验配置。正式运行前须替换为独立审核的案例清单，冻结 `baseline_id`、模型与 tokenizer 修订、事实顺序、目标层、采样步数和温度，并完成 P0 真实模型验收。

配置中的 `facts` 每项选择一个 `case_id` 和对应的编辑 `probe_id`；每项计为公式（2）的一个事实。`steps=null` 表示每个答案 token 揭示一步；每次前向只揭示当前 mask 中置信度最高的预定数量，已揭示 token 不再掩码。`temperature=0` 使用确定性贪心 token；正温度使用独立的事实种子采样。采样器只接收 mask 输入，目标答案留在监督侧。

准备好真实运行环境后，在项目根目录运行：

```bash
python -m util.collect_denoise_states configs/p1_run.json
```

一次运行在 `runs/<run_id>/round_<n>/p1/` 写入 `manifest.json`、`trajectories.jsonl`、`collection_report.json`、`fixed_states.jsonl` 和 `coverage_report.json`。原始轨迹保存每次前向前的完整 token、attention mask、剩余 mask、隐式位置及 mask logits 摘要；终态单独保存。重放逐次比对输入、揭示 token 与 logits 摘要。筛选按公式（1）区分兼容、冲突和终态，并按公式（2）（3）给兼容状态的每个剩余 mask 赋权。`coverage_report.json` 的 `formula_checks_passed` 只说明状态覆盖与权重核对通过；真实模型验收仍须查看重放记录及 P0 报告。若采集已完成但筛选尚未生成，可单独运行：

```bash
python -m util.filter_compatible_states runs/my_run/round_0/p1
```

当前产物版本是 `p1.1`：筛选入口核对配置、轨迹与采集报告摘要及事实/探针顺序，避免误用其他运行的产物。logits 摘要保留原始精度，重放按配置种子抽样。采集或筛选失败会记录原因；监督核验失败时不会生成固定监督文件。重新运行使用新的 `run_id`，旧版调试产物需要重新采集。这些是算法状态和产物核对，不调用数据集质量检查。

## 单条构造案例的 P0/P1 真实模型联调

在项目根目录、已安装的 `mdlm` 环境中，一条命令依次执行 P0 全 mask 前向、`ff_out` 输入捕获与重放，以及 P1 自主去噪、磁盘轨迹重放、兼容状态筛选和公式（1）～（3）的权重检查：

```bash
python -m util.accept_p0_p1_single_case --layer 0
```

`--layer 0` 是本次明确选择的联调层；改变目标层时显式修改该参数。命令先对本地模型权重分片和 tokenizer 文件计算 SHA-256，再分别加载 8B 模型运行 P0 和 P1，可能耗时且需要足够内存/显存。每次自动创建独立的 `runs/single_case_*/round_0/`，保存 `model_identity.json`、`environment.json`、`run_config.json`、`p0_report.json`、`p1/` 下的轨迹和覆盖报告，以及总入口 `acceptance_report.json`。失败会在总报告中留下阶段与原因；重新执行会生成新的运行目录。

默认案例仍是 `tests/fixtures/p0_case_example.json`，用途为 `example_only`。总报告中的 `passed` 只表示**这一条构造案例的技术联调通过**，不能代表正式数据集验收、跨事实权重验证或独立测试结果。2026-10-22 已在本机真实执行通过，证据位于下方 P2 联调运行的 `acceptance_report.json`。

## P2 参考键矩阵 K：公式（4）

`util/extract_reference_keys.py` 直接重放 P1 保存的全 mask 初态，在每个目标 `ff_out` 线性层的输入端捕获主体末 token 激活，按冻结事实顺序逐列组成 `K`，形状为 `d_in × N`。提取期间不重新读取案例、不重分词、不采样，也不更新模型权重。

轻量检查使用两个事实、两个不同输入维度的微型 CPU 层，覆盖输入捕获、列序、基准、非全 mask 初态、报告计数一致性、存储溢出、保存/读回失败及提取期间来源变化；另用模拟子进程检查入口编排、失败中止和已有运行保护。这条命令不加载真实 8B：

```bash
python -m unittest tests.test_reference_keys tests.test_p2_keys_smoke -v
```

一条命令运行本地真实 LLaDA 的单案例 P0/P1 与 K 提取联调：

```bash
python -m util.smoke_p2_keys --layer 0
```

命令自动创建 `runs/example_only_p2_keys_*/round_0/`，先运行已有 P0/P1 联调，再加载同一基准提取 K。对当前三 token 答案，总计加载模型 3 次、执行 10 次前向，其中 K 提取及重复核对占 2 次。总报告是 `p2_keys_acceptance_report.json`；案例仍是 `example_only`。

如果已经有完整 P1 产物，可单独执行提取：

```bash
python -m util.extract_reference_keys runs/your_run/round_0/p1 --config configs/p2_keys_example_only.json
```

`configs/p2_keys_example_only.json` 冻结存储精度 `float32` 与捕获重放容差 `atol=0.01, rtol=0.001`；存储转换不提高原始激活精度。P2 复算本地权重、模型实现和 tokenizer 的文件摘要，要求与 P1 配置中的 `local-files-sha256:...` 修订一致，并逐事实核对初态的原始 logits 摘要。`local_unverified` 等占位修订不能用于 K 提取，应先用上面的单案例入口生成冻结配置和 P1 产物。输入数据质量检查仍由独立工具负责；提取器只核对算法产物。

K 产物写入同轮的 `matrices/reference_keys/`：`keys.safetensors` 以完整模块名为张量键；`manifest.json` 记录基准、P1 来源摘要、事实列序、主体位置、层维度、捕获/存储 dtype 和文件摘要；`qa_report.json` 记录初态 logits 一致性、每列重复捕获差值及磁盘往返结果。仅 `manifest.status=complete` 的产物可用；已有成功或失败目录均不覆盖。

2026-10-22 已完成 K 的代码审核：公式（4）的计算逻辑符合方案；修复 P1 覆盖报告的重放事实数、logits 重放数和示例数未与采集报告交叉核对的问题。修复不改变 K 数值计算，也不新增输入数据质量校验。已有真实 P1 产物通过新检查；本次审核未重跑 8B，沿用下方已保存的单案例实机证据。

K、下节的 C、P2 统一验收、P3 固定 Q 和 P4 残差联合优化已实现；P5/P6 尚未实现。

2026-10-22 实机证据：`runs/example_only_p2_keys_20261022_review2/round_0/p2_keys_acceptance_report.json` 为 `passed`。第 0 层 K 为 `[12288, 1]`，原始捕获为 bfloat16、存储为 float32；初态 logits 与 P1 一致，重复捕获最大绝对差为 0，文件摘要与保存读取通过。P1 的 3 个前向前状态中保留 1 个兼容初态、诊断记录 2 个冲突状态，权重和为 1。首次运行 `review1` 的挂载盘重命名后即时读取失败已修复，失败报告原样保留。`runs/` 由 `.gitignore` 排除，仅本机保留实验产物。

## P2 保持矩阵 C：公式（5）

`util/extract_keep_statistics.py` 在与 K 相同的冻结基准和目标 `ff_out` **输入端**捕获保持向量，调用 `MDLM/keep_statistics.py` 累加：

\[
C_\ell=\sum_m\omega_{\ell,m}u_{\ell,m}u_{\ell,m}^{\top},\qquad\omega_{\ell,m}\geq0.
\]

数值上以等价的 `(sqrt(ω)u)(sqrt(ω)u)ᵀ` 累计，避免先平方或先转换权重精度造成中间溢出/下溢。不去均值、不除样本数、不强制权重和为 1，也不添加正则项。零权重、低秩和全零 C 都合法。C 在后续公式（6）～（9）中约束写回对保持方向的影响，本阶段不求 Q、不更新权重。

### 输入与配方

正式保持语料、mask 上下文、捕获位置和权重尺度尚未冻结。当前提供独立的 `tests/fixtures/p2_keep_example_only.json`：两条人工保持文本，分别追加 2/3 个 mask，捕获模板末 token 和首个 mask；显式权重为 `[0.5,1.5]` 和 `[0,2]`，总量 4。此配方仅供技术联调，不是正式统计策略；不读取编辑答案、开发题或独立测试题，也不从案例的 locality 自动选取保持语料。

`smoke_p2_keep.py` 将该示例一次性编码为 `p2.keep_inputs.1` 算法产物：保存完整 token IDs、attention mask、隐式位置、mask 位置、捕获位置、逐层逐位置权重、Tokenizer 修订及配方标识。它与原有 `0.3-proposal` 案例 JSON 分开，不改变案例字段。正式接入时提供相同的冻结保持输入产物和对应 `KeepStatisticsConfig`，由实验者提前审核数据用途；运行时不调用 `schema.py` 或扫描数据泄漏。

### 启动命令

激活 `mdlm` 后，在项目根目录一条命令新建独立运行，依次执行真实 P0/P1/K/C 和矩阵包验收：

```bash
python -m util.smoke_p2_keep --layer 0
```

该入口依次执行 **P0 → P1 → P2-K → P2-C → P2-QA**，全部调用本地 LLaDA-8B；任一阶段失败立即停止。`p2_keep_acceptance_report.json` 汇总各阶段结果。P2-QA 重新加载模型、重放 P1 初态及保持输入，再逐层复算 C 的 trace、小坐标外积和独立投影二阶矩；Q 求解及知识编辑优化仍待开发。

当前单层示例共加载模型 5 次、执行 18 次前向：上游 P0/P1/K 为 10 次，C 为 5 次，P2-QA 为 1 次 P1 初态和 2 次保持输入重放。第 0 层仅为示例选择。若已有完整 P1/K，可复用该轮提取 C 并验收：

```bash
python -m util.smoke_p2_keep --p1-dir runs/your_run/round_0/p1
```

已有 C 或失败目录不会覆盖，重复实验请使用新运行目录。正式冻结输入的提取命令为：

```bash
python -m util.extract_keep_statistics runs/your_run/round_0/p1 runs/your_run/round_0/keep_inputs.json --config configs/your_keep_config.json
```

已有同轮 P1/K/C 且尚无 `matrices/validation/` 时，可单独验收，无须重新提取矩阵：

```bash
python -m util.validate_matrix_artifacts runs/your_run/round_0/p1
```

仅运行 P2-QA 的轻量测试，不加载 8B：

```bash
python -m unittest tests.test_matrix_artifacts tests.test_p2_keep_smoke -v
```

### 产物与验证范围

`matrices/keep_statistics/` 保存逐层 `c_layer_<n>.safetensors`、冻结 `keep_inputs.json`、`manifest.json` 和 `qa_report.json`。清单绑定 P1/K 文件摘要、同一基准、目标层及完整配方，记录来源分组、向量/正权重/零权重计数、权重总量、精度和张量摘要。提取结束重查源产物；失败保留报告，仅 `status=complete` 的清单可用。

按层在 CPU 累积，默认 float64 累积、float32 保存，不保留全量激活；第 0 层仅累计方阵约 1.125 GiB、保存精度副本约 0.563 GiB，另需模型、前向和分块诊断内存。QA 核对有限性、分块对称性、对角/trace、独立坐标块外积、随机投影二阶矩/半正定及磁盘逐块读回。小坐标误差界采用独立对角几何尺度，投影误差/PSD 容差采用原二阶矩 trace 尺度；配置 `atol/rtol` 不变，报告记录尺度、误差界和归一误差，避免近零抵消误拒。只对小投影求谱，报告 `full_spectrum_checked=false`，不宣称完成全维特征值检查。

本轮 C 代码审核已完成：修复极端尺度的中间溢出/下溢、近零项误拒及失败报告非有限值；原始示例配方在分词前后核对摘要，避免旧输入绑定新来源；磁盘读回显式核对 dtype 和数值，防止清单精度与文件不符。新增 12 项回归，先复现缺陷再修复。公式（5）、非负权重及总尺度不变，不恢复数据集运行时质量检查。本次只运行 CPU 轻量回归，未重跑 8B；下述实机记录保留为此前示例证据。

2026-10-22 已复用上述真实 P1/K 运行执行 C。`runs/example_only_p2_keys_20261022_review2/round_0/p2_keep_acceptance_report.json` 为 `passed`：第 0 层 C 为 `[12288,12288]`，捕获 bfloat16、累计 float64、保存 float32；4 个保持位置中 3 个正权重、1 个零权重，总量 4。共 5 次真实前向，P1 初态 logits 一致，保持输入重复捕获差值均为 0；对称差为 0，trace 为约 12.093753，数值诊断及磁盘往返全部通过。

2026-10-23 对 `runs/example_only_p2_keep_20261023T063109Z_9628d2ee/round_0/` 的真实 P1/K/C 产物执行 P2-03，`matrices/validation/qa_report.json` 为 `complete`。额外 3 次真实前向通过 K 初态和两条保持输入重放；第 0 层 C 的对称差为 0，独立外积块最大差约 `7.28e-12`，独立投影二阶矩最大差约 `3.62e-11`。验收报告明确 `example_only=true`、`p2_stage_complete=false`、`full_spectrum_checked=false`。正式保持配方、多事实/多层运行与正式验收仍待完成。

## P3 固定写回因子 Q：公式（6）～（9）

`MDLM/writeback.py` 按冻结的逐层正则系数构造 `H = C + K Kᵀ + λ I`，用 Cholesky 解 `H X = K`，返回固定的 `Q = Xᵀ`。Q 的形状是 `[事实数, FFN 输入维度]`。不显式求逆、不修改 C 的尺度、不自动加 jitter 或增大 λ；分解失败、非有限值或残差超限均停止并保留诊断。

`WritebackConfig` 冻结各层 λ、计算/保存精度、设备和容差。示例配置 `configs/p3_writeback_example_only.json` 仅对应第 0 层，λ 为 `0.01`，CPU float64 求解、float32 保存；这是技术联调选择，正式实验需提前冻结自己的参数。计算结果和保存精度结果均用原始 K/C 核对 `HX=K` 与 `QH=Kᵀ` 的相对残差，使用相同的冻结容差；缩放范数及分块 hypot 避免平方和上溢/下溢，保存后再核对 dtype 与数值。

激活 `mdlm` 后，已有同轮 P1/K/C 和完整 `matrices/validation/` 时，用一条命令完成 P3 求解与验收：

```bash
python -u -m util.smoke_p3_writeback --p1-dir runs/your_run/round_0/p1
```

自定义逐层配置的直接入口为：

```bash
python -u -m util.solve_writeback_factor runs/your_run/round_0/p1 --config configs/your_p3_config.json
```

P3 只读取已验收的状态和矩阵产物，不加载 8B 权重，不执行模型前向。`ridge_by_layer` 必须与 P2 目标层完整对应；示例矩阵需要配置明确 `allow_example_only=true`。正式输入要求 P2 阶段完成；示例输出仍保持 `example_only=true`、`p3_stage_complete=false`。已有成功或失败目录不覆盖，重复实验应在新运行目录执行。

产物位于 `matrices/writeback_factors/`：逐层 `q_layer_<n>.safetensors`、版本 `p3.writeback_factor.2` 的 `manifest.json` 和 `qa_report.json`。记录同轮来源摘要、事实顺序、模型基准、λ、dtype、H 尺度、Cholesky 诊断、计算/保存后的线性系统与转置正规方程残差、耗时、进程内存峰值及失败原因。Q 在 P4 中固定，P3 不优化 R。版本 1 的历史产物保持原样，其报告没有新增的转置正规方程证据；P4 会用原始 K/C 重新核对旧 Q 的双侧保存残差。

2026-10-23 复用 `runs/example_only_p2_keep_20261023T063109Z_9628d2ee/round_0/` 的真实 P2 产物，P3 报告 `p3_writeback_acceptance_report.json` 为 passed。第 0 层 Q 为 `[1,12288]`，λ=`0.01`，求解相对残差约 `8.78e-16`，float32 保存后约 `1.81e-9`，磁盘往返一致。16 CPU 线程下求解约 3.37 秒，进程峰值约 4.53 GiB；该资源记录仅代表本机本次运行。结果仍为人工案例技术验收。

P3 本轮代码审核修复了近似对称 C 仅通过 `HX=K` 却违反 `QH=Kᵀ`、极小范数平方和下溢导致误接收，以及 H 行视图延长大矩阵存储生命周期的问题；不改变 λ、公式或原有容差。新增 8 项回归，P3 共 32 项、全套 132 项通过。对同一真实 K/C 重算后，Q 与旧文件逐元素及 dtype 一致；额外正规方程残差约 `8.80e-16`，float32 后约 `1.81e-9`。独立审核报告 `round_0/p3_writeback_review_report.json` 为 passed，原产物保留；本轮进程峰值约 3.40 GiB，该资源比较仅适用于本机本次记录。

轻量测试命令：

```bash
python -m unittest tests.test_writeback_math tests.test_writeback_pipeline -v
```

阅读顺序：`WritebackConfig` 与示例配置 → `solve_writeback_factor/relative_residual` → 数学测试 → `load_matrix_bundle/solve_run` → P3 冒烟入口与流水线测试。小例验证 `RQ` 满足公式（7）及公式（6）的唯一最小值；`QK` 一般不等于单位阵。

## P4 真实候选前向与多层 R 联合优化：公式（10）～（12）

`MDLM/candidate.py` 冻结基准模型参数，在每个选定线性层的全部实际调用位置加入 `R(Qk)`，后层 k 由同一次候选网络实时计算。Q 使用 P3 保存精度，`residual_dtype` 必须与其一致；R 从零开始。`forward_logits` 在 eval 模式下保留梯度；P0/P2 捕获接口仍用于无梯度重放。

`MDLM/optimize.py` 直接使用 P1 的 `1/(N·|T_i|·|M_i,t|)` 全局位置权重，优化 `J=F+βL_keep+Σρ||R||²/2`。每个完整步逐微批累积数据梯度，正则仅求导一次；Adam 不带 weight decay，也不自动更改预算或正则系数。只在冻结检查点中选择有限、非零且 `J<J(0)` 的最佳候选，保存同一步的 R 和优化器状态。无合格候选则输出 `rejected/no_candidate`，不发布 R。

已有同轮 P1、P2-QA 和 P3 complete 产物、且该轮尚无 P4 输出时执行：

```bash
python -u -m util.smoke_p4_candidate --p1-dir runs/your_new_run/round_0/p1
# 多层或正式配置：rho_by_layer 必须完整对应 P3 层名
python -u -m util.optimize_residuals runs/your_new_run/round_0/p1 --config configs/your_p4_config.json
```

以上两条是同一任务的两种入口，选择其一。示例 `configs/p4_example_only.json` 选择第 0 层、ρ=0.001、β=0、20 步、检查点 1/5/10/20；这些参数只用于技术联调。运行会真实加载 LLaDA 并执行反向传播，需要模型层常驻 CPU/GPU；推理专用 CPU/disk offload 会被明确拒绝，避免其 `no_grad` 包装切断跨层梯度。保持训练关闭时不加载 theta_0；P5 的开发保持门槛仍须执行。

启用保持训练时，设置 β>0，并显式配置 `keep_train_file`、`reference_model_path`、`reference_model_revision` 和 `reference_tokenizer_revision`。参考必须是最初的 theta_0，不能以本轮 theta_b 自动替代。保持训练文件使用独立的冻结 token 合同：

```text
artifact_version: "p4.keep_training.1"
purpose: "keep_train"
example_only: true/false
inputs: [{input: {input_ids, attention_mask, mask_positions}, positions, weights}, ...]
```

`positions/weights` 逐项对应，权重非负且全局和为 1；token IDs 属于冻结 Tokenizer，不含编辑目标监督字段。训练、开发、独立测试用途由运行前独立审核确定，算法不执行数据质量或跨划分泄漏扫描。代码只读取指定训练输入，用 theta_0 计算完整词表参考 log-probabilities 并保存，然后释放参考模型，再加载候选模型；KL 方向为 `theta_0 || candidate`。原案例 JSON 结构不变，不自动复用 locality 或 C 配方作为训练/开发输入。

输出在同轮 `residuals/`：当前版本 `p4.residual_candidate.2` 的 `manifest.json`、`optimization_report.json`、合格候选的 `residuals.safetensors` 和 `optimizer_state.pt`；β>0 时另有 `theta0_keep_log_probs.safetensors`。报告记录来源、事实/层序、有效监督数、各项损失、梯度与 R/RQ 范数、资源、保存读回及失败阶段。R 和 Adam 的步数/矩/参数组均须读回一致；模型、Tokenizer、theta_0 文件及参考缓存结束时复查原始摘要。入口总报告为 `p4_candidate_acceptance_report.json`。旧成功/失败目录均不覆盖；重复同轮实验可在直接入口指定新的 `--output-dir`，原版本 1 产物保留。P4 不合并权重。

2026-10-23 真实单案例证据：`runs/example_only_p2_keep_20261023T063109Z_9628d2ee/round_0/p4_candidate_acceptance_report.json` 为 passed。LLaDA-8B-Instruct 第 0 层、1 个兼容状态、3 个监督位置，β=0，20 步后 `J: 14.219749 → 11.173778`，选中第 20 步，R 为 `[4096,1]`，非零有限，基准和 Q 未改变、磁盘读回一致。本次加载模型 1 次，28 次前向、20 次反向；CUDA 分配峰值约 15.06 GiB，仅代表本机短输入实验。示例继续标记 `p4_stage_complete=false`，不表示目标事实生成正确、P5 接受或正式实验有效。真实多事实/多层和 β>0 训练路径的 8B 验收仍待补。

```bash
python -m unittest tests.test_candidate tests.test_optimize tests.test_residual_pipeline -v
python -m unittest discover -s tests
```

P4 开发时新增 35 项，全套 167 项通过；本轮专项审核再新增 17 项，P4 共 52 项（12 候选、18 优化、22 产物），默认 16 CPU 线程下全套 **184 项通过**。审核修复目标权重与未选中参数的共享遗漏、复合推理 hook 与数据梯度断链、极端尺度正则/范数失真、模型/参考来源复查和优化器状态读回遗漏，并使候选清理可重复调用、逐模块恢复模式。正则以 float64 的等价 `||sqrt(ρ/2)R||²` 计算，范数复用 P3 的稳定缩放实现；系数、三级权重、KL 方向、公式和优化预算不变。关键反例先复现再修复，覆盖单精度下溢、极端正则值及梯度、被正则掩盖的断链、来源/缓存/配置变化、损坏的优化器状态和失败入口。

审核后真实复验另存于同轮 `residuals_review_v2/`，独立 `p4_candidate_review_report.json` 为 passed：同一配置 20 步后 `J:14.2197494507→11.1737776872`，基准/Q 未变、优化器读回通过。R 与旧文件最大差 `2.98e-8`，编辑损失一致；原版本 1 产物原样保留。此结果仍为构造案例技术验收。对同轮产物复验时，可用新目录：

```bash
python -u -m util.optimize_residuals runs/your_run/round_0/p1 --config configs/p4_example_only.json --output-dir runs/your_run/round_0/residuals_review_v2
```

此前额外强制单线程运行时，既有 C 的 float32 大批数值 QA 测试出现线程相关差异，默认线程复核通过；此限制仍待专项数值核验，本轮未放宽容差或修改 C 算法。

依赖选择参考：[LLaDA 官方仓库](https://github.com/ML-GSAI/LLaDA/blob/main/README.md)、[PyTorch 官方历史版本安装页](https://docs.pytorch.org/get-started/previous-versions/)、[EasyEdit 当前依赖清单](https://github.com/zjunlp/EasyEdit/blob/main/requirements.txt)、[Safetensors 文档](https://huggingface.co/docs/safetensors/index)。

