# MDLM 扩散知识编辑项目

本项目按[完整算法逻辑](docs/最终逻辑%20V1.0.md)实现掩码扩散语言模型的事实知识编辑。P0 数据契约与 LLaDA 适配代码、P1 固定状态与监督代码已编写并完成本轮代码审查；真实模型前向和 P1 状态产物仍待验收。开发顺序及进度见[开发进度记录](docs/开发进度/01_开发顺序与进度记录.md)和[待办事项](docs/开发进度/02_开发待办事项.md)，修复记录见[代码审查进度](docs/开发进度/03_代码文件审查进度.md)。

## 基础扩散模型

四个已下载的指令模型位于 `model/models/`；下表链接指向各模型的原始发布页，可查看模型卡与权重文件。首版 P0 适配和接口检查使用 **LLaDA-8B-Instruct**；其余模型尚需单独完成适配与真实前向验收。

| 模型 | 本地目录 | 下载链接 |
| --- | --- | --- |
| Dream-v0-Instruct-7B | `model/models/Dream-v0-Instruct-7B/` | [Dream-org/Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) |
| LLaDA-MoE-7B-A1B-Instruct | `model/models/LLaDA-MoE-7B-A1B-Instruct/` | [inclusionAI/LLaDA-MoE-7B-A1B-Instruct](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct) |
| iLLaDA-8B-Instruct | `model/models/iLLaDA-8B-Instruct/` | [GSAI-ML/iLLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/iLLaDA-8B-Instruct) |
| LLaDA-8B-Instruct | `model/models/LLaDA-8B-Instruct/` | [GSAI-ML/LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) |

## 依赖环境

当前以 **Python 3.10 + LLaDA 8B 的基础适配路径**作为依赖基线。`requirement.txt` 只列现阶段确定的直接依赖：PyTorch 用于模型前向、梯度和矩阵求解；Transformers 用于模型及 Tokenizer 加载；Accelerate 用于大模型加载/设备管理；NumPy、SciPy 用于数据和数值校验；Safetensors 用于模型/矩阵张量读写；tqdm 用于长时间提取任务的进度显示。LLaDA 官方示例指定 `transformers==4.38.2`，所以暂不照搬 EasyEdit 当前的 Transformers 5.x 依赖组合。目标模型、驱动与硬件尚未确定，当前版本是**待实机验证的开发基线**，不是已通过完整模型实验的环境锁文件。

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

2026-10-22 本轮检查：使用 WSL `mdlm` 环境（`/home/chi/miniconda3/envs/mdlm/bin/python`），25 项测试全部通过，包含 4 项小型 CPU 适配器测试。另外，本地 LLaDA tokenizer 的人工中文、英文及字节拆分主体对齐通过。该结果与真实 8B 前向验收分别记录。

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

本轮执行了语法检查、人工轨迹逻辑测试、小型 CPU 模型测试及本地 tokenizer 检查；没有加载 8B 权重或运行 GPU 实验，尚无真实 P1 运行产物。

依赖选择参考：[LLaDA 官方仓库](https://github.com/ML-GSAI/LLaDA/blob/main/README.md)、[PyTorch 官方历史版本安装页](https://docs.pytorch.org/get-started/previous-versions/)、[EasyEdit 当前依赖清单](https://github.com/zjunlp/EasyEdit/blob/main/requirements.txt)、[Safetensors 文档](https://huggingface.co/docs/safetensors/index)。

