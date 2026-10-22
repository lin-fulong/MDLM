# MDLM 扩散知识编辑项目

本项目按[完整算法逻辑](docs/最终逻辑%20V1.0.md)实现掩码扩散语言模型的事实知识编辑。已开始实现 P0 数据契约与 LLaDA 适配接口，真实模型前向仍待验收；开发顺序及进度见[开发进度记录](docs/开发进度/01_开发顺序与进度记录.md)和[待办事项](docs/开发进度/02_开发待办事项.md)。

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

不依赖模型环境的接口检查：

```bash
python -m unittest discover -s tests -v
```

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

依赖选择参考：[LLaDA 官方仓库](https://github.com/ML-GSAI/LLaDA/blob/main/README.md)、[PyTorch 官方历史版本安装页](https://docs.pytorch.org/get-started/previous-versions/)、[EasyEdit 当前依赖清单](https://github.com/zjunlp/EasyEdit/blob/main/requirements.txt)、[Safetensors 文档](https://huggingface.co/docs/safetensors/index)。

