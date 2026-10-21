# MDLM 扩散知识编辑项目

本项目按[完整算法逻辑](docs/最终逻辑%20V1.0.md)实现掩码扩散语言模型的事实知识编辑。当前只有目录和空代码文件；开发顺序及进度见[开发进度记录](docs/开发进度/01_开发顺序与进度记录.md)和[待办事项](docs/开发进度/02_开发待办事项.md)。

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

依赖选择参考：[LLaDA 官方仓库](https://github.com/ML-GSAI/LLaDA/blob/main/README.md)、[PyTorch 官方历史版本安装页](https://docs.pytorch.org/get-started/previous-versions/)、[EasyEdit 当前依赖清单](https://github.com/zjunlp/EasyEdit/blob/main/requirements.txt)、[Safetensors 文档](https://huggingface.co/docs/safetensors/index)。

