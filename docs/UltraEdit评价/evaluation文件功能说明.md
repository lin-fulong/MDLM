# evaluation 文件夹各文件的作用

evaluation 是 MDLM 编辑完成后的独立评估模块。它读取 run_report.json、原数据和保存的目标层权重，生成短答案并汇总 ES、GS、LS 等指标；MDLM 编辑算法不依赖这个文件夹。

## 各文件职责

| 文件 | 主要作用 | 想了解什么时读它 |
| --- | --- | --- |
| types.py | 定义任务 EvalTask、生成请求 GenerationRequest、预测 Prediction，以及未来框架实现的 EvaluationFramework 接口。 | 先弄清各层之间传递什么。 |
| registry.py | 把框架名称 ultraedit 映射到 UltraEditFramework。 | 想增加自定义评估框架。 |
| frameworks/ultraedit.py | 从已尝试的事实及五类题组建立任务；把 edit、generalization、preservation、persona、portability 映射到 ES、GS、LS、person_score、mhop_score；调用逐题评分。 | 想核对“哪个问题对应哪个指标”。 |
| artifacts.py | 校验已完成的编辑报告、案例和最终权重；按编辑时读取配置重新读取数据，确定独立或连续模式该用哪个权重。 | 想知道评估数据和模型从哪里来。 |
| inference.py | 将题目转换为 LLaDA 输入；按标准答案长度建立全 mask 窗口并迭代生成；加载和恢复目标层权重。 | 想理解扩散模型怎么产生短答案。 |
| metrics.py | 比较生成 token 与标准答案，计算逐题 token 正确率、完全匹配率和前后输出一致率。 | 想核对单题公式。 |
| reporting.py | 写逐题记录，并按源案例、按题目两种口径求平均，区分 all、accepted、rejected。 | 想理解 summary.json 中的数值。 |
| runner.py | 串起配置、任务、原模型预测、编辑后预测、评分和三个输出文件；检查运行期间输入未变化。 | 想看一次评估的完整流程。 |
| __init__.py、frameworks/__init__.py | Python 包初始化文件，只有简短说明，没有指标计算。 | 通常无需单独阅读。 |

仓库根目录的 evaluate.py 是命令行入口，configs/evaluation/ultraedit.json 是默认配置；它们不在 evaluation 文件夹里。已有的数据读取器位于 data/readers，evaluation 复用其结果。

最短阅读顺序是 frameworks/ultraedit.py → inference.py → metrics.py → reporting.py → runner.py。需要追查编辑产物时再读 artifacts.py；需要扩展框架时再读 types.py 和 registry.py。

完整数学过程见 [MDLM 接入后 UltraEdit 指标的数学逻辑与实现过程](MDLM接入后UltraEdit评估指标计算逻辑.md)。
