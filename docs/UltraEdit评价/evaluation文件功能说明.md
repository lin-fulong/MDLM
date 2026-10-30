# evaluation 文件、函数与数学逻辑说明

本文按当前代码说明文件职责、每个手写函数的作用及对应公式。公式不编号；未涉及数学计算的函数明确标为读取、校验或保存操作。

建议先阅读“与既定方案是否对齐”，再看文件和函数表；需要理解一次预测时，阅读后面的数学逻辑。

## 与既定方案是否对齐

对照 [标准答案回填评价方案](MDLM置信度顺序标准答案回填评价方案.md) 和 [原自主去噪协议](MDLM接入后UltraEdit评估指标计算逻辑.md)，当前代码在有可评价任务时，预测、回填及统计定义一致。

| 方案要求 | 当前代码对应实现 | 核对结果 |
| --- | --- | --- |
| 问题按模型聊天模板渲染，答案 token 与提示边界稳定 | inference.prepare_request | 一致，不能稳定对齐时停止，不截断或改写答案 |
| 初态为问题加 L 个 mask，标准答案位置评分前不输入其标签 | inference.generate | 一致；旧协议不接收 target，新协议单独接收 target |
| 先预测全部剩余位置，按候选 token 置信度选位 | generate 的前向、采样及 selected 构造 | 一致，选位不使用标签概率或对错 |
| 温度为 0 时 argmax，正温度时独立随机流抽样 | generate | 一致，置信度使用对应分布中候选 token 的概率 |
| 每轮数量均分，余数分给前几轮，置信度并列按位置排序 | generate | 一致，默认 K=L，不强制从左到右 |
| 新协议先记录评分，再统一回填 | generate 中先更新 scored_predictions/correct/trace，再修改 state | 一致，每轮共用同一次前向 |
| 每个位置只评分一次，错误不能被覆盖 | 每轮从 remaining 移除所选位置，评分数组与 state 分离 | 一致，返回评分预测而非最终回填输入 |
| 单题分数为正确 token 比例 | framework.score → reporting.paired_scores → token_accuracy | 一致，最终重比保存的预测与求 correct 均值等价 |
| 两侧各自选位，按独立或连续模式选择模型 | runner.run_evaluation、artifacts.read_run | 一致，两侧不强制走相同路径 |
| 五类指标、两种平均、接受/拒绝子集 | framework.metrics、reporting.ScoreSummary | 一致，逐题平均不等于逐 token 混合平均 |
| 缺少可选组时报告无适用题 | build_tasks 与 ScoreSummary.export | 有其他可评价题时输出 not_applicable/null；所有所选组均无题时直接报错，不创建全空评价报告 |

最后一项是需要说明的运行边界，不是评分公式差异。方案文档中的“缺少可选组”须结合“存在至少一个评价任务”理解，当前行为保持不变。

这里的“对齐”指当前两个协议各自与项目文档一致，不表示新协议与原版 UltraEdit 完全等价。原版使用正确左前缀；新协议可提前提供右侧正确 token。它们共用单题 token 正确率，预测条件不同。

## 文件职责和完整调用链

~~~text
MDLM 编辑结束，保存 run_report.json 和目标层最终权重
    ↓
evaluate.py：解析命令行
    ↓
runner.run_evaluation：读取配置并组织 before/after
    ├─ artifacts：读取报告、恢复数据读取配置、定位模型产物
    ├─ framework：从实际尝试的事实构造题目，指定指标和评分
    ├─ inference：渲染/分词、加载权重、按协议推理
    └─ reporting：计算单题分数、累计平均、写逐题记录
    ↓
manifest.json / predictions.jsonl / summary.json
~~~

| 文件 | 职责 | 是否实现评价公式 |
| --- | --- | --- |
| __init__.py | Python 包说明，不加载模型，不执行算法 | 否，没有函数 |
| artifacts.py | 确定数据、模型和每轮 after 权重来源，校验产物及文件摘要 | 只决定使用哪个模型，不计算预测分数 |
| framework.py | 定义任务；构造五类问题；连接评分、汇总及框架注册 | 指标组映射，调用单题评分 |
| inference.py | 定义请求/结果，分词，执行两种协议及加载/恢复目标权重 | 初态、预测分布、置信度、选位、回填和逐位置评分 |
| reporting.py | 单题评分、诊断量、两种平均、覆盖状态和逐题保存 | token 正确率、完全匹配、一致率、宏平均与差值 |
| runner.py | 串起全部步骤，保证 before/after 使用对应权重与同一请求定义 | 接受率、覆盖率；调度评分而不另写评分公式 |

六个文件职责不同，保留当前划分。MDLM 编辑过程不依赖 evaluation；评价只加载已经保存的结果，不重新采样编辑状态、优化残差或进行权重合并验收。

## artifacts.py：数据和模型从哪里来

| 函数 | 输入与输出、作用 | 数学对应 |
| --- | --- | --- |
| file_digest(path) | 分块读取文件，返回 SHA-256 摘要，用来检查评价期间输入是否变化 | 文件校验，无评价公式 |
| read_run(run_dir) | 读取完整的 run_report，核对轮次及案例/事实计数，返回 report、带 checkpoint 的 rounds 和产物摘要表 | 为 before/after 确定模型状态 |
| read_run.checkpoint(reference, expected) | 内部函数：在当前运行目录下定位 case_N/final_model，核对引用，读取 model.json 和权重摘要 | 产物定位，无评分公式 |
| read_evaluation_data(report, dataset_override) | 恢复原读取器、限量、参数和选题种子，调用 data.dataset.load_dataset，返回 DatasetBundle；主体标注不影响评价题目，不要求重建编辑保持池 | 决定题目集合，不生成或修改标签 |
| model_settings(report, artifacts, config, override) | 从产物确定原始模型、目标层和加载精度，校验产物一致性；全部独立拒绝而无产物时要求配置补充层与精度 | 模型加载配置，无评分公式 |

before 一律使用原模型。after 的数学含义为：

$$
\theta_{\mathrm{after},c}=
\begin{cases}
\theta_c,&\text{independent 且接受},\\
\theta_0,&\text{independent 且拒绝},\\
\theta_{\mathrm{final}},&\text{sequential，不论该轮接受或拒绝}.
\end{cases}
$$

独立拒绝案例直接复用 before 的预测和轨迹；连续模式的拒绝事实仍在最终模型上重新评价。

## framework.py：评价哪些题目

EvalTask 是不可变数据容器，保存任务 ID、源案例 ID、事实 ID、探针 ID、问题组、问题、标准答案、接受状态和 after 权重路径。它没有手写计算方法；dataclass 自动生成的初始化等方法只处理数据字段。

| 函数 | 作用 | 数学对应 |
| --- | --- | --- |
| UltraEditFramework.build_tasks(cases, rounds, groups) | 只展开报告中实际尝试的事实；校验题目、来源和重复任务，返回 EvalTask 列表 | 构造各组题集 Q_g，决定后续分母 |
| UltraEditFramework.score(before, after, target) | 将两侧 Prediction.token_ids 交给统一配对评分，不读取回填后的输入 | 单题正确率、完全匹配及前后一致率 |
| UltraEditFramework.aggregate(tasks, groups) | 以所选题组和指标名称创建 ScoreSummary | 指定汇总对象，不执行推理 |
| get_framework(name) | 从 FRAMEWORKS 创建对应框架，未知名称报错 | 框架选择，无数学公式 |

metrics 是题组到结果名的固定映射：

| 题组 | 结果名 | 标准答案 |
| --- | --- | --- |
| edit | ES | 新事实答案 |
| generalization | GS | 改写问题对应的新事实答案 |
| preservation | LS | 无关题自己的标准答案 |
| persona | person_score | 角色化问题对应的新事实答案 |
| portability | mhop_score | 多跳题自己的答案 |

各组使用相同单题公式；LS 没有额外的 KL 或前后相等判定。自定义框架可提供同名方法及 metrics，并登记到 FRAMEWORKS，无需增加基类层次。

## inference.py：如何预测、评分和回填

两个数据容器负责明确接口：

- GenerationRequest：prompt_ids、answer_length、seed，不含标准答案内容。
- Prediction：token_ids、text 是评分预测；新协议另含 correct 和 trace。旧协议 correct 为 None、trace 为空。dataclass 自动生成的方法没有额外算法逻辑。

| 函数 | 作用 | 数学对应 |
| --- | --- | --- |
| validate_generation(config) | 校验并补齐 steps、temperature，返回归一化参数字典 | K 的定义以及 T≥0 的约束 |
| prepare_request(adapter, task, seed, protocol) | 渲染问题、联合分词、校验边界和解码，返回不含标签的请求及单独的 target；按协议和任务派生种子 | p、y、L 的定义；随机流初始化 |
| generate(model, adapter, request, steps, temperature, protocol, target) | 一次题目的完整去噪过程；两协议共用预测和选位，仅输入更新不同 | 下文全部预测、选位及状态更新公式 |
| LLaDABackend.__init__(settings, device) | 加载独立评价模型并保存原始目标层快照 | 固定 W_0，为模型切换提供恢复基准 |
| LLaDABackend.use_checkpoint(path) | 先恢复原始目标权重，校验后统一覆盖保存的最终目标权重；None 仅恢复 | W←W_0，再 W_target←W_saved；不是重新计算编辑增量 |
| LLaDABackend.close() | 恢复原始目标权重，释放后端持有的模型和快照引用 | 恢复操作，不计算分数 |

### 分词和初始状态

令 q 为问题文本，a 为标准答案文本，p 为提示 token，y 为标准答案 token：

$$
p=\operatorname{Tok}(\operatorname{Render}(q)),\qquad
y=\operatorname{Tok}(\operatorname{Render}(q)+a)[|p|:],\qquad L=|y|.
$$

代码要求联合分词结果以 p 为稳定前缀，答案非空且能精确解码。初始答案区域和未评价位置为：

$$
x^{(0)}=[\mathrm{MASK},\ldots,\mathrm{MASK}],\qquad R_1=\{1,\ldots,L\}.
$$

实际输入是 [p,x]。数学使用答案位置 1…L；代码 remaining 使用完整序列下标，trace.positions 使用答案区域从 0 开始的下标，转换关系为 position = remaining_index − len(p)。

### 分布、候选预测与置信度

将特殊 token 及 mask token 的 logits 设为负无穷，得到 \(\tilde\ell_{t,i}\)。定义：

$$
\tau=
\begin{cases}
T,&T>0,\\
1,&T=0,
\end{cases}
\qquad
P_{t,i}=\operatorname{softmax}(\tilde\ell_{t,i}/\tau).
$$

候选预测及置信度为：

$$
\hat y_{t,i}=
\begin{cases}
\arg\max_v\tilde\ell_{t,i}(v),&T=0,\\
\operatorname{Sample}(P_{t,i}),&T>0,
\end{cases}
\qquad
h_{t,i}=P_{t,i}(\hat y_{t,i}).
$$

T=0 仍计算 softmax，是为了得到选位置信度；不是按该分布随机抽样。正温度时，随机生成器按题初始化，不推进全局 RNG。

### 轮数分配和选位

$$
K=
\begin{cases}
L,&\mathrm{steps=null},\\
\min(\mathrm{steps},L),&\text{显式正整数步数},
\end{cases}
\qquad
c_t=\left\lfloor\frac{L}{K}\right\rfloor+\mathbf1[t\le L\bmod K].
$$

S_t 是剩余位置 R_t 中置信度最高的 c_t 个位置；并列按位置从小到大。不是从左到右固定选位，也不根据标签是否正确来选位。

### 两种协议的输入更新

旧协议把本轮候选写入输入：

$$
x_i^{(t)}=
\begin{cases}
\hat y_{t,i},&i\in S_t,\\
x_i^{(t-1)},&i\notin S_t.
\end{cases}
$$

新协议先保存：

$$
\hat y_i^{\mathrm{score}}\leftarrow\hat y_{t,i},
\qquad r_i\leftarrow\mathbf1[\hat y_{t,i}=y_i],\qquad i\in S_t,
$$

再统一回填：

$$
x_i^{(t)}=
\begin{cases}
y_i,&i\in S_t,\\
x_i^{(t-1)},&i\notin S_t.
\end{cases}
$$

两个协议均更新 \(R_{t+1}=R_t\setminus S_t\)。新协议每个位置恰好评分一次，同轮使用同一次前向；错误预测保留在评分记录中，不被回填覆盖。未选中位置的本轮候选不计分，下一轮重算。

新协议最终输入必然等于 y，但结果返回的是 \(\hat y^{\mathrm{score}}\)。before 和 after 各自从全 mask 开始，各自选位，路径可能不同。

## reporting.py：每道题怎么评分，整个实验怎么平均

| 函数 | 作用 | 数学对应 |
| --- | --- | --- |
| token_accuracy(predicted, target) | 对非空、等长序列逐位置比较 token ID | s=Σ_i 1[ŷ_i=y_i]/L |
| paired_scores(before, after, target) | 计算两侧正确率、两侧完全匹配，以及前后 token/整段一致率 | 下方单题及诊断量公式 |
| write_record(stream, record) | 写一条 JSONL 并刷新；不接受 NaN/Inf | 文件保存，无评分公式 |
| _mean(rows) | 对六列评分逐列等权平均；空集合返回六个 None | 等权算术平均，不进行 token 加权 |
| _pair(values, offset) | 取相邻的 before/after 数值并计算 delta；空值保持 None | Δ=after−before |
| ScoreSummary.__init__(tasks, metrics) | 固定各组/子集预期题数，初始化按源案例累计的容器 | 确定覆盖率分母 |
| ScoreSummary.add(task, scores) | 在结果成功保存后累计配对评分，同时归入 all 和该题接受/拒绝子集 | 收集各组各案例的单题分数 |
| ScoreSummary.export() | 输出两种宏平均、差值、诊断量及覆盖状态 | 下方 probe_macro 和 case_macro |

单题正确率为：

$$
s_q^u=\frac{1}{L_q}\sum_{i=1}^{L_q}\mathbf1[\hat y_{q,i}^u=y_{q,i}],
\qquad u\in\{\mathrm{before},\mathrm{after}\}.
$$

新协议中的 \(\hat y^u\) 指保存的评分预测，因此这与 \(\sum_i r_i/L\) 相同。评分函数不需要读取最终 state，也不把 correct 当成第二套独立指标。

补充诊断量为：

$$
e_q^u=\mathbf1[\hat y_q^u=y_q],\qquad
a_q=\frac{1}{L_q}\sum_i\mathbf1[\hat y_{q,i}^{\mathrm{after}}=\hat y_{q,i}^{\mathrm{before}}],
\qquad
b_q=\mathbf1[\hat y_q^{\mathrm{after}}=\hat y_q^{\mathrm{before}}].
$$

分别对应 exact_match、output_token_agreement、output_exact_agreement。新协议下它们比较评分预测，不能解释成自主生成整段答案的成功率或一致率。

设当前子集、当前题组有 N 道已完成题、C 个有题的源案例，源案例 c 的题集为 Q_c、题数 n_c：

$$
\operatorname{probe\_macro}^u=\frac{1}{N}\sum_qs_q^u,\qquad
\operatorname{case\_macro}^u=\frac{1}{C}\sum_c\left(\frac{1}{n_c}\sum_{q\in Q_c}s_q^u\right),
\qquad \Delta=\mathrm{after}-\mathrm{before}.
$$

附加诊断量按相同案例平均方式汇总；exact_match 有 before/after/delta，一致率是一个配对值。

预期无题为 not_applicable/null；有预期题但未全部完成为 incomplete；全部完成为 complete。部分均值只使用已完成题，必须同时查看完成数量及整个运行的 status，不能当成完整实验分数。

## runner.py：一次评价如何串起来

| 函数 | 作用 | 数学对应 |
| --- | --- | --- |
| run_evaluation(config_file, run_dir, device, model_path, dataset_path, eval_id) | 读取配置和产物，构造任务；执行全部 before，再按 checkpoint 分组执行 after；记录逐题预测、汇总与失败状态，并校验输入未变 | 调用上文公式；另计算接受率和覆盖率 |

$$
\operatorname{acceptance\_rate}=\frac{\text{接受的编辑源案例数}}{\text{尝试的编辑源案例数}},
\qquad
\operatorname{coverage}=\frac{\text{成功保存的配对评价题数}}{\text{预期评价题数}}.
$$

两者分母不同。拒绝案例不会从评价分母中消失。每题在配对结果成功写入后才累计 completed；中断时保存 failed 和当前任务信息，模型恢复由 ExitStack 调用 close 完成。

manifest.json 保存协议、配置、任务和输入摘要及环境信息；predictions.jsonl 保存 before/paired 预测和新协议轨迹；summary.json 保存完成状态、覆盖率和统计结果。

## 与原 UltraEdit、原 MDLM 数学逻辑的关系

- **直接沿用原 UltraEdit 的部分**：按 token ID 比较的单题正确率，以及逐题等权平均 probe_macro。五类指标通过不同题集得到。
- **扩散模型适配的部分**：全 mask 初态、候选置信度、轮数分配和非左到右选位。自主去噪规则与 MDLM.states.sample_denoise 的采样/揭示约定一致，但 evaluation 不调用该编辑函数。
- **标准答案回填新增的部分**：先记录评分再写入标准 token。这是独立评价协议，不属于 MDLM 编辑状态采样，也不改变编辑监督。
- **项目已有的补充报告部分**：before/after、case_macro、分组覆盖率、完全匹配和前后一致率；不应称为原版 UltraEdit 已经报告的全部指标。
- **本模块没有实现的 MDLM 编辑公式**：取键、二阶矩阵、写回因子、残差优化、权重增量合并及接受条件。use_checkpoint 读取的是已保存的最终目标权重，不重新计算这些公式。

## 本轮精简及验证范围

保留六文件结构与关键校验，公式注释不编号。精简重复协议检查、查字典前的额外错误分支、重复序列转换和数据类型检查；权重来源、答案对齐、有限概率、统计分母及结果保存约束仍保留。runner 使用 finally 保存失败状态，异常在恢复后自然传播，模型恢复仍由 ExitStack 完成。

迁入的数据读取器将 JSON/JSONL 共用的迭代与对象检查合并，移除多层捕获后重新抛出的包装及无用的 convert 占位方法。文件关闭、读取限量、字段映射、原始文本、案例编号、FEVER 改写选择和多跳题成对规则不变。

未知框架、缺失必要字典字段等错误直接由 Python 抛出，JSON 解析和依赖错误直接使用库的原始异常；不再保证所有错误都包装成 ValueError。错误仍会中断执行，不跳过坏记录、不回退默认答案、不改变标准答案或评分结果。

显式校验使用 assert，检查失败为 AssertionError；需要普通 Python 运行模式，不能使用移除断言的 -O/-OO。P5 的数值或等价性失败由验收模块返回明确的拒绝结果，不依赖断言决定接受/拒绝。

测试入口：python -B -m unittest discover -s tests -p test_evaluation.py -v。测试包含方案示例、温度和轮数、权重模式、旧协议回归、真实样本配模拟模型的完整入口。真实大模型验证与模拟验证分开，不以测试通过替代真实权重验收。
