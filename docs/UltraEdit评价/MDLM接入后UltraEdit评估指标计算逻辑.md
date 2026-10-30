# MDLM 接入后 UltraEdit 指标的数学逻辑与实现过程

本说明专门对应 evaluation 包保留的 diffusion_short_v1 自主去噪协议。新增的 diffusion_gold_v1 标准答案回填协议见下方方案文档。先读已完成的 MDLM 编辑产物，为实际尝试过的事实建立五类问题；同一道题分别用原始模型和编辑后模型从全 mask 答案窗口生成；计算 token 正确率，再按题目或源案例汇总。整个评估独立于 MDLM 的编辑优化过程。

若要对照原版，参见 [原版 UltraEdit 评价指标计算逻辑](原版UltraEdit评估指标计算逻辑.md)；已实现的标准答案回填协议参见 [MDLM 置信度顺序标准答案回填评价方案](MDLM置信度顺序标准答案回填评价方案.md)。

## 1. 从编辑产物建立评估任务

入口是仓库根目录的 evaluate.py，默认配置为 configs/evaluation/mdlm_ultraedit.json；其中 `protocol` 用于选择自主去噪或标准答案回填协议。运行器读取已完成且 status=complete 的 run_report.json，检查编辑模式、轮次、案例与事实 ID，并按原报告的数据读取器和读取参数重新读取数据。评估源文件、编辑报告和模型产物只读。

设编辑报告中实际尝试的事实集合为 $F$，配置所选问题组为 $G$。对每个事实 $f\in F$、问题组 $g\in G$ 以及该组的一道探针 $q$，建立一个任务：

$$
t=(\text{源案例ID},\text{事实ID},\text{问题组},\text{问题文本},\text{标准答案},\text{接受状态},\text{权重路径}).
$$

这里“源案例”可以包含多个事实，因而一条源案例可以对应多道评估题。只回测报告中实际尝试的事实；未被编辑流程尝试的事实不进入分母。edit 组必须有题；persona、portability 等可选组可以没有题。任务 ID 由事实 ID、问题组和探针 ID 生成，用来避免重复，并派生每题的随机种子。

| 问题组 $g$ | 报告中的指标名 | 标准答案 |
| --- | --- | --- |
| edit | ES | 新事实答案 |
| generalization | GS | 同一新事实答案 |
| preservation | LS | 无关问题原有的标准答案 |
| persona | person_score | 角色化问题对应的新事实答案 |
| portability | mhop_score | 多跳问题自己的标准答案 |

字段映射由 data/readers 负责，evaluation/framework.py 负责从标准化后的题组中建立任务。LS 的标准答案是无关题自己的 gold 答案，与编辑前后输出是否相同是两回事。

## 2. 确定每道题评估前后的模型

记原始模型为 $\theta_0$，某轮被接受后保存的目标层权重为 $\theta_c$，连续编辑最终保存的模型为 $\theta_{\mathrm{final}}$。所有任务的 before 都来自同一个 $\theta_0$。after 按编辑运行模式确定：

| 编辑模式和轮次 | after 使用的模型 |
| --- | --- |
| independent，案例已接受 | 该案例保存的 case_N 权重所形成的模型 |
| independent，案例被拒绝 | $\theta_0$；直接复用 before 预测 |
| sequential，不论该事实所属轮次接受或拒绝 | 全部轮次结束后的 $\theta_{\mathrm{final}}$ |

连续模式的被拒绝事实仍要用最终模型回测，因为其他已接受轮次可能改变该题的输出。独立模式切换到下一个案例权重前会先恢复原始目标层权重，避免前一案例的改动混入后一案例。评估只在自己的模型实例中加载这些权重，不重新执行 MDLM 的状态采样、残差优化或合并接受。

## 3. 把题目和标准答案变成评估输入

对任务 $q$，先通过 LLaDA 的聊天模板渲染题目文本 $x_q$，再分词得到提示序列 $p_q$。代码同时分词“渲染后的提示 + 标准答案”，验证提示 token 确实构成其前缀，剩余 token 定义为标准答案：

$$
p_q=\operatorname{Tok}(\operatorname{Render}(x_q)),\qquad
y_q=\operatorname{Tok}(\operatorname{Render}(x_q)+a_q)[|p_q|:],
\qquad L_q=|y_q|.
$$

这里的 $a_q$ 是答案文本，$L_q$ 是答案 token 数。代码还要求 $L_q>0$，并验证后缀解码后与答案文本完全一致。若边界不稳定、含特殊 token、预置 mask 或超出模型上下文长度，就报错，不自动截断。

真正送入生成器的请求只有 $(p_q,L_q,\text{seed}_q)$，不含 $y_q$ 的内容；输入序列初始化为

$$
z^{(0)}=[p_q,\underbrace{m,\ldots,m}_{L_q\text{ 个}}],
$$

其中 $m$ 是 mask token。参考答案只用于**决定窗口长度**和**事后评分**，没有作为正确答案前缀输入模型。这仍是“已知目标长度”的短答案评估，尚不是未知长度的自由生成。

评估种子默认继承编辑报告；每道题由“评估种子 + 协议名 + 任务 ID”的 SHA-256 派生独立种子。before 与 after 对同一道题使用相同的请求和派生种子；运行器只构造一次不可变请求和目标 token，并在 before/after 中复用，保证两侧输入定义一致。

## 4. 扩散模型如何逐步生成答案

设答案窗口长 $L=L_q$。配置 generation.steps 为 null 时，实际去噪步数 $S=L$，每步揭示一个位置；显式指定正整数 $s$ 时，$S=\min(s,L)$。第 $t$ 步计划揭示的数量是

$$
c_t=\left\lfloor\frac{L}{S}\right\rfloor+
\mathbf{1}[t\le L\bmod S],\qquad t=1,\ldots,S.
$$

设第 $t$ 步尚未揭示的位置集合为 $R_t$。模型在当前序列上前向传播，只读取 $R_t$ 位置的词表 logits $\ell_{t,i}(v)$。特殊 token 和 mask token 的 logits 被设为负无穷。对每个剩余位置 $i$，得到候选 token $u_{t,i}$：

$$
u_{t,i}=
\begin{cases}
\operatorname*{argmax}_v\ell_{t,i}(v),&T=0,\\
\text{从 }\operatorname{softmax}(\ell_{t,i}/T)\text{ 中抽样},&T>0,
\end{cases}
$$

其中 $T$ 是 generation.temperature。即使 $T=0$，实现也用 softmax 后候选 token 的概率定义置信度；$T>0$ 时使用每题独立的随机数生成器：

$$
h_{t,i}=\Pr_t(u_{t,i}\mid z^{(t-1)}).
$$

按 $h_{t,i}$ 从高到低选择 $c_t$ 个位置揭示；置信度相同时按位置顺序确定先后。其余位置仍为 mask，进入下一步。最终没有剩余 mask，得到长度恰为 $L$ 的预测答案 token 序列 $\hat y_q$。这种逐步揭示方式是当前评估协议的具体选择；改变步数或温度，会改变被评估的行为和可能的分数。

## 5. 五项指标的共同单题公式

同一道题在原模型和编辑后模型上分别得到 $\hat y_q^{\mathrm{before}}$、$\hat y_q^{\mathrm{after}}$。对于阶段 $u\in\{\mathrm{before},\mathrm{after}\}$，逐题 token 正确率是

$$
s_q^u=\frac{1}{L_q}\sum_{j=1}^{L_q}
\mathbf{1}[\hat y_{q,j}^{u}=y_{q,j}].
$$

例如标准答案 token 为 $[A,B,C,D]$，编辑后预测为 $[A,X,C,D]$，这题的 after 分数为 $3/4=0.75$。比较的是**相同位置的 token ID**；它不是整句正确率，也不按文字语义给部分分。

ES、GS、LS、person_score、mhop_score 只是把这一公式应用到表中对应的问题组。LS 用 preservation 题的标准答案计算；前后回答完全一样但都答错时，该题 LS 仍为 0。

## 6. 如何从单题分数得到汇总指标

汇总分别针对 all、accepted、rejected 三个子集以及每个已选择的问题组计算。设某子集、某组有 $N$ 道已完成题，分布在 $C$ 个有题的**源案例**中，第 $c$ 个源案例有 $n_c$ 道题。报告同时给出两种平均：

$$
\operatorname{probe\_macro}^{u}=\frac{1}{N}\sum_{q=1}^{N}s_q^u,
$$

$$
\operatorname{case\_macro}^{u}=\frac{1}{C}\sum_{c=1}^{C}
\left(\frac{1}{n_c}\sum_{q\in Q_c}s_q^u\right).
$$

probe_macro 给每道题相同权重，其 after 的平均顺序对应原版 UltraEdit；case_macro 先平均同一源案例的题，再让每个源案例权重相同，是当前报告的主口径。每一种口径各有 before、after、delta，其中

$$
\Delta=\text{after}-\text{before}.
$$

例如源案例 A 只有一题，得 1；源案例 B 有三题，都得 0，则 probe_macro 为 $1/4=0.25$，case_macro 为 $(1+0)/2=0.5$。看到 ES 等名称时，必须一并查看采用了哪种口径和子集。

存在至少一道可评价题时，没有预期题目的组标记 not_applicable，数值为 null；若所选组全部无题，入口报错，不创建全空评价报告。任务中断时，已完成题可能形成部分均值，但状态为 incomplete；需同时确认整个运行 status=complete 和对应指标 complete，才能当作完整实验汇总。summary.json 还记录总体 expected_tasks、completed_tasks 和 coverage；各指标也记录该组的预期与完成数量。

## 7. 额外诊断量的公式

同一题的整段 token 完全匹配率为

$$
e_q^u=\mathbf{1}[\hat y_q^u=y_q].
$$

报告的 exact_match_case_macro 对 $e_q^u$ 按“先源案例内、后源案例间”平均，并报告 before、after 与 delta。编辑前后输出一致率不使用标准答案：

$$
a_q=\frac{1}{L_q}\sum_{j=1}^{L_q}
\mathbf{1}[\hat y_{q,j}^{\mathrm{after}}=\hat y_{q,j}^{\mathrm{before}}],
\qquad
b_q=\mathbf{1}[\hat y_q^{\mathrm{after}}=\hat y_q^{\mathrm{before}}].
$$

output_token_agreement_case_macro 汇总 $a_q$，output_exact_agreement_case_macro 汇总 $b_q$。两者都按源案例平均，都是单一数值，没有 before、after 或 delta。它们只帮助判断输出变化，**不替代 LS**。

## 8. 实际执行顺序与输出

运行器先为全部任务计算原模型预测并写入 predictions.jsonl 的 before 记录；再按权重路径分组，恢复或加载相应目标层权重，为同一道题得到 after 预测。每道题计算分数后写 paired 记录，同时累计到 ScoreSummary。独立模式的拒绝案例直接使用原模型的 before 预测作为 after。

输出目录是新的 evaluations/<eval_id>，不能覆盖既有目录。三个文件的作用如下：

| 文件 | 内容 |
| --- | --- |
| manifest.json | 协议、配置、模型与数据来源、任务摘要、源文件和权重的评估时哈希 |
| predictions.jsonl | 每道题的原模型预测、编辑后预测、答案 token 与逐题分数 |
| summary.json | 完成状态、覆盖率、五项指标及附加诊断量的分组汇总 |

运行器在开始和结束时核对源数据、编辑报告及保存权重的哈希，切换权重时先校验目标层名称、形状、精度和有限性。既有编辑产物没有记录**编辑当时**的源数据和原始模型权重哈希，因此 manifest 中的评估时摘要不能证明它们与编辑当时的文件完全相同。当前 LS 是基准数据的无关题正确率，也未验证这些题与编辑时 keep_pool 完全独立。

该协议衡量给定标准答案长度时的**短答案生成**。它不等于原版自回归正确前缀评估，也不能直接检验“短答案编辑是否影响自由生成的长答案”；后一个问题需要新的长答案生成与评分协议。
