# 扩散语言模型知识编辑：统一符号与完整数学推导（讨论稿 v0.1）

> **定位**：依据《02_扩散知识编辑模型.md》（2026-10-15）第 2—3、5、7—8 节整理，**仅以“9 月 23 日三文档汇编路线”为主体**。不把第 6 节的“完整方案 v1 / 全状态梯度 / 揭示事件”等独立候选拼接到本路线。
>
> **证据标记**：【文档定义】表示原文已明确；【等价推导】表示基于原文数学式展开的推导；【待确定】表示原文没有给出唯一规定；【可选补全】表示为了构成可直接实现的算法而提出的建议，**不是原文已经确定的结论**。
>
> **重要边界**：这里的“数学链条自洽”指符号、条件化求解、优化变量和参数写入之间无明显断裂，不意味着自由长回答一定正确，不意味着整个深层网络具有全局闭式解，也不意味着完成了真实模型实验验证。

## 1. 数学符号总表

统一采用**列向量**记法；上标 `*` 表示目标或所选优化结果，下标 `0` 表示最初未编辑模型，下标 `b` 表示当前编辑轮次的冻结基准模型。

| 类别 | 符号 | 数学含义 | 类型/维度 | 来源和状态 |
|---|---|---|---|---|
| 任务 | $N$ | 待编辑事实数量 | 正整数 | 【文档定义】 |
| 任务 | $i$ | 事实索引 | $i\in\{1,\ldots,N\}$ | 【文档定义】 |
| 任务 | $(s_i,r_i,a_i^*)$ | 主体、关系、目标短答案 | 文本或 token 序列 | 【文档定义】 |
| 扩散 | $t$ | 去噪轨迹的状态索引 | 离散索引 | 【文档定义】 |
| 扩散 | $x_{i,t}$ | 某次模型前向前的**完整实际输入**，含 token/mask、位置、注意力、时间、分块条件 | 模型输入结构 | 【文档定义】 |
| 扩散 | $\mathcal A_i$ | 短答案监督位置集合 | 位置索引集合 | 【文档定义】 |
| 扩散 | $\mathcal M_{i,t}$ | 状态 $t$ 中仍被 mask 的短答案位置 | $\mathcal M_{i,t}\subseteq\mathcal A_i$ | 【文档定义】 |
| 扩散 | $a_{i,j}^*$ | 位置 $j$ 对应的目标 token | token ID | 【文档定义】 |
| 扩散 | $\mathcal T_i$ | 真实轨迹中去重后的兼容状态集合 | 有限集合 | 【文档定义】 |
| 模型 | $\theta_0$ | 最初未编辑模型参数 | 参数集合 | 【文档定义】 |
| 模型 | $\theta_b$ | 本轮冻结的基准模型参数；首轮通常为 $\theta_0$ | 参数集合 | 【文档定义】 |
| 模型 | $\mathcal L$ | 本轮被允许编辑的层/模块集合 | 模块索引集合 | 【文档定义】 |
| 模型 | $\ell$ | 编辑层或模块索引，**不是**去噪状态 | $\ell\in\mathcal L$ | 【文档定义】 |
| 模型 | $d_{i,\ell},d_{o,\ell}$ | 第 $\ell$ 层线性模块的输入、输出宽度 | 正整数 | 【文档定义】 |
| 模型 | $k_{\ell,p}(x;\theta)$ | 第 $\ell$ 层第 $p$ 次实际调用的输入向量 | $\mathbb R^{d_{i,\ell}}$ | 【等价记法】 |
| 模型 | $z_{\ell,p}(x;\theta)$ | 该调用的线性模块输出 | $\mathbb R^{d_{o,\ell}}$ | 【等价记法】 |
| 模型 | $W_{b,\ell},b_\ell$ | 本轮基准权重和固定偏置 | $\mathbb R^{d_{o,\ell}\times d_{i,\ell}}$；$\mathbb R^{d_{o,\ell}}$ | 【文档定义】 |
| 损失 | $p_{\theta,j}(a\mid x)$ | 模型在位置 $j$ 对 token $a$ 的原始预测概率 | $[0,1]$ | 【文档定义】 |
| 损失 | $L_{i,t}(\theta)$ | 单事实、单状态剩余 mask 的平均负对数概率 | 标量 | 【文档定义】 |
| 损失 | $F(\theta)$ | 先状态内/事实内、再跨事实平均的编辑目标 | 标量 | 【文档定义】 |
| 统计 | $K_\ell$ | 每事实一列的固定参考输入键 | $\mathbb R^{d_{i,\ell}\times N}$ | 【文档定义】 |
| 统计 | $C_\ell$ | 扩散保持数据上的**未中心化二阶统计**，含尺度 | $\mathbb S_+^{d_{i,\ell}}$ | 【文档定义】；数据配方待确定 |
| 统计 | $\lambda_\ell$ | 求解器的额外 ridge 系数 | $\lambda_\ell>0$ | 【文档定义】 |
| 统计 | $H_\ell$ | 条件化求解的系数矩阵 $C_\ell+K_\ell K_\ell^\top+\lambda_\ell I$ | $\mathbb R^{d_{i,\ell}\times d_{i,\ell}}$ | 【文档定义】 |
| 统计 | $X_\ell$ | 线性系统 $H_\ell X_\ell=K_\ell$ 的解 | $\mathbb R^{d_{i,\ell}\times N}$ | 【文档定义】 |
| 统计 | $Q_\ell=X_\ell^\top$ | 优化期间固定的右因子 | $\mathbb R^{N\times d_{i,\ell}}$ | 【文档定义】 |
| 优化 | $R_\ell$ | 第 $\ell$ 层**待优化**的输出空间残差 | $\mathbb R^{d_{o,\ell}\times N}$ | 【文档定义】 |
| 优化 | $\Delta W_\ell=R_\ell Q_\ell$ | 参数权重增量 | $\mathbb R^{d_{o,\ell}\times d_{i,\ell}}$ | 【文档定义】 |
| 优化 | $\theta(R)$ | 各层同时生效的真实候选模型 | 参数集合 | 【文档定义】 |
| 优化 | $L_{\mathrm{keep}}$ | 候选训练期间的知识保持损失 | 标量 | 【待确定】具体公式 |
| 优化 | $\beta,\rho_\ell$ | 保持损失权重、残差惩罚权重 | $\beta\ge0,\rho_\ell>0$ | 【文档定义】 |
| 优化 | $J(R)$ | 联合残差优化目标 | 标量 | 【文档定义】 |
| 优化 | $R_\ell^*$ | 在**有限预算**内选出的残差候选 | 与 $R_\ell$ 同形 | 【文档定义】；不保证全局最优 |
| 验收 | $\alpha$ | 所有层共享的、来自预设有限集合的试步 | $\alpha>0$ | 【文档定义】 |
| 验收 | $F_{\mathrm{dev}},G_{\mathrm{dev}},U$ | 开发编辑损失、相对初始模型的完整词表 KL 漂移、累计参数变化 | 标量 | 【文档定义】；后两项计算约定待确定 |
| 验收 | $\delta,\tau_G,\tau_U$ | 最小编辑改善、保持漂移阈值、参数变化阈值 | 预设标量，$\delta>0$ | 【文档定义】 |

> **符号辨析**：$k$ 是目标**线性模块的实际输入**，不是注意力 Key，也不是答案的 token embedding；$t$ 表示扩散输入状态，$\ell$ 表示神经网络层。$\theta_0$ 与 $\theta_b$ 在连续编辑时不能混淆。

## 2. 问题定义与建模前提

【文档定义】给定预训练掩码扩散语言模型 $f_{\theta_0}$ 和 $N$ 条待更新事实 $(s_i,r_i,a_i^*)$，寻找仅作用于白名单线性模块集合 $\mathcal L$ 的更新，使后续**不提供新答案**的自由去噪生成能表达新事实，并尽量限制其他知识的漂移。

本轮基准权重满足：

$$
 z_{\ell,p}=W_{b,\ell}k_{\ell,p}+b_\ell,\qquad
 W_{b,\ell}\in\mathbb R^{d_{o,\ell}\times d_{i,\ell}}.
$$

只更新 $W_{b,\ell}$，偏置固定。基准权重**数值冻结**，但从输出到待优化残差的自动求导链路必须保留。

【文档限定】下述“兼容状态”构造针对**受控、短答案对齐、不会重新掩码的窗口**。它不是任意 remasking 采样器的普遍性质。模型真实采样时不提前植入新答案。

## 3. 从真实扩散轨迹到训练目标

### 3.1 轨迹构造和兼容状态筛选

对事实 $i$，由本轮基准模型 $\theta_b$ 自主采样轨迹，保存每次前向的完整输入 $x_{i,t}$。令 $\mathcal A_i$ 是与目标短答案 token 对齐的位置集合，$\mathcal M_{i,t}$ 为其中仍被 mask 的位置。主训练集合定义为：

$$
\boxed{
\mathcal T_i=\big\{t:\mathcal M_{i,t}\ne\varnothing,\ \ x_{i,t,j}=a^*_{i,j}\ \forall j\in\mathcal A_i\setminus\mathcal M_{i,t}\big\}.
}
$$

即：**仍有需要预测的答案 token**，且**已经揭示的答案 token 不与目标冲突**。非空目标的全 mask 初态保留。输入去重应以实际完整条件为准，不能仅凭可见文本判重。冲突状态、终态与零监督状态仍保存为诊断数据，不进入此路线的主训练目标。

### 3.2 两级归一监督损失

定义位置 $j$ 的正确 token 为 $a^*_{i,j}$，则：

$$
\boxed{
L_{i,t}(\theta)=-\frac{1}{|\mathcal M_{i,t}|}
\sum_{j\in\mathcal M_{i,t}}
\log p_{\theta,j}(a^*_{i,j}\mid x_{i,t}).
}
$$

再令：

$$
\boxed{
F(\theta)=\frac1N\sum_{i=1}^{N}\frac{1}{|\mathcal T_i|}
\sum_{t\in\mathcal T_i}L_{i,t}(\theta).
}
$$

【等价推导】展开得：

$$
F(\theta)=-\sum_{i=1}^{N}\sum_{t\in\mathcal T_i}
\sum_{j\in\mathcal M_{i,t}}
\underbrace{\frac{1}{N|\mathcal T_i||\mathcal M_{i,t}|}}_{w_{i,t,j}}
\log p_{\theta,j}(a^*_{i,j}\mid x_{i,t}).
$$

因此每条事实总权重相等；事实内的不同兼容状态权重相等；状态内剩余 mask 位置权重相等。实现微批梯度累积时必须保留 $w_{i,t,j}$，不能把各个微批均值简单相加。这里 $F$ 是**受控短答案代理损失**，不等于最终自由生成序列的精确似然。

## 4. 参考键、保持统计和受约束参数空间

### 4.1 参考键 $K_\ell$

【文档定义】每条事实、每个编辑模块默认抽取**一列**参考键：本轮基准模型在该事实的**全 mask 初态**，主体最后一个 token 位置，目标 FFN 输出线性层的实际输入。记为 $k^{\mathrm{ref}}_{i,\ell}$，则：

$$
K_\ell=[k^{\mathrm{ref}}_{1,\ell},\ldots,k^{\mathrm{ref}}_{N,\ell}]
\in\mathbb R^{d_{i,\ell}\times N}.
$$

参考键取一次后固定。与之不同，后续候选网络中各个位置的**实际输入** $k_{\ell,p}(x;\theta(R))$ 随前层参数更新而变化；这两个对象不能互换。多去噪状态用来构造监督，并不意味着要对每个状态单独增加一列残差。

### 4.2 保持统计 $C_\ell$

【文档定义】$C_\ell\succeq0$，由**目标扩散模型**的保持输入和掩码上下文重新估计，是包含适当尺度的**未中心化二阶统计**；不能沿用其他自回归模型的矩阵。

【等价解释】若保持输入的列向量为 $u_m$，并给定非负权重 $\omega_m$，一个符合“未中心化二阶统计”含义的通用形式是：

$$
C_\ell=\sum_{m=1}^{M}\omega_m u_m u_m^\top,
\qquad \omega_m\ge0.
$$

由此：

$$
\operatorname{tr}(\Delta W_\ell C_\ell\Delta W_\ell^\top)
=\sum_m\omega_m\|\Delta W_\ell u_m\|_2^2.
$$

这说明二阶矩惩罚对应于限制更新对被选保持输入的线性层输出扰动。

**注意**：上述加权展开是解释性数学形式；保持数据集、采样粒度、各层统计尺度及权重 $\omega_m$ 的实际配方，原文**尚未唯一确定**。

## 5. 从软拟合问题推导固定求解核

这是把文档第 3.3 节公式连接成一段严格推导所需的关键环节。

对固定 $K_\ell,C_\ell$，先将 $R_\ell$ 暂时视为**给定**的目标输出残差，考虑下面的条件化线性优化问题：

$$
\boxed{
\min_{\Delta W_\ell}\quad
\frac12\|\Delta W_\ell K_\ell-R_\ell\|_F^2
+\frac12\operatorname{tr}(\Delta W_\ell C_\ell\Delta W_\ell^\top)
+\frac{\lambda_\ell}{2}\|\Delta W_\ell\|_F^2.
}
\tag{1}
$$

【等价推导】式 (1) 是与文档带 ridge MEMIT 型核相对应的**软拟合解释**：首项要求参考键上的输出变化接近期望残差；后两项控制保持输入扰动和权重范数。

对 $\Delta W_\ell$ 求导：

$$
\nabla_{\Delta W_\ell}=
(\Delta W_\ell K_\ell-R_\ell)K_\ell^\top
+\Delta W_\ell C_\ell+\lambda_\ell\Delta W_\ell.
$$

令梯度等于零，整理得正规方程：

$$
\Delta W_\ell(C_\ell+K_\ell K_\ell^\top+\lambda_\ell I)
=R_\ell K_\ell^\top.
$$

定义：

$$
H_\ell=C_\ell+K_\ell K_\ell^\top+\lambda_\ell I.
$$

因为 $C_\ell\succeq0$ 且 $\lambda_\ell>0$，对任意非零 $v$：

$$
v^\top H_\ell v=v^\top C_\ell v+\|K_\ell^\top v\|_2^2
+\lambda_\ell\|v\|_2^2>0.
$$

因此 $H_\ell\succ0$，式 (1) 关于 $\Delta W_\ell$ 是严格凸的，其**条件化**最优解存在且唯一：

$$
\boxed{\Delta W_\ell=R_\ell K_\ell^\top H_\ell^{-1}.}
$$

为避免显式计算逆矩阵，实际求解：

$$
H_\ell X_\ell=K_\ell,\qquad Q_\ell=X_\ell^\top,
$$

得到：

$$
\boxed{\Delta W_\ell(R_\ell)=R_\ell Q_\ell.}
\tag{2}
$$

其中 $Q_\ell\in\mathbb R^{N\times d_{i,\ell}}$，$R_\ell\in\mathbb R^{d_{o,\ell}\times N}$，所以 $R_\ell Q_\ell$ 恰与原权重同形。此分解还意味着 $\operatorname{rank}(\Delta W_\ell)\le N$。

**重要边界**：严格凸与唯一解只适用于“$R_\ell,K_\ell,C_\ell$ 已固定时对 $\Delta W_\ell$ 求解”的问题，不适用于后文整个深层候选模型的非凸残差优化。

## 6. 多层真实候选网络与联合残差优化

### 6.1 为什么还要优化 $R_\ell$

上节回答的是：“**给定**目标残差 $R_\ell$，怎样构造兼顾保持惩罚的权重增量？”但没有回答“什么残差能让模型真正预测新事实”。因此这条汇编路线将 $Q_\ell$ 固定，转而在**真实候选模型输出**上联合优化各层 $R_\ell$。

【文档定义】令 $R=\{R_\ell\}_{\ell\in\mathcal L}$，定义：

$$
\boxed{
\theta(R)=\theta_b\oplus\{R_\ell Q_\ell\}_{\ell\in\mathcal L}.
}
\tag{3}
$$

$\oplus$ 指仅在选定权重矩阵处合并增量，所有选定层**同时**生效。对候选前向实际输入 $k$：

$$
\begin{aligned}
\tilde z_\ell
&=(W_{b,\ell}+R_\ell Q_\ell)k+b_\ell\\
&=W_{b,\ell}k+b_\ell+R_\ell(Q_\ell k).
\end{aligned}
\tag{4}
$$

式 (4) 是代数恒等式。它要求分支覆盖目标模块在该前向中的**所有真实调用位置**，不能只对主体位置注入一个向量；冻结基准权重也不能断开反向传播。

### 6.2 优化目标和梯度链路

【文档定义】在固定训练状态和 $Q_\ell$ 的前提下：

$$
\boxed{
J(R)=F(\theta(R))
+\beta L_{\mathrm{keep}}(\theta(R))
+\frac12\sum_{\ell\in\mathcal L}\rho_\ell\|R_\ell\|_F^2.
}
\tag{5}
$$

其中 $\beta\ge0$、$\rho_\ell>0$。**$L_{\mathrm{keep}}$ 的确切输入分布、KL 方向/度量、位置归一与初始模型参照规则仍待确定，不能在本稿中冒充原文已定义。**

【等价推导】假设 $F$ 和所选 $L_{\mathrm{keep}}$ 对模型参数可微，记
$\tilde W_\ell=W_{b,\ell}+R_\ell Q_\ell$，利用 $d\tilde W_\ell=(dR_\ell)Q_\ell$ 得到：

$$
\boxed{
\nabla_{R_\ell}J
=\left[\nabla_{\tilde W_\ell}
\big(F(\theta(R))+\beta L_{\mathrm{keep}}(\theta(R))\big)\right]Q_\ell^\top
+\rho_\ell R_\ell.
}
\tag{6}
$$

其中 $\nabla_{\tilde W_\ell}$ 指**整个多层候选网络**的总梯度，而非把其他层固定、仅对某个主体位置求出的局部梯度。前层更新引起的后层真实输入变化须保留在自动求导图中。

对于具体调用位置 $p$，式 (4) 的直接残差梯度贡献还可写成：

$$
\left.\nabla_{R_\ell}\mathcal E\right|_{p}
=e_{\ell,p}(Q_\ell k_{\ell,p})^\top,
\quad e_{\ell,p}=\frac{\partial\mathcal E}{\partial\tilde z_{\ell,p}},
$$

对各调用位置、样本和状态累加即可得到总梯度的相应部分。这里 $\mathcal E$ 代表可微训练损失。

【文档定义】采用有限优化预算获得 $R^*$；同一步内多微批累积梯度时保持残差值不变，整体加权不可失真；残差正则每个优化步计一次。**具体优化器、步数、学习率和提前停止准则尚待配置**。$R^*$ 只是已选候选，不保证全局最优。

## 7. 更新合并、软拟合误差与步长验收

### 7.1 内部目标和软拟合误差

【文档定义】在**去偏置的线性输出空间**中，定义辅助目标：

$$
V_\ell^*=W_{b,\ell}K_\ell+R_\ell^*.
$$

但实际当 $\alpha=1$ 时，写回后的参考键输出为：

$$
(W_{b,\ell}+R_\ell^*Q_\ell)K_\ell
=W_{b,\ell}K_\ell+R_\ell^*Q_\ell K_\ell.
$$

【等价推导】两者之差为：

$$
\boxed{
(W_{b,\ell}+R_\ell^*Q_\ell)K_\ell-V_\ell^*
=R_\ell^*(Q_\ell K_\ell-I_N).
}
\tag{7}
$$

因此只要 $Q_\ell K_\ell\ne I_N$，就通常不能声称**精确**满足 $\Delta W_\ell K_\ell=R_\ell^*$。这与 MEMIT 型**软拟合**是相容的，并非推导错误。

### 7.2 实际候选权重与等价性检查

【文档定义】从同一个本轮基准 $\theta_b$ 出发，对来自预定有限集合的公共步长 $\alpha>0$ 形成：

$$
\boxed{W_{\alpha,\ell}=W_{b,\ell}+\alpha R_\ell^*Q_\ell.}
\tag{8}
$$

所有层共享同一 $\alpha$，不同试步不能在前一个试步的权重上累加。对同一输入，临时因子分支 $\alpha R_\ell^*(Q_\ell k)$ 和实际合并矩阵的输出应在给定数值精度容差内一致；否则不应继续验收。共享存储/绑权模块等实现问题必须单独检查。

### 7.3 接受、拒绝与回退

【文档定义】相对于**本轮基准**要求开发编辑损失改善；相对于**最初模型**约束保持漂移和累计改动：

$$
\boxed{
\begin{cases}
F_{\mathrm{dev}}(\theta_\alpha)\le F_{\mathrm{dev}}(\theta_b)-\delta,\\
G_{\mathrm{dev}}(\theta_\alpha)\le\tau_G,\\
U(\theta_\alpha)\le\tau_U.
\end{cases}}
\tag{9}
$$

其中 $\delta>0$；$G_{\mathrm{dev}}$ 是固定保持输入上相对 $\theta_0$ 的完整词表 KL 漂移；$U$ 是相对 $\theta_0$ 的累计参数变化。**两者的精确定义与归一方式尚需固定**。数值有限、矩阵更新范数、分层诊断也应检查。若所有候选试步失败，恢复上一接受模型。

默认一轮；如预先允许最多刷新一次，第二轮应重新采样、提取参考键、重建保持统计并将 $R$ 清零；累计保持参照仍为 $\theta_0$。

## 8. 面向实现的算法伪代码

```text
输入：初始模型 θ0；当前基准 θb；N 条 (主体, 关系, 短答案)；编辑模块白名单 L；
      扩散采样器及受控短窗口协议；保持数据；求解/优化参数；
      预先固定的 α 候选集、开发评价集合和接受阈值。
输出：接受后的模型 θaccepted，或保持原基准/上一接受模型。

1. 校验白名单模块、矩阵维度、偏置/共享存储以及主体 token 对齐。
2. 冻结 θb 的数值；对各事实自主采样并保存全输入 x_{i,t}，不得注入新答案。
3. 按兼容规则去重、筛选 T_i；保存冲突/终态诊断；检查每条事实有监督状态。
4. 从全 mask 初态主体最后 token 抽取每层每事实一列 K_l。
5. 在当前扩散模型保持输入上估计 C_l；构造 H_l=C_l+K_l K_l^T+λ_l I。
6. 用线性系统 H_l X_l=K_l 求解，得固定 Q_l=X_l^T；检查分解和残差。
7. 初始化所有待优化 R_l（初始化策略需固定）；基准权重仍不参与更新。
8. 临时因子分支覆盖全部有效调用位置；联合优化 J(R)，得到 R*。
9. 对预先确定的每个 α，都从 θb 重新形成所有 W_{α,l}=W_{b,l}+αR*_l Q_l。
10. 检查分支与合并前向等价、更新数值有限、开发编辑改善、保持及累计约束。
11. 如某 α 符合预定接受规则，则原子提交全部目标矩阵；否则整体回退。
12. 固定最终模型，从原生全 mask/初始状态独立自由生成；
    分开报告受控短答案、问法泛化、自由目标事实、完整语义回答。
```

实现时要把初始化、优化预算、步长候选次序、候选选择规则、容差与统计数据版本都固化到运行配置中；这些属于**尚需补齐的操作定义**。

## 9. 数学上能严格说什么、不能严格说什么

### 可由目前定义支持的性质

1. 若 $C_\ell\succeq0$ 且 $\lambda_\ell>0$，则 $H_\ell\succ0$；固定 $K_\ell,C_\ell,R_\ell$ 的软拟合子问题有唯一最优 $\Delta W_\ell$。
2. 固定 $Q_\ell$ 后，$\Delta W_\ell=R_\ell Q_\ell$ 维度正确，且其秩不超过 $N$。
3. 当分支覆盖目标模块全部真实调用位置，且候选计算条件一致，$W_bk+R(Qk)$ 与 $(W_b+RQ)k$ 在数学上等价。
4. 训练目标 $J(R)$ 的梯度可通过全网络反向传播按式 (6) 求出；即便 $W_b$ 冻结，也无需冻结对残差的求导。
5. 由式 (7) 可知本方法通常是**软拟合**，并不必然满足精确写入 $\Delta WK=R$。

### 目前不能据此声称的结论

1. $J(R)$ 在整个多层非线性网络上凸、具有全局解析最优解或一定收敛到全局最优。
2. 编辑损失下降必然意味着每条事实、所有兼容/不兼容扩散状态、自由长回答都正确。
3. KL 与权重范数验收能够保证所有旧知识绝对不变。
4. 这一路线已优于 MEMIT、AlphaEdit、LoRA 等基线，或具备已验证的资源优势。
5. 这一路线具有经确认的论文独创性；相关迁移与 partial-mask 编辑的既有研究需要单独核对。

## 10. 使本稿成为“唯一可复现算法”还需要老师确认的项目

| 优先级 | 尚未封闭的问题 | 需要写入最终算法的明确约定 |
|---|---|---|
| P0 | 最终路线身份 | 选择本汇编路线并给出确定的方法名称，其他路线仅作对照/消融 |
| P0 | $L_{\mathrm{keep}}$ | 保护样本来源、位置/状态权重、具体可微公式、参照 $\theta_0$ 或 $\theta_b$ |
| P0 | $G_{\mathrm{dev}},U$ | KL 方向、词表/位置/样本归一、累计权重变化的范数与尺度 |
| P0 | 优化和试步规则 | $R$ 初始化、优化器、步数/停止规则、$\alpha$ 试步集合和接受选择规则 |
| P0 | 训练/开发/测试隔离 | 编辑轨迹、保持统计、开发阈值、最终独立自由生成评价如何严格分离 |
| P1 | 状态与数据协议 | token 对齐、短窗口长度、去重键、采样条件、无有效状态时的处理 |
| P1 | $C_\ell$ 的构造 | 保持语料、mask 分布、权重尺度、样本量、数据版本与统计精度 |
| P1 | 编辑模块与数值计算 | 层集合、密集/MoE 选择、解法、精度、共享存储及误差容差 |
| P1 | 顺序/联合/连续编辑边界 | $\theta_b$ 与 $\theta_0$ 的更新关系、快照事务和失败回退 |

### 10.1 可供老师讨论的“补全示例”，不是原文既定算法

如果老师希望马上形成一个能写代码的封闭方案，可以**提议**采用以下一组定义，但必须标注为“新增方法选择”，固定后再验证：

- **训练保持损失（建议之一）**：在训练保持输入及指定位置集 $\mathcal D_{\mathrm{keep}}^{\mathrm{train}}$ 上，使用完整词表 KL：

  $$
  L_{\mathrm{keep}}(\theta(R))=
  \frac1{|\mathcal D_{\mathrm{keep}}^{\mathrm{train}}|}
  \sum_{(x,p)\in\mathcal D_{\mathrm{keep}}^{\mathrm{train}}}
  \mathrm{KL}\big(p_{\theta_0,p}(\cdot\mid x)\,\|\,p_{\theta(R),p}(\cdot\mid x)\big).
  $$

- **开发保持指标（建议之一）**：在不与训练重合的固定开发保持集上，使用同方向的完整词表 KL。具体 token 加权与 mask 策略需写清。
- **累计改动指标（建议之一）**：若白名单权重起点为 $W_{0,\ell}$，定义

  $$
  U(\theta)=\left(\sum_{\ell\in\mathcal L}
  \|W_{\ell}(\theta)-W_{0,\ell}\|_F^2\right)^{1/2}.
  $$

  若跨轮次改变白名单或层尺度不一致，还需规定统一集合和归一方式。
- **残差优化与选步（建议之一）**：使用固定初始化、Adam/AdamW 等明确优化器、固定总步数与全局微批权重；在预先指定且不依赖测试集的有限 $\alpha$ 集合上逐一验收，以固定规则选择首个或最优可接受候选。

这些建议能够补齐部分接口，但**还需要做数值可行性实验和方法选择讨论**，不能直接写成“已由当前文档证明”。

## 11. 最终数学链路（一页摘要）

$$
\begin{gathered}
\underbrace{\{(s_i,r_i,a_i^*)\}_{i=1}^N,\ \theta_b}_{\text{编辑输入}}
\longrightarrow\underbrace{\{x_{i,t},\mathcal T_i\}}_{\text{真实兼容状态}}\\[2mm]
\longrightarrow\underbrace{F(\theta)}_{\text{两级归一的短答案损失}}
\quad+\quad
\underbrace{(K_\ell,C_\ell)}_{\text{键和保持统计}}\\[2mm]
\longrightarrow H_\ell=C_\ell+K_\ell K_\ell^\top+\lambda_\ell I
\longrightarrow Q_\ell=(H_\ell^{-1}K_\ell)^\top\\[2mm]
\longrightarrow\theta(R)=\theta_b\oplus\{R_\ell Q_\ell\}
\longrightarrow R^*\approx\arg\min_R J(R)\\[2mm]
\longrightarrow W_{\alpha,\ell}=W_{b,\ell}+\alpha R_\ell^*Q_\ell
\longrightarrow\text{一致性检查和三重开发验收}\\[2mm]
\longrightarrow\text{接受/回退}\longrightarrow\text{独立自由生成评价}.
\end{gathered}
$$

**总结**：这份稿子把已给定的定义组成了一个**条件化求解 + 联合残差反馈优化 + 受控验收**的数学链条。最需要老师确认的是第 10 节的数学及操作性空缺。填补之后可以形成唯一可实现的算法规格，但最终方法的有效性、泛化性和创新性仍需要独立实验与文献证据。
