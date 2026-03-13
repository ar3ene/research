**先给出名词定义（按后文使用顺序）**
1. 状态变量 $\mathbf{x}$：系统在某一时刻的全部连续自由度，例如电压、位置等。
2. 哈密顿量 $H(\mathbf{x})$：把每个状态映射到一个能量标量的函数。能量越低，状态越“容易出现”。
3. 热平衡分布（玻尔兹曼分布）：
$$
p_{\text{eq}}(\mathbf{x}) = \frac{1}{Z} e^{-H(\mathbf{x})/(k_B T)}, \quad
Z = \int e^{-H(\mathbf{x})/(k_B T)} \, d\mathbf{x}
$$
其中 $Z$ 是配分函数（归一化常数），$k_B T$ 是热噪声尺度。
4. 目标分布 $p_{\text{target}}$：你想让模型生成的数据分布。
5. KL 散度：
$$
D_{\mathrm{KL}}(p\|q) = \int p(\mathbf{x}) \ln \frac{p(\mathbf{x})}{q(\mathbf{x})} \, d\mathbf{x}
$$
衡量“用 $q$ 近似 $p$”的信息损失，且 $D_{\mathrm{KL}}\ge0$，等号当且仅当两分布几乎处处相等。
6. 变分自由能目标（这里是训练目标的等价写法）：
$$
\mathcal{L}(\theta) = \langle H_{\theta}(\mathbf{x}) \rangle_{p_{\text{target}}} + k_B T \ln Z(\theta)
$$
7. 潜变量 $\mathbf{z}$：不直接观测的中间变量，用来提升表达能力。
8. 有效自由能 $F_{\text{eff}}(\mathbf{v})$：把潜变量积分掉后，仅对可见变量 $\mathbf{v}$ 的等效能量。
9. Langevin 动力学：连续状态的“梯度下降 + 热噪声”随机动力学。
10. Gibbs 采样：在各条件分布之间交替采样，构造马尔可夫链逼近平衡分布。
11. CD（Contrastive Divergence）：用有限步 Gibbs 链近似难算的模型期望项。

---

**主逻辑链（严格三段论）**

**三段论 1：为什么“物理系统”可以做生成模型**
1. 大前提：任意热平衡系统的状态分布都满足玻尔兹曼形式 $p_{\text{eq}} \propto e^{-H/(k_B T)}$。
2. 小前提：若选择
$$
H^*(\mathbf{x}) = -k_B T \ln p_{\text{target}}(\mathbf{x}),
$$
则代回可得 $p_{\text{eq}}(\mathbf{x}) = p_{\text{target}}(\mathbf{x})$。
3. 结论：只要能实现合适的 $H$，让物理系统自然弛豫到平衡，就等价于“从目标分布采样”。

对应文档位置： [doc/GenerativeThermodynamicComputing.md](doc/GenerativeThermodynamicComputing.md#L19), [doc/GenerativeThermodynamicComputing.md](doc/GenerativeThermodynamicComputing.md#L27)

---

**三段论 2：为什么训练目标是 KL 最小化**
1. 大前提：$D_{\mathrm{KL}}(p_{\text{target}}\|p_{\theta}) \ge 0$，最小值 0 在 $p_{\theta} = p_{\text{target}}$ 时达到。
2. 小前提：把
$$
p_{\theta}(\mathbf{x}) = \frac{1}{Z(\theta)} e^{-H_{\theta}(\mathbf{x})/(k_B T)}
$$
代入 KL，展开得
$$
D_{\mathrm{KL}}
= -S_{\text{target}} + \frac{1}{k_B T}\langle H_{\theta} \rangle_{p_{\text{target}}} + \ln Z(\theta).
$$
其中 $-S_{\text{target}}$ 与参数 $\theta$ 无关。
3. 结论：最小化 KL 等价于最小化
$$
\mathcal{L}(\theta) = \langle H_{\theta} \rangle_{p_{\text{target}}} + k_B T \ln Z(\theta).
$$

对应文档位置： GenerativeThermodynamicComputing.md, GenerativeThermodynamicComputing.md

---

**三段论 3：为什么梯度是“数据期望 - 模型期望”**
大前提：对数配分函数导数恒等式
$$
\frac{\partial \ln Z}{\partial \theta_k}
= -\frac{1}{k_B T}\langle \frac{\partial H}{\partial \theta_k} \rangle_{p_{\theta}}.
$$

小前提：对 $\mathcal{L}$ 求导：
$$\frac{\partial \mathcal{L}}{\partial \theta_k} = \langle \frac{\partial H}{\partial \theta_k} \rangle_{p_{\text{target}}} + k_B T \frac{\partial \ln Z}{\partial \theta_k}.$$

把大前提代入即得
$$\frac{\partial \mathcal{L}}{\partial \theta_k} = \langle \frac{\partial H}{\partial \theta_k} \rangle_{\text{data}} - \langle \frac{\partial H}{\partial \theta_k} \rangle_{\text{model}}.$$

结论：训练本质是“把模型统计量拉向数据统计量匹配”。

对应文档与实现： GenerativeThermodynamicComputing.md, training.py, training.py

---

**三段论 4：为什么潜变量能显著增强表达力**
1. 大前提：边缘化潜变量会产生新的有效能量曲面：
$$
p(\mathbf{v}) = \int p(\mathbf{v}, \mathbf{z}) \, d\mathbf{z}
= \frac{1}{Z} e^{-F_{\text{eff}}(\mathbf{v})/(k_B T)},
$$
$$
F_{\text{eff}}(\mathbf{v})
= -k_B T \ln \int e^{-H(\mathbf{v}, \mathbf{z})/(k_B T)} \, d\mathbf{z}.
$$
2. 小前提：即便 $H(\mathbf v,\mathbf z)$ 在 $(\mathbf v,\mathbf z)$ 上形式简单，积分操作本身是非线性的，会在 $\mathbf v$ 空间产生复杂势面；例如高斯积分会带来新的二次项结构。
3. 结论：通过增加潜变量维度，可以用相对简单的物理单元逼近复杂数据分布（通用近似思想）。

对应文档位置： GenerativeThermodynamicComputing.md, GenerativeThermodynamicComputing.md, GenerativeThermodynamicComputing.md

---

**三段论 5：为什么 Langevin 更新公式是这样**
1. 大前提：过阻尼 Langevin 方程
$$
d\mathbf{x}_t = -\frac{1}{\gamma}\nabla H(\mathbf{x}_t)\,dt + \sqrt{\frac{2k_B T}{\gamma}}\,d\mathbf{W}_t
$$
的平稳分布是玻尔兹曼分布。
2. 小前提：Euler 离散化得到
$$\mathbf{x}_{t+\Delta t} = \mathbf{x}_t - \frac{\Delta t}{\gamma}\nabla H(\mathbf{x}_t) + \sqrt{\frac{2k_B T\Delta t}{\gamma}}\,\xi_t,\quad \xi_t \sim \mathcal{N}(0, I).$$
3. 结论：只要 $\Delta t$ 足够小、burn-in 足够长，离散链样本就逼近目标热平衡分布。

对应实现： sampling.py, sampling.py

---

**重点公式逐条精讲（尽量“高中生可懂”但不牺牲严谨性）**

1. 玻尔兹曼分布中的指数项 $e^{-H/(k_BT)}$  
解释：同温度下，能量每升高 $\Delta H$，相对概率按因子 $e^{-\Delta H/(k_BT)}$ 衰减。  
严谨点：这是“相对概率”规律，绝对概率还需由 $Z$ 归一化。

2. 配分函数 $Z$ 的作用  
解释：把“未归一化权重”变成合法概率。  
严谨点：$Z$ 同时决定热力学量（自由能等），训练难点也主要来自 $\ln Z$ 的梯度估计。

3. KL 展开到自由能目标  
解释：KL 的第一项只由真实数据分布决定，不受模型参数影响；真正可优化的是 $\langle H \rangle + k_B T\ln Z$。
严谨点：这是“同一最优解的等价目标”，不是近似替换。

4. 梯度差分形式  
解释：模型对某统计量“估高了”，就往下调；“估低了”，就往上调。  
严谨点：这是指数族/能量模型训练的核心恒等式，不依赖具体网络结构。

5. 连续 RBM 的条件高斯  
在二部图结构下，给定一侧后另一侧条件分布可简化为高斯（或近高斯），使 Gibbs 采样可执行。  
对应实现入口： rbm.py, rbm.py

6. 有效自由能代理  
代码里
$$
F_{\text{proxy}}(\mathbf{v}) = E_v(\mathbf{v}) - \sum_j \frac{field_j(\mathbf{v})^2}{2a_{h,j}}
$$
对应把隐藏变量近似高斯积分掉后的解析项。  
对应实现： rbm.py

---

**你这个项目实现中的“精确性边界”（很关键）**
1. 代码在 $p(h\mid v)$ 里注明“忽略 $h^4$ 项”做高斯近似，所以它是可计算近似，不是严格精确后验。见 rbm.py。
2. 因为有此近似，free energy 与 KL 指标在实现里是“代理量/单调 proxy”，不是严格真值。见 metrics.py, metrics.py。
3. 这不违背理论框架，但你应把“理论等价”与“数值近似训练”分开理解。

---

**一句话把整套方法扣紧**
Generative Thermodynamic Computing 的严格核心是：先用能量模型把“生成”转写成“平衡分布匹配”，再用 KL/自由能梯度把参数学习成“数据统计 = 模型统计”，最后利用真实或模拟热动力学的自然随机性来直接采样。理论上闭环，工程上通过 CD 和条件近似实现可训练性。  

如果你愿意，我下一步可以按你 notebook 的四个实验（Langevin、Swiss Roll、高斯混合、通用近似验证）逐个做“公式-代码-结果解读”对照版。