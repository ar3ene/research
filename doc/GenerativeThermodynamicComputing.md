下面的拆解基于论文 *Generative Thermodynamic Computing*（*Physical Review Letters* 136, 037101, published on January 20, 2026）原文。目标有两个：

1. 先把所有复杂名词定义清楚。
2. 再按“三段论 + 逐式推导”的方式，把论文从公式 (1) 到公式 (14) 串成一条严密逻辑链。

全文默认读者是理科高中生：解释尽量直白，但不牺牲数学精确性。

**一、先定义名词**

1. 生成模型：从随机噪声出发，生成“像真实数据”的结果的模型。本文里的数据是手写数字图像。
2. 热力学计算：不是完全用数字逻辑门做计算，而是让一个真实物理系统在热噪声中自然演化，把“答案”编码在系统状态里。
3. Langevin 动力学：一种“确定性漂移 + 随机热噪声”的动力学方程。
4. 势能 \(V_\theta(x)\)：给系统每个状态 \(x\) 赋一个能量的函数。系统倾向往势能较低的方向运动。
5. 参数 \(\theta\)：论文里主要是耦合 \(J_{ij}\) 和偏置 \(b_i\)。训练的工作就是调整这些参数。
6. noising：把原本有结构的数据逐渐扰乱成噪声。
7. denoising：把噪声逐渐恢复成有结构的数据。
8. 正向轨迹：这里指“数字逐步退化成噪声”的轨迹。
9. 反向轨迹：把正向轨迹时间倒过来，也就是“噪声逐步恢复成数字”的轨迹。
10. 似然 / 轨迹概率：某个动力学模型生成一条给定轨迹的概率。
11. 热耗散 \(Q\)：系统向环境释放的热量。热耗散越大，通常说明过程越不可逆。
12. 熵产生：衡量不可逆性的量。熵产生越大，过程越偏离“可逆”。
13. 过阻尼：惯性可以忽略，系统基本上由阻力、势能梯度和噪声共同决定运动。
14. Onsager-Machlup 作用：给“整条随机轨迹”赋概率的一种统计物理方法。在这篇论文里，它最终变成一步步高斯概率的乘积。

**二、论文的核心主张**

这篇论文的核心主张可以浓缩成一句话：

普通扩散模型是“用数字神经网络学习反向去噪”；这篇论文改成“训练一个受热噪声驱动的物理系统，让它本身的自然动力学去完成反向去噪”。

也就是说，生成规则不再主要写在神经网络权重里，而是写在物理系统的能量地形里。

**三、主逻辑链，按三段论展开**

1. 为什么物理系统可以充当生成模型  
大前提：如果一个物理系统会在势能作用下趋向某些结构，同时又保留热噪声带来的随机性，那么它就能把“纯噪声初态”演化成“多样但有结构的终态”。  
小前提：论文构造了一个带可训练势能 \(V_\theta(x)\) 的 Langevin 系统，并专门训练它去高概率地产生 noising 轨迹的反向。  
结论：这个物理系统可以作为生成模型，从噪声出发生成结构。

2. 为什么训练目标是“最大化反向轨迹概率”  
大前提：如果一个模型更可能生成某条反向轨迹，那么这个模型的动力学就更接近正确的去噪动力学。  
小前提：训练数据正是“数字逐渐退化成噪声”的正向 noising 轨迹。  
结论：最大化反向轨迹概率，就等价于学会“从噪声恢复数字”。

3. 为什么公式里自然会出现平方项  
大前提：高斯随机变量的概率密度必然含有平方指数项 \(e^{-z^2/2}\)。  
小前提：Langevin 方程离散后，每一步真实位移和确定性漂移之间的差，正好就是一个高斯随机变量。  
结论：一步轨迹的负对数概率必然是“残差平方和”。

4. 为什么训练又等价于“最小热耗散”  
大前提：在非平衡统计物理中，正向轨迹和反向轨迹的概率比与热耗散、熵产生直接相关。  
小前提：论文推导出，最大化反向轨迹概率等价于减小对应生成过程的热排放。  
结论：这个学习过程不仅是在拟合数据，也是在寻找“最可逆、最省热”的生成动力学。

5. 为什么作者认为它可能比数字神经网络更省能  
大前提：数字神经网络的主要能耗来自大量乘加运算。  
小前提：如果去噪过程由物理系统自然演化完成，就不需要在数字芯片上一步步执行大规模乘加。  
结论：若能落地为模拟硬件，能耗可能远低于数字去噪网络。

**四、核心公式与逐式推导**

这一节是全文重点。所有块级公式统一用 `$$ ... $$`。

**4.1 公式 (1)：连续时间的 Langevin 方程**

论文的动力学起点是

$$
\dot x_i = -\mu \,\partial_i V_\theta(x) + \sqrt{2\mu k_B T}\,\eta_i(t).
$$

这里每个符号都要先读清楚：

- \(x_i\)：第 \(i\) 个自由度，例如一个电压、位移或相位。
- \(\dot x_i\)：\(x_i\) 对时间的导数，即变化速度。
- \(\partial_i V_\theta(x)\)：势能对 \(x_i\) 的偏导数。
- \(-\mu \,\partial_i V_\theta(x)\)：确定性漂移项，表示系统沿着“势能下降方向”运动。
- \(\mu\)：迁移率，控制系统对力的响应速度。
- \(k_B T\)：热能尺度。温度越高，热噪声越强。
- \(\eta_i(t)\)：均值为 0、方差为 1 的白噪声。

这条方程表示：

系统既不会只做“纯粹下坡”，也不会只做“纯随机游走”；它是在“势能拉回”与“热噪声扰动”之间共同演化。

**4.2 公式 (2)：势能函数到底长什么样**

论文取的势能是

$$V_\theta(x)=\sum_{i=1}^N \left(J_2 x_i^2 + J_4 x_i^4\right)
+
\sum_{i=1}^N b_i x_i
+
\sum_{(ij)} J_{ij} x_i x_j.
$$

这三部分各有不同角色：

- \(\sum_i (J_2 x_i^2 + J_4 x_i^4)\)：每个单元自己的局部势阱。
- \(\sum_i b_i x_i\)：偏置，相当于外部对每个单元施加的推力方向。
- \(\sum_{(ij)} J_{ij} x_i x_j\)：单元之间的耦合，决定不同位置如何相互影响。

精确点说：

- 若 \(J_4 > 0\)，则当 \(|x_i| \to \infty\) 时，四次项会让势能上升，保证体系稳定。
- 若 \(J_2 > 0\)，局部势阱是单稳态。
- 若 \(J_2 < 0\)，则局部会偏向双稳态。

论文这里选择 \(J_2 > 0\)、\(J_4 > 0\)。

**4.3 从公式 (1) 到公式 (3)：离散化**

连续方程无法直接拿来算一步概率，所以先把时间离散化。取时间步长为 \(\Delta t\)，用 Euler-Maruyama 离散：

$$x_i(t+\Delta t)=x_i(t)- \mu \,\partial_i V_\theta(x)\,\Delta t+ \sqrt{2\mu k_B T \Delta t}\,\eta_i.
$$

这就是论文的公式 (3)。

它的结构非常简单：

$$\text{下一步状态}=\text{当前状态}+ \text{确定性漂移}+ \text{随机噪声}.
$$

注意噪声项前面是 \(\sqrt{\Delta t}\) 而不是 \(\Delta t\)。  
这是布朗运动的基本尺度律，不是随便写出来的。

**4.4 从公式 (3) 到公式 (4)：把噪声显式解出来**

定义一步位移

$$
\Delta x_i \equiv x_i(t+\Delta t) - x_i(t).
$$

把上面的离散方程改写成

$$\Delta x_i=- \mu \,\partial_i V_\theta(x)\,\Delta t+ \sqrt{2\mu k_B T \Delta t}\,\eta_i.
$$

把漂移项移到左边：

$$\Delta x_i + \mu \,\partial_i V_\theta(x)\,\Delta t=
\sqrt{2\mu k_B T \Delta t}\,\eta_i.
$$

两边同时除以 \(\sqrt{2\mu k_B T \Delta t}\)，得到

$$\eta_i=\frac{
\Delta x_i + \mu \,\partial_i V_\theta(x)\,\Delta t
}{
\sqrt{2\mu k_B T \Delta t}
}.
$$

这就是论文公式 (4)。

它的物理含义是：

如果你已经观测到一步真实位移 \(\Delta x_i\)，那就能反推出“要解释这一步，系统需要抽到多大的噪声”。

**4.5 公式 (5) 与公式 (6)：一步概率为什么是残差平方和**

因为每个 \(\eta_i\) 都是独立标准高斯变量，所以它们联合概率密度是

$$P_\theta^{\text{step}}(\Delta x)=
(2\pi)^{-N/2}
\prod_{i=1}^N \exp\left(-\eta_i^2/2\right).
$$

这就是论文公式 (5) 的内容。

把乘积写成指数和：

$$P_\theta^{\text{step}}(\Delta x)=
(2\pi)^{-N/2}
\exp\left(
-\frac12 \sum_{i=1}^N \eta_i^2
\right).
$$

再代入刚刚得到的

$$\eta_i=\frac{\Delta x_i + \mu \,\partial_i V_\theta(x)\,\Delta t
}{
\sqrt{2\mu k_B T \Delta t}
},
$$

就有

$$\eta_i^2=\frac{
\left[\Delta x_i + \mu \,\partial_i V_\theta(x)\,\Delta t\right]^2
}{
2\mu k_B T \Delta t
}.
$$

于是

$$P_\theta^{\text{step}}(\Delta x)=(2\pi)^{-N/2}
\exp\left(
-\frac12
\sum_{i=1}^N
\frac{
\left[\Delta x_i + \mu \,\partial_i V_\theta(x)\,\Delta t\right]^2
}{
2\mu k_B T \Delta t
}
\right).
$$

把前面的 \(1/2\) 乘进去：

$$P_\theta^{\text{step}}(\Delta x)=(2\pi)^{-N/2}
\exp\left(
-\sum_{i=1}^N
\frac{
\left[\Delta x_i + \mu \,\partial_i V_\theta(x)\,\Delta t\right]^2
}{
4\mu k_B T \Delta t
}
\right).
$$

对两边取负对数：

$$-\ln P_\theta^{\text{step}}(\Delta x)=\frac{N}{2}\ln(2\pi)
+
\sum_{i=1}^N
\frac{
\left[\Delta x_i + \mu \,\partial_i V_\theta(x)\,\Delta t\right]^2
}{
4\mu k_B T \Delta t
}.
$$

论文说“up to an unimportant constant term”，就是把与参数 \(\theta\) 无关的常数 \(\frac{N}{2}\ln(2\pi)\) 忽略掉。于是得到公式 (6)：

$$-\ln P_\theta^{\text{step}}(\Delta x)=\sum_{i=1}^N
\frac{
\left[\Delta x_i + \mu \,\partial_i V_\theta(x)\,\Delta t\right]^2
}{
4\mu k_B T \Delta t
}.
$$

这就是“残差平方和”的来源。  
残差正是

$$
\Delta x_i - \left(-\mu \,\partial_i V_\theta(x)\,\Delta t\right).
$$

也就是“实际位移”减去“系统本来最想做出的漂移”。

所以：

- 如果一步实际位移几乎和自然漂移一致，那么残差小，这一步就很可能发生。
- 如果一步实际位移和自然漂移差很远，就必须依赖一次很罕见的大噪声，这一步概率就小。

**4.6 公式 (7)：为什么反向一步里会出现 \(-\Delta x_i\)**

正向一步是

$$
x \to x + \Delta x.
$$

反向一步是从终点走回起点，也就是

$$x' \to x,\qquad x' = x + \Delta x.
$$

因此反向位移等于

$$
x - x' = -\Delta x.
$$

只要把正向公式中的“位移”换成反向位移，并且把漂移项评价在反向步的起点 \(x'\) 上，就得到论文公式 (7)：

$$-\ln \tilde P_\theta^{\text{step}}(\Delta x)=\sum_{i=1}^N
\frac{
\left[-\Delta x_i + \mu \,\partial_i V_\theta(x')\,\Delta t\right]^2
}{
4\mu k_B T \Delta t
}.
$$

这条式子是全文训练的核心，因为训练并不是在优化正向 noising，而是在优化它的反向 denoising。

**4.7 公式 (8) 和公式 (9)：为什么训练就是梯度上升**

设整条正向轨迹为

$$
\omega = \{x(t_k)\}_{k=0}^K.
$$

那么对应的反向轨迹记为

$$
\tilde \omega = \{x(t_{K-k})\}_{k=0}^K.
$$

如果各时间步按 Markov 方式相乘，那么整条反向轨迹的对数概率就是各步对数概率之和。因此训练时对每个参数做梯度上升：

$$
J_{ij}
\leftarrow
J_{ij}
+
\alpha \sum_{k=1}^K
\frac{\partial}{\partial J_{ij}}
\ln \tilde P_\theta^{\text{step}}[\Delta x(t_k)],
$$

$$
b_i
\leftarrow
b_i
+
\alpha \sum_{k=1}^K
\frac{\partial}{\partial b_i}
\ln \tilde P_\theta^{\text{step}}[\Delta x(t_k)].
$$

这就是论文公式 (8)、(9)。

逻辑很直接：

- 如果某个参数的增大会提高反向轨迹概率，就把它调大。
- 如果某个参数的增大会降低反向轨迹概率，就把它调小。

**4.8 公式 (12)：先把势能梯度算清楚**

后面求导之前，先把 \(\partial_i V_\theta(x)\) 写出来。

从

$$V_\theta(x)=\sum_{i=1}^N \left(J_2 x_i^2 + J_4 x_i^4\right)
+
\sum_{i=1}^N b_i x_i
+
\sum_{(ij)} J_{ij} x_i x_j
$$

对 \(x_i\) 求偏导：

1. 对 \(J_2 x_i^2\) 求导，得到 \(2J_2 x_i\)。
2. 对 \(J_4 x_i^4\) 求导，得到 \(4J_4 x_i^3\)。
3. 对 \(b_i x_i\) 求导，得到 \(b_i\)。
4. 对耦合项 \(\sum_{(ij)} J_{ij}x_ix_j\) 求导时，只保留含 \(x_i\) 的那些项，于是得到 \(\sum_{j \in \mathcal N(i)} J_{ij}x_j\)。

因此

$$\partial_i V_\theta(x)=2J_2 x_i+ 4J_4 x_i^3+ b_i+ \sum_{j \in \mathcal N(i)} J_{ij}x_j.
$$

这就是论文公式 (12)。

**4.9 公式 (11)：对偏置 \(b_i\) 的梯度，逐式推导**

为了简化写法，先定义反向一步的残差

$$
r_i
\equiv
-\Delta x_i + \mu \,\partial_i V_\theta(x')\,\Delta t.
$$

于是公式 (7) 可以写成

$$-\ln \tilde P_\theta^{\text{step}}(\Delta x)=\sum_{i=1}^N \frac{r_i^2}{4\mu k_B T \Delta t}.
$$

设

$$S(\theta) \equiv -\ln \tilde P_\theta^{\text{step}}(\Delta x).$$

那么

$$S(\theta)=
\sum_{l=1}^N \frac{r_l^2}{4\mu k_B T \Delta t}.
$$

现在对 \(b_i\) 求导：

$$\frac{\partial S}{\partial b_i}=\sum_{l=1}^N
\frac{1}{4\mu k_B T \Delta t}
\cdot 2r_l
\cdot \frac{\partial r_l}{\partial b_i}.
$$

而

$$r_l=-\Delta x_l + \mu \,\partial_l V_\theta(x')\,\Delta t,
$$

所以

$$\frac{\partial r_l}{\partial b_i}=
\mu \Delta t \,
\frac{\partial}{\partial b_i}
\bigl(\partial_l V_\theta(x')\bigr).
$$

由公式 (12) 可知，\(\partial_l V_\theta\) 里只有当 \(l=i\) 时才含有 \(b_i\)，而且系数就是 1，因此

$$\frac{\partial}{\partial b_i}\bigl(\partial_l V_\theta(x')\bigr)=\delta_{li},
$$

其中 \(\delta_{li}\) 是 Kronecker delta：当 \(l=i\) 时等于 1，否则等于 0。

代回去：

$$\frac{\partial r_l}{\partial b_i}=
\mu \Delta t \,\delta_{li}.
$$

于是

$$\frac{\partial S}{\partial b_i}=
\sum_{l=1}^N
\frac{1}{4\mu k_B T \Delta t}
\cdot 2r_l \cdot \mu \Delta t \,\delta_{li}.
$$

因为 \(\delta_{li}\) 会把求和直接挑成 \(l=i\)，所以

$$\frac{\partial S}{\partial b_i}=\frac{2r_i \mu \Delta t}{4\mu k_B T \Delta t}=
\frac{r_i}{2k_B T}.
$$

又因为 \(S = -\ln \tilde P_\theta^{\text{step}}\)，所以

$$-\frac{\partial}{\partial b_i}\ln \tilde P_\theta^{\text{step}}(\Delta x)=
\frac{r_i}{2k_B T}.
$$

把 \(r_i\) 写回去，就得到论文公式 (11)：

$$-\frac{\partial}{\partial b_i}\ln \tilde P_\theta^{\text{step}}(\Delta x)=
\frac{
-\Delta x_i + \mu \,\partial_i V_\theta(x')\,\Delta t
}{
2k_B T
}.
$$

这条式子的含义是：

如果第 \(i\) 个单元在反向一步里“老是走偏”，那就通过调节局部偏置 \(b_i\) 去修正它的平均驱动力。

**4.10 公式 (10)：对耦合 \(J_{ij}\) 的梯度，逐式推导**

同样从

$$S(\theta)=\sum_{l=1}^N \frac{r_l^2}{4\mu k_B T \Delta t}
$$

开始，对 \(J_{ij}\) 求导：

$$
\frac{\partial S}{\partial J_{ij}}=\sum_{l=1}^N
\frac{1}{4\mu k_B T \Delta t}
\cdot 2r_l
\cdot \frac{\partial r_l}{\partial J_{ij}}.
$$

而

$$\frac{\partial r_l}{\partial J_{ij}}=\mu \Delta t \,
\frac{\partial}{\partial J_{ij}}
\bigl(\partial_l V_\theta(x')\bigr).
$$

现在关键是算

$$
\frac{\partial}{\partial J_{ij}}
\bigl(\partial_l V_\theta(x')\bigr).
$$

由公式 (12) 可知：

$$\partial_l V_\theta(x')=2J_2 x'_l+ 4J_4 {x'_l}^3+ b_l+ \sum_{m \in \mathcal N(l)} J_{lm}x'_m.
$$

只有最后那项含 \(J_{ij}\)。而 \(J_{ij}\) 会在两种情况下出现：

- 当 \(l=i\) 时，会给出一项 \(x'_j\)。
- 当 \(l=j\) 时，会给出一项 \(x'_i\)。

因此

$$\frac{\partial}{\partial J_{ij}}\bigl(\partial_l V_\theta(x')\bigr)=\delta_{li} x'_j + \delta_{lj} x'_i.
$$

于是

$$\frac{\partial r_l}{\partial J_{ij}}=\mu \Delta t \left(\delta_{li} x'_j + \delta_{lj} x'_i\right).
$$

代回 \(S\) 的导数：

$$\frac{\partial S}{\partial J_{ij}}=\sum_{l=1}^N
\frac{1}{4\mu k_B T \Delta t}
\cdot 2r_l
\cdot \mu \Delta t \left(\delta_{li} x'_j + \delta_{lj} x'_i\right).
$$

把常数约掉：

$$\frac{\partial S}{\partial J_{ij}}=\sum_{l=1}^N
\frac{r_l}{2k_B T}
\left(\delta_{li} x'_j + \delta_{lj} x'_i\right).
$$

Kronecker delta 再次把求和收缩掉：

$$\frac{\partial S}{\partial J_{ij}}=\frac{r_i x'_j + r_j x'_i}{2k_B T}.
$$

如果时间步 \(\Delta t\) 很小，那么

$$x'_i = x_i + \Delta x_i = x_i + O(\sqrt{\Delta t}).
$$

在论文的离散精度下，可以把 \(x'_i\) 和 \(x_i\) 的差吸收到高阶小量里，于是写成

$$\frac{\partial S}{\partial J_{ij}}\approx\frac{r_i x_j + r_j x_i}{2k_B T}.
$$

由于 \(S = -\ln \tilde P_\theta^{\text{step}}\)，所以

$$-\frac{\partial}{\partial J_{ij}}\ln \tilde P_\theta^{\text{step}}(\Delta x)=\frac{r_i x_j + r_j x_i}{2k_B T}.
$$

再把 \(r_i\) 和 \(r_j\) 展开：

$$-\frac{\partial}{\partial J_{ij}}\ln \tilde P_\theta^{\text{step}}(\Delta x)=\frac{
\left[-\Delta x_i + \mu \,\partial_i V_\theta(x')\,\Delta t\right]x_j
+
\left[-\Delta x_j + \mu \,\partial_j V_\theta(x')\,\Delta t\right]x_i
}{
2k_B T
}.
$$

这就是论文公式 (10)。

它的直观意义是：

- 如果第 \(i\) 个单元的误差和第 \(j\) 个单元当前状态经常同时出现，就应调节 \(J_{ij}\)。
- 这非常像“误差 × 输入”的学习法则，但这里不是拍脑袋设定的，而是从轨迹似然严格推出来的。

**4.11 公式 (13)：为什么轨迹概率比会联系到热耗散**

这部分是论文最深的物理意义所在。  
论文比较的是两个量：

- \(P_0^{\text{step}}(\Delta x)\)：参考系统（参数记为 \(\theta=0\)）生成正向一步的概率。
- \(\tilde P_\theta^{\text{step}}(\Delta x)\)：训练系统生成反向一步的概率。

根据公式 (6) 和公式 (7)，去掉无关常数后：

$$\ln \frac{P_0^{\text{step}}(\Delta x)}{\tilde P_\theta^{\text{step}}(\Delta x)}=\sum_{i=1}^N
\frac{\left[-\Delta x_i + \mu \,\partial_i V_\theta(x')\,\Delta t\right]^2
}{
4\mu k_B T \Delta t
}-\sum_{i=1}^N
\frac{
\left[\Delta x_i + \mu \,\partial_i V_0(x)\,\Delta t\right]^2
}{
4\mu k_B T \Delta t
}.
$$

为了简洁，记

$$
A_i \equiv -\Delta x_i + \mu \,\partial_i V_\theta(x')\,\Delta t,
\qquad
B_i \equiv \Delta x_i + \mu \,\partial_i V_0(x)\,\Delta t.
$$

那么上式变成

$$\ln \frac{P_0^{\text{step}}(\Delta x)}{\tilde P_\theta^{\text{step}}(\Delta x)}=
\sum_{i=1}^N \frac{A_i^2 - B_i^2}{4\mu k_B T \Delta t}.
$$

现在把平方展开。

先看 \(A_i^2\)：

$$A_i^2=\left(-\Delta x_i + \mu \,\partial_i V_\theta(x')\,\Delta t\right)^2=\Delta x_i^2- 2\mu \Delta t \,\Delta x_i \,\partial_i V_\theta(x')+ \mu^2 \Delta t^2 \left[\partial_i V_\theta(x')\right]^2.
$$

再看 \(B_i^2\)：

$$B_i^2=\left(\Delta x_i + \mu \,\partial_i V_0(x)\,\Delta t\right)^2=\Delta x_i^2+ 2\mu \Delta t \,\Delta x_i \,\partial_i V_0(x)+ \mu^2 \Delta t^2 \left[\partial_i V_0(x)\right]^2.
$$

两者相减：

$$
A_i^2 - B_i^2=-2\mu \Delta t \,\Delta x_i \,\partial_i V_\theta(x')-2\mu \Delta t \,\Delta x_i \,\partial_i V_0(x)+ \mu^2 \Delta t^2\left(\left[\partial_i V_\theta(x')\right]^2- \left[\partial_i V_0(x)\right]^2
\right).
$$

除以 \(4\mu k_B T \Delta t\)：

$$\frac{A_i^2 - B_i^2}{4\mu k_B T \Delta t}=-\frac{
\Delta x_i \,\partial_i V_\theta(x')+ \Delta x_i \,\partial_i V_0(x)
}{
2k_B T
}+\frac{
\mu \Delta t
}{
4k_B T
}
\left(
\left[\partial_i V_\theta(x')\right]^2- \left[\partial_i V_0(x)\right]^2
\right).
$$

当 \(\Delta t \to 0\) 时，最后那项是 \(O(\Delta t)\) 的高阶小量，可以忽略，于是

$$
\ln \frac{P_0^{\text{step}}(\Delta x)}{\tilde P_\theta^{\text{step}}(\Delta x)}
\approx
-\sum_{i=1}^N
\frac{
\Delta x_i \,\partial_i V_\theta(x')+ \Delta x_i \,\partial_i V_0(x)
}{
2k_B T
}.
$$

进一步地，因为 \(x' = x + \Delta x\)，而 \(\Delta x = O(\sqrt{\Delta t})\)，所以

$$\partial_i V_\theta(x')=\partial_i V_\theta(x)+ O(\sqrt{\Delta t}).
$$

乘上 \(\Delta x_i\) 后，误差是 \(O(\Delta t)\)，仍然可以归入高阶项，因此

$$\ln \frac{P_0^{\text{step}}(\Delta x)}{\tilde P_\theta^{\text{step}}(\Delta x)}
\approx-\sum_{i=1}^N
\frac{
\Delta x_i \,\partial_i V_\theta(x)+ \Delta x_i \,\partial_i V_0(x)
}{
2k_B T
}.
$$

在过阻尼系统里，如果这一步没有外部做功，那么一步热耗散可写成

$$\Delta Q=\sum_{i=1}^N \partial_i V(x)\,\Delta x_i.
$$

所以对参考系统和训练系统分别有

$$\Delta Q_0=\sum_{i=1}^N \partial_i V_0(x)\,\Delta x_i,
\qquad
\Delta Q_\theta=\sum_{i=1}^N \partial_i V_\theta(x)\,\Delta x_i.
$$

代回就得到论文公式 (13)：

$$
\ln \frac{P_0^{\text{step}}(\Delta x)}{\tilde P_\theta^{\text{step}}(\Delta x)}
\approx
-\frac{\Delta Q_0 + \Delta Q_\theta}{2k_B T}.
$$

这一步非常关键。  
它告诉你：反向轨迹概率越大，意味着对应热耗散越小。

**4.12 公式 (14)：从“一步”累加到“整条轨迹”**

整条轨迹由很多小步组成。对每一步的对数概率比求和：

$$\ln \frac{P_0[\omega]}{P_\theta[\tilde\omega]}=\sum_{k=1}^K
\ln \frac{P_0^{\text{step}}[\Delta x(t_k)]}{\tilde P_\theta^{\text{step}}[\Delta x(t_k)]}.
$$

再用上面的近似式：

$$\ln \frac{P_0[\omega]}{P_\theta[\tilde\omega]}\approx-\sum_{k=1}^K \frac{\Delta Q_0(t_k) + \Delta Q_\theta(t_k)}{2k_B T}.
$$

把各步热量加总成总热量：

$$Q_0(\omega) = \sum_{k=1}^K \Delta Q_0(t_k),\qquad Q_\theta(\omega) = \sum_{k=1}^K \Delta Q_\theta(t_k).
$$

并记

$$
\beta \equiv \frac{1}{k_B T},
$$

就得到论文公式 (14)：

$$\ln \frac{P_0[\omega]}{P_\theta[\tilde\omega]}=-\frac12 \left[\beta Q_0(\omega) + \beta Q_\theta(\omega)\right].
$$

这条式子的物理解释是：

- 参考过程 \(P_0[\omega]\) 是固定的。
- 所以训练时最大化 \(P_\theta[\tilde\omega]\)，等价于减小 \(Q_\theta(\omega)\)。
- 而热量在时间反演下变号，因此也等价于减小训练系统沿反向生成轨迹 \(\tilde\omega\) 的热排放。

这就是论文为什么说：  
训练是在寻找“最热力学可逆”的去噪动力学。

**五、把整套数学链条压缩成一句话**

这篇论文的数学主线可以压缩为：

1. 用 Langevin 方程描述一个受热噪声驱动的物理系统。
2. 把一步位移写成“漂移 + 高斯噪声”。
3. 因为噪声是高斯的，所以一步轨迹概率就是残差平方和。
4. 训练时最大化反向轨迹概率，于是得到对 \(J_{ij}\)、\(b_i\) 的梯度公式。
5. 轨迹概率比又和热耗散相连，所以训练同时也在最小化反向生成过程的热耗散。

**六、它和扩散模型、玻尔兹曼机到底是什么关系**

1. 和扩散模型的相同点：都是先想“正向加噪”，再学“反向去噪”。
2. 和扩散模型的不同点：普通扩散模型把反向规则写在数字神经网络里；这里把反向规则写在物理系统的势能和耦合里。
3. 和玻尔兹曼机的相同点：都借用统计物理和能量函数。
4. 和玻尔兹曼机的不同点：玻尔兹曼机通常依赖平衡分布；这里强调的是指定时间上的非平衡轨迹，不要求系统先达到平衡。

**七、这篇论文的贡献与边界**

贡献很明确：

- 它提出了一种真正“物理原生”的生成框架，不靠数字神经网络逐步控制去噪。
- 它把“学会生成”与“降低热耗散、降低熵产生”联系起来，物理意义非常强。
- 数学链条是闭合的：动力学方程、轨迹概率、参数梯度、热力学解释来自同一套框架。

但边界也要看清：

- 论文的数值演示规模很小，只用了 3 个 MNIST 数字，主要是 proof of principle。
- 论文展示的是数字模拟，不是真正落地的模拟硬件。
- 节能优势目前还是建立在理论估算和模拟结果上，真正的硬件实现仍是后续问题。

**八、一句话总结**

这篇论文最核心的思想不是“做出一个更强的扩散模型”，而是把“生成”重新定义为：

训练一个受热噪声驱动的物理系统，使它以尽可能高的概率、尽可能低的热耗散，去执行 noising 过程的时间反演。
