# 生成式热力学计算（Generative Thermodynamic Computing）

## 一、论文核心思想

本文提出了一种**用物理热力学系统进行概率生成建模**的框架。核心思路是：

> 与其用数字计算机模拟概率分布（如扩散模型、玻尔兹曼机等），不如直接让一个真实的物理系统自然地**弛豫（relax）**到其热平衡态，利用物理系统的热涨落进行采样。

关键创新：论文给出了一套**严格的数学方法**，能够设计物理系统的能量函数（哈密顿量），使其平衡分布**精确等于**我们希望生成的目标概率分布。

---

## 二、数学框架

### 2.1 基本设定

考虑一个经典热力学系统，状态由连续变量 $\mathbf{x} = (x_1, x_2, \dots, x_n)$ 描述。在温度 $T$ 下，系统的平衡分布是**玻尔兹曼分布**：

$$p_{\text{eq}}(\mathbf{x}) = \frac{1}{Z} e^{-H(\mathbf{x}) / k_B T}$$

其中 $H(\mathbf{x})$ 是哈密顿量（能量函数），$Z = \int e^{-H(\mathbf{x})/k_BT} d\mathbf{x}$ 是配分函数。

**核心问题**：给定目标分布 $p_{\text{target}}(\mathbf{x})$，如何选择 $H(\mathbf{x})$ 使得 $p_{\text{eq}} = p_{\text{target}}$？

**理论答案**：

$$H(\mathbf{x}) = -k_B T \ln p_{\text{target}}(\mathbf{x})$$

但实际物理硬件能实现的能量函数形式有限制。

### 2.2 硬件约束与可实现的哈密顿量

实际物理系统（如耦合RLC电路）能实现的哈密顿量通常是**受限形式**：

$$H(\mathbf{x}) = \sum_i V_i(x_i) + \sum_{i<j} K_{ij}(x_i, x_j)$$

即单体势能 $V_i$ 加两体耦合 $K_{ij}$。

最简单（"vanilla"）情况：

$$H(\mathbf{x}) = \sum_i \left(\frac{a_i}{2} x_i^2 + \frac{b_i}{4} x_i^4\right) + \sum_{i<j} \frac{J_{ij}}{2}(x_i - x_j)^2$$

可调参数为 $\{a_i, b_i, J_{ij}\}$。

### 2.3 核心数学方法：KL散度最小化

最小化**KL散度**：

$$D_{\text{KL}}(p_{\text{target}} \| p_H) = \int p_{\text{target}}(\mathbf{x}) \ln \frac{p_{\text{target}}(\mathbf{x})}{p_H(\mathbf{x})} \, d\mathbf{x}$$

展开后：

$$D_{\text{KL}} = \underbrace{-\int p_{\text{target}} \ln p_{\text{target}} \, d\mathbf{x}}_{-S_{\text{target}}} + \frac{1}{k_BT}\underbrace{\int p_{\text{target}} H(\mathbf{x}) \, d\mathbf{x}}_{\langle H \rangle_{\text{target}}} + \ln Z$$

最小化KL散度等价于最小化**变分自由能**：

$$\mathcal{L}(\boldsymbol{\theta}) = \langle H_{\boldsymbol{\theta}}(\mathbf{x}) \rangle_{p_{\text{target}}} + k_B T \ln Z(\boldsymbol{\theta})$$

### 2.4 梯度计算

对参数 $\theta_k$ 求梯度：

$$\frac{\partial \mathcal{L}}{\partial \theta_k} = \left\langle \frac{\partial H}{\partial \theta_k} \right\rangle_{p_{\text{target}}} - \left\langle \frac{\partial H}{\partial \theta_k} \right\rangle_{p_H}$$

梯度等于某量在**目标分布**下的期望值减去它在**当前模型分布**下的期望值。

- 第一项 $\langle \cdot \rangle_{p_{\text{target}}}$：从训练数据统计平均
- 第二项 $\langle \cdot \rangle_{p_H}$：让物理系统弛豫到平衡后采样计算

若哈密顿量中参数线性出现（$H = \sum_k \theta_k \phi_k(\mathbf{x})$），则：

$$\frac{\partial \mathcal{L}}{\partial \theta_k} = \langle \phi_k \rangle_{\text{data}} - \langle \phi_k \rangle_{\text{model}}$$

### 2.5 充分统计量与精确表示条件

**定理**：当且仅当哈密顿量中的特征函数 $\{\phi_k(\mathbf{x})\}$ 构成目标分布的**充分统计量**时，玻尔兹曼分布可以精确表示目标分布。

若目标分布是指数族：

$$p_{\text{target}}(\mathbf{x}) = h(\mathbf{x}) \exp\left(\sum_k \eta_k \phi_k(\mathbf{x}) - A(\boldsymbol{\eta})\right)$$

令 $H(\mathbf{x}) = -k_BT \sum_k \theta_k \phi_k(\mathbf{x})$，取 $\theta_k = \eta_k$ 即可精确匹配。

---

## 三、潜变量与生成模型

### 3.1 潜变量架构

引入**潜变量** $\mathbf{z}$，将物理变量分为：

- **可见变量** $\mathbf{v}$：对应数据维度
- **潜变量** $\mathbf{z}$：辅助隐藏维度

联合分布：

$$p(\mathbf{v}, \mathbf{z}) = \frac{1}{Z} e^{-H(\mathbf{v}, \mathbf{z})/k_BT}$$

边缘分布：

$$p(\mathbf{v}) = \int p(\mathbf{v}, \mathbf{z}) \, d\mathbf{z} = \frac{1}{Z} e^{-F_{\text{eff}}(\mathbf{v})/k_BT}$$

**有效自由能**：

$$F_{\text{eff}}(\mathbf{v}) = -k_BT \ln \int e^{-H(\mathbf{v}, \mathbf{z})/k_BT} d\mathbf{z}$$

即使每个物理变量只有简单势能和耦合，边缘化潜变量后的有效自由能可以是**任意复杂的非线性函数**。

### 3.2 与受限玻尔兹曼机（RBM）的类比

双线性耦合哈密顿量：

$$H(\mathbf{v}, \mathbf{z}) = \sum_i V_i(v_i) + \sum_j V_j(z_j) + \sum_{i,j} J_{ij} v_i z_j$$

等价于连续变量版本的**受限玻尔兹曼机**（RBM）。二部图结构使条件分布的采样可解析。

---

## 四、训练方法

### 4.1 对比散度（Contrastive Divergence, CD）

1. 从训练样本 $\mathbf{v}^{(0)}$ 出发
2. 执行 $k$ 步Gibbs采样（物理上对应部分弛豫）
3. 用 $k$ 步后的样本近似 $\langle \cdot \rangle_{p_H}$

### 4.2 物理Gibbs采样

对于连续变量RBM结构，条件分布是高斯的：

$$p(z_j | \mathbf{v}) \propto \exp\left(-\frac{(z_j - \mu_j(\mathbf{v}))^2}{2\sigma_j^2}\right)$$

其中均值 $\mu_j(\mathbf{v}) = -\frac{\sum_i J_{ij} v_i}{a_j^{(z)}}$，方差 $\sigma_j^2 = \frac{k_BT}{a_j^{(z)}}$。

---

## 五、物理实现

### 5.1 耦合非线性振荡器

物理实现为**耦合RLC电路**：

- 每个节点：**非线性LC振荡器**（含线性电感和非线性电容），浸泡在热噪声中
- 节点间：**线性耦合器**（电阻网络或互感器）
- 电容器两端电压 $V_i$ 对应变量 $x_i$

能量函数：

$$H = \sum_i \left(\frac{1}{2}C_i^{(1)} V_i^2 + \frac{1}{4}C_i^{(2)} V_i^4\right) + \frac{1}{2}\sum_{i \neq j} C_{ij}^{\text{coup}} (V_i - V_j)^2$$

### 5.2 时间尺度

弛豫时间在GHz电路中可达**纳秒量级**，远快于数字计算。

---

## 六、通用近似定理

**定理**：给定任意目标分布 $p_{\text{target}}(\mathbf{v})$（满足正则性条件），对于任意 $\varepsilon > 0$，存在潜变量数目 $m$ 和哈密顿量参数，使得：

$$D_{\text{KL}}(p_{\text{target}} \| p) < \varepsilon$$

**证明要点**：

1. $-\ln p_{\text{target}}(\mathbf{v})$ 可用基函数展开
2. 每个特征函数通过边缘化潜变量实现
3. 高斯积分边缘化：$\int e^{-\frac{a}{2}z^2 - Jvz} dz \propto e^{\frac{J^2 v^2}{2a}}$
4. 非线性势能（$x^4$项）和多层潜变量组合可构建更高阶项

---

## 七、实验验证

| 数据集 | 可见变量数 | FID |
|--------|-----------|-----|
| Swiss Roll（2D） | 2 | 0.11 |
| MNIST | 780 | 21.0 |
| CIFAR-10 | 3072 | 75.4 |

---

## 八、核心贡献总结

| 贡献 | 内容 |
|------|------|
| **理论框架** | 将生成建模映射为设计哈密顿量，使玻尔兹曼分布匹配目标分布 |
| **优化方法** | KL散度最小化 → 变分自由能最小化 → 梯度 = $\langle\cdot\rangle_{\text{data}} - \langle\cdot\rangle_{\text{model}}$ |
| **表示能力** | 通过潜变量边缘化，简单物理系统可表示任意复杂分布（通用近似定理） |
| **物理实现** | 耦合非线性LC电路，热噪声提供随机性 |
| **速度优势** | 物理弛豫纳秒量级完成，比数字MCMC或反向扩散快数量级 |

## 九、一句话总结

> 一个由简单非线性振荡器组成的热力学系统，通过适当选择其物理参数（由训练优化），可以在自然弛豫到热平衡的过程中精确地从任意目标概率分布中采样——这是将物理学本身作为计算引擎的范式。
