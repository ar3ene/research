"""
Generative Thermodynamic Computing
===================================
论文复现：用热力学系统的玻尔兹曼分布进行生成建模。

核心思路：
  1. 定义一个参数化的哈密顿量 H(x; θ)
  2. 对应的玻尔兹曼分布 p(x) ∝ exp(-H(x)/kT)
  3. 通过最小化 KL(p_target || p_model) 学习参数 θ
  4. 梯度 = <∂H/∂θ>_data - <∂H/∂θ>_model

本代码实现了：
  - 连续变量受限玻尔兹曼机 (Gaussian-Bernoulli 及 Gaussian-Gaussian RBM)
  - 基于 Contrastive Divergence (CD-k) 的训练
  - 基于 Langevin 动力学的物理模拟采样
  - 在 Swiss Roll、MNIST 等数据集上的演示
"""

import numpy as np
from typing import Optional, Tuple, List

# ==============================================================================
# 第一部分：哈密顿量定义
# ==============================================================================


class Hamiltonian:
    """
    参数化哈密顿量基类。

    哈密顿量的一般形式：
        H(x) = Σ_i V_i(x_i) + Σ_{i<j} K_{ij}(x_i, x_j)

    其中 V_i 是单体势能，K_{ij} 是两体耦合。
    """

    def energy(self, x: np.ndarray) -> np.ndarray:
        """计算能量 H(x)，x: (batch, dim)"""
        raise NotImplementedError

    def grad_x(self, x: np.ndarray) -> np.ndarray:
        """计算 ∂H/∂x，用于 Langevin 动力学"""
        raise NotImplementedError


class QuarticHamiltonian(Hamiltonian):
    """
    "Vanilla" 哈密顿量：二次+四次单体势能 + 二次耦合。

        H(x) = Σ_i (a_i/2 x_i² + b_i/4 x_i⁴) + Σ_{i<j} J_{ij}/2 (x_i - x_j)²

    参数：
        a: (n_dim,)     — 二次系数
        b: (n_dim,)     — 四次系数
        J: (n_dim, n_dim) — 耦合矩阵（对称，对角为零）
    """

    def __init__(self, n_dim: int, kT: float = 1.0):
        self.n_dim = n_dim
        self.kT = kT
        # 初始化参数
        self.a = np.zeros(n_dim)          # 二次系数
        self.b = np.ones(n_dim) * 0.01    # 四次系数（小正值保证势能有界）
        self.J = np.zeros((n_dim, n_dim)) # 耦合矩阵

    def energy(self, x: np.ndarray) -> np.ndarray:
        """
        x: (batch, n_dim) → 返回 (batch,) 能量值

        H(x) = Σ_i (a_i/2 x_i² + b_i/4 x_i⁴)
              + Σ_{i<j} J_{ij}/2 (x_i - x_j)²
        """
        # 单体项
        E_single = np.sum(self.a / 2 * x**2 + self.b / 4 * x**4, axis=-1)

        # 耦合项: Σ_{i<j} J_{ij}/2 (x_i - x_j)²
        # = 1/2 Σ_{ij} J_{ij}/2 (x_i - x_j)²  （因为 J 对称且对角为零）
        # = 1/2 x^T L x，其中 L 是拉普拉斯矩阵 L = diag(Σ_j J_ij) - J
        L = np.diag(np.sum(self.J, axis=1)) - self.J
        E_couple = 0.5 * np.sum(x @ L * x, axis=-1)

        return E_single + E_couple

    def grad_x(self, x: np.ndarray) -> np.ndarray:
        """
        ∂H/∂x_i = a_i x_i + b_i x_i³ + Σ_j J_{ij}(x_i - x_j)

        x: (batch, n_dim) → 返回 (batch, n_dim)
        """
        # 单体项梯度
        grad_single = self.a * x + self.b * x**3

        # 耦合项梯度：L @ x
        L = np.diag(np.sum(self.J, axis=1)) - self.J
        grad_couple = x @ L  # (batch, n_dim)

        return grad_single + grad_couple


# ==============================================================================
# 第二部分：连续变量受限玻尔兹曼机 (Continuous RBM)
# ==============================================================================


class ContinuousRBM:
    """
    连续变量受限玻尔兹曼机。

    哈密顿量：
        H(v, z) = Σ_i (a_i^v / 2 v_i²)
                + Σ_j (a_j^z / 2 z_j² + b_j^z / 4 z_j⁴)
                + Σ_{ij} J_{ij} v_i z_j

    二部图结构使得条件分布可解析：
        p(z | v) 是（近似）高斯分布（忽略 z⁴ 项时精确高斯）
        p(v | z) 是高斯分布

    参数：
        n_visible:  可见层维度
        n_hidden:   隐藏层维度
        kT:         温度 (k_B T)
    """

    def __init__(self, n_visible: int, n_hidden: int, kT: float = 1.0):
        self.n_v = n_visible
        self.n_h = n_hidden
        self.kT = kT

        # 可见层参数：二次势能
        self.a_v = np.ones(n_visible)      # > 0 保证稳定

        # 隐藏层参数：二次 + 四次势能
        self.a_h = np.ones(n_hidden)        # 二次系数
        self.b_h = np.ones(n_hidden) * 0.01 # 四次系数

        # 耦合权重
        self.W = np.random.randn(n_visible, n_hidden) * 0.01

        # 偏置
        self.bias_v = np.zeros(n_visible)
        self.bias_h = np.zeros(n_hidden)

    def energy(self, v: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        计算联合能量 H(v, h)。

        v: (batch, n_v), h: (batch, n_h) → 返回 (batch,)
        """
        # 可见层单体能量
        E_v = np.sum(self.a_v / 2 * v**2 - self.bias_v * v, axis=-1)

        # 隐藏层单体能量
        E_h = np.sum(self.a_h / 2 * h**2 + self.b_h / 4 * h**4
                     - self.bias_h * h, axis=-1)

        # 耦合能量
        E_vh = np.sum((v @ self.W) * h, axis=-1)

        return E_v + E_h + E_vh

    def _conditional_h_params(self, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算 p(h | v) 的高斯近似参数（忽略 h⁴ 项）。

        p(h_j | v) ∝ exp(-(a_j^h / 2) h_j² - (Σ_i W_{ij} v_i) h_j + bias_h_j h_j)

        这是高斯分布 N(μ_j, σ_j²)，其中：
            μ_j = (bias_h_j - Σ_i W_{ij} v_i) / a_j^h
            σ_j² = kT / a_j^h
        """
        # 有效场：bias - v @ W
        field = self.bias_h - v @ self.W  # (batch, n_h)

        mu = field / self.a_h                      # (batch, n_h)
        sigma = np.sqrt(self.kT / self.a_h)        # (n_h,)

        return mu, sigma

    def _conditional_v_params(self, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算 p(v | h) 的高斯参数。

        p(v_i | h) ∝ exp(-(a_i^v / 2) v_i² - (Σ_j W_{ij} h_j) v_i + bias_v_i v_i)

        高斯分布 N(μ_i, σ_i²)：
            μ_i = (bias_v_i - Σ_j W_{ij} h_j) / a_i^v
            σ_i² = kT / a_i^v
        """
        field = self.bias_v - h @ self.W.T  # (batch, n_v)

        mu = field / self.a_v
        sigma = np.sqrt(self.kT / self.a_v)

        return mu, sigma

    def sample_h_given_v(self, v: np.ndarray) -> np.ndarray:
        """从条件分布 p(h | v) 采样（高斯近似）"""
        mu, sigma = self._conditional_h_params(v)
        return mu + sigma * np.random.randn(*mu.shape)

    def sample_v_given_h(self, h: np.ndarray) -> np.ndarray:
        """从条件分布 p(v | h) 采样"""
        mu, sigma = self._conditional_v_params(h)
        return mu + sigma * np.random.randn(*mu.shape)

    def gibbs_step(self, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        一步 Gibbs 采样：v → h → v'

        物理含义：钳制可见层 → 隐藏层弛豫 → 钳制隐藏层 → 可见层弛豫
        """
        h = self.sample_h_given_v(v)
        v_new = self.sample_v_given_h(h)
        return v_new, h

    def gibbs_chain(self, v_init: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行 k 步 Gibbs 采样链。

        参数：
            v_init: 初始可见层状态 (batch, n_v)
            k:      Gibbs 步数

        返回：
            v_k: k 步后的可见层状态
            h_k: 最后一步的隐藏层状态
        """
        v = v_init
        h = self.sample_h_given_v(v)
        for _ in range(k):
            v, h = self.gibbs_step(v)
        return v, h

    def free_energy(self, v: np.ndarray) -> np.ndarray:
        """
        计算有效自由能 F_eff(v) = -kT ln ∫ exp(-H(v,h)/kT) dh

        对于高斯 h（忽略 h⁴），可解析积分：
            F_eff(v) = Σ_i a_i^v / 2 v_i² - bias_v · v
                     - kT/2 Σ_j ln(2π kT / a_j^h)
                     + 1/(2 a_j^h) (bias_h_j - Σ_i W_{ij} v_i)²
                     （常数项忽略）

        v: (batch, n_v) → 返回 (batch,)
        """
        # 可见层能量
        E_v = np.sum(self.a_v / 2 * v**2 - self.bias_v * v, axis=-1)

        # 隐藏层的有效贡献
        field = self.bias_h - v @ self.W  # (batch, n_h)
        # 高斯积分的结果：-kT/2 ln(2π kT/a) - field²/(2a)
        F_h = -np.sum(field**2 / (2 * self.a_h), axis=-1)

        return E_v + F_h


# ==============================================================================
# 第三部分：Contrastive Divergence 训练
# ==============================================================================


class CDTrainer:
    """
    Contrastive Divergence (CD-k) 训练器。

    核心梯度公式：
        ∂L/∂θ = <∂H/∂θ>_data - <∂H/∂θ>_model

    其中 <·>_model 用 k 步 Gibbs 采样近似。
    """

    def __init__(self, model: ContinuousRBM, lr: float = 0.001, k: int = 1,
                 momentum: float = 0.0, weight_decay: float = 0.0):
        """
        参数：
            model:        ContinuousRBM 实例
            lr:           学习率
            k:            CD 中的 Gibbs 步数
            momentum:     动量系数
            weight_decay: 权重衰减（L2正则化）
        """
        self.model = model
        self.lr = lr
        self.k = k
        self.momentum = momentum
        self.weight_decay = weight_decay

        # 动量缓存
        self._dW = np.zeros_like(model.W)
        self._d_bias_v = np.zeros_like(model.bias_v)
        self._d_bias_h = np.zeros_like(model.bias_h)
        self._d_a_v = np.zeros_like(model.a_v)
        self._d_a_h = np.zeros_like(model.a_h)

    def _compute_gradients(self, v_data: np.ndarray) -> dict:
        """
        计算 CD-k 梯度。

        步骤：
        1. 正相（positive phase）：从数据计算 <∂H/∂θ>_data
        2. 负相（negative phase）：从 k 步 Gibbs 链计算 <∂H/∂θ>_model
        3. 梯度 = 正相统计量 - 负相统计量
        """
        batch_size = v_data.shape[0]
        m = self.model

        # ---- 正相（data phase）----
        # 从数据计算隐藏层均值（用期望代替采样以减少方差）
        mu_h_data, _ = m._conditional_h_params(v_data)

        # ---- 负相（model phase）----
        # k 步 Gibbs 链
        v_model, _ = m.gibbs_chain(v_data, k=self.k)
        mu_h_model, _ = m._conditional_h_params(v_model)

        # ---- 计算梯度 ----
        # ∂H/∂W = v^T h（耦合项的梯度）
        # 梯度 = <v h^T>_data - <v h^T>_model
        grad_W = (v_data.T @ mu_h_data - v_model.T @ mu_h_model) / batch_size

        # ∂H/∂bias_v = -v
        grad_bias_v = -(np.mean(v_data, axis=0) - np.mean(v_model, axis=0))

        # ∂H/∂bias_h = -h
        grad_bias_h = -(np.mean(mu_h_data, axis=0) - np.mean(mu_h_model, axis=0))

        # ∂H/∂a_v = v²/2
        grad_a_v = (np.mean(v_data**2, axis=0) - np.mean(v_model**2, axis=0)) / 2

        # ∂H/∂a_h = h²/2
        grad_a_h = (np.mean(mu_h_data**2, axis=0) - np.mean(mu_h_model**2, axis=0)) / 2

        return {
            'W': grad_W,
            'bias_v': grad_bias_v,
            'bias_h': grad_bias_h,
            'a_v': grad_a_v,
            'a_h': grad_a_h,
        }

    def step(self, v_data: np.ndarray) -> float:
        """
        执行一步参数更新。

        返回：重构误差（作为训练监控指标）
        """
        m = self.model
        grads = self._compute_gradients(v_data)

        # 更新动量缓存并应用
        self._dW = self.momentum * self._dW - self.lr * (grads['W'] + self.weight_decay * m.W)
        self._d_bias_v = self.momentum * self._d_bias_v - self.lr * grads['bias_v']
        self._d_bias_h = self.momentum * self._d_bias_h - self.lr * grads['bias_h']
        self._d_a_v = self.momentum * self._d_a_v - self.lr * grads['a_v']
        self._d_a_h = self.momentum * self._d_a_h - self.lr * grads['a_h']

        m.W += self._dW
        m.bias_v += self._d_bias_v
        m.bias_h += self._d_bias_h
        m.a_v += self._d_a_v
        m.a_h += self._d_a_h

        # 钳制 a_v, a_h 为正值（保证势能有界）
        m.a_v = np.maximum(m.a_v, 1e-4)
        m.a_h = np.maximum(m.a_h, 1e-4)

        # 计算重构误差
        v_recon, _ = m.gibbs_chain(v_data, k=1)
        recon_error = np.mean((v_data - v_recon) ** 2)

        return float(recon_error)


# ==============================================================================
# 第四部分：Langevin 动力学采样器
# ==============================================================================


class LangevinSampler:
    """
    Langevin 动力学采样器 —— 模拟物理系统的弛豫过程。

    Langevin 方程（过阻尼极限）：
        dx/dt = -∂H/∂x / γ + √(2 k_B T / γ) · η(t)

    其中 η(t) 是高斯白噪声，γ 是阻尼系数。

    离散化（Euler-Maruyama）：
        x_{t+1} = x_t - dt/γ · ∂H/∂x + √(2 k_B T dt / γ) · ε

    其中 ε ~ N(0, I)。

    在平衡态，采样分布收敛到 p(x) ∝ exp(-H(x) / k_B T)。
    """

    def __init__(self, hamiltonian: Hamiltonian, kT: float = 1.0,
                 gamma: float = 1.0, dt: float = 0.01):
        """
        参数：
            hamiltonian: 哈密顿量对象
            kT:          温度 (k_B T)
            gamma:       阻尼系数
            dt:          时间步长
        """
        self.H = hamiltonian
        self.kT = kT
        self.gamma = gamma
        self.dt = dt
        self.noise_scale = np.sqrt(2 * kT * dt / gamma)

    def step(self, x: np.ndarray) -> np.ndarray:
        """
        一步 Langevin 更新。

        x: (batch, dim) → 返回 (batch, dim)
        """
        grad = self.H.grad_x(x)
        noise = np.random.randn(*x.shape)

        x_new = x - (self.dt / self.gamma) * grad + self.noise_scale * noise
        return x_new

    def sample(self, x_init: np.ndarray, n_steps: int = 1000,
               burn_in: int = 500, thin: int = 10) -> np.ndarray:
        """
        通过 Langevin 动力学采样。

        参数：
            x_init:   初始状态 (batch, dim)
            n_steps:  总步数
            burn_in:  丢弃的初始步数（等待系统弛豫到平衡）
            thin:     每隔 thin 步保留一个样本

        返回：
            samples: (n_samples, batch, dim)
        """
        x = x_init.copy()
        samples = []

        for t in range(n_steps):
            x = self.step(x)
            if t >= burn_in and (t - burn_in) % thin == 0:
                samples.append(x.copy())

        return np.array(samples)


# ==============================================================================
# 第五部分：KL散度与变分自由能
# ==============================================================================


def kl_divergence_estimate(p_samples: np.ndarray, model: ContinuousRBM) -> float:
    """
    估计 KL(p_target || p_model)。

    利用公式：
        KL = <-ln p_target>_target + <H/kT>_target + ln Z

    由于 ln Z 难以计算，这里用自由能差来估计：
        KL ≈ <F_eff(v)>_data / kT + const

    参数：
        p_samples: 从目标分布采样的样本 (n_samples, n_dim)
        model:     ContinuousRBM 模型

    返回：
        自由能的均值（与KL散度单调相关的代理量）
    """
    F = model.free_energy(p_samples)
    return float(np.mean(F) / model.kT)


def variational_free_energy(data: np.ndarray, model: ContinuousRBM) -> float:
    """
    计算变分自由能 L(θ) = <H_θ>_data + kT ln Z。

    由于 ln Z 不可直接计算，用代理量：
        L_proxy = mean(F_eff(v_data))

    参数：
        data:  训练数据 (n_samples, n_dim)
        model: ContinuousRBM 模型
    """
    return float(np.mean(model.free_energy(data)))


# ==============================================================================
# 第六部分：数据生成工具
# ==============================================================================


def make_swiss_roll(n_samples: int = 1000, noise: float = 0.1) -> np.ndarray:
    """
    生成 Swiss Roll 数据集（2D投影）。

    参数：
        n_samples: 样本数
        noise:     噪声强度

    返回：
        data: (n_samples, 2) 的数组
    """
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
    x = t * np.cos(t) + noise * np.random.randn(n_samples)
    y = t * np.sin(t) + noise * np.random.randn(n_samples)

    data = np.column_stack([x, y])

    # 标准化
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return data


def make_mixture_of_gaussians(n_samples: int = 1000,
                               n_components: int = 5,
                               dim: int = 2,
                               std: float = 0.3) -> np.ndarray:
    """
    生成高斯混合模型数据。

    参数：
        n_samples:    总样本数
        n_components: 高斯分量数
        dim:          数据维度
        std:          每个分量的标准差

    返回：
        data: (n_samples, dim)
    """
    # 均匀分布在单位圆上的中心
    angles = np.linspace(0, 2 * np.pi, n_components, endpoint=False)
    centers = np.column_stack([np.cos(angles), np.sin(angles)])

    if dim > 2:
        centers = np.hstack([centers, np.zeros((n_components, dim - 2))])

    # 为每个样本随机选择一个分量
    labels = np.random.randint(0, n_components, n_samples)
    data = centers[labels] + std * np.random.randn(n_samples, dim)

    return data


# ==============================================================================
# 第七部分：完整训练流程
# ==============================================================================


def train_thermodynamic_model(
    data: np.ndarray,
    n_hidden: int = 50,
    kT: float = 1.0,
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 0.001,
    cd_k: int = 1,
    momentum: float = 0.5,
    verbose: bool = True,
) -> Tuple[ContinuousRBM, List[float]]:
    """
    完整的生成式热力学计算训练流程。

    步骤：
    1. 初始化连续 RBM（哈密顿量参数化）
    2. 用 CD-k 迭代更新参数
    3. 监控重构误差和自由能

    参数：
        data:       训练数据 (n_samples, n_dim)
        n_hidden:   隐藏变量数
        kT:         温度
        n_epochs:   训练轮数
        batch_size: 批大小
        lr:         学习率
        cd_k:       CD 中的 Gibbs 步数
        momentum:   动量系数
        verbose:    是否打印训练信息

    返回：
        model:  训练好的 ContinuousRBM
        losses: 每个 epoch 的平均重构误差列表
    """
    n_samples, n_dim = data.shape

    # 1. 初始化模型
    model = ContinuousRBM(n_visible=n_dim, n_hidden=n_hidden, kT=kT)

    # 2. 初始化训练器
    trainer = CDTrainer(model, lr=lr, k=cd_k, momentum=momentum)

    # 3. 训练循环
    losses = []
    for epoch in range(n_epochs):
        # 打乱数据
        perm = np.random.permutation(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            batch = data[perm[i:i + batch_size]]
            if len(batch) < 2:
                continue

            loss = trainer.step(batch)
            epoch_loss += loss
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        # 学习率的动量增加策略
        if epoch == 5:
            trainer.momentum = 0.9

        if verbose and (epoch + 1) % 10 == 0:
            fe = variational_free_energy(data[:min(500, n_samples)], model)
            print(f"Epoch {epoch+1:4d}/{n_epochs} | "
                  f"Recon Error: {avg_loss:.6f} | "
                  f"Free Energy: {fe:.4f}")

    return model, losses


def generate_samples(model: ContinuousRBM, n_samples: int = 500,
                     n_gibbs: int = 1000) -> np.ndarray:
    """
    从训练好的模型生成样本。

    物理含义：让系统从随机初始状态弛豫到热平衡，然后采样。

    参数：
        model:     训练好的 ContinuousRBM
        n_samples: 生成样本数
        n_gibbs:   Gibbs 采样步数（对应弛豫步数）

    返回：
        samples: (n_samples, n_visible) 生成的样本
    """
    # 从随机初始化开始（对应物理系统的随机初始状态）
    v = np.random.randn(n_samples, model.n_v) * 0.1

    # Gibbs 采样 = 物理弛豫
    v, _ = model.gibbs_chain(v, k=n_gibbs)

    return v


# ==============================================================================
# 第八部分：Langevin 动力学演示
# ==============================================================================


def demo_langevin_sampling():
    """
    演示：用 Langevin 动力学从双势阱分布采样。

    目标分布：p(x) ∝ exp(-(x²-1)²)
    对应哈密顿量：H(x) = (x²-1)² = x⁴ - 2x² + 1
    即 a = -2, b = 1 的四次势能。
    """
    print("=" * 60)
    print("演示：Langevin 动力学 — 双势阱采样")
    print("=" * 60)

    # 构建哈密顿量：H(x) = x⁴ - 2x² + 1
    H = QuarticHamiltonian(n_dim=1, kT=1.0)
    H.a[0] = -4.0  # 二次系数（负值 → 双势阱）
    H.b[0] = 4.0   # 四次系数（正值 → 势能有界）

    # Langevin 采样
    sampler = LangevinSampler(H, kT=1.0, gamma=1.0, dt=0.005)
    x_init = np.random.randn(1000, 1)
    samples = sampler.sample(x_init, n_steps=5000, burn_in=2000, thin=5)

    # 统计结果
    all_samples = samples.reshape(-1)
    print(f"采样数: {len(all_samples)}")
    print(f"均值:   {np.mean(all_samples):.4f} (理论值 ≈ 0)")
    print(f"标准差: {np.std(all_samples):.4f}")
    print(f"双峰位置: x ≈ ±1")

    # 检查双峰结构
    left_peak = np.mean(all_samples[all_samples < 0])
    right_peak = np.mean(all_samples[all_samples > 0])
    print(f"左峰均值: {left_peak:.4f}, 右峰均值: {right_peak:.4f}")
    print()


# ==============================================================================
# 第九部分：Swiss Roll 演示
# ==============================================================================


def demo_swiss_roll():
    """
    演示：在 Swiss Roll 数据集上训练生成式热力学计算模型。
    """
    print("=" * 60)
    print("演示：Swiss Roll 生成")
    print("=" * 60)

    # 生成数据
    data = make_swiss_roll(n_samples=2000, noise=0.1)
    print(f"训练数据形状: {data.shape}")
    print(f"数据范围: [{data.min():.2f}, {data.max():.2f}]")

    # 训练模型
    model, losses = train_thermodynamic_model(
        data,
        n_hidden=100,
        kT=1.0,
        n_epochs=100,
        batch_size=64,
        lr=0.001,
        cd_k=10,
        momentum=0.5,
        verbose=True,
    )

    # 生成样本
    samples = generate_samples(model, n_samples=1000, n_gibbs=500)
    print(f"\n生成样本形状: {samples.shape}")
    print(f"生成范围: [{samples.min():.2f}, {samples.max():.2f}]")

    # 简单的统计比较
    print(f"\n--- 统计比较 ---")
    print(f"数据均值: {data.mean(axis=0)}")
    print(f"生成均值: {samples.mean(axis=0)}")
    print(f"数据标准差: {data.std(axis=0)}")
    print(f"生成标准差: {samples.std(axis=0)}")
    print()

    return model, data, samples


# ==============================================================================
# 第十部分：高斯混合模型演示
# ==============================================================================


def demo_gaussian_mixture():
    """
    演示：在高斯混合模型数据上训练。
    """
    print("=" * 60)
    print("演示：高斯混合模型生成")
    print("=" * 60)

    # 生成数据
    data = make_mixture_of_gaussians(n_samples=2000, n_components=5, std=0.2)
    print(f"训练数据形状: {data.shape}")

    # 训练
    model, losses = train_thermodynamic_model(
        data,
        n_hidden=100,
        kT=1.0,
        n_epochs=100,
        batch_size=64,
        lr=0.001,
        cd_k=10,
        verbose=True,
    )

    # 生成
    samples = generate_samples(model, n_samples=1000, n_gibbs=500)
    print(f"\n生成样本形状: {samples.shape}")
    print(f"\n--- 统计比较 ---")
    print(f"数据均值: {data.mean(axis=0)}")
    print(f"生成均值: {samples.mean(axis=0)}")
    print()

    return model, data, samples


# ==============================================================================
# 第十一部分：通用近似定理验证
# ==============================================================================


def verify_universal_approximation(n_hidden_list: Optional[List[int]] = None):
    """
    验证通用近似定理：随着隐藏变量数增加，模型逼近能力增强。

    测试：固定目标分布（高斯混合），增加隐藏层维度，观察自由能下降。
    """
    if n_hidden_list is None:
        n_hidden_list = [10, 25, 50, 100, 200]

    print("=" * 60)
    print("验证：通用近似定理")
    print("=" * 60)

    # 固定目标分布
    np.random.seed(42)
    data = make_mixture_of_gaussians(n_samples=1000, n_components=4, std=0.3)

    results = []
    for n_h in n_hidden_list:
        print(f"\nn_hidden = {n_h}")
        model, losses = train_thermodynamic_model(
            data, n_hidden=n_h, n_epochs=50, batch_size=64,
            lr=0.001, cd_k=5, verbose=False
        )
        fe = variational_free_energy(data, model)
        final_loss = losses[-1]
        results.append((n_h, fe, final_loss))
        print(f"  自由能: {fe:.4f}, 最终重构误差: {final_loss:.6f}")

    print(f"\n--- 结果汇总 ---")
    print(f"{'n_hidden':>10s} | {'Free Energy':>12s} | {'Recon Error':>12s}")
    print("-" * 40)
    for n_h, fe, loss in results:
        print(f"{n_h:10d} | {fe:12.4f} | {loss:12.6f}")

    print("\n理论预期：随着 n_hidden 增加，自由能和重构误差应趋于下降。")


# ==============================================================================
# 主程序
# ==============================================================================


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     Generative Thermodynamic Computing — 代码演示         ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  论文复现：用热力学系统的玻尔兹曼分布进行生成建模         ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # 设置随机种子
    np.random.seed(42)

    # 演示 1：Langevin 动力学
    demo_langevin_sampling()

    # 演示 2：Swiss Roll
    demo_swiss_roll()

    # 演示 3：高斯混合模型
    demo_gaussian_mixture()

    # 演示 4：通用近似定理验证
    verify_universal_approximation()

    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)
