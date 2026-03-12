"""Hamiltonian definitions for thermodynamic models."""

import numpy as np


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
        a: (n_dim,)       二次系数
        b: (n_dim,)       四次系数
        J: (n_dim, n_dim) 耦合矩阵（对称，对角为零）
    """

    def __init__(self, n_dim: int, kT: float = 1.0):
        self.n_dim = n_dim
        self.kT = kT
        self.a = np.zeros(n_dim)
        self.b = np.ones(n_dim) * 0.01
        self.J = np.zeros((n_dim, n_dim))

    def energy(self, x: np.ndarray) -> np.ndarray:
        """
        x: (batch, n_dim) -> 返回 (batch,) 能量值
        """
        E_single = np.sum(self.a / 2 * x**2 + self.b / 4 * x**4, axis=-1)
        L = np.diag(np.sum(self.J, axis=1)) - self.J
        E_couple = 0.5 * np.sum(x @ L * x, axis=-1)
        return E_single + E_couple

    def grad_x(self, x: np.ndarray) -> np.ndarray:
        """
        ∂H/∂x_i = a_i x_i + b_i x_i^3 + Σ_j J_{ij}(x_i - x_j)

        x: (batch, n_dim) -> 返回 (batch, n_dim)
        """
        grad_single = self.a * x + self.b * x**3
        L = np.diag(np.sum(self.J, axis=1)) - self.J
        grad_couple = x @ L
        return grad_single + grad_couple
