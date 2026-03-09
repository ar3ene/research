"""Continuous RBM model definitions."""

from typing import Tuple

import numpy as np


class ContinuousRBM:
    """
    连续变量受限玻尔兹曼机。
    """

    def __init__(self, n_visible: int, n_hidden: int, kT: float = 1.0):
        self.n_v = n_visible
        self.n_h = n_hidden
        self.kT = kT

        self.a_v = np.ones(n_visible)
        self.a_h = np.ones(n_hidden)
        self.b_h = np.ones(n_hidden) * 0.01

        self.W = np.random.randn(n_visible, n_hidden) * 0.01

        self.bias_v = np.zeros(n_visible)
        self.bias_h = np.zeros(n_hidden)

    def energy(self, v: np.ndarray, h: np.ndarray) -> np.ndarray:
        """计算联合能量 H(v, h)。"""
        E_v = np.sum(self.a_v / 2 * v**2 - self.bias_v * v, axis=-1)
        E_h = np.sum(self.a_h / 2 * h**2 + self.b_h / 4 * h**4 - self.bias_h * h, axis=-1)
        E_vh = np.sum((v @ self.W) * h, axis=-1)
        return E_v + E_h + E_vh

    def _conditional_h_params(self, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算 p(h | v) 的高斯近似参数（忽略 h^4 项）。"""
        field = self.bias_h - v @ self.W
        mu = field / self.a_h
        sigma = np.sqrt(self.kT / self.a_h)
        return mu, sigma

    def _conditional_v_params(self, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算 p(v | h) 的高斯参数。"""
        field = self.bias_v - h @ self.W.T
        mu = field / self.a_v
        sigma = np.sqrt(self.kT / self.a_v)
        return mu, sigma

    def sample_h_given_v(self, v: np.ndarray) -> np.ndarray:
        """从条件分布 p(h | v) 采样（高斯近似）。"""
        mu, sigma = self._conditional_h_params(v)
        return mu + sigma * np.random.randn(*mu.shape)

    def sample_v_given_h(self, h: np.ndarray) -> np.ndarray:
        """从条件分布 p(v | h) 采样。"""
        mu, sigma = self._conditional_v_params(h)
        return mu + sigma * np.random.randn(*mu.shape)

    def gibbs_step(self, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """一步 Gibbs 采样：v -> h -> v'。"""
        h = self.sample_h_given_v(v)
        v_new = self.sample_v_given_h(h)
        return v_new, h

    def gibbs_chain(self, v_init: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """执行 k 步 Gibbs 采样链。"""
        v = v_init
        h = self.sample_h_given_v(v)
        for _ in range(k):
            v, h = self.gibbs_step(v)
        return v, h

    def free_energy(self, v: np.ndarray) -> np.ndarray:
        """
        计算有效自由能 F_eff(v) 的代理表达。
        """
        E_v = np.sum(self.a_v / 2 * v**2 - self.bias_v * v, axis=-1)
        field = self.bias_h - v @ self.W
        F_h = -np.sum(field**2 / (2 * self.a_h), axis=-1)
        return E_v + F_h
