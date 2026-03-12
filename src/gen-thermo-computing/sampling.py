"""Sampling algorithms for thermodynamic models."""

import numpy as np

from .hamiltonians import Hamiltonian


class LangevinSampler:
    """
    Langevin 动力学采样器。
    """

    def __init__(self, hamiltonian: Hamiltonian, kT: float = 1.0, gamma: float = 1.0, dt: float = 0.01):
        self.H = hamiltonian
        self.kT = kT
        self.gamma = gamma
        self.dt = dt
        self.noise_scale = np.sqrt(2 * kT * dt / gamma)

    def step(self, x: np.ndarray) -> np.ndarray:
        """一步 Langevin 更新。"""
        grad = self.H.grad_x(x)
        noise = np.random.randn(*x.shape)
        x_new = x - (self.dt / self.gamma) * grad + self.noise_scale * noise
        return x_new

    def sample(self, x_init: np.ndarray, n_steps: int = 1000, burn_in: int = 500, thin: int = 10) -> np.ndarray:
        """通过 Langevin 动力学采样。"""
        x = x_init.copy()
        samples = []
        for t in range(n_steps):
            x = self.step(x)
            if t >= burn_in and (t - burn_in) % thin == 0:
                samples.append(x.copy())
        return np.array(samples)
