"""Synthetic data generators for demos."""

import numpy as np


def make_swiss_roll(n_samples: int = 1000, noise: float = 0.1) -> np.ndarray:
    """生成 Swiss Roll 数据集（2D 投影）。"""
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
    x = t * np.cos(t) + noise * np.random.randn(n_samples)
    y = t * np.sin(t) + noise * np.random.randn(n_samples)
    data = np.column_stack([x, y])
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return data


def make_mixture_of_gaussians(
    n_samples: int = 1000,
    n_components: int = 5,
    dim: int = 2,
    std: float = 0.3,
) -> np.ndarray:
    """生成高斯混合模型数据。"""
    angles = np.linspace(0, 2 * np.pi, n_components, endpoint=False)
    centers = np.column_stack([np.cos(angles), np.sin(angles)])

    if dim > 2:
        centers = np.hstack([centers, np.zeros((n_components, dim - 2))])

    labels = np.random.randint(0, n_components, n_samples)
    data = centers[labels] + std * np.random.randn(n_samples, dim)
    return data
