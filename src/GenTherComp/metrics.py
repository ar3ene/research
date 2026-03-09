"""Metrics and free-energy proxies."""

import numpy as np

from .rbm import ContinuousRBM


def kl_divergence_estimate(p_samples: np.ndarray, model: ContinuousRBM) -> float:
    """估计 KL(p_target || p_model) 的单调代理量。"""
    F = model.free_energy(p_samples)
    return float(np.mean(F) / model.kT)


def variational_free_energy(data: np.ndarray, model: ContinuousRBM) -> float:
    """计算变分自由能代理量 L_proxy = mean(F_eff(v_data))。"""
    return float(np.mean(model.free_energy(data)))
