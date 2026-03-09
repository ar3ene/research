"""Training utilities for continuous RBM models."""

from typing import List, Tuple

import numpy as np

from .metrics import variational_free_energy
from .rbm import ContinuousRBM


class CDTrainer:
    """Contrastive Divergence (CD-k) 训练器。"""

    def __init__(
        self,
        model: ContinuousRBM,
        lr: float = 0.001,
        k: int = 1,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        self.model = model
        self.lr = lr
        self.k = k
        self.momentum = momentum
        self.weight_decay = weight_decay

        self._dW = np.zeros_like(model.W)
        self._d_bias_v = np.zeros_like(model.bias_v)
        self._d_bias_h = np.zeros_like(model.bias_h)
        self._d_a_v = np.zeros_like(model.a_v)
        self._d_a_h = np.zeros_like(model.a_h)

    def _compute_gradients(self, v_data: np.ndarray) -> dict:
        batch_size = v_data.shape[0]
        m = self.model

        mu_h_data, _ = m._conditional_h_params(v_data)

        v_model, _ = m.gibbs_chain(v_data, k=self.k)
        mu_h_model, _ = m._conditional_h_params(v_model)

        grad_W = (v_data.T @ mu_h_data - v_model.T @ mu_h_model) / batch_size
        grad_bias_v = -(np.mean(v_data, axis=0) - np.mean(v_model, axis=0))
        grad_bias_h = -(np.mean(mu_h_data, axis=0) - np.mean(mu_h_model, axis=0))
        grad_a_v = (np.mean(v_data**2, axis=0) - np.mean(v_model**2, axis=0)) / 2
        grad_a_h = (np.mean(mu_h_data**2, axis=0) - np.mean(mu_h_model**2, axis=0)) / 2

        return {
            "W": grad_W,
            "bias_v": grad_bias_v,
            "bias_h": grad_bias_h,
            "a_v": grad_a_v,
            "a_h": grad_a_h,
        }

    def step(self, v_data: np.ndarray) -> float:
        """执行一步参数更新，并返回重构误差。"""
        m = self.model
        grads = self._compute_gradients(v_data)

        self._dW = self.momentum * self._dW - self.lr * (grads["W"] + self.weight_decay * m.W)
        self._d_bias_v = self.momentum * self._d_bias_v - self.lr * grads["bias_v"]
        self._d_bias_h = self.momentum * self._d_bias_h - self.lr * grads["bias_h"]
        self._d_a_v = self.momentum * self._d_a_v - self.lr * grads["a_v"]
        self._d_a_h = self.momentum * self._d_a_h - self.lr * grads["a_h"]

        m.W += self._dW
        m.bias_v += self._d_bias_v
        m.bias_h += self._d_bias_h
        m.a_v += self._d_a_v
        m.a_h += self._d_a_h

        m.a_v = np.maximum(m.a_v, 1e-4)
        m.a_h = np.maximum(m.a_h, 1e-4)

        v_recon, _ = m.gibbs_chain(v_data, k=1)
        recon_error = np.mean((v_data - v_recon) ** 2)
        return float(recon_error)


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
    """完整的生成式热力学计算训练流程。"""
    n_samples, n_dim = data.shape
    model = ContinuousRBM(n_visible=n_dim, n_hidden=n_hidden, kT=kT)
    trainer = CDTrainer(model, lr=lr, k=cd_k, momentum=momentum)

    losses: List[float] = []
    for epoch in range(n_epochs):
        perm = np.random.permutation(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            batch = data[perm[i : i + batch_size]]
            if len(batch) < 2:
                continue
            loss = trainer.step(batch)
            epoch_loss += loss
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if epoch == 5:
            trainer.momentum = 0.9

        if verbose and (epoch + 1) % 10 == 0:
            fe = variational_free_energy(data[: min(500, n_samples)], model)
            print(
                f"Epoch {epoch+1:4d}/{n_epochs} | "
                f"Recon Error: {avg_loss:.6f} | "
                f"Free Energy: {fe:.4f}"
            )

    return model, losses


def generate_samples(model: ContinuousRBM, n_samples: int = 500, n_gibbs: int = 1000) -> np.ndarray:
    """从训练好的模型生成样本。"""
    v = np.random.randn(n_samples, model.n_v) * 0.1
    v, _ = model.gibbs_chain(v, k=n_gibbs)
    return v
