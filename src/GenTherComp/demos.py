"""End-to-end demo workflows for thermodynamic computing experiments."""

from typing import List, Optional

import numpy as np

from .data import make_mixture_of_gaussians, make_swiss_roll
from .hamiltonians import QuarticHamiltonian
from .metrics import variational_free_energy
from .sampling import LangevinSampler
from .training import generate_samples, train_thermodynamic_model


def demo_langevin_sampling() -> None:
    """演示：用 Langevin 动力学从双势阱分布采样。"""
    print("=" * 60)
    print("演示：Langevin 动力学 - 双势阱采样")
    print("=" * 60)

    H = QuarticHamiltonian(n_dim=1, kT=1.0)
    H.a[0] = -4.0
    H.b[0] = 4.0

    sampler = LangevinSampler(H, kT=1.0, gamma=1.0, dt=0.005)
    x_init = np.random.randn(1000, 1)
    samples = sampler.sample(x_init, n_steps=5000, burn_in=2000, thin=5)

    all_samples = samples.reshape(-1)
    print(f"采样数: {len(all_samples)}")
    print(f"均值:   {np.mean(all_samples):.4f} (理论值 ~ 0)")
    print(f"标准差: {np.std(all_samples):.4f}")
    print("双峰位置: x ~ +-1")

    left_peak = np.mean(all_samples[all_samples < 0])
    right_peak = np.mean(all_samples[all_samples > 0])
    print(f"左峰均值: {left_peak:.4f}, 右峰均值: {right_peak:.4f}")
    print()


def demo_swiss_roll():
    """演示：在 Swiss Roll 数据集上训练模型并采样。"""
    print("=" * 60)
    print("演示：Swiss Roll 生成")
    print("=" * 60)

    data = make_swiss_roll(n_samples=2000, noise=0.1)
    print(f"训练数据形状: {data.shape}")
    print(f"数据范围: [{data.min():.2f}, {data.max():.2f}]")

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

    samples = generate_samples(model, n_samples=1000, n_gibbs=500)
    print(f"\n生成样本形状: {samples.shape}")
    print(f"生成范围: [{samples.min():.2f}, {samples.max():.2f}]")

    print("\n--- 统计比较 ---")
    print(f"数据均值: {data.mean(axis=0)}")
    print(f"生成均值: {samples.mean(axis=0)}")
    print(f"数据标准差: {data.std(axis=0)}")
    print(f"生成标准差: {samples.std(axis=0)}")
    print()

    return model, data, samples, losses


def demo_gaussian_mixture():
    """演示：在高斯混合数据集上训练模型并采样。"""
    print("=" * 60)
    print("演示：高斯混合模型生成")
    print("=" * 60)

    data = make_mixture_of_gaussians(n_samples=2000, n_components=5, std=0.2)
    print(f"训练数据形状: {data.shape}")

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

    samples = generate_samples(model, n_samples=1000, n_gibbs=500)
    print(f"\n生成样本形状: {samples.shape}")
    print("\n--- 统计比较 ---")
    print(f"数据均值: {data.mean(axis=0)}")
    print(f"生成均值: {samples.mean(axis=0)}")
    print()

    return model, data, samples, losses


def verify_universal_approximation(n_hidden_list: Optional[List[int]] = None):
    """验证：隐藏变量增大时模型逼近能力增强。"""
    if n_hidden_list is None:
        n_hidden_list = [10, 25, 50, 100, 200]

    print("=" * 60)
    print("验证：通用近似定理")
    print("=" * 60)

    np.random.seed(42)
    data = make_mixture_of_gaussians(n_samples=1000, n_components=4, std=0.3)

    results = []
    for n_h in n_hidden_list:
        print(f"\nn_hidden = {n_h}")
        model, losses = train_thermodynamic_model(
            data,
            n_hidden=n_h,
            n_epochs=50,
            batch_size=64,
            lr=0.001,
            cd_k=5,
            verbose=False,
        )
        fe = variational_free_energy(data, model)
        final_loss = losses[-1]
        results.append((n_h, fe, final_loss))
        print(f"  自由能: {fe:.4f}, 最终重构误差: {final_loss:.6f}")

    print("\n--- 结果汇总 ---")
    print(f"{'n_hidden':>10s} | {'Free Energy':>12s} | {'Recon Error':>12s}")
    print("-" * 40)
    for n_h, fe, loss in results:
        print(f"{n_h:10d} | {fe:12.4f} | {loss:12.6f}")

    print("\n理论预期：随着 n_hidden 增加，自由能和重构误差应趋于下降。")
    return results
