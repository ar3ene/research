"""Generative Thermodynamic Computing package."""

from .data import make_mixture_of_gaussians, make_swiss_roll
from .demos import (
    demo_gaussian_mixture,
    demo_langevin_sampling,
    demo_swiss_roll,
    verify_universal_approximation,
)
from .hamiltonians import Hamiltonian, QuarticHamiltonian
from .metrics import kl_divergence_estimate, variational_free_energy
from .rbm import ContinuousRBM
from .sampling import LangevinSampler
from .training import CDTrainer, generate_samples, train_thermodynamic_model

__all__ = [
    "Hamiltonian",
    "QuarticHamiltonian",
    "ContinuousRBM",
    "CDTrainer",
    "LangevinSampler",
    "kl_divergence_estimate",
    "variational_free_energy",
    "make_swiss_roll",
    "make_mixture_of_gaussians",
    "train_thermodynamic_model",
    "generate_samples",
    "demo_langevin_sampling",
    "demo_swiss_roll",
    "demo_gaussian_mixture",
    "verify_universal_approximation",
]
