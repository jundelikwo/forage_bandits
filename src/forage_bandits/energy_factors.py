"""forage_bandits.energy_factors
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "energy_factor_linear",
    "energy_factor_flip_linear",
    "energy_factor_exp",
    "energy_factor_flip_exp",
    "energy_factor_thr",
    "energy_factor_parabolic",
    "energy_factor_sigmoid",
]

alpha   = 3.0   # exponential decay rate                   alpha ∈ [2, 5]
tau_thr = 0.4   # hard-threshold switch-point              tau ∈ [0.3, 0.5]
lam     = 0.9   # exploration value below threshold        lamda ∈ [0.8, 1]
k       = 10.0  # logistic steepness for soft threshold    k ∈ [8, 15]
tau_l   = 0.4   # lower inflection for soft threshold
tau_h   = 0.9   # upper inflection for soft threshold
min_parabolic = 0.1 # minimum value for parabolic function

def energy_factor_linear(energy):
    """Linear: f_lin(E) = E"""
    return energy

def energy_factor_flip_linear(energy):
    """Linear: f_lin(E) = 1 - E"""
    return 1 - energy

def energy_factor_exp(energy, a=alpha):
    """Mild exponential: f_exp(E) = exp(-α(1-E))"""
    return np.exp(-a * (1 - energy))

def energy_factor_flip_exp(energy, a=alpha):
    """Mild exponential: f_exp(E) = |1 - exp(-alpha(1-E))|"""
    return np.abs(1.0 - np.exp(-a * (1 - energy)))

def energy_factor_thr(energy, tau=tau_thr, lam=lam):
    """Hard threshold: f_thr(E) = E if E>tau else lam"""
    return np.where(energy > tau, energy, lam)

def energy_factor_parabolic(energy, m=min_parabolic):
    """Parabolic: f_par(E) = m + (1 - m) * (2E-1)^2"""
    return m + (1 - m) * (2*energy - 1)**2

def energy_factor_sigmoid(energy, k=k, tau_l=tau_l, tau_h=tau_h):
    """Sigmoid combo: f_sig(E) = 1 - (1 / (1 + exp(k(E-tau_h))) + 1 / (1 + exp(-k(E-tau_l))))"""
    sigma_hi = 1 / (1 + np.exp( k * (energy - tau_h)))   # falls after tau_h
    sigma_lo = 1 / (1 + np.exp(-k * (energy - tau_l)))   # rises before tau_l
    return 1 - (sigma_hi + sigma_lo - 1)