"""forage_bandits.metrics

Vectorised helpers to compute the performance metrics used in Chapter 4 of
Jiamu's thesis with constants from the paper.

Constants (Sec 4.3.2):
    l* = 50   (maximum lifetime)
    c_m = log(l*) = log(50) ≈ 3.912
    T = 500   (maximum trials)

Metrics implemented:
1. Energy trajectory M(t) - Eq. 4.1
2. Hazard curve h(t) = exp(-c_m * M(t)) - Eq. 4.2
3. Predicted lifetime L - Eq. 4.5
4. Cumulative regret R(t) - Eq. 4.6
5. Exploration rate - Fraction of agents exploring
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

__all__ = [
    "energy_trajectory",
    "cumulative_regret",
    "predicted_lifetime",
    "hazard_curve",
    "exploration_rate",
    "MetricsBatch",
]

# Constants from thesis (Sec 4.3.2)
L_STAR = 50.0
C_M = np.log(L_STAR)  # Eq. 4.3
MAX_TRIALS = 500

# ---------------------------------------------------------------------------
# Core formulas
# ---------------------------------------------------------------------------

def energy_trajectory(
    rewards: np.ndarray,
    *,
    Mf: float = 0.0,
    M0: float = 1.0,
) -> np.ndarray:
    """Compute energy level AFTER each reward (Eq. 4.1) with step-by-step clipping."""
    rewards = np.asarray(rewards, dtype=np.float64)
    orig_ndim = rewards.ndim
    
    if orig_ndim == 1:
        rewards = rewards[np.newaxis, :]
    
    B, T = rewards.shape
    M = np.zeros((B, T))
    M[:, 0] = np.clip(M0 + rewards[:, 0] - Mf, 0.0, C_M)
    
    # Sequential update with intermediate clipping
    for t in range(1, T):
        M[:, t] = np.clip(M[:, t-1] + rewards[:, t] - Mf, 0.0, C_M)
    
    return M[0] if orig_ndim == 1 else M


def hazard_curve(
    energy: np.ndarray,
) -> np.ndarray:
    """Hazard trajectory h(t) = exp(-c_m * M(t)) (Eq. 4.2)."""
    # return np.exp(-C_M * energy)
    return np.maximum(1/L_STAR, np.exp(-energy))


def predicted_lifetime(
    hazard: np.ndarray,
    T: int = MAX_TRIALS
) -> np.ndarray:
    """Predicted lifetime L via survival analysis (Eq. 4.5)."""
    if hazard.ndim == 1:
        hazard = hazard[np.newaxis, :]
        
    B, T_sim = hazard.shape
    if T_sim < T:
        pad_width = ((0, 0), (0, T - T_sim))
        hazard = np.pad(hazard, pad_width, mode='constant', constant_values=1.0)
    
    # Compute survival function S(t) = ∏_{i=0}^{t-1} (1 - h(i))
    S = np.ones((B, T+1))
    for t in range(1, T+1):
        S[:, t] = S[:, t-1] * (1 - hazard[:, t-1])
    
    # P(t) = S(t-1) - S(t) = probability of dying at t
    P = S[:, :T] - S[:, 1:T+1]
    
    # Lifetime: L = Σ_{t=1}^{T-1} t·P(t) + T·(1 - Σ_{t=1}^{T-1} P(t))
    sum_P = np.cumsum(P[:, :-1], axis=1)
    sum_P = np.hstack([np.zeros((B, 1)), sum_P])  # Pad for t=0
    
    # Last term: 1 - sum_{t=1}^{T-1} P(t) = S(T-1)
    L = (
        np.sum(np.arange(1, T) * P[:, :T-1], axis=1) 
        + T * (1 - sum_P[:, T-1])
    )
    return L


def cumulative_regret(
    rewards: np.ndarray,
    optimal_mean: float | np.ndarray,
) -> np.ndarray:
    """Cumulative regret R(t) = Σ_{s=1}^t (μ* - r_s) (Eq. 4.6)."""
    rewards = np.asarray(rewards, dtype=np.float64)
    opt = np.asarray(optimal_mean, dtype=np.float64)
    regret_step = (opt - rewards) / C_M
    # return np.cumsum(regret_step, axis=-1)
    # regret_step = np.maximum(optimal_mean - rewards, 0) / C_M  # Scale by Emax
    return np.cumsum(regret_step, axis=-1)


def exploration_rate(
    is_exploring: np.ndarray,
) -> np.ndarray:
    """Fraction of agents exploring at each timestep."""
    return is_exploring.mean(axis=0)


# ---------------------------------------------------------------------------
# Convenience dataclass
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class MetricsBatch:
    """Collect metrics for batch of B episodes of length T."""
    regret: np.ndarray          # (B, T)
    energy: np.ndarray          # (B, T)
    hazard: np.ndarray          # (B, T)
    lifetime: np.ndarray        # (B,)
    explore_rate: np.ndarray    # (T,)

    @classmethod
    def from_rewards(
        cls,
        rewards: np.ndarray,
        is_exploring: np.ndarray,
        *,
        optimal_mean: float | np.ndarray,
        Mf: float = 0.0,
        M0: float = 1.0,
    ) -> "MetricsBatch":
        rewards = np.asarray(rewards, dtype=np.float64)
        if rewards.ndim == 1:
            rewards = rewards[None, :]
        if is_exploring.ndim == 1:
            is_exploring = is_exploring[None, :]
            
        energy = energy_trajectory(rewards, Mf=Mf, M0=M0)
        hazard = hazard_curve(energy)
        lifetime = predicted_lifetime(hazard)
        regret = cumulative_regret(rewards, optimal_mean=optimal_mean)
        exp_rate = exploration_rate(is_exploring)
        
        return cls(
            regret=regret,
            energy=energy,
            hazard=hazard,
            lifetime=lifetime,
            explore_rate=exp_rate,
        )

    def mean_regret(self) -> np.ndarray:
        return self.regret.mean(axis=0)

    def mean_energy(self) -> np.ndarray:
        return self.energy.mean(axis=0)

    def mean_hazard(self) -> np.ndarray:
        return self.hazard.mean(axis=0)

    def avg_lifetime(self) -> float:
        return float(self.lifetime.mean())