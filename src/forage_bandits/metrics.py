"""forage_bandits.metrics

Vectorised helpers to compute the performance metrics used in Chapter 4 of
Jiamu's thesis:

* **Cumulative regret**  R(t)  — Eq. 4.5.
* **Energy trajectory**  M(t) — Eq. 4.4 update:  M←clip(M + r − M_f, 0, 1).
* **Lifetime**          L     — first timestep where M(t)=0 (starvation); if
  starvation never occurs, we define L = T (episode length).
* **Hazard curve**      h(t)  — population‑level probability that M(t)=0 by
  step t (Eq. 4.6).

All routines are **NumPy‑only**, side‑effect‑free, and broadcast across batches
so they can operate on either a single episode vector *(T,)* or a batch matrix
*(B, T)*.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = [
    "energy_trajectory",
    "cumulative_regret",
    "lifetime",
    "hazard_curve",
    "MetricsBatch",
]


# ---------------------------------------------------------------------------
# Core formulas
# ---------------------------------------------------------------------------

def energy_trajectory(
    rewards: np.ndarray,
    *,
    Mf: float = 0.0,
    M0: float = 1.0,
) -> np.ndarray:
    """Return energy level **after** each reward is received.

    Parameters
    ----------
    rewards : ndarray, shape (T,) or (B, T)
        Reward at each timestep **clipped to [0, 1]**.
    Mf : float, default 0.0
        Constant foraging cost charged every timestep (Section 4.2.2).
    M0 : float, default 1.0
        Initial energy level (fully energised).

    Returns
    -------
    M : ndarray, same shape as *rewards*
        Energy level at each timestep, clipped to [0, 1].  If M hits 0 it will
        stick there for the remainder of the episode.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    dtype = rewards.dtype

    # ΔM_t = reward - Mf  (Eq. 4.4)
    delta = rewards - Mf

    # Cumulative sum with initial offset
    M = M0 + np.cumsum(delta, axis=-1, dtype=dtype)

    # Clip to [0, 1] and enforce absorbing state at 0
    M = np.clip(M, 0.0, 1.0, out=M)
    if M.ndim == 1:
        # once starved, remain at 0 (absorbing)
        starved = np.maximum.accumulate(M == 0)
        M[starved] = 0.0
    else:
        starved = np.maximum.accumulate(M == 0, axis=-1)
        M[starved] = 0.0  # type: ignore[arg-type]

    return M


def cumulative_regret(
    rewards: np.ndarray,
    optimal_mean: float | np.ndarray,
) -> np.ndarray:
    """Vectorised cumulative regret R(t) = Σ(μ* − r_t) up to each t."""
    rewards = np.asarray(rewards, dtype=np.float64)
    opt = np.asarray(optimal_mean, dtype=np.float64)
    regret_step = opt - rewards  # broadcasting handles (T,) vs scalar
    return np.cumsum(regret_step, axis=-1)


def lifetime(
    rewards: np.ndarray,
    *,
    Mf: float = 0.0,
    M0: float = 1.0,
) -> np.ndarray:
    """Episode lifetime L – first t where energy reaches 0 (1‑based).

    Returns an integer array of shape (B,) giving lifetime per episode.  When
    an episode never starves, L equals the episode length T.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    T = rewards.shape[-1]
    M = energy_trajectory(rewards, Mf=Mf, M0=M0)

    starved = M == 0
    if starved.ndim == 1:
        idx = np.argmax(starved) if starved.any() else T - 1
        return np.asarray(idx + 1, dtype=np.int64)

    # Batch mode
    first_zero = np.where(starved, np.arange(1, T + 1), T)
    L = np.min(first_zero, axis=-1)
    return L.astype(np.int64)


def hazard_curve(
    rewards: np.ndarray,
    *,
    Mf: float = 0.0,
    M0: float = 1.0,
) -> np.ndarray:
    """Population‑level hazard h(t) = P(M(t)=0).

    Input *rewards* must be **batch‑of‑episodes** shaped (B, T).  The output is
    an array of length T giving, for each timestep, the fraction of episodes
    whose energy has reached zero by (and including) that time.
    """
    if rewards.ndim != 2:
        raise ValueError("hazard_curve expects rewards with shape (B, T)")

    M = energy_trajectory(rewards, Mf=Mf, M0=M0)
    starved = M == 0
    # Cumulative: once starved, always starved ➔ use cummax
    starved_cum = np.maximum.accumulate(starved, axis=-1)
    return starved_cum.mean(axis=0)


# ---------------------------------------------------------------------------
# Convenience dataclass to bundle metrics for a batch of episodes
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class MetricsBatch:
    """Collect all metrics for a batch of *B* episodes of length *T*."""

    regret: np.ndarray      # shape (B, T)
    energy: np.ndarray      # shape (B, T)
    lifetime: np.ndarray    # shape (B,)
    hazard: np.ndarray      # shape (T,)

    @classmethod
    def from_rewards(
        cls,
        rewards: np.ndarray,
        *,
        optimal_mean: float | np.ndarray,
        Mf: float = 0.0,
        M0: float = 1.0,
    ) -> "MetricsBatch":
        """Compute all metrics from reward matrix (B, T)."""
        rewards = np.asarray(rewards, dtype=np.float64)
        if rewards.ndim == 1:
            rewards = rewards[None, :]  # promote to batch size 1

        energy = energy_trajectory(rewards, Mf=Mf, M0=M0)
        regret = cumulative_regret(rewards, optimal_mean=optimal_mean)
        L = lifetime(rewards, Mf=Mf, M0=M0)
        h = hazard_curve(rewards, Mf=Mf, M0=M0)
        return cls(regret=regret, energy=energy, lifetime=L, hazard=h)

    # ------------------------------------------------------------------
    # Convenience summarisation
    # ------------------------------------------------------------------
    def mean_regret(self) -> np.ndarray:
        """Average regret across batch (T,)."""
        return self.regret.mean(axis=0)

    def mean_energy(self) -> np.ndarray:
        return self.energy.mean(axis=0)

    def avg_lifetime(self) -> float:
        return float(self.lifetime.mean())

    # Allow dict‑like access for easy Pandas integration
    def to_dict(self) -> dict[str, np.ndarray | float]:
        return {
            "regret": self.regret,
            "energy": self.energy,
            "lifetime": self.lifetime,
            "hazard": self.hazard,
        }
