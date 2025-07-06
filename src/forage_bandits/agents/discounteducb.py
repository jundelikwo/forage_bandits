"""forage_bandits.agents.discounteducb

Implementation of the Discounted UCB algorithm for non-stationary bandits.

Discounted-UCB (a.k.a. D-UCB) is designed for non-stationary bandits, giving more 
weight to recent rewards via a discount factor γ ∈ (0,1].

Discounted counts and sums:
    N_{i,γ}(t) = Σ_{s=1}^t γ^{t-s} 1{I_s=i}
    X_{i,γ}(t) = Σ_{s=1}^t γ^{t-s} r_s 1{I_s=i}
    μ̂_{i,γ}(t) = X_{i,γ}(t) / N_{i,γ}(t)

Index:
    D-UCB_i(t) = μ̂_{i,γ}(t) + √(α ln n_γ(t) / (2 N_{i,γ}(t)))

where n_γ(t) = Σ_{k=1}^K N_{k,γ}(t) is the discounted total number of pulls.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import AgentBase
from ..energy_factors import energy_factor_linear, energy_factor_exp, energy_factor_flip_exp, energy_factor_thr, energy_factor_parabolic, energy_factor_sigmoid


class DiscountedUCB(AgentBase):
    """Discounted UCB bandit agent for non-stationary environments.

    Parameters
    ----------
    n_arms:
        Total number of arms.
    gamma:
        Discount factor γ ∈ (0,1]. Default 0.85.
    alpha:
        Tuning constant α > 0. Default 1.0.
    energy_adaptive:
        If *True* the exploration term is scaled by *M(t)*.
    forage_cost:
        Energy cost *M_f* per interaction (0 ≤ M_f ≤ 1).
    init_energy:
        Initial energy *M(0)*, default 1.0.
    Emax:
        Maximum energy level, default 1.0.
    energy_factor_alg:
        Energy factor algorithm to use. Default "linear".
    rng:
        Optional NumPy ``Generator`` or integer seed.
    eta:
        Pseudo-count for unseen arms. Default is 1 for EA, 0 for baseline.
    """

    def __init__(
        self,
        n_arms: int,
        *,
        gamma: float = 0.85,
        alpha: float = 1.0,
        energy_adaptive: bool = False,
        forage_cost: float = 0.0,
        init_energy: float = 1.0,
        Emax: float = 1.0,
        energy_factor_alg: str = "linear",
        rng: Optional[np.random.Generator | int] = None,
        eta: int | float | None = None,
    ) -> None:
        super().__init__(n_arms)

        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.energy_adaptive = bool(energy_adaptive)
        self._Mf = float(forage_cost)
        self.energy = float(init_energy)
        self.Emax: float = Emax
        self.energy_factor_alg = energy_factor_alg
        
        # Statistics - discounted counts and sums
        if eta is None:
            eta = 1 if energy_adaptive else 1e-10
        if eta == 0:
            eta = 1e-10
        self.eta: float = eta
        
        # Discounted counts: N_{i,γ}(t)
        self._discounted_counts = np.zeros(n_arms, dtype=np.float64)
        # Discounted sums: X_{i,γ}(t)
        self._discounted_sums = np.zeros(n_arms, dtype=np.float64)
        # Total discounted pulls: n_γ(t)
        self._total_discounted_pulls = 0.0

        # RNG
        self.rng: np.random.Generator
        self.rng = (
            rng
            if isinstance(rng, np.random.Generator)
            else np.random.default_rng(rng)
        )

        self._last_was_explore = False

    def _get_energy_factor(self, energy: float) -> float:
        if self.energy_factor_alg == "linear":
            return energy_factor_linear(energy)
        elif self.energy_factor_alg == "exp":
            return energy_factor_exp(energy)
        elif self.energy_factor_alg == "flip_exp":
            return energy_factor_flip_exp(energy)
        elif self.energy_factor_alg == "thr":
            return energy_factor_thr(energy)
        elif self.energy_factor_alg == "parabolic":
            return energy_factor_parabolic(energy)
        elif self.energy_factor_alg == "sigmoid":
            return energy_factor_sigmoid(energy)

    # ---------------------------------------------------------------------
    # AgentBase API
    # ---------------------------------------------------------------------

    def act(self, t: int) -> int:
        """Choose an arm according to Discounted UCB."""
        # Compute discounted empirical means
        means = np.zeros_like(self._discounted_counts, dtype=float)
        np.divide(self._discounted_sums, self._discounted_counts, out=means, 
                 where=self._discounted_counts > 0)
        
        # Energy scaling for exploration term
        energy_factor = self._get_energy_factor(self.energy / self.Emax) if self.energy_adaptive else 1.0
        energy_factor = max(energy_factor, 0)
        
        # D-UCB index: μ̂_{i,γ}(t) + √(α ln n_γ(t) / (2 N_{i,γ}(t)))
        with np.errstate(divide='ignore', invalid='ignore'):
            exploration_term = np.sqrt(
                self.alpha * np.log(self._total_discounted_pulls + 1) / 
                (2 * (self._discounted_counts + self.eta))
            )
            exploration_term = np.where(
                self._discounted_counts == 0, 
                np.inf, 
                exploration_term
            )
        
        # Scale exploration by energy if adaptive
        exploration_term *= energy_factor
        
        ucb_values = means + exploration_term
        
        # Handle numerical issues: replace NaN and inf with finite values
        ucb_values = np.nan_to_num(ucb_values, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Break ties randomly among best options
        best_arms = np.flatnonzero(ucb_values == ucb_values.max())
        
        # Safety check: if best_arms is empty, choose randomly
        if len(best_arms) == 0:
            arm = int(self.rng.integers(0, self.n_actions))
        else:
            arm = int(self.rng.choice(best_arms))

        # Check if selection was exploratory
        empirical_best = np.argmax(means)
        self._last_was_explore = (arm != empirical_best)

        return arm

    def update(self, action: int, reward: float) -> None:
        """Update discounted statistics and energy."""
        # Apply discount to all existing counts and sums
        self._discounted_counts *= self.gamma
        self._discounted_sums *= self.gamma
        self._total_discounted_pulls *= self.gamma
        
        # Add new observation (discount factor is 1 for current observation)
        self._discounted_counts[action] += 1.0
        self._discounted_sums[action] += reward
        self._total_discounted_pulls += 1.0
        
        # Energy update with clipping to [0, Emax]
        energy = self.energy + max(0, reward) - self._Mf
        energy = max(0, energy)
        energy = min(self.Emax, energy)
        self.energy = energy

    @property
    def is_exploring(self) -> bool:
        return self._last_was_explore

    # ------------------------------------------------------------------
    # Convenience helpers (not part of abstract interface)
    # ------------------------------------------------------------------

    @property
    def counts(self) -> np.ndarray:
        """Discounted counts N_{i,γ}(t) per arm."""
        return self._discounted_counts.copy()

    @property
    def discounted_sums(self) -> np.ndarray:
        """Discounted sums X_{i,γ}(t) per arm."""
        return self._discounted_sums.copy()

    @property
    def means(self) -> np.ndarray:
        """Discounted empirical mean reward per arm μ̂_{i,γ}(t)."""
        means = self._discounted_sums / (self._discounted_counts + self.eta)
        means[self._discounted_counts == 0] = 0.0  # convention
        return means

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover – for debugging only
        tag = "EA-D-UCB" if self.energy_adaptive else "D-UCB"
        return (
            f"<{tag}: γ={self.gamma}, α={self.alpha}, M={self.energy:.3f}, "
            f"discounted_counts={self._discounted_counts}, means={self.means}>"
        )
