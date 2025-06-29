"""
forage_bandits.agents.discountedegree
Implementation of Discounted ε‑Greedy and Energy‑Adaptive Discounted ε‑Greedy agents.

Key idea
---------
The agent chooses a random arm with probability
    ε_eff(t) = ε * M(t)           (Energy‑Adaptive)
    ε_eff(t) = ε                  (Baseline)

where M(t) is the current energy level scaled to [0, 1].

For exploitation, the agent uses discounted empirical means:
    N_{i,γ}(t) = Σ_{s=1}^t γ^{t-s} 1{I_s=i}
    X_{i,γ}(t) = Σ_{s=1}^t γ^{t-s} r_s 1{I_s=i}
    μ̂_{i,γ}(t) = X_{i,γ}(t) / N_{i,γ}(t)

After every interaction the agent updates its discounted statistics and—when 
energy adaptation is active—its energy via M ← M + r_t − M_f.

This file contains no external dependencies beyond NumPy.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .base import AgentBase


class DiscountedEpsilonGreedy(AgentBase):
    """Discounted ε‑Greedy and Energy‑Adaptive Discounted ε‑Greedy agent.

    Parameters
    ----------
    n_arms:
        Number of arms in the bandit environment.
    epsilon:
        Nominal exploration rate (ε). Default *0.2*, matching the paper.
    gamma:
        Discount factor γ ∈ (0,1]. Default 0.5.
    energy_adaptive:
        If *True* use the energy‑adaptive exploration rate ε_eff=ε·M(t).
    init_energy:
        Initial normalised energy level *M₀* (0 → dead, 1 → full energy).
    Emax:
        Maximum energy level, default 1.0.
    forage_cost:
        Constant energetic cost *M_f* subtracted every trial when
        *energy_adaptive* is *True*.
    rng:
        Optional NumPy random generator to make simulation seeds reproducible.
    eta:
        Pseudo-count for unseen arms. Default is 1 for EA-ε-Greedy, 0 for baseline.
    """

    def __init__(
        self,
        n_arms: int,
        epsilon: float = 0.2,
        *,
        gamma: float = 0.5,
        energy_adaptive: bool = False,
        init_energy: float = 1.0,
        Emax: float = 1.0,
        forage_cost: float = 0.0,
        rng: Optional[np.random.Generator] = None,
        eta: int | float | None = None,
    ) -> None:
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
            
        self.n_arms = n_arms
        self.epsilon = float(epsilon)
        self.gamma = float(0.85)
        self.energy_adaptive = energy_adaptive
        self.Emax: float = Emax

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
        # Discounted empirical means: μ̂_{i,γ}(t)
        self.values = np.zeros(n_arms, dtype=np.float64)

        # Energy bookkeeping
        self.energy = float(init_energy)
        self.forage_cost = float(forage_cost)

        self._rng = rng if rng is not None else np.random.default_rng()

        self._last_was_explore = False

    # ------------------------------------------------------------------
    # AgentBase interface
    # ------------------------------------------------------------------
    def act(self, t: int) -> int:  # noqa: D401, ARG002  (t can be used if needed)
        """Choose an arm index for time‑step *t*."""
        # Effective exploration rate
        energy_factor = (self.energy / self.Emax) if self.energy_adaptive else 1.0
        eps_eff = self.epsilon * energy_factor

        # Minimum exploration rate
        if eps_eff < self.epsilon * 0.1:
            eps_eff = self.epsilon * 0.1

        if self._rng.random() < eps_eff:
            # Explore uniformly at random
            self._last_was_explore = True
            return int(self._rng.integers(self.n_arms))
        
        # Exploit – tie‑break randomly among arms with maximal estimated value
        self._last_was_explore = False
        best_value = self.values.max()
        best_arms = np.flatnonzero(np.isclose(self.values, best_value))
        return int(self._rng.choice(best_arms))

    def update(self, action: int, reward: float) -> None:
        """Update discounted statistics and energy."""
        # Apply discount to all existing counts and sums
        self._discounted_counts *= self.gamma
        self._discounted_sums *= self.gamma
        
        # Add new observation (discount factor is 1 for current observation)
        self._discounted_counts[action] += 1.0
        self._discounted_sums[action] += reward
        
        # Update discounted empirical means
        self.values = self._discounted_sums / (self._discounted_counts + self.eta)
        self.values[self._discounted_counts == 0] = 0.0  # convention

        # Energy dynamics for EA variant
        energy = self.energy + max(0, reward) - self.forage_cost
        energy = max(0, energy)
        energy = min(self.Emax, energy)
        self.energy = energy

    @property
    def is_exploring(self) -> bool:
        return self._last_was_explore

    # ------------------------------------------------------------------
    # Convenience helpers (optional)
    # ------------------------------------------------------------------
    def estimated_means(self) -> np.ndarray:
        """Return current discounted empirical mean reward estimates μ̂_{i,γ}(t)."""
        return self.values.copy()

    def exploration_rate(self) -> float:
        """Return current effective ε (after energy scaling)."""
        energy_factor = (self.energy / self.Emax) if self.energy_adaptive else 1.0
        eps_eff = self.epsilon * energy_factor
        return max(eps_eff, self.epsilon * 0.1)

    @property
    def discounted_counts(self) -> np.ndarray:
        """Discounted counts N_{i,γ}(t) per arm."""
        return self._discounted_counts.copy()

    @property
    def discounted_sums(self) -> np.ndarray:
        """Discounted sums X_{i,γ}(t) per arm."""
        return self._discounted_sums.copy()

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # noqa: D401
        kind = "EA-D-ε-Greedy" if self.energy_adaptive else "D-ε-Greedy"
        return f"<{kind} ε={self.epsilon:.3f} γ={self.gamma:.3f} energy={self.energy:.3f}>"


__all__ = ["DiscountedEpsilonGreedy"]
