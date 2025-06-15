"""
forage_bandits.agents.egree
Implementation of ε‑Greedy and Energy‑Adaptive ε‑Greedy agents described in
Chapter 4 of the thesis (Algorithm 1, Eq. 4.7).

Key idea
---------
The agent chooses a random arm with probability
    ε_eff(t) = ε * M(t)           (Energy‑Adaptive)
    ε_eff(t) = ε                  (Baseline)

where M(t) is the current energy level scaled to [0, 1].
After every interaction the agent updates its empirical mean estimate μ̄_a,
the pull count n_a and—when energy adaptation is active—its energy via

    M ← M + r_t − M_f.

This file contains no external dependencies beyond NumPy.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .base import AgentBase


class EpsilonGreedy(AgentBase):
    """ε‑Greedy (baseline) and EA‑ε‑Greedy agent.

    Parameters
    ----------
    n_arms:
        Number of arms in the bandit environment.
    epsilon:
        Nominal exploration rate (ε). Default *0.2*, matching the paper.
    energy_adaptive:
        If *True* use the energy‑adaptive exploration rate ε_eff=ε·M(t).
    init_energy:
        Initial normalised energy level *M₀* (0 → dead, 1 → full energy).
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
        energy_adaptive: bool = False,
        init_energy: float = 1.0,
        forage_cost: float = 0.0,
        rng: Optional[np.random.Generator] = None,
        eta: int | float | None = None,
    ) -> None:
        self.n_arms = n_arms
        self.epsilon = float(epsilon)
        self.energy_adaptive = energy_adaptive

        # Statistics
        if eta is None:
            eta = 1
            # eta = 1 if energy_adaptive else 0
        self.counts = np.full(n_arms, eta, dtype=np.int64)
        self.values = np.zeros(n_arms, dtype=np.float64)  # empirical means μ̄_a

        # Energy bookkeeping
        self.energy = float(init_energy)
        self.forage_cost = float(forage_cost)
        if not energy_adaptive:
            # avoid pylint / mypy unused‑attribute warnings in baseline variant
            del init_energy, forage_cost

        self._rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------
    # AgentBase interface
    # ------------------------------------------------------------------
    def act(self, t: int) -> int:  # noqa: D401, ARG002  (t can be used if needed)
        """Choose an arm index for time‑step *t*."""
        # Effective exploration rate
        eps_eff = (
            self.epsilon * self.energy if self.energy_adaptive else self.epsilon
        )
        if self._rng.random() < eps_eff:
            # Explore uniformly at random
            return int(self._rng.integers(self.n_arms))
        # Exploit – tie‑break randomly among arms with maximal estimated value
        best_value = self.values.max()
        best_arms = np.flatnonzero(np.isclose(self.values, best_value))
        return int(self._rng.choice(best_arms))

    def update(self, action: int, reward: float) -> None:
        """Update internal estimates (and energy if applicable)."""
        # Incremental mean update for chosen arm
        self.counts[action] += 1
        n = self.counts[action]
        self.values[action] += (reward - self.values[action]) / n

        # Energy dynamics for EA variant
        self.energy = float(np.clip(self.energy + reward - self.forage_cost, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Convenience helpers (optional)
    # ------------------------------------------------------------------
    def estimated_means(self) -> np.ndarray:
        """Return current empirical mean reward estimates μ̄_a."""
        return self.values.copy()

    def exploration_rate(self) -> float:
        """Return current effective ε (after energy scaling)."""
        return (
            self.epsilon * self.energy if self.energy_adaptive else self.epsilon
        )

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # noqa: D401
        kind = "EA‑ε‑Greedy" if self.energy_adaptive else "ε‑Greedy"
        return f"<{kind} ε={self.epsilon:.3f} energy={self.energy:.3f}>"


__all__ = ["EpsilonGreedy"]
