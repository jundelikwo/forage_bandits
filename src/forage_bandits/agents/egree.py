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
from typing import Optional, Callable

from .base import AgentBase
from ..energy_factors import energy_factor_linear, energy_factor_exp, energy_factor_flip_exp, energy_factor_thr, energy_factor_parabolic, energy_factor_sigmoid, energy_factor_flip_linear


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
    energy_factor_alg:
        Energy factor algorithm to use. Default "linear".
    rng:
        Optional NumPy random generator to make simulation seeds reproducible.
    eta:
        Pseudo-count for unseen arms. Default is 1 for EA-ε-Greedy, 0 for baseline.
    custom_exploration_function:
        Custom exploration function to use. Default is None. Accepts energy and energy_adaptive as arguments.
    """

    def __init__(
        self,
        n_arms: int,
        epsilon: float = 0.2,
        *,
        energy_adaptive: bool = False,
        init_energy: float = 1.0,
        Emax: float = 1.0,
        forage_cost: float = 0.0,
        energy_factor_alg: str = "linear",
        rng: Optional[np.random.Generator] = None,
        eta: int | float | None = None,
        custom_exploration_function: Callable[[float, bool], float] = None,
    ) -> None:
        self.n_arms = n_arms
        self.epsilon = float(epsilon)
        self.energy_adaptive = energy_adaptive
        self.Emax: float = Emax
        self.custom_exploration_function = custom_exploration_function
        # Statistics
        if eta is None:
            eta = 1 if energy_adaptive else 1e-10
        if eta == 0:
            eta = 1e-10
        self.eta: float = eta
        self.counts = np.zeros(n_arms, dtype=np.int64)
        self.values = np.zeros(n_arms, dtype=np.float64)  # empirical means μ̄_a

        # Energy bookkeeping
        self.energy = float(init_energy)
        self.forage_cost = float(forage_cost)
        self.energy_factor_alg = energy_factor_alg
        if not energy_adaptive:
            # avoid pylint / mypy unused‑attribute warnings in baseline variant
            del init_energy, forage_cost

        self._rng = rng if rng is not None else np.random.default_rng()

        self._last_was_explore = False

    def _get_energy_factor(self, energy: float) -> float:
        if self.energy_factor_alg == "linear":
            return energy_factor_linear(energy)
        elif self.energy_factor_alg == "flip_linear":
            return energy_factor_flip_linear(energy)
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

    # ------------------------------------------------------------------
    # AgentBase interface
    # ------------------------------------------------------------------
    def act(self, t: int) -> int:  # noqa: D401, ARG002  (t can be used if needed)
        """Choose an arm index for time‑step *t*."""
        # Effective exploration rate
        energy_factor = self._get_energy_factor(self.energy / self.Emax) if self.energy_adaptive else 1.0
        eps_eff = self.epsilon * energy_factor

        if self.custom_exploration_function is not None:
            eps_eff = self.custom_exploration_function(self.energy / self.Emax, self.energy_adaptive)

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
        """Update internal estimates (and energy if applicable)."""
        # Incremental mean update for chosen arm
        self.counts[action] += 1
        n = self.counts[action]
        self.values[action] += (reward - self.values[action]) / n

        # Energy dynamics for EA variant
        # self.energy = float(np.clip(self.energy + reward - self.forage_cost, 0.0, 1.0))
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
