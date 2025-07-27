"""forage_bandits.agents.ucb

Implementation of the classic UCB1 algorithm (Auer et al., 2002) **and** the
Energy‑Adaptive variant introduced in Chapter 4 of Jiamu's thesis (Eq. 4.10).

Notation used in Jiamu's thesis:
    t            – current timestep (0‑based in code, 1‑based in paper)
    i            – arm index
    N_i(t)       – number of pulls of arm *i* before t
    \hat\mu_i(t) – empirical mean reward of arm *i*
    M(t)         – agent's internal energy, capped to [0, 1]
    M_f          – energy cost per pull (foraging cost)

Classic UCB1 chooses
    i_t = argmax_i  ( \hat\mu_i(t) + c * sqrt(2 * ln t / N_i(t)) )
where c ≥ 1 is a constant (often 1 or 2).

Energy‑Adaptive UCB simply makes the exploration coefficient **scale with the
current energy**:
    c_eff(t) = c * M(t)
This smoothly shuts off exploration as energy approaches 0.  The update rule
for M(t) follows Eq. 4.5 in the PDF:  M ← clip(M + r_t − M_f, 0, 1).

Both variants share the same incremental update logic and RNG handling.
"""
from __future__ import annotations

from typing import Optional, Callable

import numpy as np

from .base import AgentBase
from ..energy_factors import energy_factor_linear, energy_factor_exp, energy_factor_flip_exp, energy_factor_thr, energy_factor_parabolic, energy_factor_sigmoid, energy_factor_flip_linear


class UCB(AgentBase):
    """Upper‑Confidence‑Bound bandit agent.

    Parameters
    ----------
    n_arms:
        Total number of arms.
    c:
        Base exploration coefficient *c* (>= 1).  In EA‑UCB this is further
        multiplied by the current energy.
    energy_adaptive:
        If *True* the exploration term is scaled by *M(t)*
        (EA‑UCB).  If *False* we get classic UCB1.
    forage_cost:
        Energy cost *M_f* per interaction (0 ≤ M_f ≤ 1).  Ignored when
        *energy_adaptive=False* (but accepted for API consistency).
    M_init:
        Initial energy *M(0)*, default 1.0.
    energy_factor_alg:
        Energy factor algorithm to use. Default "linear".
    rng:
        Optional NumPy ``Generator`` or integer seed.
    eta:
        Pseudo-count for unseen arms. Default is 1 for EA-UCB, 0 for baseline.
    custom_exploration_function:
        Custom exploration function to use. Default is None. Accepts energy and energy_adaptive as arguments.
    """

    def __init__(
        self,
        n_arms: int,
        *,
        c: float = np.sqrt(2),
        energy_adaptive: bool = False,
        forage_cost: float = 0.0,
        init_energy: float = 1.0,
        Emax: float = 1.0,
        energy_factor_alg: str = "linear",
        rng: Optional[np.random.Generator | int] = None,
        eta: int | float | None = None,
        custom_exploration_function: Callable[[float, bool], float] = None,
    ) -> None:
        super().__init__(n_arms)

        self._c = float(c)
        self.energy_adaptive = bool(energy_adaptive)
        self._Mf = float(forage_cost)
        self.energy = float(init_energy)
        self.Emax: float = Emax
        self.energy_factor_alg = energy_factor_alg
        self.custom_exploration_function = custom_exploration_function
        # Statistics
        if eta is None:
            eta = 1 if energy_adaptive else 1e-10
        if eta == 0:
            eta = 1e-10
        self.eta: float = eta
        self._counts = np.zeros(n_arms, dtype=np.int64)
        self._sum_rwd = np.zeros(n_arms, dtype=np.float64)

        # RNG
        self.rng: np.random.Generator
        self.rng = (
            rng
            if isinstance(rng, np.random.Generator)
            else np.random.default_rng(rng)
        )

        self._last_was_explore = False

    # ---------------------------------------------------------------------
    # AgentBase API
    # ---------------------------------------------------------------------

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

    def act(self, t: int) -> int:
        """Choose an arm according to (EA‑)UCB1."""
        # Convert 0-based index to 1-based trial count
        total_trials = t + 1
        
        # Compute empirical means safely
        means = np.zeros_like(self._counts, dtype=float)
        np.divide(self._sum_rwd, self._counts, out=means, where=self._counts != 0)
        
        energy_factor = self._get_energy_factor(self.energy / self.Emax) if self.energy_adaptive else 1.0
        energy_factor = max(energy_factor, 0)

        if self.custom_exploration_function is not None:
            energy_factor = self.custom_exploration_function(self.energy / self.Emax, self.energy_adaptive)

        
        c_eff = self._c * energy_factor  # Scale by energy
        pads = c_eff * np.sqrt((np.log(total_trials)) / (self._counts + self.eta))
        
        ucb_values = means + pads
        
        # Break ties randomly among best options
        best_arms = np.flatnonzero(ucb_values == ucb_values.max())
        arm = int(self.rng.choice(best_arms))

        # Check if selection was exploratory
        empirical_best = np.argmax(self.means)
        self._last_was_explore = (arm != empirical_best)

        return arm

    def update(self, action: int, reward: float) -> None:
        """Update empirical means and energy."""
        self._counts[action] += 1
        self._sum_rwd[action] += reward
        
        # Energy update with clipping to [0, 1] (Eq. 4.1)
        # new_energy = self.energy + reward - self._Mf
        # self.energy = float(np.clip(new_energy, 0.0, 1.0))

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
        """Number of pulls per arm."""
        return self._counts.copy()

    @property
    def means(self) -> np.ndarray:
        """Empirical mean reward per arm."""
        means = self._sum_rwd / (self._counts + self.eta)
        means[self._counts == 0] = 0.0  # convention
        return means

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover – for debugging only
        tag = "EA‑UCB" if self.energy_adaptive else "UCB1"
        return (
            f"<{tag}: c={self._c}, M={self.energy:.3f}, counts={self._counts}, "
            f"means={self.means}>"
        )
