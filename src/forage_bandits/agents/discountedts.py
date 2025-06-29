"""forage_bandits.agents.discountedts

Implementation of Discounted Thompson Sampling for non-stationary bandits.

Discounted Thompson Sampling adapts the classic Thompson Sampling algorithm to 
non-stationary environments by using discounted statistics for posterior updates.

Reward model
------------
We assume arm rewards are **Gaussian** with *unknown* mean μ and (fixed but
unknown) precision τ. With a Normal‑Gamma conjugate prior NG(μ̄, n, α, β), the
posterior remains Normal‑Gamma after observing a reward.

Discounted statistics
---------------------
For each arm, we maintain discounted counts and sums:
    N_{i,γ}(t) = Σ_{s=1}^t γ^{t-s} 1{I_s=i}
    X_{i,γ}(t) = Σ_{s=1}^t γ^{t-s} r_s 1{I_s=i}
    μ̂_{i,γ}(t) = X_{i,γ}(t) / N_{i,γ}(t)

Sampling rule per trial
~~~~~~~~~~~~~~~~~~~~~~~
1.  For each arm *a* draw τₐ ← Gamma(αₐ, βₐ)
2.  Draw an estimated mean:

        μ̂ₐ ∼ Normal( μ̄ₐ,
                      σ² = energy_factor / (N_{a,γ}(t) τₐ) ),

    where energy_factor = M if *energy_adaptive=True* else 1.0.
3.  Play the arm with highest μ̂.

Posterior update after reward *r* (discounted)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```
μ̄ ← (N_{i,γ}(t-1) μ̄ + r) / N_{i,γ}(t)
α ← α + 0.5
β ← β + (r*r - μ̄_old**2) / (2 * N_{i,γ}(t))
```

Energy bookkeeping (EA‑TS only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```
M ← clip( M + r - Mf, 0, Emax )
```

This implementation is side‑effect‑free and NumPy‑only.
"""
from __future__ import annotations

from typing import Union

import numpy as np

from .base import AgentBase


class DiscountedThompsonSampling(AgentBase):
    """Discounted Thompson Sampling agent for non-stationary environments."""

    def __init__(
        self,
        n_arms: int,
        *,
        gamma: float = 0.5,
        init_energy: float = 1.0,
        Emax: float = 1.0,
        energy_adaptive: bool = False,
        forage_cost: float = 0.0,
        eta: int | float | None = None,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        rng: Union[np.random.Generator, int, None] = None,
    ) -> None:
        super().__init__(n_arms)
        
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
            
        self.gamma = float(0.85)
        self.energy_adaptive = bool(energy_adaptive)
        self.Mf = float(forage_cost)
        self._rng = np.random.default_rng(rng)

        # Prior / Posterior params per arm
        self._mu = np.zeros(n_arms, dtype=np.float64)  # μ̄ₐ
        
        # If eta not supplied, use baseline 0 or EA default 1
        if eta is None:
            eta = 1 if energy_adaptive else 1e-10
        if eta == 0:
            eta = 1e-10
        self.eta: float = eta
        
        # Discounted statistics
        self._discounted_counts = np.zeros(n_arms, dtype=np.float64)  # N_{i,γ}(t)
        self._discounted_sums = np.zeros(n_arms, dtype=np.float64)    # X_{i,γ}(t)
        
        # Gamma parameters
        self._alpha = np.full(n_arms, alpha0, dtype=np.float64)
        self._beta = np.full(n_arms, beta0, dtype=np.float64)

        # Global time step & energy
        self._t = 0
        self.energy: float = init_energy
        self.Emax: float = Emax

        self._last_was_explore = False

    # ------------------------------------------------------------------
    # Simulator API
    # ------------------------------------------------------------------
    def act(self, t: int) -> int:  # noqa: D401
        """Sample NG posterior and return arm index with biggest sample."""
        energy_factor = (self.energy / self.Emax) if self.energy_adaptive else 1.0

        # Draw from Gamma for each arm → tau (precision)
        tau = self._rng.gamma(shape=self._alpha, scale=1.0 / self._beta)

        # Use discounted counts for variance calculation
        var = energy_factor / ((self._discounted_counts + self.eta) * tau)
        sigma = np.sqrt(var)
        samples = self._rng.normal(loc=self._mu, scale=sigma)
        arm = int(np.argmax(samples))

        # Check if selection was exploratory
        empirical_best = np.argmax(self._mu)
        self._last_was_explore = (arm != empirical_best)

        return arm

    def update(self, arm: int, reward: float) -> None:
        """Discounted Bayesian posterior update plus energy bookkeeping."""
        # Apply discount to all existing statistics
        self._discounted_counts *= self.gamma
        self._discounted_sums *= self.gamma
        
        # Add new observation (discount factor is 1 for current observation)
        self._discounted_counts[arm] += 1.0
        self._discounted_sums[arm] += reward
        
        # Store old values for posterior update
        mu_old = self._mu[arm]
        n_old = self._discounted_counts[arm] - 1.0  # Count before this update
        
        # Discounted posterior update
        if self._discounted_counts[arm] > 0:
            self._mu[arm] = self._discounted_sums[arm] / self._discounted_counts[arm]
        
        # Update Gamma parameters
        self._alpha[arm] += 0.5
        if self._discounted_counts[arm] > 0:
            self._beta[arm] += (reward * reward - mu_old * mu_old) / (2.0 * self._discounted_counts[arm])

        # Energy update
        energy = self.energy + max(0, reward) - self.Mf
        energy = max(0, energy)
        energy = min(self.Emax, energy)
        self.energy = energy

        # Advance time
        self._t += 1

    @property
    def is_exploring(self) -> bool:
        return self._last_was_explore

    # ------------------------------------------------------------------
    # Diagnostics helpers
    # ------------------------------------------------------------------
    def estimated_means(self) -> np.ndarray:
        """Return current discounted posterior means μ̄."""
        return self._mu.copy()

    @property
    def discounted_counts(self) -> np.ndarray:
        """Discounted counts N_{i,γ}(t) per arm."""
        return self._discounted_counts.copy()

    @property
    def discounted_sums(self) -> np.ndarray:
        """Discounted sums X_{i,γ}(t) per arm."""
        return self._discounted_sums.copy()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def n_arms(self) -> int:  # noqa: D401
        return self._mu.size

    # ------------------------------------------------------------------
    # Reset support (optional)
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._mu.fill(0.0)
        self._discounted_counts.fill(0.0)
        self._discounted_sums.fill(0.0)
        self._alpha.fill(1.0)
        self._beta.fill(1.0)
        self._t = 0
        self.energy = 1.0

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # noqa: D401
        tag = "EA-D-TS" if self.energy_adaptive else "D-TS"
        return (
            f"<{tag}: γ={self.gamma:.3f}, M={self.energy:.3f}, "
            f"discounted_counts={self._discounted_counts}, means={self._mu}>"
        )
