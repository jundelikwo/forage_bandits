"""forage_bandits.agents.ts

Thompson Sampling (TS) and its Energy‑Adaptive variant (EA‑TS) exactly as
introduced in Chapter 4 – see Eq. 4.15 and Eq. 4.16 plus Algorithm 3 in the Jiamu's thesis .

Reward model
------------
We assume arm rewards are **Gaussian** with *unknown* mean μ and (fixed but
unknown) precision τ.  With a Normal‑Gamma conjugate prior NG(μ̄, n, α, β), the
posterior remains Normal‑Gamma after observing a reward.

Sampling rule per trial
~~~~~~~~~~~~~~~~~~~~~~~
1.  For each arm *a* draw τₐ ← Gamma(αₐ, βₐ)  (NumPy: ``rng.gamma(shape=α, scale=1/β)``)
2.  Draw an estimated mean:

        μ̂ₐ ∼ Normal( μ̄ₐ,
                      σ² = energy_factor / (nₐ τₐ) ),

    where ``energy_factor = M`` if *energy_adaptive=True* else 1.0 fileciteturn3file4.
3.  Play the arm with highest μ̂.

Posterior update after reward *r* (Eq. 4.15)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```
μ̄ ← (n μ̄ + r) / (n + 1)
n ← n + 1
α ← α + 0.5
β ← β + (r*r - μ̄_old**2) / (2 * n)
```

Energy bookkeeping (EA‑TS only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```
M ← clip( M + r - Mf, 0, 1 )
```

Novel‑arm initialisation
~~~~~~~~~~~~~~~~~~~~~~~~
Following Section 4.4.6 we optionally start with **η pseudo‑counts** so variance is
finite even for unseen arms (defaults: η=0 for baseline, η=1 when energy_adaptive).

This implementation is side‑effect‑free and NumPy‑only.
"""
from __future__ import annotations

from typing import Union

import numpy as np

from .base import AgentBase


class ThompsonSampling(AgentBase):
    """Unified Thompson Sampling / EA‑TS agent."""

    def __init__(
        self,
        n_arms: int,
        *,
        energy_adaptive: bool = False,
        forage_cost: float = 0.0,
        eta: int | float = None,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        rng: Union[np.random.Generator, int, None] = None,
    ) -> None:
        super().__init__(n_arms)
        self.energy_adaptive = bool(energy_adaptive)
        self.Mf = float(forage_cost)
        self._rng = np.random.default_rng(rng)

        # Prior / Posterior params per arm
        self._mu = np.zeros(n_arms, dtype=np.float64)  # μ̄ₐ
        # If eta not supplied, use baseline 0 or EA default 1
        if eta is None:
            eta = 1 if self.energy_adaptive else 0
        self._n = np.full(n_arms, eta, dtype=np.float64)  # nₐ (can be float)
        self._alpha = np.full(n_arms, alpha0, dtype=np.float64)
        self._beta = np.full(n_arms, beta0, dtype=np.float64)

        # Global time step & energy
        self._t = 0
        self.M: float = 1.0

    # ------------------------------------------------------------------
    # Simulator API
    # ------------------------------------------------------------------
    def act(self, t: int) -> int:  # noqa: D401
        """Sample NG posterior and return arm index with biggest sample."""
        energy_factor = self.M if self.energy_adaptive else 1.0

        # Draw from Gamma for each arm → tau (precision)
        tau = self._rng.gamma(shape=self._alpha, scale=1.0 / self._beta)
        # Avoid division by zero when n=0
        denom = np.where(self._n > 0, self._n, 1.0)
        var = energy_factor / (denom * tau)
        sigma = np.sqrt(var)
        samples = self._rng.normal(loc=self._mu, scale=sigma)
        arm = int(np.argmax(samples))
        return arm

    def update(self, arm: int, reward: float) -> None:
        """Bayesian posterior update plus energy bookkeeping."""
        mu_old = self._mu[arm]
        n_old = self._n[arm]

        # Posterior update (Eq. 4.15)
        self._mu[arm] = (n_old * mu_old + reward) / (n_old + 1)
        self._n[arm] = n_old + 1
        self._alpha[arm] += 0.5
        self._beta[arm] += (reward * reward - mu_old * mu_old) / (2.0 * self._n[arm])

        # Energy update for EA‑TS
        if self.energy_adaptive:
            self.M = float(np.clip(self.M + reward - self.Mf, 0.0, 1.0))

        # Advance time
        self._t += 1

    # ------------------------------------------------------------------
    # Diagnostics helpers
    # ------------------------------------------------------------------
    def estimated_means(self) -> np.ndarray:
        """Return current posterior means μ̄."""
        return self._mu.copy()

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
        self._n.fill(0.0)
        self._alpha.fill(1.0)
        self._beta.fill(1.0)
        self._t = 0
        self.M = 1.0
