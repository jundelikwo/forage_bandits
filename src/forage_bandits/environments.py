"""forage_bandits.environments

Stochastic bandit environments used throughout Chapter 4 experiments.
Each environment exposes a single method::

    reward = env.pull(arm)

which samples a reward from the underlying distribution of *arm* and returns it
as a **float in [0, 1]** (all experiments in the thesis normalise rewards so
that energy bookkeeping remains meaningful).

Two concrete environments are implemented out‑of‑the‑box, matching the paper:

* ``SingleOptimalEnv`` – K arms, one has mean μ⋆, the rest share μ₋ (Sec. 4.3.1).
* ``SigmoidEnv``       – mean reward follows a sigmoid curve over arm index
                          (Eq. 4.17), modelling a graded difficulty.

Both are *stationary* and *independent*: pulling an arm does not change its
future distribution.

A convenience factory ``make_env(cfg)`` allows Hydra‑style configs to instantiate
an environment by name.

All randomness is handled by ``numpy.random.Generator`` injected at init time so
simulations are fully reproducible.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Union, Dict, Any

import numpy as np
from omegaconf import DictConfig

__all__ = [
    "BanditEnvBase",
    "SingleOptimalEnv",
    "SigmoidEnv",
    "make_env",
]


class BanditEnvBase(Protocol):
    """Minimal interface every environment must implement."""

    n_arms: int

    def pull(self, arm: int) -> float:  # noqa: D401
        """Sample a reward *r ∈ [0, log(50)]* for the given arm index."""

    def true_means(self) -> np.ndarray:  # noqa: D401
        """Return the vector of mean rewards μ of shape (n_arms,)."""


# ---------------------------------------------------------------------------
# Concrete environments
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class _GaussianArm:
    mean: float
    sigma: float = 0.1

    def sample(self, rng: np.random.Generator) -> float:  # noqa: D401
        return float(np.maximum(0, rng.normal(self.mean, self.sigma)))


class SingleOptimalEnv:
    """K‑armed bandit with a single best arm.

    Parameters
    ----------
    n_arms : int
        Number of arms (*K* in the thesis).
    mu_opt : float, default 1.0
        Mean reward of the optimal arm.
    mu_sub : float, default 0.0
        Mean reward of all sub‑optimal arms.
    sigma  : float, default 0.1
        Shared standard deviation for Gaussian noise.
    opt_index : int | None
        Which arm is optimal. If *None* (default) a random index is chosen.
    rng : int | np.random.Generator | None
        Seed or Generator for reproducibility.
    """

    def __init__(
        self,
        n_arms: int,
        *,
        mu_opt: float = 0.2,
        mu_sub: float = 0.04,
        sigma: float = 0.02,
        opt_index: int | None = None,
        rng: Union[int, np.random.Generator, None] = None,
    ) -> None:
        if n_arms < 2:
            raise ValueError("SingleOptimalEnv requires at least 2 arms.")
        
        Emax = np.log(50)
        basal_cost = Emax / 10
        mu_opt = 2 * basal_cost  # Optimal arm mean
        mu_sub = 0.2 * mu_opt    # Suboptimal arm mean
        sigma = 0.1

        self._rng = np.random.default_rng(rng)
        self.n_arms = int(n_arms)
        # self.opt_index = self._rng.integers(0, n_arms) if opt_index is None else int(opt_index)
        self.opt_index = n_arms - 1 if opt_index is None else int(opt_index)

        self._arms = [
            _GaussianArm(mu_opt if i == self.opt_index else mu_sub, sigma) for i in range(n_arms)
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def pull(self, arm: int) -> float:  # noqa: D401
        return self._arms[arm].sample(self._rng)

    # ------------------------------------------------------------------
    def true_means(self) -> np.ndarray:  # noqa: D401
        return np.array([arm.mean for arm in self._arms], dtype=np.float64)

    # ------------------------------------------------------------------
    # Extras
    # ------------------------------------------------------------------
    def optimal_mean(self) -> float:  # noqa: D401
        return self._arms[self.opt_index].mean


class SigmoidEnv:
    """K‑armed bandit where μ_i follows a sigmoid curve.

    Mean reward of arm *i* (0‑based) is

        μ_i = 1 / (1 + exp(-k (i - i₀)))                          (Eq. 4.17)

    with steepness *k* and centre *i₀ = (K − 1)/2* by default.
    Rewards are Gaussian around μ_i with std *sigma*.
    """

    def __init__(
        self,
        n_arms: int,
        *,
        mu_opt: float = 0.2,
        k: float = 10.0,
        sigma: float = 0.02,
        rng: Union[int, np.random.Generator, None] = None,
    ) -> None:
        self._rng = np.random.default_rng(rng)
        self.n_arms = int(n_arms)
        self.k = float(k)
        self.sigma = float(sigma)
        self.mu_opt = float(mu_opt)
        Emax = np.log(50)
        basal_cost = Emax / 10
        mu_opt = 2 * basal_cost  # Optimal arm mean
        self.mu_opt = mu_opt

        # Compute means once
        self._means = np.array([
            # self.mu_opt / (1 + math.exp(-self.k * ((i+1)/n_arms - 0.5)))
            self.mu_opt / (1 + math.exp(-self.k * (i/(n_arms-1) - 0.5)))
            for i in range(n_arms)
        ], dtype=np.float64)

    # ------------------------------------------------------------------
    def pull(self, arm: int) -> float:  # noqa: D401
        y = 1 / (1 + np.exp(-self.k * ((arm)/(self.n_arms - 1) - 0.5)))
        reward = self._rng.normal(self.mu_opt * y, 0.1 * self.mu_opt)
        return reward
        # y = 1 / (1 + np.exp(-self.k * ((arm - 0.5)/self.n_arms - 0.5)))
        # reward = self._rng.normal(self.mu_opt * y, 0.1 * self.mu_opt)
        # return reward
        # return float(np.clip(reward, 0.0, 1.0))

    def true_means(self) -> np.ndarray:  # noqa: D401
        return self._means.copy()
    

class DynamicSingleOptimalEnv:
    """K‑armed bandit with a single best arm that changes every 20 pulls.

    Parameters
    ----------
    n_arms : int
        Number of arms (*K* in the thesis).
    mu_opt : float, default 1.0
        Mean reward of the optimal arm.
    mu_sub : float, default 0.0
        Mean reward of all sub‑optimal arms.
    sigma  : float, default 0.1
        Shared standard deviation for Gaussian noise.
    opt_index : int | None
        Which arm is optimal initially. If *None* (default) a random index is chosen.
    rng : int | np.random.Generator | None
        Seed or Generator for reproducibility.
    """

    def __init__(
        self,
        n_arms: int,
        *,
        mu_opt: float = 0.2,
        mu_sub: float = 0.04,
        sigma: float = 0.02,
        opt_index: int | None = None,
        rng: Union[int, np.random.Generator, None] = None,
    ) -> None:
        if n_arms < 2:
            raise ValueError("SingleOptimalEnv requires at least 2 arms.")
        
        Emax = np.log(50)
        basal_cost = Emax / 10
        self.mu_opt = 2 * basal_cost  # Optimal arm mean
        self.mu_sub = 0.2 * self.mu_opt    # Suboptimal arm mean
        self.sigma = 0.1

        self._rng = np.random.default_rng(rng)
        self.n_arms = int(n_arms)
        self.opt_index = n_arms - 1 if opt_index is None else int(opt_index)
        
        # Track number of pulls for dynamic changes
        self.pull_count = 0
        self.change_interval = 20

        self._update_arms()

    def _update_arms(self):
        """Update the arms based on current optimal index."""
        self._arms = [
            _GaussianArm(self.mu_opt if i == self.opt_index else self.mu_sub, self.sigma) 
            for i in range(self.n_arms)
        ]

    def _change_optimal_arm(self):
        """Change the optimal arm to a random new one."""
        old_opt = self.opt_index
        while self.opt_index == old_opt:  # Ensure we pick a different arm
            self.opt_index = self._rng.integers(0, self.n_arms)
        self._update_arms()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def pull(self, arm: int) -> float:  # noqa: D401
        # Check if we need to change the optimal arm
        self.pull_count += 1
        if self.pull_count % self.change_interval == 0:
            self._change_optimal_arm()
        
        return self._arms[arm].sample(self._rng)

    # ------------------------------------------------------------------
    def true_means(self) -> np.ndarray:  # noqa: D401
        return np.array([arm.mean for arm in self._arms], dtype=np.float64)

    # ------------------------------------------------------------------
    # Extras
    # ------------------------------------------------------------------
    def optimal_mean(self) -> float:  # noqa: D401
        return self._arms[self.opt_index].mean


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------
_ENV_REGISTRY = {
    "single_optimal": SingleOptimalEnv,
    "dynamic_single_optimal": DynamicSingleOptimalEnv,
    "sigmoid": SigmoidEnv,
}


def make_env(cfg: Union[str, DictConfig, Dict[str, Any]], seed: int | None = None) -> BanditEnvBase:
    """Instantiate an environment from a config or by name.
    
    Parameters
    ----------
    cfg : str | DictConfig | Dict[str, Any]
        Either a string name of the environment, or a config object/dict with
        environment parameters. If a string, must be one of: single_optimal, dynamic_single_optimal, sigmoid.
    seed : int | None, optional
        Random seed for reproducibility. If None, a random seed is used.
        
    Returns
    -------
    BanditEnvBase
        An instance of the requested environment.
        
    Raises
    ------
    ValueError
        If the environment name is unknown or required parameters are missing.
    """
    if isinstance(cfg, str):
        name = cfg.lower()
        try:
            cls = _ENV_REGISTRY[name]
        except KeyError as ex:
            raise ValueError(f"Unknown environment '{name}'. Available: {list(_ENV_REGISTRY)}") from ex
        return cls(rng=seed)
    
    # Handle config object/dict
    if isinstance(cfg, DictConfig):
        cfg = dict(cfg)
    
    name = cfg.pop("name", None)
    if name is None:
        raise ValueError("Config must contain 'name' field")
    
    try:
        cls = _ENV_REGISTRY[name.lower()]
    except KeyError as ex:
        raise ValueError(f"Unknown environment '{name}'. Available: {list(_ENV_REGISTRY)}") from ex
    
    # Add seed to kwargs if provided
    if seed is not None:
        cfg["rng"] = seed
    
    return cls(**cfg)
