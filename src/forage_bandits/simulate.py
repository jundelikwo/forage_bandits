"""forage_bandits.simulate
================================
Core driver that connects an *environment* with an *agent* and returns raw
roll‑out arrays **plus** convenient lazy‑computed metrics.  Everything here is
framework‑agnostic – no Hydra/OmegaConf – so you can import it from notebooks
or unit tests without heavy deps.

Quick example
-------------
>>> from forage_bandits.environments import SingleOptimalGaussian
>>> from forage_bandits.agents.ucb import UCB
>>> from forage_bandits.simulate import run_episode
>>> env = SingleOptimalGaussian(n_arms=10, mu_opt=0.8, mu_sub=0.2, rng=42)
>>> agent = UCB(n_arms=10, c=1.0, rng=43)
>>> res = run_episode(env, agent, T=1000)
>>> res.cumulative_regret[-1]
12.7

Public API
~~~~~~~~~~
* :func:`run_episode` – one (env, agent) pair → :class:`SimulationResult`
* :func:`run_batch`   – many i.i.d. env/agent pairs → :class:`SimulationResult`
* :func:`from_config` – minimal glue that instantiates env & agent from a
  *dict‑like* config (Hydra/OmegaConf friendly) and returns
  :class:`SimulationResult` (single run).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import concurrent.futures

from . import metrics as _m
from .environments import BanditEnvBase, make_env
from .agents import EpsilonGreedy, UCB, ThompsonSampling

__all__ = [
    "SimulationResult",
    "run_episode",
    "run_batch",
    "from_config",
]


# -----------------------------------------------------------------------------
# Interfaces (Protocols) for static‑type checking
# -----------------------------------------------------------------------------


class _AgentProto(Protocol):
    n_arms: int
    M: float  # Optional; EA agents expose this

    def act(self, t: int) -> int: ...

    def update(self, arm: int, reward: float) -> None: ...


# -----------------------------------------------------------------------------
# Result container
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class SimulationResult:
    """Container holding raw arrays plus lazily computed metrics."""

    rewards: np.ndarray  # shape (T,) or (N, T)
    actions: np.ndarray  # same shape as rewards
    energy: np.ndarray | None  # same shape or None if agent has no .M attr
    exploring: np.ndarray | None  # same shape or None if agent has no .M attr
    opt_mean: float  # true µ* of the environment

    # cache fields – populated on first access
    _cum_regret: np.ndarray | None = field(default=None, init=False, repr=False)
    _hazard: np.ndarray | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Properties – computed on demand & cached
    # ------------------------------------------------------------------

    @property
    def cumulative_regret(self) -> np.ndarray:
        if self._cum_regret is None:
            if self.rewards.ndim == 1:
                self._cum_regret = _m.cumulative_regret(self.rewards, self.opt_mean)
            else:
                self._cum_regret = np.vstack([
                    _m.cumulative_regret(r, self.opt_mean) for r in self.rewards
                ])
        return self._cum_regret

    @property
    def hazard(self) -> np.ndarray:
        if self.rewards.ndim == 1:
            raise ValueError("hazard is defined only for batch runs (N>1)")
        if self._hazard is None:
            # energy = _m.energy_trajectory(self.rewards, Mf=0.1, M0=1.0)
            energy = _m.energy_trajectory(self.rewards, Mf=np.log(50)/10, M0=np.log(50))
            self._hazard = _m.hazard_curve(energy)
        return self._hazard


# -----------------------------------------------------------------------------
# Core simulation loops
# -----------------------------------------------------------------------------


def run_episode(
    env: BanditEnvBase,
    agent: _AgentProto,
    T: int,
) -> SimulationResult:
    """Run *one* episode for *T* timesteps and return arrays.

    Parameters
    ----------
    env
        Environment instance whose true means determine regret baseline.
    agent
        Agent implementing ``act`` and ``update``.
    T
        Episode length.
    """
    rewards = np.empty(T, dtype=float)
    actions = np.empty(T, dtype=int)
    energy = np.empty(T, dtype=float) if hasattr(agent, "energy") else None
    exploring = np.empty(T, dtype=bool)

    for t in range(T):
        arm = agent.act(t)
        reward = env.pull(arm)

        rewards[t] = reward
        actions[t] = arm
        exploring[t] = agent.is_exploring
        if energy is not None:
            energy[t] = float(agent.energy)

        agent.update(arm, reward)

    return SimulationResult(
        rewards=rewards,
        actions=actions,
        energy=energy,
        exploring=exploring,
        opt_mean=float(env.true_means().max()),
    )


def _run_single_episode(
    env_factory: Callable[[int], BanditEnvBase],
    agent_factory: Callable[[int], _AgentProto],
    seed: int,
    T: int,
) -> SimulationResult:
    """Helper function for parallel processing of a single episode."""
    env = env_factory(seed)
    agent = agent_factory(seed)
    return run_episode(env, agent, T)


def run_batch(
    env_factory: Callable[[int], BanditEnvBase],
    agent_factory: Callable[[int], _AgentProto],
    T: int,
    n_runs: int,
    seed: int | None = None,
) -> SimulationResult:
    """Run *n_runs* episodes in parallel, each of length *T*."""
    if seed is None:
        seed = np.random.SeedSequence().entropy  # type: ignore[arg-type]
    ss = np.random.SeedSequence(int(seed))
    child_seeds = ss.spawn(n_runs)

    # Preallocate
    rewards = np.empty((n_runs, T), dtype=float)
    actions = np.empty((n_runs, T), dtype=int)
    energy = np.empty((n_runs, T), dtype=float)
    exploring = np.empty((n_runs, T), dtype=bool)
    opt_mean: float | None = None

    # Run episodes in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(_run_single_episode, env_factory, agent_factory, child_seeds[i].generate_state(1)[0], T)
            for i in range(n_runs)
        ]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    # Collect results
    for i, res_i in enumerate(results):
        rewards[i] = res_i.rewards
        actions[i] = res_i.actions
        exploring[i] = res_i.exploring
        energy[i] = res_i.energy
        if res_i.energy is None:
            raise RuntimeError("energy should not be None")
        if opt_mean is None:
            opt_mean = res_i.opt_mean

    if opt_mean is None:
        raise RuntimeError("opt_mean could not be determined (no runs executed?)")

    return SimulationResult(
        rewards=rewards,
        actions=actions,
        energy=energy,
        exploring=exploring,
        opt_mean=opt_mean,
    )


# -----------------------------------------------------------------------------
# Minimal config‑driven helper – single episode
# -----------------------------------------------------------------------------


def _make_agent_from_cfg(cfg, env: BanditEnvBase, rng: int | None = None) -> _AgentProto:  # noqa: ANN001
    """Create an agent from config, using environment for n_arms."""
    name = str(cfg.name).lower()
    n_arms = env.n_arms
    
    # Create a NumPy random generator from the seed
    agent_rng = np.random.default_rng(rng) if rng is not None else None

    # Get energy_adaptive flag from config, defaulting to False
    energy_adaptive = bool(getattr(cfg, "energy_adaptive", False))

    if name == "ucb" or name == "ea_ucb":
        return UCB(
            n_arms=n_arms,
            init_energy=np.log(50),
            Emax=np.log(50),
            c=float(getattr(cfg, "c", 1.0)),
            energy_adaptive=energy_adaptive,
            eta=float(getattr(cfg, "eta", 1.0)),
            forage_cost=np.log(50)/10,
            rng=agent_rng,
        )
    if name == "egree" or name == "ea_egree":
        return EpsilonGreedy(
            n_arms=n_arms,
            init_energy=np.log(50),
            Emax=np.log(50),
            epsilon=float(getattr(cfg, "epsilon", 0.2)),
            eta=float(getattr(cfg, "eta", 1.0)),
            energy_adaptive=energy_adaptive,
            forage_cost=np.log(50)/10,
            rng=agent_rng,
        )
    if name == "ts" or name == "ea_ts":
        return ThompsonSampling(
            n_arms=n_arms,
            init_energy=np.log(50),
            Emax=np.log(50),
            eta=float(getattr(cfg, "eta", 1.0)),
            energy_adaptive=energy_adaptive,
            forage_cost=np.log(50)/10,
            rng=agent_rng,
        )
    raise ValueError(f"Unknown agent name '{cfg.name}'.")


class EnvFactory:
    """Picklable environment factory."""
    def __init__(self, env_cfg):
        self.env_cfg = env_cfg

    def __call__(self, seed: int) -> BanditEnvBase:
        return make_env(self.env_cfg, seed)


class AgentFactory:
    """Picklable agent factory."""
    def __init__(self, env_cfg, alg_cfg):
        self.env_cfg = env_cfg
        self.alg_cfg = alg_cfg

    def __call__(self, seed: int) -> _AgentProto:
        env = make_env(self.env_cfg, seed)  # Need env for n_arms
        return _make_agent_from_cfg(self.alg_cfg, env, seed)


def from_config(cfg) -> SimulationResult:  # noqa: ANN001
    """Instantiate env & agent from a Hydra/OmegaConf‑style config.

    Expected minimal schema::

        env:
          name: single_optimal
          ...
        alg:
          name: ucb
          ...
        T: 1000
        n_runs: 1  # Optional; if > 1, runs a batch
    """
    seed = int(getattr(cfg, "seed", 1234))
    rng = np.random.default_rng(seed)
    T = int(getattr(cfg, "T", 1000))
    n_runs = int(getattr(cfg, "n_runs", 1))

    # Create factory instances
    env_factory = EnvFactory(cfg.env)
    agent_factory = AgentFactory(cfg.env, cfg.alg)

    if n_runs > 1:
        return run_batch(env_factory, agent_factory, T, n_runs, seed)
    else:
        # Single run
        env = env_factory(rng.integers(0, 2**32 - 1))
        agent = agent_factory(rng.integers(0, 2**32 - 1))
        return run_episode(env, agent, T)
