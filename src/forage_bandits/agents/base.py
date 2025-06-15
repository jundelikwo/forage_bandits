"""forage_bandits.agents.base
Core abstract interface that all bandit agents must implement.

The simulator will interact with an *Agent* in the following loop:

1. ``action = agent.act(t)`` – choose an arm for step *t*.
2. Environment executes the action and returns a reward.
3. ``agent.update(action, reward)`` – agent learns from feedback.

Sub‑classes implement concrete algorithms such as ε‑greedy, UCB, or
Thompson sampling by filling in :py:meth:`act` and :py:meth:`update`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class AgentBase(ABC):
    """Abstract base‑class for all bandit agents."""

    def __init__(
        self,
        n_actions: int,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Construct an **uninitialised** agent.

        Parameters
        ----------
        n_actions
            Number of distinct arms/actions available in the environment.
        seed
            Optional deterministic seed for this agent's RNG.  Sub‑classes can
            ignore this or use it to create a *private* NumPy generator.
        **kwargs
            Catch‑all for algorithm‑specific hyper‑parameters.  The simulator
            passes the whole Hydra/OmegaConf *alg* node; unknown keys should
            simply be ignored so that config evolution does not break older
            agents.
        """
        self.n_actions = n_actions
        self.seed = seed
        self._rng = None  # lazy NumPy generator created on first use

    # ------------------------------------------------------------------
    #  Public API to be implemented by concrete agents
    # ------------------------------------------------------------------

    @abstractmethod
    def act(self, t: int) -> int:  # noqa: D401 – imperative style
        """Return the index of the action to play at time‑step *t*.

        The returned integer **must** satisfy ``0 <= action < n_actions``.
        """

    @abstractmethod
    def update(self, action: int, reward: float) -> None:  # noqa: D401
        """Observe the reward obtained for *action* and update internal state."""

    @property
    def is_exploring(self) -> bool:
        """Return True if the last action was exploratory"""
        return False  # Default implementation

    # ------------------------------------------------------------------
    #  Optional helper hooks
    # ------------------------------------------------------------------

    def reset(self) -> None:  # noqa: D401 – imperative style
        """Clear all internal statistics so the agent can be reused."""
        pass

    # ------------------------------------------------------------------
    #  Internal utilities
    # ------------------------------------------------------------------

    def _get_rng(self):  # noqa: D401 – imperative style
        """Lazily create & return a NumPy random generator bound to *seed*."""
        if self._rng is None:
            import numpy as np

            self._rng = np.random.default_rng(self.seed)
        return self._rng
