"""
Forage Bandits: A Python package for simulating foraging behavior using multi-armed bandit algorithms.

This package implements various bandit algorithms (ϵ-greedy, UCB, Thompson Sampling) with
energy-aware modifications for modeling foraging behavior in changing environments.
"""

__version__ = "0.1.0"
__author__ = "Forage Bandits Contributors"

# Core components
from .environments import BanditEnvBase, make_env
from .metrics import cumulative_regret, energy_trajectory, lifetime, hazard_curve, MetricsBatch
from .simulate import run_episode, run_batch, from_config, SimulationResult

# Agent implementations
from .agents import EpsilonGreedy, UCB, ThompsonSampling

__all__ = [
    "BanditEnvBase",
    "make_env",
    "cumulative_regret",
    "energy_trajectory",
    "lifetime",
    "hazard_curve",
    "MetricsBatch",
    "run_episode",
    "run_batch",
    "from_config",
    "SimulationResult",
    "EpsilonGreedy",
    "UCB",
    "ThompsonSampling",
]
