from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf

from forage_bandits.plotters import (
    plot_cumulative_regret,
    plot_hazard_curve,
    plot_lifetime_distribution,
    plot_exploration_rate,
    plot_energy_trajectory,
)
from forage_bandits.simulate import from_config
from forage_bandits.metrics import predicted_lifetime, energy_trajectory

log = logging.getLogger(__name__)

def run_simulation(cfg: DictConfig, alg_name, energy_adaptive, eta=1):
    """Run batch simulation for given algorithm configuration"""
    # Create a copy of the config to modify
    sim_cfg = OmegaConf.create(cfg)

    sim_cfg.alg.eta = eta if eta == 1 else 1e-10

    sim_cfg.alg.name = alg_name
    sim_cfg.alg.energy_adaptive = energy_adaptive
    result = from_config(sim_cfg)

    hazard = result.hazard
    regret = result.cumulative_regret
    exploring = result.exploring
    lifetimes = predicted_lifetime(hazard)
    Emax = np.log(50)
    energy = energy_trajectory(result.rewards, Mf=Emax/10, M0=Emax) / Emax

    return hazard, regret, exploring, lifetimes, energy


# -----------------------------------------------------------------------------
# Hydra entry‑point
# -----------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:  # pragma: no cover
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    plot_energy = cfg.plot.energy
    fig, axes = plt.subplots(5 if plot_energy else 4, 3, figsize=(18, 25 if plot_energy else 20), dpi=300)
    fig.suptitle(f"n_arms={cfg.env.n_arms}: {cfg.env.name} environment", fontsize=22, fontweight="bold", y=0.99)

    is_discounted = cfg.discounted_agents


    # ε-Greedy
    print("  ε-Greedy (no energy)...")
    hazard, regret, exploring, lifetimes, energy = run_simulation(cfg, "discountedegree" if is_discounted else "egree", False, eta=0)
    print("  ε-Greedy (energy)...")
    hazard_ea, regret_ea, exploring_ea, lifetimes_ea, energy_ea = run_simulation(cfg, "discountedegree" if is_discounted else "egree", True)

    no_energy_label = "Discounted ε-Greedy" if is_discounted else "ε-Greedy"
    energy_label = "Discounted EA-ε-Greedy" if is_discounted else "EA-ε-Greedy"
    plot_cumulative_regret(regret, ax=axes[0, 0], energy_regret=regret_ea, label=no_energy_label, energy_label=energy_label)
    # Since I manually set the counts of all the arms to 1 at the first timestep, the exploration rate is 1.0
    plot_exploration_rate(exploring, ax=axes[1, 0], energy_exploring=exploring_ea, label=no_energy_label, energy_label=energy_label, start_explore=True)
    plot_hazard_curve(hazard, ax=axes[2, 0], energy_hazard=hazard_ea, label=no_energy_label, energy_label=energy_label)
    plot_lifetime_distribution(lifetimes, ax=axes[3, 0], energy_lifetimes=lifetimes_ea, label=no_energy_label, energy_label=energy_label)
    if plot_energy:
        plot_energy_trajectory(energy, ax=axes[4, 0], label=no_energy_label, energy_label=energy_label, energy_energy=energy_ea)


    # UCB
    print("  UCB (no energy)...")
    hazard, regret, exploring, lifetimes, energy = run_simulation(cfg, "discounteducb" if is_discounted else "ucb", False, eta=0)
    print("  UCB (energy)...")
    hazard_ea, regret_ea, exploring_ea, lifetimes_ea, energy_ea = run_simulation(cfg, "discounteducb" if is_discounted else "ucb", True)

    no_energy_label = "Discounted UCB" if is_discounted else "UCB"
    energy_label = "Discounted EA-UCB" if is_discounted else "EA-UCB"
    plot_cumulative_regret(regret, ax=axes[0, 1], energy_regret=regret_ea, label=no_energy_label, energy_label=energy_label)
    plot_exploration_rate(exploring, ax=axes[1, 1], energy_exploring=exploring_ea, label=no_energy_label, energy_label=energy_label, start_explore=True)
    plot_hazard_curve(hazard, ax=axes[2, 1], energy_hazard=hazard_ea, label=no_energy_label, energy_label=energy_label, ylim=(0.0, 0.15) if cfg.env.n_arms == 4 else None)
    plot_lifetime_distribution(lifetimes, ax=axes[3, 1], energy_lifetimes=lifetimes_ea, label=no_energy_label, energy_label=energy_label)
    if plot_energy:
        plot_energy_trajectory(energy, ax=axes[4, 1], label=no_energy_label, energy_label=energy_label, energy_energy=energy_ea)


    # TS
    print("  TS (no energy)...")
    hazard, regret, exploring, lifetimes, energy = run_simulation(cfg, "discountedts" if is_discounted else "ts", False, eta=0)
    print("  TS (energy)...")
    hazard_ea, regret_ea, exploring_ea, lifetimes_ea, energy_ea = run_simulation(cfg, "discountedts" if is_discounted else "ts", True)

    no_energy_label = "Discounted TS" if is_discounted else "TS"
    energy_label = "Discounted EA-TS" if is_discounted else "EA-TS"
    plot_cumulative_regret(regret, ax=axes[0, 2], energy_regret=regret_ea, label=no_energy_label, energy_label=energy_label)
    plot_exploration_rate(exploring, ax=axes[1, 2], energy_exploring=exploring_ea, label=no_energy_label, energy_label=energy_label, start_explore=True)
    plot_hazard_curve(hazard, ax=axes[2, 2], energy_hazard=hazard_ea, label=no_energy_label, energy_label=energy_label, ylim=(0.0, 0.15) if cfg.env.n_arms == 4 else None)
    plot_lifetime_distribution(lifetimes, ax=axes[3, 2], energy_lifetimes=lifetimes_ea, label=no_energy_label, energy_label=energy_label)
    if plot_energy:
        plot_energy_trajectory(energy, ax=axes[4, 2], label=no_energy_label, energy_label=energy_label, energy_energy=energy_ea)

    output_dir = Path("experiments/results")
    output_dir.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / f"pairwise_comparison_{cfg.env.name}_{cfg.env.n_arms}{"_discounted" if is_discounted else ""}.png")

    log.info("Done! Figures saved under %s", output_dir)


if __name__ == "__main__":
    main()
