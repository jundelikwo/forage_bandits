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

def run_simulation(cfg: DictConfig, alg_name, energy_adaptive, energy_factor, eta=1):
    """Run batch simulation for given algorithm configuration"""
    # Create a copy of the config to modify
    sim_cfg = OmegaConf.create(cfg)

    sim_cfg.alg.eta = eta if eta == 1 else 1e-10

    sim_cfg.alg.name = alg_name
    sim_cfg.alg.energy_adaptive = energy_adaptive
    sim_cfg.alg.energy_factor_alg = energy_factor
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
    first_run_energy_factors = ["linear", "flip_exp", "exp"]
    second_run_energy_factors = ["thr", "parabolic", "sigmoid"]

    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    plot_energy = cfg.plot.energy
    is_discounted = cfg.discounted_agents

    for run in range(2):
        factors = first_run_energy_factors if run == 0 else second_run_energy_factors
        fig, axes = plt.subplots(5 if plot_energy else 4, 3, figsize=(18, 25 if plot_energy else 20), dpi=200 if plot_energy else 300)
        fig.suptitle(f"n_arms={cfg.env.n_arms}: {cfg.env.name} environment", fontsize=22, fontweight="bold", y=0.99)

        # ε-Greedy
        print(f"  ε-Greedy (energy factor: {factors[0]})...")
        hazard, regret, exploring, lifetimes, energy = run_simulation(cfg, "discountedegree" if is_discounted else "egree", True, factors[0])
        print(f"  ε-Greedy (energy factor: {factors[1]})...")
        hazard_ea, regret_ea, exploring_ea, lifetimes_ea, energy_ea = run_simulation(cfg, "discountedegree" if is_discounted else "egree", True, factors[1])
        print(f"  ε-Greedy (energy factor: {factors[2]})...")
        hazard_other, regret_other, exploring_other, lifetimes_other, energy_other = run_simulation(cfg, "discountedegree" if is_discounted else "egree", True, factors[2])

        no_energy_label = f"Discounted ε-Greedy ({factors[0]})" if is_discounted else f"ε-Greedy ({factors[0]})"
        energy_label = f"Discounted ε-Greedy ({factors[1]})" if is_discounted else f"ε-Greedy ({factors[1]})"
        other_label = f"Discounted ε-Greedy ({factors[2]})" if is_discounted else f"ε-Greedy ({factors[2]})"
        plot_cumulative_regret(regret, ax=axes[0, 0], energy_regret=regret_ea, label=no_energy_label, energy_label=energy_label, other_label=other_label, other_regret=regret_other)
        # Since I manually set the counts of all the arms to 1 at the first timestep, the exploration rate is 1.0
        plot_exploration_rate(exploring, ax=axes[1, 0], energy_exploring=exploring_ea, label=no_energy_label, energy_label=energy_label, other_label=other_label, other_exploring=exploring_other, start_explore=True)
        plot_hazard_curve(hazard, ax=axes[2, 0], energy_hazard=hazard_ea, label=no_energy_label, energy_label=energy_label, other_label=other_label, other_hazard=hazard_other)
        plot_lifetime_distribution(lifetimes, ax=axes[3, 0], energy_lifetimes=lifetimes_ea, label=no_energy_label, energy_label=energy_label, other_label=other_label, other_lifetimes=lifetimes_other)
        if plot_energy:
            plot_energy_trajectory(energy, ax=axes[4, 0], label=no_energy_label, energy_label=energy_label, energy_energy=energy_ea, other_label=other_label, other_energy=energy_other)


        # UCB
        print(f"  UCB (energy factor: {factors[0]})...")
        hazard, regret, exploring, lifetimes, energy = run_simulation(cfg, "discounteducb" if is_discounted else "ucb", True, factors[0])
        print(f"  UCB (energy factor: {factors[1]})...")
        hazard_ea, regret_ea, exploring_ea, lifetimes_ea, energy_ea = run_simulation(cfg, "discounteducb" if is_discounted else "ucb", True, factors[1])
        print(f"  UCB (energy factor: {factors[2]})...")
        hazard_other, regret_other, exploring_other, lifetimes_other, energy_other = run_simulation(cfg, "discounteducb" if is_discounted else "ucb", True, factors[2])

        no_energy_label = f"Discounted UCB ({factors[0]})" if is_discounted else f"UCB ({factors[0]})"
        energy_label = f"Discounted UCB ({factors[1]})" if is_discounted else f"UCB ({factors[1]})"
        other_label = f"Discounted UCB ({factors[2]})" if is_discounted else f"UCB ({factors[2]})"
        plot_cumulative_regret(regret, ax=axes[0, 1], energy_regret=regret_ea, label=no_energy_label, energy_label=energy_label, other_label=other_label, other_regret=regret_other)
        plot_exploration_rate(exploring, ax=axes[1, 1], energy_exploring=exploring_ea, label=no_energy_label, energy_label=energy_label, other_label=other_label, other_exploring=exploring_other, start_explore=True)
        plot_hazard_curve(hazard, ax=axes[2, 1], energy_hazard=hazard_ea, label=no_energy_label, energy_label=energy_label, other_label=other_label, other_hazard=hazard_other, ylim=(0.0, 0.15) if cfg.env.n_arms == 4 else None)
        plot_lifetime_distribution(lifetimes, ax=axes[3, 1], energy_lifetimes=lifetimes_ea, label=no_energy_label, energy_label=energy_label, other_label=other_label, other_lifetimes=lifetimes_other)
        if plot_energy:
            plot_energy_trajectory(energy, ax=axes[4, 1], label=no_energy_label, energy_label=energy_label, energy_energy=energy_ea, other_label=other_label, other_energy=energy_other)


        # TS
        print(f"  TS (energy factor: {factors[0]})...")
        hazard, regret, exploring, lifetimes, energy = run_simulation(cfg, "discountedts" if is_discounted else "ts", True, factors[0])
        print(f"  TS (energy factor: {factors[1]})...")
        hazard_ea, regret_ea, exploring_ea, lifetimes_ea, energy_ea = run_simulation(cfg, "discountedts" if is_discounted else "ts", True, factors[1])
        print(f"  TS (energy factor: {factors[2]})...")
        hazard_other, regret_other, exploring_other, lifetimes_other, energy_other = run_simulation(cfg, "discountedts" if is_discounted else "ts", True, factors[2])

        no_energy_label = f"Discounted TS ({factors[0]})" if is_discounted else f"TS ({factors[0]})"
        energy_label = f"Discounted TS ({factors[1]})" if is_discounted else f"TS ({factors[1]})"
        other_label = f"Discounted TS ({factors[2]})" if is_discounted else f"TS ({factors[2]})"
        plot_cumulative_regret(regret, ax=axes[0, 2], energy_regret=regret_ea, label=no_energy_label, energy_label=energy_label, other_label=other_label, other_regret=regret_other)
        plot_exploration_rate(exploring, ax=axes[1, 2], energy_exploring=exploring_ea, label=no_energy_label, energy_label=energy_label, other_label=other_label, other_exploring=exploring_other, start_explore=True)
        plot_hazard_curve(hazard, ax=axes[2, 2], energy_hazard=hazard_ea, label=no_energy_label, energy_label=energy_label, other_label=other_label, other_hazard=hazard_other, ylim=(0.0, 0.15) if cfg.env.n_arms == 4 else None)
        plot_lifetime_distribution(lifetimes, ax=axes[3, 2], energy_lifetimes=lifetimes_ea, label=no_energy_label, energy_label=energy_label, other_label=other_label, other_lifetimes=lifetimes_other)
        if plot_energy:
            plot_energy_trajectory(energy, ax=axes[4, 2], label=no_energy_label, energy_label=energy_label, energy_energy=energy_ea, other_label=other_label, other_energy=energy_other)

        output_dir = Path("experiments/results")
        output_dir.mkdir(exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_dir / f"pairwise_comparison_energy_factors_{cfg.env.name}_{cfg.env.n_arms}_run{run}{"_discounted" if is_discounted else ""}.png")

    log.info("Done! Figures saved under %s", output_dir)


if __name__ == "__main__":
    main()
