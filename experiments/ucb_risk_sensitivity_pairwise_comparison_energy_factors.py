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
from forage_bandits.metrics import predicted_lifetime, energy_trajectory, hazard_curve

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

    energy = energy_trajectory(result.rewards, Mf=sim_cfg.alg.forage_cost, M0=sim_cfg.alg.init_energy)
    hazard = hazard_curve(energy)
    regret = result.cumulative_regret
    exploring = result.exploring
    lifetimes = predicted_lifetime(hazard)
    Emax = np.log(50)
    energy = energy / Emax
    actions = result.actions

    # Count occurrences of each integer in the actions array
    # actions_subset = actions[:, 50:]  # Shape: (1000, 450)
    unique_values, counts = np.unique(actions, return_counts=True)
    print("Action counts:")
    for value, count in zip(unique_values, counts):
        print(f"  Action {value}: {count} times ({count/actions.size*100:.1f}%)")

    return hazard, regret, exploring, lifetimes, energy


# -----------------------------------------------------------------------------
# Hydra entry‑point
# -----------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:  # pragma: no cover
    first_run_energy_factors = ["flip_linear", "flip_exp", "exp"]
    second_run_energy_factors = ["thr", "parabolic", "sigmoid"]
    cfg.alg.forage_cost = 3.912023005428146 * 0.21
    cfg.alg.init_energy = 3.912023005428146 / 2

    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    plot_energy = cfg.plot.energy
    is_discounted = cfg.discounted_agents
    fig, axes = plt.subplots(6, 3, figsize=(18, 30), dpi=200)

    print("  UCB (no energy)...")
    no_energy_label = "Discounted UCB (no energy)" if is_discounted else "UCB (no energy)"
    hazard, regret, exploring, lifetimes, energy = run_simulation(cfg, "discounteducb" if is_discounted else "ucb", False, "linear", eta=0)
    ax = plot_lifetime_distribution(lifetimes, ax=axes[0, 0], label=no_energy_label)
    ax.set_xlim(0, 15)
    plot_energy_trajectory(energy, ax=axes[1, 0], label=no_energy_label)
    ax = plot_lifetime_distribution(lifetimes, ax=axes[0, 1], label=no_energy_label)
    ax.set_xlim(0, 15)
    plot_energy_trajectory(energy, ax=axes[1, 1], label=no_energy_label)
    ax = plot_lifetime_distribution(lifetimes, ax=axes[0, 2], label=no_energy_label)
    ax.set_xlim(0, 15)
    plot_energy_trajectory(energy, ax=axes[1, 2], label=no_energy_label)

    print("  UCB (energy factor: linear)...")
    linear_energy_label = "Discounted UCB (linear)" if is_discounted else "UCB (linear)"
    hazard, regret, exploring, lifetimes_linear, energy_linear = run_simulation(cfg, "discounteducb" if is_discounted else "ucb", True, "linear")

    for run in range(2):
        lifetime_x = 2 if run == 0 else 4
        energy_x = 3 if run == 0 else 5
        factors = first_run_energy_factors if run == 0 else second_run_energy_factors
        fig.suptitle(f"n_arms={cfg.env.n_arms}: {cfg.env.name} environment", fontsize=22, fontweight="bold", y=0.99)

        # UCB
        print(f"  UCB (energy factor: {factors[0]})...")
        hazard, regret, exploring, lifetimes, energy = run_simulation(cfg, "discounteducb" if is_discounted else "ucb", True, factors[0])
        energy_label = f"Discounted UCB ({factors[0]})" if is_discounted else f"UCB ({factors[0]})"
        ax = plot_lifetime_distribution(lifetimes_linear, ax=axes[lifetime_x, 0], energy_lifetimes=lifetimes, label=linear_energy_label, energy_label=energy_label)
        ax.set_xlim(0, 15)
        plot_energy_trajectory(energy_linear, ax=axes[energy_x, 0], label=linear_energy_label, energy_label=energy_label, energy_energy=energy)
        
        print(f"  UCB (energy factor: {factors[1]})...")
        hazard, regret, exploring, lifetimes, energy = run_simulation(cfg, "discounteducb" if is_discounted else "ucb", True, factors[1])
        energy_label = f"Discounted UCB ({factors[1]})" if is_discounted else f"UCB ({factors[1]})"
        ax = plot_lifetime_distribution(lifetimes_linear, ax=axes[lifetime_x, 1], energy_lifetimes=lifetimes, label=linear_energy_label, energy_label=energy_label)
        ax.set_xlim(0, 15)
        plot_energy_trajectory(energy_linear, ax=axes[energy_x, 1], label=linear_energy_label, energy_label=energy_label, energy_energy=energy)
        
        print(f"  UCB (energy factor: {factors[2]})...")
        hazard, regret, exploring, lifetimes, energy = run_simulation(cfg, "discounteducb" if is_discounted else "ucb", True, factors[2])
        energy_label = f"Discounted UCB ({factors[2]})" if is_discounted else f"UCB ({factors[2]})"
        ax = plot_lifetime_distribution(lifetimes_linear, ax=axes[lifetime_x, 2], energy_lifetimes=lifetimes, label=linear_energy_label, energy_label=energy_label)
        ax.set_xlim(0, 15)
        plot_energy_trajectory(energy_linear, ax=axes[energy_x, 2], label=linear_energy_label, energy_label=energy_label, energy_energy=energy)
    
    
    output_dir = Path("experiments/results")
    output_dir.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / f"ucb_risk_sensitivity_pairwise_comparison_energy_factors_{cfg.env.name}_{cfg.env.n_arms}_run{run}{"_discounted" if is_discounted else ""}.png")
    
    log.info("Done! Figures saved under %s", output_dir)


if __name__ == "__main__":
    main()
