from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf

from forage_bandits.plotters import (
    plot_cumulative_regret,
    plot_hazard_curve,
    plot_lifetime_distribution,
    plot_exploration_rate,
)
from forage_bandits.simulate import from_config
from forage_bandits.metrics import predicted_lifetime

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

    return hazard, regret, exploring, lifetimes


# -----------------------------------------------------------------------------
# Hydra entry‑point
# -----------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:  # pragma: no cover
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    fig, axes = plt.subplots(4, 3, figsize=(18, 20), dpi=300)
    fig.suptitle(f"n_arms={cfg.env.n_arms}: {cfg.env.name} environment", fontsize=22, fontweight="bold", y=0.99)


    # ε-Greedy
    print("  ε-Greedy (no energy)...")
    hazard, regret, exploring, lifetimes = run_simulation(cfg, "egree", False, eta=0)
    print("  ε-Greedy (energy)...")
    hazard_ea, regret_ea, exploring_ea, lifetimes_ea = run_simulation(cfg, "egree", True)

    plot_cumulative_regret(regret, ax=axes[0, 0], energy_regret=regret_ea, label="ε-Greedy", energy_label="EA-ε-Greedy")
    # Since I manually set the counts of all the arms to 1 at the first timestep, the exploration rate is 1.0
    plot_exploration_rate(exploring, ax=axes[1, 0], energy_exploring=exploring_ea, label="ε-Greedy", energy_label="EA-ε-Greedy", start_explore=True)
    plot_hazard_curve(hazard, ax=axes[2, 0], energy_hazard=hazard_ea, label="ε-Greedy", energy_label="EA-ε-Greedy")
    plot_lifetime_distribution(lifetimes, ax=axes[3, 0], energy_lifetimes=lifetimes_ea, label="ε-Greedy", energy_label="EA-ε-Greedy")


    # UCB
    print("  UCB (no energy)...")
    hazard, regret, exploring, lifetimes = run_simulation(cfg, "ucb", False, eta=0)
    print("  UCB (energy)...")
    hazard_ea, regret_ea, exploring_ea, lifetimes_ea = run_simulation(cfg, "ucb", True)

    plot_cumulative_regret(regret, ax=axes[0, 1], energy_regret=regret_ea, label="UCB", energy_label="EA-UCB")
    plot_exploration_rate(exploring, ax=axes[1, 1], energy_exploring=exploring_ea, label="UCB", energy_label="EA-UCB", start_explore=True)
    plot_hazard_curve(hazard, ax=axes[2, 1], energy_hazard=hazard_ea, label="UCB", energy_label="EA-UCB", ylim=(0.0, 0.15) if cfg.env.n_arms == 4 else None)
    plot_lifetime_distribution(lifetimes, ax=axes[3, 1], energy_lifetimes=lifetimes_ea, label="UCB", energy_label="EA-UCB")


    # TS
    print("  TS (no energy)...")
    hazard, regret, exploring, lifetimes = run_simulation(cfg, "ts", False, eta=0)
    print("  TS (energy)...")
    hazard_ea, regret_ea, exploring_ea, lifetimes_ea = run_simulation(cfg, "ts", True)

    plot_cumulative_regret(regret, ax=axes[0, 2], energy_regret=regret_ea, label="TS", energy_label="EA-TS")
    plot_exploration_rate(exploring, ax=axes[1, 2], energy_exploring=exploring_ea, label="TS", energy_label="EA-TS", start_explore=True)
    plot_hazard_curve(hazard, ax=axes[2, 2], energy_hazard=hazard_ea, label="TS", energy_label="EA-TS", ylim=(0.0, 0.15) if cfg.env.n_arms == 4 else None)
    plot_lifetime_distribution(lifetimes, ax=axes[3, 2], energy_lifetimes=lifetimes_ea, label="TS", energy_label="EA-TS")
    
    output_dir = Path("experiments/results")
    output_dir.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / f"pairwise_comparison_{cfg.env.name}_{cfg.env.n_arms}.png")

    log.info("Done! Figures saved under %s", output_dir)


if __name__ == "__main__":
    main()
