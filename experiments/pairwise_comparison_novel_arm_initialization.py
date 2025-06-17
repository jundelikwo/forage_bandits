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

    sim_cfg.alg.eta = eta

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
    fig, axes = plt.subplots(4, 2, figsize=(18, 20), dpi=300)
    fig.suptitle(f"n_arms={cfg.env.n_arms}: {cfg.env.name} environment", fontsize=22, fontweight="bold", y=0.99)

    # UCB
    print("  UCB (no energy, eta=0)...")
    hazard, regret, exploring, lifetimes = run_simulation(cfg, "ucb", False, eta=0)
    print("  UCB (no energy, eta=1)...")
    hazard_other, regret_other, exploring_other, lifetimes_other = run_simulation(cfg, "ucb", False, eta=1)
    print("  UCB (energy)...")
    hazard_ea, regret_ea, exploring_ea, lifetimes_ea = run_simulation(cfg, "ucb", True)

    plot_cumulative_regret(regret, ax=axes[0, 0], energy_regret=regret_ea, label="UCB", energy_label="EA-UCB", other_regret=regret_other, other_label="UCB (eta=1)")
    plot_exploration_rate(exploring, ax=axes[1, 0], energy_exploring=exploring_ea, label="UCB", energy_label="EA-UCB", start_explore=True, other_exploring=exploring_other, other_label="UCB (eta=1)")
    plot_hazard_curve(hazard, ax=axes[2, 0], energy_hazard=hazard_ea, label="UCB", energy_label="EA-UCB", ylim=(0.0, 0.15) if cfg.env.n_arms == 4 else None, other_hazard=hazard_other, other_label="UCB (eta=1)")
    plot_lifetime_distribution(lifetimes, ax=axes[3, 0], energy_lifetimes=lifetimes_ea, label="UCB", energy_label="EA-UCB", other_lifetimes=lifetimes_other, other_label="UCB (eta=1)")


    # TS
    print("  TS (no energy)...")
    hazard, regret, exploring, lifetimes = run_simulation(cfg, "ts", False, eta=0)
    print("  TS (no energy, eta=1)...")
    hazard_other, regret_other, exploring_other, lifetimes_other = run_simulation(cfg, "ts", False, eta=1)
    print("  TS (energy)...")
    hazard_ea, regret_ea, exploring_ea, lifetimes_ea = run_simulation(cfg, "ts", True)

    plot_cumulative_regret(regret, ax=axes[0, 1], energy_regret=regret_ea, label="TS", energy_label="EA-TS", other_regret=regret_other, other_label="TS (eta=1)")
    plot_exploration_rate(exploring, ax=axes[1, 1], energy_exploring=exploring_ea, label="TS", energy_label="EA-TS", start_explore=True, other_exploring=exploring_other, other_label="TS (eta=1)")
    plot_hazard_curve(hazard, ax=axes[2, 1], energy_hazard=hazard_ea, label="TS", energy_label="EA-TS", ylim=(0.0, 0.15) if cfg.env.n_arms == 4 else None, other_hazard=hazard_other, other_label="TS (eta=1)")
    plot_lifetime_distribution(lifetimes, ax=axes[3, 1], energy_lifetimes=lifetimes_ea, label="TS", energy_label="EA-TS", other_lifetimes=lifetimes_other, other_label="TS (eta=1)")
    
    output_dir = Path("experiments/results")
    output_dir.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / f"pairwise_comparison_novel_arm_initialization_{cfg.env.name}_{cfg.env.n_arms}.png")

    log.info("Done! Figures saved under %s", output_dir)


if __name__ == "__main__":
    main()
