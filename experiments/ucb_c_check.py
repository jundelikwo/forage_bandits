import json
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt

from forage_bandits.simulate import from_config
from forage_bandits.metrics import hazard_curve, predicted_lifetime, energy_trajectory


def run_simulation(
    cfg: DictConfig,
    c: float,
    alg_name: str,
    energy_adaptive: bool,
    eta: float = 1.0,
) -> Tuple[float, float, float, float]:
    """Run a single simulation and return final regret and mean lifetime.

    Parameters
    ----------
    cfg
        Base configuration to use
    c
        Exploration parameter
    alg_name
        Name of the algorithm ('egree', 'ucb', or 'ts')
    energy_adaptive
        Whether to use energy-adaptive version
    eta
        Exploration count offset

    Returns
    -------
    final_regret
        Mean regret at the final timestep
    final_regret_std
        Standard deviation of regret at the final timestep
    mean_lifetime
        Mean lifetime across all runs
    mean_lifetime_std
        Standard deviation of lifetime across all runs
    """
    # Create a copy of the config to modify
    sim_cfg = OmegaConf.create(cfg)
    
    # Override algorithm config
    sim_cfg.alg.name = alg_name
    sim_cfg.alg.energy_adaptive = energy_adaptive
    sim_cfg.alg.eta = eta
    sim_cfg.alg.c = float(c)
    
    # Run simulation
    result = from_config(sim_cfg)
    
    # Get final regret (mean and std across runs)
    if result.rewards.ndim == 1:
        # Single run
        final_regret = float(result.cumulative_regret[-1])
        final_regret_std = 0.0  # No std for single run
    else:
        # Batch run - calculate regret for each run, then mean and std at final timestep
        final_regret = float(result.cumulative_regret[:, -1].mean())
        final_regret_std = float(result.cumulative_regret[:, -1].std())
    
    # Calculate mean lifetime and std
    if result.energy is not None:
        hazard = result.hazard
        lifetimes = predicted_lifetime(hazard)
        mean_lifetime = float(np.mean(lifetimes))
        mean_lifetime_std = float(np.std(lifetimes))
    else:
        raise RuntimeError("result.energy should not be None")
    
    return final_regret, final_regret_std, mean_lifetime, mean_lifetime_std

# -----------------------------------------------------------------------------
# Hydra entry‑point
# -----------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    output_dir = Path("experiments/results")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize results dictionaries
    print(f"cfg: {cfg}")
    
    # Set up x-axis values (number of arms)
    c_range = np.linspace(0, 5, 100)

    # Initialize results dictionaries
    results = {
        "ucb_no_energy": {"regret": {}, "regret_std": {}, "lifetime": {}, "lifetime_std": {}},
        "ucb_no_energy_eta_1": {"regret": {}, "regret_std": {}, "lifetime": {}, "lifetime_std": {}},
        "ucb_energy": {"regret": {}, "regret_std": {}, "lifetime": {}, "lifetime_std": {}},
    }
    
    # Run simulations for each number of arms
    for c in c_range:
        print(f"\nRunning simulations for c = {c}")
        
        # ε-Greedy
        print("  UCB (no energy)...")
        regret, regret_std, lifetime, lifetime_std = run_simulation(cfg, c, "ucb", False, eta=0)
        results["ucb_no_energy"]["lifetime"][c] = lifetime
        results["ucb_no_energy"]["lifetime_std"][c] = lifetime_std
        results["ucb_no_energy"]["regret"][c] = regret
        results["ucb_no_energy"]["regret_std"][c] = regret_std

        print("  UCB (no energy, eta=1)...")
        regret, regret_std, lifetime, lifetime_std = run_simulation(cfg, c, "ucb", False, eta=1)
        results["ucb_no_energy_eta_1"]["lifetime"][c] = lifetime
        results["ucb_no_energy_eta_1"]["lifetime_std"][c] = lifetime_std
        results["ucb_no_energy_eta_1"]["regret"][c] = regret
        results["ucb_no_energy_eta_1"]["regret_std"][c] = regret_std
        
        print("  UCB (energy)...")
        regret, regret_std, lifetime, lifetime_std = run_simulation(cfg, c, "ucb", True, eta=1)
        results["ucb_energy"]["lifetime"][c] = lifetime
        results["ucb_energy"]["lifetime_std"][c] = lifetime_std
        results["ucb_energy"]["regret"][c] = regret
        results["ucb_energy"]["regret_std"][c] = regret_std
    
    # Plot results
    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, figsize=(15, 10))
    fig.suptitle(f'Effect of varying c on Performance of UCB agent: {cfg.env.name} environment, n_arms={cfg.env.n_arms}', fontsize=16)

    # Plot lifetime (top row)
    ax = axes[0]
    
    # Plot no energy version
    no_energy = results["ucb_no_energy"]
    ax.errorbar(
        c_range,
        [no_energy['lifetime'][n] for n in c_range],
        # yerr=[no_energy['lifetime_std'][n] for n in epsilon_range],
        label='No Energy, eta=0',
        marker='o',
        capsize=5
    )
    
    # Plot no energy version
    no_energy = results["ucb_no_energy_eta_1"]
    ax.errorbar(
        c_range,
        [no_energy['lifetime'][n] for n in c_range],
        # yerr=[no_energy['lifetime_std'][n] for n in epsilon_range],
        label='No Energy, eta=1',
        marker='x',
        capsize=5
    )
    
    # Plot energy version
    energy = results["ucb_energy"]
    ax.errorbar(
        c_range,
        [energy['lifetime'][n] for n in c_range],
        # yerr=[energy['lifetime_std'][n] for n in epsilon_range],
        label='Energy',
        marker='s',
        capsize=5
    )
    
    # Plot regret (bottom row)
    ax = axes[1]
    
    # Plot no energy version
    no_energy = results["ucb_no_energy"]
    ax.errorbar(
        c_range,
        [no_energy['regret'][n] for n in c_range],
        # yerr=[no_energy['regret_std'][n] for n in epsilon_range],
        label='No Energy, eta=0',
        marker='o',
        capsize=5
    )

    no_energy = results["ucb_no_energy_eta_1"]
    ax.errorbar(
        c_range,
        [no_energy['regret'][n] for n in c_range],
        # yerr=[no_energy['regret_std'][n] for n in epsilon_range],
        label='No Energy, eta=1',
        marker='x',
        capsize=5
    )
    
    # Plot energy version
    energy = results["ucb_energy"]
    ax.errorbar(
        c_range,
        [energy['regret'][n] for n in c_range],
        # yerr=[energy['regret_std'][n] for n in epsilon_range],
        label='Energy',
        marker='s',
        capsize=5
    )
    
    # Customize subplot
    axes[0].set_xlabel('c')
    axes[0].set_ylabel('Mean Lifetime')
    axes[0].set_title(f'UCB')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].set_xlabel('c')
    axes[1].set_ylabel('Final Regret')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_dir / f"ucb_c_check_{cfg.env.name}_{cfg.env.n_arms}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    with open(output_dir / f"ucb_c_check_{cfg.env.name}_{cfg.env.n_arms}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to experiments/results/ucb_c_check_{cfg.env.name}_{cfg.env.n_arms}.json")
    print(f"Plot saved to experiments/results/ucb_c_check_{cfg.env.name}_{cfg.env.n_arms}.png")


if __name__ == "__main__":
    main() 