"""Experiment 1: Effect of number of arms on regret and lifetime.

This script runs simulations for different numbers of arms (2-15) and different
algorithms (ε-Greedy, UCB, TS) in both energy-adaptive and non-energy-adaptive
versions. For each combination, it records the final regret and mean lifetime.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt

from forage_bandits.simulate import from_config
from forage_bandits.metrics import hazard_curve, predicted_lifetime


def run_simulation(
    cfg: DictConfig,
    n_arms: int,
    alg_name: str,
    energy_adaptive: bool,
) -> Tuple[float, float, float, float]:
    """Run a single simulation and return final regret and mean lifetime.

    Parameters
    ----------
    cfg
        Base configuration to use
    n_arms
        Number of arms to use
    alg_name
        Name of the algorithm ('egree', 'ucb', or 'ts')
    energy_adaptive
        Whether to use energy-adaptive version

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
    
    # Override n_arms in environment config
    sim_cfg.env.n_arms = n_arms
    
    # Override algorithm config
    sim_cfg.alg.name = alg_name
    sim_cfg.alg.energy_adaptive = energy_adaptive
    
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
        lifetimes = []
        regrets = []
        for run in range(result.energy.shape[0]):
            energy = result.energy[run]
            hazard = hazard_curve(energy)
            lifetime = predicted_lifetime(hazard)
            lifetimes.append(lifetime)
            regrets.append(result.cumulative_regret[run, -1])
        lifetimes = np.array(lifetimes)
        regrets = np.array(regrets)
        mean_regret = float(np.mean(regrets))
        mean_regret_std = float(np.std(regrets))
        mean_lifetime = float(np.mean(lifetimes))
        mean_lifetime_std = float(np.std(lifetimes))
    else:
        raise RuntimeError("result.energy should not be None")
    
    return mean_regret, mean_regret_std, mean_lifetime, mean_lifetime_std


def main() -> None:
    """Run the experiment and save results."""
    # Get the absolute path to the configs directory
    config_dir = Path(__file__).parent.parent / "configs"
    
    # Load base config
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = hydra.compose(config_name="base")
    
    # Initialize results dictionaries
    results = {
        "e_greedy_no_energy": {"regret": {}, "regret_std": {}, "lifetime": {}, "lifetime_std": {}},
        "e_greedy_energy": {"regret": {}, "regret_std": {}, "lifetime": {}, "lifetime_std": {}},
        "ucb_no_energy": {"regret": {}, "regret_std": {}, "lifetime": {}, "lifetime_std": {}},
        "ucb_energy": {"regret": {}, "regret_std": {}, "lifetime": {}, "lifetime_std": {}},
        "ts_no_energy": {"regret": {}, "regret_std": {}, "lifetime": {}, "lifetime_std": {}},
        "ts_energy": {"regret": {}, "regret_std": {}, "lifetime": {}, "lifetime_std": {}},
    }
    print(f"cfg: {cfg}")
    
    # Run simulations for each number of arms
    for n_arms in range(2, 16):
        print(f"\nRunning simulations for n_arms = {n_arms}")
        
        # ε-Greedy
        print("  ε-Greedy (no energy)...")
        regret, regret_std, lifetime, lifetime_std = run_simulation(cfg, n_arms, "egree", False)
        results["e_greedy_no_energy"]["regret"][n_arms] = regret
        results["e_greedy_no_energy"]["regret_std"][n_arms] = regret_std
        results["e_greedy_no_energy"]["lifetime"][n_arms] = lifetime
        results["e_greedy_no_energy"]["lifetime_std"][n_arms] = lifetime_std
        
        print("  ε-Greedy (energy)...")
        regret, regret_std, lifetime, lifetime_std = run_simulation(cfg, n_arms, "egree", True)
        results["e_greedy_energy"]["regret"][n_arms] = regret
        results["e_greedy_energy"]["regret_std"][n_arms] = regret_std
        results["e_greedy_energy"]["lifetime"][n_arms] = lifetime
        results["e_greedy_energy"]["lifetime_std"][n_arms] = lifetime_std
        
        # UCB
        print("  UCB (no energy)...")
        regret, regret_std, lifetime, lifetime_std = run_simulation(cfg, n_arms, "ucb", False)
        results["ucb_no_energy"]["regret"][n_arms] = regret
        results["ucb_no_energy"]["regret_std"][n_arms] = regret_std
        results["ucb_no_energy"]["lifetime"][n_arms] = lifetime
        results["ucb_no_energy"]["lifetime_std"][n_arms] = lifetime_std
        
        print("  UCB (energy)...")
        regret, regret_std, lifetime, lifetime_std = run_simulation(cfg, n_arms, "ucb", True)
        results["ucb_energy"]["regret"][n_arms] = regret
        results["ucb_energy"]["regret_std"][n_arms] = regret_std
        results["ucb_energy"]["lifetime"][n_arms] = lifetime
        results["ucb_energy"]["lifetime_std"][n_arms] = lifetime_std
        
        # Thompson Sampling
        print("  TS (no energy)...")
        regret, regret_std, lifetime, lifetime_std = run_simulation(cfg, n_arms, "ts", False)
        results["ts_no_energy"]["regret"][n_arms] = regret
        results["ts_no_energy"]["regret_std"][n_arms] = regret_std
        results["ts_no_energy"]["lifetime"][n_arms] = lifetime
        results["ts_no_energy"]["lifetime_std"][n_arms] = lifetime_std
        
        print("  TS (energy)...")
        regret, regret_std, lifetime, lifetime_std = run_simulation(cfg, n_arms, "ts", True)
        results["ts_energy"]["regret"][n_arms] = regret
        results["ts_energy"]["regret_std"][n_arms] = regret_std
        results["ts_energy"]["lifetime"][n_arms] = lifetime
        results["ts_energy"]["lifetime_std"][n_arms] = lifetime_std
    
    # Plot results
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Effect of Number of Arms on Performance', fontsize=16)
    
    # Set up x-axis values (number of arms)
    n_arms_range = list(range(2, 16))
    
    # Plot lifetime (top row)
    for col, alg in enumerate(['e_greedy', 'ucb', 'ts']):
        ax = axes[0, col]
        
        # Plot no energy version
        no_energy = results[f"{alg}_no_energy"]
        ax.errorbar(
            n_arms_range,
            [no_energy['lifetime'][n] for n in n_arms_range],
            # yerr=[no_energy['lifetime_std'][n] for n in n_arms_range],
            label='No Energy',
            marker='o',
            capsize=5
        )
        
        # Plot energy version
        energy = results[f"{alg}_energy"]
        ax.errorbar(
            n_arms_range,
            [energy['lifetime'][n] for n in n_arms_range],
            # yerr=[energy['lifetime_std'][n] for n in n_arms_range],
            label='Energy',
            marker='s',
            capsize=5
        )
        
        # Customize subplot
        ax.set_xlabel('Number of Arms')
        ax.set_ylabel('Mean Lifetime')
        ax.set_title(f'{alg.upper()}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Plot regret (bottom row)
    for col, alg in enumerate(['e_greedy', 'ucb', 'ts']):
        ax = axes[1, col]
        
        # Plot no energy version
        no_energy = results[f"{alg}_no_energy"]
        ax.errorbar(
            n_arms_range,
            [no_energy['regret'][n] for n in n_arms_range],
            # yerr=[no_energy['regret_std'][n] for n in n_arms_range],
            label='No Energy',
            marker='o',
            capsize=5
        )
        
        # Plot energy version
        energy = results[f"{alg}_energy"]
        ax.errorbar(
            n_arms_range,
            [energy['regret'][n] for n in n_arms_range],
            # yerr=[energy['regret_std'][n] for n in n_arms_range],
            label='Energy',
            marker='s',
            capsize=5
        )
        
        # Customize subplot
        ax.set_xlabel('Number of Arms')
        ax.set_ylabel('Final Regret')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('experiments/results/experiment_1_arms.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    output_dir = Path("experiments/results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "experiment_1_arms.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to experiments/results/experiment_1_arms.json")
    print("Plot saved to experiments/results/experiment_1_arms.png")


if __name__ == "__main__":
    main() 