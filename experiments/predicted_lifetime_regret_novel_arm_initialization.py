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
    n_arms: int,
    alg_name: str,
    energy_adaptive: bool,
    eta: float = 1.0,
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
    
    # Override n_arms in environment config
    sim_cfg.env.n_arms = n_arms
    
    # Override algorithm config
    sim_cfg.alg.name = alg_name
    sim_cfg.alg.energy_adaptive = energy_adaptive
    sim_cfg.alg.eta = eta
    
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
    results = {
        "ucb_no_energy": {"regret": {}, "regret_std": {}, "lifetime": {}, "lifetime_std": {}},
        "ucb_no_energy_eta_1": {"regret": {}, "regret_std": {}, "lifetime": {}, "lifetime_std": {}},
        "ucb_energy": {"regret": {}, "regret_std": {}, "lifetime": {}, "lifetime_std": {}},
        "ts_no_energy": {"regret": {}, "regret_std": {}, "lifetime": {}, "lifetime_std": {}},
        "ts_no_energy_eta_1": {"regret": {}, "regret_std": {}, "lifetime": {}, "lifetime_std": {}},
        "ts_energy": {"regret": {}, "regret_std": {}, "lifetime": {}, "lifetime_std": {}},
    }
    print(f"cfg: {cfg}")
    
    # Run simulations for each number of arms
    for n_arms in range(2, 16):
        print(f"\nRunning simulations for n_arms = {n_arms}")
        
        # UCB
        print("  UCB (no energy)...")
        regret, regret_std, lifetime, lifetime_std = run_simulation(cfg, n_arms, "ucb", False, eta=0)
        results["ucb_no_energy"]["regret"][n_arms] = regret
        results["ucb_no_energy"]["regret_std"][n_arms] = regret_std
        results["ucb_no_energy"]["lifetime"][n_arms] = lifetime
        results["ucb_no_energy"]["lifetime_std"][n_arms] = lifetime_std
        
        print("  UCB (no energy, eta=1)...")
        regret, regret_std, lifetime, lifetime_std = run_simulation(cfg, n_arms, "ucb", False, eta=1)
        results["ucb_no_energy_eta_1"]["regret"][n_arms] = regret
        results["ucb_no_energy_eta_1"]["regret_std"][n_arms] = regret_std
        results["ucb_no_energy_eta_1"]["lifetime"][n_arms] = lifetime
        results["ucb_no_energy_eta_1"]["lifetime_std"][n_arms] = lifetime_std
        
        print("  UCB (energy)...")
        regret, regret_std, lifetime, lifetime_std = run_simulation(cfg, n_arms, "ucb", True, eta=1)
        results["ucb_energy"]["regret"][n_arms] = regret
        results["ucb_energy"]["regret_std"][n_arms] = regret_std
        results["ucb_energy"]["lifetime"][n_arms] = lifetime
        results["ucb_energy"]["lifetime_std"][n_arms] = lifetime_std
        
        # Thompson Sampling
        print("  TS (no energy)...")
        regret, regret_std, lifetime, lifetime_std = run_simulation(cfg, n_arms, "ts", False, eta=0)
        results["ts_no_energy"]["regret"][n_arms] = regret
        results["ts_no_energy"]["regret_std"][n_arms] = regret_std
        results["ts_no_energy"]["lifetime"][n_arms] = lifetime
        results["ts_no_energy"]["lifetime_std"][n_arms] = lifetime_std
        
        print("  TS (no energy, eta=1)...")
        regret, regret_std, lifetime, lifetime_std = run_simulation(cfg, n_arms, "ts", False, eta=1)
        results["ts_no_energy_eta_1"]["regret"][n_arms] = regret
        results["ts_no_energy_eta_1"]["regret_std"][n_arms] = regret_std
        results["ts_no_energy_eta_1"]["lifetime"][n_arms] = lifetime
        results["ts_no_energy_eta_1"]["lifetime_std"][n_arms] = lifetime_std
        
        print("  TS (energy)...")
        regret, regret_std, lifetime, lifetime_std = run_simulation(cfg, n_arms, "ts", True, eta=1)
        results["ts_energy"]["regret"][n_arms] = regret
        results["ts_energy"]["regret_std"][n_arms] = regret_std
        results["ts_energy"]["lifetime"][n_arms] = lifetime
        results["ts_energy"]["lifetime_std"][n_arms] = lifetime_std
    
    # Plot results
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Effect of Number of Arms on Performance: {cfg.env.name} environment', fontsize=16)
    
    # Set up x-axis values (number of arms)
    n_arms_range = list(range(2, 16))
    
    # Plot lifetime (top row)
    for col, alg in enumerate(['ucb', 'ts']):
        ax = axes[0, col]
        
        # Plot no energy version
        no_energy = results[f"{alg}_no_energy"]
        ax.errorbar(
            n_arms_range,
            [no_energy['lifetime'][n] for n in n_arms_range],
            label='No Energy',
            marker='o',
            capsize=5
        )

        no_energy_eta_1 = results[f"{alg}_no_energy_eta_1"]
        ax.errorbar(
            n_arms_range,
            [no_energy_eta_1['lifetime'][n] for n in n_arms_range],
            label='No Energy (eta=1)',
            marker='o',
            capsize=5
        )
        
        # Plot energy version
        energy = results[f"{alg}_energy"]
        ax.errorbar(
            n_arms_range,
            [energy['lifetime'][n] for n in n_arms_range],
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
    for col, alg in enumerate(['ucb', 'ts']):
        ax = axes[1, col]
        
        # Plot no energy version
        no_energy = results[f"{alg}_no_energy"]
        ax.errorbar(
            n_arms_range,
            [no_energy['regret'][n] for n in n_arms_range],
            label='No Energy',
            marker='o',
            capsize=5
        )
        
        no_energy_eta_1 = results[f"{alg}_no_energy_eta_1"]
        ax.errorbar(
            n_arms_range,
            [no_energy_eta_1['regret'][n] for n in n_arms_range],
            label='No Energy (eta=1)',
            marker='o',
            capsize=5
        )
        
        # Plot energy version
        energy = results[f"{alg}_energy"]
        ax.errorbar(
            n_arms_range,
            [energy['regret'][n] for n in n_arms_range],
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
    plt.savefig(output_dir / f"predicted_lifetime_regret_novel_arm_initialization_{cfg.env.name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    with open(output_dir / f"predicted_lifetime_regret_novel_arm_initialization_{cfg.env.name}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to experiments/results/predicted_lifetime_regret_novel_arm_initialization_{cfg.env.name}.json")
    print(f"Plot saved to experiments/results/predicted_lifetime_regret_novel_arm_initialization_{cfg.env.name}.png")


if __name__ == "__main__":
    main() 