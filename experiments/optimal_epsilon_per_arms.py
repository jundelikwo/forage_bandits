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
    epsilon: float,
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
    epsilon
        Epsilon value
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
    
    # Override epsilon in algorithm config
    sim_cfg.alg.epsilon = float(epsilon)
    
    # Override algorithm config
    sim_cfg.alg.name = alg_name
    sim_cfg.alg.energy_adaptive = energy_adaptive
    sim_cfg.alg.eta = eta

    # Override n_arms in environment config
    sim_cfg.env.n_arms = int(n_arms)
    
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

    print(f"cfg: {cfg}")
    
    # Initialize results dictionaries
    energy_results = {
        "max_lifetimes": [],
        "epsilon_max_lifetimes": [],
        "min_regrets": [],
        "epsilon_min_regrets": [],
    }

    no_energy_results = {
        "max_lifetimes": [],
        "epsilon_max_lifetimes": [],
        "min_regrets": [],
        "epsilon_min_regrets": [],
    }
    
    epsilon_range = np.linspace(0, 1, 50)
    n_arms_range = np.arange(2, 17, 1)

    # Initialize results dictionaries
    
    # Run simulations for each number of arms
    for n_arms in n_arms_range:
        print(f"\nRunning simulations for n_arms = {n_arms}")

        results = {
            "e_greedy_no_energy": {"regret": {}, "lifetime": {}},
            "e_greedy_energy": {"regret": {}, "lifetime": {}},
        }

        # Run simulations for each number of arms
        for epsilon in epsilon_range:
            print(f"\nRunning simulations for epsilon = {epsilon}")
            
            # ε-Greedy
            print("  ε-Greedy (no energy)...")
            regret, regret_std, lifetime, lifetime_std = run_simulation(cfg, epsilon, n_arms, "egree", False, eta=0)
            results["e_greedy_no_energy"]["lifetime"][epsilon] = lifetime
            results["e_greedy_no_energy"]["regret"][epsilon] = regret
            
            print("  ε-Greedy (energy)...")
            regret, regret_std, lifetime, lifetime_std = run_simulation(cfg, epsilon, n_arms, "egree", True, eta=1)
            results["e_greedy_energy"]["lifetime"][epsilon] = lifetime
            results["e_greedy_energy"]["regret"][epsilon] = regret

        no_energy = results["e_greedy_no_energy"]
        lifetime = np.array([no_energy['lifetime'][n] for n in epsilon_range])
        regret = np.array([no_energy['regret'][n] for n in epsilon_range])
        no_energy_results["max_lifetimes"].append(np.max(lifetime))
        no_energy_results["epsilon_max_lifetimes"].append(epsilon_range[np.argmax(lifetime)])
        no_energy_results["min_regrets"].append(np.min(regret))
        no_energy_results["epsilon_min_regrets"].append(epsilon_range[np.argmin(regret)])


        energy = results["e_greedy_energy"]
        lifetime = np.array([energy['lifetime'][n] for n in epsilon_range])
        regret = np.array([energy['regret'][n] for n in epsilon_range])
        energy_results["max_lifetimes"].append(np.max(lifetime))
        energy_results["epsilon_max_lifetimes"].append(epsilon_range[np.argmax(lifetime)])
        energy_results["min_regrets"].append(np.min(regret))
        energy_results["epsilon_min_regrets"].append(epsilon_range[np.argmin(regret)])
        
    
    # Plot results
    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, figsize=(15, 10))
    fig.suptitle(f'Optimal epsilon value for different number of arms for the ε-Greedy agent: {cfg.env.name} environment', fontsize=16)

    # Plot lifetime (top row)
    ax = axes[0]
    
    # Plot no energy version
    ax.errorbar(
        n_arms_range,
        no_energy_results["epsilon_max_lifetimes"],
        label='No Energy',
        marker='o',
        capsize=5
    )
    
    # Plot energy version
    energy = results["e_greedy_energy"]
    ax.errorbar(
        n_arms_range,
        energy_results["epsilon_max_lifetimes"],
        label='Energy',
        marker='s',
        capsize=5
    )
    
    # Plot regret (bottom row)
    ax = axes[1]
    
    # Plot no energy version
    no_energy = results["e_greedy_no_energy"]
    ax.errorbar(
        n_arms_range,
        no_energy_results["epsilon_min_regrets"],
        label='No Energy',
        marker='o',
        capsize=5
    )
    
    # Plot energy version
    energy = results["e_greedy_energy"]
    ax.errorbar(
        n_arms_range,
        energy_results["epsilon_min_regrets"],
        label='Energy',
        marker='s',
        capsize=5
    )
    
    # Customize subplot
    axes[0].set_xlabel('Number of Arms')
    axes[0].set_ylabel('Epsilon')
    axes[0].set_title(f'Maximum Mean Lifetime')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].set_xlabel('Number of Arms')
    axes[1].set_ylabel('Epsilon')
    axes[1].set_title(f'Minimun Final Regret')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_dir / f"optimal_epsilon_per_arms_{cfg.env.name}.png", dpi=300, bbox_inches='tight')
    plt.close()




    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, figsize=(15, 10))
    fig.suptitle(f'Maximum mean lifetime and minimum final regret for the optimal epsilon value for different number of arms for the ε-Greedy agent: {cfg.env.name} environment', fontsize=16)

    # Plot lifetime (top row)
    ax = axes[0]
    
    # Plot no energy version
    ax.errorbar(
        n_arms_range,
        no_energy_results["max_lifetimes"],
        label='No Energy',
        marker='o',
        capsize=5
    )
    
    # Plot energy version
    energy = results["e_greedy_energy"]
    ax.errorbar(
        n_arms_range,
        energy_results["max_lifetimes"],
        label='Energy',
        marker='s',
        capsize=5
    )
    
    # Plot regret (bottom row)
    ax = axes[1]
    
    # Plot no energy version
    no_energy = results["e_greedy_no_energy"]
    ax.errorbar(
        n_arms_range,
        no_energy_results["min_regrets"],
        label='No Energy',
        marker='o',
        capsize=5
    )
    
    # Plot energy version
    energy = results["e_greedy_energy"]
    ax.errorbar(
        n_arms_range,
        energy_results["min_regrets"],
        label='Energy',
        marker='s',
        capsize=5
    )
    
    # Customize subplot
    axes[0].set_xlabel('Number of Arms')
    axes[0].set_ylabel('Mean Lifetime')
    axes[0].set_title(f'Maximum Mean Lifetime')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].set_xlabel('Number of Arms')
    axes[1].set_ylabel('Final Regret')
    axes[1].set_title(f'Minimun Final Regret')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_dir / f"optimal_epsilon_per_arms_raw_values_{cfg.env.name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    results = {
        "energy": energy_results,
        "no_energy": no_energy_results,
    }
    with open(output_dir / f"optimal_epsilon_per_arms_{cfg.env.name}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to experiments/results/optimal_epsilon_per_arms_{cfg.env.name}.json")
    print(f"Plot saved to experiments/results/optimal_epsilon_per_arms_{cfg.env.name}.png")


if __name__ == "__main__":
    main() 