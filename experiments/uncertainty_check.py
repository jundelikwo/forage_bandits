import json
from pathlib import Path
from typing import Dict, List, Tuple, Callable

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt

from forage_bandits.simulate import from_config
from forage_bandits.metrics import predicted_lifetime
from forage_bandits.energy_factors import energy_factor_linear, energy_factor_flip_exp


def run_simulation(
    cfg: DictConfig,
    uncertainty_level: float,
    alg_name: str,
    energy_adaptive: bool,
    eta: float = 1.0,
    custom_exploration_function: Callable[[float, bool], float] = None,
) -> Tuple[float, float, float, float]:
    """Run a single simulation and return final regret and mean lifetime.

    Parameters
    ----------
    cfg
        Base configuration to use
    uncertainty_level
        Uncertainty level
    alg_name
        Name of the algorithm ('egree', 'ucb', or 'ts')
    energy_adaptive
        Whether to use energy-adaptive version
    eta
        Exploration count offset
    custom_exploration_function
        Custom exploration function to use
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
    
    # Override initial energy in algorithm config
    sim_cfg.env.has_uncertainty = True
    sim_cfg.env.uncertainty_level = float(uncertainty_level)
    # Override algorithm config
    sim_cfg.alg.name = alg_name
    sim_cfg.alg.energy_adaptive = energy_adaptive
    sim_cfg.alg.eta = eta

    # Run simulation
    result = from_config(sim_cfg, custom_exploration_function)
    
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

class EnergyFactorFunction:
    """A callable class that can be pickled for multiprocessing."""
    
    def __init__(self, alpha: float, beta: float, energy_factor_alg: str):
        self.alpha = alpha
        self.beta = beta
        self.energy_factor_alg = energy_factor_alg
    
    def __call__(self, energy: float, energy_adaptive: bool) -> float:
        if energy_adaptive:
            energy_factor = energy_factor_flip_exp(energy) if self.energy_factor_alg == "flip_exp" else energy_factor_linear(energy)

            value = self.alpha + self.beta * energy_factor
        else:
            value = self.alpha + self.beta

        return value

def get_ucb_factor(alpha: float, beta: float, energy_factor_alg: str) -> Callable[[float, bool], float]:
    return EnergyFactorFunction(alpha, beta, energy_factor_alg)

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
    uncertainty_range = np.linspace(0.0, 0.5, 11)
    no_energy_custom_exploration_factor = get_ucb_factor(0.2, 0, "linear")
    linear_custom_exploration_factor = get_ucb_factor(0.8, -0.7, "linear")

    # Initialize results dictionaries
    results = {
        "e_greedy_no_energy": {"regret": {}, "regret_std": {}, "lifetime": {}, "lifetime_std": {}},
        "e_greedy_energy": {"regret": {}, "regret_std": {}, "lifetime": {}, "lifetime_std": {}},
        "ucb_no_energy": {"regret": {}, "regret_std": {}, "lifetime": {}, "lifetime_std": {}},
        "ucb_energy": {"regret": {}, "regret_std": {}, "lifetime": {}, "lifetime_std": {}},
        "ts_no_energy": {"regret": {}, "regret_std": {}, "lifetime": {}, "lifetime_std": {}},
        "ts_energy": {"regret": {}, "regret_std": {}, "lifetime": {}, "lifetime_std": {}},
    }
    
    # Run simulations for each number of arms
    for uncertainty_level in uncertainty_range:
        print(f"\nRunning simulations for uncertainty_level = {uncertainty_level}")
        
        # ε-Greedy
        print("  ε-Greedy (no energy)...")
        regret, regret_std, lifetime, lifetime_std = run_simulation(cfg, uncertainty_level, "egree", False, eta=0, custom_exploration_function=no_energy_custom_exploration_factor)
        results["e_greedy_no_energy"]["lifetime"][uncertainty_level] = lifetime
        results["e_greedy_no_energy"]["lifetime_std"][uncertainty_level] = lifetime_std
        results["e_greedy_no_energy"]["regret"][uncertainty_level] = regret
        results["e_greedy_no_energy"]["regret_std"][uncertainty_level] = regret_std
        
        print("  ε-Greedy (energy)...")
        regret, regret_std, lifetime, lifetime_std = run_simulation(cfg, uncertainty_level, "egree", True, eta=1, custom_exploration_function=linear_custom_exploration_factor)
        results["e_greedy_energy"]["lifetime"][uncertainty_level] = lifetime
        results["e_greedy_energy"]["lifetime_std"][uncertainty_level] = lifetime_std
        results["e_greedy_energy"]["regret"][uncertainty_level] = regret
        results["e_greedy_energy"]["regret_std"][uncertainty_level] = regret_std

        print("  UCB (no energy)...")
        regret, regret_std, lifetime, lifetime_std = run_simulation(cfg, uncertainty_level, "ucb", False, eta=0, custom_exploration_function=no_energy_custom_exploration_factor)
        results["ucb_no_energy"]["lifetime"][uncertainty_level] = lifetime
        results["ucb_no_energy"]["lifetime_std"][uncertainty_level] = lifetime_std
        results["ucb_no_energy"]["regret"][uncertainty_level] = regret
        results["ucb_no_energy"]["regret_std"][uncertainty_level] = regret_std
        
        print("  UCB (energy)...")
        regret, regret_std, lifetime, lifetime_std = run_simulation(cfg, uncertainty_level, "ucb", True, eta=1, custom_exploration_function=linear_custom_exploration_factor)
        results["ucb_energy"]["lifetime"][uncertainty_level] = lifetime
        results["ucb_energy"]["lifetime_std"][uncertainty_level] = lifetime_std
        results["ucb_energy"]["regret"][uncertainty_level] = regret
        results["ucb_energy"]["regret_std"][uncertainty_level] = regret_std

        print("  TS (no energy)...")
        regret, regret_std, lifetime, lifetime_std = run_simulation(cfg, uncertainty_level, "ts", False, eta=0, custom_exploration_function=no_energy_custom_exploration_factor)
        results["ts_no_energy"]["lifetime"][uncertainty_level] = lifetime
        results["ts_no_energy"]["lifetime_std"][uncertainty_level] = lifetime_std
        results["ts_no_energy"]["regret"][uncertainty_level] = regret
        results["ts_no_energy"]["regret_std"][uncertainty_level] = regret_std
        
        print("  TS (energy)...")
        regret, regret_std, lifetime, lifetime_std = run_simulation(cfg, uncertainty_level, "ts", True, eta=1, custom_exploration_function=linear_custom_exploration_factor)
        results["ts_energy"]["lifetime"][uncertainty_level] = lifetime
        results["ts_energy"]["lifetime_std"][uncertainty_level] = lifetime_std
        results["ts_energy"]["regret"][uncertainty_level] = regret
        results["ts_energy"]["regret_std"][uncertainty_level] = regret_std
    
    # Plot results
    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Effect of varying Uncertainty Level on Performance of ε-Greedy, UCB, and TS agents: {cfg.env.name} environment, n_arms={cfg.env.n_arms}', fontsize=16)


    for col, alg in enumerate(['e_greedy', 'ucb', 'ts']):
        ax = axes[0, col]
        
        # Plot no energy version
        no_energy = results[f"{alg}_no_energy"]
        ax.errorbar(
            uncertainty_range * 100,
            [no_energy['lifetime'][n] for n in uncertainty_range],
            # yerr=[no_energy['lifetime_std'][n] for n in n_arms_range],
            label='No Energy',
            marker='o',
            capsize=5
        )
        
        # Plot energy version
        energy = results[f"{alg}_energy"]
        ax.errorbar(
            uncertainty_range * 100,
            [energy['lifetime'][n] for n in uncertainty_range],
            # yerr=[energy['lifetime_std'][n] for n in n_arms_range],
            label='Energy',
            marker='s',
            capsize=5
        )
        
        # Customize subplot
        ax.set_xlabel('Uncertainty Level')
        ax.set_ylabel('Mean Lifetime')
        ax.set_title(f'{alg.upper()}')
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[1, col]
        
        # Plot no energy version
        no_energy = results[f"{alg}_no_energy"]
        ax.errorbar(
            uncertainty_range * 100,
            [no_energy['regret'][n] for n in uncertainty_range],
            # yerr=[no_energy['lifetime_std'][n] for n in n_arms_range],
            label='No Energy, eta=0',
            marker='o',
            capsize=5
        )
        
        # Plot energy version
        energy = results[f"{alg}_energy"]
        ax.errorbar(
            uncertainty_range * 100,
            [energy['regret'][n] for n in uncertainty_range],
            # yerr=[energy['lifetime_std'][n] for n in n_arms_range],
            label='Energy',
            marker='s',
            capsize=5
        )
        
        # Customize subplot
        ax.set_xlabel('Uncertainty Level')
        ax.set_ylabel('Final Regret')
        ax.set_title(f'{alg.upper()}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_dir / f"ucb_uncertainty_check_{cfg.env.name}_{cfg.env.n_arms}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    with open(output_dir / f"ucb_uncertainty_check_{cfg.env.name}_{cfg.env.n_arms}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to experiments/results/ucb_uncertainty_check_{cfg.env.name}_{cfg.env.n_arms}.json")
    print(f"Plot saved to experiments/results/ucb_uncertainty_check_{cfg.env.name}_{cfg.env.n_arms}.png")


if __name__ == "__main__":
    main() 