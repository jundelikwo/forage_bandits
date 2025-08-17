import json
from pathlib import Path
from typing import Dict, List, Tuple, Callable

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt

from forage_bandits.simulate import from_config
from forage_bandits.energy_factors import energy_factor_linear, energy_factor_flip_exp
from forage_bandits.metrics import hazard_curve, predicted_lifetime, energy_trajectory


def run_simulation(
    cfg: DictConfig,
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
    ucb
        Epsilon value
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
    sim_cfg.alg.eta = int(eta)
    
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

class UCBFactorFunction:
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
    return UCBFactorFunction(alpha, beta, energy_factor_alg)

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
    eta_range = np.linspace(0, 10, 21)

    # Initialize results dictionaries
    results = {
        "ucb_no_energy": {"regret": {}, "regret_std": {}, "lifetime": {}, "lifetime_std": {}},
        "ucb_energy_flip": {"regret": {}, "regret_std": {}, "lifetime": {}, "lifetime_std": {}},
        "ucb_energy_linear": {"regret": {}, "regret_std": {}, "lifetime": {}, "lifetime_std": {}},
    }
    
    no_energy_custom_exploration_factor = get_ucb_factor(0.11, 0, "linear")
    linear_custom_exploration_factor = get_ucb_factor(1.0, -0.79, "linear")
    flip_custom_exploration_factor = get_ucb_factor(0.16, 1.0, "flip_exp")
    # Run simulations for each number of arms
    for eta in eta_range:
        print(f"\nRunning simulations for eta = {eta}")
        
        # ε-Greedy
        print("  UCB (no energy)...")
        regret, regret_std, lifetime, lifetime_std = run_simulation(cfg, "ucb", False, eta=eta, custom_exploration_function=no_energy_custom_exploration_factor)
        results["ucb_no_energy"]["lifetime"][eta] = lifetime
        results["ucb_no_energy"]["lifetime_std"][eta] = lifetime_std
        results["ucb_no_energy"]["regret"][eta] = regret
        results["ucb_no_energy"]["regret_std"][eta] = regret_std

        print("  UCB (energy, flip)...")
        regret, regret_std, lifetime, lifetime_std = run_simulation(cfg, "ucb", True, eta=eta, custom_exploration_function=flip_custom_exploration_factor)
        results["ucb_energy_flip"]["lifetime"][eta] = lifetime
        results["ucb_energy_flip"]["lifetime_std"][eta] = lifetime_std
        results["ucb_energy_flip"]["regret"][eta] = regret
        results["ucb_energy_flip"]["regret_std"][eta] = regret_std
        
        print("  UCB (energy, linear)...")
        regret, regret_std, lifetime, lifetime_std = run_simulation(cfg, "ucb", True, eta=eta, custom_exploration_function=linear_custom_exploration_factor)
        results["ucb_energy_linear"]["lifetime"][eta] = lifetime
        results["ucb_energy_linear"]["lifetime_std"][eta] = lifetime_std
        results["ucb_energy_linear"]["regret"][eta] = regret
        results["ucb_energy_linear"]["regret_std"][eta] = regret_std
    
    # Plot results
    for i in range(2):
        # Create figure with 2 subplots
        fig, axes = plt.subplots(2, figsize=(15, 10))
        fig.suptitle(f'Effect of varying eta on Performance of UCB agent: {cfg.env.name} environment, n_arms={cfg.env.n_arms}', fontsize=16)

        # Plot lifetime (top row)
        ax = axes[0]
        
        # Plot no energy version
        no_energy = results["ucb_no_energy"]
        ax.errorbar(
            eta_range,
            [no_energy['lifetime'][n] for n in eta_range],
            # yerr=[no_energy['lifetime_std'][n] for n in epsilon_range],
            label='No Energy',
            marker='o',
            capsize=5
        )
        
        # Plot linear energy version
        linear_energy = results["ucb_energy_linear"]
        ax.errorbar(
            eta_range,
            [linear_energy['lifetime'][n] for n in eta_range],
            # yerr=[linear_energy['lifetime_std'][n] for n in epsilon_range],
            label='Energy, linear',
            marker='s',
            capsize=5
        )
        
        if i == 0:
            # Plot flip energy version
            flip_energy = results["ucb_energy_flip"]
            ax.errorbar(
                eta_range,
                [flip_energy['lifetime'][n] for n in eta_range],
                # yerr=[flip_energy['lifetime_std'][n] for n in epsilon_range],
                label='Energy, flip',
                marker='x',
                capsize=5
            )
        
        # Plot regret (bottom row)
        ax = axes[1]
        
        # Plot no energy version
        no_energy = results["ucb_no_energy"]
        ax.errorbar(
            eta_range,
            [no_energy['regret'][n] for n in eta_range],
            # yerr=[no_energy['regret_std'][n] for n in epsilon_range],
            label='No Energy',
            marker='o',
            capsize=5
        )
        
        # Plot energy version
        linear_energy = results["ucb_energy_linear"]
        ax.errorbar(
            eta_range,
            [linear_energy['regret'][n] for n in eta_range],
            # yerr=[energy['regret_std'][n] for n in epsilon_range],
            label='Energy, linear',
            marker='s',
            capsize=5
        )

        if i == 0:
            flip_energy = results["ucb_energy_flip"]
            ax.errorbar(
                eta_range,
                [flip_energy['regret'][n] for n in eta_range],
                # yerr=[flip_energy['regret_std'][n] for n in epsilon_range],
                label='Energy, flip',
                marker='x',
                capsize=5
            )
        
        # Customize subplot
        axes[0].set_xlabel('eta')
        axes[0].set_ylabel('Mean Lifetime')
        axes[0].set_title(f'UCB')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        axes[1].set_xlabel('eta')
        axes[1].set_ylabel('Final Regret')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_dir / f"{i}_ucb_eta_check_{cfg.env.name}_{cfg.env.n_arms}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save results
    with open(output_dir / f"ucb_eta_check_{cfg.env.name}_{cfg.env.n_arms}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to experiments/results/ucb_eta_check_{cfg.env.name}_{cfg.env.n_arms}.json")
    print(f"Plot saved to experiments/results/ucb_eta_check_{cfg.env.name}_{cfg.env.n_arms}.png")


if __name__ == "__main__":
    main() 