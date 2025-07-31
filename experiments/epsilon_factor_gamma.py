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
    epsilon
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

class EpsilonFactorFunction:
    """A callable class that can be pickled for multiprocessing."""
    
    def __init__(self, alpha: float, beta: float, gamma: float, energy_factor_alg: str):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.energy_factor_alg = energy_factor_alg
    
    def __call__(self, energy: float, energy_adaptive: bool) -> float:
        if energy_adaptive:
            energy_factor = energy_factor_flip_exp(energy, self.gamma) if self.energy_factor_alg == "flip_exp" else energy_factor_linear(energy)

            value = self.alpha + self.beta * energy_factor
        else:
            value = self.alpha + self.beta

        if value < 0.0:
            value = 0.0
        elif value > 1.0:
            value = 1.0

        return value

def get_epsilon_factor(alpha: float, beta: float, gamma: float, energy_factor_alg: str) -> Callable[[float, bool], float]:
    return EpsilonFactorFunction(alpha, beta, gamma, energy_factor_alg)

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
    gamma_range = np.linspace(-1, 10, 20)
    beta_range = np.linspace(-1, 1, 20)
    zipped_range = np.array(np.meshgrid(gamma_range, beta_range)).T.reshape(-1, 2)

    # Initialize results dictionaries
    results = {
        "e_greedy_energy_full": {"regret": {}, "regret_std": {}, "lifetime": {}, "lifetime_std": {}},
        "e_greedy_energy_half": {"regret": {}, "regret_std": {}, "lifetime": {}, "lifetime_std": {}},
        "e_greedy_energy_half_starving": {"regret": {}, "regret_std": {}, "lifetime": {}, "lifetime_std": {}},
    }
    
    # Run simulations for each number of arms
    for gamma, beta in zipped_range:
        print(f"\nRunning simulations for gamma = {gamma}, beta = {beta}")
        
        # ε-Greedy
        print("  ε-Greedy (no energy)...")
        cfg.alg.forage_cost = 3.912023005428146 * 0.1
        cfg.alg.init_energy = 3.912023005428146
        regret, regret_std, lifetime, lifetime_std = run_simulation(cfg, "egree", True, eta=1, custom_exploration_function=get_epsilon_factor(0.0, beta, gamma, "flip_exp"))
        results["e_greedy_energy_full"]["lifetime"][gamma, beta] = lifetime
        results["e_greedy_energy_full"]["lifetime_std"][gamma, beta] = lifetime_std
        results["e_greedy_energy_full"]["regret"][gamma, beta] = regret
        results["e_greedy_energy_full"]["regret_std"][gamma, beta] = regret_std
        
        print("  ε-Greedy (energy)...")
        cfg.alg.forage_cost = 3.912023005428146 * 0.1
        cfg.alg.init_energy = 3.912023005428146 / 2
        regret, regret_std, lifetime, lifetime_std = run_simulation(cfg, "egree", True, eta=1, custom_exploration_function=get_epsilon_factor(0.0, beta, gamma, "flip_exp"))
        results["e_greedy_energy_half"]["lifetime"][gamma, beta] = lifetime
        results["e_greedy_energy_half"]["lifetime_std"][gamma, beta] = lifetime_std
        results["e_greedy_energy_half"]["regret"][gamma, beta] = regret
        results["e_greedy_energy_half"]["regret_std"][gamma, beta] = regret_std

        print("  ε-Greedy (energy, flip exp)...")
        cfg.alg.forage_cost = 3.912023005428146 * 0.21
        cfg.alg.init_energy = 3.912023005428146 / 2
        regret, regret_std, lifetime, lifetime_std = run_simulation(cfg, "egree", True, eta=1, custom_exploration_function=get_epsilon_factor(0.0, beta, gamma, "flip_exp"))
        results["e_greedy_energy_half_starving"]["lifetime"][gamma, beta] = lifetime
        results["e_greedy_energy_half_starving"]["lifetime_std"][gamma, beta] = lifetime_std
        results["e_greedy_energy_half_starving"]["regret"][gamma, beta] = regret
        results["e_greedy_energy_half_starving"]["regret_std"][gamma, beta] = regret_std
    
    # Plot results
    configs = ["e_greedy_energy_full", "e_greedy_energy_half", "e_greedy_energy_half_starving"]
    titles = ["(init energy = 1.0, forage cost = 0.1)", "(init energy = 0.5, forage cost = 0.1)", "(init energy = 0.5, forage cost = 0.21)"]

    # Create heatmaps for each configuration
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Lifetime Heatmaps for Different ε-Greedy Configurations using flip_exp model: {cfg.env.name} environment, n_arms={cfg.env.n_arms}', fontsize=16)
    
    
    for i, (config, title) in enumerate(zip(configs, titles)):
        ax = axes[i]
        
        # Extract data for heatmap
        lifetime_data = np.zeros((len(gamma_range), len(beta_range)))
        for idx, (gamma, beta) in enumerate(zipped_range):
            gamma_idx = np.where(gamma_range == gamma)[0][0]
            beta_idx = np.where(beta_range == beta)[0][0]
            lifetime_data[gamma_idx, beta_idx] = results[config]["lifetime"][(gamma, beta)]
        
        # Find and print optimal gamma, beta pair
        max_lifetime = np.max(lifetime_data)
        max_indices = np.unravel_index(np.argmax(lifetime_data), lifetime_data.shape)
        optimal_gamma = gamma_range[max_indices[0]]
        optimal_beta = beta_range[max_indices[1]]
        
        print(f"\n{title}:")
        print(f"  Optimal gamma: {optimal_gamma:.2f}")
        print(f"  Optimal beta: {optimal_beta:.2f}")
        print(f"  Maximum lifetime: {max_lifetime:.4f}")

        # Create heatmap
        im = ax.imshow(lifetime_data, cmap='plasma', aspect='auto', origin='lower')
        
        # Set ticks and labels
        # ax.set_xticks(range(len(beta_range)))
        # ax.set_yticks(range(len(gamma_range)))
        # ax.set_xticklabels([f'{b:.2f}' for b in beta_range])
        # ax.set_yticklabels([f'{a:.2f}' for a in gamma_range])
        beta_midpoint_index = int(beta_range.size/2)
        ax.set_xticks([0, beta_midpoint_index, len(beta_range)-1])
        gamma_midpoint_index = int(gamma_range.size/2)
        ax.set_yticks([0, gamma_midpoint_index, len(gamma_range)-1])
        ax.set_xticklabels([f"{beta_range[0]:.2f}", f"{beta_range[beta_midpoint_index]:.2f}", f"{beta_range[-1]:.2f}"])
        ax.set_yticklabels([f"{gamma_range[0]:.2f}", f"{gamma_range[gamma_midpoint_index]:.2f}", f"{gamma_range[-1]:.2f}"])
        
        ax.set_xlabel('Beta')
        ax.set_ylabel('Alpha')
        ax.set_title(f'{title} max lifetime: {max_lifetime:.1f}, gamma: {optimal_gamma:.2f}, beta: {optimal_beta:.2f}', fontsize=9)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Lifetime')
        
        # Mark optimal point on heatmap
        ax.plot(max_indices[1], max_indices[0], 'r*', markersize=15, markeredgecolor='white', markeredgewidth=2)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_dir / f"epsilon_factor_gamma_heatmaps_{cfg.env.name}_{cfg.env.n_arms}.png", dpi=300, bbox_inches='tight')
    plt.close()


    # Create heatmaps for each configuration
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Regret Heatmaps for Different ε-Greedy Configurations using flip_exp model: {cfg.env.name} environment, n_arms={cfg.env.n_arms}', fontsize=16)
    
    
    for i, (config, title) in enumerate(zip(configs, titles)):
        ax = axes[i]
        
        # Extract data for heatmap
        regret_data = np.zeros((len(gamma_range), len(beta_range)))
        for idx, (gamma, beta) in enumerate(zipped_range):
            gamma_idx = np.where(gamma_range == gamma)[0][0]
            beta_idx = np.where(beta_range == beta)[0][0]
            regret_data[gamma_idx, beta_idx] = results[config]["regret"][(gamma, beta)]
        
        # Find and print optimal gamma, beta pair
        min_regret = np.min(regret_data)
        min_indices = np.unravel_index(np.argmin(regret_data), regret_data.shape)
        optimal_gamma = gamma_range[min_indices[0]]
        optimal_beta = beta_range[min_indices[1]]
        
        print(f"\n{title}:")
        print(f"  Optimal gamma: {optimal_gamma:.2f}")
        print(f"  Optimal beta: {optimal_beta:.2f}")
        print(f"  Minimum regret: {min_regret:.4f}")

        # Create heatmap
        im = ax.imshow(regret_data, cmap='plasma', aspect='auto', origin='lower')
        
        # Set ticks and labels
        # ax.set_xticks(range(len(beta_range)))
        # ax.set_yticks(range(len(gamma_range)))
        # ax.set_xticklabels([f'{b:.2f}' for b in beta_range])
        # ax.set_yticklabels([f'{a:.2f}' for a in gamma_range])
        beta_midpoint_index = int(beta_range.size/2)
        ax.set_xticks([0, beta_midpoint_index, len(beta_range)-1])
        gamma_midpoint_index = int(gamma_range.size/2)
        ax.set_yticks([0, gamma_midpoint_index, len(gamma_range)-1])
        ax.set_xticklabels([f"{beta_range[0]:.2f}", f"{beta_range[beta_midpoint_index]:.2f}", f"{beta_range[-1]:.2f}"])
        ax.set_yticklabels([f"{gamma_range[0]:.2f}", f"{gamma_range[gamma_midpoint_index]:.2f}", f"{gamma_range[-1]:.2f}"])
        
        ax.set_xlabel('Beta')
        ax.set_ylabel('Gamma')
        ax.set_title(f'{title} min regret: {min_regret:.1f}, gamma: {optimal_gamma:.2f}, beta: {optimal_beta:.2f}', fontsize=9)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Regret')
        
        # Mark optimal point on heatmap
        ax.plot(min_indices[1], min_indices[0], 'r*', markersize=15, markeredgecolor='white', markeredgewidth=2)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_dir / f"epsilon_factor_gamma_heatmaps_regret_{cfg.env.name}_{cfg.env.n_arms}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    # with open(output_dir / f"epsilon_factor_gamma_{cfg.env.name}_{cfg.env.n_arms}.json", "w") as f:
    #     json.dump(results, f, indent=2)
    
    # print(f"\nResults saved to experiments/results/epsilon_factor_gamma_{cfg.env.name}_{cfg.env.n_arms}.json")
    print(f"Plot saved to experiments/results/epsilon_factor_gamma_heatmaps_{cfg.env.name}_{cfg.env.n_arms}.png")


if __name__ == "__main__":
    main() 