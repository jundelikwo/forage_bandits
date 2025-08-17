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

class TSFactorFunction:
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

        # if value < 0.0:
        #     value = 0.0
        # elif value > 1.0:
        #     value = 1.0

        return value

def get_ts_factor(alpha: float, beta: float, energy_factor_alg: str) -> Callable[[float, bool], float]:
    return TSFactorFunction(alpha, beta, energy_factor_alg)

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
    alpha_range = np.linspace(-1, 1, 21)
    beta_range = np.linspace(-1, 1, 21)
    zipped_range = np.array(np.meshgrid(alpha_range, beta_range)).T.reshape(-1, 2)

    # Initialize results dictionaries
    results = {
        # "ts_no_energy": {"regret": {}, "regret_std": {}, "lifetime": {}, "lifetime_std": {}},
        "ts_energy": {"regret": {}, "regret_std": {}, "lifetime": {}, "lifetime_std": {}},
        "ts_energy_flip_exp": {"regret": {}, "regret_std": {}, "lifetime": {}, "lifetime_std": {}},
    }
    
    # Run simulations for each number of arms
    for alpha, beta in zipped_range:
        print(f"\nRunning simulations for alpha = {alpha}, beta = {beta}")
        custom_exploration_factor = get_ts_factor(alpha, beta, "linear")
        
        # TS
        # print("  TS (no energy)...")
        # regret, regret_std, lifetime, lifetime_std = run_simulation(cfg, "ts", False, eta=0, custom_exploration_function=custom_exploration_factor)
        # results["ts_no_energy"]["lifetime"][alpha, beta] = lifetime
        # results["ts_no_energy"]["lifetime_std"][alpha, beta] = lifetime_std
        # results["ts_no_energy"]["regret"][alpha, beta] = regret
        # results["ts_no_energy"]["regret_std"][alpha, beta] = regret_std
        
        print("  TS (energy)...")
        regret, regret_std, lifetime, lifetime_std = run_simulation(cfg, "ts", True, eta=1, custom_exploration_function=custom_exploration_factor)
        results["ts_energy"]["lifetime"][alpha, beta] = lifetime
        results["ts_energy"]["lifetime_std"][alpha, beta] = lifetime_std
        results["ts_energy"]["regret"][alpha, beta] = regret
        results["ts_energy"]["regret_std"][alpha, beta] = regret_std

        print("  TS (energy, flip exp)...")
        regret, regret_std, lifetime, lifetime_std = run_simulation(cfg, "ts", True, eta=1, custom_exploration_function=get_ts_factor(alpha, beta, "flip_exp"))
        results["ts_energy_flip_exp"]["lifetime"][alpha, beta] = lifetime
        results["ts_energy_flip_exp"]["lifetime_std"][alpha, beta] = lifetime_std
        results["ts_energy_flip_exp"]["regret"][alpha, beta] = regret
        results["ts_energy_flip_exp"]["regret_std"][alpha, beta] = regret_std
    
    # Plot results
    # configs = ["ts_no_energy", "ts_energy", "ts_energy_flip_exp"]
    # titles = ["Non-Energy", "Energy", "Energy, flip_exp"]
    configs = ["ts_energy", "ts_energy_flip_exp"]
    titles = ["Linear Energy model", "Flip Exponential Energy model"]

    # Create heatmaps for each configuration
    # fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f'Lifetime Heatmaps for Different TS Configurations: {cfg.env.name} environment, n_arms={cfg.env.n_arms}', fontsize=16)
    
    
    for i, (config, title) in enumerate(zip(configs, titles)):
        ax = axes[i]
        
        # Extract data for heatmap
        lifetime_data = np.zeros((len(alpha_range), len(beta_range)))
        for idx, (alpha, beta) in enumerate(zipped_range):
            alpha_idx = np.where(alpha_range == alpha)[0][0]
            beta_idx = np.where(beta_range == beta)[0][0]
            lifetime_data[alpha_idx, beta_idx] = results[config]["lifetime"][(alpha, beta)]
        
        # Find and print optimal alpha, beta pair
        max_lifetime = np.max(lifetime_data)
        max_indices = np.unravel_index(np.argmax(lifetime_data), lifetime_data.shape)
        optimal_alpha = alpha_range[max_indices[0]]
        optimal_beta = beta_range[max_indices[1]]
        
        print(f"\n{title}:")
        print(f"  Optimal alpha: {optimal_alpha:.2f}")
        print(f"  Optimal beta: {optimal_beta:.2f}")
        print(f"  Maximum lifetime: {max_lifetime:.4f}")

        # Create heatmap
        im = ax.imshow(lifetime_data, cmap='plasma', aspect='auto', origin='lower')
        
        # Set ticks and labels
        beta_midpoint_index = int(beta_range.size/2)
        alpha_midpoint_index = int(alpha_range.size/2)
        ax.set_xticks([0, beta_midpoint_index, len(beta_range)-1])
        ax.set_yticks([0, alpha_midpoint_index, len(alpha_range)-1])
        ax.set_xticklabels([f"{beta_range[0]:.2f}", f"{beta_range[beta_midpoint_index]:.2f}", f"{beta_range[-1]:.2f}"])
        ax.set_yticklabels([f"{alpha_range[0]:.2f}", f"{alpha_range[alpha_midpoint_index]:.2f}", f"{alpha_range[-1]:.2f}"])
        
        ax.set_xlabel('Beta')
        ax.set_ylabel('Alpha')
        ax.set_title(f'{title} (max lifetime: {max_lifetime:.1f}, alpha: {optimal_alpha:.2f}, beta: {optimal_beta:.2f})')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Lifetime')
        
        # Mark optimal point on heatmap
        ax.plot(max_indices[1], max_indices[0], 'r*', markersize=15, markeredgecolor='white', markeredgewidth=2)

        if i == 0:
            beta_zero_idx = np.where(beta_range == 0.0)[0][0]  # Find index where beta=0
            alpha_values_at_beta_zero = lifetime_data[:, beta_zero_idx]  # Get all alpha values at beta=0
            optimal_alpha_at_beta_zero_idx = np.argmax(alpha_values_at_beta_zero)
            optimal_alpha_at_beta_zero = alpha_range[optimal_alpha_at_beta_zero_idx]
            max_lifetime_at_beta_zero = alpha_values_at_beta_zero[optimal_alpha_at_beta_zero_idx]

            ax.plot(beta_zero_idx, optimal_alpha_at_beta_zero_idx, 'go', markersize=15, markeredgecolor='white', markeredgewidth=2)
            
            print(f"  When beta=0.0:")
            print(f"    Optimal alpha: {optimal_alpha_at_beta_zero:.2f}")
            print(f"    Maximum lifetime: {max_lifetime_at_beta_zero:.4f}")

            # Plot blue circle for alpha=0, beta=1
            alpha_zero_idx = np.where(alpha_range == 0.0)[0][0]  # Find index where alpha=0
            beta_one_idx = np.where(beta_range == 1.0)[0][0]  # Find index where beta=1
            ax.plot(beta_one_idx, alpha_zero_idx, 'bo', markersize=15, markeredgecolor='white', markeredgewidth=2)

            print(f"  At alpha=0.0, beta=1.0:")
            print(f"    Lifetime: {lifetime_data[alpha_zero_idx, beta_one_idx]:.4f}")
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_dir / f"ts_factor_heatmaps_{cfg.env.name}_{cfg.env.n_arms}.png", dpi=300, bbox_inches='tight')
    plt.close()


    # Create heatmaps for each configuration
    # fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f'Regret Heatmaps for Different TS Configurations: {cfg.env.name} environment, n_arms={cfg.env.n_arms}', fontsize=16)
    
    
    for i, (config, title) in enumerate(zip(configs, titles)):
        ax = axes[i]
        
        # Extract data for heatmap
        regret_data = np.zeros((len(alpha_range), len(beta_range)))
        for idx, (alpha, beta) in enumerate(zipped_range):
            alpha_idx = np.where(alpha_range == alpha)[0][0]
            beta_idx = np.where(beta_range == beta)[0][0]
            regret_data[alpha_idx, beta_idx] = results[config]["regret"][(alpha, beta)]
        
        # Find and print optimal alpha, beta pair
        min_regret = np.min(regret_data)
        min_indices = np.unravel_index(np.argmin(regret_data), regret_data.shape)
        optimal_alpha = alpha_range[min_indices[0]]
        optimal_beta = beta_range[min_indices[1]]
        
        print(f"\n{title}:")
        print(f"  Optimal alpha: {optimal_alpha:.2f}")
        print(f"  Optimal beta: {optimal_beta:.2f}")
        print(f"  Minimum regret: {min_regret:.4f}")

        # Create heatmap
        im = ax.imshow(regret_data, cmap='plasma', aspect='auto', origin='lower')
        
        # Set ticks and labels
        beta_midpoint_index = int(beta_range.size/2)
        alpha_midpoint_index = int(alpha_range.size/2)
        ax.set_xticks([0, beta_midpoint_index, len(beta_range)-1])
        ax.set_yticks([0, alpha_midpoint_index, len(alpha_range)-1])
        ax.set_xticklabels([f"{beta_range[0]:.2f}", f"{beta_range[beta_midpoint_index]:.2f}", f"{beta_range[-1]:.2f}"])
        ax.set_yticklabels([f"{alpha_range[0]:.2f}", f"{alpha_range[alpha_midpoint_index]:.2f}", f"{alpha_range[-1]:.2f}"])
        
        ax.set_xlabel('Beta')
        ax.set_ylabel('Alpha')
        ax.set_title(f'{title} (min regret: {min_regret:.1f}, alpha: {optimal_alpha:.2f}, beta: {optimal_beta:.2f})')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Regret')
        
        # Mark optimal point on heatmap
        ax.plot(min_indices[1], min_indices[0], 'r*', markersize=15, markeredgecolor='white', markeredgewidth=2)

        if i == 0:
            beta_zero_idx = np.where(beta_range == 0.0)[0][0]  # Find index where beta=0
            alpha_values_at_beta_zero = regret_data[:, beta_zero_idx]  # Get all alpha values at beta=0
            optimal_alpha_at_beta_zero_idx = np.argmin(alpha_values_at_beta_zero)
            optimal_alpha_at_beta_zero = alpha_range[optimal_alpha_at_beta_zero_idx]
            min_regret_at_beta_zero = alpha_values_at_beta_zero[optimal_alpha_at_beta_zero_idx]

            ax.plot(beta_zero_idx, optimal_alpha_at_beta_zero_idx, 'go', markersize=15, markeredgecolor='white', markeredgewidth=2)
            
            print(f"  When beta=0.0:")
            print(f"    Optimal alpha: {optimal_alpha_at_beta_zero:.2f}")
            print(f"    Minimum regret: {min_regret_at_beta_zero:.4f}")

            # Plot blue circle for alpha=0, beta=1
            alpha_zero_idx = np.where(alpha_range == 0.0)[0][0]  # Find index where alpha=0
            beta_one_idx = np.where(beta_range == 1.0)[0][0]  # Find index where beta=1
            ax.plot(beta_one_idx, alpha_zero_idx, 'bo', markersize=15, markeredgecolor='white', markeredgewidth=2)
            
            print(f"  At alpha=0.0, beta=1.0:")
            print(f"    Regret: {regret_data[alpha_zero_idx, beta_one_idx]:.4f}")
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_dir / f"ts_factor_heatmaps_regret_{cfg.env.name}_{cfg.env.n_arms}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    # with open(output_dir / f"ts_factor_{cfg.env.name}_{cfg.env.n_arms}.json", "w") as f:
    #     json.dump(results, f, indent=2)
    
    # print(f"\nResults saved to experiments/results/ucb_factor_{cfg.env.name}_{cfg.env.n_arms}.json")
    print(f"Plot saved to experiments/results/ts_factor_heatmaps_{cfg.env.name}_{cfg.env.n_arms}.png")


if __name__ == "__main__":
    main() 