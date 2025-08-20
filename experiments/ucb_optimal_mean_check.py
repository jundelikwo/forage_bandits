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

def optimal_value(cfg: DictConfig):
    # Set up x-axis values (number of arms)
    alpha_range = np.linspace(-1, 1, 21)
    beta_range = np.linspace(-1, 1, 21)
    zipped_range = np.array(np.meshgrid(alpha_range, beta_range)).T.reshape(-1, 2)

    # Initialize results dictionaries
    results = {
        # "ucb_no_energy": {"regret": {}, "regret_std": {}, "lifetime": {}, "lifetime_std": {}},
        "ucb_energy": {"regret": {}, "regret_std": {}, "lifetime": {}, "lifetime_std": {}},
        "ucb_energy_flip_exp": {"regret": {}, "regret_std": {}, "lifetime": {}, "lifetime_std": {}},
    }
    
    # Run simulations for each number of arms
    for alpha, beta in zipped_range:
        # Filter out values where both alpha < -0.1 and beta < -0.1
        if alpha < -0.1 and beta < -0.1:
            print(f"Skipping alpha = {alpha}, beta = {beta} (both < -0.1)")
            continue
            
        print(f"\nRunning simulations for alpha = {alpha}, beta = {beta}")
        custom_exploration_factor = get_ucb_factor(alpha, beta, "linear")
        
        # UCB
        print("  UCB (energy)...")
        regret, regret_std, lifetime, lifetime_std = run_simulation(cfg, "ucb", True, eta=1, custom_exploration_function=custom_exploration_factor)
        results["ucb_energy"]["lifetime"][alpha, beta] = lifetime
        results["ucb_energy"]["lifetime_std"][alpha, beta] = lifetime_std
        results["ucb_energy"]["regret"][alpha, beta] = regret
        results["ucb_energy"]["regret_std"][alpha, beta] = regret_std

        print("  UCB (energy, flip exp)...")
        regret, regret_std, lifetime, lifetime_std = run_simulation(cfg, "ucb", True, eta=1, custom_exploration_function=get_ucb_factor(alpha, beta, "flip_exp"))
        results["ucb_energy_flip_exp"]["lifetime"][alpha, beta] = lifetime
        results["ucb_energy_flip_exp"]["lifetime_std"][alpha, beta] = lifetime_std
        results["ucb_energy_flip_exp"]["regret"][alpha, beta] = regret
        results["ucb_energy_flip_exp"]["regret_std"][alpha, beta] = regret_std

    linear_lifetime_data = np.full((len(alpha_range), len(beta_range)), np.nan)
    flip_lifetime_data = np.full((len(alpha_range), len(beta_range)), np.nan)
    linear_regret_data = np.full((len(alpha_range), len(beta_range)), np.nan)
    flip_regret_data = np.full((len(alpha_range), len(beta_range)), np.nan)
    for idx, (alpha, beta) in enumerate(zipped_range):
        # Skip filtered combinations
        if alpha < -0.1 and beta < -0.1:
            continue
            
        alpha_idx = np.where(alpha_range == alpha)[0][0]
        beta_idx = np.where(beta_range == beta)[0][0]
        linear_lifetime_data[alpha_idx, beta_idx] = results["ucb_energy"]["lifetime"][(alpha, beta)]
        flip_lifetime_data[alpha_idx, beta_idx] = results["ucb_energy_flip_exp"]["lifetime"][(alpha, beta)]
        linear_regret_data[alpha_idx, beta_idx] = results["ucb_energy"]["regret"][(alpha, beta)]
        flip_regret_data[alpha_idx, beta_idx] = results["ucb_energy_flip_exp"]["regret"][(alpha, beta)]
    
    # Find and print optimal alpha, beta pair
    linear_max_lifetime = np.nanmax(linear_lifetime_data)
    max_indices = np.unravel_index(np.nanargmax(linear_lifetime_data), linear_lifetime_data.shape)
    linear_lifetime_optimal_alpha = alpha_range[max_indices[0]]
    linear_lifetime_optimal_beta = beta_range[max_indices[1]]

    print(f"Linear Energy model:")
    print(f"  Optimal alpha: {linear_lifetime_optimal_alpha:.2f}")
    print(f"  Optimal beta: {linear_lifetime_optimal_beta:.2f}")
    print(f"  Maximum lifetime: {linear_max_lifetime:.4f}")

    beta_zero_idx = np.where(beta_range == 0.0)[0][0]  # Find index where beta=0
    alpha_values_at_beta_zero = linear_lifetime_data[:, beta_zero_idx]  # Get all alpha values at beta=0
    linear_lifetime_optimal_alpha_at_beta_zero_idx = np.nanargmax(alpha_values_at_beta_zero)
    linear_lifetime_optimal_alpha_at_beta_zero = alpha_range[linear_lifetime_optimal_alpha_at_beta_zero_idx]
    linear_lifetime_max_lifetime_at_beta_zero = alpha_values_at_beta_zero[linear_lifetime_optimal_alpha_at_beta_zero_idx]

    print(f"  When beta=0.0:")
    print(f"    Optimal alpha: {linear_lifetime_optimal_alpha_at_beta_zero:.2f}")
    print(f"    Maximum lifetime: {linear_lifetime_max_lifetime_at_beta_zero:.4f}")

    flip_max_lifetime = np.nanmax(flip_lifetime_data)
    max_indices = np.unravel_index(np.nanargmax(flip_lifetime_data), flip_lifetime_data.shape)
    flip_lifetime_optimal_alpha = alpha_range[max_indices[0]]
    flip_lifetime_optimal_beta = beta_range[max_indices[1]]

    print(f"Flip Energy model:")
    print(f"  Optimal alpha: {flip_lifetime_optimal_alpha:.2f}")
    print(f"  Optimal beta: {flip_lifetime_optimal_beta:.2f}")
    print(f"  Maximum lifetime: {flip_max_lifetime:.4f}")

    linear_min_regret = np.nanmin(linear_regret_data)
    min_indices = np.unravel_index(np.nanargmin(linear_regret_data), linear_regret_data.shape)
    linear_regret_optimal_alpha = alpha_range[min_indices[0]]
    linear_regret_optimal_beta = beta_range[min_indices[1]]
    
    print(f"Linear Energy model:")
    print(f"  Optimal alpha: {linear_regret_optimal_alpha:.2f}")
    print(f"  Optimal beta: {linear_regret_optimal_beta:.2f}")
    print(f"  Minimum regret: {linear_min_regret:.4f}")

    beta_zero_idx = np.where(beta_range == 0.0)[0][0]  # Find index where beta=0
    alpha_values_at_beta_zero = linear_regret_data[:, beta_zero_idx]  # Get all alpha values at beta=0
    linear_regret_optimal_alpha_at_beta_zero_idx = np.nanargmin(alpha_values_at_beta_zero)
    linear_regret_optimal_alpha_at_beta_zero = alpha_range[linear_regret_optimal_alpha_at_beta_zero_idx]
    linear_regret_min_regret_at_beta_zero = alpha_values_at_beta_zero[linear_regret_optimal_alpha_at_beta_zero_idx]

    print(f"  When beta=0.0:")
    print(f"    Optimal alpha: {linear_regret_optimal_alpha_at_beta_zero:.2f}")
    print(f"    Minimum regret: {linear_regret_min_regret_at_beta_zero:.4f}")

    flip_min_regret = np.nanmin(flip_regret_data)
    min_indices = np.unravel_index(np.nanargmin(flip_regret_data), flip_regret_data.shape)
    flip_regret_optimal_alpha = alpha_range[min_indices[0]]
    flip_regret_optimal_beta = beta_range[min_indices[1]]

    print(f"Flip Energy model:")
    print(f"  Optimal alpha: {flip_regret_optimal_alpha:.2f}")
    print(f"  Optimal beta: {flip_regret_optimal_beta:.2f}")
    print(f"  Minimum regret: {flip_min_regret:.4f}")

    return {
        "linear_max_lifetime": linear_max_lifetime,
        "linear_min_regret": linear_min_regret,
        "linear_lifetime_at_beta_zero": linear_lifetime_max_lifetime_at_beta_zero,
        "linear_regret_at_beta_zero": linear_regret_min_regret_at_beta_zero,
        "flip_max_lifetime": flip_max_lifetime,
        "flip_min_regret": flip_min_regret,
        "linear_lifetime_optimal_alpha": linear_lifetime_optimal_alpha,
        "linear_lifetime_optimal_beta": linear_lifetime_optimal_beta,
        "linear_regret_optimal_alpha": linear_regret_optimal_alpha,
        "linear_regret_optimal_beta": linear_regret_optimal_beta,
        "flip_lifetime_optimal_alpha": flip_lifetime_optimal_alpha,
        "flip_lifetime_optimal_beta": flip_lifetime_optimal_beta,
        "flip_regret_optimal_alpha": flip_regret_optimal_alpha,
        "flip_regret_optimal_beta": flip_regret_optimal_beta,
    }

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
    # optimal_mean_range = np.linspace(0.1, 0.8, 36)
    optimal_mean_range = np.linspace(0.1, 0.4, 7)

    # Initialize results dictionaries
    results = {
        "ucb_no_energy": {"regret": {}, "lifetime": {}},
        "ucb_energy_flip": {"regret": {}, "lifetime": {}},
        "ucb_energy_linear": {"regret": {}, "lifetime": {}},
        "raw_data": {},
    }
    
    # Run simulations for each number of arms
    for optimal_mean in optimal_mean_range:
        cfg.env.mu_opt = float(optimal_mean)
        print(f"\nRunning simulations for optimal_mean = {optimal_mean}")

        raw_data = optimal_value(cfg)
        results["raw_data"][optimal_mean] = raw_data

        linear_max_lifetime = raw_data["linear_max_lifetime"]
        linear_min_regret = raw_data["linear_min_regret"]
        linear_lifetime_at_beta_zero = raw_data["linear_lifetime_at_beta_zero"]
        linear_regret_at_beta_zero = raw_data["linear_regret_at_beta_zero"]
        flip_max_lifetime = raw_data["flip_max_lifetime"]
        flip_min_regret = raw_data["flip_min_regret"]
        
        results["ucb_no_energy"]["lifetime"][optimal_mean] = linear_lifetime_at_beta_zero
        results["ucb_no_energy"]["regret"][optimal_mean] = linear_regret_at_beta_zero

        results["ucb_energy_flip"]["lifetime"][optimal_mean] = flip_max_lifetime
        results["ucb_energy_flip"]["regret"][optimal_mean] = flip_min_regret
        
        results["ucb_energy_linear"]["lifetime"][optimal_mean] = linear_max_lifetime
        results["ucb_energy_linear"]["regret"][optimal_mean] = linear_min_regret
    
    # Plot results
    for i in range(2):
        # Create figure with 2 subplots
        fig, axes = plt.subplots(2, figsize=(15, 10))
        fig.suptitle(f'Effect of varying the optimal arm mean on Performance of UCB agent: {cfg.env.name} environment, n_arms={cfg.env.n_arms}', fontsize=16)

        # Plot lifetime (top row)
        ax = axes[0]
        
        # Plot no energy version
        no_energy = results["ucb_no_energy"]
        ax.errorbar(
            optimal_mean_range,
            [no_energy['lifetime'][n] for n in optimal_mean_range],
            label='No Energy',
            marker='o',
            capsize=5
        )
        
        # Plot linear energy version
        linear_energy = results["ucb_energy_linear"]
        ax.errorbar(
            optimal_mean_range,
            [linear_energy['lifetime'][n] for n in optimal_mean_range],
            label='Energy, linear',
            marker='s',
            capsize=5
        )
        
        if i == 0:
            # Plot flip energy version
            flip_energy = results["ucb_energy_flip"]
            ax.errorbar(
                optimal_mean_range,
                [flip_energy['lifetime'][n] for n in optimal_mean_range],
                label='Energy, flip',
                marker='x',
                capsize=5
            )
        
        # Plot regret (bottom row)
        ax = axes[1]
        
        # Plot no energy version
        no_energy = results["ucb_no_energy"]
        ax.errorbar(
            optimal_mean_range,
            [no_energy['regret'][n] for n in optimal_mean_range],
            label='No Energy',
            marker='o',
            capsize=5
        )
        
        # Plot energy version
        linear_energy = results["ucb_energy_linear"]
        ax.errorbar(
            optimal_mean_range,
            [linear_energy['regret'][n] for n in optimal_mean_range],
            label='Energy, linear',
            marker='s',
            capsize=5
        )

        if i == 0:
            flip_energy = results["ucb_energy_flip"]
            ax.errorbar(
                optimal_mean_range,
                [flip_energy['regret'][n] for n in optimal_mean_range],
                label='Energy, flip',
                marker='x',
                capsize=5
            )
        
        # Customize subplot
        axes[0].set_xlabel('Optimal Arm Mean')
        axes[0].set_ylabel('Mean Lifetime')
        axes[0].set_title(f'UCB')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        axes[1].set_xlabel('Optimal Arm Mean')
        axes[1].set_ylabel('Final Regret')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_dir / f"m{i}_ucb_optimal_mean_check_{cfg.env.name}_{cfg.env.n_arms}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save results
    with open(output_dir / f"mucb_optimal_mean_check_{cfg.env.name}_{cfg.env.n_arms}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to experiments/results/ucb_optimal_mean_check_{cfg.env.name}_{cfg.env.n_arms}.json")
    print(f"Plot saved to experiments/results/ucb_optimal_mean_check_{cfg.env.name}_{cfg.env.n_arms}.png")


if __name__ == "__main__":
    main() 