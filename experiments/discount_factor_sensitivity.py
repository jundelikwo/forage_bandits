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
    gamma: float,
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
    gamma
        Discount factor γ ∈ (0, 1]
    n_arms
        Number of arms to use
    alg_name
        Name of the algorithm ('discountedegree', 'discounteducb', or 'discountedts')
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
    
    # Override gamma in algorithm config
    sim_cfg.alg.gamma = float(gamma)
    
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
    if hasattr(cfg.env, 'name') and 'dynamic' in cfg.env.name:
        # Do nothing
        print("")
    else:
        raise RuntimeError("only dynamic environment for this experiment")

    output_dir = Path("experiments/results")
    output_dir.mkdir(exist_ok=True)

    print(f"cfg: {cfg}")
    
    gamma_range = np.linspace(0.01, 1.0, 30)
    
    results = {
        "d_egree_no_energy": {"regret": {}, "lifetime": {}},
        "d_egree_energy": {"regret": {}, "lifetime": {}},
        "d_ucb_no_energy": {"regret": {}, "lifetime": {}},
        "d_ucb_energy": {"regret": {}, "lifetime": {}},
        "d_ts_no_energy": {"regret": {}, "lifetime": {}},
        "d_ts_energy": {"regret": {}, "lifetime": {}},
    }
    
    # Run simulations for each gamma value
    for gamma in gamma_range:
        print(f"\nRunning simulations for gamma = {gamma:.3f}")
        
        # Test all three discounted algorithms with and without energy adaptation
        for alg_name in ["discountedegree", "discounteducb", "discountedts"]:
            alg_key = f"d_{alg_name.split('discounted')[1]}"
            
            # No energy version
            print(f"  {alg_name} (no energy)...")
            regret, regret_std, lifetime, lifetime_std = run_simulation(
                cfg, gamma, cfg.env.n_arms, alg_name, False, eta=0
            )
            results[f"{alg_key}_no_energy"]["regret"][gamma] = regret
            results[f"{alg_key}_no_energy"]["lifetime"][gamma] = lifetime
            
            # Energy version
            print(f"  {alg_name} (energy)...")
            regret, regret_std, lifetime, lifetime_std = run_simulation(
                cfg, gamma, cfg.env.n_arms, alg_name, True, eta=1
            )
            results[f"{alg_key}_energy"]["regret"][gamma] = regret
            results[f"{alg_key}_energy"]["lifetime"][gamma] = lifetime
    
    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Gamma Parameter Sweep: {cfg.env.name} environment, n_arms={cfg.env.n_arms}', fontsize=16)
    
    algorithms = ["d_egree", "d_ucb", "d_ts"]
    alg_labels = ["Discounted ε-Greedy", "Discounted UCB", "Discounted TS"]
    
    for row, (alg, label) in enumerate(zip(algorithms, alg_labels)):
        # Plot regret
        ax = axes[0, row]
        
        # No energy version
        no_energy = results[f"{alg}_no_energy"]
        ax.plot(gamma_range, [no_energy['regret'][g] for g in gamma_range], 
                'b-o', label='No Energy', linewidth=2)
        
        # Energy version
        energy = results[f"{alg}_energy"]
        ax.plot(gamma_range, [energy['regret'][g] for g in gamma_range], 
                'r-s', label='Energy', linewidth=2)
        
        ax.set_xlabel('Gamma (γ)')
        ax.set_ylabel('Final Regret')
        ax.set_title(f'{label} - Regret')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot lifetime
        ax = axes[1, row]
        
        # No energy version
        ax.plot(gamma_range, [no_energy['lifetime'][g] for g in gamma_range], 
                'b-o', label='No Energy', linewidth=2)
        
        # Energy version
        ax.plot(gamma_range, [energy['lifetime'][g] for g in gamma_range], 
                'r-s', label='Energy', linewidth=2)
        
        ax.set_xlabel('Gamma (γ)')
        ax.set_ylabel('Mean Lifetime')
        ax.set_title(f'{label} - Lifetime')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / f"gamma_sweep_{cfg.env.name}_{cfg.env.n_arms}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    with open(output_dir / f"gamma_sweep_{cfg.env.name}_{cfg.env.n_arms}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to experiments/results/gamma_sweep_{cfg.env.name}_{cfg.env.n_arms}.json")
    print(f"Plot saved to experiments/results/gamma_sweep_{cfg.env.name}_{cfg.env.n_arms}.png")


if __name__ == "__main__":
    main() 