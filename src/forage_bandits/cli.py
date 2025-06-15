"""forage_bandits.cli
===================
Command‑line entry point that glues **configs → simulate → plotters**.  It is
powered by *Hydra*, so you can compose overrides the same way as in the paper's
experiments, e.g.:

```bash
poetry run forage-run env=single_optimal alg=ucb steps=200 n_runs=100
```

By default the working directory is changed to something like
``outputs/exp‑2025‑06‑14‑11‑05`` courtesy of Hydra's ``hydra.run.dir``
setting (see ``configs/hydra.yaml``).  All generated figures/metrics live
there, making runs self‑contained and reproducible.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf

from .plotters import (
    plot_cumulative_regret,
    plot_energy_trajectory,
    plot_hazard_curve,
    plot_lifetime_distribution,
    plot_exploration_rate,
    save_figures,
)
from .simulate import from_config
from .metrics import predicted_lifetime

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Hydra entry‑point
# -----------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="../../configs", config_name="base")
def main(cfg: DictConfig) -> None:  # pragma: no cover
    """Run a single config, generate figures, and persist results.

    The active working directory is the Hydra run dir (``outputs/exp‑…``).
    """

    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    # ------------------------------------------------------------------
    # 1. Simulate
    # ------------------------------------------------------------------
    result = from_config(cfg)

    # ------------------------------------------------------------------
    # 2. Generate figures based on plot settings
    # ------------------------------------------------------------------
    figs: Dict[str, plt.Figure] = {}

    # Plot lifetime if enabled
    if cfg.plot.lifetime:
        lifetimes = predicted_lifetime(result.hazard)
        figs["fig4_5_lifetime"] = plot_lifetime_distribution(lifetimes)

    # Plot regret if enabled
    if cfg.plot.regret:
        figs["fig4_2_regret"] = plot_cumulative_regret(result.cumulative_regret)
    
    # Plot energy if enabled and available
    if cfg.plot.energy and result.energy is not None:
        figs["fig4_3_energy"] = plot_energy_trajectory(result.energy)

    # Plot hazard if enabled and we have batch runs
    if cfg.plot.hazard and result.rewards.ndim == 2:
        figs["fig4_4_hazard"] = plot_hazard_curve(result.hazard)
        
    # Plot exploration rate if enabled and we have batch runs
    if cfg.plot.explore and result.rewards.ndim == 2:
        figs["fig4_6_explore"] = plot_exploration_rate(result.exploring)

    # ------------------------------------------------------------------
    # 3. Save to disk (figures & raw arrays)
    # ------------------------------------------------------------------
    if figs:  # Only save if we have any figures
        save_figures(figs, Path.cwd() / "figs")

    # Save raw arrays for debugging / repro
    npz_path = Path("metrics.npz")
    log.info("Saving metrics → %s", npz_path.resolve())
    import numpy as np

    # Prepare save data
    save_data = {
        "rewards": result.rewards,
        "actions": result.actions,
        "energies": result.energy,
        "regret": result.cumulative_regret,
    }
    
    # Only include hazard and exploring for batch runs
    if result.rewards.ndim == 2:
        save_data["hazard"] = result.hazard
        save_data["exploring"] = result.exploring

    np.savez_compressed(npz_path, **save_data)

    log.info("Done! Figures saved under %s", Path.cwd())


if __name__ == "__main__":  # pragma: no cover
    main()
