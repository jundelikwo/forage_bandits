"""forage_bandits.plotters
========================
Utility functions that turn NumPy arrays returned by :pymod:`forage_bandits.metrics`
into Matplotlib figures.  **No disk I/O is performed by default** so the functions
are test‑friendly; calling code decides when / where to save.

Each public function follows the same contract:

```
ax = func(curve[, ax=None, label=None, kwargs…])
```

* ``curve`` may be 1‑D (single run) or 2‑D shaped *(N, T)* (batch of runs).  If
  2‑D, the mean curve plus ±1 s.e.m. shading is drawn.
* If ``ax`` is *None*, a new figure & axes are created.
* All kwargs are forwarded to the underlying Matplotlib call for quick style
  tweaks (e.g. ``color="black"``).

There is **one** optional I/O helper, :pyfunc:`save_figures`, which bulk‑writes a
``dict[str, matplotlib.figure.Figure]`` to disk.

Example
-------
```python
from pathlib import Path
from forage_bandits.plotters import (
    plot_cumulative_regret, plot_energy_trajectory, plot_hazard_curve,
    plot_lifetime_distribution, plot_exploration_rate, save_figures,
)

figs = {}
figs["fig4_2"] = plot_cumulative_regret(R).figure
figs["fig4_3"] = plot_energy_trajectory(M).figure
figs["fig4_4"] = plot_hazard_curve(h).figure
figs["fig4_5"] = plot_lifetime_distribution(L).figure
figs["fig4_6"] = plot_exploration_rate(E).figure

save_figures(figs, Path("outputs/latest/figs"))
```
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, MutableMapping, Optional

import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    "plot_cumulative_regret",
    "plot_energy_trajectory",
    "plot_hazard_curve",
    "plot_lifetime_distribution",
    "plot_exploration_rate",
    "save_figures",
]


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _prepare_ax(ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5), dpi=100)
    return ax


def _plot_curve_with_sem(
    ax: plt.Axes,
    curve: np.ndarray,
    *,
    label: str | None = None,
    sem_alpha: float = 0.2,
    **line_kws,
) -> None:
    """Plot mean ± s.e.m. if *curve* is 2‑D; otherwise a single line."""
    curve = np.asarray(curve, dtype=float)
    if curve.ndim == 2:
        mean = curve.mean(axis=0)
        sem = curve.std(axis=0, ddof=1) / np.sqrt(curve.shape[0])
        ax.fill_between(
            np.arange(mean.size),
            mean - sem,
            mean + sem,
            alpha=sem_alpha,
            linewidth=0,
        )
        ax.plot(mean, label=label, **line_kws)
    elif curve.ndim == 1:
        ax.plot(curve, label=label, **line_kws)
    else:
        raise ValueError("curve must be 1‑D or 2‑D array")


# -----------------------------------------------------------------------------
# Public plotting functions
# -----------------------------------------------------------------------------

def plot_cumulative_regret(
    regret: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    label: str | None = None,
    energy_label: str | None = None,
    energy_regret: np.ndarray | None = None,
    other_label: str | None = None,
    other_regret: np.ndarray | None = None,
    **line_kws,
) -> plt.Axes:
    """R(t) curve."""
    ax = _prepare_ax(ax)
    _plot_curve_with_sem(ax, regret, label=label, **line_kws)
    if energy_regret is not None:
        _plot_curve_with_sem(ax, energy_regret, label=energy_label, **line_kws)
    if other_regret is not None:
        _plot_curve_with_sem(ax, other_regret, label=other_label, **line_kws)
    ax.set_xlabel("Timestep $t$")
    ax.set_ylabel("Cumulative regret $R(t)$")
    ax.set_title("Cumulative Regret")
    if label or ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False)
    return ax


def plot_energy_trajectory(
    energy: np.ndarray,
    energy_energy: np.ndarray | None = None,
    *,
    ax: Optional[plt.Axes] = None,
    label: str | None = None,
    energy_label: str | None = None,
    other_label: str | None = None,
    other_energy: np.ndarray | None = None,
    **line_kws,
) -> plt.Axes:
    """M(t) curve."""
    ax = _prepare_ax(ax)
    _plot_curve_with_sem(ax, energy, label=label, **line_kws)
    if energy_energy is not None:
        _plot_curve_with_sem(ax, energy_energy, label=energy_label, **line_kws)
    if other_energy is not None:
        _plot_curve_with_sem(ax, other_energy, label=other_label, **line_kws)
    ax.set_xlabel("Timestep $t$")
    ax.set_ylabel("Energy $M(t)$")
    ax.set_title("Energy Trajectory")
    ax.set_ylim(0.0, 1.05)
    if label or ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False)
    return ax


def plot_hazard_curve(
    hazard: np.ndarray,
    energy_hazard: np.ndarray | None = None,
    *,
    ax: Optional[plt.Axes] = None,
    label: str | None = None,
    energy_label: str | None = None,
    ylim: tuple[float, float] | None = (0.0, 1.05),
    other_label: str | None = None,
    other_hazard: np.ndarray | None = None,
    **line_kws,
) -> plt.Axes:
    """Hazard curve h(t)."""
    ax = _prepare_ax(ax)
    _plot_curve_with_sem(ax, hazard, label=label, **line_kws)
    if energy_hazard is not None:
        _plot_curve_with_sem(ax, energy_hazard, label=energy_label, **line_kws)
    if other_hazard is not None:
        _plot_curve_with_sem(ax, other_hazard, label=other_label, **line_kws)
    ax.set_xlabel("Timestep $t$")
    ax.set_ylabel("Hazard $h(t)$")
    ax.set_title("Hazard Curve")
    if ylim is not None:
        ax.set_ylim(ylim)
    if label or ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False)
    return ax


def plot_lifetime_distribution(
    lifetimes: np.ndarray,
    energy_lifetimes: np.ndarray | None = None,
    *,
    ax: Optional[plt.Axes] = None,
    label: str | None = None,
    energy_label: str | None = None,
    bins: int = 30,
    other_label: str | None = None,
    other_lifetimes: np.ndarray | None = None,
    **hist_kws,
) -> plt.Axes:
    """Plot distribution of predicted lifetimes."""
    ax = _prepare_ax(ax)
    
    # Get histogram data for both distributions
    hist1, bin_edges = np.histogram(lifetimes, bins=bins)
    if energy_lifetimes is not None:
        hist2, _ = np.histogram(energy_lifetimes, bins=bin_edges)
    else:
        hist2 = None
    
    # Calculate bin centers and widths
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    # Plot first distribution (top half)
    ax.bar(bin_centers, hist1, width=bin_width, label=label, alpha=0.7, **hist_kws)
    mean_lifetime = np.mean(lifetimes)
    ax.axvline(mean_lifetime, color='blue', linestyle='--', 
               label=f'Mean: {mean_lifetime:.1f}')
    
    # Plot second distribution (bottom half) if provided
    if energy_lifetimes is not None:
        # Invert the second histogram
        ax.bar(bin_centers, -hist2, width=bin_width, label=energy_label, alpha=0.7, **hist_kws)
        energy_mean_lifetime = np.mean(energy_lifetimes)
        ax.axvline(energy_mean_lifetime, color='orange', linestyle='--', 
                   label=f'Mean: {energy_mean_lifetime:.1f}')
    
    # Plot third distribution if provided
    if other_lifetimes is not None:
        ax.bar(bin_centers, hist2, width=bin_width, label=other_label, alpha=0.7, **hist_kws)
        other_mean_lifetime = np.mean(other_lifetimes)
        ax.axvline(other_mean_lifetime, color='green', linestyle='--', 
                   label=f'Mean: {other_mean_lifetime:.1f}')
    # Customize plot
    ax.set_xlabel("Lifetime")
    ax.set_ylabel("Frequency")
    ax.set_title("Lifetime Distribution")
    ax.set_xlim(0, 50)
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Adjust y-axis limits to be symmetric and set custom tick labels
    if energy_lifetimes is not None:
        max_freq = max(np.max(hist1), np.max(hist2))
        ax.set_ylim(-max_freq * 1.1, max_freq * 1.1)
        
        # Get current y-ticks
        yticks = ax.get_yticks()
        # Create new tick labels with positive values
        yticklabels = [f'{abs(y):.0f}' for y in yticks]
        ax.set_yticklabels(yticklabels)
    
    if label or ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_exploration_rate(
    exploring: np.ndarray,
    energy_exploring: np.ndarray | None = None,
    *,
    ax: Optional[plt.Axes] = None,
    label: str | None = None,
    energy_label: str | None = None,
    start_explore: bool = False,
    other_label: str | None = None,
    other_exploring: np.ndarray | None = None,
    **line_kws,
) -> plt.Axes:
    """Plot exploration rate over time."""
    ax = _prepare_ax(ax)
    
    # Calculate mean exploration rate across runs
    mean_explore = np.mean(exploring, axis=0)
    if start_explore:
        mean_explore[0] = 1.0
        
    # Plot mean exploration rate
    ax.plot(mean_explore, label=label, **line_kws)

    if energy_exploring is not None:
        energy_mean_explore = np.mean(energy_exploring, axis=0)
        ax.plot(energy_mean_explore, label=energy_label, **line_kws)
    
    if other_exploring is not None:
        other_mean_explore = np.mean(other_exploring, axis=0)
        ax.plot(other_mean_explore, label=other_label, **line_kws)
    
    # Customize plot
    ax.set_xlabel("Timestep $t$")
    ax.set_ylabel("Exploration Rate")
    ax.set_title("Exploration Rate Over Time")
    ax.set_ylim(0, 1.05)  # Exploration rate is between 0 and 1
    if label or ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    
    return ax


# -----------------------------------------------------------------------------
# Bulk saver
# -----------------------------------------------------------------------------

def save_figures(
    figures: Mapping[str, plt.Figure] | Mapping[str, plt.Axes],
    out_dir: Path,
    *,
    dpi: int = 300,
    fmt: str = "png",
) -> None:
    """Save each figure or axes in *figures* under *out_dir* / f"{name}.{fmt}"."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in figures.items():
        fig: plt.Figure
        if isinstance(obj, plt.Axes):
            fig = obj.figure  # type: ignore[assignment]
        elif isinstance(obj, plt.Figure):
            fig = obj
        else:
            raise TypeError("values must be matplotlib Figure or Axes")
        filepath = out_dir / f"{name}.{fmt}"
        fig.savefig(filepath, dpi=dpi, format=fmt, bbox_inches="tight")
