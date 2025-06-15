# Forage Bandits

A Python package for simulating foraging behavior using energy-aware multi-armed bandit algorithms. This implementation focuses on the trade-off between exploration and energy conservation in changing environments.

## Features

- **Multiple Bandit Algorithms**:
  - ε-Greedy (baseline)
  - UCB1
  - Thompson Sampling
  - Energy-Adaptive variants of each algorithm

- **Environment Types**:
  - Single-Optimal Gaussian (Section 4.3)
  - Sigmoid Gaussian landscape (Eq. 4.17)

- **Metrics**:
  - Cumulative regret
  - Energy trajectories
  - Hazard curves
  - Lifetime analysis

## Installation

```bash
# Using Poetry (recommended)
poetry install

# Using pip
pip install -r requirements.txt
pip install -e .
```

## Quick Start

Run a single experiment:

```bash
# Using the CLI
forage-run env=single_optimal alg=egree

# Or with Python
python -m forage_bandits.cli env=single_optimal alg=egree
```

Run with energy-adaptive variant:

```bash
forage-run env=single_optimal alg=egree alg.energy_adaptive=true
```

Run experiments:

```bash
python experiments/predicted_lifetime_regret.py
python experiments/pairwise_comparison.py env.n_arms=4
python experiments/pairwise_comparison.py env.n_arms=12

python experiments/predicted_lifetime_regret.py env=sigmoid
python experiments/pairwise_comparison.py env.n_arms=10 env=sigmoid
```

## Configuration

The package uses Hydra for configuration management. Key configuration files are in the `configs/` directory:

- `base.yaml`: Global defaults
- `env/`: Environment definitions
- `alg/`: Algorithm definitions

### Environment Parameters

```yaml
# Single-Optimal Gaussian
K: 10             # number of arms
mu_opt: 0.9       # mean of the single best arm
mu_others: 0.4    # mean of every other arm
sigma: 0.05       # shared Gaussian noise std-dev

# Sigmoid Gaussian
K: 11             # arms indexed 0 … K-1
k: 1.5            # slope of the logistic
i0: 5             # index where μ ≈ 0.5
sigma: 0.05
```

### Algorithm Parameters

```yaml
# ε-Greedy
epsilon: 0.1
energy_adaptive: false

# UCB1
c: 1.0
energy_adaptive: false

# Thompson Sampling
eta: 1
energy_adaptive: false
```

## Development

```bash
# Run tests
pytest

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Linting
pylint src/
```

## Project Structure

```
forage-bandits/
├── configs/                    # Hydra configurations
│   ├── base.yaml              # Global defaults
│   ├── env/                   # Environment definitions
│   └── alg/                   # Algorithm definitions
├── experiments/               # Different experiments
├── src/forage_bandits/        # Source code
│   ├── agents/               # Bandit algorithms
│   ├── environments.py       # Reward generators
│   ├── metrics.py           # Performance metrics
│   ├── simulate.py          # Experiment runner
│   └── plotters.py          # Visualization
└── tests/                    # Test suite
```
