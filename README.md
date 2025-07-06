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

python experiments/predicted_lifetime_regret_novel_arm_initialization.py
python experiments/predicted_lifetime_regret_novel_arm_initialization.py env=sigmoid
python experiments/pairwise_comparison_novel_arm_initialization.py env.n_arms=10
python experiments/pairwise_comparison_novel_arm_initialization.py env.n_arms=10 env=sigmoid

python experiments/epsilon_check.py env.n_arms=4
python experiments/epsilon_check.py env.n_arms=12
python experiments/epsilon_check.py env.n_arms=4 env=sigmoid
python experiments/epsilon_check.py env.n_arms=12 env=sigmoid

python experiments/ucb_c_check.py env.n_arms=4 alg=ucb
python experiments/ucb_c_check.py env.n_arms=12 alg=ucb
python experiments/ucb_c_check.py env.n_arms=4 alg=ucb env=sigmoid
python experiments/ucb_c_check.py env.n_arms=12 alg=ucb env=sigmoid

python experiments/initial_energy_check.py env.n_arms=4
python experiments/initial_energy_check.py env.n_arms=12
python experiments/initial_energy_check.py env.n_arms=4 env=sigmoid
python experiments/initial_energy_check.py env.n_arms=12 env=sigmoid

python experiments/optimal_epsilon_per_arms.py
python experiments/optimal_epsilon_per_arms.py env=sigmoid

python experiments/pairwise_comparison.py env.n_arms=2 env=dynamic_single_optimal
python experiments/pairwise_comparison.py env.n_arms=2 env=dynamic_single_optimal discounted_agents=true
python experiments/pairwise_comparison.py env.n_arms=4 env=dynamic_single_optimal
python experiments/pairwise_comparison.py env.n_arms=4 env=dynamic_single_optimal discounted_agents=true
python experiments/pairwise_comparison.py env.n_arms=2 env=poisson_dynamic_single_optimal
python experiments/pairwise_comparison.py env.n_arms=2 env=poisson_dynamic_single_optimal discounted_agents=true
python experiments/pairwise_comparison.py env.n_arms=4 env=poisson_dynamic_single_optimal
python experiments/pairwise_comparison.py env.n_arms=4 env=poisson_dynamic_single_optimal discounted_agents=true

python experiments/pairwise_comparison.py env.n_arms=2 env=dynamic_sigmoid
python experiments/pairwise_comparison.py env.n_arms=2 env=dynamic_sigmoid discounted_agents=true
python experiments/pairwise_comparison.py env.n_arms=4 env=dynamic_sigmoid
python experiments/pairwise_comparison.py env.n_arms=4 env=dynamic_sigmoid discounted_agents=true

python experiments/discount_factor_sensitivity.py env.n_arms=2 env=dynamic_single_optimal
python experiments/discount_factor_sensitivity.py env.n_arms=2 env=poisson_dynamic_single_optimal
python experiments/discount_factor_sensitivity.py env.n_arms=3 env=dynamic_single_optimal
python experiments/discount_factor_sensitivity.py env.n_arms=3 env=poisson_dynamic_single_optimal
python experiments/discount_factor_sensitivity.py env.n_arms=4 env=dynamic_single_optimal
python experiments/discount_factor_sensitivity.py env.n_arms=4 env=poisson_dynamic_single_optimal
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
