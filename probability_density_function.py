import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from pathlib import Path

def plot_reward_pdf(means, filename, sigma=0.02, x_range=(-0.05, 0.25)):
    """Plot PDF of reward distributions for multiple arms"""
    plt.figure(figsize=(10, 6))
    x = np.linspace(*x_range, 1000)
    
    plt.axvline(0.1, label="Foraging cost", color='red', linestyle='--')
    
    for i, mu in enumerate(means):
        pdf = norm.pdf(x, mu, sigma)
        plt.plot(x, pdf, label=f'Arm {i+1}: μ={mu:.3f}')
    
    plt.title('Reward Distribution Probability Density Functions')
    plt.xlabel('Reward Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{filename}.png")
    plt.close()

output_dir = Path("experiments/results")
output_dir.mkdir(exist_ok=True)

# Example usage:
# For Experiment 1 (Single High-Reward)
means = [0.04] * 1 + [0.2]  # 3 low-reward arms, 1 high-reward
plot_reward_pdf(means, "single_high_reward")

# For Experiment 2 (Sigmoid)
n_arms = 2
k = 10
means = [0.2 / (1 + np.exp(-k * (i/(n_arms-1) - 0.5))) for i in range(n_arms)]
plot_reward_pdf(means, "sigmoid_2_arms")

n_arms = 3
k = 10
means = [0.2 / (1 + np.exp(-k * (i/(n_arms-1) - 0.5))) for i in range(n_arms)]
plot_reward_pdf(means, "sigmoid_3_arms")

n_arms = 5
k = 10
means = [0.2 / (1 + np.exp(-k * (i/(n_arms-1) - 0.5))) for i in range(n_arms)]
plot_reward_pdf(means, "sigmoid_5_arms")