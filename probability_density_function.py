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
    
    # plt.title('Reward Distribution Probability Density Functions')
    plt.xlabel('Reward Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{filename}.png")
    plt.close()

output_dir = Path("figs/results")
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

# For Experiment 3 (Three arms with one risky arm)
def plot_risky_arm_pdf(filename, sigma=0.02, x_range=(-0.2, 0.5)):
    """Plot PDF for three arms including one risky arm with bimodal distribution"""
    plt.figure(figsize=(10, 6))
    x = np.linspace(*x_range, 1000)
    
    plt.axvline(0.1, label="Foraging cost", color='red', linestyle='--')
    
    # Arm 1: mean=0.04, sigma=0.02
    pdf1 = norm.pdf(x, 0.04, sigma)
    plt.plot(x, pdf1, label='Arm 1: μ=0.04, σ=0.02')
    
    # Arm 2: mean=0.2, sigma=0.02
    pdf2 = norm.pdf(x, 0.2, sigma)
    plt.plot(x, pdf2, label='Arm 2: μ=0.20, σ=0.02')
    
    # Arm 3: Risky arm - 50% chance of μ=0.4, σ=0.02 and 50% chance of μ=-0.1, σ=0
    # This is a mixture distribution: 0.5 * N(0.4, 0.02²) + 0.5 * δ(-0.1)
    pdf3_high = 0.5 * norm.pdf(x, 0.4, sigma)  # 50% chance of high reward
    pdf3_low = np.zeros_like(x)  # Initialize with zeros
    # Add delta function at -0.1 (approximated as very narrow normal distribution)
    pdf3_low += 0.5 * norm.pdf(x, -0.1, 0.02)  # 50% chance of fixed low reward
    pdf3_total = pdf3_high + pdf3_low
    
    plt.plot(x, pdf3_total, label='Arm 3: 50% N(0.4, 0.02²) + 50% δ(-0.1)', linewidth=2)
    
    # plt.title('Reward Distribution PDFs: Two Standard Arms + One Risky Arm')
    plt.xlabel('Reward Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{filename}.png", dpi=300, bbox_inches='tight')
    plt.close()

# Plot the risky arm setup
plot_risky_arm_pdf("three_arms_with_risky")