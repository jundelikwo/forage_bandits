"""
forage_bandits.agents.neural_hebbian
Implementation of a biologically plausible neural agent with dopamine-modulated 
Hebbian learning, inspired by insect Mushroom Body circuits and neuromodulatory 
influences on synaptic plasticity.

Key biological concepts:
- Synaptic weights represent action values (like Kenyon cells in Mushroom Body)
- Dopamine-like reinforcement signal modulates learning rate
- Energy-dependent neuromodulation affects synaptic plasticity
- Hebbian learning rule: "neurons that fire together, wire together"
- Stress/hunger effects on learning effectiveness

This implementation simulates:
1. A simple two-layer network (input → hidden → output)
2. Reward-modulated Hebbian plasticity
3. Energy-dependent learning rate modulation
4. Synaptic weight decay (forgetting)
5. Exploration based on synaptic weight uncertainty
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Callable
import math

from .base import AgentBase
from ..energy_factors import energy_factor_linear, energy_factor_exp, energy_factor_flip_exp, energy_factor_thr, energy_factor_parabolic, energy_factor_sigmoid, energy_factor_flip_linear


class NeuralHebbianAgent(AgentBase):
    """Biologically plausible neural agent with dopamine-modulated Hebbian learning.
    
    This agent simulates a simple neural network that learns action values through
    reward-modulated synaptic plasticity, with energy-dependent learning rates
    that mimic neuromodulatory influences on learning.
    
    Parameters
    ----------
    n_arms : int
        Number of arms in the bandit environment.
    hidden_size : int, default 16
        Number of hidden neurons (simulating Kenyon cells in Mushroom Body).
    learning_rate : float, default 1.0
        Base learning rate for synaptic weight updates.
    dopamine_sensitivity : float, default 1.0
        Sensitivity to reward signals (dopamine modulation strength).
    energy_adaptive : bool, default True
        Whether to modulate learning rate based on energy level.
    energy_factor_alg : str, default "sigmoid"
        Energy factor algorithm for learning rate modulation.
    weight_decay : float, default 0.001
        Synaptic weight decay rate (forgetting).
    exploration_noise : float, default 0.1
        Standard deviation of exploration noise added to action selection.
    init_energy : float, default 1.0
        Initial normalized energy level M₀ (0 → dead, 1 → full energy).
    Emax : float, default 1.0
        Maximum energy level.
    forage_cost : float, default 0.0
        Constant energetic cost M_f subtracted every trial.
    rng : Optional[np.random.Generator], default None
        Optional NumPy random generator for reproducibility.
    custom_exploration_function:
        Custom exploration function to use. Default is None. Accepts energy and energy_adaptive as arguments.
    """

    def __init__(
        self,
        n_arms: int,
        *,
        hidden_size: int = 16,
        learning_rate: float = 1.0,
        dopamine_sensitivity: float = 1.0,
        energy_adaptive: bool = True,
        energy_factor_alg: str = "sigmoid",
        weight_decay: float = 0.001,
        exploration_noise: float = 0.2,
        init_energy: float = 1.0,
        Emax: float = 1.0,
        forage_cost: float = 0.0,
        rng: Optional[np.random.Generator] = None,
        custom_exploration_function: Callable[[float, bool], float] = None,
    ) -> None:
        # hidden_size = 18
        self.n_arms = n_arms
        self.hidden_size = hidden_size
        self.learning_rate = float(learning_rate)
        self.dopamine_sensitivity = float(dopamine_sensitivity)
        self.energy_adaptive = energy_adaptive
        self.energy_factor_alg = energy_factor_alg
        self.weight_decay = float(weight_decay)
        self.exploration_noise = float(exploration_noise)
        self.Emax = float(Emax)
        self.forage_cost = float(forage_cost)
        self.custom_exploration_function = custom_exploration_function
        
        # Energy bookkeeping
        self.energy = float(init_energy)
        
        # Random generator
        self._rng = np.random.default_rng(rng)
        
        # Neural network weights (synaptic connections)
        # Input -> Hidden weights (one-hot encoding of actions)
        self.W1 = self._rng.normal(0, 0.1, (n_arms, hidden_size))
        # Hidden -> Output weights (action values)
        self.W2 = self._rng.normal(0, 0.1, (hidden_size, n_arms))
        
        # Biases
        self.b1 = np.zeros(hidden_size)
        self.b2 = np.zeros(n_arms)
        
        # Activity tracking for Hebbian learning
        self.last_hidden_activity = None
        self.last_action = None
        
        # Statistics for compatibility
        self.counts = np.zeros(n_arms, dtype=np.int64)
        self.values = np.zeros(n_arms, dtype=np.float64)  # empirical means
        
        # Exploration tracking
        self._last_was_explore = False
        
        # Learning history for analysis
        self.learning_history = []

    def _get_energy_factor(self, energy: float) -> float:
        """Get energy-dependent modulation factor for learning rate."""
        if not self.energy_adaptive:
            return 1.0
            
        normalized_energy = energy / self.Emax
        
        if self.energy_factor_alg == "linear":
            return energy_factor_linear(normalized_energy)
        elif self.energy_factor_alg == "flip_linear":
            return energy_factor_flip_linear(normalized_energy)
        elif self.energy_factor_alg == "exp":
            return energy_factor_exp(normalized_energy)
        elif self.energy_factor_alg == "flip_exp":
            return energy_factor_flip_exp(normalized_energy)
        elif self.energy_factor_alg == "thr":
            return energy_factor_thr(normalized_energy)
        elif self.energy_factor_alg == "parabolic":
            return energy_factor_parabolic(normalized_energy)
        elif self.energy_factor_alg == "sigmoid":
            return energy_factor_sigmoid(normalized_energy)
        else:
            return energy_factor_sigmoid(normalized_energy)  # default

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)

    def _forward_pass(self, action: int) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through the neural network.
        
        Returns:
            hidden_activity: Activity of hidden layer neurons
            output_activity: Activity of output layer neurons (action values)
        """
        # One-hot encode the action
        input_activity = np.zeros(self.n_arms)
        input_activity[action] = 1.0
        
        # Hidden layer
        hidden_input = np.dot(input_activity, self.W1) + self.b1
        hidden_activity = self._sigmoid(hidden_input)
        
        # Output layer
        output_input = np.dot(hidden_activity, self.W2) + self.b2
        output_activity = self._relu(output_input)  # Action values should be positive
        
        return hidden_activity, output_activity

    def _compute_dopamine_signal(self, reward: float, expected_reward: float) -> float:
        """Compute dopamine-like reinforcement signal.
        
        This simulates the reward prediction error (RPE) that modulates
        synaptic plasticity in the brain.
        """
        # Reward prediction error
        rpe = reward - expected_reward
        
        # Dopamine signal is proportional to RPE, but clamped to prevent
        # learning rate from becoming negative
        dopamine = self.dopamine_sensitivity * rpe
        
        return dopamine

    def _hebbian_update(self, action: int, reward: float, hidden_activity: np.ndarray):
        """Update synaptic weights using reward-modulated Hebbian learning.
        
        This implements a biologically plausible learning rule where:
        1. Synaptic strength changes based on pre- and post-synaptic activity
        2. Learning rate is modulated by dopamine signal
        3. Energy level affects neuromodulatory tone
        """
        # Get current action value estimate
        _, current_output = self._forward_pass(action)
        expected_reward = current_output[action]
        
        # Compute dopamine signal
        dopamine = self._compute_dopamine_signal(reward, expected_reward)
        
        # Energy-dependent learning rate modulation
        energy_factor = self._get_energy_factor(self.energy)
        
        # Effective learning rate with dopamine modulation
        # Use a safer formulation that prevents negative learning rates
        dopamine_modulation = np.clip(1.0 + dopamine, 0.1, 2.0)  # Clamp to [0.1, 2.0]
        # effective_lr = self.learning_rate * energy_factor * dopamine_modulation
        effective_lr = energy_factor * np.abs(dopamine)
        
        # Additional safety clamp
        effective_lr = np.clip(effective_lr, 0.00001, 0.1)
        
        # One-hot encode the action
        input_activity = np.zeros(self.n_arms)
        input_activity[action] = 1.0
        
        # Hebbian update for input->hidden weights
        # ΔW1 = η * input_activity * hidden_activity * dopamine_modulation
        for i in range(self.n_arms):
            for j in range(self.hidden_size):
                if input_activity[i] > 0:  # Only update for active input
                    # Hebbian rule: strengthen connections between co-active neurons
                    weight_change = effective_lr * input_activity[i] * hidden_activity[j]
                    
                    # Modulate by dopamine signal (reward-dependent plasticity)
                    if dopamine > 0:
                        # Positive RPE: strengthen connections
                        self.W1[i, j] += weight_change
                    else:
                        # Negative RPE: weaken connections
                        self.W1[i, j] -= weight_change * 0.5  # Asymmetric learning
        
        # Hebbian update for hidden->output weights
        # Target output: reward for chosen action, current estimate for others
        target_output = current_output.copy()
        target_output[action] = reward
        
        # Update hidden->output weights
        for j in range(self.hidden_size):
            for k in range(self.n_arms):
                # Hebbian rule with target teaching signal
                weight_change = effective_lr * hidden_activity[j] * (target_output[k] - current_output[k])
                self.W2[j, k] += weight_change
        
        # Weight decay (forgetting)
        # self.W1 *= (1.0 - self.weight_decay)
        # self.W2 *= (1.0 - self.weight_decay)
        
        # Record learning event for analysis
        self.learning_history.append({
            'action': action,
            'reward': reward,
            'expected_reward': expected_reward,
            'dopamine': dopamine,
            'energy_factor': energy_factor,
            'effective_lr': effective_lr,
            'energy': self.energy
        })

    def _get_action_values(self) -> np.ndarray:
        """Get current action value estimates from the network."""
        action_values = np.zeros(self.n_arms)
        
        for action in range(self.n_arms):
            _, output_activity = self._forward_pass(action)
            action_values[action] = output_activity[action]
        
        return action_values

    # ------------------------------------------------------------------
    # AgentBase interface
    # ------------------------------------------------------------------
    def act(self, t: int) -> int:  # noqa: D401, ARG002
        """Choose an arm index for time-step t using neural network outputs."""
        # Get action values from network
        action_values = self._get_action_values()
        
        # Energy-dependent neural noise (more noise when energy is low)
        energy_factor = self._get_energy_factor(self.energy)

        if self.custom_exploration_function is not None:
            energy_factor = self.custom_exploration_function(self.energy / self.Emax, self.energy_adaptive)
            
        adaptive_noise = self.exploration_noise * energy_factor
        
        # Add exploration noise (simulating neural noise)
        noise = self._rng.normal(0, adaptive_noise, self.n_arms)
        noisy_values = action_values + noise
        
        # Find the action that would be chosen without noise (true best)
        true_best_action = np.argmax(action_values)
        
        # Choose action with highest noisy value (Thompson sampling-like)
        chosen_action = np.argmax(noisy_values)
        
        # Track exploration: if noise changed the decision, it was exploratory
        self._last_was_explore = (chosen_action != true_best_action)
        
        return int(chosen_action)

    def update(self, action: int, reward: float) -> None:
        """Update neural network weights using reward-modulated Hebbian learning."""
        # Forward pass to get neural activities
        hidden_activity, _ = self._forward_pass(action)
        
        # Update synaptic weights using Hebbian learning
        self._hebbian_update(action, reward, hidden_activity)
        
        # Update empirical statistics for compatibility
        self.counts[action] += 1
        n = self.counts[action]
        self.values[action] += (reward - self.values[action]) / n
        
        # Update energy
        energy = self.energy + max(0, reward) - self.forage_cost
        energy = max(0, energy)
        energy = min(self.Emax, energy)
        self.energy = energy

    @property
    def is_exploring(self) -> bool:
        """Return True if the last action was exploratory."""
        return self._last_was_explore

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def estimated_means(self) -> np.ndarray:
        """Return current empirical mean reward estimates."""
        return self.values.copy()

    def neural_action_values(self) -> np.ndarray:
        """Return current action values from the neural network."""
        return self._get_action_values()

    def get_synaptic_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return current synaptic weights for analysis."""
        return self.W1.copy(), self.W2.copy()

    def get_learning_history(self) -> list:
        """Return learning history for analysis."""
        return self.learning_history.copy()

    def reset_learning_history(self) -> None:
        """Clear learning history."""
        self.learning_history.clear()

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # noqa: D401
        energy_status = f"energy={self.energy:.3f}" if self.energy_adaptive else "no_energy_mod"
        return f"<NeuralHebbian hidden={self.hidden_size} lr={self.learning_rate:.4f} {energy_status}>"


__all__ = ["NeuralHebbianAgent"] 