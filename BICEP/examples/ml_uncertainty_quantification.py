#!/usr/bin/env python3
"""
Machine Learning Example: Uncertainty Quantification and Stochastic Optimization

This example demonstrates how to use BICEP for ML applications including
uncertainty quantification in neural networks, SGD trajectory visualization,
and Monte Carlo sampling for Bayesian inference.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from bicep_core import BICEPCore, BICEPConfig, NeuralBICEPLayer

def visualize_sgd_trajectories():
    """
    Visualize stochastic gradient descent paths on a 2D loss landscape.
    Shows how different random seeds lead to different optimization paths.
    """
    
    # Create a simple 2D loss landscape (Rosenbrock function)
    def rosenbrock(x, y, a=1, b=100):
        return (a - x)**2 + b * (y - x**2)**2
    
    # Generate grid for visualization
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)
    
    # BICEP configuration for SGD simulation
    config = BICEPConfig(
        device='cpu',
        use_half_precision=False
    )
    bicep = BICEPCore(config)
    
    # Simulate multiple SGD trajectories
    n_trajectories = 20
    n_steps = 200
    learning_rate = 0.01
    
    # Starting point
    start_x, start_y = -1.5, 2.5
    
    trajectories = []
    
    for i in range(n_trajectories):
        # Generate stochastic noise for SGD
        noise_x = bicep.generate_paths(
            n_paths=1,
            n_steps=n_steps,
            T=1.0,
            control_parameter=0.3,  # Some momentum-like behavior
            time_decay=0.02  # Decreasing noise (simulated annealing)
        )[0, :]
        
        noise_y = bicep.generate_paths(
            n_paths=1,
            n_steps=n_steps,
            T=1.0,
            control_parameter=0.3,
            time_decay=0.02
        )[0, :]
        
        # Simulate SGD trajectory
        trajectory = np.zeros((n_steps + 1, 2))
        trajectory[0] = [start_x, start_y]
        
        for step in range(n_steps):
            x_curr, y_curr = trajectory[step]
            
            # Compute gradients (with noise)
            grad_x = -2 * (1 - x_curr) - 400 * x_curr * (y_curr - x_curr**2)
            grad_y = 200 * (y_curr - x_curr**2)
            
            # Add stochastic noise
            grad_x += noise_x[step] * 5
            grad_y += noise_y[step] * 5
            
            # Update position
            trajectory[step + 1, 0] = x_curr - learning_rate * grad_x
            trajectory[step + 1, 1] = y_curr - learning_rate * grad_y
        
        trajectories.append(trajectory)
    
    # Visualization
    plt.figure(figsize=(12, 10))
    
    # Main plot: Loss landscape with trajectories
    plt.subplot(2, 2, 1)
    plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis', alpha=0.6)
    for traj in trajectories:
        plt.plot(traj[:, 0], traj[:, 1], 'r-', alpha=0.3, linewidth=1)
    plt.plot(1, 1, 'g*', markersize=15, label='Global Minimum')
    plt.plot(start_x, start_y, 'ro', markersize=10, label='Start')
    plt.xlabel('Parameter 1')
    plt.ylabel('Parameter 2')
    plt.title('SGD Trajectories on Loss Landscape')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss over iterations
    plt.subplot(2, 2, 2)
    for traj in trajectories:
        losses = [rosenbrock(traj[i, 0], traj[i, 1]) for i in range(len(traj))]
        plt.semilogy(losses, 'b-', alpha=0.3)
    plt.xlabel('Iteration')
    plt.ylabel('Loss (log scale)')
    plt.title('Loss Convergence')
    plt.grid(True, alpha=0.3)
    
    # Final positions distribution
    plt.subplot(2, 2, 3)
    final_positions = np.array([traj[-1] for traj in trajectories])
    plt.scatter(final_positions[:, 0], final_positions[:, 1], 
                c='red', alpha=0.6, s=50)
    plt.scatter(1, 1, c='green', s=100, marker='*', label='True Minimum')
    plt.xlabel('Final Parameter 1')
    plt.ylabel('Final Parameter 2')
    plt.title('Distribution of Final Positions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Distance to optimum over time
    plt.subplot(2, 2, 4)
    for traj in trajectories:
        distances = [np.sqrt((traj[i, 0] - 1)**2 + (traj[i, 1] - 1)**2) 
                    for i in range(len(traj))]
        plt.plot(distances, 'g-', alpha=0.3)
    plt.xlabel('Iteration')
    plt.ylabel('Distance to Optimum')
    plt.title('Convergence to Global Minimum')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sgd_trajectories.png', dpi=300)
    plt.show()
    
    # Statistics
    final_losses = [rosenbrock(traj[-1, 0], traj[-1, 1]) for traj in trajectories]
    print(f"\nSGD Trajectory Statistics:")
    print(f"Mean final loss: {np.mean(final_losses):.4f}")
    print(f"Std final loss: {np.std(final_losses):.4f}")
    print(f"Best final loss: {np.min(final_losses):.4f}")
    print(f"Success rate (loss < 0.1): {np.sum(np.array(final_losses) < 0.1) / len(final_losses):.1%}")

def uncertainty_aware_neural_network():
    """
    Build a neural network with BICEP-based uncertainty quantification.
    Demonstrates how stochastic layers can capture prediction uncertainty.
    """
    
    # Generate synthetic regression data with heteroscedastic noise
    np.random.seed(42)
    n_train = 200
    n_test = 100
    
    X_train = np.random.uniform(-3, 3, (n_train, 1)).astype(np.float32)
    noise_std = 0.3 * (1 + np.abs(X_train))  # Heteroscedastic noise
    y_train = np.sin(X_train) + np.random.normal(0, noise_std)
    
    X_test = np.linspace(-3, 3, n_test).reshape(-1, 1).astype(np.float32)
    y_true = np.sin(X_test)
    
    # Convert to torch tensors
    X_train_torch = torch.from_numpy(X_train)
    y_train_torch = torch.from_numpy(y_train.astype(np.float32))
    X_test_torch = torch.from_numpy(X_test)
    
    # Define uncertainty-aware model
    class UncertaintyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU()
            )
            
            # Stochastic layer for uncertainty
            config = BICEPConfig(device='cpu', use_memory_pool=False)
            self.stochastic = NeuralBICEPLayer(
                input_size=32,
                output_size=32,
                n_steps=20,
                config=config
            )
            
            self.decoder = nn.Linear(32, 1)
        
        def forward(self, x, n_samples=1):
            h = self.encoder(x)
            
            if n_samples == 1:
                h_stochastic = self.stochastic(h)
                return self.decoder(h + h_stochastic)
            else:
                # Multiple forward passes for uncertainty estimation
                predictions = []
                for _ in range(n_samples):
                    h_stochastic = self.stochastic(h)
                    pred = self.decoder(h + h_stochastic)
                    predictions.append(pred)
                return torch.stack(predictions)
    
    # Train the model
    model = UncertaintyNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    
    print("\nTraining uncertainty-aware neural network...")
    for epoch in range(100):
        optimizer.zero_grad()
        predictions = model(X_train_torch)
        loss = loss_fn(predictions, y_train_torch)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Generate predictions with uncertainty
    model.eval()
    with torch.no_grad():
        # Get multiple predictions
        n_mc_samples = 50
        predictions = model(X_test_torch, n_samples=n_mc_samples)
        
        # Calculate mean and uncertainty
        mean_pred = predictions.mean(dim=0).numpy()
        std_pred = predictions.std(dim=0).numpy()
    
    # Visualization
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.scatter(X_train, y_train, alpha=0.3, s=20, label='Training Data')
    plt.plot(X_test, y_true, 'k--', linewidth=2, label='True Function')
    plt.plot(X_test, mean_pred, 'r-', linewidth=2, label='Mean Prediction')
    plt.fill_between(X_test.flatten(), 
                     mean_pred.flatten() - 2*std_pred.flatten(),
                     mean_pred.flatten() + 2*std_pred.flatten(),
                     alpha=0.3, color='red', label='95% Confidence')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Uncertainty-Aware Neural Network Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(X_test, std_pred, 'b-', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('Prediction Uncertainty (Std)')
    plt.title('Uncertainty vs Input')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('uncertainty_nn.png', dpi=300)
    plt.show()
    
    print(f"\nUncertainty Quantification Results:")
    print(f"Mean prediction error: {np.mean(np.abs(mean_pred - y_true)):.4f}")
    print(f"Average uncertainty: {np.mean(std_pred):.4f}")
    print(f"Uncertainty correlation with true noise: {np.corrcoef(std_pred.flatten(), 0.3 * (1 + np.abs(X_test)).flatten())[0, 1]:.3f}")

def monte_carlo_bayesian_inference():
    """
    Use BICEP for efficient Monte Carlo sampling in Bayesian inference.
    Example: Inferring parameters of a simple probabilistic model.
    """
    
    # True parameters to infer
    true_mean = 2.5
    true_std = 1.0
    
    # Generate observed data
    np.random.seed(123)
    n_observations = 50
    observations = np.random.normal(true_mean, true_std, n_observations)
    
    # BICEP configuration for MCMC-like sampling
    config = BICEPConfig(device='cpu')
    bicep = BICEPCore(config)
    
    # Prior parameters (normal-inverse-gamma)
    prior_mean_mu = 0
    prior_mean_std = 10
    prior_std_alpha = 2
    prior_std_beta = 2
    
    # Generate MCMC chains using BICEP
    n_chains = 4
    n_samples = 5000
    
    # Generate proposal paths
    mean_paths = bicep.generate_paths(
        n_paths=n_chains,
        n_steps=n_samples,
        T=1.0,
        control_parameter=0.95,  # High autocorrelation for MCMC
        time_decay=0.0
    )
    
    std_paths = bicep.generate_paths(
        n_paths=n_chains,
        n_steps=n_samples,
        T=1.0,
        control_parameter=0.95,
        time_decay=0.0
    )
    
    # Transform to parameter space
    mean_samples = prior_mean_mu + prior_mean_std * mean_paths
    std_samples = np.exp(0.5 * std_paths)  # Ensure positive
    
    # Compute likelihoods and apply Metropolis acceptance
    accepted_means = []
    accepted_stds = []
    
    for chain in range(n_chains):
        chain_means = []
        chain_stds = []
        
        current_mean = mean_samples[chain, 0]
        current_std = std_samples[chain, 0]
        
        for i in range(1, n_samples + 1):
            proposed_mean = mean_samples[chain, i]
            proposed_std = std_samples[chain, i]
            
            # Compute log-likelihoods
            current_ll = -0.5 * np.sum((observations - current_mean)**2 / current_std**2) - n_observations * np.log(current_std)
            proposed_ll = -0.5 * np.sum((observations - proposed_mean)**2 / proposed_std**2) - n_observations * np.log(proposed_std)
            
            # Metropolis acceptance ratio
            log_ratio = proposed_ll - current_ll
            
            if np.log(np.random.uniform()) < log_ratio:
                current_mean = proposed_mean
                current_std = proposed_std
            
            chain_means.append(current_mean)
            chain_stds.append(current_std)
        
        accepted_means.extend(chain_means[1000:])  # Burn-in
        accepted_stds.extend(chain_stds[1000:])
    
    accepted_means = np.array(accepted_means)
    accepted_stds = np.array(accepted_stds)
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Trace plots
    plt.subplot(2, 3, 1)
    plt.plot(accepted_means[:1000], 'b-', alpha=0.7, linewidth=0.5)
    plt.axhline(true_mean, color='r', linestyle='--', label=f'True: {true_mean}')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Parameter')
    plt.title('MCMC Trace: Mean')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    plt.plot(accepted_stds[:1000], 'g-', alpha=0.7, linewidth=0.5)
    plt.axhline(true_std, color='r', linestyle='--', label=f'True: {true_std}')
    plt.xlabel('Iteration')
    plt.ylabel('Std Parameter')
    plt.title('MCMC Trace: Std')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Posterior distributions
    plt.subplot(2, 3, 3)
    plt.hist(accepted_means, bins=50, density=True, alpha=0.7, color='blue')
    plt.axvline(true_mean, color='r', linestyle='--', linewidth=2)
    plt.axvline(np.mean(accepted_means), color='k', linestyle='-', linewidth=2)
    plt.xlabel('Mean Parameter')
    plt.ylabel('Posterior Density')
    plt.title('Posterior: Mean')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 4)
    plt.hist(accepted_stds, bins=50, density=True, alpha=0.7, color='green')
    plt.axvline(true_std, color='r', linestyle='--', linewidth=2)
    plt.axvline(np.mean(accepted_stds), color='k', linestyle='-', linewidth=2)
    plt.xlabel('Std Parameter')
    plt.ylabel('Posterior Density')
    plt.title('Posterior: Std')
    plt.grid(True, alpha=0.3)
    
    # Joint posterior
    plt.subplot(2, 3, 5)
    plt.scatter(accepted_means[::10], accepted_stds[::10], alpha=0.5, s=1)
    plt.scatter(true_mean, true_std, color='red', s=100, marker='*', label='True')
    plt.xlabel('Mean Parameter')
    plt.ylabel('Std Parameter')
    plt.title('Joint Posterior Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Posterior predictive
    plt.subplot(2, 3, 6)
    n_predictive = 1000
    idx = np.random.choice(len(accepted_means), n_predictive)
    predictive_samples = []
    for i in idx:
        predictive_samples.extend(np.random.normal(accepted_means[i], accepted_stds[i], 10))
    
    plt.hist(predictive_samples, bins=50, density=True, alpha=0.7, color='purple', label='Posterior Predictive')
    plt.hist(observations, bins=20, density=True, alpha=0.5, color='orange', label='Observed Data')
    x_range = np.linspace(-2, 7, 100)
    plt.plot(x_range, 1/np.sqrt(2*np.pi*true_std**2) * np.exp(-(x_range-true_mean)**2/(2*true_std**2)), 
             'r--', linewidth=2, label='True Distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Posterior Predictive Check')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bayesian_inference.png', dpi=300)
    plt.show()
    
    print(f"\nBayesian Inference Results:")
    print(f"Posterior mean of μ: {np.mean(accepted_means):.3f} (True: {true_mean})")
    print(f"Posterior std of μ: {np.std(accepted_means):.3f}")
    print(f"Posterior mean of σ: {np.mean(accepted_stds):.3f} (True: {true_std})")
    print(f"Posterior std of σ: {np.std(accepted_stds):.3f}")
    print(f"95% Credible Interval for μ: [{np.percentile(accepted_means, 2.5):.3f}, {np.percentile(accepted_means, 97.5):.3f}]")
    print(f"95% Credible Interval for σ: [{np.percentile(accepted_stds, 2.5):.3f}, {np.percentile(accepted_stds, 97.5):.3f}]")

if __name__ == "__main__":
    print("BICEP Machine Learning Example: Uncertainty & Stochastic Optimization")
    print("=" * 70)
    
    # Example 1: SGD trajectory visualization
    print("\n1. Visualizing SGD trajectories on loss landscape:")
    visualize_sgd_trajectories()
    
    # Example 2: Uncertainty-aware neural network
    print("\n2. Building uncertainty-aware neural network:")
    uncertainty_aware_neural_network()
    
    # Example 3: Bayesian inference
    print("\n3. Monte Carlo Bayesian inference:")
    monte_carlo_bayesian_inference()
    
    print("\nMachine learning examples complete!")