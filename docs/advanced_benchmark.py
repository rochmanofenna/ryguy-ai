#!/usr/bin/env python3
"""
Advanced Benchmark Suite for BICEP + ENN
Includes: Transformers, Liquid Neural Networks, PPO, and more
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
try:
    import gymnasium as gym
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.evaluation import evaluate_policy
except ImportError:
    gym = None
    PPO = SAC = TD3 = None
    make_vec_env = evaluate_policy = None
    print("Warning: gymnasium/stable-baselines3 not installed. RL benchmarks will be skipped.")
import warnings
warnings.filterwarnings('ignore')


class TransformerModel(nn.Module):
    """Vision Transformer / Sequence Transformer baseline"""
    def __init__(self, input_dim, hidden_dim=256, n_heads=8, n_layers=4, output_dim=None):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.output_proj = nn.Linear(hidden_dim, output_dim or input_dim)
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        seq_len = x.shape[1]
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)
        
        return self.output_proj(x.mean(dim=1))


class LiquidNeuralNetwork(nn.Module):
    """Liquid Neural Network implementation"""
    def __init__(self, input_dim, hidden_dim=128, output_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # ODE-based continuous dynamics
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim or input_dim)
        
        # Time constants and biases
        self.tau = nn.Parameter(torch.ones(hidden_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        
        # Liquid state
        self.register_buffer('hidden_state', torch.zeros(1, hidden_dim))
        
    def ode_step(self, x, h, dt=0.01):
        """Single ODE integration step"""
        dh_dt = (-h + torch.tanh(self.input_proj(x) + self.hidden_proj(h) + self.bias)) / self.tau
        h_new = h + dt * dh_dt
        return h_new
    
    def forward(self, x, steps=10):
        batch_size = x.shape[0]
        h = self.hidden_state.expand(batch_size, -1)
        
        # Integrate ODE
        for _ in range(steps):
            h = self.ode_step(x, h)
        
        self.hidden_state = h.mean(dim=0, keepdim=True).detach()
        return self.output_proj(h)


class NeuralODE(nn.Module):
    """Neural ODE baseline"""
    def __init__(self, input_dim, hidden_dim=128, output_dim=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim or input_dim)
        )
        
    def forward(self, x, t=None):
        # Simplified forward pass - in practice would use ODE solver
        return self.net(x)


class SpikeNeuralNetwork(nn.Module):
    """Spiking Neural Network baseline"""
    def __init__(self, input_dim, hidden_dim=128, output_dim=None, threshold=1.0):
        super().__init__()
        self.threshold = threshold
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim) 
        self.output = nn.Linear(hidden_dim, output_dim or input_dim)
        
        # Membrane potentials
        self.register_buffer('v_mem', torch.zeros(1, hidden_dim))
        
    def forward(self, x, steps=20):
        batch_size = x.shape[0]
        v = self.v_mem.expand(batch_size, -1)
        spikes = []
        
        for _ in range(steps):
            v = v * 0.9 + self.input_proj(x)  # Leaky integration
            spike = (v > self.threshold).float()
            v = v * (1 - spike)  # Reset after spike
            spikes.append(spike)
            
        # Decode from spike trains
        spike_counts = torch.stack(spikes).mean(dim=0)
        return self.output(spike_counts)


class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network for relational reasoning"""
    def __init__(self, input_dim, hidden_dim=128, output_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim or input_dim)
        )
        
    def forward(self, x, adj_matrix=None):
        # Simplified GNN forward pass
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        h = self.node_encoder(x)
        
        # Simple aggregation for batch processing
        output = self.node_mlp(h)
        
        return output


class AdvancedBenchmarkSuite:
    """Comprehensive benchmark suite with modern architectures"""
    
    def __init__(self):
        self.results = defaultdict(dict)
        self.models = {
            'transformer': TransformerModel,
            'liquid_nn': LiquidNeuralNetwork,
            'neural_ode': NeuralODE,
            'spiking_nn': SpikeNeuralNetwork,
            'graph_nn': GraphNeuralNetwork,
        }
        
    def benchmark_sequence_learning(self, seq_length=100, input_dim=10, n_samples=1000):
        """Benchmark on sequence prediction tasks"""
        print("\n=== Sequence Learning Benchmark ===")
        
        # Generate synthetic sequence data
        X = torch.randn(n_samples, seq_length, input_dim)
        y = X.mean(dim=1) + 0.1 * torch.randn(n_samples, input_dim)
        
        train_size = int(0.8 * n_samples)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        for model_name, model_class in self.models.items():
            print(f"\nTesting {model_name}...")
            
            model = model_class(input_dim, output_dim=input_dim)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            # Training
            start_time = time.time()
            model.train()
            for epoch in range(10):
                optimizer.zero_grad()
                
                if model_name == 'transformer':
                    pred = model(X_train)
                else:
                    # Handle batch processing for other models
                    preds = []
                    for i in range(0, len(X_train), 32):
                        batch = X_train[i:i+32].mean(dim=1)  # Aggregate sequence
                        batch_pred = model(batch)
                        if len(batch_pred.shape) == 1:
                            batch_pred = batch_pred.unsqueeze(0)
                        preds.append(batch_pred)
                    pred = torch.cat(preds)
                    
                loss = F.mse_loss(pred, y_train)
                loss.backward()
                optimizer.step()
                
            train_time = time.time() - start_time
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                if model_name == 'transformer':
                    test_pred = model(X_test)
                else:
                    test_preds = []
                    for i in range(0, len(X_test), 32):
                        batch = X_test[i:i+32].mean(dim=1)
                        batch_pred = model(batch)
                        if len(batch_pred.shape) == 1:
                            batch_pred = batch_pred.unsqueeze(0)
                        test_preds.append(batch_pred)
                    test_pred = torch.cat(test_preds)
                    
                test_loss = F.mse_loss(test_pred, y_test).item()
                
            self.results['sequence_learning'][model_name] = {
                'test_loss': test_loss,
                'train_time': train_time,
                'params': sum(p.numel() for p in model.parameters())
            }
            
            print(f"  Test Loss: {test_loss:.4f}")
            print(f"  Train Time: {train_time:.2f}s")
            print(f"  Parameters: {self.results['sequence_learning'][model_name]['params']:,}")
            
    def benchmark_rl_with_ppo(self, env_name='CartPole-v1', n_episodes=100):
        """Benchmark RL algorithms including PPO"""
        print("\n=== Reinforcement Learning Benchmark (PPO/SAC/TD3) ===")
        
        if gym is None or PPO is None:
            print("Skipping RL benchmarks - gymnasium/stable-baselines3 not installed")
            return
            
        algorithms = {
            'PPO': PPO,
            'SAC': SAC if env_name != 'CartPole-v1' else None,  # SAC needs continuous actions
            'TD3': TD3 if env_name != 'CartPole-v1' else None,  # TD3 needs continuous actions
        }
        
        for algo_name, algo_class in algorithms.items():
            if algo_class is None:
                continue
                
            print(f"\nTesting {algo_name}...")
            
            # Create environment
            env = make_vec_env(env_name, n_envs=4)
            
            # Train model
            start_time = time.time()
            model = algo_class('MlpPolicy', env, verbose=0)
            model.learn(total_timesteps=10000)
            train_time = time.time() - start_time
            
            # Evaluate
            mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_episodes)
            
            self.results['rl_algorithms'][algo_name] = {
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'train_time': train_time,
                'env': env_name
            }
            
            print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
            print(f"  Train Time: {train_time:.2f}s")
            
    def benchmark_continual_learning(self, n_tasks=5):
        """Benchmark continual/lifelong learning capabilities"""
        print("\n=== Continual Learning Benchmark ===")
        
        task_results = defaultdict(list)
        
        for model_name, model_class in self.models.items():
            print(f"\nTesting {model_name}...")
            
            model = model_class(10, output_dim=2)  # Binary classification
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            accuracies = []
            
            for task_id in range(n_tasks):
                # Generate task-specific data
                X = torch.randn(500, 10) + task_id
                y = (X.sum(dim=1) > task_id).long()
                
                # Train on new task
                model.train()
                for _ in range(20):
                    optimizer.zero_grad()
                    pred = model(X)
                    if len(pred.shape) == 1:
                        pred = pred.unsqueeze(0)
                    loss = F.cross_entropy(pred, y)
                    loss.backward()
                    optimizer.step()
                
                # Test on all previous tasks
                model.eval()
                task_accs = []
                for prev_task in range(task_id + 1):
                    X_test = torch.randn(100, 10) + prev_task
                    y_test = (X_test.sum(dim=1) > prev_task).long()
                    
                    with torch.no_grad():
                        pred = model(X_test)
                        if len(pred.shape) == 1:
                            pred = pred.unsqueeze(0)
                        acc = (pred.argmax(dim=1) == y_test).float().mean().item()
                        task_accs.append(acc)
                
                accuracies.append(np.mean(task_accs))
                
            self.results['continual_learning'][model_name] = {
                'final_avg_accuracy': accuracies[-1],
                'forgetting': accuracies[0] - accuracies[-1] if len(accuracies) > 1 else 0,
                'accuracy_progression': accuracies
            }
            
            print(f"  Final Avg Accuracy: {accuracies[-1]:.3f}")
            print(f"  Catastrophic Forgetting: {self.results['continual_learning'][model_name]['forgetting']:.3f}")
            
    def benchmark_few_shot_learning(self, n_shots=5):
        """Benchmark few-shot learning capabilities"""
        print("\n=== Few-Shot Learning Benchmark ===")
        
        for model_name, model_class in self.models.items():
            print(f"\nTesting {model_name}...")
            
            model = model_class(10, output_dim=10)  # 10-way classification
            
            # Meta-learning phase (simplified)
            meta_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            test_accuracies = []
            
            for episode in range(10):
                # Support set (few examples per class)
                support_X = torch.randn(10 * n_shots, 10)
                support_y = torch.repeat_interleave(torch.arange(10), n_shots)
                
                # Query set
                query_X = torch.randn(100, 10)
                query_y = torch.randint(0, 10, (100,))
                
                # Adaptation
                model.train()
                for _ in range(10):
                    meta_optimizer.zero_grad()
                    pred = model(support_X)
                    loss = F.cross_entropy(pred, support_y)
                    loss.backward()
                    meta_optimizer.step()
                
                # Evaluation
                model.eval()
                with torch.no_grad():
                    pred = model(query_X)
                    acc = (pred.argmax(dim=1) == query_y).float().mean().item()
                    test_accuracies.append(acc)
                    
            self.results['few_shot_learning'][model_name] = {
                'mean_accuracy': np.mean(test_accuracies),
                'std_accuracy': np.std(test_accuracies),
                'n_shots': n_shots
            }
            
            print(f"  {n_shots}-shot Accuracy: {np.mean(test_accuracies):.3f} ± {np.std(test_accuracies):.3f}")
            
    def benchmark_adversarial_robustness(self, epsilon=0.1):
        """Benchmark adversarial robustness"""
        print("\n=== Adversarial Robustness Benchmark ===")
        
        # Generate clean data
        X = torch.randn(1000, 10)
        y = (X.sum(dim=1) > 0).long()
        
        for model_name, model_class in self.models.items():
            print(f"\nTesting {model_name}...")
            
            model = model_class(10, output_dim=2)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            # Train model
            model.train()
            for _ in range(50):
                optimizer.zero_grad()
                pred = model(X[:800])
                loss = F.cross_entropy(pred, y[:800])
                loss.backward()
                optimizer.step()
            
            # Test clean accuracy
            model.eval()
            with torch.no_grad():
                clean_pred = model(X[800:])
                clean_acc = (clean_pred.argmax(dim=1) == y[800:]).float().mean().item()
            
            # Generate adversarial examples (FGSM)
            X_test = X[800:].clone().requires_grad_(True)
            pred = model(X_test)
            loss = F.cross_entropy(pred, y[800:])
            loss.backward()
            
            if X_test.grad is not None:
                X_adv = X_test + epsilon * X_test.grad.sign()
            else:
                X_adv = X_test + epsilon * torch.randn_like(X_test).sign()
            
            # Test adversarial accuracy
            with torch.no_grad():
                adv_pred = model(X_adv)
                adv_acc = (adv_pred.argmax(dim=1) == y[800:]).float().mean().item()
                
            self.results['adversarial_robustness'][model_name] = {
                'clean_accuracy': clean_acc,
                'adversarial_accuracy': adv_acc,
                'robustness_gap': clean_acc - adv_acc,
                'epsilon': epsilon
            }
            
            print(f"  Clean Accuracy: {clean_acc:.3f}")
            print(f"  Adversarial Accuracy: {adv_acc:.3f}")
            print(f"  Robustness Gap: {clean_acc - adv_acc:.3f}")
            
    def generate_report(self):
        """Generate comprehensive benchmark report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE BENCHMARK REPORT")
        print("="*60)
        
        # Create summary DataFrame
        summary_data = []
        
        for benchmark_name, results in self.results.items():
            print(f"\n### {benchmark_name.replace('_', ' ').title()} ###")
            
            if benchmark_name == 'sequence_learning':
                print("\n| Model | Test Loss | Train Time | Parameters |")
                print("|-------|-----------|------------|------------|")
                for model, metrics in results.items():
                    print(f"| {model} | {metrics['test_loss']:.4f} | {metrics['train_time']:.2f}s | {metrics['params']:,} |")
                    
            elif benchmark_name == 'rl_algorithms':
                print("\n| Algorithm | Mean Reward | Std Reward | Train Time |")
                print("|-----------|-------------|------------|------------|")
                for algo, metrics in results.items():
                    print(f"| {algo} | {metrics['mean_reward']:.2f} | {metrics['std_reward']:.2f} | {metrics['train_time']:.2f}s |")
                    
            elif benchmark_name == 'continual_learning':
                print("\n| Model | Final Accuracy | Forgetting |")
                print("|-------|----------------|------------|")
                for model, metrics in results.items():
                    print(f"| {model} | {metrics['final_avg_accuracy']:.3f} | {metrics['forgetting']:.3f} |")
                    
            elif benchmark_name == 'few_shot_learning':
                print("\n| Model | Mean Accuracy | Std Accuracy |")
                print("|-------|---------------|--------------|")
                for model, metrics in results.items():
                    print(f"| {model} | {metrics['mean_accuracy']:.3f} | {metrics['std_accuracy']:.3f} |")
                    
            elif benchmark_name == 'adversarial_robustness':
                print("\n| Model | Clean Acc | Adversarial Acc | Robustness Gap |")
                print("|-------|-----------|-----------------|----------------|")
                for model, metrics in results.items():
                    print(f"| {model} | {metrics['clean_accuracy']:.3f} | {metrics['adversarial_accuracy']:.3f} | {metrics['robustness_gap']:.3f} |")
        
        # Save results
        import json
        with open('advanced_benchmark_results.json', 'w') as f:
            json.dump(dict(self.results), f, indent=2)
            
        print("\n✅ Results saved to 'advanced_benchmark_results.json'")
        
    def visualize_results(self):
        """Create visualization plots"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot 1: Sequence Learning Performance
        if 'sequence_learning' in self.results:
            models = list(self.results['sequence_learning'].keys())
            test_losses = [self.results['sequence_learning'][m]['test_loss'] for m in models]
            
            axes[0].bar(models, test_losses)
            axes[0].set_title('Sequence Learning Performance')
            axes[0].set_ylabel('Test Loss')
            axes[0].tick_params(axis='x', rotation=45)
            
        # Plot 2: RL Algorithm Comparison
        if 'rl_algorithms' in self.results:
            algos = list(self.results['rl_algorithms'].keys())
            rewards = [self.results['rl_algorithms'][a]['mean_reward'] for a in algos]
            stds = [self.results['rl_algorithms'][a]['std_reward'] for a in algos]
            
            axes[1].bar(algos, rewards, yerr=stds)
            axes[1].set_title('RL Algorithm Performance')
            axes[1].set_ylabel('Mean Reward')
            
        # Plot 3: Continual Learning
        if 'continual_learning' in self.results:
            models = list(self.results['continual_learning'].keys())
            final_accs = [self.results['continual_learning'][m]['final_avg_accuracy'] for m in models]
            
            axes[2].bar(models, final_accs)
            axes[2].set_title('Continual Learning Final Accuracy')
            axes[2].set_ylabel('Accuracy')
            axes[2].tick_params(axis='x', rotation=45)
            
        # Plot 4: Few-Shot Learning
        if 'few_shot_learning' in self.results:
            models = list(self.results['few_shot_learning'].keys())
            accs = [self.results['few_shot_learning'][m]['mean_accuracy'] for m in models]
            
            axes[3].bar(models, accs)
            axes[3].set_title('Few-Shot Learning Accuracy')
            axes[3].set_ylabel('Accuracy')
            axes[3].tick_params(axis='x', rotation=45)
            
        # Plot 5: Adversarial Robustness
        if 'adversarial_robustness' in self.results:
            models = list(self.results['adversarial_robustness'].keys())
            clean = [self.results['adversarial_robustness'][m]['clean_accuracy'] for m in models]
            adv = [self.results['adversarial_robustness'][m]['adversarial_accuracy'] for m in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            axes[4].bar(x - width/2, clean, width, label='Clean')
            axes[4].bar(x + width/2, adv, width, label='Adversarial')
            axes[4].set_xticks(x)
            axes[4].set_xticklabels(models, rotation=45)
            axes[4].set_title('Adversarial Robustness')
            axes[4].set_ylabel('Accuracy')
            axes[4].legend()
            
        # Plot 6: Overall Performance Radar Chart
        if len(self.results) > 0:
            # Aggregate scores for radar chart
            categories = ['Seq Learning', 'RL', 'Continual', 'Few-Shot', 'Robustness']
            
            # This would need proper normalization in practice
            axes[5].set_title('Overall Model Comparison')
            axes[5].text(0.5, 0.5, 'Radar chart would go here\nwith normalized scores', 
                        ha='center', va='center', transform=axes[5].transAxes)
            
        plt.tight_layout()
        plt.savefig('advanced_benchmark_plots.png', dpi=300, bbox_inches='tight')
        print("\n📊 Plots saved to 'advanced_benchmark_plots.png'")
        

def main():
    """Run comprehensive benchmark suite"""
    print("🚀 Starting Advanced Benchmark Suite")
    print("=" * 60)
    
    suite = AdvancedBenchmarkSuite()
    
    # Run all benchmarks
    suite.benchmark_sequence_learning()
    suite.benchmark_rl_with_ppo()
    suite.benchmark_continual_learning()
    suite.benchmark_few_shot_learning()
    suite.benchmark_adversarial_robustness()
    
    # Generate report and visualizations
    suite.generate_report()
    suite.visualize_results()
    
    print("\n✅ Benchmark suite completed!")
    

if __name__ == "__main__":
    main()