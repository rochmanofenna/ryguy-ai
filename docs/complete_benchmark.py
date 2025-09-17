#!/usr/bin/env python3
"""
Complete Benchmark Suite: BICEP + ENN vs All Modern Architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import advanced models
from advanced_benchmark import (
    TransformerModel, LiquidNeuralNetwork, NeuralODE, 
    SpikeNeuralNetwork, GraphNeuralNetwork
)

# Try importing RL libraries
try:
    import gymnasium as gym
    from stable_baselines3 import PPO, SAC, TD3, A2C, DQN
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    print("Warning: RL libraries not available")


class EnsembleNeuralNetwork(nn.Module):
    """Enhanced Ensemble Neural Network (ENN) implementation"""
    def __init__(self, input_dim, hidden_dim=128, output_dim=None, n_heads=5):
        super().__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        
        # Create ensemble of networks
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim or input_dim)
            ) for _ in range(n_heads)
        ])
        
        # Temperature parameter for uncertainty
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, x, return_uncertainty=False):
        outputs = torch.stack([head(x) for head in self.heads])
        mean_output = outputs.mean(dim=0)
        
        if return_uncertainty:
            uncertainty = outputs.std(dim=0)
            return mean_output, uncertainty
        return mean_output


class BICEPDemonstrator:
    """BICEP (Baseline Initiated Collaborative Exploration Protocol) implementation"""
    def __init__(self, env_shape, action_dim, n_demonstrations=100):
        self.env_shape = env_shape
        self.action_dim = action_dim
        self.n_demonstrations = n_demonstrations
        self.demonstrations = []
        
    def generate_demonstrations(self, env_fn=None):
        """Generate stochastic demonstrations"""
        if env_fn is None:
            # Generate synthetic demonstrations
            for _ in range(self.n_demonstrations):
                trajectory = []
                state = torch.randn(self.env_shape)
                
                for t in range(20):  # 20 steps per demo
                    # Stochastic policy with exploration
                    action = torch.randn(self.action_dim)
                    next_state = state + 0.1 * action + 0.05 * torch.randn_like(state)
                    reward = -torch.norm(next_state).item()
                    
                    trajectory.append({
                        'state': state.clone(),
                        'action': action,
                        'next_state': next_state.clone(),
                        'reward': reward
                    })
                    state = next_state
                    
                self.demonstrations.append(trajectory)
        else:
            # Use actual environment
            env = env_fn()
            for _ in range(self.n_demonstrations):
                trajectory = []
                obs, _ = env.reset()
                done = False
                
                while not done and len(trajectory) < 100:
                    action = env.action_space.sample()
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    
                    trajectory.append({
                        'state': torch.tensor(obs, dtype=torch.float32),
                        'action': torch.tensor([action], dtype=torch.float32),
                        'next_state': torch.tensor(next_obs, dtype=torch.float32),
                        'reward': reward
                    })
                    obs = next_obs
                    
                if trajectory:
                    self.demonstrations.append(trajectory)
                    
    def get_demonstration_batch(self, batch_size=32):
        """Sample batch from demonstrations"""
        if not self.demonstrations:
            self.generate_demonstrations()
            
        batch = []
        for _ in range(batch_size):
            demo_idx = np.random.randint(len(self.demonstrations))
            step_idx = np.random.randint(len(self.demonstrations[demo_idx]))
            batch.append(self.demonstrations[demo_idx][step_idx])
            
        states = torch.stack([b['state'] for b in batch])
        actions = torch.stack([b['action'] for b in batch])
        rewards = torch.tensor([b['reward'] for b in batch])
        
        return states, actions, rewards


class BICEPwithENN(nn.Module):
    """Combined BICEP + ENN model"""
    def __init__(self, input_dim, hidden_dim=128, output_dim=None, n_heads=5):
        super().__init__()
        self.enn = EnsembleNeuralNetwork(input_dim, hidden_dim, output_dim, n_heads)
        self.bicep = BICEPDemonstrator((input_dim,), output_dim or input_dim)
        
        # Additional components for RL
        self.value_head = nn.Linear(hidden_dim, 1)
        self.policy_head = nn.Linear(hidden_dim, output_dim or input_dim)
        
    def forward(self, x, return_all=False):
        # Get ENN predictions with uncertainty
        mean_pred, uncertainty = self.enn(x, return_uncertainty=True)
        
        if return_all:
            # For RL tasks, return policy and value
            features = self.enn.heads[0][:-1](x)  # Get features from first head
            value = self.value_head(features)
            policy = F.softmax(self.policy_head(features), dim=-1)
            
            return {
                'prediction': mean_pred,
                'uncertainty': uncertainty,
                'value': value,
                'policy': policy
            }
        
        return mean_pred
    
    def train_with_demonstrations(self, optimizer, epochs=10):
        """Train using BICEP demonstrations"""
        for epoch in range(epochs):
            states, actions, rewards = self.bicep.get_demonstration_batch()
            
            optimizer.zero_grad()
            pred = self.forward(states)
            loss = F.mse_loss(pred, actions) - 0.1 * rewards.mean()  # Reward shaping
            loss.backward()
            optimizer.step()


class CompleteBenchmarkSuite:
    """Comprehensive benchmark comparing all architectures"""
    
    def __init__(self):
        self.results = defaultdict(lambda: defaultdict(dict))
        
        # All models to benchmark
        self.models = {
            # Previous models
            'transformer': TransformerModel,
            'liquid_nn': LiquidNeuralNetwork,
            'neural_ode': NeuralODE,
            'spiking_nn': SpikeNeuralNetwork,
            'graph_nn': GraphNeuralNetwork,
            # New models
            'enn': EnsembleNeuralNetwork,
            'bicep_enn': BICEPwithENN,
        }
        
        # RL environments to test
        self.rl_envs = [
            'CartPole-v1',
            'MountainCar-v0',
            'Pendulum-v1',
            'LunarLander-v3',
            'BipedalWalker-v3',
        ]
        
    def benchmark_all_sequence_tasks(self):
        """Extended sequence learning benchmarks"""
        print("\n=== COMPLETE SEQUENCE LEARNING BENCHMARK ===")
        
        tasks = {
            'time_series': self._benchmark_time_series,
            'anomaly_detection': self._benchmark_anomaly_detection,
            'sequence_classification': self._benchmark_sequence_classification,
            'multi_modal': self._benchmark_multimodal,
        }
        
        for task_name, task_fn in tasks.items():
            print(f"\n--- {task_name.replace('_', ' ').title()} ---")
            task_fn()
            
    def _benchmark_time_series(self):
        """Time series prediction benchmark"""
        # Generate synthetic time series
        n_samples = 2000
        seq_length = 50
        
        # Create multiple patterns
        t = torch.linspace(0, 4*np.pi, seq_length).unsqueeze(0).repeat(n_samples, 1)
        patterns = [
            torch.sin(t),
            torch.cos(2*t),
            0.5 * torch.sin(3*t + np.pi/4),
            torch.exp(-0.1 * t) * torch.cos(t)
        ]
        
        X = torch.stack(patterns, dim=-1)  # [n_samples, seq_length, n_features]
        noise = 0.1 * torch.randn_like(X)
        X = X + noise
        
        # Target is next timestep
        y = X[:, -1, :]
        X = X[:, :-1, :]
        
        # Split data
        split_idx = int(0.8 * n_samples)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        for model_name, model_class in self.models.items():
            print(f"\nTesting {model_name}...")
            
            model = model_class(input_dim=4, hidden_dim=128, output_dim=4)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            # Special handling for BICEP+ENN
            if model_name == 'bicep_enn':
                model.bicep.generate_demonstrations()
                model.train_with_demonstrations(optimizer, epochs=5)
            
            # Training
            start_time = time.time()
            model.train()
            
            best_loss = float('inf')
            for epoch in range(20):
                # Process in batches
                for i in range(0, len(X_train), 32):
                    batch_X = X_train[i:i+32]
                    batch_y = y_train[i:i+32]
                    
                    optimizer.zero_grad()
                    
                    # Handle different model types
                    if model_name == 'transformer':
                        pred = model(batch_X)
                    else:
                        # Aggregate sequence for non-sequential models
                        batch_X_agg = batch_X.mean(dim=1)
                        pred = model(batch_X_agg)
                        
                    loss = F.mse_loss(pred, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        
            train_time = time.time() - start_time
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                if model_name == 'transformer':
                    test_pred = model(X_test)
                else:
                    test_pred = model(X_test.mean(dim=1))
                    
                test_mse = F.mse_loss(test_pred, y_test).item()
                test_mae = F.l1_loss(test_pred, y_test).item()
                
                # Calculate R² score
                ss_res = ((y_test - test_pred) ** 2).sum()
                ss_tot = ((y_test - y_test.mean()) ** 2).sum()
                r2_score = 1 - (ss_res / ss_tot).item()
                
            self.results['time_series'][model_name] = {
                'mse': test_mse,
                'mae': test_mae,
                'r2_score': r2_score,
                'train_time': train_time,
                'best_train_loss': best_loss
            }
            
            print(f"  MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, R²: {r2_score:.4f}")
            print(f"  Train Time: {train_time:.2f}s")
            
    def _benchmark_anomaly_detection(self):
        """Anomaly detection benchmark"""
        # Generate normal and anomalous sequences
        n_normal = 1500
        n_anomaly = 500
        seq_length = 30
        
        # Normal data - smooth patterns
        normal_data = torch.randn(n_normal, seq_length, 5) * 0.5
        normal_labels = torch.zeros(n_normal)
        
        # Anomalous data - spiky patterns
        anomaly_data = torch.randn(n_anomaly, seq_length, 5) * 2.0
        anomaly_data += torch.randn(n_anomaly, 1, 5) * 5.0  # Add spikes
        anomaly_labels = torch.ones(n_anomaly)
        
        # Combine and shuffle
        X = torch.cat([normal_data, anomaly_data])
        y = torch.cat([normal_labels, anomaly_labels]).long()
        
        perm = torch.randperm(len(X))
        X, y = X[perm], y[perm]
        
        # Split
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        for model_name, model_class in self.models.items():
            print(f"\nTesting {model_name}...")
            
            model = model_class(input_dim=5, hidden_dim=64, output_dim=2)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            # Training
            start_time = time.time()
            model.train()
            
            for epoch in range(15):
                for i in range(0, len(X_train), 64):
                    batch_X = X_train[i:i+64]
                    batch_y = y_train[i:i+64]
                    
                    optimizer.zero_grad()
                    
                    if model_name == 'transformer':
                        pred = model(batch_X)
                    else:
                        pred = model(batch_X.mean(dim=1))
                        
                    loss = F.cross_entropy(pred, batch_y)
                    loss.backward()
                    optimizer.step()
                    
            train_time = time.time() - start_time
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                if model_name == 'transformer':
                    test_pred = model(X_test)
                else:
                    test_pred = model(X_test.mean(dim=1))
                    
                test_probs = F.softmax(test_pred, dim=1)
                test_preds = test_pred.argmax(dim=1)
                
                # Metrics
                accuracy = (test_preds == y_test).float().mean().item()
                
                # Precision, Recall, F1
                tp = ((test_preds == 1) & (y_test == 1)).sum().item()
                fp = ((test_preds == 1) & (y_test == 0)).sum().item()
                fn = ((test_preds == 0) & (y_test == 1)).sum().item()
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                # AUC score
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(y_test.numpy(), test_probs[:, 1].numpy())
                
            self.results['anomaly_detection'][model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'train_time': train_time
            }
            
            print(f"  Accuracy: {accuracy:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")
            
    def _benchmark_sequence_classification(self):
        """Sequence classification benchmark"""
        # Generate sequences with different patterns for classification
        n_samples_per_class = 500
        seq_length = 40
        n_classes = 5
        
        X_list = []
        y_list = []
        
        for class_idx in range(n_classes):
            # Each class has a different pattern
            if class_idx == 0:  # Increasing trend
                seq = torch.linspace(0, 1, seq_length).unsqueeze(0).repeat(n_samples_per_class, 1)
            elif class_idx == 1:  # Decreasing trend
                seq = torch.linspace(1, 0, seq_length).unsqueeze(0).repeat(n_samples_per_class, 1)
            elif class_idx == 2:  # Sine wave
                t = torch.linspace(0, 2*np.pi, seq_length)
                seq = torch.sin(t).unsqueeze(0).repeat(n_samples_per_class, 1)
            elif class_idx == 3:  # Square wave
                seq = torch.sign(torch.sin(torch.linspace(0, 4*np.pi, seq_length)))
                seq = seq.unsqueeze(0).repeat(n_samples_per_class, 1)
            else:  # Random walk
                seq = torch.randn(n_samples_per_class, seq_length).cumsum(dim=1)
                seq = (seq - seq.mean(dim=1, keepdim=True)) / seq.std(dim=1, keepdim=True)
                
            # Add noise and expand dimensions
            seq = seq.unsqueeze(-1) + 0.1 * torch.randn(n_samples_per_class, seq_length, 3)
            
            X_list.append(seq)
            y_list.append(torch.full((n_samples_per_class,), class_idx))
            
        X = torch.cat(X_list)
        y = torch.cat(y_list).long()
        
        # Shuffle
        perm = torch.randperm(len(X))
        X, y = X[perm], y[perm]
        
        # Split
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        for model_name, model_class in self.models.items():
            print(f"\nTesting {model_name}...")
            
            model = model_class(input_dim=3, hidden_dim=128, output_dim=n_classes)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            # Training
            start_time = time.time()
            model.train()
            
            for epoch in range(20):
                for i in range(0, len(X_train), 32):
                    batch_X = X_train[i:i+32]
                    batch_y = y_train[i:i+32]
                    
                    optimizer.zero_grad()
                    
                    if model_name == 'transformer':
                        pred = model(batch_X)
                    else:
                        pred = model(batch_X.mean(dim=1))
                        
                    loss = F.cross_entropy(pred, batch_y)
                    loss.backward()
                    optimizer.step()
                    
            train_time = time.time() - start_time
            
            # Evaluation
            model.eval()
            all_preds = []
            all_probs = []
            
            with torch.no_grad():
                for i in range(0, len(X_test), 32):
                    batch_X = X_test[i:i+32]
                    
                    if model_name == 'transformer':
                        pred = model(batch_X)
                    else:
                        pred = model(batch_X.mean(dim=1))
                        
                    all_preds.append(pred.argmax(dim=1))
                    all_probs.append(F.softmax(pred, dim=1))
                    
            test_preds = torch.cat(all_preds)
            test_probs = torch.cat(all_probs)
            
            # Metrics
            accuracy = (test_preds == y_test).float().mean().item()
            
            # Per-class accuracy
            per_class_acc = []
            for c in range(n_classes):
                mask = y_test == c
                if mask.sum() > 0:
                    class_acc = (test_preds[mask] == y_test[mask]).float().mean().item()
                    per_class_acc.append(class_acc)
                    
            self.results['sequence_classification'][model_name] = {
                'accuracy': accuracy,
                'per_class_accuracy': per_class_acc,
                'train_time': train_time
            }
            
            print(f"  Accuracy: {accuracy:.3f}, Per-class: {[f'{a:.2f}' for a in per_class_acc]}")
            
    def _benchmark_seq2seq(self):
        """Sequence-to-sequence benchmark"""
        # Simple sequence reversal task
        n_samples = 1000
        seq_length = 20
        vocab_size = 10
        
        # Generate random sequences
        X = torch.randint(0, vocab_size, (n_samples, seq_length))
        y = torch.flip(X, dims=[1])  # Reverse sequences
        
        # Convert to one-hot
        X_onehot = F.one_hot(X, vocab_size).float()
        
        # Split
        split_idx = int(0.8 * n_samples)
        X_train, X_test = X_onehot[:split_idx], X_onehot[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        for model_name, model_class in self.models.items():
            if model_name in ['spiking_nn']:  # Skip models not suitable for seq2seq
                continue
                
            print(f"\nTesting {model_name}...")
            
            # For seq2seq, we need models that can output sequences
            if model_name == 'transformer':
                model = model_class(input_dim=vocab_size, hidden_dim=128, output_dim=vocab_size)
            else:
                # Simple approach: predict one token at a time
                model = model_class(input_dim=vocab_size * seq_length, hidden_dim=256, output_dim=vocab_size)
                
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            # Training
            start_time = time.time()
            model.train()
            
            for epoch in range(30):
                for i in range(0, len(X_train), 32):
                    batch_X = X_train[i:i+32]
                    batch_y = y_train[i:i+32]
                    
                    optimizer.zero_grad()
                    
                    if model_name == 'transformer':
                        # Transformer can handle sequences naturally
                        pred = model(batch_X)  # [batch, seq_length, vocab_size]
                        # Reshape for loss calculation
                        pred_flat = pred.view(-1, vocab_size)  # [batch*seq_length, vocab_size]
                        y_flat = batch_y.view(-1)  # [batch*seq_length]
                        loss = F.cross_entropy(pred_flat, y_flat)
                    else:
                        # Flatten input and predict first token
                        batch_X_flat = batch_X.reshape(batch_X.shape[0], -1)
                        pred = model(batch_X_flat)
                        loss = F.cross_entropy(pred, batch_y[:, 0])  # Just first token for simplicity
                        
                    loss.backward()
                    optimizer.step()
                    
            train_time = time.time() - start_time
            
            # Evaluation
            model.eval()
            correct_sequences = 0
            
            with torch.no_grad():
                for i in range(len(X_test)):
                    if model_name == 'transformer':
                        pred = model(X_test[i:i+1])
                        pred_seq = pred.argmax(dim=-1).squeeze()
                        if torch.equal(pred_seq, y_test[i]):
                            correct_sequences += 1
                    else:
                        # Simple evaluation - just check first token
                        pred = model(X_test[i:i+1].reshape(1, -1))
                        pred_seq = pred.argmax(dim=-1)
                        # Only check first token for non-sequential models
                        if pred_seq == y_test[i, 0]:
                            correct_sequences += 0.05  # Partial credit
                        
            accuracy = correct_sequences / len(X_test)
            
            self.results['seq2seq'][model_name] = {
                'sequence_accuracy': accuracy,
                'train_time': train_time
            }
            
            print(f"  Sequence Accuracy: {accuracy:.3f}")
            
    def _benchmark_multimodal(self):
        """Multi-modal learning benchmark"""
        # Combine vision-like and text-like features
        n_samples = 1000
        img_features = 64  # Simulated image features
        text_features = 32  # Simulated text features
        
        # Generate correlated multi-modal data
        shared_factor = torch.randn(n_samples, 16)
        
        # Image features
        img_projection = torch.randn(16, img_features)
        img_data = torch.matmul(shared_factor, img_projection) + 0.5 * torch.randn(n_samples, img_features)
        
        # Text features  
        text_projection = torch.randn(16, text_features)
        text_data = torch.matmul(shared_factor, text_projection) + 0.5 * torch.randn(n_samples, text_features)
        
        # Concatenate modalities
        X = torch.cat([img_data, text_data], dim=1)
        
        # Task: predict if shared factor norm is above median
        y = (shared_factor.norm(dim=1) > shared_factor.norm(dim=1).median()).long()
        
        # Split
        split_idx = int(0.8 * n_samples)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        for model_name, model_class in self.models.items():
            print(f"\nTesting {model_name}...")
            
            model = model_class(input_dim=img_features + text_features, hidden_dim=128, output_dim=2)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            # Training
            start_time = time.time()
            model.train()
            
            for epoch in range(20):
                for i in range(0, len(X_train), 32):
                    batch_X = X_train[i:i+32]
                    batch_y = y_train[i:i+32]
                    
                    optimizer.zero_grad()
                    pred = model(batch_X)
                    loss = F.cross_entropy(pred, batch_y)
                    loss.backward()
                    optimizer.step()
                    
            train_time = time.time() - start_time
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                test_pred = model(X_test)
                test_preds = test_pred.argmax(dim=1)
                accuracy = (test_preds == y_test).float().mean().item()
                
                # Test on single modalities
                img_only_pred = model(torch.cat([X_test[:, :img_features], 
                                                torch.zeros_like(X_test[:, img_features:])], dim=1))
                img_only_acc = (img_only_pred.argmax(dim=1) == y_test).float().mean().item()
                
                text_only_pred = model(torch.cat([torch.zeros_like(X_test[:, :img_features]),
                                                 X_test[:, img_features:]], dim=1))
                text_only_acc = (text_only_pred.argmax(dim=1) == y_test).float().mean().item()
                
            self.results['multimodal'][model_name] = {
                'full_accuracy': accuracy,
                'image_only_accuracy': img_only_acc,
                'text_only_accuracy': text_only_acc,
                'train_time': train_time
            }
            
            print(f"  Full: {accuracy:.3f}, Image-only: {img_only_acc:.3f}, Text-only: {text_only_acc:.3f}")
            
    def benchmark_rl_comprehensive(self):
        """Comprehensive RL benchmarks across multiple environments"""
        if not RL_AVAILABLE:
            print("\n=== RL BENCHMARKS SKIPPED (libraries not available) ===")
            return
            
        print("\n=== COMPREHENSIVE RL BENCHMARKS ===")
        
        # RL algorithms to test
        rl_algorithms = {
            'PPO': PPO,
            'A2C': A2C,
            'DQN': DQN,  # For discrete action spaces
            'SAC': SAC,  # For continuous action spaces
            'TD3': TD3,  # For continuous action spaces
        }
        
        for env_name in self.rl_envs:
            print(f"\n--- Environment: {env_name} ---")
            
            # Check action space type
            temp_env = gym.make(env_name)
            is_discrete = isinstance(temp_env.action_space, gym.spaces.Discrete)
            temp_env.close()
            
            for algo_name, algo_class in rl_algorithms.items():
                # Skip incompatible algorithms
                if is_discrete and algo_name in ['SAC', 'TD3']:
                    continue
                if not is_discrete and algo_name == 'DQN':
                    continue
                    
                print(f"\nTesting {algo_name}...")
                
                try:
                    # Create vectorized environment
                    env = make_vec_env(env_name, n_envs=4)
                    
                    # Train model
                    start_time = time.time()
                    model = algo_class('MlpPolicy', env, verbose=0)
                    
                    # Adjust timesteps based on environment difficulty
                    timesteps = {
                        'CartPole-v1': 10000,
                        'MountainCar-v0': 50000,
                        'Pendulum-v1': 20000,
                        'LunarLander-v3': 100000,
                        'BipedalWalker-v3': 300000,
                    }.get(env_name, 50000)
                    
                    model.learn(total_timesteps=timesteps)
                    train_time = time.time() - start_time
                    
                    # Evaluate
                    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
                    
                    # Test sample efficiency
                    rewards_over_time = []
                    test_env = make_vec_env(env_name, n_envs=1)
                    
                    for i in range(5):
                        r, _ = evaluate_policy(model, test_env, n_eval_episodes=5)
                        rewards_over_time.append(r)
                        
                    self.results[f'rl_{env_name}'][algo_name] = {
                        'mean_reward': mean_reward,
                        'std_reward': std_reward,
                        'train_time': train_time,
                        'rewards_progression': rewards_over_time,
                        'timesteps': timesteps
                    }
                    
                    print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
                    print(f"  Train Time: {train_time:.2f}s")
                    
                    env.close()
                    test_env.close()
                    
                except Exception as e:
                    print(f"  Error: {str(e)}")
                    
        # Also test custom BICEP+ENN on RL tasks
        print("\n--- Testing BICEP+ENN on RL ---")
        for env_name in ['CartPole-v1', 'MountainCar-v0']:
            print(f"\nEnvironment: {env_name}")
            
            env = gym.make(env_name)
            obs_dim = env.observation_space.shape[0]
            
            if isinstance(env.action_space, gym.spaces.Discrete):
                act_dim = env.action_space.n
            else:
                act_dim = env.action_space.shape[0]
                
            # Create BICEP+ENN model
            model = BICEPwithENN(input_dim=obs_dim, hidden_dim=128, output_dim=act_dim)
            
            # Generate demonstrations using random policy
            model.bicep.generate_demonstrations(lambda: gym.make(env_name))
            
            # Simple training loop
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            start_time = time.time()
            
            # Train on demonstrations first
            model.train_with_demonstrations(optimizer, epochs=10)
            
            # Then do some RL training
            total_rewards = []
            for episode in range(100):
                obs, _ = env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    
                    with torch.no_grad():
                        output = model(obs_tensor, return_all=True)
                        if isinstance(env.action_space, gym.spaces.Discrete):
                            action = output['policy'].argmax().item()
                        else:
                            action = output['prediction'].squeeze().numpy()
                            
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    
                    # Simple online update
                    optimizer.zero_grad()
                    pred = model(obs_tensor)
                    target = torch.tensor([action], dtype=torch.float32)
                    loss = F.mse_loss(pred, target) - 0.01 * reward
                    loss.backward()
                    optimizer.step()
                    
                    obs = next_obs
                    
                total_rewards.append(episode_reward)
                
            train_time = time.time() - start_time
            
            # Evaluate
            eval_rewards = []
            for _ in range(10):
                obs, _ = env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    
                    with torch.no_grad():
                        output = model(obs_tensor, return_all=True)
                        if isinstance(env.action_space, gym.spaces.Discrete):
                            action = output['policy'].argmax().item()
                        else:
                            action = output['prediction'].squeeze().numpy()
                            
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    
                eval_rewards.append(episode_reward)
                
            env.close()
            
            self.results[f'rl_{env_name}']['bicep_enn'] = {
                'mean_reward': np.mean(eval_rewards),
                'std_reward': np.std(eval_rewards),
                'train_time': train_time,
                'training_rewards': total_rewards[-20:]  # Last 20 episodes
            }
            
            print(f"  Mean Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
            
    def benchmark_scalability(self):
        """Test scalability with different input sizes"""
        print("\n=== SCALABILITY BENCHMARK ===")
        
        input_sizes = [10, 50, 100, 500, 1000]
        batch_sizes = [1, 32, 128, 512]
        
        for model_name, model_class in self.models.items():
            print(f"\n{model_name}:")
            
            results = []
            
            for input_size in input_sizes:
                for batch_size in batch_sizes:
                    try:
                        model = model_class(input_dim=input_size, hidden_dim=128, output_dim=10)
                        
                        # Measure forward pass time
                        X = torch.randn(batch_size, input_size)
                        
                        # Warmup
                        for _ in range(10):
                            _ = model(X)
                            
                        # Time forward passes
                        start_time = time.time()
                        for _ in range(100):
                            _ = model(X)
                        forward_time = (time.time() - start_time) / 100
                        
                        # Measure memory
                        import tracemalloc
                        tracemalloc.start()
                        _ = model(X)
                        current, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        
                        results.append({
                            'input_size': input_size,
                            'batch_size': batch_size,
                            'forward_time_ms': forward_time * 1000,
                            'memory_mb': peak / 1024 / 1024
                        })
                        
                    except Exception as e:
                        results.append({
                            'input_size': input_size,
                            'batch_size': batch_size,
                            'forward_time_ms': -1,
                            'memory_mb': -1,
                            'error': str(e)
                        })
                        
            self.results['scalability'][model_name] = results
            
            # Print summary
            for r in results[:5]:  # Show first few results
                if r['forward_time_ms'] > 0:
                    print(f"  Input: {r['input_size']}, Batch: {r['batch_size']} -> "
                          f"{r['forward_time_ms']:.2f}ms, {r['memory_mb']:.1f}MB")
                          
    def generate_comprehensive_report(self):
        """Generate detailed comparison report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE BENCHMARK REPORT")
        print("="*80)
        
        # Save raw results
        with open('complete_benchmark_results.json', 'w') as f:
            json.dump(dict(self.results), f, indent=2, default=str)
            
        # Generate summary tables
        print("\n### OVERALL PERFORMANCE SUMMARY ###")
        
        # Aggregate scores across all tasks
        model_scores = defaultdict(list)
        
        for task, task_results in self.results.items():
            if task.startswith('rl_'):
                continue  # Handle RL separately
                
            for model, metrics in task_results.items():
                # Normalize scores based on task
                if 'accuracy' in metrics:
                    model_scores[model].append(metrics['accuracy'])
                elif 'mse' in metrics:
                    # Invert MSE so lower is better -> higher score
                    model_scores[model].append(1 / (1 + metrics['mse']))
                elif 'f1_score' in metrics:
                    model_scores[model].append(metrics['f1_score'])
                    
        # Calculate average scores
        print("\n| Model | Avg Score | # Tasks | Best At |")
        print("|-------|-----------|---------|---------|")
        
        for model in self.models.keys():
            if model in model_scores:
                avg_score = np.mean(model_scores[model])
                n_tasks = len(model_scores[model])
                
                # Find what this model is best at
                best_tasks = []
                for task, results in self.results.items():
                    if task.startswith('rl_'):
                        continue
                    if model in results:
                        # Check if this model has best score for this task
                        task_scores = []
                        for m, metrics in results.items():
                            if 'accuracy' in metrics:
                                task_scores.append((m, metrics['accuracy']))
                            elif 'mse' in metrics:
                                task_scores.append((m, -metrics['mse']))  # Negative for sorting
                                
                        if task_scores:
                            task_scores.sort(key=lambda x: x[1], reverse=True)
                            if task_scores[0][0] == model:
                                best_tasks.append(task)
                                
                best_at = ', '.join(best_tasks[:2]) if best_tasks else 'None'
                
                print(f"| {model} | {avg_score:.3f} | {n_tasks} | {best_at} |")
                
        # RL Summary
        if any(k.startswith('rl_') for k in self.results.keys()):
            print("\n### RL PERFORMANCE SUMMARY ###")
            
            rl_scores = defaultdict(list)
            
            for task, results in self.results.items():
                if task.startswith('rl_'):
                    env_name = task.replace('rl_', '')
                    print(f"\n{env_name}:")
                    
                    for algo, metrics in results.items():
                        if 'mean_reward' in metrics:
                            print(f"  {algo}: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
                            rl_scores[algo].append(metrics['mean_reward'])
                            
        # Efficiency Analysis
        print("\n### EFFICIENCY ANALYSIS ###")
        
        print("\n| Model | Avg Train Time | Params | Time/Param Efficiency |")
        print("|-------|----------------|--------|----------------------|")
        
        efficiency_data = []
        
        for model in self.models.keys():
            train_times = []
            param_count = None
            
            for task, results in self.results.items():
                if model in results and 'train_time' in results[model]:
                    train_times.append(results[model]['train_time'])
                    
            # Get parameter count from sequence learning task
            if 'sequence_learning' in self.results and model in self.results['sequence_learning']:
                if 'params' in self.results['sequence_learning'][model]:
                    param_count = self.results['sequence_learning'][model]['params']
                    
            if train_times and param_count:
                avg_time = np.mean(train_times)
                efficiency = avg_time / (param_count / 1000)  # Time per 1K params
                
                efficiency_data.append({
                    'model': model,
                    'avg_time': avg_time,
                    'params': param_count,
                    'efficiency': efficiency
                })
                
        # Sort by efficiency
        efficiency_data.sort(key=lambda x: x['efficiency'])
        
        for data in efficiency_data:
            print(f"| {data['model']} | {data['avg_time']:.2f}s | {data['params']:,} | {data['efficiency']:.6f} |")
            
        # Generate plots
        self._generate_comparison_plots()
        
        print("\n✅ Complete benchmark finished!")
        print("📊 Results saved to 'complete_benchmark_results.json'")
        print("📈 Plots saved to 'complete_benchmark_plots.png'")
        
    def _generate_comparison_plots(self):
        """Generate comprehensive comparison plots"""
        fig = plt.figure(figsize=(20, 15))
        
        # Plot 1: Performance across sequence tasks
        ax1 = plt.subplot(3, 3, 1)
        task_names = ['time_series', 'anomaly_detection', 'sequence_classification']
        model_names = list(self.models.keys())
        
        scores_matrix = []
        for model in model_names:
            scores = []
            for task in task_names:
                if task in self.results and model in self.results[task]:
                    if 'accuracy' in self.results[task][model]:
                        scores.append(self.results[task][model]['accuracy'])
                    elif 'mse' in self.results[task][model]:
                        scores.append(1 / (1 + self.results[task][model]['mse']))
                    else:
                        scores.append(0)
                else:
                    scores.append(0)
            scores_matrix.append(scores)
            
        scores_matrix = np.array(scores_matrix)
        
        im = ax1.imshow(scores_matrix, cmap='RdYlGn', aspect='auto')
        ax1.set_xticks(range(len(task_names)))
        ax1.set_xticklabels(task_names, rotation=45)
        ax1.set_yticks(range(len(model_names)))
        ax1.set_yticklabels(model_names)
        ax1.set_title('Performance Heatmap')
        plt.colorbar(im, ax=ax1)
        
        # Plot 2: Training efficiency
        ax2 = plt.subplot(3, 3, 2)
        train_times = []
        model_labels = []
        
        for model in model_names:
            times = []
            for task, results in self.results.items():
                if model in results and 'train_time' in results[model]:
                    times.append(results[model]['train_time'])
            if times:
                train_times.append(np.mean(times))
                model_labels.append(model)
                
        ax2.bar(model_labels, train_times)
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Avg Training Time (s)')
        ax2.set_title('Training Efficiency')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Scalability
        if 'scalability' in self.results:
            ax3 = plt.subplot(3, 3, 3)
            
            for model in ['transformer', 'liquid_nn', 'bicep_enn']:
                if model in self.results['scalability']:
                    data = self.results['scalability'][model]
                    
                    # Extract data for batch_size=32
                    input_sizes = []
                    forward_times = []
                    
                    for item in data:
                        if item['batch_size'] == 32 and item['forward_time_ms'] > 0:
                            input_sizes.append(item['input_size'])
                            forward_times.append(item['forward_time_ms'])
                            
                    if input_sizes:
                        ax3.plot(input_sizes, forward_times, marker='o', label=model)
                        
            ax3.set_xlabel('Input Size')
            ax3.set_ylabel('Forward Time (ms)')
            ax3.set_title('Scalability Analysis')
            ax3.legend()
            ax3.set_xscale('log')
            ax3.set_yscale('log')
            
        # Plot 4-6: RL Performance
        rl_envs_to_plot = ['CartPole-v1', 'MountainCar-v0', 'LunarLander-v3']
        
        for idx, env in enumerate(rl_envs_to_plot):
            if f'rl_{env}' in self.results:
                ax = plt.subplot(3, 3, 4 + idx)
                
                algos = []
                rewards = []
                stds = []
                
                for algo, metrics in self.results[f'rl_{env}'].items():
                    if 'mean_reward' in metrics:
                        algos.append(algo)
                        rewards.append(metrics['mean_reward'])
                        stds.append(metrics['std_reward'])
                        
                if algos:
                    ax.bar(algos, rewards, yerr=stds, capsize=5)
                    ax.set_xlabel('Algorithm')
                    ax.set_ylabel('Mean Reward')
                    ax.set_title(f'RL Performance: {env}')
                    ax.tick_params(axis='x', rotation=45)
                    
        # Plot 7: Multi-modal performance
        if 'multimodal' in self.results:
            ax7 = plt.subplot(3, 3, 7)
            
            model_names_mm = []
            full_acc = []
            img_acc = []
            text_acc = []
            
            for model, metrics in self.results['multimodal'].items():
                model_names_mm.append(model)
                full_acc.append(metrics['full_accuracy'])
                img_acc.append(metrics['image_only_accuracy'])
                text_acc.append(metrics['text_only_accuracy'])
                
            x = np.arange(len(model_names_mm))
            width = 0.25
            
            ax7.bar(x - width, full_acc, width, label='Full')
            ax7.bar(x, img_acc, width, label='Image Only')
            ax7.bar(x + width, text_acc, width, label='Text Only')
            
            ax7.set_xlabel('Model')
            ax7.set_ylabel('Accuracy')
            ax7.set_title('Multi-modal Learning')
            ax7.set_xticks(x)
            ax7.set_xticklabels(model_names_mm, rotation=45)
            ax7.legend()
            
        # Plot 8: Anomaly detection ROC curves (simplified)
        if 'anomaly_detection' in self.results:
            ax8 = plt.subplot(3, 3, 8)
            
            models_ad = []
            aucs = []
            
            for model, metrics in self.results['anomaly_detection'].items():
                if 'auc' in metrics:
                    models_ad.append(model)
                    aucs.append(metrics['auc'])
                    
            ax8.bar(models_ad, aucs)
            ax8.set_xlabel('Model')
            ax8.set_ylabel('AUC Score')
            ax8.set_title('Anomaly Detection Performance')
            ax8.tick_params(axis='x', rotation=45)
            ax8.set_ylim(0.5, 1.0)
            
        # Plot 9: Overall ranking
        ax9 = plt.subplot(3, 3, 9)
        
        # Calculate overall ranking
        rankings = defaultdict(list)
        
        for task, results in self.results.items():
            if task == 'scalability':
                continue
                
            # Get scores for this task
            task_scores = []
            for model, metrics in results.items():
                score = 0
                if 'accuracy' in metrics:
                    score = metrics['accuracy']
                elif 'mse' in metrics:
                    score = 1 / (1 + metrics['mse'])
                elif 'mean_reward' in metrics:
                    score = metrics['mean_reward'] / 100  # Normalize
                elif 'f1_score' in metrics:
                    score = metrics['f1_score']
                    
                task_scores.append((model, score))
                
            # Rank models for this task
            task_scores.sort(key=lambda x: x[1], reverse=True)
            for rank, (model, score) in enumerate(task_scores):
                rankings[model].append(rank + 1)
                
        # Calculate average ranking
        avg_rankings = []
        for model in self.models.keys():
            if model in rankings:
                avg_rank = np.mean(rankings[model])
                avg_rankings.append((model, avg_rank))
                
        avg_rankings.sort(key=lambda x: x[1])  # Lower rank is better
        
        models_ranked = [x[0] for x in avg_rankings]
        ranks = [x[1] for x in avg_rankings]
        
        ax9.barh(models_ranked, ranks)
        ax9.set_xlabel('Average Ranking (lower is better)')
        ax9.set_title('Overall Model Ranking')
        ax9.invert_xaxis()  # Invert so best (lowest) is on right
        
        plt.tight_layout()
        plt.savefig('complete_benchmark_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        

def main():
    """Run complete benchmark suite"""
    print("🚀 Starting Complete Benchmark Suite")
    print("This includes: Transformers, Liquid NN, Neural ODE, Spiking NN,")
    print("Graph NN, ENN, BICEP+ENN, and multiple RL algorithms")
    print("="*60)
    
    suite = CompleteBenchmarkSuite()
    
    # Run all benchmarks
    suite.benchmark_all_sequence_tasks()
    suite.benchmark_rl_comprehensive()
    suite.benchmark_scalability()
    
    # Generate comprehensive report
    suite.generate_comprehensive_report()
    
    print("\n✨ All benchmarks completed successfully!")
    

if __name__ == "__main__":
    main()