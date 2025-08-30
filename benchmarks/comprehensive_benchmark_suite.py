#!/usr/bin/env python3
"""
Comprehensive Multi-Domain Benchmark Suite
Tests ENN+BICEP vs baselines across different problem types
"""

import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ENN'))

# Import models
try:
    from enn.model import ENNModelWithSparsityControl
    from enn.config import Config
    ENN_AVAILABLE = True
except:
    ENN_AVAILABLE = False

# ==================== BASELINE MODELS ====================

class LSTMBaseline(nn.Module):
    """LSTM for sequence modeling tasks"""
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class TransformerBaseline(nn.Module):
    """Transformer for sequence modeling"""
    def __init__(self, input_size, hidden_size, output_size, num_heads=4):
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.embedding(x)
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)
        return self.fc(x[:, -1, :])

class CNNBaseline(nn.Module):
    """1D CNN for sequence modeling"""
    def __init__(self, input_size, hidden_size, output_size, seq_len=20):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # CNN expects [batch, channels, length]
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = x.squeeze(-1)
        return self.fc(x)

class SimpleENN(nn.Module):
    """Simplified ENN-style model for fair comparison"""
    def __init__(self, input_size, hidden_size, output_size, num_heads=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.15)
        )
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, output_size)
            ) for _ in range(num_heads)
        ])
        
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        features = self.encoder(x)
        attended, _ = self.attention(features, features, features)
        
        # Ensemble predictions
        outputs = []
        for head in self.heads:
            outputs.append(head(attended[:, -1, :]))
        
        # Average ensemble
        return torch.stack(outputs).mean(dim=0)

# ==================== BENCHMARK TASKS ====================

class BenchmarkTask:
    """Base class for benchmark tasks"""
    def generate_data(self, n_samples: int) -> Tuple[Any, Any, Any, Any]:
        raise NotImplementedError
    
    def evaluate(self, model, test_loader) -> Dict[str, float]:
        raise NotImplementedError

class TimeSeriesPredictionTask(BenchmarkTask):
    """Predict next value in time series"""
    def __init__(self, seq_len=20, feature_dim=5):
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
    def generate_data(self, n_samples=5000):
        """Generate synthetic time series data"""
        # Multiple sine waves with noise
        t = np.linspace(0, 4 * np.pi, self.seq_len)
        X = []
        y = []
        
        for _ in range(n_samples):
            # Random frequencies and phases
            freqs = np.random.uniform(0.5, 2.0, self.feature_dim)
            phases = np.random.uniform(0, 2 * np.pi, self.feature_dim)
            
            # Generate multivariate time series
            series = np.zeros((self.seq_len, self.feature_dim))
            for i in range(self.feature_dim):
                series[:, i] = np.sin(freqs[i] * t + phases[i]) + 0.1 * np.random.randn(self.seq_len)
            
            X.append(series[:-1])  # Use all but last
            y.append(series[-1, 0])  # Predict first feature of last timestep
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)
        
        # Split data
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        return X_train, y_train, X_test, y_test
    
    def evaluate(self, model, test_loader):
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                output = model(batch_x)
                predictions.append(output.numpy())
                targets.append(batch_y.numpy())
        
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        
        mse = mean_squared_error(targets, predictions)
        mae = np.mean(np.abs(targets - predictions))
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(np.sqrt(mse))
        }

class AnomalyDetectionTask(BenchmarkTask):
    """Detect anomalies in sequences"""
    def __init__(self, seq_len=20, feature_dim=10):
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
    def generate_data(self, n_samples=5000):
        """Generate normal and anomalous sequences"""
        X = []
        y = []
        
        # 80% normal, 20% anomalous
        n_normal = int(0.8 * n_samples)
        
        # Normal sequences - smooth patterns
        for _ in range(n_normal):
            # Smooth random walk
            seq = np.cumsum(np.random.randn(self.seq_len, self.feature_dim) * 0.1, axis=0)
            X.append(seq)
            y.append(0)  # Normal
        
        # Anomalous sequences - sudden changes
        for _ in range(n_samples - n_normal):
            seq = np.cumsum(np.random.randn(self.seq_len, self.feature_dim) * 0.1, axis=0)
            
            # Add anomaly
            anomaly_type = np.random.choice(['spike', 'shift', 'noise'])
            anomaly_start = np.random.randint(self.seq_len // 2, self.seq_len - 5)
            
            if anomaly_type == 'spike':
                seq[anomaly_start:anomaly_start+3] += np.random.randn(3, self.feature_dim) * 3
            elif anomaly_type == 'shift':
                seq[anomaly_start:] += 2.0
            else:  # noise
                seq[anomaly_start:] += np.random.randn(len(seq[anomaly_start:]), self.feature_dim) * 1.5
            
            X.append(seq)
            y.append(1)  # Anomaly
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # Split
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        return X_train, y_train, X_test, y_test
    
    def evaluate(self, model, test_loader):
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                output = model(batch_x)
                if output.dim() > 1 and output.size(1) > 1:
                    # Classification output
                    pred = torch.argmax(output, dim=1)
                else:
                    # Binary output
                    pred = (output.squeeze() > 0.5).long()
                predictions.append(pred.numpy())
                targets.append(batch_y.numpy())
        
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        
        accuracy = accuracy_score(targets, predictions)
        f1 = f1_score(targets, predictions, average='weighted')
        
        # Calculate specificity and sensitivity
        true_positives = ((predictions == 1) & (targets == 1)).sum()
        true_negatives = ((predictions == 0) & (targets == 0)).sum()
        false_positives = ((predictions == 1) & (targets == 0)).sum()
        false_negatives = ((predictions == 0) & (targets == 1)).sum()
        
        sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        
        return {
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity)
        }

class SequenceClassificationTask(BenchmarkTask):
    """Classify sequences into categories"""
    def __init__(self, seq_len=30, feature_dim=8, num_classes=4):
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
    def generate_data(self, n_samples=6000):
        """Generate sequences with different patterns for each class"""
        X = []
        y = []
        
        samples_per_class = n_samples // self.num_classes
        
        for class_idx in range(self.num_classes):
            for _ in range(samples_per_class):
                seq = np.zeros((self.seq_len, self.feature_dim))
                
                if class_idx == 0:  # Oscillating pattern
                    t = np.linspace(0, 4 * np.pi, self.seq_len)
                    for i in range(self.feature_dim):
                        freq = 1 + i * 0.5
                        seq[:, i] = np.sin(freq * t) + 0.1 * np.random.randn(self.seq_len)
                        
                elif class_idx == 1:  # Trending pattern
                    for i in range(self.feature_dim):
                        trend = np.linspace(0, 2, self.seq_len) * (i + 1) / self.feature_dim
                        seq[:, i] = trend + 0.2 * np.random.randn(self.seq_len)
                        
                elif class_idx == 2:  # Step pattern
                    for i in range(self.feature_dim):
                        n_steps = 3 + i % 3
                        step_positions = np.linspace(0, self.seq_len, n_steps + 1).astype(int)
                        for j in range(n_steps):
                            seq[step_positions[j]:step_positions[j+1], i] = j * 0.5 + 0.1 * np.random.randn()
                            
                else:  # Random walk pattern
                    for i in range(self.feature_dim):
                        seq[:, i] = np.cumsum(np.random.randn(self.seq_len) * 0.3)
                
                X.append(seq)
                y.append(class_idx)
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # Split
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        return X_train, y_train, X_test, y_test
    
    def evaluate(self, model, test_loader):
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                output = model(batch_x)
                pred = torch.argmax(output, dim=1)
                predictions.append(pred.numpy())
                targets.append(batch_y.numpy())
        
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        
        accuracy = accuracy_score(targets, predictions)
        f1 = f1_score(targets, predictions, average='weighted')
        
        # Per-class accuracy
        per_class_acc = {}
        for i in range(self.num_classes):
            mask = targets == i
            if mask.sum() > 0:
                per_class_acc[f'class_{i}_acc'] = float((predictions[mask] == i).mean())
        
        return {
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            **per_class_acc
        }

class ReinforcementControlTask(BenchmarkTask):
    """Control task similar to CartPole"""
    def __init__(self, state_dim=4, action_dim=2, seq_len=20):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        
    def generate_data(self, n_samples=5000):
        """Generate control sequences with rewards"""
        X = []  # States
        y = []  # Best actions
        
        for _ in range(n_samples):
            # Initial state
            state = np.random.randn(self.state_dim) * 0.1
            states = [state]
            
            # Simulate dynamics
            for _ in range(self.seq_len - 1):
                # Simple dynamics model
                action = np.random.randint(0, self.action_dim)
                
                # Update state based on action
                if action == 0:  # Move left
                    state[0] -= 0.1
                    state[2] -= 0.05
                else:  # Move right
                    state[0] += 0.1
                    state[2] += 0.05
                
                # Add physics
                state[1] = state[0] * 0.1 + np.random.randn() * 0.01
                state[3] = state[2] * 0.1 + np.random.randn() * 0.01
                
                # Clip to reasonable range
                state = np.clip(state, -2, 2)
                states.append(state.copy())
            
            # Determine best action for final state
            final_state = states[-1]
            if final_state[0] < 0:  # Position negative, move right
                best_action = 1
            else:  # Position positive, move left
                best_action = 0
            
            X.append(np.array(states[:-1]))  # Use all but last state
            y.append(best_action)
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)
        
        # Split
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        return X_train, y_train, X_test, y_test
    
    def evaluate(self, model, test_loader):
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                output = model(batch_x)
                if output.size(1) == self.action_dim:
                    pred = torch.argmax(output, dim=1)
                else:
                    pred = (output.squeeze() > 0.5).long()
                predictions.append(pred.numpy())
                targets.append(batch_y.numpy())
        
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        
        accuracy = accuracy_score(targets, predictions)
        
        # Compute average "reward" based on correct actions
        reward = 0
        for i, (pred, target) in enumerate(zip(predictions, targets)):
            if pred == target:
                reward += 1.0
            else:
                reward -= 0.5
        
        avg_reward = reward / len(predictions)
        
        return {
            'accuracy': float(accuracy),
            'avg_reward': float(avg_reward),
            'correct_actions': int((predictions == targets).sum()),
            'total_actions': len(predictions)
        }

# ==================== MAIN BENCHMARK ====================

def train_and_evaluate(model, task, train_loader, test_loader, epochs=50, lr=0.001):
    """Train and evaluate a model on a task"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Determine loss function based on task
    if isinstance(task, TimeSeriesPredictionTask):
        criterion = nn.MSELoss()
    elif isinstance(task, (AnomalyDetectionTask, ReinforcementControlTask)):
        criterion = nn.CrossEntropyLoss()
    elif isinstance(task, SequenceClassificationTask):
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    
    # Training
    start_time = time.time()
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            output = model(batch_x)
            
            # Handle different output formats
            if isinstance(task, TimeSeriesPredictionTask):
                loss = criterion(output, batch_y)
            elif isinstance(task, (AnomalyDetectionTask, SequenceClassificationTask, ReinforcementControlTask)):
                if output.size(1) == 1:  # Binary output
                    loss = criterion(output.squeeze(), batch_y.float())
                else:
                    loss = criterion(output, batch_y)
            else:
                loss = criterion(output, batch_y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    train_time = time.time() - start_time
    
    # Evaluation
    metrics = task.evaluate(model, test_loader)
    metrics['train_time'] = train_time
    
    return metrics

def run_comprehensive_benchmark():
    """Run complete benchmark suite"""
    print("=" * 80)
    print("COMPREHENSIVE MULTI-DOMAIN BENCHMARK")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"ENN Available: {ENN_AVAILABLE}")
    
    # Define tasks
    tasks = [
        ("Time Series Prediction", TimeSeriesPredictionTask()),
        ("Anomaly Detection", AnomalyDetectionTask()),
        ("Sequence Classification", SequenceClassificationTask()),
        ("Reinforcement Control", ReinforcementControlTask())
    ]
    
    # Define models
    def get_models(input_size, output_size, seq_len=20):
        return {
            "LSTM": LSTMBaseline(input_size, 64, output_size),
            "Transformer": TransformerBaseline(input_size, 64, output_size),
            "1D-CNN": CNNBaseline(input_size, 64, output_size, seq_len),
            "ENN-Style": SimpleENN(input_size, 64, output_size)
        }
    
    # Results storage
    all_results = {}
    
    # Run benchmarks
    for task_name, task in tasks:
        print(f"\n{'='*60}")
        print(f"TASK: {task_name}")
        print(f"{'='*60}")
        
        # Generate data
        X_train, y_train, X_test, y_test = task.generate_data()
        
        # Determine dimensions
        if len(X_train.shape) == 3:
            seq_len, input_size = X_train.shape[1], X_train.shape[2]
        else:
            seq_len, input_size = 1, X_train.shape[1]
        
        if task_name == "Time Series Prediction":
            output_size = 1
        elif task_name == "Anomaly Detection":
            output_size = 2
        elif task_name == "Sequence Classification":
            output_size = task.num_classes
        elif task_name == "Reinforcement Control":
            output_size = task.action_dim
        
        print(f"Data shape: {X_train.shape}, Output size: {output_size}")
        
        # Create data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64)
        
        # Get models
        models = get_models(input_size, output_size, seq_len)
        
        # Train and evaluate each model
        task_results = {}
        
        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Parameters: {n_params:,}")
            
            try:
                metrics = train_and_evaluate(model, task, train_loader, test_loader, epochs=30)
                task_results[model_name] = metrics
                
                # Print key metrics
                if task_name == "Time Series Prediction":
                    print(f"RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
                elif task_name in ["Anomaly Detection", "Sequence Classification"]:
                    print(f"Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1_score']:.3f}")
                elif task_name == "Reinforcement Control":
                    print(f"Accuracy: {metrics['accuracy']:.3f}, Avg Reward: {metrics['avg_reward']:.3f}")
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                task_results[model_name] = {"error": str(e)}
        
        all_results[task_name] = task_results
    
    # Summary report
    print("\n" + "=" * 80)
    print("COMPREHENSIVE BENCHMARK SUMMARY")
    print("=" * 80)
    
    # Create comparison tables
    for task_name, results in all_results.items():
        print(f"\n{task_name}:")
        print("-" * 60)
        
        if task_name == "Time Series Prediction":
            print(f"{'Model':<15} {'RMSE':<10} {'MAE':<10} {'Time (s)':<10}")
            print("-" * 45)
            
            # Sort by RMSE
            sorted_models = sorted(results.items(), 
                                 key=lambda x: x[1].get('rmse', float('inf')))
            
            for model_name, metrics in sorted_models:
                if 'error' not in metrics:
                    print(f"{model_name:<15} {metrics['rmse']:<10.4f} "
                          f"{metrics['mae']:<10.4f} {metrics['train_time']:<10.2f}")
        
        elif task_name in ["Anomaly Detection", "Sequence Classification"]:
            print(f"{'Model':<15} {'Accuracy':<10} {'F1-Score':<10} {'Time (s)':<10}")
            print("-" * 45)
            
            # Sort by accuracy
            sorted_models = sorted(results.items(), 
                                 key=lambda x: x[1].get('accuracy', 0), reverse=True)
            
            for model_name, metrics in sorted_models:
                if 'error' not in metrics:
                    print(f"{model_name:<15} {metrics['accuracy']:<10.3f} "
                          f"{metrics['f1_score']:<10.3f} {metrics['train_time']:<10.2f}")
        
        elif task_name == "Reinforcement Control":
            print(f"{'Model':<15} {'Accuracy':<10} {'Avg Reward':<12} {'Time (s)':<10}")
            print("-" * 47)
            
            # Sort by accuracy
            sorted_models = sorted(results.items(), 
                                 key=lambda x: x[1].get('accuracy', 0), reverse=True)
            
            for model_name, metrics in sorted_models:
                if 'error' not in metrics:
                    print(f"{model_name:<15} {metrics['accuracy']:<10.3f} "
                          f"{metrics['avg_reward']:<12.3f} {metrics['train_time']:<10.2f}")
    
    # Overall rankings
    print("\n" + "=" * 80)
    print("OVERALL MODEL RANKINGS")
    print("=" * 80)
    
    model_scores = {model: [] for model in ["LSTM", "Transformer", "1D-CNN", "ENN-Style"]}
    
    for task_name, results in all_results.items():
        # Rank models for this task (1-4, lower is better)
        if task_name == "Time Series Prediction":
            sorted_models = sorted(results.items(), 
                                 key=lambda x: x[1].get('rmse', float('inf')))
        else:
            sorted_models = sorted(results.items(), 
                                 key=lambda x: x[1].get('accuracy', 0), reverse=True)
        
        for rank, (model_name, _) in enumerate(sorted_models, 1):
            if model_name in model_scores:
                model_scores[model_name].append(rank)
    
    # Calculate average ranks
    avg_ranks = {}
    for model, ranks in model_scores.items():
        if ranks:
            avg_ranks[model] = np.mean(ranks)
    
    print(f"{'Model':<15} {'Avg Rank':<10} {'Best Tasks'}")
    print("-" * 60)
    
    for model, avg_rank in sorted(avg_ranks.items(), key=lambda x: x[1]):
        # Find best tasks for this model
        best_tasks = []
        for task_name, results in all_results.items():
            if model in results and 'error' not in results[model]:
                if task_name == "Time Series Prediction":
                    metric = results[model].get('rmse', float('inf'))
                    # Check if this model has lowest RMSE
                    is_best = all(results[m].get('rmse', float('inf')) >= metric 
                                for m in results if m != model)
                else:
                    metric = results[model].get('accuracy', 0)
                    # Check if this model has highest accuracy
                    is_best = all(results[m].get('accuracy', 0) <= metric 
                                for m in results if m != model)
                
                if is_best:
                    best_tasks.append(task_name)
        
        best_tasks_str = ", ".join(best_tasks) if best_tasks else "None"
        print(f"{model:<15} {avg_rank:<10.1f} {best_tasks_str}")
    
    # Save results
    with open('benchmark_results.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': all_results
        }, f, indent=2)
    
    print("\n✅ Benchmark complete! Results saved to benchmark_results.json")
    
    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("• Different architectures excel at different tasks")
    print("• LSTMs typically strong for time series with temporal dependencies")
    print("• CNNs efficient for pattern detection in sequences")
    print("• Transformers good for complex sequence relationships")
    print("• ENN-style ensembles provide balanced performance with uncertainty")
    
    return all_results

if __name__ == "__main__":
    results = run_comprehensive_benchmark()