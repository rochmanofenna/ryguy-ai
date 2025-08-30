#!/usr/bin/env python3
"""
Enhanced Statistical Benchmark Suite for BICEP + ENN
Includes proper statistical testing, multiple seeds, error bars, and confidence intervals
"""

import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime
from dataclasses import dataclass
from scipy import stats
import yaml
import warnings
warnings.filterwarnings('ignore')

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ENN'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'BICEP'))

# Import models
try:
    from enn.model import ENNModelWithSparsityControl
    from enn.config import Config
    ENN_AVAILABLE = True
except:
    ENN_AVAILABLE = False

try:
    from bicep_core import BICEPCore
    BICEP_AVAILABLE = True
except:
    BICEP_AVAILABLE = False

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs"""
    n_seeds: int = 20  # Number of random seeds to test
    n_trials: int = 50  # Hyperparameter search budget per model
    test_split: float = 0.2
    confidence_level: float = 0.95
    epochs: int = 50
    batch_size: int = 64
    early_stopping: bool = True
    patience: int = 10
    
class ModelRegistry:
    """Registry for fair baseline comparisons with equal hyperparameter budgets"""
    
    @staticmethod
    def get_lstm_config():
        return {
            'hidden_size': [32, 64, 128, 256],
            'num_layers': [1, 2, 3],
            'lr': [0.001, 0.003, 0.01],
            'dropout': [0.1, 0.2, 0.3]
        }
    
    @staticmethod
    def get_transformer_config():
        return {
            'hidden_size': [64, 128, 256],
            'num_heads': [2, 4, 8],
            'num_layers': [2, 4, 6],
            'lr': [0.0003, 0.001, 0.003]
        }
    
    @staticmethod
    def get_cnn_config():
        return {
            'hidden_size': [32, 64, 128],
            'num_layers': [2, 3, 4],
            'kernel_size': [3, 5, 7],
            'lr': [0.001, 0.003, 0.01]
        }
    
    @staticmethod 
    def get_enn_config():
        return {
            'hidden_size': [64, 128, 256],
            'num_heads': [3, 5, 7],
            'dropout': [0.1, 0.15, 0.2],
            'lr': [0.0003, 0.001, 0.003]
        }

class EnhancedLSTM(nn.Module):
    """Enhanced LSTM with configurable architecture"""
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        x = self.dropout(lstm_out[:, -1, :])
        return self.fc(x)

class EnhancedTransformer(nn.Module):
    """Enhanced Transformer with configurable architecture"""
    def __init__(self, input_size, hidden_size, output_size, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.embedding(x)
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)
        return self.fc(x[:, -1, :])

class EnhancedCNN(nn.Module):
    """Enhanced 1D CNN with configurable architecture"""
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, kernel_size=3):
        super().__init__()
        layers = []
        in_channels = input_size
        
        for i in range(num_layers):
            layers.extend([
                nn.Conv1d(in_channels, hidden_size, kernel_size=kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.1)
            ])
            if i < num_layers - 1:
                layers.append(nn.MaxPool1d(2))
            in_channels = hidden_size
            
        self.conv_layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)

class EnhancedENN(nn.Module):
    """Enhanced ENN-style ensemble model"""
    def __init__(self, input_size, hidden_size, output_size, num_heads=5, dropout=0.15):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, output_size)
            ) for _ in range(num_heads)
        ])
        
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=min(4, num_heads), batch_first=True)
        self.uncertainty_head = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        features = self.encoder(x)
        attended, _ = self.attention(features, features, features)
        
        # Ensemble predictions with uncertainty
        outputs = []
        for head in self.heads:
            outputs.append(head(attended[:, -1, :]))
        
        # Mean prediction
        mean_pred = torch.stack(outputs).mean(dim=0)
        
        # Uncertainty estimation (variance across heads)
        uncertainty = torch.stack(outputs).var(dim=0)
        
        return mean_pred
    
    def predict_with_uncertainty(self, x):
        """Return both prediction and uncertainty estimates"""
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        features = self.encoder(x)
        attended, _ = self.attention(features, features, features)
        
        outputs = []
        for head in self.heads:
            outputs.append(head(attended[:, -1, :]))
        
        predictions = torch.stack(outputs)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        
        return mean_pred, uncertainty

class HyperparameterOptimizer:
    """Grid search optimizer with equal budget across models"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        
    def optimize_hyperparameters(self, model_class, param_grid, X_train, y_train, X_val, y_val, task_type='regression'):
        """Find best hyperparameters within budget"""
        best_score = float('inf') if task_type == 'regression' else 0
        best_params = None
        
        # Generate parameter combinations
        import itertools
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        # Randomly sample within budget
        np.random.shuffle(param_combinations)
        param_combinations = param_combinations[:self.config.n_trials]
        
        for params in param_combinations:
            param_dict = dict(zip(param_names, params))
            
            try:
                # Create model with these parameters
                model = self._create_model(model_class, param_dict, X_train.shape, y_train.shape)
                score = self._evaluate_model(model, X_train, y_train, X_val, y_val, 
                                           param_dict.get('lr', 0.001), task_type)
                
                if task_type == 'regression':
                    if score < best_score:
                        best_score = score
                        best_params = param_dict
                else:
                    if score > best_score:
                        best_score = score
                        best_params = param_dict
                        
            except Exception as e:
                continue  # Skip failed configurations
                
        return best_params or param_dict
    
    def _create_model(self, model_class, params, X_shape, y_shape):
        """Create model instance with given parameters"""
        input_size = X_shape[-1] if len(X_shape) > 2 else X_shape[1]
        output_size = y_shape[1] if len(y_shape) > 1 else 1
        
        # Remove lr from model params
        model_params = {k: v for k, v in params.items() if k != 'lr'}
        
        if model_class == EnhancedLSTM:
            return model_class(input_size, **model_params, output_size=output_size)
        elif model_class == EnhancedTransformer:
            return model_class(input_size, **model_params, output_size=output_size)
        elif model_class == EnhancedCNN:
            return model_class(input_size, **model_params, output_size=output_size)
        elif model_class == EnhancedENN:
            return model_class(input_size, **model_params, output_size=output_size)
    
    def _evaluate_model(self, model, X_train, y_train, X_val, y_val, lr, task_type):
        """Quick evaluation for hyperparameter search"""
        train_dataset = TensorDataset(torch.FloatTensor(X_train), 
                                    torch.FloatTensor(y_train) if task_type == 'regression' else torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), 
                                  torch.FloatTensor(y_val) if task_type == 'regression' else torch.LongTensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss() if task_type == 'regression' else nn.CrossEntropyLoss()
        
        # Quick training (fewer epochs for hyperparameter search)
        for epoch in range(min(10, self.config.epochs // 5)):
            model.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                output = model(batch_x)
                if task_type == 'classification' and output.size(1) == 1:
                    loss = criterion(output.squeeze(), batch_y.float())
                else:
                    loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
        
        # Validation
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                output = model(batch_x)
                if task_type == 'regression':
                    loss = criterion(output, batch_y)
                    total_loss += loss.item()
                else:
                    if output.size(1) > 1:
                        pred = torch.argmax(output, dim=1)
                    else:
                        pred = (output.squeeze() > 0.5).long()
                    correct += (pred == batch_y).sum().item()
                    total += batch_y.size(0)
        
        return total_loss / len(val_loader) if task_type == 'regression' else correct / total

class StatisticalAnalyzer:
    """Statistical analysis and confidence interval calculation"""
    
    @staticmethod
    def calculate_confidence_interval(data, confidence=0.95):
        """Calculate confidence interval for data"""
        n = len(data)
        mean = np.mean(data)
        sem = stats.sem(data)
        h = sem * stats.t.ppf((1 + confidence) / 2., n-1)
        return mean, mean - h, mean + h
    
    @staticmethod
    def compare_models(results1, results2, metric='accuracy'):
        """Statistical comparison between two model results"""
        data1 = [r[metric] for r in results1 if metric in r]
        data2 = [r[metric] for r in results2 if metric in r]
        
        if len(data1) == 0 or len(data2) == 0:
            return {'p_value': 1.0, 'significant': False, 'effect_size': 0}
        
        # Welch's t-test (unequal variances)
        statistic, p_value = stats.ttest_ind(data1, data2, equal_var=False)
        
        # Cohen's d effect size
        pooled_std = np.sqrt((np.var(data1) + np.var(data2)) / 2)
        effect_size = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0
        
        return {
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size': abs(effect_size),
            'statistic': statistic
        }
    
    @staticmethod
    def create_performance_plot(results, metric='accuracy', save_path='performance_comparison.png'):
        """Create performance comparison plot with error bars"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        models = list(results.keys())
        means = []
        errors = []
        
        for model in models:
            data = [r[metric] for r in results[model] if metric in r]
            if data:
                mean, ci_lower, ci_upper = StatisticalAnalyzer.calculate_confidence_interval(data)
                means.append(mean)
                errors.append([mean - ci_lower, ci_upper - mean])
            else:
                means.append(0)
                errors.append([0, 0])
        
        x_pos = np.arange(len(models))
        bars = ax.bar(x_pos, means, yerr=np.array(errors).T, capsize=5, 
                     alpha=0.7, error_kw={'elinewidth': 2, 'capthick': 2})
        
        ax.set_xlabel('Model')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Model Performance Comparison - {metric.replace("_", " ").title()}')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45)
        
        # Add value labels on bars
        for i, (bar, mean) in enumerate(zip(bars, means)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + errors[i][1]/2,
                   f'{mean:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

class EnhancedBenchmarkRunner:
    """Enhanced benchmark runner with statistical rigor"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.optimizer = HyperparameterOptimizer(config)
        self.analyzer = StatisticalAnalyzer()
        
    def run_benchmark_with_seeds(self, task, n_samples=5000):
        """Run benchmark across multiple seeds"""
        print(f"Running benchmark with {self.config.n_seeds} seeds...")
        
        all_results = {
            'LSTM': [],
            'Transformer': [], 
            'CNN': [],
            'ENN': []
        }
        
        model_configs = {
            'LSTM': (EnhancedLSTM, ModelRegistry.get_lstm_config()),
            'Transformer': (EnhancedTransformer, ModelRegistry.get_transformer_config()),
            'CNN': (EnhancedCNN, ModelRegistry.get_cnn_config()),
            'ENN': (EnhancedENN, ModelRegistry.get_enn_config())
        }
        
        for seed in range(self.config.n_seeds):
            print(f"\nSeed {seed + 1}/{self.config.n_seeds}")
            
            # Set seed for reproducibility
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            # Generate data with current seed
            X_train, y_train, X_test, y_test = task.generate_data(n_samples)
            
            # Split training data for validation
            val_split = int(0.8 * len(X_train))
            X_val, y_val = X_train[val_split:], y_train[val_split:]
            X_train, y_train = X_train[:val_split], y_train[:val_split]
            
            task_type = 'regression' if hasattr(task, '__class__') and 'Prediction' in task.__class__.__name__ else 'classification'
            
            for model_name, (model_class, param_grid) in model_configs.items():
                print(f"  Training {model_name}...")
                
                # Optimize hyperparameters
                best_params = self.optimizer.optimize_hyperparameters(
                    model_class, param_grid, X_train, y_train, X_val, y_val, task_type
                )
                
                # Train final model with best parameters
                model = self.optimizer._create_model(model_class, best_params, X_train.shape, y_train.shape)
                metrics = self._train_and_evaluate_model(
                    model, task, X_train, y_train, X_test, y_test, 
                    best_params.get('lr', 0.001), seed
                )
                
                metrics['seed'] = seed
                metrics['best_params'] = best_params
                all_results[model_name].append(metrics)
        
        return all_results
    
    def _train_and_evaluate_model(self, model, task, X_train, y_train, X_test, y_test, lr, seed):
        """Train and evaluate a single model"""
        # Determine task type and setup
        task_type = 'regression' if hasattr(task, '__class__') and 'Prediction' in task.__class__.__name__ else 'classification'
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.FloatTensor(y_train) if task_type == 'regression' else torch.LongTensor(y_train)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test), 
            torch.FloatTensor(y_test) if task_type == 'regression' else torch.LongTensor(y_test)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss() if task_type == 'regression' else nn.CrossEntropyLoss()
        
        # Training with early stopping
        start_time = time.time()
        best_val_loss = float('inf')
        patience_counter = 0
        
        model.train()
        for epoch in range(self.config.epochs):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                output = model(batch_x)
                
                if task_type == 'classification' and output.size(1) == 1:
                    loss = criterion(output.squeeze(), batch_y.float())
                else:
                    loss = criterion(output, batch_y)
                    
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Early stopping check (simplified)
            val_loss = total_loss / len(train_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if self.config.early_stopping and patience_counter >= self.config.patience:
                break
        
        train_time = time.time() - start_time
        
        # Evaluation
        model.eval()
        start_time = time.time()
        metrics = task.evaluate(model, test_loader)
        inference_time = time.time() - start_time
        
        metrics['train_time'] = train_time
        metrics['inference_time'] = inference_time
        metrics['n_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return metrics
    
    def generate_comprehensive_report(self, all_results, task_name, save_dir='benchmark_results'):
        """Generate comprehensive statistical report"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE STATISTICAL REPORT - {task_name}")
        print(f"{'='*80}")
        
        # Calculate statistics for each model
        report = {
            'task_name': task_name,
            'config': self.config.__dict__,
            'timestamp': datetime.now().isoformat(),
            'models': {}
        }
        
        for model_name, results in all_results.items():
            if not results:
                continue
                
            model_stats = {}
            
            # Extract metrics
            metrics = list(results[0].keys())
            metrics = [m for m in metrics if m not in ['seed', 'best_params'] and isinstance(results[0][m], (int, float))]
            
            for metric in metrics:
                values = [r[metric] for r in results if metric in r]
                if values:
                    mean, ci_lower, ci_upper = self.analyzer.calculate_confidence_interval(values, self.config.confidence_level)
                    model_stats[metric] = {
                        'mean': float(mean),
                        'std': float(np.std(values)),
                        'ci_lower': float(ci_lower),
                        'ci_upper': float(ci_upper),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'n_samples': len(values)
                    }
            
            report['models'][model_name] = model_stats
            
            # Print summary
            print(f"\n{model_name} Results (n={len(results)}):")
            print("-" * 50)
            for metric, stats in model_stats.items():
                if metric in ['accuracy', 'f1_score', 'rmse', 'mae']:
                    print(f"{metric:>15}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                          f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")
        
        # Statistical comparisons
        print(f"\n{'='*60}")
        print("STATISTICAL COMPARISONS")
        print(f"{'='*60}")
        
        models = list(all_results.keys())
        comparisons = {}
        
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models[i+1:], i+1):
                if all_results[model1] and all_results[model2]:
                    for metric in ['accuracy', 'f1_score', 'rmse', 'mae']:
                        if metric in report['models'].get(model1, {}) and metric in report['models'].get(model2, {}):
                            comparison = self.analyzer.compare_models(
                                all_results[model1], all_results[model2], metric
                            )
                            
                            key = f"{model1}_vs_{model2}_{metric}"
                            comparisons[key] = comparison
                            
                            if comparison['significant']:
                                effect_desc = 'small' if comparison['effect_size'] < 0.5 else 'medium' if comparison['effect_size'] < 0.8 else 'large'
                                print(f"{model1} vs {model2} ({metric}): p={comparison['p_value']:.4f} "
                                      f"({'significant' if comparison['significant'] else 'not significant'}, "
                                      f"effect size: {comparison['effect_size']:.3f} - {effect_desc})")
        
        report['statistical_comparisons'] = comparisons
        
        # Generate plots
        if 'accuracy' in report['models'][list(all_results.keys())[0]]:
            self.analyzer.create_performance_plot(
                all_results, 'accuracy', 
                os.path.join(save_dir, f'{task_name}_accuracy.png')
            )
        
        if 'rmse' in report['models'][list(all_results.keys())[0]]:
            self.analyzer.create_performance_plot(
                all_results, 'rmse', 
                os.path.join(save_dir, f'{task_name}_rmse.png')
            )
        
        # Save detailed report
        with open(os.path.join(save_dir, f'{task_name}_detailed_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save raw results
        with open(os.path.join(save_dir, f'{task_name}_raw_results.json'), 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Save configuration
        with open(os.path.join(save_dir, 'benchmark_config.yaml'), 'w') as f:
            yaml.dump(self.config.__dict__, f, default_flow_style=False)
        
        print(f"\n✅ Comprehensive report saved to {save_dir}/")
        
        return report

# Import your existing task classes here
from comprehensive_benchmark_suite import (
    TimeSeriesPredictionTask, 
    AnomalyDetectionTask, 
    SequenceClassificationTask, 
    ReinforcementControlTask
)

def run_enhanced_benchmark():
    """Run the enhanced statistical benchmark"""
    config = BenchmarkConfig(
        n_seeds=20,
        n_trials=50,
        confidence_level=0.95,
        epochs=50,
        early_stopping=True
    )
    
    runner = EnhancedBenchmarkRunner(config)
    
    # Define tasks
    tasks = [
        ("Time_Series_Prediction", TimeSeriesPredictionTask()),
        ("Anomaly_Detection", AnomalyDetectionTask()),
        ("Sequence_Classification", SequenceClassificationTask()),
        ("Reinforcement_Control", ReinforcementControlTask())
    ]
    
    print("=" * 80)
    print("ENHANCED STATISTICAL BENCHMARK SUITE")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Seeds: {config.n_seeds}")
    print(f"  Hyperparameter trials per model: {config.n_trials}")
    print(f"  Confidence level: {config.confidence_level}")
    print(f"  Early stopping: {config.early_stopping}")
    
    all_task_results = {}
    
    for task_name, task in tasks:
        print(f"\n{'='*60}")
        print(f"RUNNING TASK: {task_name}")
        print(f"{'='*60}")
        
        results = runner.run_benchmark_with_seeds(task)
        report = runner.generate_comprehensive_report(results, task_name)
        all_task_results[task_name] = report
    
    # Generate overall summary
    print(f"\n{'='*80}")
    print("OVERALL BENCHMARK SUMMARY")
    print(f"{'='*80}")
    
    for task_name, report in all_task_results.items():
        print(f"\n{task_name}:")
        best_model = None
        best_score = float('inf') if 'rmse' in str(report) else 0
        
        for model_name, stats in report['models'].items():
            primary_metric = 'accuracy' if 'accuracy' in stats else 'rmse'
            score = stats[primary_metric]['mean']
            
            if primary_metric == 'rmse':
                if score < best_score:
                    best_score = score
                    best_model = model_name
            else:
                if score > best_score:
                    best_score = score
                    best_model = model_name
        
        print(f"  Best model: {best_model} ({primary_metric}: {best_score:.4f})")
    
    print(f"\n✅ Enhanced benchmark complete!")
    print("📊 Results include:")
    print("  • Statistical significance tests")
    print("  • 95% confidence intervals") 
    print("  • Fair hyperparameter optimization")
    print("  • Multiple random seeds")
    print("  • Effect size calculations")

if __name__ == "__main__":
    run_enhanced_benchmark()