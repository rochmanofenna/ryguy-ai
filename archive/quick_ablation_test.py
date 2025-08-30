#!/usr/bin/env python3
"""
Quick Ablation Test - Component Importance
"""

import numpy as np
import torch
import torch.nn as nn

class BaselineModel(nn.Module):
    def __init__(self, input_size=20, hidden_size=64, output_size=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.net(x)

class ENNStyleModel(nn.Module):
    def __init__(self, input_size=20, hidden_size=64, output_size=2, num_heads=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, output_size) for _ in range(num_heads)
        ])
        self.num_heads = num_heads
    
    def forward(self, x):
        features = self.encoder(x)
        outputs = [head(features) for head in self.heads]
        return torch.stack(outputs).mean(dim=0)

class BICEPStyleModel(nn.Module):
    def __init__(self, input_size=20, hidden_size=64, output_size=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.stochastic_layer = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        features = self.encoder(x)
        
        if self.training:
            # Add stochastic variation during training
            noise = torch.randn_like(features) * 0.1
            stochastic_features = self.stochastic_layer(features + noise)
        else:
            stochastic_features = self.stochastic_layer(features)
        
        return self.output(stochastic_features)

class HybridModel(nn.Module):
    def __init__(self, input_size=20, hidden_size=64, output_size=2):
        super().__init__()
        # BICEP-style processing
        self.bicep_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.stochastic_layer = nn.Linear(hidden_size, hidden_size)
        
        # ENN-style ensemble
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, output_size) for _ in range(3)
        ])
    
    def forward(self, x):
        # BICEP processing
        features = self.bicep_encoder(x)
        if self.training:
            noise = torch.randn_like(features) * 0.05
            features = self.stochastic_layer(features + noise)
        else:
            features = self.stochastic_layer(features)
        
        # ENN ensemble
        outputs = [head(features) for head in self.heads]
        return torch.stack(outputs).mean(dim=0)

def generate_data(n_samples=1000):
    """Generate test data"""
    X = np.random.randn(n_samples, 20)
    # More complex pattern for better differentiation
    y = ((X[:, :5].sum(axis=1) > 0) & (X[:, 10:15].sum(axis=1) < 0)).astype(int)
    return X.astype(np.float32), y

def train_and_evaluate(model, X_train, y_train, X_test, y_test, epochs=25):
    """Train and evaluate model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)
    
    # Training
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        pred = torch.argmax(outputs, dim=1)
        accuracy = (pred == y_test_t).float().mean().item()
        
        # Calculate uncertainty (variance across multiple forward passes)
        uncertainties = []
        for _ in range(10):
            model.train()  # Enable dropout/stochastic elements
            out = model(X_test_t)
            uncertainties.append(torch.softmax(out, dim=1))
        
        model.eval()
        uncertainty_var = torch.stack(uncertainties).var(dim=0).mean().item()
    
    return accuracy, uncertainty_var

def run_ablation_study():
    """Run component ablation study"""
    print("=" * 60)
    print("COMPONENT ABLATION STUDY")
    print("=" * 60)
    
    # Models to test
    models = {
        'Baseline_MLP': BaselineModel,
        'ENN_Ensemble': ENNStyleModel,
        'BICEP_Stochastic': BICEPStyleModel,
        'Hybrid_BICEP_ENN': HybridModel
    }
    
    results = {}
    n_runs = 5
    
    for model_name, model_class in models.items():
        print(f"\nTesting {model_name}...")
        
        accuracies = []
        uncertainties = []
        
        for run in range(n_runs):
            # Set seeds
            np.random.seed(run)
            torch.manual_seed(run)
            
            # Generate data
            X_train, y_train = generate_data(800)
            X_test, y_test = generate_data(200)
            
            # Create and train model
            model = model_class()
            accuracy, uncertainty = train_and_evaluate(
                model, X_train, y_train, X_test, y_test
            )
            
            accuracies.append(accuracy)
            uncertainties.append(uncertainty)
        
        # Statistics
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        mean_unc = np.mean(uncertainties)
        
        results[model_name] = {
            'accuracy_mean': mean_acc,
            'accuracy_std': std_acc,
            'uncertainty_mean': mean_unc,
            'parameter_count': sum(p.numel() for p in model_class().parameters())
        }
        
        print(f"  Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"  Uncertainty: {mean_unc:.4f}")
        print(f"  Parameters: {results[model_name]['parameter_count']:,}")
    
    return results

def analyze_results(results):
    """Analyze ablation results"""
    print(f"\n{'='*60}")
    print("ABLATION ANALYSIS")
    print(f"{'='*60}")
    
    # Performance ranking
    ranked = sorted(results.items(), key=lambda x: x[1]['accuracy_mean'], reverse=True)
    
    print("\nPerformance Ranking:")
    for i, (name, metrics) in enumerate(ranked, 1):
        acc = metrics['accuracy_mean']
        std = metrics['accuracy_std']
        params = metrics['parameter_count']
        print(f"{i}. {name}: {acc:.4f} ± {std:.4f} ({params:,} params)")
    
    # Component contributions
    baseline = results['Baseline_MLP']['accuracy_mean']
    
    print(f"\nComponent Contributions (vs Baseline {baseline:.4f}):")
    
    if 'ENN_Ensemble' in results:
        enn_gain = results['ENN_Ensemble']['accuracy_mean'] - baseline
        print(f"  ENN Ensemble: +{enn_gain:.4f} ({enn_gain/baseline*100:+.1f}%)")
    
    if 'BICEP_Stochastic' in results:
        bicep_gain = results['BICEP_Stochastic']['accuracy_mean'] - baseline
        print(f"  BICEP Stochastic: +{bicep_gain:.4f} ({bicep_gain/baseline*100:+.1f}%)")
    
    if 'Hybrid_BICEP_ENN' in results:
        hybrid_gain = results['Hybrid_BICEP_ENN']['accuracy_mean'] - baseline
        print(f"  Hybrid BICEP+ENN: +{hybrid_gain:.4f} ({hybrid_gain/baseline*100:+.1f}%)")
    
    # Uncertainty analysis
    print(f"\nUncertainty Quantification:")
    for name, metrics in results.items():
        unc = metrics['uncertainty_mean']
        print(f"  {name}: {unc:.4f}")
    
    # Efficiency analysis
    print(f"\nParameter Efficiency:")
    for name, metrics in results.items():
        acc = metrics['accuracy_mean']
        params = metrics['parameter_count']
        efficiency = acc / (params / 1000)  # Accuracy per 1k parameters
        print(f"  {name}: {efficiency:.4f} acc/1k params")

if __name__ == "__main__":
    results = run_ablation_study()
    analyze_results(results)
    
    print(f"\n{'='*60}")
    print("KEY ABLATION INSIGHTS")
    print(f"{'='*60}")
    print("✅ Component contributions quantified")
    print("✅ Ensemble effects measured")
    print("✅ Stochastic processing benefits assessed")
    print("✅ Parameter efficiency compared")
    print("✅ Uncertainty quantification evaluated")