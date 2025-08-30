#!/usr/bin/env python3
"""
Quick Performance Test - Core Metrics Only
"""

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
import time

# Simple models for comparison
class SimpleENN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, output_size) for _ in range(num_heads)
        ])
    
    def forward(self, x):
        features = self.encoder(x)
        outputs = [head(features) for head in self.heads]
        return torch.stack(outputs).mean(dim=0)

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def generate_test_data(n_samples=1000):
    """Generate simple classification task"""
    X = np.random.randn(n_samples, 20)
    # Create pattern: positive if sum of first 10 features > sum of last 10
    y = (X[:, :10].sum(axis=1) > X[:, 10:].sum(axis=1)).astype(int)
    return X.astype(np.float32), y

def train_and_evaluate(model, X_train, y_train, X_test, y_test, epochs=20):
    """Quick training and evaluation"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)
    
    # Training
    model.train()
    start_time = time.time()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
    
    train_time = time.time() - start_time
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        pred = torch.argmax(outputs, dim=1)
        accuracy = (pred == y_test_t).float().mean().item()
    
    return accuracy, train_time

def run_statistical_comparison():
    """Run statistical comparison with multiple seeds"""
    
    print("=" * 60)
    print("ENHANCED STATISTICAL PERFORMANCE TEST")
    print("=" * 60)
    
    n_seeds = 10
    models = {
        'SimpleENN': lambda: SimpleENN(20, 64, 2, num_heads=5),
        'SimpleLSTM': lambda: SimpleLSTM(20, 64, 2),
        'Baseline_MLP': lambda: nn.Sequential(
            nn.Linear(20, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 2)
        )
    }
    
    results = {name: {'accuracies': [], 'times': []} for name in models.keys()}
    
    for seed in range(n_seeds):
        print(f"\nSeed {seed + 1}/{n_seeds}")
        
        # Set seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Generate data
        X, y = generate_test_data(800)
        split = 600
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Test each model
        for model_name, model_fn in models.items():
            model = model_fn()
            accuracy, train_time = train_and_evaluate(
                model, X_train, y_train, X_test, y_test
            )
            
            results[model_name]['accuracies'].append(accuracy)
            results[model_name]['times'].append(train_time)
    
    # Statistical analysis
    print(f"\n{'='*60}")
    print("STATISTICAL RESULTS")
    print(f"{'='*60}")
    
    for model_name, data in results.items():
        accuracies = data['accuracies']
        times = data['times']
        
        # Calculate statistics
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        # 95% confidence interval
        n = len(accuracies)
        se = std_acc / np.sqrt(n)
        ci_margin = 1.96 * se
        
        print(f"\n{model_name}:")
        print(f"  Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"  95% CI: [{mean_acc - ci_margin:.4f}, {mean_acc + ci_margin:.4f}]")
        print(f"  Training Time: {np.mean(times):.3f}s ± {np.std(times):.3f}s")
        print(f"  Parameters: {sum(p.numel() for p in models[model_name]().parameters()):,}")
    
    # Statistical significance testing
    print(f"\n{'='*50}")
    print("STATISTICAL SIGNIFICANCE TESTS")
    print(f"{'='*50}")
    
    model_names = list(results.keys())
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names[i+1:], i+1):
            acc1 = results[model1]['accuracies']
            acc2 = results[model2]['accuracies']
            
            # Welch's t-test
            t_stat, p_value = stats.ttest_ind(acc1, acc2, equal_var=False)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(acc1) + np.var(acc2)) / 2)
            cohens_d = abs(np.mean(acc1) - np.mean(acc2)) / pooled_std if pooled_std > 0 else 0
            
            significance = "significant" if p_value < 0.05 else "not significant"
            effect_size = "small" if cohens_d < 0.5 else "medium" if cohens_d < 0.8 else "large"
            
            print(f"{model1} vs {model2}:")
            print(f"  p-value: {p_value:.4f} ({significance})")
            print(f"  Effect size: {cohens_d:.3f} ({effect_size})")
            print()
    
    # Performance ranking
    print(f"{'='*50}")
    print("PERFORMANCE RANKING")
    print(f"{'='*50}")
    
    ranking = sorted(results.items(), 
                    key=lambda x: np.mean(x[1]['accuracies']), 
                    reverse=True)
    
    for i, (model_name, data) in enumerate(ranking, 1):
        mean_acc = np.mean(data['accuracies'])
        print(f"{i}. {model_name}: {mean_acc:.4f}")
    
    return results

if __name__ == "__main__":
    results = run_statistical_comparison()
    
    print(f"\n{'='*60}")
    print("KEY FINDINGS")
    print(f"{'='*60}")
    print("✅ Enhanced statistical testing implemented")
    print("✅ Multiple seeds provide reliable estimates")
    print("✅ Confidence intervals show result precision") 
    print("✅ Statistical significance testing validates claims")
    print("✅ Effect sizes quantify practical importance")