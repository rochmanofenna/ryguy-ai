#!/usr/bin/env python3
"""
Simplified benchmark comparison between LSTM, Transformer, and basic MLP
"""

import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class SimpleTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nhead=4):
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead, batch_first=True
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

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        if x.dim() == 3:
            x = x[:, -1, :]  # Use last timestep
        return self.net(x)

def create_data(n_samples=1000, seq_len=20, input_dim=5):
    """Create simple synthetic data"""
    X = torch.randn(n_samples, seq_len, input_dim)
    y = X[:, -1, :].sum(dim=1, keepdim=True) + 0.1 * torch.randn(n_samples, 1)
    return X, y

def train_and_evaluate(model, train_loader, test_loader, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Training
    start_time = time.time()
    train_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))
    
    train_time = time.time() - start_time
    
    # Evaluation
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output = model(X_batch)
            loss = criterion(output, y_batch)
            test_loss += loss.item()
    
    test_loss /= len(test_loader)
    
    return {
        'train_time': train_time,
        'final_train_loss': train_losses[-1],
        'test_loss': test_loss,
        'convergence_epoch': np.argmin(train_losses) + 1
    }

def main():
    print("=" * 60)
    print("BASELINE MODEL COMPARISON")
    print("=" * 60)
    
    # Parameters
    input_dim = 5
    hidden_dim = 64
    output_dim = 1
    batch_size = 32
    epochs = 30
    
    # Create data
    print("\nGenerating data...")
    X_train, y_train = create_data(2000, 20, input_dim)
    X_test, y_test = create_data(500, 20, input_dim)
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Models
    models = {
        'MLP': SimpleMLP(input_dim, hidden_dim, output_dim),
        'LSTM': SimpleLSTM(input_dim, hidden_dim, output_dim),
        'Transformer': SimpleTransformer(input_dim, hidden_dim, output_dim)
    }
    
    # Train and evaluate
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}")
        
        results[name] = train_and_evaluate(model, train_loader, test_loader, epochs)
        results[name]['parameters'] = n_params
        
        print(f"Test Loss: {results[name]['test_loss']:.4f}")
        print(f"Train Time: {results[name]['train_time']:.2f}s")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<15} {'Parameters':<12} {'Test Loss':<12} {'Train Time':<12} {'Speed Rank'}")
    print("-" * 70)
    
    # Sort by test loss
    sorted_results = sorted(results.items(), key=lambda x: x[1]['test_loss'])
    train_times = sorted([(k, v['train_time']) for k, v in results.items()], key=lambda x: x[1])
    speed_ranks = {k: i+1 for i, (k, _) in enumerate(train_times)}
    
    for name, res in sorted_results:
        print(f"{name:<15} {res['parameters']:<12,} {res['test_loss']:<12.4f} "
              f"{res['train_time']:<12.2f} {speed_ranks[name]}")
    
    # Performance comparison
    print("\n" + "=" * 70)
    print("RELATIVE PERFORMANCE")
    print("=" * 70)
    
    baseline_name = 'LSTM'
    baseline_loss = results[baseline_name]['test_loss']
    baseline_time = results[baseline_name]['train_time']
    
    for name, res in sorted_results:
        loss_ratio = res['test_loss'] / baseline_loss
        time_ratio = res['train_time'] / baseline_time
        
        print(f"{name:<15} Loss: {loss_ratio:.2f}x LSTM   "
              f"Speed: {1/time_ratio:.2f}x faster than LSTM")

if __name__ == "__main__":
    main()