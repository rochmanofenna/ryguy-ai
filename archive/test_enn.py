#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ENN'))

import torch
import torch.nn as nn
from enn.model import ENNModelWithSparsityControl  
from enn.config import Config

# Test ENN integration
print("Testing ENN integration...")

# Create config
config = Config()
config.input_dim = 27
config.num_neurons = 10
config.num_states = 5
config.num_layers = 3

print(f"Config: input_dim={config.input_dim}, num_neurons={config.num_neurons}, num_states={config.num_states}")

# Create ENN model
enn_model = ENNModelWithSparsityControl(config)

# Create test input [batch=2, features=27]
test_input = torch.randn(2, 27)
print(f"Test input shape: {test_input.shape}")

# Test forward pass
try:
    enn_output, p_t, contradiction, diagnostics = enn_model(test_input)
    print(f"ENN output shape: {enn_output.shape}")
    print(f"ENN output content: {enn_output}")
    
    # Need to aggregate across neurons - take mean
    # ENN output is [batch, num_neurons, num_states] -> [batch, num_states]
    aggregated_output = enn_output.mean(dim=1)  # Average across neurons
    print(f"Aggregated output shape: {aggregated_output.shape}")
    
    # Test output layer
    output_layer = nn.Linear(config.num_states, 4)
    final_output = output_layer(aggregated_output)
    print(f"Final output shape: {final_output.shape}")
    print(f"Final output: {final_output}")
    
    # Test with CrossEntropyLoss
    targets = torch.LongTensor([0, 1])  # 2 samples, class indices
    criterion = nn.CrossEntropyLoss()
    loss = criterion(final_output, targets)
    print(f"CrossEntropy loss: {loss.item()}")
    
    print("✓ ENN integration test successful!")

except Exception as e:
    print(f"✗ ENN integration test failed: {e}")
    import traceback
    traceback.print_exc()