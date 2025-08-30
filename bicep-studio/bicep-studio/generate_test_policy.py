#!/usr/bin/env python3
"""
Generate a test ENN policy file for the BICEP Studio demo.
"""

import json
import numpy as np

def generate_test_policy():
    """Generate a 5-head ensemble for navigation with 27 inputs (5x5 grid + 2 pos)."""
    
    np.random.seed(42)  # Reproducible
    
    input_dim = 27  # 5x5 local grid + 2 position features
    hidden_dim = 128
    output_dim = 4  # 4 actions: right, down, left, up
    num_heads = 5
    
    heads = []
    
    for i in range(num_heads):
        # Initialize with small random weights
        W1 = np.random.normal(0, 0.1, (input_dim, hidden_dim))
        b1 = np.random.normal(0, 0.01, hidden_dim)
        W2 = np.random.normal(0, 0.1, (hidden_dim, output_dim))
        b2 = np.random.normal(0, 0.01, output_dim)
        
        # Add some bias toward right/down movement (positive gradient)
        b2[0] += 0.1  # slight bias toward right
        b2[1] += 0.05  # slight bias toward down
        
        heads.append({
            "W1": W1.tolist(),
            "b1": b1.tolist(), 
            "W2": W2.tolist(),
            "b2": b2.tolist()
        })
    
    policy_data = {
        "heads": heads,
        "metadata": {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "num_heads": num_heads,
            "description": "Test ensemble policy for BICEP Studio navigation demo"
        }
    }
    
    # Save to JSON file
    with open("test_policy.json", "w") as f:
        json.dump(policy_data, f, indent=2)
    
    print(f"Generated test policy: {num_heads} heads, {input_dim}→{hidden_dim}→{output_dim}")
    print("Saved as test_policy.json")

if __name__ == "__main__":
    generate_test_policy()