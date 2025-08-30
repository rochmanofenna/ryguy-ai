#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'BICEP'))

import torch
from bicep_core import BICEPCore, StreamingBICEP, BICEPConfig

# Test BICEP integration
print("Testing BICEP integration...")

# Create config
config = BICEPConfig(
    device='cpu',
    use_half_precision=False,
    use_memory_pool=True,
    max_paths=1000,
    max_steps=100
)

print(f"Config: device={config.device}, max_paths={config.max_paths}")

try:
    # Create BICEP core
    bicep_core = BICEPCore(config)
    print("✓ BICEP Core created successfully")

    # Generate some paths
    n_paths = 10
    n_steps = 50
    paths = bicep_core.generate_paths(n_paths, n_steps)
    print(f"✓ Generated {n_paths} paths with {n_steps} steps each")
    print(f"Paths shape: {paths.shape}")
    print(f"Sample path values: {paths[0, :5].tolist()}")

    # Test streaming version
    streaming_bicep = StreamingBICEP(config)
    print("✓ StreamingBICEP created successfully")
    
    total_generated = 0
    for batch in streaming_bicep.stream_generate(50, n_steps):
        total_generated += batch.shape[0]
        if total_generated >= 20:  # Just test a small amount
            break
    print(f"✓ Streaming generated {total_generated} paths")
    
    print("✓ BICEP integration test successful!")

except Exception as e:
    print(f"✗ BICEP integration test failed: {e}")
    import traceback
    traceback.print_exc()