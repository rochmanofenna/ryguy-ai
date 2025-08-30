# BICEP: Brownian Compute Engine for Paths

A high-performance stochastic path generation library optimized for GPU acceleration, designed for scientific computing, research applications, and any domain requiring random walk or diffusion process simulation.

## Overview

BICEP provides state-of-the-art implementations for generating Brownian motion paths with minimal latency and maximum throughput. The library supports multiple hardware backends including NVIDIA CUDA, Apple Metal Performance Shaders (MPS), and CPU, with automatic optimization for each platform.

### Key Features

- **Sub-millisecond latency**: Single path generation in under 0.4ms on modern GPUs
- **High throughput**: Over 2.5 million paths per second on consumer hardware
- **Multiple backends**: Optimized implementations for CUDA, Metal, and CPU
- **Memory efficient**: Advanced memory pooling and streaming capabilities
- **Neural network integration**: Seamless integration with PyTorch models
- **Precision options**: Support for both FP16 and FP32 computation
- **Domain agnostic**: Applicable to physics, biology, engineering, ML, and more

## Application Domains

### Physics & Chemistry
- **Particle diffusion**: Model Brownian motion of particles in fluids
- **Quantum walks**: Simulate quantum random walks and decoherence
- **Thermal dynamics**: Study heat diffusion and energy transfer
- **Molecular dynamics**: Track molecular movement and interactions

### Biology & Medicine
- **Cell migration**: Model random walk behavior of cells
- **Genetic drift**: Simulate neutral evolution in populations
- **Protein folding**: Study conformational changes over time
- **Drug diffusion**: Analyze pharmaceutical dispersion in tissue

### Engineering & Signal Processing
- **Noise modeling**: Generate realistic noise for signal analysis
- **Reliability analysis**: Monte Carlo methods for failure prediction
- **Random vibrations**: Study stochastic mechanical systems
- **Network traffic**: Model packet arrival and congestion

### Machine Learning & AI
- **SGD visualization**: Visualize stochastic gradient descent paths
- **Uncertainty quantification**: Generate ensemble predictions
- **Monte Carlo methods**: Efficient sampling for Bayesian inference
- **Diffusion models**: Foundation for generative AI applications

## Performance Benchmarks

Performance metrics on Apple M3 (Metal Performance Shaders):

| Batch Size | Time per Path | Throughput |
|------------|---------------|------------|
| 1          | 0.390 ms      | 2,564 paths/sec |
| 100        | 0.009 ms      | 111,111 paths/sec |
| 1,000      | 0.004 ms      | 250,000 paths/sec |
| 10,000     | 0.0004 ms     | 2,500,000 paths/sec |

Projected performance on NVIDIA A100: ~0.036 ms per path (10.8x faster)
Projected performance on NVIDIA H100: ~0.020 ms per path (20x faster)

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- CUDA toolkit (for NVIDIA GPU support)
- Metal Performance Shaders (for Apple Silicon)

### Install from source

```bash
git clone https://github.com/rochmanofenna/BICEP.git
cd BICEP
pip install -e .
```

### Dependencies

```bash
pip install torch numpy
```

For CUDA support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### Basic Usage

```python
from bicep_core import BICEPCore, BICEPConfig

# Initialize with default configuration
config = BICEPConfig(device='mps')  # or 'cuda' for NVIDIA GPUs
bicep = BICEPCore(config)

# Generate 1000 paths with 1000 time steps
paths = bicep.generate_paths(n_paths=1000, n_steps=1000)
print(f"Generated paths shape: {paths.shape}")  # [1000, 1001]
```

### Domain-Specific Examples

#### Physics: Particle Diffusion
```python
# Simulate diffusion of particles in a medium
config = BICEPConfig(device='cuda')
bicep = BICEPCore(config)

# Parameters for particle diffusion
diffusion_coefficient = 2e-9  # m²/s
time_step = 0.001  # seconds
n_particles = 10000

paths = bicep.generate_paths(
    n_paths=n_particles,
    n_steps=1000,
    T=1.0,  # Total time
    control_parameter=0.0,  # No external force
    time_decay=0.0  # Constant diffusion
)

# Convert to physical units
positions = paths * np.sqrt(2 * diffusion_coefficient * time_step)
```

#### Biology: Cell Migration
```python
# Model random walk of immune cells
config = BICEPConfig(device='mps', use_half_precision=False)
bicep = BICEPCore(config)

# Cell migration parameters
cell_speed = 10  # μm/min
persistence = 0.8  # Directional persistence

paths = bicep.generate_paths(
    n_paths=100,  # Track 100 cells
    n_steps=720,  # 12 hours at 1-minute intervals
    T=720.0,
    control_parameter=persistence,  # Persistence in direction
    time_decay=0.05  # Gradual randomization
)

# Scale to cell movement
trajectories = paths * cell_speed
```

#### Engineering: Signal Noise
```python
# Generate noise for signal processing
config = BICEPConfig(device='cpu')
bicep = BICEPCore(config)

# Noise characteristics
sampling_rate = 44100  # Hz
duration = 10  # seconds

noise_paths = bicep.generate_paths(
    n_paths=2,  # Stereo channels
    n_steps=sampling_rate * duration,
    T=duration,
    control_parameter=0.5,  # Colored noise
    time_decay=0.001  # Frequency-dependent decay
)

# Apply to signal processing pipeline
noisy_signal = clean_signal + 0.1 * noise_paths
```

#### ML/AI: Uncertainty Quantification
```python
from bicep_core import NeuralBICEPLayer
import torch.nn as nn

class UncertaintyAwareNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        # Add stochastic layer for uncertainty
        self.stochastic = NeuralBICEPLayer(
            hidden_dim, hidden_dim, 
            n_steps=50,  # Monte Carlo samples
            control_parameter=0.3  # Uncertainty level
        )
        self.decoder = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, n_samples=10):
        h = torch.relu(self.encoder(x))
        # Generate multiple stochastic samples
        predictions = []
        for _ in range(n_samples):
            h_uncertain = self.stochastic(h)
            pred = self.decoder(h + h_uncertain)
            predictions.append(pred)
        return torch.stack(predictions)

# Use for uncertainty estimation
model = UncertaintyAwareNN(64, 128, 10)
predictions = model(input_data, n_samples=100)
mean_pred = predictions.mean(dim=0)
uncertainty = predictions.std(dim=0)
```

### Advanced Configuration

```python
# Custom configuration for optimal performance
config = BICEPConfig(
    device='cuda',
    max_paths=100000,
    max_steps=4000,
    use_half_precision=True,  # Use FP16 for faster computation
    use_memory_pool=True,     # Enable memory pooling
    tile_size=32              # Optimize for GPU warp size
)

bicep = BICEPCore(config)

# Generate paths with stochastic controls
paths = bicep.generate_paths(
    n_paths=10000,
    n_steps=1000,
    T=1.0,                    # Time horizon
    control_parameter=0.7,    # Control strength [0, 1]
    time_decay=0.1           # Temporal decay rate
)
```

### Streaming Generation

For generating large numbers of paths with limited memory:

```python
from bicep_core import StreamingBICEP

streaming = StreamingBICEP(config, buffer_size=20000)

# Generate 1 million paths in chunks
total_generated = 0
for chunk in streaming.stream_generate(total_paths=1000000, n_steps=1000):
    # Process each chunk
    process_paths(chunk)
    total_generated += chunk.shape[0]
```

## API Reference

### BICEPConfig

Configuration dataclass for BICEP engine:

- `device`: Computing device ('cuda', 'mps', 'cpu')
- `max_paths`: Maximum number of paths for memory pre-allocation
- `max_steps`: Maximum time steps per path
- `tile_size`: Memory alignment size for optimal GPU access
- `use_half_precision`: Enable FP16 computation
- `use_memory_pool`: Enable memory pooling
- `warmup_iterations`: Number of warmup iterations

### BICEPCore

Core path generation class:

#### Methods

- `generate_paths(n_paths, n_steps, T=1.0, control_parameter=0.5, time_decay=0.1)`: Generate Brownian motion paths
  - `n_paths`: Number of paths to generate
  - `n_steps`: Number of time steps
  - `T`: Time horizon
  - `control_parameter`: Process control parameter [0, 1]
  - `time_decay`: Temporal decay for time-dependent behavior

### StreamingBICEP

Streaming interface for large-scale generation:

#### Methods

- `stream_generate(total_paths, n_steps, **kwargs)`: Generate paths in streaming fashion
  - Yields chunks of paths for memory-efficient processing

### NeuralBICEPLayer

PyTorch module for neural network integration:

#### Parameters

- `input_size`: Input dimension
- `output_size`: Output dimension  
- `n_steps`: Number of time steps for path generation
- `control_parameter`: Default control parameter
- `config`: Optional BICEPConfig instance

## Architecture

BICEP employs several optimization techniques:

1. **Memory Pooling**: Pre-allocated memory pools eliminate allocation overhead
2. **Precision Optimization**: Optional FP16 computation for improved throughput
3. **Vectorized Operations**: Fully vectorized Box-Muller transformation
4. **Tiled Memory Access**: Aligned memory access patterns for GPU efficiency
5. **Kernel Fusion**: Combined operations to minimize memory bandwidth usage

## Benchmarking

Run the included benchmark suite:

```bash
python bicep_core.py
```

For domain-specific benchmarks:
```bash
python benchmarks/bench_physics.py     # Physics simulations
python benchmarks/bench_biology.py     # Biological systems
python benchmarks/bench_engineering.py # Engineering applications
python benchmarks/bench_ml.py          # ML/AI use cases
```

## Research Applications

BICEP has been designed to support cutting-edge research across multiple domains:

- **Climate Science**: Stochastic weather models and uncertainty propagation
- **Neuroscience**: Neural spike train generation and synaptic noise
- **Ecology**: Population dynamics and species dispersal
- **Materials Science**: Defect migration and grain boundary motion
- **Quantum Computing**: Decoherence modeling and error simulation

## Contributing

Contributions are welcome. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

### Development Setup

```bash
# Clone repository
git clone https://github.com/rochmanofenna/BICEP.git
cd BICEP

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/
```

## Citation

If you use BICEP in your research, please cite:

```bibtex
@software{bicep2025,
  title = {BICEP: Brownian Compute Engine for Paths},
  author = {Rochman, Ryan},
  year = {2025},
  url = {https://github.com/rochmanofenna/BICEP}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

- Author: Ryan Rochman
- Email: [Contact via GitHub]
- GitHub: [@rochmanofenna](https://github.com/rochmanofenna)

## Acknowledgments

This work leverages optimizations for modern GPU architectures including NVIDIA CUDA and Apple Metal Performance Shaders. Special thanks to the PyTorch team for their excellent GPU abstraction layer.