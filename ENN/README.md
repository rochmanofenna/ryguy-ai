# Entangled Neural Networks (ENN)
## General Sequence Modeling Architecture

Beyond trading! Novel neural architecture for any temporal/sequential data:
- **NLP**: Text generation, dialogue systems, sentiment analysis
- **Audio**: Speech recognition, music generation, sound classification  
- **Video**: Action recognition, temporal event detection
- **IoT/Sensors**: Time-series analysis, anomaly detection, predictive maintenance
- **Science**: Climate modeling, seismic analysis, biological rhythms

A neural architecture featuring entangled neuron dynamics, multi-head attention mechanisms, and adaptive sparsity control for sequence modeling tasks.

## Installation

```bash
pip install torch numpy matplotlib pandas seaborn
git clone [repository]
cd ENN
export PYTHONPATH=$(pwd):$PYTHONPATH
```

## Architecture

ENN implements a novel approach to sequence modeling through entangled neuron dynamics, where neurons share state evolution patterns. The architecture incorporates:

- Entangled neuron dynamics with shared state evolution
- Multi-head attention mechanisms for neuron-state and temporal processing
- Adaptive sparsity control with dynamic pruning
- Memory architecture with short-term buffers and temporal proximity scaling

## Usage

```python
from enn.enhanced_model import create_attention_enn
from enn.config import Config

# Example: Audio feature analysis
config = Config()
config.input_dim = 13  # MFCC features
model = create_attention_enn(config, 'full')
audio_output = model(torch.randn(32, 100, 13))  # [batch, time_frames, mfcc_features]

# Example: Text sequence processing  
config.input_dim = 768  # BERT embeddings
text_model = create_attention_enn(config, 'minimal')
text_output = text_model(torch.randn(16, 50, 768))  # [batch, tokens, embeddings]

# Example: IoT sensor data
config.input_dim = 6  # Multi-sensor readings
sensor_model = create_attention_enn(config, 'neuron_only')
sensor_output = sensor_model(torch.randn(64, 200, 6))  # [batch, timesteps, sensors]
```

### Architecture Variants

```python
# Lightweight attention (5K parameters)
model = create_attention_enn(config, 'minimal')

# Neuron-focused attention (18K parameters)  
model = create_attention_enn(config, 'neuron_only')

# Full attention mechanisms (148K parameters)
model = create_attention_enn(config, 'full')
```

## Configuration

```python
config = Config()
# Architecture parameters
config.num_layers = 3        # Processing layers
config.num_neurons = 10      # Entangled neurons
config.num_states = 5        # States per neuron
config.input_dim = 128       # Input feature dimension (adapt to your data)

# Memory dynamics
config.decay_rate = 0.1      # Memory decay rate
config.recency_factor = 0.9  # Temporal weighting
config.buffer_size = 5       # Short-term memory

# Training parameters
config.epochs = 114          # Training epochs
config.batch_size = 32       # Batch size
```

## Performance

| Model | Parameters | Validation Loss | Relative Performance |
|-------|------------|-----------------|---------------------|
| ENN Original | 431 | 0.000016 | 1.0x |
| ENN + Attention | 148,152 | 0.000066 | 0.24x |
| Transformer | 56,881 | 0.000710 | 0.023x |
| LSTM | 30,193 | 0.001869 | 0.009x |
| CNN | 11,065 | 0.020181 | 0.0008x |

Benchmarked on synthetic sequence modeling dataset (1000 samples, 20 timesteps, 3 features).

## Benchmarking

```bash
# Quick benchmark
python benchmarks/run_comprehensive_benchmark.py --quick

# Full benchmark suite
python benchmarks/run_comprehensive_benchmark.py --epochs 114 --runs 3
```

## Project Structure

```
enn/
├── model.py              # Core ENN architecture
├── enhanced_model.py     # ENN with attention mechanisms
├── multihead_attention.py # Attention implementations
├── memory.py             # Memory and buffer systems
├── config.py             # Configuration management
└── training.py           # Training utilities

baselines/
└── baseline_models.py    # Comparison models (LSTM, Transformer, CNN)

benchmarks/
├── benchmark_framework.py # Evaluation framework
└── run_comprehensive_benchmark.py # Benchmark runner

examples/
└── usage.py              # Usage examples

tests/
└── test_*.py            # Unit tests
```

## Technical Details

### Input Formats
- **Temporal sequences**: `[batch, time_steps, features]` - Most common for sequence data
- **Direct neuron format**: `[batch, num_neurons, num_states]` - For specialized applications  
- **Single timestep**: `[batch, features]` - For static or instantaneous predictions

### Application Examples
- **Time Series**: Stock prices, sensor readings, weather data
- **NLP**: Token sequences, sentence embeddings, document analysis
- **Audio**: Spectrograms, MFCC features, raw waveforms
- **Video**: Frame sequences, optical flow, motion patterns
- **Scientific**: Gene sequences, chemical reactions, physical simulations

### Memory Dynamics
- Exponential state decay with configurable rate
- Short-term buffers with FIFO eviction
- Recency-weighted temporal proximity scaling
- State collapse with autoencoder compression

### Attention Architecture
- Neuron-State Attention: Cross-attention between entangled neurons
- Temporal Attention: Memory buffer processing with positional encoding
- Sparse Attention: Top-k attention for computational efficiency
- Attention Pooling: Sequence-to-vector aggregation

## Citation

```bibtex
@misc{enn2024,
  title={Entangled Neural Networks: General Sequence Modeling Architecture},
  author={[Author]},
  year={2024},
  note={Domain-agnostic neural architecture for temporal and sequential data processing}
}
```