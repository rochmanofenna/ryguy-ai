"""
Entangled Neural Networks (ENN) - Core Module

A neural architecture featuring entangled neuron dynamics, multi-head attention 
mechanisms, and adaptive sparsity control for sequence modeling tasks.
"""

from .model import ENNModelWithSparsityControl
from .enhanced_model import create_attention_enn
from .config import Config

__all__ = [
    'ENNModelWithSparsityControl',
    'create_attention_enn',
    'Config'
]

__version__ = "1.0.0"