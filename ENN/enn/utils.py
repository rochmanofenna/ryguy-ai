"""
Core utility functions for ENN operations.

Contains essential functions for data complexity analysis, neuron activation,
weight sharing, and parameter initialization.
"""

import torch
from typing import Union


def compute_data_complexity(x: torch.Tensor) -> torch.Tensor:
    """
    Compute complexity score for input data.
    
    Uses variance-based approach for better sensitivity to data distribution.
    
    Args:
        x: Input tensor
        
    Returns:
        Complexity score clamped to [0, 1]
    """
    complexity_score = torch.var(x)
    return torch.clamp(complexity_score, 0, 1)


def dynamic_scaling(complexity_score: torch.Tensor, max_states: int = 5) -> int:
    """
    Map complexity score to number of active states.
    
    Args:
        complexity_score: Complexity score tensor [0, 1]
        max_states: Maximum number of states to activate
        
    Returns:
        Number of active states [1, max_states]
    """
    c = float(torch.clamp(complexity_score, 0.0, 1.0).item())
    k = int(c * max_states)
    return max(1, min(max_states, k))


def activate_neuron(neuron_state: torch.Tensor, data_input: torch.Tensor) -> torch.Tensor:
    """
    Activate neuron based on input data complexity.
    
    Args:
        neuron_state: Current neuron state
        data_input: Input data for activation
        
    Returns:
        Updated neuron state
    """
    complexity_score = compute_data_complexity(data_input)
    active_states = dynamic_scaling(complexity_score)
    neuron_state[:active_states] = torch.sigmoid(data_input[:active_states])
    return neuron_state


def dynamic_weight_sharing(neuron_activations: torch.Tensor, weights: torch.Tensor, 
                          attention_threshold: float = 0.6) -> torch.Tensor:
    """
    Apply dynamic weight sharing based on attention scores.
    
    Args:
        neuron_activations: Neuron activation patterns
        weights: Weight matrix to modify
        attention_threshold: Threshold for attention-based sharing
        
    Returns:
        Modified weights with attention-based sharing
    """
    attention_scores = torch.matmul(neuron_activations, neuron_activations.T) / neuron_activations.size(-1) ** 0.5
    attention_mask = (attention_scores > attention_threshold).float()[:, :weights.size(1)]
    shared_weights = weights * attention_mask
    return shared_weights


def context_aware_initialization(num_neurons: int, num_states: int, 
                                entanglement_matrix: torch.Tensor, 
                                method: str = "xavier") -> torch.Tensor:
    """
    Initialize weights with context-aware entanglement patterns.
    
    Args:
        num_neurons: Number of neurons
        num_states: Number of states per neuron
        entanglement_matrix: Matrix defining entanglement patterns
        method: Initialization method ('xavier' or 'he')
        
    Returns:
        Initialized weight tensor
    """
    if method == "xavier":
        scale = torch.sqrt(torch.tensor(2.0) / (num_neurons + num_states))
    elif method == "he":
        scale = torch.sqrt(torch.tensor(2.0) / num_neurons)
    else:
        raise ValueError(f"Unsupported initialization method: {method}")
    
    weights = scale * torch.randn(num_neurons, num_states)
    ent_matrix_t = torch.tensor(entanglement_matrix) if not isinstance(entanglement_matrix, torch.Tensor) else entanglement_matrix
    mask = ent_matrix_t > 0  
    weights[mask] *= ent_matrix_t[mask]
    return weights


def entropy_based_prune(neuron_state: torch.Tensor, importance_threshold: float = 0.1) -> torch.Tensor:
    """
    Prune neuron states based on importance threshold.
    
    Args:
        neuron_state: Current neuron state
        importance_threshold: Minimum importance to retain
        
    Returns:
        Pruned neuron state
    """
    return torch.where(neuron_state > importance_threshold, neuron_state, torch.zeros_like(neuron_state))


def context_collapse(neuron_state: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Collapse neuron state based on entropy analysis.
    
    Args:
        neuron_state: Current neuron state
        threshold: Entropy threshold for collapse
        
    Returns:
        Collapsed or original neuron state
    """
    entropy = -torch.sum(neuron_state * torch.log(neuron_state + 1e-9))
    if entropy < threshold:
        mean_val = torch.mean(neuron_state)
        return mean_val.expand_as(neuron_state)
    return neuron_state