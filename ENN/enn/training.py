"""
Training utilities for ENN models.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from typing import Dict, Any, Optional
from enn.utils import context_aware_initialization


def create_optimizer(model: nn.Module, config) -> torch.optim.Optimizer:
    """Create optimizer with model parameters."""
    return AdamW(
        model.parameters(),
        lr=getattr(config, 'learning_rate', 0.001),
        weight_decay=getattr(config, 'weight_decay', 1e-5)
    )


def initialize_model_weights(model: nn.Module, entanglement_matrix, config):
    """Initialize model weights with context-aware entanglement patterns."""
    for name, param in model.named_parameters():
        if param.requires_grad:
            shape = param.data.shape
            if len(shape) == 2 and shape == (config.num_neurons, config.num_states):
                param.data = context_aware_initialization(
                    config.num_neurons, 
                    config.num_states, 
                    entanglement_matrix, 
                    method=getattr(config, 'init_method', 'xavier')
                )
            else:
                nn.init.xavier_uniform_(param.data)


def train_epoch(model: nn.Module, data_loader, optimizer: torch.optim.Optimizer, 
                criterion: nn.Module, device: str = 'cpu') -> Dict[str, float]:
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Ensure output and target shapes match
        if outputs.dim() > targets.dim():
            outputs = outputs.view(outputs.size(0), -1)
        if targets.dim() == 1:
            targets = targets.view(-1, 1)
            
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return {'loss': total_loss / num_batches}


def validate_model(model: nn.Module, data_loader, criterion: nn.Module, 
                  device: str = 'cpu') -> Dict[str, float]:
    """Validate model performance."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Ensure output and target shapes match
            if outputs.dim() > targets.dim():
                outputs = outputs.view(outputs.size(0), -1)
            if targets.dim() == 1:
                targets = targets.view(-1, 1)
                
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            num_batches += 1
    
    return {'val_loss': total_loss / num_batches}


def train_model(model: nn.Module, train_loader, val_loader, config, 
                device: str = 'cpu') -> Dict[str, Any]:
    """Complete training loop for ENN model."""
    optimizer = create_optimizer(model, config)
    criterion = nn.MSELoss()
    
    # Initialize weights if entanglement matrix provided
    if hasattr(config, 'entanglement_matrix') and config.entanglement_matrix is not None:
        initialize_model_weights(model, config.entanglement_matrix, config)
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    for epoch in range(getattr(config, 'epochs', 50)):
        # Training
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validation
        val_metrics = validate_model(model, val_loader, criterion, device)
        
        # Record metrics
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['val_loss'])
        
        # Track best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
        
        # Print progress
        if epoch % 10 == 0 or epoch == getattr(config, 'epochs', 50) - 1:
            print(f"Epoch {epoch}: Train Loss = {train_metrics['loss']:.6f}, "
                  f"Val Loss = {val_metrics['val_loss']:.6f}")
    
    return {
        'history': history,
        'best_val_loss': best_val_loss,
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1]
    }