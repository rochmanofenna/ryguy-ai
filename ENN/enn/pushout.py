import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class ContextCollapseHead(nn.Module):
    """
    Implements the pushout/context synthesis operation for ENN.
    
    This module takes multi-state neuron tensors and produces:
    1. p_t: A context symbol representing the synthesized state
    2. contradiction_score: A scalar measuring the degree of contradiction
    
    The pushout operation implements categorical theory's pushout concept,
    synthesizing contradictory states into a unified context representation.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_states: int,
        context_dim: int = 128,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Dimension of each neuron (N)
            num_states: Number of states/personas per neuron (K)
            context_dim: Output dimension of p_t context symbol (d)
            hidden_dim: Hidden layer dimension (defaults to 2*context_dim)
            dropout: Dropout probability
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_states = num_states
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim or (2 * context_dim)
        
        # Entropy computation for contradiction scoring
        self.temperature = nn.Parameter(torch.ones(1))
        
        # State attention mechanism - learns which states to emphasize
        self.state_attention = nn.Sequential(
            nn.Linear(num_states, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, num_states),
            nn.Softmax(dim=-1)
        )
        
        # Interference detection network
        self.interference_net = nn.Sequential(
            nn.Linear(num_states, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Context synthesis network (the actual pushout operation)
        # Input: weighted states + attention weights + contradiction score
        synthesis_input_dim = num_states + num_states + 1
        self.synthesis_net = nn.Sequential(
            nn.Linear(synthesis_input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, context_dim)
        )
        
        # Final normalization
        self.output_norm = nn.LayerNorm(context_dim)
        
    def compute_state_entropy(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy across states as a measure of contradiction.
        
        Args:
            states: Tensor of shape (B, N, K) 
        
        Returns:
            entropy: Tensor of shape (B, N)
        """
        # Normalize states to probability distribution
        states_norm = F.softmax(states / self.temperature, dim=-1)
        
        # Compute entropy: -sum(p * log(p))
        log_probs = torch.log(states_norm + 1e-8)
        entropy = -torch.sum(states_norm * log_probs, dim=-1)
        
        # Normalize by log(K) to get value in [0, 1]
        entropy = entropy / torch.log(torch.tensor(self.num_states, dtype=torch.float32))
        
        return entropy
    
    def compute_interference_score(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute interference pattern between states.
        
        Args:
            states: Tensor of shape (B, N, K)
            
        Returns:
            interference: Tensor of shape (B, N)
        """
        B, N, K = states.shape
        
        # Process each neuron's states independently
        # Reshape to (B*N, K) for batch processing
        states_reshaped = states.reshape(B * N, K)
        
        # Compute interference score for each neuron
        interference = self.interference_net(states_reshaped).squeeze(-1)  # (B*N,)
        
        # Reshape back to (B, N)
        interference = interference.view(B, N)
        
        return interference
    
    def forward(
        self, 
        states: torch.Tensor,
        return_diagnostics: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass implementing the pushout operation.
        
        Args:
            states: Multi-state neuron tensor of shape (B, N, K)
                    B = batch size, N = num neurons, K = num states
            return_diagnostics: Whether to return diagnostic information
            
        Returns:
            p_t: Context symbol tensor of shape (B, N, context_dim)
            contradiction_score: Scalar contradiction measure of shape (B, N)
            diagnostics: Optional dict with entropy, interference, attention weights
        """
        B, N, K = states.shape
        device = states.device
        
        # Compute entropy-based contradiction score
        entropy = self.compute_state_entropy(states)
        
        # Compute interference patterns
        interference = self.compute_interference_score(states)
        
        # Combined contradiction score
        contradiction_score = (entropy + interference) / 2.0
        
        # Compute state attention weights per neuron
        # Process each neuron independently: (B*N, K) -> (B*N, K)
        states_reshaped = states.reshape(B * N, K)
        attention_weights = self.state_attention(states_reshaped)  # (B*N, K)
        attention_weights = attention_weights.view(B, N, K)  # (B, N, K)
        
        # Apply attention-weighted combination
        states_weighted = (states * attention_weights).sum(dim=-1)  # (B, N)
        
        # Prepare synthesis input for each neuron
        # We'll process each neuron independently again
        synthesis_inputs = []
        for b in range(B):
            for n in range(N):
                # For each neuron, concatenate:
                # 1. attention-weighted state (scalar -> expanded to num_states)
                # 2. attention weights
                # 3. contradiction score
                neuron_input = torch.cat([
                    states_weighted[b, n].repeat(K),  # Expand scalar to num_states
                    attention_weights[b, n],  # (K,)
                    contradiction_score[b, n].unsqueeze(0)  # (1,)
                ])  # Total: K + K + 1 = 2*K + 1
                synthesis_inputs.append(neuron_input)
        
        synthesis_input = torch.stack(synthesis_inputs).view(B, N, -1)  # (B, N, 2*K+1)
        
        # Synthesize context symbol p_t
        p_t = self.synthesis_net(synthesis_input)
        p_t = self.output_norm(p_t)
        
        # Prepare diagnostics if requested
        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                'entropy': entropy,
                'interference': interference,
                'attention_weights': attention_weights,
                'states_norm': F.normalize(states, p=2, dim=-1)
            }
        
        return p_t, contradiction_score, diagnostics


class DeterministicContextHead(ContextCollapseHead):
    """
    A deterministic version of the context collapse head for testing.
    Uses fixed operations instead of learned parameters.
    """
    
    def __init__(self, input_dim: int, num_states: int, context_dim: int = 128):
        super().__init__(input_dim, num_states, context_dim)
        
        # Override with deterministic operations
        self.use_deterministic = True
        
    def forward(
        self,
        states: torch.Tensor,
        return_diagnostics: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Deterministic forward pass for testing.
        """
        B, N, K = states.shape
        
        # Simple mean pooling for p_t
        p_t_simple = states.mean(dim=2)  # (B, N)
        
        # Expand to context dimension by repeating
        p_t = p_t_simple.unsqueeze(-1).expand(B, N, self.context_dim)
        
        # Simple variance as contradiction score
        contradiction_score = states.var(dim=2)
        
        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                'entropy': torch.zeros_like(contradiction_score),
                'interference': torch.zeros_like(contradiction_score),
                'attention_weights': torch.ones(B, N, K) / K,
                'states_norm': F.normalize(states, p=2, dim=-1)
            }
        
        return p_t, contradiction_score, diagnostics