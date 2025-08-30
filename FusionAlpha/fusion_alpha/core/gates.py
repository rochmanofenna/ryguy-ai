#!/usr/bin/env python3
"""
Pluggable Gates System

Four gate types for routing decisions:
- rule: threshold-based routing (fast baseline)
- mlp: learned 2-layer MLP on [a,b,Δ,aux]
- bandit: contextual bandit (Thompson/UCB) for online regime selection
- rl: PPO/SAC for long-horizon choices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

@dataclass
class GateDecision:
    """Decision output from a gate"""
    expert_choice: str
    confidence: float
    routing_scores: Dict[str, float]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseGate(ABC):
    """Base class for routing gates"""
    
    def __init__(self, expert_names: List[str]):
        self.expert_names = expert_names
        self.num_experts = len(expert_names)
        
    @abstractmethod
    def route(self, features: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> GateDecision:
        """Route input to expert"""
        pass
    
    @abstractmethod
    def update(self, decision: GateDecision, reward: float, context: Optional[Dict[str, Any]] = None):
        """Update gate based on feedback"""
        pass
    
    @property
    @abstractmethod
    def gate_type(self) -> str:
        """Type identifier for the gate"""
        pass

class RuleGate(BaseGate):
    """Rule-based gate using threshold logic"""
    
    def __init__(self, expert_names: List[str], 
                 thresholds: Dict[str, float] = None,
                 default_expert: str = None):
        super().__init__(expert_names)
        
        if thresholds is None:
            # Default thresholds based on contradiction scores
            thresholds = {
                "high_contradiction": 0.8,
                "medium_contradiction": 0.5,
                "low_contradiction": 0.2
            }
        
        self.thresholds = thresholds
        self.default_expert = default_expert or expert_names[0]
        
        # Map threshold ranges to experts
        self.threshold_to_expert = {}
        sorted_thresholds = sorted(thresholds.items(), key=lambda x: x[1], reverse=True)
        
        for i, (name, threshold) in enumerate(sorted_thresholds):
            if i < len(expert_names):
                self.threshold_to_expert[threshold] = expert_names[i]
        
        logger.info(f"Initialized RuleGate with thresholds: {thresholds}")
    
    def route(self, features: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> GateDecision:
        """Route based on simple threshold rules"""
        # Assume first feature is the primary contradiction score
        if len(features.shape) > 1:
            contradiction_score = features[0, 0].item()  # First sample, first feature
        else:
            contradiction_score = features[0].item()
        
        # Find appropriate expert based on thresholds
        chosen_expert = self.default_expert
        routing_scores = {expert: 0.0 for expert in self.expert_names}
        
        for threshold, expert in self.threshold_to_expert.items():
            if contradiction_score >= threshold:
                chosen_expert = expert
                break
        
        # Set routing scores (winner-take-all)
        routing_scores[chosen_expert] = 1.0
        
        return GateDecision(
            expert_choice=chosen_expert,
            confidence=min(contradiction_score, 1.0),
            routing_scores=routing_scores,
            metadata={"threshold_used": contradiction_score, "rule_type": "threshold"}
        )
    
    def update(self, decision: GateDecision, reward: float, context: Optional[Dict[str, Any]] = None):
        """Rules don't learn, but we log performance"""
        logger.debug(f"Rule gate routed to {decision.expert_choice} with reward {reward}")
    
    @property
    def gate_type(self) -> str:
        return "rule"

class MLPGate(BaseGate):
    """Learned MLP gate on [a,b,Δ,aux] features"""
    
    def __init__(self, expert_names: List[str], 
                 input_dim: int,
                 hidden_dim: int = 64,
                 temperature: float = 1.0):
        super().__init__(expert_names)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        
        # MLP architecture
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.num_experts)
        )
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.loss_history = deque(maxlen=1000)
        
        logger.info(f"Initialized MLPGate with {input_dim}→{hidden_dim}→{self.num_experts}")
    
    def route(self, features: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> GateDecision:
        """Route using learned MLP"""
        self.net.eval()
        
        with torch.no_grad():
            if len(features.shape) == 1:
                features = features.unsqueeze(0)
            
            # Forward pass
            logits = self.net(features)
            
            # Apply temperature scaling
            scaled_logits = logits / self.temperature
            
            # Softmax to get routing probabilities
            routing_probs = F.softmax(scaled_logits, dim=-1)
            
            # Choose expert (can be stochastic or deterministic)
            expert_idx = torch.argmax(routing_probs, dim=-1).item()
            chosen_expert = self.expert_names[expert_idx]
            
            # Create routing scores dict
            routing_scores = {}
            for i, expert in enumerate(self.expert_names):
                routing_scores[expert] = routing_probs[0, i].item()
            
            confidence = routing_probs[0, expert_idx].item()
        
        return GateDecision(
            expert_choice=chosen_expert,
            confidence=confidence,
            routing_scores=routing_scores,
            metadata={"logits": logits[0].tolist(), "temperature": self.temperature}
        )
    
    def update(self, decision: GateDecision, reward: float, context: Optional[Dict[str, Any]] = None):
        """Update MLP based on reward"""
        if context is None or 'features' not in context:
            logger.warning("MLPGate update requires features in context")
            return
        
        features = context['features']
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        
        self.net.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        logits = self.net(features)
        expert_idx = self.expert_names.index(decision.expert_choice)
        
        # Loss based on reward (higher reward = lower loss)
        target = torch.zeros_like(logits)
        target[0, expert_idx] = reward  # Set target for chosen expert
        
        loss = F.mse_loss(torch.softmax(logits, dim=-1), torch.softmax(target, dim=-1))
        
        loss.backward()
        self.optimizer.step()
        
        self.loss_history.append(loss.item())
        logger.debug(f"Updated MLPGate with reward {reward}, loss {loss.item():.4f}")
    
    @property
    def gate_type(self) -> str:
        return "mlp"

class BanditGate(BaseGate):
    """Contextual bandit gate (Thompson Sampling / UCB)"""
    
    def __init__(self, expert_names: List[str],
                 algorithm: str = "thompson",  # "thompson" or "ucb"
                 exploration_param: float = 1.0,
                 window_size: int = 1000):
        super().__init__(expert_names)
        
        self.algorithm = algorithm
        self.exploration_param = exploration_param
        self.window_size = window_size
        
        # Statistics for each expert
        self.expert_stats = {}
        for expert in expert_names:
            self.expert_stats[expert] = {
                'rewards': deque(maxlen=window_size),
                'counts': 0,
                'sum_rewards': 0.0,
                'sum_squared_rewards': 0.0
            }
        
        logger.info(f"Initialized BanditGate with {algorithm} algorithm")
    
    def route(self, features: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> GateDecision:
        """Route using bandit algorithm"""
        
        if self.algorithm == "thompson":
            chosen_expert, routing_scores = self._thompson_sampling()
        elif self.algorithm == "ucb":
            chosen_expert, routing_scores = self._ucb_selection()
        else:
            # Random fallback
            chosen_expert = np.random.choice(self.expert_names)
            routing_scores = {expert: 1.0/len(self.expert_names) for expert in self.expert_names}
            routing_scores[chosen_expert] = 1.0
        
        confidence = routing_scores[chosen_expert]
        
        return GateDecision(
            expert_choice=chosen_expert,
            confidence=confidence,
            routing_scores=routing_scores,
            metadata={"algorithm": self.algorithm, "total_pulls": sum(s['counts'] for s in self.expert_stats.values())}
        )
    
    def _thompson_sampling(self) -> Tuple[str, Dict[str, float]]:
        """Thompson sampling for expert selection"""
        samples = {}
        
        for expert, stats in self.expert_stats.items():
            if stats['counts'] == 0:
                # No data, sample from prior
                samples[expert] = np.random.beta(1, 1)
            else:
                # Sample from posterior Beta distribution
                mean_reward = stats['sum_rewards'] / stats['counts']
                # Simple Beta approximation
                alpha = max(1, mean_reward * stats['counts'] + 1)
                beta = max(1, (1 - mean_reward) * stats['counts'] + 1)
                samples[expert] = np.random.beta(alpha, beta)
        
        # Choose expert with highest sample
        chosen_expert = max(samples, key=samples.get)
        
        # Convert to routing scores (softmax of samples)
        routing_scores = {}
        sample_values = list(samples.values())
        exp_values = np.exp(np.array(sample_values))
        softmax_values = exp_values / np.sum(exp_values)
        
        for i, expert in enumerate(self.expert_names):
            routing_scores[expert] = softmax_values[i]
        
        return chosen_expert, routing_scores
    
    def _ucb_selection(self) -> Tuple[str, Dict[str, float]]:
        """Upper Confidence Bound selection"""
        total_pulls = sum(stats['counts'] for stats in self.expert_stats.values())
        ucb_values = {}
        
        for expert, stats in self.expert_stats.items():
            if stats['counts'] == 0:
                ucb_values[expert] = float('inf')  # Explore unexplored experts
            else:
                mean_reward = stats['sum_rewards'] / stats['counts']
                confidence_interval = self.exploration_param * np.sqrt(
                    2 * np.log(total_pulls) / stats['counts']
                )
                ucb_values[expert] = mean_reward + confidence_interval
        
        # Choose expert with highest UCB
        chosen_expert = max(ucb_values, key=ucb_values.get)
        
        # Convert to routing scores (softmax of UCB values)
        routing_scores = {}
        ucb_vals = [ucb_values[expert] for expert in self.expert_names]
        
        # Handle infinite UCB values
        if any(np.isinf(val) for val in ucb_vals):
            routing_scores = {expert: (1.0 if np.isinf(ucb_values[expert]) else 0.0) 
                            for expert in self.expert_names}
            # Normalize among infinite values
            inf_count = sum(1 for val in routing_scores.values() if val > 0)
            if inf_count > 0:
                for expert in routing_scores:
                    if routing_scores[expert] > 0:
                        routing_scores[expert] = 1.0 / inf_count
        else:
            exp_values = np.exp(np.array(ucb_vals))
            softmax_values = exp_values / np.sum(exp_values)
            routing_scores = {expert: softmax_values[i] for i, expert in enumerate(self.expert_names)}
        
        return chosen_expert, routing_scores
    
    def update(self, decision: GateDecision, reward: float, context: Optional[Dict[str, Any]] = None):
        """Update bandit statistics"""
        expert = decision.expert_choice
        stats = self.expert_stats[expert]
        
        # Update statistics
        stats['rewards'].append(reward)
        stats['counts'] += 1
        stats['sum_rewards'] += reward
        stats['sum_squared_rewards'] += reward ** 2
        
        logger.debug(f"Updated {expert}: count={stats['counts']}, avg_reward={stats['sum_rewards']/stats['counts']:.3f}")
    
    @property
    def gate_type(self) -> str:
        return "bandit"

class RLGate(BaseGate):
    """RL-based gate for long-horizon decisions (simplified PPO-style)"""
    
    def __init__(self, expert_names: List[str],
                 state_dim: int = None,
                 input_dim: int = None,
                 hidden_dim: int = 128,
                 lr: float = 3e-4):
        super().__init__(expert_names)
        
        # Handle both parameter names
        self.state_dim = state_dim or input_dim or 32  # Default fallback
        self.hidden_dim = hidden_dim
        
        # Policy network (actor)
        self.policy_net = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_experts)
        )
        
        # Value network (critic)
        self.value_net = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.optimizer = torch.optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()), 
            lr=lr
        )
        
        # Experience buffer (simplified)
        self.experience_buffer = deque(maxlen=10000)
        
        logger.info(f"Initialized RLGate with policy and value networks")
    
    def route(self, features: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> GateDecision:
        """Route using RL policy"""
        self.policy_net.eval()
        
        with torch.no_grad():
            if len(features.shape) == 1:
                features = features.unsqueeze(0)
            
            # Get action probabilities from policy
            logits = self.policy_net(features)
            action_probs = F.softmax(logits, dim=-1)
            
            # Sample action
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            expert_idx = action.item()
            
            chosen_expert = self.expert_names[expert_idx]
            
            # Create routing scores
            routing_scores = {}
            for i, expert in enumerate(self.expert_names):
                routing_scores[expert] = action_probs[0, i].item()
            
            confidence = action_probs[0, expert_idx].item()
        
        return GateDecision(
            expert_choice=chosen_expert,
            confidence=confidence,
            routing_scores=routing_scores,
            metadata={
                "action_logits": logits[0].tolist(),
                "action_probs": action_probs[0].tolist()
            }
        )
    
    def update(self, decision: GateDecision, reward: float, context: Optional[Dict[str, Any]] = None):
        """Update RL networks (simplified)"""
        if context is None or 'features' not in context:
            logger.warning("RLGate update requires features in context")
            return
        
        features = context['features']
        expert_idx = self.expert_names.index(decision.expert_choice)
        
        # Store experience
        self.experience_buffer.append({
            'state': features,
            'action': expert_idx,
            'reward': reward,
            'done': context.get('done', False)
        })
        
        # Simple policy gradient update (not full PPO)
        if len(self.experience_buffer) > 32:
            self._update_networks()
    
    def _update_networks(self):
        """Update policy and value networks"""
        batch_size = min(32, len(self.experience_buffer))
        batch = list(self.experience_buffer)[-batch_size:]
        
        states = torch.stack([exp['state'] for exp in batch])
        actions = torch.tensor([exp['action'] for exp in batch])
        rewards = torch.tensor([exp['reward'] for exp in batch], dtype=torch.float32)
        
        self.optimizer.zero_grad()
        
        # Policy loss (REINFORCE-style)
        logits = self.policy_net(states)
        action_probs = F.softmax(logits, dim=-1)
        
        selected_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze()
        policy_loss = -torch.mean(torch.log(selected_probs + 1e-8) * rewards)
        
        # Value loss
        values = self.value_net(states).squeeze()
        value_loss = F.mse_loss(values, rewards)
        
        total_loss = policy_loss + 0.5 * value_loss
        total_loss.backward()
        self.optimizer.step()
        
        logger.debug(f"Updated RL networks: policy_loss={policy_loss.item():.4f}, value_loss={value_loss.item():.4f}")
    
    @property
    def gate_type(self) -> str:
        return "rl"

class GateFactory:
    """Factory for creating different gate types"""
    
    @staticmethod
    def create_gate(gate_type: str, expert_names: List[str], **kwargs) -> BaseGate:
        """Create gate of specified type"""
        
        if gate_type == "rule":
            return RuleGate(expert_names, **kwargs)
        elif gate_type == "mlp":
            return MLPGate(expert_names, **kwargs)
        elif gate_type == "bandit":
            return BanditGate(expert_names, **kwargs)
        elif gate_type == "rl":
            return RLGate(expert_names, **kwargs)
        else:
            raise ValueError(f"Unknown gate type: {gate_type}. Available: rule, mlp, bandit, rl")
    
    @staticmethod
    def get_available_gates() -> List[str]:
        """Get list of available gate types"""
        return ["rule", "mlp", "bandit", "rl"]

if __name__ == "__main__":
    # Demo the gate system
    print("Pluggable Gates System Demo")
    print("=" * 35)
    
    expert_names = ["AlignedExpert", "AntiAExpert", "AntiBExpert", "SafeFallbackExpert"]
    feature_dim = 16
    
    # Test features
    features = torch.randn(1, feature_dim)
    
    # Test each gate type
    for gate_type in GateFactory.get_available_gates():
        print(f"\nTesting {gate_type.upper()} Gate:")
        
        kwargs = {}
        if gate_type in ["mlp", "rl"]:
            kwargs["input_dim"] = feature_dim
        
        gate = GateFactory.create_gate(gate_type, expert_names, **kwargs)
        
        # Route decision
        decision = gate.route(features)
        print(f"  Chose: {decision.expert_choice}")
        print(f"  Confidence: {decision.confidence:.3f}")
        print(f"  Routing scores: {decision.routing_scores}")
        
        # Simulate update
        reward = np.random.uniform(0, 1)
        context = {"features": features} if gate_type in ["mlp", "rl"] else None
        gate.update(decision, reward, context)
        print(f"  Updated with reward: {reward:.3f}")
    
    print("\nGate system working correctly!")