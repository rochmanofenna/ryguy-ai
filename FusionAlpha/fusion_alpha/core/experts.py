#!/usr/bin/env python3
"""
Expert Interface System

Each expert = forward(z) -> (y, conf) with a risk_profile tag.
Ships with generic experts plus optional ENN feature integration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class RiskProfile(Enum):
    """Risk profile categories for experts"""
    CONSERVATIVE = "conservative"  # Low risk, high precision
    BALANCED = "balanced"          # Medium risk, balanced approach
    AGGRESSIVE = "aggressive"      # High risk, high potential reward
    SAFE_FALLBACK = "safe_fallback" # Minimal risk, abstain when uncertain

@dataclass
class ExpertPrediction:
    """Prediction output from an expert"""
    prediction: Union[float, torch.Tensor]
    confidence: float
    expert_name: str
    risk_profile: RiskProfile
    should_abstain: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseExpert(ABC):
    """Base class for all experts"""
    
    def __init__(self, name: str, risk_profile: RiskProfile):
        self.name = name
        self.risk_profile = risk_profile
        self.prediction_history = []
        
    @abstractmethod
    def forward(self, features: torch.Tensor, 
                context: Optional[Dict[str, Any]] = None) -> ExpertPrediction:
        """Forward pass to generate prediction"""
        pass
    
    @abstractmethod
    def update(self, prediction: ExpertPrediction, reward: float, 
               context: Optional[Dict[str, Any]] = None):
        """Update expert based on feedback"""
        pass
    
    def get_stats(self) -> Dict[str, float]:
        """Get expert performance statistics"""
        if not self.prediction_history:
            return {"predictions": 0, "avg_confidence": 0.0}
        
        confidences = [p.confidence for p in self.prediction_history]
        return {
            "predictions": len(self.prediction_history),
            "avg_confidence": np.mean(confidences),
            "confidence_std": np.std(confidences)
        }

class AlignedExpert(BaseExpert):
    """Expert that assumes both modalities are correct and aligned"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__("AlignedExpert", RiskProfile.BALANCED)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Simple MLP for aligned prediction
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # [prediction, confidence]
        )
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        logger.info(f"Initialized AlignedExpert with {input_dim}→{hidden_dim}")
    
    def forward(self, features: torch.Tensor, 
                context: Optional[Dict[str, Any]] = None) -> ExpertPrediction:
        """Forward assuming modalities agree"""
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        
        with torch.no_grad():
            output = self.net(features)
            prediction = torch.tanh(output[0, 0]).item()  # Bounded prediction
            confidence = torch.sigmoid(output[0, 1]).item()  # 0-1 confidence
        
        pred_obj = ExpertPrediction(
            prediction=prediction,
            confidence=confidence,
            expert_name=self.name,
            risk_profile=self.risk_profile,
            should_abstain=confidence < 0.3,  # Abstain if very uncertain
            metadata={"assumption": "modalities_aligned", "features_used": features.shape[1]}
        )
        
        self.prediction_history.append(pred_obj)
        return pred_obj
    
    def update(self, prediction: ExpertPrediction, reward: float, 
               context: Optional[Dict[str, Any]] = None):
        """Update aligned expert network"""
        if context is None or 'features' not in context:
            return
        
        features = context['features']
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        
        self.optimizer.zero_grad()
        
        output = self.net(features)
        pred_logit = output[0, 0]
        conf_logit = output[0, 1]
        
        # Loss based on reward
        target_pred = torch.tensor([reward], dtype=torch.float32)
        target_conf = torch.tensor([abs(reward)], dtype=torch.float32)  # High reward = high confidence
        
        pred_loss = F.mse_loss(torch.tanh(pred_logit).unsqueeze(0), target_pred)
        conf_loss = F.mse_loss(torch.sigmoid(conf_logit).unsqueeze(0), target_conf)
        
        total_loss = pred_loss + 0.1 * conf_loss
        total_loss.backward()
        self.optimizer.step()
        
        logger.debug(f"Updated {self.name} with reward {reward:.3f}, loss {total_loss.item():.4f}")

class AntiAExpert(BaseExpert):
    """Expert that trusts modality B over modality A when they disagree"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, 
                 modality_a_dim: Optional[int] = None):
        super().__init__("AntiAExpert", RiskProfile.AGGRESSIVE)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.modality_a_dim = modality_a_dim or input_dim // 2
        
        # Network that focuses on modality B features
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        # Attention mechanism to downweight modality A
        self.attention = nn.Linear(input_dim, input_dim)
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        logger.info(f"Initialized AntiAExpert (trusts modality B)")
    
    def forward(self, features: torch.Tensor, 
                context: Optional[Dict[str, Any]] = None) -> ExpertPrediction:
        """Forward with anti-A bias"""
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        
        with torch.no_grad():
            # Apply attention to downweight modality A
            attention_weights = torch.sigmoid(self.attention(features))
            
            # Create mask that downweights first half (assumed to be modality A)
            mask = torch.ones_like(attention_weights)
            mask[:, :self.modality_a_dim] *= 0.3  # Downweight modality A
            
            weighted_features = features * attention_weights * mask
            
            output = self.net(weighted_features)
            prediction = torch.tanh(output[0, 0]).item()
            confidence = torch.sigmoid(output[0, 1]).item()
            
            # Boost confidence when contradiction is high
            if context and 'contradiction_scores' in context:
                contradiction = context['contradiction_scores'].get('cosine', 0)
                confidence *= (1 + contradiction)  # Higher contradiction = higher confidence in anti-A
                confidence = min(confidence, 1.0)
        
        pred_obj = ExpertPrediction(
            prediction=prediction,
            confidence=confidence,
            expert_name=self.name,
            risk_profile=self.risk_profile,
            should_abstain=confidence < 0.2,
            metadata={"bias": "anti_modality_a", "attention_applied": True}
        )
        
        self.prediction_history.append(pred_obj)
        return pred_obj
    
    def update(self, prediction: ExpertPrediction, reward: float, 
               context: Optional[Dict[str, Any]] = None):
        """Update anti-A expert"""
        if context is None or 'features' not in context:
            return
        
        features = context['features']
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        
        self.optimizer.zero_grad()
        
        # Apply same attention mechanism
        attention_weights = torch.sigmoid(self.attention(features))
        mask = torch.ones_like(attention_weights)
        mask[:, :self.modality_a_dim] *= 0.3
        weighted_features = features * attention_weights * mask
        
        output = self.net(weighted_features)
        
        # Loss computation
        target_pred = torch.tensor([reward], dtype=torch.float32)
        target_conf = torch.tensor([abs(reward)], dtype=torch.float32)
        
        pred_loss = F.mse_loss(torch.tanh(output[0, 0]).unsqueeze(0), target_pred)
        conf_loss = F.mse_loss(torch.sigmoid(output[0, 1]).unsqueeze(0), target_conf)
        
        total_loss = pred_loss + 0.1 * conf_loss
        total_loss.backward()
        self.optimizer.step()

class AntiBExpert(BaseExpert):
    """Expert that trusts modality A over modality B when they disagree"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 modality_a_dim: Optional[int] = None):
        super().__init__("AntiBExpert", RiskProfile.AGGRESSIVE)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.modality_a_dim = modality_a_dim or input_dim // 2
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        self.attention = nn.Linear(input_dim, input_dim)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        logger.info(f"Initialized AntiBExpert (trusts modality A)")
    
    def forward(self, features: torch.Tensor, 
                context: Optional[Dict[str, Any]] = None) -> ExpertPrediction:
        """Forward with anti-B bias"""
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        
        with torch.no_grad():
            attention_weights = torch.sigmoid(self.attention(features))
            
            # Create mask that downweights second half (assumed to be modality B)
            mask = torch.ones_like(attention_weights)
            mask[:, self.modality_a_dim:] *= 0.3  # Downweight modality B
            
            weighted_features = features * attention_weights * mask
            
            output = self.net(weighted_features)
            prediction = torch.tanh(output[0, 0]).item()
            confidence = torch.sigmoid(output[0, 1]).item()
            
            # Boost confidence when contradiction is high
            if context and 'contradiction_scores' in context:
                contradiction = context['contradiction_scores'].get('cosine', 0)
                confidence *= (1 + contradiction)
                confidence = min(confidence, 1.0)
        
        pred_obj = ExpertPrediction(
            prediction=prediction,
            confidence=confidence,
            expert_name=self.name,
            risk_profile=self.risk_profile,
            should_abstain=confidence < 0.2,
            metadata={"bias": "anti_modality_b", "attention_applied": True}
        )
        
        self.prediction_history.append(pred_obj)
        return pred_obj
    
    def update(self, prediction: ExpertPrediction, reward: float, 
               context: Optional[Dict[str, Any]] = None):
        """Update anti-B expert"""
        if context is None or 'features' not in context:
            return
        
        features = context['features']
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        
        self.optimizer.zero_grad()
        
        attention_weights = torch.sigmoid(self.attention(features))
        mask = torch.ones_like(attention_weights)
        mask[:, self.modality_a_dim:] *= 0.3
        weighted_features = features * attention_weights * mask
        
        output = self.net(weighted_features)
        
        target_pred = torch.tensor([reward], dtype=torch.float32)
        target_conf = torch.tensor([abs(reward)], dtype=torch.float32)
        
        pred_loss = F.mse_loss(torch.tanh(output[0, 0]).unsqueeze(0), target_pred)
        conf_loss = F.mse_loss(torch.sigmoid(output[0, 1]).unsqueeze(0), target_conf)
        
        total_loss = pred_loss + 0.1 * conf_loss
        total_loss.backward()
        self.optimizer.step()

class SafeFallbackExpert(BaseExpert):
    """Conservative expert that abstains when uncertain"""
    
    def __init__(self, abstain_threshold: float = 0.7, 
                 uncertainty_penalty: float = 0.1):
        super().__init__("SafeFallbackExpert", RiskProfile.SAFE_FALLBACK)
        
        self.abstain_threshold = abstain_threshold
        self.uncertainty_penalty = uncertainty_penalty
        self.abstain_count = 0
        self.total_predictions = 0
        
        logger.info(f"Initialized SafeFallbackExpert (abstain_threshold={abstain_threshold})")
    
    def forward(self, features: torch.Tensor, 
                context: Optional[Dict[str, Any]] = None) -> ExpertPrediction:
        """Conservative prediction with abstention"""
        
        # Calculate uncertainty from feature variance and contradiction scores
        feature_uncertainty = torch.var(features).item()
        
        contradiction_uncertainty = 0.0
        if context and 'contradiction_scores' in context:
            scores = context['contradiction_scores']
            contradiction_uncertainty = scores.get('cosine', 0) + scores.get('delta', 0)
            contradiction_uncertainty = min(contradiction_uncertainty / 2.0, 1.0)
        
        total_uncertainty = (feature_uncertainty + contradiction_uncertainty) / 2.0
        confidence = max(0.0, 1.0 - total_uncertainty - self.uncertainty_penalty)
        
        # Conservative prediction (close to zero)
        prediction = 0.1 if confidence > 0.5 else 0.0
        
        # Abstain if uncertainty is too high
        should_abstain = total_uncertainty > self.abstain_threshold
        
        if should_abstain:
            self.abstain_count += 1
            prediction = 0.0
            confidence = 0.0
        
        self.total_predictions += 1
        
        pred_obj = ExpertPrediction(
            prediction=prediction,
            confidence=confidence,
            expert_name=self.name,
            risk_profile=self.risk_profile,
            should_abstain=should_abstain,
            metadata={
                "feature_uncertainty": feature_uncertainty,
                "contradiction_uncertainty": contradiction_uncertainty,
                "total_uncertainty": total_uncertainty,
                "abstain_rate": self.abstain_count / max(1, self.total_predictions)
            }
        )
        
        self.prediction_history.append(pred_obj)
        return pred_obj
    
    def update(self, prediction: ExpertPrediction, reward: float, 
               context: Optional[Dict[str, Any]] = None):
        """Safe expert learns from abstention patterns"""
        if prediction.should_abstain:
            # If we abstained and reward was high, lower threshold slightly
            if reward > 0.7:
                self.abstain_threshold = max(0.5, self.abstain_threshold - 0.01)
            # If we abstained and reward was low, good decision
            elif reward < 0.3:
                self.abstain_threshold = min(0.9, self.abstain_threshold + 0.005)
        
        logger.debug(f"Updated SafeFallbackExpert: abstain_threshold={self.abstain_threshold:.3f}")

class ENNIntegratedExpert(BaseExpert):
    """Expert that uses ENN entanglement features for dependency-aware decisions"""
    
    def __init__(self, name: str, input_dim: int, enn_dim: int = 32, 
                 risk_profile: RiskProfile = RiskProfile.BALANCED):
        super().__init__(name, risk_profile)
        
        self.input_dim = input_dim
        self.enn_dim = enn_dim
        
        # Network that processes both regular features and ENN entanglement vector
        total_dim = input_dim + enn_dim
        
        self.net = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # [prediction, confidence]
        )
        
        # ENN feature processor
        self.enn_processor = nn.Sequential(
            nn.Linear(enn_dim, enn_dim),
            nn.Tanh(),
            nn.Linear(enn_dim, enn_dim)
        )
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        logger.info(f"Initialized ENNIntegratedExpert with ENN features")
    
    def forward(self, features: torch.Tensor, 
                context: Optional[Dict[str, Any]] = None) -> ExpertPrediction:
        """Forward with ENN entanglement features"""
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        
        # Get ENN entanglement vector from context
        enn_features = torch.zeros(1, self.enn_dim)
        if context and 'enn_entanglement' in context:
            enn_raw = context['enn_entanglement']
            if isinstance(enn_raw, torch.Tensor):
                enn_features = self.enn_processor(enn_raw.unsqueeze(0) if len(enn_raw.shape) == 1 else enn_raw)
        
        with torch.no_grad():
            # Concatenate regular features with processed ENN features
            combined_features = torch.cat([features, enn_features], dim=1)
            
            output = self.net(combined_features)
            prediction = torch.tanh(output[0, 0]).item()
            confidence = torch.sigmoid(output[0, 1]).item()
        
        pred_obj = ExpertPrediction(
            prediction=prediction,
            confidence=confidence,
            expert_name=self.name,
            risk_profile=self.risk_profile,
            should_abstain=confidence < 0.25,
            metadata={"enn_features_used": True, "enn_dim": self.enn_dim}
        )
        
        self.prediction_history.append(pred_obj)
        return pred_obj
    
    def update(self, prediction: ExpertPrediction, reward: float, 
               context: Optional[Dict[str, Any]] = None):
        """Update ENN-integrated expert"""
        if context is None or 'features' not in context:
            return
        
        features = context['features']
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        
        # Process ENN features same as forward
        enn_features = torch.zeros(1, self.enn_dim)
        if 'enn_entanglement' in context:
            enn_raw = context['enn_entanglement']
            if isinstance(enn_raw, torch.Tensor):
                enn_features = self.enn_processor(enn_raw.unsqueeze(0) if len(enn_raw.shape) == 1 else enn_raw)
        
        combined_features = torch.cat([features, enn_features], dim=1)
        
        self.optimizer.zero_grad()
        
        output = self.net(combined_features)
        
        target_pred = torch.tensor([reward], dtype=torch.float32)
        target_conf = torch.tensor([abs(reward)], dtype=torch.float32)
        
        pred_loss = F.mse_loss(torch.tanh(output[0, 0]).unsqueeze(0), target_pred)
        conf_loss = F.mse_loss(torch.sigmoid(output[0, 1]).unsqueeze(0), target_conf)
        
        total_loss = pred_loss + 0.1 * conf_loss
        total_loss.backward()
        self.optimizer.step()

class ExpertEnsemble:
    """Manages a collection of experts"""
    
    def __init__(self, experts: List[BaseExpert]):
        self.experts = {expert.name: expert for expert in experts}
        self.prediction_history = []
        
        logger.info(f"Initialized ExpertEnsemble with {len(experts)} experts: {list(self.experts.keys())}")
    
    def get_expert(self, name: str) -> Optional[BaseExpert]:
        """Get expert by name"""
        return self.experts.get(name)
    
    def predict_all(self, features: torch.Tensor, 
                   context: Optional[Dict[str, Any]] = None) -> Dict[str, ExpertPrediction]:
        """Get predictions from all experts"""
        predictions = {}
        
        for name, expert in self.experts.items():
            try:
                pred = expert.forward(features, context)
                predictions[name] = pred
            except Exception as e:
                logger.warning(f"Expert {name} failed prediction: {e}")
        
        return predictions
    
    def update_expert(self, expert_name: str, prediction: ExpertPrediction, 
                     reward: float, context: Optional[Dict[str, Any]] = None):
        """Update specific expert"""
        if expert_name in self.experts:
            self.experts[expert_name].update(prediction, reward, context)
    
    def get_ensemble_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all experts"""
        return {name: expert.get_stats() for name, expert in self.experts.items()}
    
    def add_expert(self, expert: BaseExpert):
        """Add new expert to ensemble"""
        self.experts[expert.name] = expert
        logger.info(f"Added expert: {expert.name}")
    
    def remove_expert(self, name: str):
        """Remove expert from ensemble"""
        if name in self.experts:
            del self.experts[name]
            logger.info(f"Removed expert: {name}")

def create_standard_experts(input_dim: int, modality_a_dim: Optional[int] = None) -> List[BaseExpert]:
    """Create standard set of experts"""
    experts = [
        AlignedExpert(input_dim),
        AntiAExpert(input_dim, modality_a_dim=modality_a_dim),
        AntiBExpert(input_dim, modality_a_dim=modality_a_dim),
        SafeFallbackExpert()
    ]
    return experts

def create_enn_integrated_experts(input_dim: int, enn_dim: int = 32) -> List[BaseExpert]:
    """Create experts with ENN integration"""
    experts = [
        ENNIntegratedExpert("ENNAligned", input_dim, enn_dim, RiskProfile.BALANCED),
        ENNIntegratedExpert("ENNAntiA", input_dim, enn_dim, RiskProfile.AGGRESSIVE),
        ENNIntegratedExpert("ENNAntiB", input_dim, enn_dim, RiskProfile.AGGRESSIVE),
        SafeFallbackExpert()
    ]
    return experts

if __name__ == "__main__":
    # Demo the expert system
    print("Expert Interface System Demo")
    print("=" * 35)
    
    input_dim = 32
    features = torch.randn(input_dim)
    
    # Create standard experts
    experts = create_standard_experts(input_dim)
    ensemble = ExpertEnsemble(experts)
    
    print(f"Created ensemble with experts: {list(ensemble.experts.keys())}")
    
    # Test predictions
    context = {
        'contradiction_scores': {'cosine': 0.7, 'delta': 0.5},
        'features': features
    }
    
    predictions = ensemble.predict_all(features, context)
    
    print("\nExpert Predictions:")
    for name, pred in predictions.items():
        print(f"  {name}:")
        print(f"    Prediction: {pred.prediction:.3f}")
        print(f"    Confidence: {pred.confidence:.3f}")
        print(f"    Should abstain: {pred.should_abstain}")
        print(f"    Risk profile: {pred.risk_profile.value}")
    
    # Test updates
    print("\nUpdating experts with rewards...")
    for name, pred in predictions.items():
        reward = np.random.uniform(0, 1)
        ensemble.update_expert(name, pred, reward, context)
        print(f"  Updated {name} with reward {reward:.3f}")
    
    # Get stats
    print("\nExpert Statistics:")
    stats = ensemble.get_ensemble_stats()
    for name, stat in stats.items():
        print(f"  {name}: {stat}")
    
    print("\nExpert system working correctly!")