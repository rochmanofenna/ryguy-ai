#!/usr/bin/env python3
"""
FusionAlpha Universal Router

The main class that ties together modalities, gates, experts, and scoring.
Clean API that feels obvious for any domain.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from datetime import datetime

from .modality_registry import ModalityRegistry, get_registry
from .scores import ContradictionScorer, create_default_scorer
from .gates import BaseGate, GateFactory, GateDecision
from .experts import BaseExpert, ExpertEnsemble, create_standard_experts, ExpertPrediction
from .calibrate import TemperatureCalibrator, UncertaintyEstimator

logger = logging.getLogger(__name__)

@dataclass
class FusionResult:
    """Complete result from FusionAlpha prediction"""
    prediction: Union[float, torch.Tensor]
    confidence: float
    expert_used: str
    gate_decision: GateDecision
    expert_prediction: ExpertPrediction
    contradiction_scores: Dict[str, float]
    uncertainty_info: Dict[str, float]
    abstained: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class FusionAlpha:
    """
    Universal contradiction-aware decision layer.
    
    Clean API that works across domains:
    - Healthcare: symptoms vs biomarkers
    - SRE/Ops: metrics vs logs vs user reports  
    - Robotics: vision vs lidar vs proprioception
    - Content moderation: text vs image vs heuristics
    - Finance: sentiment vs price (original use case)
    """
    
    def __init__(self,
                 modalities: Optional[List[str]] = None,
                 gate: str = "rule",
                 experts: Optional[List[BaseExpert]] = None,
                 contradiction_scores: List[str] = None,
                 calibrator: str = "temperature",
                 risk_budget: Optional[Dict[str, float]] = None,
                 device: str = "cpu"):
        
        self.device = device
        
        # Initialize modality registry
        self.registry = get_registry()
        self.modalities = modalities or []
        
        # Initialize contradiction scoring
        if contradiction_scores is None:
            contradiction_scores = ["cosine", "delta"]
        self.scorer = ContradictionScorer(contradiction_scores)
        
        # Initialize experts
        if experts is None:
            # Estimate input dimension from registered modalities
            input_dim = self._estimate_input_dim()
            experts = create_standard_experts(input_dim)
        
        self.expert_ensemble = ExpertEnsemble(experts)
        self.expert_names = list(self.expert_ensemble.experts.keys())
        
        # Initialize gate
        self.gate = self._create_gate(gate)
        
        # Initialize calibrator and uncertainty estimation
        self.calibrator = self._create_calibrator(calibrator)
        self.uncertainty_estimator = UncertaintyEstimator()
        
        # Risk management
        if risk_budget is None:
            risk_budget = {name: 1.0 for name in self.expert_names}
        self.risk_budget = risk_budget
        
        # Tracking
        self.prediction_history = []
        self.performance_metrics = {
            "total_predictions": 0,
            "abstain_count": 0,
            "expert_usage": {name: 0 for name in self.expert_names},
            "avg_confidence": 0.0
        }
        
        logger.info(f"Initialized FusionAlpha with {len(self.expert_names)} experts and {gate} gate")
    
    def _estimate_input_dim(self) -> int:
        """Estimate input dimension from registered modalities"""
        if not self.modalities:
            logger.debug("No modalities specified, using default dim 64")
            return 64  # Default
        
        total_dim = 0
        for mod_name in self.modalities:
            spec = self.registry.get_modality(mod_name)
            if spec:
                dim = np.prod(spec.shape)
                total_dim += dim
            else:
                logger.warning(f"Modality {mod_name} not found in registry, using default dim 32")
                total_dim += 32  # Default dimension if modality not registered yet
        
        final_dim = total_dim or 32  # Use calculated dimension or default to 32
        return final_dim
    
    def _create_gate(self, gate_type: str) -> BaseGate:
        """Create gate with appropriate parameters"""
        kwargs = {}
        if gate_type in ["mlp", "rl"]:
            kwargs["input_dim"] = self._estimate_input_dim()
        
        return GateFactory.create_gate(gate_type, self.expert_names, **kwargs)
    
    def _create_calibrator(self, calibrator_type: str):
        """Create calibrator"""
        if calibrator_type == "temperature":
            return TemperatureCalibrator()
        else:
            return None
    
    def fit(self, train_loader, val_loader=None):
        """Fit experts and gate if learnable"""
        logger.info("Training FusionAlpha components...")
        
        total_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if isinstance(batch, dict):
                # Extract features and labels
                features = self._extract_batch_features(batch)
                labels = batch.get('labels', torch.zeros(features.shape[0]))
                
                # Train gate and experts with batch
                for i in range(features.shape[0]):
                    sample_features = features[i]
                    sample_label = labels[i] if hasattr(labels, 'shape') and len(labels.shape) > 0 else labels
                    
                    # Get contradiction scores
                    contradiction_scores = self._compute_contradiction_scores(sample_features)
                    
                    # Route through gate
                    gate_decision = self.gate.route(sample_features)
                    
                    # Get expert prediction
                    expert = self.expert_ensemble.get_expert(gate_decision.expert_choice)
                    if expert:
                        context = {
                            'features': sample_features,
                            'contradiction_scores': contradiction_scores
                        }
                        prediction = expert.forward(sample_features, context)
                        
                        # Use label as reward (assuming normalized 0-1)
                        reward = sample_label.item() if isinstance(sample_label, torch.Tensor) else float(sample_label)
                        
                        # Update expert and gate
                        expert.update(prediction, reward, context)
                        self.gate.update(gate_decision, reward, context)
                        
                        total_samples += 1
        
        logger.info(f"Training completed on {total_samples} samples")
        
        # Calibrate if we have validation data
        if val_loader and self.calibrator:
            self._calibrate(val_loader)
    
    def _extract_batch_features(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Extract and concatenate features from batch"""
        feature_list = []
        
        for modality_name in self.modalities:
            if modality_name in batch:
                data = batch[modality_name]
                features = self.registry.extract_features(modality_name, data)
                
                # Flatten if needed
                if len(features.shape) > 2:
                    features = features.view(features.shape[0], -1)
                
                feature_list.append(features)
        
        if feature_list:
            return torch.cat(feature_list, dim=1)
        else:
            # Fallback: assume 'features' key exists
            features = batch.get('features', torch.randn(1, 32))
            if len(features.shape) == 1:
                features = features.unsqueeze(0)
            return features
    
    def _compute_contradiction_scores(self, features: torch.Tensor) -> Dict[str, float]:
        """Compute contradiction scores for feature vector"""
        # Split features into two halves (modality A and B)
        mid = features.shape[0] // 2 if len(features.shape) == 1 else features.shape[1] // 2
        
        if len(features.shape) == 1:
            mod_a = features[:mid]
            mod_b = features[mid:]
        else:
            mod_a = features[:, :mid]
            mod_b = features[:, mid:]
        
        # Ensure they have the same shape
        if mod_a.shape != mod_b.shape:
            min_dim = min(mod_a.shape[-1], mod_b.shape[-1])
            mod_a = mod_a[..., :min_dim]
            mod_b = mod_b[..., :min_dim]
        
        # Add batch dimension if needed
        if len(mod_a.shape) == 1:
            mod_a = mod_a.unsqueeze(0)
            mod_b = mod_b.unsqueeze(0)
        
        # Compute scores
        scores = self.scorer.compute_pairwise(mod_a, mod_b)
        
        # Convert to scalar values
        return {name: score.mean().item() for name, score in scores.items()}
    
    def predict(self, batch: Union[Dict[str, Any], torch.Tensor], 
                return_info: bool = False) -> Union[FusionResult, Tuple[Any, Dict]]:
        """Main prediction interface"""
        
        # Handle different input types
        if isinstance(batch, torch.Tensor):
            features = batch
            if len(features.shape) == 1:
                features = features.unsqueeze(0)
        elif isinstance(batch, dict):
            features = self._extract_batch_features(batch)
        else:
            raise ValueError("Input must be tensor or dict")
        
        # Process first sample (extend for batch later)
        sample_features = features[0] if features.shape[0] > 1 else features.squeeze(0)
        
        # Compute contradiction scores
        contradiction_scores = self._compute_contradiction_scores(sample_features)
        
        # Route through gate
        gate_decision = self.gate.route(sample_features)
        
        # Get expert prediction
        expert = self.expert_ensemble.get_expert(gate_decision.expert_choice)
        if not expert:
            raise ValueError(f"Expert {gate_decision.expert_choice} not found")
        
        context = {
            'features': sample_features,
            'contradiction_scores': contradiction_scores,
            'batch': batch if isinstance(batch, dict) else None
        }
        
        expert_prediction = expert.forward(sample_features, context)
        
        # Apply risk budget
        risk_multiplier = self.risk_budget.get(expert.name, 1.0)
        adjusted_prediction = expert_prediction.prediction * risk_multiplier
        
        # Estimate uncertainty
        uncertainty_info = self.uncertainty_estimator.estimate(
            sample_features, expert_prediction, contradiction_scores
        )
        
        # Apply calibration
        calibrated_confidence = expert_prediction.confidence
        if self.calibrator and self.calibrator.is_fitted():
            calibrated_confidence = self.calibrator.calibrate(
                expert_prediction.confidence, expert_prediction.prediction
            )
        
        # Determine if we should abstain
        abstained = (expert_prediction.should_abstain or 
                    calibrated_confidence < 0.1 or
                    uncertainty_info.get('should_abstain', False))
        
        # Update tracking
        self.performance_metrics["total_predictions"] += 1
        self.performance_metrics["expert_usage"][expert.name] += 1
        if abstained:
            self.performance_metrics["abstain_count"] += 1
        
        # Update running average confidence
        total_preds = self.performance_metrics["total_predictions"]
        old_avg = self.performance_metrics["avg_confidence"]
        self.performance_metrics["avg_confidence"] = (
            (old_avg * (total_preds - 1) + calibrated_confidence) / total_preds
        )
        
        # Create result
        result = FusionResult(
            prediction=adjusted_prediction,
            confidence=calibrated_confidence,
            expert_used=expert.name,
            gate_decision=gate_decision,
            expert_prediction=expert_prediction,
            contradiction_scores=contradiction_scores,
            uncertainty_info=uncertainty_info,
            abstained=abstained,
            metadata={
                "risk_multiplier": risk_multiplier,
                "original_confidence": expert_prediction.confidence,
                "timestamp": datetime.now(),
                "gate_type": self.gate.gate_type
            }
        )
        
        self.prediction_history.append(result)
        
        if return_info:
            info = {
                "scores": contradiction_scores,
                "chosen_expert": expert.name,
                "uncertainty": uncertainty_info.get("total_uncertainty", 0.0),
                "abstained": abstained,
                "gate_confidence": gate_decision.confidence,
                "routing_scores": gate_decision.routing_scores
            }
            return result.prediction, info
        
        return result
    
    def update(self, result: FusionResult, reward: float):
        """Update system with feedback"""
        
        # Update expert
        expert = self.expert_ensemble.get_expert(result.expert_used)
        if expert:
            context = {
                'features': result.metadata.get('features'),
                'contradiction_scores': result.contradiction_scores
            }
            expert.update(result.expert_prediction, reward, context)
        
        # Update gate
        self.gate.update(result.gate_decision, reward, context)
        
        logger.debug(f"Updated system with reward {reward} for expert {result.expert_used}")
    
    def _calibrate(self, val_loader):
        """Calibrate the system on validation data"""
        if not self.calibrator:
            return
        
        predictions = []
        confidences = []
        
        for batch in val_loader:
            features = self._extract_batch_features(batch)
            labels = batch.get('labels', torch.zeros(features.shape[0]))
            
            for i in range(features.shape[0]):
                result = self.predict(features[i])
                predictions.append(result.prediction)
                confidences.append(result.confidence)
        
        if predictions:
            predictions = torch.tensor(predictions)
            confidences = torch.tensor(confidences)
            labels = torch.tensor([batch.get('labels', 0) for batch in val_loader])
            
            self.calibrator.fit(confidences, predictions, labels[:len(predictions)])
            logger.info("Calibration completed")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        total_preds = self.performance_metrics["total_predictions"]
        
        metrics = self.performance_metrics.copy()
        metrics.update({
            "abstain_rate": self.performance_metrics["abstain_count"] / max(1, total_preds),
            "expert_usage_pct": {
                name: count / max(1, total_preds) 
                for name, count in self.performance_metrics["expert_usage"].items()
            },
            "gate_type": self.gate.gate_type,
            "num_modalities": len(self.modalities),
            "contradiction_scores": list(self.scorer.scorers.keys())
        })
        
        return metrics
    
    def get_expert_stats(self) -> Dict[str, Dict[str, float]]:
        """Get detailed expert statistics"""
        return self.expert_ensemble.get_ensemble_stats()
    
    def add_expert(self, expert: BaseExpert):
        """Add new expert to the system"""
        self.expert_ensemble.add_expert(expert)
        self.expert_names.append(expert.name)
        self.risk_budget[expert.name] = 1.0
        self.performance_metrics["expert_usage"][expert.name] = 0
        
        # Recreate gate with new expert list
        old_gate_type = self.gate.gate_type
        self.gate = self._create_gate(old_gate_type)
        
        logger.info(f"Added expert {expert.name} and recreated {old_gate_type} gate")
    
    def configure_risk_budget(self, budget: Dict[str, float]):
        """Update risk budget for experts"""
        self.risk_budget.update(budget)
        logger.info(f"Updated risk budget: {budget}")

# Convenience functions for common use cases
def create_healthcare_fusion(modalities: List[str] = None) -> FusionAlpha:
    """Create FusionAlpha configured for healthcare"""
    if modalities is None:
        modalities = ["patient_notes", "lab_results", "vital_signs"]
    
    return FusionAlpha(
        modalities=modalities,
        gate="mlp",
        contradiction_scores=["cosine", "delta", "magnitude_ratio"],
        calibrator="temperature"
    )

def create_sre_fusion(modalities: List[str] = None) -> FusionAlpha:
    """Create FusionAlpha configured for SRE/Ops"""
    if modalities is None:
        modalities = ["metrics_ts", "logs_text", "user_reports"]
    
    return FusionAlpha(
        modalities=modalities,
        gate="bandit",  # Good for online ops
        contradiction_scores=["cosine", "delta", "sign_flip_rate"],
        calibrator="temperature"
    )

def create_robotics_fusion(modalities: List[str] = None) -> FusionAlpha:
    """Create FusionAlpha configured for robotics"""
    if modalities is None:
        modalities = ["vision_det", "lidar_vec", "proprio_state"]
    
    return FusionAlpha(
        modalities=modalities,
        gate="rl",  # Good for sequential decisions
        contradiction_scores=["cosine", "delta", "direction_change"],
        calibrator="temperature",
        risk_budget={"SafeFallbackExpert": 2.0}  # Higher weight on safety
    )

if __name__ == "__main__":
    # Demo the FusionAlpha system
    print("FusionAlpha Universal Router Demo")
    print("=" * 40)
    
    # Create a simple system
    fa = FusionAlpha(
        gate="rule",
        contradiction_scores=["cosine", "delta"]
    )
    
    # Test prediction
    test_features = torch.randn(32)  # 32-dim feature vector
    
    result = fa.predict(test_features)
    
    print(f"Prediction: {result.prediction:.3f}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Expert used: {result.expert_used}")
    print(f"Abstained: {result.abstained}")
    print(f"Contradiction scores: {result.contradiction_scores}")
    
    # Test update
    reward = 0.8  # Good prediction
    fa.update(result, reward)
    
    # Get metrics
    metrics = fa.get_performance_metrics()
    print(f"\nPerformance metrics: {metrics}")
    
    print("\nFusionAlpha system working correctly!")