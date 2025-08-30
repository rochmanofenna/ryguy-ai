#!/usr/bin/env python3
"""
Contradiction Score Library

A collection of metrics to quantify "disagreement" between modalities.
Supports pairwise and groupwise contradiction scoring.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from abc import ABC, abstractmethod
import logging
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import mutual_info_score

logger = logging.getLogger(__name__)

class BaseContradictionScore(ABC):
    """Base class for contradiction scoring functions"""
    
    @abstractmethod
    def compute(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute contradiction score between two tensors"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the score"""
        pass
    
    @property 
    def higher_is_more_contradictory(self) -> bool:
        """Whether higher scores indicate more contradiction"""
        return True

class DeltaScore(BaseContradictionScore):
    """L2 distance between embeddings: ||a-b||"""
    
    def compute(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.norm(a - b, dim=-1)
    
    @property
    def name(self) -> str:
        return "delta"

class CosineScore(BaseContradictionScore):
    """Cosine disagreement: 1 - cos(a,b)"""
    
    def compute(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        cos_sim = F.cosine_similarity(a, b, dim=-1, eps=1e-8)
        return 1.0 - cos_sim
    
    @property
    def name(self) -> str:
        return "cosine"

class RankDivergenceScore(BaseContradictionScore):
    """Spearman rank correlation disagreement for ordinal data"""
    
    def compute(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        batch_size = a.shape[0]
        scores = torch.zeros(batch_size)
        
        for i in range(batch_size):
            a_np = a[i].detach().cpu().numpy()
            b_np = b[i].detach().cpu().numpy()
            
            # Handle edge cases
            if len(np.unique(a_np)) < 2 or len(np.unique(b_np)) < 2:
                scores[i] = 0.5  # Neutral score for constant values
            else:
                try:
                    corr, _ = spearmanr(a_np, b_np)
                    if np.isnan(corr):
                        scores[i] = 0.5
                    else:
                        scores[i] = 1.0 - abs(corr)  # 0 = perfect correlation, 1 = no correlation
                except:
                    scores[i] = 0.5
        
        return scores
    
    @property
    def name(self) -> str:
        return "rank_divergence"

class MIShiftScore(BaseContradictionScore):
    """Mutual information shift detection"""
    
    def __init__(self, bins: int = 10, normalize: bool = True):
        self.bins = bins
        self.normalize = normalize
    
    def compute(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        batch_size = a.shape[0]
        scores = torch.zeros(batch_size)
        
        for i in range(batch_size):
            a_np = a[i].detach().cpu().numpy()
            b_np = b[i].detach().cpu().numpy()
            
            try:
                # Discretize continuous values for MI computation
                a_discrete = np.digitize(a_np, np.linspace(a_np.min(), a_np.max(), self.bins))
                b_discrete = np.digitize(b_np, np.linspace(b_np.min(), b_np.max(), self.bins))
                
                mi = mutual_info_score(a_discrete, b_discrete)
                
                if self.normalize:
                    # Normalize by max possible MI
                    max_mi = min(np.log(len(np.unique(a_discrete))), 
                                np.log(len(np.unique(b_discrete))))
                    if max_mi > 0:
                        mi = mi / max_mi
                
                scores[i] = 1.0 - mi  # Convert to disagreement score
            except:
                scores[i] = 0.5
        
        return scores
    
    @property
    def name(self) -> str:
        return "mi_shift"

class SignFlipScore(BaseContradictionScore):
    """Rate of sign disagreements between sequences"""
    
    def compute(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Sign of each element
        sign_a = torch.sign(a)
        sign_b = torch.sign(b)
        
        # Count disagreements
        disagreements = (sign_a != sign_b).float()
        
        # Return proportion of sign flips
        return torch.mean(disagreements, dim=-1)
    
    @property
    def name(self) -> str:
        return "sign_flip_rate"

class MagnitudeRatioScore(BaseContradictionScore):
    """Ratio of magnitudes: max(||a||, ||b||) / min(||a||, ||b||)"""
    
    def compute(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        norm_a = torch.norm(a, dim=-1)
        norm_b = torch.norm(b, dim=-1)
        
        max_norm = torch.max(norm_a, norm_b)
        min_norm = torch.min(norm_a, norm_b)
        
        # Avoid division by zero
        ratio = max_norm / (min_norm + 1e-8)
        
        # Convert to 0-1 scale using tanh
        return torch.tanh((ratio - 1.0) / 2.0)
    
    @property
    def name(self) -> str:
        return "magnitude_ratio"

class DirectionChangeScore(BaseContradictionScore):
    """Angular difference between vectors in radians"""
    
    def compute(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Normalize vectors
        a_norm = F.normalize(a, dim=-1, eps=1e-8)
        b_norm = F.normalize(b, dim=-1, eps=1e-8)
        
        # Dot product gives cos(angle)
        cos_angle = torch.sum(a_norm * b_norm, dim=-1)
        
        # Clamp to avoid numerical issues with acos
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        
        # Convert to angle in radians
        angle = torch.acos(cos_angle)
        
        # Normalize to 0-1 (0 to pi)
        return angle / np.pi
    
    @property
    def name(self) -> str:
        return "direction_change"

class ContradictionScorer:
    """Manager for multiple contradiction scoring methods"""
    
    def __init__(self, score_names: List[str] = None):
        if score_names is None:
            score_names = ["cosine", "delta"]
        
        self.scorers = {}
        for name in score_names:
            self.scorers[name] = self._create_scorer(name)
        
        logger.info(f"Initialized ContradictionScorer with: {list(self.scorers.keys())}")
    
    def _create_scorer(self, name: str) -> BaseContradictionScore:
        """Factory method for scoring functions"""
        scorers = {
            "delta": DeltaScore(),
            "cosine": CosineScore(),
            "rank_divergence": RankDivergenceScore(),
            "mi_shift": MIShiftScore(),
            "sign_flip_rate": SignFlipScore(),
            "magnitude_ratio": MagnitudeRatioScore(),
            "direction_change": DirectionChangeScore(),
        }
        
        if name not in scorers:
            raise ValueError(f"Unknown scorer: {name}. Available: {list(scorers.keys())}")
        
        return scorers[name]
    
    def compute_pairwise(self, a: torch.Tensor, b: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute all registered scores between two tensors"""
        results = {}
        
        for name, scorer in self.scorers.items():
            try:
                score = scorer.compute(a, b)
                results[name] = score
            except Exception as e:
                logger.warning(f"Failed to compute {name} score: {e}")
                results[name] = torch.zeros(a.shape[0])
        
        return results
    
    def compute_groupwise(self, tensors: Dict[str, torch.Tensor]) -> Dict[Tuple[str, str], Dict[str, torch.Tensor]]:
        """Compute pairwise scores for all combinations in a group"""
        results = {}
        tensor_names = list(tensors.keys())
        
        for i, name_a in enumerate(tensor_names):
            for j, name_b in enumerate(tensor_names):
                if i < j:  # Only compute upper triangle
                    pair_key = (name_a, name_b)
                    results[pair_key] = self.compute_pairwise(tensors[name_a], tensors[name_b])
        
        return results
    
    def aggregate_scores(self, score_dict: Dict[str, torch.Tensor], 
                        weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """Aggregate multiple scores into a single contradiction score"""
        if weights is None:
            weights = {name: 1.0 for name in score_dict.keys()}
        
        weighted_scores = []
        total_weight = 0
        
        for name, score in score_dict.items():
            weight = weights.get(name, 1.0)
            weighted_scores.append(weight * score)
            total_weight += weight
        
        if weighted_scores:
            return torch.stack(weighted_scores).sum(dim=0) / total_weight
        else:
            return torch.zeros(1)
    
    def get_contradiction_summary(self, a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
        """Get summary statistics for contradiction scores"""
        scores = self.compute_pairwise(a, b)
        
        summary = {}
        for name, score_tensor in scores.items():
            summary[f"{name}_mean"] = score_tensor.mean().item()
            summary[f"{name}_std"] = score_tensor.std().item()
            summary[f"{name}_max"] = score_tensor.max().item()
            summary[f"{name}_min"] = score_tensor.min().item()
        
        return summary
    
    def add_scorer(self, name: str, scorer: BaseContradictionScore):
        """Add a custom scorer"""
        self.scorers[name] = scorer
        logger.info(f"Added custom scorer: {name}")
    
    def remove_scorer(self, name: str):
        """Remove a scorer"""
        if name in self.scorers:
            del self.scorers[name]
            logger.info(f"Removed scorer: {name}")

def create_default_scorer() -> ContradictionScorer:
    """Create scorer with sensible defaults"""
    return ContradictionScorer(["cosine", "delta", "magnitude_ratio"])

def create_ordinal_scorer() -> ContradictionScorer:
    """Create scorer optimized for ordinal data"""
    return ContradictionScorer(["rank_divergence", "sign_flip_rate", "direction_change"])

def create_comprehensive_scorer() -> ContradictionScorer:
    """Create scorer with all available methods"""
    return ContradictionScorer([
        "cosine", "delta", "rank_divergence", "mi_shift", 
        "sign_flip_rate", "magnitude_ratio", "direction_change"
    ])

# Convenience functions
def quick_contradiction_score(a: torch.Tensor, b: torch.Tensor, 
                             method: str = "cosine") -> torch.Tensor:
    """Quick single-method scoring"""
    scorer = ContradictionScorer([method])
    results = scorer.compute_pairwise(a, b)
    return results[method]

def contradiction_heatmap(tensors: Dict[str, torch.Tensor], 
                         method: str = "cosine") -> np.ndarray:
    """Create contradiction heatmap for visualization"""
    scorer = ContradictionScorer([method])
    results = scorer.compute_groupwise(tensors)
    
    names = list(tensors.keys())
    n = len(names)
    heatmap = np.zeros((n, n))
    
    # Fill upper triangle
    for (name_a, name_b), scores in results.items():
        i = names.index(name_a)
        j = names.index(name_b)
        heatmap[i, j] = scores[method].mean().item()
        heatmap[j, i] = heatmap[i, j]  # Mirror to lower triangle
    
    return heatmap

if __name__ == "__main__":
    # Demo the scoring system
    print("Contradiction Score Library Demo")
    print("=" * 40)
    
    # Create test data
    batch_size = 4
    dim = 8
    
    # Similar vectors (low contradiction)
    a = torch.randn(batch_size, dim)
    b = a + 0.1 * torch.randn(batch_size, dim)
    
    # Very different vectors (high contradiction) 
    c = torch.randn(batch_size, dim)
    d = -c + torch.randn(batch_size, dim)
    
    scorer = create_default_scorer()
    
    print("Similar vectors (should have low contradiction):")
    scores_low = scorer.compute_pairwise(a, b)
    for name, score in scores_low.items():
        print(f"  {name}: {score.mean().item():.3f} ± {score.std().item():.3f}")
    
    print("\nVery different vectors (should have high contradiction):")
    scores_high = scorer.compute_pairwise(c, d)
    for name, score in scores_high.items():
        print(f"  {name}: {score.mean().item():.3f} ± {score.std().item():.3f}")
    
    # Test groupwise scoring
    print("\nGroupwise scoring:")
    tensors = {"modality_1": a, "modality_2": b, "modality_3": c}
    groupwise = scorer.compute_groupwise(tensors)
    
    for (name_a, name_b), scores in groupwise.items():
        print(f"  {name_a} vs {name_b}:")
        for score_name, score_tensor in scores.items():
            print(f"    {score_name}: {score_tensor.mean().item():.3f}")
    
    print("\nScoring system working correctly!")