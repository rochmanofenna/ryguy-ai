"""
FusionAlpha - Universal Contradiction Detection Framework

A general contradiction-aware decision layer that routes "when A and B disagree" 
across any domain. Clean modality registry, pluggable gates, and dead-simple eval harness.

Supported domains:
- Healthcare: symptoms vs biomarkers → triage decisions
- SRE/Ops: metrics vs logs vs user reports → incident detection  
- Robotics: vision vs lidar vs proprioception → safety decisions
- Content moderation: text vs image vs heuristics → moderation actions
- Finance: sentiment vs price movement → trading signals (original use case)

Core architecture:
- Modality registry: pluggable extractors for any data type
- Contradiction scores: cosine, delta, rank divergence, MI shift, etc.
- Pluggable gates: rule/mlp/bandit/rl for routing decisions
- Expert ensemble: Aligned, AntiA, AntiB, SafeFallback + custom
- Calibration & uncertainty: temperature scaling, ECE, selective risk
- One-command eval: `fa eval --gate mlp --dataset healthcare --metrics auroc,ece`
"""

# Main API - the obvious way to use FusionAlpha
from .core.router import FusionAlpha, FusionResult
from .core.router import create_healthcare_fusion, create_sre_fusion, create_robotics_fusion

# Modality registration
from .core.modality_registry import (
    register_modality, get_registry, 
    create_text_modality, create_series_modality, 
    create_image_modality, create_structured_modality,
    TextEmbedder, SeriesStats, ImageExtractor, StructuredExtractor
)

# Gate types  
from .core.gates import GateFactory

# Expert types
from .core.experts import (
    create_standard_experts, create_enn_integrated_experts,
    AlignedExpert, AntiAExpert, AntiBExpert, SafeFallbackExpert, ENNIntegratedExpert
)

# Contradiction scoring
from .core.scores import (
    ContradictionScorer, create_default_scorer, create_ordinal_scorer,
    quick_contradiction_score, contradiction_heatmap
)

# Calibration & uncertainty
from .core.calibrate import (
    create_calibrator, TemperatureCalibrator, UncertaintyEstimator, CalibrationMetrics
)

# Evaluation
from .eval import EvalHarness, EvalConfig, run_evaluation_from_args

# Backward compatibility (old finance-focused API)
from .api import (
    ContradictionDetector as LegacyDetector,
    detect_financial_underhype
)
from .models.universal_signal_model import TradingModel

# Version and metadata
__version__ = "1.0.0"
__author__ = "Ryan Rochman"  
__description__ = "Universal Contradiction Detection Framework"

__all__ = [
    # Main API (start here)
    "FusionAlpha",
    "FusionResult", 
    
    # Domain-specific factories
    "create_healthcare_fusion",
    "create_sre_fusion", 
    "create_robotics_fusion",
    
    # Modality system
    "register_modality",
    "get_registry",
    "create_text_modality",
    "create_series_modality",
    "create_image_modality", 
    "create_structured_modality",
    
    # Gate system
    "GateFactory",
    
    # Expert system  
    "create_standard_experts",
    "AlignedExpert",
    "AntiAExpert", 
    "AntiBExpert",
    "SafeFallbackExpert",
    
    # Scoring system
    "ContradictionScorer",
    "create_default_scorer",
    "quick_contradiction_score",
    
    # Calibration
    "create_calibrator",
    "UncertaintyEstimator",
    "CalibrationMetrics",
    
    # Evaluation
    "EvalHarness",
    "EvalConfig",
    "run_evaluation_from_args",
    
    # Backward compatibility
    "LegacyDetector",
    "detect_financial_underhype", 
    "TradingModel"
]