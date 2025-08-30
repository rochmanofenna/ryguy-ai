# Universal Signal Models for Multi-Domain Contradiction Detection
from .universal_signal_model import (
    UniversalSignalModel,
    EncoderSignalA,
    EncoderSignalB,
    ProjectionHead,
    AdaptiveFusion,
    DecisionHead,
    ContradictionLoss
)

# Backward compatibility imports for financial applications
from .universal_signal_model import (
    TradingModel,
    EncoderTechnical,
    EncoderSentiment
)

# Existing models
from .contradiction_graph import *
from .finbert import *
from .fusionnet import *
from .improved_fusion import *
from .temporal_encoder import *

# Keep the original trading_model for backward compatibility
from .trading_model import *

__all__ = [
    # Universal models
    'UniversalSignalModel',
    'EncoderSignalA', 
    'EncoderSignalB',
    'ProjectionHead',
    'AdaptiveFusion',
    'DecisionHead',
    'ContradictionLoss',
    
    # Backward compatibility
    'TradingModel',
    'EncoderTechnical',
    'EncoderSentiment'
]