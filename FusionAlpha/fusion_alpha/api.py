#!/usr/bin/env python3
"""
FusionAlpha Universal Contradiction Detection API

This module provides a clean, domain-agnostic public API for contradiction detection
across multiple domains including healthcare, cybersecurity, media analysis, 
manufacturing, and financial applications.

Example usage:
    from fusion_alpha.api import ContradictionDetector
    
    # Initialize for healthcare domain
    detector = ContradictionDetector(domain='healthcare', confidence_threshold=0.7)
    
    # Detect contradictions between symptoms and lab results
    result = detector.detect(
        signal_a=[8, 9, 7, 8],  # Patient-reported symptom severity
        signal_b=[0.1, 0.2, 0.1, 0.15],  # Biomarker levels
        identifier='PATIENT_001',
        context='chest_pain_evaluation'
    )
    
    print(f"Contradiction detected: {result.is_contradictory}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Type: {result.contradiction_type}")
"""

import torch
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass
import logging

# Import core components
try:
    from .models.universal_signal_model import UniversalSignalModel
except ImportError:
    UniversalSignalModel = None

try:
    from .core.underhype_engine import UniversalContradictionEngine, ContradictionSignal
except ImportError:
    # Use new architecture components
    from .core.router import FusionAlpha, FusionResult
    from .core.scores import ContradictionScorer
    UniversalContradictionEngine = None
    ContradictionSignal = None

try:
    from .config.underhype_config import get_production_config, get_domain_specific_config
except ImportError:
    def get_production_config(domain='general'):
        return {'domain': domain}
    def get_domain_specific_config(domain='general'):
        return {'domain': domain, 'threshold': 0.7}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ContradictionResult:
    """
    Universal contradiction detection result container
    """
    is_contradictory: bool
    confidence: float
    contradiction_type: str
    signal_a_value: float
    signal_b_value: float
    identifier: str
    domain: str
    timestamp: datetime
    context: str = ""
    expected_outcome: float = 0.0
    signal_strength: str = "medium"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class ContradictionDetector:
    """
    Universal contradiction detector with domain-specific configurations
    
    Supports multiple domains:
    - healthcare: symptom vs. biomarker contradiction detection
    - cybersecurity: behavior vs. baseline anomaly detection
    - manufacturing: specification vs. measurement deviation detection
    - media: claim vs. evidence contradiction detection
    - finance: sentiment vs. price movement analysis
    """
    
    def __init__(self, 
                 domain: str = 'general',
                 confidence_threshold: float = 0.7,
                 device: str = 'auto',
                 config: Optional[Dict] = None):
        """
        Initialize the contradiction detector
        
        Args:
            domain: Application domain ('healthcare', 'cybersecurity', 'manufacturing', 'media', 'finance', 'general')
            confidence_threshold: Minimum confidence for contradiction detection (0.0-1.0)
            device: Computing device ('auto', 'cpu', 'cuda', 'mps')
            config: Optional custom configuration dictionary
        """
        self.domain = domain
        self.confidence_threshold = confidence_threshold
        self.device = self._resolve_device(device)
        
        # Load domain-specific configuration
        if config is None:
            self.config = get_production_config(domain)
        else:
            self.config = config
            
        # Initialize the contradiction engine
        if UniversalContradictionEngine is not None:
            self.engine = UniversalContradictionEngine(
                confidence_threshold=confidence_threshold,
                domain=domain
            )
        else:
            # Use new architecture
            from .core.router import create_healthcare_fusion, create_sre_fusion, create_robotics_fusion
            if domain == 'healthcare':
                self.engine = create_healthcare_fusion()
            elif domain in ['cybersecurity', 'sre']:
                self.engine = create_sre_fusion()
            elif domain == 'robotics':
                self.engine = create_robotics_fusion()
            else:
                # Default general configuration
                self.engine = FusionAlpha(
                    modalities=['general_signal'],
                    gate='rule',
                    experts=['aligned', 'anti_a', 'anti_b', 'fallback'],
                    contradiction_scores=['cosine', 'delta']
                )
        
        # Domain-specific settings
        self.domain_config = get_domain_specific_config(domain)
        
        logger.info(f"ContradictionDetector initialized for domain: {domain}")
        
    def _resolve_device(self, device: str) -> str:
        """Resolve the optimal computing device"""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        return device
    
    def detect(self,
               signal_a: Union[List[float], np.ndarray, float],
               signal_b: Union[List[float], np.ndarray, float],
               identifier: str,
               context: str = "",
               domain_override: Optional[str] = None) -> ContradictionResult:
        """
        Detect contradictions between two signals
        
        Args:
            signal_a: Primary signal values (domain-specific interpretation)
            signal_b: Reference/baseline signal values 
            identifier: Entity identifier (patient_id, user_id, ticker, etc.)
            context: Descriptive context for the signals
            domain_override: Temporarily override the detector's domain
            
        Returns:
            ContradictionResult with detection details
        """
        
        # Convert inputs to scalars if needed
        if isinstance(signal_a, (list, np.ndarray)):
            signal_a_value = float(np.mean(signal_a))
        else:
            signal_a_value = float(signal_a)
            
        if isinstance(signal_b, (list, np.ndarray)):
            signal_b_value = float(np.mean(signal_b))
        else:
            signal_b_value = float(signal_b)
        
        # Use the engine to detect contradictions
        active_domain = domain_override or self.domain
        
        if UniversalContradictionEngine is not None and hasattr(self.engine, 'detect_contradiction'):
            # Legacy engine
            contradiction_signal = self.engine.detect_contradiction(
                identifier=identifier,
                signal_a=signal_a_value,
                signal_b=signal_b_value,
                context=context,
                domain=active_domain
            )
            
            # Convert to public API result format
            if contradiction_signal:
                return ContradictionResult(
                    is_contradictory=True,
                    confidence=contradiction_signal.confidence,
                    contradiction_type=contradiction_signal.contradiction_type,
                    signal_a_value=signal_a_value,
                    signal_b_value=signal_b_value,
                    identifier=identifier,
                    domain=active_domain,
                    timestamp=contradiction_signal.date,
                    context=context,
                    expected_outcome=contradiction_signal.expected_outcome,
                    signal_strength=contradiction_signal.signal_strength,
                    metadata={
                        'domain_config': self.domain_config,
                        'engine_confidence': contradiction_signal.confidence
                    }
                )
            else:
                return ContradictionResult(
                    is_contradictory=False,
                    confidence=0.0,
                    contradiction_type='none',
                    signal_a_value=signal_a_value,
                    signal_b_value=signal_b_value,
                    identifier=identifier,
                    domain=active_domain,
                    timestamp=datetime.now(),
                    context=context,
                    metadata={'domain_config': self.domain_config}
                )
        else:
            # New architecture
            features = torch.tensor([signal_a_value, signal_b_value] + [0.0] * 30, dtype=torch.float32)  # Pad to 32 dims
            result = self.engine.predict(features)
            
            # Determine if contradiction based on confidence and prediction
            is_contradictory = result.confidence > self.confidence_threshold and abs(result.prediction - 0.5) > 0.3
            contradiction_type = "signal_divergence" if is_contradictory else "none"
            
            return ContradictionResult(
                is_contradictory=is_contradictory,
                confidence=result.confidence,
                contradiction_type=contradiction_type,
                signal_a_value=signal_a_value,
                signal_b_value=signal_b_value,
                identifier=identifier,
                domain=active_domain,
                timestamp=datetime.now(),
                context=context,
                expected_outcome=result.prediction,
                signal_strength="high" if result.confidence > 0.8 else "medium" if result.confidence > 0.5 else "low",
                metadata={
                    'domain_config': self.domain_config,
                    'expert_used': result.expert_used,
                    'contradiction_scores': result.contradiction_scores,
                    'abstained': result.abstained
                }
            )
    
    def batch_detect(self,
                     signal_pairs: List[Dict[str, Any]]) -> List[ContradictionResult]:
        """
        Batch detection for multiple signal pairs
        
        Args:
            signal_pairs: List of dictionaries with keys:
                - signal_a: Primary signal values
                - signal_b: Reference signal values  
                - identifier: Entity identifier
                - context: Optional context string
                
        Returns:
            List of ContradictionResult objects
        """
        results = []
        
        for pair in signal_pairs:
            try:
                result = self.detect(
                    signal_a=pair['signal_a'],
                    signal_b=pair['signal_b'],
                    identifier=pair['identifier'],
                    context=pair.get('context', '')
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Error processing {pair.get('identifier', 'unknown')}: {e}")
                # Create a failed result
                results.append(ContradictionResult(
                    is_contradictory=False,
                    confidence=0.0,
                    contradiction_type='error',
                    signal_a_value=0.0,
                    signal_b_value=0.0,
                    identifier=pair.get('identifier', 'unknown'),
                    domain=self.domain,
                    timestamp=datetime.now(),
                    context=f"Error: {str(e)}",
                    metadata={'error': str(e)}
                ))
        
        return results
    
    def configure_domain(self, domain: str, custom_thresholds: Optional[Dict] = None):
        """
        Reconfigure the detector for a different domain
        
        Args:
            domain: New domain to configure for
            custom_thresholds: Optional custom threshold configuration
        """
        self.domain = domain
        self.domain_config = get_domain_specific_config(domain)
        
        # Reinitialize engine with new domain
        if UniversalContradictionEngine is not None:
            self.engine = UniversalContradictionEngine(
                confidence_threshold=self.confidence_threshold,
                domain=domain
            )
            
            if custom_thresholds:
                # Update engine thresholds
                if domain not in self.engine.domain_configs:
                    self.engine.domain_configs[domain] = {}
                self.engine.domain_configs[domain].update(custom_thresholds)
        else:
            # Use new architecture
            from .core.router import create_healthcare_fusion, create_sre_fusion, create_robotics_fusion
            if domain == 'healthcare':
                self.engine = create_healthcare_fusion()
            elif domain in ['cybersecurity', 'sre']:
                self.engine = create_sre_fusion()
            elif domain == 'robotics':
                self.engine = create_robotics_fusion()
            else:
                # Default general configuration
                self.engine = FusionAlpha(
                    modalities=['general_signal'],
                    gate='rule',
                    experts=['aligned', 'anti_a', 'anti_b', 'fallback'],
                    contradiction_scores=['cosine', 'delta']
                )
        
        logger.info(f"Reconfigured detector for domain: {domain}")
    
    def get_domain_examples(self) -> Dict[str, str]:
        """Get example use cases for the current domain"""
        examples = {
            'healthcare': "Detect contradictions between patient-reported symptoms and objective biomarker levels",
            'cybersecurity': "Identify anomalies between stated user behavior and actual network activity",
            'manufacturing': "Flag deviations between design specifications and quality control measurements", 
            'media': "Detect mismatches between claim strength and supporting evidence",
            'finance': "Identify underhype opportunities where negative sentiment contradicts positive price movement",
            'general': "Detect any significant divergence between two signal types"
        }
        return {self.domain: examples.get(self.domain, examples['general'])}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance characteristics for the current configuration"""
        return {
            'domain': self.domain,
            'confidence_threshold': self.confidence_threshold,
            'device': self.device,
            'expected_latency_ms': 25,  # Sub-25ms inference
            'memory_usage_mb': 512,     # ~512MB VRAM
            'supported_batch_size': 32,
            'domain_config': self.domain_config
        }

# Convenience functions for specific domains
def detect_healthcare_contradiction(symptom_severity: Union[List[float], float],
                                  biomarker_levels: Union[List[float], float],
                                  patient_id: str,
                                  context: str = "") -> ContradictionResult:
    """Convenience function for healthcare contradiction detection"""
    detector = ContradictionDetector(domain='healthcare')
    return detector.detect(symptom_severity, biomarker_levels, patient_id, context)

def detect_cybersecurity_anomaly(stated_behavior: Union[List[float], float],
                                actual_behavior: Union[List[float], float], 
                                user_id: str,
                                context: str = "") -> ContradictionResult:
    """Convenience function for cybersecurity anomaly detection"""
    detector = ContradictionDetector(domain='cybersecurity')
    return detector.detect(stated_behavior, actual_behavior, user_id, context)

def detect_manufacturing_deviation(design_specs: Union[List[float], float],
                                 measurements: Union[List[float], float],
                                 batch_id: str,
                                 context: str = "") -> ContradictionResult:
    """Convenience function for manufacturing quality control"""
    detector = ContradictionDetector(domain='manufacturing')
    return detector.detect(design_specs, measurements, batch_id, context)

def detect_media_contradiction(claim_strength: Union[List[float], float],
                              evidence_support: Union[List[float], float],
                              content_id: str, 
                              context: str = "") -> ContradictionResult:
    """Convenience function for media fact-checking"""
    detector = ContradictionDetector(domain='media')
    return detector.detect(claim_strength, evidence_support, content_id, context)

def detect_financial_underhype(sentiment: Union[List[float], float],
                              price_movement: Union[List[float], float],
                              ticker: str,
                              context: str = "") -> ContradictionResult:
    """Convenience function for financial underhype detection"""
    detector = ContradictionDetector(domain='finance')
    return detector.detect(sentiment, price_movement, ticker, context)

# Export main classes and functions
__all__ = [
    'ContradictionDetector',
    'ContradictionResult', 
    'detect_healthcare_contradiction',
    'detect_cybersecurity_anomaly',
    'detect_manufacturing_deviation',
    'detect_media_contradiction',
    'detect_financial_underhype'
]