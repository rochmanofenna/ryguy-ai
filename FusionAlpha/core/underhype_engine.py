#!/usr/bin/env python3
"""
Universal Contradiction Detection Engine

Core engine for detecting contradictions between any two signal types.
Supports multiple domains including healthcare, cybersecurity, media analysis,
manufacturing, and financial applications.

Originally optimized for financial underhype detection with proven performance.
Now extended for universal contradiction detection across domains.
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ContradictionSignal:
    """Universal contradiction signal data structure"""
    identifier: str  # entity identifier (ticker, patient_id, user_id, etc.)
    date: datetime
    confidence: float
    signal_a: float  # primary signal value
    signal_b: float  # reference/baseline signal value
    context: str  # descriptive context
    expected_outcome: float  # expected result or impact
    signal_strength: str  # 'high', 'medium', 'low'
    domain: str = 'general'  # application domain
    contradiction_type: str = 'general'  # type of contradiction detected

# Backward compatibility alias for financial applications
UnderhypeSignal = ContradictionSignal
    
class UniversalContradictionEngine:
    """
    Universal contradiction detection engine
    
    Detects contradictions between any two signal types across domains:
    - Healthcare: symptoms vs. lab results
    - Cybersecurity: stated behavior vs. actual activity
    - Media: claims vs. evidence
    - Manufacturing: specifications vs. measurements
    - Finance: sentiment vs. price movement (original use case)
    """
    
    def __init__(self, confidence_threshold: float = 2.5, domain: str = 'general'):
        self.confidence_threshold = confidence_threshold
        self.domain = domain
        
        # Domain-specific thresholds and configurations
        self.domain_configs = {
            'finance': {
                # Financial thresholds (original underhype detection)
                'ORCL': {'signal_a': -0.15, 'signal_b': 0.015, 'expected_outcome': 8.03},
                'TSLA': {'signal_a': -0.12, 'signal_b': 0.025, 'expected_outcome': 5.48}, 
                'GOOG': {'signal_a': -0.10, 'signal_b': 0.020, 'expected_outcome': 5.42},
                'GOOGL': {'signal_a': -0.10, 'signal_b': 0.020, 'expected_outcome': 5.42},
                'NVDA': {'signal_a': -0.11, 'signal_b': 0.022, 'expected_outcome': 4.83},
                'AVGO': {'signal_a': -0.13, 'signal_b': 0.018, 'expected_outcome': 5.07},
                'AAPL': {'signal_a': -0.10, 'signal_b': 0.020, 'expected_outcome': 4.50},
                'MSFT': {'signal_a': -0.10, 'signal_b': 0.020, 'expected_outcome': 4.50},
                'AMZN': {'signal_a': -0.12, 'signal_b': 0.025, 'expected_outcome': 4.20},
                'META': {'signal_a': -0.11, 'signal_b': 0.022, 'expected_outcome': 4.30},
                'DEFAULT': {'signal_a': -0.10, 'signal_b': 0.020, 'expected_outcome': 4.00}
            },
            'healthcare': {
                'DEFAULT': {'signal_a': 0.7, 'signal_b': 0.3, 'expected_outcome': 0.8}  # High symptoms, low biomarkers
            },
            'cybersecurity': {
                'DEFAULT': {'signal_a': 0.2, 'signal_b': 0.8, 'expected_outcome': 0.9}  # Low stated, high actual
            },
            'manufacturing': {
                'DEFAULT': {'signal_a': 0.05, 'signal_b': 0.05, 'expected_outcome': 0.7}  # Tolerance thresholds
            },
            'media': {
                'DEFAULT': {'signal_a': 0.8, 'signal_b': 0.2, 'expected_outcome': 0.8}  # High claims, low evidence
            },
            'general': {
                'DEFAULT': {'signal_a': 0.5, 'signal_b': 0.5, 'expected_outcome': 0.5}  # Generic thresholds
            }
        }
        
        logger.info(f"UniversalContradictionEngine initialized with confidence threshold: {confidence_threshold}, domain: {domain}")
    
    def detect_contradiction(self, identifier: str, signal_a: float, signal_b: float, 
                           context: str = "", domain: str = None) -> Optional[ContradictionSignal]:
        """
        Detect universal contradiction between two signals
        
        Args:
            identifier: Entity identifier (ticker, patient_id, user_id, etc.)
            signal_a: Primary signal value
            signal_b: Reference/baseline signal value
            context: Descriptive context
            domain: Override the default domain for this detection
            
        Returns:
            ContradictionSignal if detected, None otherwise
        """
        
        # Use provided domain or fall back to instance domain
        active_domain = domain or self.domain
        domain_config = self.domain_configs.get(active_domain, self.domain_configs['general'])
        
        # Get entity-specific thresholds or use default
        thresh = domain_config.get(identifier, domain_config['DEFAULT'])
        
        # Domain-specific contradiction logic
        contradiction_detected = False
        contradiction_type = 'general'
        
        if active_domain == 'finance':
            # Original underhype: negative sentiment + positive price
            contradiction_detected = signal_a < thresh['signal_a'] and signal_b > thresh['signal_b']
            contradiction_type = 'underhype'
        elif active_domain == 'healthcare':
            # High symptoms + low biomarkers
            contradiction_detected = signal_a > thresh['signal_a'] and signal_b < thresh['signal_b']
            contradiction_type = 'symptom_biomarker_mismatch'
        elif active_domain == 'cybersecurity':
            # Low stated activity + high actual activity
            contradiction_detected = signal_a < thresh['signal_a'] and signal_b > thresh['signal_b']
            contradiction_type = 'behavioral_anomaly'
        elif active_domain == 'manufacturing':
            # Deviation from specifications
            deviation_a = abs(signal_a - thresh['signal_a'])
            deviation_b = abs(signal_b - thresh['signal_b'])
            contradiction_detected = deviation_a > thresh['signal_a'] or deviation_b > thresh['signal_b']
            contradiction_type = 'specification_deviation'
        elif active_domain == 'media':
            # High claims + low evidence
            contradiction_detected = signal_a > thresh['signal_a'] and signal_b < thresh['signal_b']
            contradiction_type = 'claim_evidence_mismatch'
        else:
            # Generic contradiction: significant divergence between signals
            divergence = abs(signal_a - signal_b)
            contradiction_detected = divergence > max(thresh['signal_a'], thresh['signal_b'])
            contradiction_type = 'signal_divergence'
        
        if contradiction_detected:
            # Calculate confidence score
            if active_domain == 'finance':
                signal_a_strength = abs(signal_a) / abs(thresh['signal_a']) if thresh['signal_a'] != 0 else 1
                signal_b_strength = signal_b / thresh['signal_b'] if thresh['signal_b'] != 0 else 1
            else:
                signal_a_strength = abs(signal_a - thresh['signal_a']) / (abs(thresh['signal_a']) + 1e-8)
                signal_b_strength = abs(signal_b - thresh['signal_b']) / (abs(thresh['signal_b']) + 1e-8)
            
            confidence = min(signal_a_strength, signal_b_strength) * (signal_a_strength + signal_b_strength) / 2
            
            # Only return signals above confidence threshold
            if confidence >= self.confidence_threshold:
                
                # Determine signal strength
                if confidence >= 3.5:
                    signal_strength = 'high'
                elif confidence >= 3.0:
                    signal_strength = 'medium'
                else:
                    signal_strength = 'low'
                
                return ContradictionSignal(
                    identifier=identifier,
                    date=datetime.now(),
                    confidence=confidence,
                    signal_a=signal_a,
                    signal_b=signal_b,
                    context=context,
                    expected_outcome=thresh['expected_outcome'],
                    signal_strength=signal_strength,
                    domain=active_domain,
                    contradiction_type=contradiction_type
                )
        
        return None
    
    # Backward compatibility method for financial applications
    def detect_underhype(self, ticker: str, sentiment: float, price_movement: float, 
                        headline: str = "") -> Optional[ContradictionSignal]:
        """Backward compatibility wrapper for financial underhype detection"""
        return self.detect_contradiction(
            identifier=ticker,
            signal_a=sentiment,
            signal_b=price_movement,
            context=headline,
            domain='finance'
        )
    
    def batch_detect_underhype(self, data: List[Dict]) -> List[UnderhypeSignal]:
        """
        Batch process multiple potential underhype scenarios
        
        Args:
            data: List of dicts with keys: ticker, sentiment, price_movement, headline, date
            
        Returns:
            List of UnderhypeSignal objects
        """
        
        signals = []
        
        for item in data:
            try:
                signal = self.detect_underhype(
                    ticker=item['ticker'],
                    sentiment=item['sentiment'], 
                    price_movement=item['price_movement'],
                    headline=item.get('headline', '')
                )
                
                if signal:
                    # Override date if provided
                    if 'date' in item:
                        signal.date = item['date']
                    signals.append(signal)
                    
            except Exception as e:
                logger.warning(f"Error processing {item.get('ticker', 'unknown')}: {e}")
                continue
        
        logger.info(f"Detected {len(signals)} underhype signals from {len(data)} scenarios")
        return signals
    
    def get_position_size_recommendation(self, signal: UnderhypeSignal, 
                                       portfolio_value: float) -> Dict[str, float]:
        """
        Get recommended position sizing based on signal confidence and expected returns
        
        Args:
            signal: UnderhypeSignal object
            portfolio_value: Total portfolio value
            
        Returns:
            Dict with position sizing recommendations
        """
        
        # Base position sizes by signal strength (as % of portfolio)
        base_sizes = {
            'high': 0.025,    # 2.5% for high confidence (≥3.5)
            'medium': 0.020,  # 2.0% for medium confidence (≥3.0)
            'low': 0.015      # 1.5% for low confidence (≥2.5)
        }
        
        base_pct = base_sizes[signal.signal_strength]
        
        # Adjust based on expected return
        if signal.expected_return > 7.0:
            adjustment = 1.2  # Increase for high expected return tickers
        elif signal.expected_return > 5.0:
            adjustment = 1.1
        else:
            adjustment = 1.0
        
        final_pct = min(base_pct * adjustment, 0.025)  # Cap at 2.5%
        position_value = portfolio_value * final_pct
        
        return {
            'percentage': final_pct,
            'dollar_amount': position_value,
            'signal_strength': signal.signal_strength,
            'expected_return': signal.expected_return,
            'confidence': signal.confidence,
            'rationale': f"{signal.signal_strength.title()} confidence underhype signal"
        }
    
    def get_portfolio_allocation_limit(self, current_underhype_positions: int) -> float:
        """
        Get maximum portfolio allocation for underhype signals
        
        Args:
            current_underhype_positions: Number of current underhype positions
            
        Returns:
            Maximum portfolio percentage for underhype signals
        """
        
        # Conservative scaling based on number of positions
        if current_underhype_positions <= 5:
            return 0.10  # 10% max
        elif current_underhype_positions <= 10:
            return 0.15  # 15% max  
        else:
            return 0.20  # 20% max (aggressive)
    
    def validate_signal_quality(self, signal: UnderhypeSignal) -> Dict[str, Union[bool, str]]:
        """
        Validate signal quality and provide feedback
        
        Args:
            signal: UnderhypeSignal to validate
            
        Returns:
            Dict with validation results
        """
        
        validations = {
            'passes_confidence': signal.confidence >= self.confidence_threshold,
            'strong_sentiment_divergence': abs(signal.finbert_sentiment) > 0.2,
            'significant_price_movement': signal.price_movement > 0.025,
            'preferred_ticker': signal.ticker in ['ORCL', 'TSLA', 'GOOG', 'GOOGL', 'NVDA'],
            'high_expected_return': signal.expected_return > 5.0
        }
        
        score = sum(validations.values())
        
        if score >= 4:
            quality = 'excellent'
        elif score >= 3:
            quality = 'good'
        elif score >= 2:
            quality = 'acceptable'
        else:
            quality = 'poor'
        
        return {
            'overall_quality': quality,
            'quality_score': score,
            'max_score': len(validations),
            'validations': validations,
            'recommendation': 'STRONG BUY' if score >= 4 else 'BUY' if score >= 2 else 'HOLD'
        }

class UnderhypePortfolioManager:
    """
    Portfolio management specifically for underhype signals
    """
    
    def __init__(self, max_allocation: float = 0.15):
        self.max_allocation = max_allocation  # 15% default max allocation
        self.active_positions = {}
        self.position_history = []
        
    def can_add_position(self, signal: UnderhypeSignal, portfolio_value: float) -> bool:
        """Check if new position can be added within risk limits"""
        
        current_allocation = sum(pos['value'] for pos in self.active_positions.values()) / portfolio_value
        
        engine = UnderhypeEngine()
        position_rec = engine.get_position_size_recommendation(signal, portfolio_value)
        new_position_pct = position_rec['percentage']
        
        return (current_allocation + new_position_pct) <= self.max_allocation
    
    def add_position(self, signal: UnderhypeSignal, portfolio_value: float) -> Dict:
        """Add new underhype position to portfolio"""
        
        if not self.can_add_position(signal, portfolio_value):
            return {'success': False, 'reason': 'Exceeds allocation limit'}
        
        engine = UnderhypeEngine()
        position_rec = engine.get_position_size_recommendation(signal, portfolio_value)
        
        position_id = f"{signal.ticker}_{signal.date.strftime('%Y%m%d_%H%M%S')}"
        
        position = {
            'id': position_id,
            'ticker': signal.ticker,
            'entry_date': signal.date,
            'confidence': signal.confidence,
            'expected_return': signal.expected_return,
            'position_size': position_rec['percentage'],
            'value': position_rec['dollar_amount'],
            'status': 'active'
        }
        
        self.active_positions[position_id] = position
        
        logger.info(f"Added underhype position: {signal.ticker} ({position_rec['percentage']:.1%} allocation)")
        
        return {'success': True, 'position_id': position_id, 'position': position}
    
    def close_position(self, position_id: str, exit_date: datetime, actual_return: float) -> Dict:
        """Close underhype position and record performance"""
        
        if position_id not in self.active_positions:
            return {'success': False, 'reason': 'Position not found'}
        
        position = self.active_positions.pop(position_id)
        position['exit_date'] = exit_date
        position['actual_return'] = actual_return
        position['status'] = 'closed'
        position['days_held'] = (exit_date - position['entry_date']).days
        
        self.position_history.append(position)
        
        logger.info(f"Closed underhype position: {position['ticker']} ({actual_return:.2%} return)")
        
        return {'success': True, 'position': position}

# Backward compatibility alias for financial applications
UnderhypeEngine = UniversalContradictionEngine
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""
        
        active_count = len(self.active_positions)
        total_value = sum(pos['value'] for pos in self.active_positions.values())
        
        if self.position_history:
            closed_returns = [pos['actual_return'] for pos in self.position_history]
            avg_return = np.mean(closed_returns)
            win_rate = sum(1 for r in closed_returns if r > 0) / len(closed_returns)
        else:
            avg_return = 0
            win_rate = 0
        
        return {
            'active_positions': active_count,
            'active_value': total_value,
            'closed_positions': len(self.position_history),
            'average_return': avg_return,
            'win_rate': win_rate,
            'max_allocation': self.max_allocation
        }