#!/usr/bin/env python3
"""
FusionAlpha Unified API Demonstration

This example showcases the new unified API for contradiction detection
across multiple domains using the same interface.
"""

import sys
import os
import numpy as np
from datetime import datetime

# Add the parent directory to sys.path to import fusion_alpha
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the unified API
import fusion_alpha as fa

def main():
    """Demonstrate the unified API across multiple domains"""
    
    print("FusionAlpha Universal Contradiction Detection API Demo")
    print("=" * 60)
    
    # Example 1: Healthcare - Symptom vs. Biomarker Contradiction
    print("\n🏥 Healthcare Example:")
    print("-" * 30)
    
    healthcare_result = fa.detect_healthcare_contradiction(
        symptom_severity=[8, 9, 7, 8],  # High patient-reported pain
        biomarker_levels=[0.1, 0.2, 0.1, 0.15],  # Normal inflammatory markers
        patient_id="PATIENT_001",
        context="chronic_pain_evaluation"
    )
    
    print(f"Patient: {healthcare_result.identifier}")
    print(f"Contradiction detected: {healthcare_result.is_contradictory}")
    print(f"Confidence: {healthcare_result.confidence:.3f}")
    print(f"Type: {healthcare_result.contradiction_type}")
    print(f"Signal A (symptoms): {healthcare_result.signal_a_value:.2f}")
    print(f"Signal B (biomarkers): {healthcare_result.signal_b_value:.2f}")
    
    # Example 2: Cybersecurity - Behavior Anomaly Detection
    print("\n🔒 Cybersecurity Example:")
    print("-" * 30)
    
    cybersec_result = fa.detect_cybersecurity_anomaly(
        stated_behavior=[0.1, 0.1, 0.2, 0.1],  # Low claimed activity
        actual_behavior=[0.9, 0.8, 0.9, 0.7],  # High actual network usage
        user_id="USER_001",
        context="weekend_access_anomaly"
    )
    
    print(f"User: {cybersec_result.identifier}")
    print(f"Anomaly detected: {cybersec_result.is_contradictory}")
    print(f"Confidence: {cybersec_result.confidence:.3f}")
    print(f"Type: {cybersec_result.contradiction_type}")
    print(f"Signal A (stated): {cybersec_result.signal_a_value:.2f}")
    print(f"Signal B (actual): {cybersec_result.signal_b_value:.2f}")
    
    # Example 3: Manufacturing - Quality Control Deviation
    print("\n🏭 Manufacturing Example:")
    print("-" * 30)
    
    manufacturing_result = fa.detect_manufacturing_deviation(
        design_specs=[0.95, 0.96, 0.94, 0.95],  # Target specifications
        measurements=[0.85, 0.82, 0.88, 0.81],  # Actual measurements
        batch_id="BATCH_2024_001",
        context="precision_component_qa"
    )
    
    print(f"Batch: {manufacturing_result.identifier}")
    print(f"Deviation detected: {manufacturing_result.is_contradictory}")
    print(f"Confidence: {manufacturing_result.confidence:.3f}")
    print(f"Type: {manufacturing_result.contradiction_type}")
    print(f"Signal A (specs): {manufacturing_result.signal_a_value:.2f}")
    print(f"Signal B (measurements): {manufacturing_result.signal_b_value:.2f}")
    
    # Example 4: Media - Fact Checking
    print("\n📰 Media Analysis Example:")
    print("-" * 30)
    
    media_result = fa.detect_media_contradiction(
        claim_strength=[0.9, 0.85, 0.9, 0.95],  # Strong claims
        evidence_support=[0.1, 0.05, 0.1, 0.0],  # Weak evidence
        content_id="POST_001",
        context="health_misinformation_check"
    )
    
    print(f"Content: {media_result.identifier}")
    print(f"Misinformation detected: {media_result.is_contradictory}")
    print(f"Confidence: {media_result.confidence:.3f}")
    print(f"Type: {media_result.contradiction_type}")
    print(f"Signal A (claims): {media_result.signal_a_value:.2f}")
    print(f"Signal B (evidence): {media_result.signal_b_value:.2f}")
    
    # Example 5: Finance - Underhype Detection (Original Use Case)
    print("\n💰 Financial Example:")
    print("-" * 30)
    
    finance_result = fa.detect_financial_underhype(
        sentiment=[-0.15, -0.12, -0.18, -0.14],  # Negative sentiment
        price_movement=[0.025, 0.018, 0.022, 0.020],  # Positive price movement
        ticker="AAPL",
        context="earnings_underhype_opportunity"
    )
    
    print(f"Ticker: {finance_result.identifier}")
    print(f"Underhype detected: {finance_result.is_contradictory}")
    print(f"Confidence: {finance_result.confidence:.3f}")
    print(f"Type: {finance_result.contradiction_type}")
    print(f"Signal A (sentiment): {finance_result.signal_a_value:.3f}")
    print(f"Signal B (price): {finance_result.signal_b_value:.3f}")
    
    # Example 6: Using the Generic Detector Class
    print("\n🔧 Generic Detector Example:")
    print("-" * 30)
    
    # Initialize a generic detector
    detector = fa.ContradictionDetector(domain='general', confidence_threshold=0.6)
    
    # Batch processing example
    signal_pairs = [
        {
            'signal_a': [0.8, 0.9, 0.7],
            'signal_b': [0.2, 0.1, 0.3],
            'identifier': 'ENTITY_001',
            'context': 'test_scenario_1'
        },
        {
            'signal_a': [0.3, 0.4, 0.3],
            'signal_b': [0.3, 0.4, 0.3],
            'identifier': 'ENTITY_002',
            'context': 'test_scenario_2'
        }
    ]
    
    batch_results = detector.batch_detect(signal_pairs)
    
    for i, result in enumerate(batch_results):
        print(f"Entity {i+1}: {result.identifier}")
        print(f"  Contradiction: {result.is_contradictory}")
        print(f"  Confidence: {result.confidence:.3f}")
    
    # Example 7: Domain Reconfiguration
    print("\n⚙️  Domain Reconfiguration Example:")
    print("-" * 30)
    
    # Start with healthcare
    adaptive_detector = fa.ContradictionDetector(domain='healthcare')
    print(f"Initial domain: {adaptive_detector.domain}")
    print(f"Example use case: {adaptive_detector.get_domain_examples()}")
    
    # Switch to cybersecurity
    adaptive_detector.configure_domain('cybersecurity')
    print(f"Reconfigured domain: {adaptive_detector.domain}")
    print(f"New use case: {adaptive_detector.get_domain_examples()}")
    
    # Performance metrics
    metrics = adaptive_detector.get_performance_metrics()
    print(f"Performance: {metrics['expected_latency_ms']}ms latency, {metrics['memory_usage_mb']}MB memory")
    
    print("\n✅ API demonstration complete!")
    print("\nKey Benefits:")
    print("• Unified interface across all domains")
    print("• Domain-specific optimizations")
    print("• Backward compatibility with financial models")
    print("• Batch processing capabilities")
    print("• Runtime domain switching")
    print("• Performance monitoring")

if __name__ == "__main__":
    main()