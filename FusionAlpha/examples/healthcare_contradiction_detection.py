#!/usr/bin/env python3
"""
Healthcare Contradiction Detection Example

This example demonstrates how to use FusionAlpha for medical applications,
detecting contradictions between patient-reported symptoms and objective
clinical measurements.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to sys.path to import fusion_alpha
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# For now, we'll create a simplified interface that demonstrates the concept
# In the actual refactored version, this would import from fusion_alpha

class HealthcareContradictionDetector:
    """
    Healthcare-specific contradiction detection using FusionAlpha framework.
    
    Detects contradictions between:
    - Patient-reported symptoms vs. lab results
    - Vital signs vs. patient condition reports  
    - Imaging findings vs. clinical assessments
    - Treatment response vs. biomarkers
    """
    
    def __init__(self, confidence_threshold=0.8, temporal_window=24):
        self.confidence_threshold = confidence_threshold
        self.temporal_window = temporal_window  # hours
        
    def detect_symptom_lab_contradiction(self, symptom_scores, lab_values, 
                                       symptom_names, lab_names, patient_id=None):
        """
        Detect contradictions between patient-reported symptoms and lab results.
        
        Args:
            symptom_scores: Patient-reported symptom severity (0-10 scale)
            lab_values: Normalized lab values (0-1 scale, higher = more abnormal)
            symptom_names: Names of symptoms being reported
            lab_names: Names of lab tests
            patient_id: Optional patient identifier
            
        Returns:
            ContradictionResult with detection details
        """
        
        # Normalize symptom scores to 0-1 scale
        normalized_symptoms = np.array(symptom_scores) / 10.0
        normalized_labs = np.array(lab_values)
        
        # Calculate contradiction score
        # High symptoms + normal labs = potential contradiction
        symptom_severity = np.mean(normalized_symptoms)
        lab_abnormality = np.mean(normalized_labs)
        
        # Contradiction logic: inverse correlation suggests mismatch
        correlation = np.corrcoef(normalized_symptoms, normalized_labs)[0, 1]
        contradiction_score = max(0, -correlation)  # Negative correlation = contradiction
        
        # Additional factors
        severity_gap = abs(symptom_severity - lab_abnormality)
        confidence = min(1.0, contradiction_score * severity_gap * 2)
        
        is_contradictory = confidence > self.confidence_threshold
        
        # Determine contradiction type
        if symptom_severity > 0.7 and lab_abnormality < 0.3:
            contradiction_type = "OVERHYPE"  # High symptoms, normal labs
        elif symptom_severity < 0.3 and lab_abnormality > 0.7:
            contradiction_type = "UNDERHYPE"  # Low symptoms, abnormal labs
        else:
            contradiction_type = "TEMPORAL"  # Timing mismatch
            
        return {
            'is_contradictory': is_contradictory,
            'confidence': confidence,
            'type': contradiction_type,
            'symptom_severity': symptom_severity,
            'lab_abnormality': lab_abnormality,
            'correlation': correlation,
            'patient_id': patient_id,
            'timestamp': datetime.now(),
            'details': {
                'symptoms': dict(zip(symptom_names, symptom_scores)),
                'labs': dict(zip(lab_names, lab_values))
            }
        }

def simulate_patient_data():
    """Generate realistic patient data for demonstration."""
    
    # Patient 1: High pain, normal inflammation markers (potential contradiction)
    patient1 = {
        'patient_id': 'P001',
        'condition': 'Chronic Pain - Possible Functional',
        'symptoms': {
            'pain_level': [8, 9, 7, 8, 9],      # High pain reports
            'fatigue': [7, 8, 6, 7, 8],         # High fatigue
            'mobility': [3, 2, 4, 3, 2]         # Low mobility (reversed scale)
        },
        'labs': {
            'crp': [0.1, 0.2, 0.1, 0.15, 0.1],    # Normal C-reactive protein  
            'esr': [0.2, 0.3, 0.2, 0.25, 0.2],    # Normal ESR
            'wbc': [0.1, 0.1, 0.2, 0.1, 0.1]      # Normal white blood cells
        }
    }
    
    # Patient 2: Low symptoms, high inflammatory markers (underhype contradiction)
    patient2 = {
        'patient_id': 'P002', 
        'condition': 'Early Rheumatoid Arthritis - Underreported',
        'symptoms': {
            'pain_level': [3, 2, 4, 3, 2],      # Low pain reports
            'stiffness': [2, 3, 2, 3, 2],       # Minimal stiffness
            'swelling': [1, 2, 1, 2, 1]         # Little perceived swelling
        },
        'labs': {
            'crp': [0.8, 0.9, 0.7, 0.8, 0.9],     # High inflammation
            'rf': [0.9, 0.8, 0.9, 0.8, 0.9],      # High rheumatoid factor
            'anti_ccp': [0.7, 0.8, 0.7, 0.8, 0.7] # High anti-CCP antibodies
        }
    }
    
    # Patient 3: Consistent symptoms and labs (no contradiction)
    patient3 = {
        'patient_id': 'P003',
        'condition': 'Active Inflammatory Disease - Consistent',
        'symptoms': {
            'pain_level': [7, 8, 7, 8, 7],      # High pain
            'swelling': [8, 7, 8, 7, 8],        # High swelling
            'stiffness': [8, 9, 7, 8, 9]        # High stiffness
        },
        'labs': {
            'crp': [0.9, 0.8, 0.9, 0.8, 0.9],     # High inflammation
            'esr': [0.8, 0.9, 0.7, 0.8, 0.9],     # High ESR
            'wbc': [0.7, 0.8, 0.7, 0.8, 0.7]      # Elevated WBC
        }
    }
    
    return [patient1, patient2, patient3]

def analyze_patient_contradictions():
    """Analyze contradictions across multiple patients."""
    
    detector = HealthcareContradictionDetector(confidence_threshold=0.6)
    patients = simulate_patient_data()
    
    results = []
    
    print("Healthcare Contradiction Detection Analysis")
    print("=" * 60)
    
    for patient in patients:
        print(f"\nPatient: {patient['patient_id']} - {patient['condition']}")
        print("-" * 40)
        
        # Combine all symptoms and labs for analysis
        all_symptoms = []
        all_labs = []
        symptom_names = []
        lab_names = []
        
        for symptom, values in patient['symptoms'].items():
            all_symptoms.extend(values)
            symptom_names.extend([f"{symptom}_{i}" for i in range(len(values))])
            
        for lab, values in patient['labs'].items():
            all_labs.extend(values) 
            lab_names.extend([f"{lab}_{i}" for i in range(len(values))])
        
        # Detect contradictions
        result = detector.detect_symptom_lab_contradiction(
            symptom_scores=all_symptoms,
            lab_values=all_labs,
            symptom_names=symptom_names,
            lab_names=lab_names,
            patient_id=patient['patient_id']
        )
        
        results.append(result)
        
        # Display results
        print(f"Contradiction detected: {result['is_contradictory']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Type: {result['type']}")
        print(f"Symptom severity: {result['symptom_severity']:.3f}")
        print(f"Lab abnormality: {result['lab_abnormality']:.3f}")
        print(f"Correlation: {result['correlation']:.3f}")
        
        # Clinical interpretation
        if result['is_contradictory']:
            if result['type'] == 'OVERHYPE':
                print("⚠️  Clinical Alert: High symptoms with normal lab values")
                print("   Consider: Functional disorder, psychological factors, or early disease")
            elif result['type'] == 'UNDERHYPE':
                print("⚠️  Clinical Alert: Low symptoms with abnormal lab values") 
                print("   Consider: Patient minimizing symptoms, early disease stage")
            else:
                print("⚠️  Clinical Alert: Temporal mismatch between symptoms and labs")
                print("   Consider: Disease progression or treatment effects")
        else:
            print("✅ No significant contradiction detected")
    
    return results

def visualize_contradictions(results):
    """Create visualizations of contradiction patterns."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract data for plotting
    patient_ids = [r['patient_id'] for r in results]
    confidences = [r['confidence'] for r in results]
    symptom_scores = [r['symptom_severity'] for r in results]
    lab_scores = [r['lab_abnormality'] for r in results]
    contradictory = [r['is_contradictory'] for r in results]
    
    # Plot 1: Confidence scores
    colors = ['red' if c else 'green' for c in contradictory]
    ax1.bar(patient_ids, confidences, color=colors, alpha=0.7)
    ax1.set_ylabel('Contradiction Confidence')
    ax1.set_title('Contradiction Detection Confidence by Patient')
    ax1.axhline(y=0.6, color='black', linestyle='--', label='Threshold')
    ax1.legend()
    
    # Plot 2: Symptom vs Lab scatter
    for i, (x, y) in enumerate(zip(symptom_scores, lab_scores)):
        color = 'red' if contradictory[i] else 'blue'
        ax2.scatter(x, y, c=color, s=100, alpha=0.7, label=patient_ids[i])
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Agreement')
    ax2.set_xlabel('Symptom Severity (Normalized)')
    ax2.set_ylabel('Lab Abnormality (Normalized)')
    ax2.set_title('Symptom-Lab Correlation Analysis')
    ax2.legend()
    
    # Plot 3: Contradiction types
    types = [r['type'] for r in results]
    type_counts = {t: types.count(t) for t in set(types)}
    ax3.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
    ax3.set_title('Distribution of Contradiction Types')
    
    # Plot 4: Timeline analysis (simulated)
    times = pd.date_range(start='2024-01-01', periods=len(results), freq='D')
    ax4.plot(times, confidences, 'o-', color='purple', linewidth=2, markersize=8)
    ax4.set_ylabel('Contradiction Confidence')
    ax4.set_title('Contradiction Detection Over Time')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('healthcare_contradictions.png', dpi=300, bbox_inches='tight')
    plt.show()

def clinical_decision_support_demo():
    """Demonstrate real-time clinical decision support."""
    
    print("\n" + "="*60)
    print("CLINICAL DECISION SUPPORT DEMO")
    print("="*60)
    
    detector = HealthcareContradictionDetector(confidence_threshold=0.7)
    
    # Simulate real-time patient data input
    print("\n📊 New Patient Assessment:")
    print("Patient ID: P004")
    print("Chief Complaint: Severe chest pain, shortness of breath")
    
    # Patient reports severe symptoms
    symptoms = {
        'chest_pain': 9,
        'shortness_of_breath': 8, 
        'fatigue': 7,
        'anxiety': 8
    }
    
    # But initial labs/vitals are normal
    labs = {
        'troponin': 0.1,      # Normal cardiac marker
        'ekg_abnormal': 0.1,  # Normal EKG
        'heart_rate': 0.2,    # Slightly elevated but normal
        'bp_systolic': 0.3    # Slightly elevated
    }
    
    print(f"\nSymptoms reported: {symptoms}")
    print(f"Initial test results: {labs}")
    
    result = detector.detect_symptom_lab_contradiction(
        symptom_scores=list(symptoms.values()),
        lab_values=list(labs.values()),
        symptom_names=list(symptoms.keys()),
        lab_names=list(labs.keys()),
        patient_id='P004'
    )
    
    print(f"\n🔍 Contradiction Analysis:")
    print(f"Contradiction detected: {result['is_contradictory']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Type: {result['type']}")
    
    if result['is_contradictory']:
        print(f"\n⚠️  CLINICAL DECISION SUPPORT ALERT:")
        print(f"   High symptom severity with normal initial tests")
        print(f"   Recommendations:")
        print(f"   • Consider anxiety/panic disorder")
        print(f"   • Rule out conditions with delayed lab changes")
        print(f"   • Monitor for evolving symptoms")
        print(f"   • Consider stress testing if chest pain persists")
        print(f"   • Evaluate psychological factors")
    
    return result

if __name__ == "__main__":
    print("FusionAlpha Healthcare Contradiction Detection")
    print("=" * 50)
    
    # Run main analysis
    results = analyze_patient_contradictions()
    
    # Create visualizations  
    visualize_contradictions(results)
    
    # Demonstrate clinical decision support
    clinical_result = clinical_decision_support_demo()
    
    print(f"\n✅ Healthcare analysis complete!")
    print(f"   Analyzed {len(results)} patients")
    print(f"   Detected {sum(r['is_contradictory'] for r in results)} contradictions")
    print(f"   Average confidence: {np.mean([r['confidence'] for r in results]):.3f}")
    
    print(f"\n💡 Clinical Applications:")
    print(f"   • Early disease detection")
    print(f"   • Patient compliance monitoring") 
    print(f"   • Diagnostic accuracy improvement")
    print(f"   • Treatment response assessment")
    print(f"   • Quality assurance in healthcare delivery")