#!/usr/bin/env python3
"""
Cybersecurity Contradiction Detection Example

This example demonstrates how to use FusionAlpha for cybersecurity applications,
detecting contradictions between user behavior patterns and stated intentions,
or between normal baselines and current activity.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import sys
import os

# Add the parent directory to sys.path to import fusion_alpha
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class CybersecurityContradictionDetector:
    """
    Cybersecurity-specific contradiction detection using FusionAlpha framework.
    
    Detects contradictions between:
    - User stated activity vs. actual network behavior
    - Historical baseline vs. current activity patterns
    - Authentication claims vs. behavioral biometrics
    - System logs vs. expected operational patterns
    """
    
    def __init__(self, confidence_threshold=0.6, temporal_window=300):
        self.confidence_threshold = confidence_threshold
        self.temporal_window = temporal_window  # seconds
        
    def detect_behavior_contradiction(self, stated_activity, actual_behavior,
                                    behavior_names, user_id=None, session_id=None):
        """
        Detect contradictions between stated user activity and observed behavior.
        
        Args:
            stated_activity: User's claimed activity levels (0-1 scale)
            actual_behavior: Observed network/system behavior (0-1 scale)  
            behavior_names: Names of behavior metrics being tracked
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            ContradictionResult with detection details
        """
        
        stated_array = np.array(stated_activity)
        actual_array = np.array(actual_behavior)
        
        # Calculate contradiction metrics
        stated_level = np.mean(stated_array)
        actual_level = np.mean(actual_array)
        
        # Behavioral divergence analysis
        divergence = np.mean(np.abs(stated_array - actual_array))
        correlation = np.corrcoef(stated_array, actual_array)[0, 1] if len(stated_array) > 1 else 0
        
        # Anomaly scoring
        anomaly_score = max(0, actual_level - stated_level)  # Higher actual than stated = suspicious
        confidence = min(1.0, divergence + anomaly_score)
        
        is_contradictory = confidence > self.confidence_threshold
        
        # Determine threat type
        if stated_level < 0.3 and actual_level > 0.7:
            threat_type = "COVERT_ACTIVITY"  # Low stated, high actual
        elif stated_level > 0.7 and actual_level > 0.8:
            threat_type = "EXCESSIVE_BEHAVIOR"  # Both high but behavior exceeds claims
        elif correlation < -0.5:
            threat_type = "DECEPTIVE_PATTERN"  # Inverse correlation
        else:
            threat_type = "TEMPORAL_ANOMALY"  # Timing-based inconsistency
            
        return {
            'is_contradictory': is_contradictory,
            'confidence': confidence,
            'threat_type': threat_type,
            'stated_level': stated_level,
            'actual_level': actual_level,
            'divergence': divergence,
            'correlation': correlation,
            'anomaly_score': anomaly_score,
            'user_id': user_id,
            'session_id': session_id,
            'timestamp': datetime.now(),
            'risk_level': 'HIGH' if confidence > 0.8 else 'MEDIUM' if confidence > 0.6 else 'LOW',
            'details': {
                'stated': dict(zip(behavior_names, stated_activity)),
                'actual': dict(zip(behavior_names, actual_behavior))
            }
        }

def simulate_network_scenarios():
    """Generate realistic cybersecurity scenarios for demonstration."""
    
    # Scenario 1: Insider threat - User claims light activity but heavy data access
    insider_threat = {
        'user_id': 'U001',
        'scenario': 'Potential Insider Threat - Data Exfiltration',
        'stated_activity': {
            'work_intensity': [0.2, 0.1, 0.3, 0.2],    # Claims light work
            'file_access_need': [0.1, 0.2, 0.1, 0.1],  # Claims minimal file access
            'network_usage': [0.2, 0.3, 0.2, 0.2]      # Claims normal usage
        },
        'actual_behavior': {
            'data_transfer': [0.9, 0.8, 0.9, 0.8],     # Heavy data transfers
            'file_access_rate': [0.8, 0.9, 0.7, 0.8],  # Excessive file access
            'off_hours_activity': [0.7, 0.8, 0.6, 0.7] # High off-hours activity
        }
    }
    
    # Scenario 2: Account compromise - Normal user claims vs. malicious behavior
    compromised_account = {
        'user_id': 'U002',
        'scenario': 'Compromised Account - Credential Stuffing',
        'stated_activity': {
            'login_frequency': [0.3, 0.3, 0.4, 0.3],   # Normal login claims
            'geographic_access': [0.1, 0.1, 0.1, 0.1], # Claims single location
            'application_usage': [0.4, 0.3, 0.4, 0.3]  # Claims normal app usage
        },
        'actual_behavior': {
            'failed_logins': [0.9, 0.8, 0.9, 0.7],     # High failed login attempts
            'geo_anomalies': [0.8, 0.9, 0.8, 0.9],     # Multiple geographic locations
            'privilege_escalation': [0.7, 0.8, 0.6, 0.7] # Attempting privilege escalation
        }
    }
    
    # Scenario 3: Normal user - Consistent behavior (no contradiction)
    normal_user = {
        'user_id': 'U003',
        'scenario': 'Normal User - Consistent Behavior',
        'stated_activity': {
            'work_intensity': [0.6, 0.7, 0.6, 0.7],    # Moderate work claims
            'collaboration': [0.5, 0.6, 0.5, 0.6],     # Normal collaboration
            'resource_usage': [0.4, 0.5, 0.4, 0.5]     # Moderate resource use
        },
        'actual_behavior': {
            'network_activity': [0.6, 0.7, 0.5, 0.6],  # Matches stated activity
            'file_operations': [0.5, 0.6, 0.5, 0.6],   # Consistent file access
            'system_calls': [0.4, 0.5, 0.4, 0.5]       # Normal system usage
        }
    }
    
    # Scenario 4: Advanced Persistent Threat (APT)
    apt_scenario = {
        'user_id': 'U004',
        'scenario': 'APT - Lateral Movement',
        'stated_activity': {
            'routine_tasks': [0.4, 0.4, 0.5, 0.4],     # Claims routine work
            'system_admin': [0.1, 0.1, 0.2, 0.1],      # Claims no admin work
            'external_comm': [0.2, 0.2, 0.3, 0.2]      # Claims minimal external comm
        },
        'actual_behavior': {
            'lateral_movement': [0.8, 0.7, 0.8, 0.9],  # High lateral movement
            'admin_commands': [0.9, 0.8, 0.9, 0.8],    # Excessive admin commands
            'c2_communication': [0.7, 0.8, 0.7, 0.8]   # Command & control traffic
        }
    }
    
    return [insider_threat, compromised_account, normal_user, apt_scenario]

def analyze_security_contradictions():
    """Analyze contradictions across multiple security scenarios."""
    
    detector = CybersecurityContradictionDetector(confidence_threshold=0.5)
    scenarios = simulate_network_scenarios()
    
    results = []
    
    print("Cybersecurity Contradiction Detection Analysis")
    print("=" * 60)
    
    for scenario in scenarios:
        print(f"\nUser: {scenario['user_id']} - {scenario['scenario']}")
        print("-" * 50)
        
        # Combine all stated and actual behaviors
        stated_values = []
        actual_values = []
        behavior_names = []
        
        # Process stated activity
        for behavior, values in scenario['stated_activity'].items():
            stated_values.extend(values)
            behavior_names.extend([f"stated_{behavior}_{i}" for i in range(len(values))])
            
        # Process actual behavior (align with stated)
        actual_combined = []
        for behavior, values in scenario['actual_behavior'].items():
            actual_combined.extend(values)
        
        # Ensure equal length arrays
        min_length = min(len(stated_values), len(actual_combined))
        stated_values = stated_values[:min_length]
        actual_combined = actual_combined[:min_length]
        behavior_names = behavior_names[:min_length]
        
        # Detect contradictions
        result = detector.detect_behavior_contradiction(
            stated_activity=stated_values,
            actual_behavior=actual_combined,
            behavior_names=behavior_names,
            user_id=scenario['user_id'],
            session_id=f"SES_{scenario['user_id']}_001"
        )
        
        results.append(result)
        
        # Display results
        print(f"🔍 Threat detected: {result['is_contradictory']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Risk level: {result['risk_level']}")
        print(f"Threat type: {result['threat_type']}")
        print(f"Stated activity level: {result['stated_level']:.3f}")
        print(f"Actual behavior level: {result['actual_level']:.3f}")
        print(f"Behavioral divergence: {result['divergence']:.3f}")
        
        # Security recommendations
        if result['is_contradictory']:
            print(f"\n🚨 SECURITY ALERT:")
            if result['threat_type'] == 'COVERT_ACTIVITY':
                print("   • Potential insider threat detected")
                print("   • Recommend: Enhanced monitoring, access review")
            elif result['threat_type'] == 'EXCESSIVE_BEHAVIOR':
                print("   • Abnormal activity levels detected")
                print("   • Recommend: Immediate investigation, session termination")
            elif result['threat_type'] == 'DECEPTIVE_PATTERN':
                print("   • Deceptive behavior pattern identified")
                print("   • Recommend: Multi-factor authentication, behavioral analysis")
            else:
                print("   • Temporal anomaly in user behavior")
                print("   • Recommend: Time-based access controls, pattern analysis")
                
            print(f"   • Risk score: {result['confidence']:.1%}")
            print(f"   • Suggested actions: Log retention, forensic capture")
        else:
            print("✅ No significant security threat detected")
    
    return results

def threat_intelligence_analysis(results):
    """Analyze threat patterns across multiple users."""
    
    print("\n" + "="*60)
    print("THREAT INTELLIGENCE ANALYSIS")
    print("="*60)
    
    # Aggregate threat statistics
    total_users = len(results)
    threats_detected = sum(r['is_contradictory'] for r in results)
    avg_confidence = np.mean([r['confidence'] for r in results])
    
    threat_types = [r['threat_type'] for r in results if r['is_contradictory']]
    risk_levels = [r['risk_level'] for r in results]
    
    print(f"\n📊 Threat Summary:")
    print(f"   Users analyzed: {total_users}")
    print(f"   Threats detected: {threats_detected}")
    print(f"   Detection rate: {threats_detected/total_users:.1%}")
    print(f"   Average confidence: {avg_confidence:.3f}")
    
    # Risk distribution
    risk_counts = {level: risk_levels.count(level) for level in ['LOW', 'MEDIUM', 'HIGH']}
    print(f"\n🎯 Risk Distribution:")
    for level, count in risk_counts.items():
        print(f"   {level}: {count} users ({count/total_users:.1%})")
    
    # Threat type analysis
    if threat_types:
        threat_counts = {t: threat_types.count(t) for t in set(threat_types)}
        print(f"\n⚠️  Threat Types Detected:")
        for threat_type, count in threat_counts.items():
            print(f"   {threat_type}: {count} incidents")
    
    return {
        'total_users': total_users,
        'threats_detected': threats_detected,
        'detection_rate': threats_detected/total_users,
        'avg_confidence': avg_confidence,
        'risk_distribution': risk_counts,
        'threat_types': threat_counts if threat_types else {}
    }

def visualize_security_analysis(results):
    """Create visualizations for cybersecurity analysis."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract data for plotting
    user_ids = [r['user_id'] for r in results]
    confidences = [r['confidence'] for r in results]
    stated_levels = [r['stated_level'] for r in results]
    actual_levels = [r['actual_level'] for r in results]
    is_threat = [r['is_contradictory'] for r in results]
    
    # Plot 1: Threat confidence by user
    colors = ['red' if threat else 'green' for threat in is_threat]
    ax1.bar(user_ids, confidences, color=colors, alpha=0.7)
    ax1.set_ylabel('Threat Confidence')
    ax1.set_title('Cybersecurity Threat Detection by User')
    ax1.axhline(y=0.5, color='black', linestyle='--', label='Threshold')
    ax1.legend()
    
    # Plot 2: Stated vs. Actual behavior scatter
    for i, (x, y) in enumerate(zip(stated_levels, actual_levels)):
        color = 'red' if is_threat[i] else 'blue'
        ax2.scatter(x, y, c=color, s=100, alpha=0.7, label=user_ids[i])
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Honesty Line')
    ax2.set_xlabel('Stated Activity Level')
    ax2.set_ylabel('Actual Behavior Level')
    ax2.set_title('Behavioral Contradiction Analysis')
    ax2.legend()
    
    # Plot 3: Risk level distribution
    risk_levels = [r['risk_level'] for r in results]
    risk_counts = {level: risk_levels.count(level) for level in ['LOW', 'MEDIUM', 'HIGH']}
    colors_risk = ['green', 'orange', 'red']
    ax3.pie(risk_counts.values(), labels=risk_counts.keys(), autopct='%1.1f%%', 
            colors=colors_risk)
    ax3.set_title('Risk Level Distribution')
    
    # Plot 4: Behavioral divergence over time (simulated timeline)
    divergences = [r['divergence'] for r in results]
    times = pd.date_range(start='2024-01-01 09:00', periods=len(results), freq='H')
    ax4.plot(times, divergences, 'o-', color='purple', linewidth=2, markersize=8)
    ax4.set_ylabel('Behavioral Divergence')
    ax4.set_title('Threat Activity Timeline')
    ax4.tick_params(axis='x', rotation=45)
    
    # Highlight threat periods
    for i, (time, divergence) in enumerate(zip(times, divergences)):
        if is_threat[i]:
            ax4.axvspan(time - timedelta(minutes=30), time + timedelta(minutes=30), 
                       alpha=0.3, color='red')
    
    plt.tight_layout()
    plt.savefig('cybersecurity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def real_time_monitoring_demo():
    """Demonstrate real-time cybersecurity monitoring."""
    
    print("\n" + "="*60)
    print("REAL-TIME SECURITY MONITORING DEMO")
    print("="*60)
    
    detector = CybersecurityContradictionDetector(confidence_threshold=0.6)
    
    print("\n🖥️  Monitoring active user session...")
    print("User ID: U005")
    print("Session: Emergency weekend work")
    
    # Simulate real-time activity stream
    time_points = ['10:00', '10:15', '10:30', '10:45']
    
    for i, time_point in enumerate(time_points):
        print(f"\n⏰ Time: {time_point}")
        
        # User claims normal weekend work
        stated = {
            'urgency_level': 0.4,      # Claims moderate urgency
            'file_access_need': 0.3,   # Claims limited file access
            'external_contact': 0.1    # Claims no external contact
        }
        
        # But behavior shows escalating suspicious activity
        actual = {
            'data_exfiltration': 0.2 + i * 0.2,    # Escalating data transfer
            'privilege_attempts': 0.1 + i * 0.25,  # Increasing privilege attempts
            'network_scanning': 0.0 + i * 0.3      # Growing network reconnaissance
        }
        
        print(f"Stated activity: {stated}")
        print(f"Observed behavior: {actual}")
        
        result = detector.detect_behavior_contradiction(
            stated_activity=list(stated.values()),
            actual_behavior=list(actual.values()),
            behavior_names=list(stated.keys()),
            user_id='U005',
            session_id=f'SES_U005_{time_point}'
        )
        
        print(f"🔍 Threat confidence: {result['confidence']:.3f}")
        print(f"Risk level: {result['risk_level']}")
        
        if result['is_contradictory']:
            print(f"🚨 SECURITY ALERT at {time_point}!")
            print(f"   Threat type: {result['threat_type']}")
            print(f"   Recommended action: Immediate investigation")
            break
        else:
            print("✅ No threat detected")
    
    return result

if __name__ == "__main__":
    print("FusionAlpha Cybersecurity Contradiction Detection")
    print("=" * 50)
    
    # Run main security analysis
    results = analyze_security_contradictions()
    
    # Threat intelligence analysis
    threat_intel = threat_intelligence_analysis(results)
    
    # Create visualizations
    visualize_security_analysis(results)
    
    # Demonstrate real-time monitoring
    monitoring_result = real_time_monitoring_demo()
    
    print(f"\n✅ Cybersecurity analysis complete!")
    print(f"   Analyzed {len(results)} users/sessions")
    print(f"   Detected {sum(r['is_contradictory'] for r in results)} potential threats")
    print(f"   Average confidence: {np.mean([r['confidence'] for r in results]):.3f}")
    
    print(f"\n🛡️  Security Applications:")
    print(f"   • Insider threat detection")
    print(f"   • Account compromise identification")
    print(f"   • Advanced persistent threat (APT) detection")
    print(f"   • Behavioral anomaly monitoring")
    print(f"   • Real-time security orchestration")